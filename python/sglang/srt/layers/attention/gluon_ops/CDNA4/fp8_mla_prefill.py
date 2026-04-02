# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 KV Gluon MLA D512 prefill kernel for DeepSeek on MI350X.

Target config: BM64, BN32, 4 warps, manual pipeline, FP8 KV cache.
MMA: v_mfma_f32_16x16x32_fp8_fp8 (prefix), v_mfma_f32_16x16x32_bf16 (extend).
waves_per_eu=1, matrix_instr_nonkdim=16.

This kernel handles ONLY the D512 MLA shapes (Lq=576, Lv=512) with FP8
KV cache. The 4-warp prefix path uses FP8 MFMA with a staged K/V async
pipeline and pair-wise KPE super-tiles in shared memory. Extend phase
uses BF16 with 2-stage pipeline.

Derived from f16_mla_prefill.py (BF16 variant).
"""

import math

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language.amd.cdna3 import buffer_load as cdna_buffer_load
from triton.experimental.gluon.language._layouts import (
    DistributedLinearLayout,
    DotOperandLayout,
    PaddedSharedLayout,
)

LOG2E = tl.constexpr(1.4426950408889634)


# ===-----------------------------------------------------------------------===#
# Primitives
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _nan_propagating_max(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def nan_propagating_max(x, axis):
    return gl.reduce(x, axis, _nan_propagating_max)


@gluon.jit
def do_mma(a, b, c):
    if b.dtype == tl.float8e4b8 or b.dtype == tl.float8e4nv:
        a_fp8 = tl.cast(a, tl.float8e4nv, bitcast=(a.dtype != tl.bfloat16 and a.dtype != tl.float16))
        b_fp8 = tl.cast(b, tl.float8e4nv, bitcast=True)
        return mfma_cdna4(a_fp8, b_fp8, c)
    return mfma_cdna4(a.to(tl.bfloat16), b.to(tl.bfloat16), c)


@gluon.jit
def _load_kpe_from_global_fp8(
    qk, qpe_dot, kv_base, kv_indices, kv_start, start_n, causal_kv_end,
    stride_kvbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    kt_dot_layout: gl.constexpr,
):
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_dot_layout))
    offs_dpe = gl.arange(0, BLOCK_DPE)
    n_idx = start_n + offs_n
    mask_n = n_idx < causal_kv_end
    kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)
    kpe_ptrs = kv_base + kv_locs[None, :] * stride_kvbs + BLOCK_DMODEL + offs_dpe[:, None]
    kpe = gl.load(kpe_ptrs, mask=mask_n[None, :], other=0.0)
    kpe_dot = gl.convert_layout(kpe, kt_dot_layout)
    qk = do_mma(qpe_dot, kpe_dot, qk)
    return qk


@gluon.jit
def _softmax_prefill_fp8(
    acc, l_i, m_i, qk, start_n, causal_kv_end, q_abs_pos,
    qk_scale, LOGIT_CAP: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr, SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,
    Mask, mask_base_idx, mask_row_stride,
    q_m_idx, seq_len_prefix,
    BLOCK_N: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
):
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)
    n_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
    valid = n_offs[None, :] < causal_kv_end
    if IS_CAUSAL:
        valid = valid & (q_abs_pos[:, None] >= n_offs[None, :])
    if USE_CUSTOM_MASK:
        is_prefix = n_offs[None, :] < seq_len_prefix
        need_mask = (~is_prefix) | (is_prefix & (not SKIP_PREFIX_CUSTOM_MASK))
        mask_ptrs = (
            Mask + mask_base_idx
            + q_m_idx[:, None] * mask_row_stride
            + n_offs[None, :].to(tl.int64)
        )
        mask_vals = gl.load(mask_ptrs, mask=valid & need_mask, other=1)
        valid = valid & gl.where(need_mask, mask_vals != 0, True)
    qk_scaled = gl.where(
        valid,
        qk_scaled,
        gl.full([qk.shape[0], qk.shape[1]], float("-inf"), dtype=gl.float32, layout=mma_layout),
    )

    m_ij = nan_propagating_max(qk_scaled, axis=1)
    m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
    p = gl.exp2(qk_scaled - m_new[:, None])
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


# ===-----------------------------------------------------------------------===#
# Softmax helpers (identical to BF16 -- operate on f32 QK accumulators)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _softmax_prefix(
    acc, l_i, m_i, qk, start_n, seq_len_prefix, q_abs_pos,
    qk_scale, LOGIT_CAP: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr, SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,
    Mask, mask_base_idx, mask_row_stride,
    q_extend_offs,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
):
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)

    n_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
    valid = n_offs[None, :] < seq_len_prefix
    if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
        mask_ptrs = (
            Mask + mask_base_idx
            + q_extend_offs[:, None] * mask_row_stride
            + start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)[None, :]
        )
        mask_vals = gl.load(mask_ptrs, mask=valid, other=0)
        valid = valid & (mask_vals != 0)
    qk_scaled = gl.where(
        valid,
        qk_scaled,
        gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout),
    )

    m_ij = nan_propagating_max(qk_scaled, axis=1)
    m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
    p = gl.exp2(qk_scaled - m_new[:, None])
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


@gluon.jit
def _softmax_extend(
    acc, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
    qk_scale, LOGIT_CAP: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr,
    Mask, mask_base_idx, mask_row_stride,
    seq_len_prefix,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
):
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)

    n_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
    q_offs = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)

    valid = q_offs[:, None] < seq_len_extend
    valid = valid & (n_offs[None, :] < seq_len_extend)
    if IS_CAUSAL:
        valid = valid & (q_offs[:, None] >= n_offs[None, :])
    if USE_CUSTOM_MASK:
        mask_ptrs = (
            Mask + mask_base_idx
            + q_offs[:, None] * mask_row_stride
            + seq_len_prefix.to(tl.int64) + start_n
            + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)[None, :]
        )
        mask_vals = gl.load(mask_ptrs, mask=valid, other=0)
        valid = valid & (mask_vals != 0)
    qk_scaled = gl.where(
        valid,
        qk_scaled,
        gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout),
    )

    m_ij = nan_propagating_max(qk_scaled, axis=1)
    m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
    p = gl.exp2(qk_scaled - m_new[:, None])
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


# ===-----------------------------------------------------------------------===#
# Async load helpers (shared between prefix/extend, dtype-agnostic via SMEM)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _load_prefix_kv_locs(
    kv_indices, kv_start, start_n, seq_len,
    BLOCK_N: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    n_idx = start_n + kt_offs_n
    mask_n = n_idx < seq_len
    safe_idx = gl.where(mask_n, kv_start + n_idx, gl.zeros([BLOCK_N], dtype=tl.int32, layout=kt_offs_n_layout))
    kv_locs = cdna_buffer_load(kv_indices, safe_idx.to(tl.int32), mask=mask_n, other=0)
    return kv_locs, mask_n


@gluon.jit
def _issue_prefix_k_async(
    kt_smem, k_base, kv_locs, mask_n_kt,
    stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kv_locs_kt = gl.convert_layout(kv_locs, gl.SliceLayout(dim=0, parent=kt_async_layout))
    kt_offsets = (kt_offs_d[:, None] + kv_locs_kt[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        kt_smem, k_base, kt_offsets, mask=mask_n_kt[None, :], other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_prefix_kpe_async(
    kpe_smem, k_base, kv_locs, mask_n_kpe,
    stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    kpe_async_layout: gl.constexpr,
):
    kpe_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
    kpe_offs_d = gl.arange(0, BLOCK_DPE, layout=kpe_offs_d_layout)
    kv_locs_kpe = gl.convert_layout(kv_locs, gl.SliceLayout(dim=0, parent=kpe_async_layout))
    kpe_offsets = ((kpe_offs_d[:, None] + BLOCK_DMODEL) + kv_locs_kpe[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        kpe_smem, k_base, kpe_offsets, mask=mask_n_kpe[None, :], other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_kpe_super_dma(
    kpe_smem, k_base, kv_indices, kv_start, start_n, seq_len,
    stride_kbs,
    BLOCK_DPE: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_N2: gl.constexpr,
    kpe_super_layout: gl.constexpr,
):
    """Issue DMA for a [64, 64] KPE super-tile covering 2 N-blocks. No commit."""
    kpe_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_super_layout)
    kpe_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kpe_super_layout)
    kpe_offs_d = gl.arange(0, BLOCK_DPE, layout=kpe_offs_d_layout)
    offs_n = gl.arange(0, BLOCK_N2, layout=kpe_offs_n_layout)
    n_idx = start_n + offs_n
    mask_n = n_idx < seq_len
    safe_idx = gl.where(mask_n, kv_start + n_idx, gl.zeros([BLOCK_N2], dtype=tl.int32, layout=kpe_offs_n_layout))
    kv_locs = cdna_buffer_load(kv_indices, safe_idx.to(tl.int32), mask=mask_n, other=0)
    kv_locs_kpe = gl.convert_layout(kv_locs, kpe_offs_n_layout)
    kpe_offsets = ((kpe_offs_d[:, None] + BLOCK_DMODEL) + kv_locs_kpe[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        kpe_smem, k_base, kpe_offsets, mask=mask_n[None, :], other=0.0
    )


@gluon.jit
def _issue_prefix_kvkpe_async(
    kt_smem, kpe_smem, k_base, kv_locs, mask_n_kt, mask_n_kpe,
    stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    kt_async_layout: gl.constexpr, kpe_async_layout: gl.constexpr,
):
    _issue_prefix_k_async(
        kt_smem, k_base, kv_locs, mask_n_kt,
        stride_kbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
    )
    _issue_prefix_kpe_async(
        kpe_smem, k_base, kv_locs, mask_n_kpe,
        stride_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kpe_async_layout,
    )


@gluon.jit
def _issue_prefix_v_async(
    v_smem, v_base, kv_locs, mask_n_v,
    stride_vbs,
    BLOCK_N: gl.constexpr, BLOCK_DV: gl.constexpr,
    v_async_layout: gl.constexpr,
):
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    kv_locs_v = gl.convert_layout(kv_locs, gl.SliceLayout(dim=1, parent=v_async_layout))
    v_offsets = (kv_locs_v[:, None] * stride_vbs + v_offs_d[None, :]).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        v_smem, v_base, v_offsets, mask=mask_n_v[:, None], other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_prefix_kv_async(
    kt_smem, v_smem, kv_base, kv_locs, mask_n,
    stride_kbs, stride_vbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_DV: gl.constexpr,
    kt_async_layout: gl.constexpr, v_async_layout: gl.constexpr,
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kv_locs_kt = gl.convert_layout(kv_locs, gl.SliceLayout(dim=0, parent=kt_async_layout))
    kt_offsets = (kt_offs_d[:, None] + kv_locs_kt[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        kt_smem, kv_base, kt_offsets, mask=mask_n[None, :], other=0.0
    )

    mask_n_v = gl.convert_layout(mask_n, gl.SliceLayout(dim=1, parent=v_async_layout))
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    kv_locs_v = gl.convert_layout(kv_locs, gl.SliceLayout(dim=1, parent=v_async_layout))
    v_offsets = (kv_locs_v[:, None] * stride_vbs + v_offs_d[None, :]).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        v_smem, kv_base, v_offsets, mask=mask_n_v[:, None], other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_kpe_super_async(
    kpe_smem, k_base, kv_indices, kv_start, start_n, seq_len,
    stride_kbs,
    BLOCK_DPE: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_N2: gl.constexpr,
    kpe_super_layout: gl.constexpr,
):
    _issue_kpe_super_dma(
        kpe_smem, k_base, kv_indices, kv_start, start_n, seq_len,
        stride_kbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout,
    )
    cdna4_async.commit_group()


@gluon.jit
def _load_kpe_super_half(
    kpe_smem, half_start: gl.constexpr,
    BLOCK_N: gl.constexpr,
    kt_dot_layout: gl.constexpr,
):
    return cdna4_async.load_shared_relaxed(
        kpe_smem.slice(half_start, BLOCK_N, dim=1),
        kt_dot_layout,
    )


@gluon.jit
def _issue_async_k_extend(
    kt_smem, k_base, start_n, seq_len, stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    kt_offsets = (kt_offs_d[:, None] + (start_n + kt_offs_n[None, :]) * stride_kbs).to(tl.int32)
    kt_mask = (start_n + kt_offs_n[None, :]) < seq_len
    cdna4_async.buffer_load_to_shared(
        kt_smem, k_base, kt_offsets, mask=kt_mask, other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_async_kpe_extend(
    kpe_smem, k_base, start_n, seq_len, stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    kpe_async_layout: gl.constexpr,
):
    kpe_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
    kpe_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kpe_async_layout)
    kpe_offs_d = gl.arange(0, BLOCK_DPE, layout=kpe_offs_d_layout)
    kpe_offs_n = gl.arange(0, BLOCK_N, layout=kpe_offs_n_layout)
    kpe_offsets = (
        (kpe_offs_d[:, None] + BLOCK_DMODEL)
        + (start_n + kpe_offs_n[None, :]) * stride_kbs
    ).to(tl.int32)
    kpe_mask = (start_n + kpe_offs_n[None, :]) < seq_len
    cdna4_async.buffer_load_to_shared(
        kpe_smem, k_base, kpe_offsets, mask=kpe_mask, other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_async_v_extend(
    v_smem, v_base, start_n, seq_len, stride_vbs,
    BLOCK_N: gl.constexpr, BLOCK_DV: gl.constexpr,
    v_async_layout: gl.constexpr,
):
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    v_offsets = ((start_n + v_offs_n[:, None]) * stride_vbs + v_offs_d[None, :]).to(tl.int32)
    v_mask = (start_n + v_offs_n[:, None]) < seq_len
    cdna4_async.buffer_load_to_shared(
        v_smem, v_base, v_offsets, mask=v_mask, other=0.0
    )
    cdna4_async.commit_group()


# ===-----------------------------------------------------------------------===#
# Extend inner loops (BF16 -- reused for extend phase after prefix)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _extend_pipelined(
    acc, l_i, m_i, q_dot, qpe_dot,
    k_ext_base, v_ext_base,
    cur_block_m, seq_len_extend, seq_len_prefix,
    stride_kbs, stride_vbs,
    block_start, block_end,
    kt_smem, kpe_smem, v_smem,
    qk_scale, LOGIT_CAP: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr,
    Mask, mask_base_idx, mask_row_stride,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr, BLOCK_DPE: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    NUM_STAGES: gl.constexpr, STREAMS: gl.constexpr,
    kt_async_layout: gl.constexpr, kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
):
    for _s in gl.static_range(NUM_STAGES):
        issue_n = (block_start + _s) * BLOCK_N
        _issue_async_k_extend(
            kt_smem.index(_s), k_ext_base, issue_n, seq_len_extend,
            stride_kbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
        )
        _issue_async_kpe_extend(
            kpe_smem.index(_s), k_ext_base, issue_n, seq_len_extend,
            stride_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kpe_async_layout,
        )
        _issue_async_v_extend(
            v_smem.index(_s), v_ext_base, issue_n, seq_len_extend,
            stride_vbs, BLOCK_N, BLOCK_DV, v_async_layout,
        )

    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    main_end = block_end - block_start - NUM_STAGES
    cdna4_async.wait_group(WAIT_K)

    for iter_n in tl.range(0, main_end):
        stage = (iter_n % NUM_STAGES).to(tl.int32)
        start_n = ((block_start + iter_n) * BLOCK_N).to(tl.int32)
        future_n = ((block_start + iter_n + NUM_STAGES) * BLOCK_N).to(tl.int32)

        kt_d = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt_d, qk)

        kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        cdna4_async.wait_group(WAIT_V)
        v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)

        _issue_async_k_extend(
            kt_smem.index(stage), k_ext_base, future_n, seq_len_extend,
            stride_kbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
        )
        _issue_async_kpe_extend(
            kpe_smem.index(stage), k_ext_base, future_n, seq_len_extend,
            stride_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kpe_async_layout,
        )
        _issue_async_v_extend(
            v_smem.index(stage), v_ext_base, future_n, seq_len_extend,
            stride_vbs, BLOCK_N, BLOCK_DV, v_async_layout,
        )

        acc, l_i, m_i, p = _softmax_extend(
            acc, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )

        p_cast = p.to(v_d.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)
        acc = mfma_cdna4(p_d, v_d, acc)

        cdna4_async.wait_group(WAIT_K)

    cdna4_async.wait_group(0)
    for tail_i in gl.static_range(NUM_STAGES):
        stage = ((main_end + tail_i) % NUM_STAGES).to(tl.int32)
        start_n = (block_start + main_end + tail_i) * BLOCK_N

        kt_d = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt_d, qk)
        kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        acc, l_i, m_i, p = _softmax_extend(
            acc, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )
        v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
        p_cast = p.to(v_d.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)
        acc = mfma_cdna4(p_d, v_d, acc)

    return acc, l_i, m_i


@gluon.jit
def _extend_serial(
    acc, l_i, m_i, q_dot, qpe_dot,
    k_ext_base, v_ext_base,
    cur_block_m, seq_len_extend, seq_len_prefix,
    stride_kbs, stride_vbs,
    block_start, block_end,
    kt_smem, kpe_smem, v_smem,
    qk_scale, LOGIT_CAP: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr,
    Mask, mask_base_idx, mask_row_stride,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr, BLOCK_DPE: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    kt_async_layout: gl.constexpr, kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
):
    cdna4_async.wait_group(0)
    n_local = block_end - block_start
    for local_i in tl.range(0, n_local):
        start_n = ((block_start + local_i) * BLOCK_N).to(tl.int32)

        _issue_async_k_extend(
            kt_smem.index(0), k_ext_base, start_n, seq_len_extend,
            stride_kbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
        )
        _issue_async_kpe_extend(
            kpe_smem.index(0), k_ext_base, start_n, seq_len_extend,
            stride_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kpe_async_layout,
        )
        _issue_async_v_extend(
            v_smem.index(0), v_ext_base, start_n, seq_len_extend,
            stride_vbs, BLOCK_N, BLOCK_DV, v_async_layout,
        )
        cdna4_async.wait_group(0)

        kt_d = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt_d, qk)
        kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(0), kt_dot_layout)
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        acc, l_i, m_i, p = _softmax_extend(
            acc, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )
        v_d = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
        p_cast = p.to(v_d.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)
        acc = mfma_cdna4(p_d, v_d, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# GQA Extend inner loops (BF16 -- using kv_indices for unified KV buffer)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _gqa_extend_pipelined(
    acc, l_i, m_i, q_dot, qpe_dot,
    kv_base, kv_indices, kv_start, ext_offset,
    cur_block_m, seq_len_extend, seq_len_prefix,
    stride_kvbs,
    block_start, block_end,
    kt_smem, kpe_smem, v_smem,
    Mask,
    qk_scale, LOGIT_CAP: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr, BLOCK_DPE: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    NUM_STAGES: gl.constexpr, STREAMS: gl.constexpr,
    kt_async_layout: gl.constexpr, kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
):
    for _s in gl.static_range(NUM_STAGES):
        kv_locs, mask_n = _load_prefix_kv_locs(
            kv_indices, kv_start, (block_start + _s) * BLOCK_N + ext_offset,
            seq_len_prefix + seq_len_extend,
            BLOCK_N, kt_async_layout,
        )
        mask_n_kpe = gl.convert_layout(mask_n, gl.SliceLayout(dim=0, parent=kpe_async_layout))
        mask_n_v = gl.convert_layout(mask_n, gl.SliceLayout(dim=1, parent=v_async_layout))
        _issue_prefix_kvkpe_async(
            kt_smem.index(_s), kpe_smem.index(_s), kv_base,
            kv_locs, mask_n, mask_n_kpe,
            stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
            kt_async_layout, kpe_async_layout,
        )
        _issue_prefix_v_async(
            v_smem.index(_s), kv_base,
            kv_locs, mask_n_v,
            stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
        )

    main_end = block_end - block_start - NUM_STAGES
    cdna4_async.wait_group(STREAMS)

    for iter_n in tl.range(0, main_end):
        stage = (iter_n % NUM_STAGES).to(tl.int32)
        start_n = ((block_start + iter_n) * BLOCK_N).to(tl.int32)
        future_abs = (block_start + iter_n + NUM_STAGES) * BLOCK_N + ext_offset

        kv_locs, mask_n = _load_prefix_kv_locs(
            kv_indices, kv_start, future_abs,
            seq_len_prefix + seq_len_extend,
            BLOCK_N, kt_async_layout,
        )
        mask_n_kpe = gl.convert_layout(mask_n, gl.SliceLayout(dim=0, parent=kpe_async_layout))
        mask_n_v = gl.convert_layout(mask_n, gl.SliceLayout(dim=1, parent=v_async_layout))

        kt_d = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
        _issue_prefix_k_async(
            kt_smem.index(stage), kv_base,
            kv_locs, mask_n,
            stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt_d, qk)

        kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
        _issue_prefix_kpe_async(
            kpe_smem.index(stage), kv_base,
            kv_locs, mask_n_kpe,
            stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kpe_async_layout,
        )
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        acc, l_i, m_i, p = _softmax_extend(
            acc, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            False, Mask,
            tl.cast(0, tl.int64), tl.cast(0, tl.int64),
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )

        v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
        _issue_prefix_v_async(
            v_smem.index(stage), kv_base,
            kv_locs, mask_n_v,
            stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
        )
        p_cast = p.to(v_d.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)
        acc = mfma_cdna4(p_d, v_d, acc)

        cdna4_async.wait_group(STREAMS)

    cdna4_async.wait_group(0)
    for tail_i in gl.static_range(NUM_STAGES):
        stage = ((main_end + tail_i) % NUM_STAGES).to(tl.int32)
        start_n = (block_start + main_end + tail_i) * BLOCK_N

        kt_d = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt_d, qk)
        kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        acc, l_i, m_i, p = _softmax_extend(
            acc, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            False, Mask,
            tl.cast(0, tl.int64), tl.cast(0, tl.int64),
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )
        v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
        p_cast = p.to(v_d.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)
        acc = mfma_cdna4(p_d, v_d, acc)

    return acc, l_i, m_i


@gluon.jit
def _gqa_extend_serial(
    acc, l_i, m_i, q_dot, qpe_dot,
    kv_base, kv_indices, kv_start, ext_offset,
    cur_block_m, seq_len_extend, seq_len_prefix,
    stride_kvbs,
    block_start, block_end,
    kt_smem, kpe_smem, v_smem,
    Mask,
    qk_scale, LOGIT_CAP: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr, BLOCK_DPE: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    kt_async_layout: gl.constexpr, kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
):
    cdna4_async.wait_group(0)
    n_local = block_end - block_start
    for local_i in tl.range(0, n_local):
        start_n = ((block_start + local_i) * BLOCK_N).to(tl.int32)
        abs_n = start_n + ext_offset

        kv_locs, mask_n = _load_prefix_kv_locs(
            kv_indices, kv_start, abs_n,
            seq_len_prefix + seq_len_extend,
            BLOCK_N, kt_async_layout,
        )
        mask_n_kpe = gl.convert_layout(mask_n, gl.SliceLayout(dim=0, parent=kpe_async_layout))
        mask_n_v = gl.convert_layout(mask_n, gl.SliceLayout(dim=1, parent=v_async_layout))

        _issue_prefix_kvkpe_async(
            kt_smem.index(0), kpe_smem.index(0), kv_base,
            kv_locs, mask_n, mask_n_kpe,
            stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
            kt_async_layout, kpe_async_layout,
        )
        _issue_prefix_v_async(
            v_smem.index(0), kv_base,
            kv_locs, mask_n_v,
            stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
        )
        cdna4_async.wait_group(0)

        kt_d = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt_d, qk)
        kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(0), kt_dot_layout)
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        acc, l_i, m_i, p = _softmax_extend(
            acc, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            False, Mask,
            tl.cast(0, tl.int64), tl.cast(0, tl.int64),
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )
        v_d = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
        p_cast = p.to(v_d.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)
        acc = mfma_cdna4(p_d, v_d, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# FP8 4-Warp Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def mla_d512_gqa_fwd_fp8(
    Q,
    KV_Buffer,
    O,
    qo_indptr,
    kv_indptr,
    kv_indices,
    Mask,
    MaskIndptr,
    sm_scale,
    stride_qbs,
    stride_qh,
    stride_kvbs,
    stride_kvh,
    stride_obs,
    stride_oh,
    IS_CAUSAL: gl.constexpr,
    LOGIT_CAP: gl.constexpr,
    GQA_RATIO: gl.constexpr,
    V_SCALE: gl.constexpr = 1.0,
    NUM_PFX_STAGES: gl.constexpr = 3,
    USE_CUSTOM_MASK: gl.constexpr = False,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr = True,
):
    """MLA D512 FP8 KV prefill kernel with unified KV buffer.

    Grid: (batch, num_q_heads, m_blocks).
    KV_Buffer: [total_kv, num_kv_heads, 576] in fp8e4m3.
    Prefix phase: FP8 MFMA with K/V async pipeline; KPE from global.
    Extend phase: BF16 MFMA with 2-stage pipeline (Q/K/V extend are bf16).
    k_scale is folded into sm_scale by the wrapper.
    V_SCALE applied after final normalization.
    """
    BLOCK_M: gl.constexpr = 64
    BLOCK_N: gl.constexpr = 32
    BLOCK_DMODEL: gl.constexpr = 512
    BLOCK_DPE: gl.constexpr = 64
    BLOCK_DV: gl.constexpr = 512
    EXT_NUM_STAGES: gl.constexpr = 2
    num_warps: gl.constexpr = gl.num_warps()

    MMA_INSTR_M: gl.constexpr = 16
    MMA_INSTR_N: gl.constexpr = 16
    MMA_INSTR_K: gl.constexpr = 32
    BF16_QK_K_WIDTH: gl.constexpr = 8
    BF16_PV_K_WIDTH: gl.constexpr = 4
    FP8_QK_K_WIDTH: gl.constexpr = 16
    FP8_PV_K_WIDTH: gl.constexpr = 8
    ASYNC_PAD_K: gl.constexpr = 8
    ASYNC_PAD_V: gl.constexpr = 32

    cur_seq = gl.program_id(0)
    cur_q_head = gl.program_id(1)
    cur_block_m = gl.program_id(2)
    cur_kv_head = cur_q_head // GQA_RATIO

    cur_seq_q_start = gl.load(qo_indptr + cur_seq)
    seq_len_extend = gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start
    cur_seq_kv_start = gl.load(kv_indptr + cur_seq)
    seq_len_kv = gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start

    if cur_block_m * BLOCK_M >= seq_len_extend:
        return

    seq_len_prefix = seq_len_kv - seq_len_extend
    if USE_CUSTOM_MASK:
        mask_base_idx = gl.load(MaskIndptr + cur_seq).to(tl.int64)
        mask_row_stride = seq_len_kv.to(tl.int64)
    else:
        mask_base_idx = tl.cast(0, tl.int64)
        mask_row_stride = tl.cast(0, tl.int64)

    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    # BF16 dot layouts (extend phase)
    bf16_q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=BF16_QK_K_WIDTH
    )
    bf16_kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=BF16_QK_K_WIDTH
    )
    bf16_p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=BF16_PV_K_WIDTH
    )
    bf16_v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=BF16_PV_K_WIDTH
    )

    # FP8 dot layouts (prefix phase -- k_width doubled for 1-byte elements)
    fp8_q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=FP8_QK_K_WIDTH
    )
    fp8_kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=FP8_QK_K_WIDTH
    )
    fp8_p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=FP8_PV_K_WIDTH
    )
    fp8_v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=FP8_PV_K_WIDTH
    )

    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    # FP8 DLLs for D512 (vec16 fp8 = 128-bit for direct-to-LDS)
    # Kt: 4 contiguous D-bits in regs (vec16), 5 D-bits in lanes, 2 N-bits in warps
    kt_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 16]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    kt_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
        [0, 16],
        [0, 1], [0, 2], [0, 4], [0, 8],
    ]

    # V: 4 contiguous DV-bits in regs (vec16), 5 DV-bits in lanes, 2 N-bits in warps
    v_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_N, BLOCK_DV],
    )
    v_offset_bases: gl.constexpr = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
        [16, 0],
        [1, 0], [2, 0], [4, 0], [8, 0],
    ]

    # --- FP8 prefix SMEM (K, V, and KPE super-tile) ---
    fp8_kt_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[1024, ASYNC_PAD_K], [2048, 32]],
        offset_bases=kt_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    fp8_v_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[1024, ASYNC_PAD_V], [2048, 32]],
        offset_bases=v_offset_bases,
        cga_layout=[],
        shape=[BLOCK_N, BLOCK_DV],
    )

    # KPE super-tile: [64, 64] = 2 N-blocks, 128-bit DMA (vec16 FP8)
    BLOCK_N2: gl.constexpr = 64
    kpe_super_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0]],
        lane_bases=[[16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 32]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[], shape=[BLOCK_DPE, BLOCK_N2],
    )
    kpe_super_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        [0, 4], [0, 8], [0, 16], [0, 32],
        [0, 1], [0, 2],
    ]
    fp8_kpe_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[1024, ASYNC_PAD_K], [2048, 32]],
        offset_bases=kpe_super_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DPE, BLOCK_N2],
    )

    PFX_SMEM_TY: gl.constexpr = KV_Buffer.dtype.element_ty

    kt_smem = gl.allocate_shared_memory(
        PFX_SMEM_TY,
        [NUM_PFX_STAGES, BLOCK_DMODEL, BLOCK_N],
        layout=fp8_kt_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        PFX_SMEM_TY,
        [NUM_PFX_STAGES, BLOCK_N, BLOCK_DV],
        layout=fp8_v_smem_layout,
    )
    NUM_KPE_STAGES: gl.constexpr = 2
    kpe_smem = gl.allocate_shared_memory(
        PFX_SMEM_TY,
        [NUM_KPE_STAGES, BLOCK_DPE, BLOCK_N2],
        layout=fp8_kpe_smem_layout,
    )

    for _s in gl.static_range(NUM_PFX_STAGES):
        v_zero = gl.zeros(
            [BLOCK_N, BLOCK_DV],
            dtype=PFX_SMEM_TY,
            layout=v_async_layout,
        )
        v_smem.index(_s).store(v_zero)
    gl.barrier()

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=offs_d_layout)
    offs_dpe = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=offs_d_layout)

    q_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend

    qk_scale = sm_scale * LOG2E
    kv_base = KV_Buffer + cur_kv_head * stride_kvh

    if IS_CAUSAL:
        causal_kv_end = seq_len_kv - seq_len_extend + tl.minimum(seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    else:
        causal_kv_end = seq_len_kv
    n_kv_blocks = (causal_kv_end + BLOCK_N - 1) // BLOCK_N

    q_abs_pos = (
        (seq_len_kv - seq_len_extend)
        + cur_block_m * BLOCK_M
        + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    )

    q_m_idx = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    PAIR_WAIT_READY: gl.constexpr = 3
    PAIR_WAIT_EPILOGUE_ODD: gl.constexpr = 2

    q_base = (
        Q
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_q_head * stride_qh
    )
    q_reg = gl.load(q_base + offs_d[None, :], mask=q_mask, other=0.0)
    qpe_reg = gl.load(q_base + offs_dpe[None, :], mask=q_mask, other=0.0)

    # FP8 Q for prefix MFMA
    fp8_q_dot = gl.convert_layout(q_reg.to(tl.float8e4nv), fp8_q_dot_layout)
    fp8_qpe_dot = gl.convert_layout(qpe_reg.to(tl.float8e4nv), fp8_q_dot_layout)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

    # ===--- PREFIX PHASE (FP8 MFMA, NUM_PFX_STAGES-deep pipeline) ---===
    if n_kv_blocks <= 0:
        pass
    elif n_kv_blocks <= NUM_PFX_STAGES:
        for tail_i in gl.static_range(NUM_PFX_STAGES):
            kv_locs_t, mask_n_t = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, tail_i * BLOCK_N, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_v_t = gl.convert_layout(mask_n_t, gl.SliceLayout(dim=1, parent=v_async_layout))
            _issue_prefix_k_async(
                kt_smem.index(tail_i), kv_base,
                kv_locs_t, mask_n_t,
                stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(tail_i), kv_base,
                kv_locs_t, mask_n_v_t,
                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

        cdna4_async.wait_group(0)

        for tail_i in gl.static_range(NUM_PFX_STAGES):
            start_n_t = tail_i * BLOCK_N
            if tail_i < n_kv_blocks:
                kt_d_t = cdna4_async.load_shared_relaxed(kt_smem.index(tail_i), fp8_kt_dot_layout)
                qk_t = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk_t = do_mma(fp8_q_dot, kt_d_t, qk_t)
                qk_t = _load_kpe_from_global_fp8(
                    qk_t, fp8_qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                    start_n_t, causal_kv_end, stride_kvbs,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, fp8_kt_dot_layout,
                )

                qk_s_t = qk_t * qk_scale
                if LOGIT_CAP > 0:
                    log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
                    inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
                    e_neg_t = tl.math.exp2(-qk_s_t * inv_cap)
                    sig_t = 1.0 / (1.0 + e_neg_t)
                    qk_s_t = log2_cap * (2.0 * sig_t - 1.0)
                n_offs_t = start_n_t + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                valid_t = n_offs_t[None, :] < causal_kv_end
                if IS_CAUSAL:
                    valid_t = valid_t & (q_abs_pos[:, None] >= n_offs_t[None, :])
                if USE_CUSTOM_MASK:
                    q_m_idx_t = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
                    is_prefix_t = n_offs_t[None, :] < seq_len_prefix
                    need_mask_t = (~is_prefix_t) | (is_prefix_t & (not SKIP_PREFIX_CUSTOM_MASK))
                    mask_ptrs_t = (
                        Mask + mask_base_idx
                        + q_m_idx_t[:, None] * mask_row_stride
                        + n_offs_t[None, :].to(tl.int64)
                    )
                    mask_vals_t = gl.load(mask_ptrs_t, mask=valid_t & need_mask_t, other=1)
                    valid_t = valid_t & gl.where(need_mask_t, mask_vals_t != 0, True)
                qk_s_t = gl.where(valid_t, qk_s_t, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

                m_ij_t = nan_propagating_max(qk_s_t, axis=1)
                m_new_t = gl.maximum(m_i, m_ij_t, propagate_nan=tl.PropagateNan.ALL)
                p_t = gl.exp2(qk_s_t - m_new_t[:, None])
                l_ij_t = gl.sum(p_t, axis=1)
                alpha_t = gl.exp2(m_i - m_new_t)
                l_i = l_i * alpha_t + l_ij_t
                acc = acc * alpha_t[:, None]
                m_i = m_new_t

                v_d_t = cdna4_async.load_shared_relaxed(v_smem.index(tail_i), fp8_v_dot_layout)
                p_cast_t = p_t.to(v_d_t.dtype)
                p_d_t = gl.convert_layout(p_cast_t, fp8_p_dot_layout)
                acc = do_mma(p_d_t, v_d_t, acc)
    else:
        kv_locs_0, mask_n_0 = _load_prefix_kv_locs(
            kv_indices, cur_seq_kv_start, 0, causal_kv_end,
            BLOCK_N, kt_async_layout,
        )
        _issue_prefix_kv_async(
            kt_smem.index(0), v_smem.index(0), kv_base,
            kv_locs_0, mask_n_0,
            stride_kvbs, stride_kvbs,
            BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
            kt_async_layout, v_async_layout,
        )
        _issue_kpe_super_async(
            kpe_smem.index(0), kv_base, kv_indices, cur_seq_kv_start, 0, causal_kv_end,
            stride_kvbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout,
        )

        kv_locs_1, mask_n_1 = _load_prefix_kv_locs(
            kv_indices, cur_seq_kv_start, BLOCK_N, causal_kv_end,
            BLOCK_N, kt_async_layout,
        )
        _issue_prefix_kv_async(
            kt_smem.index(1), v_smem.index(1), kv_base,
            kv_locs_1, mask_n_1,
            stride_kvbs, stride_kvbs,
            BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
            kt_async_layout, v_async_layout,
        )
        _issue_kpe_super_async(
            kpe_smem.index(1), kv_base, kv_indices, cur_seq_kv_start, 2 * BLOCK_N, causal_kv_end,
            stride_kvbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout,
        )

        kv_locs_2, mask_n_2 = _load_prefix_kv_locs(
            kv_indices, cur_seq_kv_start, 2 * BLOCK_N, causal_kv_end,
            BLOCK_N, kt_async_layout,
        )
        _issue_prefix_kv_async(
            kt_smem.index(2), v_smem.index(2), kv_base,
            kv_locs_2, mask_n_2,
            stride_kvbs, stride_kvbs,
            BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
            kt_async_layout, v_async_layout,
        )

        n_pair_blocks = n_kv_blocks // 2
        main_pair_count = n_pair_blocks - 1

        for pair_i in tl.range(0, main_pair_count):
            even_block = (pair_i * 2).to(tl.int32)
            odd_block = even_block + 1
            even_stage = (even_block % NUM_PFX_STAGES).to(tl.int32)
            odd_stage = (odd_block % NUM_PFX_STAGES).to(tl.int32)
            kpe_stage = (pair_i % NUM_KPE_STAGES).to(tl.int32)

            cdna4_async.wait_group(PAIR_WAIT_READY)
            kt_even = cdna4_async.load_shared_relaxed(kt_smem.index(even_stage), fp8_kt_dot_layout)
            kpe_lo = _load_kpe_super_half(
                kpe_smem.index(kpe_stage), 0, BLOCK_N, fp8_kt_dot_layout,
            )
            qk_even = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk_even = do_mma(fp8_q_dot, kt_even, qk_even)
            qk_even = do_mma(fp8_qpe_dot, kpe_lo, qk_even)
            v_even = cdna4_async.load_shared_relaxed(v_smem.index(even_stage), fp8_v_dot_layout)

            future_odd_start = (even_block + NUM_PFX_STAGES) * BLOCK_N
            kv_locs_odd_next, mask_n_odd_next = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, future_odd_start, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            _issue_prefix_kv_async(
                kt_smem.index(even_stage), v_smem.index(even_stage), kv_base,
                kv_locs_odd_next, mask_n_odd_next,
                stride_kvbs, stride_kvbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
                kt_async_layout, v_async_layout,
            )

            acc, l_i, m_i, p_even = _softmax_prefill_fp8(
                acc, l_i, m_i, qk_even, even_block * BLOCK_N, causal_kv_end, q_abs_pos,
                qk_scale, LOGIT_CAP, IS_CAUSAL,
                USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
                Mask, mask_base_idx, mask_row_stride,
                q_m_idx, seq_len_prefix,
                BLOCK_N, mma_layout, mma_offs_n_col,
            )
            p_cast_even = p_even.to(v_even.dtype)
            p_d_even = gl.convert_layout(p_cast_even, fp8_p_dot_layout)
            acc = do_mma(p_d_even, v_even, acc)

            cdna4_async.wait_group(PAIR_WAIT_READY)
            kt_odd = cdna4_async.load_shared_relaxed(kt_smem.index(odd_stage), fp8_kt_dot_layout)
            kpe_hi = _load_kpe_super_half(
                kpe_smem.index(kpe_stage), BLOCK_N, BLOCK_N, fp8_kt_dot_layout,
            )
            qk_odd = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk_odd = do_mma(fp8_q_dot, kt_odd, qk_odd)
            qk_odd = do_mma(fp8_qpe_dot, kpe_hi, qk_odd)
            v_odd = cdna4_async.load_shared_relaxed(v_smem.index(odd_stage), fp8_v_dot_layout)

            future_pair_start = (even_block + 2 * NUM_KPE_STAGES) * BLOCK_N
            _issue_kpe_super_async(
                kpe_smem.index(kpe_stage), kv_base, kv_indices, cur_seq_kv_start,
                future_pair_start, causal_kv_end,
                stride_kvbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout,
            )
            future_even_start = (odd_block + NUM_PFX_STAGES) * BLOCK_N
            kv_locs_even_next, mask_n_even_next = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, future_even_start, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            _issue_prefix_kv_async(
                kt_smem.index(odd_stage), v_smem.index(odd_stage), kv_base,
                kv_locs_even_next, mask_n_even_next,
                stride_kvbs, stride_kvbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
                kt_async_layout, v_async_layout,
            )

            acc, l_i, m_i, p_odd = _softmax_prefill_fp8(
                acc, l_i, m_i, qk_odd, odd_block * BLOCK_N, causal_kv_end, q_abs_pos,
                qk_scale, LOGIT_CAP, IS_CAUSAL,
                USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
                Mask, mask_base_idx, mask_row_stride,
                q_m_idx, seq_len_prefix,
                BLOCK_N, mma_layout, mma_offs_n_col,
            )
            p_cast_odd = p_odd.to(v_odd.dtype)
            p_d_odd = gl.convert_layout(p_cast_odd, fp8_p_dot_layout)
            acc = do_mma(p_d_odd, v_odd, acc)

        final_pair = (n_pair_blocks - 1).to(tl.int32)
        final_even_block = final_pair * 2
        final_odd_block = final_even_block + 1
        final_even_stage = (final_even_block % NUM_PFX_STAGES).to(tl.int32)
        final_odd_stage = (final_odd_block % NUM_PFX_STAGES).to(tl.int32)
        final_kpe_stage = (final_pair % NUM_KPE_STAGES).to(tl.int32)

        cdna4_async.wait_group(PAIR_WAIT_READY)
        kt_even_tail = cdna4_async.load_shared_relaxed(kt_smem.index(final_even_stage), fp8_kt_dot_layout)
        kpe_lo_tail = _load_kpe_super_half(
            kpe_smem.index(final_kpe_stage), 0, BLOCK_N, fp8_kt_dot_layout,
        )
        qk_even_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk_even_tail = do_mma(fp8_q_dot, kt_even_tail, qk_even_tail)
        qk_even_tail = do_mma(fp8_qpe_dot, kpe_lo_tail, qk_even_tail)
        v_even_tail = cdna4_async.load_shared_relaxed(v_smem.index(final_even_stage), fp8_v_dot_layout)
        acc, l_i, m_i, p_even_tail = _softmax_prefill_fp8(
            acc, l_i, m_i, qk_even_tail, final_even_block * BLOCK_N, causal_kv_end, q_abs_pos,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
            Mask, mask_base_idx, mask_row_stride,
            q_m_idx, seq_len_prefix,
            BLOCK_N, mma_layout, mma_offs_n_col,
        )
        p_cast_even_tail = p_even_tail.to(v_even_tail.dtype)
        p_d_even_tail = gl.convert_layout(p_cast_even_tail, fp8_p_dot_layout)
        acc = do_mma(p_d_even_tail, v_even_tail, acc)

        cdna4_async.wait_group(PAIR_WAIT_EPILOGUE_ODD)
        kt_odd_tail = cdna4_async.load_shared_relaxed(kt_smem.index(final_odd_stage), fp8_kt_dot_layout)
        kpe_hi_tail = _load_kpe_super_half(
            kpe_smem.index(final_kpe_stage), BLOCK_N, BLOCK_N, fp8_kt_dot_layout,
        )
        qk_odd_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk_odd_tail = do_mma(fp8_q_dot, kt_odd_tail, qk_odd_tail)
        qk_odd_tail = do_mma(fp8_qpe_dot, kpe_hi_tail, qk_odd_tail)
        v_odd_tail = cdna4_async.load_shared_relaxed(v_smem.index(final_odd_stage), fp8_v_dot_layout)
        acc, l_i, m_i, p_odd_tail = _softmax_prefill_fp8(
            acc, l_i, m_i, qk_odd_tail, final_odd_block * BLOCK_N, causal_kv_end, q_abs_pos,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
            Mask, mask_base_idx, mask_row_stride,
            q_m_idx, seq_len_prefix,
            BLOCK_N, mma_layout, mma_offs_n_col,
        )
        p_cast_odd_tail = p_odd_tail.to(v_odd_tail.dtype)
        p_d_odd_tail = gl.convert_layout(p_cast_odd_tail, fp8_p_dot_layout)
        acc = do_mma(p_d_odd_tail, v_odd_tail, acc)

        if (n_kv_blocks % 2) != 0:
            tail_block = n_pair_blocks * 2
            tail_stage = (tail_block % NUM_PFX_STAGES).to(tl.int32)

            cdna4_async.wait_group(0)
            kt_tail = cdna4_async.load_shared_relaxed(kt_smem.index(tail_stage), fp8_kt_dot_layout)
            qk_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk_tail = do_mma(fp8_q_dot, kt_tail, qk_tail)
            qk_tail = _load_kpe_from_global_fp8(
                qk_tail, fp8_qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                tail_block * BLOCK_N, causal_kv_end, stride_kvbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, fp8_kt_dot_layout,
            )
            v_tail = cdna4_async.load_shared_relaxed(v_smem.index(tail_stage), fp8_v_dot_layout)
            acc, l_i, m_i, p_tail = _softmax_prefill_fp8(
                acc, l_i, m_i, qk_tail, tail_block * BLOCK_N, causal_kv_end, q_abs_pos,
                qk_scale, LOGIT_CAP, IS_CAUSAL,
                USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
                Mask, mask_base_idx, mask_row_stride,
                q_m_idx, seq_len_prefix,
                BLOCK_N, mma_layout, mma_offs_n_col,
            )
            p_cast_tail = p_tail.to(v_tail.dtype)
            p_d_tail = gl.convert_layout(p_cast_tail, fp8_p_dot_layout)
            acc = do_mma(p_d_tail, v_tail, acc)

    # ===--- EXTEND PHASE (BF16 MFMA, 2-stage pipeline) ---===
    # Transition FP8 prefix SMEM -> BF16 extend SMEM
    # (extend Q/K/V are always bf16)
    # TODO: extend phase -- currently this kernel only handles prefix-only (prefill).
    # When used with extend tokens, SMEM must be reallocated with bf16 dtype
    # and extend inner loops called with bf16 layouts.
    # For now, the MLA prefill path (mla_d512_gqa_attention_fwd) only processes
    # unified KV buffer where all tokens go through the prefix path.

    if V_SCALE != 1.0:
        acc = acc * V_SCALE

    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]
    out_bf16 = gl.convert_layout(acc, blocked_layout).to(O.dtype.element_ty)
    o_base = (
        O
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_q_head * stride_oh
    )
    o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    gl.store(o_base + offs_dv[None, :], out_bf16, mask=o_mask)


# ===-----------------------------------------------------------------------===#
# WCA Persistent + Split-K Kernel (FP8 prefix, same inner loops)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def mla_d512_gqa_fwd_wca_fp8(
    Q, KV_Buffer, O,
    qo_indptr, kv_indptr, kv_indices,
    sm_scale,
    stride_qbs, stride_qh, stride_kvbs, stride_kvh, stride_obs, stride_oh,
    partial_out, partial_lse,
    total_valid_tiles, total_programs,
    num_heads, n_m_tiles,
    IS_CAUSAL: gl.constexpr, LOGIT_CAP: gl.constexpr,
    GQA_RATIO: gl.constexpr, SPLIT_K: gl.constexpr,
    V_SCALE: gl.constexpr = 1.0,
    NUM_PFX_STAGES: gl.constexpr = 3,
):
    """Persistent WCA MLA D512 FP8 kernel with optional split-K.

    Same inner loops as mla_d512_gqa_fwd_fp8 but with WCA tile scheduling.
    KPE still loads from global memory here; only the 4w path uses KPE SMEM
    super-tiles today.
    """
    BLOCK_M: gl.constexpr = 64
    BLOCK_N: gl.constexpr = 32
    BLOCK_DMODEL: gl.constexpr = 512
    BLOCK_DPE: gl.constexpr = 64
    BLOCK_DV: gl.constexpr = 512
    num_warps: gl.constexpr = gl.num_warps()

    MMA_INSTR_M: gl.constexpr = 16
    MMA_INSTR_N: gl.constexpr = 16
    MMA_INSTR_K: gl.constexpr = 32
    FP8_QK_K_WIDTH: gl.constexpr = 16
    FP8_PV_K_WIDTH: gl.constexpr = 8
    ASYNC_PAD_K: gl.constexpr = 8
    ASYNC_PAD_V: gl.constexpr = 32

    cta_id = gl.program_id(0)

    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4, instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True, warps_per_cta=[num_warps, 1],
    )
    fp8_q_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=FP8_QK_K_WIDTH)
    fp8_kt_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=FP8_QK_K_WIDTH)
    fp8_p_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=FP8_PV_K_WIDTH)
    fp8_v_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=FP8_PV_K_WIDTH)
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1], order=[1, 0],
    )
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    kt_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[16, 0], [32, 0], [64, 0], [128, 0], [256, 0], [0, 16]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[], shape=[BLOCK_DMODEL, BLOCK_N],
    )
    kt_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
        [0, 16], [0, 1], [0, 2], [0, 4], [0, 8],
    ]
    v_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [16, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[], shape=[BLOCK_N, BLOCK_DV],
    )
    v_offset_bases: gl.constexpr = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
        [16, 0], [1, 0], [2, 0], [4, 0], [8, 0],
    ]

    BLOCK_N2: gl.constexpr = 64
    kpe_super_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [8, 0]],
        lane_bases=[[16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 32]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[], shape=[BLOCK_DPE, BLOCK_N2],
    )
    kpe_super_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        [0, 4], [0, 8], [0, 16], [0, 32],
        [0, 1], [0, 2],
    ]

    fp8_kt_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[1024, ASYNC_PAD_K], [2048, 32]],
        offset_bases=kt_offset_bases, cga_layout=[], shape=[BLOCK_DMODEL, BLOCK_N],
    )
    fp8_v_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[1024, ASYNC_PAD_V], [2048, 32]],
        offset_bases=v_offset_bases, cga_layout=[], shape=[BLOCK_N, BLOCK_DV],
    )
    fp8_kpe_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[1024, ASYNC_PAD_K], [2048, 32]],
        offset_bases=kpe_super_offset_bases, cga_layout=[], shape=[BLOCK_DPE, BLOCK_N2],
    )

    PFX_SMEM_TY: gl.constexpr = KV_Buffer.dtype.element_ty
    kt_smem = gl.allocate_shared_memory(PFX_SMEM_TY, [NUM_PFX_STAGES, BLOCK_DMODEL, BLOCK_N], layout=fp8_kt_smem_layout)
    v_smem = gl.allocate_shared_memory(PFX_SMEM_TY, [NUM_PFX_STAGES, BLOCK_N, BLOCK_DV], layout=fp8_v_smem_layout)
    NUM_KPE_STAGES: gl.constexpr = 2
    kpe_smem = gl.allocate_shared_memory(PFX_SMEM_TY, [NUM_KPE_STAGES, BLOCK_DPE, BLOCK_N2], layout=fp8_kpe_smem_layout)

    for _s in gl.static_range(NUM_PFX_STAGES):
        v_zero = gl.zeros([BLOCK_N, BLOCK_DV], dtype=PFX_SMEM_TY, layout=v_async_layout)
        v_smem.index(_s).store(v_zero)
    gl.barrier()

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=offs_d_layout)
    offs_dpe = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=offs_d_layout)

    STREAMS: gl.constexpr = 2
    WAIT_K: gl.constexpr = STREAMS * NUM_PFX_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_PFX_STAGES - STREAMS
    PAIR_WAIT_READY: gl.constexpr = 3
    PAIR_WAIT_EPILOGUE_ODD: gl.constexpr = 2

    tile_idx = cta_id
    while tile_idx < total_valid_tiles:
        if SPLIT_K > 1:
            output_tile = tile_idx // SPLIT_K
            k_split_id = tile_idx % SPLIT_K
        else:
            output_tile = tile_idx
            k_split_id = 0

        tiles_per_seq = num_heads * n_m_tiles
        cur_seq = output_tile // tiles_per_seq
        rem = output_tile % tiles_per_seq
        cur_q_head_idx = rem // n_m_tiles
        cur_block_m = rem % n_m_tiles
        cur_kv_head = cur_q_head_idx // GQA_RATIO

        cur_seq_q_start = gl.load(qo_indptr + cur_seq)
        seq_len_extend = gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start
        cur_seq_kv_start = gl.load(kv_indptr + cur_seq)
        seq_len_kv = gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start

        if IS_CAUSAL:
            causal_kv_end = seq_len_kv - seq_len_extend + tl.minimum(seq_len_extend, (cur_block_m + 1) * BLOCK_M)
        else:
            causal_kv_end = seq_len_kv

        if SPLIT_K > 1:
            n_kv_blocks_total = (causal_kv_end + BLOCK_N - 1) // BLOCK_N
            blocks_per_split = (n_kv_blocks_total + SPLIT_K - 1) // SPLIT_K
            my_block_start = k_split_id * blocks_per_split
            my_block_end = tl.minimum((k_split_id + 1) * blocks_per_split, n_kv_blocks_total)
            split_kv_offset = my_block_start * BLOCK_N
            causal_kv_end = tl.minimum(my_block_end * BLOCK_N, causal_kv_end) - split_kv_offset
            causal_kv_end = tl.maximum(causal_kv_end, 0)
        else:
            split_kv_offset = 0

        n_kv_blocks = (causal_kv_end + BLOCK_N - 1) // BLOCK_N
        kv_base = KV_Buffer + cur_kv_head * stride_kvh
        qk_scale = sm_scale * LOG2E

        q_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
        q_base = (
            Q + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
            + cur_q_head_idx * stride_qh
        )
        q_reg = gl.load(q_base + offs_d[None, :], mask=q_mask, other=0.0)
        qpe_reg = gl.load(q_base + offs_dpe[None, :], mask=q_mask, other=0.0)
        fp8_q_dot = gl.convert_layout(q_reg.to(tl.float8e4nv), fp8_q_dot_layout)
        fp8_qpe_dot = gl.convert_layout(qpe_reg.to(tl.float8e4nv), fp8_q_dot_layout)

        q_abs_pos = (
            (seq_len_kv - seq_len_extend)
            + cur_block_m * BLOCK_M
            + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        )

        m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
        l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
        acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

        if n_kv_blocks > 0:
            if n_kv_blocks <= NUM_PFX_STAGES:
                for tail_i in gl.static_range(NUM_PFX_STAGES):
                    kv_locs_t, mask_n_t = _load_prefix_kv_locs(
                        kv_indices, cur_seq_kv_start + split_kv_offset,
                        tail_i * BLOCK_N, causal_kv_end, BLOCK_N, kt_async_layout,
                    )
                    mask_n_v_t = gl.convert_layout(mask_n_t, gl.SliceLayout(dim=1, parent=v_async_layout))
                    _issue_prefix_k_async(
                        kt_smem.index(tail_i), kv_base,
                        kv_locs_t, mask_n_t,
                        stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
                    )
                    _issue_prefix_v_async(
                        v_smem.index(tail_i), kv_base,
                        kv_locs_t, mask_n_v_t,
                        stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
                    )
                cdna4_async.wait_group(0)

                for tail_i in gl.static_range(NUM_PFX_STAGES):
                    start_n_t = tail_i * BLOCK_N
                    if tail_i < n_kv_blocks:
                        kt_d_t = cdna4_async.load_shared_relaxed(kt_smem.index(tail_i), fp8_kt_dot_layout)
                        qk_t = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                        qk_t = do_mma(fp8_q_dot, kt_d_t, qk_t)
                        qk_t = _load_kpe_from_global_fp8(
                            qk_t, fp8_qpe_dot, kv_base, kv_indices,
                            cur_seq_kv_start + split_kv_offset,
                            start_n_t, causal_kv_end, stride_kvbs,
                            BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, fp8_kt_dot_layout,
                        )
                        qk_s_t = qk_t * qk_scale
                        if LOGIT_CAP > 0:
                            log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
                            inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
                            e_neg_t = tl.math.exp2(-qk_s_t * inv_cap)
                            sig_t = 1.0 / (1.0 + e_neg_t)
                            qk_s_t = log2_cap * (2.0 * sig_t - 1.0)
                        abs_n = split_kv_offset + start_n_t + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                        valid_t = (start_n_t + gl.arange(0, BLOCK_N, layout=mma_offs_n_col))[None, :] < causal_kv_end
                        if IS_CAUSAL:
                            valid_t = valid_t & (q_abs_pos[:, None] >= abs_n[None, :])
                        qk_s_t = gl.where(valid_t, qk_s_t, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

                        m_ij_t = nan_propagating_max(qk_s_t, axis=1)
                        m_new_t = gl.maximum(m_i, m_ij_t, propagate_nan=tl.PropagateNan.ALL)
                        p_t = gl.exp2(qk_s_t - m_new_t[:, None])
                        l_ij_t = gl.sum(p_t, axis=1)
                        alpha_t = gl.exp2(m_i - m_new_t)
                        l_i = l_i * alpha_t + l_ij_t
                        acc = acc * alpha_t[:, None]
                        m_i = m_new_t
                        v_d_t = cdna4_async.load_shared_relaxed(v_smem.index(tail_i), fp8_v_dot_layout)
                        p_cast_t = p_t.to(v_d_t.dtype)
                        p_d_t = gl.convert_layout(p_cast_t, fp8_p_dot_layout)
                        acc = do_mma(p_d_t, v_d_t, acc)
            else:
                kv_start_base = cur_seq_kv_start + split_kv_offset
                kv_locs_0, mask_n_0 = _load_prefix_kv_locs(
                    kv_indices, kv_start_base,
                    0, causal_kv_end, BLOCK_N, kt_async_layout,
                )
                _issue_prefix_kv_async(
                    kt_smem.index(0), v_smem.index(0), kv_base,
                    kv_locs_0, mask_n_0,
                    stride_kvbs, stride_kvbs,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
                    kt_async_layout, v_async_layout,
                )
                _issue_kpe_super_async(
                    kpe_smem.index(0), kv_base, kv_indices, kv_start_base, 0, causal_kv_end,
                    stride_kvbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout,
                )

                kv_locs_1, mask_n_1 = _load_prefix_kv_locs(
                    kv_indices, kv_start_base,
                    BLOCK_N, causal_kv_end, BLOCK_N, kt_async_layout,
                )
                _issue_prefix_kv_async(
                    kt_smem.index(1), v_smem.index(1), kv_base,
                    kv_locs_1, mask_n_1,
                    stride_kvbs, stride_kvbs,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
                    kt_async_layout, v_async_layout,
                )
                _issue_kpe_super_async(
                    kpe_smem.index(1), kv_base, kv_indices, kv_start_base, 2 * BLOCK_N, causal_kv_end,
                    stride_kvbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout,
                )

                kv_locs_2, mask_n_2 = _load_prefix_kv_locs(
                    kv_indices, kv_start_base,
                    2 * BLOCK_N, causal_kv_end, BLOCK_N, kt_async_layout,
                )
                _issue_prefix_kv_async(
                    kt_smem.index(2), v_smem.index(2), kv_base,
                    kv_locs_2, mask_n_2,
                    stride_kvbs, stride_kvbs,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
                    kt_async_layout, v_async_layout,
                )

                n_pair_blocks = n_kv_blocks // 2
                main_pair_count = n_pair_blocks - 1

                for pair_i in tl.range(0, main_pair_count):
                    even_block = (pair_i * 2).to(tl.int32)
                    odd_block = even_block + 1
                    even_stage = (even_block % NUM_PFX_STAGES).to(tl.int32)
                    odd_stage = (odd_block % NUM_PFX_STAGES).to(tl.int32)
                    kpe_stage = (pair_i % NUM_KPE_STAGES).to(tl.int32)

                    cdna4_async.wait_group(PAIR_WAIT_READY)
                    kt_even = cdna4_async.load_shared_relaxed(kt_smem.index(even_stage), fp8_kt_dot_layout)
                    kpe_lo = _load_kpe_super_half(
                        kpe_smem.index(kpe_stage), 0, BLOCK_N, fp8_kt_dot_layout,
                    )
                    qk_even = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                    qk_even = do_mma(fp8_q_dot, kt_even, qk_even)
                    qk_even = do_mma(fp8_qpe_dot, kpe_lo, qk_even)
                    v_even = cdna4_async.load_shared_relaxed(v_smem.index(even_stage), fp8_v_dot_layout)

                    future_odd_start = (even_block + NUM_PFX_STAGES) * BLOCK_N
                    kv_locs_odd_next, mask_n_odd_next = _load_prefix_kv_locs(
                        kv_indices, kv_start_base,
                        future_odd_start, causal_kv_end, BLOCK_N, kt_async_layout,
                    )
                    _issue_prefix_kv_async(
                        kt_smem.index(even_stage), v_smem.index(even_stage), kv_base,
                        kv_locs_odd_next, mask_n_odd_next,
                        stride_kvbs, stride_kvbs,
                        BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
                        kt_async_layout, v_async_layout,
                    )

                    qk_even_scaled = qk_even * qk_scale
                    if LOGIT_CAP > 0:
                        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
                        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
                        e_neg_even = tl.math.exp2(-qk_even_scaled * inv_cap)
                        sig_even = 1.0 / (1.0 + e_neg_even)
                        qk_even_scaled = log2_cap * (2.0 * sig_even - 1.0)
                    local_even = even_block * BLOCK_N + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                    abs_even = split_kv_offset + local_even
                    valid_even = local_even[None, :] < causal_kv_end
                    if IS_CAUSAL:
                        valid_even = valid_even & (q_abs_pos[:, None] >= abs_even[None, :])
                    qk_even_scaled = gl.where(
                        valid_even,
                        qk_even_scaled,
                        gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout),
                    )
                    m_even = nan_propagating_max(qk_even_scaled, axis=1)
                    m_new_even = gl.maximum(m_i, m_even, propagate_nan=tl.PropagateNan.ALL)
                    p_even = gl.exp2(qk_even_scaled - m_new_even[:, None])
                    l_even = gl.sum(p_even, axis=1)
                    alpha_even = gl.exp2(m_i - m_new_even)
                    l_i = l_i * alpha_even + l_even
                    acc = acc * alpha_even[:, None]
                    m_i = m_new_even
                    p_cast_even = p_even.to(v_even.dtype)
                    p_d_even = gl.convert_layout(p_cast_even, fp8_p_dot_layout)
                    acc = do_mma(p_d_even, v_even, acc)

                    cdna4_async.wait_group(PAIR_WAIT_READY)
                    kt_odd = cdna4_async.load_shared_relaxed(kt_smem.index(odd_stage), fp8_kt_dot_layout)
                    kpe_hi = _load_kpe_super_half(
                        kpe_smem.index(kpe_stage), BLOCK_N, BLOCK_N, fp8_kt_dot_layout,
                    )
                    qk_odd = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                    qk_odd = do_mma(fp8_q_dot, kt_odd, qk_odd)
                    qk_odd = do_mma(fp8_qpe_dot, kpe_hi, qk_odd)
                    v_odd = cdna4_async.load_shared_relaxed(v_smem.index(odd_stage), fp8_v_dot_layout)

                    future_pair_start = (even_block + 2 * NUM_KPE_STAGES) * BLOCK_N
                    _issue_kpe_super_async(
                        kpe_smem.index(kpe_stage), kv_base, kv_indices, kv_start_base,
                        future_pair_start, causal_kv_end,
                        stride_kvbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout,
                    )
                    future_even_start = (odd_block + NUM_PFX_STAGES) * BLOCK_N
                    kv_locs_even_next, mask_n_even_next = _load_prefix_kv_locs(
                        kv_indices, kv_start_base,
                        future_even_start, causal_kv_end, BLOCK_N, kt_async_layout,
                    )
                    _issue_prefix_kv_async(
                        kt_smem.index(odd_stage), v_smem.index(odd_stage), kv_base,
                        kv_locs_even_next, mask_n_even_next,
                        stride_kvbs, stride_kvbs,
                        BLOCK_N, BLOCK_DMODEL, BLOCK_DV,
                        kt_async_layout, v_async_layout,
                    )

                    qk_odd_scaled = qk_odd * qk_scale
                    if LOGIT_CAP > 0:
                        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
                        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
                        e_neg_odd = tl.math.exp2(-qk_odd_scaled * inv_cap)
                        sig_odd = 1.0 / (1.0 + e_neg_odd)
                        qk_odd_scaled = log2_cap * (2.0 * sig_odd - 1.0)
                    local_odd = odd_block * BLOCK_N + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                    abs_odd = split_kv_offset + local_odd
                    valid_odd = local_odd[None, :] < causal_kv_end
                    if IS_CAUSAL:
                        valid_odd = valid_odd & (q_abs_pos[:, None] >= abs_odd[None, :])
                    qk_odd_scaled = gl.where(
                        valid_odd,
                        qk_odd_scaled,
                        gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout),
                    )
                    m_odd = nan_propagating_max(qk_odd_scaled, axis=1)
                    m_new_odd = gl.maximum(m_i, m_odd, propagate_nan=tl.PropagateNan.ALL)
                    p_odd = gl.exp2(qk_odd_scaled - m_new_odd[:, None])
                    l_odd = gl.sum(p_odd, axis=1)
                    alpha_odd = gl.exp2(m_i - m_new_odd)
                    l_i = l_i * alpha_odd + l_odd
                    acc = acc * alpha_odd[:, None]
                    m_i = m_new_odd
                    p_cast_odd = p_odd.to(v_odd.dtype)
                    p_d_odd = gl.convert_layout(p_cast_odd, fp8_p_dot_layout)
                    acc = do_mma(p_d_odd, v_odd, acc)

                final_pair = (n_pair_blocks - 1).to(tl.int32)
                final_even_block = final_pair * 2
                final_odd_block = final_even_block + 1
                final_even_stage = (final_even_block % NUM_PFX_STAGES).to(tl.int32)
                final_odd_stage = (final_odd_block % NUM_PFX_STAGES).to(tl.int32)
                final_kpe_stage = (final_pair % NUM_KPE_STAGES).to(tl.int32)

                cdna4_async.wait_group(PAIR_WAIT_READY)
                kt_even_tail = cdna4_async.load_shared_relaxed(kt_smem.index(final_even_stage), fp8_kt_dot_layout)
                kpe_lo_tail = _load_kpe_super_half(
                    kpe_smem.index(final_kpe_stage), 0, BLOCK_N, fp8_kt_dot_layout,
                )
                qk_even_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk_even_tail = do_mma(fp8_q_dot, kt_even_tail, qk_even_tail)
                qk_even_tail = do_mma(fp8_qpe_dot, kpe_lo_tail, qk_even_tail)
                v_even_tail = cdna4_async.load_shared_relaxed(v_smem.index(final_even_stage), fp8_v_dot_layout)
                qk_even_tail = qk_even_tail * qk_scale
                if LOGIT_CAP > 0:
                    log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
                    inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
                    e_neg_even_tail = tl.math.exp2(-qk_even_tail * inv_cap)
                    sig_even_tail = 1.0 / (1.0 + e_neg_even_tail)
                    qk_even_tail = log2_cap * (2.0 * sig_even_tail - 1.0)
                local_even_tail = final_even_block * BLOCK_N + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                abs_even_tail = split_kv_offset + local_even_tail
                valid_even_tail = local_even_tail[None, :] < causal_kv_end
                if IS_CAUSAL:
                    valid_even_tail = valid_even_tail & (q_abs_pos[:, None] >= abs_even_tail[None, :])
                qk_even_tail = gl.where(
                    valid_even_tail,
                    qk_even_tail,
                    gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout),
                )
                m_even_tail = nan_propagating_max(qk_even_tail, axis=1)
                m_new_even_tail = gl.maximum(m_i, m_even_tail, propagate_nan=tl.PropagateNan.ALL)
                p_even_tail = gl.exp2(qk_even_tail - m_new_even_tail[:, None])
                l_even_tail = gl.sum(p_even_tail, axis=1)
                alpha_even_tail = gl.exp2(m_i - m_new_even_tail)
                l_i = l_i * alpha_even_tail + l_even_tail
                acc = acc * alpha_even_tail[:, None]
                m_i = m_new_even_tail
                p_cast_even_tail = p_even_tail.to(v_even_tail.dtype)
                p_d_even_tail = gl.convert_layout(p_cast_even_tail, fp8_p_dot_layout)
                acc = do_mma(p_d_even_tail, v_even_tail, acc)

                cdna4_async.wait_group(PAIR_WAIT_EPILOGUE_ODD)
                kt_odd_tail = cdna4_async.load_shared_relaxed(kt_smem.index(final_odd_stage), fp8_kt_dot_layout)
                kpe_hi_tail = _load_kpe_super_half(
                    kpe_smem.index(final_kpe_stage), BLOCK_N, BLOCK_N, fp8_kt_dot_layout,
                )
                qk_odd_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk_odd_tail = do_mma(fp8_q_dot, kt_odd_tail, qk_odd_tail)
                qk_odd_tail = do_mma(fp8_qpe_dot, kpe_hi_tail, qk_odd_tail)
                v_odd_tail = cdna4_async.load_shared_relaxed(v_smem.index(final_odd_stage), fp8_v_dot_layout)
                qk_odd_tail = qk_odd_tail * qk_scale
                if LOGIT_CAP > 0:
                    log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
                    inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
                    e_neg_odd_tail = tl.math.exp2(-qk_odd_tail * inv_cap)
                    sig_odd_tail = 1.0 / (1.0 + e_neg_odd_tail)
                    qk_odd_tail = log2_cap * (2.0 * sig_odd_tail - 1.0)
                local_odd_tail = final_odd_block * BLOCK_N + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                abs_odd_tail = split_kv_offset + local_odd_tail
                valid_odd_tail = local_odd_tail[None, :] < causal_kv_end
                if IS_CAUSAL:
                    valid_odd_tail = valid_odd_tail & (q_abs_pos[:, None] >= abs_odd_tail[None, :])
                qk_odd_tail = gl.where(
                    valid_odd_tail,
                    qk_odd_tail,
                    gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout),
                )
                m_odd_tail = nan_propagating_max(qk_odd_tail, axis=1)
                m_new_odd_tail = gl.maximum(m_i, m_odd_tail, propagate_nan=tl.PropagateNan.ALL)
                p_odd_tail = gl.exp2(qk_odd_tail - m_new_odd_tail[:, None])
                l_odd_tail = gl.sum(p_odd_tail, axis=1)
                alpha_odd_tail = gl.exp2(m_i - m_new_odd_tail)
                l_i = l_i * alpha_odd_tail + l_odd_tail
                acc = acc * alpha_odd_tail[:, None]
                m_i = m_new_odd_tail
                p_cast_odd_tail = p_odd_tail.to(v_odd_tail.dtype)
                p_d_odd_tail = gl.convert_layout(p_cast_odd_tail, fp8_p_dot_layout)
                acc = do_mma(p_d_odd_tail, v_odd_tail, acc)

                if (n_kv_blocks % 2) != 0:
                    tail_block = n_pair_blocks * 2
                    tail_stage = (tail_block % NUM_PFX_STAGES).to(tl.int32)

                    cdna4_async.wait_group(0)
                    kt_tail = cdna4_async.load_shared_relaxed(kt_smem.index(tail_stage), fp8_kt_dot_layout)
                    qk_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                    qk_tail = do_mma(fp8_q_dot, kt_tail, qk_tail)
                    qk_tail = _load_kpe_from_global_fp8(
                        qk_tail, fp8_qpe_dot, kv_base, kv_indices,
                        kv_start_base,
                        tail_block * BLOCK_N, causal_kv_end, stride_kvbs,
                        BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, fp8_kt_dot_layout,
                    )
                    qk_tail = qk_tail * qk_scale
                    if LOGIT_CAP > 0:
                        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
                        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
                        e_neg_tail = tl.math.exp2(-qk_tail * inv_cap)
                        sig_tail = 1.0 / (1.0 + e_neg_tail)
                        qk_tail = log2_cap * (2.0 * sig_tail - 1.0)
                    local_tail = tail_block * BLOCK_N + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                    abs_tail = split_kv_offset + local_tail
                    valid_tail = local_tail[None, :] < causal_kv_end
                    if IS_CAUSAL:
                        valid_tail = valid_tail & (q_abs_pos[:, None] >= abs_tail[None, :])
                    qk_tail = gl.where(
                        valid_tail,
                        qk_tail,
                        gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout),
                    )
                    m_tail = nan_propagating_max(qk_tail, axis=1)
                    m_new_tail = gl.maximum(m_i, m_tail, propagate_nan=tl.PropagateNan.ALL)
                    p_tail = gl.exp2(qk_tail - m_new_tail[:, None])
                    l_tail = gl.sum(p_tail, axis=1)
                    alpha_tail = gl.exp2(m_i - m_new_tail)
                    l_i = l_i * alpha_tail + l_tail
                    acc = acc * alpha_tail[:, None]
                    m_i = m_new_tail
                    v_tail = cdna4_async.load_shared_relaxed(v_smem.index(tail_stage), fp8_v_dot_layout)
                    p_cast_tail = p_tail.to(v_tail.dtype)
                    p_d_tail = gl.convert_layout(p_cast_tail, fp8_p_dot_layout)
                    acc = do_mma(p_d_tail, v_tail, acc)

                cdna4_async.wait_group(0)

        _v_scale: gl.constexpr = 1.0 if SPLIT_K > 1 else V_SCALE
        if _v_scale != 1.0:
            acc = acc * _v_scale

        if SPLIT_K > 1:
            no_valid = m_i == gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
            safe_l_i = gl.where(no_valid, gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout), l_i)
            l_recip_sk = 1.0 / safe_l_i
            safe_acc = gl.where(
                no_valid[:, None],
                gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout),
                acc,
            )
            acc_normed = safe_acc * l_recip_sk[:, None]
            lse = gl.where(
                no_valid,
                gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout),
                m_i + tl.log2(safe_l_i),
            )
            split_idx = output_tile * SPLIT_K + k_split_id

            po_base = partial_out + split_idx * BLOCK_M * BLOCK_DV
            po_ptrs = po_base + offs_m[:, None] * BLOCK_DV + offs_dv[None, :]
            po_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
            po_val = gl.convert_layout(acc_normed, blocked_layout)
            gl.store(po_ptrs, po_val, mask=po_mask)

            pl_base = partial_lse + split_idx * BLOCK_M
            pl_ptrs = pl_base + offs_m
            pl_mask = (cur_block_m * BLOCK_M + offs_m) < seq_len_extend
            lse_val = gl.convert_layout(lse, offs_m_layout)
            gl.store(pl_ptrs, lse_val, mask=pl_mask)
        else:
            l_recip = 1.0 / l_i
            acc = acc * l_recip[:, None]
            out_bf16 = gl.convert_layout(acc, blocked_layout).to(O.dtype.element_ty)
            o_base = (
                O + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
                + cur_q_head_idx * stride_oh
            )
            o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
            gl.store(o_base + offs_dv[None, :], out_bf16, mask=o_mask)

        tile_idx += total_programs


# ===-----------------------------------------------------------------------===#
# Split-K Reduce (reused from BF16 file)
# ===-----------------------------------------------------------------------===#


@triton.jit
def _mla_splitk_reduce(
    partial_out_ptr, partial_lse_ptr,
    O, qo_indptr,
    num_heads, n_m_tiles,
    stride_obs, stride_oh,
    SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DV: tl.constexpr,
    V_SCALE: tl.constexpr = 1.0,
):
    tile_id = tl.program_id(0)
    tiles_per_seq = num_heads * n_m_tiles
    cur_seq = tile_id // tiles_per_seq
    rem = tile_id % tiles_per_seq
    cur_head = rem // n_m_tiles
    cur_block_m = rem % n_m_tiles
    cur_seq_q_start = tl.load(qo_indptr + cur_seq)
    seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start

    offs_m = tl.arange(0, BLOCK_M)
    offs_dv = tl.arange(0, BLOCK_DV)
    m_mask = (cur_block_m * BLOCK_M + offs_m) < seq_len_extend

    base_0 = tile_id * SPLIT_K
    lse_0 = tl.load(partial_lse_ptr + base_0 * BLOCK_M + offs_m, mask=m_mask, other=float("-inf"))
    acc = tl.load(
        partial_out_ptr + base_0 * BLOCK_M * BLOCK_DV + offs_m[:, None] * BLOCK_DV + offs_dv[None, :],
        mask=m_mask[:, None], other=0.0,
    )

    for k in tl.static_range(1, SPLIT_K):
        base_k = base_0 + k
        lse_k = tl.load(partial_lse_ptr + base_k * BLOCK_M + offs_m, mask=m_mask, other=float("-inf"))
        acc_k = tl.load(
            partial_out_ptr + base_k * BLOCK_M * BLOCK_DV + offs_m[:, None] * BLOCK_DV + offs_dv[None, :],
            mask=m_mask[:, None], other=0.0,
        )
        max_lse = tl.maximum(lse_0, lse_k)
        w_old = tl.exp2(lse_0 - max_lse)
        w_new = tl.exp2(lse_k - max_lse)
        denom = w_old + w_new
        acc = (acc * w_old[:, None] + acc_k * w_new[:, None]) / denom[:, None]
        lse_0 = max_lse + tl.log2(denom)

    if V_SCALE != 1.0:
        acc = acc * V_SCALE

    o_ptrs = (
        O + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_dv[None, :]
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_mask[:, None])


# ===-----------------------------------------------------------------------===#
# Python Wrappers
# ===-----------------------------------------------------------------------===#


_splitk_workspace_cache = {}


def _get_mla_splitk_workspace(device, total_splits, block_m, block_dv):
    key = (device, block_m, block_dv)
    if key in _splitk_workspace_cache:
        po, pl = _splitk_workspace_cache[key]
        if po.shape[0] >= total_splits:
            po[:total_splits].zero_()
            pl[:total_splits].fill_(float("-inf"))
            return po, pl
    cap = max(total_splits, 512)
    po = torch.zeros(cap, block_m, block_dv, dtype=torch.float32, device=device)
    pl = torch.full((cap, block_m), float("-inf"), dtype=torch.float32, device=device)
    _splitk_workspace_cache[key] = (po, pl)
    return po, pl


def mla_d512_gqa_attention_fwd_fp8(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend=None,
    is_causal=True,
    sm_scale=None,
    logit_cap=0.0,
    k_scale=1.0,
    v_scale=1.0,
    custom_mask=None,
    mask_indptr=None,
    skip_prefix_custom_mask=True,
):
    """FP8 MLA prefill wrapper.

    q:          [total_q_tokens, num_q_heads, 576]  (bf16)
    kv_buffer:  [total_kv_tokens, num_kv_heads, 576] (fp8e4m3)
    o:          [total_q_tokens, num_q_heads, 512]  (bf16)
    k_scale/v_scale: dequant scales for FP8 KV.
    """
    Lq = q.shape[-1]
    assert Lq == 576, f"GQA MLA D512 kernel requires Lq=576, got {Lq}"
    Lv = o.shape[-1]
    assert Lv == 512, f"GQA MLA D512 kernel requires Lv=512, got {Lv}"

    batch_size = qo_indptr.shape[0] - 1
    num_q_heads = q.shape[1]
    num_kv_heads = kv_buffer.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads

    if max_len_extend is None:
        extend_lens = qo_indptr[1:] - qo_indptr[:-1]
        max_len_extend = int(extend_lens.max().item())

    sm_scale = (sm_scale or (1.0 / math.sqrt(Lq))) * k_scale

    BLOCK_M = 64
    n_m_blocks = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    grid = (batch_size, num_q_heads, n_m_blocks)

    USE_CUSTOM_MASK = custom_mask is not None
    dummy_mask = torch.empty(0, dtype=torch.uint8, device=q.device)
    dummy_mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int64, device=q.device)
    if not USE_CUSTOM_MASK:
        custom_mask = dummy_mask
        mask_indptr = dummy_mask_indptr

    mla_d512_gqa_fwd_fp8[grid](
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sm_scale,
        q.stride(0), q.stride(1),
        kv_buffer.stride(0), kv_buffer.stride(1),
        o.stride(0), o.stride(1),
        IS_CAUSAL=is_causal,
        LOGIT_CAP=logit_cap,
        GQA_RATIO=gqa_ratio,
        V_SCALE=v_scale,
        NUM_PFX_STAGES=3,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
    )


def mla_d512_gqa_attention_fwd_wca_fp8(
    q, kv_buffer, o,
    qo_indptr, kv_indptr, kv_indices,
    max_len_extend=None,
    is_causal=True,
    sm_scale=None,
    logit_cap=0.0,
    k_scale=1.0,
    v_scale=1.0,
    split_k=None,
):
    """WCA (persistent + split-K) FP8 MLA prefill wrapper."""
    Lq = q.shape[-1]
    assert Lq == 576
    Lv = o.shape[-1]
    assert Lv == 512

    BLOCK_M = 64
    batch_size = qo_indptr.shape[0] - 1
    num_q_heads = q.shape[1]
    num_kv_heads = kv_buffer.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads
    device = q.device

    if max_len_extend is None:
        extend_lens = qo_indptr[1:] - qo_indptr[:-1]
        max_len_extend = int(extend_lens.max().item())

    sm_scale = (sm_scale or (1.0 / math.sqrt(Lq))) * k_scale

    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    total_output_tiles = batch_size * num_q_heads * n_m_tiles
    if total_output_tiles == 0:
        return

    num_CUs = torch.cuda.get_device_properties(device).multi_processor_count
    if split_k is None:
        if total_output_tiles >= num_CUs:
            SPLIT_K = 1
        else:
            target = (num_CUs + total_output_tiles - 1) // total_output_tiles
            if target >= 8:
                SPLIT_K = 8
            elif target >= 4:
                SPLIT_K = 4
            elif target >= 2:
                SPLIT_K = 2
            else:
                SPLIT_K = 1
    else:
        SPLIT_K = split_k

    total_valid_tiles = total_output_tiles * SPLIT_K
    total_programs = min(total_valid_tiles, 2 * num_CUs)
    grid = (total_programs,)

    if SPLIT_K > 1:
        partial_out, partial_lse = _get_mla_splitk_workspace(
            device, total_output_tiles * SPLIT_K, BLOCK_M, 512
        )
    else:
        partial_out = torch.empty(1, dtype=torch.float32, device=device)
        partial_lse = torch.empty(1, dtype=torch.float32, device=device)

    po_flat = partial_out.reshape(-1) if SPLIT_K > 1 else partial_out
    pl_flat = partial_lse.reshape(-1) if SPLIT_K > 1 else partial_lse

    mla_d512_gqa_fwd_wca_fp8[grid](
        q, kv_buffer, o,
        qo_indptr, kv_indptr, kv_indices,
        sm_scale,
        q.stride(0), q.stride(1),
        kv_buffer.stride(0), kv_buffer.stride(1),
        o.stride(0), o.stride(1),
        po_flat, pl_flat,
        total_valid_tiles, total_programs,
        num_q_heads, n_m_tiles,
        IS_CAUSAL=is_causal, LOGIT_CAP=logit_cap,
        GQA_RATIO=gqa_ratio, SPLIT_K=SPLIT_K,
        V_SCALE=1.0 if SPLIT_K > 1 else v_scale,
        NUM_PFX_STAGES=3,
        num_warps=4, num_stages=1, waves_per_eu=1, matrix_instr_nonkdim=16,
    )

    if SPLIT_K > 1:
        reduce_grid = (total_output_tiles,)
        _mla_splitk_reduce[reduce_grid](
            po_flat, pl_flat,
            o, qo_indptr,
            num_q_heads, n_m_tiles,
            o.stride(0), o.stride(1),
            SPLIT_K=SPLIT_K, BLOCK_M=BLOCK_M, BLOCK_DV=512,
            V_SCALE=v_scale,
            num_warps=4,
        )
