# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental FP8 MLA D512 prefill kernel -- stripped for iteration.

Single Q-head per CTA, FP8 KV cache, 4 warps, BM64/BN32.
Prefix-only (no extend phase). No custom mask. No logit cap.
All shapes hardcoded for DeepSeek MLA: Lq=576 (512+64), Lv=512.

Derived from fp8_mla_prefill.py by removing unused features.
"""

import math

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
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
def _nan_max(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def _row_max(x, axis):
    return gl.reduce(x, axis, _nan_max)


@gluon.jit
def _fp8_mma(a, b, c):
    """FP8 MFMA: cast both operands to fp8e4nv and accumulate into fp32 c."""
    a_fp8 = tl.cast(a, tl.float8e4nv, bitcast=(a.dtype != tl.bfloat16 and a.dtype != tl.float16))
    b_fp8 = tl.cast(b, tl.float8e4nv, bitcast=True)
    return mfma_cdna4(a_fp8, b_fp8, c)


# ===-----------------------------------------------------------------------===#
# KPE global load (fallback for tail / short-sequence paths)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _load_kpe_global(
    qk, qpe_dot, kv_base, kv_indices, kv_start, start_n, seq_len,
    stride_kvbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    kt_dot_layout: gl.constexpr,
):
    """Load KPE from global memory and fuse QK_PE dot product."""
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_dot_layout))
    offs_dpe = gl.arange(0, BLOCK_DPE)
    n_idx = start_n + offs_n
    mask_n = n_idx < seq_len
    kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)
    kpe_ptrs = kv_base + kv_locs[None, :] * stride_kvbs + BLOCK_DMODEL + offs_dpe[:, None]
    kpe = gl.load(kpe_ptrs, mask=mask_n[None, :], other=0.0)
    kpe_dot = gl.convert_layout(kpe, kt_dot_layout)
    return _fp8_mma(qpe_dot, kpe_dot, qk)


# ===-----------------------------------------------------------------------===#
# Online softmax (no custom mask, no logit cap)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _softmax(
    acc, l_i, m_i, qk, start_n, seq_len, q_abs_pos, qk_scale,
    IS_CAUSAL: gl.constexpr,
    BLOCK_N: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
):
    qk_scaled = qk * qk_scale
    n_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
    valid = n_offs[None, :] < seq_len
    if IS_CAUSAL:
        valid = valid & (q_abs_pos[:, None] >= n_offs[None, :])
    qk_scaled = gl.where(
        valid, qk_scaled,
        gl.full([qk.shape[0], qk.shape[1]], float("-inf"), dtype=gl.float32, layout=mma_layout),
    )

    m_ij = _row_max(qk_scaled, axis=1)
    m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
    p = gl.exp2(qk_scaled - m_new[:, None])
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


# ===-----------------------------------------------------------------------===#
# Async DMA helpers
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _load_kv_locs(
    kv_indices, kv_start, start_n, seq_len,
    BLOCK_N: gl.constexpr, kt_async_layout: gl.constexpr,
):
    """Load paged KV indices for one N-block."""
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    n_idx = start_n + kt_offs_n
    mask_n = n_idx < seq_len
    safe_idx = gl.where(mask_n, kv_start + n_idx, gl.zeros([BLOCK_N], dtype=tl.int32, layout=kt_offs_n_layout))
    kv_locs = cdna_buffer_load(kv_indices, safe_idx.to(tl.int32), mask=mask_n, other=0)
    return kv_locs, mask_n


@gluon.jit
def _issue_k_async(
    kt_smem, k_base, kv_locs, mask_n_kt, stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kv_locs_kt = gl.convert_layout(kv_locs, gl.SliceLayout(dim=0, parent=kt_async_layout))
    kt_offsets = (kt_offs_d[:, None] + kv_locs_kt[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(kt_smem, k_base, kt_offsets, mask=mask_n_kt[None, :], other=0.0)
    cdna4_async.commit_group()


@gluon.jit
def _issue_v_async(
    v_smem, v_base, kv_locs, mask_n_v, stride_vbs,
    BLOCK_N: gl.constexpr, BLOCK_DV: gl.constexpr,
    v_async_layout: gl.constexpr,
):
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    kv_locs_v = gl.convert_layout(kv_locs, gl.SliceLayout(dim=1, parent=v_async_layout))
    v_offsets = (kv_locs_v[:, None] * stride_vbs + v_offs_d[None, :]).to(tl.int32)
    cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets, mask=mask_n_v[:, None], other=0.0)
    cdna4_async.commit_group()


@gluon.jit
def _issue_kv_async(
    kt_smem, v_smem, kv_base, kv_locs, mask_n,
    stride_kbs, stride_vbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_DV: gl.constexpr,
    kt_async_layout: gl.constexpr, v_async_layout: gl.constexpr,
):
    """Issue K + V DMA as a single commit group."""
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kv_locs_kt = gl.convert_layout(kv_locs, gl.SliceLayout(dim=0, parent=kt_async_layout))
    kt_offsets = (kt_offs_d[:, None] + kv_locs_kt[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(kt_smem, kv_base, kt_offsets, mask=mask_n[None, :], other=0.0)

    mask_n_v = gl.convert_layout(mask_n, gl.SliceLayout(dim=1, parent=v_async_layout))
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    kv_locs_v = gl.convert_layout(kv_locs, gl.SliceLayout(dim=1, parent=v_async_layout))
    v_offsets = (kv_locs_v[:, None] * stride_vbs + v_offs_d[None, :]).to(tl.int32)
    cdna4_async.buffer_load_to_shared(v_smem, kv_base, v_offsets, mask=mask_n_v[:, None], other=0.0)
    cdna4_async.commit_group()


@gluon.jit
def _issue_kpe_super_dma(
    kpe_smem, k_base, kv_indices, kv_start, start_n, seq_len, stride_kbs,
    BLOCK_DPE: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_N2: gl.constexpr, kpe_super_layout: gl.constexpr,
):
    """Issue DMA for a [64, 64] KPE super-tile covering 2 N-blocks."""
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
    cdna4_async.buffer_load_to_shared(kpe_smem, k_base, kpe_offsets, mask=mask_n[None, :], other=0.0)


@gluon.jit
def _issue_kpe_super_async(
    kpe_smem, k_base, kv_indices, kv_start, start_n, seq_len, stride_kbs,
    BLOCK_DPE: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_N2: gl.constexpr, kpe_super_layout: gl.constexpr,
):
    _issue_kpe_super_dma(
        kpe_smem, k_base, kv_indices, kv_start, start_n, seq_len,
        stride_kbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout,
    )
    cdna4_async.commit_group()


@gluon.jit
def _load_kpe_super_half(kpe_smem, half_start: gl.constexpr, BLOCK_N: gl.constexpr, kt_dot_layout: gl.constexpr):
    return cdna4_async.load_shared_relaxed(kpe_smem.slice(half_start, BLOCK_N, dim=1), kt_dot_layout)


# ===-----------------------------------------------------------------------===#
# Main kernel: FP8 MLA D512 prefill, 1 Q-head per CTA
# ===-----------------------------------------------------------------------===#


@gluon.jit
def mla_d512_fp8_prefill(
    Q, KV_Buffer, O,
    qo_indptr, kv_indptr, kv_indices,
    sm_scale,
    stride_qbs, stride_qh,
    stride_kvbs, stride_kvh,
    stride_obs, stride_oh,
    IS_CAUSAL: gl.constexpr,
    GQA_RATIO: gl.constexpr,
    V_SCALE: gl.constexpr = 1.0,
    NUM_PFX_STAGES: gl.constexpr = 3,
):
    """FP8 MLA D512 prefill. Grid: (batch, num_q_heads, m_blocks).

    Q:          [total_q, num_q_heads, 576] bf16
    KV_Buffer:  [total_kv, num_kv_heads, 576] fp8e4m3
    O:          [total_q, num_q_heads, 512] bf16
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

    # --- Layouts ---
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

    # --- FP8 async DMA layouts ---
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
        [0, 4], [0, 8], [0, 16], [0, 32], [0, 1], [0, 2],
    ]

    # --- SMEM allocation (FP8) ---
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

    # --- Load Q (global -> register, stays resident) ---
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

    q_base = (
        Q
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_q_head * stride_qh
    )
    q_reg = gl.load(q_base + offs_d[None, :], mask=q_mask, other=0.0)
    qpe_reg = gl.load(q_base + offs_dpe[None, :], mask=q_mask, other=0.0)
    fp8_q_dot = gl.convert_layout(q_reg.to(tl.float8e4nv), fp8_q_dot_layout)
    fp8_qpe_dot = gl.convert_layout(qpe_reg.to(tl.float8e4nv), fp8_q_dot_layout)

    # --- Accumulator init ---
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

    PAIR_WAIT_READY: gl.constexpr = 3
    PAIR_WAIT_EPILOGUE_ODD: gl.constexpr = 2

    # ===--- SHORT PATH (n_kv_blocks <= NUM_PFX_STAGES) ---===
    if n_kv_blocks <= 0:
        pass
    elif n_kv_blocks <= NUM_PFX_STAGES:
        for tail_i in gl.static_range(NUM_PFX_STAGES):
            kv_locs_t, mask_n_t = _load_kv_locs(
                kv_indices, cur_seq_kv_start, tail_i * BLOCK_N, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_v_t = gl.convert_layout(mask_n_t, gl.SliceLayout(dim=1, parent=v_async_layout))
            _issue_k_async(kt_smem.index(tail_i), kv_base, kv_locs_t, mask_n_t, stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout)
            _issue_v_async(v_smem.index(tail_i), kv_base, kv_locs_t, mask_n_v_t, stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout)

        cdna4_async.wait_group(0)

        for tail_i in gl.static_range(NUM_PFX_STAGES):
            if tail_i < n_kv_blocks:
                start_n_t = tail_i * BLOCK_N
                kt_d_t = cdna4_async.load_shared_relaxed(kt_smem.index(tail_i), fp8_kt_dot_layout)
                qk_t = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk_t = _fp8_mma(fp8_q_dot, kt_d_t, qk_t)
                qk_t = _load_kpe_global(
                    qk_t, fp8_qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                    start_n_t, causal_kv_end, stride_kvbs,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, fp8_kt_dot_layout,
                )
                acc, l_i, m_i, p_t = _softmax(
                    acc, l_i, m_i, qk_t, start_n_t, causal_kv_end, q_abs_pos, qk_scale,
                    IS_CAUSAL, BLOCK_N, mma_layout, mma_offs_n_col,
                )
                v_d_t = cdna4_async.load_shared_relaxed(v_smem.index(tail_i), fp8_v_dot_layout)
                p_cast_t = p_t.to(v_d_t.dtype)
                p_d_t = gl.convert_layout(p_cast_t, fp8_p_dot_layout)
                acc = _fp8_mma(p_d_t, v_d_t, acc)

    # ===--- MAIN PIPELINED PATH (3-stage K/V, 2-stage KPE super-tiles) ---===
    else:
        # Prefill pipeline: issue first 3 K/V stages + first 2 KPE super-tiles
        kv_locs_0, mask_n_0 = _load_kv_locs(kv_indices, cur_seq_kv_start, 0, causal_kv_end, BLOCK_N, kt_async_layout)
        _issue_kv_async(kt_smem.index(0), v_smem.index(0), kv_base, kv_locs_0, mask_n_0,
                        stride_kvbs, stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DV, kt_async_layout, v_async_layout)
        _issue_kpe_super_async(kpe_smem.index(0), kv_base, kv_indices, cur_seq_kv_start, 0, causal_kv_end,
                               stride_kvbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout)

        kv_locs_1, mask_n_1 = _load_kv_locs(kv_indices, cur_seq_kv_start, BLOCK_N, causal_kv_end, BLOCK_N, kt_async_layout)
        _issue_kv_async(kt_smem.index(1), v_smem.index(1), kv_base, kv_locs_1, mask_n_1,
                        stride_kvbs, stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DV, kt_async_layout, v_async_layout)
        _issue_kpe_super_async(kpe_smem.index(1), kv_base, kv_indices, cur_seq_kv_start, 2 * BLOCK_N, causal_kv_end,
                               stride_kvbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout)

        kv_locs_2, mask_n_2 = _load_kv_locs(kv_indices, cur_seq_kv_start, 2 * BLOCK_N, causal_kv_end, BLOCK_N, kt_async_layout)
        _issue_kv_async(kt_smem.index(2), v_smem.index(2), kv_base, kv_locs_2, mask_n_2,
                        stride_kvbs, stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DV, kt_async_layout, v_async_layout)

        n_pair_blocks = n_kv_blocks // 2
        main_pair_count = n_pair_blocks - 1

        # --- Main even/odd pair loop ---
        for pair_i in tl.range(0, main_pair_count):
            even_block = (pair_i * 2).to(tl.int32)
            odd_block = even_block + 1
            even_stage = (even_block % NUM_PFX_STAGES).to(tl.int32)
            odd_stage = (odd_block % NUM_PFX_STAGES).to(tl.int32)
            kpe_stage = (pair_i % NUM_KPE_STAGES).to(tl.int32)

            # --- Even block ---
            cdna4_async.wait_group(PAIR_WAIT_READY)
            kt_even = cdna4_async.load_shared_relaxed(kt_smem.index(even_stage), fp8_kt_dot_layout)
            kpe_lo = _load_kpe_super_half(kpe_smem.index(kpe_stage), 0, BLOCK_N, fp8_kt_dot_layout)
            qk_even = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk_even = _fp8_mma(fp8_q_dot, kt_even, qk_even)
            qk_even = _fp8_mma(fp8_qpe_dot, kpe_lo, qk_even)
            v_even = cdna4_async.load_shared_relaxed(v_smem.index(even_stage), fp8_v_dot_layout)

            # Re-issue DMA for future odd block
            future_odd_start = (even_block + NUM_PFX_STAGES) * BLOCK_N
            kv_locs_on, mask_n_on = _load_kv_locs(kv_indices, cur_seq_kv_start, future_odd_start, causal_kv_end, BLOCK_N, kt_async_layout)
            _issue_kv_async(kt_smem.index(even_stage), v_smem.index(even_stage), kv_base, kv_locs_on, mask_n_on,
                            stride_kvbs, stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DV, kt_async_layout, v_async_layout)

            acc, l_i, m_i, p_even = _softmax(
                acc, l_i, m_i, qk_even, even_block * BLOCK_N, causal_kv_end, q_abs_pos, qk_scale,
                IS_CAUSAL, BLOCK_N, mma_layout, mma_offs_n_col,
            )
            p_cast_even = p_even.to(v_even.dtype)
            p_d_even = gl.convert_layout(p_cast_even, fp8_p_dot_layout)
            acc = _fp8_mma(p_d_even, v_even, acc)

            # --- Odd block ---
            cdna4_async.wait_group(PAIR_WAIT_READY)
            kt_odd = cdna4_async.load_shared_relaxed(kt_smem.index(odd_stage), fp8_kt_dot_layout)
            kpe_hi = _load_kpe_super_half(kpe_smem.index(kpe_stage), BLOCK_N, BLOCK_N, fp8_kt_dot_layout)
            qk_odd = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk_odd = _fp8_mma(fp8_q_dot, kt_odd, qk_odd)
            qk_odd = _fp8_mma(fp8_qpe_dot, kpe_hi, qk_odd)
            v_odd = cdna4_async.load_shared_relaxed(v_smem.index(odd_stage), fp8_v_dot_layout)

            # Re-issue KPE super-tile + future even K/V
            future_pair_start = (even_block + 2 * NUM_KPE_STAGES) * BLOCK_N
            _issue_kpe_super_async(kpe_smem.index(kpe_stage), kv_base, kv_indices, cur_seq_kv_start,
                                   future_pair_start, causal_kv_end,
                                   stride_kvbs, BLOCK_DPE, BLOCK_DMODEL, BLOCK_N2, kpe_super_layout)
            future_even_start = (odd_block + NUM_PFX_STAGES) * BLOCK_N
            kv_locs_en, mask_n_en = _load_kv_locs(kv_indices, cur_seq_kv_start, future_even_start, causal_kv_end, BLOCK_N, kt_async_layout)
            _issue_kv_async(kt_smem.index(odd_stage), v_smem.index(odd_stage), kv_base, kv_locs_en, mask_n_en,
                            stride_kvbs, stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DV, kt_async_layout, v_async_layout)

            acc, l_i, m_i, p_odd = _softmax(
                acc, l_i, m_i, qk_odd, odd_block * BLOCK_N, causal_kv_end, q_abs_pos, qk_scale,
                IS_CAUSAL, BLOCK_N, mma_layout, mma_offs_n_col,
            )
            p_cast_odd = p_odd.to(v_odd.dtype)
            p_d_odd = gl.convert_layout(p_cast_odd, fp8_p_dot_layout)
            acc = _fp8_mma(p_d_odd, v_odd, acc)

        # --- Final pair epilogue ---
        final_pair = (n_pair_blocks - 1).to(tl.int32)
        fe = final_pair * 2
        fo = fe + 1
        fe_stage = (fe % NUM_PFX_STAGES).to(tl.int32)
        fo_stage = (fo % NUM_PFX_STAGES).to(tl.int32)
        fk_stage = (final_pair % NUM_KPE_STAGES).to(tl.int32)

        cdna4_async.wait_group(PAIR_WAIT_READY)
        kt_fe = cdna4_async.load_shared_relaxed(kt_smem.index(fe_stage), fp8_kt_dot_layout)
        kpe_lo_f = _load_kpe_super_half(kpe_smem.index(fk_stage), 0, BLOCK_N, fp8_kt_dot_layout)
        qk_fe = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk_fe = _fp8_mma(fp8_q_dot, kt_fe, qk_fe)
        qk_fe = _fp8_mma(fp8_qpe_dot, kpe_lo_f, qk_fe)
        v_fe = cdna4_async.load_shared_relaxed(v_smem.index(fe_stage), fp8_v_dot_layout)
        acc, l_i, m_i, p_fe = _softmax(
            acc, l_i, m_i, qk_fe, fe * BLOCK_N, causal_kv_end, q_abs_pos, qk_scale,
            IS_CAUSAL, BLOCK_N, mma_layout, mma_offs_n_col,
        )
        p_d_fe = gl.convert_layout(p_fe.to(v_fe.dtype), fp8_p_dot_layout)
        acc = _fp8_mma(p_d_fe, v_fe, acc)

        cdna4_async.wait_group(PAIR_WAIT_EPILOGUE_ODD)
        kt_fo = cdna4_async.load_shared_relaxed(kt_smem.index(fo_stage), fp8_kt_dot_layout)
        kpe_hi_f = _load_kpe_super_half(kpe_smem.index(fk_stage), BLOCK_N, BLOCK_N, fp8_kt_dot_layout)
        qk_fo = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk_fo = _fp8_mma(fp8_q_dot, kt_fo, qk_fo)
        qk_fo = _fp8_mma(fp8_qpe_dot, kpe_hi_f, qk_fo)
        v_fo = cdna4_async.load_shared_relaxed(v_smem.index(fo_stage), fp8_v_dot_layout)
        acc, l_i, m_i, p_fo = _softmax(
            acc, l_i, m_i, qk_fo, fo * BLOCK_N, causal_kv_end, q_abs_pos, qk_scale,
            IS_CAUSAL, BLOCK_N, mma_layout, mma_offs_n_col,
        )
        p_d_fo = gl.convert_layout(p_fo.to(v_fo.dtype), fp8_p_dot_layout)
        acc = _fp8_mma(p_d_fo, v_fo, acc)

        # --- Odd tail (when n_kv_blocks is odd) ---
        if (n_kv_blocks % 2) != 0:
            tail_block = n_pair_blocks * 2
            tail_stage = (tail_block % NUM_PFX_STAGES).to(tl.int32)
            cdna4_async.wait_group(0)
            kt_tail = cdna4_async.load_shared_relaxed(kt_smem.index(tail_stage), fp8_kt_dot_layout)
            qk_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk_tail = _fp8_mma(fp8_q_dot, kt_tail, qk_tail)
            qk_tail = _load_kpe_global(
                qk_tail, fp8_qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                tail_block * BLOCK_N, causal_kv_end, stride_kvbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, fp8_kt_dot_layout,
            )
            v_tail = cdna4_async.load_shared_relaxed(v_smem.index(tail_stage), fp8_v_dot_layout)
            acc, l_i, m_i, p_tail = _softmax(
                acc, l_i, m_i, qk_tail, tail_block * BLOCK_N, causal_kv_end, q_abs_pos, qk_scale,
                IS_CAUSAL, BLOCK_N, mma_layout, mma_offs_n_col,
            )
            p_d_tail = gl.convert_layout(p_tail.to(v_tail.dtype), fp8_p_dot_layout)
            acc = _fp8_mma(p_d_tail, v_tail, acc)

    # ===--- Output ---===
    if V_SCALE != 1.0:
        acc = acc * V_SCALE
    acc = acc * (1.0 / l_i)[:, None]
    out_bf16 = gl.convert_layout(acc, blocked_layout).to(O.dtype.element_ty)
    o_base = (
        O
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_q_head * stride_oh
    )
    o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    gl.store(o_base + offs_dv[None, :], out_bf16, mask=o_mask)


# ===-----------------------------------------------------------------------===#
# Python wrapper
# ===-----------------------------------------------------------------------===#


def mla_d512_fp8_prefill_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend=None,
    is_causal=True,
    sm_scale=None,
    k_scale=1.0,
    v_scale=1.0,
):
    """Launch FP8 MLA D512 prefill kernel.

    q:          [total_q_tokens, num_q_heads, 576]  bf16
    kv_buffer:  [total_kv_tokens, num_kv_heads, 576] fp8e4m3
    o:          [total_q_tokens, num_q_heads, 512]  bf16
    """
    Lq = q.shape[-1]
    assert Lq == 576, f"Requires Lq=576, got {Lq}"
    Lv = o.shape[-1]
    assert Lv == 512, f"Requires Lv=512, got {Lv}"

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

    mla_d512_fp8_prefill[grid](
        q, kv_buffer, o,
        qo_indptr, kv_indptr, kv_indices,
        sm_scale,
        q.stride(0), q.stride(1),
        kv_buffer.stride(0), kv_buffer.stride(1),
        o.stride(0), o.stride(1),
        IS_CAUSAL=is_causal,
        GQA_RATIO=gqa_ratio,
        V_SCALE=v_scale,
        NUM_PFX_STAGES=3,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
    )
