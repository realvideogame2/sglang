# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused Gluon MLA D512 extend-attention kernel for DeepSeek on MI350X.

Target config: BM64, BN32, 4 warps, 2-stage manual pipeline, BF16.
MMA: v_mfma_f32_16x16x32_{bf16,f16} (CDNA4 native, dtype-inferred).
waves_per_eu=1, matrix_instr_nonkdim=16.

This kernel handles ONLY the D512 MLA shapes (Lq=576, Lv=512) with
BLOCK_DMODEL=512, BLOCK_DPE=64, BLOCK_DV=512. It is a separate,
focused implementation for rapid iteration on MLA-specific optimizations.
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


@gluon.jit
def _nan_propagating_max(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def nan_propagating_max(x, axis):
    return gl.reduce(x, axis, _nan_propagating_max)


# ===-----------------------------------------------------------------------===#
# Extend inner loops (pipelined + serial, shared by full/masked dispatch)
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
    # Prologue: fill pipeline
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

    main_end = block_end - block_start - NUM_STAGES
    cdna4_async.wait_group(STREAMS)

    # Main loop: consume current, issue future
    for iter_n in tl.range(0, main_end):
        stage = (iter_n % NUM_STAGES).to(tl.int32)
        start_n = ((block_start + iter_n) * BLOCK_N).to(tl.int32)
        future_n = ((block_start + iter_n + NUM_STAGES) * BLOCK_N).to(tl.int32)

        kt_d = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
        _issue_async_k_extend(
            kt_smem.index(stage), k_ext_base, future_n, seq_len_extend,
            stride_kbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot, kt_d, qk)

        kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
        _issue_async_kpe_extend(
            kpe_smem.index(stage), k_ext_base, future_n, seq_len_extend,
            stride_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kpe_async_layout,
        )
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        acc, l_i, m_i, p = _softmax_extend(
            acc, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )

        v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
        _issue_async_v_extend(
            v_smem.index(stage), v_ext_base, future_n, seq_len_extend,
            stride_vbs, BLOCK_N, BLOCK_DV, v_async_layout,
        )
        p_cast = p.to(v_d.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)
        acc = mfma_cdna4(p_d, v_d, acc)

        cdna4_async.wait_group(STREAMS)

    # Epilogue: drain remaining stages
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
# Subtile extend inner loop (D-split: 256-wide halves, NUM_STAGES=3)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _issue_async_k_extend_half(
    kt_half_smem, k_base, start_n, seq_len, stride_kbs, d_offset,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL_HALF: gl.constexpr,
    kt_half_async_layout: gl.constexpr,
):
    d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_half_async_layout)
    n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_half_async_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL_HALF, layout=d_layout)
    offs_n = gl.arange(0, BLOCK_N, layout=n_layout)
    offsets = ((d_offset + offs_d)[:, None] + (start_n + offs_n[None, :]) * stride_kbs).to(tl.int32)
    mask = (start_n + offs_n[None, :]) < seq_len
    cdna4_async.buffer_load_to_shared(kt_half_smem, k_base, offsets, mask=mask, other=0.0)
    cdna4_async.commit_group()


@gluon.jit
def _issue_async_v_extend_half(
    v_half_smem, v_base, start_n, seq_len, stride_vbs, d_offset,
    BLOCK_N: gl.constexpr, BLOCK_DV_HALF: gl.constexpr,
    v_half_async_layout: gl.constexpr,
):
    n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_half_async_layout)
    d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_half_async_layout)
    offs_n = gl.arange(0, BLOCK_N, layout=n_layout)
    offs_d = gl.arange(0, BLOCK_DV_HALF, layout=d_layout)
    offsets = ((start_n + offs_n)[:, None] * stride_vbs + d_offset + offs_d[None, :]).to(tl.int32)
    mask = (start_n + offs_n)[:, None] < seq_len
    cdna4_async.buffer_load_to_shared(v_half_smem, v_base, offsets, mask=mask, other=0.0)
    cdna4_async.commit_group()


@gluon.jit
def _extend_subtile_pipelined(
    acc_lo, acc_hi, l_i, m_i,
    q_dot_lo, q_dot_hi, qpe_dot,
    k_ext_base, v_ext_base,
    cur_block_m, seq_len_extend, seq_len_prefix,
    stride_kbs, stride_vbs,
    block_start, block_end,
    kt_half_smem, v_half_smem,
    qk_scale, LOGIT_CAP: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr,
    Mask, mask_base_idx, mask_row_stride,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr, BLOCK_DPE: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    BLOCK_DMODEL_HALF: gl.constexpr, BLOCK_DV_HALF: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    kt_half_async_layout: gl.constexpr, kpe_async_layout: gl.constexpr,
    v_half_async_layout: gl.constexpr,
    kt_half_dot_layout: gl.constexpr, kpe_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr, v_half_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
):
    """D-subtiled extend inner loop: K/V split into 256-wide halves.

    K_lo is prefetched across N-blocks via async DMA. K_hi and V halves are
    loaded synchronously. KPE is loaded via global memory (no SMEM) to save
    LDS for deeper pipelines.
    """
    cdna4_async.wait_group(0)

    kpe_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kpe_async_layout)
    kpe_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
    offs_dpe = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=kpe_d_layout)
    offs_n_kpe = gl.arange(0, BLOCK_N, layout=kpe_n_layout)

    # Prologue: prefetch K_lo for first NUM_STAGES blocks
    for stage in gl.static_range(NUM_STAGES):
        pf_n = (block_start + stage) * BLOCK_N
        _issue_async_k_extend_half(
            kt_half_smem.index(stage), k_ext_base, pf_n, seq_len_extend,
            stride_kbs, 0, BLOCK_N, BLOCK_DMODEL_HALF,
            kt_half_async_layout,
        )

    # Main loop
    main_end = block_end - block_start - NUM_STAGES
    for iter_n in tl.range(0, main_end, loop_unroll_factor=1):
        si = (iter_n % NUM_STAGES).to(tl.int32)
        start_n = ((block_start + iter_n) * BLOCK_N).to(tl.int32)
        future_n = ((block_start + iter_n + NUM_STAGES) * BLOCK_N).to(tl.int32)

        cdna4_async.wait_group(NUM_STAGES - 1)
        kt_lo_d = cdna4_async.load_shared_relaxed(kt_half_smem.index(si), kt_half_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot_lo, kt_lo_d, qk)

        _issue_async_k_extend_half(
            kt_half_smem.index(si), k_ext_base, start_n, seq_len_extend,
            stride_kbs, BLOCK_DMODEL_HALF, BLOCK_N, BLOCK_DMODEL_HALF,
            kt_half_async_layout,
        )
        cdna4_async.wait_group(0)
        kt_hi_d = cdna4_async.load_shared_relaxed(kt_half_smem.index(si), kt_half_dot_layout)
        qk = mfma_cdna4(q_dot_hi, kt_hi_d, qk)

        kpe_mask = (start_n + offs_n_kpe[None, :]) < seq_len_extend
        kpe_t = gl.load(k_ext_base + (start_n + offs_n_kpe[None, :]) * stride_kbs + offs_dpe[:, None],
                        mask=kpe_mask, other=0.0)
        kpe_d = gl.convert_layout(kpe_t, kpe_dot_layout)
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        m_i_old = m_i
        acc_lo, l_i, m_i, p = _softmax_extend(
            acc_lo, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )
        acc_hi = acc_hi * gl.exp2(m_i_old - m_i)[:, None]

        p_cast = p.to(q_dot_lo.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)

        _issue_async_v_extend_half(
            v_half_smem.index(si), v_ext_base, start_n, seq_len_extend,
            stride_vbs, 0, BLOCK_N, BLOCK_DV_HALF,
            v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_lo_d = cdna4_async.load_shared_relaxed(v_half_smem.index(si), v_half_dot_layout)
        acc_lo = mfma_cdna4(p_d, v_lo_d, acc_lo)

        _issue_async_v_extend_half(
            v_half_smem.index(si), v_ext_base, start_n, seq_len_extend,
            stride_vbs, BLOCK_DV_HALF, BLOCK_N, BLOCK_DV_HALF,
            v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_hi_d = cdna4_async.load_shared_relaxed(v_half_smem.index(si), v_half_dot_layout)
        acc_hi = mfma_cdna4(p_d, v_hi_d, acc_hi)

        # Prefetch K_lo for future block
        _issue_async_k_extend_half(
            kt_half_smem.index(si), k_ext_base, future_n, seq_len_extend,
            stride_kbs, 0, BLOCK_N, BLOCK_DMODEL_HALF,
            kt_half_async_layout,
        )

    # Tail: drain NUM_STAGES blocks
    for tail_i in gl.static_range(NUM_STAGES):
        si = ((main_end + tail_i) % NUM_STAGES).to(tl.int32)
        start_n = (block_start + main_end + tail_i) * BLOCK_N

        cdna4_async.wait_group(NUM_STAGES - tail_i - 1)
        kt_lo_d = cdna4_async.load_shared_relaxed(kt_half_smem.index(si), kt_half_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot_lo, kt_lo_d, qk)

        _issue_async_k_extend_half(
            kt_half_smem.index(si), k_ext_base, start_n, seq_len_extend,
            stride_kbs, BLOCK_DMODEL_HALF, BLOCK_N, BLOCK_DMODEL_HALF,
            kt_half_async_layout,
        )
        cdna4_async.wait_group(0)
        kt_hi_d = cdna4_async.load_shared_relaxed(kt_half_smem.index(si), kt_half_dot_layout)
        qk = mfma_cdna4(q_dot_hi, kt_hi_d, qk)

        kpe_mask = (start_n + offs_n_kpe[None, :]) < seq_len_extend
        kpe_t = gl.load(k_ext_base + (start_n + offs_n_kpe[None, :]) * stride_kbs + offs_dpe[:, None],
                        mask=kpe_mask, other=0.0)
        kpe_d = gl.convert_layout(kpe_t, kpe_dot_layout)
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        m_i_old = m_i
        acc_lo, l_i, m_i, p = _softmax_extend(
            acc_lo, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )
        acc_hi = acc_hi * gl.exp2(m_i_old - m_i)[:, None]

        p_cast = p.to(q_dot_lo.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)

        _issue_async_v_extend_half(
            v_half_smem.index(si), v_ext_base, start_n, seq_len_extend,
            stride_vbs, 0, BLOCK_N, BLOCK_DV_HALF,
            v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_lo_d = cdna4_async.load_shared_relaxed(v_half_smem.index(si), v_half_dot_layout)
        acc_lo = mfma_cdna4(p_d, v_lo_d, acc_lo)

        _issue_async_v_extend_half(
            v_half_smem.index(si), v_ext_base, start_n, seq_len_extend,
            stride_vbs, BLOCK_DV_HALF, BLOCK_N, BLOCK_DV_HALF,
            v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_hi_d = cdna4_async.load_shared_relaxed(v_half_smem.index(si), v_half_dot_layout)
        acc_hi = mfma_cdna4(p_d, v_hi_d, acc_hi)

    return acc_lo, acc_hi, l_i, m_i


@gluon.jit
def _extend_subtile_serial(
    acc_lo, acc_hi, l_i, m_i,
    q_dot_lo, q_dot_hi, qpe_dot,
    k_ext_base, v_ext_base,
    cur_block_m, seq_len_extend, seq_len_prefix,
    stride_kbs, stride_vbs,
    block_start, block_end,
    kt_half_smem, v_half_smem,
    qk_scale, LOGIT_CAP: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr,
    Mask, mask_base_idx, mask_row_stride,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr, BLOCK_DPE: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    BLOCK_DMODEL_HALF: gl.constexpr, BLOCK_DV_HALF: gl.constexpr,
    kt_half_async_layout: gl.constexpr, kpe_async_layout: gl.constexpr,
    v_half_async_layout: gl.constexpr,
    kt_half_dot_layout: gl.constexpr, kpe_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr, v_half_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
):
    cdna4_async.wait_group(0)

    kpe_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kpe_async_layout)
    kpe_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
    offs_dpe = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=kpe_d_layout)
    offs_n_kpe = gl.arange(0, BLOCK_N, layout=kpe_n_layout)

    n_local = block_end - block_start
    for local_i in tl.range(0, n_local):
        start_n = ((block_start + local_i) * BLOCK_N).to(tl.int32)

        _issue_async_k_extend_half(
            kt_half_smem.index(0), k_ext_base, start_n, seq_len_extend,
            stride_kbs, 0, BLOCK_N, BLOCK_DMODEL_HALF, kt_half_async_layout,
        )
        cdna4_async.wait_group(0)
        kt_lo_d = cdna4_async.load_shared_relaxed(kt_half_smem.index(0), kt_half_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = mfma_cdna4(q_dot_lo, kt_lo_d, qk)

        _issue_async_k_extend_half(
            kt_half_smem.index(0), k_ext_base, start_n, seq_len_extend,
            stride_kbs, BLOCK_DMODEL_HALF, BLOCK_N, BLOCK_DMODEL_HALF, kt_half_async_layout,
        )
        cdna4_async.wait_group(0)
        kt_hi_d = cdna4_async.load_shared_relaxed(kt_half_smem.index(0), kt_half_dot_layout)
        qk = mfma_cdna4(q_dot_hi, kt_hi_d, qk)

        kpe_mask = (start_n + offs_n_kpe[None, :]) < seq_len_extend
        kpe_t = gl.load(k_ext_base + (start_n + offs_n_kpe[None, :]) * stride_kbs + offs_dpe[:, None],
                        mask=kpe_mask, other=0.0)
        kpe_d = gl.convert_layout(kpe_t, kpe_dot_layout)
        qk = mfma_cdna4(qpe_dot, kpe_d, qk)

        m_i_old = m_i
        acc_lo, l_i, m_i, p = _softmax_extend(
            acc_lo, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, IS_CAUSAL,
            USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
            seq_len_prefix,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )
        acc_hi = acc_hi * gl.exp2(m_i_old - m_i)[:, None]

        p_cast = p.to(q_dot_lo.dtype)
        p_d = gl.convert_layout(p_cast, p_dot_layout)

        _issue_async_v_extend_half(
            v_half_smem.index(0), v_ext_base, start_n, seq_len_extend,
            stride_vbs, 0, BLOCK_N, BLOCK_DV_HALF, v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_lo_d = cdna4_async.load_shared_relaxed(v_half_smem.index(0), v_half_dot_layout)
        acc_lo = mfma_cdna4(p_d, v_lo_d, acc_lo)

        _issue_async_v_extend_half(
            v_half_smem.index(0), v_ext_base, start_n, seq_len_extend,
            stride_vbs, BLOCK_DV_HALF, BLOCK_N, BLOCK_DV_HALF, v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_hi_d = cdna4_async.load_shared_relaxed(v_half_smem.index(0), v_half_dot_layout)
        acc_hi = mfma_cdna4(p_d, v_hi_d, acc_hi)

    return acc_lo, acc_hi, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def mla_d512_extend_fwd(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    Mask,
    MaskIndptr,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    IS_CAUSAL: gl.constexpr,
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,
    LOGIT_CAP: gl.constexpr,
    ENABLE_MASK_SPLIT: gl.constexpr = True,
    USE_SUBTILE: gl.constexpr = False,
):
    # --- Fixed D512 config ---
    BLOCK_M: gl.constexpr = 64
    BLOCK_N: gl.constexpr = 32
    BLOCK_DMODEL: gl.constexpr = 512
    BLOCK_DPE: gl.constexpr = 64
    BLOCK_DV: gl.constexpr = 512
    NUM_STAGES: gl.constexpr = 4 if USE_SUBTILE else 2
    num_warps: gl.constexpr = gl.num_warps()

    # MMA = 16x16x32 bf16 (CDNA4 native)
    MMA_INSTR_M: gl.constexpr = 16
    MMA_INSTR_N: gl.constexpr = 16
    MMA_INSTR_K: gl.constexpr = 32
    QK_K_WIDTH: gl.constexpr = 8
    PV_K_WIDTH: gl.constexpr = 4
    ASYNC_PAD_K: gl.constexpr = 8
    ASYNC_PAD_V: gl.constexpr = 32

    # --- Grid decode ---
    cur_seq = gl.program_id(0)
    cur_head = gl.program_id(1)
    cur_block_m = gl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_q_start = gl.load(qo_indptr + cur_seq)
    seq_len_extend = gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start
    cur_seq_kv_start = gl.load(kv_indptr + cur_seq)
    seq_len_prefix = gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start

    if cur_block_m * BLOCK_M >= seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        mask_base_idx = gl.load(MaskIndptr + cur_seq).to(tl.int64)
        cur_seq_len = seq_len_prefix + seq_len_extend
        mask_row_stride = cur_seq_len.to(tl.int64)
    else:
        mask_base_idx = tl.cast(0, tl.int64)
        mask_row_stride = tl.cast(0, tl.int64)

    # --- Layout definitions ---
    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=QK_K_WIDTH
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=QK_K_WIDTH
    )
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=PV_K_WIDTH
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=PV_K_WIDTH
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

    # --- K^T async layout: [512, 32] ---
    kt_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    kt_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
        [0, 16],
        [0, 1], [0, 2], [0, 4], [0, 8],
    ]

    # --- V async layout: [32, 512] ---
    v_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_N, BLOCK_DV],
    )
    v_offset_bases: gl.constexpr = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
        [16, 0],
        [1, 0], [2, 0], [4, 0], [8, 0],
    ]

    # --- K-rope async layout: [64, 32] ---
    kpe_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_DPE, BLOCK_N],
    )
    kpe_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        [0, 4], [0, 8], [0, 16],
        [0, 1], [0, 2],
    ]

    # --- Shared memory allocation ---
    kpe_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_K]],
        offset_bases=kpe_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DPE, BLOCK_N],
    )

    if USE_SUBTILE:
        BLOCK_DMODEL_HALF: gl.constexpr = BLOCK_DMODEL // 2
        BLOCK_DV_HALF: gl.constexpr = BLOCK_DV // 2

        # Half-K^T layout: [256, 32] for BLOCK_N=32
        kt_half_offset_bases: gl.constexpr = [
            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0],
            [0, 16],
            [0, 1], [0, 2], [0, 4], [0, 8],
        ]
        kt_half_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]],
            warp_bases=[[0, 1], [0, 2]],
            block_bases=[],
            shape=[BLOCK_DMODEL_HALF, BLOCK_N],
        )
        # Half-V layout: [32, 256] for BLOCK_N=32
        v_half_offset_bases: gl.constexpr = [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128],
            [16, 0],
            [1, 0], [2, 0], [4, 0], [8, 0],
        ]
        v_half_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]],
            warp_bases=[[1, 0], [2, 0]],
            block_bases=[],
            shape=[BLOCK_N, BLOCK_DV_HALF],
        )

        kt_half_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[512, ASYNC_PAD_K]],
            offset_bases=kt_half_offset_bases,
            cga_layout=[],
            shape=[BLOCK_DMODEL_HALF, BLOCK_N],
        )
        v_half_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[512, ASYNC_PAD_V]],
            offset_bases=v_half_offset_bases,
            cga_layout=[],
            shape=[BLOCK_N, BLOCK_DV_HALF],
        )

        kt_half_smem = gl.allocate_shared_memory(
            Q_Extend.dtype.element_ty,
            [NUM_STAGES, BLOCK_DMODEL_HALF, BLOCK_N],
            layout=kt_half_smem_layout,
        )
        v_half_smem = gl.allocate_shared_memory(
            Q_Extend.dtype.element_ty,
            [NUM_STAGES, BLOCK_N, BLOCK_DV_HALF],
            layout=v_half_smem_layout,
        )

        for _s in gl.static_range(NUM_STAGES):
            vz = gl.zeros(
                [BLOCK_N, BLOCK_DV_HALF],
                dtype=Q_Extend.dtype.element_ty,
                layout=v_half_async_layout,
            )
            v_half_smem.index(_s).store(vz)
        gl.barrier()

        kt_half_dot_layout: gl.constexpr = kt_dot_layout
        v_half_dot_layout: gl.constexpr = v_dot_layout

        # Subtile mode: no KPE SMEM allocated. KPE loaded via global memory
        # in both prefix and extend phases to save LDS for deeper pipeline.
    else:
        kpe_smem = gl.allocate_shared_memory(
            Q_Extend.dtype.element_ty,
            [NUM_STAGES, BLOCK_DPE, BLOCK_N],
            layout=kpe_smem_layout,
        )

        kt_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[512, ASYNC_PAD_K]],
            offset_bases=kt_offset_bases,
            cga_layout=[],
            shape=[BLOCK_DMODEL, BLOCK_N],
        )
        v_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[512, ASYNC_PAD_V]],
            offset_bases=v_offset_bases,
            cga_layout=[],
            shape=[BLOCK_N, BLOCK_DV],
        )

        kt_smem = gl.allocate_shared_memory(
            Q_Extend.dtype.element_ty,
            [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
            layout=kt_smem_layout,
        )
        v_smem = gl.allocate_shared_memory(
            Q_Extend.dtype.element_ty,
            [NUM_STAGES, BLOCK_N, BLOCK_DV],
            layout=v_smem_layout,
        )

        for _s in gl.static_range(NUM_STAGES):
            v_zero = gl.zeros(
                [BLOCK_N, BLOCK_DV],
                dtype=Q_Extend.dtype.element_ty,
                layout=v_async_layout,
            )
            v_smem.index(_s).store(v_zero)
        gl.barrier()

    # --- Load Q nope [BM, 512] and Q rope [BM, 64] ---
    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=offs_d_layout)

    q_base = (
        Q_Extend
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
    )
    q_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    q = gl.load(q_base + offs_d[None, :], mask=q_mask, other=0.0)

    offs_dpe = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=offs_d_layout)
    qpe = gl.load(q_base + offs_dpe[None, :], mask=q_mask, other=0.0)

    q_dot = gl.convert_layout(q, q_dot_layout)
    qpe_dot = gl.convert_layout(qpe, q_dot_layout)

    # --- Softmax state ---
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)
    qk_scale = sm_scale * LOG2E

    q_abs_pos = (
        seq_len_prefix
        + cur_block_m * BLOCK_M
        + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    )

    # ===========================
    # PREFIX phase (KV cache)
    # ===========================
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N

    k_pfx_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_pfx_base = V_Buffer + cur_kv_head * stride_buf_vh

    STREAMS: gl.constexpr = 3
    PFX_NUM_STAGES: gl.constexpr = 2

    if USE_SUBTILE:
        acc_lo = gl.zeros([BLOCK_M, BLOCK_DV_HALF], dtype=gl.float32, layout=mma_layout)
        acc_hi = gl.zeros([BLOCK_M, BLOCK_DV_HALF], dtype=gl.float32, layout=mma_layout)

        if n_prefix_blocks > 0:
            kt_pfx_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_half_async_layout)
            kt_pfx_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_half_async_layout)
            kpe_pfx_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
            kpe_pfx_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kpe_async_layout)
            v_pfx_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_half_async_layout)
            v_pfx_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_half_async_layout)

            offs_d_kt_lo = gl.arange(0, BLOCK_DMODEL // 2, layout=kt_pfx_d_layout)
            offs_d_kt_hi = BLOCK_DMODEL // 2 + gl.arange(0, BLOCK_DMODEL // 2, layout=kt_pfx_d_layout)
            offs_dpe_kt = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=kpe_pfx_d_layout)

            offs_d_q_lo = gl.arange(0, BLOCK_DMODEL // 2, layout=offs_d_layout)
            offs_d_q_hi = BLOCK_DMODEL // 2 + gl.arange(0, BLOCK_DMODEL // 2, layout=offs_d_layout)
            q_lo_pfx = gl.load(q_base + offs_d_q_lo[None, :], mask=q_mask, other=0.0)
            q_hi_pfx = gl.load(q_base + offs_d_q_hi[None, :], mask=q_mask, other=0.0)
            q_dot_lo_pfx = gl.convert_layout(q_lo_pfx, q_dot_layout)
            q_dot_hi_pfx = gl.convert_layout(q_hi_pfx, q_dot_layout)

            offs_dv_lo = gl.arange(0, BLOCK_DV_HALF, layout=v_pfx_d_layout)
            offs_dv_hi = BLOCK_DV_HALF + gl.arange(0, BLOCK_DV_HALF, layout=v_pfx_d_layout)

            for pfx_bn in tl.range(0, n_prefix_blocks):
                start_n = (pfx_bn * BLOCK_N).to(tl.int32)
                n_idx = start_n + gl.arange(0, BLOCK_N, layout=kt_pfx_n_layout)
                mask_n = n_idx < seq_len_prefix
                safe_idx = gl.where(mask_n, cur_seq_kv_start + n_idx, gl.zeros([BLOCK_N], dtype=tl.int32, layout=kt_pfx_n_layout))
                kv_locs = cdna_buffer_load(kv_indices, safe_idx.to(tl.int32), mask=mask_n, other=0)

                kv_locs_kt = gl.convert_layout(kv_locs, kt_pfx_n_layout)
                mask_n_kt = gl.convert_layout(mask_n, kt_pfx_n_layout)
                k_lo_t = gl.load(
                    k_pfx_base + kv_locs_kt[None, :] * stride_buf_kbs + offs_d_kt_lo[:, None],
                    mask=mask_n_kt[None, :], other=0.0
                )
                k_hi_t = gl.load(
                    k_pfx_base + kv_locs_kt[None, :] * stride_buf_kbs + offs_d_kt_hi[:, None],
                    mask=mask_n_kt[None, :], other=0.0
                )
                kt_lo_d = gl.convert_layout(k_lo_t, kt_half_dot_layout)
                kt_hi_d = gl.convert_layout(k_hi_t, kt_half_dot_layout)

                qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk = mfma_cdna4(q_dot_lo_pfx, kt_lo_d, qk)
                qk = mfma_cdna4(q_dot_hi_pfx, kt_hi_d, qk)

                kv_locs_kpe = gl.convert_layout(kv_locs, kpe_pfx_n_layout)
                mask_n_kpe = gl.convert_layout(mask_n, kpe_pfx_n_layout)
                kpe_t = gl.load(
                    k_pfx_base + kv_locs_kpe[None, :] * stride_buf_kbs + offs_dpe_kt[:, None],
                    mask=mask_n_kpe[None, :], other=0.0
                )
                kpe_d = gl.convert_layout(kpe_t, kt_dot_layout)
                qk = mfma_cdna4(qpe_dot, kpe_d, qk)

                m_old = m_i
                acc_lo, l_i, m_i, p = _softmax_prefix(
                    acc_lo, l_i, m_i, qk, start_n, seq_len_prefix, q_abs_pos,
                    qk_scale, LOGIT_CAP,
                    USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
                    Mask, mask_base_idx, mask_row_stride,
                    cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row),
                    BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col,
                )
                alpha = gl.exp2(m_old - m_i)
                acc_hi = acc_hi * alpha[:, None]

                kv_locs_v = gl.convert_layout(kv_locs, v_pfx_n_layout)
                mask_n_v = gl.convert_layout(mask_n, v_pfx_n_layout)
                v_lo = gl.load(
                    v_pfx_base + kv_locs_v[:, None] * stride_buf_vbs + offs_dv_lo[None, :],
                    mask=mask_n_v[:, None], other=0.0
                )
                v_hi = gl.load(
                    v_pfx_base + kv_locs_v[:, None] * stride_buf_vbs + offs_dv_hi[None, :],
                    mask=mask_n_v[:, None], other=0.0
                )
                v_lo_d = gl.convert_layout(v_lo, v_half_dot_layout)
                v_hi_d = gl.convert_layout(v_hi, v_half_dot_layout)

                p_cast = p.to(q.dtype)
                p_d = gl.convert_layout(p_cast, p_dot_layout)
                acc_lo = mfma_cdna4(p_d, v_lo_d, acc_lo)
                acc_hi = mfma_cdna4(p_d, v_hi_d, acc_hi)

    elif n_prefix_blocks >= PFX_NUM_STAGES:
        # --- Pipelined prefix path ---
        for _s in gl.static_range(PFX_NUM_STAGES):
            kv_locs, mask_n = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, _s * BLOCK_N, seq_len_prefix,
                BLOCK_N, kt_async_layout,
            )
            mask_n_kpe = gl.convert_layout(mask_n, gl.SliceLayout(dim=0, parent=kpe_async_layout))
            mask_n_v = gl.convert_layout(mask_n, gl.SliceLayout(dim=1, parent=v_async_layout))
            _issue_prefix_kvkpe_async(
                kt_smem.index(_s), kpe_smem.index(_s), k_pfx_base,
                kv_locs, mask_n, mask_n_kpe,
                stride_buf_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
                kt_async_layout, kpe_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(_s), v_pfx_base,
                kv_locs, mask_n_v,
                stride_buf_vbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

        pfx_main_end = n_prefix_blocks - PFX_NUM_STAGES
        cdna4_async.wait_group(STREAMS)

        for block_n in tl.range(0, pfx_main_end):
            stage = (block_n % PFX_NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)
            future_n = ((block_n + PFX_NUM_STAGES) * BLOCK_N).to(tl.int32)

            kv_locs, mask_n = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, future_n, seq_len_prefix,
                BLOCK_N, kt_async_layout,
            )
            mask_n_kpe = gl.convert_layout(mask_n, gl.SliceLayout(dim=0, parent=kpe_async_layout))
            mask_n_v = gl.convert_layout(mask_n, gl.SliceLayout(dim=1, parent=v_async_layout))

            kt_d = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
            _issue_prefix_k_async(
                kt_smem.index(stage), k_pfx_base,
                kv_locs, mask_n,
                stride_buf_kbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
            )
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = mfma_cdna4(q_dot, kt_d, qk)

            kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
            _issue_prefix_kpe_async(
                kpe_smem.index(stage), k_pfx_base,
                kv_locs, mask_n_kpe,
                stride_buf_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kpe_async_layout,
            )
            qk = mfma_cdna4(qpe_dot, kpe_d, qk)

            acc, l_i, m_i, p = _softmax_prefix(
                acc, l_i, m_i, qk, start_n, seq_len_prefix, q_abs_pos,
                qk_scale, LOGIT_CAP,
                USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
                Mask, mask_base_idx, mask_row_stride,
                cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row),
                BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col,
            )

            v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
            _issue_prefix_v_async(
                v_smem.index(stage), v_pfx_base,
                kv_locs, mask_n_v,
                stride_buf_vbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )
            p_cast = p.to(v_d.dtype)
            p_d = gl.convert_layout(p_cast, p_dot_layout)
            acc = mfma_cdna4(p_d, v_d, acc)

            cdna4_async.wait_group(STREAMS)

        # Epilogue: drain all outstanding, then consume remaining stages
        cdna4_async.wait_group(0)
        for tail_i in gl.static_range(PFX_NUM_STAGES):
            stage = ((pfx_main_end + tail_i) % PFX_NUM_STAGES).to(tl.int32)
            start_n = (pfx_main_end + tail_i) * BLOCK_N

            kt_d_tail = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = mfma_cdna4(q_dot, kt_d_tail, qk)
            kpe_d_tail = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
            qk = mfma_cdna4(qpe_dot, kpe_d_tail, qk)

            acc, l_i, m_i, p = _softmax_prefix(
                acc, l_i, m_i, qk, start_n, seq_len_prefix, q_abs_pos,
                qk_scale, LOGIT_CAP,
                USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
                Mask, mask_base_idx, mask_row_stride,
                cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row),
                BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col,
            )
            v_d_tail = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
            p_cast = p.to(v_d_tail.dtype)
            p_d = gl.convert_layout(p_cast, p_dot_layout)
            acc = mfma_cdna4(p_d, v_d_tail, acc)

    elif n_prefix_blocks > 0:
        # --- Serial prefix path (< NUM_STAGES blocks) ---
        for block_n in tl.range(0, n_prefix_blocks):
            start_n = (block_n * BLOCK_N).to(tl.int32)

            kv_locs, mask_n = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, start_n, seq_len_prefix,
                BLOCK_N, kt_async_layout,
            )
            mask_n_kpe = gl.convert_layout(mask_n, gl.SliceLayout(dim=0, parent=kpe_async_layout))
            mask_n_v = gl.convert_layout(mask_n, gl.SliceLayout(dim=1, parent=v_async_layout))

            _issue_prefix_kvkpe_async(
                kt_smem.index(0), kpe_smem.index(0), k_pfx_base,
                kv_locs, mask_n, mask_n_kpe,
                stride_buf_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
                kt_async_layout, kpe_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(0), v_pfx_base,
                kv_locs, mask_n_v,
                stride_buf_vbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )
            cdna4_async.wait_group(0)

            kt_d = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = mfma_cdna4(q_dot, kt_d, qk)
            kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(0), kt_dot_layout)
            qk = mfma_cdna4(qpe_dot, kpe_d, qk)

            acc, l_i, m_i, p = _softmax_prefix(
                acc, l_i, m_i, qk, start_n, seq_len_prefix, q_abs_pos,
                qk_scale, LOGIT_CAP,
                USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
                Mask, mask_base_idx, mask_row_stride,
                cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row),
                BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col,
            )
            v_d = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
            p_cast = p.to(v_d.dtype)
            p_d = gl.convert_layout(p_cast, p_dot_layout)
            acc = mfma_cdna4(p_d, v_d, acc)

    # ===========================
    # EXTEND phase (contiguous)
    # ===========================
    if IS_CAUSAL:
        causal_end = tl.minimum(seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    else:
        causal_end = seq_len_extend
    n_extend_blocks = (causal_end + BLOCK_N - 1) // BLOCK_N

    if ENABLE_MASK_SPLIT and IS_CAUSAL and not USE_CUSTOM_MASK:
        partial_block = ((causal_end % BLOCK_N) != 0).to(tl.int32)
        masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
        masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
        n_full_blocks = n_extend_blocks - masked_blocks
    else:
        n_full_blocks = 0

    k_ext_base = K_Extend + cur_seq_q_start * stride_kbs + cur_kv_head * stride_kh
    v_ext_base = V_Extend + cur_seq_q_start * stride_vbs + cur_kv_head * stride_vh

    if USE_SUBTILE:
        offs_d_lo = gl.arange(0, BLOCK_DMODEL_HALF, layout=offs_d_layout)
        offs_d_hi = BLOCK_DMODEL_HALF + gl.arange(0, BLOCK_DMODEL_HALF, layout=offs_d_layout)
        q_lo = gl.load(q_base + offs_d_lo[None, :], mask=q_mask, other=0.0)
        q_hi = gl.load(q_base + offs_d_hi[None, :], mask=q_mask, other=0.0)
        q_dot_lo = gl.convert_layout(q_lo, q_dot_layout)
        q_dot_hi = gl.convert_layout(q_hi, q_dot_layout)

        if n_full_blocks >= NUM_STAGES:
            acc_lo, acc_hi, l_i, m_i = _extend_subtile_pipelined(
                acc_lo, acc_hi, l_i, m_i,
                q_dot_lo, q_dot_hi, qpe_dot,
                k_ext_base, v_ext_base,
                cur_block_m, seq_len_extend, seq_len_prefix,
                stride_kbs, stride_vbs,
                0, n_full_blocks,
                kt_half_smem, v_half_smem,
                qk_scale, LOGIT_CAP, False,
                USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
                BLOCK_M, BLOCK_N,
                BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV,
                BLOCK_DMODEL_HALF, BLOCK_DV_HALF,
                NUM_STAGES,
                kt_half_async_layout, kpe_async_layout, v_half_async_layout,
                kt_half_dot_layout, kt_dot_layout,
                p_dot_layout, v_half_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
        elif n_full_blocks > 0:
            acc_lo, acc_hi, l_i, m_i = _extend_subtile_serial(
                acc_lo, acc_hi, l_i, m_i,
                q_dot_lo, q_dot_hi, qpe_dot,
                k_ext_base, v_ext_base,
                cur_block_m, seq_len_extend, seq_len_prefix,
                stride_kbs, stride_vbs,
                0, n_full_blocks,
                kt_half_smem, v_half_smem,
                qk_scale, LOGIT_CAP, False,
                USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
                BLOCK_M, BLOCK_N,
                BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV,
                BLOCK_DMODEL_HALF, BLOCK_DV_HALF,
                kt_half_async_layout, kpe_async_layout, v_half_async_layout,
                kt_half_dot_layout, kt_dot_layout,
                p_dot_layout, v_half_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )

        remaining_blocks = n_extend_blocks - n_full_blocks
        if remaining_blocks >= NUM_STAGES:
            acc_lo, acc_hi, l_i, m_i = _extend_subtile_pipelined(
                acc_lo, acc_hi, l_i, m_i,
                q_dot_lo, q_dot_hi, qpe_dot,
                k_ext_base, v_ext_base,
                cur_block_m, seq_len_extend, seq_len_prefix,
                stride_kbs, stride_vbs,
                n_full_blocks, n_extend_blocks,
                kt_half_smem, v_half_smem,
                qk_scale, LOGIT_CAP, IS_CAUSAL,
                USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
                BLOCK_M, BLOCK_N,
                BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV,
                BLOCK_DMODEL_HALF, BLOCK_DV_HALF,
                NUM_STAGES,
                kt_half_async_layout, kpe_async_layout, v_half_async_layout,
                kt_half_dot_layout, kt_dot_layout,
                p_dot_layout, v_half_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
        elif remaining_blocks > 0:
            acc_lo, acc_hi, l_i, m_i = _extend_subtile_serial(
                acc_lo, acc_hi, l_i, m_i,
                q_dot_lo, q_dot_hi, qpe_dot,
                k_ext_base, v_ext_base,
                cur_block_m, seq_len_extend, seq_len_prefix,
                stride_kbs, stride_vbs,
                n_full_blocks, n_extend_blocks,
                kt_half_smem, v_half_smem,
                qk_scale, LOGIT_CAP, IS_CAUSAL,
                USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
                BLOCK_M, BLOCK_N,
                BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV,
                BLOCK_DMODEL_HALF, BLOCK_DV_HALF,
                kt_half_async_layout, kpe_async_layout, v_half_async_layout,
                kt_half_dot_layout, kt_dot_layout,
                p_dot_layout, v_half_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )

        l_recip = 1.0 / l_i
        acc_lo = acc_lo * l_recip[:, None]
        acc_hi = acc_hi * l_recip[:, None]

        o_base = (
            O_Extend
            + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
            + cur_head * stride_oh
        )
        o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend

        out_lo = gl.convert_layout(acc_lo, blocked_layout).to(O_Extend.dtype.element_ty)
        out_hi = gl.convert_layout(acc_hi, blocked_layout).to(O_Extend.dtype.element_ty)
        offs_dv_lo = gl.arange(0, BLOCK_DV_HALF, layout=offs_d_layout)
        offs_dv_hi = BLOCK_DV_HALF + gl.arange(0, BLOCK_DV_HALF, layout=offs_d_layout)
        gl.store(o_base + offs_dv_lo[None, :], out_lo, mask=o_mask)
        gl.store(o_base + offs_dv_hi[None, :], out_hi, mask=o_mask)

    else:
        # --- Full-width extend path ---
        if n_full_blocks >= NUM_STAGES:
            acc, l_i, m_i = _extend_pipelined(
                acc, l_i, m_i, q_dot, qpe_dot,
                k_ext_base, v_ext_base,
                cur_block_m, seq_len_extend, seq_len_prefix,
                stride_kbs, stride_vbs,
                0, n_full_blocks,
                kt_smem, kpe_smem, v_smem,
                qk_scale, LOGIT_CAP, False,
                USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV,
                NUM_STAGES, STREAMS,
                kt_async_layout, kpe_async_layout, v_async_layout,
                kt_dot_layout, v_dot_layout, p_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
        elif n_full_blocks > 0:
            acc, l_i, m_i = _extend_serial(
                acc, l_i, m_i, q_dot, qpe_dot,
                k_ext_base, v_ext_base,
                cur_block_m, seq_len_extend, seq_len_prefix,
                stride_kbs, stride_vbs,
                0, n_full_blocks,
                kt_smem, kpe_smem, v_smem,
                qk_scale, LOGIT_CAP, False,
                USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV,
                kt_async_layout, kpe_async_layout, v_async_layout,
                kt_dot_layout, v_dot_layout, p_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )

        remaining_blocks = n_extend_blocks - n_full_blocks
        if remaining_blocks >= NUM_STAGES:
            acc, l_i, m_i = _extend_pipelined(
                acc, l_i, m_i, q_dot, qpe_dot,
                k_ext_base, v_ext_base,
                cur_block_m, seq_len_extend, seq_len_prefix,
                stride_kbs, stride_vbs,
                n_full_blocks, n_extend_blocks,
                kt_smem, kpe_smem, v_smem,
                qk_scale, LOGIT_CAP, IS_CAUSAL,
                USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV,
                NUM_STAGES, STREAMS,
                kt_async_layout, kpe_async_layout, v_async_layout,
                kt_dot_layout, v_dot_layout, p_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
        elif remaining_blocks > 0:
            acc, l_i, m_i = _extend_serial(
                acc, l_i, m_i, q_dot, qpe_dot,
                k_ext_base, v_ext_base,
                cur_block_m, seq_len_extend, seq_len_prefix,
                stride_kbs, stride_vbs,
                n_full_blocks, n_extend_blocks,
                kt_smem, kpe_smem, v_smem,
                qk_scale, LOGIT_CAP, IS_CAUSAL,
                USE_CUSTOM_MASK, Mask, mask_base_idx, mask_row_stride,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV,
                kt_async_layout, kpe_async_layout, v_async_layout,
                kt_dot_layout, v_dot_layout, p_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )

        # --- Output ---
        l_recip = 1.0 / l_i
        acc = acc * l_recip[:, None]
        out_bf16 = gl.convert_layout(acc, blocked_layout).to(O_Extend.dtype.element_ty)
        o_base = (
            O_Extend
            + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
            + cur_head * stride_oh
        )
        o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
        gl.store(o_base + offs_dv[None, :], out_bf16, mask=o_mask)


# ===-----------------------------------------------------------------------===#
# Async load helpers
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _issue_prefix_k_async(
    kt_smem, k_base, kv_locs, mask_n_kt,
    stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    """Issue K-nope load, one commit."""
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kv_locs_kt = gl.convert_layout(kv_locs, gl.SliceLayout(dim=0, parent=kt_async_layout))
    kt_offsets = (kt_offs_d[:, None] + kv_locs_kt[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        kt_smem, k_base, kt_offsets, mask=mask_n_kt[None, :], other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_k_native(
    kt_smem, k_base, kv_locs_native, mask_n_native,
    stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    """Issue K-nope load -- kv_locs already in kt N-slice layout."""
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offsets = (kt_offs_d[:, None] + kv_locs_native[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        kt_smem, k_base, kt_offsets, mask=mask_n_native[None, :], other=0.0
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
    """Issue K-rope load, one commit."""
    kpe_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
    kpe_offs_d = gl.arange(0, BLOCK_DPE, layout=kpe_offs_d_layout)
    kv_locs_kpe = gl.convert_layout(kv_locs, gl.SliceLayout(dim=0, parent=kpe_async_layout))
    kpe_offsets = ((kpe_offs_d[:, None] + BLOCK_DMODEL) + kv_locs_kpe[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        kpe_smem, k_base, kpe_offsets, mask=mask_n_kpe[None, :], other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_kpe_native(
    kpe_smem, k_base, kv_locs_native, mask_n_native,
    stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    kpe_async_layout: gl.constexpr,
):
    """Issue K-rope load -- kv_locs already in kpe N-slice layout."""
    kpe_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
    kpe_offs_d = gl.arange(0, BLOCK_DPE, layout=kpe_offs_d_layout)
    kpe_offsets = ((kpe_offs_d[:, None] + BLOCK_DMODEL) + kv_locs_native[None, :] * stride_kbs).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        kpe_smem, k_base, kpe_offsets, mask=mask_n_native[None, :], other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_prefix_kvkpe_async(
    kt_smem, kpe_smem, k_base, kv_locs, mask_n_kt, mask_n_kpe,
    stride_kbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    kt_async_layout: gl.constexpr, kpe_async_layout: gl.constexpr,
):
    """Issue K-nope and K-rope loads, one commit each (prologue helper)."""
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
    """Issue V load, one commit."""
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    kv_locs_v = gl.convert_layout(kv_locs, gl.SliceLayout(dim=1, parent=v_async_layout))
    v_offsets = (kv_locs_v[:, None] * stride_vbs + v_offs_d[None, :]).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        v_smem, v_base, v_offsets, mask=mask_n_v[:, None], other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _issue_v_native(
    v_smem, v_base, kv_locs_native, mask_n_native,
    stride_vbs,
    BLOCK_N: gl.constexpr, BLOCK_DV: gl.constexpr,
    v_async_layout: gl.constexpr,
):
    """Issue V load -- kv_locs already in v N-slice layout."""
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    v_offsets = (kv_locs_native[:, None] * stride_vbs + v_offs_d[None, :]).to(tl.int32)
    cdna4_async.buffer_load_to_shared(
        v_smem, v_base, v_offsets, mask=mask_n_native[:, None], other=0.0
    )
    cdna4_async.commit_group()


@gluon.jit
def _load_prefix_kv_locs(
    kv_indices, kv_start, start_n, seq_len,
    BLOCK_N: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    """Load kv_indices via buffer_load (scalar base + element offsets → register).

    Clamp offsets for masked lanes to 0 to avoid OOB buffer descriptor faults.
    """
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    n_idx = start_n + kt_offs_n
    mask_n = n_idx < seq_len
    safe_idx = gl.where(mask_n, kv_start + n_idx, gl.zeros([BLOCK_N], dtype=tl.int32, layout=kt_offs_n_layout))
    kv_locs = cdna_buffer_load(kv_indices, safe_idx.to(tl.int32), mask=mask_n, other=0)
    return kv_locs, mask_n


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
# Softmax helpers
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
# GQA Head-Sharing Kernel
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
    """Pipelined extend inner loop using kv_indices (unified buffer)."""
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
    """Serial extend inner loop using kv_indices (unified buffer)."""
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


@gluon.jit
def mla_d512_gqa_fwd(
    Q,
    KV_Buffer,
    O,
    qo_indptr,
    kv_indptr,
    kv_indices,
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
):
    """MLA D512 prefill kernel with unified KV buffer.

    Grid: (batch, num_q_heads, m_blocks).
    Each CTA processes one Q-head for one M-block.
    KV_Buffer: [total_kv, num_kv_heads, 576] where V = KV[:512].
    """
    BLOCK_M: gl.constexpr = 64
    BLOCK_N: gl.constexpr = 32
    BLOCK_DMODEL: gl.constexpr = 512
    BLOCK_DPE: gl.constexpr = 64
    BLOCK_DV: gl.constexpr = 512
    NUM_STAGES: gl.constexpr = 2
    num_warps: gl.constexpr = gl.num_warps()

    MMA_INSTR_M: gl.constexpr = 16
    MMA_INSTR_N: gl.constexpr = 16
    MMA_INSTR_K: gl.constexpr = 32
    QK_K_WIDTH: gl.constexpr = 8
    PV_K_WIDTH: gl.constexpr = 4
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

    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=QK_K_WIDTH
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=QK_K_WIDTH
    )
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=PV_K_WIDTH
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=PV_K_WIDTH
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

    kt_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    kt_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
        [0, 16],
        [0, 1], [0, 2], [0, 4], [0, 8],
    ]

    v_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_N, BLOCK_DV],
    )
    v_offset_bases: gl.constexpr = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
        [16, 0],
        [1, 0], [2, 0], [4, 0], [8, 0],
    ]

    kpe_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_DPE, BLOCK_N],
    )
    kpe_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        [0, 4], [0, 8], [0, 16],
        [0, 1], [0, 2],
    ]

    kpe_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_K]],
        offset_bases=kpe_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DPE, BLOCK_N],
    )
    kt_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_K]],
        offset_bases=kt_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    v_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_V]],
        offset_bases=v_offset_bases,
        cga_layout=[],
        shape=[BLOCK_N, BLOCK_DV],
    )

    kpe_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_DPE, BLOCK_N],
        layout=kpe_smem_layout,
    )
    kt_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
        layout=kt_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_N, BLOCK_DV],
        layout=v_smem_layout,
    )

    for _s in gl.static_range(NUM_STAGES):
        v_zero = gl.zeros(
            [BLOCK_N, BLOCK_DV],
            dtype=Q.dtype.element_ty,
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

    STREAMS: gl.constexpr = 3
    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    q_base = (
        Q
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_q_head * stride_qh
    )
    q_reg = gl.load(q_base + offs_d[None, :], mask=q_mask, other=0.0)
    qpe_reg = gl.load(q_base + offs_dpe[None, :], mask=q_mask, other=0.0)
    q_dot = gl.convert_layout(q_reg, q_dot_layout)
    qpe_dot = gl.convert_layout(qpe_reg, q_dot_layout)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

    if n_kv_blocks <= 0:
        pass
    elif n_kv_blocks <= NUM_STAGES:
        for tail_i in gl.static_range(NUM_STAGES):
            kv_locs_t, mask_n_t = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, tail_i * BLOCK_N, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_kpe_t = gl.convert_layout(mask_n_t, gl.SliceLayout(dim=0, parent=kpe_async_layout))
            mask_n_v_t = gl.convert_layout(mask_n_t, gl.SliceLayout(dim=1, parent=v_async_layout))
            _issue_prefix_kvkpe_async(
                kt_smem.index(tail_i), kpe_smem.index(tail_i), kv_base,
                kv_locs_t, mask_n_t, mask_n_kpe_t,
                stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
                kt_async_layout, kpe_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(tail_i), kv_base,
                kv_locs_t, mask_n_v_t,
                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

        cdna4_async.wait_group(0)

        for tail_i in gl.static_range(NUM_STAGES):
            start_n_t = tail_i * BLOCK_N
            if tail_i < n_kv_blocks:
                kt_d_t = cdna4_async.load_shared_relaxed(kt_smem.index(tail_i), kt_dot_layout)
                qk_t = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk_t = mfma_cdna4(q_dot, kt_d_t, qk_t)
                kpe_d_t = cdna4_async.load_shared_relaxed(kpe_smem.index(tail_i), kt_dot_layout)
                qk_t = mfma_cdna4(qpe_dot, kpe_d_t, qk_t)

                qk_s_t = qk_t * qk_scale
                n_offs_t = start_n_t + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                valid_t = n_offs_t[None, :] < causal_kv_end
                if IS_CAUSAL:
                    valid_t = valid_t & (q_abs_pos[:, None] >= n_offs_t[None, :])
                qk_s_t = gl.where(valid_t, qk_s_t, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

                m_ij_t = nan_propagating_max(qk_s_t, axis=1)
                m_new_t = gl.maximum(m_i, m_ij_t, propagate_nan=tl.PropagateNan.ALL)
                p_t = gl.exp2(qk_s_t - m_new_t[:, None])
                l_ij_t = gl.sum(p_t, axis=1)
                alpha_t = gl.exp2(m_i - m_new_t)
                l_i = l_i * alpha_t + l_ij_t
                acc = acc * alpha_t[:, None]
                m_i = m_new_t

                v_d_t = cdna4_async.load_shared_relaxed(v_smem.index(tail_i), v_dot_layout)
                p_cast_t = p_t.to(v_d_t.dtype)
                p_d_t = gl.convert_layout(p_cast_t, p_dot_layout)
                acc = mfma_cdna4(p_d_t, v_d_t, acc)
    else:
        for _s in gl.static_range(NUM_STAGES):
            kv_locs_pf, mask_n_pf = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, _s * BLOCK_N, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_kpe_pf = gl.convert_layout(mask_n_pf, gl.SliceLayout(dim=0, parent=kpe_async_layout))
            mask_n_v_pf = gl.convert_layout(mask_n_pf, gl.SliceLayout(dim=1, parent=v_async_layout))
            _issue_prefix_kvkpe_async(
                kt_smem.index(_s), kpe_smem.index(_s), kv_base,
                kv_locs_pf, mask_n_pf, mask_n_kpe_pf,
                stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
                kt_async_layout, kpe_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(_s), kv_base,
                kv_locs_pf, mask_n_v_pf,
                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

        main_loop_end = n_kv_blocks - NUM_STAGES
        cdna4_async.wait_group(WAIT_K)
        kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

        for block_n in tl.range(0, main_loop_end):
            stage = (block_n % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)

            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = mfma_cdna4(q_dot, kt_dot_reg, qk)
            kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
            qk = mfma_cdna4(qpe_dot, kpe_d, qk)

            cdna4_async.wait_group(WAIT_V)
            v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)

            future_n = (block_n + NUM_STAGES) * BLOCK_N
            kv_locs_nxt, mask_n_nxt = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, future_n, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_kpe_nxt = gl.convert_layout(mask_n_nxt, gl.SliceLayout(dim=0, parent=kpe_async_layout))
            mask_n_v_nxt = gl.convert_layout(mask_n_nxt, gl.SliceLayout(dim=1, parent=v_async_layout))

            _issue_prefix_kvkpe_async(
                kt_smem.index(stage), kpe_smem.index(stage), kv_base,
                kv_locs_nxt, mask_n_nxt, mask_n_kpe_nxt,
                stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
                kt_async_layout, kpe_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(stage), kv_base,
                kv_locs_nxt, mask_n_v_nxt,
                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

            qk_scaled = qk * qk_scale
            n_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            valid = n_offs[None, :] < causal_kv_end
            if IS_CAUSAL:
                valid = valid & (q_abs_pos[:, None] >= n_offs[None, :])
            qk_scaled = gl.where(valid, qk_scaled, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

            m_ij = nan_propagating_max(qk_scaled, axis=1)
            m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = gl.exp2(qk_scaled - m_new[:, None])
            l_ij = gl.sum(p, axis=1)
            alpha = gl.exp2(m_i - m_new)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            m_i = m_new

            p_cast = p.to(v_d.dtype)
            p_d = gl.convert_layout(p_cast, p_dot_layout)
            acc = mfma_cdna4(p_d, v_d, acc)

            cdna4_async.wait_group(WAIT_K)
            next_stage = ((block_n + 1) % NUM_STAGES).to(tl.int32)
            kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(next_stage), kt_dot_layout)

        cdna4_async.wait_group(0)
        for tail_i in gl.static_range(NUM_STAGES):
            stage = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
            start_n = ((main_loop_end + tail_i) * BLOCK_N).to(tl.int32)

            kt_d_tail = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
            qk_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk_tail = mfma_cdna4(q_dot, kt_d_tail, qk_tail)
            kpe_d_tail = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
            qk_tail = mfma_cdna4(qpe_dot, kpe_d_tail, qk_tail)

            qk_s_tail = qk_tail * qk_scale
            n_offs_tail = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            valid_tail = n_offs_tail[None, :] < causal_kv_end
            if IS_CAUSAL:
                valid_tail = valid_tail & (q_abs_pos[:, None] >= n_offs_tail[None, :])
            qk_s_tail = gl.where(valid_tail, qk_s_tail, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

            m_ij_tail = nan_propagating_max(qk_s_tail, axis=1)
            m_new_tail = gl.maximum(m_i, m_ij_tail, propagate_nan=tl.PropagateNan.ALL)
            p_tail = gl.exp2(qk_s_tail - m_new_tail[:, None])
            l_ij_tail = gl.sum(p_tail, axis=1)
            alpha_tail = gl.exp2(m_i - m_new_tail)
            l_i = l_i * alpha_tail + l_ij_tail
            acc = acc * alpha_tail[:, None]
            m_i = m_new_tail

            v_d_tail = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
            p_cast_tail = p_tail.to(v_d_tail.dtype)
            p_d_tail = gl.convert_layout(p_cast_tail, p_dot_layout)
            acc = mfma_cdna4(p_d_tail, v_d_tail, acc)

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


@gluon.jit
def mla_d512_gqa_fwd_large_tile(
    Q,
    KV_Buffer,
    O,
    qo_indptr,
    kv_indptr,
    kv_indices,
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
):
    """MLA D512 prefill kernel with BLOCK_M=128 (ASM-style large M-tile).

    Identical to mla_d512_gqa_fwd but with BLOCK_M=128 instead of 64.
    Better amortization of KV loads for long sequences.
    Grid: (batch, num_q_heads, m_blocks).
    """
    BLOCK_M: gl.constexpr = 128
    BLOCK_N: gl.constexpr = 32
    BLOCK_DMODEL: gl.constexpr = 512
    BLOCK_DPE: gl.constexpr = 64
    BLOCK_DV: gl.constexpr = 512
    NUM_STAGES: gl.constexpr = 2
    num_warps: gl.constexpr = gl.num_warps()

    MMA_INSTR_M: gl.constexpr = 16
    MMA_INSTR_N: gl.constexpr = 16
    MMA_INSTR_K: gl.constexpr = 32
    QK_K_WIDTH: gl.constexpr = 8
    PV_K_WIDTH: gl.constexpr = 4
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

    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=QK_K_WIDTH
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=QK_K_WIDTH
    )
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=PV_K_WIDTH
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=PV_K_WIDTH
    )

    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[2, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    kt_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    kt_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
        [0, 16],
        [0, 1], [0, 2], [0, 4], [0, 8],
    ]

    v_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_N, BLOCK_DV],
    )
    v_offset_bases: gl.constexpr = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
        [16, 0],
        [1, 0], [2, 0], [4, 0], [8, 0],
    ]

    kpe_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_DPE, BLOCK_N],
    )
    kpe_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
        [0, 4], [0, 8], [0, 16],
        [0, 1], [0, 2],
    ]

    kpe_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_K]],
        offset_bases=kpe_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DPE, BLOCK_N],
    )
    kt_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_K]],
        offset_bases=kt_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    v_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_V]],
        offset_bases=v_offset_bases,
        cga_layout=[],
        shape=[BLOCK_N, BLOCK_DV],
    )

    kpe_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_DPE, BLOCK_N],
        layout=kpe_smem_layout,
    )
    kt_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
        layout=kt_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_N, BLOCK_DV],
        layout=v_smem_layout,
    )

    for _s in gl.static_range(NUM_STAGES):
        v_zero = gl.zeros(
            [BLOCK_N, BLOCK_DV],
            dtype=Q.dtype.element_ty,
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

    STREAMS: gl.constexpr = 3
    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    q_base = (
        Q
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_q_head * stride_qh
    )
    q_reg = gl.load(q_base + offs_d[None, :], mask=q_mask, other=0.0)
    qpe_reg = gl.load(q_base + offs_dpe[None, :], mask=q_mask, other=0.0)
    q_dot = gl.convert_layout(q_reg, q_dot_layout)
    qpe_dot = gl.convert_layout(qpe_reg, q_dot_layout)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

    if n_kv_blocks <= 0:
        pass
    elif n_kv_blocks <= NUM_STAGES:
        for tail_i in gl.static_range(NUM_STAGES):
            kv_locs_t, mask_n_t = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, tail_i * BLOCK_N, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_kpe_t = gl.convert_layout(mask_n_t, gl.SliceLayout(dim=0, parent=kpe_async_layout))
            mask_n_v_t = gl.convert_layout(mask_n_t, gl.SliceLayout(dim=1, parent=v_async_layout))
            _issue_prefix_kvkpe_async(
                kt_smem.index(tail_i), kpe_smem.index(tail_i), kv_base,
                kv_locs_t, mask_n_t, mask_n_kpe_t,
                stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
                kt_async_layout, kpe_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(tail_i), kv_base,
                kv_locs_t, mask_n_v_t,
                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

        cdna4_async.wait_group(0)

        for tail_i in gl.static_range(NUM_STAGES):
            start_n_t = tail_i * BLOCK_N
            if tail_i < n_kv_blocks:
                kt_d_t = cdna4_async.load_shared_relaxed(kt_smem.index(tail_i), kt_dot_layout)
                qk_t = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk_t = mfma_cdna4(q_dot, kt_d_t, qk_t)
                kpe_d_t = cdna4_async.load_shared_relaxed(kpe_smem.index(tail_i), kt_dot_layout)
                qk_t = mfma_cdna4(qpe_dot, kpe_d_t, qk_t)

                qk_s_t = qk_t * qk_scale
                n_offs_t = start_n_t + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                valid_t = n_offs_t[None, :] < causal_kv_end
                if IS_CAUSAL:
                    valid_t = valid_t & (q_abs_pos[:, None] >= n_offs_t[None, :])
                qk_s_t = gl.where(valid_t, qk_s_t, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

                m_ij_t = nan_propagating_max(qk_s_t, axis=1)
                m_new_t = gl.maximum(m_i, m_ij_t, propagate_nan=tl.PropagateNan.ALL)
                p_t = gl.exp2(qk_s_t - m_new_t[:, None])
                l_ij_t = gl.sum(p_t, axis=1)
                alpha_t = gl.exp2(m_i - m_new_t)
                l_i = l_i * alpha_t + l_ij_t
                acc = acc * alpha_t[:, None]
                m_i = m_new_t

                v_d_t = cdna4_async.load_shared_relaxed(v_smem.index(tail_i), v_dot_layout)
                p_cast_t = p_t.to(v_d_t.dtype)
                p_d_t = gl.convert_layout(p_cast_t, p_dot_layout)
                acc = mfma_cdna4(p_d_t, v_d_t, acc)
    else:
        for _s in gl.static_range(NUM_STAGES):
            kv_locs_pf, mask_n_pf = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, _s * BLOCK_N, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_kpe_pf = gl.convert_layout(mask_n_pf, gl.SliceLayout(dim=0, parent=kpe_async_layout))
            mask_n_v_pf = gl.convert_layout(mask_n_pf, gl.SliceLayout(dim=1, parent=v_async_layout))
            _issue_prefix_kvkpe_async(
                kt_smem.index(_s), kpe_smem.index(_s), kv_base,
                kv_locs_pf, mask_n_pf, mask_n_kpe_pf,
                stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
                kt_async_layout, kpe_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(_s), kv_base,
                kv_locs_pf, mask_n_v_pf,
                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

        main_loop_end = n_kv_blocks - NUM_STAGES
        cdna4_async.wait_group(WAIT_K)
        kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

        for block_n in tl.range(0, main_loop_end):
            stage = (block_n % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)

            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = mfma_cdna4(q_dot, kt_dot_reg, qk)
            kpe_d = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
            qk = mfma_cdna4(qpe_dot, kpe_d, qk)

            cdna4_async.wait_group(WAIT_V)
            v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)

            future_n = (block_n + NUM_STAGES) * BLOCK_N
            kv_locs_nxt, mask_n_nxt = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, future_n, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_kpe_nxt = gl.convert_layout(mask_n_nxt, gl.SliceLayout(dim=0, parent=kpe_async_layout))
            mask_n_v_nxt = gl.convert_layout(mask_n_nxt, gl.SliceLayout(dim=1, parent=v_async_layout))

            _issue_prefix_kvkpe_async(
                kt_smem.index(stage), kpe_smem.index(stage), kv_base,
                kv_locs_nxt, mask_n_nxt, mask_n_kpe_nxt,
                stride_kvbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
                kt_async_layout, kpe_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(stage), kv_base,
                kv_locs_nxt, mask_n_v_nxt,
                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

            qk_scaled = qk * qk_scale
            n_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            valid = n_offs[None, :] < causal_kv_end
            if IS_CAUSAL:
                valid = valid & (q_abs_pos[:, None] >= n_offs[None, :])
            qk_scaled = gl.where(valid, qk_scaled, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

            m_ij = nan_propagating_max(qk_scaled, axis=1)
            m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = gl.exp2(qk_scaled - m_new[:, None])
            l_ij = gl.sum(p, axis=1)
            alpha = gl.exp2(m_i - m_new)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            m_i = m_new

            p_cast = p.to(v_d.dtype)
            p_d = gl.convert_layout(p_cast, p_dot_layout)
            acc = mfma_cdna4(p_d, v_d, acc)

            cdna4_async.wait_group(WAIT_K)
            next_stage = ((block_n + 1) % NUM_STAGES).to(tl.int32)
            kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(next_stage), kt_dot_layout)

        cdna4_async.wait_group(0)
        for tail_i in gl.static_range(NUM_STAGES):
            stage = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
            start_n = ((main_loop_end + tail_i) * BLOCK_N).to(tl.int32)

            kt_d_tail = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
            qk_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk_tail = mfma_cdna4(q_dot, kt_d_tail, qk_tail)
            kpe_d_tail = cdna4_async.load_shared_relaxed(kpe_smem.index(stage), kt_dot_layout)
            qk_tail = mfma_cdna4(qpe_dot, kpe_d_tail, qk_tail)

            qk_s_tail = qk_tail * qk_scale
            n_offs_tail = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            valid_tail = n_offs_tail[None, :] < causal_kv_end
            if IS_CAUSAL:
                valid_tail = valid_tail & (q_abs_pos[:, None] >= n_offs_tail[None, :])
            qk_s_tail = gl.where(valid_tail, qk_s_tail, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

            m_ij_tail = nan_propagating_max(qk_s_tail, axis=1)
            m_new_tail = gl.maximum(m_i, m_ij_tail, propagate_nan=tl.PropagateNan.ALL)
            p_tail = gl.exp2(qk_s_tail - m_new_tail[:, None])
            l_ij_tail = gl.sum(p_tail, axis=1)
            alpha_tail = gl.exp2(m_i - m_new_tail)
            l_i = l_i * alpha_tail + l_ij_tail
            acc = acc * alpha_tail[:, None]
            m_i = m_new_tail

            v_d_tail = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
            p_cast_tail = p_tail.to(v_d_tail.dtype)
            p_d_tail = gl.convert_layout(p_cast_tail, p_dot_layout)
            acc = mfma_cdna4(p_d_tail, v_d_tail, acc)

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


@gluon.jit
def mla_d512_gqa_fwd_bn16(
    Q,
    KV_Buffer,
    O,
    qo_indptr,
    kv_indptr,
    kv_indices,
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
):
    """MLA D512 prefill kernel with BLOCK_N=16 (ASM-style small N-tile).

    BLOCK_M=64, BLOCK_N=16, NUM_STAGES=3, 4 warps.
    KPE loaded from global (no async), K+V async pipelined.
    3-stage pipeline fits in LDS: K 48KB + V 48KB + pad ≈ 105KB < 160KB.
    QK uses v_mfma_f32_16x16x32_bf16, PV uses v_mfma_f32_16x16x16_bf16.
    """
    BLOCK_M: gl.constexpr = 64
    BLOCK_N: gl.constexpr = 16
    BLOCK_DMODEL: gl.constexpr = 512
    BLOCK_DPE: gl.constexpr = 64
    BLOCK_DV: gl.constexpr = 512
    NUM_STAGES: gl.constexpr = 3
    num_warps: gl.constexpr = gl.num_warps()

    MMA_INSTR_M: gl.constexpr = 16
    MMA_INSTR_N: gl.constexpr = 16
    QK_MMA_INSTR_K: gl.constexpr = 32
    PV_MMA_INSTR_K: gl.constexpr = 16
    QK_K_WIDTH: gl.constexpr = 8
    PV_K_WIDTH: gl.constexpr = 4
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

    # QK: [M, 512] x [512, 16] → uses 16x16x32 MFMA
    qk_mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, QK_MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    # PV: [M, 16] x [16, 512] → uses 16x16x16 MFMA
    pv_mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, PV_MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=qk_mma_layout, k_width=QK_K_WIDTH
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=qk_mma_layout, k_width=QK_K_WIDTH
    )
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=pv_mma_layout, k_width=PV_K_WIDTH
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=pv_mma_layout, k_width=PV_K_WIDTH
    )

    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=qk_mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=qk_mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=qk_mma_layout)
    pv_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=pv_mma_layout)

    # K^T async layout for [512, 16]: 3 reg D bits + 2 reg N bits,
    # 6 lane D bits, 2 warp N bits.  128-bit loads (8 bf16 contiguous on D).
    kt_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 1], [0, 2]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
        warp_bases=[[0, 4], [0, 8]],
        block_bases=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    kt_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
        [0, 1], [0, 2], [0, 4], [0, 8],
    ]

    # V async layout for [16, 512]: 3 reg D bits + 2 reg N bits,
    # 6 lane D bits, 2 warp N bits.  128-bit loads (8 bf16 contiguous on D).
    v_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
        warp_bases=[[4, 0], [8, 0]],
        block_bases=[],
        shape=[BLOCK_N, BLOCK_DV],
    )
    v_offset_bases: gl.constexpr = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
        [1, 0], [2, 0], [4, 0], [8, 0],
    ]

    kt_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_K]],
        offset_bases=kt_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    v_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_V]],
        offset_bases=v_offset_bases,
        cga_layout=[],
        shape=[BLOCK_N, BLOCK_DV],
    )

    kt_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
        layout=kt_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_N, BLOCK_DV],
        layout=v_smem_layout,
    )

    for _s in gl.static_range(NUM_STAGES):
        v_zero = gl.zeros(
            [BLOCK_N, BLOCK_DV],
            dtype=Q.dtype.element_ty,
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

    STREAMS: gl.constexpr = 2
    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    q_base = (
        Q
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_q_head * stride_qh
    )
    q_reg = gl.load(q_base + offs_d[None, :], mask=q_mask, other=0.0)
    qpe_reg = gl.load(q_base + offs_dpe[None, :], mask=q_mask, other=0.0)
    q_dot = gl.convert_layout(q_reg, q_dot_layout)
    qpe_dot = gl.convert_layout(qpe_reg, q_dot_layout)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=pv_mma_layout)

    if n_kv_blocks <= 0:
        pass
    elif n_kv_blocks <= NUM_STAGES:
        for tail_i in gl.static_range(NUM_STAGES):
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

        for tail_i in gl.static_range(NUM_STAGES):
            start_n_t = tail_i * BLOCK_N
            if tail_i < n_kv_blocks:
                kt_d_t = cdna4_async.load_shared_relaxed(kt_smem.index(tail_i), kt_dot_layout)
                qk_t = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=qk_mma_layout)
                qk_t = mfma_cdna4(q_dot, kt_d_t, qk_t)
                qk_t = _load_kpe_from_global(
                    qk_t, qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                    start_n_t, causal_kv_end, stride_kvbs,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kt_dot_layout,
                )

                qk_s_t = qk_t * qk_scale
                n_offs_t = start_n_t + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                valid_t = n_offs_t[None, :] < causal_kv_end
                if IS_CAUSAL:
                    valid_t = valid_t & (q_abs_pos[:, None] >= n_offs_t[None, :])
                qk_s_t = gl.where(valid_t, qk_s_t, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_mma_layout))

                m_ij_t = nan_propagating_max(qk_s_t, axis=1)
                m_new_t = gl.maximum(m_i, m_ij_t, propagate_nan=tl.PropagateNan.ALL)
                p_t = gl.exp2(qk_s_t - m_new_t[:, None])
                l_ij_t = gl.sum(p_t, axis=1)
                alpha_t = gl.exp2(m_i - m_new_t)
                l_i = l_i * alpha_t + l_ij_t
                alpha_t_pv = gl.convert_layout(alpha_t, pv_m_layout)
                acc = acc * alpha_t_pv[:, None]
                m_i = m_new_t

                v_d_t = cdna4_async.load_shared_relaxed(v_smem.index(tail_i), v_dot_layout)
                p_cast_t = p_t.to(v_d_t.dtype)
                p_d_t = gl.convert_layout(p_cast_t, p_dot_layout)
                acc = mfma_cdna4(p_d_t, v_d_t, acc)
    else:
        # Prologue: fill 3 stages
        for _s in gl.static_range(NUM_STAGES):
            kv_locs_pf, mask_n_pf = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, _s * BLOCK_N, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_v_pf = gl.convert_layout(mask_n_pf, gl.SliceLayout(dim=1, parent=v_async_layout))
            _issue_prefix_k_async(
                kt_smem.index(_s), kv_base,
                kv_locs_pf, mask_n_pf,
                stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(_s), kv_base,
                kv_locs_pf, mask_n_v_pf,
                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

        main_loop_end = n_kv_blocks - NUM_STAGES
        cdna4_async.wait_group(WAIT_K)
        kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

        for block_n in tl.range(0, main_loop_end):
            stage = (block_n % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)

            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=qk_mma_layout)
            qk = mfma_cdna4(q_dot, kt_dot_reg, qk)
            qk = _load_kpe_from_global(
                qk, qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                start_n, causal_kv_end, stride_kvbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kt_dot_layout,
            )

            cdna4_async.wait_group(WAIT_V)
            v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)

            future_n = (block_n + NUM_STAGES) * BLOCK_N
            kv_locs_nxt, mask_n_nxt = _load_prefix_kv_locs(
                kv_indices, cur_seq_kv_start, future_n, causal_kv_end,
                BLOCK_N, kt_async_layout,
            )
            mask_n_v_nxt = gl.convert_layout(mask_n_nxt, gl.SliceLayout(dim=1, parent=v_async_layout))

            _issue_prefix_k_async(
                kt_smem.index(stage), kv_base,
                kv_locs_nxt, mask_n_nxt,
                stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
            )
            _issue_prefix_v_async(
                v_smem.index(stage), kv_base,
                kv_locs_nxt, mask_n_v_nxt,
                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
            )

            qk_scaled = qk * qk_scale
            n_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            valid = n_offs[None, :] < causal_kv_end
            if IS_CAUSAL:
                valid = valid & (q_abs_pos[:, None] >= n_offs[None, :])
            qk_scaled = gl.where(valid, qk_scaled, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_mma_layout))

            m_ij = nan_propagating_max(qk_scaled, axis=1)
            m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
            p = gl.exp2(qk_scaled - m_new[:, None])
            l_ij = gl.sum(p, axis=1)
            alpha = gl.exp2(m_i - m_new)
            l_i = l_i * alpha + l_ij
            alpha_pv = gl.convert_layout(alpha, pv_m_layout)
            acc = acc * alpha_pv[:, None]
            m_i = m_new

            p_cast = p.to(v_d.dtype)
            p_d = gl.convert_layout(p_cast, p_dot_layout)
            acc = mfma_cdna4(p_d, v_d, acc)

            cdna4_async.wait_group(WAIT_K)
            next_stage = ((block_n + 1) % NUM_STAGES).to(tl.int32)
            kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(next_stage), kt_dot_layout)

        # Tail: drain pipeline
        cdna4_async.wait_group(0)
        for tail_i in gl.static_range(NUM_STAGES):
            stage = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
            start_n = ((main_loop_end + tail_i) * BLOCK_N).to(tl.int32)

            kt_d_tail = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
            qk_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=qk_mma_layout)
            qk_tail = mfma_cdna4(q_dot, kt_d_tail, qk_tail)
            qk_tail = _load_kpe_from_global(
                qk_tail, qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                start_n, causal_kv_end, stride_kvbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kt_dot_layout,
            )

            qk_s_tail = qk_tail * qk_scale
            n_offs_tail = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            valid_tail = n_offs_tail[None, :] < causal_kv_end
            if IS_CAUSAL:
                valid_tail = valid_tail & (q_abs_pos[:, None] >= n_offs_tail[None, :])
            qk_s_tail = gl.where(valid_tail, qk_s_tail, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_mma_layout))

            m_ij_tail = nan_propagating_max(qk_s_tail, axis=1)
            m_new_tail = gl.maximum(m_i, m_ij_tail, propagate_nan=tl.PropagateNan.ALL)
            p_tail = gl.exp2(qk_s_tail - m_new_tail[:, None])
            l_ij_tail = gl.sum(p_tail, axis=1)
            alpha_tail = gl.exp2(m_i - m_new_tail)
            l_i = l_i * alpha_tail + l_ij_tail
            alpha_tail_pv = gl.convert_layout(alpha_tail, pv_m_layout)
            acc = acc * alpha_tail_pv[:, None]
            m_i = m_new_tail

            v_d_tail = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
            p_cast_tail = p_tail.to(v_d_tail.dtype)
            p_d_tail = gl.convert_layout(p_cast_tail, p_dot_layout)
            acc = mfma_cdna4(p_d_tail, v_d_tail, acc)

    l_recip = 1.0 / l_i
    l_recip_pv = gl.convert_layout(l_recip, pv_m_layout)
    acc = acc * l_recip_pv[:, None]
    out_bf16 = gl.convert_layout(acc, blocked_layout).to(O.dtype.element_ty)
    o_base = (
        O
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_q_head * stride_oh
    )
    o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    gl.store(o_base + offs_dv[None, :], out_bf16, mask=o_mask)


@gluon.jit
def _load_kpe_from_global(
    qk, qpe_dot, kv_base, kv_indices, kv_start, start_n, causal_kv_end,
    stride_kvbs,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    kt_dot_layout: gl.constexpr,
):
    """Load KPE from global memory and accumulate into qk."""
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_dot_layout))
    offs_dpe = gl.arange(0, BLOCK_DPE)
    n_idx = start_n + offs_n
    mask_n = n_idx < causal_kv_end
    kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)
    kpe_ptrs = kv_base + kv_locs[None, :] * stride_kvbs + BLOCK_DMODEL + offs_dpe[:, None]
    kpe = gl.load(kpe_ptrs, mask=mask_n[None, :], other=0.0)
    kpe_dot = gl.convert_layout(kpe, kt_dot_layout)
    qk = mfma_cdna4(qpe_dot, kpe_dot, qk)
    return qk


@gluon.jit
def mla_d512_gqa_fwd_8w(
    Q,
    KV_Buffer,
    O,
    qo_indptr,
    kv_indptr,
    kv_indices,
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
):
    """MLA D512 prefill kernel -- 8 warp pingpong variant.

    Grid: (batch, num_q_heads, m_blocks).
    BLOCK_M=128, 8 warps, STREAMS=2 (K+V async, KPE from global).
    Uses warp_pipeline_stage for compute/memory overlap.
    """
    BLOCK_M: gl.constexpr = 128
    BLOCK_N: gl.constexpr = 32
    BLOCK_DMODEL: gl.constexpr = 512
    BLOCK_DPE: gl.constexpr = 64
    BLOCK_DV: gl.constexpr = 512
    NUM_STAGES: gl.constexpr = 2
    num_warps: gl.constexpr = gl.num_warps()

    MMA_INSTR_M: gl.constexpr = 16
    MMA_INSTR_N: gl.constexpr = 16
    MMA_INSTR_K: gl.constexpr = 32
    QK_K_WIDTH: gl.constexpr = 8
    PV_K_WIDTH: gl.constexpr = 4
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

    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )

    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=QK_K_WIDTH
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=QK_K_WIDTH
    )
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=PV_K_WIDTH
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=PV_K_WIDTH
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

    # 8-warp K async: BLOCK_DMODEL>=512, BLOCK_N<64
    kt_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
        [0, 16],
        [0, 1], [0, 2], [0, 4], [0, 8],
    ]
    kt_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 16]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
        warp_bases=[[0, 1], [0, 2], [0, 4]],
        block_bases=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )

    # 8-warp V async: BLOCK_DV>=512, BLOCK_N<64
    v_offset_bases: gl.constexpr = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
        [16, 0],
        [1, 0], [2, 0], [4, 0], [8, 0],
    ]
    v_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [16, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
        warp_bases=[[1, 0], [2, 0], [4, 0]],
        block_bases=[],
        shape=[BLOCK_N, BLOCK_DV],
    )

    kt_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_K]],
        offset_bases=kt_offset_bases,
        cga_layout=[],
        shape=[BLOCK_DMODEL, BLOCK_N],
    )
    v_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_V]],
        offset_bases=v_offset_bases,
        cga_layout=[],
        shape=[BLOCK_N, BLOCK_DV],
    )

    kt_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
        layout=kt_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty,
        [NUM_STAGES, BLOCK_N, BLOCK_DV],
        layout=v_smem_layout,
    )

    for _s in gl.static_range(NUM_STAGES):
        v_zero = gl.zeros(
            [BLOCK_N, BLOCK_DV],
            dtype=Q.dtype.element_ty,
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

    STREAMS: gl.constexpr = 2
    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    q_base = (
        Q
        + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_q_head * stride_qh
    )
    q_reg = gl.load(q_base + offs_d[None, :], mask=q_mask, other=0.0)
    qpe_reg = gl.load(q_base + offs_dpe[None, :], mask=q_mask, other=0.0)
    q_dot = gl.convert_layout(q_reg, q_dot_layout)
    qpe_dot = gl.convert_layout(qpe_reg, q_dot_layout)

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

    kt_offs_n_pf = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_async_layout))
    v_offs_n_pf = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=v_async_layout))

    if n_kv_blocks <= 0:
        pass
    elif n_kv_blocks <= NUM_STAGES:
        for tail_i in gl.static_range(NUM_STAGES):
            sn = tail_i * BLOCK_N
            kt_idx = sn + kt_offs_n_pf
            kt_mask = kt_idx < causal_kv_end
            kt_locs = gl.load(kv_indices + cur_seq_kv_start + kt_idx, mask=kt_mask, other=0).to(tl.int32)
            _issue_k_native(kt_smem.index(tail_i), kv_base, kt_locs, kt_mask,
                            stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout)
            v_idx = sn + v_offs_n_pf
            v_mask = v_idx < causal_kv_end
            v_locs = gl.load(kv_indices + cur_seq_kv_start + v_idx, mask=v_mask, other=0).to(tl.int32)
            _issue_v_native(v_smem.index(tail_i), kv_base, v_locs, v_mask,
                            stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout)

        cdna4_async.wait_group(0)

        for tail_i in gl.static_range(NUM_STAGES):
            start_n_t = tail_i * BLOCK_N
            if tail_i < n_kv_blocks:
                kt_d_t = cdna4_async.load_shared_relaxed(kt_smem.index(tail_i), kt_dot_layout)
                qk_t = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk_t = mfma_cdna4(q_dot, kt_d_t, qk_t)
                qk_t = _load_kpe_from_global(
                    qk_t, qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                    start_n_t, causal_kv_end, stride_kvbs,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kt_dot_layout,
                )

                qk_s_t = qk_t * qk_scale
                n_offs_t = start_n_t + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                valid_t = n_offs_t[None, :] < causal_kv_end
                if IS_CAUSAL:
                    valid_t = valid_t & (q_abs_pos[:, None] >= n_offs_t[None, :])
                qk_s_t = gl.where(valid_t, qk_s_t, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

                m_ij_t = nan_propagating_max(qk_s_t, axis=1)
                m_new_t = gl.maximum(m_i, m_ij_t, propagate_nan=tl.PropagateNan.ALL)
                p_t = gl.exp2(qk_s_t - m_new_t[:, None])
                l_ij_t = gl.sum(p_t, axis=1)
                alpha_t = gl.exp2(m_i - m_new_t)
                l_i = l_i * alpha_t + l_ij_t
                acc = acc * alpha_t[:, None]
                m_i = m_new_t

                v_d_t = cdna4_async.load_shared_relaxed(v_smem.index(tail_i), v_dot_layout)
                p_cast_t = p_t.to(v_d_t.dtype)
                p_d_t = gl.convert_layout(p_cast_t, p_dot_layout)
                acc = mfma_cdna4(p_d_t, v_d_t, acc)
    else:
        for _s in gl.static_range(NUM_STAGES):
            sn = _s * BLOCK_N
            kt_idx = sn + kt_offs_n_pf
            kt_mask = kt_idx < causal_kv_end
            kt_locs = gl.load(kv_indices + cur_seq_kv_start + kt_idx, mask=kt_mask, other=0).to(tl.int32)
            _issue_k_native(kt_smem.index(_s), kv_base, kt_locs, kt_mask,
                            stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout)
            v_idx = sn + v_offs_n_pf
            v_mask = v_idx < causal_kv_end
            v_locs = gl.load(kv_indices + cur_seq_kv_start + v_idx, mask=v_mask, other=0).to(tl.int32)
            _issue_v_native(v_smem.index(_s), kv_base, v_locs, v_mask,
                            stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout)

        main_loop_end = n_kv_blocks - NUM_STAGES
        cdna4_async.wait_group(WAIT_K)
        kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

        # === Pingpong main loop (2 streams: K+V) ===
        for block_n in tl.range(0, main_loop_end, loop_unroll_factor=2):

            with warp_pipeline_stage("compute0", priority=0):
                stage = (block_n % NUM_STAGES).to(tl.int32)
                start_n = (block_n * BLOCK_N).to(tl.int32)
                qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk = mfma_cdna4(q_dot, kt_dot_reg, qk)
                qk = _load_kpe_from_global(
                    qk, qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                    start_n, causal_kv_end, stride_kvbs,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kt_dot_layout,
                )

            cdna4_async.wait_group(WAIT_V)

            with warp_pipeline_stage("memory0", priority=1):
                v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
                future_n = ((block_n + NUM_STAGES) * BLOCK_N).to(tl.int32)
                fut_kt_idx = future_n + kt_offs_n_pf
                fut_kt_mask = fut_kt_idx < causal_kv_end
                fut_kt_locs = gl.load(kv_indices + cur_seq_kv_start + fut_kt_idx, mask=fut_kt_mask, other=0).to(tl.int32)
                _issue_k_native(kt_smem.index(stage), kv_base, fut_kt_locs, fut_kt_mask,
                                stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout)

            with warp_pipeline_stage("compute1", priority=0):
                qk_scaled = qk * qk_scale
                n_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                valid = n_offs[None, :] < causal_kv_end
                if IS_CAUSAL:
                    valid = valid & (q_abs_pos[:, None] >= n_offs[None, :])
                qk_scaled = gl.where(valid, qk_scaled, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

                m_ij = nan_propagating_max(qk_scaled, axis=1)
                m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
                p = gl.exp2(qk_scaled - m_new[:, None])
                l_ij = gl.sum(p, axis=1)
                alpha = gl.exp2(m_i - m_new)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                m_i = m_new

            with warp_pipeline_stage("memory1", priority=1):
                fut_v_idx = future_n + v_offs_n_pf
                fut_v_mask = fut_v_idx < causal_kv_end
                fut_v_locs = gl.load(kv_indices + cur_seq_kv_start + fut_v_idx, mask=fut_v_mask, other=0).to(tl.int32)
                _issue_v_native(v_smem.index(stage), kv_base, fut_v_locs, fut_v_mask,
                                stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout)

            with warp_pipeline_stage("compute2", priority=0):
                p_cast = p.to(v_d.dtype)
                p_d = gl.convert_layout(p_cast, p_dot_layout)
                acc = mfma_cdna4(p_d, v_d, acc)

            cdna4_async.wait_group(WAIT_K)

            with warp_pipeline_stage("memory2", priority=1):
                next_stage = ((block_n + 1) % NUM_STAGES).to(tl.int32)
                kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(next_stage), kt_dot_layout)

        # === Tail: drain pipeline ===
        for tail_i in gl.static_range(NUM_STAGES):
            cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - (STREAMS - 1))
            stage = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
            start_n = ((main_loop_end + tail_i) * BLOCK_N).to(tl.int32)

            kt_d_tail = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
            qk_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk_tail = mfma_cdna4(q_dot, kt_d_tail, qk_tail)
            qk_tail = _load_kpe_from_global(
                qk_tail, qpe_dot, kv_base, kv_indices, cur_seq_kv_start,
                start_n, causal_kv_end, stride_kvbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kt_dot_layout,
            )

            qk_s_tail = qk_tail * qk_scale
            n_offs_tail = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            valid_tail = n_offs_tail[None, :] < causal_kv_end
            if IS_CAUSAL:
                valid_tail = valid_tail & (q_abs_pos[:, None] >= n_offs_tail[None, :])
            qk_s_tail = gl.where(valid_tail, qk_s_tail, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

            m_ij_tail = nan_propagating_max(qk_s_tail, axis=1)
            m_new_tail = gl.maximum(m_i, m_ij_tail, propagate_nan=tl.PropagateNan.ALL)
            p_tail = gl.exp2(qk_s_tail - m_new_tail[:, None])
            l_ij_tail = gl.sum(p_tail, axis=1)
            alpha_tail = gl.exp2(m_i - m_new_tail)
            l_i = l_i * alpha_tail + l_ij_tail
            acc = acc * alpha_tail[:, None]
            m_i = m_new_tail

            cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - STREAMS)
            v_d_tail = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
            p_cast_tail = p_tail.to(v_d_tail.dtype)
            p_d_tail = gl.convert_layout(p_cast_tail, p_dot_layout)
            acc = mfma_cdna4(p_d_tail, v_d_tail, acc)

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
# Python wrappers
# ===-----------------------------------------------------------------------===#


def mla_d512_gqa_attention_fwd(
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
    use_8warp=False,
    use_large_tile=False,
    use_bn16=False,
):
    """MLA prefill wrapper matching production mla_prefill_fwd interface.

    q:          [total_q_tokens, num_q_heads, 576]
    kv_buffer:  [total_kv_tokens, num_kv_heads, 576]  (unified: K=[:576], V=[:512])
    o:          [total_q_tokens, num_q_heads, 512]
    use_8warp:      Use 8-warp pingpong variant.
    use_large_tile: Use BLOCK_M=128 variant.
    use_bn16:       Use BLOCK_N=16 (ASM-style) variant with 3-stage pipeline.
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

    sm_scale = sm_scale or (1.0 / math.sqrt(Lq))

    if use_8warp:
        BLOCK_M = 128
    elif use_large_tile:
        BLOCK_M = 128
    else:
        BLOCK_M = 64

    n_m_blocks = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    grid = (batch_size, num_q_heads, n_m_blocks)

    _kernel_args = dict(
        IS_CAUSAL=is_causal,
        LOGIT_CAP=logit_cap,
        GQA_RATIO=gqa_ratio,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
    )
    _tensor_args = (
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        q.stride(0), q.stride(1),
        kv_buffer.stride(0), kv_buffer.stride(1),
        o.stride(0), o.stride(1),
    )

    if use_8warp:
        mla_d512_gqa_fwd_8w[grid](*_tensor_args, **_kernel_args, num_warps=8)
    elif use_large_tile:
        mla_d512_gqa_fwd_large_tile[grid](*_tensor_args, **_kernel_args, num_warps=4)
    elif use_bn16:
        mla_d512_gqa_fwd_bn16[grid](*_tensor_args, **_kernel_args, num_warps=4)
    else:
        mla_d512_gqa_fwd[grid](*_tensor_args, **_kernel_args, num_warps=4)


@gluon.jit
def mla_d512_gqa_fwd_wca(
    Q, KV_Buffer, O,
    qo_indptr, kv_indptr, kv_indices,
    sm_scale,
    stride_qbs, stride_qh, stride_kvbs, stride_kvh, stride_obs, stride_oh,
    partial_out, partial_lse,
    total_valid_tiles, total_programs,
    num_heads, n_m_tiles,
    IS_CAUSAL: gl.constexpr, GQA_RATIO: gl.constexpr, SPLIT_K: gl.constexpr,
):
    """Persistent WCA MLA D512 kernel with optional split-K over unified KV range.

    Uses uniform arithmetic tile schedule: tile = (seq, head, m_block).
    """
    BLOCK_M: gl.constexpr = 64
    BLOCK_N: gl.constexpr = 32
    BLOCK_DMODEL: gl.constexpr = 512
    BLOCK_DPE: gl.constexpr = 64
    BLOCK_DV: gl.constexpr = 512
    NUM_STAGES: gl.constexpr = 2
    num_warps: gl.constexpr = gl.num_warps()

    MMA_INSTR_M: gl.constexpr = 16
    MMA_INSTR_N: gl.constexpr = 16
    MMA_INSTR_K: gl.constexpr = 32
    QK_K_WIDTH: gl.constexpr = 8
    PV_K_WIDTH: gl.constexpr = 4
    ASYNC_PAD_K: gl.constexpr = 8
    ASYNC_PAD_V: gl.constexpr = 32

    cta_id = gl.program_id(0)

    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4, instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True, warps_per_cta=[num_warps, 1],
    )
    q_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=QK_K_WIDTH)
    kt_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=QK_K_WIDTH)
    p_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=PV_K_WIDTH)
    v_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=PV_K_WIDTH)
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1], order=[1, 0],
    )
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    kt_offset_bases: gl.constexpr = [
        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
        [0, 16],
        [0, 1], [0, 2], [0, 4], [0, 8],
    ]
    kt_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[], shape=[BLOCK_DMODEL, BLOCK_N],
    )
    v_offset_bases: gl.constexpr = [
        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
        [16, 0],
        [1, 0], [2, 0], [4, 0], [8, 0],
    ]
    v_async_layout: gl.constexpr = DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[], shape=[BLOCK_N, BLOCK_DV],
    )
    kt_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_K]], offset_bases=kt_offset_bases,
        cga_layout=[], shape=[BLOCK_DMODEL, BLOCK_N],
    )
    v_smem_layout: gl.constexpr = PaddedSharedLayout(
        interval_padding_pairs=[[512, ASYNC_PAD_V]], offset_bases=v_offset_bases,
        cga_layout=[], shape=[BLOCK_N, BLOCK_DV],
    )
    kt_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty, [NUM_STAGES, BLOCK_DMODEL, BLOCK_N], layout=kt_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        Q.dtype.element_ty, [NUM_STAGES, BLOCK_N, BLOCK_DV], layout=v_smem_layout,
    )
    for _s in gl.static_range(NUM_STAGES):
        v_zero = gl.zeros([BLOCK_N, BLOCK_DV], dtype=Q.dtype.element_ty, layout=v_async_layout)
        v_smem.index(_s).store(v_zero)
    gl.barrier()

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=offs_d_layout)
    offs_dpe = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=offs_d_layout)

    STREAMS: gl.constexpr = 2
    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

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

        orig_causal_kv_end = causal_kv_end

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
        q_dot = gl.convert_layout(q_reg, q_dot_layout)
        qpe_dot = gl.convert_layout(qpe_reg, q_dot_layout)

        q_abs_pos = (
            (seq_len_kv - seq_len_extend)
            + cur_block_m * BLOCK_M
            + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        )

        m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
        l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
        acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

        if n_kv_blocks > 0:
            if n_kv_blocks <= NUM_STAGES:
                for tail_i in gl.static_range(NUM_STAGES):
                    kv_locs_t, mask_n_t = _load_prefix_kv_locs(
                        kv_indices, cur_seq_kv_start + split_kv_offset,
                        tail_i * BLOCK_N, causal_kv_end, BLOCK_N, kt_async_layout,
                    )
                    mask_n_v_t = gl.convert_layout(mask_n_t, gl.SliceLayout(dim=1, parent=v_async_layout))
                    _issue_prefix_k_async(
                        kt_smem.index(tail_i), kv_base, kv_locs_t, mask_n_t,
                        stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
                    )
                    _issue_prefix_v_async(
                        v_smem.index(tail_i), kv_base, kv_locs_t, mask_n_v_t,
                        stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
                    )
                cdna4_async.wait_group(0)

                for tail_i in gl.static_range(NUM_STAGES):
                    start_n_t = tail_i * BLOCK_N
                    if tail_i < n_kv_blocks:
                        kt_d_t = cdna4_async.load_shared_relaxed(kt_smem.index(tail_i), kt_dot_layout)
                        qk_t = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                        qk_t = mfma_cdna4(q_dot, kt_d_t, qk_t)
                        qk_t = _load_kpe_from_global(
                            qk_t, qpe_dot, kv_base, kv_indices,
                            cur_seq_kv_start + split_kv_offset,
                            start_n_t, causal_kv_end, stride_kvbs,
                            BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kt_dot_layout,
                        )
                        qk_s_t = qk_t * qk_scale
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
                        v_d_t = cdna4_async.load_shared_relaxed(v_smem.index(tail_i), v_dot_layout)
                        p_cast_t = p_t.to(v_d_t.dtype)
                        p_d_t = gl.convert_layout(p_cast_t, p_dot_layout)
                        acc = mfma_cdna4(p_d_t, v_d_t, acc)
            else:
                for _s in gl.static_range(NUM_STAGES):
                    kv_locs_pf, mask_n_pf = _load_prefix_kv_locs(
                        kv_indices, cur_seq_kv_start + split_kv_offset,
                        _s * BLOCK_N, causal_kv_end, BLOCK_N, kt_async_layout,
                    )
                    mask_n_v_pf = gl.convert_layout(mask_n_pf, gl.SliceLayout(dim=1, parent=v_async_layout))
                    _issue_prefix_k_async(
                        kt_smem.index(_s), kv_base, kv_locs_pf, mask_n_pf,
                        stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
                    )
                    _issue_prefix_v_async(
                        v_smem.index(_s), kv_base, kv_locs_pf, mask_n_v_pf,
                        stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
                    )

                main_loop_end = n_kv_blocks - NUM_STAGES
                cdna4_async.wait_group(WAIT_K)
                kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

                for block_n in tl.range(0, main_loop_end):
                    stage = (block_n % NUM_STAGES).to(tl.int32)
                    start_n = (block_n * BLOCK_N).to(tl.int32)

                    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                    qk = mfma_cdna4(q_dot, kt_dot_reg, qk)
                    qk = _load_kpe_from_global(
                        qk, qpe_dot, kv_base, kv_indices,
                        cur_seq_kv_start + split_kv_offset,
                        start_n, causal_kv_end, stride_kvbs,
                        BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kt_dot_layout,
                    )
                    cdna4_async.wait_group(WAIT_V)
                    v_d = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)

                    future_n = (block_n + NUM_STAGES) * BLOCK_N
                    kv_locs_nxt, mask_n_nxt = _load_prefix_kv_locs(
                        kv_indices, cur_seq_kv_start + split_kv_offset,
                        future_n, causal_kv_end, BLOCK_N, kt_async_layout,
                    )
                    mask_n_v_nxt = gl.convert_layout(mask_n_nxt, gl.SliceLayout(dim=1, parent=v_async_layout))
                    _issue_prefix_k_async(
                        kt_smem.index(stage), kv_base, kv_locs_nxt, mask_n_nxt,
                        stride_kvbs, BLOCK_N, BLOCK_DMODEL, kt_async_layout,
                    )
                    _issue_prefix_v_async(
                        v_smem.index(stage), kv_base, kv_locs_nxt, mask_n_v_nxt,
                        stride_kvbs, BLOCK_N, BLOCK_DV, v_async_layout,
                    )

                    qk_scaled = qk * qk_scale
                    abs_n = split_kv_offset + start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                    valid = (start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col))[None, :] < causal_kv_end
                    if IS_CAUSAL:
                        valid = valid & (q_abs_pos[:, None] >= abs_n[None, :])
                    qk_scaled = gl.where(valid, qk_scaled, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

                    m_ij = nan_propagating_max(qk_scaled, axis=1)
                    m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
                    p = gl.exp2(qk_scaled - m_new[:, None])
                    l_ij = gl.sum(p, axis=1)
                    alpha = gl.exp2(m_i - m_new)
                    l_i = l_i * alpha + l_ij
                    acc = acc * alpha[:, None]
                    m_i = m_new
                    p_cast = p.to(v_d.dtype)
                    p_d = gl.convert_layout(p_cast, p_dot_layout)
                    acc = mfma_cdna4(p_d, v_d, acc)

                    cdna4_async.wait_group(WAIT_K)
                    next_stage = ((block_n + 1) % NUM_STAGES).to(tl.int32)
                    kt_dot_reg = cdna4_async.load_shared_relaxed(kt_smem.index(next_stage), kt_dot_layout)

                cdna4_async.wait_group(0)
                for tail_i in gl.static_range(NUM_STAGES):
                    stage = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
                    start_n = ((main_loop_end + tail_i) * BLOCK_N).to(tl.int32)

                    kt_d_tail = cdna4_async.load_shared_relaxed(kt_smem.index(stage), kt_dot_layout)
                    qk_tail = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                    qk_tail = mfma_cdna4(q_dot, kt_d_tail, qk_tail)
                    qk_tail = _load_kpe_from_global(
                        qk_tail, qpe_dot, kv_base, kv_indices,
                        cur_seq_kv_start + split_kv_offset,
                        start_n, causal_kv_end, stride_kvbs,
                        BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, kt_dot_layout,
                    )
                    qk_s_tail = qk_tail * qk_scale
                    abs_n_tail = split_kv_offset + start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                    valid_tail = (start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col))[None, :] < causal_kv_end
                    if IS_CAUSAL:
                        valid_tail = valid_tail & (q_abs_pos[:, None] >= abs_n_tail[None, :])
                    qk_s_tail = gl.where(valid_tail, qk_s_tail, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))

                    m_ij_tail = nan_propagating_max(qk_s_tail, axis=1)
                    m_new_tail = gl.maximum(m_i, m_ij_tail, propagate_nan=tl.PropagateNan.ALL)
                    p_tail = gl.exp2(qk_s_tail - m_new_tail[:, None])
                    l_ij_tail = gl.sum(p_tail, axis=1)
                    alpha_tail = gl.exp2(m_i - m_new_tail)
                    l_i = l_i * alpha_tail + l_ij_tail
                    acc = acc * alpha_tail[:, None]
                    m_i = m_new_tail
                    v_d_tail = cdna4_async.load_shared_relaxed(v_smem.index(stage), v_dot_layout)
                    p_cast_tail = p_tail.to(v_d_tail.dtype)
                    p_d_tail = gl.convert_layout(p_cast_tail, p_dot_layout)
                    acc = mfma_cdna4(p_d_tail, v_d_tail, acc)

        if SPLIT_K > 1:
            # Sanitize NaN from -inf - (-inf) in rows where all tokens were masked.
            # For those rows: m_i=-inf, acc=NaN, l_i=NaN.
            # The reduction handles lse=-inf (zero weight) correctly.
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


@triton.jit
def _mla_splitk_reduce(
    partial_out_ptr, partial_lse_ptr,
    O, qo_indptr,
    num_heads, n_m_tiles,
    stride_obs, stride_oh,
    SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DV: tl.constexpr,
):
    """Combine SPLIT_K partial MLA attention results via log-sum-exp reduction."""
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

    o_ptrs = (
        O + (cur_seq_q_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh + offs_dv[None, :]
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_mask[:, None])


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


def mla_d512_gqa_attention_fwd_wca(
    q, kv_buffer, o,
    qo_indptr, kv_indptr, kv_indices,
    max_len_extend=None,
    is_causal=True,
    sm_scale=None,
    split_k=None,
):
    """WCA (persistent + split-K) MLA prefill wrapper.

    When split_k is None, auto-selects based on CU utilization.
    """
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

    sm_scale = sm_scale or (1.0 / math.sqrt(Lq))

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

    mla_d512_gqa_fwd_wca[grid](
        q, kv_buffer, o,
        qo_indptr, kv_indptr, kv_indices,
        sm_scale,
        q.stride(0), q.stride(1),
        kv_buffer.stride(0), kv_buffer.stride(1),
        o.stride(0), o.stride(1),
        po_flat, pl_flat,
        total_valid_tiles, total_programs,
        num_q_heads, n_m_tiles,
        IS_CAUSAL=is_causal, GQA_RATIO=gqa_ratio, SPLIT_K=SPLIT_K,
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
            num_warps=4,
        )


def mla_d512_extend_attention_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask=None,
    mask_indptr=None,
    max_len_extend=None,
    is_causal=True,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    use_subtile=False,
):
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    assert Lq == 576 and Lv == 512, f"mla_d512 kernel requires Lq=576, Lv=512, got Lq={Lq}, Lv={Lv}"

    batch_size = qo_indptr.shape[0] - 1
    head_num = q_extend.shape[1]
    kv_group_num = head_num // k_extend.shape[1]

    if max_len_extend is None:
        extend_lens = qo_indptr[1:] - qo_indptr[:-1]
        max_len_extend = int(extend_lens.max().item())

    BLOCK_M = 64
    sm_scale = sm_scale or (1.0 / math.sqrt(Lq))
    USE_CUSTOM_MASK = custom_mask is not None

    dummy_mask = torch.empty(0, dtype=torch.uint8, device=q_extend.device)
    dummy_mask_indptr = torch.zeros(batch_size + 1, dtype=torch.int64, device=q_extend.device)
    if not USE_CUSTOM_MASK:
        custom_mask = dummy_mask
        mask_indptr = dummy_mask_indptr

    n_m_blocks = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    grid = (batch_size, head_num, n_m_blocks)

    mla_d512_extend_fwd[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sm_scale,
        kv_group_num,
        q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        o_extend.stride(0), o_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
        LOGIT_CAP=logit_cap,
        ENABLE_MASK_SPLIT=is_causal and not USE_CUSTOM_MASK,
        USE_SUBTILE=use_subtile,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
    )
