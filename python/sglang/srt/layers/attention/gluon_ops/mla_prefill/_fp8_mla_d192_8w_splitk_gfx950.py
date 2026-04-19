#!/usr/bin/env python3
"""FP8 D192 MLA prefill — 8-warp persistent + split-K (gfx950).

Persistent scheduling with KV-dimension split-K parallelism and inline
log-sum-exp reduction.  When SPLIT_K == 1 this is a pure persistent kernel
with zero workspace overhead.

Inner loop is identical to the base 8-warp kernel:
  3-stream async DMA (K_nope + K_rope + V), warp_pipeline_stage,
  BLOCK_M=128, BLOCK_N=128, NUM_STAGES=2.

Split-K strategy:
  total_valid_tiles = output_tiles * SPLIT_K
  tile_idx → (output_tile, split_id)
  Each split processes blocks [split_lo, split_hi) of the KV range.
  Partials stored as FP32 (normalized O + log2-LSE) in workspace.
  Last split to finish does inline merge via atomic counter.

Workspace (only when SPLIT_K > 1):
  partial_o   [output_tiles * SPLIT_K, BLOCK_M, D_V]  FP32
  partial_lse [output_tiles * SPLIT_K, BLOCK_M]        FP32
  sync_count  [output_tiles]                            INT32 (zeroed)
"""

import math
import os
import sys

import torch
import triton
import triton.language as tl

try:
    from ._common import *  # noqa: F403
    from ._layouts import (
        make_mfma_dot_layouts,
        make_fp8_dot_layouts,
        make_blocked_and_slice_layouts,
        make_padded_smem,
        make_dll,
        make_offset_bases,
        SERIAL_KT_SMEM,
        SERIAL_V_SMEM,
        SERIAL_Q_SMEM,
    )
except ImportError:
    _EXTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extend")
    if _EXTEND_DIR not in sys.path:
        sys.path.insert(0, _EXTEND_DIR)
    from _common import *  # noqa: F403
    from _layouts import (
        make_mfma_dot_layouts,
        make_fp8_dot_layouts,
        make_blocked_and_slice_layouts,
        make_padded_smem,
        make_dll,
        make_offset_bases,
        SERIAL_KT_SMEM,
        SERIAL_V_SMEM,
        SERIAL_Q_SMEM,
    )


D_NOPE = 128
D_ROPE = 64
D_FULL = D_NOPE + D_ROPE
D_V = 128


# ── DMA helpers (same as base 8w kernel) ─────────────────────────────────────


@gluon.jit
def _dma_kt_nope(kt_smem, kv_base, start_n, seqlen_kv, stride_kv_tok,
                  BLOCK_N: gl.constexpr, D_NOPE: gl.constexpr,
                  kt_async_layout: gl.constexpr,
                  SKIP_BOUNDS_CHECK: gl.constexpr = False):
    d_ly: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    n_ly: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    offs_d = gl.arange(0, D_NOPE, layout=d_ly)
    offs_n = gl.arange(0, BLOCK_N, layout=n_ly)
    offsets = (offs_d[:, None] + (start_n + offs_n[None, :]) * stride_kv_tok).to(tl.int32)
    if SKIP_BOUNDS_CHECK:
        cdna4_async.buffer_load_to_shared(kt_smem, kv_base, offsets)
    else:
        mask = (start_n + offs_n[None, :]) < seqlen_kv
        cdna4_async.buffer_load_to_shared(kt_smem, kv_base, offsets, mask=mask, other=0.0)
    cdna4_async.commit_group()


@gluon.jit
def _dma_kt_rope(kpe_smem, kv_base, start_n, seqlen_kv, stride_kv_tok,
                  BLOCK_N: gl.constexpr, D_NOPE: gl.constexpr,
                  D_ROPE: gl.constexpr,
                  kpe_async_layout: gl.constexpr,
                  SKIP_BOUNDS_CHECK: gl.constexpr = False):
    d_ly: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
    n_ly: gl.constexpr = gl.SliceLayout(dim=0, parent=kpe_async_layout)
    offs_d = gl.arange(0, D_ROPE, layout=d_ly)
    offs_n = gl.arange(0, BLOCK_N, layout=n_ly)
    offsets = ((D_NOPE + offs_d)[:, None] + (start_n + offs_n[None, :]) * stride_kv_tok).to(tl.int32)
    if SKIP_BOUNDS_CHECK:
        cdna4_async.buffer_load_to_shared(kpe_smem, kv_base, offsets)
    else:
        mask = (start_n + offs_n[None, :]) < seqlen_kv
        cdna4_async.buffer_load_to_shared(kpe_smem, kv_base, offsets, mask=mask, other=0.0)
    cdna4_async.commit_group()


@gluon.jit
def _dma_v(v_smem, kv_base, start_n, seqlen_kv, stride_kv_tok,
           BLOCK_N: gl.constexpr, D_V: gl.constexpr,
           v_async_layout: gl.constexpr,
           SKIP_BOUNDS_CHECK: gl.constexpr = False):
    n_ly: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    d_ly: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    offs_n = gl.arange(0, BLOCK_N, layout=n_ly)
    offs_d = gl.arange(0, D_V, layout=d_ly)
    offsets = ((start_n + offs_n)[:, None] * stride_kv_tok + offs_d[None, :]).to(tl.int32)
    if SKIP_BOUNDS_CHECK:
        cdna4_async.buffer_load_to_shared(v_smem, kv_base, offsets)
    else:
        mask = (start_n + offs_n)[:, None] < seqlen_kv
        cdna4_async.buffer_load_to_shared(v_smem, kv_base, offsets, mask=mask, other=0.0)
    cdna4_async.commit_group()


# ── Main kernel ───────────────────────────────────────────────────────────────


@gluon.jit
def fp8_mla_d192_8w_splitk_fwd(
    Q, KV, V_SEP, O,
    qo_indptr, kv_indptr,
    sm_scale,
    stride_q_tok, stride_q_h, stride_kv_tok, stride_kv_h, stride_v_tok, stride_v_h, stride_o_tok, stride_o_h,
    # Workspace (unused when SPLIT_K == 1)
    Partial_O, Partial_LSE, Sync_Count,
    stride_po_tile, stride_po_m,
    stride_pl_tile,
    # Grid / scheduling
    num_heads,
    n_m_tiles,
    total_valid_tiles,
    total_programs,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    D_NOPE: gl.constexpr,
    D_ROPE: gl.constexpr,
    D_V: gl.constexpr,
    NUM_WARPS_CONSTEXPR: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    Q_SCALE: gl.constexpr,
    KV_SCALE: gl.constexpr,
    SPLIT_K: gl.constexpr,
):
    # ── Layouts (identical to base 8w) ────────────────────────────────────
    _mfma: gl.constexpr = make_mfma_dot_layouts(NUM_WARPS_CONSTEXPR, 16, 16, 32, 8, 4)
    mma_layout: gl.constexpr = _mfma[0]
    _fp8: gl.constexpr = make_fp8_dot_layouts(mma_layout, 16, 8)
    fp8_q_dot: gl.constexpr = _fp8[0]
    fp8_kt_dot: gl.constexpr = _fp8[1]
    fp8_p_dot: gl.constexpr = _fp8[2]
    fp8_v_dot: gl.constexpr = _fp8[3]

    _blk: gl.constexpr = make_blocked_and_slice_layouts(NUM_WARPS_CONSTEXPR, mma_layout)
    blocked_layout: gl.constexpr = _blk[0]
    offs_m_layout: gl.constexpr = _blk[1]
    offs_d_layout: gl.constexpr = _blk[2]
    mma_n_col: gl.constexpr = _blk[3]
    mma_m_layout: gl.constexpr = _blk[5]

    qk_scale = sm_scale * Q_SCALE * KV_SCALE * LOG2E

    # ── Serial SMEM (causal tail, serial fallback) ────────────────────────
    USE_SERIAL_SMEM: gl.constexpr = NUM_STAGES <= 2
    if USE_SERIAL_SMEM:
        kt_blocked: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[8, 4], threads_per_warp=[4, 16],
            warps_per_cta=[4, 2], order=[0, 1])
        kpe_blocked: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[4, 4], threads_per_warp=[4, 16],
            warps_per_cta=[4, 2], order=[0, 1])
        kt_offs_d = gl.arange(0, D_NOPE, layout=gl.SliceLayout(dim=1, parent=kt_blocked))
        kt_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_blocked))
        kpe_offs_d = gl.arange(0, D_ROPE, layout=gl.SliceLayout(dim=1, parent=kpe_blocked))
        kpe_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kpe_blocked))
        v_offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=blocked_layout))
        v_offs_d = gl.arange(0, D_V, layout=gl.SliceLayout(dim=0, parent=blocked_layout))
        s_kt = gl.allocate_shared_memory(tl.float8e4nv, [D_NOPE, BLOCK_N], layout=SERIAL_KT_SMEM)
        s_kpe = gl.allocate_shared_memory(tl.float8e4nv, [D_ROPE, BLOCK_N], layout=SERIAL_KT_SMEM)
        s_v = gl.allocate_shared_memory(tl.float8e4nv, [BLOCK_N, D_V], layout=SERIAL_V_SMEM)

    # ── Async pipeline SMEM + layouts ─────────────────────────────────────
    if NUM_STAGES >= 2:
        kt_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32, 64], [1, 2, 4, 8], 0)
        kt_async_layout: gl.constexpr = make_dll(
            [D_NOPE, BLOCK_N],
            [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
            [[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]],
            [[0, 1], [0, 2], [0, 4]])
        kpe_offset_bases: gl.constexpr = make_offset_bases(32, [8, 16, 32, 64], [1, 2, 4], 0)
        kpe_async_layout: gl.constexpr = make_dll(
            [D_ROPE, BLOCK_N],
            [[1, 0], [2, 0], [4, 0], [8, 0]],
            [[16, 0], [32, 0], [0, 8], [0, 16], [0, 32], [0, 64]],
            [[0, 1], [0, 2], [0, 4]])
        v_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32, 64], [1, 2, 4, 8], 1)
        v_async_layout: gl.constexpr = make_dll(
            [BLOCK_N, D_V],
            [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
            [[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
            [[1, 0], [2, 0], [4, 0]])

        kt_smem_a = gl.allocate_shared_memory(
            tl.float8e4nv, [NUM_STAGES, D_NOPE, BLOCK_N],
            layout=make_padded_smem([D_NOPE, BLOCK_N], kt_offset_bases, [[1024, 32]]))
        kpe_smem_a = gl.allocate_shared_memory(
            tl.float8e4nv, [NUM_STAGES, D_ROPE, BLOCK_N],
            layout=make_padded_smem([D_ROPE, BLOCK_N], kpe_offset_bases, [[1024, 32]]))
        v_smem_a = gl.allocate_shared_memory(
            tl.float8e4nv, [NUM_STAGES, BLOCK_N, D_V],
            layout=make_padded_smem([BLOCK_N, D_V], v_offset_bases, [[1024, 32]]))
        for _s in gl.static_range(NUM_STAGES):
            v_zero = gl.zeros([BLOCK_N, D_V], dtype=tl.float8e4nv, layout=v_async_layout)
            v_smem_a.index(_s).store(v_zero)
        gl.barrier()
        STREAMS: gl.constexpr = 3
        WAIT_INIT: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
        WAIT_LOOP: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    # ══════════════════════════════════════════════════════════════════════
    #  Persistent tile loop
    # ══════════════════════════════════════════════════════════════════════

    tile_idx = gl.program_id(0)
    while tile_idx < total_valid_tiles:

        # ── Tile decomposition ────────────────────────────────────────────
        if SPLIT_K > 1:
            output_tile = tile_idx // SPLIT_K
            split_id = tile_idx % SPLIT_K
        else:
            output_tile = tile_idx
            split_id = 0

        pid_mb = output_tile % n_m_tiles
        rem = output_tile // n_m_tiles
        pid_h = rem % num_heads
        pid_seq = rem // num_heads

        seq_q_start = gl.load(qo_indptr + pid_seq)
        seqlen_q = gl.load(qo_indptr + pid_seq + 1) - seq_q_start
        kv_start = gl.load(kv_indptr + pid_seq)
        seqlen_kv = gl.load(kv_indptr + pid_seq + 1) - kv_start

        q_start = pid_mb * BLOCK_M
        if q_start < seqlen_q:
            kv_offset = seqlen_kv - seqlen_q
            n_blocks = (seqlen_kv + BLOCK_N - 1) // BLOCK_N
            causal_end = tl.minimum(
                (kv_offset + q_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N, n_blocks)
            unmasked_end = (kv_offset + q_start) // BLOCK_N
            kv_base = KV + kv_start * stride_kv_tok + pid_h * stride_kv_h
            v_base = V_SEP + kv_start * stride_v_tok + pid_h * stride_v_h

            # ── Split-K: partition causal_end blocks across splits ────────
            if SPLIT_K > 1:
                blocks_per_split = (causal_end + SPLIT_K - 1) // SPLIT_K
                split_lo = split_id * blocks_per_split
                split_hi = tl.minimum((split_id + 1) * blocks_per_split, causal_end)
                split_unmasked_lo = split_lo
                split_unmasked_hi = tl.minimum(split_hi, unmasked_end)
                split_causal_lo = tl.maximum(split_lo, unmasked_end)
                split_causal_hi = split_hi
            else:
                split_lo = 0
                split_hi = causal_end
                split_unmasked_lo = 0
                split_unmasked_hi = unmasked_end
                split_causal_lo = unmasked_end
                split_causal_hi = causal_end

            # ── Load Q (once per tile, before KV loop) ────────────────────
            offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
            q_mask = (q_start + offs_m)[:, None] < seqlen_q
            q_base = (Q + (seq_q_start + q_start + offs_m)[:, None] * stride_q_tok
                      + pid_h * stride_q_h)
            q_nope = gl.load(q_base + gl.arange(0, D_NOPE, layout=offs_d_layout)[None, :],
                             mask=q_mask, other=0.0)
            q_nope_dot = gl.convert_layout(q_nope, fp8_q_dot)
            qpe = gl.load(q_base + (D_NOPE + gl.arange(0, D_ROPE, layout=offs_d_layout))[None, :],
                           mask=q_mask, other=0.0)
            qpe_dot = gl.convert_layout(qpe, fp8_q_dot)

            acc = gl.zeros([BLOCK_M, D_V], dtype=gl.float32, layout=mma_layout)
            m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
            l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)

            n_unmasked = split_unmasked_hi - split_unmasked_lo

            # ══════════════════════════════════════════════════════════════
            #  Unmasked KV blocks for this split
            # ══════════════════════════════════════════════════════════════

            if NUM_STAGES >= 2:
                if n_unmasked < NUM_STAGES:
                    cdna4_async.wait_group(0)
                    for block_n in tl.range(split_unmasked_lo, split_unmasked_hi):
                        start_n = block_n * BLOCK_N
                        s_kt.store(gl.load(kv_base + kt_offs_d[:, None] + (start_n + kt_offs_n[None, :]) * stride_kv_tok))
                        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                        qk = do_mma(q_nope_dot, s_kt.load(fp8_kt_dot), qk)
                        s_kpe.store(gl.load(kv_base + (D_NOPE + kpe_offs_d)[:, None] + (start_n + kpe_offs_n[None, :]) * stride_kv_tok))
                        qk = do_mma(qpe_dot, s_kpe.load(fp8_kt_dot), qk)
                        qk_scaled = qk * qk_scale
                        m_ij = gl.maximum(m_i, nan_propagating_max(qk_scaled, axis=1),
                                          propagate_nan=tl.PropagateNan.ALL)
                        p = gl.exp2(qk_scaled - m_ij[:, None])
                        alpha = gl.exp2(m_i - m_ij)
                        acc = acc * alpha[:, None]
                        l_i = l_i * alpha + gl.sum(p, axis=1)
                        m_i = m_ij
                        p_c = gl.convert_layout(p.to(tl.float8e4nv), fp8_p_dot)
                        s_v.store(gl.load(v_base + (start_n + v_offs_n[:, None]) * stride_v_tok + v_offs_d[None, :]))
                        acc = do_mma(p_c, s_v.load(fp8_v_dot), acc)
                else:
                    # ── Pipeline prologue ──────────────────────────────────
                    cdna4_async.wait_group(0)
                    for stage in gl.static_range(NUM_STAGES):
                        pf_n = (split_unmasked_lo + stage) * BLOCK_N
                        _dma_kt_nope(kt_smem_a.index(stage), kv_base, pf_n, seqlen_kv,
                                     stride_kv_tok, BLOCK_N, D_NOPE, kt_async_layout,
                                     SKIP_BOUNDS_CHECK=True)
                        _dma_kt_rope(kpe_smem_a.index(stage), kv_base, pf_n, seqlen_kv,
                                     stride_kv_tok, BLOCK_N, D_NOPE, D_ROPE, kpe_async_layout,
                                     SKIP_BOUNDS_CHECK=True)
                        _dma_v(v_smem_a.index(stage), v_base, pf_n, seqlen_kv,
                               stride_v_tok, BLOCK_N, D_V, v_async_layout,
                               SKIP_BOUNDS_CHECK=True)
                    cdna4_async.wait_group(WAIT_INIT)
                    kt_reg = cdna4_async.load_shared_relaxed(kt_smem_a.index(0), fp8_kt_dot)
                    kpe_reg = cdna4_async.load_shared_relaxed(kpe_smem_a.index(0), fp8_kt_dot)

                    # ── Main pipelined loop ────────────────────────────────
                    main_count = n_unmasked - NUM_STAGES
                    for rel_n in tl.range(0, main_count, loop_unroll_factor=2):
                        si = (rel_n % NUM_STAGES).to(tl.int32)
                        future_n = (split_unmasked_lo + rel_n + NUM_STAGES) * BLOCK_N
                        with warp_pipeline_stage("dot1", priority=0):
                            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                            qk = do_mma(q_nope_dot, kt_reg, qk)
                            qk = do_mma(qpe_dot, kpe_reg, qk)
                        cdna4_async.wait_group(WAIT_LOOP)
                        with warp_pipeline_stage("mem1", priority=1):
                            v_dot = cdna4_async.load_shared_relaxed(v_smem_a.index(si), fp8_v_dot)
                            _dma_kt_nope(kt_smem_a.index(si), kv_base, future_n, seqlen_kv,
                                         stride_kv_tok, BLOCK_N, D_NOPE, kt_async_layout,
                                         SKIP_BOUNDS_CHECK=True)
                            _dma_kt_rope(kpe_smem_a.index(si), kv_base, future_n, seqlen_kv,
                                         stride_kv_tok, BLOCK_N, D_NOPE, D_ROPE, kpe_async_layout,
                                         SKIP_BOUNDS_CHECK=True)
                        with warp_pipeline_stage("dot2a", priority=0):
                            qk_scaled = qk * qk_scale
                            m_ij = gl.maximum(m_i, nan_propagating_max(qk_scaled, axis=1),
                                          propagate_nan=tl.PropagateNan.ALL)
                            p = gl.exp2(qk_scaled - m_ij[:, None])
                            alpha = gl.exp2(m_i - m_ij)
                            acc = acc * alpha[:, None]
                            l_i = l_i * alpha + gl.sum(p, axis=1)
                            m_i = m_ij
                        with warp_pipeline_stage("dot2b", priority=0):
                            p_c = gl.convert_layout(p.to(tl.float8e4nv), fp8_p_dot)
                            acc = do_mma(p_c, v_dot, acc)
                        cdna4_async.wait_group(WAIT_LOOP)
                        with warp_pipeline_stage("mem2", priority=1):
                            nsi = ((rel_n + 1) % NUM_STAGES).to(tl.int32)
                            kt_reg = cdna4_async.load_shared_relaxed(kt_smem_a.index(nsi), fp8_kt_dot)
                            kpe_reg = cdna4_async.load_shared_relaxed(kpe_smem_a.index(nsi), fp8_kt_dot)
                            _dma_v(v_smem_a.index(si), v_base, future_n, seqlen_kv,
                                   stride_v_tok, BLOCK_N, D_V, v_async_layout,
                                   SKIP_BOUNDS_CHECK=True)

                    # ── Pipeline epilogue (drain) ──────────────────────────
                    for tail_i in gl.static_range(NUM_STAGES):
                        si = ((main_count + tail_i) % NUM_STAGES).to(tl.int32)
                        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - 1)
                        kt_tail = cdna4_async.load_shared_relaxed(kt_smem_a.index(si), fp8_kt_dot)
                        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - 2)
                        kpe_tail = cdna4_async.load_shared_relaxed(kpe_smem_a.index(si), fp8_kt_dot)
                        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                        qk = do_mma(q_nope_dot, kt_tail, qk)
                        qk = do_mma(qpe_dot, kpe_tail, qk)
                        qk_scaled = qk * qk_scale
                        m_ij = gl.maximum(m_i, nan_propagating_max(qk_scaled, axis=1),
                                          propagate_nan=tl.PropagateNan.ALL)
                        p = gl.exp2(qk_scaled - m_ij[:, None])
                        alpha = gl.exp2(m_i - m_ij)
                        acc = acc * alpha[:, None]
                        l_i = l_i * alpha + gl.sum(p, axis=1)
                        m_i = m_ij
                        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - 3)
                        v_tail = cdna4_async.load_shared_relaxed(v_smem_a.index(si), fp8_v_dot)
                        p_c = gl.convert_layout(p.to(tl.float8e4nv), fp8_p_dot)
                        acc = do_mma(p_c, v_tail, acc)
                    cdna4_async.wait_group(0)
            else:
                for block_n in tl.range(split_unmasked_lo, split_unmasked_hi):
                    start_n = block_n * BLOCK_N
                    s_kt.store(gl.load(kv_base + kt_offs_d[:, None] + (start_n + kt_offs_n[None, :]) * stride_kv_tok))
                    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                    qk = do_mma(q_nope_dot, s_kt.load(fp8_kt_dot), qk)
                    s_kpe.store(gl.load(kv_base + (D_NOPE + kpe_offs_d)[:, None] + (start_n + kpe_offs_n[None, :]) * stride_kv_tok))
                    qk = do_mma(qpe_dot, s_kpe.load(fp8_kt_dot), qk)
                    qk_scaled = qk * qk_scale
                    m_ij = gl.maximum(m_i, nan_propagating_max(qk_scaled, axis=1),
                                          propagate_nan=tl.PropagateNan.ALL)
                    p = gl.exp2(qk_scaled - m_ij[:, None])
                    alpha = gl.exp2(m_i - m_ij)
                    acc = acc * alpha[:, None]
                    l_i = l_i * alpha + gl.sum(p, axis=1)
                    m_i = m_ij
                    p_c = gl.convert_layout(p.to(tl.float8e4nv), fp8_p_dot)
                    s_v.store(gl.load(v_base + (start_n + v_offs_n[:, None]) * stride_v_tok + v_offs_d[None, :]))
                    acc = do_mma(p_c, s_v.load(fp8_v_dot), acc)

            # ══════════════════════════════════════════════════════════════
            #  Causal blocks for this split
            # ══════════════════════════════════════════════════════════════

            causal_row = gl.arange(0, BLOCK_M, layout=mma_m_layout)
            causal_limit = gl.minimum(kv_offset + q_start + causal_row + 1, seqlen_kv)
            n_col = gl.arange(0, BLOCK_N, layout=mma_n_col)
            q_row_valid = (q_start + causal_row) < seqlen_q

            if NUM_STAGES >= 2:
                for block_n in tl.range(split_causal_lo, split_causal_hi):
                    start_n = block_n * BLOCK_N
                    _dma_kt_nope(kt_smem_a.index(0), kv_base, start_n, seqlen_kv,
                                 stride_kv_tok, BLOCK_N, D_NOPE, kt_async_layout)
                    _dma_kt_rope(kpe_smem_a.index(0), kv_base, start_n, seqlen_kv,
                                 stride_kv_tok, BLOCK_N, D_NOPE, D_ROPE, kpe_async_layout)
                    _dma_v(v_smem_a.index(0), v_base, start_n, seqlen_kv,
                           stride_v_tok, BLOCK_N, D_V, v_async_layout)
                    cdna4_async.wait_group(0)
                    kt_c = cdna4_async.load_shared_relaxed(kt_smem_a.index(0), fp8_kt_dot)
                    kpe_c = cdna4_async.load_shared_relaxed(kpe_smem_a.index(0), fp8_kt_dot)
                    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                    qk = do_mma(q_nope_dot, kt_c, qk)
                    qk = do_mma(qpe_dot, kpe_c, qk)
                    valid = q_row_valid[:, None] & ((start_n + n_col)[None, :] < causal_limit[:, None])
                    qk_scaled = qk * qk_scale
                    qk_scaled = gl.where(valid, qk_scaled,
                        gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))
                    m_ij = gl.maximum(m_i, nan_propagating_max(qk_scaled, axis=1),
                                          propagate_nan=tl.PropagateNan.ALL)
                    p = gl.exp2(qk_scaled - m_ij[:, None])
                    alpha = gl.exp2(m_i - m_ij)
                    acc = acc * alpha[:, None]
                    l_i = l_i * alpha + gl.sum(p, axis=1)
                    m_i = m_ij
                    v_c = cdna4_async.load_shared_relaxed(v_smem_a.index(0), fp8_v_dot)
                    p_c = gl.convert_layout(p.to(tl.float8e4nv), fp8_p_dot)
                    acc = do_mma(p_c, v_c, acc)
            else:
                for block_n in tl.range(split_causal_lo, split_causal_hi):
                    start_n = block_n * BLOCK_N
                    kt_mask = (start_n + kt_offs_n[None, :]) < seqlen_kv
                    s_kt.store(gl.load(kv_base + kt_offs_d[:, None] + (start_n + kt_offs_n[None, :]) * stride_kv_tok,
                                       mask=kt_mask, other=0.0))
                    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                    qk = do_mma(q_nope_dot, s_kt.load(fp8_kt_dot), qk)
                    kpe_mask = (start_n + kpe_offs_n[None, :]) < seqlen_kv
                    s_kpe.store(gl.load(kv_base + (D_NOPE + kpe_offs_d)[:, None] + (start_n + kpe_offs_n[None, :]) * stride_kv_tok,
                                        mask=kpe_mask, other=0.0))
                    qk = do_mma(qpe_dot, s_kpe.load(fp8_kt_dot), qk)
                    valid = q_row_valid[:, None] & ((start_n + n_col)[None, :] < causal_limit[:, None])
                    qk_scaled = qk * qk_scale
                    qk_scaled = gl.where(valid, qk_scaled,
                        gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout))
                    m_ij = gl.maximum(m_i, nan_propagating_max(qk_scaled, axis=1),
                                          propagate_nan=tl.PropagateNan.ALL)
                    p = gl.exp2(qk_scaled - m_ij[:, None])
                    alpha = gl.exp2(m_i - m_ij)
                    acc = acc * alpha[:, None]
                    l_i = l_i * alpha + gl.sum(p, axis=1)
                    m_i = m_ij
                    p_c = gl.convert_layout(p.to(tl.float8e4nv), fp8_p_dot)
                    v_mask = (start_n + v_offs_n[:, None]) < seqlen_kv
                    s_v.store(gl.load(v_base + (start_n + v_offs_n[:, None]) * stride_v_tok + v_offs_d[None, :],
                                      mask=v_mask, other=0.0))
                    acc = do_mma(p_c, s_v.load(fp8_v_dot), acc)

            # ══════════════════════════════════════════════════════════════
            #  Output: direct write (SK==1) or partial + inline reduce
            # ══════════════════════════════════════════════════════════════

            o_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
            o_d = gl.arange(0, D_V, layout=offs_d_layout)
            o_mask = (q_start + o_m)[:, None] < seqlen_q

            if SPLIT_K == 1:
                rcp_l = (1.0 / l_i)[:, None]
                out = gl.convert_layout(acc * rcp_l, blocked_layout).to(gl.bfloat16)
                o_base = O + (seq_q_start + q_start) * stride_o_tok + pid_h * stride_o_h
                o_offsets = (o_m[:, None] * stride_o_tok + o_d[None, :]).to(tl.int32)
                cdna4_buffer_store(out, o_base, o_offsets, mask=o_mask)
            else:
                # All splits write normalized partial + log2-LSE to workspace
                rcp_l = (1.0 / l_i)[:, None]
                normed_o = gl.convert_layout(acc * rcp_l, blocked_layout)
                lse = gl.convert_layout(m_i + tl.math.log2(l_i), offs_m_layout)

                partial_idx = output_tile * SPLIT_K + split_id
                po_base = Partial_O + partial_idx * stride_po_tile + o_m[:, None] * stride_po_m
                gl.store(po_base + o_d[None, :], normed_o, mask=o_mask)
                pl_base = Partial_LSE + partial_idx * stride_pl_tile
                gl.store(pl_base + o_m, lse, mask=(q_start + o_m) < seqlen_q)

        tile_idx += total_programs


# ═══════════════════════════════════════════════════════════════════════════════
#  Separate reduce kernel (merges SPLIT_K partials per output tile)
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit
def _splitk_reduce(
    Partial_O, Partial_LSE,
    O,
    qo_indptr,
    stride_po_tile, stride_po_m,
    stride_pl_tile,
    stride_o_tok, stride_o_h,
    num_heads, n_m_tiles,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    D_V: tl.constexpr,
):
    """One workgroup per output tile — merge SPLIT_K partials with stable LSE."""
    output_tile = tl.program_id(0)

    pid_mb = output_tile % n_m_tiles
    rem = output_tile // n_m_tiles
    pid_h = rem % num_heads
    pid_seq = rem // num_heads

    seq_q_start = tl.load(qo_indptr + pid_seq)
    seqlen_q = tl.load(qo_indptr + pid_seq + 1) - seq_q_start
    q_start = pid_mb * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_V)
    o_mask = (q_start + offs_m)[:, None] < seqlen_q
    pl_mask = (q_start + offs_m) < seqlen_q
    base_idx = output_tile * SPLIT_K

    # Load first partial as running accumulator
    r_po = Partial_O + base_idx * stride_po_tile + offs_m[:, None] * stride_po_m
    running_o = tl.load(r_po + offs_d[None, :], mask=o_mask, other=0.0)
    r_pl = Partial_LSE + base_idx * stride_pl_tile
    running_lse = tl.load(r_pl + offs_m, mask=pl_mask, other=float("-inf"))

    for ki in tl.static_range(1, SPLIT_K):
        k_po = Partial_O + (base_idx + ki) * stride_po_tile + offs_m[:, None] * stride_po_m
        o_ki = tl.load(k_po + offs_d[None, :], mask=o_mask, other=0.0)
        k_pl = Partial_LSE + (base_idx + ki) * stride_pl_tile
        lse_ki = tl.load(k_pl + offs_m, mask=pl_mask, other=float("-inf"))

        max_lse = tl.maximum(running_lse, lse_ki)
        w_old = tl.exp2(running_lse - max_lse)
        w_new = tl.exp2(lse_ki - max_lse)
        denom = w_old + w_new
        running_o = (running_o * w_old[:, None]
                     + o_ki * w_new[:, None]) / denom[:, None]
        running_lse = max_lse + tl.math.log2(denom)

    out = running_o.to(tl.bfloat16)
    o_base = (O + (seq_q_start + q_start + offs_m)[:, None] * stride_o_tok
              + pid_h * stride_o_h)
    tl.store(o_base + offs_d[None, :], out, mask=o_mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  Python launch helpers
# ═══════════════════════════════════════════════════════════════════════════════


NUM_CUS = 256


def select_split_k(batch, num_heads, max_q_blocks, min_kv_seqlen, block_n=128):
    """Pick SPLIT_K to maximize CU occupancy without over-splitting."""
    total_output_tiles = batch * num_heads * max_q_blocks
    if total_output_tiles >= NUM_CUS:
        return 1
    target = NUM_CUS * 2
    sk = max(1, min(8, target // max(total_output_tiles, 1)))
    max_splits = max(1, min_kv_seqlen // (block_n * 2))
    return max(1, min(sk, max_splits))


def launch_splitk(q, kv, o, qo_indptr, kv_indptr, sm_scale,
                  num_heads, split_k=None, num_stages=2,
                  q_scale=1.0, kv_scale=1.0, v=None):
    """Launch persistent split-K prefill + separate reduce (dev-only helper).

    Production path is `mla_prefill_d192_splitk_fwd` in the sibling
    dispatch module; this helper is kept alive for the in-file test
    harness under `__main__`.
    """
    BM, BN, NW = 128, 128, 8
    batch = qo_indptr.shape[0] - 1
    q_seqlens = (qo_indptr[1:] - qo_indptr[:-1]).tolist()
    kv_seqlens = (kv_indptr[1:] - kv_indptr[:-1]).tolist()
    max_q_blocks = max((sq + BM - 1) // BM for sq in q_seqlens)

    if split_k is None:
        split_k = select_split_k(batch, num_heads, max_q_blocks, min(kv_seqlens), BN)

    total_output_tiles = batch * num_heads * max_q_blocks
    total_valid_tiles = total_output_tiles * split_k
    total_programs = min(total_valid_tiles, NUM_CUS)

    v_sep = kv if v is None else v
    stride_kv_h = kv.stride(1) if kv.ndim >= 3 else 0
    stride_v_h = v_sep.stride(1) if v_sep.ndim >= 3 else 0
    stride_v_tok = v_sep.stride(0) if v_sep is not kv else kv.stride(0)

    device = q.device
    if split_k > 1:
        partial_o = torch.empty(total_output_tiles * split_k, BM, D_V,
                                dtype=torch.float32, device=device)
        partial_lse = torch.full((total_output_tiles * split_k, BM),
                                 float("-inf"), dtype=torch.float32, device=device)
        sync_count = torch.empty(0, dtype=torch.int32, device=device)
    else:
        partial_o = torch.empty(1, dtype=torch.float32, device=device)
        partial_lse = torch.empty(1, dtype=torch.float32, device=device)
        sync_count = torch.empty(1, dtype=torch.int32, device=device)

    grid = (total_programs,)
    fp8_mla_d192_8w_splitk_fwd[grid](
        q, kv, v_sep, o,
        qo_indptr, kv_indptr,
        sm_scale,
        q.stride(0), q.stride(1),
        kv.stride(0), stride_kv_h, stride_v_tok, stride_v_h,
        o.stride(0), o.stride(1),
        partial_o, partial_lse, sync_count,
        partial_o.stride(0) if split_k > 1 else 0,
        partial_o.stride(1) if split_k > 1 else 0,
        partial_lse.stride(0) if split_k > 1 else 0,
        num_heads, max_q_blocks,
        total_valid_tiles, total_programs,
        BLOCK_M=BM, BLOCK_N=BN,
        D_NOPE=D_NOPE, D_ROPE=D_ROPE, D_V=D_V,
        NUM_WARPS_CONSTEXPR=NW, NUM_STAGES=num_stages,
        Q_SCALE=q_scale, KV_SCALE=kv_scale,
        SPLIT_K=split_k,
        num_warps=NW, num_stages=1,
    )

    if split_k > 1:
        _splitk_reduce[(total_output_tiles,)](
            partial_o, partial_lse,
            o, qo_indptr,
            partial_o.stride(0), partial_o.stride(1),
            partial_lse.stride(0),
            o.stride(0), o.stride(1),
            num_heads, max_q_blocks,
            SPLIT_K=split_k, BLOCK_M=BM, D_V=D_V,
            num_warps=4,
        )

    return split_k


# ═══════════════════════════════════════════════════════════════════════════════
#  Correctness + benchmark
# ═══════════════════════════════════════════════════════════════════════════════


def ref_attention(q_fp8_flat, kv_fp8_flat, q_seqlens, kv_seqlens, sm_scale, num_q_heads):
    """Reference attention using FP8 Q and FP8 KV (matching SGLang production path).

    Both q_fp8_flat and kv_fp8_flat should already be float8_e4m3fn tensors.
    Computation is done in float32 after upcast (same as what the MFMA produces).
    """
    batch_size = len(q_seqlens)
    q_offset, kv_offset = 0, 0
    out_list = []
    for b in range(batch_size):
        sq, skv = q_seqlens[b], kv_seqlens[b]
        q_b = q_fp8_flat[q_offset : q_offset + sq].float()
        kv_b = kv_fp8_flat[kv_offset : kv_offset + skv].squeeze(1).float()
        off = skv - sq
        q_all = q_b[:, :, :D_FULL]                      # [sq, nh, D_FULL]
        k_all = kv_b[:, :D_FULL]                         # [skv, D_FULL]
        v_all = kv_b[:, :D_V]                             # [skv, D_V]
        qk = torch.einsum("qhd,kd->hqk", q_all, k_all) * sm_scale  # [nh, sq, skv]
        row_idx = torch.arange(sq).unsqueeze(1)           # [sq, 1]
        col_idx = torch.arange(skv).unsqueeze(0)          # [1, skv]
        causal_mask = (off + row_idx) >= col_idx           # [sq, skv]
        qk.masked_fill_(~causal_mask.unsqueeze(0), float("-inf"))
        p = torch.softmax(qk, dim=-1)                     # [nh, sq, skv]
        o_b = torch.einsum("hqk,kd->qhd", p, v_all)      # [sq, nh, D_V]
        out_list.append(o_b.to(torch.bfloat16))
        q_offset += sq
        kv_offset += skv
    return torch.cat(out_list, dim=0)


def run_test(q_seqlens, kv_seqlens, num_q_heads=1, split_k=1,
             num_stages=2, seed=42):
    batch = len(q_seqlens)
    total_q, total_kv = sum(q_seqlens), sum(kv_seqlens)
    sm_scale = 1.0 / math.sqrt(D_FULL)
    torch.manual_seed(seed)

    q_bf16 = torch.randn(total_q, num_q_heads, D_FULL, dtype=torch.bfloat16, device="cuda")
    q = q_bf16.to(torch.float8_e4m3fn)
    kv = torch.randn(total_kv, 1, D_FULL, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    o = torch.zeros(total_q, num_q_heads, D_V, dtype=torch.bfloat16, device="cuda")

    qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    for b in range(batch):
        qo_indptr[b + 1] = qo_indptr[b] + q_seqlens[b]
        kv_indptr[b + 1] = kv_indptr[b] + kv_seqlens[b]

    used_sk = launch_splitk(q, kv, o, qo_indptr, kv_indptr, sm_scale,
                            num_q_heads, split_k=split_k, num_stages=num_stages)
    torch.cuda.synchronize()

    ref = ref_attention(q.cpu(), kv.cpu(), q_seqlens, kv_seqlens, sm_scale, num_q_heads)
    out = o.cpu()
    diff = (out.float() - ref.float()).abs()
    max_d, mean_d = diff.max().item(), diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        out.float().reshape(-1), ref.float().reshape(-1), dim=0).item()
    ok = max_d < 0.05 and cos > 0.999
    tag = "PASS" if ok else "FAIL"
    ctx_str = str(kv_seqlens) if batch > 1 else str(kv_seqlens[0])
    print(f"  [{tag}] SK={used_sk} NS={num_stages} nh={num_q_heads} ctx={ctx_str}  "
          f"max={max_d:.4f}  mean={mean_d:.6f}  cos={cos:.6f}")
    return ok


def bench(q_seqlens, kv_seqlens, num_q_heads=128, split_k=None,
          num_stages=2, warmup=10, iters=30):
    batch = len(q_seqlens)
    total_q, total_kv = sum(q_seqlens), sum(kv_seqlens)
    sm_scale = 1.0 / math.sqrt(D_FULL)
    torch.manual_seed(42)

    q = torch.randn(total_q, num_q_heads, D_FULL, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    kv = torch.randn(total_kv, 1, D_FULL, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    o = torch.zeros(total_q, num_q_heads, D_V, dtype=torch.bfloat16, device="cuda")

    qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    for b in range(batch):
        qo_indptr[b + 1] = qo_indptr[b] + q_seqlens[b]
        kv_indptr[b + 1] = kv_indptr[b] + kv_seqlens[b]

    used_sk = 0
    def _run():
        nonlocal used_sk
        used_sk = launch_splitk(q, kv, o, qo_indptr, kv_indptr, sm_scale,
                                num_q_heads, split_k=split_k, num_stages=num_stages)

    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        _run()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    flops = sum(2 * sq * num_q_heads * skv * (D_FULL + D_V)
                for sq, skv in zip(q_seqlens, kv_seqlens))
    tflops = flops / ms / 1e9
    print(f"  SK={used_sk} B{batch} nh{num_q_heads} ctx{kv_seqlens[0]}  "
          f"{ms:.3f} ms  {tflops:.0f} TF/s")
    return ms, tflops


if __name__ == "__main__":
    print(f"\n{'='*72}")
    print(f"  D192 MLA prefill — 8w persistent split-K (gfx950)")
    print(f"  BM=128 BN=128 NW=8 NS=2  D_QK={D_FULL} D_V={D_V}")
    print(f"{'='*72}")

    all_ok = True

    print("\n--- Correctness: SPLIT_K=1 (pure persistent) ---")
    for seed in [42, 0, 12345]:
        all_ok &= run_test([256], [1024], 16, split_k=1, seed=seed)
    all_ok &= run_test([128], [128], 1, split_k=1)
    all_ok &= run_test([256], [4096], 16, split_k=1)
    all_ok &= run_test([512], [2048], 128, split_k=1)

    print("\n--- Correctness: SPLIT_K=2 ---")
    all_ok &= run_test([256], [1024], 16, split_k=2)
    all_ok &= run_test([256], [4096], 16, split_k=2)
    all_ok &= run_test([512], [2048], 128, split_k=2)

    print("\n--- Correctness: SPLIT_K=4 ---")
    all_ok &= run_test([256], [4096], 16, split_k=4)
    all_ok &= run_test([512], [4096], 128, split_k=4)

    print("\n--- Correctness: DeepSeek V3 params (nh=128, causal) ---")
    for ctx in [1024, 2048, 4096]:
        all_ok &= run_test([ctx], [ctx], 128, split_k=1)
        all_ok &= run_test([ctx], [ctx], 128, split_k=2)

    print("\n--- Performance: SK=1 (pure persistent) vs auto ---")
    for ctx in [1024, 2048, 4096, 8192, 16384]:
        bench([ctx], [ctx], 128, split_k=1)
        bench([ctx], [ctx], 128, split_k=None)

    print(f"\n{'='*72}")
    print(f"  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print(f"{'='*72}")
