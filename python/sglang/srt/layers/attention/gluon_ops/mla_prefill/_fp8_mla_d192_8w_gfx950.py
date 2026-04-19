#!/usr/bin/env python3
"""FP8 D192 MLA prefill — 8-warp warp-pipeline kernel (gfx950).

QK = qk_nope(128) + qk_rope(64) = 192,  V = kv_lora_rank = 128
3-stream async DMA pipeline. Supports IS_PERSISTENT and basic dispatch.

FP8 DLL layouts taken directly from the working FP8 extend kernel
(_fp8_kv_extend_basic_gfx950.py), 8-warp path.
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


@gluon.jit
def fp8_mla_d192_8w_fwd(
    Q, KV, V_SEP, O,
    qo_indptr, kv_indptr,
    sm_scale,
    stride_q_tok, stride_q_h, stride_kv_tok, stride_kv_h, stride_v_tok, stride_v_h, stride_o_tok, stride_o_h,
    num_heads,
    n_m_tiles,
    total_valid_tiles,
    total_programs,
    work_indptr_ptr,
    work_info_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    stride_po_tok, stride_po_h,
    stride_pl_tok, stride_pl_h,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    D_NOPE: gl.constexpr,
    D_ROPE: gl.constexpr,
    D_V: gl.constexpr,
    NUM_WARPS_CONSTEXPR: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    Q_SCALE: gl.constexpr,
    KV_SCALE: gl.constexpr,
    IS_PERSISTENT: gl.constexpr = False,
    IS_PS_PERSISTENT: gl.constexpr = False,
    IS_WCA: gl.constexpr = False,
    batch_size = 0,
):
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

    if NUM_STAGES >= 2:
        # FP8 async DMA layouts — from working extend kernel (8w, D=128, N=128)
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
        # V [128, 128]: transposed — from extend kernel (8w, DV=128, N=128)
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

    if IS_PS_PERSISTENT:
        cta_id = gl.program_id(0)
        ps_start = gl.load(work_indptr_ptr + cta_id)
        ps_end = gl.load(work_indptr_ptr + cta_id + 1)
        ps_idx = ps_start
    elif IS_PERSISTENT or IS_WCA:
        tile_idx = gl.program_id(0)
    else:
        tile_idx = 0

    while (ps_idx < ps_end if IS_PS_PERSISTENT else
           tile_idx < (total_valid_tiles if (IS_PERSISTENT or IS_WCA) else 1)):

        if IS_PS_PERSISTENT:
            info_base = work_info_ptr + ps_idx * 8
            w_batch = gl.load(info_base).to(tl.int32)
            w_partial = gl.load(info_base + 1).to(tl.int32)
            w_qo_start = gl.load(info_base + 2).to(tl.int32)
            w_qo_end = gl.load(info_base + 3).to(tl.int32)
            w_kv_start_g = gl.load(info_base + 4).to(tl.int32)
            w_kv_end_g = gl.load(info_base + 5).to(tl.int32)
            w_kv_off = gl.load(info_base + 6).to(tl.int32)
            w_hrange = gl.load(info_base + 7).to(tl.int32)

            pid_h = w_hrange & 0xFFFF
            seq_q_start = w_qo_start
            seqlen_q = w_qo_end - w_qo_start
            seqlen_kv = w_kv_end_g - w_kv_start_g
            q_start = 0
            kv_base = KV + w_kv_start_g * stride_kv_tok + pid_h * stride_kv_h
            v_base = V_SEP + w_kv_start_g * stride_v_tok + pid_h * stride_v_h

            n_blocks = (seqlen_kv + BLOCK_N - 1) // BLOCK_N
            if w_kv_off > 0:
                kv_offset = seqlen_kv
                unmasked_end = n_blocks
                causal_end = n_blocks
            else:
                kv_batch_end = gl.load(kv_indptr + w_batch + 1)
                qo_batch_end = gl.load(qo_indptr + w_batch + 1)
                kv_offset = kv_batch_end - qo_batch_end + w_qo_start - w_kv_start_g
                causal_end = tl.minimum(
                    (kv_offset + BLOCK_M + BLOCK_N - 1) // BLOCK_N, n_blocks)
                unmasked_end = tl.minimum(kv_offset // BLOCK_N, n_blocks)
        else:
            if IS_WCA:
                cur_seq: tl.int32 = 0
                cum_tiles: tl.int32 = 0
                found: tl.int32 = 0
                while (cur_seq < batch_size) & (found == 0):
                    s_ext = (gl.load(qo_indptr + cur_seq + 1)
                             - gl.load(qo_indptr + cur_seq)).to(tl.int32)
                    s_tiles = tl.maximum(
                        (s_ext + BLOCK_M - 1) // BLOCK_M, 0) * num_heads
                    next_cum = cum_tiles + s_tiles
                    if next_cum > tile_idx:
                        found = 1
                    else:
                        cum_tiles = next_cum
                        cur_seq += 1
                local_tile = tile_idx - cum_tiles
                seq_ext_len = (gl.load(qo_indptr + cur_seq + 1)
                               - gl.load(qo_indptr + cur_seq)).to(tl.int32)
                tiles_per_head = tl.maximum(
                    (seq_ext_len + BLOCK_M - 1) // BLOCK_M, 1)
                pid_h = local_tile // tiles_per_head
                pid_mb = local_tile % tiles_per_head
                pid_seq = cur_seq
            elif not IS_PERSISTENT:
                pid_seq = gl.program_id(0)
                pid_h = gl.program_id(1)
                pid_mb = gl.program_id(2)
            else:
                pid_mb = tile_idx % n_m_tiles
                rem = tile_idx // n_m_tiles
                pid_h = rem % num_heads
                pid_seq = rem // num_heads

            seq_q_start = gl.load(qo_indptr + pid_seq)
            seqlen_q = gl.load(qo_indptr + pid_seq + 1) - seq_q_start
            kv_start = gl.load(kv_indptr + pid_seq)
            seqlen_kv = gl.load(kv_indptr + pid_seq + 1) - kv_start
            q_start = pid_mb * BLOCK_M
            kv_base = KV + kv_start * stride_kv_tok + pid_h * stride_kv_h
            v_base = V_SEP + kv_start * stride_v_tok + pid_h * stride_v_h
            kv_offset = seqlen_kv - seqlen_q
            n_blocks = (seqlen_kv + BLOCK_N - 1) // BLOCK_N
            causal_end = tl.minimum(
                (kv_offset + q_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N, n_blocks)
            unmasked_end = (kv_offset + q_start) // BLOCK_N

        if q_start < seqlen_q:

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

            if NUM_STAGES >= 2:
                if unmasked_end < NUM_STAGES:
                    cdna4_async.wait_group(0)
                    if USE_SERIAL_SMEM:
                        for block_n in tl.range(0, unmasked_end):
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
                        for block_n in tl.range(0, unmasked_end):
                            start_n = block_n * BLOCK_N
                            _dma_kt_nope(kt_smem_a.index(0), kv_base, start_n, seqlen_kv,
                                         stride_kv_tok, BLOCK_N, D_NOPE, kt_async_layout)
                            _dma_kt_rope(kpe_smem_a.index(0), kv_base, start_n, seqlen_kv,
                                         stride_kv_tok, BLOCK_N, D_NOPE, D_ROPE, kpe_async_layout)
                            _dma_v(v_smem_a.index(0), v_base, start_n, seqlen_kv,
                                   stride_v_tok, BLOCK_N, D_V, v_async_layout)
                            cdna4_async.wait_group(0)
                            kt_r = cdna4_async.load_shared_relaxed(kt_smem_a.index(0), fp8_kt_dot)
                            kpe_r = cdna4_async.load_shared_relaxed(kpe_smem_a.index(0), fp8_kt_dot)
                            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                            qk = do_mma(q_nope_dot, kt_r, qk)
                            qk = do_mma(qpe_dot, kpe_r, qk)
                            qk_scaled = qk * qk_scale
                            m_ij = gl.maximum(m_i, nan_propagating_max(qk_scaled, axis=1),
                                          propagate_nan=tl.PropagateNan.ALL)
                            p = gl.exp2(qk_scaled - m_ij[:, None])
                            alpha = gl.exp2(m_i - m_ij)
                            acc = acc * alpha[:, None]
                            l_i = l_i * alpha + gl.sum(p, axis=1)
                            m_i = m_ij
                            v_r = cdna4_async.load_shared_relaxed(v_smem_a.index(0), fp8_v_dot)
                            p_c = gl.convert_layout(p.to(tl.float8e4nv), fp8_p_dot)
                            acc = do_mma(p_c, v_r, acc)
                else:
                    cdna4_async.wait_group(0)
                    for stage in gl.static_range(NUM_STAGES):
                        pf_n = stage * BLOCK_N
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
                    main_end = unmasked_end - NUM_STAGES
                    for block_n in tl.range(0, main_end, loop_unroll_factor=2):
                        si = (block_n % NUM_STAGES).to(tl.int32)
                        future_n = (block_n + NUM_STAGES) * BLOCK_N
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
                            nsi = ((block_n + 1) % NUM_STAGES).to(tl.int32)
                            kt_reg = cdna4_async.load_shared_relaxed(kt_smem_a.index(nsi), fp8_kt_dot)
                            kpe_reg = cdna4_async.load_shared_relaxed(kpe_smem_a.index(nsi), fp8_kt_dot)
                            _dma_v(v_smem_a.index(si), v_base, future_n, seqlen_kv,
                                   stride_v_tok, BLOCK_N, D_V, v_async_layout,
                                   SKIP_BOUNDS_CHECK=True)
                    for tail_i in gl.static_range(NUM_STAGES):
                        si = ((main_end + tail_i) % NUM_STAGES).to(tl.int32)
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
                for block_n in tl.range(0, unmasked_end):
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

            causal_row = gl.arange(0, BLOCK_M, layout=mma_m_layout)
            causal_limit = gl.minimum(kv_offset + q_start + causal_row + 1, seqlen_kv)
            n_col = gl.arange(0, BLOCK_N, layout=mma_n_col)
            q_row_valid = (q_start + causal_row) < seqlen_q

            if NUM_STAGES >= 2:
                for block_n in tl.range(unmasked_end, causal_end):
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
                for block_n in tl.range(unmasked_end, causal_end):
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

            rcp_l = (1.0 / l_i)[:, None]
            o_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
            o_d = gl.arange(0, D_V, layout=offs_d_layout)
            o_mask = (q_start + o_m)[:, None] < seqlen_q

            if IS_PS_PERSISTENT:
                is_partial = w_partial >= 0
                if is_partial:
                    out_f32 = gl.convert_layout(acc * rcp_l, blocked_layout)
                    po_base = (partial_out_ptr
                               + (w_partial + o_m)[:, None] * stride_po_tok
                               + pid_h * stride_po_h)
                    gl.store(po_base + o_d[None, :], out_f32, mask=o_mask)
                    lse_val = gl.convert_layout(
                        m_i + gl.log2(l_i + 1e-10), offs_m_layout)
                    lse_mask = (q_start + o_m) < seqlen_q
                    pl_base = (partial_lse_ptr
                               + (w_partial + o_m) * stride_pl_tok
                               + pid_h * stride_pl_h)
                    gl.store(pl_base, lse_val, mask=lse_mask)
                else:
                    out = gl.convert_layout(acc * rcp_l, blocked_layout).to(gl.bfloat16)
                    o_base = O + seq_q_start * stride_o_tok + pid_h * stride_o_h
                    o_offsets = (o_m[:, None] * stride_o_tok + o_d[None, :]).to(tl.int32)
                    cdna4_buffer_store(out, o_base, o_offsets, mask=o_mask)
            else:
                out = gl.convert_layout(acc * rcp_l, blocked_layout).to(gl.bfloat16)
                o_base = O + (seq_q_start + q_start) * stride_o_tok + pid_h * stride_o_h
                o_offsets = (o_m[:, None] * stride_o_tok + o_d[None, :]).to(tl.int32)
                cdna4_buffer_store(out, o_base, o_offsets, mask=o_mask)

        if IS_PS_PERSISTENT:
            ps_idx += 1
        elif IS_PERSISTENT or IS_WCA:
            tile_idx += total_programs
        else:
            tile_idx = 1
