# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention kernel for DeepSeek/MLA (Lq != Lv).

Handles split head dims with BLOCK_DPE > 0, persistent CTA scheduling,
and split-K WCA for tile-starved decode shapes.
"""

from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_common import *  # noqa: F403

# ===-----------------------------------------------------------------------===#
# Main Kernel
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_fwd(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,  #
    K_Buffer,
    V_Buffer,  #
    qo_indptr,
    kv_indptr,
    kv_indices,  #
    Mask,
    MaskIndptr,
    WindowKvOffsets,  #
    SM_SCALE: gl.constexpr,
    kv_group_num,  #
    stride_qbs,
    stride_qh,  #
    stride_kbs,
    stride_kh,  #
    stride_vbs,
    stride_vh,  #
    stride_obs,
    stride_oh,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    IS_CAUSAL: gl.constexpr,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    ENABLE_MASK_SPLIT: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    MMA_INSTR_M: gl.constexpr,
    MMA_INSTR_N: gl.constexpr,
    MMA_INSTR_K: gl.constexpr,  #
    QK_K_WIDTH: gl.constexpr,
    PV_K_WIDTH: gl.constexpr,  #
    ASYNC_PAD_K: gl.constexpr,
    ASYNC_PAD_V: gl.constexpr,  #
    Sinks,
    HAS_SINK: gl.constexpr,  #
    LOGIT_CAP: gl.constexpr,  #
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    V_SCALE: gl.constexpr,  #
    GRID_NUM_HEADS: gl.constexpr = 0,  #
    GRID_NUM_M_BLOCKS: gl.constexpr = 0,  #
    NUM_XCDS: gl.constexpr = 0,  #
    V_PRELOAD: gl.constexpr = False,  #
    USE_SUBTILE: gl.constexpr = False,  #
):
    num_warps: gl.constexpr = gl.num_warps()

    if NUM_XCDS > 0:
        # 1D grid with XCD-aware head remapping for load balancing.
        # Grid is linearized as (seq * GRID_NUM_HEADS * GRID_NUM_M_BLOCKS +
        #                         head * GRID_NUM_M_BLOCKS + block_m).
        pid = gl.program_id(0)
        cur_block_m = pid % GRID_NUM_M_BLOCKS
        pid_hb = pid // GRID_NUM_M_BLOCKS
        raw_head = pid_hb % GRID_NUM_HEADS
        cur_seq = pid_hb // GRID_NUM_HEADS
        # Round-robin remap head index across XCDs.
        # For DeepSeek 16 heads / 8 XCDs: each XCD gets exactly 2 heads.
        xcd = raw_head % NUM_XCDS
        local_pid = raw_head // NUM_XCDS
        pids_per_xcd: gl.constexpr = (GRID_NUM_HEADS + NUM_XCDS - 1) // NUM_XCDS
        _rem: gl.constexpr = GRID_NUM_HEADS % NUM_XCDS
        tall_xcds: gl.constexpr = NUM_XCDS if _rem == 0 else _rem
        if _rem == 0:
            # All XCDs have equal load; every raw_head maps to the tall branch.
            cur_head = xcd * pids_per_xcd + local_pid
        else:
            if xcd < tall_xcds:
                cur_head = xcd * pids_per_xcd + local_pid
            else:
                cur_head = (
                    tall_xcds * pids_per_xcd
                    + (xcd - tall_xcds) * (pids_per_xcd - 1)
                    + local_pid
                )
    else:
        cur_seq = gl.program_id(0)
        cur_head = gl.program_id(1)
        cur_block_m = gl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_q_start_idx = gl.load(qo_indptr + cur_seq)
    seq_len_extend = (gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx).to(tl.int32)
    cur_seq_kv_start_idx = gl.load(kv_indptr + cur_seq)
    seq_len_prefix = (gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx).to(tl.int32)

    if cur_block_m * BLOCK_M >= seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        mask_base_idx = gl.load(MaskIndptr + cur_seq).to(tl.int64)
        window_kv_offset = 0
        if SLIDING_WINDOW_SIZE > 0:
            window_kv_offset = gl.load(WindowKvOffsets + cur_seq)
        cur_seq_len = seq_len_prefix + seq_len_extend
        mask_row_stride = (cur_seq_len + window_kv_offset).to(tl.int64)
        mask_base_idx = mask_base_idx + window_kv_offset.to(tl.int64)
        mask_kv_col_offset = (seq_len_prefix).to(tl.int64)
    else:
        mask_base_idx = tl.cast(0, tl.int64)
        mask_row_stride = tl.cast(0, tl.int64)
        mask_kv_col_offset = tl.cast(0, tl.int64)

    # layouts (same as v1)
    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    k_width: gl.constexpr = QK_K_WIDTH
    threads_per_warp: gl.constexpr = 64
    pv_k_width: gl.constexpr = PV_K_WIDTH

    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=k_width
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=k_width
    )
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=pv_k_width
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=pv_k_width
    )

    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[threads_per_warp // 4, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=offs_d_layout)

    USE_SERIAL: gl.constexpr = num_warps < 8

    # Q load
    q_ptrs = (
        Q_Extend
        + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=offs_d_layout)
        qpe_ptrs = (
            Q_Extend
            + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
        if ACTUAL_BLOCK_DPE != BLOCK_DPE:
            qpe_mask = qpe_mask & (offs_dpe[None, :] < (BLOCK_DMODEL + ACTUAL_BLOCK_DPE))
        qpe = gl.load(qpe_ptrs, mask=qpe_mask, other=0.0)
    else:
        qpe = q
    qpe_dot = gl.convert_layout(qpe, q_dot_layout)

    # softmax state
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)
    qk_scale: gl.constexpr = SM_SCALE * LOG2E

    q_abs_pos = (
        seq_len_prefix
        + cur_block_m * BLOCK_M
        + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    )
    q_extend_raw = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    if USE_CUSTOM_MASK:
        q_extend_offs = tl.minimum(q_extend_raw, tl.maximum(seq_len_extend - 1, 0))
    else:
        q_extend_offs = q_extend_raw

    if XAI_TEMPERATURE_LEN > 0:
        inv_log2_len = 1.0 / tl.log2(float(XAI_TEMPERATURE_LEN))
        xai_temperature_reg = gl.where(
            q_abs_pos > XAI_TEMPERATURE_LEN,
            tl.log2(q_abs_pos.to(gl.float32)) * inv_log2_len,
            gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row),
        )
    else:
        xai_temperature_reg = gl.full(
            [BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row
        )

    # SWA prefix skip: jump past prefix tiles entirely outside the window.
    # For the M-tile, min q_abs_pos = seq_len_prefix + cur_block_m * BLOCK_M.
    # Any prefix block whose last key position < (min_q - SWS) is fully masked.
    pfx_kv_start = cur_seq_kv_start_idx
    pfx_seq_len = seq_len_prefix
    pfx_q_abs_pos = q_abs_pos
    pfx_mask_base = mask_base_idx
    if SLIDING_WINDOW_SIZE > 0:
        q_min_abs = seq_len_prefix + cur_block_m * BLOCK_M
        first_useful_pos = tl.maximum(q_min_abs - SLIDING_WINDOW_SIZE, 0)
        prefix_skip_n = (first_useful_pos // BLOCK_N) * BLOCK_N
        pfx_kv_start = cur_seq_kv_start_idx + prefix_skip_n
        pfx_seq_len = seq_len_prefix - prefix_skip_n
        pfx_q_abs_pos = q_abs_pos - prefix_skip_n
        if USE_CUSTOM_MASK:
            pfx_mask_base = mask_base_idx + prefix_skip_n.to(tl.int64)

    if USE_SERIAL:
        if USE_SUBTILE and BLOCK_DMODEL >= 512 and NUM_STAGES >= 2:
            # D-subtiled path: split K/V along d into 256-wide halves.
            # Enables BLOCK_N=64 + NUM_STAGES=2 within 160KB LDS.
            BLOCK_DMODEL_HALF: gl.constexpr = BLOCK_DMODEL // 2
            BLOCK_DV_HALF: gl.constexpr = BLOCK_DV // 2

            # Half-K^T layout: [256, BLOCK_N] — based on existing BLOCK_DMODEL>=256 pattern
            # Must have 6 lane_bases for 64-thread wavefronts (CDNA4)
            kt_half_offset_bases: gl.constexpr = [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0],
                [0, 16], [0, 32],
                [0, 1], [0, 2], [0, 4], [0, 8],
            ] if BLOCK_N >= 64 else [
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0],
                [0, 16],
                [0, 1], [0, 2], [0, 4], [0, 8],
            ]
            kt_half_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [128, 0]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                warp_bases=[[0, 1], [0, 2]],
                block_bases=[],
                shape=[BLOCK_DMODEL_HALF, BLOCK_N],
            ) if BLOCK_N >= 64 else DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
                lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]],
                warp_bases=[[0, 1], [0, 2]],
                block_bases=[],
                shape=[BLOCK_DMODEL_HALF, BLOCK_N],
            )

            # Half-V layout: [BLOCK_N, 256]
            v_half_offset_bases: gl.constexpr = [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128],
                [16, 0], [32, 0],
                [1, 0], [2, 0], [4, 0], [8, 0],
            ] if BLOCK_N >= 64 else [
                [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128],
                [16, 0],
                [1, 0], [2, 0], [4, 0], [8, 0],
            ]
            v_half_async_layout: gl.constexpr = DistributedLinearLayout(
                reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [0, 128]],
                lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                warp_bases=[[1, 0], [2, 0]],
                block_bases=[],
                shape=[BLOCK_N, BLOCK_DV_HALF],
            ) if BLOCK_N >= 64 else DistributedLinearLayout(
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

            ASYNC_KPE: gl.constexpr = (
                BLOCK_DPE > 0 and Q_Extend.dtype.element_ty != tl.float32
            )
            if ASYNC_KPE:
                if BLOCK_DPE >= 64:
                    if BLOCK_N >= 64:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 32],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 32]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                    else:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                else:
                    kpe_offset_bases: gl.constexpr = [
                        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                        [0, 16], [0, 32],
                        [0, 1], [0, 2], [0, 4], [0, 8],
                    ]
                    kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [0, 4]],
                        lane_bases=[[4, 0], [8, 0], [16, 0], [0, 8], [0, 16], [0, 32]],
                        warp_bases=[[0, 1], [0, 2]],
                        block_bases=[],
                        shape=[BLOCK_DPE, BLOCK_N],
                    )
                kpe_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_K]],
                    offset_bases=kpe_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_DPE, BLOCK_N],
                )
                kpe_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DPE, BLOCK_N],
                    layout=kpe_smem_layout,
                )
            else:
                kpe_smem = kt_lo_smem
                kpe_async_layout: gl.constexpr = kt_half_async_layout

            for _s in gl.static_range(NUM_STAGES):
                vz = gl.zeros(
                    [BLOCK_N, BLOCK_DV_HALF],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_half_async_layout,
                )
                v_half_smem.index(_s).store(vz)
            gl.barrier()

            # Split Q into lo [BLOCK_M, 256] and hi [BLOCK_M, 256]
            offs_d_lo = gl.arange(0, BLOCK_DMODEL_HALF, layout=offs_d_layout)
            offs_d_hi = BLOCK_DMODEL_HALF + gl.arange(0, BLOCK_DMODEL_HALF, layout=offs_d_layout)

            q_lo_ptrs = (
                Q_Extend
                + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
                + cur_head * stride_qh
                + offs_d_lo[None, :]
            )
            q_hi_ptrs = (
                Q_Extend
                + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
                + cur_head * stride_qh
                + offs_d_hi[None, :]
            )
            q_m_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
            q_lo = gl.load(q_lo_ptrs, mask=q_m_mask, other=0.0)
            q_hi = gl.load(q_hi_ptrs, mask=q_m_mask, other=0.0)
            q_dot_lo = gl.convert_layout(q_lo, q_dot_layout)
            q_dot_hi = gl.convert_layout(q_hi, q_dot_layout)

            # Split accumulators
            acc_lo = gl.zeros([BLOCK_M, BLOCK_DV_HALF], dtype=gl.float32, layout=mma_layout)
            acc_hi = gl.zeros([BLOCK_M, BLOCK_DV_HALF], dtype=gl.float32, layout=mma_layout)

            # Prefix (non-subtiled — use full-D smem that's separate from extend smem)
            # For simplicity, skip prefix subtiling; prefix uses global loads.
            if pfx_seq_len > 0:
                n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
                for pfx_bn in tl.range(0, n_prefix_blocks):
                    pfx_start_n = pfx_bn * BLOCK_N
                    kv_loc_ptrs = kv_indices + pfx_kv_start + pfx_start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                    pfx_n_mask = (pfx_start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)) < pfx_seq_len
                    kv_locs = gl.load(kv_loc_ptrs, mask=pfx_n_mask, other=0).to(tl.int32)

                    # Load K prefix: use kt_half_async_layout-based slices for K^T [D_half, N]
                    kt_pfx_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_half_async_layout)
                    kt_pfx_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_half_async_layout)
                    k_pfx_d_lo = gl.arange(0, BLOCK_DMODEL_HALF, layout=kt_pfx_d_layout)
                    k_pfx_d_hi = BLOCK_DMODEL_HALF + gl.arange(0, BLOCK_DMODEL_HALF, layout=kt_pfx_d_layout)
                    kv_locs_kt = gl.convert_layout(kv_locs, kt_pfx_n_layout)
                    k_pfx_lo_ptrs = K_Buffer + kv_locs_kt[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + k_pfx_d_lo[:, None]
                    k_pfx_hi_ptrs = K_Buffer + kv_locs_kt[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + k_pfx_d_hi[:, None]
                    pfx_n_mask_kt = gl.convert_layout(pfx_n_mask, kt_pfx_n_layout)
                    k_lo_t = gl.load(k_pfx_lo_ptrs, mask=pfx_n_mask_kt[None, :], other=0.0)
                    k_hi_t = gl.load(k_pfx_hi_ptrs, mask=pfx_n_mask_kt[None, :], other=0.0)
                    kt_lo_dot_pfx = gl.convert_layout(k_lo_t, kt_dot_layout)
                    kt_hi_dot_pfx = gl.convert_layout(k_hi_t, kt_dot_layout)

                    qk_pfx = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                    qk_pfx = do_mma(q_dot_lo, kt_lo_dot_pfx, qk_pfx)
                    qk_pfx = do_mma(q_dot_hi, kt_hi_dot_pfx, qk_pfx)

                    if BLOCK_DPE > 0:
                        kpe_pfx_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
                        kpe_pfx_d = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=kpe_pfx_d_layout)
                        kv_locs_kpe = gl.convert_layout(kv_locs, gl.SliceLayout(dim=0, parent=kpe_async_layout))
                        kpe_ptrs = K_Buffer + kv_locs_kpe[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + kpe_pfx_d[:, None]
                        kpe_pfx_mask = gl.convert_layout(pfx_n_mask, gl.SliceLayout(dim=0, parent=kpe_async_layout))
                        kpe_t = gl.load(kpe_ptrs, mask=kpe_pfx_mask[None, :], other=0.0)
                        kpe_dot_pfx = gl.convert_layout(kpe_t, kt_dot_layout)
                        qk_pfx = do_mma(qpe_dot, kpe_dot_pfx, qk_pfx)

                    # Load V prefix halves: V is [BLOCK_N, BLOCK_DV_HALF]
                    v_pfx_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_half_async_layout)
                    v_pfx_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_half_async_layout)
                    v_pfx_n_offs = gl.arange(0, BLOCK_N, layout=v_pfx_n_layout)
                    kv_locs_v = gl.load(kv_indices + pfx_kv_start + pfx_start_n + v_pfx_n_offs, mask=(pfx_start_n + v_pfx_n_offs) < pfx_seq_len, other=0).to(tl.int32)
                    v_pfx_d_lo = gl.arange(0, BLOCK_DV_HALF, layout=v_pfx_d_layout)
                    v_pfx_d_hi = BLOCK_DV_HALF + gl.arange(0, BLOCK_DV_HALF, layout=v_pfx_d_layout)
                    v_pfx_lo_ptrs = V_Buffer + kv_locs_v[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + v_pfx_d_lo[None, :]
                    v_pfx_hi_ptrs = V_Buffer + kv_locs_v[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + v_pfx_d_hi[None, :]
                    v_pfx_mask = (pfx_start_n + v_pfx_n_offs) < pfx_seq_len
                    v_lo_pfx = gl.load(v_pfx_lo_ptrs, mask=v_pfx_mask[:, None], other=0.0)
                    v_hi_pfx = gl.load(v_pfx_hi_ptrs, mask=v_pfx_mask[:, None], other=0.0)
                    v_lo_dot_pfx = gl.convert_layout(v_lo_pfx, v_dot_layout)
                    v_hi_dot_pfx = gl.convert_layout(v_hi_pfx, v_dot_layout)

                    # Softmax (prefix uses absolute positions for causal masking)
                    m_i_old = m_i
                    acc_lo, l_i, m_i, p = compute_softmax_prefix(
                        acc_lo, l_i, m_i, qk_pfx, pfx_start_n,
                        pfx_seq_len, qk_scale, LOGIT_CAP,
                        xai_temperature_reg, XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos, SLIDING_WINDOW_SIZE,
                        Mask, pfx_mask_base,
                        mask_row_stride, q_extend_offs,
                        USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED, BLOCK_M, BLOCK_N,
                        mma_layout, mma_offs_n_col,
                    )
                    acc_hi = acc_hi * gl.exp2(m_i_old - m_i)[:, None]

                    p_cast = p.to(v_lo_dot_pfx.dtype)
                    p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
                    acc_lo = do_mma(p_dot_reg, v_lo_dot_pfx, acc_lo)
                    acc_hi = do_mma(p_dot_reg, v_hi_dot_pfx, acc_hi)

            # Extend: subtiled inner loop
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc_lo, acc_hi, l_i, m_i = attn_fwd_inner_extend_subtiled(
                    acc_lo, acc_hi, l_i, m_i,
                    q_dot_lo, q_dot_hi, qpe_dot,
                    k_extend_base, v_extend_base,
                    cur_block_m, seq_len_extend,
                    stride_kbs, stride_vbs,
                    0, n_full_blocks,
                    kt_lo_smem, kt_hi_smem, kpe_smem,
                    v_lo_smem, v_hi_smem,
                    qk_scale, LOGIT_CAP,
                    xai_temperature_reg, XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE, IS_CAUSAL,
                    Mask, mask_base_idx, mask_row_stride, mask_kv_col_offset,
                    USE_CUSTOM_MASK, False,
                    BLOCK_M, BLOCK_N,
                    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE, ACTUAL_BLOCK_DPE,
                    BLOCK_DV, ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_half_async_layout, kpe_async_layout, v_half_async_layout,
                    kt_dot_layout, kt_dot_layout,
                    p_dot_layout, v_dot_layout,
                    mma_layout, mma_offs_n_col, mma_offs_m_row,
                    ASYNC_KPE,
                )

            # Masked tail blocks (use global loads, same pattern as prefix)
            masked_start = n_full_blocks
            for ext_bn in tl.range(masked_start, n_extend_blocks):
                ext_start_n = ext_bn * BLOCK_N
                # Load K extend halves from global
                ext_n_offs = ext_start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
                ext_n_mask = ext_n_offs < seq_len_extend

                k_ext_base = K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
                k_ext_lo_ptrs = k_ext_base + ext_n_offs[None, :] * stride_kbs + gl.arange(0, BLOCK_DMODEL_HALF, layout=mma_offs_m_row)[:, None]
                k_ext_hi_ptrs = k_ext_base + ext_n_offs[None, :] * stride_kbs + (BLOCK_DMODEL_HALF + gl.arange(0, BLOCK_DMODEL_HALF, layout=mma_offs_m_row)[:, None])
                k_lo_ext = gl.load(k_ext_lo_ptrs, mask=ext_n_mask[None, :], other=0.0)
                k_hi_ext = gl.load(k_ext_hi_ptrs, mask=ext_n_mask[None, :], other=0.0)
                kt_lo_dot_ext = gl.convert_layout(k_lo_ext, kt_dot_layout)
                kt_hi_dot_ext = gl.convert_layout(k_hi_ext, kt_dot_layout)

                qk_ext = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk_ext = do_mma(q_dot_lo, kt_lo_dot_ext, qk_ext)
                qk_ext = do_mma(q_dot_hi, kt_hi_dot_ext, qk_ext)

                if BLOCK_DPE > 0:
                    kpe_ext_ptrs = k_ext_base + ext_n_offs[None, :] * stride_kbs + (BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=mma_offs_m_row)[:, None])
                    kpe_ext = gl.load(kpe_ext_ptrs, mask=ext_n_mask[None, :], other=0.0)
                    kpe_dot_ext = gl.convert_layout(kpe_ext, kt_dot_layout)
                    qk_ext = do_mma(qpe_dot, kpe_dot_ext, qk_ext)

                v_ext_base = V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
                v_ext_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_half_async_layout)
                v_ext_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_half_async_layout)
                v_ext_n_offs = ext_start_n + gl.arange(0, BLOCK_N, layout=v_ext_n_layout)
                v_ext_n_mask = v_ext_n_offs < seq_len_extend
                v_ext_d_lo = gl.arange(0, BLOCK_DV_HALF, layout=v_ext_d_layout)
                v_ext_d_hi = BLOCK_DV_HALF + gl.arange(0, BLOCK_DV_HALF, layout=v_ext_d_layout)
                v_ext_lo_ptrs = v_ext_base + v_ext_n_offs[:, None] * stride_vbs + v_ext_d_lo[None, :]
                v_ext_hi_ptrs = v_ext_base + v_ext_n_offs[:, None] * stride_vbs + v_ext_d_hi[None, :]
                v_lo_ext = gl.load(v_ext_lo_ptrs, mask=v_ext_n_mask[:, None], other=0.0)
                v_hi_ext = gl.load(v_ext_hi_ptrs, mask=v_ext_n_mask[:, None], other=0.0)
                v_lo_dot_ext = gl.convert_layout(v_lo_ext, v_dot_layout)
                v_hi_dot_ext = gl.convert_layout(v_hi_ext, v_dot_layout)

                m_i_old = m_i
                acc_lo, l_i, m_i, p = compute_softmax_extend(
                    acc_lo, l_i, m_i, qk_ext, ext_start_n,
                    cur_block_m, seq_len_extend, qk_scale, LOGIT_CAP,
                    xai_temperature_reg, XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE, IS_CAUSAL,
                    Mask, mask_base_idx, mask_row_stride, mask_kv_col_offset,
                    USE_CUSTOM_MASK, True,
                    BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
                )
                acc_hi = acc_hi * gl.exp2(m_i_old - m_i)[:, None]

                p_cast = p.to(v_lo_dot_ext.dtype)
                p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
                acc_lo = do_mma(p_dot_reg, v_lo_dot_ext, acc_lo)
                acc_hi = do_mma(p_dot_reg, v_hi_dot_ext, acc_hi)

            # Normalize and store both halves
            l_recip = 1.0 / l_i
            acc_lo = acc_lo * l_recip[:, None]
            acc_hi = acc_hi * l_recip[:, None]
            if V_SCALE != 1.0:
                acc_lo = acc_lo * V_SCALE
                acc_hi = acc_hi * V_SCALE

            o_base = (
                O_Extend
                + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
                + cur_head * stride_oh
            )
            o_m_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
            offs_dv_lo = gl.arange(0, BLOCK_DV_HALF, layout=offs_d_layout)
            offs_dv_hi = BLOCK_DV_HALF + gl.arange(0, BLOCK_DV_HALF, layout=offs_d_layout)

            out_lo = gl.convert_layout(acc_lo, blocked_layout).to(O_Extend.dtype.element_ty)
            gl.store(o_base + offs_dv_lo[None, :], out_lo, mask=o_m_mask)
            out_hi = gl.convert_layout(acc_hi, blocked_layout).to(O_Extend.dtype.element_ty)
            gl.store(o_base + offs_dv_hi[None, :], out_hi, mask=o_m_mask)

        elif NUM_STAGES >= 2 and BLOCK_DMODEL >= 128:
            # 4-warp DMA path
            if BLOCK_DMODEL >= 512:
                if BLOCK_N >= 64:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
                        [0, 16], [0, 32],
                        [0, 1], [0, 2], [0, 4], [0, 8],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16], [0, 32]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
                        warp_bases=[[0, 1], [0, 2]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                else:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
                        [0, 16],
                        [0, 1], [0, 2], [0, 4], [0, 8],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
                        warp_bases=[[0, 1], [0, 2]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
            elif BLOCK_DMODEL >= 256:
                kt_offset_bases: gl.constexpr = [
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                    [16, 0],
                    [32, 0],
                    [64, 0],
                    [128, 0],
                    [0, 16],
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                ]
                kt_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
                    lane_bases=[
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [128, 0],
                        [0, 16],
                    ],
                    warp_bases=[[0, 1], [0, 2]],
                    block_bases=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
            else:
                if BLOCK_N >= 128:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 64]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                        warp_bases=[[0, 1], [0, 2]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                else:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [0, 16],
                        [0, 32],
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                        warp_bases=[[0, 1], [0, 2]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )

            if BLOCK_DV >= 512:
                if BLOCK_N >= 64:
                    v_offset_bases: gl.constexpr = [
                        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                        [16, 0], [32, 0],
                        [1, 0], [2, 0], [4, 0], [8, 0],
                    ]
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0], [32, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
                        warp_bases=[[1, 0], [2, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
                else:
                    v_offset_bases: gl.constexpr = [
                        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                        [16, 0],
                        [1, 0], [2, 0], [4, 0], [8, 0],
                    ]
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
                        warp_bases=[[1, 0], [2, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
            elif BLOCK_DV >= 256:
                v_offset_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                    [0, 16],
                    [0, 32],
                    [0, 64],
                    [0, 128],
                    [16, 0],
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                ]
                v_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                    lane_bases=[
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [0, 128],
                        [16, 0],
                    ],
                    warp_bases=[[1, 0], [2, 0]],
                    block_bases=[],
                    shape=[BLOCK_N, BLOCK_DV],
                )
            else:
                if BLOCK_N >= 128:
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [64, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                        warp_bases=[[1, 0], [2, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
                else:
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [16, 0],
                        [32, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                        warp_bases=[[1, 0], [2, 0]],
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
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=v_smem_layout,
            )

            ASYNC_KPE: gl.constexpr = (
                BLOCK_DPE > 0
                and Q_Extend.dtype.element_ty != tl.float32
            )

            if ASYNC_KPE:
                if BLOCK_DPE >= 64:
                    if BLOCK_N >= 128:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 32], [0, 64],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 32], [0, 64]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                    elif BLOCK_N >= 64:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 32],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 32]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                    else:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                elif BLOCK_DPE >= 32:
                    if BLOCK_N >= 128:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                            [0, 16], [0, 32], [0, 64],
                            [0, 1], [0, 2], [0, 4], [0, 8],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [0, 4], [0, 64]],
                            lane_bases=[[4, 0], [8, 0], [16, 0], [0, 8], [0, 16], [0, 32]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                    else:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                            [0, 16], [0, 32],
                            [0, 1], [0, 2], [0, 4], [0, 8],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [0, 4]],
                            lane_bases=[[4, 0], [8, 0], [16, 0], [0, 8], [0, 16], [0, 32]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                else:
                    if BLOCK_N >= 128:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0],
                            [0, 16], [0, 32], [0, 64],
                            [0, 1], [0, 2], [0, 4], [0, 8],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [0, 4], [0, 64]],
                            lane_bases=[[2, 0], [4, 0], [8, 0], [0, 8], [0, 16], [0, 32]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                    else:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0],
                            [0, 16], [0, 32],
                            [0, 1], [0, 2], [0, 4], [0, 8],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [0, 4]],
                            lane_bases=[[2, 0], [4, 0], [8, 0], [0, 8], [0, 16], [0, 32]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                kpe_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_K]],
                    offset_bases=kpe_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_DPE, BLOCK_N],
                )
                kpe_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DPE, BLOCK_N],
                    layout=kpe_smem_layout,
                )
            else:
                kpe_smem = kt_smem
                kpe_async_layout: gl.constexpr = kt_async_layout

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix (same as v1)
            if pfx_seq_len > 0:
                n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
                n_extend_est = (seq_len_extend + BLOCK_N - 1) // BLOCK_N
                use_pipe_prefix = n_prefix_blocks >= NUM_STAGES
                if LOGIT_CAP > 0:
                    use_pipe_prefix = use_pipe_prefix and (n_extend_est < NUM_STAGES)
                if use_pipe_prefix:
                    acc, l_i, m_i = attn_fwd_inner_prefix_dma_simple(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        ASYNC_KPE,
                    )
                else:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        V_PRELOAD=V_PRELOAD,
                    )

            cdna4_async.wait_group(0)

            # EXTEND: per-CTA dispatch (v2 change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_dma(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )
            masked_start = n_full_blocks
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_dma(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )

        else:
            # 4-warp serial path (unchanged -- already correct for any n_extend_blocks)
            kt_blocked_layout: gl.constexpr = gl.BlockedLayout(
                size_per_thread=[1, 8],
                threads_per_warp=[threads_per_warp // 4, 4],
                warps_per_cta=[1, num_warps],
                order=[0, 1],
            )
            kt_serial_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                vec=8,
                per_phase=1,
                max_phase=16,
                order=[0, 1],
            )
            v_serial_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                vec=8,
                per_phase=1,
                max_phase=16,
                order=[1, 0],
            )
            q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                vec=8,
                per_phase=1,
                max_phase=16,
                order=[1, 0],
            )

            kt_serial_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_DMODEL, BLOCK_N],
                layout=kt_serial_smem_layout,
            )
            if BLOCK_DPE > 0:
                kt_dpe_serial_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_DPE, BLOCK_N],
                    layout=kt_serial_smem_layout,
                )
            else:
                kt_dpe_serial_smem = kt_serial_smem
            v_serial_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_N, BLOCK_DV],
                layout=v_serial_smem_layout,
            )
            q_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_M, BLOCK_DMODEL],
                layout=q_smem_layout,
            )

            q_smem.store(q)
            q_dot = q_smem.load(q_dot_layout)
            qpe_dot = gl.convert_layout(qpe, q_dot_layout)

            if pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_serial(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_serial_smem,
                    kt_dpe_serial_smem,
                    v_serial_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_blocked_layout,
                    blocked_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    V_PRELOAD,
                )

            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_serial(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_serial_smem,
                    kt_dpe_serial_smem,
                    v_serial_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_blocked_layout,
                    blocked_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD,
                )
            masked_start = n_full_blocks
            if n_extend_blocks > masked_start:
                acc, l_i, m_i = attn_fwd_inner_extend_serial(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_serial_smem,
                    kt_dpe_serial_smem,
                    v_serial_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_blocked_layout,
                    blocked_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD,
                )

    else:
        # 8-warp DMA path

        if BLOCK_DMODEL >= 128:
            if BLOCK_DMODEL >= 512:
                if BLOCK_N >= 64:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
                        [0, 16], [0, 32],
                        [0, 1], [0, 2], [0, 4], [0, 8],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
                        warp_bases=[[0, 1], [0, 2], [0, 32]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                else:
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
                if BLOCK_DV >= 512:
                    if BLOCK_N >= 64:
                        v_offset_bases: gl.constexpr = [
                            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                            [16, 0], [32, 0],
                            [1, 0], [2, 0], [4, 0], [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
                            warp_bases=[[1, 0], [2, 0], [32, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    else:
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
                elif BLOCK_DV >= 256:
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [0, 128],
                        [16, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                        lane_bases=[
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [0, 128],
                            [16, 0],
                        ],
                        warp_bases=[[1, 0], [2, 0], [4, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
                else:
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [16, 0],
                        [32, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                        warp_bases=[[1, 0], [2, 0], [4, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
            elif BLOCK_DMODEL >= 256:
                kt_offset_bases: gl.constexpr = [
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                    [16, 0],
                    [32, 0],
                    [64, 0],
                    [128, 0],
                    [0, 16],
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                ]
                kt_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
                    lane_bases=[
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [128, 0],
                        [0, 16],
                    ],
                    warp_bases=[[0, 1], [0, 2], [0, 4]],
                    block_bases=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                if BLOCK_DV >= 512:
                    v_offset_bases: gl.constexpr = [
                        [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                        [16, 0], [32, 0],
                        [1, 0], [2, 0], [4, 0], [8, 0],
                    ]
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
                        warp_bases=[[1, 0], [2, 0], [32, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
                elif BLOCK_DV >= 256:
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [0, 128],
                        [16, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                        lane_bases=[
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [0, 128],
                            [16, 0],
                        ],
                        warp_bases=[[1, 0], [2, 0], [4, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
                else:
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [16, 0],
                        [32, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                        warp_bases=[[1, 0], [2, 0], [4, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
            else:
                if BLOCK_N >= 128:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                    ]
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 64]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                        warp_bases=[[0, 1], [0, 2], [0, 4]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [64, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                        warp_bases=[[1, 0], [2, 0], [4, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
                else:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [0, 16],
                        [0, 32],
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                    ]
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [16, 0],
                        [32, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                        warp_bases=[[0, 1], [0, 2], [0, 4]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                        warp_bases=[[1, 0], [2, 0], [4, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )

            kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_K]],
                offset_bases=kt_offset_bases,
                cga_layout=[],
                shape=[BLOCK_DMODEL, BLOCK_N],
            )
            v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_V]],
                offset_bases=v_offset_bases,
                cga_layout=[],
                shape=[BLOCK_N, BLOCK_DV],
            )

            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=v_async_smem_layout,
            )

            ASYNC_KPE: gl.constexpr = (
                BLOCK_DPE > 0
                and Q_Extend.dtype.element_ty != tl.float32
            )

            if ASYNC_KPE:
                if BLOCK_DPE >= 64:
                    if BLOCK_N >= 128:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 32], [0, 64],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 64]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2], [0, 32]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                    elif BLOCK_N >= 64:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 32],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2], [0, 32]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                    else:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                            [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 4],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0]],
                            lane_bases=[[4, 0], [8, 0], [16, 0], [32, 0], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2], [0, 4]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                elif BLOCK_DPE >= 32:
                    if BLOCK_N >= 128:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 32], [0, 64],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [0, 64]],
                            lane_bases=[[4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2], [0, 32]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                    else:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 32],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0]],
                            lane_bases=[[4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2], [0, 32]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                else:
                    if BLOCK_N >= 128:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 32], [0, 64],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [0, 64]],
                            lane_bases=[[2, 0], [4, 0], [8, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2], [0, 32]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                    else:
                        kpe_offset_bases: gl.constexpr = [
                            [1, 0], [2, 0], [4, 0], [8, 0],
                            [0, 4], [0, 8], [0, 16],
                            [0, 1], [0, 2], [0, 32],
                        ]
                        kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0]],
                            lane_bases=[[2, 0], [4, 0], [8, 0], [0, 4], [0, 8], [0, 16]],
                            warp_bases=[[0, 1], [0, 2], [0, 32]],
                            block_bases=[],
                            shape=[BLOCK_DPE, BLOCK_N],
                        )
                kpe_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_K]],
                    offset_bases=kpe_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_DPE, BLOCK_N],
                )
                kpe_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DPE, BLOCK_N],
                    layout=kpe_async_smem_layout,
                )
            else:
                kpe_smem = kt_smem
                kpe_async_layout: gl.constexpr = kt_async_layout

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix dispatch (same as v1)
            n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
            if n_prefix_blocks >= NUM_STAGES:
                if NUM_STAGES >= 3:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined_scalar_mask(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        ASYNC_KPE,
                    )
                else:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        ASYNC_KPE,
                    )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    V_PRELOAD=V_PRELOAD,
                )

            # EXTEND: per-CTA dispatch (v2 core change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )
            masked_start = n_full_blocks
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )

        else:
            # 8-warp BLOCK_DMODEL < 128
            if BLOCK_N >= 128:
                kt_offset_bases: gl.constexpr = [
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                    [16, 0],
                    [32, 0],
                    [0, 16],
                    [0, 32],
                    [0, 64],
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                ]
                v_offset_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                    [0, 16],
                    [0, 32],
                    [16, 0],
                    [32, 0],
                    [64, 0],
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                ]
                kt_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[1, 0], [2, 0], [4, 0], [0, 64]],
                    lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
                    warp_bases=[[0, 2], [0, 4], [0, 8]],
                    block_bases=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[0, 1], [0, 2], [0, 4], [64, 0]],
                    lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
                    warp_bases=[[2, 0], [4, 0], [8, 0]],
                    block_bases=[],
                    shape=[BLOCK_N, BLOCK_DV],
                )
            else:
                kt_offset_bases: gl.constexpr = [
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                    [16, 0],
                    [32, 0],
                    [0, 16],
                    [0, 32],
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                ]
                v_offset_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 4],
                    [0, 8],
                    [0, 16],
                    [0, 32],
                    [16, 0],
                    [32, 0],
                    [1, 0],
                    [2, 0],
                    [4, 0],
                    [8, 0],
                ]
                kt_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[1, 0], [2, 0], [4, 0]],
                    lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
                    warp_bases=[[0, 2], [0, 4], [0, 8]],
                    block_bases=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_async_layout: gl.constexpr = DistributedLinearLayout(
                    reg_bases=[[0, 1], [0, 2], [0, 4]],
                    lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
                    warp_bases=[[2, 0], [4, 0], [8, 0]],
                    block_bases=[],
                    shape=[BLOCK_N, BLOCK_DV],
                )

            kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_K]],
                offset_bases=kt_offset_bases,
                cga_layout=[],
                shape=[BLOCK_DMODEL, BLOCK_N],
            )
            v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                interval_padding_pairs=[[512, ASYNC_PAD_V]],
                offset_bases=v_offset_bases,
                cga_layout=[],
                shape=[BLOCK_N, BLOCK_DV],
            )

            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=v_async_smem_layout,
            )

            ASYNC_KPE: gl.constexpr = False
            kpe_smem = kt_smem
            kpe_async_layout: gl.constexpr = kt_async_layout

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix
            n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
            if n_prefix_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    ASYNC_KPE,
                )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    V_PRELOAD=V_PRELOAD,
                )

            # EXTEND: per-CTA dispatch (v2 core change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )
            masked_start = n_full_blocks
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )

    # sinks
    if HAS_SINK:
        cur_sink = gl.load(Sinks + cur_head)
        l_i = l_i + gl.exp2(cur_sink * LOG2E - m_i)

    # normalize and store
    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]
    if V_SCALE != 1.0:
        acc = acc * V_SCALE

    o_ptrs = (
        O_Extend
        + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    if ACTUAL_BLOCK_DV != BLOCK_DV:
        o_mask = o_mask & (offs_dv[None, :] < ACTUAL_BLOCK_DV)
    out = gl.convert_layout(acc, blocked_layout).to(O_Extend.dtype.element_ty)
    gl.store(o_ptrs, out, mask=o_mask)


# ===-----------------------------------------------------------------------===#
# Persistent-CTA Kernel (work-centric scheduling)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_fwd_persistent(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,  #
    K_Buffer,
    V_Buffer,  #
    qo_indptr,
    kv_indptr,
    kv_indices,  #
    Mask,
    MaskIndptr,
    WindowKvOffsets,  #
    SM_SCALE: gl.constexpr,
    kv_group_num,  #
    stride_qbs,
    stride_qh,  #
    stride_kbs,
    stride_kh,  #
    stride_vbs,
    stride_vh,  #
    stride_obs,
    stride_oh,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    num_heads,  #         int32 scalar -- total Q heads
    n_m_tiles,  #         int32 scalar -- ceil(max_len_extend / BLOCK_M)
    total_valid_tiles,  # int32 scalar -- batch * num_heads * n_m_tiles [* SPLIT_K]
    total_programs,  #    int32 scalar (= grid dim 0)
    IS_CAUSAL: gl.constexpr,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    ENABLE_MASK_SPLIT: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    MMA_INSTR_M: gl.constexpr,
    MMA_INSTR_N: gl.constexpr,
    MMA_INSTR_K: gl.constexpr,  #
    QK_K_WIDTH: gl.constexpr,
    PV_K_WIDTH: gl.constexpr,  #
    ASYNC_PAD_K: gl.constexpr,
    ASYNC_PAD_V: gl.constexpr,  #
    Sinks,
    HAS_SINK: gl.constexpr,  #
    LOGIT_CAP: gl.constexpr,  #
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    V_SCALE: gl.constexpr,  #
    SPLIT_K: gl.constexpr,  #  1 = normal, >1 = split-K across prefix
    partial_out,  #             workspace [total_output_tiles * SPLIT_K, BLOCK_M, BLOCK_DV] fp32
    partial_lse,  #             workspace [total_output_tiles * SPLIT_K, BLOCK_M] fp32
    V_PRELOAD: gl.constexpr = False,  #
):
    num_warps: gl.constexpr = gl.num_warps()
    cta_id = gl.program_id(0)

    mma_layout: gl.constexpr = AMDMFMALayout(
        version=4,
        instr_shape=[MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    k_width: gl.constexpr = QK_K_WIDTH
    threads_per_warp: gl.constexpr = 64
    pv_k_width: gl.constexpr = PV_K_WIDTH

    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=k_width
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=k_width
    )
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=mma_layout, k_width=pv_k_width
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=mma_layout, k_width=pv_k_width
    )

    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[threads_per_warp // 4, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=offs_d_layout)

    USE_SERIAL: gl.constexpr = num_warps < 8
    qk_scale: gl.constexpr = SM_SCALE * LOG2E

    # === Persistent tile loop ===
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
        cur_head = rem // n_m_tiles
        cur_block_m = rem % n_m_tiles
        cur_kv_head = cur_head // kv_group_num

        cur_seq_q_start_idx = gl.load(qo_indptr + cur_seq)
        seq_len_extend = (gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx).to(tl.int32)

        # For ragged batches: padded tiles beyond the actual sequence length
        # have their lengths zeroed so prefix/extend loops iterate 0 blocks
        # and output stores are fully masked.
        is_valid_tile = cur_block_m * BLOCK_M < seq_len_extend
        seq_len_extend = tl.where(is_valid_tile, seq_len_extend, 0)

        cur_seq_kv_start_idx = gl.load(kv_indptr + cur_seq)
        seq_len_prefix_raw = (gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx).to(tl.int32)
        seq_len_prefix = tl.where(is_valid_tile, seq_len_prefix_raw, 0)

        if USE_CUSTOM_MASK:
            mask_base_idx = gl.load(MaskIndptr + cur_seq).to(tl.int64)
            window_kv_offset = 0
            if SLIDING_WINDOW_SIZE > 0:
                window_kv_offset = gl.load(WindowKvOffsets + cur_seq)
            cur_seq_len = seq_len_prefix + seq_len_extend
            mask_row_stride = (cur_seq_len + window_kv_offset).to(tl.int64)
            mask_base_idx = mask_base_idx + window_kv_offset.to(tl.int64)
            mask_kv_col_offset = (seq_len_prefix).to(tl.int64)
        else:
            mask_base_idx = tl.cast(0, tl.int64)
            mask_row_stride = tl.cast(0, tl.int64)
            mask_kv_col_offset = tl.cast(0, tl.int64)

        q_ptrs = (
            Q_Extend
            + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        q = gl.load(q_ptrs, mask=q_mask, other=0.0)
        if BLOCK_DPE > 0:
            offs_dpe = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=offs_d_layout)
            qpe_ptrs = (
                Q_Extend
                + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                * stride_qbs
                + cur_head * stride_qh
                + offs_dpe[None, :]
            )
            qpe_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
            if ACTUAL_BLOCK_DPE != BLOCK_DPE:
                qpe_mask = qpe_mask & (offs_dpe[None, :] < (BLOCK_DMODEL + ACTUAL_BLOCK_DPE))
            qpe = gl.load(qpe_ptrs, mask=qpe_mask, other=0.0)
        else:
            qpe = q
        qpe_dot = gl.convert_layout(qpe, q_dot_layout)

        m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
        l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
        acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

        q_abs_pos = (
            seq_len_prefix
            + cur_block_m * BLOCK_M
            + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        )
        q_extend_raw = cur_block_m * BLOCK_M + gl.arange(
            0, BLOCK_M, layout=mma_offs_m_row
        )
        if USE_CUSTOM_MASK:
            q_extend_offs = tl.minimum(q_extend_raw, tl.maximum(seq_len_extend - 1, 0))
        else:
            q_extend_offs = q_extend_raw

        if XAI_TEMPERATURE_LEN > 0:
            inv_log2_len = 1.0 / tl.log2(float(XAI_TEMPERATURE_LEN))
            xai_temperature_reg = gl.where(
                q_abs_pos > XAI_TEMPERATURE_LEN,
                tl.log2(q_abs_pos.to(gl.float32)) * inv_log2_len,
                gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row),
            )
        else:
            xai_temperature_reg = gl.full(
                [BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row
            )

        pfx_kv_start = cur_seq_kv_start_idx
        pfx_seq_len = seq_len_prefix
        pfx_q_abs_pos = q_abs_pos
        pfx_mask_base = mask_base_idx
        if SLIDING_WINDOW_SIZE > 0:
            q_min_abs = seq_len_prefix + cur_block_m * BLOCK_M
            first_useful_pos = tl.maximum(q_min_abs - SLIDING_WINDOW_SIZE, 0)
            prefix_skip_n = (first_useful_pos // BLOCK_N) * BLOCK_N
            pfx_kv_start = cur_seq_kv_start_idx + prefix_skip_n
            pfx_seq_len = seq_len_prefix - prefix_skip_n
            pfx_q_abs_pos = q_abs_pos - prefix_skip_n
            if USE_CUSTOM_MASK:
                pfx_mask_base = mask_base_idx + prefix_skip_n.to(tl.int64)

        # Split-K: partition the prefix KV range across K splits.
        # Non-last splits only process their prefix chunk; last split also
        # handles the extend tokens.
        # Save original extend length for the partial-output write mask —
        # all splits must write their results even if they only saw prefix.
        orig_seq_len_extend = seq_len_extend
        if SPLIT_K > 1:
            n_pfx_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
            blocks_per_split = (n_pfx_blocks + SPLIT_K - 1) // SPLIT_K
            my_block_start = k_split_id * blocks_per_split
            my_block_end = tl.minimum((k_split_id + 1) * blocks_per_split, n_pfx_blocks)
            split_start_offset = my_block_start * BLOCK_N
            pfx_kv_start = pfx_kv_start + split_start_offset
            pfx_seq_len = tl.minimum(my_block_end * BLOCK_N, pfx_seq_len) - split_start_offset
            pfx_seq_len = tl.maximum(pfx_seq_len, 0)
            pfx_q_abs_pos = pfx_q_abs_pos - split_start_offset
            if USE_CUSTOM_MASK:
                pfx_mask_base = pfx_mask_base + split_start_offset.to(tl.int64)
            # Only last split processes the extend tokens.
            if k_split_id < SPLIT_K - 1:
                seq_len_extend = 0

        if USE_SERIAL:
            if NUM_STAGES >= 2 and BLOCK_DMODEL >= 128:
                if BLOCK_DMODEL >= 512:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
                        [0, 16], [0, 32],
                        [0, 1], [0, 2], [0, 4], [0, 8],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16], [0, 32]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
                        warp_bases=[[0, 1], [0, 2]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                    if BLOCK_DV >= 512:
                        v_offset_bases: gl.constexpr = [
                            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                            [16, 0], [32, 0],
                            [1, 0], [2, 0], [4, 0], [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0], [32, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
                            warp_bases=[[1, 0], [2, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    elif BLOCK_DV >= 256:
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [0, 128],
                            [16, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                            lane_bases=[
                                [0, 8],
                                [0, 16],
                                [0, 32],
                                [0, 64],
                                [0, 128],
                                [16, 0],
                            ],
                            warp_bases=[[1, 0], [2, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    else:
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [16, 0],
                            [32, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                            warp_bases=[[1, 0], [2, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                elif BLOCK_DMODEL >= 256:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [128, 0],
                        [0, 16],
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
                        lane_bases=[
                            [8, 0],
                            [16, 0],
                            [32, 0],
                            [64, 0],
                            [128, 0],
                            [0, 16],
                        ],
                        warp_bases=[[0, 1], [0, 2]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                    if BLOCK_DV >= 512:
                        v_offset_bases: gl.constexpr = [
                            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                            [16, 0], [32, 0],
                            [1, 0], [2, 0], [4, 0], [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0], [32, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
                            warp_bases=[[1, 0], [2, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    elif BLOCK_DV >= 256:
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [0, 128],
                            [16, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                            lane_bases=[
                                [0, 8],
                                [0, 16],
                                [0, 32],
                                [0, 64],
                                [0, 128],
                                [16, 0],
                            ],
                            warp_bases=[[1, 0], [2, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    else:
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [16, 0],
                            [32, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                            warp_bases=[[1, 0], [2, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                else:
                    if BLOCK_N >= 128:
                        kt_offset_bases: gl.constexpr = [
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                            [16, 0],
                            [32, 0],
                            [64, 0],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                        ]
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [16, 0],
                            [32, 0],
                            [64, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        kt_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 64]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DMODEL, BLOCK_N],
                        )
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [64, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                            warp_bases=[[1, 0], [2, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    else:
                        kt_offset_bases: gl.constexpr = [
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                            [16, 0],
                            [32, 0],
                            [64, 0],
                            [0, 16],
                            [0, 32],
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                        ]
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [16, 0],
                            [32, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        kt_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                            warp_bases=[[0, 1], [0, 2]],
                            block_bases=[],
                            shape=[BLOCK_DMODEL, BLOCK_N],
                        )
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                            warp_bases=[[1, 0], [2, 0]],
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
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                    layout=kt_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_N, BLOCK_DV],
                    layout=v_smem_layout,
                )

                ASYNC_KPE: gl.constexpr = (
                    BLOCK_DPE > 0
                    and Q_Extend.dtype.element_ty != tl.float32
                )

                if ASYNC_KPE:
                    if BLOCK_DPE >= 64:
                        if BLOCK_N >= 128:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32], [0, 64],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 32], [0, 64]],
                                lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                                warp_bases=[[0, 1], [0, 2]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                        else:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 32]],
                                lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                                warp_bases=[[0, 1], [0, 2]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                    elif BLOCK_DPE >= 32:
                        if BLOCK_N >= 128:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32], [0, 64],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [2, 0], [0, 4], [0, 64]],
                                lane_bases=[[4, 0], [8, 0], [16, 0], [0, 8], [0, 16], [0, 32]],
                                warp_bases=[[0, 1], [0, 2]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                        else:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [2, 0], [0, 4]],
                                lane_bases=[[4, 0], [8, 0], [16, 0], [0, 8], [0, 16], [0, 32]],
                                warp_bases=[[0, 1], [0, 2]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                    else:
                        if BLOCK_N >= 128:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32], [0, 64],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [0, 4], [0, 64]],
                                lane_bases=[[2, 0], [4, 0], [8, 0], [0, 8], [0, 16], [0, 32]],
                                warp_bases=[[0, 1], [0, 2]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                        else:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [0, 4]],
                                lane_bases=[[2, 0], [4, 0], [8, 0], [0, 8], [0, 16], [0, 32]],
                                warp_bases=[[0, 1], [0, 2]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                    kpe_smem_layout: gl.constexpr = PaddedSharedLayout(
                        interval_padding_pairs=[[512, ASYNC_PAD_K]],
                        offset_bases=kpe_offset_bases,
                        cga_layout=[],
                        shape=[BLOCK_DPE, BLOCK_N],
                    )
                    kpe_smem = gl.allocate_shared_memory(
                        Q_Extend.dtype.element_ty,
                        [NUM_STAGES, BLOCK_DPE, BLOCK_N],
                        layout=kpe_smem_layout,
                    )
                else:
                    kpe_smem = kt_smem
                    kpe_async_layout: gl.constexpr = kt_async_layout

                for _s in gl.static_range(NUM_STAGES):
                    v_zero = gl.zeros(
                        [BLOCK_N, BLOCK_DV],
                        dtype=Q_Extend.dtype.element_ty,
                        layout=v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

                q_dot = gl.convert_layout(q, q_dot_layout)

                if pfx_seq_len > 0:
                    n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
                    n_extend_est = (seq_len_extend + BLOCK_N - 1) // BLOCK_N
                    use_pipe_prefix = n_prefix_blocks >= NUM_STAGES
                    if LOGIT_CAP > 0:
                        use_pipe_prefix = use_pipe_prefix and (
                            n_extend_est < NUM_STAGES
                        )
                    if use_pipe_prefix:
                        acc, l_i, m_i = attn_fwd_inner_prefix_dma_simple(
                            acc,
                            l_i,
                            m_i,
                            q_dot,
                            qpe_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            kpe_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            ACTUAL_BLOCK_DMODEL,
                            BLOCK_DPE,
                            ACTUAL_BLOCK_DPE,
                            BLOCK_DV,
                            ACTUAL_BLOCK_DV,
                            NUM_STAGES,
                            kt_async_layout,
                            kpe_async_layout,
                            v_async_layout,
                            kt_dot_layout,
                            p_dot_layout,
                            v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                        )
                    else:
                        acc, l_i, m_i = attn_fwd_inner_prefix_short(
                            acc,
                            l_i,
                            m_i,
                            q_dot,
                            qpe_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            ACTUAL_BLOCK_DMODEL,
                            BLOCK_DPE,
                            ACTUAL_BLOCK_DPE,
                            BLOCK_DV,
                            ACTUAL_BLOCK_DV,
                            kt_async_layout,
                            v_async_layout,
                            kt_dot_layout,
                            p_dot_layout,
                            v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                            V_PRELOAD=V_PRELOAD,
                        )

                cdna4_async.wait_group(0)

                if IS_CAUSAL:
                    causal_kv_end = (cur_block_m + 1) * BLOCK_M
                    effective_end = tl.minimum(seq_len_extend, causal_kv_end)
                else:
                    effective_end = seq_len_extend
                n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
                if (
                    (not ENABLE_MASK_SPLIT)
                    or USE_CUSTOM_MASK
                    or SLIDING_WINDOW_SIZE > 0
                ):
                    n_full_blocks = 0
                else:
                    partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                    if IS_CAUSAL:
                        masked_blocks = (
                            (BLOCK_M + BLOCK_N - 1) // BLOCK_N
                        ) + partial_block
                    else:
                        masked_blocks = partial_block
                    masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                    n_full_blocks = n_extend_blocks - masked_blocks

                k_extend_base = (
                    K_Extend
                    + cur_seq_q_start_idx * stride_kbs
                    + cur_kv_head * stride_kh
                )
                v_extend_base = (
                    V_Extend
                    + cur_seq_q_start_idx * stride_vbs
                    + cur_kv_head * stride_vh
                )

                if n_full_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_dma(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                elif n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        V_PRELOAD=V_PRELOAD,
                    )
                masked_start = n_full_blocks
                remaining_blocks = n_extend_blocks - masked_start
                if remaining_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_dma(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                elif remaining_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        V_PRELOAD=V_PRELOAD,
                    )

            else:
                kt_blocked_layout: gl.constexpr = gl.BlockedLayout(
                    size_per_thread=[1, 8],
                    threads_per_warp=[threads_per_warp // 4, 4],
                    warps_per_cta=[1, num_warps],
                    order=[0, 1],
                )
                kt_serial_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                    vec=8,
                    per_phase=1,
                    max_phase=16,
                    order=[0, 1],
                )
                v_serial_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                    vec=8,
                    per_phase=1,
                    max_phase=16,
                    order=[1, 0],
                )
                q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
                    vec=8,
                    per_phase=1,
                    max_phase=16,
                    order=[1, 0],
                )

                kt_serial_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_DMODEL, BLOCK_N],
                    layout=kt_serial_smem_layout,
                )
                if BLOCK_DPE > 0:
                    kt_dpe_serial_smem = gl.allocate_shared_memory(
                        Q_Extend.dtype.element_ty,
                        [BLOCK_DPE, BLOCK_N],
                        layout=kt_serial_smem_layout,
                    )
                else:
                    kt_dpe_serial_smem = kt_serial_smem
                v_serial_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_N, BLOCK_DV],
                    layout=v_serial_smem_layout,
                )
                q_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_M, BLOCK_DMODEL],
                    layout=q_smem_layout,
                )

                q_smem.store(q)
                q_dot = q_smem.load(q_dot_layout)
                qpe_dot = gl.convert_layout(qpe, q_dot_layout)

                if pfx_seq_len > 0:
                    acc, l_i, m_i = attn_fwd_inner_prefix_serial(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_serial_smem,
                        kt_dpe_serial_smem,
                        v_serial_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_blocked_layout,
                        blocked_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )

                if IS_CAUSAL:
                    causal_kv_end = (cur_block_m + 1) * BLOCK_M
                    effective_end = tl.minimum(seq_len_extend, causal_kv_end)
                else:
                    effective_end = seq_len_extend
                n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
                if (
                    (not ENABLE_MASK_SPLIT)
                    or USE_CUSTOM_MASK
                    or SLIDING_WINDOW_SIZE > 0
                ):
                    n_full_blocks = 0
                else:
                    partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                    if IS_CAUSAL:
                        masked_blocks = (
                            (BLOCK_M + BLOCK_N - 1) // BLOCK_N
                        ) + partial_block
                    else:
                        masked_blocks = partial_block
                    masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                    n_full_blocks = n_extend_blocks - masked_blocks

                k_extend_base = (
                    K_Extend
                    + cur_seq_q_start_idx * stride_kbs
                    + cur_kv_head * stride_kh
                )
                v_extend_base = (
                    V_Extend
                    + cur_seq_q_start_idx * stride_vbs
                    + cur_kv_head * stride_vh
                )

                if n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_serial(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_serial_smem,
                        kt_dpe_serial_smem,
                        v_serial_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_blocked_layout,
                        blocked_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                masked_start = n_full_blocks
                remaining_blocks = n_extend_blocks - masked_start
                if remaining_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_serial(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_serial_smem,
                        kt_dpe_serial_smem,
                        v_serial_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_blocked_layout,
                        blocked_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )

        else:
            # 8-warp path -- identical inner dispatch to basic kernel but in persistent loop
            if BLOCK_DMODEL >= 128:
                if BLOCK_DMODEL >= 512:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
                        [0, 16], [0, 32],
                        [0, 1], [0, 2], [0, 4], [0, 8],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8], [0, 16]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]],
                        warp_bases=[[0, 1], [0, 2], [0, 32]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                    if BLOCK_DV >= 512:
                        v_offset_bases: gl.constexpr = [
                            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                            [16, 0], [32, 0],
                            [1, 0], [2, 0], [4, 0], [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
                            warp_bases=[[1, 0], [2, 0], [32, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    elif BLOCK_DV >= 256:
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [0, 128],
                            [16, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                            lane_bases=[
                                [0, 8],
                                [0, 16],
                                [0, 32],
                                [0, 64],
                                [0, 128],
                                [16, 0],
                            ],
                            warp_bases=[[1, 0], [2, 0], [4, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    else:
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [16, 0],
                            [32, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                            warp_bases=[[1, 0], [2, 0], [4, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                elif BLOCK_DMODEL >= 256:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [128, 0],
                        [0, 16],
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
                        lane_bases=[
                            [8, 0],
                            [16, 0],
                            [32, 0],
                            [64, 0],
                            [128, 0],
                            [0, 16],
                        ],
                        warp_bases=[[0, 1], [0, 2], [0, 4]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                    if BLOCK_DV >= 512:
                        v_offset_bases: gl.constexpr = [
                            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256],
                            [16, 0], [32, 0],
                            [1, 0], [2, 0], [4, 0], [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0], [16, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [0, 256]],
                            warp_bases=[[1, 0], [2, 0], [32, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    elif BLOCK_DV >= 256:
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [0, 128],
                            [16, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                            lane_bases=[
                                [0, 8],
                                [0, 16],
                                [0, 32],
                                [0, 64],
                                [0, 128],
                                [16, 0],
                            ],
                            warp_bases=[[1, 0], [2, 0], [4, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    else:
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [16, 0],
                            [32, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                            warp_bases=[[1, 0], [2, 0], [4, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                else:
                    if BLOCK_N >= 128:
                        kt_offset_bases: gl.constexpr = [
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                            [16, 0],
                            [32, 0],
                            [64, 0],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                        ]
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [16, 0],
                            [32, 0],
                            [64, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        kt_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8], [0, 64]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                            warp_bases=[[0, 1], [0, 2], [0, 4]],
                            block_bases=[],
                            shape=[BLOCK_DMODEL, BLOCK_N],
                        )
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [64, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                            warp_bases=[[1, 0], [2, 0], [4, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )
                    else:
                        kt_offset_bases: gl.constexpr = [
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                            [16, 0],
                            [32, 0],
                            [64, 0],
                            [0, 16],
                            [0, 32],
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                        ]
                        v_offset_bases: gl.constexpr = [
                            [0, 1],
                            [0, 2],
                            [0, 4],
                            [0, 8],
                            [0, 16],
                            [0, 32],
                            [0, 64],
                            [16, 0],
                            [32, 0],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [8, 0],
                        ]
                        kt_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
                            lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
                            warp_bases=[[0, 1], [0, 2], [0, 4]],
                            block_bases=[],
                            shape=[BLOCK_DMODEL, BLOCK_N],
                        )
                        v_async_layout: gl.constexpr = DistributedLinearLayout(
                            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
                            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
                            warp_bases=[[1, 0], [2, 0], [4, 0]],
                            block_bases=[],
                            shape=[BLOCK_N, BLOCK_DV],
                        )

                kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_K]],
                    offset_bases=kt_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_V]],
                    offset_bases=v_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_N, BLOCK_DV],
                )

                kt_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                    layout=kt_async_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_N, BLOCK_DV],
                    layout=v_async_smem_layout,
                )

                ASYNC_KPE: gl.constexpr = (
                    BLOCK_DPE > 0
                    and Q_Extend.dtype.element_ty != tl.float32
                )

                if ASYNC_KPE:
                    if BLOCK_DPE >= 64:
                        if BLOCK_N >= 128:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32], [0, 64],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [2, 0], [4, 0], [0, 64]],
                                lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                                warp_bases=[[0, 1], [0, 2], [0, 32]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                        else:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [2, 0], [4, 0]],
                                lane_bases=[[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]],
                                warp_bases=[[0, 1], [0, 2], [0, 32]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                    elif BLOCK_DPE >= 32:
                        if BLOCK_N >= 128:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32], [0, 64],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [2, 0], [0, 64]],
                                lane_bases=[[4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [0, 16]],
                                warp_bases=[[0, 1], [0, 2], [0, 32]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                        else:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [2, 0]],
                                lane_bases=[[4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [0, 16]],
                                warp_bases=[[0, 1], [0, 2], [0, 32]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                    else:
                        if BLOCK_N >= 128:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32], [0, 64],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0], [0, 64]],
                                lane_bases=[[2, 0], [4, 0], [8, 0], [0, 4], [0, 8], [0, 16]],
                                warp_bases=[[0, 1], [0, 2], [0, 32]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                        else:
                            kpe_offset_bases: gl.constexpr = [
                                [1, 0], [2, 0], [4, 0], [8, 0],
                                [0, 4], [0, 8], [0, 16],
                                [0, 1], [0, 2], [0, 32],
                            ]
                            kpe_async_layout: gl.constexpr = DistributedLinearLayout(
                                reg_bases=[[1, 0]],
                                lane_bases=[[2, 0], [4, 0], [8, 0], [0, 4], [0, 8], [0, 16]],
                                warp_bases=[[0, 1], [0, 2], [0, 32]],
                                block_bases=[],
                                shape=[BLOCK_DPE, BLOCK_N],
                            )
                    kpe_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                        interval_padding_pairs=[[512, ASYNC_PAD_K]],
                        offset_bases=kpe_offset_bases,
                        cga_layout=[],
                        shape=[BLOCK_DPE, BLOCK_N],
                    )
                    kpe_smem = gl.allocate_shared_memory(
                        Q_Extend.dtype.element_ty,
                        [NUM_STAGES, BLOCK_DPE, BLOCK_N],
                        layout=kpe_async_smem_layout,
                    )
                else:
                    kpe_smem = kt_smem
                    kpe_async_layout: gl.constexpr = kt_async_layout

                for _s in gl.static_range(NUM_STAGES):
                    v_zero = gl.zeros(
                        [BLOCK_N, BLOCK_DV],
                        dtype=Q_Extend.dtype.element_ty,
                        layout=v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

                q_dot = gl.convert_layout(q, q_dot_layout)

                n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
                if n_prefix_blocks >= NUM_STAGES:
                    if NUM_STAGES >= 3:
                        acc, l_i, m_i = attn_fwd_inner_prefix_pipelined_scalar_mask(
                            acc,
                            l_i,
                            m_i,
                            q_dot,
                            qpe_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            kpe_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            ACTUAL_BLOCK_DMODEL,
                            BLOCK_DPE,
                            ACTUAL_BLOCK_DPE,
                            BLOCK_DV,
                            ACTUAL_BLOCK_DV,
                            NUM_STAGES,
                            kt_async_layout,
                            kpe_async_layout,
                            v_async_layout,
                            kt_dot_layout,
                            p_dot_layout,
                            v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                            ASYNC_KPE,
                        )
                    else:
                        acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                            acc,
                            l_i,
                            m_i,
                            q_dot,
                            qpe_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            kpe_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            ACTUAL_BLOCK_DMODEL,
                            BLOCK_DPE,
                            ACTUAL_BLOCK_DPE,
                            BLOCK_DV,
                            ACTUAL_BLOCK_DV,
                            NUM_STAGES,
                            kt_async_layout,
                            kpe_async_layout,
                            v_async_layout,
                            kt_dot_layout,
                            p_dot_layout,
                            v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                            ASYNC_KPE,
                        )
                elif pfx_seq_len > 0:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        V_PRELOAD=V_PRELOAD,
                    )

                if IS_CAUSAL:
                    causal_kv_end = (cur_block_m + 1) * BLOCK_M
                    effective_end = tl.minimum(seq_len_extend, causal_kv_end)
                else:
                    effective_end = seq_len_extend
                n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
                if (
                    (not ENABLE_MASK_SPLIT)
                    or USE_CUSTOM_MASK
                    or SLIDING_WINDOW_SIZE > 0
                ):
                    n_full_blocks = 0
                else:
                    partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                    if IS_CAUSAL:
                        masked_blocks = (
                            (BLOCK_M + BLOCK_N - 1) // BLOCK_N
                        ) + partial_block
                    else:
                        masked_blocks = partial_block
                    masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                    n_full_blocks = n_extend_blocks - masked_blocks

                k_extend_base = (
                    K_Extend
                    + cur_seq_q_start_idx * stride_kbs
                    + cur_kv_head * stride_kh
                )
                v_extend_base = (
                    V_Extend
                    + cur_seq_q_start_idx * stride_vbs
                    + cur_kv_head * stride_vh
                )

                if n_full_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        ASYNC_KPE,
                    )
                elif n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        V_PRELOAD=V_PRELOAD,
                    )
                masked_start = n_full_blocks
                remaining_blocks = n_extend_blocks - masked_start
                if remaining_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        ASYNC_KPE,
                    )
                elif remaining_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        V_PRELOAD=V_PRELOAD,
                    )

            else:
                # 8-warp BLOCK_DMODEL < 128
                if BLOCK_N >= 128:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                    ]
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [16, 0],
                        [32, 0],
                        [64, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 64]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
                        warp_bases=[[0, 2], [0, 4], [0, 8]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4], [64, 0]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
                        warp_bases=[[2, 0], [4, 0], [8, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )
                else:
                    kt_offset_bases: gl.constexpr = [
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                        [16, 0],
                        [32, 0],
                        [0, 16],
                        [0, 32],
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                    ]
                    v_offset_bases: gl.constexpr = [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [16, 0],
                        [32, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ]
                    kt_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[1, 0], [2, 0], [4, 0]],
                        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]],
                        warp_bases=[[0, 2], [0, 4], [0, 8]],
                        block_bases=[],
                        shape=[BLOCK_DMODEL, BLOCK_N],
                    )
                    v_async_layout: gl.constexpr = DistributedLinearLayout(
                        reg_bases=[[0, 1], [0, 2], [0, 4]],
                        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]],
                        warp_bases=[[2, 0], [4, 0], [8, 0]],
                        block_bases=[],
                        shape=[BLOCK_N, BLOCK_DV],
                    )

                kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_K]],
                    offset_bases=kt_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_DMODEL, BLOCK_N],
                )
                v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
                    interval_padding_pairs=[[512, ASYNC_PAD_V]],
                    offset_bases=v_offset_bases,
                    cga_layout=[],
                    shape=[BLOCK_N, BLOCK_DV],
                )

                kt_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                    layout=kt_async_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_N, BLOCK_DV],
                    layout=v_async_smem_layout,
                )
                ASYNC_KPE: gl.constexpr = False
                kpe_smem = kt_smem
                kpe_async_layout: gl.constexpr = kt_async_layout

                for _s in gl.static_range(NUM_STAGES):
                    v_zero = gl.zeros(
                        [BLOCK_N, BLOCK_DV],
                        dtype=Q_Extend.dtype.element_ty,
                        layout=v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

                q_dot = gl.convert_layout(q, q_dot_layout)

                n_prefix_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
                if n_prefix_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        ASYNC_KPE,
                    )
                elif pfx_seq_len > 0:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        V_PRELOAD=V_PRELOAD,
                    )

                if IS_CAUSAL:
                    causal_kv_end = (cur_block_m + 1) * BLOCK_M
                    effective_end = tl.minimum(seq_len_extend, causal_kv_end)
                else:
                    effective_end = seq_len_extend
                n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
                if (
                    (not ENABLE_MASK_SPLIT)
                    or USE_CUSTOM_MASK
                    or SLIDING_WINDOW_SIZE > 0
                ):
                    n_full_blocks = 0
                else:
                    partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                    if IS_CAUSAL:
                        masked_blocks = (
                            (BLOCK_M + BLOCK_N - 1) // BLOCK_N
                        ) + partial_block
                    else:
                        masked_blocks = partial_block
                    masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                    n_full_blocks = n_extend_blocks - masked_blocks

                k_extend_base = (
                    K_Extend
                    + cur_seq_q_start_idx * stride_kbs
                    + cur_kv_head * stride_kh
                )
                v_extend_base = (
                    V_Extend
                    + cur_seq_q_start_idx * stride_vbs
                    + cur_kv_head * stride_vh
                )

                if n_full_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        ASYNC_KPE,
                    )
                elif n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        V_PRELOAD=V_PRELOAD,
                    )
                masked_start = n_full_blocks
                remaining_blocks = n_extend_blocks - masked_start
                if remaining_blocks >= NUM_STAGES:
                    acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        ASYNC_KPE,
                    )
                elif remaining_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        V_PRELOAD=V_PRELOAD,
                    )

        if HAS_SINK:
            cur_sink = gl.load(Sinks + cur_head)
            l_i = l_i + gl.exp2(cur_sink * LOG2E - m_i)

        if SPLIT_K > 1:
            # Write normalized partial output and lse to workspace for fixup.
            # Use orig_seq_len_extend for the mask — all splits (including
            # prefix-only ones) must write their partial results.
            l_recip_sk = 1.0 / l_i
            acc_normed = acc * l_recip_sk[:, None]
            lse = m_i + tl.log2(l_i)
            split_idx = output_tile * SPLIT_K + k_split_id

            po_base = partial_out + split_idx * BLOCK_M * BLOCK_DV
            po_ptrs = po_base + offs_m[:, None] * BLOCK_DV + offs_dv[None, :]
            po_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < orig_seq_len_extend
            if ACTUAL_BLOCK_DV != BLOCK_DV:
                po_mask = po_mask & (offs_dv[None, :] < ACTUAL_BLOCK_DV)
            po_val = gl.convert_layout(acc_normed, blocked_layout)
            gl.store(po_ptrs, po_val, mask=po_mask)

            pl_base = partial_lse + split_idx * BLOCK_M
            pl_ptrs = pl_base + offs_m
            pl_mask = (cur_block_m * BLOCK_M + offs_m) < orig_seq_len_extend
            lse_val = gl.convert_layout(lse, offs_m_layout)
            gl.store(pl_ptrs, lse_val, mask=pl_mask)
        else:
            l_recip = 1.0 / l_i
            acc = acc * l_recip[:, None]
            if V_SCALE != 1.0:
                acc = acc * V_SCALE

            o_ptrs = (
                O_Extend
                + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
                * stride_obs
                + cur_head * stride_oh
                + offs_dv[None, :]
            )
            o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
            if ACTUAL_BLOCK_DV != BLOCK_DV:
                o_mask = o_mask & (offs_dv[None, :] < ACTUAL_BLOCK_DV)
            out = gl.convert_layout(acc, blocked_layout).to(O_Extend.dtype.element_ty)
            gl.store(o_ptrs, out, mask=o_mask)

        tile_idx += total_programs


# ===-----------------------------------------------------------------------===#
# Persistent-CTA Tile Scheduling
# ===-----------------------------------------------------------------------===#


def _build_tile_schedule_uniform(batch_size, head_num, n_m_tiles, device):
    """GPU-only fast path: all requests have the same number of M-tiles.

    Avoids the CPU<->GPU sync of the general path.  For decode (ext <= BLOCK_M)
    every request has exactly 1 M-tile, so n_m_tiles == 1.
    """
    total_valid_tiles = batch_size * head_num * n_m_tiles
    if total_valid_tiles == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, empty, 0

    b_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    h_idx = torch.arange(head_num, dtype=torch.int32, device=device)
    m_idx = torch.arange(n_m_tiles, dtype=torch.int32, device=device)

    # Order: for each batch, for each head, for each m-tile
    tile_b = b_idx.repeat_interleave(head_num * n_m_tiles)
    tile_h = h_idx.repeat(batch_size * n_m_tiles)
    if n_m_tiles > 1:
        tile_h = h_idx.unsqueeze(1).expand(head_num, n_m_tiles).reshape(-1).repeat(batch_size)
        tile_m = m_idx.repeat(batch_size * head_num)
    else:
        tile_m = torch.zeros(total_valid_tiles, dtype=torch.int32, device=device)
    return tile_b, tile_h, tile_m, total_valid_tiles


def _build_tile_schedule(qo_indptr, head_num, BLOCK_M, device):
    extend_lens_cpu = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    batch_size = len(extend_lens_cpu)
    n_m_tiles_per_batch = [(el + BLOCK_M - 1) // BLOCK_M for el in extend_lens_cpu]

    # Fast path: uniform tile count (includes all-decode and uniform-prefill).
    if batch_size > 0 and min(n_m_tiles_per_batch) == max(n_m_tiles_per_batch):
        return _build_tile_schedule_uniform(
            batch_size, head_num, n_m_tiles_per_batch[0], device
        )

    total_m_tiles = sum(n_m_tiles_per_batch)
    total_valid_tiles = total_m_tiles * head_num

    if total_valid_tiles == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, empty, 0

    tile_b = torch.empty(total_valid_tiles, dtype=torch.int32)
    tile_h = torch.empty(total_valid_tiles, dtype=torch.int32)
    tile_m = torch.empty(total_valid_tiles, dtype=torch.int32)

    idx = 0
    for b in range(batch_size):
        n_m = n_m_tiles_per_batch[b]
        seg = n_m * head_num
        m_range = torch.arange(n_m, dtype=torch.int32)
        h_range = torch.arange(head_num, dtype=torch.int32)
        tile_b[idx : idx + seg] = b
        tile_h[idx : idx + seg] = h_range.repeat_interleave(n_m)
        tile_m[idx : idx + seg] = m_range.repeat(head_num)
        idx += seg

    return (tile_b.to(device), tile_h.to(device), tile_m.to(device), total_valid_tiles)


# ===-----------------------------------------------------------------------===#
# Persistent Grid Selection (origami-inspired, compile-time heuristics)
# ===-----------------------------------------------------------------------===#

_cached_num_CUs = {}


def _get_num_CUs(device):
    idx = device.index if hasattr(device, 'index') and device.index is not None else 0
    if idx not in _cached_num_CUs:
        _cached_num_CUs[idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return _cached_num_CUs[idx]


def _select_persistent_grid(total_valid_tiles: int, num_CUs: int) -> int:
    """Pick CTA count for the persistent kernel.

    When there are more tiles than CUs we cap at 2*CUs for good occupancy.
    When there are fewer tiles than CUs (decode, spec-decode) we use all
    available tiles — split-K will eventually fill the remaining CUs.

    Inspired by origami/streamk grid_k_split_aware but baked into
    static if-branches so there is no runtime library dependency.
    """
    if total_valid_tiles >= 2 * num_CUs:
        return 2 * num_CUs
    if total_valid_tiles >= num_CUs:
        return num_CUs
    # Tile-starved: use all tiles.  Split-K (Phase 3) will later
    # multiply this by a k_split factor to improve CU utilization.
    return total_valid_tiles


# ===-----------------------------------------------------------------------===#
# Python Wrappers
# ===-----------------------------------------------------------------------===#


def _launch_persistent(
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
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale=1.0,
    v_scale=1.0,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    enable_mask_split=True,
    enable_prefix_unmasked=True,
    _force_block_m=None,
    _force_num_warps=None,
    _force_num_stages=None,
    _force_mma_shape=None,
    _force_waves_per_eu=None,
    _force_async_pad_k=None,
    _force_async_pad_v=None,
    _force_block_n=None,
    _ck_v_preload=False,
    min_len_extend=None,
):
    from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_entrypoints import (
        _resolve_qk_split_dims,
    )
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE = _resolve_qk_split_dims(
        Lq
    )

    USE_CUSTOM_MASK = custom_mask is not None
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask
    if not USE_CUSTOM_MASK:
        custom_mask = torch.empty(0, dtype=torch.uint8, device=q_extend.device)
        mask_indptr = torch.zeros(
            q_extend.shape[0] + 1, dtype=torch.int64, device=q_extend.device
        )
    if window_kv_offsets is None:
        window_kv_offsets = torch.zeros(
            qo_indptr.shape[0] - 1, dtype=torch.int32, device=q_extend.device
        )
    assert q_extend.shape[1] % k_extend.shape[1] == 0

    BLOCK_DV = max(triton.next_power_of_2(Lv), 16)
    BLOCK_N = 32 if max(BLOCK_DMODEL, BLOCK_DV) >= 256 else 64
    if Lq != Lv:
        BLOCK_N = 32 if max(BLOCK_DMODEL, BLOCK_DV) >= 256 else 64
    if _force_block_n is not None:
        BLOCK_N = _force_block_n
    batch_size = qo_indptr.shape[0] - 1

    if min_len_extend is None:
        extend_lens = qo_indptr[1:] - qo_indptr[:-1]
        min_len_extend = int(extend_lens.min().item())
    head_num = q_extend.shape[1]

    if _force_block_m is not None and _force_num_warps is not None:
        BLOCK_M = _force_block_m
        num_warps = _force_num_warps
    elif max(BLOCK_DMODEL, BLOCK_DV) >= 256:
        BLOCK_M = 64
        num_warps = 4
    elif max_len_extend <= 128:
        BLOCK_M = 128
        num_warps = 8
    elif batch_size <= 4:
        BLOCK_M = 128
        num_warps = 8
    elif BLOCK_DMODEL >= 128 and min_len_extend >= 64 and max_len_extend >= 256:
        BLOCK_M = 256
        num_warps = 8
    else:
        BLOCK_M = 128
        num_warps = 8

    if _force_num_stages is not None:
        NUM_STAGES = _force_num_stages
    elif max(BLOCK_DMODEL, BLOCK_DV) >= 256:
        NUM_STAGES = 1
    elif BLOCK_M == 64:
        NUM_STAGES = 1
    else:
        NUM_STAGES = 4

    if (
        BLOCK_M == 128
        and num_warps == 8
        and NUM_STAGES == 2
        and _force_num_stages is None
    ):
        NUM_STAGES = 3
    if Lq != Lv and _force_num_warps is None:
        BLOCK_M = 64
        num_warps = 4
        NUM_STAGES = 2

    if _force_mma_shape == "32x32x16":
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 32, 32, 16
        QK_K_WIDTH, PV_K_WIDTH = 32, 4
    else:
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 16, 16, 32
        QK_K_WIDTH, PV_K_WIDTH = 8, 4

    if _force_async_pad_k is not None:
        ASYNC_PAD_K = _force_async_pad_k
    else:
        ASYNC_PAD_K = 8 if BLOCK_DMODEL >= 256 else 16
    if _force_async_pad_v is not None:
        ASYNC_PAD_V = _force_async_pad_v
    else:
        ASYNC_PAD_V = 32 if BLOCK_DV >= 256 else 16

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    device = q_extend.device
    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    total_output_tiles = batch_size * head_num * n_m_tiles
    if total_output_tiles == 0:
        return

    num_CUs = _get_num_CUs(device)

    SPLIT_K = 1
    if total_output_tiles < num_CUs:
        avg_kv_len = int((kv_indptr[-1] - kv_indptr[0]).item()) // max(1, batch_size)
        if avg_kv_len >= 4 * BLOCK_N:
            SPLIT_K = _select_k_splits(total_output_tiles, num_CUs)

    if SPLIT_K > 1:
        total_splits = total_output_tiles * SPLIT_K
        partial_out, partial_lse = _ensure_splitk_workspace(
            total_splits, BLOCK_M, BLOCK_DV, device,
        )
        total_valid_tiles = total_splits
    else:
        partial_out = _ensure_splitk_dummy(device)
        partial_lse = _ensure_splitk_dummy(device)
        total_valid_tiles = total_output_tiles

    total_programs = _select_persistent_grid(total_valid_tiles, num_CUs)
    grid = (total_programs,)

    gluon_extend_attn_fwd_persistent[grid](
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
        window_kv_offsets,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        head_num,
        n_m_tiles,
        total_valid_tiles,
        total_programs,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=enable_prefix_unmasked,
        ENABLE_MASK_SPLIT=enable_mask_split,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        ACTUAL_BLOCK_DPE=ACTUAL_BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        ACTUAL_BLOCK_DV=Lv,
        NUM_STAGES=NUM_STAGES,
        MMA_INSTR_M=MMA_INSTR_M,
        MMA_INSTR_N=MMA_INSTR_N,
        MMA_INSTR_K=MMA_INSTR_K,
        QK_K_WIDTH=QK_K_WIDTH,
        PV_K_WIDTH=PV_K_WIDTH,
        ASYNC_PAD_K=ASYNC_PAD_K,
        ASYNC_PAD_V=ASYNC_PAD_V,
        Sinks=sinks,
        HAS_SINK=sinks is not None,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        V_SCALE=1.0 if SPLIT_K > 1 else v_scale,
        SPLIT_K=SPLIT_K,
        partial_out=partial_out,
        partial_lse=partial_lse,
        V_PRELOAD=False,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=_force_waves_per_eu if _force_waves_per_eu is not None else 2,
        matrix_instr_nonkdim=32,
    )

    if SPLIT_K > 1:
        reduce_grid = (total_output_tiles,)
        _splitk_reduce[reduce_grid](
            partial_out, partial_lse,
            o_extend, qo_indptr,
            head_num, n_m_tiles,
            o_extend.stride(0), o_extend.stride(1),
            SPLIT_K=SPLIT_K,
            BLOCK_M=BLOCK_M,
            BLOCK_DV=BLOCK_DV,
            ACTUAL_BLOCK_DV=Lv,
            V_SCALE=v_scale,
            num_warps=4,
        )


# ===-----------------------------------------------------------------------===#
# Split-K WCA: Fixup Kernel & Launcher
# ===-----------------------------------------------------------------------===#

_splitk_dummy = None


def _ensure_splitk_dummy(device):
    global _splitk_dummy
    if _splitk_dummy is None or _splitk_dummy.device != device:
        _splitk_dummy = torch.empty(1, dtype=torch.float32, device=device)
    return _splitk_dummy


_splitk_ws_out = None
_splitk_ws_lse = None


def _ensure_splitk_workspace(total_splits, BLOCK_M, BLOCK_DV, device):
    """Return (partial_out, partial_lse) workspace tensors, re-using cached
    allocations when possible.  Only re-allocates when the required size
    exceeds the cached capacity.
    """
    global _splitk_ws_out, _splitk_ws_lse
    need_out = (total_splits, BLOCK_M, BLOCK_DV)
    need_lse = (total_splits, BLOCK_M)
    if (
        _splitk_ws_out is not None
        and _splitk_ws_out.device == device
        and _splitk_ws_out.shape[0] >= total_splits
        and _splitk_ws_out.shape[1] >= BLOCK_M
        and _splitk_ws_out.shape[2] >= BLOCK_DV
    ):
        po = _splitk_ws_out[:total_splits, :BLOCK_M, :BLOCK_DV]
        pl = _splitk_ws_lse[:total_splits, :BLOCK_M]
    else:
        cap = max(total_splits, 2048)
        _splitk_ws_out = torch.empty(cap, BLOCK_M, BLOCK_DV, dtype=torch.float32, device=device)
        _splitk_ws_lse = torch.empty(cap, BLOCK_M, dtype=torch.float32, device=device)
        po = _splitk_ws_out[:total_splits]
        pl = _splitk_ws_lse[:total_splits]
    po.zero_()
    pl.fill_(float("-inf"))
    return po, pl


@triton.jit
def _splitk_reduce(
    partial_out_ptr,   # [total_output_tiles * SPLIT_K, BLOCK_M, BLOCK_DV] fp32
    partial_lse_ptr,   # [total_output_tiles * SPLIT_K, BLOCK_M] fp32
    O_Extend,          # output tensor
    qo_indptr,
    num_heads,
    n_m_tiles,
    stride_obs,
    stride_oh,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    ACTUAL_BLOCK_DV: tl.constexpr,
    V_SCALE: tl.constexpr,
):
    """Combine SPLIT_K partial attention results via log-sum-exp reduction."""
    tile_id = tl.program_id(0)

    tiles_per_seq = num_heads * n_m_tiles
    cur_seq = tile_id // tiles_per_seq
    rem = tile_id % tiles_per_seq
    cur_head = rem // n_m_tiles
    cur_block_m = rem % n_m_tiles
    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    seq_len_extend = (tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx).to(tl.int32)

    offs_m = tl.arange(0, BLOCK_M)
    offs_dv = tl.arange(0, BLOCK_DV)
    m_mask = (cur_block_m * BLOCK_M + offs_m) < seq_len_extend

    # Load first split's partial results.
    base_0 = tile_id * SPLIT_K
    lse_0 = tl.load(
        partial_lse_ptr + base_0 * BLOCK_M + offs_m,
        mask=m_mask, other=float("-inf"),
    )
    acc = tl.load(
        partial_out_ptr + base_0 * BLOCK_M * BLOCK_DV
        + offs_m[:, None] * BLOCK_DV + offs_dv[None, :],
        mask=m_mask[:, None], other=0.0,
    )

    # Iteratively combine remaining splits.
    for k in tl.static_range(1, SPLIT_K):
        base_k = base_0 + k
        lse_k = tl.load(
            partial_lse_ptr + base_k * BLOCK_M + offs_m,
            mask=m_mask, other=float("-inf"),
        )
        acc_k = tl.load(
            partial_out_ptr + base_k * BLOCK_M * BLOCK_DV
            + offs_m[:, None] * BLOCK_DV + offs_dv[None, :],
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
        O_Extend
        + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    o_mask = m_mask[:, None]
    if ACTUAL_BLOCK_DV != BLOCK_DV:
        o_mask = o_mask & (offs_dv[None, :] < ACTUAL_BLOCK_DV)
    tl.store(o_ptrs, acc.to(O_Extend.dtype.element_ty), mask=o_mask)


def _select_k_splits(total_output_tiles, num_CUs, min_prefix_blocks=4):
    """Choose SPLIT_K for prefix partitioning across CTAs.

    Goal: fill the GPU when there are fewer output tiles than CUs.
    Each split multiplies the grid by SPLIT_K, so we pick the smallest
    power-of-two that brings CU utilization above ~75%.

    Only splits the prefix (KV-cache) dimension -- the last split also
    handles extend tokens, so correctness is preserved for any SPLIT_K.
    """
    if total_output_tiles >= num_CUs:
        return 1
    for sk in (2, 4, 8):
        if total_output_tiles * sk >= num_CUs:
            return sk
    return 8


def _launch_splitk(
    q_extend, k_extend, v_extend, o_extend,
    k_buffer, v_buffer,
    qo_indptr, kv_indptr, kv_indices,
    custom_mask, mask_indptr, window_kv_offsets,
    sm_scale, k_scale, v_scale, logit_cap,
    Lq, Lv, is_causal, max_len_extend, min_len_extend,
    sinks, xai_temperature_len, sliding_window_size,
    BLOCK_M, BLOCK_N, num_warps, NUM_STAGES,
    _force_mma_shape=None, _force_async_pad_k=None, _force_async_pad_v=None,
    _force_waves_per_eu=None,
):
    """Split-K persistent kernel: partitions prefix across CTAs, then reduces."""
    head_num = q_extend.shape[1]
    device = q_extend.device
    batch_size = qo_indptr.shape[0] - 1

    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    total_output_tiles = batch_size * head_num * n_m_tiles
    if total_output_tiles == 0:
        return

    num_CUs = _get_num_CUs(device)
    avg_kv_len = int((kv_indptr[-1] - kv_indptr[0]).item()) // max(1, batch_size)
    need_real_splitk = (
        total_output_tiles < num_CUs and avg_kv_len >= 4 * BLOCK_N
    )

    if not need_real_splitk:
        _launch_persistent(
            q_extend, k_extend, v_extend, o_extend,
            k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices,
            custom_mask, is_causal, mask_indptr,
            max_len_extend,
            k_scale=k_scale, v_scale=v_scale, sm_scale=sm_scale,
            logit_cap=logit_cap,
            min_len_extend=min_len_extend,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            sliding_window_size=sliding_window_size,
            window_kv_offsets=window_kv_offsets,
        )
        return

    from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_entrypoints import (
        _resolve_qk_split_dims,
    )
    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE = _resolve_qk_split_dims(Lq)
    BLOCK_DV = max(triton.next_power_of_2(Lv), 64) if Lv < 256 else 256

    SPLIT_K = _select_k_splits(total_output_tiles, num_CUs)

    if max(BLOCK_DMODEL, BLOCK_DV) < 256:
        BLOCK_M = 128
        num_warps = 8
        NUM_STAGES = 4

    USE_CUSTOM_MASK = custom_mask is not None
    SKIP_PREFIX_CUSTOM_MASK = not USE_CUSTOM_MASK
    if not USE_CUSTOM_MASK:
        custom_mask = torch.empty(0, dtype=torch.uint8, device=device)
        mask_indptr = torch.zeros(
            q_extend.shape[0] + 1, dtype=torch.int64, device=device
        )
    if window_kv_offsets is None:
        window_kv_offsets = torch.zeros(
            qo_indptr.shape[0] - 1, dtype=torch.int32, device=device
        )
    enable_prefix_unmasked = True
    enable_mask_split = (not USE_CUSTOM_MASK) and (sliding_window_size <= 0) and is_causal

    if _force_mma_shape == "32x32x16":
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 32, 32, 16
        QK_K_WIDTH, PV_K_WIDTH = 32, 4
    else:
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 16, 16, 32
        QK_K_WIDTH, PV_K_WIDTH = 8, 4

    if _force_async_pad_k is not None:
        ASYNC_PAD_K = _force_async_pad_k
    else:
        ASYNC_PAD_K = 8 if BLOCK_DMODEL >= 256 else 16
    if _force_async_pad_v is not None:
        ASYNC_PAD_V = _force_async_pad_v
    else:
        ASYNC_PAD_V = 32 if BLOCK_DV >= 256 else 16

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    total_splits = total_output_tiles * SPLIT_K
    partial_out, partial_lse = _ensure_splitk_workspace(
        total_splits, BLOCK_M, BLOCK_DV, device,
    )

    total_valid_tiles = total_output_tiles * SPLIT_K
    total_programs = min(total_valid_tiles, 2 * num_CUs)
    grid = (total_programs,)

    gluon_extend_attn_fwd_persistent[grid](
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        custom_mask, mask_indptr, window_kv_offsets,
        sm_scale, kv_group_num,
        q_extend.stride(0), q_extend.stride(1),
        k_extend.stride(0), k_extend.stride(1),
        v_extend.stride(0), v_extend.stride(1),
        o_extend.stride(0), o_extend.stride(1),
        k_buffer.stride(0), k_buffer.stride(1),
        v_buffer.stride(0), v_buffer.stride(1),
        head_num, n_m_tiles,
        total_valid_tiles, total_programs,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        ENABLE_PREFIX_UNMASKED=enable_prefix_unmasked,
        ENABLE_MASK_SPLIT=enable_mask_split,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE, ACTUAL_BLOCK_DPE=ACTUAL_BLOCK_DPE,
        BLOCK_DV=BLOCK_DV, ACTUAL_BLOCK_DV=Lv,
        NUM_STAGES=NUM_STAGES,
        MMA_INSTR_M=MMA_INSTR_M, MMA_INSTR_N=MMA_INSTR_N, MMA_INSTR_K=MMA_INSTR_K,
        QK_K_WIDTH=QK_K_WIDTH, PV_K_WIDTH=PV_K_WIDTH,
        ASYNC_PAD_K=ASYNC_PAD_K, ASYNC_PAD_V=ASYNC_PAD_V,
        Sinks=sinks, HAS_SINK=sinks is not None,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        V_SCALE=1.0,
        SPLIT_K=SPLIT_K,
        partial_out=partial_out, partial_lse=partial_lse,
        V_PRELOAD=False,
        num_warps=num_warps, num_stages=1,
        waves_per_eu=_force_waves_per_eu if _force_waves_per_eu is not None else 2,
        matrix_instr_nonkdim=32,
    )

    reduce_grid = (total_output_tiles,)
    _splitk_reduce[reduce_grid](
        partial_out, partial_lse,
        o_extend, qo_indptr,
        head_num, n_m_tiles,
        o_extend.stride(0), o_extend.stride(1),
        SPLIT_K=SPLIT_K,
        BLOCK_M=BLOCK_M,
        BLOCK_DV=BLOCK_DV,
        ACTUAL_BLOCK_DV=Lv,
        V_SCALE=v_scale,
        num_warps=4,
    )


_dummy_cm = None
_dummy_mi = None
_dummy_mi_size = 0
_dummy_wkvo = None
_dummy_wkvo_size = 0


