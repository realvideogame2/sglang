# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention kernel for symmetric head dims (Lq == Lv).

Handles D=64, D=128, D=256 where BLOCK_DPE is always 0.
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
    V_PRELOAD: gl.constexpr = False,  #
):
    num_warps: gl.constexpr = gl.num_warps()

    cur_seq = gl.program_id(0)
    cur_head = gl.program_id(1)
    cur_block_m = gl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_q_start_idx = gl.load(qo_indptr + cur_seq)
    seq_len_extend = gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx
    cur_seq_kv_start_idx = gl.load(kv_indptr + cur_seq)
    seq_len_prefix = gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx

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
        if NUM_STAGES >= 2 and BLOCK_DMODEL >= 128:
            # 4-warp DMA path
            if BLOCK_DMODEL >= 256:
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

            if BLOCK_DV >= 256:
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
            if BLOCK_DMODEL >= 256:
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
                if BLOCK_DV >= 256:
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


