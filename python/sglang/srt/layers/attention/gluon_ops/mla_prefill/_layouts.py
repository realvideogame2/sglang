# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout factory functions for Gluon extend-attention kernels.

Consolidates repeated layout definitions shared by the symmetric, DeepSeek,
and FP8 kernel files.  Each factory returns layout tuples suitable for
unpacking via indexed ``gl.constexpr`` assignments inside ``@gluon.jit``
kernels.

All factories must be decorated with ``@gluon.constexpr_function`` so the
Gluon JIT can evaluate them at compile time.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language._layouts import (
    DistributedLinearLayout,
    DotOperandLayout,
    PaddedSharedLayout,
)

# ===-----------------------------------------------------------------------===#
# Phase 1 -- Header layouts (MFMA + dot operands + blocked + slices)
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_mfma_dot_layouts(num_warps, mma_m, mma_n, mma_k, qk_kw, pv_kw):
    """MFMA accumulator layout and QK / PV dot-operand layouts.

    Returns (mma_layout, q_dot, kt_dot, p_dot, v_dot).
    """
    mma = AMDMFMALayout(
        version=4,
        instr_shape=[mma_m, mma_n, mma_k],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    q_dot = DotOperandLayout(operand_index=0, parent=mma, k_width=qk_kw)
    kt_dot = DotOperandLayout(operand_index=1, parent=mma, k_width=qk_kw)
    p_dot = DotOperandLayout(operand_index=0, parent=mma, k_width=pv_kw)
    v_dot = DotOperandLayout(operand_index=1, parent=mma, k_width=pv_kw)
    return mma, q_dot, kt_dot, p_dot, v_dot


@gluon.constexpr_function
def make_fp8_dot_layouts(mma_layout, fp8_qk_kw, fp8_pv_kw):
    """Extra dot-operand layouts for the FP8 prefix MFMA path.

    Returns (fp8_q_dot, fp8_kt_dot, fp8_p_dot, fp8_v_dot).
    """
    fp8_q = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=fp8_qk_kw)
    fp8_kt = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=fp8_qk_kw)
    fp8_p = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=fp8_pv_kw)
    fp8_v = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=fp8_pv_kw)
    return fp8_q, fp8_kt, fp8_p, fp8_v


@gluon.constexpr_function
def make_blocked_and_slice_layouts(num_warps, mma_layout):
    """Output blocked layout and 1-D slice helpers.

    Returns (blocked, offs_m_layout, offs_d_layout,
             mma_offs_n_col, mma_offs_m_row, mma_m_layout).
    """
    blocked = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    offs_m = gl.SliceLayout(dim=1, parent=blocked)
    offs_d = gl.SliceLayout(dim=0, parent=blocked)
    mma_n_col = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_m_row = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_ly = gl.SliceLayout(dim=1, parent=mma_layout)
    return blocked, offs_m, offs_d, mma_n_col, mma_m_row, mma_m_ly


# ===-----------------------------------------------------------------------===#
# Phase 2 -- Serial shared-memory layouts (SwizzledShared + serial BlockedLayout)
# ===-----------------------------------------------------------------------===#

SERIAL_KT_SMEM = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[0, 1])
SERIAL_V_SMEM = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[1, 0])
SERIAL_Q_SMEM = SERIAL_V_SMEM


# ===-----------------------------------------------------------------------===#
# Phase 3 -- PaddedSharedLayout factory
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_padded_smem(shape, offset_bases, padding_pairs):
    """Padded shared-memory layout for async DMA tiles.

    ``padding_pairs`` is e.g. ``[[512, pad]]`` (BF16) or
    ``[[1024, pad], [2048, 32]]`` (FP8).
    """
    return PaddedSharedLayout(
        interval_padding_pairs=padding_pairs,
        offset_bases=offset_bases,
        cga_layout=[],
        shape=shape,
    )


# ===-----------------------------------------------------------------------===#
# Phase 4 -- DistributedLinearLayout (DLL) wrapper
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_dll(shape, reg_bases, lane_bases, warp_bases):
    """Async DMA layout (DistributedLinearLayout) for K^T / V / KPE tiles."""
    return DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=lane_bases,
        warp_bases=warp_bases,
        block_bases=[],
        shape=shape,
    )


@gluon.constexpr_function
def make_serial_kt_blocked(num_warps):
    """Blocked layout for serial K^T tile loads (warps spread along N)."""
    return gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )


# ===-----------------------------------------------------------------------===#
# Phase 5 -- Offset-bases factory for PaddedSharedLayout
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_offset_bases(major_max, minor_coarse, minor_fine, major_dim):
    """Compute offset_bases for PaddedSharedLayout async DMA tiles.

    Generates a power-of-2 ladder ``[1 .. major_max]`` along ``major_dim``
    followed by two groups of minor-dimension basis vectors.

    Args:
        major_max: largest power-of-2 in the fast-varying dimension
                   (e.g. ``BLOCK_DMODEL // 2`` for K^T, ``BLOCK_DV // 2`` for V).
        minor_coarse: coarse basis values placed in the high address bits.
        minor_fine: fine basis values placed in the low address bits.
        major_dim: ``0`` for K^T / KPE (row-major), ``1`` for V (col-major).
    """
    bases = []
    d = 1
    while d <= major_max:
        bases.append([d, 0] if major_dim == 0 else [0, d])
        d *= 2
    for v in minor_coarse:
        bases.append([0, v] if major_dim == 0 else [v, 0])
    for v in minor_fine:
        bases.append([0, v] if major_dim == 0 else [v, 0])
    return bases


# ===-----------------------------------------------------------------------===#
# Phase 6 -- High-level async DMA layout factories
#
# These replace the duplicated if/elif trees in every kernel file.
# Call with (num_warps, BLOCK_DIM, BLOCK_N) and get back the finished
# (offset_bases, async_layout) pair ready for PaddedSharedLayout and DMA.
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N):
    """Offset bases for K^T PaddedSharedLayout."""
    D_half = BLOCK_DMODEL // 2
    if BLOCK_DMODEL >= 512:
        mc = [16, 32] if BLOCK_N >= 64 else [16]
    elif BLOCK_DMODEL >= 256:
        mc = [16]
    else:
        mc = [16, 32, 64] if BLOCK_N >= 128 else [16, 32]
    ob = []
    d = 1
    while d <= D_half:
        ob.append([d, 0])
        d *= 2
    for v in mc:
        ob.append([0, v])
    for v in [1, 2, 4, 8]:
        ob.append([0, v])
    return ob


@gluon.constexpr_function
def make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N):
    """Async DMA DistributedLinearLayout for a [BLOCK_DMODEL, BLOCK_N] K^T tile."""
    is_4w = num_warps < 8

    if is_4w:
        if BLOCK_DMODEL >= 512:
            reg = [[1,0],[2,0],[4,0],[0,4],[0,8],[0,16],[0,32]] if BLOCK_N >= 64 \
             else [[1,0],[2,0],[4,0],[0,4],[0,8],[0,16]]
            lane = [[8,0],[16,0],[32,0],[64,0],[128,0],[256,0]]
        elif BLOCK_DMODEL >= 256:
            reg  = [[1,0],[2,0],[4,0],[0,4],[0,8]]
            lane = [[8,0],[16,0],[32,0],[64,0],[128,0],[0,16]]
        else:
            reg = [[1,0],[2,0],[4,0],[0,4],[0,8],[0,64]] if BLOCK_N >= 128 \
             else [[1,0],[2,0],[4,0],[0,4],[0,8]]
            lane = [[8,0],[16,0],[32,0],[64,0],[0,16],[0,32]]
        warp = [[0,1],[0,2]]
    else:
        if BLOCK_DMODEL >= 512:
            lane = [[8,0],[16,0],[32,0],[64,0],[128,0],[256,0]]
            if BLOCK_N >= 64:
                reg  = [[1,0],[2,0],[4,0],[0,4],[0,8],[0,16]]
                warp = [[0,1],[0,2],[0,32]]
            else:
                reg  = [[1,0],[2,0],[4,0],[0,8],[0,16]]
                warp = [[0,1],[0,2],[0,4]]
        elif BLOCK_DMODEL >= 256:
            reg  = [[1,0],[2,0],[4,0],[0,8]]
            lane = [[8,0],[16,0],[32,0],[64,0],[128,0],[0,16]]
            warp = [[0,1],[0,2],[0,4]]
        elif BLOCK_DMODEL >= 128:
            reg = [[1,0],[2,0],[4,0],[0,8],[0,64]] if BLOCK_N >= 128 \
             else [[1,0],[2,0],[4,0],[0,8]]
            lane = [[8,0],[16,0],[32,0],[64,0],[0,16],[0,32]]
            warp = [[0,1],[0,2],[0,4]]
        else:
            reg = [[1,0],[2,0],[4,0],[0,64]] if BLOCK_N >= 128 \
             else [[1,0],[2,0],[4,0]]
            lane = [[8,0],[16,0],[32,0],[0,16],[0,32],[0,1]]
            warp = [[0,2],[0,4],[0,8]]

    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp,
        block_bases=[], shape=[BLOCK_DMODEL, BLOCK_N],
    )


@gluon.constexpr_function
def make_v_offset_bases(BLOCK_DV, BLOCK_N):
    """Offset bases for V PaddedSharedLayout."""
    Dv_half = BLOCK_DV // 2
    if BLOCK_DV >= 512:
        mc = [16, 32] if BLOCK_N >= 64 else [16]
    elif BLOCK_DV >= 256:
        mc = [16]
    else:
        mc = [16, 32, 64] if BLOCK_N >= 128 else [16, 32]
    ob = []
    d = 1
    while d <= Dv_half:
        ob.append([0, d])
        d *= 2
    for v in mc:
        ob.append([v, 0])
    for v in [1, 2, 4, 8]:
        ob.append([v, 0])
    return ob


@gluon.constexpr_function
def make_v_dll(num_warps, BLOCK_DV, BLOCK_N):
    """Async DMA DistributedLinearLayout for a [BLOCK_N, BLOCK_DV] V tile."""
    is_4w = num_warps < 8

    if is_4w:
        if BLOCK_DV >= 512:
            reg = [[0,1],[0,2],[0,4],[4,0],[8,0],[16,0],[32,0]] if BLOCK_N >= 64 \
             else [[0,1],[0,2],[0,4],[4,0],[8,0],[16,0]]
            lane = [[0,8],[0,16],[0,32],[0,64],[0,128],[0,256]]
        elif BLOCK_DV >= 256:
            reg  = [[0,1],[0,2],[0,4],[4,0],[8,0]]
            lane = [[0,8],[0,16],[0,32],[0,64],[0,128],[16,0]]
        else:
            reg = [[0,1],[0,2],[0,4],[4,0],[8,0],[64,0]] if BLOCK_N >= 128 \
             else [[0,1],[0,2],[0,4],[4,0],[8,0]]
            lane = [[0,8],[0,16],[0,32],[0,64],[16,0],[32,0]]
        warp = [[1,0],[2,0]]
    else:
        if BLOCK_DV >= 512:
            lane = [[0,8],[0,16],[0,32],[0,64],[0,128],[0,256]]
            if BLOCK_N >= 64:
                reg  = [[0,1],[0,2],[0,4],[4,0],[8,0],[16,0]]
                warp = [[1,0],[2,0],[32,0]]
            else:
                reg  = [[0,1],[0,2],[0,4],[8,0],[16,0]]
                warp = [[1,0],[2,0],[4,0]]
        elif BLOCK_DV >= 256:
            reg  = [[0,1],[0,2],[0,4],[8,0]]
            lane = [[0,8],[0,16],[0,32],[0,64],[0,128],[16,0]]
            warp = [[1,0],[2,0],[4,0]]
        elif BLOCK_DV >= 128:
            reg = [[0,1],[0,2],[0,4],[8,0],[64,0]] if BLOCK_N >= 128 \
             else [[0,1],[0,2],[0,4],[8,0]]
            lane = [[0,8],[0,16],[0,32],[0,64],[16,0],[32,0]]
            warp = [[1,0],[2,0],[4,0]]
        else:
            reg = [[0,1],[0,2],[0,4],[64,0]] if BLOCK_N >= 128 \
             else [[0,1],[0,2],[0,4]]
            lane = [[0,8],[0,16],[0,32],[16,0],[32,0],[1,0]]
            warp = [[2,0],[4,0],[8,0]]

    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp,
        block_bases=[], shape=[BLOCK_N, BLOCK_DV],
    )


# ===-----------------------------------------------------------------------===#
# Phase 7 -- FP8 kernel BF16-extend V layout factories
#
# The FP8 symmetric kernel's BF16 extend phase uses non-standard V layouts
# in three corners (4w Dv>=256 N>=128, 4w Dv<256 N<128, 8w Dv<256 N<128).
# The bf16 kt layouts match the standard factories above.
# ===-----------------------------------------------------------------------===#


@gluon.constexpr_function
def make_fp8_bf16_v_offset_bases(num_warps, BLOCK_DV, BLOCK_N):
    """Offset bases for FP8 kernel's BF16-extend V PaddedSharedLayout.

    Differs from standard only for 4-warp Dv>=256 N>=128 (uses [16,32,64]
    coarse instead of [16]).
    """
    Dv_half = BLOCK_DV // 2
    is_4w = num_warps < 8
    if is_4w and BLOCK_DV >= 256 and BLOCK_N >= 128:
        mc = [16, 32, 64]
    elif BLOCK_DV >= 256:
        mc = [16]
    else:
        mc = [16, 32, 64] if BLOCK_N >= 128 else [16, 32]
    ob = []
    d = 1
    while d <= Dv_half:
        ob.append([0, d])
        d *= 2
    for v in mc:
        ob.append([v, 0])
    for v in [1, 2, 4, 8]:
        ob.append([v, 0])
    return ob


@gluon.constexpr_function
def make_fp8_bf16_v_dll(num_warps, BLOCK_DV, BLOCK_N):
    """Async DMA DLL for FP8 kernel's BF16-extend V tile [BLOCK_N, BLOCK_DV].

    Three non-standard variants vs the standard ``make_v_dll``:
    - 4w Dv>=256 N>=128: extra [32,0],[64,0] in reg
    - 4w Dv<256  N<128:  shuffled lane/warp for FP8 shared-memory pressure
    - 8w Dv<256  N<128:  same shuffled pattern, 3-warp stride
    """
    is_4w = num_warps < 8

    if is_4w:
        if BLOCK_DV >= 256:
            if BLOCK_N >= 128:
                reg = [[0,1],[0,2],[0,4],[4,0],[8,0],[32,0],[64,0]]
            else:
                reg = [[0,1],[0,2],[0,4],[4,0],[8,0]]
            lane = [[0,8],[0,16],[0,32],[0,64],[0,128],[16,0]]
            warp = [[1,0],[2,0]]
        else:
            if BLOCK_N >= 128:
                reg  = [[0,1],[0,2],[0,4],[4,0],[8,0],[64,0]]
                lane = [[0,8],[0,16],[0,32],[0,64],[16,0],[32,0]]
                warp = [[1,0],[2,0]]
            else:
                reg  = [[0,1],[0,2],[0,4],[0,8],[8,0]]
                lane = [[0,16],[0,32],[0,64],[1,0],[2,0],[4,0]]
                warp = [[16,0],[32,0]]
    else:
        if BLOCK_DV >= 256:
            reg  = [[0,1],[0,2],[0,4],[8,0]]
            lane = [[0,8],[0,16],[0,32],[0,64],[0,128],[16,0]]
            warp = [[1,0],[2,0],[4,0]]
        elif BLOCK_DV >= 128:
            if BLOCK_N >= 128:
                reg  = [[0,1],[0,2],[0,4],[8,0],[64,0]]
                lane = [[0,8],[0,16],[0,32],[0,64],[16,0],[32,0]]
                warp = [[1,0],[2,0],[4,0]]
            else:
                reg  = [[0,1],[0,2],[0,4],[0,8]]
                lane = [[0,16],[0,32],[0,64],[1,0],[2,0],[4,0]]
                warp = [[8,0],[16,0],[32,0]]
        else:
            reg = [[0,1],[0,2],[0,4],[64,0]] if BLOCK_N >= 128 \
             else [[0,1],[0,2],[0,4]]
            lane = [[0,8],[0,16],[0,32],[16,0],[32,0],[1,0]]
            warp = [[2,0],[4,0],[8,0]]

    return DistributedLinearLayout(
        reg_bases=reg, lane_bases=lane, warp_bases=warp,
        block_bases=[], shape=[BLOCK_N, BLOCK_DV],
    )
