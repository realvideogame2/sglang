# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon FP8 MLA prefill dispatch for D192 on gfx950 (MI350X / CDNA4).

DeepSeek V3/R1 MHA-path dimensions (post kv_b_proj expansion):
    QK = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192
    V  = v_head_dim = 128

Public entry points:
    mla_prefill_d192_fwd()     — non-persistent (3D grid, one CTA per tile)
    mla_prefill_d192_ps_fwd()  — AITER-style metadata-driven persistent scheduling
    prewarm_mla_d192()         — JIT-compile all kernel variants (call at model load)
"""

import math
import os
import sys

import torch
import triton
import triton.language as tl

try:
    from ._fp8_mla_d192_8w_gfx950 import fp8_mla_d192_8w_fwd as _fp8_8w
    from ._fp8_mla_d192_8w_splitk_gfx950 import (
        fp8_mla_d192_8w_splitk_fwd as _fp8_splitk,
        _splitk_reduce as _fp8_splitk_reduce,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _fp8_mla_d192_8w_gfx950 import fp8_mla_d192_8w_fwd as _fp8_8w
    from _fp8_mla_d192_8w_splitk_gfx950 import (
        fp8_mla_d192_8w_splitk_fwd as _fp8_splitk,
        _splitk_reduce as _fp8_splitk_reduce,
    )

D_NOPE = 128
D_ROPE = 64
D_FULL = D_NOPE + D_ROPE  # 192
D_V = 128
BLOCK_M = 128
BLOCK_N = 128
NUM_WARPS = 8
NUM_STAGES = 2
SCHED_HINT = "none"

_dummy_i32 = None
_dummy_f32 = None


def _get_dummy(device):
    global _dummy_i32, _dummy_f32
    if _dummy_i32 is None or _dummy_i32.device != device:
        _dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        _dummy_f32 = torch.zeros(1, dtype=torch.float32, device=device)
    return _dummy_i32, _dummy_f32


def _launch_non_persistent(q, kv, v_sep, o, qo_indptr, kv_indptr, sm_scale,
                            num_heads, n_m_tiles, batch_size,
                            q_scale, kv_scale, return_compiled=False):
    dummy_i32, dummy_f32 = _get_dummy(q.device)
    stride_v_tok = v_sep.stride(0) if v_sep is not kv else kv.stride(0)
    stride_kv_h = kv.stride(1) if kv.ndim >= 3 else 0
    stride_v_h = v_sep.stride(1) if v_sep.ndim >= 3 else 0
    grid = (batch_size, num_heads, n_m_tiles)
    # .run(...) is the `return the CompiledKernel` variant of `[grid](...)`
    # and is what the SGLang-side fast-path cache uses to capture the
    # compiled kernel for direct HIPLauncher invocation on subsequent
    # calls (saving ~40us of JITFunction specialization per call).
    kwargs = dict(
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        D_NOPE=D_NOPE, D_ROPE=D_ROPE, D_V=D_V,
        NUM_WARPS_CONSTEXPR=NUM_WARPS, NUM_STAGES=NUM_STAGES,
        Q_SCALE=q_scale, KV_SCALE=kv_scale,
        IS_PERSISTENT=False, IS_PS_PERSISTENT=False,
        num_warps=NUM_WARPS, num_stages=1, schedule_hint=SCHED_HINT,
    )
    if return_compiled:
        kwargs["grid"] = grid
        kwargs["warmup"] = False
        launch = _fp8_8w.run
    else:
        launch = _fp8_8w[grid]
    return launch(
        q, kv, v_sep, o,
        qo_indptr, kv_indptr,
        sm_scale,
        q.stride(0), q.stride(1),
        kv.stride(0), stride_kv_h, stride_v_tok, stride_v_h,
        o.stride(0), o.stride(1),
        num_heads, n_m_tiles, 1, 1,
        dummy_i32, dummy_i32, dummy_f32, dummy_f32,
        0, 0, 0, 0,
        **kwargs,
    )


def mla_prefill_d192_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float = None,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    v: torch.Tensor = None,
):
    """FP8 MLA prefill attention (D192) — non-persistent dispatch.

    Args:
        kv: Key tensor [tokens, (heads,) D_FULL] — K_nope(128) + K_rope(64).
        v:  Optional separate value tensor [tokens, (heads,) D_V].
            If None, V is read from kv[:, :D_V] (original single-tensor mode).
    """
    assert q.dtype == torch.float8_e4m3fn, f"Q must be float8_e4m3fn, got {q.dtype}"
    assert kv.dtype == torch.float8_e4m3fn, f"KV must be float8_e4m3fn, got {kv.dtype}"
    assert o.dtype == torch.bfloat16, f"O must be bfloat16, got {o.dtype}"
    assert q.shape[-1] == D_FULL and kv.shape[-1] == D_FULL and o.shape[-1] == D_V
    assert q.ndim >= 2 and kv.ndim >= 2 and o.ndim >= 2
    assert qo_indptr.dtype == torch.int32 and kv_indptr.dtype == torch.int32

    v_sep = kv if v is None else v
    if v is not None:
        assert v.dtype == torch.float8_e4m3fn, f"V must be float8_e4m3fn, got {v.dtype}"
        assert v.shape[-1] == D_V

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D_FULL)

    batch_size = qo_indptr.shape[0] - 1
    num_heads = q.shape[1]
    max_seq_q = int((qo_indptr[1:] - qo_indptr[:-1]).max().item()) if batch_size > 0 else 0
    n_m_tiles = (max_seq_q + BLOCK_M - 1) // BLOCK_M

    _launch_non_persistent(q, kv, v_sep, o, qo_indptr, kv_indptr, sm_scale,
                           num_heads, n_m_tiles, batch_size, q_scale, kv_scale)
    return o


@triton.jit
def _gen_work_info_kernel(
    qo_indptr_ptr, kv_indptr_ptr,
    prefix_ptr,
    work_info_ptr,
    work_indptr_ptr,
    batch_size,
    total_tiles,
    NUM_HEADS: tl.constexpr,
    NUM_CUS: tl.constexpr,
    BLOCK_M_C: tl.constexpr,
):
    cta_id = tl.program_id(0)

    tpc = total_tiles // NUM_CUS
    rem = total_tiles % NUM_CUS
    extra = tl.where(cta_id < rem, 1, 0)
    my_start = cta_id * tpc + tl.minimum(cta_id, rem)
    my_count = tpc + extra

    tl.store(work_indptr_ptr + cta_id, my_start)
    if cta_id == 0:
        tl.store(work_indptr_ptr + NUM_CUS, total_tiles)

    cur_b: tl.int32 = 0
    cur_end: tl.int32 = tl.load(prefix_ptr + 1)

    for i in tl.range(my_count):
        tid = my_start + i

        while tid >= cur_end:
            cur_b += 1
            cur_end = tl.load(prefix_ptr + cur_b + 1)

        local_id = tid - tl.load(prefix_ptr + cur_b)
        h = local_id % NUM_HEADS
        m = local_id // NUM_HEADS

        qo_s_b = tl.load(qo_indptr_ptr + cur_b)
        qo_e_b = tl.load(qo_indptr_ptr + cur_b + 1)
        kv_s = tl.load(kv_indptr_ptr + cur_b)
        kv_e = tl.load(kv_indptr_ptr + cur_b + 1)

        qo_s = qo_s_b + m * BLOCK_M_C
        qo_e = tl.minimum(qo_s_b + (m + 1) * BLOCK_M_C, qo_e_b)

        base = tid * 8
        tl.store(work_info_ptr + base + 0, cur_b)
        tl.store(work_info_ptr + base + 1, tl.full([], -1, tl.int32))
        tl.store(work_info_ptr + base + 2, qo_s)
        tl.store(work_info_ptr + base + 3, qo_e)
        tl.store(work_info_ptr + base + 4, kv_s)
        tl.store(work_info_ptr + base + 5, kv_e)
        tl.store(work_info_ptr + base + 6, tl.full([], 0, tl.int32))
        tl.store(work_info_ptr + base + 7, h | ((h + 1) << 16))


def _gen_metadata_gpu(qo_indptr, kv_indptr, num_heads, num_CUs):
    """Generate PS work metadata on GPU via a lightweight Triton kernel.

    No KV splitting: every tile gets partial_o_loc = -1.
    Cost: 3 tiny PyTorch ops on [B] tensors + 1 .item() sync + 1 Triton launch.
    """
    device = qo_indptr.device
    B = qo_indptr.shape[0] - 1

    seqlen_q = qo_indptr[1:] - qo_indptr[:-1]
    n_m = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    tpb = n_m * num_heads

    prefix = torch.zeros(B + 1, dtype=torch.int32, device=device)
    torch.cumsum(tpb, dim=0, out=prefix[1:])
    total_tiles = int(prefix[-1].item())

    if total_tiles == 0:
        return (torch.zeros(num_CUs + 1, dtype=torch.int32, device=device),
                torch.empty((0, 8), dtype=torch.int32, device=device))

    work_info_flat = torch.empty(total_tiles * 8, dtype=torch.int32, device=device)
    work_indptr = torch.empty(num_CUs + 1, dtype=torch.int32, device=device)

    _gen_work_info_kernel[(num_CUs,)](
        qo_indptr, kv_indptr,
        prefix, work_info_flat, work_indptr,
        B, total_tiles,
        NUM_HEADS=num_heads, NUM_CUS=num_CUs, BLOCK_M_C=BLOCK_M,
        num_warps=1, num_stages=1,
    )

    work_info = work_info_flat.view(total_tiles, 8)
    return work_indptr, work_info


def _launch_ps(q, kv, v_sep, o, qo_indptr, kv_indptr, sm_scale,
               num_heads, num_CUs, work_indptr, work_info,
               q_scale, kv_scale, return_compiled=False):
    """Launch the PS persistent kernel with pre-built metadata (no reduce).

    When `return_compiled=True`, goes through `JITFunction.run` (which
    returns the CompiledKernel alongside launching) so the SGLang-side
    dispatch can cache a direct HIPLauncher fast runner.
    """
    dummy_f32 = torch.empty(1, dtype=torch.float32, device=q.device)
    stride_v_tok = v_sep.stride(0) if v_sep is not kv else kv.stride(0)
    stride_kv_h = kv.stride(1) if kv.ndim >= 3 else 0
    stride_v_h = v_sep.stride(1) if v_sep.ndim >= 3 else 0
    grid = (num_CUs,)
    kwargs = dict(
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        D_NOPE=D_NOPE, D_ROPE=D_ROPE, D_V=D_V,
        NUM_WARPS_CONSTEXPR=NUM_WARPS, NUM_STAGES=NUM_STAGES,
        Q_SCALE=q_scale, KV_SCALE=kv_scale,
        IS_PERSISTENT=False, IS_PS_PERSISTENT=True,
        num_warps=NUM_WARPS, num_stages=1, schedule_hint=SCHED_HINT,
    )
    if return_compiled:
        kwargs["grid"] = grid
        kwargs["warmup"] = False
        launch = _fp8_8w.run
    else:
        launch = _fp8_8w[grid]
    return launch(
        q, kv, v_sep, o,
        qo_indptr, kv_indptr,
        sm_scale,
        q.stride(0), q.stride(1),
        kv.stride(0), stride_kv_h, stride_v_tok, stride_v_h,
        o.stride(0), o.stride(1),
        num_heads, 1, 1, 1,
        work_indptr, work_info,
        dummy_f32, dummy_f32,
        0, 0, 0, 0,
        **kwargs,
    )


def mla_prefill_d192_ps_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float = None,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    v: torch.Tensor = None,
):
    """FP8 MLA prefill attention (D192) — persistent scheduling.

    Args:
        kv: Key tensor [tokens, (heads,) D_FULL] — K_nope(128) + K_rope(64).
        v:  Optional separate value tensor [tokens, (heads,) D_V].
            If None, V is read from kv[:, :D_V] (original single-tensor mode).
    """
    assert q.dtype == torch.float8_e4m3fn, f"Q must be float8_e4m3fn, got {q.dtype}"
    assert kv.dtype == torch.float8_e4m3fn, f"KV must be float8_e4m3fn, got {kv.dtype}"
    assert o.dtype == torch.bfloat16, f"O must be bfloat16, got {o.dtype}"
    assert q.shape[-1] == D_FULL and kv.shape[-1] == D_FULL and o.shape[-1] == D_V
    assert q.ndim >= 2 and kv.ndim >= 2 and o.ndim >= 2
    assert qo_indptr.dtype == torch.int32 and kv_indptr.dtype == torch.int32

    v_sep = kv if v is None else v
    if v is not None:
        assert v.dtype == torch.float8_e4m3fn, f"V must be float8_e4m3fn, got {v.dtype}"
        assert v.shape[-1] == D_V

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D_FULL)

    batch_size = qo_indptr.shape[0] - 1
    num_heads = q.shape[1]
    device = q.device
    num_CUs = torch.cuda.get_device_properties(device).multi_processor_count

    if batch_size == 0:
        return o

    work_indptr, work_info = _gen_metadata_gpu(
        qo_indptr, kv_indptr, num_heads, num_CUs)
    _launch_ps(q, kv, v_sep, o, qo_indptr, kv_indptr, sm_scale,
               num_heads, num_CUs, work_indptr, work_info,
               q_scale, kv_scale)
    return o


def _launch_wca(q, kv, v_sep, o, qo_indptr, kv_indptr, sm_scale,
                num_heads, num_CUs, total_tiles, batch_size,
                q_scale, kv_scale, return_compiled=False):
    """Launch WCA persistent kernel — inline serial scan, no metadata."""
    dummy_f32 = torch.empty(1, dtype=torch.float32, device=q.device)
    dummy_i32 = torch.zeros(1, dtype=torch.int32, device=q.device)
    stride_v_tok = v_sep.stride(0) if v_sep is not kv else kv.stride(0)
    stride_kv_h = kv.stride(1) if kv.ndim >= 3 else 0
    stride_v_h = v_sep.stride(1) if v_sep.ndim >= 3 else 0
    total_programs = min(total_tiles, num_CUs)
    grid = (total_programs,)
    kwargs = dict(
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        D_NOPE=D_NOPE, D_ROPE=D_ROPE, D_V=D_V,
        NUM_WARPS_CONSTEXPR=NUM_WARPS, NUM_STAGES=NUM_STAGES,
        Q_SCALE=q_scale, KV_SCALE=kv_scale,
        IS_PERSISTENT=False, IS_PS_PERSISTENT=False,
        IS_WCA=True, batch_size=batch_size,
        num_warps=NUM_WARPS, num_stages=1, schedule_hint=SCHED_HINT,
    )
    if return_compiled:
        kwargs["grid"] = grid
        kwargs["warmup"] = False
        launch = _fp8_8w.run
    else:
        launch = _fp8_8w[grid]
    return launch(
        q, kv, v_sep, o,
        qo_indptr, kv_indptr,
        sm_scale,
        q.stride(0), q.stride(1),
        kv.stride(0), stride_kv_h, stride_v_tok, stride_v_h,
        o.stride(0), o.stride(1),
        num_heads, 1, total_tiles, total_programs,
        dummy_i32, dummy_i32, dummy_f32, dummy_f32,
        0, 0, 0, 0,
        **kwargs,
    )


def mla_prefill_d192_wca_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float = None,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    v: torch.Tensor = None,
):
    """FP8 MLA prefill attention (D192) — WCA persistent (no metadata)."""
    assert q.dtype == torch.float8_e4m3fn
    assert kv.dtype == torch.float8_e4m3fn
    assert o.dtype == torch.bfloat16
    assert q.shape[-1] == D_FULL and kv.shape[-1] == D_FULL and o.shape[-1] == D_V
    assert qo_indptr.dtype == torch.int32 and kv_indptr.dtype == torch.int32

    v_sep = kv if v is None else v
    if v is not None:
        assert v.dtype == torch.float8_e4m3fn and v.shape[-1] == D_V

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D_FULL)

    batch_size = qo_indptr.shape[0] - 1
    num_heads = q.shape[1]
    device = q.device
    num_CUs = torch.cuda.get_device_properties(device).multi_processor_count

    if batch_size == 0:
        return o

    seqlen_q = qo_indptr[1:] - qo_indptr[:-1]
    n_m = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    total_tiles = int((n_m * num_heads).sum().item())

    _launch_wca(q, kv, v_sep, o, qo_indptr, kv_indptr, sm_scale,
                num_heads, num_CUs, total_tiles, batch_size,
                q_scale, kv_scale)
    return o


def _launch_splitk(q, kv, v_sep, o, qo_indptr, kv_indptr, sm_scale,
                   num_heads, n_m_tiles, num_CUs, batch_size, split_k,
                   q_scale, kv_scale, return_compiled=False,
                   partial_o=None, partial_lse=None, sync_count=None):
    """Launch the split-K persistent kernel (no reduce; caller handles merge).

    When `return_compiled=True`, goes through `JITFunction.run` so the
    SGLang wrapper can cache a direct HIPLauncher fast runner that
    bypasses the Python-level Triton binder.

    Workspace tensors (`partial_o`, `partial_lse`, `sync_count`) must be
    supplied by the caller when `split_k > 1`. The launcher leaves them
    alloc'd up-front so the fast path never allocates during dispatch.
    """
    stride_v_tok = v_sep.stride(0) if v_sep is not kv else kv.stride(0)
    stride_kv_h = kv.stride(1) if kv.ndim >= 3 else 0
    stride_v_h = v_sep.stride(1) if v_sep.ndim >= 3 else 0

    output_tiles = batch_size * num_heads * n_m_tiles
    total_valid_tiles = output_tiles * split_k
    total_programs = min(total_valid_tiles, num_CUs)

    device = q.device
    if split_k > 1:
        if partial_o is None:
            partial_o = torch.empty(output_tiles * split_k, BLOCK_M, D_V,
                                    dtype=torch.float32, device=device)
        if partial_lse is None:
            partial_lse = torch.empty(output_tiles * split_k, BLOCK_M,
                                      dtype=torch.float32, device=device)
        if sync_count is None:
            sync_count = torch.zeros(output_tiles, dtype=torch.int32, device=device)
        po_stride_tile = partial_o.stride(0)
        po_stride_m = partial_o.stride(1)
        pl_stride_tile = partial_lse.stride(0)
    else:
        if partial_o is None:
            partial_o = torch.empty(1, dtype=torch.float32, device=device)
        if partial_lse is None:
            partial_lse = torch.empty(1, dtype=torch.float32, device=device)
        if sync_count is None:
            sync_count = torch.empty(1, dtype=torch.int32, device=device)
        po_stride_tile = 0
        po_stride_m = 0
        pl_stride_tile = 0

    grid = (total_programs,)
    kwargs = dict(
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        D_NOPE=D_NOPE, D_ROPE=D_ROPE, D_V=D_V,
        NUM_WARPS_CONSTEXPR=NUM_WARPS, NUM_STAGES=NUM_STAGES,
        Q_SCALE=q_scale, KV_SCALE=kv_scale,
        SPLIT_K=split_k,
        num_warps=NUM_WARPS, num_stages=1, schedule_hint=SCHED_HINT,
    )
    if return_compiled:
        kwargs["grid"] = grid
        kwargs["warmup"] = False
        launch = _fp8_splitk.run
    else:
        launch = _fp8_splitk[grid]
    compiled = launch(
        q, kv, v_sep, o,
        qo_indptr, kv_indptr,
        sm_scale,
        q.stride(0), q.stride(1),
        kv.stride(0), stride_kv_h, stride_v_tok, stride_v_h,
        o.stride(0), o.stride(1),
        partial_o, partial_lse, sync_count,
        po_stride_tile, po_stride_m, pl_stride_tile,
        num_heads, n_m_tiles, total_valid_tiles, total_programs,
        **kwargs,
    )
    return compiled, partial_o, partial_lse, sync_count


def _launch_splitk_reduce(partial_o, partial_lse, o, qo_indptr,
                          num_heads, n_m_tiles, batch_size, split_k,
                          return_compiled=False):
    """Launch the split-K reduce kernel that merges partial_o/lse into o.

    One workgroup per output tile; D_V=128 NT=8x16 per-CTA is cheap.
    """
    output_tiles = batch_size * num_heads * n_m_tiles
    grid = (output_tiles,)
    kwargs = dict(
        SPLIT_K=split_k, BLOCK_M=BLOCK_M, D_V=D_V,
        num_warps=4,
    )
    if return_compiled:
        kwargs["grid"] = grid
        kwargs["warmup"] = False
        launch = _fp8_splitk_reduce.run
    else:
        launch = _fp8_splitk_reduce[grid]
    return launch(
        partial_o, partial_lse,
        o, qo_indptr,
        partial_o.stride(0), partial_o.stride(1),
        partial_lse.stride(0),
        o.stride(0), o.stride(1),
        num_heads, n_m_tiles,
        **kwargs,
    )


def mla_prefill_d192_splitk_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float = None,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    v: torch.Tensor = None,
    split_k: int = 2,
):
    """FP8 MLA prefill (D192) — persistent + split-K over KV dimension.

    split_k=1: pure persistent (no workspace, no reduce).
    split_k>=2: KV-split with partial-O + log2-LSE written to workspace,
    then merged by `_splitk_reduce` in a second kernel launch.
    """
    assert q.dtype == torch.float8_e4m3fn
    assert kv.dtype == torch.float8_e4m3fn
    assert o.dtype == torch.bfloat16
    assert q.shape[-1] == D_FULL and kv.shape[-1] == D_FULL and o.shape[-1] == D_V
    assert qo_indptr.dtype == torch.int32 and kv_indptr.dtype == torch.int32

    v_sep = kv if v is None else v
    if v is not None:
        assert v.dtype == torch.float8_e4m3fn and v.shape[-1] == D_V

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D_FULL)

    batch_size = qo_indptr.shape[0] - 1
    if batch_size == 0:
        return o
    num_heads = q.shape[1]
    max_seq_q = int((qo_indptr[1:] - qo_indptr[:-1]).max().item())
    n_m_tiles = (max_seq_q + BLOCK_M - 1) // BLOCK_M
    if n_m_tiles == 0:
        return o
    device = q.device
    num_CUs = torch.cuda.get_device_properties(device).multi_processor_count

    _, partial_o, partial_lse, _ = _launch_splitk(
        q, kv, v_sep, o, qo_indptr, kv_indptr, sm_scale,
        num_heads, n_m_tiles, num_CUs, batch_size, split_k,
        q_scale, kv_scale,
    )
    if split_k > 1:
        _launch_splitk_reduce(
            partial_o, partial_lse, o, qo_indptr,
            num_heads, n_m_tiles, batch_size, split_k,
        )
    return o


def prewarm_mla_d192(device=None, num_heads=16, verbose=True,
                      return_compiled=False, warm_wca=False, warm_sk1=True):
    """JIT-compile every MLA D192 kernel variant the dispatch will select.

    Warms by default:
      - _gen_work_info_kernel (PS metadata generation)
      - _fp8_8w IS_PS_PERSISTENT=True  (PS)
      - _fp8_8w IS_PERSISTENT=False IS_PS_PERSISTENT=False IS_WCA=False (NP)
      - _fp8_splitk SPLIT_K=1          (sk1 — pure persistent, long-tail win)

    When `warm_wca=True`, additionally warms:
      - _fp8_8w IS_WCA=True            (WCA)

    WCA is off by default because SGLang's wrapper no longer dispatches
    to it — bench data showed it never beats PS/NP on MLA prefill. The
    knob is kept so microbench code (which does still import _launch_wca)
    can opt back in without compile-on-first-call latency.

    SK1 (the split-K kernel with SPLIT_K=1 baked in) is a dedicated
    persistent variant that beats both PS and NP on long-tail mixed
    batches — the SGLang wrapper uses it when length variance is high.

    All variants share BLOCK_M=128 BLOCK_N=128 D_NOPE=128 D_ROPE=64 D_V=128
    NUM_WARPS=8 NUM_STAGES=2 Q_SCALE=KV_SCALE=1.0 so the constexpr
    footprint stays tight.

    `num_heads` must match the runtime (DeepSeek-V3/R1 TP8 = 16). The
    kernel's `num_heads` is a RUNTIME arg, not a constexpr, but Triton
    still specializes on divisibility hints for the stride computations
    that depend on it; warming with the wrong value would force a
    recompile on first live call.

    Returns a dict `{"ps": CompiledKernel, "np": ..., "metadata": ...,
    optional "sk1": ..., optional "wca": ...}` when `return_compiled=True`,
    else None.
    """
    import time

    if device is None:
        device = torch.device("cuda:0")
    t0 = time.time()

    NH = num_heads
    BM = BLOCK_M
    # Use bs=2 with a 2-tile shape so both q_start==0 and q_start!=0 code
    # paths get touched during warmup.
    N_TOK = BM * 2
    dummy_q = torch.zeros(N_TOK, NH, D_FULL, dtype=torch.float8_e4m3fn, device=device)
    dummy_kv = torch.zeros(N_TOK, NH, D_FULL, dtype=torch.float8_e4m3fn, device=device)
    dummy_v = torch.zeros(N_TOK, NH, D_V, dtype=torch.float8_e4m3fn, device=device)
    dummy_o = torch.zeros(N_TOK, NH, D_V, dtype=torch.bfloat16, device=device)
    qo_indptr = torch.tensor([0, BM, N_TOK], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, BM, N_TOK], dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(D_FULL)

    num_CUs = torch.cuda.get_device_properties(device).multi_processor_count

    compiled = {}

    # 1. Metadata-gen kernel. Its own constexprs (NUM_HEADS, NUM_CUS,
    # BLOCK_M_C) match the live path exactly.
    work_indptr, work_info = _gen_metadata_gpu(qo_indptr, kv_indptr, NH, num_CUs)

    # 2. PS variant (the hot path for batched prefill).
    ck_ps = _launch_ps(
        dummy_q, dummy_kv, dummy_v, dummy_o,
        qo_indptr, kv_indptr, sm_scale,
        NH, num_CUs, work_indptr, work_info,
        q_scale=1.0, kv_scale=1.0,
        return_compiled=True,
    )
    compiled["ps"] = ck_ps

    # 3. NP variant (routed for tiny-work single-seq).
    n_m_tiles = (BM + BLOCK_M - 1) // BLOCK_M
    batch_size = 2
    ck_np = _launch_non_persistent(
        dummy_q, dummy_kv, dummy_v, dummy_o,
        qo_indptr, kv_indptr, sm_scale,
        NH, n_m_tiles, batch_size,
        1.0, 1.0,
        return_compiled=True,
    )
    compiled["np"] = ck_np

    # 4. SK1 variant — pure-persistent split-K kernel with SPLIT_K=1
    # baked. Wins on long-tail mixed batches (microbench shows 1.5-1.8x
    # vs NP on [512,1024,2048,4096] and bs=8 mixed-length). Does not use
    # workspace tensors (sk=1 path writes output directly).
    #
    # CRITICAL: sk1 reads `n_m_tiles` as a runtime arg to decompose
    # `tile_idx` into `(pid_seq, pid_h, pid_mb)` via `output_tile %
    # n_m_tiles` and `output_tile // n_m_tiles`. If Triton specializes
    # `n_m_tiles == 1` (i.e. we prewarm with a single-tile shape), it
    # bakes `x % 1 = 0` and `x // 1 = x` into the SASS, so every tile
    # decodes `pid_mb = 0` and the kernel only writes `q_start = 0`
    # rows at runtime. To avoid this we prewarm with a dedicated
    # 2-m-tile shape (seq = BLOCK_M*2) so `n_m_tiles = 2`, which falls
    # into Triton's "no specialization hint" class and treats
    # `n_m_tiles` as a real runtime integer for all later invocations.
    # The validator in the SGLang wrapper would catch a regression via
    # a differential bs=4 mixed-length shape, but only if the output
    # buffer is zeroed — torch.empty-based validators miss this because
    # the allocator reuses PS/NP's correct output as stale memory.
    if warm_sk1:
        sk1_N_TOK = BM * 2
        sk1_q = torch.zeros(sk1_N_TOK, NH, D_FULL, dtype=torch.float8_e4m3fn, device=device)
        sk1_kv = torch.zeros(sk1_N_TOK, NH, D_FULL, dtype=torch.float8_e4m3fn, device=device)
        sk1_v = torch.zeros(sk1_N_TOK, NH, D_V, dtype=torch.float8_e4m3fn, device=device)
        sk1_o = torch.zeros(sk1_N_TOK, NH, D_V, dtype=torch.bfloat16, device=device)
        sk1_qo_indptr = torch.tensor([0, sk1_N_TOK], dtype=torch.int32, device=device)
        sk1_kv_indptr = torch.tensor([0, sk1_N_TOK], dtype=torch.int32, device=device)
        sk1_n_m_tiles = (sk1_N_TOK + BLOCK_M - 1) // BLOCK_M  # == 2
        sk1_batch_size = 1
        ck_sk1, _, _, _ = _launch_splitk(
            sk1_q, sk1_kv, sk1_v, sk1_o,
            sk1_qo_indptr, sk1_kv_indptr, sm_scale,
            NH, sk1_n_m_tiles, num_CUs, sk1_batch_size, 1,
            1.0, 1.0,
            return_compiled=True,
        )
        compiled["sk1"] = ck_sk1

    # 5. WCA variant is off by default — SGLang's wrapper no longer
    # dispatches to it. Keep the code path here for microbench-only
    # re-enablement via warm_wca=True.
    if warm_wca:
        seqlen_q = qo_indptr[1:] - qo_indptr[:-1]
        n_m = (seqlen_q + BLOCK_M - 1) // BLOCK_M
        total_tiles = int((n_m * NH).sum().item())
        ck_wca = _launch_wca(
            dummy_q, dummy_kv, dummy_v, dummy_o,
            qo_indptr, kv_indptr, sm_scale,
            NH, num_CUs, total_tiles, batch_size,
            1.0, 1.0,
            return_compiled=True,
        )
        compiled["wca"] = ck_wca

    torch.cuda.synchronize(device)

    dt = time.time() - t0
    if verbose:
        n_attn = 2 + (1 if warm_wca else 0) + (1 if warm_sk1 else 0)
        print(
            f"[Gluon MLA D192] prewarm done: {n_attn} attn variants + metadata "
            f"in {dt:.1f}s (NH={NH}, num_CUs={num_CUs})"
        )
    if return_compiled:
        return compiled
    return None
