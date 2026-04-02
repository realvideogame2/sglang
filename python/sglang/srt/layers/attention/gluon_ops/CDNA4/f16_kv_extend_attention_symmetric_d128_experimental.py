# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental D128 symmetric Gluon kernel fork point.

Goals for this module:
1) Keep an independent copy of `gluon_extend_attn_fwd` so we can diverge.
2) Constrain experimental bring-up to D128 (Lq == Lv == 128).
3) Add an extend (ext>=1) WCA-style split/fix-up prototype with:
   - partial (m/l/acc) per contributing CTA
   - host CTA merge protocol
   - lock-based synchronization
"""

import inspect
import linecache
import math

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_common import *  # noqa: F403
from sglang.srt.layers.attention.gluon_ops.CDNA4 import (
    f16_kv_extend_attention_symmetric as _baseline_sym,
)


def _build_copied_d128_kernel():
    """Build an independent copy of baseline `gluon_extend_attn_fwd`.

    We copy the function source from `f16_kv_extend_attention_symmetric.py`, rename it,
    and inject D128-only static assertions so this fork is intentionally narrow.
    """

    src = inspect.getsource(_baseline_sym.gluon_extend_attn_fwd.fn)
    src = src.replace(
        "def gluon_extend_attn_fwd(",
        "def _gluon_extend_attn_fwd_d128_copied(",
        1,
    )
    src = src.replace(
        "    num_warps: gl.constexpr = gl.num_warps()",
        (
            "    num_warps: gl.constexpr = gl.num_warps()\n"
            "    tl.static_assert(BLOCK_DPE == 0)\n"
            "    tl.static_assert(ACTUAL_BLOCK_DMODEL == 128)\n"
            "    tl.static_assert(ACTUAL_BLOCK_DV == 128)"
        ),
        1,
    )
    virtual_filename = f"{__file__}::d128_copied_kernel"
    linecache.cache[virtual_filename] = (
        len(src),
        None,
        src.splitlines(keepends=True),
        virtual_filename,
    )
    ns = dict(globals())
    exec(compile(src, virtual_filename, "exec"), ns)
    copied = ns["_gluon_extend_attn_fwd_d128_copied"]
    globals()["_gluon_extend_attn_fwd_d128_copied"] = copied
    return copied


# Independent copy of baseline body, constrained to D128.
gluon_extend_attn_fwd = _build_copied_d128_kernel()


@triton.jit
def _cal_num_split_wgs_d128(
    pid: tl.int32,
    tile_iter_end: tl.int32,
    cta_end_tile_gid: tl.int32,
    max_tiles_per_wg: tl.constexpr,
    high_load_wgs: tl.constexpr,
    num_splits: tl.constexpr,
):
    """Port of Lean decode split counting for host-CTA merge fan-in."""
    zero_i = tl.full((), 0, dtype=tl.int32)
    start_cta = tl.cast(pid + 1, tl.int32)
    remaining = tl.maximum(tl.cast(tile_iter_end - cta_end_tile_gid, tl.int32), zero_i)
    cap_high = tl.cast(max_tiles_per_wg, tl.int32)
    cap_low = tl.cast(max_tiles_per_wg - 1, tl.int32)
    cap_low = tl.where(cap_low > 0, cap_low, tl.full((), 1, dtype=tl.int32))
    ctas_high_avail = tl.maximum(tl.cast(high_load_wgs, tl.int32) - start_cta, zero_i)
    total_high_capacity = ctas_high_avail * cap_high
    need_high_only = (remaining + cap_high - 1) // cap_high
    rem_after_high = tl.maximum(remaining - total_high_capacity, zero_i)
    need_low_after_high = (rem_after_high + cap_low - 1) // cap_low
    ctas_needed = tl.where(
        remaining <= total_high_capacity, need_high_only, ctas_high_avail + need_low_after_high
    )
    max_ctas_allowed = tl.maximum(tl.cast(num_splits - 1, tl.int32), zero_i)
    ctas_to_use = tl.minimum(ctas_needed, max_ctas_allowed)
    last_cta = start_cta + ctas_to_use
    last_cta = tl.where(ctas_to_use == 0, start_cta - 1, last_cta)
    return last_cta


@triton.jit
def _d128_ext_wca_persistent_fixup(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    token_q_idx,  # [total_tokens], global q index per output token
    token_ext_start,  # [total_tokens], start q-index of owning request
    token_prefix_start,  # [total_tokens], start in kv_indices
    token_prefix_len,  # [total_tokens], prefix length in tokens
    token_task_start,  # [total_tokens], inclusive start in task space
    token_task_end,  # [total_tokens], exclusive end in task space
    task_token_idx,  # [total_task_blocks], task->token mapping
    kv_indices,
    ready_tags,  # int32 tag per CTA (tile_gid) for synchronization
    partial_m,
    partial_l,
    partial_o,
    sm_scale,
    v_scale,
    total_task_blocks,
    total_tokens,
    kv_group_num: tl.constexpr,
    num_query_heads: tl.constexpr,
    total_programs: tl.constexpr,
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    num_splits: tl.constexpr,
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
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """D128 persistent WCA for extend tokens with host-CTA fix-up merge."""

    pid = tl.program_id(0)

    if pid < high_load_wgs:
        iter_idx = max_tiles_per_wg * pid
        cta_end_tile_gid = iter_idx + max_tiles_per_wg
    else:
        iter_idx = (max_tiles_per_wg - 1) * (pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter_idx + (max_tiles_per_wg - 1)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    while iter_idx < cta_end_tile_gid:
        tile_head_idx = iter_idx // total_task_blocks
        if tile_head_idx < num_query_heads:
            head_iter_base = tile_head_idx * total_task_blocks
            local_head_iter = iter_idx - head_iter_base

            cur_token_idx = tl.load(task_token_idx + local_head_iter)
            token_start = tl.load(token_task_start + cur_token_idx)
            token_end = tl.load(token_task_end + cur_token_idx)
            tile_iter = head_iter_base + token_start
            tile_iter_end = head_iter_base + token_end

            local_iter = iter_idx - tile_iter
            local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter
            host_block = iter_idx == tile_iter
            finishing_block = cta_end_tile_gid >= tile_iter_end

            cur_head = tile_head_idx
            cur_kv_head = cur_head // kv_group_num
            tile_gid = cur_head * total_tokens + cur_token_idx

            q_idx = tl.load(token_q_idx + cur_token_idx)
            q_ptr = Q_Extend + q_idx * stride_qbs + cur_head * stride_qh + offs_d
            q = tl.load(q_ptr)

            kv_start = tl.load(token_prefix_start + cur_token_idx)
            seq_len_prefix = tl.load(token_prefix_len + cur_token_idx)

            m_i = -float("inf")
            l_i = 0.0
            acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

            # Prefix split phase: each CTA works on its assigned prefix block span.
            start_n = local_iter * BLOCK_N
            end_n = local_iter_end * BLOCK_N
            while start_n < end_n:
                if start_n < seq_len_prefix:
                    offs_n = start_n + tl.arange(0, BLOCK_N)
                    mask_n = offs_n < seq_len_prefix
                    kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0)

                    k_ptrs = (
                        K_Buffer
                        + kv_loc[:, None] * stride_buf_kbs
                        + cur_kv_head * stride_buf_kh
                        + offs_d[None, :]
                    )
                    v_ptrs = (
                        V_Buffer
                        + kv_loc[:, None] * stride_buf_vbs
                        + cur_kv_head * stride_buf_vh
                        + offs_d[None, :]
                    )
                    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
                    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

                    qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
                    qk = tl.where(mask_n, qk, -float("inf"))

                    m_new = tl.maximum(m_i, tl.max(qk, axis=0))
                    alpha = tl.exp(m_i - m_new)
                    p = tl.exp(qk - m_new)
                    acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                    l_i = l_i * alpha + tl.sum(p, axis=0)
                    m_i = m_new
                start_n += BLOCK_N

            if not host_block:
                tl.store(partial_m + pid, m_i)
                tl.store(partial_l + pid, l_i)
                tl.store(partial_o + pid * BLOCK_DMODEL + offs_d, acc)
                tl.atomic_xchg(ready_tags + pid, tile_gid)
            else:
                if not finishing_block:
                    last_cta = _cal_num_split_wgs_d128(
                        pid=pid,
                        tile_iter_end=tile_iter_end,
                        cta_end_tile_gid=cta_end_tile_gid,
                        max_tiles_per_wg=max_tiles_per_wg,
                        high_load_wgs=high_load_wgs,
                        num_splits=num_splits,
                    )
                    temp_pid = pid
                    for cta in range((pid + 1), last_cta):
                        temp_pid = temp_pid + 1
                        while tl.atomic_cas(ready_tags + temp_pid, tile_gid, tile_gid) != tile_gid:
                            pass
                        m_cta = tl.load(partial_m + temp_pid)
                        l_cta = tl.load(partial_l + temp_pid)
                        acc_cta = tl.load(partial_o + temp_pid * BLOCK_DMODEL + offs_d)
                        m_new = tl.maximum(m_cta, m_i)
                        alpha = tl.exp(m_cta - m_new)
                        beta = tl.exp(m_i - m_new)
                        l_new = alpha * l_cta + beta * l_i
                        acc = acc_cta * alpha + acc * beta
                        m_i = m_new
                        l_i = l_new

                # Causal extend phase: host adds all prior extend tokens + self.
                ext_start = tl.load(token_ext_start + cur_token_idx)
                ext_len = q_idx - ext_start + 1
                ext_off = 0
                while ext_off < ext_len:
                    offs_n = ext_off + tl.arange(0, BLOCK_N)
                    mask_n = offs_n < ext_len
                    ext_idx = ext_start + offs_n
                    k_ptrs = (
                        K_Extend
                        + ext_idx[:, None] * stride_kbs
                        + cur_kv_head * stride_kh
                        + offs_d[None, :]
                    )
                    v_ptrs = (
                        V_Extend
                        + ext_idx[:, None] * stride_vbs
                        + cur_kv_head * stride_vh
                        + offs_d[None, :]
                    )
                    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
                    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

                    qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
                    qk = tl.where(mask_n, qk, -float("inf"))
                    m_new = tl.maximum(m_i, tl.max(qk, axis=0))
                    alpha = tl.exp(m_i - m_new)
                    p = tl.exp(qk - m_new)
                    acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                    l_i = l_i * alpha + tl.sum(p, axis=0)
                    m_i = m_new
                    ext_off += BLOCK_N

                out = acc / l_i
                if v_scale != 1.0:
                    out = out * v_scale
                o_ptr = O_Extend + q_idx * stride_obs + cur_head * stride_oh + offs_d
                tl.store(o_ptr, out.to(O_Extend.dtype.element_ty))

            delta = local_iter_end - local_iter
            iter_idx = iter_idx + tl.maximum(delta, 1)
        else:
            iter_idx = cta_end_tile_gid


def _calc_persistent_params(
    total_task_blocks: int,
    head_num: int,
    total_programs: int,
):
    total_tiles = total_task_blocks * head_num
    max_tiles_per_wg = (total_tiles + total_programs - 1) // total_programs
    high_load_wgs = total_tiles - ((max_tiles_per_wg - 1) * total_programs)
    high_load_wgs = max(0, min(high_load_wgs, total_programs))
    num_splits = max(1, total_programs)
    return high_load_wgs, max_tiles_per_wg, num_splits


def _build_ext_token_metadata(qo_indptr, kv_indptr, block_n: int, device: torch.device):
    qo_cpu = qo_indptr.to(device="cpu", dtype=torch.int64)
    kv_cpu = kv_indptr.to(device="cpu", dtype=torch.int64)
    batch_size = int(qo_cpu.numel() - 1)

    token_q_idx: list[int] = []
    token_ext_start: list[int] = []
    token_prefix_start: list[int] = []
    token_prefix_len: list[int] = []
    token_task_start: list[int] = []
    token_task_end: list[int] = []
    task_token_idx: list[int] = []

    task_cursor = 0
    for b in range(batch_size):
        q_start = int(qo_cpu[b].item())
        q_end = int(qo_cpu[b + 1].item())
        ext_len = q_end - q_start
        kv_start = int(kv_cpu[b].item())
        kv_end = int(kv_cpu[b + 1].item())
        prefix_len = kv_end - kv_start
        # Keep at least one host task per output token so ext-only tokens
        # (prefix_len == 0) still execute the causal extend phase.
        prefix_blocks = max(1, (prefix_len + (block_n - 1)) // block_n)

        for q_off in range(ext_len):
            token_id = len(token_q_idx)
            token_q_idx.append(q_start + q_off)
            token_ext_start.append(q_start)
            token_prefix_start.append(kv_start)
            token_prefix_len.append(prefix_len)
            token_task_start.append(task_cursor)
            task_cursor += prefix_blocks
            token_task_end.append(task_cursor)
            if prefix_blocks > 0:
                task_token_idx.extend([token_id] * prefix_blocks)

    def _to_i32(values):
        if len(values) == 0:
            return torch.empty((0,), dtype=torch.int32, device=device)
        return torch.tensor(values, dtype=torch.int32, device=device).contiguous()

    return (
        _to_i32(token_q_idx),
        _to_i32(token_ext_start),
        _to_i32(token_prefix_start),
        _to_i32(token_prefix_len),
        _to_i32(token_task_start),
        _to_i32(token_task_end),
        _to_i32(task_token_idx),
    )


def launch_d128_decode1_wca(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    k_scale=1.0,
    v_scale=1.0,
    block_n=64,
):
    """Launch D128 experimental persistent WCA/fix-up variant for ext>=1.

    Returns True when the experimental path is launched, False if the input
    does not match the constrained bring-up domain.
    """

    if q_extend.shape[-1] != 128 or v_extend.shape[-1] != 128:
        return False
    if q_extend.shape[0] == 0:
        return True

    head_num = q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]
    total_tokens = int(q_extend.shape[0])
    total_output_tiles = total_tokens * head_num
    if total_output_tiles <= 0:
        return True

    (
        token_q_idx,
        token_ext_start,
        token_prefix_start,
        token_prefix_len,
        token_task_start,
        token_task_end,
        task_token_idx,
    ) = _build_ext_token_metadata(qo_indptr, kv_indptr, block_n=block_n, device=q_extend.device)

    total_task_blocks = int(task_token_idx.numel())
    if total_task_blocks <= 0:
        # No prefix blocks to split: keep baseline path for this edge case.
        return False

    total_tile_tasks = total_task_blocks * head_num
    num_cus = torch.cuda.get_device_properties(q_extend.device).multi_processor_count
    total_programs = min(total_tile_tasks, num_cus)
    total_programs = max(1, int(total_programs))

    high_load_wgs, max_tiles_per_wg, num_splits = _calc_persistent_params(
        total_task_blocks=total_task_blocks,
        head_num=head_num,
        total_programs=total_programs,
    )
    ready_tags = torch.full((total_programs,), -1, dtype=torch.int32, device=q_extend.device)
    partial_m = torch.full((total_programs,), float("-inf"), dtype=torch.float32, device=q_extend.device)
    partial_l = torch.zeros((total_programs,), dtype=torch.float32, device=q_extend.device)
    partial_o = torch.zeros(total_programs, 128, dtype=torch.float32, device=q_extend.device)

    sm_scale = (sm_scale or (1.0 / math.sqrt(128.0))) * k_scale

    _d128_ext_wca_persistent_fixup[(total_programs,)](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        token_q_idx,
        token_ext_start,
        token_prefix_start,
        token_prefix_len,
        token_task_start,
        token_task_end,
        task_token_idx,
        kv_indices,
        ready_tags,
        partial_m,
        partial_l,
        partial_o,
        sm_scale,
        v_scale,
        total_task_blocks,
        total_tokens,
        kv_group_num=kv_group_num,
        num_query_heads=head_num,
        total_programs=total_programs,
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        num_splits=num_splits,
        stride_qbs=q_extend.stride(0),
        stride_qh=q_extend.stride(1),
        stride_kbs=k_extend.stride(0),
        stride_kh=k_extend.stride(1),
        stride_vbs=v_extend.stride(0),
        stride_vh=v_extend.stride(1),
        stride_obs=o_extend.stride(0),
        stride_oh=o_extend.stride(1),
        stride_buf_kbs=k_buffer.stride(0),
        stride_buf_kh=k_buffer.stride(1),
        stride_buf_vbs=v_buffer.stride(0),
        stride_buf_vh=v_buffer.stride(1),
        BLOCK_N=block_n,
        BLOCK_DMODEL=128,
        num_warps=4,
        num_stages=1,
    )
    return True

