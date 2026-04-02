# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention dispatch for gfx950 (MI350X / CDNA4).

Routes to symmetric kernel (Lq == Lv) or DeepSeek kernel (Lq != Lv)
based on head dimensions.  The public entry point is
gluon_extend_attention_fwd().
"""

import logging
import math
import os

import torch
import triton

from sglang.srt.layers.attention.gluon_ops.CDNA4.f16_kv_extend_attention_symmetric import (
    gluon_extend_attn_fwd as _gluon_extend_attn_fwd_symmetric,
)
from sglang.srt.layers.attention.gluon_ops.CDNA4.f16_kv_extend_attention_symmetric_d128_experimental import (
    gluon_extend_attn_fwd as _gluon_extend_attn_fwd_symmetric_d128_experimental,
    launch_d128_decode1_wca as _launch_d128_decode1_wca_experimental,
)
from sglang.srt.layers.attention.gluon_ops.CDNA4.fp8_kv_extend_attention_symmetric import (
    gluon_extend_attn_fwd as _gluon_extend_attn_fwd_symmetric_fp8,
    gluon_extend_attn_fwd_persistent_fp8 as _gluon_extend_attn_fwd_persistent_fp8,
)
from sglang.srt.layers.attention.gluon_ops.CDNA4.f16_kv_extend_attention_mixed import (
    gluon_extend_attn_fwd as _gluon_extend_attn_fwd_deepseek,
    gluon_extend_attn_fwd_persistent,
    _select_persistent_grid,
    _launch_persistent,
    _ensure_splitk_dummy,
    _ensure_splitk_workspace,
    _splitk_reduce,
    _select_k_splits,
    _launch_splitk,
    _get_num_CUs,
)
from sglang.srt.layers.attention.gluon_ops.CDNA4.fp8_kv_extend_attention_mixed import (
    gluon_extend_attn_fwd as _gluon_extend_attn_fwd_deepseek_fp8,
    _launch_persistent as _launch_persistent_deepseek_fp8,
    _launch_splitk as _launch_splitk_deepseek_fp8,
)

_dummy_cm = None
_dummy_mi = None
_dummy_mi_size = 0
_dummy_wkvo = None
_dummy_wkvo_size = 0

_NUM_XCDS = None
_LOGGED_FP8_KV_MODE = False
logger = logging.getLogger(__name__)

def _get_num_xcds():
    """Return number of XCDs for the current GPU (8 for gfx942/gfx950, 0 to disable).

    Disabled by default; benchmarking shows the 1D-grid decomposition overhead
    outweighs the load-balancing benefit for DeepSeek 16-head shapes on MI350X
    where heads already divide evenly across 8 XCDs.  Set SGLANG_GLUON_NUM_XCDS=8
    to enable for experimentation.
    """
    global _NUM_XCDS
    if _NUM_XCDS is not None:
        return _NUM_XCDS
    override = os.environ.get("SGLANG_GLUON_NUM_XCDS")
    if override is not None:
        _NUM_XCDS = int(override)
    else:
        _NUM_XCDS = 0
    return _NUM_XCDS


def _ensure_dummies(device, mi_size, wkvo_size):
    """Lazy-init module-level singleton dummy tensors on first use."""
    global _dummy_cm, _dummy_mi, _dummy_mi_size, _dummy_wkvo, _dummy_wkvo_size
    if _dummy_cm is None:
        _dummy_cm = torch.empty(0, dtype=torch.uint8, device=device)
    if _dummy_mi is None or _dummy_mi_size < mi_size:
        _dummy_mi = torch.zeros(mi_size, dtype=torch.int64, device=device)
        _dummy_mi_size = mi_size
    if _dummy_wkvo is None or _dummy_wkvo_size < wkvo_size:
        _dummy_wkvo = torch.zeros(wkvo_size, dtype=torch.int32, device=device)
        _dummy_wkvo_size = wkvo_size


_CACHED_ENV_MIXED_DIMS = None
_CACHED_ENV_DEEPSEEK = None
_CACHED_ENV_BLOCK_DPE = None
_CACHED_ENV_FP8_KV_FORCE_BF16 = None
_CACHED_ENV_D128_EXPERIMENTAL = None


def _get_env_flags():
    """Cache os.getenv() lookups for hot-path performance."""
    global _CACHED_ENV_MIXED_DIMS, _CACHED_ENV_DEEPSEEK, _CACHED_ENV_BLOCK_DPE
    if _CACHED_ENV_DEEPSEEK is None:
        _CACHED_ENV_MIXED_DIMS = int(os.getenv("AITER_ENABLE_GLUON_MIXED_DIMS", "0")) != 0
        _CACHED_ENV_DEEPSEEK = int(os.getenv("AITER_ENABLE_GLUON_DEEPSEEK", "1")) != 0
        _CACHED_ENV_BLOCK_DPE = int(os.getenv("AITER_ENABLE_GLUON_BLOCK_DPE", "0")) != 0
    return _CACHED_ENV_MIXED_DIMS, _CACHED_ENV_DEEPSEEK, _CACHED_ENV_BLOCK_DPE


def _use_fp8_kv_bf16_bridge():
    """Whether to force fp8 KV -> bf16-kernel bridge path.

    When enabled, fp8 prefix KV cache tensors are cast to bf16 and dispatched
    through the bf16 Gluon kernels. This is intended as a global fallback /
    perf-regression escape hatch during fp8 rollout.
    """
    global _CACHED_ENV_FP8_KV_FORCE_BF16
    if _CACHED_ENV_FP8_KV_FORCE_BF16 is None:
        _CACHED_ENV_FP8_KV_FORCE_BF16 = (
            int(os.getenv("SGLANG_GLUON_FP8_KV_FORCE_BF16", "0")) != 0
        )
    return _CACHED_ENV_FP8_KV_FORCE_BF16


def _use_d128_experimental_kernel():
    """Whether to route D128 symmetric path to the experimental kernel fork."""
    global _CACHED_ENV_D128_EXPERIMENTAL
    if _CACHED_ENV_D128_EXPERIMENTAL is None:
        _CACHED_ENV_D128_EXPERIMENTAL = (
            int(os.getenv("SGLANG_GLUON_D128_EXPERIMENTAL", "0")) != 0
        )
    return _CACHED_ENV_D128_EXPERIMENTAL


# Pre-computed results for _resolve_qk_split_dims for known head dims.
# Avoids repeated os.getenv() + branching on the hot path.
_QK_SPLIT_CACHE = {}


def _infer_block_dpe(Lq: int):
    """Map head-dim to Triton-style split-DPE metadata."""
    _, allow_deepseek_auto, allow_block_dpe = _get_env_flags()
    if allow_deepseek_auto and Lq in (192, 288, 576):
        allow_block_dpe = True
    if not allow_block_dpe:
        return 0, 0
    if Lq == 576:
        return 64, 64
    if Lq == 288:
        return 32, 32
    if Lq == 192:
        return 64, 64
    return 0, 0


def _resolve_qk_split_dims(Lq: int):
    """Return (BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE)."""
    cached = _QK_SPLIT_CACHE.get(Lq)
    if cached is not None:
        return cached
    block_dpe, actual_block_dpe = _infer_block_dpe(Lq)
    if block_dpe > 0:
        block_dmodel = Lq - block_dpe
        result = (block_dmodel, block_dmodel, block_dpe, actual_block_dpe)
    else:
        block_dmodel = max(triton.next_power_of_2(Lq), 16)
        result = (block_dmodel, Lq, 0, 0)
    _QK_SPLIT_CACHE[Lq] = result
    return result


def _select_d256_dispatch(
    batch_size: int,
    max_len_extend: int,
    min_len_extend: int,
    total_prefix_len: int,
    total_extend_len: int,
):
    """Data-driven d256 launch policy from correctness+perf sweeps.

    Chosen candidates are restricted to configurations that passed all tested
    correctness domains, then ranked by measured Triton-relative speedup.
    """
    total_tokens = max(1, total_prefix_len + total_extend_len)
    prefix_frac = total_prefix_len / total_tokens
    ext_ratio = max_len_extend / max(1, min_len_extend)
    avg_pfx = total_prefix_len // max(1, batch_size)

    # Extend-heavy / long-extend domain.
    if max_len_extend >= 768:
        # Smaller tile wins for low-batch heavy-extend mixes.
        if batch_size <= 2:
            return 64, 4, 4, 16, 16
        return 128, 8, 3, 16, 16

    if prefix_frac <= 0.55:
        return 128, 8, 3, 16, 16

    # Prefix-heavy short-extend domain.
    if max_len_extend <= 128 and avg_pfx >= 2048:
        return 64, 8, 4, 16, 16

    # Mixed domain default.
    if ext_ratio <= 2.5:
        return 64, 4, 2, 16, 16

    # Fallback to a safe high-throughput config.
    return 64, 8, 4, 16, 16


def gluon_extend_attention_fwd(
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
    _force_block_m=None,
    _force_num_warps=None,
    _force_num_stages=None,
    _force_mma_shape=None,
    _force_waves_per_eu=None,
    _force_async_pad_k=None,
    _force_async_pad_v=None,
    _force_use_persistent=None,
    _force_use_splitk=None,
    _force_block_n=None,
    _ck_v_preload=False,
    _mask_split_ext_threshold=1024,
    min_len_extend=None,
    total_prefix_len=None,
    total_extend_len=None,
):
    global _LOGGED_FP8_KV_MODE
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    _kv_is_fp8 = k_buffer.dtype in (torch.float8_e4m3fnuz, torch.float8_e4m3fn)
    _kv_was_fp8 = _kv_is_fp8
    # Global escape hatch: bridge fp8 KV to bf16 kernels when requested.
    _force_bridge = _kv_is_fp8 and _use_fp8_kv_bf16_bridge()
    _force_mixed_dim_bridge = _force_bridge and (Lq != Lv)
    if _force_bridge:
        # Keep fp8 KV cache externally, but bridge to non-fp8 Gluon kernels by
        # casting prefix KV on entry to the active compute dtype.
        bridge_dtype = q_extend.dtype
        k_buffer = k_buffer.to(bridge_dtype)
        v_buffer = v_buffer.to(bridge_dtype)
        _kv_is_fp8 = False
    if _kv_was_fp8 and not _LOGGED_FP8_KV_MODE:
        if _kv_is_fp8:
            if Lq != Lv:
                logger.info(
                    "Gluon FP8 KV path active: native fp8 mixed-dim kernels enabled."
                )
            else:
                logger.info(
                    "Gluon FP8 KV path active: native fp8 symmetric kernels enabled."
                )
        elif _force_mixed_dim_bridge:
            logger.warning(
                "Gluon FP8 mixed-dim bridge enabled by env flag: casting fp8 "
                "KV buffers to non-fp8 kernels."
            )
        else:
            logger.warning(
                "Gluon FP8 KV bridge enabled: casting fp8 KV buffers to non-fp8 kernels."
            )
        _LOGGED_FP8_KV_MODE = True
    if Lq != Lv:
        _kernel_fn = (
            _gluon_extend_attn_fwd_deepseek_fp8
            if _kv_is_fp8
            else _gluon_extend_attn_fwd_deepseek
        )
    elif _kv_is_fp8:
        _kernel_fn = _gluon_extend_attn_fwd_symmetric_fp8
    else:
        if Lq == 128 and _use_d128_experimental_kernel():
            _kernel_fn = _gluon_extend_attn_fwd_symmetric_d128_experimental
        else:
            _kernel_fn = _gluon_extend_attn_fwd_symmetric
    allow_mixed_dims, allow_deepseek_auto, _ = _get_env_flags()
    if allow_deepseek_auto and Lq in (192, 288, 576):
        allow_mixed_dims = True
    if Lq != Lv and not allow_mixed_dims:
        # Direct callers of the Gluon wrapper still get a valid result for mixed dims.
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd as _fallback_extend_attention_fwd,
        )

        return _fallback_extend_attention_fwd(
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
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            skip_prefix_custom_mask=skip_prefix_custom_mask,
            k_scale=k_scale,
            v_scale=v_scale,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=xai_temperature_len,
            config=None,
            min_len_extend=min_len_extend,
            total_prefix_len=total_prefix_len,
            total_extend_len=total_extend_len,
        )
    batch_size = qo_indptr.shape[0] - 1
    head_num = q_extend.shape[1]

    # Experimental D128 ext>=1 split/fix-up path:
    # - independent D128 kernel fork
    # - WCA-style partial-state fix-up with host merge + locks
    if (
        Lq == 128
        and Lv == 128
        and _use_d128_experimental_kernel()
        and (not _kv_is_fp8)
        and custom_mask is None
        and mask_indptr is None
        and is_causal
        and sliding_window_size <= 0
    ):
        _used = _launch_d128_decode1_wca_experimental(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        if _used:
            return

    # -- Fast path: no custom mask, no test overrides --
    # For uniform/B=1: launches with hardcoded constants.
    # For ragged batches with significant imbalance: routes to WCA persistent.
    _is_fast_eligible = (
        _force_block_m is None
        and _force_block_n is None
        and not _ck_v_preload
        and _force_use_persistent is None
        and _force_use_splitk is None
        and custom_mask is None
    )
    _is_uniform = batch_size <= 1 or min_len_extend == max_len_extend
    _is_ragged = _is_fast_eligible and not _is_uniform and batch_size >= 2

    # Route ragged batches to WCA persistent when extend imbalance is significant.
    # Only for D64/D128 (BLOCK_DPE == 0); D256/DeepSeek use the full dispatch path.
    if _is_ragged and Lq <= 128 and Lq == Lv:
        _BLOCK_DMODEL, _ACTUAL_BLOCK_DMODEL, _BLOCK_DPE, _ACTUAL_BLOCK_DPE = (
            _resolve_qk_split_dims(Lq)
        )
        if _BLOCK_DPE == 0:
            ext_ratio = max_len_extend / max(1, min_len_extend) if min_len_extend else float('inf')
            if Lq == 128:
                # D128 ragged: WCA wins 26-175x over Triton on high-imbalance
                # ragged batches. Use WCA for any batch with imbalance >= 2x
                # or when max_extend is meaningful (>= 128).
                _use_wca = (
                    (ext_ratio >= 2.0 and max_len_extend >= 128)
                    or max_len_extend >= 256
                )
            else:
                _use_wca = (
                    (ext_ratio > 4.0 and max_len_extend >= 256)
                    or (ext_ratio > 20.0 and max_len_extend >= 64)
                )
            if _use_wca:
                if _kv_is_fp8 and Lq != Lv:
                    _wca_fn = _launch_persistent_deepseek_fp8
                elif _kv_is_fp8:
                    _wca_fn = _launch_persistent_fp8
                else:
                    _wca_fn = _launch_persistent
                _wca_fn(
                    q_extend, k_extend, v_extend, o_extend,
                    k_buffer, v_buffer,
                    qo_indptr, kv_indptr, kv_indices,
                    custom_mask, is_causal, mask_indptr, max_len_extend,
                    k_scale=k_scale, v_scale=v_scale, sm_scale=sm_scale,
                    logit_cap=logit_cap,
                    skip_prefix_custom_mask=skip_prefix_custom_mask,
                    sliding_window_size=sliding_window_size,
                    sinks=sinks, window_kv_offsets=window_kv_offsets,
                    xai_temperature_len=xai_temperature_len,
                    min_len_extend=min_len_extend,
                )
                return

    # D512 long extend now uses subtiled BN=64+NS=2 in the fast path.
    _d512_long_extend = False

    if _is_fast_eligible and _is_uniform and not _d512_long_extend:
        _BLOCK_DMODEL, _ACTUAL_BLOCK_DMODEL, _BLOCK_DPE, _ACTUAL_BLOCK_DPE = (
            _resolve_qk_split_dims(Lq)
        )

        if _BLOCK_DPE == 0 and Lq == Lv and Lq <= 128:
            _BM_est = 128
            _n_m_est = (max_len_extend + _BM_est - 1) // _BM_est
            _total_tiles_est = batch_size * head_num * _n_m_est
            _num_CUs = _get_num_CUs(q_extend.device)
            _total_ext = batch_size * max_len_extend
            if Lq == 128:
                # D128 WCA dispatch from 102-shape MI350X clean sweep (GPU events)
                # + Llama 8B E2E TTFT validation. WCA wins 75% of kernel shapes
                # and 27/39 E2E cases. BUT for short no-prefix extends (B=1
                # ext<=256), WCA overhead causes 2-3x E2E regression vs Triton.
                # Fix: require minimum work (total_ext >= 512 or has prefix) to
                # use WCA; fall through to basic 4w/8w kernel otherwise.
                _has_prefix = (
                    total_prefix_len is not None and total_prefix_len > 0
                )
                _need_persistent = (
                    (_total_ext >= 512 or _has_prefix or batch_size >= 4)
                    and (
                        _total_ext < 16384
                        or _total_tiles_est < _num_CUs
                    )
                )
            elif Lq == 64:
                _need_persistent = (
                    _total_tiles_est < _num_CUs
                    and max_len_extend >= 128
                    and (batch_size >= 8 or max_len_extend >= 1024)
                )
            else:
                _need_persistent = (
                    _total_tiles_est < _num_CUs
                    and max_len_extend >= 128
                    and (batch_size >= 8 or max_len_extend >= 1024)
                )
            if _need_persistent:
                if min_len_extend is None:
                    min_len_extend = int((qo_indptr[1:] - qo_indptr[:-1]).min().item())
                if _kv_is_fp8 and Lq != Lv:
                    _wca_fn = _launch_persistent_deepseek_fp8
                elif _kv_is_fp8:
                    _wca_fn = _launch_persistent_fp8
                else:
                    _wca_fn = _launch_persistent
                _wca_fn(
                    q_extend, k_extend, v_extend, o_extend,
                    k_buffer, v_buffer,
                    qo_indptr, kv_indptr, kv_indices,
                    custom_mask, is_causal, mask_indptr, max_len_extend,
                    k_scale=k_scale, v_scale=v_scale, sm_scale=sm_scale,
                    logit_cap=logit_cap,
                    sliding_window_size=sliding_window_size,
                    sinks=sinks, window_kv_offsets=window_kv_offsets,
                    xai_temperature_len=xai_temperature_len,
                    min_len_extend=min_len_extend,
                )
                return

        _sm = (sm_scale if sm_scale is not None else Lq**-0.5) * k_scale
        _kv_gn = head_num // k_extend.shape[1]
        _BLOCK_DV = (
            Lv
            if (Lv & (Lv - 1) == 0 and Lv >= 16)
            else max(triton.next_power_of_2(Lv), 16)
        )
        _wkvo = window_kv_offsets
        if _wkvo is None or _dummy_cm is None:
            _ensure_dummies(q_extend.device, q_extend.shape[0] + 1, batch_size)
        if _wkvo is None:
            _wkvo = _dummy_wkvo[:batch_size]

        # Data-driven dispatch from MI350X kernel config sweep.
        _USE_SUBTILE = False
        if Lq != Lv:
            _BN = 64
            if max(_BLOCK_DMODEL, _BLOCK_DV) >= 512:
                _BM, _NW, _NS = 64, 4, 2
                _USE_SUBTILE = True
            else:
                _total_ext = batch_size * max_len_extend
                _has_prefix = (
                    total_prefix_len is not None and total_prefix_len > 0
                )
                if _has_prefix and max_len_extend < 1024:
                    _BM, _NW, _NS = 64, 8, 3
                elif (
                    _total_ext >= 2048
                    or (batch_size >= 4 and max_len_extend >= 512)
                ):
                    _BM, _NW, _NS = 128, 8, 2
                elif batch_size >= 2 and max_len_extend < 512:
                    _BM, _NW, _NS = 64, 4, 3
                else:
                    _BM, _NW, _NS = 64, 4, 2
        elif Lq == 256:
            _BN = 32
            _total_pfx = kv_indices.shape[0]
            _total_ext = batch_size * max_len_extend
            _BM, _NW, _NS, _PAD_K, _PAD_V = _select_d256_dispatch(
                batch_size, max_len_extend, max_len_extend, _total_pfx, _total_ext,
            )
        elif Lq == 64:
            _BN = 64
            _total_ext = batch_size * max_len_extend
            if batch_size >= 16:
                if max_len_extend >= 512:
                    _BM, _NW, _NS = 256, 8, 2
                else:
                    _BM, _NW, _NS = 128, 8, 4
            elif batch_size >= 4:
                if _total_ext >= 2048 or max_len_extend >= 512:
                    _BM, _NW, _NS = 256, 8, 2
                else:
                    _BM, _NW, _NS = 128, 8, 4
            else:
                if max_len_extend >= 2048:
                    _BM, _NW, _NS = 256, 8, 2
                else:
                    _BM, _NW, _NS = 128, 8, 4
        else:
            # D128 basic kernel dispatch. Reached when WCA isn't used:
            # either large extends (total_ext >= 16384) with enough tile
            # parallelism, or short extends (total_ext < 512, B<=3, no pfx)
            # where WCA overhead exceeds the kernel savings.
            _BN = 64
            _total_ext = batch_size * max_len_extend
            if batch_size == 1 and max_len_extend <= 256:
                _BM, _NW, _NS = 128, 8, 4
            elif batch_size == 1:
                _BM, _NW, _NS = 64, 4, 2
            elif _total_ext >= 32768:
                _BM, _NW, _NS = 256, 8, 2
            else:
                _BM, _NW, _NS = 128, 8, 4
        if Lq != Lv:
            _PAD_K, _PAD_V = 16, 16
        elif Lq == 256:
            pass  # _PAD_K, _PAD_V set by _select_d256_dispatch
        else:
            _PAD_K, _PAD_V = 16, 16

        if _kv_is_fp8:
            if Lq <= 128:
                _BN = int(os.environ.get('_GLUON_FP8_BN', '128'))
            _NS = int(os.environ.get('_GLUON_FP8_NS', '2'))
            if Lq == 128:
                # Benchmark sweeps show D128 fp8 is predominantly 8w-optimal.
                _NW = _force_num_warps or 8
            elif Lq >= 128:
                _NW = _force_num_warps or _NW
            else:
                _NW = _force_num_warps or min(_NW, 4)
            _EXT_BN = int(os.environ.get('_GLUON_FP8_EXT_BN', '64'))
            _EXT_NS = int(os.environ.get('_GLUON_FP8_EXT_NS', '3'))
        else:
            _EXT_BN = _BN
            _EXT_NS = _NS

        if _force_async_pad_k is not None:
            _PAD_K = _force_async_pad_k
        if _force_async_pad_v is not None:
            _PAD_V = _force_async_pad_v
        if _force_block_n is not None:
            _BN = _force_block_n

        _kernel_fn.run(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            _dummy_cm,
            _dummy_mi[: q_extend.shape[0] + 1],
            _wkvo,
            _sm,
            _kv_gn,
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
            IS_CAUSAL=is_causal,
            USE_CUSTOM_MASK=False,
            SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
            ENABLE_PREFIX_UNMASKED=is_causal and sliding_window_size <= 0 and (_NW < 8 or _BLOCK_DPE > 0),
            ENABLE_MASK_SPLIT=is_causal and sliding_window_size <= 0 and (_NW < 8 or _BLOCK_DPE > 0),
            BLOCK_M=_BM,
            BLOCK_N=_BN,
            BLOCK_DMODEL=_BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=_ACTUAL_BLOCK_DMODEL,
            BLOCK_DPE=_BLOCK_DPE,
            ACTUAL_BLOCK_DPE=_ACTUAL_BLOCK_DPE,
            BLOCK_DV=_BLOCK_DV,
            ACTUAL_BLOCK_DV=Lv,
            NUM_STAGES=_NS,
            **({
                "EXT_BLOCK_N": _EXT_BN,
                "EXT_NUM_STAGES": _EXT_NS,
            } if (_kv_is_fp8 and Lq == Lv) else {}),
            MMA_INSTR_M=16,
            MMA_INSTR_N=16,
            MMA_INSTR_K=32,
            QK_K_WIDTH=8,
            PV_K_WIDTH=4,
            **({
                "FP8_QK_K_WIDTH": 16,
                "FP8_PV_K_WIDTH": 8,
            } if (_kv_is_fp8 and Lq == Lv) else {}),
            ASYNC_PAD_K=_PAD_K,
            ASYNC_PAD_V=_PAD_V,
            Sinks=sinks,
            HAS_SINK=sinks is not None,
            LOGIT_CAP=logit_cap,
            XAI_TEMPERATURE_LEN=xai_temperature_len,
            SLIDING_WINDOW_SIZE=sliding_window_size,
            V_SCALE=v_scale,
            **({"GRID_NUM_HEADS": head_num if _get_num_xcds() > 0 else 0,
                "GRID_NUM_M_BLOCKS": ((max_len_extend + _BM - 1) // _BM) if _get_num_xcds() > 0 else 0,
                "NUM_XCDS": _get_num_xcds(),
                "USE_SUBTILE": _USE_SUBTILE} if Lq != Lv else {}),
            V_PRELOAD=False,
            num_warps=_NW,
            num_stages=1,
            waves_per_eu=2,
            matrix_instr_nonkdim=32,
            grid=(batch_size * head_num * ((max_len_extend + _BM - 1) // _BM),) if (_get_num_xcds() > 0 and Lq != Lv) else (batch_size, head_num, (max_len_extend + _BM - 1) // _BM),
            warmup=False,
        )
        return

    # -- Full dispatch path (heterogeneous batches, custom mask, test overrides) --

    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE = (
        _resolve_qk_split_dims(Lq)
    )

    if min_len_extend is None:
        min_len_extend = int((qo_indptr[1:] - qo_indptr[:-1]).min().item())

    _ensure_dummies(q_extend.device, q_extend.shape[0] + 1, batch_size)

    # -----------------------------------------------------------------------
    # Dispatch decision: split-K, WCA persistent, or basic kernel.
    #
    # Decision tree derived from D64/D128 oracle analysis (327 shapes, MI350X):
    #   split-K:  tile-starved (tiles < CUs), or high-prefix + few tiles
    #   WCA:      medium/large extends, moderate-to-large prefix
    #   4w:       small ext (<=64), moderate tiles, large prefix
    #   8w:       large ext with small prefix, or very large batches
    # -----------------------------------------------------------------------
    is_deepseek = Lq != Lv
    num_CUs = _get_num_CUs(q_extend.device)

    _BM_est = _force_block_m or (64 if Lq < 256 else 128)
    n_m_tiles_est = (max_len_extend + _BM_est - 1) // _BM_est
    total_tiles_est = batch_size * head_num * n_m_tiles_est

    use_splitk = False
    use_persistent = False

    if _force_use_splitk is not None:
        use_splitk = bool(_force_use_splitk)
    elif _force_use_persistent is not None:
        use_persistent = bool(_force_use_persistent)
    elif Lq >= 256:
        use_persistent = False
    elif is_deepseek:
        ext_ratio = max_len_extend / max(1, min_len_extend)
        use_persistent = (
            batch_size >= 4
            and ext_ratio > 4.0
            and max_len_extend >= 256
        )
    elif BLOCK_DPE == 0 and custom_mask is None and Lq == Lv and Lq <= 128:
        use_splitk = True

    if use_splitk:
        _BN = 32 if max(BLOCK_DMODEL, Lv) >= 256 else 64
        if _kv_is_fp8 and Lq != Lv:
            _sk_fn = _launch_splitk_deepseek_fp8
        elif _kv_is_fp8:
            _sk_fn = _launch_splitk_fp8
        else:
            _sk_fn = _launch_splitk
        _sk_fn(
            q_extend, k_extend, v_extend, o_extend,
            k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices,
            custom_mask, mask_indptr,
            window_kv_offsets if window_kv_offsets is not None else _dummy_wkvo[:batch_size],
            sm_scale, k_scale, v_scale, logit_cap,
            Lq, Lv, is_causal, max_len_extend, min_len_extend,
            sinks, xai_temperature_len, sliding_window_size,
            BLOCK_M=_force_block_m or 64,
            BLOCK_N=_BN,
            num_warps=_force_num_warps or 4,
            NUM_STAGES=_force_num_stages or 2,
            _force_mma_shape=_force_mma_shape,
            _force_async_pad_k=_force_async_pad_k,
            _force_async_pad_v=_force_async_pad_v,
            _force_waves_per_eu=_force_waves_per_eu,
        )
        return

    if use_persistent:
        if _kv_is_fp8 and Lq != Lv:
            _wca_fn = _launch_persistent_deepseek_fp8
        elif _kv_is_fp8:
            _wca_fn = _launch_persistent_fp8
        else:
            _wca_fn = _launch_persistent
        _wca_fn(
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
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            skip_prefix_custom_mask=skip_prefix_custom_mask,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=xai_temperature_len,
            _force_block_m=_force_block_m,
            _force_num_warps=_force_num_warps,
            _force_num_stages=_force_num_stages,
            _force_mma_shape=_force_mma_shape,
            _force_waves_per_eu=_force_waves_per_eu,
            _force_async_pad_k=_force_async_pad_k,
            _force_async_pad_v=_force_async_pad_v,
            min_len_extend=min_len_extend,
        )
        return

    # Mask-split dispatch.
    needs_detailed_check = False
    if max_len_extend >= 2048 and batch_size >= 2:
        if not needs_detailed_check:
            if total_prefix_len is not None and total_extend_len is not None:
                total_prefix = total_prefix_len
                total_extend = total_extend_len
            else:
                total_prefix = int((kv_indptr[-1] - kv_indptr[0]).item())
                total_extend = int((qo_indptr[-1] - qo_indptr[0]).item())
            ppct = total_prefix / max(1, total_prefix + total_extend)
        enable_mask_split = ppct < 0.60
    else:
        enable_mask_split = False
    enable_prefix_unmasked = enable_mask_split

    USE_CUSTOM_MASK = custom_mask is not None
    if not USE_CUSTOM_MASK:
        custom_mask = _dummy_cm
        mask_indptr = _dummy_mi[: q_extend.shape[0] + 1]
    if window_kv_offsets is None:
        window_kv_offsets = _dummy_wkvo[:batch_size]

    BLOCK_DV = max(triton.next_power_of_2(Lv), 16)
    if max(BLOCK_DMODEL, BLOCK_DV) >= 512:
        if max_len_extend >= 1024:
            BLOCK_N = 32
        else:
            BLOCK_N = 64
    elif max(BLOCK_DMODEL, BLOCK_DV) >= 256:
        BLOCK_N = 32
    else:
        BLOCK_N = 64
    if Lq != Lv and max(BLOCK_DMODEL, BLOCK_DV) < 512:
        BLOCK_N = 32 if max(BLOCK_DMODEL, BLOCK_DV) >= 256 else 64
    if _force_block_n is not None:
        BLOCK_N = _force_block_n
    if _kv_is_fp8 and max(BLOCK_DMODEL, BLOCK_DV) < 256:
        BLOCK_N = 128
    EXT_BLOCK_N = BLOCK_N
    AUTO_PAD_K, AUTO_PAD_V = (16, 16)
    NUM_STAGES = 1

    if _force_block_m is not None and _force_num_warps is not None:
        BLOCK_M = _force_block_m
        num_warps = _force_num_warps
    elif Lq != Lv and max(BLOCK_DMODEL, BLOCK_DV) >= 512:
        if BLOCK_N <= 32:
            BLOCK_M, num_warps = 64, 4
        else:
            BLOCK_M, num_warps = 64, 8
    elif max(BLOCK_DMODEL, BLOCK_DV) >= 256:
        if total_prefix_len is None or total_extend_len is None:
            total_prefix_len = int((kv_indptr[-1] - kv_indptr[0]).item())
            total_extend_len = int((qo_indptr[-1] - qo_indptr[0]).item())
        BLOCK_M, num_warps, NUM_STAGES, AUTO_PAD_K, AUTO_PAD_V = _select_d256_dispatch(
            batch_size,
            max_len_extend,
            min_len_extend,
            total_prefix_len,
            total_extend_len,
        )
    elif Lq != Lv:
        BLOCK_M, num_warps = 64, 4
    elif BLOCK_DMODEL == 64:
        # D64 full-path dispatch.
        _total_ext = batch_size * max_len_extend
        if batch_size >= 16:
            if max_len_extend <= 64:
                BLOCK_M, num_warps = 64, 4
            elif max_len_extend <= 256:
                BLOCK_M, num_warps = 128, 4
            elif max_len_extend <= 512:
                BLOCK_M, num_warps = 256, 4
            else:
                BLOCK_M, num_warps = 256, 8
        elif batch_size >= 4:
            if _total_ext >= 2048 or max_len_extend >= 512:
                BLOCK_M, num_warps = 256, 8
            else:
                BLOCK_M, num_warps = 128, 8
        else:
            if max_len_extend >= 2048:
                BLOCK_M, num_warps = 256, 8
            else:
                BLOCK_M, num_warps = 128, 8
    elif BLOCK_DMODEL == 128 and Lq == Lv:
        _total_ext_full = batch_size * max_len_extend
        if batch_size == 1:
            BLOCK_M, num_warps = 64, 4
        elif _total_ext_full >= 32768:
            BLOCK_M, num_warps = 256, 8
        else:
            BLOCK_M, num_warps = 128, 8
    else:
        BLOCK_M = 128
        num_warps = 8

    if _force_num_stages is not None:
        NUM_STAGES = _force_num_stages
    elif max(BLOCK_DMODEL, BLOCK_DV) >= 256:
        if _force_block_m is not None and _force_num_warps is not None:
            NUM_STAGES = 4 if BLOCK_M >= 128 else 2
    elif Lq != Lv:
        NUM_STAGES = 2
    elif BLOCK_DMODEL == 64:
        if batch_size >= 16 and max_len_extend <= 512:
            NUM_STAGES = 4
        elif BLOCK_M == 256:
            NUM_STAGES = 2
        else:
            NUM_STAGES = 4
    elif BLOCK_DMODEL == 128 and Lq == Lv:
        if BLOCK_M >= 256:
            NUM_STAGES = 2
        elif num_warps == 8 and BLOCK_M == 128:
            NUM_STAGES = 4
        elif num_warps == 4 and BLOCK_M == 64:
            NUM_STAGES = 2
        else:
            NUM_STAGES = 2
    elif BLOCK_M == 64:
        NUM_STAGES = 1
    else:
        NUM_STAGES = 4

    if _kv_is_fp8:
        NUM_STAGES = _force_num_stages or 2
        if _force_num_warps is not None:
            num_warps = _force_num_warps
        elif BLOCK_DMODEL >= 128:
            num_warps = 8
        else:
            num_warps = 4
        total_pfx_tokens = kv_indices.shape[0] if kv_indices is not None else 0
        if _force_block_m is not None:
            BLOCK_M = _force_block_m
        elif batch_size * max_len_extend >= 4096 and total_pfx_tokens >= 1024 * batch_size:
            BLOCK_M = 128
        else:
            BLOCK_M = 64
    EXT_NUM_STAGES = NUM_STAGES

    # Correctness guards.
    if BLOCK_DMODEL == 64 and num_warps == 8:
        if BLOCK_M in (64, 128) and NUM_STAGES == 1:
            NUM_STAGES = 2
        if BLOCK_M == 256 and NUM_STAGES in (1, 3, 4):
            NUM_STAGES = 2
    if max(BLOCK_DMODEL, BLOCK_DV) >= 256 and max(BLOCK_DMODEL, BLOCK_DV) < 512 and num_warps == 8 and NUM_STAGES == 1:
        if _force_num_stages is None:
            NUM_STAGES = 2
    _USE_SUBTILE_FULL = False
    if max(BLOCK_DMODEL, BLOCK_DV) >= 512 and _force_num_stages is None:
        if Lq != Lv and BLOCK_N >= 64:
            NUM_STAGES = 2
            num_warps = 4
            _USE_SUBTILE_FULL = True
        elif BLOCK_N <= 32:
            NUM_STAGES = 2
        else:
            NUM_STAGES = 1
            num_warps = max(num_warps, 8)

    if _force_mma_shape == "32x32x16":
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 32, 32, 16
        QK_K_WIDTH, PV_K_WIDTH = 32, 4
    else:
        MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 16, 16, 32
        QK_K_WIDTH, PV_K_WIDTH = 8, 4

    if _force_async_pad_k is not None:
        ASYNC_PAD_K = _force_async_pad_k
    else:
        ASYNC_PAD_K = AUTO_PAD_K if BLOCK_DMODEL >= 256 else 16
    if _force_async_pad_v is not None:
        ASYNC_PAD_V = _force_async_pad_v
    else:
        ASYNC_PAD_V = AUTO_PAD_V if BLOCK_DV >= 256 else 16

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    _kernel_fn.run(
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
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
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
        **({
            "EXT_BLOCK_N": EXT_BLOCK_N,
            "EXT_NUM_STAGES": EXT_NUM_STAGES,
        } if (_kv_is_fp8 and Lq == Lv) else {}),
        MMA_INSTR_M=MMA_INSTR_M,
        MMA_INSTR_N=MMA_INSTR_N,
        MMA_INSTR_K=MMA_INSTR_K,
        QK_K_WIDTH=QK_K_WIDTH,
        PV_K_WIDTH=PV_K_WIDTH,
        **({
            "FP8_QK_K_WIDTH": 16,
            "FP8_PV_K_WIDTH": 8,
        } if (_kv_is_fp8 and Lq == Lv) else {}),
        ASYNC_PAD_K=ASYNC_PAD_K,
        ASYNC_PAD_V=ASYNC_PAD_V,
        Sinks=sinks,
        HAS_SINK=sinks is not None,
        LOGIT_CAP=logit_cap,
        XAI_TEMPERATURE_LEN=xai_temperature_len,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        V_SCALE=v_scale,
        **({"GRID_NUM_HEADS": head_num if _get_num_xcds() > 0 else 0,
            "GRID_NUM_M_BLOCKS": triton.cdiv(max_len_extend, BLOCK_M) if _get_num_xcds() > 0 else 0,
            "NUM_XCDS": _get_num_xcds(),
            "USE_SUBTILE": _USE_SUBTILE_FULL} if Lq != Lv else {}),
        V_PRELOAD=_ck_v_preload,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=_force_waves_per_eu if _force_waves_per_eu is not None else 2,
        matrix_instr_nonkdim=32,
        grid=(batch_size * head_num * triton.cdiv(max_len_extend, BLOCK_M),) if (_get_num_xcds() > 0 and Lq != Lv) else (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M)),
        warmup=False,
    )


# ===-----------------------------------------------------------------------===#
# Test / Benchmark Helpers
# ===-----------------------------------------------------------------------===#


def _run_gluon(q, k, v, kb, vb, qo, kv, ki, o, elens, causal, **kw):
    gluon_extend_attention_fwd(
        q,
        k,
        v,
        o,
        kb,
        vb,
        qo,
        kv,
        ki,
        custom_mask=None,
        is_causal=causal,
        mask_indptr=None,
        max_len_extend=max(elens),
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        min_len_extend=min(elens),
        **kw,
    )


def _run_gluon_persistent(q, k, v, kb, vb, qo, kv, ki, o, elens, causal, **kw):
    _launch_persistent(
        q,
        k,
        v,
        o,
        kb,
        vb,
        qo,
        kv,
        ki,
        custom_mask=None,
        is_causal=causal,
        mask_indptr=None,
        max_len_extend=max(elens),
        sm_scale=1.0 / math.sqrt(q.shape[-1]),
        min_len_extend=min(elens),
        **kw,
    )


# ===-----------------------------------------------------------------------===#
# FP8 Persistent / Split-K Launchers
# ===-----------------------------------------------------------------------===#


def _launch_persistent_fp8(
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
    SPLIT_K=1,
):
    """FP8 persistent launcher: mirrors _launch_persistent but calls the FP8
    persistent kernel with FP8-specific layout parameters."""
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE = _resolve_qk_split_dims(Lq)

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
    BLOCK_N = 128 if Lq <= 128 else (32 if max(BLOCK_DMODEL, BLOCK_DV) >= 256 else 64)
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

    if Lq < 128:
        num_warps = min(num_warps, 4)

    if _force_num_stages is not None:
        NUM_STAGES = _force_num_stages
    else:
        NUM_STAGES = int(os.environ.get('_GLUON_FP8_NS', '2'))

    MMA_INSTR_M, MMA_INSTR_N, MMA_INSTR_K = 16, 16, 32
    QK_K_WIDTH, PV_K_WIDTH = 8, 4
    FP8_QK_K_WIDTH, FP8_PV_K_WIDTH = 16, 8
    EXT_BLOCK_N = int(os.environ.get('_GLUON_FP8_EXT_BN', '64'))
    EXT_NUM_STAGES = int(os.environ.get('_GLUON_FP8_EXT_NS', '3'))

    ASYNC_PAD_K = _force_async_pad_k if _force_async_pad_k is not None else (8 if BLOCK_DMODEL >= 256 else 16)
    ASYNC_PAD_V = _force_async_pad_v if _force_async_pad_v is not None else (32 if BLOCK_DV >= 256 else 16)

    sm_scale = sm_scale or 1.0 / math.sqrt(Lq)
    sm_scale = sm_scale * k_scale
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    device = q_extend.device
    n_m_tiles = (max_len_extend + BLOCK_M - 1) // BLOCK_M
    total_output_tiles = batch_size * head_num * n_m_tiles
    if total_output_tiles == 0:
        return

    num_CUs = _get_num_CUs(device)

    if SPLIT_K <= 1:
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

    _gluon_extend_attn_fwd_persistent_fp8[grid](
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
        EXT_BLOCK_N=EXT_BLOCK_N,
        EXT_NUM_STAGES=EXT_NUM_STAGES,
        MMA_INSTR_M=MMA_INSTR_M,
        MMA_INSTR_N=MMA_INSTR_N,
        MMA_INSTR_K=MMA_INSTR_K,
        QK_K_WIDTH=QK_K_WIDTH,
        PV_K_WIDTH=PV_K_WIDTH,
        FP8_QK_K_WIDTH=FP8_QK_K_WIDTH,
        FP8_PV_K_WIDTH=FP8_PV_K_WIDTH,
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


def _launch_splitk_fp8(
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
    """FP8 split-K launcher: determines SPLIT_K and delegates to
    _launch_persistent_fp8 with the right split factor."""
    head_num = q_extend.shape[1]
    device = q_extend.device
    batch_size = qo_indptr.shape[0] - 1

    BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, _, _ = _resolve_qk_split_dims(Lq)
    BLOCK_DV = max(triton.next_power_of_2(Lv), 16)

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
        _launch_persistent_fp8(
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

    SPLIT_K = _select_k_splits(total_output_tiles, num_CUs)

    if max(BLOCK_DMODEL, BLOCK_DV) < 256:
        BLOCK_M = 128
        num_warps = 8
        NUM_STAGES = int(os.environ.get('_GLUON_FP8_NS', '2'))

    enable_prefix_unmasked = True
    enable_mask_split = (custom_mask is None) and (sliding_window_size <= 0) and is_causal

    _launch_persistent_fp8(
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
        enable_mask_split=enable_mask_split,
        enable_prefix_unmasked=enable_prefix_unmasked,
        _force_block_m=BLOCK_M,
        _force_num_warps=num_warps,
        _force_num_stages=NUM_STAGES,
        _force_async_pad_k=_force_async_pad_k,
        _force_async_pad_v=_force_async_pad_v,
        _force_waves_per_eu=_force_waves_per_eu,
        SPLIT_K=SPLIT_K,
    )
