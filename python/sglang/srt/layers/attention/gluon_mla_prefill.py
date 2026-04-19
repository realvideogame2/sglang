"""Gluon MLA prefill (D192) wrapper for SGLang — opt-in ASM replacement.

Drop-in replacement for AITER's ASM `mla_prefill_ps_asm_fwd + mla_reduce_v1`
pair, writing the final BF16 output directly (no separate reduce kernel).

Gated by `SGLANG_AITER_USE_GLUON_MLA_PREFILL=1`. When import or shape
requirements fail, returns False so the caller keeps the ASM path.

Scheduling strategy via `SGLANG_AITER_GLUON_MLA_SCHED`:
  - `ps`     : metadata-driven persistent (matches ASM closest)
  - `np`     : non-persistent 3D grid (best when total tiles >> num_CUs
               AND tile work is roughly uniform)
  - `sk1`    : pure-persistent split-K kernel with SPLIT_K=1 baked —
               beats PS/NP by 1.5-1.8x on long-tail mixed batches where
               NP's 3D grid over-launches empty CTAs for shorter seqs.
  - `hybrid` : pick per-forward (default). See `_select_scheduler` for
               the exact thresholds.

  WCA (work-centric attention) was previously supported here as a third
  mode but bench data showed it was never faster than PS or NP in MLA
  prefill — it's been removed from the wrapper. The kernel module still
  exposes it for microbenchmarking but the wrapper no longer dispatches
  to it.

Fast-path cache (mirrors `gluon_kernels.cdna4.fa.extend.extend_attention_gfx950`):
after the first call for a given (mode, num_heads, dtypes) tuple we install a
closure that invokes the HIP launcher directly, bypassing Triton's
JITFunction.run specialization (~40us/call). For DeepSeek-R1 TP8 with
MTP enabled (7440 MLA calls per bench run) this saves ~300ms per run, most
of which shows up as a lower mean TTFT.

The Gluon kernel sources live in-tree at
``sglang.srt.layers.attention.gluon_ops.mla_prefill`` so the wrapper is
self-contained. The canonical research copy is in the AMD-Triton/gluon-
kernels repository on branch ``tussingh/mla-d192-prefill`` (same files).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


_PS_FN: Optional[Callable] = None
_PS_LAUNCH_FN: Optional[Callable] = None
_PS_GEN_META_FN: Optional[Callable] = None
_NP_FN: Optional[Callable] = None
_NP_LAUNCH_FN: Optional[Callable] = None
_SK1_FN: Optional[Callable] = None
_SK1_LAUNCH_FN: Optional[Callable] = None
_PREWARM_FN: Optional[Callable] = None
_FP8_8W_JIT = None  # The @gluon.jit CompiledKernel factory (set at import)
_PREWARMED: bool = False

# Metadata cache for PS: keyed on (qo_indptr ptr+shape, kv_indptr ptr+shape,
# num_heads, num_CUs) PLUS a content fingerprint of the two indptrs. SGLang
# reuses slot tensors across forward passes (aiter_backend.py:284) and
# mutates their values in place; without the fingerprint a previously-cached
# metadata tuple would be reused with new sequence lengths, causing OOB
# reads in the attention kernel (see repro_cache_stale.py).
_META_CACHE: dict = {}
_META_CACHE_KEY: Optional[Tuple] = None
_META_CACHE_ORDER: list = []
_META_CACHE_MAX = 4

# Fast-path CompiledKernel cache (see module docstring). Keyed on the
# constexpr/dtype tuple; value is a closure that takes the runtime-varying
# args and invokes HIPLauncher directly. Installed on first call for a given
# key; evicted only on shape changes that don't actually happen in practice
# (DeepSeek-R1 TP8 has one (num_heads, num_CUs, dtype) triple for the
# entire server lifetime, so this cache stays at 3 entries total — one per
# scheduler mode — for all 61 MLA prefill layers * all forward passes).
_FP_CACHE: dict = {}
_FP_CACHE_DISABLED = False  # Escape hatch: set via SGLANG_GLUON_MLA_FASTPATH=0
# Tombstones: keys that failed startup correctness validation. Once tombstoned,
# `_install_fp_entry` refuses to re-install so live dispatches stay on the
# always-correct slow path for that (mode, num_heads, num_CUs, dtypes) tuple.
# Without tombstoning, the first slow-path call after eviction would install
# a new fast-runner whose specialization is tied to *that* call's shape, and
# any subsequent call with a different shape would hit the same baked-constexpr
# bug that caused the original eviction.
_FP_CACHE_TOMBSTONES: set = set()

# Kernel-level constants mirrored from the Gluon module so the dispatch
# heuristic can reason about tile counts without recomputing inside the
# kernel wrapper. Kept in sync with
# sglang/srt/layers/attention/gluon_ops/mla_prefill/mla_prefill_d192_gfx950.py
_BLOCK_M = 128
_BLOCK_N = 128
_D_NOPE = 128
_D_ROPE = 64
_D_V = 128
_NUM_WARPS = 8
_NUM_STAGES = 2

_wrapper_counters = {
    "total": 0,
    "ps": 0,
    "np": 0,
    "sk1": 0,
    "fast_hit": 0,
    "slow_install": 0,
    "fallback_asm": 0,
}


def get_wrapper_counters() -> dict:
    return dict(_wrapper_counters)


def reset_wrapper_counters() -> None:
    for k in list(_wrapper_counters):
        _wrapper_counters[k] = 0


def _try_import_gluon() -> bool:
    """Populate module-level function handles from the vendored Gluon
    kernel sources under ``sglang.srt.layers.attention.gluon_ops.mla_prefill``.

    Safe to call repeatedly. Returns True on success, False if Triton /
    Gluon itself isn't importable on this platform.
    """
    global _PS_FN, _PS_LAUNCH_FN, _PS_GEN_META_FN
    global _NP_FN, _NP_LAUNCH_FN
    global _SK1_FN, _SK1_LAUNCH_FN
    global _PREWARM_FN, _FP8_8W_JIT
    if _PS_FN is not None:
        return True
    try:
        from sglang.srt.layers.attention.gluon_ops.mla_prefill import (
            mla_prefill_d192_fwd,
            mla_prefill_d192_ps_fwd,
            mla_prefill_d192_splitk_fwd,
            prewarm_mla_d192,
            _gen_metadata_gpu,
            _launch_ps,
            _launch_non_persistent,
            _launch_splitk,
            _fp8_8w,
        )
        _PS_FN = mla_prefill_d192_ps_fwd
        _PS_LAUNCH_FN = _launch_ps
        _PS_GEN_META_FN = _gen_metadata_gpu
        _NP_FN = mla_prefill_d192_fwd
        _NP_LAUNCH_FN = _launch_non_persistent
        _SK1_FN = mla_prefill_d192_splitk_fwd
        _SK1_LAUNCH_FN = _launch_splitk
        _PREWARM_FN = prewarm_mla_d192
        _FP8_8W_JIT = _fp8_8w
        logger.info("Gluon MLA prefill enabled (vendored)")
        return True
    except Exception as e:
        logger.warning(
            f"Failed to import Gluon MLA prefill: {e!r}. Falling back to ASM."
        )
        return False


def is_gluon_mla_available() -> bool:
    return _try_import_gluon()


# ---------------------------------------------------------------------------
# Fast-path (CompiledKernel direct-invoke) infrastructure
# ---------------------------------------------------------------------------
# The kernel's constexpr signature is fixed for our usage (BLOCK_M=128,
# BLOCK_N=128, D_NOPE=128, D_ROPE=64, D_V=128, NUM_WARPS=8, NUM_STAGES=2,
# Q_SCALE=1.0, KV_SCALE=1.0). Only IS_PS_PERSISTENT / IS_PERSISTENT vary
# across dispatch modes; IS_WCA is always False post-WCA-removal. Two
# cache entries (ps, np) cover both dispatch modes.
#
# The cache key includes dtypes because Triton specializes on every tensor
# argument's dtype (the pointer element type is baked into the SASS at
# compile time). A dtype mismatch doesn't raise at launch — the kernel
# reads wrong-sized strides and async-faults with HSA_STATUS_ERROR_MEMORY
# _APERTURE_VIOLATION on the next kernel dispatch.

_Q_SCALE = 1.0
_KV_SCALE = 1.0


def _make_mla_fast_runner(compiled_kernel, mode: str):
    """Build a closure that launches the MLA kernel via HIPLauncher directly.

    Bypasses Triton's `JITFunction.run` specialization (arg type inspection +
    CompiledKernel lookup) and `compiled_kernel[grid]` closure indirection.
    The remaining per-call cost is dominated by `_get_current_device` and
    `_get_current_stream` lookups plus the launcher's C++ trampoline.

    Assumes `knobs.runtime.launch_enter_hook / launch_exit_hook` are None at
    closure construction time (the common case without a profiler attached).
    If hooks get installed later, the cached closure won't see them; callers
    that need profiler integration should clear `_FP_CACHE`.
    """
    from triton.runtime import driver as _triton_driver
    from triton import knobs as _triton_knobs

    compiled_kernel._init_handles()
    _hip_launcher = compiled_kernel.run
    _fn_handle = compiled_kernel.function
    _packed_md = compiled_kernel.packed_metadata
    _active = _triton_driver.active
    _get_dev = _active.get_current_device
    _get_stream = _active.get_current_stream
    _enter_hook = _triton_knobs.runtime.launch_enter_hook
    _exit_hook = _triton_knobs.runtime.launch_exit_hook

    # All constexprs are baked into the CompiledKernel; we still pass them
    # through HIPLauncher because the Triton binder expects the full arg
    # list (constexpr + non-constexpr). Constant strides for the unused
    # partial_out/lse buffers ride through as zeros.
    IS_PS_PERSISTENT = (mode == "ps")
    IS_WCA = (mode == "wca")
    IS_PERSISTENT = False  # Reserved path, always False in current dispatch
    BLOCK_M = _BLOCK_M
    BLOCK_N = _BLOCK_N
    D_NOPE = _D_NOPE
    D_ROPE = _D_ROPE
    D_V = _D_V
    NUM_WARPS_CONSTEXPR = _NUM_WARPS
    NUM_STAGES = _NUM_STAGES
    Q_SCALE = _Q_SCALE
    KV_SCALE = _KV_SCALE

    def _fast_run(q, kv, v_sep, o,
                  qo_indptr, kv_indptr, sm_scale,
                  strides8,
                  num_heads, n_m_tiles, total_valid_tiles, total_programs,
                  work_indptr, work_info,
                  dummy_f32_a, dummy_f32_b,
                  batch_size, grid):
        dev = _get_dev()
        stream = _get_stream(dev)
        _hip_launcher(
            grid[0],
            grid[1] if len(grid) > 1 else 1,
            grid[2] if len(grid) > 2 else 1,
            stream, _fn_handle, _packed_md, None,
            _enter_hook, _exit_hook,
            q, kv, v_sep, o,
            qo_indptr, kv_indptr,
            sm_scale,
            strides8[0], strides8[1], strides8[2], strides8[3],
            strides8[4], strides8[5], strides8[6], strides8[7],
            num_heads, n_m_tiles, total_valid_tiles, total_programs,
            work_indptr, work_info,
            dummy_f32_a, dummy_f32_b,
            0, 0, 0, 0,  # stride_po_tok/h stride_pl_tok/h (dummy)
            BLOCK_M, BLOCK_N, D_NOPE, D_ROPE, D_V,
            NUM_WARPS_CONSTEXPR, NUM_STAGES,
            Q_SCALE, KV_SCALE,
            IS_PERSISTENT, IS_PS_PERSISTENT, IS_WCA,
            batch_size,
        )

    return _fast_run


def _make_splitk_fast_runner(compiled_kernel, split_k: int):
    """Build a closure that launches the split-K MLA kernel via HIPLauncher.

    Signature differs from the main _fp8_8w kernel: the split-K kernel
    has `Partial_O, Partial_LSE, Sync_Count` tensor args + three
    partial-stride ints, omits `work_indptr/work_info`, and has an extra
    `SPLIT_K` constexpr. For our fast-path we only cache SPLIT_K=1 (no
    workspace, no reduce kernel needed).
    """
    from triton.runtime import driver as _triton_driver
    from triton import knobs as _triton_knobs

    compiled_kernel._init_handles()
    _hip_launcher = compiled_kernel.run
    _fn_handle = compiled_kernel.function
    _packed_md = compiled_kernel.packed_metadata
    _active = _triton_driver.active
    _get_dev = _active.get_current_device
    _get_stream = _active.get_current_stream
    _enter_hook = _triton_knobs.runtime.launch_enter_hook
    _exit_hook = _triton_knobs.runtime.launch_exit_hook

    BLOCK_M = _BLOCK_M
    BLOCK_N = _BLOCK_N
    D_NOPE = _D_NOPE
    D_ROPE = _D_ROPE
    D_V = _D_V
    NUM_WARPS_CONSTEXPR = _NUM_WARPS
    NUM_STAGES = _NUM_STAGES
    Q_SCALE = _Q_SCALE
    KV_SCALE = _KV_SCALE
    SPLIT_K = int(split_k)

    def _fast_run(q, kv, v_sep, o,
                  qo_indptr, kv_indptr, sm_scale,
                  strides8,
                  partial_o, partial_lse, sync_count,
                  num_heads, n_m_tiles, total_valid_tiles, total_programs,
                  grid):
        dev = _get_dev()
        stream = _get_stream(dev)
        _hip_launcher(
            grid[0],
            grid[1] if len(grid) > 1 else 1,
            grid[2] if len(grid) > 2 else 1,
            stream, _fn_handle, _packed_md, None,
            _enter_hook, _exit_hook,
            q, kv, v_sep, o,
            qo_indptr, kv_indptr,
            sm_scale,
            strides8[0], strides8[1], strides8[2], strides8[3],
            strides8[4], strides8[5], strides8[6], strides8[7],
            partial_o, partial_lse, sync_count,
            0, 0, 0,  # stride_po_tile, stride_po_m, stride_pl_tile — zero for SPLIT_K=1
            num_heads, n_m_tiles, total_valid_tiles, total_programs,
            BLOCK_M, BLOCK_N, D_NOPE, D_ROPE, D_V,
            NUM_WARPS_CONSTEXPR, NUM_STAGES,
            Q_SCALE, KV_SCALE,
            SPLIT_K,
        )

    return _fast_run


def _fp_cache_key(mode, num_heads, num_CUs, q, kv, v_sep, o,
                   qo_indptr, kv_indptr, work_indptr_dtype):
    """Build the fast-path cache key.

    Dtype fields matter because Triton specializes CompiledKernel on each
    tensor arg's dtype (pointer element type is baked into the SASS). Shape
    fields don't matter — the kernel is shape-generic at fixed BLOCK_M/N
    (num_heads is a RUNTIME arg but affects stride specialization, so we
    still key on it).
    """
    return (
        mode,
        int(num_heads),
        int(num_CUs),
        q.dtype, kv.dtype, v_sep.dtype, o.dtype,
        qo_indptr.dtype, kv_indptr.dtype,
        work_indptr_dtype,
    )


def _install_fp_entry(key, compiled_kernel, mode):
    # Respect tombstones from earlier startup validation failures.
    if key in _FP_CACHE_TOMBSTONES:
        return
    # sk1 uses a different kernel (_fp8_splitk) with a different argument
    # signature than the _fp8_8w-based PS/NP modes, so it gets its own
    # fast-runner closure builder.
    if mode == "sk1":
        runner = _make_splitk_fast_runner(compiled_kernel, split_k=1)
    else:
        runner = _make_mla_fast_runner(compiled_kernel, mode)
    _FP_CACHE[key] = (compiled_kernel, runner)


def _fastpath_disabled() -> bool:
    global _FP_CACHE_DISABLED
    if _FP_CACHE_DISABLED:
        return True
    if os.environ.get("SGLANG_GLUON_MLA_FASTPATH", "1") == "0":
        _FP_CACHE_DISABLED = True
        return True
    return False


def clear_fast_path_cache() -> None:
    _FP_CACHE.clear()
    _FP_CACHE_TOMBSTONES.clear()


# ---------------------------------------------------------------------------
# Metadata-gen content fingerprint (batched D2H)
# ---------------------------------------------------------------------------


def _indptr_fingerprint(qo_indptr: torch.Tensor, kv_indptr: torch.Tensor):
    """Compute a content-hash of two small int32 indptrs with a SINGLE D2H.

    The previous implementation did 4 separate `.item()` calls (sum/last of
    each indptr), each forcing its own stream sync (~30us each). Batching
    them into one `cat.cpu()` trades one launch for ~90us saved per MLA
    prefill call. At 7440 calls/bench * 90us = ~670ms saved over a run.
    """
    # Slab both tensors through a 4-int32 staging tensor and D2H in one shot.
    # .sum() and indexing with [-1] each go through a tiny GPU kernel; we
    # stage into a single torch tensor so there's only one device->host
    # synchronisation point.
    dev = qo_indptr.device
    sig = torch.empty(4, dtype=torch.int32, device=dev)
    sig[0] = qo_indptr.sum()
    sig[1] = qo_indptr[-1] if qo_indptr.numel() > 0 else 0
    sig[2] = kv_indptr.sum()
    sig[3] = kv_indptr[-1] if kv_indptr.numel() > 0 else 0
    v = sig.cpu().tolist()
    return (v[0], v[1]), (v[2], v[3])


def _get_cached_ps_metadata(
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    num_heads: int,
    num_CUs: int,
):
    """Cache (work_indptr, work_info) across the 61 MLA-prefill calls in a
    single forward pass.

    CORRECTNESS NOTE — this was a BUG previously. SGLang's AiterAttnBackend
    pre-allocates a qo_indptr slot tensor (aiter_backend.py:284) and mutates
    its values IN PLACE each forward pass. Keying on data_ptr alone meant
    two forward passes with the same batch size but different per-sequence
    lengths would collide on the cache, returning metadata computed for the
    previous forward's sequence lengths — leading to OOB reads/writes and
    ultimately a downstream HIP illegal memory access. We now include a
    device-side content fingerprint as part of the key, batched into a
    single D2H sync.
    """
    global _META_CACHE_ORDER

    qo_sig, kv_sig = _indptr_fingerprint(qo_indptr, kv_indptr)
    key = (
        qo_indptr.data_ptr(),
        qo_indptr.shape[0],
        qo_sig,
        kv_indptr.data_ptr(),
        kv_indptr.shape[0],
        kv_sig,
        num_heads,
        num_CUs,
    )
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    work_indptr, work_info = _PS_GEN_META_FN(
        qo_indptr, kv_indptr, num_heads, num_CUs
    )
    _META_CACHE[key] = (work_indptr, work_info)
    _META_CACHE_ORDER.append(key)
    while len(_META_CACHE_ORDER) > _META_CACHE_MAX:
        drop = _META_CACHE_ORDER.pop(0)
        _META_CACHE.pop(drop, None)
    return work_indptr, work_info


def clear_metadata_cache() -> None:
    global _META_CACHE_ORDER
    _META_CACHE.clear()
    _META_CACHE_ORDER = []


# ---------------------------------------------------------------------------
# Prewarm (phase 1: compile all variants; phase 2: populate fast-path cache)
# ---------------------------------------------------------------------------


def _bf16_mla_reference(q_bf, k_bf, v_bf, qo_indptr, num_heads, d_v=128,
                        d_qk=192):
    """Eager BF16 MLA reference — causal, no KV-split. Used only for
    startup validation of the fast-path cache; not on the hot path.
    """
    device = q_bf.device
    total_q = q_bf.shape[0]
    bs = qo_indptr.shape[0] - 1
    out = torch.empty(total_q, num_heads, d_v, dtype=torch.bfloat16, device=device)
    sm_scale = 1.0 / math.sqrt(d_qk)
    for i in range(bs):
        s = int(qo_indptr[i].item())
        e = int(qo_indptr[i + 1].item())
        if s == e:
            continue
        qh = q_bf[s:e].transpose(0, 1).float()
        kh = k_bf[s:e].transpose(0, 1).float()
        vh = v_bf[s:e].transpose(0, 1).float()
        scores = qh @ kh.transpose(-1, -2) * sm_scale
        mask = torch.triu(
            torch.ones(e - s, e - s, device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        out[s:e] = (torch.softmax(scores, -1) @ vh
                    ).transpose(0, 1).to(torch.bfloat16)
    return out


def _validate_fast_path_correctness(device, num_heads, num_CUs, installed):
    """Per-mode differential test run once at prewarm time.

    For every (mode, cache_key) in `installed`, invoke the fast-runner on
    a shape that differs from the prewarm shape (bs=2, seq_lens=[128,128])
    and compare against a BF16 reference. If the relative error exceeds
    `_FP_VALIDATE_TOL`, evict that cache entry so live traffic will fall
    through to the slow `_launch_*` path (which goes through JIT
    specialization and will recompile correctly for any shape).

    Covers the scenario where a future kernel edit reads one of the
    currently-unused constexpr-specialized args (slots 16/17/18 baked as
    `1` at prewarm). Also catches any drift in the kernel that makes the
    fast-path diverge from the slow-path; the test is cheap (~50 ms
    total across 3 modes) and runs only at server startup.

    Returns the list of evicted mode names.
    """
    # Differential shape: bs=4 with seq_lens [128, 256, 192, 200] so
    #   - total_q = 776 (not multiple of BLOCK_M, not a prewarm size)
    #   - max n_m_tiles across seqs = 2 (prewarm had 1)
    #   - tiles_per_CU tiny so NP/PS paths are both exercisable
    tol = 0.1  # well above the ~3.2e-2 FP8-quant noise floor
    seq_lens = [128, 256, 192, 200]
    total_q = sum(seq_lens)
    bs = len(seq_lens)

    offs = [0]
    for s in seq_lens:
        offs.append(offs[-1] + s)
    qo_indptr = torch.tensor(offs, dtype=torch.int32, device=device)
    kv_indptr = qo_indptr.clone()

    # Deterministic sample so evictions are reproducible.
    g = torch.Generator(device=device).manual_seed(0x5FA17A57)
    q_bf = torch.randn(total_q, num_heads, _D_NOPE + _D_ROPE,
                       dtype=torch.bfloat16, device=device, generator=g) * 0.2
    k_bf = torch.randn(total_q, num_heads, _D_NOPE + _D_ROPE,
                       dtype=torch.bfloat16, device=device, generator=g) * 0.2
    v_bf = torch.randn(total_q, num_heads, _D_V,
                       dtype=torch.bfloat16, device=device, generator=g) * 0.2
    q = q_bf.to(torch.float8_e4m3fn)
    k = k_bf.to(torch.float8_e4m3fn)
    v = v_bf.to(torch.float8_e4m3fn)
    sm_scale = 1.0 / math.sqrt(_D_NOPE + _D_ROPE)

    ref = _bf16_mla_reference(q_bf, k_bf, v_bf, qo_indptr, num_heads,
                              d_v=_D_V, d_qk=_D_NOPE + _D_ROPE)
    ref_norm = ref.float().norm().item() + 1e-6

    # Lazy dummies (validation happens on the same device as prewarm).
    dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
    dummy_f32 = torch.zeros(1, dtype=torch.float32, device=device)

    evicted = []
    for mode, key in installed:
        entry = _FP_CACHE.get(key)
        if entry is None:
            continue
        _ck, fast_run = entry
        # IMPORTANT: torch.zeros, not torch.empty. PyTorch's caching
        # allocator reuses freed memory, so `torch.empty` for mode=sk1
        # would silently inherit the correct output from the prior PS/NP
        # validation pass — masking any partial-write bug in the fast
        # kernel. Zeroing forces the validator to see only what the fast
        # kernel actually wrote.
        o = torch.zeros(total_q, num_heads, _D_V,
                        dtype=torch.bfloat16, device=device)
        try:
            if mode == "ps":
                # Need fresh metadata for the differential shape.
                work_indptr, work_info = _get_cached_ps_metadata(
                    qo_indptr, kv_indptr, num_heads, num_CUs)
                _run_ps_fastpath(
                    fast_run, q, k, v, o,
                    qo_indptr, kv_indptr, sm_scale,
                    num_heads, num_CUs,
                    work_indptr, work_info,
                    dummy_f32,
                )
            elif mode == "np":
                max_seq_q = max(seq_lens)
                n_m_tiles = (max_seq_q + _BLOCK_M - 1) // _BLOCK_M
                _run_np_fastpath(
                    fast_run, q, k, v, o,
                    qo_indptr, kv_indptr, sm_scale,
                    num_heads, n_m_tiles, bs,
                    dummy_i32, dummy_f32,
                )
            elif mode == "sk1":
                max_seq_q = max(seq_lens)
                n_m_tiles = (max_seq_q + _BLOCK_M - 1) // _BLOCK_M
                _run_sk1_fastpath(
                    fast_run, q, k, v, o,
                    qo_indptr, kv_indptr, sm_scale,
                    num_heads, n_m_tiles, num_CUs, bs,
                    dummy_i32, dummy_f32,
                )
            torch.cuda.synchronize(device)
            err = (o.float() - ref.float()).norm().item() / ref_norm
            logger.debug(f"[Gluon MLA] fast-path {mode} validation rel_err={err:.3e} (tol={tol})")
            if not math.isfinite(err) or err > tol:
                _FP_CACHE.pop(key, None)
                _FP_CACHE_TOMBSTONES.add(key)
                evicted.append(mode)
                logger.warning(
                    f"[Gluon MLA] fast-path {mode} failed correctness check "
                    f"(rel_err={err:.3e} > tol={tol}); "
                    "evicted+tombstoned, falling back to slow path for life of process."
                )
        except Exception as e:
            _FP_CACHE.pop(key, None)
            _FP_CACHE_TOMBSTONES.add(key)
            evicted.append(mode)
            logger.warning(
                f"[Gluon MLA] fast-path {mode} raised during validation "
                f"({e!r}); evicted+tombstoned."
            )
    return evicted


def prewarm_mla(device: Optional[torch.device] = None,
                num_heads: int = 16,
                verbose: bool = False) -> None:
    """Compile all Gluon MLA kernel variants up-front AND populate the
    fast-path CompiledKernel cache so the first live call is already at
    steady-state overhead.

    Warms PS + NP + sk1 attention kernels plus the PS metadata-gen kernel.
    For MI350X with DeepSeek-V3/R1 TP8 (num_heads=16, num_CUs=256) this
    runs in ~4 seconds.

    CORRECTNESS — the fast-path is only safe as long as the kernel doesn't
    read any integer arg that Triton specialized as a compile-time
    constant at prewarm time. Audit of mla_prefill_d192_gfx950.py (current):
      * PS (IS_PS_PERSISTENT=True) : work_indptr-driven; positions 16/17/18
        are never read -> baked `1` values are harmless.
      * NP (all IS_* = False)      : gl.program_id-driven; same three args
        are never read -> harmless.
      * sk1 (_fp8_splitk, SPLIT_K=1 baked as constexpr): persistent
        tile_idx loop that reads `n_m_tiles` at runtime to decompose
        `output_tile // n_m_tiles` and `output_tile % n_m_tiles` into
        `(pid_seq, pid_h, pid_mb)`. Prewarm_mla_d192 uses a 2-m-tile
        dummy (seq = BLOCK_M*2) so `n_m_tiles = 2`, which is neither
        `equal_to_1` nor `multiple_of_16` - Triton leaves it as a plain
        runtime integer. Prewarming with a 1-tile shape would bake
        `n_m_tiles == 1` and the kernel would only write `q_start = 0`
        rows at runtime.

    Validator uses `torch.zeros` for the output buffer (not
    `torch.empty`) — the PyTorch allocator reuses freed blocks, so a
    stale PS/NP output would mask partial-write bugs in sk1 by leaving
    the correct values in the unwritten rows. Zeroing forces the
    validator to see only what the current kernel actually wrote.

    We validate the above at prewarm time by running a differential test
    (see `_validate_fast_path_correctness`): each populated cache entry
    is invoked on a shape that differs from the prewarm shape and its
    output compared against a BF16 reference. If any variant fails the
    check we evict that cache entry so subsequent calls go through the
    slow path. This catches kernel updates that accidentally widen the
    set of constexpr-specialized read sites.

    No-op if Gluon is unavailable or already warmed. Safe to call
    repeatedly; subsequent calls are cheap.
    """
    global _PREWARMED
    if _PREWARMED:
        return
    if not _try_import_gluon():
        return
    try:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())

        # Phase 1 + 2: prewarm_mla_d192 with return_compiled=True compiles
        # every variant AND launches each one against dummy inputs (which
        # forces hipModuleLoadData, so HIP module load latency is paid
        # here rather than on the first live request). We install the
        # captured CompiledKernels directly into _FP_CACHE so the first
        # real dispatch call skips even the JITFunction.run lookup.
        compiled = _PREWARM_FN(
            device=device, num_heads=num_heads, verbose=verbose,
            return_compiled=True,
        )
        if compiled is not None and not _fastpath_disabled():
            num_CUs = torch.cuda.get_device_properties(device).multi_processor_count
            # The fp_cache keys on dtypes; prewarm always uses FP8_E4M3 for
            # Q/KV/V and BF16 for O, with int32 indptrs (matching AITER's
            # conventions in aiter_backend.py). Non-PS modes use int32
            # dummy tensors for work_indptr/work_info as well, so one
            # work_indptr dtype covers all three modes.
            q_dt = torch.float8_e4m3fn
            kv_dt = torch.float8_e4m3fn
            v_dt = torch.float8_e4m3fn
            o_dt = torch.bfloat16
            ip_dt = torch.int32
            wi_dt = torch.int32
            installed = []
            for mode in ("ps", "np", "sk1"):
                ck = compiled.get(mode)
                if ck is None:
                    continue
                key = (
                    mode, int(num_heads), int(num_CUs),
                    q_dt, kv_dt, v_dt, o_dt,
                    ip_dt, ip_dt, wi_dt,
                )
                _install_fp_entry(key, ck, mode)
                installed.append((mode, key))
            # Defense-in-depth: validate each cached fast-runner against a
            # shape that didn't match prewarm. Evict anything that
            # produces wrong output so live traffic falls through to the
            # verified slow path.
            if installed:
                evicted = _validate_fast_path_correctness(
                    device=device, num_heads=num_heads,
                    num_CUs=num_CUs, installed=installed,
                )
                if verbose:
                    logger.info(
                        f"Gluon MLA fast-path cache populated: "
                        f"{len(_FP_CACHE)}/{len(installed)} entries "
                        f"(NH={num_heads}, num_CUs={num_CUs}), "
                        f"evicted={evicted or 'none'}"
                    )
        _PREWARMED = True
    except Exception as e:
        logger.warning(f"Gluon MLA prewarm failed (non-fatal): {e!r}")


# ---------------------------------------------------------------------------
# Scheduler heuristic (unchanged)
# ---------------------------------------------------------------------------


def _select_scheduler(total_q: int, batch_size: int, max_seq_q: int,
                       num_heads: int, num_CUs: int) -> str:
    """Pick PS / NP / sk1 based on workload shape.

    Caller must have already computed `max_seq_q = max_i(seq_q_i)` (one
    D2H sync that every path needs anyway for n_m_tiles).

    Heuristic summary:
      * Uniform batches (all seqs equal) -> NP  (HW scheduler + 3D grid
        rebalance causal tiles well; split-K's reduce overhead never
        pays off when NP already fills 256 CUs).
      * Long-tail mixed batches (NP's implicit grid would over-launch
        empty tiles for shorter seqs) -> sk1 (persistent split-K with
        SPLIT_K=1 baked; dynamic work stealing via tile_idx += NUM_CUS
        balances heterogeneous tile costs across CUs).
      * Small total work (tiles_per_CU tiny, not enough parallelism for
        NP over-launch to matter) -> NP still.
      * Moderate tiles_per_CU with uniform lengths -> PS (rarely the
        winner post-sk1 but kept as a fallback for the in-between
        regime).

    The sk1 threshold is `uniform_ratio = total_q / (batch_size * max_seq_q)`.
    When this is < ~0.7, NP's grid would launch `batch * num_heads *
    max_n_m` CTAs but ~30%+ of them exit immediately (q_start >= seqlen_q);
    sk1's persistent loop skips those without launch overhead.
    """
    mode = os.environ.get("SGLANG_AITER_GLUON_MLA_SCHED", "hybrid").lower()
    if mode in ("ps", "np", "sk1"):
        return mode
    # Back-compat: old "wca" value silently coerces to hybrid.
    if mode == "wca":
        mode = "hybrid"

    if batch_size <= 0 or max_seq_q <= 0:
        return "np"

    n_m_tiles = (max_seq_q + _BLOCK_M - 1) // _BLOCK_M
    np_slots = batch_size * num_heads * n_m_tiles

    # Uniform-ratio gate. Works without per-seq iteration: if every seq
    # hit `max_seq_q` we'd have `total_q == batch_size * max_seq_q`; any
    # ratio < 1.0 means at least one seq is shorter than the max.
    uniform_total = batch_size * max_seq_q
    uniform_ratio = total_q / uniform_total if uniform_total > 0 else 1.0

    # sk1 window: mixed lengths AND enough work for NP's over-launch to
    # bite. The np_slots > num_CUs gate prevents sk1 from firing on
    # tiny batches where NP launches < num_CUs CTAs to begin with.
    if uniform_ratio < 0.7 and np_slots > num_CUs:
        return "sk1"

    upper_tiles = n_m_tiles * num_heads * batch_size
    if upper_tiles <= num_CUs:
        return "np"
    tiles_per_cu = upper_tiles / num_CUs
    if tiles_per_cu <= 4.0:
        return "ps"
    return "np"


# ---------------------------------------------------------------------------
# Dispatch (with fast-path cache install + hit)
# ---------------------------------------------------------------------------


_DEBUG = os.environ.get("SGLANG_GLUON_MLA_DEBUG", "").lower() in ("1", "true", "yes")
_SYNC_AFTER = os.environ.get("SGLANG_GLUON_MLA_SYNC", "").lower() in ("1", "true", "yes")
# Fence mode: kernel-launch-based "read-then-discard" that forces the
# previous gluon kernel's writes to be globally visible before any
# subsequent GEMM reads the same memory. Values:
#   "off"   : disabled
#   "cast"  : an explicit BF16->FP32 cast of o (kernel reads all of o)
#   "d2h"   : cast + a single-element .item() D2H (strongest; forces
#             a full device-side memory fence)
#   "zero"  : pre-zero `o` before kernel (no post-ops)
_FENCE_MODE = os.environ.get("SGLANG_GLUON_MLA_FENCE", "off").lower()
_TRACE = os.environ.get("SGLANG_GLUON_MLA_TRACE", "").lower() in ("1", "true", "yes")
_DEBUG_CALL_COUNT = 0
_TRACE_CALL_COUNT = 0


def _run_ps_fastpath(fast_run, q, kv, v_sep, o,
                     qo_indptr, kv_indptr, sm_scale,
                     num_heads, num_CUs,
                     work_indptr, work_info,
                     dummy_f32):
    strides8 = (
        q.stride(0), q.stride(1),
        kv.stride(0),
        kv.stride(1) if kv.ndim >= 3 else 0,
        v_sep.stride(0) if v_sep is not kv else kv.stride(0),
        v_sep.stride(1) if v_sep.ndim >= 3 else 0,
        o.stride(0), o.stride(1),
    )
    fast_run(
        q, kv, v_sep, o,
        qo_indptr, kv_indptr, sm_scale,
        strides8,
        num_heads, 1, 1, 1,
        work_indptr, work_info,
        dummy_f32, dummy_f32,
        0, (num_CUs,),
    )


def _run_np_fastpath(fast_run, q, kv, v_sep, o,
                     qo_indptr, kv_indptr, sm_scale,
                     num_heads, n_m_tiles, batch_size,
                     dummy_i32, dummy_f32):
    strides8 = (
        q.stride(0), q.stride(1),
        kv.stride(0),
        kv.stride(1) if kv.ndim >= 3 else 0,
        v_sep.stride(0) if v_sep is not kv else kv.stride(0),
        v_sep.stride(1) if v_sep.ndim >= 3 else 0,
        o.stride(0), o.stride(1),
    )
    fast_run(
        q, kv, v_sep, o,
        qo_indptr, kv_indptr, sm_scale,
        strides8,
        num_heads, n_m_tiles, 1, 1,
        dummy_i32, dummy_i32,
        dummy_f32, dummy_f32,
        0, (batch_size, num_heads, n_m_tiles),
    )


def _run_sk1_fastpath(fast_run, q, kv, v_sep, o,
                      qo_indptr, kv_indptr, sm_scale,
                      num_heads, n_m_tiles, num_CUs, batch_size,
                      dummy_i32, dummy_f32):
    """Launch the split-K kernel's fast-runner with SPLIT_K=1 (no reduce).

    Derived args mirror `_launch_splitk`'s SPLIT_K=1 path but avoid
    touching the `.stride()` accessors on dummy tensors (which would
    re-lookup every call); we build `strides8` in-line and feed zero
    partial-strides through the runner's closure.
    """
    strides8 = (
        q.stride(0), q.stride(1),
        kv.stride(0),
        kv.stride(1) if kv.ndim >= 3 else 0,
        v_sep.stride(0) if v_sep is not kv else kv.stride(0),
        v_sep.stride(1) if v_sep.ndim >= 3 else 0,
        o.stride(0), o.stride(1),
    )
    output_tiles = batch_size * num_heads * n_m_tiles
    total_valid_tiles = output_tiles  # SPLIT_K=1
    total_programs = output_tiles if output_tiles < num_CUs else num_CUs
    fast_run(
        q, kv, v_sep, o,
        qo_indptr, kv_indptr, sm_scale,
        strides8,
        dummy_f32, dummy_f32, dummy_i32,
        num_heads, n_m_tiles, total_valid_tiles, total_programs,
        (total_programs,),
    )


# Tiny per-device dummy tensors (int32 all-zeros, f32 one-element). Reused
# across fast-path calls so we don't allocate on every dispatch. One pair
# per device; created lazily on first use.
_FP_DUMMY_I32: dict = {}
_FP_DUMMY_F32: dict = {}


def _get_fp_dummies(device):
    d_i32 = _FP_DUMMY_I32.get(device)
    if d_i32 is None:
        d_i32 = torch.zeros(1, dtype=torch.int32, device=device)
        _FP_DUMMY_I32[device] = d_i32
    d_f32 = _FP_DUMMY_F32.get(device)
    if d_f32 is None:
        d_f32 = torch.zeros(1, dtype=torch.float32, device=device)
        _FP_DUMMY_F32[device] = d_f32
    return d_i32, d_f32


def gluon_mla_fp8_prefill_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float,
    num_heads: int,
    v_head_dim: int,
    input_dtype: torch.dtype,
) -> Tuple[torch.Tensor, str]:
    """Run Gluon MLA prefill and return (output, sched_tag_used).

    Inputs mirror the ASM path:
      q: [total_q, num_heads, qk_head_dim=192]  (FP8 e4m3)
      k: [total_kv, num_heads, qk_head_dim=192] (FP8 e4m3)
      v: [total_kv, num_heads, v_head_dim=128]  (FP8 e4m3)
      qo_indptr: int32 [B+1]
      kv_indptr: int32 [B+1]

    Output:
      o: [total_q, num_heads, v_head_dim=128]  (BF16 / input_dtype)
    """
    global _DEBUG_CALL_COUNT, _TRACE_CALL_COUNT
    _wrapper_counters["total"] += 1
    total_q = q.shape[0]
    device = q.device
    num_CUs = torch.cuda.get_device_properties(device).multi_processor_count

    if _TRACE:
        _TRACE_CALL_COUNT += 1
        if _TRACE_CALL_COUNT <= 256 or _TRACE_CALL_COUNT % 100 == 0:
            bs = qo_indptr.shape[0] - 1
            qo_cpu = qo_indptr.cpu().tolist() if qo_indptr.numel() < 20 else None
            kv_cpu = kv_indptr.cpu().tolist() if kv_indptr.numel() < 20 else None
            logger.warning(
                f"[gluon_trace] call={_TRACE_CALL_COUNT} "
                f"total_q={total_q} H={num_heads} bs={bs} "
                f"q={tuple(q.shape)} k={tuple(k.shape)} "
                f"v={tuple(v.shape) if v is not None else None} "
                f"qo={qo_cpu} kv={kv_cpu}"
            )

    # Pre-kernel init: "inf" pre-fills with +inf; "zero" pre-zeros.
    _pre_inf = _DEBUG or _FENCE_MODE in ("inf", "inf_only") or _FENCE_MODE.startswith("inf_")
    _pre_zero = _FENCE_MODE == "zero" or _FENCE_MODE.startswith("zero_")
    if _pre_inf:
        o = torch.full(
            (total_q, num_heads, v_head_dim),
            float("inf"),
            device=device,
            dtype=input_dtype,
        )
    elif _pre_zero:
        o = torch.zeros(
            (total_q, num_heads, v_head_dim),
            device=device,
            dtype=input_dtype,
        )
    else:
        o = q.new_empty((total_q, num_heads, v_head_dim), dtype=input_dtype)

    # Pre-compute max_seq_q / n_m_tiles once. Every dispatch path needs
    # these (NP for grid-Z, sk1 for total_valid_tiles, scheduler for the
    # uniform-ratio heuristic). One .max().item() sync total.
    batch_size = qo_indptr.shape[0] - 1
    if batch_size > 0:
        seqlen_q = qo_indptr[1:] - qo_indptr[:-1]
        max_seq_q = int(seqlen_q.max().item())
    else:
        max_seq_q = 0
    n_m_tiles = (max_seq_q + _BLOCK_M - 1) // _BLOCK_M

    mode = _select_scheduler(total_q, batch_size, max_seq_q, num_heads, num_CUs)

    v_sep = v if v is not None else k
    dummy_i32, dummy_f32 = _get_fp_dummies(device)

    # Fast-path: if we have a cached CompiledKernel for this (mode, dtype)
    # combo, invoke HIPLauncher directly. Otherwise go through the kernel
    # module's _launch_* function with return_compiled=True, capture the
    # CompiledKernel, install in the fast-path cache, and from the next
    # call onwards we hit the fast path.
    fp_enabled = not _fastpath_disabled()
    if fp_enabled:
        fp_key = _fp_cache_key(
            mode, num_heads, num_CUs, q, k, v_sep, o,
            qo_indptr, kv_indptr,
            work_indptr_dtype=torch.int32,
        )
    else:
        fp_key = None
    fp_entry = _FP_CACHE.get(fp_key) if fp_key is not None else None

    if mode == "ps":
        work_indptr, work_info = _get_cached_ps_metadata(
            qo_indptr, kv_indptr, num_heads, num_CUs
        )
        if fp_entry is not None:
            _, fast_run = fp_entry
            _run_ps_fastpath(
                fast_run, q, k, v_sep, o,
                qo_indptr, kv_indptr, sm_scale,
                num_heads, num_CUs,
                work_indptr, work_info,
                dummy_f32,
            )
            _wrapper_counters["fast_hit"] += 1
        else:
            compiled = _PS_LAUNCH_FN(
                q, k, v_sep, o,
                qo_indptr, kv_indptr, sm_scale,
                num_heads, num_CUs, work_indptr, work_info,
                1.0, 1.0, return_compiled=fp_enabled,
            )
            if fp_enabled and compiled is not None and fp_key is not None:
                _install_fp_entry(fp_key, compiled, "ps")
                _wrapper_counters["slow_install"] += 1
    elif mode == "np":
        if fp_entry is not None:
            _, fast_run = fp_entry
            _run_np_fastpath(
                fast_run, q, k, v_sep, o,
                qo_indptr, kv_indptr, sm_scale,
                num_heads, n_m_tiles, batch_size,
                dummy_i32, dummy_f32,
            )
            _wrapper_counters["fast_hit"] += 1
        else:
            compiled = _NP_LAUNCH_FN(
                q, k, v_sep, o, qo_indptr, kv_indptr, sm_scale,
                num_heads, n_m_tiles, batch_size,
                1.0, 1.0, return_compiled=fp_enabled,
            )
            if fp_enabled and compiled is not None and fp_key is not None:
                _install_fp_entry(fp_key, compiled, "np")
                _wrapper_counters["slow_install"] += 1
    elif mode == "sk1":
        if fp_entry is not None:
            _, fast_run = fp_entry
            _run_sk1_fastpath(
                fast_run, q, k, v_sep, o,
                qo_indptr, kv_indptr, sm_scale,
                num_heads, n_m_tiles, num_CUs, batch_size,
                dummy_i32, dummy_f32,
            )
            _wrapper_counters["fast_hit"] += 1
        else:
            # Slow first call: go through _launch_splitk with SPLIT_K=1
            # (no workspace, no reduce needed). Capture CompiledKernel
            # for the fast-path cache.
            compiled, _, _, _ = _SK1_LAUNCH_FN(
                q, k, v_sep, o, qo_indptr, kv_indptr, sm_scale,
                num_heads, n_m_tiles, num_CUs, batch_size, 1,
                1.0, 1.0, return_compiled=fp_enabled,
            )
            if fp_enabled and compiled is not None and fp_key is not None:
                _install_fp_entry(fp_key, compiled, "sk1")
                _wrapper_counters["slow_install"] += 1
    else:
        # Defensive: _select_scheduler can only emit ps/np/sk1. Anything
        # else is a bug; fall back to NP which is correct for any shape.
        logger.warning(f"[gluon_mla] unknown sched mode {mode!r}; falling back to NP")
        mode = "np"
        compiled = _NP_LAUNCH_FN(
            q, k, v_sep, o, qo_indptr, kv_indptr, sm_scale,
            num_heads, n_m_tiles, batch_size,
            1.0, 1.0, return_compiled=False,
        )

    _wrapper_counters[mode] += 1

    if _SYNC_AFTER:
        # WORKAROUND probe: force a device-side sync after the Gluon kernel.
        torch.cuda.synchronize()

    # Post-kernel ops (fence modes used for debugging).
    _post_tokens = _FENCE_MODE.split("_")
    if "sync" in _post_tokens:
        torch.cuda.synchronize()
    if any(t in _post_tokens for t in ("cast", "d2h", "scan")):
        _fence_buf = o.float()  # noqa: F841  -- intentional forced read
        if "d2h" in _post_tokens:
            _ = _fence_buf.view(-1)[0].item()
        if "scan" in _post_tokens:
            _has_nan = torch.isnan(_fence_buf).any().item()
            _has_inf = torch.isinf(_fence_buf).any().item()
            if _has_nan or _has_inf:
                raise RuntimeError(
                    f"[gluon_mla_scan] NaN/Inf in o: "
                    f"total_q={total_q} H={num_heads}"
                )

    if _DEBUG:
        _DEBUG_CALL_COUNT += 1
        torch.cuda.synchronize()
        of = o.float()
        has_nan = torch.isnan(of).any().item()
        has_inf = torch.isinf(of).any().item()
        if has_nan or has_inf or _DEBUG_CALL_COUNT < 5 or _DEBUG_CALL_COUNT % 200 == 0:
            bs = qo_indptr.shape[0] - 1
            qo_cpu = qo_indptr.cpu().tolist() if qo_indptr.numel() < 20 else None
            kv_cpu = kv_indptr.cpu().tolist() if kv_indptr.numel() < 20 else None
            logger.warning(
                f"[gluon_mla_dbg] call={_DEBUG_CALL_COUNT} mode={mode} "
                f"total_q={total_q} H={num_heads} bs={bs} "
                f"q={tuple(q.shape)}/{q.dtype} k={tuple(k.shape)}/{k.dtype} "
                f"v={tuple(v.shape) if v is not None else None}/{v.dtype if v is not None else None} "
                f"qo={qo_cpu} kv={kv_cpu} "
                f"sm_scale={sm_scale} NaN={has_nan} Inf={has_inf} "
                f"o_min={of.min().item():.3g} o_max={of.max().item():.3g}"
            )
        if has_nan or has_inf:
            raise RuntimeError(
                f"[gluon_mla_dbg] Gluon produced NaN/Inf on call {_DEBUG_CALL_COUNT}: "
                f"mode={mode} total_q={total_q} H={num_heads} bs={bs}"
            )

    return o, mode
