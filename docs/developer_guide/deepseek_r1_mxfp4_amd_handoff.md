# DeepSeek-R1-MXFP4 AMD Handoff Notes

This document summarizes the upstream-ported branch state for debugging DeepSeek-R1-MXFP4 on MI350X (gfx950), focused on Aiter/Triton/Gluon attention behavior in speculative EAGLE mode.

## Scope of This Branch

The following changes from the local WCA workspace were ported onto upstream SGLang:

- Attention backends and routing:
  - `python/sglang/srt/layers/attention/attention_registry.py`
  - `python/sglang/srt/layers/attention/triton_backend.py`
  - `python/sglang/srt/layers/attention/aiter_backend.py`
  - `python/sglang/srt/server_args.py`
- DeepSeek MLA forward path:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- Gluon ops package:
  - `python/sglang/srt/layers/attention/gluon_ops/*`
  - Includes FP16 + FP8 symmetric and mixed-dim kernels, D512 wrappers, and dispatch wrapper.
- Runtime stability fixes previously needed on this machine:
  - `python/sglang/jit_kernel/clamp_position.py` (native HIP fallback when `hipcc` unavailable)
  - `python/sglang/jit_kernel/resolve_future_token_ids.py` (native HIP fallback when `hipcc` unavailable)
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` (`e8m0_shuffle` import safety on HIP)
- Repro tooling:
  - `scripts/compare_aiter_triton_token_divergence.py`
  - `scripts/gsm8k_probe_sweep.py`
  - `scripts/gsm8k_aiter_watchdog.py`
  - `scripts/prompt_backend_quality_matrix.py`
  - `scripts/aiter_native_matrix_watchdog.py`
  - `scripts/start_aiter_matrix_watchdog.sh`

## Kernel Correctness Verification (Current Dispatch Paths)

Executed with:

```bash
PYTHONPATH=/home/tussingh/sglang_upstream_clean/python \
/home/tussingh/venv-triton/bin/python -m unittest -v \
  python.sglang.test.attention.test_gluon_attention_kernels.TestGluonExtendKernelAccuracy.test_fp16_d192_v128_deepseek \
  python.sglang.test.attention.test_gluon_attention_kernels.TestGluonExtendKernelAccuracy.test_fp8_kv_d192_v128_deepseek \
  python.sglang.test.attention.test_gluon_attention_kernels.TestGluonExtendKernelAccuracy.test_native_fp8_kv_cache_d192_v128_deepseek \
  python.sglang.test.attention.test_gluon_mla_d512_kernels.TestGluonMLAD512KernelAccuracy.test_mla_basic_fp16 \
  python.sglang.test.attention.test_gluon_mla_d512_kernels.TestGluonMLAD512KernelAccuracy.test_mla_basic_fp8_kv
```

Result:

- `D192/Lv128` wrapper path (`gluon_extend_attention_fwd`) passed:
  - fp16
  - fp8-style KV
  - native fp8 KV cache dtype
- `D576/Lv512` dedicated D512 wrappers passed:
  - fp16 (`CDNA4/f16_mla_prefill.py`)
  - fp8 (`CDNA4/fp8_mla_prefill.py`)

## Non-DeepSeek Symmetric Kernels (D64/D128) Verification

Executed with:

```bash
PYTHONPATH=/home/tussingh/sglang_upstream_clean/python \
/home/tussingh/venv-triton/bin/python -m unittest -v \
  python.sglang.test.attention.test_gluon_attention_kernels.TestGluonExtendKernelAccuracy.test_fp16_d64 \
  python.sglang.test.attention.test_gluon_attention_kernels.TestGluonExtendKernelAccuracy.test_fp16_d128 \
  python.sglang.test.attention.test_gluon_attention_kernels.TestGluonExtendKernelAccuracy.test_fp8_kv_d64 \
  python.sglang.test.attention.test_gluon_attention_kernels.TestGluonExtendKernelAccuracy.test_fp8_kv_d128 \
  python.sglang.test.attention.test_gluon_attention_kernels.TestGluonExtendKernelAccuracy.test_native_fp8_kv_cache_d64 \
  python.sglang.test.attention.test_gluon_attention_kernels.TestGluonExtendKernelAccuracy.test_native_fp8_kv_cache_d128 \
  python.sglang.test.attention.test_gluon_backend_routing.TestGluonKernelRouting.test_symmetric_kernel_used_for_supported_equal_dims \
  python.sglang.test.attention.test_gluon_backend_routing.TestGluonKernelRouting.test_ragged_d128_routes_to_persistent \
  python.sglang.test.attention.test_gluon_backend_routing.TestGluonKernelRouting.test_splitk_override_routes_to_splitk_path \
  python.sglang.test.attention.test_gluon_backend_routing.TestGluonKernelRouting.test_v_scale_propagates_to_kernel_kwargs \
  python.sglang.test.attention.test_gluon_backend_routing.TestGluonKernelRouting.test_symmetric_fp8_routes_to_symmetric_fp8_kernel \
  python.sglang.test.attention.test_gluon_backend_routing.TestGluonKernelRouting.test_symmetric_fp8_bridge_forces_bf16_kernel
```

Result:

- Numerical correctness:
  - `D64/Lv64`: fp16, fp8-style KV, native fp8 KV cache all passed.
  - `D128/Lv128`: fp16, fp8-style KV, native fp8 KV cache all passed.
- Additional bf16 KV sanity (single-run metrics):
  - `D64/Lv64`: `cos=0.999998`, `max_abs=0.004079`
  - `D64/Lv64` native fp8 KV: `cos=0.999464`, `max_abs=0.045566`
  - `D128/Lv128`: `cos=0.999998`, `max_abs=0.001788`
  - `D128/Lv128` native fp8 KV: `cos=0.999410`, `max_abs=0.025242`
- Expected dispatch/routing behavior:
  - Symmetric kernel selection for equal dims passed.
  - Ragged `D128` -> persistent path routing passed.
  - Split-K override routing (`_force_use_splitk`) passed.
  - `v_scale` propagation into symmetric kernel kwargs passed.
  - Symmetric fp8 routing and fp8->bf16 bridge behavior passed.

## End-to-End Behavior Snapshot

Latest prompt-matrix status on this branch:

- Strict 1k prompt coverage across `(spec|nospec) x (aiter_native|triton_probe|gluon_probe)` is now **5/6**.
  - Successful 1k sets:
    - `nospec + aiter_native`
    - `nospec + triton_probe`
    - `nospec + gluon_probe`
    - `spec + aiter_native`
    - `spec + gluon_probe`
  - Still blocked:
    - `spec + triton_probe`
- Native Aiter path now runs reliably in both nospec/spec prompt sweeps, but long outputs still show repetition/looping degradation.
- Gluon probe path also runs in both nospec/spec 1k sweeps, but quality remains degenerate on long outputs.
- Triton probe path is stable enough in non-spec runs for output collection, but speculative EAGLE remains unstable.
- GSM8K on this model/backend combination remains significantly below expected target; issue is not explained by FP8 KV alone (similar degradation seen in BF16 KV tests).

## Runtime Unblock Patch (Apr 2026)

To get the shareable upstream branch running end-to-end again (at least in non-spec mode), the following changes were applied:

- `python/sglang/srt/models/deepseek_common/utils.py`
  - Restored MI350X/gfx95 behavior:
    - `_use_aiter_gfx95 = _is_hip and _is_gfx95_supported`
  - This avoids relying on global `SGLANG_USE_AITER` env for gfx95-specific DeepSeek paths.
- `python/sglang/srt/layers/attention/aiter_backend.py` (temporary diagnostic guards)
  - Extend probe guard: if target-verify `q` shape is incompatible for probe reshape, force that call to native Aiter extend.
  - Decode guard: if decode `q` shape is incompatible, return a zero tensor for that call instead of crashing.
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` (probe safety experiments)
  - Added triton-probe detection helper and probe-only fallbacks to avoid known crashing paths while debugging spec launch failures.
  - These changes are diagnostic and do not yet make `spec + triton_probe` stable.

### Why this was needed

- Before the gfx95 restore, clean branch crashed in non-spec decode with:
  - `RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [16, 512] but got: [16, 256]`
  - Source: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` (`torch.bmm` on `attn_output` x `w_vc`).
- After restoring `_use_aiter_gfx95`, non-spec prompt matrix runs complete again.

### Current Repro Artifacts

- **Non-spec, 3-backend, 1k prompts (all successful):**
  - JSON: `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_000652_nospec_1k_bundle_after_gfx95fix.json`
  - Logs (gzip-compressed for shareable branch size limits):
    - `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_000652_nospec_1k_bundle_after_gfx95fix_aiter_native.server.log.gz`
    - `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_000652_nospec_1k_bundle_after_gfx95fix_triton_probe.server.log.gz`
    - `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_000652_nospec_1k_bundle_after_gfx95fix_gluon_probe.server.log.gz`
- **Spec, 1k prompts, Aiter native (successful):**
  - JSON: `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_211402_spec_1k_aiter_gluon_after_gfx95fix.json`
  - Log: `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_211402_spec_1k_aiter_gluon_after_gfx95fix_aiter_native.server.log`
- **Spec, 1k prompts, Gluon probe (successful):**
  - JSON: `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_211926_spec_1k_gluon_only_after_triton_guard.json`
  - Log: `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_211926_spec_1k_gluon_only_after_triton_guard_gluon_probe.server.log`
- **Spec, Triton probe (still blocked in smoke runs):**
  - `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_212240_spec_triton_smoke128_after_triton_guard.json`
  - `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_212358_spec_triton_smoke128_core_guard_only.json`
  - `e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_20260402_212533_spec_triton_smoke128_zero_fallback.json`

### Is the temporary guard causing the garbage outputs?

Evidence so far says **no** for successful non-spec and spec (aiter/gluon) runs:

- The guard markers
  - `probe_extend_shape_guard_fallback_aiter`
  - `mla_decode_shape_guard_zero_fallback`
  were introduced for crash prevention and not as a quality fix.
- Repetition/degeneration remains visible even in runs that do not depend on those fallback paths.

Interpretation:

- The guard is a crash-prevention shim for bad-shape edge cases.
- The observed low-quality looping appears to come from underlying model/backend behavior, not from these guards.

### Spec Triton Probe Failure Chain (Current)

In latest speculative triton-probe smoke attempts, failures remain launch-time and include:

- `RuntimeError` shape mismatch in DeepSeek MLA fallback paths:
  - `[16, 128] vs [16, 64]` in `forward_absorb_prepare`
  - `[16, 512] vs [16, 256]` in `forward_absorb_core`
- HIP illegal memory access during verify/extend path, later surfacing in rotary (`rotate_gptj`) and/or Triton/Aiter quant kernels (`fused_rms_mxfp4_quant`).

This indicates cascading incompatibilities in the speculative triton-probe path; a single guard is insufficient.

## Known Caveats

- The generic Gluon extend wrapper is stable for D192 mixed-dim routes used by current probe policy.
- D512 production probe route currently uses dedicated D512 wrappers selected in `aiter_backend.py` (not the generic mixed-dim extend kernel path).
- D128 experimental import is guarded to avoid hard initialization failures when Triton JIT source inspection is unavailable.

## Suggested Next Maintainer Steps

1. Instrument end-to-end KV write/read invariants for DeepSeek-R1 speculative flow (`attn_mqa`) across Aiter native vs probe routes.
2. Validate per-step token agreement against a known-good backend before long-horizon quality eval.
3. Stabilize `spec + triton_probe` first (shape contract + fused quant/rotary safety in draft/verify flow), then regenerate a full 1k 6/6 bundle.
4. Re-run GSM8K subset (`50/100`) with strict environment sanitation and archived server logs.
5. Keep D192 wrapper and D512 dedicated wrapper correctness tests in CI on HIP/gfx950.

