# Gluon attention kernels for MI350X — performance handoff

**Branches**

| purpose | sglang branch | gluon-kernels source branch |
|---|---|---|
| MLA prefill (D192) replacement for DeepSeek V3 / R1 | `tussingh/gluon-mla-prefill` | `tussingh/mla-d192-prefill` |
| Extend attention (BF16 + FP8-KV) | `tussingh/gluon-extend-attn` | `tussingh/extend-attention-experiments` |

Both branches are based on upstream `sgl-project/sglang:main` and vendor the Gluon kernel sources under `python/sglang/srt/layers/attention/gluon_ops/`, so the sglang wheel is self-contained — no external `gluon-kernels` checkout required at runtime.

**Docker image**

`docker/rocm-gluon.Dockerfile` layers either branch onto the reviewer-pinned base `rocm/sgl-dev:v0.5.10rc0-rocm700-mi35x-20260409` (ROCm 7.0.0, Triton 3.7, aiter 0.1.12, sgl-kernel for gfx950). Pass `--build-arg SGL_BRANCH=<branch>` to select which Gluon feature to ship.

**Runtime switches**

| env var | effect | default |
|---|---|---|
| `SGLANG_AITER_USE_GLUON_MLA_PREFILL` | replace `mla_prefill_ps_asm_fwd`+`mla_reduce_v1` with the Gluon FP8 D192 kernel | off |
| `SGLANG_AITER_GLUON_MLA_SCHED` | `ps` (persistent) / `np` (3D grid) / `sk1` (split-K=1 work-stealing) / `hybrid` | `hybrid` |
| `SGLANG_AITER_USE_GLUON_EXTEND` | replace the Triton extend-attention path | off |
| `SGLANG_AITER_FP8_PREFILL_ATTN` | enable FP8 prefill (prereq for MLA Gluon) | off |
| `SGLANG_AITER_MLA_PERSIST` | aiter's persistent metadata cache (orthogonal, keep on) | off |

All Gluon switches are opt-in. With them unset the image runs the pure aiter ASM / Triton baseline from the base container, so the same image is used for A/B comparison.

---

## MLA prefill (D192) — final benchmark

Captured **2026-04-19** on an 8× MI350X node (gfx950) inside `rocm/sgl-dev:v0.5.10rc0-rocm700-mi35x-20260409` (ROCm 7.0.0, Triton 3.7, aiter 0.1.12). TP8, DeepSeek-R1, mxfp4 weights, fp8_e4m3 KV cache.

### Server config

```
python3 -m sglang.launch_server \
  --model-path ${MODEL} --tp-size 8 --trust-remote-code \
  --attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 131072 \
  --mem-fraction-static 0.8 \
  --max-running-requests 64 \
  --disable-radix-cache \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num_draft_tokens 4
```

Env for the Gluon run adds `SGLANG_AITER_USE_GLUON_MLA_PREFILL=1 SGLANG_AITER_GLUON_MLA_SCHED=hybrid SGLANG_AITER_FP8_PREFILL_ATTN=1 SGLANG_AITER_MLA_PERSIST=1`. ASM run leaves the Gluon var unset.

### Client

```
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 0.0.0.0 --port 9000 \
  --dataset-name random \
  --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.8 \
  --num-prompts 64 --request-rate 16 --warmup-requests 4 \
  --max-concurrency 32
```

Eight sequential runs of the same 64-prompt batch, same seed, different `--output-file`. Run-1 is warmup (all backends pay JIT cost for the rest of the stack — extend attention, rope, LN, mxfp4 GEMM — even with our MLA prewarm, because SGLang JITs other Triton kernels on first batch).

### Headline (runs 2–8, n=7)

| metric | ASM | Gluon (hybrid, sk1 enabled) | Δ |
|---|---:|---:|---:|
| throughput (tok/s, mean) | 17072 | 17054 | **-0.1% (tie)** |
| TPOT (ms, mean) | 21.23 | **20.91** | **-1.5% (Gluon win)** |
| TTFT (ms, mean, n=7) | 520 | 570 | +9.6% (one outlier) |
| TTFT (ms, mean, n=6 drop outlier) | 517 | **508** | **-1.7% (tie)** |
| median ITL (ms) | 10.84 | 10.88 | +0.4% (tie) |

Both backends pin median inter-token latency at ~10.85 ms — the decoder (mxfp4 MoE GEMMs, sampling, rope, LN) dominates ITL on this workload, not MLA. On an MI355 with +9% clock headroom both numbers should drop proportionally.

### Raw per-run throughput (tok/s)

```
        run1*   run2   run3   run4   run5   run6   run7   run8
ASM     4825  16226  16454  17091  17864  16295  17573  18001
Gluon   5026  17327  16391  16360  17388  17225  17095  17592
```
(*) run 1 is JIT warmup for non-MLA Triton ops in SGLang.

### Dispatch trace (`SGLANG_GLUON_MLA_TRACE=1`, 2184 calls)

| mode | calls | % | dominant shape |
|---|---:|---:|---|
| PS | 1520 | 69.6% | bs=1, uniform long |
| NP | 632 | 28.9% | bs=1, max≤128 |
| sk1 | 32 | 1.4% | bs=4/8, long-tail mixed (`uniform_ratio<0.75`) |

sk1 fires rarely on this workload but contributes a measurable per-call win on the mixed shapes it claims (1.5-1.8× vs NP per microbench). It's the safety net for heterogeneous batches where PS's pre-baked work table and NP's 3D grid both underutilize the 256 CUs.

---

## Split-K scan (for the record)

We tested `SPLIT_K ∈ {1, 2, 4, 8}` per shape. `sk1` refers to `SPLIT_K=1` as a compile-time constant — the kernel still uses the split-K *body* (persistent strided-fetch schedule, `tile_idx += total_programs`), but the reduce path and workspace constant-fold away. The scheduling is what makes it fast, not KV splitting.

Microbench on representative shapes (ms, lower is better):

| shape | NP best | sk1 | sk2 | sk4 | sk8 |
|---|---:|---:|---:|---:|---:|
| [2048] | 0.108 | 0.116 | 0.146 | 0.164 | 0.197 |
| [8192] | 0.438 | 0.621 | 0.702 | 0.814 | 0.874 |
| [16384] | 1.574 | 2.324 | 2.583 | 2.276 | 2.335 |
| bs=8 mixed | 0.877 | **0.561** | 0.729 | 0.946 | — |

SPLIT_K > 1 loses on every MLA D192 shape tested because MLA already has `16 heads × batch × n_m_tiles` of inherent parallelism — for any non-trivial shape this saturates 256 CUs without KV splitting, and the BF16→FP32→BF16 reduce kernel is pure cost.

---

## Recovery vs the April-10 regression

Reviewer's April-10 image (`docker.io/library/sglang-gluon:latest`, MI355 MTP) vs current:

| metric vs ASM | April-10 gluon | Current gluon (this branch) | recovery |
|---|---:|---:|---:|
| throughput | 84.6% (-15.4%) | 99.9% (tie) | **+15.3%** |
| TTFT | 97.9% (-2.1%) | ~100% (tie) | — |
| ITL | 87.9% (+12%) | ~100% (tie) | — |

The April-10 throughput regression is fully closed. Root cause was the kernel not contiguous-izing KV before launch + ROCm version downgrade (7.2 → 7.0 in reviewer's rebuild). We fixed the former; the latter is the container's baseline, and both backends pay it equally, so apples-to-apples Gluon-vs-ASM on ROCm 7.0 is a tie.

---

## Correctness

- 38-shape differential sweep vs the `torch_gemm`-equivalent reference: max rel_err ≤ 3.22e-2 (FP8 quantization noise floor).
- Startup validation in the wrapper: before any fast-path `CompiledKernel` is reused for a *new* specialization key, a one-shot differential check against the Triton JIT call validates output to the same tolerance. On mismatch the fast-path entry is tombstoned and the JIT path is used for that shape thereafter.
- sk1 passes the same differential validator on startup.

---

## First-request TTFT (warmup)

Run 1 TTFT ≈ 8.7s on *both* backends. Our MLA prewarm closes the MLA JIT stall, but SGLang still JITs extend-attention, rope, LN and several mxfp4 GEMM shapes on the first live batch. That's orthogonal to MLA; it's why run-1 is excluded from steady-state numbers. The `tussingh/gluon-extend-attn` branch's `prewarm_for_model` hook already covers extend; a matching prewarm for the rope/LN path would shave warmup further.

---

## Testing methodology

1. **Workload**: `sglang.bench_serving --dataset-name random --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.8 --num-prompts 64 --request-rate 16 --warmup-requests 4 --max-concurrency 32`. Seed held fixed across backends.
2. **Sequence**: 8 consecutive runs per backend, drop run-1 (JIT), aggregate runs 2-8.
3. **Metrics**: prefer TPOT (mean) and throughput (mean); TTFT is noisy under HTTP jitter and MTP scheduling, report with and without outliers.
4. **Outlier policy**: one run (Gluon run-4, TTFT=941 ms) is marked as an outlier because its tail TTFT is >2σ from the rest of the run-2-to-run-8 distribution for that backend. Reporting both `n=7` (with outlier) and `n=6` (without).
5. **A/B harness**: same container, toggle env vars, restart server between backends. Hot `hipblaslt`/`rccl` caches are left alone — both backends benefit equally.
6. **Dispatch trace**: `SGLANG_GLUON_MLA_TRACE=1` emits per-call `(mode, bs, max_q, total_q, uniform_ratio)` so we can attribute perf to scheduling decisions.
7. **Microbench**: `bench_shape_overhead.py` runs the kernel standalone over a batch of synthetic shapes to isolate kernel time from SGLang wrapper / CUDA graph overhead.

---

## PR readiness

Ready to propose upstreaming `tussingh/gluon-mla-prefill` as an opt-in backend:

1. **Correctness**: 38/38 shape sweep at rel_err ≤ 3.22e-2 (FP8 noise floor); startup validation + tombstone protect the fast-path against future kernel drift.
2. **Throughput**: tie with ASM (-0.1%).
3. **TPOT**: slight Gluon win (-1.5%).
4. **TTFT**: tie with ASM (within 1.7% after outlier removal).
5. **First-request TTFT**: matches ASM (non-MLA JIT dominates both; prewarm has closed the MLA JIT gap).
6. **sk1 fallback**: long-tail mixed batches (~1-2% of traffic on this benchmark; higher on agentic / chat workloads with variable context length).
7. **Surface**: single env var (`SGLANG_AITER_USE_GLUON_MLA_PREFILL`) guards all changes; shape guard (D_QK=192 ∧ D_V=128) means non-DeepSeek MLA models take the existing ASM path unconditionally.

---

## Repro quick-start

```bash
docker build \
  --build-arg SGL_FORK=https://github.com/tussingh/sglang.git \
  --build-arg SGL_BRANCH=tussingh/gluon-mla-prefill \
  -t sglang-gluon-mla:rocm700-mi35x \
  -f docker/rocm-gluon.Dockerfile .

# ASM baseline
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --network=host --shm-size=32G \
  -v ${MODEL_DIR}:/models \
  -e SGLANG_AITER_FP8_PREFILL_ATTN=1 \
  -e SGLANG_AITER_MLA_PERSIST=1 \
  sglang-gluon-mla:rocm700-mi35x \
  bash launch_sglang.sh

# Gluon A/B (same image)
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --network=host --shm-size=32G \
  -v ${MODEL_DIR}:/models \
  -e SGLANG_AITER_USE_GLUON_MLA_PREFILL=1 \
  -e SGLANG_AITER_GLUON_MLA_SCHED=hybrid \
  -e SGLANG_AITER_FP8_PREFILL_ATTN=1 \
  -e SGLANG_AITER_MLA_PERSIST=1 \
  sglang-gluon-mla:rocm700-mi35x \
  bash launch_sglang.sh
```

Then drive it with the bench_serving invocation above.
