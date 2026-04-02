# Gluon Extend-Attention Kernels for AMD MI350X (gfx950)

Optimized Triton/Gluon extend-attention kernels targeting AMD CDNA4 GPUs, integrated into SGLang's serving path.

## Supported Configurations

| Head Dim (Lq x Lv) | Kernel File | Models |
|---|---|---|
| 64 x 64 | `f16_kv_extend_attention_symmetric.py` | GPT-OSS 20B/120B |
| 128 x 128 | `f16_kv_extend_attention_symmetric.py` | Llama 3, Qwen 2/2.5 |
| 192 x 128 | `f16_kv_extend_attention_mixed.py` | DeepSeek V2 Lite |
| 576 x 512 | `f16_mla_prefill.py` | DeepSeek V3/R1 |

## File Layout

- `extend_attention_entrypoints.py` — Extend-attention dispatch entry points.
- `extend_attention_common.py` — Shared extend-attention inner loops.
- `f16_kv_extend_attention_symmetric.py` — f16 KV symmetric extend kernels.
- `fp8_kv_extend_attention_symmetric.py` — fp8 KV symmetric extend kernels.
- `f16_kv_extend_attention_mixed.py` — f16 KV mixed-dim extend kernels.
- `fp8_kv_extend_attention_mixed.py` — fp8 KV mixed-dim extend kernels.
- `f16_mla_prefill.py` — f16 MLA prefill kernels (D512 path).
- `fp8_mla_prefill.py` — fp8 MLA prefill kernels (D512 path).

## Quick Start — GPT-OSS on MI350X

### Prerequisites

- AMD MI350X (gfx950) GPU(s)
- Python venv with Triton (ROCm/gfx950), PyTorch (ROCm), and Gluon installed
- SGLang from this branch (`gluon-v3-port`)
- Dummy or real model weights (GPT-OSS 20B for TP=1, 120B for TP=2)

### Launching a Server with Gluon

```bash
# GPT-OSS 20B (TP=1, single GPU)
SGLANG_USE_GLUON_EXTEND=1 python -m sglang.launch_server \
    --model-path /path/to/gpt-oss-20b \
    --tp-size 1 \
    --trust-remote-code \
    --host 0.0.0.0 --port 9100 \
    --attention-backend triton \
    --disable-cuda-graph

# GPT-OSS 120B (TP=2, two GPUs)
SGLANG_USE_GLUON_EXTEND=1 python -m sglang.launch_server \
    --model-path /path/to/gpt-oss-120b \
    --tp-size 2 \
    --trust-remote-code \
    --host 0.0.0.0 --port 9200 \
    --attention-backend triton \
    --disable-cuda-graph
```

Key env vars:
- `SGLANG_USE_GLUON_EXTEND=1` — Routes extend-attention through Gluon (required).
- `SGLANG_GLUON_AUTO_SPLITK=1` — Enables auto split-K for tile-starved shapes (default off, recommended on).
- `AITER_ENABLE_GLUON_DEEPSEEK=1` — Enables Gluon for DeepSeek asymmetric dims (default on).

### Running Benchmarks

**Kernel-level correctness (1973 cases, D64):**
```bash
HIP_VISIBLE_DEVICES=0 python verify_correctness_d64.py
# Output: e2e_results/correctness_verify_d64.csv
```

**Kernel-level performance (1973 cases, D64, wall-clock):**
```bash
HIP_VISIBLE_DEVICES=0 python bench_dispatch_verify_d64.py
# Output: e2e_results/dispatch_verify_d64.csv
```

**E2E TTFT (server-level, requires running server):**
```bash
python bench_ttft_matrix.py --port 9100 --model gptoss-20b --backend gluon
# Output: e2e_results/gptoss-20b_gluon.csv, gptoss-20b_gluon_batched.csv
```

**3-way kernel comparison (Triton vs Gluon vs CK):**
```bash
HIP_VISIBLE_DEVICES=0 python bench_3way_d64.py
# Runs each backend in separate subprocesses for isolation
```

## Performance Summary (D64, GPT-OSS, MI350X)

### E2E TTFT (Gluon vs Triton, geomean speedup)

| Model | B=1 Cold TTFT | Batched | Overall |
|---|---|---|---|
| GPT-OSS 20B (TP=1) | **1.043x** | 0.946x | 0.995x |
| GPT-OSS 120B (TP=2) | **1.278x** | **1.410x** | **1.345x** |

### Kernel-level (1973 cases, wall-clock including dispatch)

| Dispatch Path | N | Geomean | Wins / Ties / Losses |
|---|---|---|---|
| WCA/persistent | 299 | **1.749x** | 299 / 0 / 0 |
| basic_BM256_8w | 712 | **2.437x** | 712 / 0 / 0 |
| basic_BM128_8w | 962 | 1.059x | 391 / 22 / 549 |
| **Overall** | **1973** | **1.543x** | **1402 / 22 / 549** |

Losses are concentrated in the tiny-extend regime (ext <= 32); for ext >= 129, win rate is 99.4%.

### Correctness (1973 cases, D64 bf16)

- 100% cases have cosine similarity >= 0.99 vs Triton
- 95.8% pass strict threshold (max_abs < 0.05)
- Zero NaN/Inf, zero runtime errors

## Dispatch Logic

The dispatch in `extend_attention_entrypoints.py` selects:

1. **WCA/persistent** — When tiles < CUs and extend >= 128 (tile-starved regime)
2. **basic_BM256_8w** — Large extends (>= 512) or high total_ext
3. **basic_BM128_8w** — Everything else (small/medium extends)

For DeepSeek asymmetric dims, the dispatch uses BM128/8w/2s for large shapes and BM64/4w/2s for small shapes.
