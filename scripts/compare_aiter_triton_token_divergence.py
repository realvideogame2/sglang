#!/usr/bin/env python3
"""Compare token-by-token divergence between AITER and Triton backends."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer


def wait_health(
    port: int, proc: subprocess.Popen | None = None, timeout_s: int = 240
) -> tuple[bool, str]:
    t0 = time.time()
    last_report = 0.0
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False, f"server process exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True, "ok"
        except Exception:
            pass
        elapsed = time.time() - t0
        if elapsed - last_report >= 15:
            print(f"  waiting for /health on port {port} ({elapsed:.0f}s)...", flush=True)
            last_report = elapsed
        time.sleep(1)
    return False, f"health timeout after {timeout_s}s"


def stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


def generate(port: int, input_ids: list[int], max_new_tokens: int) -> dict:
    payload = json.dumps(
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
                "ignore_eos": False,
            },
            "stream": True,
            "log_metrics": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    out_ids = []
    meta = {}
    with urllib.request.urlopen(req, timeout=300) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if not data or data == b"[DONE]":
                continue
            obj = json.loads(data.decode("utf-8"))
            if "error" in obj:
                return {"status": "error", "error": str(obj["error"])}
            if isinstance(obj.get("meta_info"), dict):
                meta = obj["meta_info"]
            if obj.get("output_ids"):
                out_ids = obj["output_ids"]
    if not out_ids:
        return {"status": "error", "error": "no_output_ids"}
    return {"status": "ok", "output_ids": out_ids, "meta_info": meta}


def run_one(
    backend: str,
    model_path: str,
    port: int,
    speculative: bool,
    kv_cache_dtype: str,
    disable_cuda_graph: bool,
    disable_overlap_schedule: bool,
    force_use_aiter_env: bool,
    triton_shadow_max_calls: int,
    triton_shadow_layer: int,
    prompt_ids: list[int],
    tokenizer: AutoTokenizer,
    log_path: Path,
    max_new_tokens: int,
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/tussingh/sglang_wca/python:" + env.get("PYTHONPATH", "")
    env["SGLANG_DISABLE_AITER_LAYERNORM"] = "1"
    env["HIP_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    if backend == "aiter" or force_use_aiter_env:
        env["SGLANG_USE_AITER"] = "1"
    if backend == "triton":
        env["SGLANG_DEBUG_MLA_RECON_COMPARE"] = "1"
        env["SGLANG_DEBUG_TRITON_KV_STATE"] = "1"
        env["SGLANG_DEBUG_TRITON_SHADOW_AITER"] = "1"
        env["SGLANG_DEBUG_TRITON_SHADOW_MAX_CALLS"] = str(triton_shadow_max_calls)
        if triton_shadow_layer >= 0:
            env["SGLANG_DEBUG_TRITON_SHADOW_LAYER"] = str(triton_shadow_layer)
        env["SGLANG_DEBUG_TRITON_ASSERTS"] = "1"

    cmd = [
        "/home/tussingh/venv-triton/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        "8",
        "--port",
        str(port),
        "--trust-remote-code",
        "--chunked-prefill-size",
        "131072",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "64",
        "--max-total-tokens",
        "131072",
        "--attention-backend",
        backend,
        "--kv-cache-dtype",
        kv_cache_dtype,
    ]
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")
    if disable_overlap_schedule:
        cmd.append("--disable-overlap-schedule")
    if speculative:
        cmd.extend(
            [
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--speculative-attention-mode",
                "decode",
                "--speculative-draft-attention-backend",
                "triton",
            ]
        )

    proc = None
    result = {"backend": backend, "speculative": speculative, "log": str(log_path)}
    try:
        with log_path.open("w", encoding="utf-8") as lf:
            print(
                f"Launching backend={backend} speculative={speculative} on port {port} ...",
                flush=True,
            )
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        healthy, reason = wait_health(port=port, proc=proc)
        if not healthy:
            result["status"] = "launch_failed"
            result["error"] = reason
            return result
        print(f"Server healthy for backend={backend}. Generating...", flush=True)
        out = generate(port=port, input_ids=prompt_ids, max_new_tokens=max_new_tokens)
        result.update(out)
        if out.get("status") == "ok":
            result["text"] = tokenizer.decode(out["output_ids"], skip_special_tokens=True)
    finally:
        stop_proc(proc)
    return result


def compare_tokens(a: list[int], b: list[int]) -> dict[str, int | None]:
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return {"first_divergence_idx": i}
    if len(a) != len(b):
        return {"first_divergence_idx": m}
    return {"first_divergence_idx": None}


def parse_backends(raw: str) -> list[str]:
    backends = [b.strip() for b in raw.split(",") if b.strip()]
    if not backends:
        raise ValueError("No backends provided")
    allowed = {"aiter", "triton"}
    invalid = [b for b in backends if b not in allowed]
    if invalid:
        raise ValueError(f"Unsupported backend(s): {invalid}. Allowed: {sorted(allowed)}")
    return backends


def successful_runs_by_backend(runs: list[dict]) -> dict[str, dict]:
    return {
        r["backend"]: r
        for r in runs
        if isinstance(r, dict) and r.get("status") == "ok" and "backend" in r
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/tussingh/DeepSeek-R1-MXFP4")
    parser.add_argument("--base-port", type=int, default=31020)
    parser.add_argument(
        "--output-json",
        default="/home/tussingh/sglang_wca/e2e_results/aiter_triton_token_divergence.json",
    )
    parser.add_argument(
        "--prompt",
        default="What is 19 + 23? Answer with only the number.",
    )
    parser.add_argument(
        "--backends",
        default="aiter,triton",
        help="Comma-separated backends to run (supported: aiter,triton).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--kv-cache-dtype",
        default="bfloat16",
        help="KV cache dtype passed to launch_server (e.g., bfloat16, fp8_e4m3).",
    )
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Use decode/cuda-graph path instead of forcing extend path.",
    )
    parser.add_argument(
        "--enable-overlap-schedule",
        action="store_true",
        help="Enable overlap scheduler; by default this script disables it for stability.",
    )
    parser.add_argument(
        "--force-use-aiter-env-for-triton",
        action="store_true",
        help="Set SGLANG_USE_AITER=1 even for triton backend to isolate non-attention stack drift.",
    )
    parser.add_argument(
        "--triton-shadow-max-calls",
        type=int,
        default=3,
        help="Max shadow compare calls per (stage,layer).",
    )
    parser.add_argument(
        "--triton-shadow-layer",
        type=int,
        default=-1,
        help="If >=0, only shadow-compare this layer.",
    )
    parser.add_argument(
        "--reuse-baseline-json",
        default="",
        help="Optional existing JSON output to reuse as token baseline.",
    )
    parser.add_argument(
        "--reuse-baseline-backend",
        default="aiter",
        choices=["aiter", "triton"],
        help="Backend from --reuse-baseline-json used as comparison baseline.",
    )
    args = parser.parse_args()

    out_json = Path(args.output_json)
    log_dir = out_json.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    backends = parse_backends(args.backends)

    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )

    runs = []
    for i, backend in enumerate(backends):
        runs.append(
            run_one(
                backend=backend,
                model_path=args.model_path,
                port=args.base_port + i,
                speculative=args.speculative,
                kv_cache_dtype=args.kv_cache_dtype,
                disable_cuda_graph=not args.enable_cuda_graph,
                disable_overlap_schedule=not args.enable_overlap_schedule,
                force_use_aiter_env=(
                    args.force_use_aiter_env_for_triton and backend == "triton"
                ),
                triton_shadow_max_calls=args.triton_shadow_max_calls,
                triton_shadow_layer=args.triton_shadow_layer,
                prompt_ids=prompt_ids,
                tokenizer=tok,
                log_path=log_dir / f"token_divergence_{backend}.log",
                max_new_tokens=args.max_new_tokens,
            )
        )

    out = {
        "prompt": args.prompt,
        "speculative": args.speculative,
        "max_new_tokens": args.max_new_tokens,
        "runs": runs,
    }
    run_map = successful_runs_by_backend(runs)
    token_compare_by_backend = {}
    if "aiter" in run_map and "triton" in run_map:
        token_compare_by_backend["aiter_vs_triton"] = compare_tokens(
            run_map["aiter"]["output_ids"], run_map["triton"]["output_ids"]
        )

    if args.reuse_baseline_json:
        baseline_path = Path(args.reuse_baseline_json)
        try:
            baseline_obj = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_runs = baseline_obj.get("runs", [])
            baseline_map = successful_runs_by_backend(baseline_runs)
            baseline_backend = args.reuse_baseline_backend
            baseline_run = baseline_map.get(baseline_backend)
            if baseline_run is None:
                print(
                    f"Warning: baseline backend={baseline_backend} missing/not-ok in {baseline_path}",
                    flush=True,
                )
            else:
                out["reused_baseline_json"] = str(baseline_path)
                out["reused_baseline_backend"] = baseline_backend
                baseline_ids = baseline_run["output_ids"]
                for backend, run in run_map.items():
                    if backend == baseline_backend:
                        continue
                    key = f"{baseline_backend}_vs_{backend}"
                    token_compare_by_backend[key] = compare_tokens(
                        baseline_ids, run["output_ids"]
                    )
        except Exception as e:
            print(f"Warning: failed to load --reuse-baseline-json: {e}", flush=True)

    if token_compare_by_backend:
        out["token_compare_by_backend"] = token_compare_by_backend
        if "aiter_vs_triton" in token_compare_by_backend:
            out["token_compare"] = token_compare_by_backend["aiter_vs_triton"]
        elif len(token_compare_by_backend) == 1:
            out["token_compare"] = next(iter(token_compare_by_backend.values()))

    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_json}", flush=True)
    for r in runs:
        print(f"{r['backend']} status={r.get('status')}", flush=True)
    if "token_compare_by_backend" in out:
        print(f"token_compare_by_backend={out['token_compare_by_backend']}", flush=True)


if __name__ == "__main__":
    main()
