#!/usr/bin/env python3
"""Run GSM8K accuracy sweeps across AITER probe variants.

This launcher keeps the server stack identical (AITER backend) and only swaps
attention kernels via probe env vars:
  - aiter_native
  - triton_probe
  - gluon_probe
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

REPO_PYTHON_PATH = "/home/tussingh/sglang_wca/python"
if REPO_PYTHON_PATH not in sys.path:
    sys.path.insert(0, REPO_PYTHON_PATH)

from sglang.test.few_shot_gsm8k import run_eval  # pylint: disable=import-error


def is_server_healthy(port: int, timeout_s: int = 3) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_health(
    port: int,
    proc: subprocess.Popen | None,
    timeout_s: int = 300,
    status_every_s: int = 10,
    label: str = "server",
) -> tuple[bool, str]:
    t0 = time.time()
    next_status_print_t = t0
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False, f"server process exited with code {proc.returncode}"
        if time.time() >= next_status_print_t:
            elapsed = int(time.time() - t0)
            print(
                f"[HEALTH] {label}: waiting for /health "
                f"({elapsed}s/{timeout_s}s elapsed)",
                flush=True,
            )
            next_status_print_t = time.time() + max(1, status_every_s)
        if is_server_healthy(port=port, timeout_s=3):
            elapsed = time.time() - t0
            print(f"[HEALTH] {label}: ready in {elapsed:.1f}s", flush=True)
            return True, "ok"
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


def parse_csv_ints(raw: str) -> list[int]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("No counts provided")
    return out


def fetch_server_info(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/server_info", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def eval_heartbeat_loop(
    *,
    stop_event: threading.Event,
    port: int,
    proc: subprocess.Popen | None,
    variant: str,
    num_questions: int,
    interval_s: int,
) -> None:
    t0 = time.time()
    while not stop_event.wait(timeout=max(1, interval_s)):
        elapsed = int(time.time() - t0)
        proc_alive = proc is None or proc.poll() is None
        healthy = is_server_healthy(port=port, timeout_s=3) if proc_alive else False
        print(
            "[EVAL] "
            f"variant={variant} n={num_questions} elapsed={elapsed}s "
            f"proc_alive={proc_alive} health={healthy}",
            flush=True,
        )
        if not proc_alive:
            return


def variant_probe_env(variant: str) -> dict[str, str]:
    if variant == "aiter_native":
        return {
            "SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND": "aiter",
            "SGLANG_AITER_MLA_PROBE_DECODE_BACKEND": "aiter",
        }
    if variant == "triton_probe":
        return {
            "SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND": "triton",
            "SGLANG_AITER_MLA_PROBE_DECODE_BACKEND": "triton",
        }
    if variant == "gluon_probe":
        return {
            "SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND": "gluon",
            "SGLANG_AITER_MLA_PROBE_DECODE_BACKEND": "aiter",
        }
    raise ValueError(f"Unknown variant: {variant}")


def run_one(
    *,
    model_path: str,
    port: int,
    variant: str,
    num_questions: int,
    max_new_tokens: int,
    parallel: int,
    kv_cache_dtype: str,
    disable_overlap_schedule: bool,
    disable_cuda_graph: bool,
    health_timeout_s: int,
    health_status_every_s: int,
    eval_heartbeat_s: int,
    log_path: Path,
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_PYTHON_PATH + ":" + env.get("PYTHONPATH", "")
    env["SGLANG_AITER_MLA_PERSIST"] = "1"
    env["ROCM_HOME"] = env.get("ROCM_HOME", "/opt/rocm")
    env["PATH"] = f"/opt/rocm/bin:{env.get('PATH', '')}"
    env.update(variant_probe_env(variant))

    cmd = [
        "/home/tussingh/venv-triton/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp",
        "8",
        "--host",
        "127.0.0.1",
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
        "--attention-backend",
        "aiter",
        "--kv-cache-dtype",
        kv_cache_dtype,
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
    if disable_overlap_schedule:
        cmd.append("--disable-overlap-schedule")
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")

    proc: subprocess.Popen | None = None
    result: dict = {
        "variant": variant,
        "num_questions": num_questions,
        "port": port,
        "log": str(log_path),
    }
    try:
        with log_path.open("w", encoding="utf-8") as lf:
            print(
                f"[RUN] variant={variant} questions={num_questions} port={port} log={log_path}",
                flush=True,
            )
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

        healthy, reason = wait_health(
            port=port,
            proc=proc,
            timeout_s=health_timeout_s,
            status_every_s=health_status_every_s,
            label=f"{variant}:{port}",
        )
        if not healthy:
            result["status"] = "launch_failed"
            result["error"] = reason
            return result

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=num_questions,
            max_new_tokens=max_new_tokens,
            parallel=parallel,
            host="http://127.0.0.1",
            port=port,
            temperature=0.0,
        )
        hb_stop = threading.Event()
        hb_thread = None
        if eval_heartbeat_s > 0:
            hb_thread = threading.Thread(
                target=eval_heartbeat_loop,
                kwargs={
                    "stop_event": hb_stop,
                    "port": port,
                    "proc": proc,
                    "variant": variant,
                    "num_questions": num_questions,
                    "interval_s": eval_heartbeat_s,
                },
                daemon=True,
            )
            hb_thread.start()
        try:
            metrics = run_eval(args)
        finally:
            if hb_thread is not None:
                hb_stop.set()
                hb_thread.join(timeout=5)
        result["status"] = "ok"
        result["metrics"] = metrics

        server_info = fetch_server_info(port)
        if server_info is not None:
            states = server_info.get("internal_states")
            if isinstance(states, list) and states:
                s0 = states[0] if isinstance(states[0], dict) else {}
                result["server_spec"] = {
                    "avg_spec_accept_length": s0.get("avg_spec_accept_length"),
                    "avg_spec_accept_tokens": s0.get("avg_spec_accept_tokens"),
                }
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = str(e)
    finally:
        stop_proc(proc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--counts",
        default="50,100",
        help="Comma-separated GSM8K question counts per variant.",
    )
    parser.add_argument(
        "--variants",
        default="aiter_native,triton_probe,gluon_probe",
        help="Comma-separated variants to run.",
    )
    parser.add_argument(
        "--output-json",
        default="/home/tussingh/sglang_wca/e2e_results/ctrl_flow_probe/gsm8k_probe_sweep.json",
    )
    parser.add_argument("--base-port", type=int, default=31120)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--health-timeout-s",
        type=int,
        default=300,
        help="Server launch health timeout in seconds.",
    )
    parser.add_argument(
        "--health-status-every-s",
        type=int,
        default=10,
        help="How often to print launch health wait status.",
    )
    parser.add_argument(
        "--eval-heartbeat-s",
        type=int,
        default=20,
        help="How often to print eval heartbeat while GSM8K is running (0 disables).",
    )
    args = parser.parse_args()

    counts = parse_csv_ints(args.counts)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants:
        raise ValueError("No variants provided")

    out_path = Path(args.output_json)
    log_dir = out_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    port = args.base_port
    for num_questions in counts:
        for variant in variants:
            run = run_one(
                model_path=args.model_path,
                port=port,
                variant=variant,
                num_questions=num_questions,
                max_new_tokens=args.max_new_tokens,
                parallel=args.parallel,
                kv_cache_dtype=args.kv_cache_dtype,
                disable_overlap_schedule=args.disable_overlap_schedule,
                disable_cuda_graph=args.disable_cuda_graph,
                health_timeout_s=args.health_timeout_s,
                health_status_every_s=args.health_status_every_s,
                eval_heartbeat_s=args.eval_heartbeat_s,
                log_path=log_dir / f"gsm8k_{variant}_{num_questions}.log",
            )
            runs.append(run)
            port += 1

    out = {
        "model_path": args.model_path,
        "counts": counts,
        "variants": variants,
        "kv_cache_dtype": args.kv_cache_dtype,
        "runs": runs,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    for r in runs:
        status = r.get("status")
        acc = (r.get("metrics") or {}).get("accuracy")
        print(
            f"variant={r.get('variant')} n={r.get('num_questions')} status={status} accuracy={acc}",
            flush=True,
        )


if __name__ == "__main__":
    main()
