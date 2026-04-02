#!/usr/bin/env python3
"""Run GSM8K on pure AITER backend with liveness heartbeats.

This script intentionally clears all AITER probe env vars so the run is
"native AITER only" (no Triton/Gluon probe overrides).
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
    *,
    port: int,
    proc: subprocess.Popen | None,
    timeout_s: int,
    status_every_s: int,
    label: str,
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
    mode: str,
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
            f"mode={mode} n={num_questions} elapsed={elapsed}s "
            f"proc_alive={proc_alive} health={healthy}",
            flush=True,
        )
        if not proc_alive:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=31190)
    parser.add_argument("--num-questions", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument(
        "--mode",
        choices=["spec", "nospec"],
        default="spec",
        help="spec: original EAGLE-style launch, nospec: no speculative decode.",
    )
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-aiter-persist", action="store_true")
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
        help="How often to print eval heartbeat while GSM8K runs (0 disables).",
    )
    parser.add_argument(
        "--output-json",
        default="/home/tussingh/sglang_wca/e2e_results/ctrl_flow_probe/gsm8k_aiter_watchdog.json",
    )
    parser.add_argument(
        "--server-log",
        default="/home/tussingh/sglang_wca/e2e_results/ctrl_flow_probe/logs/gsm8k_aiter_watchdog.log",
    )
    args = parser.parse_args()

    out_path = Path(args.output_json)
    log_path = Path(args.server_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_PYTHON_PATH + ":" + env.get("PYTHONPATH", "")
    env["ROCM_HOME"] = env.get("ROCM_HOME", "/opt/rocm")
    env["PATH"] = f"/opt/rocm/bin:{env.get('PATH', '')}"
    if not args.disable_aiter_persist:
        env["SGLANG_AITER_MLA_PERSIST"] = "1"

    # Ensure native AITER path only (no probe overrides).
    for key in [
        "SGLANG_AITER_MLA_PROBE_BACKEND",
        "SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND",
        "SGLANG_AITER_MLA_PROBE_DECODE_BACKEND",
        "SGLANG_AITER_MLA_TRITON_KERNEL_PROBE",
        "SGLANG_AITER_MLA_TRITON_PROBE_EXTEND",
        "SGLANG_AITER_MLA_TRITON_PROBE_DECODE",
    ]:
        env.pop(key, None)

    cmd = [
        "/home/tussingh/venv-triton/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--tp",
        "8",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
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
        args.kv_cache_dtype,
    ]
    if args.mode == "spec":
        cmd += [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]
    if args.disable_overlap_schedule:
        cmd.append("--disable-overlap-schedule")
    if args.disable_cuda_graph:
        cmd.append("--disable-cuda-graph")

    print(
        f"[RUN] mode={args.mode} questions={args.num_questions} port={args.port} log={log_path}",
        flush=True,
    )
    print(f"[RUN] command={' '.join(cmd)}", flush=True)

    proc: subprocess.Popen | None = None
    result: dict = {
        "mode": args.mode,
        "num_questions": args.num_questions,
        "port": args.port,
        "server_log": str(log_path),
        "output_json": str(out_path),
    }
    try:
        with log_path.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

        healthy, reason = wait_health(
            port=args.port,
            proc=proc,
            timeout_s=args.health_timeout_s,
            status_every_s=args.health_status_every_s,
            label=f"aiter:{args.port}:{args.mode}",
        )
        if not healthy:
            result["status"] = "launch_failed"
            result["error"] = reason
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
            return

        eval_args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=args.num_questions,
            max_new_tokens=args.max_new_tokens,
            parallel=args.parallel,
            host="http://127.0.0.1",
            port=args.port,
            temperature=0.0,
        )

        hb_stop = threading.Event()
        hb_thread = None
        if args.eval_heartbeat_s > 0:
            hb_thread = threading.Thread(
                target=eval_heartbeat_loop,
                kwargs={
                    "stop_event": hb_stop,
                    "port": args.port,
                    "proc": proc,
                    "mode": args.mode,
                    "num_questions": args.num_questions,
                    "interval_s": args.eval_heartbeat_s,
                },
                daemon=True,
            )
            hb_thread.start()
        try:
            metrics = run_eval(eval_args)
        finally:
            if hb_thread is not None:
                hb_stop.set()
                hb_thread.join(timeout=5)

        result["status"] = "ok"
        result["metrics"] = metrics

        server_info = fetch_server_info(args.port)
        if server_info is not None:
            states = server_info.get("internal_states")
            if isinstance(states, list) and states:
                s0 = states[0] if isinstance(states[0], dict) else {}
                result["server_spec"] = {
                    "avg_spec_accept_length": s0.get("avg_spec_accept_length"),
                    "avg_spec_accept_tokens": s0.get("avg_spec_accept_tokens"),
                }

    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        stop_proc(proc)

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
