#!/usr/bin/env python3
"""Run prompt-quality checks across AITER probe backends.

This script launches the AITER backend and toggles probe routing:
- aiter_native
- triton_probe
- gluon_probe

It sends a fixed prompt set and writes full outputs to JSON so we can compare
looping/repetition behavior across backends.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

DEFAULT_REPO = str(Path(__file__).resolve().parents[1])

DEFAULT_PROMPTS = [
    ("q1_2plus2", "What is 2+2? Give only the number.", 128),
    (
        "q2_robe",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?",
        256,
    ),
    ("q3_sky_2sent", "Explain why the sky is blue in 2 sentences.", 1000),
]


@dataclass(frozen=True)
class BackendCase:
    name: str
    probe_extend: str | None
    probe_decode: str | None
    port: int


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_healthy(port: int, timeout_s: int = 3) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def _stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=35)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _base_env(repo: str, persist: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_HOME"] = env.get("ROCM_HOME", "/opt/rocm")
    env["PATH"] = f"/opt/rocm/bin:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{repo}/python"
    if persist:
        env["SGLANG_AITER_MLA_PERSIST"] = "1"
    else:
        env.pop("SGLANG_AITER_MLA_PERSIST", None)

    for key in [
        "SGLANG_AITER_MLA_PROBE_BACKEND",
        "SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND",
        "SGLANG_AITER_MLA_PROBE_DECODE_BACKEND",
        "SGLANG_AITER_MLA_TRITON_KERNEL_PROBE",
        "SGLANG_AITER_MLA_TRITON_PROBE_EXTEND",
        "SGLANG_AITER_MLA_TRITON_PROBE_DECODE",
        "SGLANG_FORCE_SAVE_KV_CACHE",
    ]:
        env.pop(key, None)
    return env


def _build_cmd(model_path: str, port: int, kv_cache_dtype: str, spec: bool) -> list[str]:
    cmd = [
        "/home/tussingh/venv-triton/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--tp-size",
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
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
    ]
    if spec:
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
    return cmd


def _wait_for_health(
    proc: subprocess.Popen,
    port: int,
    timeout_s: int,
    status_every_s: int,
    case_name: str,
) -> tuple[bool, str]:
    t0 = time.time()
    next_status_t = t0
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False, f"server exited code={proc.returncode}"
        now = time.time()
        if now >= next_status_t:
            elapsed = int(now - t0)
            print(
                f"[HEALTH] case={case_name} waiting ({elapsed}s/{timeout_s}s)",
                flush=True,
            )
            next_status_t = now + max(1, status_every_s)
        if _is_healthy(port):
            return True, "ok"
        time.sleep(1)
    return False, f"health timeout after {timeout_s}s"


def _run_prompt(port: int, model_path: str, prompt: str, max_tokens: int, timeout_s: int) -> dict[str, Any]:
    payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload,
            timeout=timeout_s,
        )
        row: dict[str, Any] = {"status_code": resp.status_code, "max_tokens": max_tokens}
        if resp.status_code != 200:
            row["error_body"] = resp.text[:1200]
            return row
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})
        usage = data.get("usage", {})
        row["finish_reason"] = choice.get("finish_reason")
        row["completion_tokens"] = usage.get("completion_tokens")
        row["content"] = msg.get("content")
        row["reasoning_content"] = msg.get("reasoning_content")
        return row
    except Exception as exc:  # noqa: BLE001
        return {"request_error": str(exc), "max_tokens": max_tokens}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--base-port", type=int, default=31720)
    parser.add_argument("--health-timeout-s", type=int, default=320)
    parser.add_argument("--health-status-every-s", type=int, default=10)
    parser.add_argument("--request-timeout-s", type=int, default=220)
    parser.add_argument(
        "--max-tokens-all",
        type=int,
        default=None,
        help="If set, override max_tokens for all default prompts.",
    )
    parser.add_argument("--spec", action="store_true")
    parser.add_argument("--disable-aiter-persist", action="store_true")
    parser.add_argument(
        "--cases",
        default="aiter_native,triton_probe,gluon_probe",
        help="Comma-separated: aiter_native,triton_probe,gluon_probe",
    )
    parser.add_argument(
        "--output-json",
        default=f"{DEFAULT_REPO}/e2e_results/ctrl_flow_probe/runs/prompt_backend_matrix_{_now_stamp()}.json",
    )
    args = parser.parse_args()

    run_dir = Path(args.output_json).parent
    run_dir.mkdir(parents=True, exist_ok=True)

    all_cases: dict[str, BackendCase] = {
        "aiter_native": BackendCase(
            name="aiter_native",
            probe_extend=None,
            probe_decode=None,
            port=args.base_port,
        ),
        "triton_probe": BackendCase(
            name="triton_probe",
            probe_extend="triton",
            probe_decode="triton",
            port=args.base_port + 1,
        ),
        "gluon_probe": BackendCase(
            name="gluon_probe",
            probe_extend="gluon",
            probe_decode="gluon",
            port=args.base_port + 2,
        ),
    }

    selected_names = [x.strip() for x in args.cases.split(",") if x.strip()]
    selected = []
    for name in selected_names:
        if name not in all_cases:
            raise ValueError(f"Unknown case: {name}")
        selected.append(all_cases[name])
    prompts = (
        [(name, text, args.max_tokens_all) for name, text, _ in DEFAULT_PROMPTS]
        if args.max_tokens_all is not None
        else DEFAULT_PROMPTS
    )

    output_rows: list[dict[str, Any]] = []
    for case in selected:
        case_row: dict[str, Any] = {
            "case": case.name,
            "spec": args.spec,
            "kv_cache_dtype": args.kv_cache_dtype,
            "port": case.port,
        }
        print(f"\n[CASE] {case.name}", flush=True)
        env = _base_env(args.repo, persist=not args.disable_aiter_persist)
        if case.probe_extend is not None:
            env["SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND"] = case.probe_extend
        if case.probe_decode is not None:
            env["SGLANG_AITER_MLA_PROBE_DECODE_BACKEND"] = case.probe_decode

        cmd = _build_cmd(
            model_path=args.model_path,
            port=case.port,
            kv_cache_dtype=args.kv_cache_dtype,
            spec=args.spec,
        )
        case_row["cmd"] = " ".join(cmd)
        log_path = run_dir / f"{Path(args.output_json).stem}_{case.name}.server.log"
        case_row["server_log"] = str(log_path)

        proc: subprocess.Popen | None = None
        try:
            with log_path.open("w", encoding="utf-8") as lf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=args.repo,
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            case_row["server_pid"] = proc.pid
            ok, reason = _wait_for_health(
                proc=proc,
                port=case.port,
                timeout_s=args.health_timeout_s,
                status_every_s=args.health_status_every_s,
                case_name=case.name,
            )
            if not ok:
                case_row["status"] = "launch_failed"
                case_row["error"] = reason
                output_rows.append(case_row)
                continue

            case_row["status"] = "ok"
            rows = []
            for prompt_name, prompt_text, max_tokens in prompts:
                prompt_row = {
                    "prompt_name": prompt_name,
                    "prompt": prompt_text,
                }
                prompt_row.update(
                    _run_prompt(
                        port=case.port,
                        model_path=args.model_path,
                        prompt=prompt_text,
                        max_tokens=max_tokens,
                        timeout_s=args.request_timeout_s,
                    )
                )
                rows.append(prompt_row)
            case_row["outputs"] = rows
        finally:
            _stop_proc(proc)
        output_rows.append(case_row)

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(output_rows, indent=2), encoding="utf-8")
    print(f"\nWrote results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
