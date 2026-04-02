#!/usr/bin/env python3
"""Run long native-AITER repo/model checks with durable status logs.

This runner is designed for long experiments:
- writes incremental status to status.json
- writes incremental results to results.json
- stores per-case server logs
- cleans up child server on SIGINT/SIGTERM

Typical use:
  /home/tussingh/venv-triton/bin/python scripts/aiter_native_matrix_watchdog.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

PROMPTS = [
    "Explain why the sky is blue in one short paragraph.",
    "What is 2+2? Give only the number.",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?",
]

DEFAULT_CASES = (
    "current_r1_spec",
    "clean_r1_spec",
    "current_v2_nospec",
    "clean_v2_nospec",
)

DEFAULT_REPO_CURRENT = "/home/tussingh/sglang_wca"
DEFAULT_REPO_CLEAN = "/home/tussingh/sglang_upstream_clean"
DEFAULT_MODEL_R1 = "/home/tussingh/DeepSeek-R1-MXFP4"
DEFAULT_MODEL_V2 = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"


@dataclass(frozen=True)
class CaseDef:
    name: str
    repo: str
    model: str
    spec: bool
    port: int


class RunState:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.status_path = run_dir / "status.json"
        self.results_path = run_dir / "results.json"
        self._status: dict[str, Any] = {}
        self._results: list[dict[str, Any]] = []

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    def set_status(self, **kwargs: Any) -> None:
        self._status.update(kwargs)
        self._status["updated_at"] = now_iso()
        self._atomic_write_json(self.status_path, self._status)

    def add_result(self, item: dict[str, Any]) -> None:
        self._results.append(item)
        self._atomic_write_json(self.results_path, self._results)

    def load_done_cases(self) -> set[str]:
        if not self.results_path.exists():
            return set()
        try:
            payload = json.loads(self.results_path.read_text(encoding="utf-8"))
        except Exception:
            return set()
        if not isinstance(payload, list):
            return set()
        done = set()
        for row in payload:
            if isinstance(row, dict) and isinstance(row.get("case"), str):
                done.add(row["case"])
        self._results = [x for x in payload if isinstance(x, dict)]
        return done


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_server_healthy(port: int, timeout_s: int = 3) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


def stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


def wait_health(
    *,
    port: int,
    proc: subprocess.Popen | None,
    timeout_s: int,
    status_every_s: int,
    state: RunState,
    case_name: str,
) -> tuple[bool, str]:
    t0 = time.time()
    next_status_t = t0
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False, f"server exited code={proc.returncode}"
        now = time.time()
        if now >= next_status_t:
            elapsed = int(now - t0)
            msg = f"[HEALTH] case={case_name} waiting ({elapsed}s/{timeout_s}s)"
            print(msg, flush=True)
            state.set_status(phase="wait_health", case=case_name, message=msg, elapsed_s=elapsed)
            next_status_t = now + max(1, status_every_s)
        if is_server_healthy(port):
            elapsed = time.time() - t0
            msg = f"[HEALTH] case={case_name} ready in {elapsed:.1f}s"
            print(msg, flush=True)
            state.set_status(phase="wait_health_done", case=case_name, message=msg, elapsed_s=elapsed)
            return True, "ok"
        time.sleep(1)
    return False, f"health timeout after {timeout_s}s"


def build_cases(args: argparse.Namespace) -> list[CaseDef]:
    all_cases = {
        "current_r1_spec": CaseDef(
            name="current_r1_spec",
            repo=args.current_repo,
            model=args.model_r1,
            spec=True,
            port=args.base_port,
        ),
        "clean_r1_spec": CaseDef(
            name="clean_r1_spec",
            repo=args.clean_repo,
            model=args.model_r1,
            spec=True,
            port=args.base_port + 1,
        ),
        "current_v2_nospec": CaseDef(
            name="current_v2_nospec",
            repo=args.current_repo,
            model=args.model_v2,
            spec=False,
            port=args.base_port + 2,
        ),
        "clean_v2_nospec": CaseDef(
            name="clean_v2_nospec",
            repo=args.clean_repo,
            model=args.model_v2,
            spec=False,
            port=args.base_port + 3,
        ),
    }
    out = []
    for name in args.cases.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in all_cases:
            raise ValueError(f"Unknown case: {name}")
        out.append(all_cases[name])
    return out


def launch_cmd(case: CaseDef) -> list[str]:
    cmd = [
        "/home/tussingh/venv-triton/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        case.model,
        "--tp-size",
        "8",
        "--host",
        "127.0.0.1",
        "--port",
        str(case.port),
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
        "fp8_e4m3",
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
    ]
    if case.spec:
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


def case_env(case: CaseDef, keep_aiter_persist: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_HOME"] = env.get("ROCM_HOME", "/opt/rocm")
    env["PATH"] = f"/opt/rocm/bin:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{case.repo}/python"
    if keep_aiter_persist:
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
    ]:
        env.pop(key, None)
    return env


def run_prompts(case: CaseDef, req_timeout_s: int) -> list[dict[str, Any]]:
    outputs = []
    for prompt in PROMPTS:
        payload = {
            "model": case.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        try:
            resp = requests.post(
                f"http://127.0.0.1:{case.port}/v1/chat/completions",
                json=payload,
                timeout=req_timeout_s,
            )
            if resp.status_code != 200:
                outputs.append(
                    {
                        "prompt": prompt,
                        "status": "http_error",
                        "code": resp.status_code,
                        "body": resp.text[:500],
                    }
                )
                continue
            msg = resp.json()["choices"][0]["message"]
            outputs.append(
                {
                    "prompt": prompt,
                    "status": "ok",
                    "content_preview": (msg.get("content") or "")[:500],
                    "reasoning_preview": (msg.get("reasoning_content") or "")[:300],
                }
            )
        except Exception as exc:  # noqa: BLE001
            outputs.append({"prompt": prompt, "status": "request_error", "error": str(exc)})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-repo", default=DEFAULT_REPO_CURRENT)
    parser.add_argument("--clean-repo", default=DEFAULT_REPO_CLEAN)
    parser.add_argument("--model-r1", default=DEFAULT_MODEL_R1)
    parser.add_argument("--model-v2", default=DEFAULT_MODEL_V2)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--base-port", type=int, default=31540)
    parser.add_argument("--health-timeout-s", type=int, default=420)
    parser.add_argument("--health-status-every-s", type=int, default=10)
    parser.add_argument("--request-timeout-s", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-aiter-persist", action="store_true")
    parser.add_argument(
        "--run-dir",
        default=f"/home/tussingh/sglang_wca/e2e_results/ctrl_flow_probe/runs/matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(run_dir=run_dir)
    state.set_status(
        phase="init",
        run_dir=str(run_dir),
        pid=os.getpid(),
        args=vars(args),
        started_at=now_iso(),
    )

    done_cases = state.load_done_cases() if args.resume else set()
    cases = build_cases(args)
    current_proc: subprocess.Popen | None = None

    def _handle_signal(signum: int, _frame: Any) -> None:
        msg = f"received signal {signum}, stopping child process"
        print(msg, flush=True)
        state.set_status(phase="signal", message=msg, signal=signum)
        stop_proc(current_proc)
        sys.exit(130)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    for case in cases:
        if case.name in done_cases:
            print(f"[SKIP] already done: {case.name}", flush=True)
            continue

        server_log = run_dir / f"{case.name}.server.log"
        cmd = launch_cmd(case)
        env = case_env(case, keep_aiter_persist=not args.disable_aiter_persist)
        row: dict[str, Any] = {
            "case": case.name,
            "repo": case.repo,
            "model": case.model,
            "spec": case.spec,
            "port": case.port,
            "command": " ".join(cmd),
            "server_log": str(server_log),
            "started_at": now_iso(),
        }

        print(f"\n[CASE] {case.name}", flush=True)
        state.set_status(
            phase="launch",
            case=case.name,
            message=f"launching {case.name}",
            server_log=str(server_log),
        )
        try:
            with server_log.open("w", encoding="utf-8") as lf:
                current_proc = subprocess.Popen(
                    cmd,
                    cwd=case.repo,
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
            row["server_pid"] = current_proc.pid

            ok, reason = wait_health(
                port=case.port,
                proc=current_proc,
                timeout_s=args.health_timeout_s,
                status_every_s=args.health_status_every_s,
                state=state,
                case_name=case.name,
            )
            if not ok:
                row["status"] = "launch_failed"
                row["error"] = reason
                row["ended_at"] = now_iso()
                state.add_result(row)
                print(f"[FAIL] {case.name}: {reason}", flush=True)
                continue

            state.set_status(phase="eval", case=case.name, message=f"running prompts on {case.name}")
            row["status"] = "ok"
            row["outputs"] = run_prompts(case, req_timeout_s=args.request_timeout_s)
            row["ended_at"] = now_iso()
            state.add_result(row)
            print(f"[DONE] {case.name}", flush=True)
        finally:
            stop_proc(current_proc)
            current_proc = None

    state.set_status(phase="done", message="all requested cases processed", ended_at=now_iso())
    print(f"\nRun complete. status={state.status_path} results={state.results_path}", flush=True)


if __name__ == "__main__":
    main()
