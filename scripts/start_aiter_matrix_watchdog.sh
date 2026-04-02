#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tussingh/sglang_wca"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR_DEFAULT="$ROOT/e2e_results/ctrl_flow_probe/runs/matrix_live_$STAMP"

RUN_DIR="${1:-$RUN_DIR_DEFAULT}"
if [[ $# -gt 0 ]]; then
  shift
fi

mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/orchestrator.log"
PID_FILE="$RUN_DIR/orchestrator.pid"
STATUS_FILE="$RUN_DIR/status.json"
RESULTS_FILE="$RUN_DIR/results.json"

cd "$ROOT"
nohup /home/tussingh/venv-triton/bin/python scripts/aiter_native_matrix_watchdog.py --run-dir "$RUN_DIR" "$@" >"$LOG_FILE" 2>&1 &
PID="$!"
echo "$PID" >"$PID_FILE"

echo "Started matrix watchdog."
echo "run_dir:   $RUN_DIR"
echo "pid:       $PID"
echo "log:       $LOG_FILE"
echo "status:    $STATUS_FILE"
echo "results:   $RESULTS_FILE"
