#!/usr/bin/env bash
# Launch DeepSeek-R1-MXFP4 with Gluon attention kernels on MI350X (8xTP)
#
# Usage (inside the Docker container):
#   ./scripts/launch_dsr1_gluon.sh [MODEL_PATH] [PORT]
#
# With Docker directly:
#   docker run --rm -it --device /dev/kfd --device /dev/dri \
#     --group-add video --security-opt seccomp=unconfined \
#     -v /data:/data -v /data2:/data2 \
#     -p 9000:9000 \
#     sglang-gluon:latest \
#     bash scripts/launch_dsr1_gluon.sh /data/models/DeepSeek-R1-0528-MXFP4 9000

set -euo pipefail

MODEL="${1:-/data/models/DeepSeek-R1-0528-MXFP4}"
PORT="${2:-9000}"

# -- Gluon probe: swap Gluon in for Aiter's extend (prefill) kernel.
# Decode stays on Aiter's native ASM MLA decode kernel.
export SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND=gluon

exec python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tp-size 8 \
    --trust-remote-code \
    --chunked-prefill-size 131072 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --log-requests \
    --disable-radix-cache \
    --mem-fraction-static 0.8 \
    --max-running-requests 64 \
    --kv-cache-dtype fp8_e4m3 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --attention-backend aiter \
    --disable-overlap-schedule \
    2>&1 | tee /tmp/sglang_gluon.log
