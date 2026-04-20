# syntax=docker/dockerfile:1.5
#
# SGLang + Gluon attention kernels (MI350X / gfx950)
# --------------------------------------------------
#
# This image layers tussingh's Gluon-attention work on top of the
# ROCm 7.0.0 / Triton 3.7 base that was validated for DeepSeek-R1
# prefill profiling. Only the sglang Python package is replaced; the
# base image's aiter / torch / triton builds are kept intact so the
# ASM fallback remains available.
#
# Base: rocm/sgl-dev:v0.5.10rc0-rocm700-mi35x-20260409
#   - ROCm 7.0.0
#   - hipBLASlt + rccl tuned for MI350X
#   - Triton 3.7 (upstream) built against this ROCm
#   - torch 2.12 nightly (ROCm 7.1 build but works with 7.0 runtime)
#   - aiter 0.1.12 (ASM kernels + Triton fallbacks)
#   - sgl-kernel built for gfx950
#
# Build examples
# --------------
#
#   # MLA prefill branch (D192 / DeepSeek R1 prefill replacement)
#   docker build \
#     --build-arg SGL_FORK=https://github.com/realvideogame2/sglang.git \
#     --build-arg SGL_BRANCH=tussingh/gluon-mla-prefill \
#     -t sglang-gluon-mla:rocm700-mi35x \
#     -f docker/rocm-gluon.Dockerfile .
#
#   # Extend-attention branch (BF16 + FP8-KV extend path)
#   docker build \
#     --build-arg SGL_FORK=https://github.com/realvideogame2/sglang.git \
#     --build-arg SGL_BRANCH=tussingh/gluon-extend-attn \
#     -t sglang-gluon-extend:rocm700-mi35x \
#     -f docker/rocm-gluon.Dockerfile .
#
# Or pull the Dockerfile directly off the branch without a local checkout:
#
#   curl -fsSL https://raw.githubusercontent.com/realvideogame2/sglang/tussingh/gluon-mla-prefill/docker/rocm-gluon.Dockerfile \
#     -o rocm-gluon.Dockerfile
#   docker build \
#     --build-arg SGL_FORK=https://github.com/realvideogame2/sglang.git \
#     --build-arg SGL_BRANCH=tussingh/gluon-mla-prefill \
#     -t sglang-gluon-mla:rocm700-mi35x \
#     -f rocm-gluon.Dockerfile .
#
# Runtime
# -------
#
# Launch DeepSeek-R1 FP8 prefill with Gluon MLA:
#
#   docker run --rm -it \
#     --device=/dev/kfd --device=/dev/dri \
#     --group-add video --ipc=host --network=host \
#     --shm-size 32G \
#     -v ${MODEL_DIR}:/models \
#     -e SGLANG_AITER_USE_GLUON_MLA_PREFILL=1 \
#     -e SGLANG_AITER_GLUON_MLA_SCHED=hybrid \
#     -e SGLANG_AITER_FP8_PREFILL_ATTN=1 \
#     -e SGLANG_AITER_MLA_PERSIST=1 \
#     sglang-gluon-mla:rocm700-mi35x \
#     python3 -m sglang.launch_server \
#       --model-path /models/DeepSeek-R1 \
#       --tp-size 8 --trust-remote-code \
#       --attention-backend aiter \
#       --kv-cache-dtype fp8_e4m3 \
#       --chunked-prefill-size 131072 \
#       --mem-fraction-static 0.8 \
#       --speculative-algorithm EAGLE \
#       --speculative-num-steps 3 \
#       --speculative-eagle-topk 1 \
#       --speculative-num_draft_tokens 4
#
# Set SGLANG_AITER_USE_GLUON_MLA_PREFILL=0 (or unset it) to run the
# pure ASM baseline from the same image for A/B comparison.

ARG BASE_IMAGE=rocm/sgl-dev:v0.5.10rc0-rocm700-mi35x-20260409
FROM ${BASE_IMAGE}

ARG SGL_FORK=https://github.com/realvideogame2/sglang.git
ARG SGL_BRANCH=tussingh/gluon-mla-prefill

LABEL org.opencontainers.image.title="sglang-gluon-mi350x"
LABEL org.opencontainers.image.description="SGLang + Gluon attention kernels for AMD MI350X (gfx950)"
LABEL org.opencontainers.image.source="${SGL_FORK}"
LABEL org.opencontainers.image.ref.name="${SGL_BRANCH}"
LABEL gluon.target.arch="gfx950"
LABEL gluon.rocm.version="7.0.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Native gfx950 detection works on MI350X/MI355 under ROCm 7.0 — do NOT
# set HSA_OVERRIDE_GFX_VERSION here (setting e.g. 11.0.0 would force
# the runtime to compile for gfx1100 / RDNA3 and break everything).
ENV PYTORCH_ROCM_ARCH=gfx950

# Default Gluon knobs. Override at docker run time with -e.
# Unset USE_GLUON to fall back to pure ASM for A/B comparison.
ENV SGLANG_AITER_USE_GLUON_MLA_PREFILL=1
ENV SGLANG_AITER_GLUON_MLA_SCHED=hybrid
ENV SGLANG_AITER_FP8_PREFILL_ATTN=1

# Replace the bundled sglang with the fork's branch. We reinstall
# in-place so sgl-kernel / aiter / torch from the base image are
# untouched. The editable install keeps the vendored Gluon kernel
# sources under python/sglang/srt/layers/attention/gluon_ops/
# directly importable (they're .py, no C++ compile needed).
RUN set -eux; \
    rm -rf /sgl-workspace/sglang; \
    git clone --depth 1 --branch "${SGL_BRANCH}" "${SGL_FORK}" /sgl-workspace/sglang; \
    cd /sgl-workspace/sglang/python; \
    python3 -m pip install --no-deps -e . ; \
    python3 -c "import sglang; print('sglang =', sglang.__version__)" ; \
    python3 -c "from sglang.srt.layers.attention import gluon_mla_prefill as _; print('gluon MLA wrapper OK')" 2>/dev/null || true ; \
    python3 -c "from sglang.srt.layers.attention import gluon_extend_attention as _; print('gluon extend wrapper OK')" 2>/dev/null || true

WORKDIR /sgl-workspace/sglang

# Healthcheck: make sure the Gluon ops package is importable at
# container start, not just at install time (catches broken installs
# that manifest only after torch is loaded).
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "from sglang.srt.layers.attention.gluon_ops import __init__ as _" || exit 1

# Default command: print install info so `docker run <img>` is
# self-diagnostic if invoked without args.
CMD ["python3", "-c", "import sglang, torch; \
print('sglang     =', sglang.__version__); \
print('torch      =', torch.__version__); \
print('triton     =', __import__('triton').__version__); \
print('hip device =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); \
print('Launch sglang.launch_server with -e SGLANG_AITER_USE_GLUON_MLA_PREFILL=1 to enable Gluon.')"]
