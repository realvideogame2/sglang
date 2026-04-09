# Gluon-enabled SGLang image for DeepSeek-R1-MXFP4 on MI350X (gfx950)
#
# Builds on the official SGLang ROCm dev image and upgrades Triton to
# upstream main so that Gluon kernels (which need PaddedSharedLayout
# cga_layout support from triton-lang/triton PR #9336+) compile correctly
# during CUDA-graph capture.
#
# Usage:
#   docker build -f docker/rocm-gluon.Dockerfile -t sglang-gluon:latest .
#
# The build context is the sglang repo root (this branch).

ARG BASE_IMAGE="rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260331"
FROM ${BASE_IMAGE}

ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG TRITON_COMMIT="8d4c6cd55"

USER root
ENV HOME=/root
ENV TRITON_CACHE_PATH=/root/.triton

# ---------- 1. Upgrade Triton (AMD backend only) ----------
RUN set -eux; \
    git clone --depth=200 ${TRITON_REPO} /tmp/triton-build; \
    cd /tmp/triton-build; \
    git checkout ${TRITON_COMMIT}; \
    TRITON_CODEGEN_BACKENDS=amd pip install --no-build-isolation .; \
    rm -rf /tmp/triton-build; \
    python3 -c "import triton; print('triton', triton.__version__); \
                 from triton.backends.amd import driver; print('AMD backend OK'); \
                 from triton.experimental.gluon.language._layouts import PaddedSharedLayout; \
                 import inspect; sig = inspect.signature(PaddedSharedLayout.__init__); \
                 assert 'cga_layout' in str(sig), 'missing cga_layout'; \
                 print('PaddedSharedLayout cga_layout OK')"

# ---------- 2. Install Gluon-enabled SGLang from this branch ----------
COPY python /sgl-workspace/sglang-gluon/python

# ---------- 3. Env vars for ROCm perf + aiter ----------
ENV PYTHONPATH="/sgl-workspace/sglang-gluon/python:/sgl-workspace/aiter:${PYTHONPATH}"
ENV HIP_FORCE_DEV_KERNARG=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV SGLANG_USE_AITER=1
ENV SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
ENV SGLANG_SET_CPU_AFFINITY=1
ENV SGLANG_AITER_MLA_PERSIST=1

CMD ["/bin/bash"]
