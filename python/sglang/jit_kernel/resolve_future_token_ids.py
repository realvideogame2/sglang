from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HIPCC_PATH = "/opt/rocm/bin/hipcc"
_HIPCC_PERL_DRIVER_PATH = "/opt/rocm/bin/hipcc.pl"
# Some ROCm installs expose hipcc but miss hipcc.pl; treat this as
# unavailable JIT toolchain and use native fallback.
_USE_NATIVE_HIP_FALLBACK = is_hip() and (
    (not os.path.exists(_HIPCC_PATH)) or (not os.path.exists(_HIPCC_PERL_DRIVER_PATH))
)


@cache_once
def _jit_resolve_future_token_ids_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT module for a given dtype."""
    args = make_cpp_args(dtype)
    return load_jit(
        "resolve_future_token_ids",
        *args,
        cuda_files=["elementwise/resolve_future_token_ids.cuh"],
        cuda_wrappers=[
            (
                "resolve_future_token_ids",
                f"ResolveFutureTokenIds<{args}>::run",
            )
        ],
    )


def resolve_future_token_ids_cuda(
    input_ids: torch.Tensor, future_token_ids_map: torch.Tensor
) -> None:
    """Resolve future token IDs in-place on CUDA.

    For each negative value in input_ids, replaces it with
    future_token_ids_map[-value]. Non-negative values are unchanged.

    Supported dtypes: torch.int32, torch.int64.
    """
    if _USE_NATIVE_HIP_FALLBACK:
        input_ids[:] = torch.where(
            input_ids < 0,
            future_token_ids_map[torch.clamp(-input_ids, min=0)],
            input_ids,
        )
        return
    module = _jit_resolve_future_token_ids_module(input_ids.dtype)
    module.resolve_future_token_ids(input_ids, future_token_ids_map)
