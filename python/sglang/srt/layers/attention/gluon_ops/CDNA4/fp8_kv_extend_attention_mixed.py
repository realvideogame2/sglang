# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 DeepSeek/mixed-dim Gluon extend-attention kernels.

This module reuses the DeepSeek kernel implementation from
`f16_kv_extend_attention_mixed.py` and keeps import-plumbing for the
fp8 variant name.

That keeps BF16 kernels untouched while enabling FP8-specific MMA and
mixed fp8/bf16 shared-load handling in the FP8 variant.
"""

from pathlib import Path

_SRC_PATH = Path(__file__).with_name("f16_kv_extend_attention_mixed.py")
_SRC_TEXT = _SRC_PATH.read_text(encoding="utf-8")

_COMMON_IMPORT = (
    "from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_common import *  # noqa: F403"
)
_COMMON_FP8_IMPORT = (
    "from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_common import *  # noqa: F403"
)

if _COMMON_IMPORT not in _SRC_TEXT:
    raise RuntimeError(
        f"Failed to locate common import in {_SRC_PATH}; fp8 kernel shim cannot apply."
    )

_SRC_TEXT = _SRC_TEXT.replace(_COMMON_IMPORT, _COMMON_FP8_IMPORT, 1)

# Compile/exec into this module namespace so callers can import the same symbols
# (kernel funcs + launch helpers) from the fp8-specific module.
exec(compile(_SRC_TEXT, str(_SRC_PATH), "exec"), globals(), globals())
