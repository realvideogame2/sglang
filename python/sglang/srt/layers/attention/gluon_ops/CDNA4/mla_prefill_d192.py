# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Thin D192 MLA prefill wrapper around the Gluon extend-attention kernel.

Exposes a simplified API for DeepSeek D192 MLA prefill (Lq=192, Lv=128)
by calling gluon_extend_attention_fwd with all non-MLA features disabled
(no sliding window, custom mask, logit cap, xai temperature, sinks).
"""

import math
import logging

import torch

logger = logging.getLogger(__name__)

from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_entrypoints import (
    gluon_extend_attention_fwd,
)


def mla_d192_prefill_fwd(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    is_causal=True,
    sm_scale=None,
    k_scale=1.0,
    v_scale=1.0,
    max_len_extend=None,
    min_len_extend=None,
    total_prefix_len=None,
    total_extend_len=None,
):
    """D192 MLA prefill: Lq=192 (128 nope + 64 rope), Lv=128.

    q_extend:   [total_q, num_heads, 192]  bf16
    k_extend:   [total_q, num_heads, 192]  bf16 or fp8
    v_extend:   [total_q, num_heads, 128]  bf16 or fp8
    o_extend:   [total_q, num_heads, 128]  bf16
    k_buffer:   [pool, num_heads, 192]     bf16 or fp8 (prefix KV via kv_indices)
    v_buffer:   [pool, num_heads, 128]     bf16 or fp8 (prefix KV via kv_indices)
    qo_indptr:  [batch+1] int32 -- extend token boundaries
    kv_indptr:  [batch+1] int32 -- prefix token boundaries
    kv_indices: [total_prefix] int32 -- prefix token locations in k/v_buffer
    """
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]
    assert Lq in (192, 288), f"Expected Lq in (192,288), got {Lq}"
    assert Lv == 128, f"Expected Lv=128, got {Lv}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Lq)

    if max_len_extend is None:
        extend_lens = qo_indptr[1:] - qo_indptr[:-1]
        max_len_extend = int(extend_lens.max().item())

    gluon_extend_attention_fwd(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask=None,
        is_causal=is_causal,
        mask_indptr=None,
        max_len_extend=max_len_extend,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=sm_scale,
        logit_cap=0.0,
        min_len_extend=min_len_extend,
        total_prefix_len=total_prefix_len,
        total_extend_len=total_extend_len,
    )
