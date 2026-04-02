import math
import os
import random
import unittest
from typing import Sequence

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_entrypoints import (
    gluon_extend_attention_fwd,
)


def _get_fp8_dtype():
    for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
        dt = getattr(torch, name, None)
        if dt is not None:
            return dt
    return None


def _quantize_to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype):
    # e4m3 finite range is ~448; keep a small safety margin.
    max_abs = float(x.detach().abs().max().item())
    scale = max(max_abs / 440.0, 1e-6)
    q = (x / scale).to(fp8_dtype)
    return q, scale


def _quantize_fp8_style(x: torch.Tensor):
    # Use fp8-like dynamic range and integerized mantissa in fp16/bf16 tensors.
    max_abs = float(x.detach().abs().max().item())
    scale = max(max_abs / 440.0, 1e-6)
    q = torch.round((x / scale).clamp(-440, 440)).to(x.dtype)
    return q, scale


def _to_indptr(lengths: Sequence[int], device: torch.device):
    out = [0]
    for length in lengths:
        out.append(out[-1] + int(length))
    return torch.tensor(out, dtype=torch.int32, device=device)


def _build_inputs(
    *,
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    q_heads: int,
    kv_heads: int,
    qk_dim: int,
    v_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    batch_size = len(prefix_lens)
    assert len(extend_lens) == batch_size

    total_prefix = sum(prefix_lens)
    total_extend = sum(extend_lens)
    total_kv = total_prefix + total_extend

    q_extend = torch.randn(total_extend, q_heads, qk_dim, dtype=dtype, device=device)
    k_extend = torch.randn(total_extend, kv_heads, qk_dim, dtype=dtype, device=device)
    v_extend = torch.randn(total_extend, kv_heads, v_dim, dtype=dtype, device=device)

    k_buffer = torch.randn(total_kv, kv_heads, qk_dim, dtype=dtype, device=device)
    v_buffer = torch.randn(total_kv, kv_heads, v_dim, dtype=dtype, device=device)

    # Prefix indices are per-sequence contiguous regions in the packed buffer.
    kv_index_chunks = []
    cursor = 0
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        kv_index_chunks.append(
            torch.arange(cursor, cursor + prefix_len, dtype=torch.int32, device=device)
        )
        cursor += prefix_len + extend_len
    kv_indices = (
        torch.cat(kv_index_chunks)
        if kv_index_chunks
        else torch.empty((0,), dtype=torch.int32, device=device)
    )

    qo_indptr = _to_indptr(extend_lens, device)
    kv_indptr = _to_indptr(prefix_lens, device)

    # Mirror production packed-layout behavior: current extend tokens are also in cache.
    buf_cursor = 0
    ext_cursor = 0
    for prefix_len, extend_len in zip(prefix_lens, extend_lens):
        buf_start = buf_cursor + prefix_len
        buf_end = buf_start + extend_len
        ext_end = ext_cursor + extend_len
        k_buffer[buf_start:buf_end] = k_extend[ext_cursor:ext_end]
        v_buffer[buf_start:buf_end] = v_extend[ext_cursor:ext_end]
        buf_cursor += prefix_len + extend_len
        ext_cursor = ext_end

    return {
        "q_extend": q_extend,
        "k_extend": k_extend,
        "v_extend": v_extend,
        "k_buffer": k_buffer,
        "v_buffer": v_buffer,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "max_len_extend": max(extend_lens),
        "min_len_extend": min(extend_lens),
        "total_prefix_len": total_prefix,
        "total_extend_len": total_extend,
    }


def _ref_extend_attention(
    *,
    q_extend: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
):
    batch_size = qo_indptr.shape[0] - 1
    q_heads = q_extend.shape[1]
    kv_heads = k_extend.shape[1]
    gqa = q_heads // kv_heads
    qk_dim = q_extend.shape[-1]
    v_dim = v_extend.shape[-1]

    o = torch.zeros(
        q_extend.shape[0], q_heads, v_dim, dtype=torch.float32, device=q_extend.device
    )

    for i in range(batch_size):
        q_s = int(qo_indptr[i].item())
        q_e = int(qo_indptr[i + 1].item())
        kv_s = int(kv_indptr[i].item())
        kv_e = int(kv_indptr[i + 1].item())

        q_i = q_extend[q_s:q_e].float()
        k_prefix = k_buffer[kv_indices[kv_s:kv_e]].float()
        v_prefix = v_buffer[kv_indices[kv_s:kv_e]].float()
        k_new = k_extend[q_s:q_e].float()
        v_new = v_extend[q_s:q_e].float()

        k_full = torch.cat([k_prefix, k_new], dim=0)
        v_full = torch.cat([v_prefix, v_new], dim=0)
        if gqa > 1:
            k_full = k_full.repeat_interleave(gqa, dim=1)
            v_full = v_full.repeat_interleave(gqa, dim=1)

        sq = q_i.shape[0]
        sk = k_full.shape[0]
        scores = torch.einsum("qhd,khd->qhk", q_i, k_full[..., :qk_dim]) * sm_scale

        # Causal mask over prefix + current extend tokens.
        pos_keys = torch.arange(sk, device=q_i.device)
        prefix_len = sk - sq
        t = prefix_len + torch.arange(sq, device=q_i.device)
        causal_mask = pos_keys.unsqueeze(0) <= t.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))

        probs = F.softmax(scores, dim=-1)
        o[q_s:q_e] = torch.einsum("qhk,khd->qhd", probs, v_full[..., :v_dim])

    return o


@unittest.skipIf(
    (not torch.cuda.is_available()) or (not torch.version.hip),
    "ROCm GPU is required",
)
class TestGluonExtendKernelAccuracy(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        os.environ.setdefault("AITER_ENABLE_GLUON_DEEPSEEK", "1")
        os.environ.setdefault("AITER_ENABLE_GLUON_MIXED_DIMS", "1")
        os.environ.setdefault("SGLANG_GLUON_FP8_KV_FORCE_BF16", "0")

    def _run_case(
        self,
        *,
        qk_dim: int,
        v_dim: int,
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
        q_heads: int,
        kv_heads: int,
        use_fp8_kv: bool,
        atol: float,
        rtol: float,
        min_cos: float,
    ):
        inp = _build_inputs(
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            q_heads=q_heads,
            kv_heads=kv_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            dtype=self.dtype,
            device=self.device,
        )
        sm_scale = 1.0 / math.sqrt(qk_dim)
        k_scale = 1.0
        v_scale = 1.0

        if use_fp8_kv:
            # Gluon extend kernels currently consume fp16/bf16 tensors.
            # Emulate FP8 KV cache numerics by passing quantized-value tensors
            # together with explicit k_scale/v_scale descales.
            k_buffer_q, k_scale = _quantize_fp8_style(inp["k_buffer"])
            v_buffer_q, v_scale = _quantize_fp8_style(inp["v_buffer"])
            k_extend_q, _ = _quantize_fp8_style(inp["k_extend"])
            v_extend_q, _ = _quantize_fp8_style(inp["v_extend"])

            k_buffer_kernel = k_buffer_q
            v_buffer_kernel = v_buffer_q
            k_extend_kernel = k_extend_q
            v_extend_kernel = v_extend_q

            k_buffer_ref = k_buffer_q.float() * k_scale
            v_buffer_ref = v_buffer_q.float() * v_scale
            k_extend_ref = k_extend_q.float() * k_scale
            v_extend_ref = v_extend_q.float() * v_scale
        else:
            k_buffer_kernel = inp["k_buffer"]
            v_buffer_kernel = inp["v_buffer"]
            k_extend_kernel = inp["k_extend"]
            v_extend_kernel = inp["v_extend"]
            k_buffer_ref = inp["k_buffer"]
            v_buffer_ref = inp["v_buffer"]
            k_extend_ref = inp["k_extend"]
            v_extend_ref = inp["v_extend"]

        o_gluon = torch.empty(
            inp["q_extend"].shape[0],
            q_heads,
            v_dim,
            dtype=self.dtype,
            device=self.device,
        )
        gluon_extend_attention_fwd(
            inp["q_extend"],
            k_extend_kernel,
            v_extend_kernel,
            o_gluon,
            k_buffer_kernel,
            v_buffer_kernel,
            inp["qo_indptr"],
            inp["kv_indptr"],
            inp["kv_indices"],
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=inp["max_len_extend"],
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=sm_scale,
            min_len_extend=inp["min_len_extend"],
            total_prefix_len=inp["total_prefix_len"],
            total_extend_len=inp["total_extend_len"],
        )
        o_ref = _ref_extend_attention(
            q_extend=inp["q_extend"],
            k_extend=k_extend_ref,
            v_extend=v_extend_ref,
            k_buffer=k_buffer_ref,
            v_buffer=v_buffer_ref,
            qo_indptr=inp["qo_indptr"],
            kv_indptr=inp["kv_indptr"],
            kv_indices=inp["kv_indices"],
            sm_scale=sm_scale,
        )

        max_abs = (o_gluon.float() - o_ref).abs().max().item()
        cos = F.cosine_similarity(
            o_gluon.float().flatten(), o_ref.flatten(), dim=0
        ).item()
        self.assertGreaterEqual(
            cos,
            min_cos,
            msg=f"cos={cos:.6f} max_abs={max_abs:.6f}",
        )
        self.assertTrue(
            torch.allclose(o_gluon.float(), o_ref, atol=atol, rtol=rtol),
            msg=f"allclose failed max_abs={max_abs:.6f}",
        )

    def _run_native_fp8_cache_case(
        self,
        *,
        qk_dim: int,
        v_dim: int,
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
        q_heads: int,
        kv_heads: int,
        atol: float,
        rtol: float,
        min_cos: float,
    ):
        fp8_dtype = _get_fp8_dtype()
        if fp8_dtype is None:
            self.skipTest("FP8 dtype is unavailable on this torch build")

        inp = _build_inputs(
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            q_heads=q_heads,
            kv_heads=kv_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            dtype=self.dtype,
            device=self.device,
        )
        sm_scale = 1.0 / math.sqrt(qk_dim)
        o_gluon = torch.empty(
            inp["q_extend"].shape[0],
            q_heads,
            v_dim,
            dtype=self.dtype,
            device=self.device,
        )

        k_buffer_fp8 = inp["k_buffer"].to(fp8_dtype)
        v_buffer_fp8 = inp["v_buffer"].to(fp8_dtype)
        gluon_extend_attention_fwd(
            inp["q_extend"],
            inp["k_extend"],
            inp["v_extend"],
            o_gluon,
            k_buffer_fp8,
            v_buffer_fp8,
            inp["qo_indptr"],
            inp["kv_indptr"],
            inp["kv_indices"],
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=inp["max_len_extend"],
            k_scale=1.0,
            v_scale=1.0,
            sm_scale=sm_scale,
            logit_cap=0.0,
            min_len_extend=inp["min_len_extend"],
            total_prefix_len=inp["total_prefix_len"],
            total_extend_len=inp["total_extend_len"],
        )
        o_ref = _ref_extend_attention(
            q_extend=inp["q_extend"],
            k_extend=inp["k_extend"],
            v_extend=inp["v_extend"],
            k_buffer=k_buffer_fp8.float(),
            v_buffer=v_buffer_fp8.float(),
            qo_indptr=inp["qo_indptr"],
            kv_indptr=inp["kv_indptr"],
            kv_indices=inp["kv_indices"],
            sm_scale=sm_scale,
        )

        max_abs = (o_gluon.float() - o_ref).abs().max().item()
        cos = F.cosine_similarity(o_gluon.float().flatten(), o_ref.flatten(), dim=0).item()
        self.assertGreaterEqual(cos, min_cos, msg=f"cos={cos:.6f} max_abs={max_abs:.6f}")
        self.assertTrue(
            torch.allclose(o_gluon.float(), o_ref, atol=atol, rtol=rtol),
            msg=f"allclose failed max_abs={max_abs:.6f}",
        )

    def test_fp16_d64(self):
        self._run_case(
            qk_dim=64,
            v_dim=64,
            prefix_lens=[128, 96, 64, 160],
            extend_lens=[8, 16, 4, 12],
            q_heads=32,
            kv_heads=8,
            use_fp8_kv=False,
            atol=1e-2,
            rtol=1e-2,
            min_cos=0.999,
        )

    def test_fp16_d128(self):
        self._run_case(
            qk_dim=128,
            v_dim=128,
            prefix_lens=[256, 64, 192],
            extend_lens=[16, 8, 12],
            q_heads=32,
            kv_heads=8,
            use_fp8_kv=False,
            atol=1e-2,
            rtol=1e-2,
            min_cos=0.999,
        )

    def test_fp16_d192_v128_deepseek(self):
        self._run_case(
            qk_dim=192,
            v_dim=128,
            prefix_lens=[128, 256],
            extend_lens=[8, 12],
            q_heads=16,
            kv_heads=2,
            use_fp8_kv=False,
            atol=1e-2,
            rtol=1e-2,
            min_cos=0.998,
        )

    def test_fp8_kv_d128(self):
        self._run_case(
            qk_dim=128,
            v_dim=128,
            prefix_lens=[128, 192],
            extend_lens=[8, 8],
            q_heads=32,
            kv_heads=8,
            use_fp8_kv=True,
            atol=8e-2,
            rtol=8e-2,
            min_cos=0.98,
        )

    def test_fp8_kv_d64(self):
        self._run_case(
            qk_dim=64,
            v_dim=64,
            prefix_lens=[160, 96, 128],
            extend_lens=[8, 12, 10],
            q_heads=32,
            kv_heads=8,
            use_fp8_kv=True,
            atol=9e-2,
            rtol=9e-2,
            min_cos=0.98,
        )

    def test_fp8_kv_d192_v128_deepseek(self):
        self._run_case(
            qk_dim=192,
            v_dim=128,
            prefix_lens=[256],
            extend_lens=[12],
            q_heads=16,
            kv_heads=2,
            use_fp8_kv=True,
            atol=1.2e-1,
            rtol=1.2e-1,
            min_cos=0.97,
        )

    def test_native_fp8_kv_cache_d192_v128_deepseek(self):
        self._run_native_fp8_cache_case(
            qk_dim=192,
            v_dim=128,
            prefix_lens=[256],
            extend_lens=[12],
            q_heads=16,
            kv_heads=2,
            atol=1.8e-1,
            rtol=1.8e-1,
            min_cos=0.95,
        )

    def test_native_fp8_kv_cache_d64(self):
        self._run_native_fp8_cache_case(
            qk_dim=64,
            v_dim=64,
            prefix_lens=[128, 192, 96],
            extend_lens=[8, 10, 12],
            q_heads=32,
            kv_heads=8,
            atol=1.8e-1,
            rtol=1.8e-1,
            min_cos=0.95,
        )

    def test_native_fp8_kv_cache_d128(self):
        self._run_native_fp8_cache_case(
            qk_dim=128,
            v_dim=128,
            prefix_lens=[160, 224],
            extend_lens=[12, 10],
            q_heads=32,
            kv_heads=8,
            atol=1.8e-1,
            rtol=1.8e-1,
            min_cos=0.95,
        )


if __name__ == "__main__":
    unittest.main()

