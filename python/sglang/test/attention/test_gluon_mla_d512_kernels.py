import math
import random
import unittest
from typing import Sequence

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.gluon_ops.CDNA4.f16_mla_prefill import (
    mla_d512_gqa_attention_fwd,
    mla_d512_gqa_attention_fwd_wca,
)
from sglang.srt.layers.attention.gluon_ops.CDNA4.fp8_mla_prefill import (
    mla_d512_gqa_attention_fwd_fp8,
    mla_d512_gqa_attention_fwd_wca_fp8,
)


LQ = 576
LV = 512


def _get_fp8_dtype():
    for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
        dt = getattr(torch, name, None)
        if dt is not None:
            return dt
    return None


def _quantize_to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype):
    max_abs = float(x.detach().abs().max().item())
    scale = max(max_abs / 440.0, 1e-6)
    q = (x / scale).to(fp8_dtype)
    return q, scale


def _to_indptr(lengths: Sequence[int], device: torch.device):
    out = [0]
    for length in lengths:
        out.append(out[-1] + int(length))
    return torch.tensor(out, dtype=torch.int32, device=device)


def _build_inputs(
    *,
    seq_lens: Sequence[int],
    q_heads: int,
    kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
):
    total = sum(seq_lens)
    q = torch.randn(total, q_heads, LQ, dtype=dtype, device=device)
    kv_buffer = torch.randn(total, kv_heads, LQ, dtype=dtype, device=device)
    o = torch.empty(total, q_heads, LV, dtype=dtype, device=device)
    qo_indptr = _to_indptr(seq_lens, device)
    kv_indptr = _to_indptr(seq_lens, device)
    kv_indices = torch.arange(total, dtype=torch.int32, device=device)
    return {
        "q": q,
        "kv_buffer": kv_buffer,
        "o": o,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "max_len_extend": max(seq_lens),
    }


def _mla_ref(
    *,
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
):
    batch_size = qo_indptr.shape[0] - 1
    q_heads = q.shape[1]
    kv_heads = kv_buffer.shape[1]
    gqa = q_heads // kv_heads
    o = torch.zeros(q.shape[0], q_heads, LV, dtype=torch.float32, device=q.device)

    for i in range(batch_size):
        q_s = int(qo_indptr[i].item())
        q_e = int(qo_indptr[i + 1].item())
        kv_s = int(kv_indptr[i].item())
        kv_e = int(kv_indptr[i + 1].item())

        q_i = q[q_s:q_e].float()
        kv_i = kv_buffer[kv_indices[kv_s:kv_e]].float()
        k_i = kv_i[..., :LQ]
        v_i = kv_i[..., :LV]

        if gqa > 1:
            k_i = k_i.repeat_interleave(gqa, dim=1)
            v_i = v_i.repeat_interleave(gqa, dim=1)

        sq = q_i.shape[0]
        sk = k_i.shape[0]
        scores = torch.einsum("qhd,khd->qhk", q_i, k_i) * sm_scale

        pos_keys = torch.arange(sk, device=q.device)
        prefix_len = sk - sq
        t = prefix_len + torch.arange(sq, device=q.device)
        causal = pos_keys.unsqueeze(0) <= t.unsqueeze(1)
        scores = scores.masked_fill(~causal.unsqueeze(1), float("-inf"))

        probs = F.softmax(scores, dim=-1)
        o[q_s:q_e] = torch.einsum("qhk,khd->qhd", probs, v_i)

    return o


@unittest.skipIf(
    (not torch.cuda.is_available()) or (not torch.version.hip),
    "ROCm GPU is required",
)
class TestGluonMLAD512KernelAccuracy(unittest.TestCase):
    def setUp(self):
        random.seed(1)
        torch.manual_seed(1)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def _assert_close(self, out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float, min_cos: float):
        max_abs = (out.float() - ref).abs().max().item()
        cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
        self.assertGreaterEqual(cos, min_cos, msg=f"cos={cos:.6f} max_abs={max_abs:.6f}")
        self.assertTrue(
            torch.allclose(out.float(), ref, atol=atol, rtol=rtol),
            msg=f"allclose failed max_abs={max_abs:.6f}",
        )

    def test_mla_basic_fp16(self):
        for seq_lens in ([64], [16, 32, 24]):
            with self.subTest(seq_lens=seq_lens):
                inp = _build_inputs(
                    seq_lens=seq_lens,
                    q_heads=16,
                    kv_heads=1,
                    dtype=self.dtype,
                    device=self.device,
                )
                sm_scale = 1.0 / math.sqrt(LQ)
                mla_d512_gqa_attention_fwd(
                    inp["q"],
                    inp["kv_buffer"],
                    inp["o"],
                    inp["qo_indptr"],
                    inp["kv_indptr"],
                    inp["kv_indices"],
                    max_len_extend=inp["max_len_extend"],
                    is_causal=True,
                    sm_scale=sm_scale,
                )
                ref = _mla_ref(
                    q=inp["q"],
                    kv_buffer=inp["kv_buffer"],
                    qo_indptr=inp["qo_indptr"],
                    kv_indptr=inp["kv_indptr"],
                    kv_indices=inp["kv_indices"],
                    sm_scale=sm_scale,
                )
                self._assert_close(inp["o"], ref, atol=1e-2, rtol=1e-2, min_cos=0.999)

    def test_mla_wca_fp16(self):
        inp = _build_inputs(
            seq_lens=[32, 64, 16, 48],
            q_heads=16,
            kv_heads=1,
            dtype=self.dtype,
            device=self.device,
        )
        sm_scale = 1.0 / math.sqrt(LQ)
        mla_d512_gqa_attention_fwd_wca(
            inp["q"],
            inp["kv_buffer"],
            inp["o"],
            inp["qo_indptr"],
            inp["kv_indptr"],
            inp["kv_indices"],
            max_len_extend=inp["max_len_extend"],
            is_causal=True,
            sm_scale=sm_scale,
            split_k=2,
        )
        ref = _mla_ref(
            q=inp["q"],
            kv_buffer=inp["kv_buffer"],
            qo_indptr=inp["qo_indptr"],
            kv_indptr=inp["kv_indptr"],
            kv_indices=inp["kv_indices"],
            sm_scale=sm_scale,
        )
        self._assert_close(inp["o"], ref, atol=1.5e-2, rtol=1.5e-2, min_cos=0.998)

    def test_mla_basic_fp8_kv(self):
        fp8_dtype = _get_fp8_dtype()
        if fp8_dtype is None:
            self.skipTest("FP8 dtype is unavailable on this torch build")

        inp = _build_inputs(
            seq_lens=[64],
            q_heads=16,
            kv_heads=1,
            dtype=self.dtype,
            device=self.device,
        )
        kv_q, kv_scale = _quantize_to_fp8(inp["kv_buffer"], fp8_dtype)

        mla_d512_gqa_attention_fwd_fp8(
            inp["q"],
            kv_q,
            inp["o"],
            inp["qo_indptr"],
            inp["kv_indptr"],
            inp["kv_indices"],
            max_len_extend=inp["max_len_extend"],
            is_causal=True,
            sm_scale=1.0 / math.sqrt(LQ),
            k_scale=kv_scale,
            v_scale=kv_scale,
        )
        ref = _mla_ref(
            q=inp["q"],
            kv_buffer=kv_q.float() * kv_scale,
            qo_indptr=inp["qo_indptr"],
            kv_indptr=inp["kv_indptr"],
            kv_indices=inp["kv_indices"],
            sm_scale=(1.0 / math.sqrt(LQ)),
        )
        self._assert_close(inp["o"], ref, atol=1.2e-1, rtol=1.2e-1, min_cos=0.97)

    def test_mla_wca_fp8_kv(self):
        fp8_dtype = _get_fp8_dtype()
        if fp8_dtype is None:
            self.skipTest("FP8 dtype is unavailable on this torch build")

        inp = _build_inputs(
            seq_lens=[32, 48, 16, 24],
            q_heads=16,
            kv_heads=1,
            dtype=self.dtype,
            device=self.device,
        )
        kv_q, kv_scale = _quantize_to_fp8(inp["kv_buffer"], fp8_dtype)

        mla_d512_gqa_attention_fwd_wca_fp8(
            inp["q"],
            kv_q,
            inp["o"],
            inp["qo_indptr"],
            inp["kv_indptr"],
            inp["kv_indices"],
            max_len_extend=inp["max_len_extend"],
            is_causal=True,
            sm_scale=1.0 / math.sqrt(LQ),
            k_scale=kv_scale,
            v_scale=kv_scale,
            split_k=2,
        )
        ref = _mla_ref(
            q=inp["q"],
            kv_buffer=kv_q.float() * kv_scale,
            qo_indptr=inp["qo_indptr"],
            kv_indptr=inp["kv_indptr"],
            kv_indices=inp["kv_indices"],
            sm_scale=(1.0 / math.sqrt(LQ)),
        )
        self._assert_close(inp["o"], ref, atol=2.0e-1, rtol=2.0e-1, min_cos=0.96)


if __name__ == "__main__":
    unittest.main()

