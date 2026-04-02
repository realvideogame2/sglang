import os
import unittest
from contextlib import contextmanager
from typing import Sequence
from unittest import mock

import torch

from sglang.srt.layers.attention.attention_registry import create_gluon_backend
from sglang.srt.layers.attention.gluon_ops.CDNA4 import (
    extend_attention_entrypoints as eag,
)


def _get_fp8_dtype():
    for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
        dt = getattr(torch, name, None)
        if dt is not None:
            return dt
    return None


def _reset_dispatch_caches():
    eag._CACHED_ENV_MIXED_DIMS = None
    eag._CACHED_ENV_DEEPSEEK = None
    eag._CACHED_ENV_BLOCK_DPE = None
    eag._CACHED_ENV_FP8_KV_FORCE_BF16 = None
    eag._QK_SPLIT_CACHE.clear()
    eag._NUM_XCDS = None


@contextmanager
def _temp_env(**updates):
    old = {k: os.environ.get(k) for k in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    _reset_dispatch_caches()
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _reset_dispatch_caches()


class _KernelSpy:
    def __init__(self):
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _to_indptr(lengths: Sequence[int], device: torch.device):
    out = [0]
    for length in lengths:
        out.append(out[-1] + int(length))
    return torch.tensor(out, dtype=torch.int32, device=device)


def _make_inputs(
    *,
    batch_size: int,
    ext_lens: Sequence[int],
    prefix_len: int,
    q_heads: int,
    kv_heads: int,
    qk_dim: int,
    v_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    total_q = sum(ext_lens)
    total_prefix = prefix_len * batch_size
    total_kv = total_prefix + total_q

    q_extend = torch.randn(total_q, q_heads, qk_dim, dtype=dtype, device=device)
    k_extend = torch.randn(total_q, kv_heads, qk_dim, dtype=dtype, device=device)
    v_extend = torch.randn(total_q, kv_heads, v_dim, dtype=dtype, device=device)
    o_extend = torch.zeros(total_q, q_heads, v_dim, dtype=dtype, device=device)

    k_buffer = torch.randn(total_kv, kv_heads, qk_dim, dtype=dtype, device=device)
    v_buffer = torch.randn(total_kv, kv_heads, v_dim, dtype=dtype, device=device)

    qo_indptr = _to_indptr(ext_lens, device)
    kv_indptr = _to_indptr([prefix_len] * batch_size, device)
    kv_indices = torch.arange(total_prefix, dtype=torch.int32, device=device)

    return (
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        total_prefix,
        total_q,
    )


def _make_registry_runner():
    return type(
        "RunnerStub",
        (),
        {
            "model_config": type("ModelConfig", (), {"is_encoder_decoder": False})(),
            "server_args": type("ServerArgs", (), {"enable_double_sparsity": False})(),
        },
    )()


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGluonKernelRouting(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.q_heads = 8
        self.kv_heads = 8

    def _invoke(
        self,
        *,
        qk_dim: int,
        v_dim: int,
        ext_lens: Sequence[int],
        prefix_len: int = 16,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        sm_scale: float | None = None,
        min_len_extend: int | None = None,
        force_splitk: bool | None = None,
        force_persistent: bool | None = None,
        kv_cache_dtype: torch.dtype | None = None,
    ):
        batch_size = len(ext_lens)
        (
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            total_prefix,
            total_q,
        ) = _make_inputs(
            batch_size=batch_size,
            ext_lens=ext_lens,
            prefix_len=prefix_len,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            dtype=self.dtype,
            device=self.device,
        )
        if kv_cache_dtype is not None:
            k_buffer = k_buffer.to(kv_cache_dtype)
            v_buffer = v_buffer.to(kv_cache_dtype)

        eag.gluon_extend_attention_fwd(
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
            is_causal=True,
            mask_indptr=None,
            max_len_extend=max(ext_lens),
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=sm_scale,
            min_len_extend=min_len_extend,
            total_prefix_len=total_prefix,
            total_extend_len=total_q,
            _force_use_persistent=force_persistent,
            _force_use_splitk=force_splitk,
        )

    def test_symmetric_kernel_used_for_supported_equal_dims(self):
        for qk_dim in (64, 128, 192):
            with self.subTest(qk_dim=qk_dim):
                sym = _KernelSpy()
                deep = _KernelSpy()
                with _temp_env(
                    AITER_ENABLE_GLUON_MIXED_DIMS=0,
                    AITER_ENABLE_GLUON_DEEPSEEK=1,
                    SGLANG_GLUON_D128_EXPERIMENTAL=0,
                ), mock.patch.object(eag, "_gluon_extend_attn_fwd_symmetric", sym), mock.patch.object(
                    eag, "_gluon_extend_attn_fwd_deepseek", deep
                ), mock.patch.object(
                    eag, "_get_num_CUs", return_value=256
                ):
                    self._invoke(
                        qk_dim=qk_dim,
                        v_dim=qk_dim,
                        ext_lens=[32],
                        force_persistent=False,
                    )
                self.assertEqual(len(sym.calls), 1)
                self.assertEqual(len(deep.calls), 0)

    def test_deepseek_kernel_used_for_mixed_dims_192_128(self):
        sym = _KernelSpy()
        deep = _KernelSpy()
        with _temp_env(
            AITER_ENABLE_GLUON_MIXED_DIMS=1,
            AITER_ENABLE_GLUON_DEEPSEEK=1,
        ), mock.patch.object(eag, "_gluon_extend_attn_fwd_symmetric", sym), mock.patch.object(
            eag, "_gluon_extend_attn_fwd_deepseek", deep
        ), mock.patch.object(
            eag, "_get_num_CUs", return_value=256
        ):
            self._invoke(qk_dim=192, v_dim=128, ext_lens=[64])
        self.assertEqual(len(sym.calls), 0)
        self.assertEqual(len(deep.calls), 1)

    def test_mixed_dims_falls_back_to_triton_when_disabled(self):
        with _temp_env(
            AITER_ENABLE_GLUON_MIXED_DIMS=0,
            AITER_ENABLE_GLUON_DEEPSEEK=0,
        ), mock.patch(
            "sglang.srt.layers.attention.triton_ops.extend_attention.extend_attention_fwd"
        ) as fallback:
            self._invoke(qk_dim=192, v_dim=128, ext_lens=[32], k_scale=3.0, v_scale=5.0)
        self.assertTrue(fallback.called)
        _, kwargs = fallback.call_args
        self.assertAlmostEqual(kwargs["k_scale"], 3.0)
        self.assertAlmostEqual(kwargs["v_scale"], 5.0)

    def test_ragged_d128_routes_to_persistent(self):
        sym = _KernelSpy()
        with _temp_env(
            AITER_ENABLE_GLUON_MIXED_DIMS=0,
            AITER_ENABLE_GLUON_DEEPSEEK=1,
        ), mock.patch.object(eag, "_gluon_extend_attn_fwd_symmetric", sym), mock.patch.object(
            eag, "_launch_persistent"
        ) as launch_persistent, mock.patch.object(
            eag, "_get_num_CUs", return_value=256
        ):
            self._invoke(qk_dim=128, v_dim=128, ext_lens=[16, 1024], min_len_extend=None)
        self.assertTrue(launch_persistent.called)
        self.assertEqual(len(sym.calls), 0)

    def test_splitk_override_routes_to_splitk_path(self):
        sym = _KernelSpy()
        with _temp_env(
            AITER_ENABLE_GLUON_MIXED_DIMS=0,
            AITER_ENABLE_GLUON_DEEPSEEK=1,
        ), mock.patch.object(eag, "_gluon_extend_attn_fwd_symmetric", sym), mock.patch.object(
            eag, "_launch_splitk"
        ) as launch_splitk, mock.patch.object(
            eag, "_get_num_CUs", return_value=256
        ):
            self._invoke(qk_dim=64, v_dim=64, ext_lens=[128, 128], force_splitk=True)
        self.assertTrue(launch_splitk.called)
        self.assertEqual(len(sym.calls), 0)

    def test_v_scale_propagates_to_kernel_kwargs(self):
        sym = _KernelSpy()
        with _temp_env(
            AITER_ENABLE_GLUON_MIXED_DIMS=0,
            AITER_ENABLE_GLUON_DEEPSEEK=1,
            SGLANG_GLUON_D128_EXPERIMENTAL=0,
        ), mock.patch.object(eag, "_gluon_extend_attn_fwd_symmetric", sym), mock.patch.object(
            eag, "_get_num_CUs", return_value=256
        ):
            self._invoke(
                qk_dim=128,
                v_dim=128,
                ext_lens=[64],
                v_scale=7.0,
                force_persistent=False,
            )
        self.assertEqual(len(sym.calls), 1)
        _, kernel_kwargs = sym.calls[0]
        self.assertAlmostEqual(kernel_kwargs["V_SCALE"], 7.0)

    def test_symmetric_fp8_routes_to_symmetric_fp8_kernel(self):
        fp8_dtype = _get_fp8_dtype()
        if fp8_dtype is None:
            self.skipTest("FP8 dtype is unavailable on this torch build")
        for qk_dim in (64, 128):
            with self.subTest(qk_dim=qk_dim):
                sym = _KernelSpy()
                sym_fp8 = _KernelSpy()
                with _temp_env(
                    AITER_ENABLE_GLUON_MIXED_DIMS=0,
                    AITER_ENABLE_GLUON_DEEPSEEK=1,
                    SGLANG_GLUON_FP8_KV_FORCE_BF16=0,
                    SGLANG_GLUON_D128_EXPERIMENTAL=0,
                ), mock.patch.object(eag, "_gluon_extend_attn_fwd_symmetric", sym), mock.patch.object(
                    eag, "_gluon_extend_attn_fwd_symmetric_fp8", sym_fp8
                ), mock.patch.object(
                    eag, "_get_num_CUs", return_value=256
                ):
                    self._invoke(
                        qk_dim=qk_dim,
                        v_dim=qk_dim,
                        ext_lens=[64],
                        force_persistent=False,
                        kv_cache_dtype=fp8_dtype,
                    )
                self.assertEqual(len(sym.calls), 0)
                self.assertEqual(len(sym_fp8.calls), 1)

    def test_symmetric_fp8_bridge_forces_bf16_kernel(self):
        fp8_dtype = _get_fp8_dtype()
        if fp8_dtype is None:
            self.skipTest("FP8 dtype is unavailable on this torch build")
        sym = _KernelSpy()
        sym_fp8 = _KernelSpy()
        with _temp_env(
            AITER_ENABLE_GLUON_MIXED_DIMS=0,
            AITER_ENABLE_GLUON_DEEPSEEK=1,
            SGLANG_GLUON_FP8_KV_FORCE_BF16=1,
            SGLANG_GLUON_D128_EXPERIMENTAL=0,
        ), mock.patch.object(eag, "_gluon_extend_attn_fwd_symmetric", sym), mock.patch.object(
            eag, "_gluon_extend_attn_fwd_symmetric_fp8", sym_fp8
        ), mock.patch.object(
            eag, "_get_num_CUs", return_value=256
        ):
            self._invoke(
                qk_dim=128,
                v_dim=128,
                ext_lens=[64],
                force_persistent=False,
                kv_cache_dtype=fp8_dtype,
            )
        self.assertEqual(len(sym.calls), 1)
        self.assertEqual(len(sym_fp8.calls), 0)

    def test_mixed_dims_fp8_routes_to_deepseek_fp8_kernel(self):
        fp8_dtype = _get_fp8_dtype()
        if fp8_dtype is None:
            self.skipTest("FP8 dtype is unavailable on this torch build")
        sym = _KernelSpy()
        deep = _KernelSpy()
        deep_fp8 = _KernelSpy()
        with _temp_env(
            AITER_ENABLE_GLUON_MIXED_DIMS=1,
            AITER_ENABLE_GLUON_DEEPSEEK=1,
            SGLANG_GLUON_FP8_KV_FORCE_BF16=0,
        ), mock.patch.object(eag, "_gluon_extend_attn_fwd_symmetric", sym), mock.patch.object(
            eag, "_gluon_extend_attn_fwd_deepseek", deep
        ), mock.patch.object(
            eag, "_gluon_extend_attn_fwd_deepseek_fp8", deep_fp8
        ), mock.patch.object(
            eag, "_get_num_CUs", return_value=256
        ):
            self._invoke(
                qk_dim=192,
                v_dim=128,
                ext_lens=[64],
                kv_cache_dtype=fp8_dtype,
            )
        self.assertEqual(len(sym.calls), 0)
        self.assertEqual(len(deep.calls), 0)
        self.assertEqual(len(deep_fp8.calls), 1)

    def test_mixed_dims_fp8_bridge_forces_bf16_kernel(self):
        fp8_dtype = _get_fp8_dtype()
        if fp8_dtype is None:
            self.skipTest("FP8 dtype is unavailable on this torch build")
        deep = _KernelSpy()
        deep_fp8 = _KernelSpy()
        with _temp_env(
            AITER_ENABLE_GLUON_MIXED_DIMS=1,
            AITER_ENABLE_GLUON_DEEPSEEK=1,
            SGLANG_GLUON_FP8_KV_FORCE_BF16=1,
        ), mock.patch.object(
            eag, "_gluon_extend_attn_fwd_deepseek", deep
        ), mock.patch.object(
            eag, "_gluon_extend_attn_fwd_deepseek_fp8", deep_fp8
        ), mock.patch.object(
            eag, "_get_num_CUs", return_value=256
        ):
            self._invoke(
                qk_dim=192,
                v_dim=128,
                ext_lens=[64],
                kv_cache_dtype=fp8_dtype,
            )
        self.assertEqual(len(deep.calls), 1)
        self.assertEqual(len(deep_fp8.calls), 0)

    def test_registry_fallback_when_not_gfx950(self):
        runner = _make_registry_runner()
        sentinel = object()
        with mock.patch("sglang.srt.utils.is_hip", return_value=True), mock.patch(
            "sglang.srt.utils.is_gfx95_supported", return_value=False
        ), mock.patch(
            "sglang.srt.layers.attention.attention_registry.create_triton_backend",
            return_value=sentinel,
        ) as create_triton:
            backend = create_gluon_backend(runner)
        self.assertIs(backend, sentinel)
        create_triton.assert_called_once_with(runner)

    def test_registry_uses_force_gluon_on_gfx950(self):
        runner = _make_registry_runner()
        sentinel = object()
        with mock.patch("sglang.srt.utils.is_hip", return_value=True), mock.patch(
            "sglang.srt.utils.is_gfx95_supported", return_value=True
        ), mock.patch(
            "sglang.srt.layers.attention.triton_backend.TritonAttnBackend",
            return_value=sentinel,
        ) as triton_cls:
            backend = create_gluon_backend(runner)
        self.assertIs(backend, sentinel)
        triton_cls.assert_called_once_with(runner, force_gluon=True)


if __name__ == "__main__":
    unittest.main()

