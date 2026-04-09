from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_core_count,
    get_int_env_var,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)

_FP8_TORCH_DTYPES = {
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
}


def logit_capping_mod(logit_capping_method, logit_cap):
    # positive logit_cap -> tanh cap
    if logit_capping_method == "tanh":
        return logit_cap
    else:
        raise ValueError()


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
    # Separate attn_logits for SWA layers when v_head_dim differs
    swa_attn_logits: Optional[torch.Tensor] = None


class TritonAttnBackend(AttentionBackend):
    @staticmethod
    def _is_cuda_graph_capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        force_gluon: bool = False,
    ):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            build_unified_kv_indices,
            extend_attention_fwd,
            extend_attention_fwd_unified,
        )

        super().__init__()

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
        self.extend_attention_fwd_unified = torch.compiler.disable(
            extend_attention_fwd_unified
        )
        self.build_unified_kv_indices = torch.compiler.disable(build_unified_kv_indices)

        import os

        self._force_gluon = force_gluon
        self._use_gluon = False
        self._gluon_fn = None
        enable_gluon_extend = force_gluon or os.environ.get(
            "SGLANG_USE_GLUON_EXTEND", "0"
        ) == "1"
        if enable_gluon_extend:
            try:
                from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_entrypoints import (
                    gluon_extend_attention_fwd,
                )

                self._gluon_fn = gluon_extend_attention_fwd
                self._use_gluon = True
            except (ImportError, AttributeError):
                logger.warning(
                    "Gluon extend kernel is unavailable; falling back to triton extend."
                )

        self._gluon_mla_fn = None
        self._gluon_mla_wca_fn = None
        self._gluon_mla_fp8_fn = None
        self._gluon_mla_wca_fp8_fn = None
        self._logged_missing_gluon_fp8 = False
        self._debug_gluon_dispatch = get_bool_env_var(
            "SGLANG_DEBUG_GLUON_DISPATCH", "false"
        )
        self._logged_gluon_dispatch = set()
        self._gluon_signature_cache = {}
        self._debug_mla_path = get_bool_env_var("SGLANG_DEBUG_TRITON_MLA_PATH", "false")
        self._logged_mla_path = set()
        self._allow_non_unified_mixed_mla = get_bool_env_var(
            "SGLANG_TRITON_MLA_ALLOW_NON_UNIFIED_MIXED", "false"
        )
        self._mla_reconstruct_extend = get_bool_env_var(
            "SGLANG_TRITON_MLA_RECONSTRUCT_EXTEND", "true"
        )
        self._mla_reconstruct_decode = get_bool_env_var(
            "SGLANG_TRITON_MLA_RECONSTRUCT_DECODE", "true"
        )
        self._debug_mla_recon_compare = get_bool_env_var(
            "SGLANG_DEBUG_MLA_RECON_COMPARE", "false"
        )
        self._debug_kv_state = get_bool_env_var("SGLANG_DEBUG_TRITON_KV_STATE", "false")
        self._kv_seen_write_locs: dict[int, set[int]] = {}
        self._debug_shadow_aiter = get_bool_env_var(
            "SGLANG_DEBUG_TRITON_SHADOW_AITER", "false"
        )
        self._debug_asserts = get_bool_env_var("SGLANG_DEBUG_TRITON_ASSERTS", "false")
        self._debug_ctrl_flow = get_bool_env_var("SGLANG_DEBUG_ATTN_CTRL_FLOW", "false")
        self._ctrl_flow_seen: set[tuple] = set()
        self._force_mla_no_save_kv = get_bool_env_var(
            "SGLANG_TRITON_MLA_FORCE_NO_SAVE_KV", "false"
        )
        self._shadow_layer = get_int_env_var("SGLANG_DEBUG_TRITON_SHADOW_LAYER", -1)
        self._shadow_max_calls = get_int_env_var("SGLANG_DEBUG_TRITON_SHADOW_MAX_CALLS", 4)
        self._shadow_calls: dict[tuple[str, int], int] = {}
        self._shadow_flash_attn_varlen = None
        if self._debug_shadow_aiter:
            try:
                from aiter import flash_attn_varlen_func

                self._shadow_flash_attn_varlen = flash_attn_varlen_func
            except ImportError:
                logger.warning(
                    "SGLANG_DEBUG_TRITON_SHADOW_AITER=1 but aiter flash_attn_varlen_func "
                    "is unavailable; shadow compare disabled."
                )
                self._debug_shadow_aiter = False
        gluon_mla_default = "hybrid" if force_gluon else "0"
        gluon_mla_mode = os.environ.get("SGLANG_GLUON_MLA", gluon_mla_default).lower()
        if gluon_mla_mode in ("gluon", "hybrid"):
            try:
                from sglang.srt.layers.attention.gluon_ops.CDNA4.f16_mla_prefill import (
                    mla_d512_gqa_attention_fwd,
                    mla_d512_gqa_attention_fwd_wca,
                )

                self._gluon_mla_fn = mla_d512_gqa_attention_fwd
                self._gluon_mla_wca_fn = mla_d512_gqa_attention_fwd_wca
            except (ImportError, AttributeError):
                logger.warning(
                    "Gluon MLA D576 kernels unavailable; "
                    "falling back to triton for MLA prefill."
                )

            try:
                from sglang.srt.layers.attention.gluon_ops.CDNA4.fp8_mla_prefill import (
                    mla_d512_gqa_attention_fwd_fp8,
                    mla_d512_gqa_attention_fwd_wca_fp8,
                )

                self._gluon_mla_fp8_fn = mla_d512_gqa_attention_fwd_fp8
                self._gluon_mla_wca_fp8_fn = mla_d512_gqa_attention_fwd_wca_fp8
            except (ImportError, AttributeError):
                self._gluon_mla_fp8_fn = None
                self._gluon_mla_wca_fp8_fn = None

        # Parse args
        self.skip_prefill = skip_prefill
        max_bs = model_runner.req_to_token_pool.size
        self.sliding_window_size = model_runner.sliding_window_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        hf_config = getattr(model_runner.model_config, "hf_config", None)
        self._mla_kv_lora_rank = (
            getattr(hf_config, "kv_lora_rank", None) if hf_config is not None else None
        )
        self._mla_qk_rope_head_dim = (
            getattr(hf_config, "qk_rope_head_dim", None)
            if hf_config is not None
            else None
        )
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        # The decode triton kernel derives attn_lse offsets from attn_logits
        # strides via integer division by v_head_dim (the "// Lv" trick in
        # _fwd_kernel_stage1/stage2), so attn_logits.shape[-1] must exactly
        # match the layer's v_head_dim. For hybrid SWA models where SWA and
        # full-attention layers use different v_head_dim (e.g. Gemma 4:
        # swa=256, full=512), we allocate a second buffer for SWA layers.
        full_v_head_dim = model_runner.model_config.v_head_dim
        swa_v_head_dim = model_runner.model_config.swa_v_head_dim
        if self.sliding_window_size is not None and swa_v_head_dim != full_v_head_dim:
            self.v_head_dim = full_v_head_dim
            self.swa_v_head_dim = swa_v_head_dim
        elif (
            model_runner.hybrid_gdn_config is not None
            or model_runner.kimi_linear_config is not None
            or model_runner.linear_attn_model_spec is not None
        ):
            # For hybrid linear models, layer_id = 0 may not be full attention
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
            self.swa_v_head_dim = None
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]
            self.swa_v_head_dim = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.device_core_count = get_device_core_count(model_runner.gpu_id)
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits

        self.allow_bidirectional_attention_in_extend = (
            model_runner.server_args.disable_cuda_graph
            and (model_runner.server_args.chunked_prefill_size == -1)
        )

        # Decide whether enable deterministic inference with batch-invariant operations
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )

        # Configure deterministic inference settings
        if self.enable_deterministic:
            # Use fixed split tile size for batch invariance
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256
            )
            # Set static_kv_splits to False to use deterministic logic instead
            self.static_kv_splits = False
        else:
            self.split_tile_size = (
                model_runner.server_args.triton_attention_split_tile_size
            )

        if self.split_tile_size is not None:
            self.max_kv_splits = (
                self.max_context_len + self.split_tile_size - 1
            ) // self.split_tile_size

        # Check arguments
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # Initialize buffers
        # TODO(Jianan Ji): Make sure it behaves as expected when kv_indptr_buf is provided and sliding window is enabled
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # If sliding window is enabled, we might need two sets of buffers
        # because of interleaved attention types (e.g. for Gemma3)
        self.window_kv_indptr = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indptr_buf is None:
                self.window_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
            else:
                # When provided a buffer, create a clone for the second buffer
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        # Initialize forward metadata
        self.forward_metadata: ForwardMetadata = None

        self.cuda_graph_custom_mask = None

    def _needs_mla_kv_reconstruction(self, layer: RadixAttention) -> bool:
        if not self.use_mla:
            return False
        if self._mla_kv_lora_rank is None or self._mla_qk_rope_head_dim is None:
            return False
        return layer.qk_head_dim != (
            self._mla_kv_lora_rank + self._mla_qk_rope_head_dim
        )

    def _maybe_log_ctrl_flow(
        self,
        stage: str,
        path: str,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        kernel: str = "",
        save_kv_cache: Optional[bool] = None,
        needs_reconstruct: Optional[bool] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        extra: str = "",
    ) -> None:
        if not self._debug_ctrl_flow or self._is_cuda_graph_capturing():
            return
        mode_name = (
            forward_batch.forward_mode.name
            if hasattr(forward_batch.forward_mode, "name")
            else str(forward_batch.forward_mode)
        )
        kv_last = -1
        kv_n = -1
        if kv_indptr is not None and kv_indptr.numel() > 0:
            kv_last = int(kv_indptr[-1].item())
        if kv_indices is not None:
            kv_n = int(kv_indices.numel())
        key = (
            stage,
            path,
            mode_name,
            layer.layer_id,
            kernel,
            save_kv_cache,
            needs_reconstruct,
            layer.qk_head_dim,
            layer.v_head_dim,
            kv_last,
            kv_n,
            extra,
        )
        if key in self._ctrl_flow_seen:
            return
        self._ctrl_flow_seen.add(key)
        logger.info(
            "ATTN_CTRL backend=triton stage=%s path=%s kernel=%s mode=%s layer=%s "
            "qk_dim=%s v_dim=%s save_kv=%s needs_reconstruct=%s kv_last=%s kv_n=%s %s",
            stage,
            path,
            kernel,
            mode_name,
            layer.layer_id,
            layer.qk_head_dim,
            layer.v_head_dim,
            save_kv_cache,
            needs_reconstruct,
            kv_last,
            kv_n,
            extra,
        )

    def _reconstruct_mla_kv_from_latent(
        self,
        layer: RadixAttention,
        kv_indices: torch.Tensor,
        target_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if kv_indices.numel() == 0:
            tp_k_heads = getattr(layer, "tp_k_head_num", layer.tp_q_head_num)
            k = torch.empty(
                (0, tp_k_heads, layer.qk_head_dim),
                dtype=target_dtype,
                device=self.device,
            )
            v = torch.empty(
                (0, tp_k_heads, layer.v_head_dim),
                dtype=target_dtype,
                device=self.device,
            )
            remapped_indices = torch.empty(0, dtype=torch.int64, device=self.device)
            return k, v, remapped_indices

        kv_lora_rank = self._mla_kv_lora_rank
        qk_rope_head_dim = self._mla_qk_rope_head_dim
        qk_nope_head_dim = getattr(
            layer, "qk_nope_head_dim", layer.qk_head_dim - qk_rope_head_dim
        )
        tp_k_heads = getattr(layer, "tp_k_head_num", layer.tp_q_head_num)

        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        selected = torch.index_select(kv_cache, 0, kv_indices.to(torch.long))
        kvc, k_pe = torch.split(selected, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        if kvc.dtype in _FP8_TORCH_DTYPES and target_dtype not in _FP8_TORCH_DTYPES:
            kvc = kvc.to(target_dtype)
            k_pe = k_pe.to(target_dtype)

        kv_b_weight = getattr(layer.kv_b_proj, "weight", None)
        if kv_b_weight is not None and kv_b_weight.dtype == torch.uint8:
            fp8_out_dtype = getattr(torch, "float8_e4m3fn", target_dtype)
            k, v = layer.kv_b_proj(
                (
                    kvc.squeeze(1),
                    k_pe.expand(-1, tp_k_heads, -1),
                    qk_nope_head_dim,
                    layer.v_head_dim,
                    fp8_out_dtype,
                )
            )[0]
            if target_dtype not in _FP8_TORCH_DTYPES:
                if k.dtype in _FP8_TORCH_DTYPES:
                    k = k.to(target_dtype)
                if v.dtype in _FP8_TORCH_DTYPES:
                    v = v.to(target_dtype)
        else:
            kv = layer.kv_b_proj(kvc.contiguous())[0]
            kv = kv.view(-1, tp_k_heads, qk_nope_head_dim + layer.v_head_dim)
            k_nope, v = torch.split(kv, [qk_nope_head_dim, layer.v_head_dim], dim=-1)
            k = torch.cat(
                [k_nope, k_pe.expand(-1, tp_k_heads, -1)],
                dim=-1,
            )
        k = k.contiguous()
        v = v.contiguous()
        remapped_indices = torch.arange(k.shape[0], dtype=torch.int64, device=k.device)
        return k, v, remapped_indices

    @staticmethod
    def _tensor_diff_stats_full(
        a: torch.Tensor, b: torch.Tensor
    ) -> tuple[float, float, float, float, float, float]:
        af = a.float().reshape(-1)
        bf = b.float().reshape(-1)
        d = af - bf
        max_abs = float(d.abs().max().item())
        mean_abs = float(d.abs().mean().item())
        a_norm = float(af.norm().item())
        b_norm = float(bf.norm().item())
        denom = max(a_norm * b_norm, 1e-12)
        cosine = float((af @ bf).item() / denom)
        rel_l2 = float(d.norm().item() / max(b_norm, 1e-12))
        return max_abs, mean_abs, cosine, rel_l2, a_norm, b_norm

    @staticmethod
    def _tensor_diff_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
        max_abs, mean_abs, cosine, _, _, _ = TritonAttnBackend._tensor_diff_stats_full(a, b)
        return max_abs, mean_abs, cosine

    def _maybe_log_mla_recon_compare(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        k_ref: torch.Tensor,
        v_ref: torch.Tensor,
    ) -> None:
        if self._is_cuda_graph_capturing():
            return
        if not self._debug_mla_recon_compare:
            return
        if not self._needs_mla_kv_reconstruction(layer):
            return
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None or out_cache_loc.numel() == 0:
            return
        tp_k_heads = getattr(layer, "tp_k_head_num", layer.tp_q_head_num)
        k_ref = k_ref.view(-1, tp_k_heads, layer.qk_head_dim).contiguous()
        v_ref = v_ref.view(-1, tp_k_heads, layer.v_head_dim).contiguous()
        k_cmp, v_cmp, _ = self._reconstruct_mla_kv_from_latent(
            layer=layer,
            kv_indices=out_cache_loc.to(torch.int64),
            target_dtype=k_ref.dtype,
            forward_batch=forward_batch,
        )
        n = min(k_ref.shape[0], k_cmp.shape[0], v_ref.shape[0], v_cmp.shape[0])
        if n <= 0:
            return
        k_max, k_mean, k_cos = self._tensor_diff_stats(k_ref[:n], k_cmp[:n])
        v_max, v_mean, v_cos = self._tensor_diff_stats(v_ref[:n], v_cmp[:n])
        mode_name = (
            forward_batch.forward_mode.name
            if hasattr(forward_batch.forward_mode, "name")
            else str(forward_batch.forward_mode)
        )
        pos = getattr(forward_batch, "positions", None)
        if pos is not None and pos.numel() > 0:
            pos_min = int(pos.min().item())
            pos_max = int(pos.max().item())
        else:
            pos_min = -1
            pos_max = -1
        logger.info(
            "MLA_RECON_COMPARE mode=%s layer=%s pos=[%s,%s] "
            "k_max=%.6f k_mean=%.6f k_cos=%.8f "
            "v_max=%.6f v_mean=%.6f v_cos=%.8f",
            mode_name,
            layer.layer_id,
            pos_min,
            pos_max,
            k_max,
            k_mean,
            k_cos,
            v_max,
            v_mean,
            v_cos,
        )

    def _maybe_log_kv_state(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        stage: str,
    ) -> None:
        if self._is_cuda_graph_capturing():
            return
        if not self._debug_kv_state:
            return
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None or out_cache_loc.numel() == 0:
            return
        loc_i64 = out_cache_loc.to(torch.long)
        loc_list = loc_i64.tolist()
        unique_locs = set(int(x) for x in loc_list)
        dup_in_batch = len(loc_list) - len(unique_locs)

        seen = self._kv_seen_write_locs.setdefault(layer.layer_id, set())
        repeat_write = sum(1 for x in unique_locs if x in seen)
        seen.update(unique_locs)

        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        k_sel = torch.index_select(k_buf, 0, loc_i64)
        v_sel = torch.index_select(v_buf, 0, loc_i64)
        kf = k_sel.float().reshape(-1)
        vf = v_sel.float().reshape(-1)

        mode_name = (
            forward_batch.forward_mode.name
            if hasattr(forward_batch.forward_mode, "name")
            else str(forward_batch.forward_mode)
        )
        pos = getattr(forward_batch, "positions", None)
        if pos is not None and pos.numel() > 0:
            pos_min = int(pos.min().item())
            pos_max = int(pos.max().item())
        else:
            pos_min = -1
            pos_max = -1

        logger.info(
            "TRITON_KV_STATE stage=%s mode=%s layer=%s pos=[%s,%s] "
            "loc_n=%s loc_min=%s loc_max=%s dup_in_batch=%s repeat_write=%s "
            "k_mean_abs=%.6f v_mean_abs=%.6f",
            stage,
            mode_name,
            layer.layer_id,
            pos_min,
            pos_max,
            len(loc_list),
            min(unique_locs) if unique_locs else -1,
            max(unique_locs) if unique_locs else -1,
            dup_in_batch,
            repeat_write,
            float(kf.abs().mean().item()) if kf.numel() > 0 else 0.0,
            float(vf.abs().mean().item()) if vf.numel() > 0 else 0.0,
        )

    def _maybe_shadow_compare_aiter(
        self,
        stage: str,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        k_current: Optional[torch.Tensor],
        v_current: Optional[torch.Tensor],
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        o_triton: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        sm_scale: float,
        causal: bool,
    ) -> None:
        if self._is_cuda_graph_capturing():
            return
        if not self._debug_shadow_aiter or self._shadow_flash_attn_varlen is None:
            return
        if self._shadow_layer >= 0 and layer.layer_id != self._shadow_layer:
            return
        if stage == "extend":
            # flash_attn_varlen reference does not model custom speculative masks.
            if (
                self.forward_metadata.custom_mask is not None
                or not forward_batch.forward_mode.is_extend_without_speculative()
            ):
                return
        key = (stage, layer.layer_id)
        count = self._shadow_calls.get(key, 0)
        if count >= self._shadow_max_calls:
            return
        self._shadow_calls[key] = count + 1

        q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        q_ref = q3.to(torch.bfloat16)
        if kv_indices.numel() == 0:
            # No-prefix extend uses current K/V tokens directly.
            if k_current is None or v_current is None:
                return
            k_ref = k_current.to(torch.bfloat16).contiguous()
            v_ref = v_current.to(torch.bfloat16).contiguous()
            kv_indptr_i32 = self.forward_metadata.qo_indptr.to(torch.int32)
        else:
            active_n = int(kv_indptr[-1].item()) if kv_indptr.numel() > 0 else kv_indices.numel()
            active_n = max(0, min(active_n, kv_indices.numel()))
            kv_indices_active = kv_indices[:active_n]
            # Prefix path: gather explicit per-token K/V in kv_indices order.
            k_ref = torch.index_select(k_buffer, 0, kv_indices_active.to(torch.long)).to(
                torch.bfloat16
            )
            v_ref = torch.index_select(v_buffer, 0, kv_indices_active.to(torch.long)).to(
                torch.bfloat16
            )
            kv_indptr_i32 = kv_indptr.to(torch.int32)
        q_norm = float(q_ref.float().norm().item())
        k_norm = float(k_ref.float().norm().item())
        v_norm = float(v_ref.float().norm().item())

        if stage == "decode":
            bs = q_ref.shape[0]
            qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=q_ref.device)
            max_q_len = 1
        else:
            qo_indptr = self.forward_metadata.qo_indptr.to(torch.int32)
            q_lens = qo_indptr[1:] - qo_indptr[:-1]
            max_q_len = int(q_lens.max().item()) if q_lens.numel() > 0 else 0
        kv_lens = kv_indptr_i32[1:] - kv_indptr_i32[:-1]
        max_kv_len = int(kv_lens.max().item()) if kv_lens.numel() > 0 else 0

        o_ref = self._shadow_flash_attn_varlen(
            q_ref,
            k_ref,
            v_ref,
            qo_indptr,
            kv_indptr_i32,
            max_q_len,
            max_kv_len,
            softmax_scale=sm_scale,
            causal=causal,
        )
        o_ref = o_ref.to(o_triton.dtype)
        max_abs, mean_abs, cosine, rel_l2, triton_norm, aiter_norm = (
            self._tensor_diff_stats_full(o_triton, o_ref)
        )
        mode_name = (
            forward_batch.forward_mode.name
            if hasattr(forward_batch.forward_mode, "name")
            else str(forward_batch.forward_mode)
        )
        pos = getattr(forward_batch, "positions", None)
        if pos is not None and pos.numel() > 0:
            pos_min = int(pos.min().item())
            pos_max = int(pos.max().item())
        else:
            pos_min = -1
            pos_max = -1
        logger.info(
            "TRITON_SHADOW_AITER stage=%s mode=%s layer=%s pos=[%s,%s] "
            "max_abs=%.6f mean_abs=%.6f cosine=%.8f rel_l2=%.8f "
            "triton_norm=%.6f aiter_norm=%.6f "
            "q_norm=%.6f k_norm=%.6f v_norm=%.6f "
            "q_n=%s kv_n=%s max_q_len=%s max_kv_len=%s "
            "q_dtype=%s k_dtype=%s v_dtype=%s call=%s/%s",
            stage,
            mode_name,
            layer.layer_id,
            pos_min,
            pos_max,
            max_abs,
            mean_abs,
            cosine,
            rel_l2,
            triton_norm,
            aiter_norm,
            q_norm,
            k_norm,
            v_norm,
            q_ref.shape[0],
            k_ref.shape[0] if k_ref is not None else 0,
            max_q_len,
            max_kv_len,
            str(q_ref.dtype),
            str(k_ref.dtype) if k_ref is not None else "None",
            str(v_ref.dtype) if v_ref is not None else "None",
            count + 1,
            self._shadow_max_calls,
        )
        if aiter_norm < 1e-9:
            logger.warning(
                "TRITON_SHADOW_AITER_ZERO_REF stage=%s mode=%s layer=%s "
                "q_n=%s kv_n=%s max_q_len=%s max_kv_len=%s",
                stage,
                mode_name,
                layer.layer_id,
                q_ref.shape[0],
                k_ref.shape[0] if k_ref is not None else 0,
                max_q_len,
                max_kv_len,
            )

    def _assert_attention_debug_state(
        self,
        stage: str,
        layer: RadixAttention,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        qo_indptr: Optional[torch.Tensor] = None,
    ) -> None:
        if self._is_cuda_graph_capturing():
            return
        if not self._debug_asserts:
            return
        layer_id = layer.layer_id
        q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if q3.shape[-1] != layer.qk_head_dim:
            raise RuntimeError(
                f"[TRITON_ASSERT] stage={stage} layer={layer_id} "
                f"q_head_dim={q3.shape[-1]} expected={layer.qk_head_dim}"
            )
        if k_buffer.shape[-1] != layer.qk_head_dim:
            raise RuntimeError(
                f"[TRITON_ASSERT] stage={stage} layer={layer_id} "
                f"k_head_dim={k_buffer.shape[-1]} expected={layer.qk_head_dim}"
            )
        if v_buffer.shape[-1] != layer.v_head_dim:
            raise RuntimeError(
                f"[TRITON_ASSERT] stage={stage} layer={layer_id} "
                f"v_head_dim={v_buffer.shape[-1]} expected={layer.v_head_dim}"
            )
        if not torch.isfinite(q3.float()).all():
            raise RuntimeError(
                f"[TRITON_ASSERT] stage={stage} layer={layer_id} q has non-finite values"
            )

        if kv_indptr.numel() > 1:
            if not torch.all(kv_indptr[1:] >= kv_indptr[:-1]):
                raise RuntimeError(
                    f"[TRITON_ASSERT] stage={stage} layer={layer_id} "
                    "kv_indptr is not monotonic"
                )
        if kv_indices.numel() > 0:
            active_n = int(kv_indptr[-1].item()) if kv_indptr.numel() > 0 else kv_indices.numel()
            active_n = max(0, min(active_n, kv_indices.numel()))
            active_indices = kv_indices[:active_n] if active_n > 0 else kv_indices[:0]
            if active_indices.numel() > 0:
                idx_min = int(active_indices.min().item())
                idx_max = int(active_indices.max().item())
            else:
                idx_min = 0
                idx_max = -1
            if idx_min < 0 or idx_max >= k_buffer.shape[0]:
                raise RuntimeError(
                    f"[TRITON_ASSERT] stage={stage} layer={layer_id} "
                    f"kv_indices out of range min={idx_min} max={idx_max} "
                    f"buf_n={k_buffer.shape[0]}"
                )
        if kv_indptr.numel() > 0:
            kv_last = int(kv_indptr[-1].item())
            if stage == "extend":
                expect_equal = kv_last == kv_indices.numel()
            else:
                # decode path may use preallocated kv_indices buffers for cuda-graph capture.
                expect_equal = kv_last <= kv_indices.numel()
            if not expect_equal:
                raise RuntimeError(
                    f"[TRITON_ASSERT] stage={stage} layer={layer_id} "
                    f"kv_indptr[-1]={kv_last} != kv_indices_n={kv_indices.numel()}"
                )
        if qo_indptr is not None:
            if qo_indptr.numel() > 1 and not torch.all(qo_indptr[1:] >= qo_indptr[:-1]):
                raise RuntimeError(
                    f"[TRITON_ASSERT] stage={stage} layer={layer_id} "
                    "qo_indptr is not monotonic"
                )
            if qo_indptr.numel() > 0 and int(qo_indptr[-1].item()) != q3.shape[0]:
                raise RuntimeError(
                    f"[TRITON_ASSERT] stage={stage} layer={layer_id} "
                    f"qo_indptr[-1]={int(qo_indptr[-1].item())} != q_n={q3.shape[0]}"
                )

        # Validate gathered K/V slices used by attention.
        if kv_indices.numel() > 0:
            active_n = int(kv_indptr[-1].item()) if kv_indptr.numel() > 0 else kv_indices.numel()
            active_n = max(0, min(active_n, kv_indices.numel()))
            active_indices = kv_indices[:active_n]
            kg = torch.index_select(k_buffer, 0, active_indices.to(torch.long))
            vg = torch.index_select(v_buffer, 0, active_indices.to(torch.long))
            if not torch.isfinite(kg.float()).all():
                raise RuntimeError(
                    f"[TRITON_ASSERT] stage={stage} layer={layer_id} gathered K has non-finite values"
                )
            if not torch.isfinite(vg.float()).all():
                raise RuntimeError(
                    f"[TRITON_ASSERT] stage={stage} layer={layer_id} gathered V has non-finite values"
                )

    def _should_use_gluon_extend(
        self,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        layer: RadixAttention,
        force_triton_fallback: bool,
    ) -> bool:
        if force_triton_fallback:
            return False
        if not self._use_gluon or self._gluon_fn is None:
            return False
        if self.enable_deterministic:
            return False
        if not forward_batch.forward_mode.is_extend():
            return False
        # Gluon extend kernels are tuned/validated for these dhead families.
        is_supported_shape = (
            (layer.qk_head_dim == layer.v_head_dim and layer.qk_head_dim in (64, 128))
            or (layer.qk_head_dim == 192 and layer.v_head_dim in (128, 192))
        )
        if not is_supported_shape:
            return False
        # FP8 KV cache paths still use bf16/fp16 Q in the model code paths.
        return q.dtype in (torch.float16, torch.bfloat16)

    def _get_gluon_extend_stats(
        self,
        forward_batch: ForwardBatch,
        q_tokens: int,
    ) -> tuple[int, int, int]:
        cached = getattr(forward_batch, "_gluon_extend_stats", None)
        if cached is not None:
            return cached
        ext_lens = forward_batch.extend_seq_lens_cpu or []
        pfx_lens = forward_batch.extend_prefix_lens_cpu or []
        min_ext = min(ext_lens) if ext_lens else self.forward_metadata.max_extend_len
        total_ext = sum(ext_lens) if ext_lens else q_tokens
        total_pfx = sum(pfx_lens) if pfx_lens else 0
        cached = (min_ext, total_ext, total_pfx)
        setattr(forward_batch, "_gluon_extend_stats", cached)
        return cached

    def _log_gluon_dispatch(
        self,
        path: str,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        extra: str = "",
    ) -> None:
        if not self._debug_gluon_dispatch:
            return
        mode_name = (
            forward_batch.forward_mode.name
            if hasattr(forward_batch.forward_mode, "name")
            else str(forward_batch.forward_mode)
        )
        key = (
            path,
            mode_name,
            layer.layer_id,
            layer.qk_head_dim,
            layer.v_head_dim,
            str(self.kv_cache_dtype),
            extra,
        )
        if key in self._logged_gluon_dispatch:
            return
        self._logged_gluon_dispatch.add(key)
        logger.info(
            "GLUON_DISPATCH path=%s mode=%s layer=%s qk_dim=%s v_dim=%s kv_cache_dtype=%s %s",
            path,
            mode_name,
            layer.layer_id,
            layer.qk_head_dim,
            layer.v_head_dim,
            self.kv_cache_dtype,
            extra,
        )

    def _gluon_fn_supports_kwarg(self, fn, kwarg: str) -> bool:
        """Return whether a Gluon wrapper accepts a keyword argument."""
        key = (id(fn), kwarg)
        if key in self._gluon_signature_cache:
            return self._gluon_signature_cache[key]
        try:
            supported = kwarg in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            supported = False
        self._gluon_signature_cache[key] = supported
        return supported

    def _try_gluon_mla_prefill(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ) -> tuple[bool, bool]:
        """Try D576 Gluon MLA prefill. Returns (handled, force_triton_fallback)."""
        if (
            not self.use_mla
            or not self._use_gluon
            or self.enable_deterministic
            or not forward_batch.forward_mode.is_extend_without_speculative()
            or layer.qk_head_dim != 576
            or layer.v_head_dim != 512
            or self.forward_metadata is None
            or self.forward_metadata.qo_indptr is None
        ):
            return False, False
        if (
            (layer.sliding_window_size is not None and layer.sliding_window_size > -1)
            or sinks is not None
            or self.forward_metadata.custom_mask is not None
        ):
            return False, False

        kv_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_is_fp8 = kv_buffer.dtype in _FP8_TORCH_DTYPES or (
            self.kv_cache_dtype in _FP8_TORCH_DTYPES
        )
        max_q_len = self.forward_metadata.max_extend_len
        qo_indptr = self.forward_metadata.qo_indptr
        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices

        bs = qo_indptr.shape[0] - 1
        n_m_tiles = (max_q_len + 64 - 1) // 64
        total_output_tiles = bs * self.num_head * n_m_tiles
        max_prefix = getattr(forward_batch, "_gluon_max_prefix_len", None)
        if max_prefix is None:
            prefix_lens = forward_batch.extend_prefix_lens_cpu or []
            max_prefix = max(prefix_lens) if prefix_lens else 0
            setattr(forward_batch, "_gluon_max_prefix_len", max_prefix)
        use_wca = max_prefix >= 512 and total_output_tiles < self.device_core_count

        q3 = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o3 = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        if kv_is_fp8:
            if self._gluon_mla_fp8_fn is None:
                if self._force_gluon and not self._logged_missing_gluon_fp8:
                    logger.warning(
                        "FP8 KV cache detected but Gluon FP8 MLA kernels are unavailable; "
                        "falling back to triton kernels for this path."
                    )
                    self._logged_missing_gluon_fp8 = True
                return False, True

            gluon_fn = (
                self._gluon_mla_wca_fp8_fn
                if use_wca and self._gluon_mla_wca_fp8_fn is not None
                else self._gluon_mla_fp8_fn
            )
            k_scale = layer.k_scale_float if layer.k_scale is not None else 1.0
            v_scale = layer.v_scale_float if layer.v_scale is not None else 1.0
            gluon_kwargs = {
                "max_len_extend": max_q_len,
                "is_causal": True,
                "sm_scale": layer.scaling,
            }
            if self._gluon_fn_supports_kwarg(gluon_fn, "logit_cap"):
                gluon_kwargs["logit_cap"] = logits_soft_cap
            if self._gluon_fn_supports_kwarg(gluon_fn, "k_scale"):
                gluon_kwargs["k_scale"] = k_scale
            if self._gluon_fn_supports_kwarg(gluon_fn, "v_scale"):
                gluon_kwargs["v_scale"] = v_scale
            gluon_fn(
                q3,
                kv_buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                o3,
                qo_indptr,
                kv_indptr,
                kv_indices,
                **gluon_kwargs,
            )
            self._log_gluon_dispatch(
                path=(
                    "mla_fp8_wca"
                    if use_wca and gluon_fn is self._gluon_mla_wca_fp8_fn
                    else "mla_fp8_4w"
                ),
                layer=layer,
                forward_batch=forward_batch,
                extra=f"k_scale={k_scale} v_scale={v_scale}",
            )
            return True, False

        if self._gluon_mla_fn is None:
            return False, False

        compute_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        gluon_fn = (
            self._gluon_mla_wca_fn
            if use_wca and self._gluon_mla_wca_fn is not None
            else self._gluon_mla_fn
        )
        gluon_kwargs = {
            "max_len_extend": max_q_len,
            "is_causal": True,
            "sm_scale": layer.scaling,
        }
        if self._gluon_fn_supports_kwarg(gluon_fn, "logit_cap"):
            gluon_kwargs["logit_cap"] = logits_soft_cap
        gluon_fn(
            q3.to(compute_dtype),
            kv_buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim).to(compute_dtype),
            o3,
            qo_indptr,
            kv_indptr,
            kv_indices,
            **gluon_kwargs,
        )
        self._log_gluon_dispatch(
            path=(
                "mla_bf16_wca"
                if use_wca and gluon_fn is self._gluon_mla_wca_fn
                else "mla_bf16_4w"
            ),
            layer=layer,
            forward_batch=forward_batch,
        )
        return True, False

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        # Legacy dynamic splitting logic (non-deterministic)
        if (
            self.static_kv_splits or self.device_core_count <= 0
        ) and not self.enable_deterministic:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        # deterministic
        if self.split_tile_size is not None and self.enable_deterministic:
            # expand seq_lens to match num_token
            if num_group > 1:
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)
            else:
                expanded_seq_lens = seq_lens

            num_kv_splits[:] = (
                expanded_seq_lens + self.split_tile_size - 1
            ) // self.split_tile_size
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        swa_attn_logits = None
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    forward_batch.seq_lens_sum, dtype=torch.int64, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                # Sliding window
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            self.req_to_token,
                            self.sliding_window_size,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            bs,
                            self.device,
                            self.token_to_kv_pool_allocator,
                        )
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if self.swa_v_head_dim is not None:
                swa_attn_logits = torch.empty(
                    (bs, self.num_head, self.max_kv_splits, self.swa_v_head_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                swa_attn_logits = None
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            self.get_num_kv_splits(num_kv_splits, forward_batch.seq_lens)

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            bs = len(forward_batch.req_pool_indices)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # Different with flashinfer kv_indptr and kv_indices construction
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_indptr[-1], dtype=torch.int64, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                # window_kv_offsets is used to calculate the start position in custom mask
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool_allocator,
                )

            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.seq_lens + self.num_draft_tokens
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        elif forward_batch.forward_mode.is_draft_extend():
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    self.req_to_token,
                )
            )
            kv_indices = kv_indices.to(torch.int64)
            mask_indptr = None
            # TODO(FIXME): This will trigger an invalid Eagle tree when using
            # `max(spec_info.accept_length_cpu)`.
            # It might have been forgotten to update somewhere.
            max_extend_len = torch.max(spec_info.accept_length).item()
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            kv_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                sum(forward_batch.extend_prefix_lens_cpu),
                dtype=torch.int64,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            # Sliding window
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.extend_prefix_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool_allocator,
                )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            max_extend_len = max(forward_batch.extend_seq_lens_cpu)
            num_kv_splits = None

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
            swa_attn_logits=swa_attn_logits,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if self.swa_v_head_dim is not None:
            self.cuda_graph_swa_attn_logits = torch.zeros(
                (
                    max_num_tokens,
                    self.num_head,
                    self.max_kv_splits,
                    self.swa_v_head_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.cuda_graph_swa_attn_logits = None
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )

        if cuda_graph_num_kv_splits_buf is None:
            self.cuda_graph_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_num_kv_splits = cuda_graph_num_kv_splits_buf

        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_num_tokens * self.sliding_window_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
                device=self.device,
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        assert encoder_lens is None, "Not supported"
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        swa_attn_logits = None

        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indptr, window_kv_indices, _, _ = (
                        update_sliding_window_buffer_cuda_graph(
                            self.window_kv_indptr,
                            window_kv_indices,
                            self.req_to_token,
                            self.sliding_window_size,
                            seq_lens[:bs],
                            req_pool_indices,
                            bs,
                            self.token_to_kv_pool_allocator,
                        )
                    )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            attn_logits = self.cuda_graph_attn_logits
            swa_attn_logits = self.cuda_graph_swa_attn_logits
            attn_lse = self.cuda_graph_attn_lse
            max_extend_len = None
            num_kv_splits = self.cuda_graph_num_kv_splits
            qo_indptr = None
            custom_mask = None
            mask_indptr = None
        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )

            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        elif forward_mode.is_draft_extend(include_v2=True):
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            custom_mask = None
            mask_indptr = None
            max_extend_len = num_tokens_per_bs
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph capture."
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
            swa_attn_logits=swa_attn_logits,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            # Update kv_indptr, kv_indices
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            num_kv_splits = self.cuda_graph_num_kv_splits
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                num_token = bs
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    _, _, window_kv_lens, _ = update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices[:bs],
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                    self.get_num_kv_splits(
                        window_num_kv_splits[:num_token], window_kv_lens[:bs]
                    )

            else:
                assert False, "Multi-step cuda graph init is not done here."
            self.get_num_kv_splits(num_kv_splits[:num_token], seq_lens[:bs])

        elif forward_mode.is_target_verify():
            # Update qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                _, _, window_kv_lens, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )
            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        elif forward_mode.is_draft_extend(include_v2=True):
            seq_lens = seq_lens[:bs]
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [self.cuda_graph_custom_mask, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        needs_mla_kv_reconstruct = (
            self._mla_reconstruct_extend and self._needs_mla_kv_reconstruction(layer)
        )
        if needs_mla_kv_reconstruct:
            save_kv_cache = False
            self._maybe_log_mla_recon_compare(layer, forward_batch, k, v)
        if self._force_mla_no_save_kv and self.use_mla:
            save_kv_cache = False
        self._maybe_log_ctrl_flow(
            stage="extend",
            path="entry",
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            needs_reconstruct=needs_mla_kv_reconstruct,
            extra=f"use_mla={self.use_mla} deterministic={self.enable_deterministic}",
        )

        if k is None and v is None:
            pool = forward_batch.token_to_kv_pool
            cache_loc = forward_batch.out_cache_loc
            if isinstance(pool, SWAKVPool) and pool.layers_mapping[layer.layer_id][1]:
                cache_loc = pool.translate_loc_from_full_to_swa(cache_loc)
            k_buffer, v_buffer = pool.get_kv_buffer(layer.layer_id)
            k = k_buffer[cache_loc]
            v = v_buffer[cache_loc]
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            if save_kv_cache:
                if (
                    self.use_mla or layer.k_scale is None
                ):
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k,
                        v,
                    )
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer,
                        forward_batch.out_cache_loc,
                        k.clone(),
                        v.clone(),
                        layer.k_scale,
                        layer.v_scale,
                    )
        self._maybe_log_kv_state(layer, forward_batch, stage="extend_after_write")

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
            or (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
                and self.allow_bidirectional_attention_in_extend
            )
        ):
            causal = False

        gluon_handled, force_triton_fallback = self._try_gluon_mla_prefill(
            q=q,
            o=o,
            layer=layer,
            forward_batch=forward_batch,
            logits_soft_cap=logits_soft_cap,
            sinks=sinks,
        )
        if gluon_handled:
            return o

        # Deterministic mode uses unified kernel. By default we also force
        # unified for MLA mixed-dimension heads (e.g., DeepSeek D576/D512).
        # The env flag SGLANG_TRITON_MLA_ALLOW_NON_UNIFIED_MIXED=1 is a debug
        # override to analyze quality/perf differences in this routing choice.
        # For DeepSeek MHA-on-MLA latent-cache layers we disable unified mode,
        # because those layers need KV reconstruction before attention.
        force_unified_mla = (
            self.use_mla
            and layer.qk_head_dim != layer.v_head_dim
            and not self._allow_non_unified_mixed_mla
            and not needs_mla_kv_reconstruct
        )
        if self._debug_mla_path and self.use_mla:
            mode_name = (
                forward_batch.forward_mode.name
                if hasattr(forward_batch.forward_mode, "name")
                else str(forward_batch.forward_mode)
            )
            key = (
                mode_name,
                layer.layer_id,
                layer.qk_head_dim,
                layer.v_head_dim,
                self.enable_deterministic,
                force_unified_mla,
            )
            if key not in self._logged_mla_path:
                self._logged_mla_path.add(key)
                logger.info(
                    "TRITON_MLA_PATH mode=%s layer=%s qk_dim=%s v_dim=%s "
                    "deterministic=%s force_unified=%s allow_non_unified_mixed=%s",
                    mode_name,
                    layer.layer_id,
                    layer.qk_head_dim,
                    layer.v_head_dim,
                    self.enable_deterministic,
                    force_unified_mla,
                    self._allow_non_unified_mixed_mla,
                )
        if self.enable_deterministic or force_unified_mla:
            self._maybe_log_ctrl_flow(
                stage="extend",
                path="dispatch",
                kernel="extend_attention_fwd_unified",
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                needs_reconstruct=needs_mla_kv_reconstruct,
                extra=f"force_unified={force_unified_mla}",
            )
            return self._forward_extend_unified(
                q, o, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        # Normal mode: use original 2-stage kernel
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = (
                layer.sliding_window_size
            )  # Needed for sliding window mask
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        _q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        _k = k.contiguous()
        _v = v.contiguous()
        # Triton extend kernels do not accept fp8 RHS in dot(). Keep KV-cache writes
        # as-is, but cast compute-side K/V tensors to bf16 for compatibility.
        if _q.dtype in _FP8_TORCH_DTYPES:
            _q = _q.to(torch.bfloat16)
        if _k.dtype in _FP8_TORCH_DTYPES:
            _k = _k.to(torch.bfloat16)
        if _v.dtype in _FP8_TORCH_DTYPES:
            _v = _v.to(torch.bfloat16)
        _o = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        if needs_mla_kv_reconstruct:
            _kb, _vb, kv_indices = self._reconstruct_mla_kv_from_latent(
                layer=layer,
                kv_indices=kv_indices,
                target_dtype=_q.dtype,
                forward_batch=forward_batch,
            )
        else:
            _kb = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            _vb = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        self._assert_attention_debug_state(
            stage="extend",
            layer=layer,
            q=_q,
            k_buffer=_kb,
            v_buffer=_vb,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=self.forward_metadata.qo_indptr,
        )

        use_gluon_extend = self._should_use_gluon_extend(
            forward_batch=forward_batch,
            q=q,
            layer=layer,
            force_triton_fallback=force_triton_fallback,
        ) and not needs_mla_kv_reconstruct
        if use_gluon_extend:
            self._maybe_log_ctrl_flow(
                stage="extend",
                path="dispatch",
                kernel="gluon_extend_attention_fwd",
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                needs_reconstruct=needs_mla_kv_reconstruct,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                extra=f"sliding_window={sliding_window_size}",
            )
            _min_ext, _total_ext, _total_pfx = self._get_gluon_extend_stats(
                forward_batch=forward_batch,
                q_tokens=_q.shape[0],
            )
            self._log_gluon_dispatch(
                path="extend",
                layer=layer,
                forward_batch=forward_batch,
                extra=f"k_scale={k_descale} v_scale={v_descale}",
            )
            self._gluon_fn(
                _q, _k, _v, _o, _kb, _vb,
                self.forward_metadata.qo_indptr,
                kv_indptr, kv_indices,
                self.forward_metadata.custom_mask,
                causal,
                self.forward_metadata.mask_indptr,
                self.forward_metadata.max_extend_len,
                k_scale=k_descale, v_scale=v_descale,
                sm_scale=layer.scaling,
                logit_cap=logits_soft_cap,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=layer.xai_temperature_len,
                min_len_extend=_min_ext,
                total_prefix_len=_total_pfx,
                total_extend_len=_total_ext,
            )
        else:
            self._maybe_log_ctrl_flow(
                stage="extend",
                path="dispatch",
                kernel="triton_extend_attention_fwd",
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                needs_reconstruct=needs_mla_kv_reconstruct,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                extra=f"sliding_window={sliding_window_size}",
            )
            self.extend_attention_fwd(
                _q, _k, _v, _o, _kb, _vb,
                self.forward_metadata.qo_indptr,
                kv_indptr, kv_indices,
                self.forward_metadata.custom_mask,
                causal,
                self.forward_metadata.mask_indptr,
                self.forward_metadata.max_extend_len,
                k_descale, v_descale, layer.scaling,
                logit_cap=logits_soft_cap,
                sliding_window_size=sliding_window_size,
                sinks=sinks,
                window_kv_offsets=window_kv_offsets,
                xai_temperature_len=layer.xai_temperature_len,
            )
        self._maybe_shadow_compare_aiter(
            stage="extend",
            layer=layer,
            forward_batch=forward_batch,
            q=_q,
            k_current=_k,
            v_current=_v,
            k_buffer=_kb,
            v_buffer=_vb,
            o_triton=_o,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            sm_scale=layer.scaling,
            causal=causal,
        )
        if (
            self._debug_asserts
            and (not self._is_cuda_graph_capturing())
            and not torch.isfinite(_o.float()).all()
        ):
            raise RuntimeError(
                f"[TRITON_ASSERT] stage=extend layer={layer.layer_id} output has non-finite values"
            )
        return o

    def _forward_extend_unified(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ):
        """
        Unified 1-stage extend attention for deterministic inference.
        Both prefix and extend KV are accessed through unified kv_indices.
        """
        bs = forward_batch.batch_size

        # Determine sliding window settings
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = layer.sliding_window_size
            # Note: for unified kernel, we use full kv_indptr (not window)
            prefix_kv_indptr = self.forward_metadata.window_kv_indptr
            prefix_kv_indices = self.forward_metadata.window_kv_indices
            # Compute window start positions (absolute position of first key in window)
            # window_start_pos = seq_len - window_len
            window_kv_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
            # Handle TARGET_VERIFY mode where extend_prefix_lens might not be set
            if forward_batch.extend_prefix_lens is not None:
                window_start_pos = (
                    forward_batch.extend_prefix_lens[:bs] - window_kv_lens
                )
            else:
                # Infer from spec_info: prefix_len = seq_len - draft_token_num
                if forward_batch.spec_info is not None and hasattr(
                    forward_batch.spec_info, "draft_token_num"
                ):
                    extend_prefix_lens = (
                        forward_batch.seq_lens[:bs]
                        - forward_batch.spec_info.draft_token_num
                    )
                    window_start_pos = extend_prefix_lens - window_kv_lens
                else:
                    window_start_pos = None
        else:
            sliding_window_size = -1
            prefix_kv_indptr = self.forward_metadata.kv_indptr
            prefix_kv_indices = self.forward_metadata.kv_indices
            window_start_pos = None

        # Build unified kv_indices using fused Triton kernel
        extend_kv_indices = forward_batch.out_cache_loc

        # Handle cases where extend_seq_lens or extend_start_loc might not be set
        # In speculative decoding, we can infer these from spec_info or compute them
        if forward_batch.extend_seq_lens is None:
            # TARGET_VERIFY mode: infer extend_seq_lens from spec_info
            if forward_batch.spec_info is not None and hasattr(
                forward_batch.spec_info, "draft_token_num"
            ):
                draft_token_num = forward_batch.spec_info.draft_token_num
                extend_seq_lens = torch.full(
                    (bs,), draft_token_num, dtype=torch.int32, device=self.device
                )
            else:
                raise RuntimeError(
                    "extend_seq_lens is None but cannot infer from spec_info. "
                    "This should not happen in TARGET_VERIFY mode."
                )
        else:
            extend_seq_lens = forward_batch.extend_seq_lens

        # Check extend_start_loc separately - it might be None even when extend_seq_lens is set
        if forward_batch.extend_start_loc is None:
            # Compute extend_start_loc from extend_seq_lens
            # extend_start_loc[i] = sum(extend_seq_lens[0:i])
            extend_start_loc = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=self.device),
                    torch.cumsum(extend_seq_lens[:-1], dim=0),
                ]
            )
        else:
            extend_start_loc = forward_batch.extend_start_loc

        unified_kv_indptr, unified_kv_indices, prefix_lens = (
            self.build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                bs,
            )
        )

        # Convert prefix_lens to int32 for the kernel
        prefix_lens = prefix_lens.to(torch.int32)

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Call unified kernel
        self.extend_attention_fwd_unified(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            k_descale,
            v_descale,
            self.forward_metadata.qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens,
            self.forward_metadata.max_extend_len,
            custom_mask=self.forward_metadata.custom_mask,
            mask_indptr=self.forward_metadata.mask_indptr,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            is_causal=causal,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_start_pos=window_start_pos,
            xai_temperature_len=layer.xai_temperature_len,
        )

        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        needs_mla_kv_reconstruct = (
            self._mla_reconstruct_decode and self._needs_mla_kv_reconstruction(layer)
        )
        if needs_mla_kv_reconstruct:
            # DeepSeek MHA-on-MLA path stores latent KV separately; do not
            # overwrite with expanded K/V tensors here.
            save_kv_cache = False
            self._maybe_log_mla_recon_compare(layer, forward_batch, k, v)
        if self._force_mla_no_save_kv and self.use_mla:
            # Debug override: align with AITER MLA policy where fused RoPE+cache
            # path owns KV updates and backend avoids duplicate writes.
            save_kv_cache = False
        self._maybe_log_ctrl_flow(
            stage="decode",
            path="entry",
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            needs_reconstruct=needs_mla_kv_reconstruct,
            extra=f"use_mla={self.use_mla}",
        )

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if save_kv_cache:
            if self.use_mla:  # Triton MLA currently doesn't support quantized kv cache
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )
        self._maybe_log_kv_state(layer, forward_batch, stage="decode_after_write")

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        if needs_mla_kv_reconstruct:
            k_buffer, v_buffer, kv_indices = self._reconstruct_mla_kv_from_latent(
                layer=layer,
                kv_indices=kv_indices,
                target_dtype=q.dtype,
                forward_batch=forward_batch,
            )
        else:
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        self._assert_attention_debug_state(
            stage="decode",
            layer=layer,
            q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=None,
        )
        self._maybe_log_ctrl_flow(
            stage="decode",
            path="dispatch",
            kernel="triton_decode_attention_fwd",
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            needs_reconstruct=needs_mla_kv_reconstruct,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
        )

        attn_logits = self.forward_metadata.attn_logits
        if (
            self.forward_metadata.swa_attn_logits is not None
            and layer.v_head_dim == self.swa_v_head_dim
        ):
            attn_logits = self.forward_metadata.swa_attn_logits

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_buffer,
            v_buffer,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr,
            kv_indices,
            attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            k_descale,
            v_descale,
            logit_cap=logits_soft_cap,
            sinks=sinks,
            xai_temperature_len=layer.xai_temperature_len,
        )
        self._maybe_shadow_compare_aiter(
            stage="decode",
            layer=layer,
            forward_batch=forward_batch,
            q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_current=None,
            v_current=None,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            o_triton=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            sm_scale=layer.scaling,
            causal=False,
        )
        if (
            self._debug_asserts
            and (not self._is_cuda_graph_capturing())
            and not torch.isfinite(o.float()).all()
        ):
            raise RuntimeError(
                f"[TRITON_ASSERT] stage=decode layer={layer.layer_id} output has non-finite values"
            )
        return o


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends: List[TritonAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: Optional[torch.Tensor],
        call_fn: int,
    ):
        if kv_indices_buffer is None:
            kv_indices_buffer = self.cuda_graph_kv_indices

        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        if call_fn is None:
            return

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int64,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int64,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,),
            self.attn_backends[0].max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs,
                max_num_tokens,
                kv_indices_buf=self.cuda_graph_kv_indices[i],
                cuda_graph_num_kv_splits_buf=self.cuda_graph_num_kv_splits,
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, None, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        self.common_template(forward_batch, None, None)

        # NOTE: Multi-step's attention backends use the slice of
        # - kv_indptr buffer (cuda graph and non-cuda graph)
        # - kv_indices buffer (cuda graph only)
        # So we don't need to assign the KV indices inside the attention backend.

        # Compute num_kv_splits only once
        num_token = forward_batch.batch_size * self.topk
        self.attn_backends[-1].get_num_kv_splits(
            self.attn_backends[-1].cuda_graph_num_kv_splits[:num_token],
            forward_batch.seq_lens[:bs],
        )


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


def update_sliding_window_buffer(
    window_kv_indptr,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_indices = torch.empty(
        window_kv_indptr[-1], dtype=torch.int64, device=device
    )
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx


def update_sliding_window_buffer_cuda_graph(
    window_kv_indptr,
    window_kv_indices,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
