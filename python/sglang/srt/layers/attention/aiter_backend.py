from __future__ import annotations

"""
end to end attention solution with aiter kernels
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch
import triton

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
    create_flashmla_kv_indices_triton,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_gfx95_supported

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

try:
    from aiter import (
        flash_attn_varlen_func,
        get_mla_metadata_info_v1,
        get_mla_metadata_v1,
        get_ps_metadata_info_v1,
        get_ps_metadata_v1,
        mha_batch_prefill_func,
        mla_prefill_ps_asm_fwd,
        mla_reduce_v1,
        paged_attention_ragged,
    )
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
    from aiter.ops.triton.attention.unified_attention import unified_attention
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.utils import (
    launch_reshape_and_cache_flash,
    pad_sequence_with_mask,
)
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

# Use aiter mla persist design for fp8-kv cache
_use_mla_ps_kernel = get_bool_env_var("SGLANG_AITER_MLA_PERSIST", "True")

# Use fp8 prefill only on gfx95
_use_fp8_prefill_attn = (
    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True") and is_gfx95_supported()
)

# Persist
# fast_mode=True if _use_mla_ps_kernel else False
# intra_batch_mode=False if _use_mla_ps_kernel else True

# fake non-ps, intra_batch_mode needs to be True for non-ps-mode
fast_mode = False
intra_batch_mode = True if _use_mla_ps_kernel else False


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class ForwardMetadata:
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
    max_q_len: int
    max_kv_len: Optional[int]
    work_metadata: Optional[torch.Tensor] = None
    work_info_set: Optional[torch.Tensor] = None
    work_indptr: Optional[torch.Tensor] = None
    reduce_indptr: Optional[torch.Tensor] = None
    reduce_final_map: Optional[torch.Tensor] = None
    reduce_partial_map: Optional[torch.Tensor] = None
    num_kv_splits: Optional[int] = None
    run_graph: Optional[bool] = True
    custom_mask: Optional[torch.Tensor] = None
    mask_indptr: Optional[torch.Tensor] = None
    max_extend_len: Optional[int] = None
    fp8_prefill_kv_indices: Optional[torch.Tensor] = None
    swa_page_table: Optional[torch.Tensor] = None


global_workspace_buffer = None

_AITER_PARTITION_SIZE_ROCM = 256


class AiterAttnBackend(AttentionBackend):
    @staticmethod
    def _is_cuda_graph_capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    @staticmethod
    def _parse_probe_backend(
        value: Optional[str], default: str, allowed: tuple[str, ...]
    ) -> str:
        if value is None:
            return default
        parsed = value.strip().lower()
        if parsed in allowed:
            return parsed
        logger.warning(
            "Invalid probe backend '%s'; falling back to '%s' (allowed=%s).",
            value,
            default,
            ",".join(allowed),
        )
        return default

    @staticmethod
    def _dispatch_num_cus(device: torch.device) -> int:
        # MI35x/MI45x path is fixed at 256 CUs for dispatch heuristics.
        if is_gfx95_supported():
            return 256
        try:
            return max(1, int(torch.cuda.get_device_properties(device).multi_processor_count))
        except Exception:
            return 1

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        topk: int = 1,
    ):
        super().__init__()
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )
        try:
            from sglang.srt.layers.attention.gluon_ops.CDNA4.extend_attention_entrypoints import (
                gluon_extend_attention_fwd,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gluon extend import unavailable in Aiter backend; disabling gluon probe path: %s",
                exc,
            )
            gluon_extend_attention_fwd = None
        try:
            from sglang.srt.layers.attention.gluon_ops.CDNA4 import (
                f16_mla_prefill as _gluon_mla_d512_mod,
            )

            mla_d512_gqa_attention_fwd = getattr(
                _gluon_mla_d512_mod, "mla_d512_gqa_attention_fwd", None
            )
            mla_d512_gqa_attention_fwd_wca = getattr(
                _gluon_mla_d512_mod, "mla_d512_gqa_attention_fwd_wca", None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gluon D512 import unavailable in Aiter backend; disabling gluon probe path: %s",
                exc,
            )
            mla_d512_gqa_attention_fwd = None
            mla_d512_gqa_attention_fwd_wca = None
        try:
            from sglang.srt.layers.attention.gluon_ops.CDNA4 import (
                fp8_mla_prefill as _gluon_mla_d512_fp8_mod,
            )

            mla_d512_gqa_attention_fwd_fp8 = getattr(
                _gluon_mla_d512_fp8_mod, "mla_d512_gqa_attention_fwd_fp8", None
            )
            mla_d512_gqa_attention_fwd_wca_fp8 = getattr(
                _gluon_mla_d512_fp8_mod, "mla_d512_gqa_attention_fwd_wca_fp8", None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gluon D512 FP8 import unavailable in Aiter backend; disabling gluon probe path: %s",
                exc,
            )
            mla_d512_gqa_attention_fwd_fp8 = None
            mla_d512_gqa_attention_fwd_wca_fp8 = None

        self.input_dtype = model_runner.model_config.dtype

        self.page_size = model_runner.server_args.page_size

        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
        self.decode_attention_fwd_triton = torch.compiler.disable(decode_attention_fwd)
        self.gluon_extend_attention_fwd = (
            torch.compiler.disable(gluon_extend_attention_fwd)
            if gluon_extend_attention_fwd is not None
            else None
        )
        self.gluon_mla_prefill_fwd = (
            torch.compiler.disable(mla_d512_gqa_attention_fwd)
            if mla_d512_gqa_attention_fwd is not None
            else None
        )
        self.gluon_mla_prefill_wca_fwd = (
            torch.compiler.disable(mla_d512_gqa_attention_fwd_wca)
            if mla_d512_gqa_attention_fwd_wca is not None
            else None
        )
        self.gluon_mla_prefill_fp8_fwd = (
            torch.compiler.disable(mla_d512_gqa_attention_fwd_fp8)
            if mla_d512_gqa_attention_fwd_fp8 is not None
            else None
        )
        self.gluon_mla_prefill_wca_fp8_fwd = (
            torch.compiler.disable(mla_d512_gqa_attention_fwd_wca_fp8)
            if mla_d512_gqa_attention_fwd_wca_fp8 is not None
            else None
        )
        self.mla_triton_kernel_probe = get_bool_env_var(
            "SGLANG_AITER_MLA_TRITON_KERNEL_PROBE", "False"
        )
        self.mla_triton_probe_extend = get_bool_env_var(
            "SGLANG_AITER_MLA_TRITON_KERNEL_PROBE_EXTEND",
            "True" if self.mla_triton_kernel_probe else "False",
        )
        self.mla_triton_probe_decode = get_bool_env_var(
            "SGLANG_AITER_MLA_TRITON_KERNEL_PROBE_DECODE",
            "True" if self.mla_triton_kernel_probe else "False",
        )
        # Stage-wise probe backend selection inside AITER MLA flow.
        # Supported values: aiter | triton | gluon
        default_probe_backend = "triton" if self.mla_triton_kernel_probe else "aiter"
        global_probe_backend = self._parse_probe_backend(
            os.getenv("SGLANG_AITER_MLA_PROBE_BACKEND"),
            default=default_probe_backend,
            allowed=("aiter", "triton", "gluon"),
        )
        self.mla_probe_extend_backend = self._parse_probe_backend(
            os.getenv("SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND"),
            default=global_probe_backend,
            allowed=("aiter", "triton", "gluon"),
        )
        self.mla_probe_decode_backend = self._parse_probe_backend(
            os.getenv("SGLANG_AITER_MLA_PROBE_DECODE_BACKEND"),
            default=global_probe_backend,
            allowed=("aiter", "triton", "gluon"),
        )
        # Preserve legacy booleans unless explicit stage backend is provided.
        if (
            "SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND" not in os.environ
            and self.mla_triton_probe_extend
        ):
            self.mla_probe_extend_backend = "triton"
        if (
            "SGLANG_AITER_MLA_PROBE_DECODE_BACKEND" not in os.environ
            and self.mla_triton_probe_decode
        ):
            self.mla_probe_decode_backend = "triton"
        self._logged_decode_gluon_probe_fallback = False
        self._logged_gluon_d192_probe_failure = False
        self._gluon_d192_probe_failed = False
        self._gluon_d192_probe_failure_reason: Optional[str] = None
        self._logged_probe_extend_shape_guard = False
        self._logged_decode_shape_guard = False
        self._debug_ctrl_flow = get_bool_env_var("SGLANG_DEBUG_ATTN_CTRL_FLOW", "false")
        self._ctrl_flow_seen: set[tuple] = set()

        self.device = model_runner.device
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = topk
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        if is_gfx95_supported():
            # MI350/gfx95 uses e4m3fn (not fnuz) in this stack.
            expected_fp8 = getattr(torch, "float8_e4m3fn", None)
            if expected_fp8 is not None and fp8_dtype != expected_fp8:
                logger.warning(
                    "gfx95 detected but fp8_dtype=%s (expected=%s).",
                    fp8_dtype,
                    expected_fp8,
                )

        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA

        # Get v_head_dim based on model type
        if self.use_mla:
            # For MLA models, get v_head_dim from model config
            self.v_head_dim = model_runner.model_config.v_head_dim
        elif hasattr(model_runner.token_to_kv_pool, "get_v_head_dim"):
            # For hybrid models (Mamba+attention, GDN, Kimi linear),
            # layer_id=0 may not be a full attention layer
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
        else:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[
                -1
            ]

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill

        max_bs = model_runner.req_to_token_pool.size

        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        self.mask_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int64, device=model_runner.device
        )
        self._kv_indices_scratch: Optional[torch.Tensor] = None

        # Create prefill indices updater
        if not skip_prefill:
            self.indices_updater_prefill = AiterIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.use_mla:
                self.mla_indices_updater_prefill = AiterMlaIndicesUpdaterPrefill(
                    model_runner, self
                )

        # sliding window attention
        self.use_sliding_window_kv_pool = (
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)
            and model_runner.token_to_kv_pool.swa_layer_nums > 0
        )

        if self.use_sliding_window_kv_pool:
            self.token_to_kv_pool = model_runner.token_to_kv_pool
            self.use_triton_unified_attention = True
        else:
            self.use_triton_unified_attention = get_bool_env_var(
                "SGLANG_USE_AITER_UNIFIED_ATTN"
            )

        # aiter kernel related initialization
        self.max_num_partitions = (
            self.max_context_len + _AITER_PARTITION_SIZE_ROCM - 1
        ) // _AITER_PARTITION_SIZE_ROCM

        nbyes_per_qo_elem = torch.finfo(torch.float32).bits // 8

        if not (self.use_mla or self.use_triton_unified_attention):
            self.workspace_buffer = torch.empty(
                (max_bs * self.num_head * self.max_num_partitions * self.head_dim)
                * nbyes_per_qo_elem
                + 2 * (max_bs * self.num_head * self.max_num_partitions) * 4,
                dtype=torch.uint8,
                device=self.device,
            )

        self.scale = float(1.0 / (self.head_dim**0.5))
        self.k_scale = self.v_scale = torch.tensor([1.0], dtype=torch.float32).to(
            self.device
        )

        self.logits_soft_cap = 0.0

        self.forward_metadata: ForwardMetadata = None

        if self.use_mla:
            _valid_heads = self.num_head in (4, 8) or (
                self.num_head % 16 == 0 and 16 <= self.num_head <= 128
            )
            assert _valid_heads, (
                f"Aiter MLA supports num_head of 4, 8, or multiples of 16 "
                f"in [16, 128].\n"
                f"Provided {self.num_head} number of heads.\n"
                "Try adjusting tensor_parallel_size value."
            )
            self.num_head_padded = 16 if self.num_head < 16 else self.num_head
            self.head_repeat_factor = 16 // self.num_head if self.num_head < 16 else 1

            self.enable_dp_attention = is_dp_attention_enabled()
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
            global _use_mla_ps_kernel, fast_mode, intra_batch_mode

            # current mla_decode_fwd only support fake-nps in self.num_head == 16
            # so all num_head size does not use qh16 kernel to simulate
            # it should not use fake-nps (fast_mode = False, intra_batch_mode = True)
            # it will cause gpu-fault or accuracy issue
            if self.num_head == 32 or self.num_head == 128:
                fast_mode = True
                intra_batch_mode = False

            # current persist a16w16 mla_decode kernel does not support head_num = 128
            # need to fall back to non-persist
            # only use mla_ps_kernel when fp8 kv_cache
            # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
            # head_num=16 (tp8 perf issue), head_num=128 (unsupported, like tp1 or --enable-dp-attention with tp8-dp8)
            if (
                self.num_head_padded == 16 or self.num_head_padded == 128
            ) and self.kv_cache_dtype is not fp8_dtype:
                _use_mla_ps_kernel = False
                fast_mode = False
                intra_batch_mode = False

            self.max_split_per_batch = 32 if _use_mla_ps_kernel else None

            if self.num_draft_tokens is None and _use_mla_ps_kernel:
                self.max_split_per_batch = 64

            self.fix_max_split_per_batch = self.max_split_per_batch

    def make_mla_decode_meta_data_buffer(self, max_seqlen_qo, batch_size):
        nhead = self.num_head_padded
        dtype = self.kv_cache_dtype

        if self.enable_dp_attention:
            gpu = torch.cuda.current_device()
            device_properties = torch.cuda.get_device_properties(gpu)
            cu_num = device_properties.multi_processor_count
            self.max_split_per_batch = min(
                (cu_num + batch_size - 1) // batch_size, self.fix_max_split_per_batch
            )

        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            batch_size,
            max_seqlen_qo,
            nhead,
            dtype,
            dtype,
            is_sparse=False,
            fast_mode=fast_mode,
            num_kv_splits=self.max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
        )

        # aiter implementation
        # the tensor's meaning please refer aiter/ops/attention.py
        work_metadata = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device="cuda"
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device="cuda"
        )
        work_info_set = torch.empty(
            work_info_set_size,
            dtype=work_info_set_type,
            device="cuda",
        )
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
        )

        return (
            work_metadata,
            work_indptr,
            work_info_set,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_meta_data(
        self,
        qo_indptr,
        kv_indptr,
        kv_last_page_len,
        work_metadata,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_q_len,
        fast_mode,
        max_split_per_batch,
        intra_batch_mode,
    ):

        nhead_kv = 1
        page_size = self.page_size
        dtype = self.kv_cache_dtype

        meta = get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            kv_last_page_len,
            self.num_head_padded // nhead_kv,
            nhead_kv,
            False,
            work_metadata,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            kv_granularity=max(page_size, 16),
            max_seqlen_qo=max_q_len,
            uni_seqlen_qo=max_q_len,
            fast_mode=fast_mode,
            max_split_per_batch=max_split_per_batch,
            intra_batch_mode=intra_batch_mode,
            dtype_q=dtype,
            dtype_kv=dtype,
        )

    def make_mla_prefill_ps_meta_data_buffer(
        self, batch_size: int, max_qlen: int, qlen_granularity: int
    ):
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_size, work_info_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_ps_metadata_info_v1(
            batch_size=batch_size,
            num_head_k=self.num_kv_head,
            max_qlen=max_qlen,
            qlen_granularity=qlen_granularity,
        )

        device = self.device
        work_metadata_ptrs = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
        reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        reduce_partial_map = torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        )

        return (
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
        )

    def make_mla_prefill_ps_meta_data(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        seq_lens: torch.Tensor,
        work_metadata: torch.Tensor,
        work_indptr: torch.Tensor,
        work_info: torch.Tensor,
        reduce_indptr: torch.Tensor,
        reduce_final_map: torch.Tensor,
        reduce_partial_map: torch.Tensor,
        is_causal: bool = True,
    ):
        gqa_ratio = self.num_head // self.num_kv_head
        num_heads_k = self.num_kv_head
        tile_q = 256
        qhead_granularity = gqa_ratio
        qlen_granularity = tile_q // qhead_granularity
        kvlen_granularity = max(128, self.page_size)
        block_size = self.page_size

        qo_indptr_cpu = qo_indptr.to("cpu", dtype=torch.int32)
        kv_indptr_cpu = kv_indptr.to("cpu", dtype=torch.int32)
        seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int32)

        get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_heads_k,
            work_metadata,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=is_causal,
        )

    # for page size > 1 useful conversion function
    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def _resolve_v2_num_draft_tokens(
        self,
        extend_seq_lens: Optional[torch.Tensor] = None,
        extend_seq_lens_cpu: Optional[list[int]] = None,
    ) -> int:
        """Resolve fixed per-request extend length for DRAFT_EXTEND_V2."""
        num_draft_tokens = self.num_draft_tokens
        if num_draft_tokens is None:
            if extend_seq_lens is not None and extend_seq_lens.numel() > 0:
                # Avoid list scans in hot path when tensor lengths are already available.
                num_draft_tokens = int(extend_seq_lens[0].item())
            elif extend_seq_lens_cpu:
                num_draft_tokens = max(extend_seq_lens_cpu)
            else:
                raise ValueError(
                    "DRAFT_EXTEND_V2 requires speculative_num_draft_tokens or "
                    "non-empty extend_seq_lens/extend_seq_lens_cpu."
                )

        num_draft_tokens = int(num_draft_tokens)
        if extend_seq_lens is not None and extend_seq_lens.numel() > 0:
            if not torch.all(extend_seq_lens == num_draft_tokens):
                raise ValueError(
                    "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                    f"extend_seq_lens={extend_seq_lens}, expected all == {num_draft_tokens}."
                )
        if extend_seq_lens_cpu and any(
            x != num_draft_tokens for x in extend_seq_lens_cpu
        ):
            raise ValueError(
                "DRAFT_EXTEND_V2 expects fixed extend length per request; got "
                f"{extend_seq_lens_cpu}, expected all == {num_draft_tokens}."
            )
        return num_draft_tokens

    def _get_kv_indices_scratch(
        self, required_tokens: int, device: torch.device
    ) -> torch.Tensor:
        if (
            self._kv_indices_scratch is None
            or self._kv_indices_scratch.device != device
            or self._kv_indices_scratch.numel() < required_tokens
        ):
            self._kv_indices_scratch = torch.empty(
                required_tokens, dtype=torch.int32, device=device
            )
        return self._kv_indices_scratch[:required_tokens]

    def _set_uniform_qo_indptr(
        self, bs: int, tokens_per_req: int, device: torch.device
    ) -> torch.Tensor:
        qo_indptr = self.qo_indptr[: bs + 1]
        qo_indptr[: bs + 1] = torch.arange(
            0,
            bs * tokens_per_req + 1,
            step=tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        return qo_indptr

    def _ensure_spec_v2_topk_supported(self):
        if self.topk > 1:
            raise NotImplementedError(
                "AiterAttnBackend SPEC_V2 path currently supports topk <= 1 only. "
                f"Got topk={self.topk}."
            )

    def _mla_decode_fwd_with_head_pad(
        self,
        q: torch.Tensor,
        k_buffer_flat: torch.Tensor,
        layer,
        **kwargs,
    ):
        """Wrap mla_decode_fwd with head-dimension padding for num_head < 16.

        When head_repeat_factor > 1 (i.e. num_head is 4 or 8), q is
        repeat-interleaved to reach num_head_padded (16) before the kernel
        call, and the corresponding output columns are sliced back afterward.
        q / o must already be shaped (..., num_head, head_dim).
        """
        if self.head_repeat_factor > 1:
            q_in = q.repeat_interleave(self.head_repeat_factor, dim=1)
            o = q.new_empty(
                (q.shape[0], self.num_head_padded, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q_in, k_buffer_flat, o, **kwargs)
            return o[:, :: self.head_repeat_factor, :]
        else:
            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                dtype=self.input_dtype,
            )
            mla_decode_fwd(q, k_buffer_flat, o, **kwargs)
            return o

    def mla_fp8_prefill_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
    ):
        total_q = q.shape[0]
        nhead = layer.tp_q_head_num
        v_head_dim = layer.v_head_dim

        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)
        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        tile_q = 256
        reduce_indptr = self.forward_metadata.reduce_indptr
        reduce_final_map = self.forward_metadata.reduce_final_map
        reduce_partial_map = self.forward_metadata.reduce_partial_map

        logits = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead, v_head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        attn_lse = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        final_lse = torch.empty(
            (total_q, nhead),
            dtype=torch.float32,
            device=q.device,
        )
        output = q.new_empty(
            (total_q, nhead, v_head_dim),
            dtype=self.input_dtype,
        )

        mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            self.forward_metadata.qo_indptr,
            self.forward_metadata.kv_indptr,
            self.forward_metadata.fp8_prefill_kv_indices,
            self.forward_metadata.work_indptr,
            self.forward_metadata.work_info_set,
            self.forward_metadata.max_q_len,
            layer.scaling,
            True,
            logits,
            attn_lse,
            output,
            one_scale,
            one_scale,
            one_scale,
        )
        mla_reduce_v1(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            tile_q,
            output,
            final_lse,
        )
        return output

    def _maybe_log_ctrl_flow(
        self,
        stage: str,
        path: str,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        kernel: str = "",
        save_kv_cache: Optional[bool] = None,
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
            "ATTN_CTRL backend=aiter stage=%s path=%s kernel=%s mode=%s layer=%s "
            "qk_dim=%s v_dim=%s save_kv=%s kv_last=%s kv_n=%s %s",
            stage,
            path,
            kernel,
            mode_name,
            layer.layer_id,
            layer.qk_head_dim,
            layer.v_head_dim,
            save_kv_cache,
            kv_last,
            kv_n,
            extra,
        )

    def _select_gluon_d192_probe_policy(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        max_len_extend: int,
    ) -> tuple[dict[str, bool], str, int, int, int, str]:
        """Select auto/WCA/split-K policy for D192 mixed-dim probe extend."""
        batch_size = max(0, qo_indptr.shape[0] - 1)
        if batch_size == 0:
            return {}, "gluon_extend_d192_auto_probe", 1, 0, 0, "empty_batch=1"

        # Prefer CPU metadata to avoid GPU reduction + .item() sync in dispatch.
        ext_lens_cpu = forward_batch.extend_seq_lens_cpu or []
        if ext_lens_cpu and len(ext_lens_cpu) == batch_size:
            min_len_extend = int(min(ext_lens_cpu))
            total_extend_len = int(sum(ext_lens_cpu))
        else:
            extend_lens = qo_indptr[1:] - qo_indptr[:-1]
            min_len_extend = int(extend_lens.min().item())
            total_extend_len = int(extend_lens.sum().item())

        prefix_lens = forward_batch.extend_prefix_lens_cpu or []
        if prefix_lens and len(prefix_lens) == batch_size:
            max_prefix = int(max(prefix_lens))
            total_prefix_len = int(sum(prefix_lens))
            avg_prefix = total_prefix_len // max(1, batch_size)
        else:
            total_prefix_len = int((kv_indptr[-1] - kv_indptr[0]).item())
            max_prefix = int(max(prefix_lens)) if prefix_lens else 0
            avg_prefix = total_prefix_len // max(1, batch_size)

        # Tile estimate follows existing WCA policy for D512 wrappers.
        n_m_tiles = (max_len_extend + 64 - 1) // 64
        total_output_tiles = batch_size * layer.tp_q_head_num * n_m_tiles
        num_cus = self._dispatch_num_cus(qo_indptr.device)
        tile_starved = total_output_tiles < num_cus
        severe_tile_starved = (total_output_tiles * 4) < num_cus

        # Keep split-K very narrow for mixed-dim D192.
        use_splitk = (
            severe_tile_starved
            and avg_prefix >= 4096
            and max_len_extend >= 512
        )
        # WCA helps prefix-heavy tile-starved extend, but avoid decode/very-short.
        use_wca = (
            tile_starved
            and max_prefix >= 1024
            and max_len_extend >= 512
            and not use_splitk
        )

        if use_splitk:
            kwargs = {"_force_use_splitk": True}
            kernel = "gluon_extend_d192_splitk_probe"
        elif use_wca:
            kwargs = {"_force_use_persistent": True}
            kernel = "gluon_extend_d192_wca_probe"
        else:
            kwargs = {}
            kernel = "gluon_extend_d192_auto_probe"

        extra = (
            f"tile_starved={int(tile_starved)} total_tiles={total_output_tiles} "
            f"severe_tile_starved={int(severe_tile_starved)} num_cus={num_cus} "
            f"max_prefix={max_prefix} avg_prefix={avg_prefix} max_ext={max_len_extend}"
        )
        return kwargs, kernel, min_len_extend, total_prefix_len, total_extend_len, extra

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for aiter attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info
        qo_indptr = None
        kv_last_page_len = None
        max_q_len = None
        max_kv_len = None

        work_metadata = None
        work_indptr = None
        work_info_set = None
        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        num_kv_splits = None
        swa_page_table = None
        max_kv_len = forward_batch.seq_lens_cpu.max().item()

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None or forward_batch.forward_mode.is_idle():
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]

                if not self.use_triton_unified_attention:
                    kv_indices = self._get_kv_indices_scratch(
                        forward_batch.seq_lens_sum, forward_batch.seq_lens.device
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
                else:
                    max_q_len = 1
                    page_size = self.page_size
                    max_num_blocks_per_seq = (max_kv_len + page_size - 1) // page_size
                    kv_indices = torch.zeros(
                        bs, max_kv_len, dtype=torch.int32, device=self.device
                    )

                    create_flashmla_kv_indices_triton[(bs,)](
                        self.req_to_token,
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        None,
                        kv_indices,
                        self.req_to_token.stride(0),
                        max_kv_len,
                        1,
                    )

                    if self.use_sliding_window_kv_pool:
                        swa_page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                kv_indices
                            )
                        )

                        kv_indices = self._transform_table_1_to_real(kv_indices)
                        swa_page_table = self._transform_table_1_to_real(swa_page_table)
                    elif self.page_size > 1:
                        kv_indices = self._transform_table_1_to_real(kv_indices)

                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.kv_last_page_len[:bs], dim=0
                    )

            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
                kv_last_page_len = self.kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_q_len, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                run_graph=False,
                swa_page_table=swa_page_table,
            )

        elif forward_batch.forward_mode.is_draft_extend_v2():
            # EAGLE V2: DRAFT_EXTEND_V2 mode - extend draft KV cache with all predicted tokens
            self._ensure_spec_v2_topk_supported()
            if self.use_mla:
                device = forward_batch.seq_lens.device
                num_draft_tokens = self._resolve_v2_num_draft_tokens(
                    extend_seq_lens=forward_batch.extend_seq_lens
                )
                qo_indptr = self._set_uniform_qo_indptr(bs, num_draft_tokens, device)

                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)

                kv_indices = self._get_kv_indices_scratch(
                    forward_batch.seq_lens_sum, device
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

                if _use_mla_ps_kernel:
                    max_seqlen_qo = num_draft_tokens
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    num_draft_tokens,
                    forward_batch.seq_lens_cpu.max().item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                )
        elif forward_batch.forward_mode.is_draft_extend():
            # EAGLE V1: DRAFT_EXTEND mode - uses spec_info.accept_length
            if self.use_mla:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )

                if _use_mla_ps_kernel:
                    max_seqlen_qo = max(forward_batch.extend_seq_lens_cpu)
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                # Non-MLA draft_extend: use triton extend kernel with causal masking
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )
                kv_indices = kv_indices.to(torch.int64)
                draft_max_extend_len = torch.max(spec_info.accept_length).item()

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    draft_max_extend_len,
                    None,
                    custom_mask=custom_mask,
                    mask_indptr=None,
                    max_extend_len=draft_max_extend_len,
                )
        elif forward_batch.forward_mode.is_target_verify():
            if self.use_mla:
                draft_num = spec_info.draft_token_num
                kv_lens = forward_batch.seq_lens + draft_num
                kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs
                device = forward_batch.seq_lens.device

                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=device,
                )
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                kv_indices = self._get_kv_indices_scratch(
                    kv_lens_sum,
                    device,
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    kv_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )

                # if self.kv_cache_dtype == fp8_dtype:
                if _use_mla_ps_kernel:
                    max_seqlen_qo = draft_num
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, bs)

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        self.kv_last_page_len[:bs],
                        work_metadata,
                        work_info_set,
                        work_indptr,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        max_seqlen_qo,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    draft_num,
                    None,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                    run_graph=False,
                )
            else:
                # Non-MLA target_verify: use triton extend kernel with custom mask
                bs = len(forward_batch.req_pool_indices)
                draft_num = spec_info.draft_token_num

                qo_indptr = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=self.device,
                )

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

                custom_mask = spec_info.custom_mask
                seq_mask_len = draft_num * (forward_batch.seq_lens + draft_num)
                mask_indptr = self.mask_indptr
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                mask_indptr = mask_indptr[: bs + 1]

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    draft_num,
                    None,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=draft_num,
                )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            if self.is_multimodal:
                extend_no_prefix = False
            else:
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            if self.use_mla:
                self.mla_indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    forward_batch.extend_seq_lens,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    spec_info=None,
                )

                max_q_len = self.mla_indices_updater_prefill.max_q_len
                qo_indptr = self.mla_indices_updater_prefill.qo_indptr
                kv_indptr = self.mla_indices_updater_prefill.kv_indptr

                work_metadata = None
                work_indptr = None
                work_info_set = None
                reduce_indptr = None
                reduce_final_map = None
                reduce_partial_map = None
                fp8_prefill_kv_indices = None

                if _use_fp8_prefill_attn:
                    tile_q = 256
                    qlen_granularity = tile_q // (self.num_head // self.num_kv_head)
                    (
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                    ) = self.make_mla_prefill_ps_meta_data_buffer(
                        bs, max_q_len, qlen_granularity
                    )

                    self.make_mla_prefill_ps_meta_data(
                        qo_indptr,
                        kv_indptr,
                        forward_batch.seq_lens,
                        work_metadata,
                        work_indptr,
                        work_info_set,
                        reduce_indptr,
                        reduce_final_map,
                        reduce_partial_map,
                        is_causal=True,
                    )

                    total_s = forward_batch.seq_lens_sum
                    fp8_prefill_kv_indices = torch.arange(
                        total_s, device=self.device, dtype=torch.int32
                    )

                self.forward_metadata = ForwardMetadata(
                    self.mla_indices_updater_prefill.kv_indptr,
                    self.mla_indices_updater_prefill.kv_indices,
                    qo_indptr,
                    self.kv_last_page_len[:bs],
                    max_q_len,
                    self.mla_indices_updater_prefill.max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    fp8_prefill_kv_indices=fp8_prefill_kv_indices,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )

                if self.use_sliding_window_kv_pool:
                    swa_page_table = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            self.indices_updater_prefill.kv_indices
                        )
                    )

                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    swa_page_table=swa_page_table,
                )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        # Keep CUDA-graph metadata on device for aiter MLA decode kernels.
        self.cuda_graph_kv_last_page_len = torch.ones(
            max_bs, dtype=torch.int32, device=self.device
        )
        if kv_indices_buf is None:
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * max_num_blocks_per_seq),
                dtype=torch.int32,
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

        # if self.use_mla and (_use_mla_ps_kernel or self.kv_cache_dtype == fp8_dtype):
        if self.use_mla and _use_mla_ps_kernel:
            # for persistent mla_decode_fwd
            max_seqlen_qo = (
                1 if self.num_draft_tokens is None else self.num_draft_tokens
            )

            (
                self.work_metadata,
                self.work_indptr,
                self.work_info_set,
                self.reduce_indptr,
                self.reduce_final_map,
                self.reduce_partial_map,
            ) = self.make_mla_decode_meta_data_buffer(max_seqlen_qo, max_bs)

        else:
            self.work_metadata = None
            self.work_indptr = None
            self.work_info_set = None

            self.reduce_indptr = None
            self.reduce_final_map = None
            self.reduce_partial_map = None

        if self.use_sliding_window_kv_pool:
            max_num_blocks_per_seq = (
                self.max_context_len + self.page_size - 1
            ) // self.page_size
            self.cuda_graph_swa_page_table = torch.zeros(
                (max_bs, max_num_blocks_per_seq),
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

        num_kv_splits = None
        # num_kv_splits_indptr = None

        work_metadata = None
        work_info_set = None
        work_indptr = None

        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        swa_page_table = None

        max_kv_len = torch.max(seq_lens).item()

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:

                if not self.use_triton_unified_attention:
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
                else:
                    max_q_len = 1
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    kv_indices = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )

                    page_indices = self.req_to_token[req_pool_indices[:bs], :max_kv_len]

                    if self.use_sliding_window_kv_pool:
                        swa_page_indices = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                page_indices
                            )
                        )

                        page_indices = self._transform_table_1_to_real(page_indices)
                        swa_page_indices = self._transform_table_1_to_real(
                            swa_page_indices
                        )

                        new_rows = swa_page_indices.shape[0]
                        new_cols = swa_page_indices.shape[1]

                        kv_indices[:new_rows, :new_cols].copy_(page_indices)
                        swa_page_table = self.cuda_graph_swa_page_table
                        swa_page_table[:new_rows, :new_cols].copy_(swa_page_indices)
                    elif self.page_size > 1:
                        page_indices = self._transform_table_1_to_real(page_indices)
                        new_rows = page_indices.shape[0]
                        new_cols = page_indices.shape[1]
                        kv_indices[:new_rows, :new_cols].copy_(page_indices)

                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.cuda_graph_kv_last_page_len[:bs], dim=0
                    )

                    kv_indptr = None
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                swa_page_table=swa_page_table,
            )

        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self.use_mla:
                kv_lens = seq_lens + self.num_draft_tokens
            else:
                kv_lens = seq_lens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _use_mla_ps_kernel:

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                custom_mask = self.cuda_graph_custom_mask
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = max_q_len * (seq_lens + max_q_len)
                mask_indptr = self.mask_indptr
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
                mask_indptr = mask_indptr[: bs + 1]

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=max_q_len,
                )
        elif forward_mode.is_draft_extend_v2():
            # EAGLE V2: Uses fixed num_draft_tokens per batch
            self._ensure_spec_v2_topk_supported()
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            qo_indptr = self._set_uniform_qo_indptr(bs, num_tokens_per_bs, self.device)
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
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs

            if self.use_mla and _use_mla_ps_kernel:
                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )
        elif forward_mode.is_draft_extend():
            # EAGLE V1: Uses speculative_num_steps + 1
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

            if self.use_mla:
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = num_tokens_per_bs

                if _use_mla_ps_kernel:

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                # Non-MLA draft_extend cuda graph: use triton extend kernel
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    None,
                    num_tokens_per_bs,
                    None,
                    custom_mask=None,
                    mask_indptr=None,
                    max_extend_len=num_tokens_per_bs,
                )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

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

        num_kv_splits = None
        # num_kv_splits_indptr = None

        work_metadata = None
        work_info_set = None
        work_indptr = None

        reduce_indptr = None
        reduce_final_map = None
        reduce_partial_map = None

        swa_page_table = None
        max_kv_len = seq_lens_cpu.max().item()

        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:
                if not self.use_triton_unified_attention:
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
                else:
                    max_q_len = 1
                    max_num_blocks_per_seq = (
                        self.max_context_len + self.page_size - 1
                    ) // self.page_size
                    kv_indices = self.cuda_graph_kv_indices.view(
                        -1, max_num_blocks_per_seq
                    )

                    page_indices = self.req_to_token[req_pool_indices[:bs], :max_kv_len]

                    if self.use_sliding_window_kv_pool:
                        swa_page_indices = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                page_indices
                            )
                        )

                        page_indices = self._transform_table_1_to_real(page_indices)
                        swa_page_indices = self._transform_table_1_to_real(
                            swa_page_indices
                        )

                        new_rows = swa_page_indices.shape[0]
                        new_cols = swa_page_indices.shape[1]

                        kv_indices[:new_rows, :new_cols].copy_(page_indices)
                        swa_page_table = self.cuda_graph_swa_page_table
                        swa_page_table[:new_rows, :new_cols].copy_(swa_page_indices)
                    elif self.page_size > 1:
                        page_indices = self._transform_table_1_to_real(page_indices)
                        new_rows = page_indices.shape[0]
                        new_cols = page_indices.shape[1]
                        kv_indices[:new_rows, :new_cols].copy_(page_indices)

                    qo_indptr = self.qo_indptr[: bs + 1]
                    qo_indptr[1 : bs + 1] = torch.cumsum(
                        self.cuda_graph_kv_last_page_len[:bs], dim=0
                    )

                    kv_indptr = None
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1

                if _use_mla_ps_kernel:
                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
                swa_page_table=swa_page_table,
                # num_kv_splits_indptr=num_kv_splits_indptr,
            )

        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            if self.use_mla:
                kv_lens = seq_lens + self.num_draft_tokens
            else:
                kv_lens = seq_lens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = self.num_draft_tokens

            if self.use_mla:
                if _use_mla_ps_kernel:

                    num_kv_splits = self.max_split_per_batch

                    self.make_mla_meta_data(
                        qo_indptr,
                        kv_indptr,
                        kv_last_page_len,
                        self.work_metadata,
                        self.work_info_set,
                        self.work_indptr,
                        self.reduce_indptr,
                        self.reduce_final_map,
                        self.reduce_partial_map,
                        max_q_len,
                        fast_mode=fast_mode,
                        max_split_per_batch=num_kv_splits,
                        intra_batch_mode=intra_batch_mode,
                    )

                    work_metadata = self.work_metadata
                    work_info_set = self.work_info_set
                    work_indptr = self.work_indptr

                    reduce_indptr = self.reduce_indptr
                    reduce_final_map = self.reduce_final_map
                    reduce_partial_map = self.reduce_partial_map

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    work_metadata=work_metadata,
                    work_info_set=work_info_set,
                    work_indptr=work_indptr,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    num_kv_splits=num_kv_splits,
                )
            else:
                custom_mask = self.cuda_graph_custom_mask
                custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
                seq_mask_len = max_q_len * (seq_lens + max_q_len)
                mask_indptr = self.mask_indptr[: bs + 1]
                mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    max_kv_len,
                    custom_mask=custom_mask,
                    mask_indptr=mask_indptr,
                    max_extend_len=max_q_len,
                )
        elif forward_mode.is_draft_extend_v2():
            # EAGLE V2: Fixed num_draft_tokens per batch
            self._ensure_spec_v2_topk_supported()
            seq_lens = seq_lens[:bs]
            num_tokens_per_bs = self._resolve_v2_num_draft_tokens()
            extend_lens = torch.full(
                (bs,), num_tokens_per_bs, dtype=torch.int32, device=seq_lens.device
            )

            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
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

            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs

            if self.use_mla and _use_mla_ps_kernel:

                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )
        elif forward_mode.is_draft_extend():
            # EAGLE V1: Uses spec_info.accept_length
            num_tokens_per_bs = self.speculative_num_steps + 1
            seq_lens = seq_lens[:bs]
            accept_lens = spec_info.accept_length[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(accept_lens, dim=0)
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

            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs

            if self.use_mla and _use_mla_ps_kernel:

                num_kv_splits = self.max_split_per_batch

                self.make_mla_meta_data(
                    qo_indptr,
                    kv_indptr,
                    kv_last_page_len,
                    self.work_metadata,
                    self.work_info_set,
                    self.work_indptr,
                    self.reduce_indptr,
                    self.reduce_final_map,
                    self.reduce_partial_map,
                    max_q_len,
                    fast_mode=fast_mode,
                    max_split_per_batch=num_kv_splits,
                    intra_batch_mode=intra_batch_mode,
                )

                work_metadata = self.work_metadata
                work_info_set = self.work_info_set
                work_indptr = self.work_indptr

                reduce_indptr = self.reduce_indptr
                reduce_final_map = self.reduce_final_map
                reduce_partial_map = self.reduce_partial_map

            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                max_kv_len,
                work_metadata=work_metadata,
                work_info_set=work_info_set,
                work_indptr=work_indptr,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                num_kv_splits=num_kv_splits,
            )

        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1 if self.num_draft_tokens is None else self.num_draft_tokens

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        # AITER verify path does not require post-draft buffer patching currently.
        # This override prevents overlap-plan stream mode from failing with the
        # base class NotImplementedError.
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
        self.logits_soft_cap = layer.logit_cap
        self._maybe_log_ctrl_flow(
            stage="extend",
            path="entry",
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            kv_indptr=(
                self.forward_metadata.kv_indptr if self.forward_metadata is not None else None
            ),
            kv_indices=(
                self.forward_metadata.kv_indices if self.forward_metadata is not None else None
            ),
            extra=f"use_mla={self.use_mla} probe_extend={self.mla_probe_extend_backend}",
        )

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
                # both unified attention and sliding window kv pool are active.
                # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
                # use standard set_kv_buffer, as they lack SWA-specific attributes
                # like full_to_swa_index_mapping.
                if (
                    self.use_triton_unified_attention
                    and self.use_sliding_window_kv_pool
                ):
                    token_to_kv_pool = forward_batch.token_to_kv_pool
                    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    slot_mapping_swa = token_to_kv_pool.full_to_swa_index_mapping

                    launch_reshape_and_cache_flash(
                        k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                        v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                        k_cache.view(
                            -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                        ),
                        v_cache.view(
                            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                        ),
                        cache_loc,
                        (
                            slot_mapping_swa.long()
                            if layer.sliding_window_size > 0
                            else None
                        ),
                    )

                elif self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
                else:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )

        if self.use_mla:
            probe_extend_backend = self.mla_probe_extend_backend
            expected_q_width = layer.tp_q_head_num * layer.qk_head_dim
            if (
                probe_extend_backend != "aiter"
                and q is not None
                and q.numel() % expected_q_width != 0
            ):
                if not self._logged_probe_extend_shape_guard:
                    logger.warning(
                        "AITER MLA probe extend disabled for incompatible q shape "
                        "(numel=%s, expected multiple of %s). Falling back to native AITER extend.",
                        q.numel(),
                        expected_q_width,
                    )
                    self._logged_probe_extend_shape_guard = True
                self._maybe_log_ctrl_flow(
                    stage="extend",
                    path="dispatch",
                    kernel="probe_extend_shape_guard_fallback_aiter",
                    layer=layer,
                    forward_batch=forward_batch,
                    save_kv_cache=save_kv_cache,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                    extra=f"q_numel={q.numel()} expected_multiple={expected_q_width}",
                )
                probe_extend_backend = "aiter"

            if probe_extend_backend != "aiter" and k is not None and v is not None:
                fp8_probe_dtypes = {fp8_dtype}
                for _name in (
                    "float8_e4m3fn",
                    "float8_e4m3fnuz",
                    "float8_e5m2",
                    "float8_e5m2fnuz",
                ):
                    _dtype = getattr(torch, _name, None)
                    if _dtype is not None:
                        fp8_probe_dtypes.add(_dtype)

                q_probe = q
                k_probe = k
                v_probe = v
                k_scale_probe = layer.k_scale_float if layer.k_scale is not None else 1.0
                v_scale_probe = layer.v_scale_float if layer.v_scale is not None else 1.0
                if (
                    q.dtype in fp8_probe_dtypes
                    or k.dtype in fp8_probe_dtypes
                    or v.dtype in fp8_probe_dtypes
                ):
                    q_probe = q.to(torch.bfloat16)
                    k_probe = k.to(torch.bfloat16)
                    v_probe = v.to(torch.bfloat16)
                    k_scale_probe = 1.0
                    v_scale_probe = 1.0

                if layer.qk_head_dim != layer.v_head_dim:
                    o = q_probe.new_empty(
                        (q_probe.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                        dtype=self.input_dtype,
                    )
                else:
                    o = torch.empty_like(q_probe, dtype=self.input_dtype)
                max_len_extend = (
                    self.forward_metadata.max_extend_len
                    if self.forward_metadata.max_extend_len is not None
                    else self.forward_metadata.max_q_len
                )
                q3 = q_probe.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                o3 = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buf = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

                def _run_triton_probe_extend(kernel_name: str, extra: str = "") -> torch.Tensor:
                    self._maybe_log_ctrl_flow(
                        stage="extend",
                        path="dispatch",
                        kernel=kernel_name,
                        layer=layer,
                        forward_batch=forward_batch,
                        save_kv_cache=save_kv_cache,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        kv_indices=self.forward_metadata.kv_indices,
                        extra=extra,
                    )
                    self.extend_attention_fwd(
                        q3,
                        k_probe.contiguous(),
                        v_probe.contiguous(),
                        o3,
                        k_buf,
                        v_buf,
                        self.forward_metadata.qo_indptr,
                        self.forward_metadata.kv_indptr,
                        self.forward_metadata.kv_indices,
                        self.forward_metadata.custom_mask,
                        True,
                        self.forward_metadata.mask_indptr,
                        max_len_extend,
                        k_scale_probe,
                        v_scale_probe,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                    )
                    return o

                probe_backend = probe_extend_backend
                if probe_backend == "triton":
                    return _run_triton_probe_extend("triton_extend_attention_fwd_probe")

                if probe_backend == "gluon":
                    # Prefer dedicated Gluon MLA common kernels for DeepSeek D576/D512.
                    # They have explicit bf16/fp8 variants and should be selected by KV dtype.
                    use_gluon_mla_common = (
                        layer.qk_head_dim == 576
                        and layer.v_head_dim == 512
                        and self.forward_metadata.custom_mask is None
                    )
                    if use_gluon_mla_common:
                        kv_buffer = k_buf.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                        prefix_lens = forward_batch.extend_prefix_lens_cpu or []
                        max_prefix = max(prefix_lens) if prefix_lens else 0
                        batch_size = max(0, self.forward_metadata.qo_indptr.shape[0] - 1)
                        n_m_tiles = (max_len_extend + 64 - 1) // 64
                        total_output_tiles = batch_size * layer.tp_q_head_num * n_m_tiles
                        num_cus = self._dispatch_num_cus(q3.device)
                        use_wca = (
                            max_prefix >= 512
                            and max_len_extend >= 128
                            and total_output_tiles < num_cus
                        )

                        kv_is_fp8 = (
                            kv_buffer.dtype in fp8_probe_dtypes
                            or self.kv_cache_dtype == fp8_dtype
                        )
                        if kv_is_fp8:
                            if self.gluon_mla_prefill_fp8_fwd is None:
                                raise RuntimeError(
                                    "SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND=gluon with fp8 KV "
                                    "requires fp8_mla_prefill wrapper, but it is unavailable."
                                )
                            gluon_fn = (
                                self.gluon_mla_prefill_wca_fp8_fwd
                                if use_wca and self.gluon_mla_prefill_wca_fp8_fwd is not None
                                else self.gluon_mla_prefill_fp8_fwd
                            )
                            kernel_name = (
                                "gluon_mla_d512_fp8_wca_probe"
                                if use_wca
                                and gluon_fn is self.gluon_mla_prefill_wca_fp8_fwd
                                else "gluon_mla_d512_fp8_probe"
                            )
                            self._maybe_log_ctrl_flow(
                                stage="extend",
                                path="dispatch",
                                kernel=kernel_name,
                                layer=layer,
                                forward_batch=forward_batch,
                                save_kv_cache=save_kv_cache,
                                kv_indptr=self.forward_metadata.kv_indptr,
                                kv_indices=self.forward_metadata.kv_indices,
                            )
                            gluon_fn(
                                q3,
                                kv_buffer,
                                o3,
                                self.forward_metadata.qo_indptr,
                                self.forward_metadata.kv_indptr,
                                self.forward_metadata.kv_indices,
                                max_len_extend=max_len_extend,
                                is_causal=True,
                                sm_scale=layer.scaling,
                                logit_cap=layer.logit_cap,
                                k_scale=k_scale_probe,
                                v_scale=v_scale_probe,
                            )
                            return o
                        if self.gluon_mla_prefill_fwd is None:
                            raise RuntimeError(
                                "SGLANG_AITER_MLA_PROBE_EXTEND_BACKEND=gluon with bf16 KV "
                                "requires f16_mla_prefill wrapper, but it is unavailable."
                            )
                        compute_dtype = (
                            q3.dtype
                            if q3.dtype in (torch.float16, torch.bfloat16)
                            else torch.bfloat16
                        )
                        gluon_fn = (
                            self.gluon_mla_prefill_wca_fwd
                            if use_wca and self.gluon_mla_prefill_wca_fwd is not None
                            else self.gluon_mla_prefill_fwd
                        )
                        kernel_name = (
                            "gluon_mla_d512_bf16_wca_probe"
                            if use_wca and gluon_fn is self.gluon_mla_prefill_wca_fwd
                            else "gluon_mla_d512_bf16_probe"
                        )
                        self._maybe_log_ctrl_flow(
                            stage="extend",
                            path="dispatch",
                            kernel=kernel_name,
                            layer=layer,
                            forward_batch=forward_batch,
                            save_kv_cache=save_kv_cache,
                            kv_indptr=self.forward_metadata.kv_indptr,
                            kv_indices=self.forward_metadata.kv_indices,
                        )
                        gluon_fn(
                            q3.to(compute_dtype),
                            kv_buffer.to(compute_dtype),
                            o3,
                            self.forward_metadata.qo_indptr,
                            self.forward_metadata.kv_indptr,
                            self.forward_metadata.kv_indices,
                            max_len_extend=max_len_extend,
                            is_causal=True,
                            sm_scale=layer.scaling,
                            logit_cap=layer.logit_cap,
                        )
                        return o

                    use_gluon_d192 = (
                        self.gluon_extend_attention_fwd is not None
                        and layer.qk_head_dim == 192
                        and layer.v_head_dim in (128, 192)
                    )
                    if use_gluon_d192:
                        if self._gluon_d192_probe_failed:
                            fallback_reason = (
                                self._gluon_d192_probe_failure_reason or "prior_failure"
                            )
                            return _run_triton_probe_extend(
                                "triton_extend_attention_fwd_probe_gluon_d192_disabled",
                                extra=f"reason={fallback_reason}",
                            )
                        (
                            policy_kwargs,
                            policy_kernel,
                            min_len_extend,
                            total_prefix_len,
                            total_extend_len,
                            policy_extra,
                        ) = self._select_gluon_d192_probe_policy(
                            layer=layer,
                            forward_batch=forward_batch,
                            qo_indptr=self.forward_metadata.qo_indptr,
                            kv_indptr=self.forward_metadata.kv_indptr,
                            max_len_extend=max_len_extend,
                        )
                        self._maybe_log_ctrl_flow(
                            stage="extend",
                            path="dispatch",
                            kernel=policy_kernel,
                            layer=layer,
                            forward_batch=forward_batch,
                            save_kv_cache=save_kv_cache,
                            kv_indptr=self.forward_metadata.kv_indptr,
                            kv_indices=self.forward_metadata.kv_indices,
                            extra=policy_extra,
                        )
                        try:
                            self.gluon_extend_attention_fwd(
                                q3,
                                k_probe.contiguous(),
                                v_probe.contiguous(),
                                o3,
                                k_buf,
                                v_buf,
                                self.forward_metadata.qo_indptr,
                                self.forward_metadata.kv_indptr,
                                self.forward_metadata.kv_indices,
                                custom_mask=self.forward_metadata.custom_mask,
                                is_causal=True,
                                mask_indptr=self.forward_metadata.mask_indptr,
                                max_len_extend=max_len_extend,
                                k_scale=k_scale_probe,
                                v_scale=v_scale_probe,
                                sm_scale=layer.scaling,
                                logit_cap=layer.logit_cap,
                                min_len_extend=min_len_extend,
                                total_prefix_len=total_prefix_len,
                                total_extend_len=total_extend_len,
                                **policy_kwargs,
                            )
                            return o
                        except Exception as e:  # noqa: BLE001
                            failure_reason = type(e).__name__
                            failure_msg = str(e).strip().replace("\n", " ")
                            self._gluon_d192_probe_failed = True
                            self._gluon_d192_probe_failure_reason = failure_reason
                            if not self._logged_gluon_d192_probe_failure:
                                logger.warning(
                                    "Gluon D192 probe extend failed (%s): %s. "
                                    "Disabling Gluon D192 probe for this worker and "
                                    "falling back to Triton.",
                                    failure_reason,
                                    failure_msg if failure_msg else "<no message>",
                                )
                                self._logged_gluon_d192_probe_failure = True
                            return _run_triton_probe_extend(
                                "triton_extend_attention_fwd_probe_gluon_exception_fallback",
                                extra=f"reason={failure_reason}",
                            )

                    # Unsupported Gluon probe shape/wrapper -> Triton fallback.
                    return _run_triton_probe_extend(
                        "triton_extend_attention_fwd_probe_gluon_fallback"
                    )

            max_q_len = self.forward_metadata.max_q_len
            max_kv_len = self.forward_metadata.max_kv_len
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            qo_indptr = self.forward_metadata.qo_indptr
            K_Buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            V_Buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            kv_lora_rank = V_Buffer.shape[-1]
            qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
            qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
            assert len(q.shape) == 3
            assert len(k.shape) == 3
            assert len(v.shape) == 3

            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
                if kv_indices.shape[0] == 0 or extend_no_prefix:
                    if _use_fp8_prefill_attn:
                        self._maybe_log_ctrl_flow(
                            stage="extend",
                            path="dispatch",
                            kernel="mla_fp8_prefill_attn",
                            layer=layer,
                            forward_batch=forward_batch,
                            save_kv_cache=save_kv_cache,
                            kv_indptr=qo_indptr,
                            kv_indices=kv_indices,
                            extra="extend_no_prefix=1",
                        )
                        output = self.mla_fp8_prefill_attn(
                            q,
                            k,
                            v,
                            layer,
                        )
                    else:
                        self._maybe_log_ctrl_flow(
                            stage="extend",
                            path="dispatch",
                            kernel="flash_attn_varlen_func",
                            layer=layer,
                            forward_batch=forward_batch,
                            save_kv_cache=save_kv_cache,
                            kv_indptr=qo_indptr,
                            kv_indices=kv_indices,
                            extra="extend_no_prefix=1",
                        )
                        output = flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            qo_indptr,
                            max_q_len,
                            max_q_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )
                    return output
                elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
                    K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
                    kvc, k_pe = torch.split(
                        K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
                    )

                    if self.kv_cache_dtype == fp8_dtype:
                        dtype = q.dtype

                        kvc = kvc.to(dtype)
                        k_pe = k_pe.to(dtype)

                    if (
                        _use_fp8_prefill_attn
                        and layer.kv_b_proj.weight.dtype == torch.uint8
                    ):
                        # MXFP4 weights + FP8 prefill: fuse GEMM, nope/v split, and k_pe cat
                        # into a single kernel (fused_gemm_afp4wfp4_split_cat) that writes k and v
                        # directly in FP8, avoiding a separate elementwise cast
                        k, v = layer.kv_b_proj(
                            (
                                kvc.squeeze(1),
                                k_pe.expand(-1, layer.tp_k_head_num, -1),
                                qk_nope_head_dim,
                                layer.v_head_dim,
                                fp8_dtype,
                            )
                        )[0]
                    else:
                        kv = layer.kv_b_proj(kvc.contiguous())[0]

                        kv = kv.view(
                            -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
                        )
                        k, v = torch.split(
                            kv, [qk_nope_head_dim, layer.v_head_dim], dim=-1
                        )
                        k = torch.cat(
                            [
                                k,
                                torch.broadcast_to(
                                    k_pe,
                                    (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),
                                ),
                            ],
                            dim=-1,
                        )

                    assert (
                        forward_batch.extend_prefix_lens.shape
                        == forward_batch.extend_seq_lens.shape
                    )

                    if _use_fp8_prefill_attn:
                        self._maybe_log_ctrl_flow(
                            stage="extend",
                            path="dispatch",
                            kernel="mla_fp8_prefill_attn",
                            layer=layer,
                            forward_batch=forward_batch,
                            save_kv_cache=save_kv_cache,
                            kv_indptr=kv_indptr,
                            kv_indices=kv_indices,
                            extra="extend_no_prefix=0 reconstructed=1",
                        )
                        return self.mla_fp8_prefill_attn(q, k, v, layer)
                    else:
                        self._maybe_log_ctrl_flow(
                            stage="extend",
                            path="dispatch",
                            kernel="flash_attn_varlen_func",
                            layer=layer,
                            forward_batch=forward_batch,
                            save_kv_cache=save_kv_cache,
                            kv_indptr=kv_indptr,
                            kv_indices=kv_indices,
                            extra="extend_no_prefix=0 reconstructed=1",
                        )
                        return flash_attn_varlen_func(
                            q,
                            k,
                            v,
                            qo_indptr,
                            kv_indptr,
                            max_q_len,
                            max_kv_len,
                            softmax_scale=layer.scaling,
                            causal=True,
                        )

                else:
                    self._maybe_log_ctrl_flow(
                        stage="extend",
                        path="dispatch",
                        kernel="mla_prefill_fwd",
                        layer=layer,
                        forward_batch=forward_batch,
                        save_kv_cache=save_kv_cache,
                        kv_indptr=kv_indptr,
                        kv_indices=kv_indices,
                        extra="extend_no_prefix=0 reconstructed=0",
                    )
                    if layer.qk_head_dim != layer.v_head_dim:
                        o = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        o = torch.empty_like(q)
                    mla_prefill_fwd(
                        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    return o
            elif forward_batch.forward_mode.is_target_verify():
                self._maybe_log_ctrl_flow(
                    stage="extend",
                    path="dispatch",
                    kernel="mla_decode_fwd",
                    layer=layer,
                    forward_batch=forward_batch,
                    save_kv_cache=save_kv_cache,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                    extra="target_verify=1",
                )
                o = q.new_empty(
                    (q.shape[0], layer.tp_q_head_num, layer.v_head_dim),
                    dtype=self.input_dtype,
                )

                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                o = self._mla_decode_fwd_with_head_pad(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    layer,
                    qo_indptr=self.forward_metadata.qo_indptr,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                    kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    sm_scale=layer.scaling,
                    logit_cap=layer.logit_cap,
                    work_meta_data=work_metadata,
                    work_indptr=work_indptr,
                    work_info_set=work_info_set,
                    reduce_indptr=reduce_indptr,
                    reduce_final_map=reduce_final_map,
                    reduce_partial_map=reduce_partial_map,
                    q_scale=(
                        layer.k_scale if layer.k_scale is not None else self.k_scale
                    ),
                    kv_scale=(
                        layer.k_scale if layer.k_scale is not None else self.k_scale
                    ),
                    intra_batch_mode=intra_batch_mode,
                    num_kv_splits=num_kv_splits,
                )
                return o
            elif (
                forward_batch.forward_mode.is_draft_extend()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                self._maybe_log_ctrl_flow(
                    stage="extend",
                    path="dispatch",
                    kernel="mla_decode_fwd",
                    layer=layer,
                    forward_batch=forward_batch,
                    save_kv_cache=save_kv_cache,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                    extra=f"draft_extend=1 run_graph={self.forward_metadata.run_graph}",
                )

                work_metadata = self.forward_metadata.work_metadata
                work_indptr = self.forward_metadata.work_indptr
                work_info_set = self.forward_metadata.work_info_set

                reduce_indptr = self.forward_metadata.reduce_indptr
                reduce_final_map = self.forward_metadata.reduce_final_map
                reduce_partial_map = self.forward_metadata.reduce_partial_map

                num_kv_splits = self.forward_metadata.num_kv_splits

                if self.forward_metadata.run_graph is not True:
                    bs, q_pad, q_mask = pad_sequence_with_mask(
                        q.view(q.shape[0], -1),
                        qo_indptr[:-1],
                        forward_batch.extend_seq_lens,
                        self.forward_metadata.max_q_len,
                    )
                    o = self._mla_decode_fwd_with_head_pad(
                        q_pad.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        layer,
                        qo_indptr=self.forward_metadata.qo_indptr,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        kv_indices=self.forward_metadata.kv_indices,
                        kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=(
                            layer.k_scale if layer.k_scale is not None else self.k_scale
                        ),
                        kv_scale=(
                            layer.k_scale if layer.k_scale is not None else self.k_scale
                        ),
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )

                    total_valid_q = int(qo_indptr[-1].item())
                    return o[:total_valid_q]
                else:
                    o = self._mla_decode_fwd_with_head_pad(
                        q,
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        layer,
                        qo_indptr=self.forward_metadata.qo_indptr,
                        kv_indptr=self.forward_metadata.kv_indptr,
                        kv_indices=self.forward_metadata.kv_indices,
                        kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                        max_seqlen_q=self.forward_metadata.max_q_len,
                        sm_scale=layer.scaling,
                        logit_cap=layer.logit_cap,
                        work_meta_data=work_metadata,
                        work_indptr=work_indptr,
                        work_info_set=work_info_set,
                        reduce_indptr=reduce_indptr,
                        reduce_final_map=reduce_final_map,
                        reduce_partial_map=reduce_partial_map,
                        q_scale=(
                            layer.k_scale if layer.k_scale is not None else self.k_scale
                        ),
                        kv_scale=(
                            layer.k_scale if layer.k_scale is not None else self.k_scale
                        ),
                        intra_batch_mode=intra_batch_mode,
                        num_kv_splits=num_kv_splits,
                    )
                    return o
            else:
                raise ValueError(
                    f"Invalid forward mode for MLA prefill: {forward_batch.forward_mode=}"
                )
        else:
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                # Use triton extend kernel which supports custom masks and causal masking
                self._maybe_log_ctrl_flow(
                    stage="extend",
                    path="dispatch",
                    kernel="triton_extend_attention_fwd",
                    layer=layer,
                    forward_batch=forward_batch,
                    save_kv_cache=save_kv_cache,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                )
                if layer.qk_head_dim != layer.v_head_dim:
                    o = q.new_empty(
                        (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                    )
                else:
                    o = torch.empty_like(q)

                self.extend_attention_fwd(
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k.contiguous(),
                    v.contiguous(),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                    forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.custom_mask,
                    True,  # causal
                    self.forward_metadata.mask_indptr,
                    self.forward_metadata.max_extend_len,
                    1.0,  # k_scale
                    1.0,  # v_scale
                    layer.scaling,
                    logit_cap=layer.logit_cap,
                )
                return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            bs0 = forward_batch.batch_size + 1

            # TODO kkhuang-amd need to remove it when mha_batch_prefill_func support fp8-kv
            if self.kv_cache_dtype == fp8_dtype:
                dtype = q.dtype
                k_cache = k_cache.to(dtype)
                v_cache = v_cache.to(dtype)

            window_size = (-1, -1)
            page_table = self.forward_metadata.kv_indices

            if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
                window_size = (layer.sliding_window_size, -1)
                if self.forward_metadata.swa_page_table is not None:
                    page_table = self.forward_metadata.swa_page_table

            self._maybe_log_ctrl_flow(
                stage="extend",
                path="dispatch",
                kernel="mha_batch_prefill_func",
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                kv_indptr=self.forward_metadata.kv_indptr[:bs0],
                kv_indices=page_table,
            )
            o = mha_batch_prefill_func(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache,
                v_cache,
                self.qo_indptr[:bs0],
                self.forward_metadata.kv_indptr[:bs0],
                page_table,
                self.forward_metadata.max_q_len,
                self.forward_metadata.max_kv_len,
                causal=True,
                logits_soft_cap=self.logits_soft_cap,
                alibi_slopes=None,
                return_lse=False,
                return_attn_probs=False,
                window_size=window_size,
                sink_ptr=sinks,
            )

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

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
        expected_q_width = layer.tp_q_head_num * layer.qk_head_dim
        if self.use_mla and q.numel() % expected_q_width != 0:
            token_count = (
                q.shape[0]
                if q.ndim >= 2
                else max(1, self.forward_metadata.qo_indptr.shape[0] - 1)
            )
            if not self._logged_decode_shape_guard:
                logger.warning(
                    "AITER MLA decode encountered incompatible q shape "
                    "(numel=%s, expected multiple of %s). Using zero-output fallback "
                    "for this decode call to keep server alive.",
                    q.numel(),
                    expected_q_width,
                )
                self._logged_decode_shape_guard = True
            self._maybe_log_ctrl_flow(
                stage="decode",
                path="dispatch",
                kernel="mla_decode_shape_guard_zero_fallback",
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                kv_indptr=(
                    self.forward_metadata.kv_indptr
                    if self.forward_metadata is not None
                    else None
                ),
                kv_indices=(
                    self.forward_metadata.kv_indices
                    if self.forward_metadata is not None
                    else None
                ),
                extra=f"q_numel={q.numel()} expected_multiple={expected_q_width}",
            )
            return q.new_zeros(
                (token_count, layer.tp_q_head_num * layer.v_head_dim),
                dtype=self.input_dtype,
            )

        q = q.reshape(-1, expected_q_width)
        self._maybe_log_ctrl_flow(
            stage="decode",
            path="entry",
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            kv_indptr=(
                self.forward_metadata.kv_indptr if self.forward_metadata is not None else None
            ),
            kv_indices=(
                self.forward_metadata.kv_indices if self.forward_metadata is not None else None
            ),
            extra=f"use_mla={self.use_mla} probe_decode={self.mla_probe_decode_backend}",
        )

        k_descale = None
        v_descale = None
        if self.kv_cache_dtype == fp8_dtype:
            k_descale = layer.k_scale if layer.k_scale is not None else self.k_scale
            v_descale = layer.v_scale if layer.v_scale is not None else self.k_scale

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=self.input_dtype,
            )
        else:
            o = torch.empty_like(q, dtype=self.input_dtype)

        if save_kv_cache:
            # Only use SWA-specific kv cache write (reshape_and_cache_flash) when
            # both unified attention and sliding window kv pool are active.
            # Non-SWA models (e.g. Qwen3-VL) enabled via SGLANG_USE_AITER_UNIFIED_ATTN
            # use standard set_kv_buffer, as they lack SWA-specific attributes
            # like full_to_swa_index_mapping.
            if self.use_triton_unified_attention and self.use_sliding_window_kv_pool:
                token_to_kv_pool = forward_batch.token_to_kv_pool
                k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                    layer.layer_id
                )
                slot_mapping_swa = token_to_kv_pool.full_to_swa_index_mapping

                launch_reshape_and_cache_flash(
                    k.view(-1, layer.tp_k_head_num, layer.qk_head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    forward_batch.out_cache_loc,
                    slot_mapping_swa.long() if layer.sliding_window_size > 0 else None,
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

        if self.use_mla:
            decode_probe_backend = self.mla_probe_decode_backend
            if decode_probe_backend == "gluon":
                if not self._logged_decode_gluon_probe_fallback:
                    logger.warning(
                        "SGLANG_AITER_MLA_PROBE_DECODE_BACKEND=gluon is not supported; "
                        "falling back to AITER MLA decode kernel."
                    )
                    self._logged_decode_gluon_probe_fallback = True
                decode_probe_backend = "aiter"
            if decode_probe_backend == "triton":
                self._maybe_log_ctrl_flow(
                    stage="decode",
                    path="dispatch",
                    kernel="triton_decode_attention_fwd_probe",
                    layer=layer,
                    forward_batch=forward_batch,
                    save_kv_cache=save_kv_cache,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                )
                k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
                bs = self.forward_metadata.kv_indptr.shape[0] - 1
                max_kv_splits = 1
                attn_logits = torch.empty(
                    (bs, layer.tp_q_head_num, max_kv_splits, layer.v_head_dim),
                    dtype=torch.float32,
                    device=q.device,
                )
                attn_lse = torch.empty(
                    (bs, layer.tp_q_head_num, max_kv_splits),
                    dtype=torch.float32,
                    device=q.device,
                )
                num_kv_splits = torch.ones(bs, dtype=torch.int32, device=q.device)
                self.decode_attention_fwd_triton(
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k_buffer,
                    v_buffer,
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices.to(torch.int64),
                    attn_logits,
                    attn_lse,
                    num_kv_splits,
                    max_kv_splits,
                    layer.scaling,
                    layer.k_scale_float if layer.k_scale is not None else 1.0,
                    layer.v_scale_float if layer.v_scale is not None else 1.0,
                    logit_cap=layer.logit_cap,
                    sinks=sinks,
                    xai_temperature_len=layer.xai_temperature_len,
                )
                return o

            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            self._maybe_log_ctrl_flow(
                stage="decode",
                path="dispatch",
                kernel="mla_decode_fwd",
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                kv_indptr=self.forward_metadata.kv_indptr,
                kv_indices=self.forward_metadata.kv_indices,
            )

            work_metadata = self.forward_metadata.work_metadata
            work_indptr = self.forward_metadata.work_indptr
            work_info_set = self.forward_metadata.work_info_set

            reduce_indptr = self.forward_metadata.reduce_indptr
            reduce_final_map = self.forward_metadata.reduce_final_map
            reduce_partial_map = self.forward_metadata.reduce_partial_map

            num_kv_splits = self.forward_metadata.num_kv_splits

            o = self._mla_decode_fwd_with_head_pad(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                layer,
                qo_indptr=self.forward_metadata.qo_indptr,
                kv_indptr=self.forward_metadata.kv_indptr,
                kv_indices=self.forward_metadata.kv_indices,
                kv_last_page_lens=self.forward_metadata.kv_last_page_len,
                max_seqlen_q=self.forward_metadata.max_q_len,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                work_meta_data=work_metadata,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=layer.k_scale if layer.k_scale is not None else self.k_scale,
                kv_scale=layer.k_scale if layer.k_scale is not None else self.k_scale,
                intra_batch_mode=intra_batch_mode,
                num_kv_splits=num_kv_splits,
            )
        else:
            self.logits_soft_cap = layer.logit_cap

            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            o = torch.empty_like(q, dtype=self.input_dtype)

            if self.kv_cache_dtype == fp8_dtype:
                dtype = q.dtype
                k_cache = k_cache.to(dtype)
                v_cache = v_cache.to(dtype)

            if self.use_triton_unified_attention:
                bs = forward_batch.batch_size
                window_size = (-1, -1)
                page_table = self.forward_metadata.kv_indices

                if (
                    layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                ):
                    window_size = (layer.sliding_window_size - 1, 0)
                    if self.forward_metadata.swa_page_table is not None:
                        page_table = self.forward_metadata.swa_page_table
                self._maybe_log_ctrl_flow(
                    stage="decode",
                    path="dispatch",
                    kernel="unified_attention",
                    layer=layer,
                    forward_batch=forward_batch,
                    save_kv_cache=save_kv_cache,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=page_table,
                )

                max_kv_len = page_table.shape[1] * self.page_size

                unified_attention(
                    q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k=k_cache.view(
                        -1, self.page_size, layer.tp_k_head_num, layer.qk_head_dim
                    ),
                    v=v_cache.view(
                        -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                    ),
                    out=o.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    cu_seqlens_q=self.forward_metadata.qo_indptr,
                    seqused_k=forward_batch.seq_lens,
                    max_seqlen_q=self.forward_metadata.max_q_len,
                    max_seqlen_k=max_kv_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=window_size,
                    block_table=page_table,
                    softcap=0,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                    sinks=sinks,
                )
            else:
                self._maybe_log_ctrl_flow(
                    stage="decode",
                    path="dispatch",
                    kernel="paged_attention_ragged",
                    layer=layer,
                    forward_batch=forward_batch,
                    save_kv_cache=save_kv_cache,
                    kv_indptr=self.forward_metadata.kv_indptr,
                    kv_indices=self.forward_metadata.kv_indices,
                )
                paged_attention_ragged(
                    o.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    self.workspace_buffer,
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k_cache.view(-1, 1, layer.tp_k_head_num, layer.qk_head_dim),
                    v_cache.view(-1, 1, layer.tp_v_head_num, layer.v_head_dim),
                    self.scale,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.kv_last_page_len,
                    1,
                    self.max_num_partitions,
                    None,
                    "auto",
                    "NHD",
                    self.logits_soft_cap,
                    self.k_scale,
                    self.v_scale,
                    None,
                    _AITER_PARTITION_SIZE_ROCM,
                )

        return o


class AiterIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indices = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):

        kv_start_idx = None
        kv_indptr = self.kv_indptr
        qo_indptr = self.qo_indptr
        paged_kernel_lens = seq_lens
        paged_kernel_lens_sum = seq_lens_sum

        bs = len(req_pool_indices)
        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            # (TODO: Kk) WA - CI test_moe_eval_accuracy_large.py
            # mha_batch_prefill reads 128 data to do computatoin
            # if real data is not long enough then original padding value 0 is used
            # but the 0 location will be made nan (noqa) in cuda graph capture mode
            # this will cause the output tensor value becomes nan
            # WA is to assure that last index of pool not changed
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )

            token_num = kv_indptr[-1]
            kv_indices[token_num:] = kv_indices[0]

            # self.max_kv_len = torch.max(paged_kernel_lens).item()

            extend_lens = seq_lens - prefix_lens

            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indices = kv_indices


class AiterMlaIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indptr = None
        self.kv_indices = None
        self.qo_indptr = None
        self.kv_last_page_len = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        bs = len(req_pool_indices)

        kv_indptr = self.attn_backend.kv_indptr

        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            qo_indptr = self.attn_backend.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    kv_lens,
                    kv_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.max_q_len = max_q_len
        self.max_kv_len = max_kv_len


class AiterMultiStepDraftBackend:
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
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                AiterAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    topk=topk,
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
        self, forward_batch: ForwardBatch, kv_indices_buffer: torch.Tensor, call_fn: int
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
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
            triton.next_power_of_2(num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(bs),
            self.page_size,
        )

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
            dtype=torch.int32,
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
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
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

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)
