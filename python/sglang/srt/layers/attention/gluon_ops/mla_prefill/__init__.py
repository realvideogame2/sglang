"""FP8 D192 MLA prefill kernel sources for MI350X (gfx950 / CDNA 4).

Ported from ``AMD-Triton/gluon-kernels`` branch
``tussingh/mla-d192-prefill``. The canonical research copy still lives
there; this is the subset SGLang imports at runtime.

Public re-exports (consumed by
``sglang.srt.layers.attention.gluon_mla_prefill``):

* ``mla_prefill_d192_fwd``          non-persistent 3D-grid launcher
* ``mla_prefill_d192_ps_fwd``       AITER-style metadata-driven PS launcher
* ``mla_prefill_d192_splitk_fwd``   split-K (sk1 = SPLIT_K=1 work-stealing)
* ``prewarm_mla_d192``              compile + launch dummy instances of every
                                    variant at model-load time
* ``_gen_metadata_gpu``             metadata kernel for PS
* ``_launch_ps`` / ``_launch_non_persistent`` / ``_launch_splitk``
                                    inner launchers that optionally return
                                    the underlying ``CompiledKernel``
* ``_fp8_8w``                       the ``@gluon.jit`` factory for the
                                    8-warp FP8 kernel (used for the fast
                                    path)
"""

from .mla_prefill_d192_gfx950 import (  # noqa: F401
    mla_prefill_d192_fwd,
    mla_prefill_d192_ps_fwd,
    mla_prefill_d192_splitk_fwd,
    prewarm_mla_d192,
    _gen_metadata_gpu,
    _launch_ps,
    _launch_non_persistent,
    _launch_splitk,
    _fp8_8w,
)
