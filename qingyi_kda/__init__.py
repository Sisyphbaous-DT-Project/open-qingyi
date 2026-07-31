"""qingyi-kda: linearizing Qwen3-0.6B into the KDA architecture."""

from .kda_ref import kda_recurrence
from .layer import KDAConfig, KDALayer
from .surgery import (
    FULL_ATTN_LAYERS,
    KDA_LAYERS,
    build_hybrid_model,
    get_attention_pairs,
    load_hybrid,
    save_hybrid,
    verify_surgery,
)

__all__ = [
    "kda_recurrence",
    "KDAConfig",
    "KDALayer",
    "FULL_ATTN_LAYERS",
    "KDA_LAYERS",
    "build_hybrid_model",
    "save_hybrid",
    "load_hybrid",
    "verify_surgery",
    "get_attention_pairs",
]
