"""Hybrid cache for Qwen3+KDA hybrid models.

Standard DynamicCache carries KV for the remaining full-attention layers;
this subclass additionally carries per-KDA-layer recurrent state:
    S       [b, h, d_k, d_v]  delta-rule state (fp32, from the kernels)
    conv_*  [b, proj, k-1]    ShortConvolution input windows (q/k/v)

Usage: model.generate(..., past_key_values=HybridKDACache())
"""
from transformers.cache_utils import DynamicCache

__all__ = ["HybridKDACache"]


class HybridKDACache(DynamicCache):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        # layer_idx -> {"S": ..., "conv_q": ..., "conv_k": ..., "conv_v": ...}
        self.kda_states = {}

    def get_kda_state(self, layer_idx: int):
        return self.kda_states.get(layer_idx)

    def set_kda_state(self, layer_idx: int, state: dict):
        self.kda_states[layer_idx] = state
