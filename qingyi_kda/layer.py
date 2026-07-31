"""Full KDA (Kimi Delta Attention) layer, mirroring the official
``KimiDeltaAttention`` from moonshotai/Kimi-Linear-48B-A3B-Instruct
(modeling_kimi.py) and the Kimi Linear paper (arXiv:2510.26692, section 4).

fla-core 0.5.1 ships kernels only (``fla.ops``) -- no ``fla.layers`` module --
so this layer is assembled here from fla primitives:

- ``fla.modules.ShortConvolution``       (causal depthwise conv1d + SiLU)
- ``fla.ops.kda.gate.fused_kda_gate``    (g = -exp(A_log) * softplus(x + dt_bias))
- ``fla.ops.kda.chunk_kda``              (chunkwise KDA kernel, verified vs reference)
- ``fla.modules.FusedRMSNormGated``      (head-wise gated RMSNorm, sigmoid gate)

Structure per token (paper Eq. 10 and official code):
    q, k = L2Norm(SiLU(ShortConv(W_{q/k} x)))     (L2 norm inside the kernel)
    v    = SiLU(ShortConv(W_v x))
    g    = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)   (log-space decay)
    beta = sigmoid(W_b x)
    o    = KDA(q, k, v, g, beta)                  (chunk_kda)
    o    = W_o [ RMSNorm(o) * sigmoid(g_b(g_a(x))) ]

No rotary position embedding is used: KDA encodes position through its
data-dependent decay (paper section 3 / 6.1). ``position_embeddings`` is
accepted for drop-in compatibility with Qwen3DecoderLayer and ignored.

Incremental decoding (2026-07-28): when ``past_key_values`` is a
``HybridKDACache`` the layer carries its recurrent state S and the three
ShortConvolution windows across steps -- prefill runs ``chunk_kda`` with
``output_final_state=True``, single-token steps run ``fused_recurrent_kda``.
Plain ``DynamicCache``/None keeps the old stateless behavior (full forward
only; generation then needs ``use_cache=False``).
"""

import torch
import torch.nn as nn
from einops import rearrange
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate

__all__ = ["KDALayer", "KDAConfig"]


class KDAConfig:
    """Hyperparameters of one KDA layer (serializable for layout.json)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 16,
        head_dim: int = 64,
        short_conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.short_conv_kernel_size = short_conv_kernel_size
        self.rms_norm_eps = rms_norm_eps

    def to_dict(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "short_conv_kernel_size": self.short_conv_kernel_size,
            "rms_norm_eps": self.rms_norm_eps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KDAConfig":
        return cls(**d)


class KDALayer(nn.Module):
    """Drop-in replacement for Qwen3Attention (same forward signature).

    Dimension choice for Qwen3-0.6B (hidden=1024, native attention:
    16 Q heads x 128 + 8 KV heads x 128, ~6.29M params):
    H=16 heads, head_k_dim = head_v_dim = 64 -> key/value total dim 1024 each.
    This keeps every head's state S [64, 64] small, gives 16 independent
    channel-wise decay gates, and lands at ~5.49M params (~87% of the native
    attention layer, same order of magnitude) while matching the 3:1 hybrid
    layout of the paper.
    """

    def __init__(self, config: KDAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        d = config.hidden_size
        H = config.num_heads
        D = config.head_dim
        proj = H * D

        # q/k/v projections (no bias, as in the official implementation).
        self.q_proj = nn.Linear(d, proj, bias=False)
        self.k_proj = nn.Linear(d, proj, bias=False)
        self.v_proj = nn.Linear(d, proj, bias=False)

        # Short causal depthwise conv + SiLU on q/k/v (paper Fig. 3).
        self.q_conv1d = ShortConvolution(proj, config.short_conv_kernel_size, activation="silu")
        self.k_conv1d = ShortConvolution(proj, config.short_conv_kernel_size, activation="silu")
        self.v_conv1d = ShortConvolution(proj, config.short_conv_kernel_size, activation="silu")

        # Forget gate: low-rank (rank = head_dim) projection, then
        # g = -exp(A_log) * softplus(raw + dt_bias) computed by fused_kda_gate.
        self.f_a_proj = nn.Linear(d, D, bias=False)
        self.f_b_proj = nn.Linear(D, proj, bias=False)
        # fp32 parameters, matching the official init: A ~ U(1, 16).
        self.A_log = nn.Parameter(
            torch.log(torch.empty(H, dtype=torch.float32).uniform_(1, 16))
        )
        self.dt_bias = nn.Parameter(torch.zeros(proj, dtype=torch.float32))

        # Write gate beta (sigmoid applied in forward).
        self.b_proj = nn.Linear(d, H, bias=False)

        # Output gate: low-rank (rank = head_dim), sigmoid, applied inside the
        # gated RMSNorm (paper Eq. 10).
        self.g_a_proj = nn.Linear(d, D, bias=False)
        self.g_b_proj = nn.Linear(D, proj, bias=False)
        self.o_norm = FusedRMSNormGated(D, eps=config.rms_norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(proj, d, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,   # unused: KDA is NoPE by design
        attention_mask=None,        # unused: no padding in our training setup
        past_key_values=None,       # HybridKDACache for incremental decoding
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        state = (past_key_values.get_kda_state(self.layer_idx)
                 if past_key_values is not None
                 and hasattr(past_key_values, "get_kda_state") else None)
        S = state.get("S") if state else None
        want_state = (past_key_values is not None
                      and hasattr(past_key_values, "set_kda_state"))
        # decode step = single token with an existing recurrent state
        decoding = hidden_states.shape[1] == 1 and S is not None

        conv_cache = lambda key: state.get(key) if state else None
        q, cq = self.q_conv1d(x=self.q_proj(hidden_states),
                              cache=conv_cache("conv_q"),
                              output_final_state=True)
        k, ck = self.k_conv1d(x=self.k_proj(hidden_states),
                              cache=conv_cache("conv_k"),
                              output_final_state=True)
        v, cv = self.v_conv1d(x=self.v_proj(hidden_states),
                              cache=conv_cache("conv_v"),
                              output_final_state=True)

        H, D = self.config.num_heads, self.config.head_dim
        q = rearrange(q, "b t (h d) -> b t h d", d=D)
        k = rearrange(k, "b t (h d) -> b t h d", d=D)
        v = rearrange(v, "b t (h d) -> b t h d", d=D)

        # Forget gate in log space (fp32 out), write gate post-sigmoid.
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = rearrange(g, "b t (h d) -> b t h d", d=D)
        g = fused_kda_gate(g, self.A_log, self.dt_bias)
        beta = self.b_proj(hidden_states).float().sigmoid()

        # Core KDA operator; q/k L2 norm fused in the kernel (required for
        # eigenvalue stability of the delta-rule transition).
        if decoding:
            o, S = fused_recurrent_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=S, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            o, S = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=S, output_final_state=want_state,
                use_qk_l2norm_in_kernel=True,
            )

        if state is not None:
            state.update({"S": S, "conv_q": cq, "conv_k": ck, "conv_v": cv})
        elif want_state:
            past_key_values.set_kda_state(
                self.layer_idx,
                {"S": S, "conv_q": cq, "conv_k": ck, "conv_v": cv})

        # Gated head-wise RMSNorm + output projection.
        gate = self.g_b_proj(self.g_a_proj(hidden_states))
        gate = rearrange(gate, "b t (h d) -> b t h d", d=D)
        o = self.o_norm(o, gate)
        o = self.o_proj(rearrange(o, "b t h d -> b t (h d)"))
        return o, None
