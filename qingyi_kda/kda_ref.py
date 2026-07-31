"""Pure PyTorch reference implementation of the KDA (Kimi Delta Attention) recurrence.

This implements the recurrent form of KDA as defined in the Kimi Linear paper
(arXiv:2510.26692), Eq. (1):

    S_t = (I - beta_t * k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t * k_t v_t^T
    o_t = S_t^T q_t

which expands to the delta-rule form actually computed here:

    S~_t = Diag(alpha_t) S_{t-1}                     (per-channel decay)
    S_t  = S~_t + beta_t * k_t (v_t - S~_t^T k_t)^T  (delta-rule write)
    o_t  = S_t^T q_t                                 (readout, post-update)

Conventions follow fla-core's `fla.ops.kda.chunk_kda` exactly (fla is the source
of truth because pretrained weights must stay compatible with it):

- ``alpha_t = exp(g_t)``: the forget gate is given in log space, one scalar per
  key channel (fine-grained / channel-wise decay, DPLR diagonal part).
- ``beta_t`` is expected in post-sigmoid space, i.e. already in (0, 1).
- The state is stored as ``S[B, HV, K, V]`` (keys along dim -2, values along -1).
- ``scale`` (default ``K ** -0.5``) is applied to q before the recurrence.
- GVA (grouped value attention): if ``HV > H``, q/k heads are repeated
  ``G = HV // H`` times so every value head gets its own decay/beta state.

Everything is computed in fp32 regardless of the input dtype; the g accumulation
and exponentiation are the numerically sensitive parts. This is a slow O(T)
token-by-token loop meant only as a ground-truth reference for testing.
"""

import torch

__all__ = ["kda_recurrence"]


def kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Token-by-token KDA recurrence in fp32.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            Per-channel decay gates in log space of shape ``[B, T, HV, K]``.
            Values should be <= 0 so that ``exp(g)`` is a decay in (0, 1].
        beta (torch.Tensor):
            Write gates (post-sigmoid, in (0, 1)) of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor applied to q. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[B, HV, K, V]``.
        output_final_state (bool):
            Whether to return the final state.

    Returns:
        A tuple ``(o, S)`` where ``o`` has shape ``[B, T, HV, V]`` (cast back to
        the dtype of ``v``) and ``S`` has shape ``[B, HV, K, V]`` (fp32) if
        ``output_final_state`` else ``None``.
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    assert HV % H == 0, f"HV ({HV}) must be divisible by H ({H})"
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    # Work in fp32 end to end: exp(g) and the running state are numerically
    # sensitive and must not be accumulated in low precision.
    q, k, v, g, beta = (x.to(torch.float32) for x in (q, k, v, g, beta))

    # GVA: expand q/k heads to the value-head count so that every value head
    # owns an independent [K, V] state with its own decay and beta.
    q = q.repeat_interleave(G, dim=2) * scale  # [B, T, HV, K]
    k = k.repeat_interleave(G, dim=2)          # [B, T, HV, K]

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    for t in range(T):
        q_t, k_t, v_t, g_t, b_t = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        # Decay: Diag(alpha_t) S with alpha_t = exp(g_t) applied per key channel.
        S = S * g_t[..., None].exp()
        # Delta-rule write: S += beta_t * k_t (v_t - S^T k_t)^T.
        # (k_t[..., None] * S).sum(-2) contracts the K dim -> [B, HV, V].
        S = S + torch.einsum(
            "b h k, b h v -> b h k v",
            b_t[..., None] * k_t,
            v_t - (k_t[..., None] * S).sum(-2),
        )
        # Readout from the post-update state: o_t = S_t^T q_t.
        o[:, t] = torch.einsum("b h k, b h k v -> b h v", q_t, S)

    if not output_final_state:
        S = None
    return o.to(dtype), S
