"""Surgery: turn Qwen3-0.6B into a 3:1 KDA/full-attention hybrid.

Every 4th layer (0-indexed 3, 7, 11, 15, 19, 23, 27) keeps the native Qwen3
GQA full attention (with RoPE, weights inherited). All other 21 layers have
their ``self_attn`` replaced by a ``KDALayer`` (NoPE --
KDA layers never see the rotary embeddings, the decoder layer passes them in
but ``KDALayer.forward`` ignores ``position_embeddings``). KDA layers inherit
the teacher's Q/K/V/O projections via ``teacher_init_kda`` (v2 surgery;
``teacher_init=False`` reproduces v1's all-random init).

Embedding, MLP, RMSNorms and lm_head are untouched. Since the module types
change, ``save_pretrained`` is not usable; ``save_hybrid``/``load_hybrid``
persist a safetensors state dict plus a ``layout.json`` describing which
layers are KDA and their hyperparameters.
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from .layer import KDAConfig, KDALayer

__all__ = [
    "FULL_ATTN_LAYERS",
    "KDA_LAYERS",
    "build_hybrid_model",
    "save_hybrid",
    "load_hybrid",
    "verify_surgery",
    "get_attention_pairs",
]

# 3:1 hybrid over 28 layers: every 4th layer keeps full attention.
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27]
NUM_LAYERS = 28
KDA_LAYERS = [i for i in range(NUM_LAYERS) if i not in FULL_ATTN_LAYERS]

# KDA layer hyperparameters for Qwen3-0.6B (hidden 1024, H=16, head_dim=64).
KDA_HYPERPARAMS = {
    "num_heads": 16,
    "head_dim": 64,
    "short_conv_kernel_size": 4,
}

# v3: match the teacher's native head_dim (16 Q heads x 128). Per-head
# recurrent state 128x128 (4x v2), q/k/v/o projections isomorphic to the
# teacher's -- no coordinate mismatch anywhere (see teacher_init_kda_v3).
KDA_HYPERPARAMS_V3 = {
    "num_heads": 16,
    "head_dim": 128,
    "short_conv_kernel_size": 4,
}


def _kda_config_for(model) -> KDAConfig:
    return KDAConfig(
        hidden_size=model.config.hidden_size,
        rms_norm_eps=model.config.rms_norm_eps,
        **KDA_HYPERPARAMS,
    )


def teacher_init_kda(kda_layer: KDALayer, qwen_attn) -> None:
    """Inherit Q/K/V/O projection weights from a Qwen3 attention module.

    Weight mapping (teacher: 16 Q heads x 128 + 8 KV heads x 128;
    KDA: 16 heads x 64 for q/k/v):

    - ``k_proj`` / ``v_proj``: flat copy [1024, 1024] -- every 128-dim KV head
      splits into two 64-dim KDA heads (8x128 = 16x64).
    - ``q_proj``: keep the first 64 dims of each of the 16 Q heads
      (16x128 -> 16x64), preserving all heads' diversity.
    - ``o_proj``: keep the first 64 input dims of each Q head, the exact
      transpose mapping of q_proj.

    Gates, conv1d, A_log/dt_bias keep their own (official) init -- the teacher
    has no counterpart for them.
    """
    H, D = kda_layer.config.num_heads, kda_layer.config.head_dim
    tq = qwen_attn.q_proj.weight.data  # [H*128, hidden]
    tk = qwen_attn.k_proj.weight.data  # [8*128, hidden] == [H*64, hidden]
    tv = qwen_attn.v_proj.weight.data  # [8*128, hidden]
    to = qwen_attn.o_proj.weight.data  # [hidden, H*128]
    td = tq.shape[0] // H              # teacher head_dim = 128
    with torch.no_grad():
        kda_layer.q_proj.weight.copy_(
            tq.view(H, td, -1)[:, :D, :].reshape(H * D, -1))
        kda_layer.k_proj.weight.copy_(tk)
        kda_layer.v_proj.weight.copy_(tv)
        kda_layer.o_proj.weight.copy_(
            to.view(-1, H, td)[:, :, :D].reshape(-1, H * D))


def teacher_init_kda_v3(
    kda_layer: KDALayer,
    qwen_attn,
    gate_scale: float = 0.02,
    dt_bias: float = 0.0,
    a_low: float = 0.03,
    a_high: float = 0.3,
    out_scale: float = 1.0,
) -> None:
    """Function-aligned inheritance for KDA128 (v3 surgery).

    With head_dim=128 the KDA projections are *isomorphic* to the teacher's
    (16 heads x 128), so the mapping has no coordinate mismatch anywhere:

    - ``q_proj``: direct copy [2048, 1024] (16 Q heads x 128).
    - ``k_proj`` / ``v_proj``: each 128-dim KV head is duplicated for the two
      KDA heads that match the GQA pair sharing it (teacher: Q heads 2j/2j+1
      read KV head j; KDA: heads 2j/2j+1 get a copy of KV head j).
    - ``o_proj``: direct copy [1024, 2048] (same head ordering semantics).

    Parts with no teacher counterpart get a conversion-friendly init
    (GenDistill-style; from-scratch random gates made v2 "amnesiac at birth",
    median retention ~0.0014):

    - short convs: identity kernel (causal depthwise, last tap = 1, bias = 0);
    - gate projections: default init scaled by ``gate_scale`` (gates start
      near-neutral instead of near-random);
    - ``dt_bias`` < 0 so softplus(dt_bias) ~ 0 -> high state retention;
    - ``A_log`` ~ log U(a_low, a_high) (slow decay), kept fp32.
    """
    H, D = kda_layer.config.num_heads, kda_layer.config.head_dim
    tq = qwen_attn.q_proj.weight.data  # [H*128, hidden]
    tk = qwen_attn.k_proj.weight.data  # [8*128, hidden]
    tv = qwen_attn.v_proj.weight.data  # [8*128, hidden]
    to = qwen_attn.o_proj.weight.data  # [hidden, H*128]
    assert tq.shape[0] == H * D, f"teacher q_proj {tq.shape} != {H}x{D}"
    assert to.shape[1] == H * D, f"teacher o_proj {to.shape} != hidden x {H}x{D}"
    with torch.no_grad():
        kda_layer.q_proj.weight.copy_(tq)
        kda_layer.k_proj.weight.copy_(
            tk.view(-1, D, tk.shape[1]).repeat_interleave(2, dim=0).reshape(H * D, -1))
        kda_layer.v_proj.weight.copy_(
            tv.view(-1, D, tv.shape[1]).repeat_interleave(2, dim=0).reshape(H * D, -1))
        kda_layer.o_proj.weight.copy_(to)
        # Identity short convs (fla ShortConvolution: weight [dim, 1, width]).
        for conv in (kda_layer.q_conv1d, kda_layer.k_conv1d, kda_layer.v_conv1d):
            conv.weight.zero_()
            conv.weight[:, 0, -1] = 1.0
            if conv.bias is not None:
                conv.bias.zero_()
        # Near-neutral gates.
        for lin in (kda_layer.f_a_proj, kda_layer.f_b_proj,
                    kda_layer.g_a_proj, kda_layer.g_b_proj, kda_layer.b_proj):
            lin.weight.mul_(gate_scale)
        kda_layer.dt_bias.fill_(dt_bias)
        kda_layer.A_log.copy_(
            torch.log(torch.empty(H, dtype=torch.float32).uniform_(a_low, a_high)))
        if out_scale != 1.0:
            # The sigmoid output gate sits at ~0.5 when gate inputs are
            # near-neutral (reviewer blocker #4): scale the gated output
            # RMSNorm weight to compensate, restoring output magnitude.
            kda_layer.o_norm.weight.mul_(out_scale)


def build_hybrid_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    seed: int = 0,
    teacher_init: bool = True,
    kda_config: "KDAConfig | None" = None,
    init_scheme: str = "v2",
    gate_init: dict | None = None,
) -> torch.nn.Module:
    """Load Qwen3-0.6B-Base and replace non-kept attention layers with KDA.

    With ``teacher_init=True`` (default, v2 surgery) the KDA layers inherit the
    teacher's Q/K/V/O projections (see ``teacher_init_kda``); gates/convs are
    randomly initialized (seeded for reproducibility). ``teacher_init=False``
    reproduces the v1 all-random surgery. Everything else inherits the
    pretrained weights.

    ``kda_config`` overrides the default v2 hyperparameters (e.g.
    ``KDAConfig(hidden_size, **KDA_HYPERPARAMS_V3)``); ``init_scheme="v3"``
    selects the function-aligned KDA128 init (``teacher_init_kda_v3``,
    ``gate_init`` forwards its knob overrides).
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.config.use_cache = False  # KDA layers have no recurrent-state cache yet

    kda_cfg = kda_config or _kda_config_for(model)
    gen = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for idx in KDA_LAYERS:
            layer = KDALayer(kda_cfg, layer_idx=idx)
            layer = layer.to(dtype=dtype)
            # A_log / dt_bias must stay fp32 (gate parameterization is
            # numerically sensitive; fla kernels expect them in fp32).
            layer.A_log.data = layer.A_log.data.float()
            layer.dt_bias.data = layer.dt_bias.data.float()
            old = model.model.layers[idx].self_attn
            if teacher_init:
                if init_scheme == "v3":
                    teacher_init_kda_v3(layer, old, **(gate_init or {}))
                else:
                    teacher_init_kda(layer, old)
            model.model.layers[idx].self_attn = layer
            del old
    del gen

    model._kda_layout = {  # runtime annotation, also written to layout.json
        "kda_layers": KDA_LAYERS,
        "full_attn_layers": FULL_ATTN_LAYERS,
        "kda_config": kda_cfg.to_dict(),
        "base_model_path": model_path,
        # init provenance (2026-07-30 reviewer blocker #3): the artifact must
        # carry HOW its KDA layers were initialized, not just the geometry.
        "init_scheme": init_scheme if teacher_init else "none",
        "gate_init": (gate_init or {}) if (teacher_init and init_scheme == "v3") else None,
        "seed": seed,
    }
    return model.to(device)


def save_hybrid(model: torch.nn.Module, save_dir: str) -> None:
    """Save hybrid model as state_dict + layout.json (no save_pretrained)."""
    os.makedirs(save_dir, exist_ok=True)
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    # Tied weights (Qwen3-0.6B ties lm_head to embed_tokens) break safetensors
    # shared-tensor checks; keep only one copy.
    if model.config.tie_word_embeddings and "lm_head.weight" in sd:
        del sd["lm_head.weight"]
    save_file(sd, os.path.join(save_dir, "model.safetensors"))
    layout = getattr(model, "_kda_layout", None) or {
        "kda_layers": KDA_LAYERS,
        "full_attn_layers": FULL_ATTN_LAYERS,
        "kda_config": None,
        "base_model_path": None,
    }
    with open(os.path.join(save_dir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=2)


def load_hybrid(
    save_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Rebuild the hybrid from layout.json + base weights + saved state dict.

    The KDA geometry comes from layout.json's ``kda_config`` (falls back to
    the v2 default for pre-v3 checkpoints). Rebuild skips teacher-init --
    load_state_dict overwrites everything anyway (and the v2 slicing would
    shape-mismatch a v3 128-dim layer).
    """
    with open(os.path.join(save_dir, "layout.json")) as f:
        layout = json.load(f)
    kda_cfg = (KDAConfig.from_dict(layout["kda_config"])
               if layout.get("kda_config") else None)
    model = build_hybrid_model(layout["base_model_path"], dtype=dtype,
                               device=device, teacher_init=False,
                               kda_config=kda_cfg)
    sd = load_file(os.path.join(save_dir, "model.safetensors"))
    if model.config.tie_word_embeddings and "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    model.load_state_dict(sd, strict=True)
    return model.to(device)


@torch.no_grad()
def verify_surgery(original: torch.nn.Module, hybrid: torch.nn.Module) -> bool:
    """Check that every untouched weight is elementwise identical."""
    ok = True

    def check(name, a, b):
        nonlocal ok
        same = a.shape == b.shape and torch.equal(a, b)
        if not same:
            ok = False
        return f"{'OK ' if same else 'MISMATCH'} {name}"

    # Embedding / final norm / lm_head (tied).
    print(check("embed_tokens", original.model.embed_tokens.weight,
                hybrid.model.embed_tokens.weight))
    print(check("final_norm", original.model.norm.weight, hybrid.model.norm.weight))

    for i in range(original.config.num_hidden_layers):
        lo, lh = original.model.layers[i], hybrid.model.layers[i]
        for tag, mo, mh in [
            ("mlp", lo.mlp, lh.mlp),
            ("input_layernorm", lo.input_layernorm, lh.input_layernorm),
            ("post_attention_layernorm", lo.post_attention_layernorm,
             lh.post_attention_layernorm),
        ]:
            for (n, po), (n2, ph) in zip(mo.named_parameters(), mh.named_parameters()):
                assert n == n2, f"param name mismatch: {n} vs {n2}"
                r = check(f"layer{i:2d}.{tag}.{n}", po, ph)
                if "MISMATCH" in r:
                    print(r)
        if i in FULL_ATTN_LAYERS:
            for (n, po), (n2, ph) in zip(
                lo.self_attn.named_parameters(), lh.self_attn.named_parameters()
            ):
                assert n == n2, f"param name mismatch: {n} vs {n2}"
                r = check(f"layer{i:2d}.self_attn.{n} (kept)", po, ph)
                if "MISMATCH" in r:
                    print(r)
        else:
            assert isinstance(lh.self_attn, KDALayer), f"layer {i} not replaced"
    print(f"kept full-attn layers: {FULL_ATTN_LAYERS}")
    print(f"replaced KDA layers ({len(KDA_LAYERS)}): {KDA_LAYERS}")
    print("surgery verification:", "PASS" if ok else "FAIL")
    return ok


def param_count(model: torch.nn.Module) -> dict:
    """Parameter counts: total / attention modules / one KDA vs one native attn."""
    total = sum(p.numel() for p in model.parameters())
    attn = 0
    kda_one = native_one = None
    for i, layer in enumerate(model.model.layers):
        n = sum(p.numel() for p in layer.self_attn.parameters())
        attn += n
        if i in FULL_ATTN_LAYERS and native_one is None:
            native_one = n
        if i in KDA_LAYERS and kda_one is None:
            kda_one = n
    return {"total": total, "attn_total": attn,
            "per_kda_layer": kda_one, "per_native_attn_layer": native_one}


def get_attention_pairs(teacher_model: torch.nn.Module, hybrid_model: torch.nn.Module):
    """Return [(layer_idx, teacher_attn_module, student_kda_module)] for every
    replaced layer -- hook these to capture (input hidden states, teacher attn
    output) vs student KDA output for the distillation step.
    """
    pairs = []
    for idx in KDA_LAYERS:
        teacher_attn = teacher_model.model.layers[idx].self_attn
        student_kda = hybrid_model.model.layers[idx].self_attn
        assert isinstance(student_kda, KDALayer)
        pairs.append((idx, teacher_attn, student_kda))
    return pairs
