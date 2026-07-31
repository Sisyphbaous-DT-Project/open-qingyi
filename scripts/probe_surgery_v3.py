#!/usr/bin/env python
"""Isolated functional acceptance probe (HALO-style) for surgery checkpoints.

For every replaced layer: take the TEACHER's layer input hidden state
(detached), feed it to both the teacher attention module and the student KDA
module, compare outputs (cosine / relative L2). This tests *function*
inheritance -- unlike v2's circular weight-slice cosine check.

Also exercises chunk_kda at the checkpoint's head_dim (T=2048), doubling as
the fla kernel smoke test.

Usage:
  .venv/bin/python scripts/probe_surgery_v3.py models/qingyi-hybrid-init-v3
  .venv/bin/python scripts/probe_surgery_v3.py models/qingyi-hybrid-init-v2
"""
import argparse
import sys

sys.path.insert(0, "/root/projects/qingyi-kda")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from qingyi_kda.surgery import KDA_LAYERS, load_hybrid  # noqa: E402

ROOT = "/root/projects/qingyi-kda"
BASE = f"{ROOT}/models/Qwen3-0.6B-Base"
HELDOUT_CACHE = f"{ROOT}/data/heldout_v2.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    args = ap.parse_args()
    device = "cuda"

    blob = torch.load(HELDOUT_CACHE, weights_only=False)
    ids = blob["batches"]["zhwiki"][0][:1].to(device)  # [1, 2048] real text

    teacher = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map=device)
    teacher.eval().requires_grad_(False)
    student = load_hybrid(args.checkpoint, dtype=torch.bfloat16, device=device)
    student.eval()

    # Capture each replaced layer's input (args incl. position_embeddings and
    # mask kwargs) during a teacher forward, via pre-hooks.
    captured = {}
    handles = []

    def make_hook(idx):
        def hook(module, args, kwargs):
            kwargs = dict(kwargs)
            if args:  # hidden_states positional
                x, rest = args[0], list(args[1:])
            else:     # hidden_states as kwarg (this transformers version)
                x = kwargs.pop("hidden_states")
                rest = []
            captured[idx] = (
                x.detach(),
                [a.detach() if torch.is_tensor(a) else a for a in rest],
                {k: (v.detach() if torch.is_tensor(v) else v)
                 for k, v in kwargs.items()},
            )
        return hook

    for idx in KDA_LAYERS:
        h = teacher.model.layers[idx].self_attn.register_forward_pre_hook(
            make_hook(idx), with_kwargs=True)
        handles.append(h)
    with torch.no_grad():
        teacher.model(ids, use_cache=False)
    for h in handles:
        h.remove()
    assert len(captured) == len(KDA_LAYERS), \
        f"captured {len(captured)} != {len(KDA_LAYERS)}"

    print(f"\n=== probe: {args.checkpoint} ===")
    print(f"{'layer':>5} {'cosine':>8} {'relL2':>8}")
    cos_all, l2_all = [], []
    with torch.no_grad():
        for idx in KDA_LAYERS:
            x, t_rest, t_kwargs = captured[idx]
            t_out = teacher.model.layers[idx].self_attn(x, *t_rest, **t_kwargs)
            t_out = t_out[0] if isinstance(t_out, tuple) else t_out
            s_out = student.model.layers[idx].self_attn(x)[0]
            tf = t_out.float().reshape(-1, t_out.shape[-1])
            sf = s_out.float().reshape(-1, s_out.shape[-1])
            cos = F.cosine_similarity(tf, sf, dim=-1).mean().item()
            rel = (tf - sf).norm().item() / tf.norm().item()
            cos_all.append(cos)
            l2_all.append(rel)
            print(f"{idx:>5} {cos:>8.4f} {rel:>8.4f}")
    print(f"  mean cosine {sum(cos_all)/len(cos_all):.4f}  "
          f"min {min(cos_all):.4f} | mean relL2 {sum(l2_all)/len(l2_all):.4f}")


if __name__ == "__main__":
    main()
