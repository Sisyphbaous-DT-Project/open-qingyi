#!/usr/bin/env python
"""Smoke test for the 3:1 KDA hybrid model.

- Load the hybrid on GPU (bf16), greedy-generate 20 tokens from
  "The capital of France is". The KDA layers are randomly initialized, so the
  output text is expected to be garbage -- we only check shapes and numeric
  health (no NaN/Inf, finite logits).
- One next-token loss + backward + a single bitsandbytes Adam8bit step.
  Reports loss, grad norm, peak VRAM, and wall time of the step.
- Demonstrates get_attention_pairs() for the upcoming distillation stage.
"""

import sys
import time

import torch

sys.path.insert(0, "/root/projects/qingyi-kda")

import bitsandbytes as bnb
from transformers import AutoTokenizer

from qingyi_kda.surgery import (
    FULL_ATTN_LAYERS,
    KDA_LAYERS,
    build_hybrid_model,
    get_attention_pairs,
)

MODEL_PATH = "/root/projects/qingyi-kda/models/Qwen3-0.6B-Base"
PROMPT = "The capital of France is"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = build_hybrid_model(MODEL_PATH, dtype=torch.bfloat16, device="cuda")
    print(f"KDA layers ({len(KDA_LAYERS)}): {KDA_LAYERS}")
    print(f"kept full-attn layers: {FULL_ATTN_LAYERS}")

    # ---- generation smoke (no cache: KDA layers have no state cache yet) ----
    print("=" * 70)
    print("greedy generation (20 tokens, use_cache=False)")
    print("=" * 70)
    model.eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=20, do_sample=False, use_cache=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"prompt : {PROMPT!r}")
    print(f"output : {text!r}")
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits
    finite = torch.isfinite(logits).all().item()
    print(f"logits finite: {finite}  shape={tuple(logits.shape)}")

    # ---- one training step: loss + backward + Adam8bit ----
    print("=" * 70)
    print("single training step (next-token loss, bf16, Adam8bit)")
    print("=" * 70)
    model.train()
    # Only the KDA layers are trainable in this smoke test; freeze the rest to
    # mirror the distillation setup and keep VRAM low.
    for name, p in model.named_parameters():
        p.requires_grad = any(f"layers.{i}." in name and ".self_attn." in name
                              for i in KDA_LAYERS)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params (KDA layers only): {trainable:,}")

    opt = bnb.optim.Adam8bit(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    text = "The capital of France is Paris. The capital of Germany is Berlin."
    batch = tok(text, return_tensors="pt").input_ids.cuda()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    logits = model(batch, use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)),
        batch[:, 1:].reshape(-1),
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1e9
    )
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"loss          : {loss.item():.4f}")
    print(f"grad norm     : {grad_norm.item():.4f} (finite: {torch.isfinite(grad_norm).item()})")
    print(f"peak VRAM     : {peak:.2f} GiB")
    print(f"step time     : {dt:.2f} s (first step incl. triton autotune)")

    # ---- distillation hook plumbing ----
    print("=" * 70)
    print("get_attention_pairs (for distillation)")
    print("=" * 70)
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    pairs = get_attention_pairs(teacher, model)
    print(f"pairs: {len(pairs)} (expect {len(KDA_LAYERS)})")
    for idx, t_attn, s_kda in pairs[:3]:
        print(f"  layer {idx:2d}: teacher={type(t_attn).__name__}  student={type(s_kda).__name__}")
    print("  ...")

    ok = finite and torch.isfinite(grad_norm).item() and loss.item() == loss.item()
    print("=" * 70)
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
