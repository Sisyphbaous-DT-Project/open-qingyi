#!/usr/bin/env python
"""Decisive fp32 check: is batch>1 vs batch=1 divergence pure bf16 rounding?

Loads the hybrid in float32 and repeats:
  TEST1 same sample x2, all-ones mask (no padding at all)
  TEST2 padded pair, row0 (shorter seq, real right-padding)
If fp32 diffs are ~1e-5, the bf16 divergence was numerics, batching is safe.
"""
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, "/root/projects/qingyi-kda")
from qingyi_kda.surgery import load_hybrid  # noqa: E402

ROOT = "/root/projects/qingyi-kda"
model = load_hybrid(f"{ROOT}/models/cpt-checkpoints/best",
                    dtype=torch.float32, device="cuda")
model.eval()
data = torch.load(f"{ROOT}/data/sft/sft_dataset.pt", weights_only=False)
d0, d1 = data[200], data[201]
ids1 = torch.tensor([d0["input_ids"]], device="cuda")

with torch.no_grad():
    h_single = model.model(ids1, use_cache=False).last_hidden_state
    # TEST1: identical twins, all-ones mask
    ids2 = ids1.repeat(2, 1)
    mask2 = torch.ones_like(ids2)
    h_b = model.model(ids2, attention_mask=mask2,
                      use_cache=False).last_hidden_state
    d_t1 = (h_b[0] - h_single[0]).abs().max().item()
    scale = h_single.abs().max().item()
    print(f"TEST1 fp32 identical-batch: max|diff| {d_t1:.3e} "
          f"(hidden scale {scale:.1f})")

    # TEST2: real right-padding pair
    la, lb = len(d0["input_ids"]), len(d1["input_ids"])
    maxlen = max(la, lb)
    ids = torch.zeros((2, maxlen), dtype=torch.long)
    mask = torch.zeros((2, maxlen), dtype=torch.long)
    ids[0, :la] = torch.tensor(d0["input_ids"]); mask[0, :la] = 1
    ids[1, :lb] = torch.tensor(d1["input_ids"]); mask[1, :lb] = 1
    h_p = model.model(ids.cuda(), attention_mask=mask.cuda(),
                      use_cache=False).last_hidden_state
    d_t2 = (h_p[0, :la].cpu() - h_single[0].cpu()).abs().max().item()
    print(f"TEST2 fp32 padded-pair row0: max|diff| {d_t2:.3e}")

ok = max(d_t1, d_t2) < 1e-4
print("VERDICT:", "PASS (bf16 noise, batching safe)" if ok
      else "FAIL (real bug, do NOT batch)")
sys.exit(0 if ok else 1)
