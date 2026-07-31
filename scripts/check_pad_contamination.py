#!/usr/bin/env python
"""Semantic contamination test (bitwise): does pad CONTENT change real tokens?

Same padded pair, same mask, but pads filled with eos vs random tokens.
Shapes/kernel paths identical, so any diff in real-position outputs means
padding leaked into the computation (bad); bitwise-equal means safe.
Runs in bf16 (the training dtype) on purpose.
"""
import sys

import torch

sys.path.insert(0, "/root/projects/qingyi-kda")
from qingyi_kda.surgery import load_hybrid  # noqa: E402

ROOT = "/root/projects/qingyi-kda"
model = load_hybrid(f"{ROOT}/models/cpt-checkpoints/best",
                    dtype=torch.bfloat16, device="cuda")
model.eval()
data = torch.load(f"{ROOT}/data/sft/sft_dataset.pt", weights_only=False)
d0, d1 = data[200], data[201]
la, lb = len(d0["input_ids"]), len(d1["input_ids"])
maxlen = max(la, lb)

def run(pad_ids):
    ids = pad_ids.clone()
    mask = torch.zeros((2, maxlen), dtype=torch.long)
    ids[0, :la] = torch.tensor(d0["input_ids"]); mask[0, :la] = 1
    ids[1, :lb] = torch.tensor(d1["input_ids"]); mask[1, :lb] = 1
    with torch.no_grad():
        return model.model(ids.cuda(), attention_mask=mask.cuda(),
                           use_cache=False).last_hidden_state.cpu()

eos = 151643  # <|endoftext|>
pads_eos = torch.full((2, maxlen), eos, dtype=torch.long)
g = torch.Generator().manual_seed(42)
pads_rand = torch.randint(1000, 50000, (2, maxlen), generator=g)

h_eos = run(pads_eos)
h_rand = run(pads_rand)
d_row0 = (h_eos[0, :la].float() - h_rand[0, :la].float()).abs().max().item()
d_row1 = (h_eos[1, :lb].float() - h_rand[1, :lb].float()).abs().max().item()
print(f"row0 real positions (len {la}): max|diff| {d_row0:.3e}")
print(f"row1 real positions (len {lb}): max|diff| {d_row1:.3e}")
ok = d_row0 == 0.0 and d_row1 == 0.0
print("VERDICT:", "PASS (pads inert, batching semantically safe)" if ok
      else "FAIL (pad content leaks into real tokens)")
sys.exit(0 if ok else 1)
