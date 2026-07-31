#!/usr/bin/env python
"""Verify micro_batch>1 padding+mask equals micro_batch=1 per-sample forward.

Critical gate before cloud restart: KDA layers must unpad via cu_seqlens so
padded positions never contaminate the recurrent state. Compares, for the
same 8 samples: per-sample last_hidden_state vs batched (max abs diff on
non-pad positions), and per-sample CE vs batched CE.
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
batch = data[200:208]  # arbitrary slice, varied lengths
lens = [len(d["input_ids"]) for d in batch]
print("sample lengths:", lens)

# per-sample forward (reference)
ref_h, ref_l = [], []
with torch.no_grad():
    for d in batch:
        ids = torch.tensor([d["input_ids"]], device="cuda")
        lab = torch.tensor([d["labels"]], device="cuda")
        h = model.model(ids, use_cache=False).last_hidden_state
        ref_h.append(h[0].float().cpu())
        logits = model.lm_head(h[:, :-1])
        l = torch.nn.functional.cross_entropy(
            logits[0].float(), lab[0, 1:], ignore_index=-100)
        ref_l.append(l.item())

# batched forward with padding + mask
maxlen = max(lens)
ids = torch.zeros((8, maxlen), dtype=torch.long)
lab = torch.full((8, maxlen), -100, dtype=torch.long)
mask = torch.zeros((8, maxlen), dtype=torch.long)
for i, d in enumerate(batch):
    ids[i, :lens[i]] = torch.tensor(d["input_ids"])
    lab[i, :lens[i]] = torch.tensor(d["labels"])
    mask[i, :lens[i]] = 1
ids, lab, mask = ids.cuda(), lab.cuda(), mask.cuda()
with torch.no_grad():
    hb = model.model(ids, attention_mask=mask, use_cache=False).last_hidden_state
    logits = model.lm_head(hb[:, :-1])
    lb = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), lab[:, 1:].reshape(-1),
        ignore_index=-100)

max_diff = 0.0
for i, n in enumerate(lens):
    d = (hb[i, :n].float().cpu() - ref_h[i]).abs().max().item()
    max_diff = max(max_diff, d)
    print(f"sample {i} (len {n:4d}): hidden max|diff| {d:.2e} | "
          f"ce per-sample {ref_l[i]:.4f}")
print(f"batched token-mean ce {lb.item():.4f}")
ok = max_diff < 2e-3
print("VERDICT:", "PASS" if ok else "FAIL", f"(max hidden diff {max_diff:.2e})")
sys.exit(0 if ok else 1)
