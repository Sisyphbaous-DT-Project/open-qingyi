#!/usr/bin/env python
"""Bisect the batch-vs-single mismatch found by check_batch_equiv.py.

Test 1: batch the SAME sample twice, all-ones mask -> if hidden differs,
        batch>1 itself is broken somewhere.
Test 2: layer-by-layer hook diff for a padded batch vs single -> first
        divergent layer, tagged KDA/GQA.
"""
import sys

import torch

sys.path.insert(0, "/root/projects/qingyi-kda")
from qingyi_kda.surgery import FULL_ATTN_LAYERS, load_hybrid  # noqa: E402

ROOT = "/root/projects/qingyi-kda"
model = load_hybrid(f"{ROOT}/models/cpt-checkpoints/best",
                    dtype=torch.bfloat16, device="cuda")
model.eval()
data = torch.load(f"{ROOT}/data/sft/sft_dataset.pt", weights_only=False)
d0 = data[200]
ids1 = torch.tensor([d0["input_ids"]], device="cuda")

# ---- test 1: same sample x2, all-ones mask ----
ids2 = ids1.repeat(2, 1)
mask2 = torch.ones_like(ids2)
with torch.no_grad():
    h_single = model.model(ids1, use_cache=False).last_hidden_state
    h_batch = model.model(ids2, attention_mask=mask2,
                          use_cache=False).last_hidden_state
diff1 = (h_batch[0].float() - h_single[0].float()).abs().max().item()
diff1b = (h_batch[1].float() - h_single[0].float()).abs().max().item()
print(f"TEST1 same-sample batch: row0 diff {diff1:.2e}, row1 diff {diff1b:.2e}")

# ---- test 2: layer-wise divergence with real padding ----
d1 = data[201]
la, lb = len(d0["input_ids"]), len(d1["input_ids"])
maxlen = max(la, lb)
ids = torch.zeros((2, maxlen), dtype=torch.long)
mask = torch.zeros((2, maxlen), dtype=torch.long)
ids[0, :la] = torch.tensor(d0["input_ids"]); mask[0, :la] = 1
ids[1, :lb] = torch.tensor(d1["input_ids"]); mask[1, :lb] = 1
ids, mask = ids.cuda(), mask.cuda()

saved = {}
def hook(name):
    def fn(mod, args, out):
        saved[name] = out[0].detach() if isinstance(out, tuple) else out.detach()
    return fn

handles = []
for i, layer in enumerate(model.model.layers):
    handles.append(layer.register_forward_hook(hook(f"L{i}")))
with torch.no_grad():
    model.model(ids, attention_mask=mask, use_cache=False)
    hs = {}
    for i, layer in enumerate(model.model.layers):
        saved.clear()
        model.model(ids1, use_cache=False)
for h in handles:
    h.remove()

# rerun capturing singles layer by layer is heavy; instead capture both in one pass
saved_b, saved_s = {}, {}
def hook2(store, name):
    def fn(mod, args, out):
        store[name] = (out[0] if isinstance(out, tuple) else out).detach()
    return fn
handles = []
for i, layer in enumerate(model.model.layers):
    handles.append(layer.register_forward_hook(hook2(saved_b, f"L{i}")))
with torch.no_grad():
    model.model(ids, attention_mask=mask, use_cache=False)
for h in handles:
    h.remove()
handles = []
for i, layer in enumerate(model.model.layers):
    handles.append(layer.register_forward_hook(hook2(saved_s, f"L{i}")))
with torch.no_grad():
    model.model(ids1, use_cache=False)
for h in handles:
    h.remove()

print(f"TEST2 padded batch vs single (row0, len {la}):")
for i in range(len(model.model.layers)):
    a = saved_b[f"L{i}"][0, :la].float()
    b = saved_s[f"L{i}"][0].float()
    diff = (a - b).abs().max().item()
    tag = "GQA" if i in FULL_ATTN_LAYERS else "KDA"
    flag = " <-- diverges" if diff > 1e-2 else ""
    print(f"  layer {i:2d} ({tag}): max|diff| {diff:.2e}{flag}")
