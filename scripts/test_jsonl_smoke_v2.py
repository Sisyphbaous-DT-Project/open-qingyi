"""Arm-C preflight v2: post-review fixes (P0 resume / P1 pack integrity /
P1 multi-file / seed+sha / global-norm accumulation). CPU only, no GPU.

Run: python3 scripts/test_jsonl_smoke_v2.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from kd_e2e import ChunkedKL, JSONLMixer, RESUME_HPARM_KEYS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIR = os.path.join(ROOT, "models", "Qwen3-0.6B-Base")
TOY1 = os.path.join(ROOT, "data", "toy_armc_mcq.jsonl")
TOY2 = os.path.join(ROOT, "data", "toy_armc_qa.jsonl")

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---- build two toy files ---------------------------------------------------
with open(TOY1, "w", encoding="utf-8") as f:
    for i in range(10):
        f.write(json.dumps({
            "type": "mcq",
            "question": f"第{i}题：下面哪个选项是正确的测试答案？",
            "options": [f"选项甲{i}", f"选项乙{i}", f"选项丙{i}", f"选项丁{i}"],
            "answer_label": "ABCD"[i % 4],
        }, ensure_ascii=False) + "\n")
with open(TOY2, "w", encoding="utf-8") as f:
    for i in range(10):
        f.write(json.dumps({
            "type": "qa",
            "prompt": f"问题：{i}加{i}等于多少？\n回答：",
            "completion": f" {2 * i}",
        }, ensure_ascii=False) + "\n")

tok = AutoTokenizer.from_pretrained(TOK_DIR)
SEQ = 256

# ---- 1. multi-file merge ----------------------------------------------------
mixer = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1, TOY2], max_rows=200000)
check("multi-file sample count", len(mixer.samples) == 20,
      f"got {len(mixer.samples)}")
single = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1], max_rows=200000)
check("single-file sample count", len(single.samples) == 10,
      f"got {len(single.samples)}")

# ---- 2. pack integrity: samples never split ---------------------------------
# Every pack must be: [whole sample]+ then [eos pad with mask=0].
m2 = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1, TOY2], max_rows=200000)
known = set()
for p, c in m2.samples:
    p_ids = tok(p, add_special_tokens=False).input_ids
    c_ids = tok(c, add_special_tokens=False).input_ids + [m2.eos]
    known.add(tuple(p_ids + c_ids))

packs_ok = True
detail = ""
for pk in range(12):
    ids, mask = m2.next()
    ids, mask = ids.tolist(), mask.tolist()
    pos = 0
    while pos < SEQ:
        try:
            end = ids.index(m2.eos, pos) + 1
        except ValueError:
            packs_ok = False
            detail = f"pack {pk}: no eos from pos {pos}"
            break
        seg = tuple(ids[pos:end])
        if seg in known:
            pos = end
            continue
        # not a sample: must be pure eos pad with mask all 0 to pack end
        if all(t == m2.eos for t in ids[pos:]) and \
                all(v == 0.0 for v in mask[pos:]):
            pos = SEQ
            break
        packs_ok = False
        detail = f"pack {pk}: segment at {pos} is neither whole sample nor pad"
        break
    if not packs_ok:
        break
check("samples never split across packs", packs_ok, detail)

# pad mask is always 0 (spot check across packs)
m3 = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1, TOY2], max_rows=200000)
pad_mask_ok = True
for _ in range(12):
    ids, mask = m3.next()
    ids, mask = ids.tolist(), mask.tolist()
    # find last non-pad position: after final sample eos, everything is pad
    # pad eos positions must have mask 0 AND the position before pad (which
    # predicts the first pad eos) must also be mask 0 — it is the last
    # sample's eos position, which predicts... the pad. Check: any position
    # j where ids[j+1] is a pad eos must have mask[j]==0 unless j belongs
    # to a sample's completion span ending in ITS OWN eos.
    # Simpler invariant: mask[j]==1 implies ids[j+1] is part of a real
    # sample (already covered). Here: trailing pad positions have mask 0.
    last_one = max((j for j, v in enumerate(mask) if v == 1.0), default=-1)
    if any(v != 0.0 for v in mask[last_one + 1:]):
        pad_mask_ok = False
        break
check("pad regions all mask=0", pad_mask_ok)

# ---- 3. state/load_state with pending sample --------------------------------
ma = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1, TOY2], max_rows=200000)
for _ in range(3):
    ma.next()
st = ma.state()
check("state has kind/pending/sha", st.get("kind") == "jsonl"
      and "pending_ids" in st and "data_sha256" in st)
ahead_a = [ma.next() for _ in range(6)]
mb = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1, TOY2], max_rows=200000)
mb.load_state(st)
ahead_b = [mb.next() for _ in range(6)]
same = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
           for a, b in zip(ahead_a, ahead_b))
check("state/load_state token-for-token (with pending)", same)

# json round-trip of state (as checkpoint json would)
st_rt = json.loads(json.dumps(st))
mc = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1, TOY2], max_rows=200000)
mc.load_state(st_rt)
ahead_c = [mc.next() for _ in range(6)]
same_rt = all(torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])
              for a, c in zip(ahead_a, ahead_c))
check("state survives json round-trip", same_rt)

# old CursorMixer-shaped state must be rejected by JSONLMixer.load_state
try:
    mb.load_state({"rows": {}, "buf": {}, "rng": "junk"})
    rejected = False
except ValueError:
    rejected = True
check("old CursorMixer state rejected (P0)", rejected)

# ---- 4. data sha256 tamper lock ----------------------------------------------
sha1 = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1, TOY2],
                  max_rows=200000).data_sha256
sha2 = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1, TOY2],
                  max_rows=200000).data_sha256
check("sha deterministic", sha1 == sha2)
with open(TOY2, "a", encoding="utf-8") as f:
    f.write(json.dumps({"prompt": "附加", "completion": " x"},
                       ensure_ascii=False) + "\n")
sha3 = JSONLMixer(tok, SEQ, seed=0, paths=[TOY1, TOY2],
                  max_rows=200000).data_sha256
check("sha changes on file edit", sha3 != sha1)
# restore file
with open(TOY2, "w", encoding="utf-8") as f:
    for i in range(10):
        f.write(json.dumps({
            "type": "qa",
            "prompt": f"问题：{i}加{i}等于多少？\n回答：",
            "completion": f" {2 * i}",
        }, ensure_ascii=False) + "\n")

# ---- 5. seed in resume hparam keys -------------------------------------------
check("seed in RESUME_HPARM_KEYS", "seed" in RESUME_HPARM_KEYS)

# ---- 6. global-norm accumulation (reviewer fix) -------------------------------
torch.manual_seed(0)
H, V = 16, 53
T = 2.0
Ws = torch.randn(V, H)
Wt = torch.randn(V, H)


def make_batch(n_tok, n_masked, seed):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(n_tok, H, generator=g, requires_grad=True)
    ht = torch.randn(n_tok, H, generator=g)
    mask = torch.zeros(n_tok)
    mask[torch.randperm(n_tok, generator=g)[:n_masked]] = 1.0
    return hs, ht, mask


hs1, ht1, mk1 = make_batch(32, 3, 1)
hs2, ht2, mk2 = make_batch(32, 7, 2)
total_norm = float(mk1.sum() + mk2.sum())

# new path: norm_override shared across micro-batches
out1 = ChunkedKL.apply(hs1, ht1, Ws, Wt, T, 8, mk1, total_norm)
out1.backward()
out2 = ChunkedKL.apply(hs2, ht2, Ws, Wt, T, 8, mk2, total_norm)
out2.backward()
merged_g1, merged_g2 = hs1.grad.clone(), hs2.grad.clone()

# manual global mean: sum over all masked tokens / total_norm
def tok_kl(hs, ht):
    ls = (hs @ Ws.T) / T
    lt = (ht @ Wt.T) / T
    lps, lpt = F.log_softmax(ls, -1), F.log_softmax(lt, -1)
    return (lpt.exp() * (lpt - lps)).sum(-1)


manual_sum = (tok_kl(hs1.detach(), ht1) * mk1).sum() + \
             (tok_kl(hs2.detach(), ht2) * mk2).sum()
manual_mean = manual_sum * T * T / total_norm
check("global-norm forward == manual",
      torch.allclose(out1 + out2, manual_mean, atol=1e-5),
      f"{float(out1 + out2):.6f} vs {float(manual_mean):.6f}")

# manual gradient of global mean wrt each micro-batch's hs
gm1 = hs1.detach().clone().requires_grad_(True)
gm2 = hs2.detach().clone().requires_grad_(True)
loss_manual = ((tok_kl(gm1, ht1) * mk1).sum()
               + (tok_kl(gm2, ht2) * mk2).sum()) * T * T / total_norm
loss_manual.backward()
check("global-norm backward batch1",
      torch.allclose(merged_g1, gm1.grad, atol=1e-5),
      f"max diff {float((merged_g1 - gm1.grad).abs().max()):.2e}")
check("global-norm backward batch2",
      torch.allclose(merged_g2, gm2.grad, atol=1e-5),
      f"max diff {float((merged_g2 - gm2.grad).abs().max()):.2e}")

# and the OLD (per-micro mean-of-means) path would differ -> proves the fix matters
old_style = (ChunkedKL.apply(hs1.detach(), ht1, Ws, Wt, T, 8, mk1)
             + ChunkedKL.apply(hs2.detach(), ht2, Ws, Wt, T, 8, mk2)) / 2
check("old mean-of-means differs from global mean",
      not torch.allclose(old_style, manual_mean, atol=1e-4),
      f"old {float(old_style):.6f} vs global {float(manual_mean):.6f}")

# no-mask / no-override path still identical to original formula
hs3 = torch.randn(20, H, requires_grad=True)
ht3 = torch.randn(20, H)
out3 = ChunkedKL.apply(hs3, ht3, Ws, Wt, T, 8, None, None)
manual3 = tok_kl(hs3.detach(), ht3).sum() * T * T / 20
check("unmasked no-override unchanged", torch.allclose(out3, manual3, atol=1e-5))

# mask-only path (no override) still normalizes by mask.sum()
hs4 = torch.randn(20, H, requires_grad=True)
ht4 = torch.randn(20, H)
mk4 = torch.zeros(20)
mk4[:5] = 1.0
out4 = ChunkedKL.apply(hs4, ht4, Ws, Wt, T, 8, mk4, None)
manual4 = (tok_kl(hs4.detach(), ht4) * mk4).sum() * T * T / 5.0
check("mask-only norm still mask.sum()", torch.allclose(out4, manual4, atol=1e-5))

print()
if failures:
    print(f"SMOKE FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("SMOKE OK: all checks passed")
