"""Arm-C preflight: JSONLMixer alignment + ChunkedKL masked math (CPU, no GPU).

Run: python3 scripts/test_jsonl_smoke.py
Validates the four arm-C edits to kd_e2e.py before any cloud run:
  1. JSONLMixer renders mcq/QA rows, mask lands exactly on completion spans
  2. state/load_state restores the stream token-for-token
  3. ChunkedKL masked forward == manual KL over masked positions only
  4. ChunkedKL masked backward: zero grad at mask=0, correct scale at mask=1
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from kd_e2e import ChunkedKL, JSONLMixer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIR = os.path.join(ROOT, "models", "Qwen3-0.6B-Base")
TOY = os.path.join(ROOT, "data", "toy_armc.jsonl")

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---- build toy JSONL: 10 mcq + 10 generic QA -------------------------------
rows = []
for i in range(10):
    rows.append({
        "type": "mcq",
        "question": f"第{i}题：下面哪个选项是正确的测试答案？",
        "options": [f"选项甲{i}", f"选项乙{i}", f"选项丙{i}", f"选项丁{i}"],
        "answer_label": "ABCD"[i % 4],
    })
for i in range(10):
    rows.append({
        "type": "qa",
        "prompt": f"问题：{i}加{i}等于多少？\n回答：",
        "completion": f" {2 * i}",
    })
with open(TOY, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

tok = AutoTokenizer.from_pretrained(TOK_DIR)
SEQ = 256

# ---- 1. mixer basics -------------------------------------------------------
mixer = JSONLMixer(tok, SEQ, seed=0, path=TOY, max_rows=200000)
check("sample count", len(mixer.samples) == 20, f"got {len(mixer.samples)}")

# render check: mcq row format
p, c = mixer.samples[0] if mixer.samples[0][0].startswith(JSONLMixer.MCQ_PREFIX) else mixer.samples[1]
mcq_render = [s for s in mixer.samples if s[0].startswith(JSONLMixer.MCQ_PREFIX)]
qa_render = [s for s in mixer.samples if not s[0].startswith(JSONLMixer.MCQ_PREFIX)]
check("mcq render count", len(mcq_render) == 10, f"got {len(mcq_render)}")
check("qa render count", len(qa_render) == 10, f"got {len(qa_render)}")
mp, mc = mcq_render[0]
check("mcq prompt shape",
      "答案：" in mp and "A. " in mp and "D. " in mp,
      mp.splitlines()[-1])
check("mcq completion is label", mc.strip() in list("ABCD"), repr(mc))

# ---- 2. mask alignment over packed sequences --------------------------------
mixer2 = JSONLMixer(tok, SEQ, seed=0, path=TOY, max_rows=200000)
ids, mask = mixer2.next()
check("next() shapes", ids.shape == (SEQ,) and mask.shape == (SEQ,),
      f"{ids.shape} {mask.shape}")
check("mask binary", set(mask.unique().tolist()) <= {0.0, 1.0},
      str(mask.unique().tolist()))
frac = float(mask.mean())
check("mask fraction small", 0.001 < frac < 0.5, f"{frac:.4f}")

# For every masked position j, token ids[j+1] must belong to a completion
# span. Verify by re-tokenizing each sample and matching spans inside ids.
spans = []
for prompt, completion in mixer2.samples:
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    c_ids = tok(completion, add_special_tokens=False).input_ids + [mixer2.eos]
    spans.append((p_ids, c_ids))

# reconstruct: walk the buffer from a fresh mixer, sample by sample
mixer3 = JSONLMixer(tok, SEQ, seed=0, path=TOY, max_rows=200000)
ok_align = True
detail = ""
for _ in range(40):  # consume 40 samples worth
    mixer3._refill()
    # find the sample boundaries by scanning buf for eos
    buf = mixer3.buf_ids
    # take the first sample: its tokens end at first eos
    try:
        end = buf.index(mixer3.eos) + 1
    except ValueError:
        ok_align = False
        detail = "no eos in buffer"
        break
    sample_ids = buf[:end]
    sample_mask = mixer3.buf_mask[:end]
    # match against known spans
    hit = None
    for p_ids, c_ids in spans:
        if p_ids + c_ids == sample_ids:
            hit = (p_ids, c_ids)
            break
    if hit is None:
        ok_align = False
        detail = f"unmatched sample len={end}"
        break
    p_ids, c_ids = hit
    expect = [0.0] * end
    for j in range(len(p_ids) - 1, end - 1):
        expect[j] = 1.0
    if sample_mask != expect:
        ok_align = False
        bad = [k for k in range(end) if sample_mask[k] != expect[k]][:5]
        detail = f"mask mismatch at {bad}"
        break
    # masked positions predict completion tokens: ids[j+1] in c_ids
    pred_tokens = [sample_ids[j + 1] for j in range(len(p_ids) - 1, end - 1)]
    if pred_tokens != c_ids:
        ok_align = False
        detail = "predicted tokens != completion ids"
        break
    mixer3.buf_ids = mixer3.buf_ids[end:]
    mixer3.buf_mask = mixer3.buf_mask[end:]
check("mask lands on completion span", ok_align, detail)

# decode check: completion text recoverable from masked targets
ids4, mask4 = mixer2.next() if False else (None, None)
mixer4 = JSONLMixer(tok, SEQ, seed=0, path=TOY, max_rows=200000)
ids4, mask4 = mixer4.next()
masked_targets = ids4[1:][mask4[:-1] == 1.0]
txt = tok.decode(masked_targets.tolist())
check("masked targets decode to completions", ("答案" not in txt) and txt.strip() != "",
      repr(txt[:60]))

# ---- 3. state / load_state token-for-token ----------------------------------
ma = JSONLMixer(tok, SEQ, seed=0, path=TOY, max_rows=200000)
for _ in range(3):
    ma.next()
st = ma.state()
ahead_a = [ma.next() for _ in range(5)]
mb = JSONLMixer(tok, SEQ, seed=0, path=TOY, max_rows=200000)
mb.load_state(st)
ahead_b = [mb.next() for _ in range(5)]
same = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
           for a, b in zip(ahead_a, ahead_b))
check("state/load_state token-for-token", same)

# wrap-around determinism
mc1 = JSONLMixer(tok, 64, seed=0, path=TOY, max_rows=200000)
seq_a = [mc1.next()[0] for _ in range(12)]
mc2 = JSONLMixer(tok, 64, seed=0, path=TOY, max_rows=200000)
seq_b = [mc2.next()[0] for _ in range(12)]
check("wrap deterministic", all(torch.equal(a, b) for a, b in zip(seq_a, seq_b)))

# ---- 4. ChunkedKL masked math ------------------------------------------------
torch.manual_seed(0)
N, H, V = 37, 16, 53
hs = torch.randn(N, H, dtype=torch.float32, requires_grad=True)
ht = torch.randn(N, H, dtype=torch.float32)
Ws = torch.randn(V, H, dtype=torch.float32)
Wt = torch.randn(V, H, dtype=torch.float32)
T = 2.0
mask = torch.zeros(N)
idx = random.Random(0).sample(range(N), 11)
mask[idx] = 1.0

# manual forward on masked positions only
ls = (hs[idx] @ Ws.T) / T
lt = (ht[idx] @ Wt.T) / T
log_ps = F.log_softmax(ls, -1)
log_pt = F.log_softmax(lt, -1)
manual = ((log_pt.exp() * (log_pt - log_ps)).sum(-1)).sum() * T * T / mask.sum()

out = ChunkedKL.apply(hs, ht, Ws, Wt, T, 8, mask)
check("masked forward == manual", torch.allclose(out, manual, atol=1e-5),
      f"{float(out):.6f} vs {float(manual):.6f}")

# backward: grad zero at mask=0 positions
out.backward()
g = hs.grad
check("grad zero at mask=0", float(g[mask == 0].abs().max()) == 0.0)

# grad at mask=1 positions == unmasked formula / mask.sum()
hs2 = hs.detach().clone().requires_grad_(True)
ls2 = (hs2[idx] @ Ws.T) / T
lt2 = (ht[idx] @ Wt.T) / T
ps2 = F.softmax(ls2, -1)
pt2 = F.softmax(lt2, -1)
dz = (ps2 - pt2) * T
expect_g = (dz @ Ws) / mask.sum()
got_g = g[idx]
check("masked backward scale", torch.allclose(got_g, expect_g, atol=1e-5),
      f"max diff {float((got_g - expect_g).abs().max()):.2e}")

# no-mask path unchanged (norm = n)
out_nomask = ChunkedKL.apply(hs.detach(), ht, Ws, Wt, T, 8, None)
ls3 = (hs.detach() @ Ws.T) / T
lt3 = (ht @ Wt.T) / T
manual_nomask = ((F.log_softmax(lt3, -1).exp()
                  * (F.log_softmax(lt3, -1) - F.log_softmax(ls3, -1))).sum(-1)
                 .sum() * T * T / N)
check("unmasked forward unchanged", torch.allclose(out_nomask, manual_nomask, atol=1e-5),
      f"{float(out_nomask):.6f} vs {float(manual_nomask):.6f}")

# all-masked degenerate: norm clamps to >=1, no NaN
mask0 = torch.zeros(N)
out0 = ChunkedKL.apply(hs.detach(), ht, Ws, Wt, T, 8, mask0)
check("all-zero mask finite", torch.isfinite(out0).item() and float(out0) == 0.0,
      f"{float(out0)}")

print()
if failures:
    print(f"SMOKE FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("SMOKE OK: all checks passed")
