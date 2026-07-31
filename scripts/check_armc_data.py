"""Read-only check: real arm-C final data through JSONLMixer. CPU only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from kd_e2e import JSONLMixer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "work", "stage3b-format-data", "final")
PATHS = [os.path.join(FINAL, f) for f in
         ("mcq_train.jsonl", "poetry_qa.jsonl", "translate_qa.jsonl")]

tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models", "Qwen3-0.6B-Base"))
mixer = JSONLMixer(tok, 2048, seed=0, paths=PATHS, max_rows=200000)
print(f"\ntotal samples: {len(mixer.samples)} (expect 31000)")
print(f"data_sha256: {mixer.data_sha256}")

# token length distribution per sample type
import random as _r
rng = _r.Random(0)
idx = rng.sample(range(len(mixer.samples)), 2000)
lens = []
overlong = 0
full_lens = []
for i, (p, c) in enumerate(mixer.samples):
    n = len(tok(p, add_special_tokens=False).input_ids) \
        + len(tok(c, add_special_tokens=False).input_ids) + 1
    full_lens.append(n)
    if n > 2048:
        overlong += 1
full_lens.sort()
n = len(full_lens)
print(f"len median={full_lens[n//2]} p95={full_lens[int(n*0.95)]} "
      f"p99={full_lens[int(n*0.99)]} max={full_lens[-1]}")
print(f"overlong (>2048, would be skipped): {overlong}")

# render one example per kind
seen = set()
for p, c in mixer.samples:
    kind = "mcq" if p.startswith(JSONLMixer.MCQ_PREFIX) else \
        ("poetry" if "续写" in p else "translate")
    if kind not in seen:
        seen.add(kind)
        print(f"\n--- {kind} sample ---")
        print(p[:400])
        print(f"[completion]{c}[/completion]")

# simulate packs: mask / pad fractions
tot_mask = tot_pad = tot = 0
for _ in range(20):
    ids, mask = mixer.next()
    tot += len(ids)
    tot_mask += float(mask.sum())
    # pad = trailing eos run beyond last mask==1
    last_one = max(j for j, v in enumerate(mask.tolist()) if v == 1.0)
    tot_pad += len(ids) - last_one - 2  # rough
print(f"\n20 packs: mask frac {tot_mask/tot:.4f} ({tot_mask:.0f} valid loss "
      f"tokens per {tot} packed tokens)")
print(f"approx valid loss tokens per optimizer step (micro2 x accum2 x 2048): "
      f"{tot_mask/20*4:.0f}")
print("\nDATA CHECK DONE")
