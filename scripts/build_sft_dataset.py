#!/usr/bin/env python
"""Build the tokenized SFT dataset.

Mix (ROADMAP 2026-07-28):
- personal pairs (data/sft/all_pairs_v4.jsonl, 14,605) -- the personality core
- persona QA (data/persona/persona_qa.json) x REPEAT_PQA -- identity facts
- general anchor: smoltalk2 smol-magpie-ultra (12k) + COIG-CQIA coig_pc (6k)
  -> personal share ~45% (target 40-60%)

Format: Qwen3 chat markup, NO system prompt (persona is baked into weights):
  <|im_start|>user\n{ctx lines "name: text"}<|im_end|>\n
  <|im_start|>assistant\n{target}<|im_end|>\n
Labels: -100 everywhere except assistant tokens + trailing <|im_end|>.
Output: data/sft/sft_dataset.pt  (list of {"input_ids", "labels"}, len<=1024)

General sets are fetched once (streaming, take N) and cached as jsonl under
data/sft/general_cache/ so rebuilds are offline and deterministic.
"""
import json
import os
import sys

sys.path.insert(0, "/root/projects/qingyi-kda")

import torch
from transformers import AutoTokenizer

ROOT = "/root/projects/qingyi-kda"
PAIRS_V4 = f"{ROOT}/data/sft/all_pairs_v4.jsonl"
PERSONA_QA = f"{ROOT}/data/persona/persona_qa.json"
CACHE_DIR = f"{ROOT}/data/sft/general_cache"
OUT = f"{ROOT}/data/sft/sft_dataset.pt"
MAX_LEN = 1024
MAX_CTX_CHARS = 1500  # trim oldest group-context lines first
REPEAT_PQA = 8
N_SMOL = 12000
N_COIG = 6000

tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-0.6B-Base")


def encode(user_text, answer_text):
    """-> (input_ids, labels) or None if over MAX_LEN."""
    head = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    tail = f"{answer_text}<|im_end|>\n"
    h = tok(head, add_special_tokens=False).input_ids
    t = tok(tail, add_special_tokens=False).input_ids
    ids = h + t
    if len(ids) > MAX_LEN or not t:
        return None
    labels = [-100] * len(h) + t
    return {"input_ids": ids, "labels": labels}


def encode_multiturn(messages):
    """messages: [{'role':..., 'content':...}] -> labeled example or None.

    The <|im_start|>role header is ALWAYS masked; only assistant content +
    trailing <|im_end|> carry loss (bug fix 2026-07-28: the header was
    previously labeled, teaching the model to emit <|im_start|> tokens).
    """
    ids, labels = [], []
    for m in messages:
        h = tok(f"<|im_start|>{m['role']}\n",
                add_special_tokens=False).input_ids
        b = tok(f"{m['content']}<|im_end|>\n",
                add_special_tokens=False).input_ids
        ids += h + b
        labels += ([-100] * len(h) + b if m["role"] == "assistant"
                   else [-100] * (len(h) + len(b)))
    if len(ids) > MAX_LEN or not any(x != -100 for x in labels):
        return None
    return {"input_ids": ids, "labels": labels}


# ---- 1. personal pairs ----
personal, skipped = [], 0
with open(PAIRS_V4) as f:
    for line in f:
        r = json.loads(line)
        ctx = r["context"]
        # trim oldest context messages by char budget
        while ctx and sum(len(c["text"]) + len(c.get("name", "")) + 2
                          for c in ctx) > MAX_CTX_CHARS:
            ctx = ctx[1:]
        if not ctx:
            skipped += 1
            continue
        user_text = "\n".join(f"{c.get('name', '群友')}: {c['text']}" for c in ctx)
        ex = encode(user_text, r["target"])
        if ex is None:
            skipped += 1
            continue
        personal.append(ex)
print(f"personal pairs: {len(personal)} (skipped {skipped})")

# ---- 2. persona QA (repeated) ----
pqa = []
with open(PERSONA_QA) as f:
    for r in json.load(f):
        ex = encode(r["q"], r["a"])
        if ex:
            pqa.append(ex)
pqa = pqa * REPEAT_PQA
print(f"persona QA: {len(pqa)} (x{REPEAT_PQA})")

# ---- 3. general anchor (cached fetch) ----
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_or_fetch(name, n, loader):
    path = os.path.join(CACHE_DIR, f"{name}.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            return [json.loads(l) for l in f]
    rows = loader(n)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows

def load_smol(n):
    # streaming over the proxy is flaky (SSL EOF) -- download one shard
    # (~66k rows) and read locally; n = raw pool size (keep target is smaller)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        "HuggingFaceTB/smoltalk2",
        "SFT/smoltalk_smollm3_smol_magpie_ultra_no_think-00000-of-00006.parquet",
        repo_type="dataset")
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
        for msgs in batch.column("messages").to_pylist():
            rows.append({"messages": [
                {"role": m["role"], "content": m["content"]} for m in msgs]})
            if len(rows) >= n:
                return rows
    return rows

def load_coig(n):
    # coig_pc core sample + xhs (小红书 casual social-media Chinese) until n
    import json as _json
    from huggingface_hub import hf_hub_download
    rows = []
    for fn in ("coig_pc/coig_pc_core_sample.jsonl", "xhs/xhs.jsonl"):
        path = hf_hub_download("m-a-p/COIG-CQIA", fn, repo_type="dataset")
        with open(path) as f:
            for line in f:
                r = _json.loads(line)
                q = r.get("instruction", "")
                if r.get("input"):
                    q = q + "\n" + r["input"]
                a = r.get("output", "")
                if not q or not a:
                    continue
                rows.append({"messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]})
                if len(rows) >= n:
                    return rows
    return rows

def encode_fit(messages):
    """Drop oldest MIDDLE turns (keep first user msg + tail) until <=MAX_LEN."""
    msgs = [m for m in messages
            if m.get("role") in ("user", "assistant") and m.get("content")]
    while len(msgs) >= 2:
        if sum(len(m["content"]) for m in msgs) > 3500 and len(msgs) > 2:
            msgs.pop(1)
            continue
        ex = encode_multiturn(msgs)
        if ex is not None:
            return ex
        if len(msgs) <= 2:
            return None
        msgs.pop(1)
    return None

general, gskip = [], 0
for name, target, loader in [("smoltalk2_magpie", N_SMOL, load_smol),
                             ("coig_cqia_pc", N_COIG, load_coig)]:
    rows = cache_or_fetch(name, target * 4, loader)  # raw pool, trim to target
    ok = 0
    for r in rows:
        if ok >= target:
            break
        ex = encode_fit(r["messages"])
        if ex is None:
            gskip += 1
            continue
        general.append(ex)
        ok += 1
    print(f"general[{name}]: {ok} kept")

dataset = personal + pqa + general
total = len(dataset)
share = (len(personal) + len(pqa)) / total
print(f"total {total} | personal share {share:.1%} | general skipped {gskip}")
torch.save(dataset, OUT)
print(f"saved {OUT}")
