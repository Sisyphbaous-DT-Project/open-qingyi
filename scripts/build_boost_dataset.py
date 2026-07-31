#!/usr/bin/env python
"""Build the tokenized identity-boost dataset from boost_pairs.jsonl.

Same ChatML encoding as build_sft_dataset.py (no system prompt, labels only
on assistant content + <|im_end|>), but sourced from the boost mix:
identity QA x20 + 4k chat pairs (build_identity_boost.py output).
Output: data/sft/boost_dataset.pt
"""
import json
import sys

sys.path.insert(0, "/root/projects/qingyi-kda")

import torch
from transformers import AutoTokenizer

ROOT = "/root/projects/qingyi-kda"
PAIRS = f"{ROOT}/data/sft/boost_pairs.jsonl"
OUT = f"{ROOT}/data/sft/boost_dataset.pt"
MAX_LEN = 1024
MAX_CTX_CHARS = 1500

tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-0.6B-Base")


def encode(user_text, answer_text):
    head = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    tail = f"{answer_text}<|im_end|>\n"
    h = tok(head, add_special_tokens=False).input_ids
    t = tok(tail, add_special_tokens=False).input_ids
    ids = h + t
    if len(ids) > MAX_LEN or not t:
        return None
    labels = [-100] * len(h) + t
    return {"input_ids": ids, "labels": labels}


dataset, skipped, n_id, n_chat = [], 0, 0, 0
with open(PAIRS) as f:
    for line in f:
        r = json.loads(line)
        ctx = r["context"]
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
        dataset.append(ex)
        if r.get("source") == "identity":
            n_id += 1
        else:
            n_chat += 1

print(f"boost dataset: {len(dataset)} (identity {n_id} / chat {n_chat}, "
      f"skipped {skipped})")
torch.save(dataset, OUT)
print(f"saved {OUT}")
