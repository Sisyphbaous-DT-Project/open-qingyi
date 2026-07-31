#!/usr/bin/env python
"""DPO-v2 用 on-policy 采样（云端版）：让 maxsteps-300 对坑题生成候选。

池子：
  - identity：boost_pairs.jsonl source=identity 全部独特问题（~249 条，
    含拒密/身份/QOSP/DT-Project/GitHub 新三组），每条采 4 样（多抓失败）
  - chat：随机 300 条正常聊天（chosen=真实回复），每条采 2 样（保持应答意愿）
采样：temperature 0.9 / top_p 0.95，等长批次零 padding（KDA 不读 mask）。
输出: data/dpo/v2_samples.jsonl {"user_text","target","qa_answer","kind","samples"}
"""
import json
import random
import sys

sys.path.insert(0, "/root/autodl-tmp/qingyi-kda")

import torch
from transformers import AutoTokenizer
from qingyi_kda.surgery import load_hybrid
from qingyi_kda.cache import HybridKDACache

ROOT = "/root/autodl-tmp/qingyi-kda"
CKPT = f"{ROOT}/models/boost-v2/maxsteps-300"
OUT = f"{ROOT}/data/dpo/v2_samples.jsonl"
N_CHAT = 300
MAX_NEW = 60
BS = 8
random.seed(11)

tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-0.6B-Base")
tok.padding_side = "left"  # 生成必须左 pad；KDA 层不读 attention_mask
model = load_hybrid(CKPT, dtype=torch.bfloat16, device="cuda")
model.eval()

pool = []

# identity 全池（按问题文本去重：同题的不同群友名变体只留一条），chosen=标准答案
seen = set()
for line in open(f"{ROOT}/data/sft/boost_pairs.jsonl"):
    r = json.loads(line)
    if r.get("source") != "identity":
        continue
    q = r["context"][0]["text"]
    name = r["context"][0].get("name", "群友")
    if q in seen:
        continue
    seen.add(q)
    pool.append({"user_text": f"{name}: {q}", "target": None,
                 "qa_answer": r["target"], "kind": "identity", "n": 4})
n_identity = len(pool)

# chat 平衡池，chosen=真实回复
pairs = [json.loads(l) for l in open(f"{ROOT}/data/sft/boost_pairs.jsonl")
         if json.loads(l).get("source") == "chat"]
random.shuffle(pairs)
n_chat = 0
for r in pairs:
    if n_chat >= N_CHAT:
        break
    ctx = r["context"]
    while ctx and sum(len(c["text"]) + len(c.get("name", "")) + 2
                      for c in ctx) > 1500:
        ctx = ctx[1:]
    if not ctx or not r.get("target"):
        continue
    user_text = "\n".join(f"{c.get('name', '群友')}: {c['text']}" for c in ctx)
    pool.append({"user_text": user_text, "target": r["target"],
                 "qa_answer": None, "kind": "chat", "n": 2})
    n_chat += 1

print(f"prompt 池: {len(pool)} (identity {n_identity} / chat {n_chat})")


def chatml(user_text):
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


for b in pool:
    b["ids"] = tok(chatml(b["user_text"]), add_special_tokens=False).input_ids
pool.sort(key=lambda b: len(b["ids"]))

# 等长精确分组：同批等长 -> 零 padding -> 不传 attention_mask
batches = []
i = 0
while i < len(pool):
    j = i + 1
    while (j < len(pool) and len(pool[j]["ids"]) == len(pool[i]["ids"])
           and j - i < BS):
        j += 1
    batches.append(pool[i:j])
    i = j
print(f"等长批次: {len(batches)} (均批 {len(pool) / len(batches):.1f})")

with open(OUT, "w") as f:
    for bi, batch in enumerate(batches):
        n = max(b["n"] for b in batch)
        gens = []
        for _ in range(n):
            cache = HybridKDACache()
            ids = torch.tensor([b["ids"] for b in batch],
                               dtype=torch.long, device="cuda")
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=MAX_NEW, do_sample=True,
                    temperature=0.9, top_p=0.95,
                    past_key_values=cache,
                    pad_token_id=tok.eos_token_id)
            gens.append((out, ids.shape[1]))
        for j, b in enumerate(batch):
            samples = []
            for g, plen in gens[:b["n"]]:
                text = tok.decode(g[j, plen:], skip_special_tokens=False)
                samples.append(text.split("<|im_end|>")[0].strip())
            f.write(json.dumps({
                "user_text": b["user_text"], "target": b["target"],
                "qa_answer": b["qa_answer"], "kind": b["kind"],
                "samples": samples}, ensure_ascii=False) + "\n")
        if bi % 25 == 0:
            print(f"  batch {bi}/{len(batches)}", flush=True)

print(f"saved {OUT}")
