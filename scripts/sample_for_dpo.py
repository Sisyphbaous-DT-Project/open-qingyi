#!/usr/bin/env python
"""DPO 用 on-policy 采样：让当前模型对一批 prompt 生成候选回复。

prompt 池：
  - 1000 条真实聊天 context（all_pairs_v4 随机抽）
  - 222 条身份 QA 变体（boost 小灶同池）
  - persona_qa.json 47 条
每 prompt 采 2 条（temperature 0.9 / top_p 0.95），HybridKDACache 增量解码。
输出: data/dpo/samples.jsonl  {"context": [...], "target": 真实回复|None,
                               "qa_answer": 标准答案|None, "samples": [...]}
"""
import json
import random
import sys

sys.path.insert(0, "/root/projects/qingyi-kda")

import torch
from transformers import AutoTokenizer
from qingyi_kda.surgery import load_hybrid
from qingyi_kda.cache import HybridKDACache

ROOT = "/root/projects/qingyi-kda"
CKPT = f"{ROOT}/models/boost-checkpoints/best"
OUT = f"{ROOT}/data/dpo/samples.jsonl"
N_CHAT = 1000
SAMPLES_PER_PROMPT = 2
MAX_NEW = 60
BS = 8
random.seed(7)

tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-0.6B-Base")
tok.padding_side = "left"  # 生成必须左 pad；KDA 层不读 attention_mask，
                           # 右 pad 会把 pad token 扫进循环状态
model = load_hybrid(CKPT, dtype=torch.bfloat16, device="cuda")
model.eval()

# ---- prompt 池 ----
pool = []
pairs = [json.loads(l) for l in open(f"{ROOT}/data/sft/all_pairs_v4.jsonl")]
random.shuffle(pairs)
for r in pairs[:N_CHAT]:
    ctx = r["context"]
    while ctx and sum(len(c["text"]) + len(c.get("name", "")) + 2
                      for c in ctx) > 1500:
        ctx = ctx[1:]
    if not ctx:
        continue
    user_text = "\n".join(f"{c.get('name', '群友')}: {c['text']}" for c in ctx)
    pool.append({"user_text": user_text, "target": r["target"],
                 "qa_answer": None, "kind": "chat"})

boost = [json.loads(l) for l in open(f"{ROOT}/data/sft/boost_pairs.jsonl")
         if json.loads(l).get("source") == "identity"]
seen = set()
for r in boost:
    q = r["context"][0]["text"]
    name = r["context"][0].get("name", "群友")
    if q in seen:
        continue
    seen.add(q)
    pool.append({"user_text": f"{name}: {q}", "target": None,
                 "qa_answer": r["target"], "kind": "identity"})

for e in json.load(open(f"{ROOT}/data/persona/persona_qa.json")):
    pool.append({"user_text": f"哥哥: {e['q']}", "target": None,
                 "qa_answer": e["a"], "kind": "persona"})

# 按 token 长度精确分组：同批等长 -> 零 padding -> 不传 attention_mask。
# （左 pad + sdpa 在缓存解码时有 contiguous bug；且 KDA 不读 mask，
#  pad token 会污染循环状态，等长分组两个问题一起解决）
def chatml(user_text):
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


for b in pool:
    b["ids"] = tok(chatml(b["user_text"]),
                   add_special_tokens=False).input_ids
pool.sort(key=lambda b: len(b["ids"]))
print(f"prompt 池: {len(pool)} (chat {N_CHAT} / identity {len(seen)} / persona 47)")


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
        gens = []
        for _ in range(SAMPLES_PER_PROMPT):
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
            for g, plen in gens:
                text = tok.decode(g[j, plen:],
                                  skip_special_tokens=False)
                samples.append(text.split("<|im_end|>")[0].strip())
            f.write(json.dumps({
                "user_text": b["user_text"], "target": b["target"],
                "qa_answer": b["qa_answer"], "kind": b["kind"],
                "samples": samples}, ensure_ascii=False) + "\n")
        if bi % 25 == 0:
            print(f"  batch {bi}/{len(batches)}")

print(f"saved {OUT}")
