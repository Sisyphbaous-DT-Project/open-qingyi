#!/usr/bin/env python
"""把 sample_for_dpo.py 的采样结果组装成 DPO 偏好对。

规则：
- chat:     chosen=真实回复 target; rejected=两条采样里更差的（复读分高者优先）
- identity/persona: chosen=标准答案; rejected=与标准答案不同的采样
- 过滤：空文本、chosen==rejected、rejected 与 chosen 前 8 字相同（说明模型
  已经答对，不是合格的负例）、rejected 少于 4 字
输出: data/dpo/dpo_pairs.jsonl  {"user_text","chosen","rejected","kind"}
"""
import json
import re
import sys
from collections import Counter

ROOT = "/root/projects/qingyi-kda"
SRC = f"{ROOT}/data/dpo/samples.jsonl"
OUT = f"{ROOT}/data/dpo/dpo_pairs.jsonl"


def rep_score(text):
    """4-gram 复读率：越高越复读。短文本返回 0。"""
    toks = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+|[^\s\w]", text)
    if len(toks) < 12:
        return 0.0
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    c = Counter(grams)
    repeated = sum(n for n in c.values() if n > 1)
    return repeated / len(grams)


def norm(t):
    return re.sub(r"\s+", "", t or "")


pairs, skip = [], Counter()
for line in open(SRC):
    r = json.loads(line)
    chosen = r["target"] if r["kind"] == "chat" else r["qa_answer"]
    if not chosen or not norm(chosen):
        skip["no_chosen"] += 1
        continue
    cands = []
    for s in r["samples"]:
        ns = norm(s)
        if len(ns) < 4:
            continue
        if norm(chosen)[:8] == ns[:8]:   # 模型已答对，不是负例
            continue
        cands.append((rep_score(s), s))
    if not cands:
        skip["no_bad_sample"] += 1
        continue
    cands.sort(key=lambda x: -x[0])      # 复读最厉害的当 rejected
    rejected = cands[0][1]
    pairs.append({"user_text": r["user_text"], "chosen": chosen,
                  "rejected": rejected, "kind": r["kind"],
                  "rep": round(cands[0][0], 3)})

with open(OUT, "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

by_kind = Counter(p["kind"] for p in pairs)
print(f"pairs: {len(pairs)}  {dict(by_kind)}  跳过: {dict(skip)}")
print(f"rejected 平均复读分: {sum(p['rep'] for p in pairs) / len(pairs):.3f}")
for p in pairs[:3] + pairs[-2:]:
    print("---", p["kind"], f"(rep {p['rep']})")
    print("  Q:", p["user_text"][:60].replace("\n", " / "))
    print("  chosen:", p["chosen"][:60])
    print("  rejected:", p["rejected"][:60])
