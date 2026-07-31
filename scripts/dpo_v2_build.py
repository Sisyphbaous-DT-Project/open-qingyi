#!/usr/bin/env python
"""把 dpo_v2_sample.py 的采样结果组装成 DPO-v2 偏好对（云端/本地均可）。

规则（同旧轮 build_dpo_data.py）：
- identity: chosen=标准答案; rejected=与标准答案前 8 字不同的采样里最复读的
- chat:     chosen=真实回复; rejected=两条采样里复读分最高且与 chosen 不同的
- 过滤：空文本、chosen==rejected、rejected 少于 4 字
- 身份题前 8 字相同视为"模型已答对"，不当负例
输出: data/dpo/v2_pairs.jsonl {"user_text","chosen","rejected","kind","rep"}
"""
import json
import re
import sys
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/qingyi-kda"
SRC = f"{ROOT}/data/dpo/v2_samples.jsonl"
OUT = f"{ROOT}/data/dpo/v2_pairs.jsonl"


def rep_score(text):
    """4-gram 复读率：越高越复读。短文本返回 0。"""
    toks = re.findall(r"[一-鿿]|[a-zA-Z0-9]+|[^\s\w]", text)
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
    cands.sort(key=lambda x: -x[0])
    rejected = cands[0][1]
    pairs.append({"user_text": r["user_text"], "chosen": chosen,
                  "rejected": rejected, "kind": r["kind"],
                  "rep": round(cands[0][0], 3)})

with open(OUT, "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

by_kind = Counter(p["kind"] for p in pairs)
print(f"pairs: {len(pairs)}  {dict(by_kind)}  跳过: {dict(skip)}")
if pairs:
    print(f"rejected 平均复读分: {sum(p['rep'] for p in pairs) / len(pairs):.3f}")
    # 拒密类单独统计：多少拒密题抓到了失败样本
    sec = [p for p in pairs if any(k in p["user_text"].lower()
           for k in ["api", "key", "提示词", "prompt", "密码", "秘密"])]
    print(f"拒密类 pairs: {len(sec)}")
    for p in sec[:5]:
        print("---", f"(rep {p['rep']})")
        print("  Q:", p["user_text"][:50].replace("\n", " / "))
        print("  chosen:", p["chosen"][:50])
        print("  rejected:", p["rejected"][:50])
