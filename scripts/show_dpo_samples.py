#!/usr/bin/env python
"""从 samples.jsonl 挑有趣的采样展示。"""
import json
import re
from collections import Counter


def rep(t):
    toks = re.findall(r"[一-鿿]|[a-zA-Z0-9]+|[^\s\w]", t)
    if len(toks) < 12:
        return 0.0
    g = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    c = Counter(g)
    return sum(n for n in c.values() if n > 1) / len(g)


rows = [json.loads(l) for l in open("/root/projects/qingyi-kda/data/dpo/samples.jsonl")]
flat = []
for r in rows:
    for s in r["samples"]:
        flat.append((rep(s), r, s))
flat.sort(key=lambda x: -x[0])

print("=== 复读王 TOP3 ===")
for sc, r, s in flat[:3]:
    print(f"[rep {sc:.2f}][{r['kind']}] Q: {r['user_text'][:50]}")
    print(f"  A: {s[:100]}")

print()
print("=== 聊天类：真实回复 vs 采样（2 条）===")
chat = [x for x in flat if x[1]["kind"] == "chat"]
for sc, r, s in (chat[100], chat[400]):
    print(f"[rep {sc:.2f}] Q: {r['user_text'][:60]}")
    print(f"  真实: {(r['target'] or '')[:60]}")
    print(f"  采样: {s[:80]}")

print()
print("=== 身份类：答得好的 2 条 ===")
good = [x for x in flat if x[1]["kind"] in ("identity", "persona")
        and r and (x[1]["qa_answer"] or "")[:6] == x[2].replace(" ", "")[:6]]
for sc, r, s in good[:2]:
    print(f"[{r['kind']}] Q: {r['user_text'][:50]}")
    print(f"  标准: {r['qa_answer'][:60]}")
    print(f"  采样: {s[:80]}")

print()
print(f"统计: 共 {len(rows)} prompt / {len(flat)} 采样, "
      f"复读分>0.3 的 {sum(1 for x in flat if x[0] > 0.3)} 条 "
      f"({sum(1 for x in flat if x[0] > 0.3) / len(flat):.0%})")
