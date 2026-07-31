"""v4 cleaning: pseudonym mapping + 哥哥 replacement + truncation.

Input : data/sft/all_pairs_v3.jsonl  (context/target pairs, credentials already
        stripped by v2 SECRET_RE)
Map   : data/sft/pseudonym_map.json  (approved pseudonym pool)
Output: data/sft/all_pairs_v4.jsonl

Rules (user decisions 2026-07-28):
- User variants (name field + text bodies): user nick variants -> 哥哥
- QQ 123456789 stays (identity binding, intentionally public)
- 林清漪 stays
- Top-30 pseudonyms applied to name fields; text-body replacement only for
  distinctive names (common-word nicknames like 哈哈/值得/安 are skipped so we
  don't corrupt ordinary text)
- No sensitive-word list (user: 让清漪完完整整的)
- Truncate over-long targets to ~1024 tokens
"""
import json
import re

MAP_PATH = "data/sft/pseudonym_map.json"
IN_PATH = "data/sft/all_pairs_v3.jsonl"
OUT_PATH = "data/sft/all_pairs_v4.jsonl"
MAX_TARGET_CHARS = 2048  # ~1024 tokens for CJK-heavy text, avoids loading tokenizer

# Nicknames that are also ordinary words -- never replaced inside text bodies.
AMBIGUOUS = {"安", "缪", "墨", "汐", "哈哈", "值得", "无名", "冰糖", "柚子",
             "柠檬", "团子", "小鹿", "杏子", "十一", "拾叁", "溪云", "桃酥"}

with open(MAP_PATH) as f:
    pmap = json.load(f)

# Text-body replacement table: distinctive original names only.
# (user nick variants are already normalized to 哥哥 in the released data)
text_repl = {}
for orig, fake in pmap.items():
    if orig in ("哥哥", "林清漪") or fake.startswith("群友"):
        continue
    if orig in AMBIGUOUS or len(orig) < 2:
        continue
    text_repl[orig] = fake

pat = re.compile("|".join(re.escape(k) for k in sorted(text_repl, key=len, reverse=True)))

# Textual @mentions of real nicknames not covered by the speaker map (they
# rarely speak, so v2's [@QQ]->群友 hashing never caught them). Anonymize with
# the same md5 rule as the low-frequency pool; Qingyi's bot account normalizes
# to her real name.
import hashlib
for nick in ("钢板日穿", "强碱范", "变态男酮"):
    text_repl[nick] = "群友" + hashlib.md5(nick.encode()).hexdigest()[:3]
text_repl["Thinkmore_林清漪"] = "林清漪"
pat = re.compile("|".join(re.escape(k) for k in sorted(text_repl, key=len, reverse=True)))

stats = {"pairs": 0, "name_mapped": 0, "text_repl": 0, "truncated": 0}

def clean_text(t):
    t, n = pat.subn(lambda m: text_repl[m.group(0)], t)
    stats["text_repl"] += n
    return t

with open(IN_PATH) as fin, open(OUT_PATH, "w") as fout:
    for line in fin:
        r = json.loads(line)
        for c in r["context"]:
            n = c.get("name", "")
            if n in pmap:
                c["name"] = pmap[n]
                stats["name_mapped"] += 1
            c["text"] = clean_text(c["text"])
        t = clean_text(r["target"])
        if len(t) > MAX_TARGET_CHARS:
            t = t[:MAX_TARGET_CHARS]
            stats["truncated"] += 1
        r["target"] = t
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["pairs"] += 1

print(json.dumps(stats, ensure_ascii=False, indent=1))
