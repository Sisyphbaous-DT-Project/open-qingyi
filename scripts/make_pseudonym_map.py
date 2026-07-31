"""Draft the pseudonym mapping table for v4 cleaning.

High-frequency speakers (top ~30, >=200 msgs) get human-style pseudonyms;
everyone else gets a deterministic 群友+hash label. The user (哥哥) maps
to 哥哥; 林清漪 stays. Output: data/sft/pseudonym_map_draft.json for review.
"""
import json
import collections
import hashlib

FAKE_NAMES = [
    "小葵", "阿澈", "老白", "柠檬", "星野", "团子", "洛洛", "阿福", "晚晴", "十一",
    "栗子", "南絮", "阿棠", "白鹿", "溪云", "桃酥", "阿眠", "青柠", "远舟", "小鹿",
    "知夏", "墨白", "杏子", "拾叁", "江眠", "沐沐", "阿屿", "千寻", "时雨", "柚子",
]

cnt = collections.Counter()
with open("data/sft/all_pairs_v3.jsonl") as f:
    for line in f:
        for c in json.loads(line)["context"]:
            n = c.get("name", "")
            if n:
                cnt[n] += 1

mapping = {}
i = 0
for name, c in cnt.most_common():
    if name in ("林清漪", "哥哥"):
        continue
    if i < 30 and c >= 200:
        mapping[name] = FAKE_NAMES[i]
        i += 1
    else:
        h = hashlib.md5(name.encode()).hexdigest()[:3]
        mapping[name] = f"群友{h}"

mapping["哥哥"] = "哥哥"
mapping["林清漪"] = "林清漪"

with open("data/sft/pseudonym_map_draft.json", "w") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=1)

print("=== 高频真人假名 ===")
for name, c in cnt.most_common():
    v = mapping.get(name, "")
    if v and not v.startswith("群友") and name != "林清漪":
        print(f"{c:6d}  {name[:28]:30s} -> {v}")

print()
print("低频 -> 群友+哈希:", sum(1 for v in mapping.values() if v.startswith("群友")), "人")
