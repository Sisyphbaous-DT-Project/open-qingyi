"""v3 quality pass over SFT pairs (rule-based, no GPU).

Filters/dedups the pairs produced by clean_chat.py:
  1. exact + near duplicate targets (Qingyi repeats herself a lot)
  2. degenerate targets: single char / pure punctuation / repetition loops
  3. misfire pairs: context has no real user utterance (bot chains, system noise)
  4. overlong targets (will be truncated at SFT formatting; here we just flag-drop >1500 chars)
  5. context dedup: keep at most N pairs sharing the same final context utterance

Usage:
    python scripts/filter_pairs.py data/sft/*_pairs.jsonl --out data/sft/all_pairs_v3.jsonl
"""

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict

PUNCT_ONLY = re.compile(r"^[\s\W_]+$")
MAX_TARGET = 1500
MAX_SAME_CTX = 3


def norm(t):
    return re.sub(r"\s+", "", t)


def is_loop(t):
    # target that is one short fragment repeated 3+ times
    for n in (2, 3, 4):
        if len(t) >= n * 3 and len(t) % n == 0:
            seg = t[: len(t) // n]
            if seg * n == t:
                return True
    words = t.split("\n")
    if len(words) >= 4 and len(set(words)) == 1:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = []
    for p in args.inputs:
        paths.extend(glob.glob(p))
    pairs = []
    for path in sorted(set(paths)):
        for line in open(path, encoding="utf-8"):
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    stats = Counter()
    out = []
    seen_target = {}
    ctx_count = defaultdict(int)

    for p in pairs:
        tgt = p["target"].strip()
        ctx_texts = [c["text"] for c in p["context"] if c["role"] == "user"]
        if not tgt or PUNCT_ONLY.match(tgt) or len(norm(tgt)) < 2:
            stats["degenerate_target"] += 1
            continue
        if is_loop(norm(tgt)):
            stats["loop_target"] += 1
            continue
        if len(tgt) > MAX_TARGET:
            stats["overlong_target"] += 1
            continue
        if not ctx_texts:
            stats["no_user_context"] += 1
            continue
        key = norm(tgt)
        if key in seen_target:
            stats["dup_target"] += 1
            continue
        ctx_key = norm(ctx_texts[-1])[:80]
        if ctx_count[ctx_key] >= MAX_SAME_CTX:
            stats["dup_context"] += 1
            continue
        seen_target[key] = 1
        ctx_count[ctx_key] += 1
        out.append(p)
        stats["kept"] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    total = len(pairs)
    print(f"in: {total} pairs", file=sys.stderr)
    for k, v in stats.most_common():
        print(f"  {k}: {v} ({v/total:.1%})", file=sys.stderr)
    lens = sorted(len(p["target"]) for p in out)
    if lens:
        print(f"kept target len: p50={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
