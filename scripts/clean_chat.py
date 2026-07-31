"""Clean exported QQ chat logs into SFT pairs (context -> Qingyi reply).

Pipeline per ROADMAP.md data tiers:
  1. load exported jsonl (simplified NapCat messages)
  2. drop bot commands / placeholder-only / empty messages
  3. merge consecutive same-sender messages within MERGE_WINDOW seconds
     (QQ bots split one reply into many short messages)
  4. for each Qingyi utterance, emit {context: [...utterances], target: reply}

Usage:
    python scripts/clean_chat.py data/chat/private_example.jsonl --out data/sft/private_pairs.jsonl
"""

import argparse
import json
import re
import sys

QINGYI_ID = 0000000000
MERGE_WINDOW = 120  # seconds; bot fragments arrive within seconds of each other
MAX_CONTEXT = 8     # utterances of context kept per pair

CMD_RE = re.compile(r"^\s*[/#]")          # bot commands like /reset /new
PLACEHOLDER_RE = re.compile(r"^\s*(\[[a-zA-Z一-鿿 ]+\]\s*)+$")  # [image] [face] only
BOT_NOISE = ("[收到红包", "[CQ:")
ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")  # bot inserts zero-width chars
# credentials / secrets that must never reach training data (号池群重点)
SECRET_RE = re.compile(
    r"(密码|passwd|password|卡密|cdkey|token|api[-_]?key|secret)"
    r"|[A-Za-z0-9_\-]{24,}"  # long credential-looking strings
    r"|sk-[A-Za-z0-9]{10,}", re.I)
AT_QQ_RE = re.compile(r"\[@(\d+)\]")


def _anon_qq(m):
    # deterministic pseudonym: same QQ -> same short tag, no real number kept
    return "[@群友%s]" % (format(int(m.group(1)) % 1000, "03d"))


def clean_text(text):
    text = ZERO_WIDTH_RE.sub("", text)
    text = AT_QQ_RE.sub(_anon_qq, text)
    return text.strip()


def load_msgs(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not m.get("text"):
                continue
            m["text"] = clean_text(m["text"])
            if not m["text"]:
                continue
            rows.append(m)
    rows.sort(key=lambda m: (m["time"], m.get("msg_seq") or 0))
    return rows


def keep(text):
    if CMD_RE.match(text):
        return False
    if PLACEHOLDER_RE.match(text):
        return False
    if any(n in text for n in BOT_NOISE):
        return False
    if SECRET_RE.search(text):
        return False
    return True


def merge_utterances(msgs):
    """Merge consecutive same-sender messages into utterances."""
    utts = []
    for m in msgs:
        if (
            utts
            and m["user_id"] == utts[-1]["user_id"]
            and m["time"] - utts[-1]["end_time"] <= MERGE_WINDOW
        ):
            utts[-1]["text"] += "\n" + m["text"]
            utts[-1]["end_time"] = m["time"]
            utts[-1]["n"] += 1
        else:
            utts.append({
                "user_id": m["user_id"],
                "nickname": m.get("nickname") or "",
                "text": m["text"],
                "time": m["time"],
                "end_time": m["time"],
                "n": 1,
            })
    return utts


def extract_pairs(utts):
    pairs = []
    for i, u in enumerate(utts):
        if u["user_id"] != QINGYI_ID:
            continue
        ctx = []
        for c in utts[max(0, i - MAX_CONTEXT):i]:
            role = "qingyi" if c["user_id"] == QINGYI_ID else "user"
            ctx.append({"role": role, "name": c["nickname"], "text": c["text"]})
        if not ctx:
            continue
        pairs.append({"context": ctx, "target": u["text"],
                      "time": u["time"], "merged_from": u["n"]})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_pairs = []
    for path in args.inputs:
        msgs = load_msgs(path)
        kept = [m for m in msgs if keep(m["text"])]
        utts = merge_utterances(kept)
        pairs = extract_pairs(utts)
        qy_msgs = sum(1 for m in kept if m["user_id"] == QINGYI_ID)
        print(f"{path}: {len(msgs)} msgs -> {len(kept)} kept ({qy_msgs} qingyi) "
              f"-> {len(utts)} utterances -> {len(pairs)} pairs", file=sys.stderr)
        all_pairs.extend(pairs)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_pairs)} pairs -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
