#!/usr/bin/env python
"""验证复核指控：held-out 号称 600 篇，实际进入评分窗口的文档数。

评分窗口 = batches_per_source(2) x batch(4) x seq_len(2048) = 16384 token/源。
统计每个源 kept 200 篇里有多少篇的开头落在前 16384 token 之内。
"""
import hashlib
import sys

sys.path.insert(0, "/root/projects/qingyi-kda")

from transformers import AutoTokenizer  # noqa: E402

from qingyi_kda.data import SOURCES, _open_stream, build_contamination_fps  # noqa: E402

WINDOW = 2 * 4 * 2048

tok = AutoTokenizer.from_pretrained(
    "/root/projects/qingyi-kda/models/Qwen3-0.6B-Base")
contam = build_contamination_fps(100_000)
for name, spec in SOURCES.items():
    stream, field = _open_stream(spec)
    kept = in_window = toks = 0
    for i, row in enumerate(stream):
        if i < 50_000:
            continue
        if kept >= 200:
            break
        text = row.get(field)
        if not text:
            continue
        fp = hashlib.sha1(text[:512].encode("utf-8", "ignore")).hexdigest()
        if fp in contam:
            continue
        kept += 1
        if toks < WINDOW:
            in_window += 1
        toks += len(tok(text, add_special_tokens=False).input_ids) + 1
    print(f"{name}: kept={kept} in_score_window={in_window} "
          f"packed_tokens={toks} (window {WINDOW})")
