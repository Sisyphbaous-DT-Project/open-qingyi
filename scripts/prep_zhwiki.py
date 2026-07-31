#!/usr/bin/env python
# 下载中文维基百科 (wikimedia/wikipedia 20231101.zh)，清洗后写成 text 单列 parquet 分片
# 供 qingyi_kda/data.py 的 local_glob 管道直接使用
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq

OUT_DIR = os.environ.get("ZHWIKI_OUT", "/root/autodl-tmp/qingyi-kda/data/raw/zhwiki")
BATCH = 20000
MIN_CHARS = 200  # 跳过词条 stub/消歧页

os.makedirs(OUT_DIR, exist_ok=True)
print("loading wikimedia/wikipedia 20231101.zh ...", flush=True)
ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train")
print(f"loaded {len(ds)} rows, streaming to parquet shards", flush=True)

buf, shard, kept, skipped = [], 0, 0, 0
for row in ds:
    t = row.get("text")
    if not t or len(t) < MIN_CHARS:
        skipped += 1
        continue
    buf.append(t)
    kept += 1
    if len(buf) >= BATCH:
        pq.write_table(pa.table({"text": buf}), f"{OUT_DIR}/train-{shard:05d}.parquet")
        print(f"shard {shard} written ({BATCH} docs, kept={kept} skipped={skipped})", flush=True)
        shard += 1
        buf = []
if buf:
    pq.write_table(pa.table({"text": buf}), f"{OUT_DIR}/train-{shard:05d}.parquet")
    shard += 1
print(f"DONE shards={shard} kept={kept} skipped={skipped}", flush=True)
