#!/usr/bin/env python
"""尺子自检：新 held-out（build_held_out_v2 + 污染集过滤）的四项断言。

必须先于一切复测通过：
  A1 零重合（文档级）：held-out 文档指纹与"训练可及范围"（各源前
     CONTAM_K=100k 篇，>6x 安全边际）的污染集交集为空——第一次扫描时
     语料自身就有 24 篇重复文档，所以光靠 skip_docs 不够，必须过滤；
  A2 零重合（序列级）：训练流（make_train_iterator, seed=0）前 3000 条
     打包序列与全部 held-out 序列无一条相同；
  A3 分层与形状：每个源恰好 batches_per_source 个 [batch, seq_len] batch；
  A4 确定性：同一污染集两次构建的 held-out 指纹一致。
"""
import hashlib
import sys

sys.path.insert(0, "/root/projects/qingyi-kda")

from transformers import AutoTokenizer  # noqa: E402

from qingyi_kda.data import (  # noqa: E402
    SOURCES,
    build_contamination_fps,
    build_held_out_v2,
    make_train_iterator,
)

ROOT = "/root/projects/qingyi-kda"
SKIP = 50_000
CONTAM_K = 100_000
N_DOCS = 200
SEQ_LEN = 2048
BATCH = 4
BPS = 2


def seq_hash(t) -> str:
    return hashlib.sha1(t.numpy().tobytes()).hexdigest()


def main():
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-0.6B-Base")

    # --- contamination set: everything training could ever have seen ---
    print(f"scanning first {CONTAM_K} docs per source for contamination set...")
    contam = build_contamination_fps(n_docs=CONTAM_K)
    print(f"contamination set: {len(contam)} unique docs")

    # --- build the ruler (with duplicate/contamination filtering) ---
    batches_by_src, held_fps = build_held_out_v2(
        tok, SEQ_LEN, BATCH, n_docs_per_source=N_DOCS,
        batches_per_source=BPS, skip_docs=SKIP, exclude_fps=contam)

    # --- A1: doc-level zero overlap (by construction + explicit check) ---
    overlap = held_fps & contam
    assert not overlap, f"LEAK: {len(overlap)} held-out docs inside training range"
    print(f"A1 OK: 0/{len(held_fps)} held-out docs in first "
          f"{CONTAM_K} docs of any source")

    # --- A2: sequence-level zero overlap vs the actual training stream ---
    print("A2: hashing held-out seqs + first 3000 training seqs...")
    held_seqs = {
        seq_hash(row)
        for bl in batches_by_src.values() for b in bl for row in b
    }
    it = make_train_iterator(tok, SEQ_LEN, seed=0)
    train_seqs = {seq_hash(next(it)) for _ in range(3000)}
    both = held_seqs & train_seqs
    assert not both, f"LEAK: {len(both)} identical packed sequences"
    print(f"A2 OK: 0/{len(held_seqs)} held-out seqs in first 3000 train seqs")

    # --- A3: stratification + shapes ---
    for name, bl in batches_by_src.items():
        assert len(bl) == BPS, f"{name}: {len(bl)} batches != {BPS}"
        for b in bl:
            assert tuple(b.shape) == (BATCH, SEQ_LEN), f"{name}: bad shape {b.shape}"
    print(f"A3 OK: { {k: len(v) for k, v in batches_by_src.items()} }, "
          f"each [{BATCH}, {SEQ_LEN}]")

    # --- A4: determinism ---
    _, fps2 = build_held_out_v2(
        tok, SEQ_LEN, BATCH, n_docs_per_source=N_DOCS,
        batches_per_source=BPS, skip_docs=SKIP, exclude_fps=contam)
    assert fps2 == held_fps, "non-deterministic held-out construction"
    print("A4 OK: deterministic across rebuilds")

    print("\nALL ASSERTIONS PASSED -- ruler is clean.")


if __name__ == "__main__":
    main()
