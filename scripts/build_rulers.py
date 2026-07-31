#!/usr/bin/env python
"""Build the v3 ruler trio with a 200k contamination set (one-shot script).

Reviewer round-2 findings addressed here:
- 100k contamination margin was only ~3.0x (1.26x at 3000 steps) -> 200k.
- tune/accept caches shared 2 document fingerprints (in-corpus duplicates
  deduped only within each build) -> all splits are built in ONE process and
  each split's fingerprints are added to the exclusion set of the next, so
  cross-split overlap is zero by construction, duplicates included.
- Three splits, not two: tune (docs 0-199, dev), valid (docs 200-799,
  checkpoint selection), release (docs 800-1399, LOCKED, final go/no-go only).
- Every cache carries a manifest (params, fp counts, zero-overlap assertion)
  so a cache is auditable without the building code.

Output: data/ruler_{tune,valid,release}.pt  (supersedes heldout_v2*.pt)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer

from qingyi_kda.data import build_contamination_fps, build_held_out_v2

CONTAM_N = 200_000
SPLITS = {
    # name       doc_offset  n_docs  batches/src
    "tune":    dict(doc_offset=0,   n_docs=200, batches=2),
    "valid":   dict(doc_offset=200, n_docs=600, batches=8),
    "release": dict(doc_offset=800, n_docs=600, batches=8),
}


def main():
    tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B-Base")
    print(f"[rulers] building contamination fingerprints ({CONTAM_N} docs/source)...")
    contam = build_contamination_fps(CONTAM_N)

    seen_fps: set[str] = set()
    built_at = time.strftime("%F %T")
    for name, cfg in SPLITS.items():
        print(f"[rulers] building split {name} (docs {cfg['doc_offset']}.."
              f"{cfg['doc_offset'] + cfg['n_docs'] - 1})...")
        batches, fps = build_held_out_v2(
            tok, seq_len=2048, batch_size=4,
            n_docs_per_source=cfg["n_docs"], batches_per_source=cfg["batches"],
            doc_offset=cfg["doc_offset"],
            exclude_fps=contam | seen_fps)  # cross-split exclusion: structural
        overlap = fps & seen_fps
        assert not overlap, f"cross-split fingerprint overlap in {name}: {len(overlap)}"
        n_batches = {k: len(v) for k, v in batches.items()}
        manifest = dict(
            split=name, built_at=built_at, contamination_n=CONTAM_N,
            skip_docs=50_000, doc_offset=cfg["doc_offset"],
            n_docs_per_source=cfg["n_docs"],
            batches_per_source=cfg["batches"], seq_len=2048, batch_size=4,
            docs_fingerprinted=len(fps), batches=n_batches,
            cross_split_overlap=0,
        )
        path = f"data/ruler_{name}.pt"
        torch.save({"batches": {k: [t.cpu() for t in v] for k, v in batches.items()},
                    "fps": sorted(fps), "manifest": manifest}, path)
        print(f"[rulers] {name}: {len(fps)} docs, batches={n_batches} -> {path}")
        seen_fps |= fps

    print("[rulers] all splits disjoint by construction. DONE")


if __name__ == "__main__":
    main()
