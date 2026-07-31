"""Streaming data pipeline for attention distillation.

Two sources mixed 60/40:
- EN (~60%): HuggingFaceTB/smollm-corpus, config ``fineweb-edu-dedup``,
  text field ``text``.
- ZH (~40%): BAAI/IndustryCorpus2 (gated -- the HF account must have accepted
  the terms on the dataset page), text field auto-detected
  (``text`` or ``content``).

Documents are tokenized, joined with <|endoftext|>, and packed into fixed
seq_len token sequences (standard practice for linearization distillation:
no doc-boundary attention mask). The mixer draws each packed sequence from a
source with the source's probability (infinite iterators). A small held-out
set (first N docs of each source, packed once at startup) is used for
teacher-vs-student CE evaluation.

If a source fails to load (e.g. gated dataset not yet accepted), the pipeline
warns and renormalizes over the remaining sources -- smoke runs can proceed
with English only.
"""

import glob
import hashlib
import random
import sys
import warnings
from pathlib import Path

import torch
from datasets import load_dataset

__all__ = ["SOURCES", "make_train_iterator", "build_held_out"]

_SOURCES_LOCAL_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    "zhwiki": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "prob": 0.5,
        "local_glob": str(_SOURCES_LOCAL_DIR / "zhwiki" / "*.parquet"),
    },
    "en": {
        "dataset": "HuggingFaceTB/smollm-corpus",
        "config": "fineweb-edu-dedup",
        "split": "train",
        "prob": 0.35,
        # local shards (preferred when present; avoids flaky hub streaming)
        "local_glob": str(_SOURCES_LOCAL_DIR / "smollm-corpus" / "fineweb-edu-dedup" / "*.parquet"),
    },
    "zh": {
        "dataset": "BAAI/IndustryCorpus2",
        "config": None,
        "split": "train",
        "prob": 0.15,
        "local_glob": str(_SOURCES_LOCAL_DIR / "IndustryCorpus2" / "*" / "chinese" / "high" / "*.parquet"),
    },
}

_TEXT_FIELDS = ("text", "content", "raw_content", "body")


def _open_stream(spec: dict):
    """Open a streaming dataset; returns (iterator, text_field).

    Prefers locally downloaded parquet shards (spec["local_glob"]) over hub
    streaming: the hub path is flaky behind the proxy.
    """
    local = sorted(glob.glob(spec.get("local_glob", "")))
    if local:
        ds = load_dataset("parquet", data_files=local, split="train", streaming=True)
    else:
        ds = load_dataset(
            spec["dataset"], spec["config"], split=spec["split"], streaming=True
        )
    fields = set(ds.features.keys())
    for f in _TEXT_FIELDS:
        if f in fields:
            return iter(ds), f
    raise ValueError(f"no known text field in {spec['dataset']}: {sorted(fields)}")


def _packed_iterator(spec: dict, tokenizer, seq_len: int):
    """Infinite iterator of packed [seq_len] token tensors from one source."""
    eos = tokenizer.eos_token_id  # <|endoftext|> = 151643 for Qwen3
    while True:  # re-open the stream when it is exhausted
        stream, field = _open_stream(spec)
        buf: list[int] = []
        for row in stream:
            text = row.get(field)
            if not text:
                continue
            buf.extend(tokenizer(text, add_special_tokens=False).input_ids)
            buf.append(eos)
            while len(buf) >= seq_len:
                yield torch.tensor(buf[:seq_len], dtype=torch.long)
                buf = buf[seq_len:]


def make_train_iterator(
    tokenizer,
    seq_len: int,
    seed: int = 0,
    sources: dict | None = None,
):
    """Infinite iterator yielding packed [seq_len] tensors, mixed by source
    probability. Sources that fail to open are dropped with a warning."""
    sources = dict(sources or SOURCES)
    rng = random.Random(seed)

    iters, probs = {}, {}
    for name, spec in sources.items():
        try:
            # Eager probe: _packed_iterator is a generator, so opening the
            # stream lazily inside it would defer failures to the first
            # next() call, outside this try/except.
            _open_stream(spec)
            iters[name] = _packed_iterator(spec, tokenizer, seq_len)
            probs[name] = spec["prob"]
        except Exception as e:
            warnings.warn(
                f"source {name!r} ({spec['dataset']}) unavailable: "
                f"{type(e).__name__}: {str(e)[:200]} -- dropping it from the mix"
            )
    if not iters:
        raise RuntimeError("no data source available")
    total = sum(probs.values())
    names = list(iters)
    weights = [probs[n] / total for n in names]
    if len(names) < len(sources):
        print(f"[data] active sources: {names} (weights {weights})", file=sys.stderr)

    while True:
        name = rng.choices(names, weights=weights, k=1)[0]
        yield next(iters[name])


def build_held_out(
    tokenizer,
    seq_len: int,
    batch_size: int,
    n_docs_per_source: int = 200,
    sources: dict | None = None,
) -> list[torch.Tensor]:
    """DEPRECATED (leaked ruler, kept only so legacy scripts still run).

    Two fatal flaws, confirmed 2026-07-30 by independent audit:
    1) LEAK: packs the FIRST n_docs_per_source docs of each source -- the same
       stream prefix the training iterator consumes (no skip, no shuffle), so
       "held-out" CE/KL measured training-prefix memorization.
    2) BIAS: batches are appended source-by-source in dict order, so
       ``held_out[:2]`` (the default eval) was 100% zhwiki.
    Use ``build_held_out_v2`` for any new measurement."""
    sources = sources or SOURCES
    eos = tokenizer.eos_token_id
    batches = []
    for name, spec in sources.items():
        try:
            stream, field = _open_stream(spec)
        except Exception as e:
            warnings.warn(
                f"held-out: source {name!r} unavailable ({type(e).__name__}), skipped"
            )
            continue
        buf: list[int] = []
        rows: list[int] = []
        for i, row in enumerate(stream):
            if i >= n_docs_per_source:
                break
            text = row.get(field)
            if not text:
                continue
            rows.extend(tokenizer(text, add_special_tokens=False).input_ids)
            rows.append(eos)
        buf = rows
        seqs = [buf[i:i + seq_len] for i in range(0, len(buf) - seq_len + 1, seq_len)]
        for j in range(0, len(seqs) - batch_size + 1, batch_size):
            batches.append(torch.stack([torch.tensor(s, dtype=torch.long)
                                        for s in seqs[j:j + batch_size]]))
    return batches


def build_held_out_v2(
    tokenizer,
    seq_len: int,
    batch_size: int,
    n_docs_per_source: int = 200,
    batches_per_source: int = 2,
    skip_docs: int = 50_000,
    doc_offset: int = 0,
    exclude_fps: set[str] | None = None,
    sources: dict | None = None,
) -> tuple[dict[str, list[torch.Tensor]], set[str]]:
    """Leak-free, stratified held-out set (the fixed ruler).

    For each source, skips the first ``skip_docs`` stream rows, then (after
    contamination filtering) skips a further ``doc_offset`` *clean* docs, then
    packs the next ``n_docs_per_source`` docs into exactly
    ``batches_per_source`` batches -- so every source is evaluated equally.
    ``doc_offset`` > 0 builds a disjoint acceptance set (tuning used clean
    docs 0..199; acceptance should use doc_offset=200).

    NOTE (2026-07-30 reviewer blocker #2): only the first
    ``batches_per_source * batch_size * seq_len`` tokens per source actually
    enter the score (~52 docs total at the default 2x4x2048). Fingerprinted
    docs beyond the window are padding. Choose batch counts accordingly.

    ``skip_docs`` alone is NOT sufficient (the corpora contain duplicate
    documents: 24 docs in the 50k..50.2k window also appear in the first
    50k), so callers should pass ``exclude_fps`` = fingerprints of every doc
    training could ever have seen; any doc whose fingerprint is in that set
    is dropped here too.

    Returns ``(batches_by_source, fingerprints)`` where fingerprints is the
    set of SHA1 hex digests (first 512 chars) of every held-out doc, for the
    zero-overlap assertions in ``scripts/test_heldout.py``.
    """
    sources = sources or SOURCES
    exclude_fps = exclude_fps or set()
    eos = tokenizer.eos_token_id
    out: dict[str, list[torch.Tensor]] = {}
    fingerprints: set[str] = set()
    for name, spec in sources.items():
        try:
            stream, field = _open_stream(spec)
        except Exception as e:
            warnings.warn(
                f"held-out v2: source {name!r} unavailable ({type(e).__name__}), skipped"
            )
            continue
        buf: list[int] = []
        kept = dropped = clean_seen = 0
        for i, row in enumerate(stream):
            if i < skip_docs:
                continue
            if kept >= n_docs_per_source:
                break
            text = row.get(field)
            if not text:
                continue
            fp = hashlib.sha1(text[:512].encode("utf-8", "ignore")).hexdigest()
            if fp in exclude_fps or fp in fingerprints:
                dropped += 1
                continue
            if clean_seen < doc_offset:
                clean_seen += 1
                continue
            fingerprints.add(fp)
            buf.extend(tokenizer(text, add_special_tokens=False).input_ids)
            buf.append(eos)
            kept += 1
        if dropped:
            print(f"[held-out v2] {name}: dropped {dropped} duplicate/"
                  f"contaminated docs", file=sys.stderr)
        seqs = [buf[j:j + seq_len] for j in range(0, len(buf) - seq_len + 1, seq_len)]
        batches = [
            torch.stack([torch.tensor(s, dtype=torch.long) for s in seqs[j:j + batch_size]])
            for j in range(0, len(seqs) - batch_size + 1, batch_size)
        ]
        out[name] = batches[:batches_per_source]
    return out, fingerprints


def build_contamination_fps(
    n_docs: int = 100_000,
    sources: dict | None = None,
) -> set[str]:
    """Fingerprints of the first ``n_docs`` docs of every source.

    This is the "everything training could ever have seen" set: all v2 stages
    (and any resume) read each source sequentially from doc 0, and the largest
    unique consumption was ~16k docs/source (62.67M tokens, 50% zhwiki), so
    100k is a >6x safety margin. Held-out docs must be disjoint from this set.
    """
    sources = sources or SOURCES
    fps: set[str] = set()
    for name, spec in sources.items():
        try:
            stream, field = _open_stream(spec)
        except Exception as e:
            warnings.warn(
                f"contamination scan: source {name!r} unavailable "
                f"({type(e).__name__}), skipped"
            )
            continue
        for i, row in enumerate(stream):
            if i >= n_docs:
                break
            text = row.get(field)
            if not text:
                continue
            fps.add(hashlib.sha1(text[:512].encode("utf-8", "ignore")).hexdigest())
        print(f"[contamination] {name}: scanned {n_docs} docs "
              f"(set size {len(fps)})", file=sys.stderr)
    return fps
