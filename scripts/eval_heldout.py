"""Evaluate checkpoints on the v3 ruler trio (leak-free, 200k contamination).

Appends one line per eval to results/heldout_v2_log.jsonl and prints a table.

  python scripts/eval_heldout.py <ckpt_dir> [--tag NAME] [--set tune|valid|release]

Splits (--set), built once by scripts/build_rulers.py:
  tune    : dev/tuning ruler — clean docs 0..199 of each source, 2 batches/src.
            Use for grid search, hyperparameter tuning.
  valid   : validation ruler — clean docs 200..799, 8 batches/src. Use for
            checkpoint selection during a run.
  release : release ruler — clean docs 800..1399, 8 batches/src. LOCKED:
            requires --unlock-release. Final go/no-go before publishing only;
            never tune against it.

All three caches carry a manifest and are disjoint by construction (built in
one process, cross-split fingerprint exclusion + zero-overlap assertion).
Caches must exist; this script never builds rulers itself.

The old data/heldout_v2*.pt caches (100k contamination, 2 cross-split
fingerprint overlaps) are SUPERSEDED and must not be used for new decisions.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from qingyi_kda.surgery import load_hybrid

TEACHER_DIR = "models/Qwen3-0.6B-Base"
SPLIT_CFG = {
    "tune":    dict(cache="data/ruler_tune.pt"),
    "valid":   dict(cache="data/ruler_valid.pt"),
    "release": dict(cache="data/ruler_release.pt"),
}


def CE(log_probs: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over non-masked positions."""
    return F.cross_entropy(
        log_probs.reshape(-1, log_probs.size(-1)), tgt.reshape(-1), ignore_index=-100)


def KL(student_lp: torch.Tensor, teacher_lp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Mean forward KL(teacher || student) over non-masked positions."""
    mask = tgt != -100
    t = teacher_lp[mask].exp()
    return (t * (teacher_lp[mask] - student_lp[mask])).sum(-1).mean()


def load_ruler(split: str, device: str) -> dict[str, list[torch.Tensor]]:
    cache = SPLIT_CFG[split]["cache"]
    if not os.path.exists(cache):
        raise SystemExit(f"ruler cache missing: {cache} — run scripts/build_rulers.py first")
    d = torch.load(cache, weights_only=True)
    m = d.get("manifest", {})
    print(f"ruler manifest: split={m.get('split')} contamination_n={m.get('contamination_n')} "
          f"docs={m.get('docs_fingerprinted')} cross_split_overlap={m.get('cross_split_overlap')}")
    return {k: [t.to(device) for t in v] for k, v in d["batches"].items()}


@torch.no_grad()
def eval_model(model, batches_by_src, teacher=None):
    tot_ce = tot_kl = tot_n = 0
    for src, batches in batches_by_src.items():
        for ids in batches:
            tgt = torch.cat([ids[:, 1:], torch.full((ids.size(0), 1), -100, dtype=ids.dtype, device=ids.device)], 1)
            lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits.float(), -1)
            ce = CE(lp, tgt).item()
            kl = 0.0
            if teacher is not None:
                tlp = torch.log_softmax(teacher(input_ids=ids, use_cache=False).logits.float(), -1)
                kl = KL(lp, tlp, tgt).item()
            n = int((tgt != -100).sum())
            tot_ce += ce * n
            tot_kl += kl * n
            tot_n += n
    return tot_ce / tot_n, tot_kl / tot_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="checkpoint dirs to evaluate")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", dest="split", choices=["tune", "valid", "release"], default="tune")
    ap.add_argument("--unlock-release", action="store_true",
                    help="required for --set release (final go/no-go only)")
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.split == "release" and not a.unlock_release:
        raise SystemExit("release ruler is LOCKED: pass --unlock-release "
                         "(final go/no-go before publishing only)")

    seqs = load_ruler(a.split, a.device)
    print(f"ruler split: {a.split}  ({SPLIT_CFG[a.split]['cache']}, "
          f"{ {k: len(v) for k, v in seqs.items()} } batches/source)")

    teacher = None
    rows = []
    if not a.no_teacher:
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER_DIR, dtype=torch.bfloat16, attn_implementation="eager").to(a.device)
        teacher.eval()
        tce, _ = eval_model(teacher, seqs)
        rows.append(("teacher", tce, 0.0, 0.0))

    for c in a.ckpts:
        model = load_hybrid(c, dtype=torch.bfloat16, device=a.device)
        model.eval()
        ce, kl = eval_model(model, seqs, teacher)
        rows.append((a.tag or os.path.basename(c.rstrip("/")), ce, ce - rows[0][1], kl))
        del model
        torch.cuda.empty_cache()

    print(f"\n{'model':<38} {'CE':>8} {'gap':>7} {'KL':>8}")
    for name, ce, gap, kl in rows:
        print(f"{name:<38} {ce:8.4f} {gap:+7.4f} {kl:8.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/heldout_v2_log.jsonl", "a", encoding="utf-8") as f:
        for name, ce, gap, kl in rows[1:]:
            f.write(json.dumps({"ts": time.strftime("%F %T"), "split": a.split,
                                "ckpt": name, "ce": round(ce, 6), "gap": round(gap, 6),
                                "kl": round(kl, 6)}) + "\n")


if __name__ == "__main__":
    main()
