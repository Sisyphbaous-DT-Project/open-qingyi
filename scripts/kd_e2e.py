#!/usr/bin/env python
"""Stage-3 end-to-end forward-KL distillation (GenDistill recipe, alpha_CE=0).

Stage 2 (align_layers.py) repaired the KDA layers' mechanics in isolation;
C-Eval stayed at the random line (25.48%) because knowledge retrieval is an
END-TO-END property. This stage distills the whole student stream so its
next-token distribution tracks the frozen teacher's — pure forward KL, no
CE term. GenDistill (arXiv:2603.26556) measured that even alpha_CE=0.01
drops C-Eval by 8.4 points, so CE is exactly zero here.

Loss: mean over tokens of KL(teacher || student) at temperature T, computed
by ChunkedKL (custom autograd from the reviewed cpt_kl.py — never
materializes the [B,T,V] logits graph; backward recomputes per-chunk).

Reviewer round-4 fixes (2026-07-30):
- P0-1 TRAINABLE SCOPE = GenDistill stage-3a: 21 KDA self_attn modules +
  ALL MLPs + ALL block LayerNorms + final norm (~457.9M params). Frozen:
  teacher, embeddings, tied lm_head, the 7 kept attention layers (incl.
  their q/k norms). GenDistill's freeze-MLP ablation: C-Eval 37.9 -> 31.4
  at 250M tokens — MLP/norm hold and route knowledge; freezing them was
  judged likely to re-break the knowledge pathway.
- P0-2 BYPASS FULL LOGITS: hidden states come from
  `model.model(input_ids=...).last_hidden_state` (verified elementwise
  identical to hidden_states[-1], max diff 0) — the CausalLM wrapper no
  longer materializes [B,T,151936] logits (~2.3 GiB/step) or stores all
  29 layer hiddens.
- INIT HARD LOCK: a fresh run REQUIRES --expect-init-hash matching the
  stage-2 checkpoint's per-tensor hash; on resume the stored hash is
  RE-VERIFIED against --expect-init-hash when the flag is passed again.
- RESUME SAFETY: best_ce/best_kl are restored from trainer_state (a
  resumed run can no longer overwrite a better historical best); stored
  hparams (lr, min_lr, warmup, weight_decay, temperature, kl_chunk,
  seq_len, micro_batch, grad_accum, max_rows, total_steps) are validated
  against the CLI — mismatch aborts unless --allow-hparam-override.
- DUAL BEST, SLIM: best-ce and best-KL checkpoints store ONLY the model +
  layout + a small pointer JSON (~1.3 GB each, no optimizer/master/rng —
  those live in step-N checkpoints only, ~6.8 GB each). Both thresholds
  are updated BEFORE any save so a later resume never sees a stale best.
- EVAL MEMORY: full_ruler_eval (align_layers.py) now runs per-sequence
  with token-chunked fp32 math — peak stays small even with ~8.2 GB of
  resident training state (fp32 master + Adam for 457.9M params).
- HPARAMS aligned to GenDistill where load-bearing: lr 2.5e-5, 10%
  warmup, weight_decay 0.1.

Reviewer round-5 fixes (2026-07-30, post-smoke):
- CLEAN SHUTDOWN: mixer.close() (explicit stream teardown), CUDA sync,
  model/optimizer deletion and cache release all happen BEFORE process
  exit — the r5 smoke ended in "terminate called without an active
  exception" (non-zero exit) after DONE, likely a background stream /
  teardown thread dying late. TOKENIZERS_PARALLELISM disabled as well.
- DEVICE-SIDE FP32 LOSS ACCUMULATION: ChunkedKL.forward now accumulates
  the KL sum as a device fp32 scalar (no per-chunk .item() host syncs);
  the returned loss is exact fp32 math rounded once at the end, so
  training-log KL is no longer bf16-coarse.

KNOWN DEVIATIONS from GenDistill (declared, not bugs):
- effective batch = micro_batch * grad_accum = 4 sequences (paper: 64) —
  memory-bound; if OOM use --micro-batch 2 --grad-accum 2 to keep 8192
  tokens/optimizer-step (r5 smoke: micro4 OOMs deterministically,
  micro2+accum2 leaves ~10 GiB headroom on a 32G card).
- LR schedule: full cosine to min_lr = 0.1 * lr (paper: warmup-stable-
  decay, last-10% decay, min ratio 0.02).
- Stage-3a on pretraining data only (paper also has a 3b instruction
  phase, seq 4096, completion-only KL — later).
- 57M tokens (7000 steps) is a PILOT budget: paper main recipe 500M, min
  ablation 100M. Gate: C-Eval off the random line before extending.

Usage:
  python scripts/kd_e2e.py --total-steps 7000 --run-until 4 --eval-every 4 \
      --save-every 4 --micro-batch 2 --grad-accum 2 \
      --expect-init-hash <stage2-best-hash>            # smoke (incl. eval)
  python scripts/kd_e2e.py --total-steps 7000 --run-until 250 \
      --micro-batch 2 --grad-accum 2 \
      --expect-init-hash <stage2-best-hash>            # pilot segment
  python scripts/kd_e2e.py --resume models/kd-checkpoints/step-250 \
      --expect-init-hash <stage2-best-hash>
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.environ.get("QINGYI_ROOT", "/root/projects/qingyi-kda"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_layers import (CursorMixer, MasterWeights,  # noqa: E402
                          full_ruler_eval, tensor_hash)
from qingyi_kda.surgery import KDA_LAYERS, load_hybrid  # noqa: E402

TEACHER_DIR = "models/Qwen3-0.6B-Base"
DEFAULT_INIT = "models/align-checkpoints/best"
VALID_RULER = "data/ruler_valid.pt"

RESUME_HPARM_KEYS = ["lr", "min_lr", "warmup", "weight_decay", "temperature",
                     "kl_chunk", "seq_len", "micro_batch", "grad_accum",
                     "max_rows", "total_steps", "data_jsonl", "seed"]


class ChunkedKL(torch.autograd.Function):
    """KL(teacher || student) * T^2, chunked over tokens, custom backward.

    Reviewed implementation carried over from cpt_kl.py / kl_distill.py:
    forward accumulates without a graph (never materializes full logits);
    backward computes d/dz_s = T*(p_s - p_t)/n per token, grad_h = dz @ W_s,
    also chunked. Round-5: the forward sum is a DEVICE-side fp32 scalar
    (no per-chunk .item() host syncs; bf16-coarse log values eliminated).

    Completion-only mode: optional per-token `mask` (fp32, same length as
    tokens). Positions with mask=0 contribute nothing to forward or
    backward; the normalizer n becomes mask.sum() (clamped to >=1).

    Gradient-accumulation mode: optional `norm_override` (float). When set,
    it replaces the normalizer in BOTH forward and backward, so several
    micro-batches can share one global token count — the accumulated
    gradient then equals the mean over ALL completion tokens in the whole
    optimizer step, not the mean of per-micro-batch means.
    """

    @staticmethod
    def forward(ctx, hs, ht, Ws, Wt, T, chunk, mask=None, norm_override=None):
        ctx.save_for_backward(hs, ht, Ws, Wt)
        ctx.T, ctx.chunk = T, chunk
        ctx.mask = mask
        n = hs.size(0)
        if norm_override is not None:
            norm = float(norm_override)
        elif mask is not None:
            norm = float(mask.sum().clamp(min=1.0))
        else:
            norm = float(n)
        ctx.norm = norm
        total = hs.new_zeros((), dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, n, chunk):
                ls = (hs[i:i + chunk] @ Ws.T).float() / T
                lt = (ht[i:i + chunk] @ Wt.T).float() / T
                log_ps = F.log_softmax(ls, -1)
                log_pt = F.log_softmax(lt, -1)
                pt = log_pt.exp()
                per_tok = (pt * (log_pt - log_ps)).sum(-1)
                if mask is not None:
                    per_tok = per_tok * mask[i:i + chunk]
                total += per_tok.sum()
        return total * T * T / norm  # fp32 scalar: fine-grained training-log KL

    @staticmethod
    def backward(ctx, grad_out):
        hs, ht, Ws, Wt = ctx.saved_tensors
        T, chunk, mask = ctx.T, ctx.chunk, ctx.mask
        norm = ctx.norm
        n = hs.size(0)
        g = torch.empty_like(hs)
        for i in range(0, n, chunk):
            ls = (hs[i:i + chunk] @ Ws.T).float() / T
            lt = (ht[i:i + chunk] @ Wt.T).float() / T
            ps = F.softmax(ls, -1)
            pt = F.softmax(lt, -1)
            dz = (ps - pt) * T
            gi = dz @ Ws.float()
            if mask is not None:
                gi = gi * mask[i:i + chunk, None]
            g[i:i + chunk] = gi.to(hs.dtype) / norm
        return ((g * grad_out).to(hs.dtype), None, None, None, None, None,
                None, None)


def freeze_gendistill_scope(model) -> int:
    """GenDistill stage-3a scope: KDA layers + all MLPs + all norms trainable.

    Frozen: embeddings, tied lm_head, the 7 kept attention layers (their
    q/k norms live under self_attn and stay frozen with them).
    """
    trainable = 0
    for n, p in model.named_parameters():
        on = (any(f"layers.{i}.self_attn." in n for i in KDA_LAYERS)
              or ".mlp." in n
              or "input_layernorm" in n
              or "post_attention_layernorm" in n
              or n == "model.norm.weight")
        p.requires_grad = on
        trainable += p.numel() if on else 0
    return trainable


class JSONLMixer:
    """Completion-masked sample mixer for arm-C style data (MCQ / QA pairs).

    Reads one or more JSONL files, renders each row to (prompt, completion),
    tokenizes as ids = prompt + completion + [eos], and packs samples into
    seq_len with a per-position mask marking the positions whose NEXT token
    belongs to a completion span (completion-only KL).

    Packing rule (reviewer P1 fix): a sample is NEVER split across packs.
    If the next sample does not fit the remaining space, the pack is padded
    with eos tokens at mask=0 and the sample moves to the next pack whole.
    Samples longer than seq_len are skipped (counted in self.skipped).

    Deterministic: single seed-shuffled pass over merged rows, wrap-around
    with the same order; state/load_state restore cursor + pending sample.
    self.data_sha256 fingerprints the input file bytes (resume tamper lock).
    """

    MCQ_PREFIX = "以下是一道单项选择题，请选出唯一正确答案。"
    LABELS = "ABCD"

    def __init__(self, tokenizer, seq_len: int, seed: int, paths: list[str],
                 max_rows: int):
        self.tok = tokenizer
        self.seq_len = seq_len
        self.max_rows = max_rows
        self.eos = tokenizer.eos_token_id
        self.rng = random.Random(seed)
        self.paths = list(paths)
        self.samples: list[tuple[str, str]] = []
        sha = hashlib.sha256()
        for p in self.paths:
            n0 = len(self.samples)
            with open(p, "rb") as f:
                data = f.read()
            sha.update(len(data).to_bytes(8, "big"))
            sha.update(hashlib.sha256(data).digest())
            for line in data.decode("utf-8").splitlines():
                if line.strip():
                    self.samples.append(self._render(json.loads(line)))
            print(f"data: {p} -> {len(self.samples) - n0} samples")
        if not self.samples:
            raise SystemExit(f"no usable rows in {self.paths}")
        self.data_sha256 = sha.hexdigest()
        self.order = list(range(len(self.samples)))
        self.rng.shuffle(self.order)
        self.cursor = 0           # rows consumed (into self.order)
        self.pending: tuple[list[int], list[float]] | None = None
        self.skipped = 0          # samples dropped for exceeding seq_len

    def _render(self, row: dict) -> tuple[str, str]:
        if row.get("type") == "mcq" or "options" in row:
            lines = [self.MCQ_PREFIX, "", row["question"]]
            lines.extend(f"{lab}. {opt}" for lab, opt
                         in zip(self.LABELS, row["options"]))
            lines.append("答案：")
            label = row.get("answer_label")
            if label is None:
                label = self.LABELS[row["answer_index"]]
            return "\n".join(lines), " " + label
        return row["prompt"], row["completion"]

    def _tokenize_next(self) -> None:
        """Advance the cursor and set self.pending to the next sample."""
        if self.cursor >= min(len(self.order), self.max_rows):
            self.cursor = 0  # wrap: same shuffled order
        prompt, completion = self.samples[self.order[self.cursor]]
        self.cursor += 1
        p_ids = self.tok(prompt, add_special_tokens=False).input_ids
        c_ids = self.tok(completion, add_special_tokens=False).input_ids
        c_ids = c_ids + [self.eos]
        ids = p_ids + c_ids
        # mask=1 at positions predicting completion/eos tokens:
        # position j predicts token j+1, so span is len(p)-1 .. len(ids)-2
        mask = [0.0] * len(ids)
        for j in range(len(p_ids) - 1, len(ids) - 1):
            mask[j] = 1.0
        self.pending = (ids, mask)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        ids: list[int] = []
        mask: list[float] = []
        while len(ids) < self.seq_len:
            if self.pending is None:
                self._tokenize_next()
            s_ids, s_mask = self.pending
            if len(s_ids) > self.seq_len:
                self.pending = None
                self.skipped += 1
                print(f"[mixer] skipped overlong sample "
                      f"({len(s_ids)} > {self.seq_len}), total {self.skipped}")
                continue
            room = self.seq_len - len(ids)
            if len(s_ids) <= room:
                ids.extend(s_ids)
                mask.extend(s_mask)
                self.pending = None
            else:
                # sample does not fit: pad the pack, keep sample whole
                pad = self.seq_len - len(ids)
                ids.extend([self.eos] * pad)
                mask.extend([0.0] * pad)
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32))

    def state(self) -> dict:
        pend_ids, pend_mask = self.pending if self.pending else ([], [])
        return {"kind": "jsonl", "cursor": self.cursor,
                "pending_ids": list(pend_ids),
                "pending_mask": list(pend_mask),
                "skipped": self.skipped,
                "data_sha256": self.data_sha256}

    def load_state(self, st: dict) -> None:
        if st.get("kind") != "jsonl":
            raise ValueError("not a JSONLMixer state")
        self.cursor = st["cursor"]
        pend_ids = list(st.get("pending_ids", []))
        pend_mask = list(st.get("pending_mask", []))
        self.pending = (pend_ids, pend_mask) if pend_ids else None
        self.skipped = st.get("skipped", 0)

    def close(self) -> None:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=7000,
                    help="schedule horizon (cosine denominator); stored in "
                         "checkpoints, never recomputed on resume")
    ap.add_argument("--run-until", type=int, default=None,
                    help="stop after this step (default: total-steps)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient accumulation steps; tokens/optimizer-step "
                         "= micro-batch * grad-accum * seq-len")
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--min-lr", type=float, default=2.5e-6)
    ap.add_argument("--warmup", type=int, default=None,
                    help="default: 10%% of total-steps (GenDistill)")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kl-chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--data-jsonl", default=None,
                    help="arm-C mode: comma-separated JSONL file(s) of "
                         "mcq/QA rows (merged, shuffled together); switches "
                         "the mixer to JSONLMixer and enables "
                         "completion-only KL (prompt positions masked out)")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--keep", type=int, default=2,
                    help="how many newest step-N checkpoints to keep (best* exempt)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="models/kd-checkpoints")
    ap.add_argument("--init", default=DEFAULT_INIT,
                    help="stage-2 aligned checkpoint (align best/step-N)")
    ap.add_argument("--expect-init-hash", default=None,
                    help="REQUIRED on a fresh run: per-tensor hash of the "
                         "stage-2 init checkpoint (hard lock, no default); "
                         "on resume, re-verified against the stored hash")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--allow-hparam-override", action="store_true",
                    help="resume even if stored hparams differ from CLI")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.warmup is None:
        a.warmup = max(1, int(0.1 * a.total_steps))

    init_hash = None
    if not a.resume:
        init_sf = os.path.join(a.init, "model.safetensors")
        if not os.path.exists(init_sf):
            raise SystemExit(f"init checkpoint not found: {init_sf}")
        init_hash = tensor_hash(init_sf)
        if a.expect_init_hash is None:
            raise SystemExit(
                "fresh run requires --expect-init-hash (hard lock on the "
                f"stage-2 artifact); computed hash is: {init_hash}")
        if init_hash != a.expect_init_hash:
            raise SystemExit(
                f"INIT HASH MISMATCH: {init_hash} != {a.expect_init_hash}; "
                "refusing to train from an unapproved artifact")
        print(f"init: {a.init} hash {init_hash[:16]}... (verified against "
              "--expect-init-hash)")

    tok = AutoTokenizer.from_pretrained(TEACHER_DIR)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_DIR, dtype=torch.bfloat16, attn_implementation="eager").to(a.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = load_hybrid(a.resume or a.init, dtype=torch.bfloat16, device=a.device)
    student.train()
    n_trainable = freeze_gendistill_scope(student)
    print(f"trainable params (GenDistill stage-3a scope): "
          f"{n_trainable/1e6:.1f}M (end-to-end KL, alpha_CE=0)")

    master = MasterWeights(student)
    opt = torch.optim.AdamW(list(master.params.values()),
                            lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=a.weight_decay)
    if a.data_jsonl:
        paths = [p.strip() for p in a.data_jsonl.split(",") if p.strip()]
        mixer = JSONLMixer(tok, a.seq_len, a.seed, paths, a.max_rows)
        print(f"data: {a.data_jsonl} ({len(mixer.samples)} samples total, "
              f"sha256 {mixer.data_sha256[:16]}..., completion-only KL)")
    else:
        mixer = CursorMixer(tok, a.seq_len, a.seed, a.max_rows)

    Ws = student.lm_head.weight
    Wt = teacher.lm_head.weight

    start_step = 0
    total_steps = a.total_steps
    best_ce = float("inf")
    best_kl = float("inf")
    if a.resume:
        st = json.load(open(os.path.join(a.resume, "trainer_state.json")))
        stored = st.get("args", {})
        mism = {k: (stored.get(k), getattr(a, k))
                for k in RESUME_HPARM_KEYS
                if k in stored and stored[k] != getattr(a, k)}
        if mism and not a.allow_hparam_override:
            raise SystemExit(f"hparam mismatch vs stored checkpoint: {mism}; "
                             "pass --allow-hparam-override to proceed anyway")
        init_hash = st.get("init_hash")
        if a.expect_init_hash is not None and init_hash != a.expect_init_hash:
            raise SystemExit(
                f"INIT HASH MISMATCH on resume: stored {init_hash} != "
                f"{a.expect_init_hash}; refusing to continue this lineage")
        opt.load_state_dict(torch.load(os.path.join(a.resume, "optimizer.pt"),
                                       map_location=a.device))
        master.load_state_dict(torch.load(os.path.join(a.resume, "master.pt"),
                                          map_location="cpu"))
        master.push_weights(student)
        # Mixer-state compatibility (reviewer P0): a checkpoint written by a
        # different data mode (CursorMixer continuous corpus vs JSONLMixer
        # completion-masked samples) must NOT be force-loaded — arm-C starts
        # from a stage-3a checkpoint and intentionally SWITCHES data mode.
        cur_state = st.get("cursor")
        state_is_jsonl = isinstance(cur_state, dict) and \
            cur_state.get("kind") == "jsonl"
        want_jsonl = a.data_jsonl is not None
        if cur_state is not None and state_is_jsonl == want_jsonl:
            if want_jsonl:
                stored_sha = cur_state.get("data_sha256")
                if stored_sha and stored_sha != mixer.data_sha256:
                    raise SystemExit(
                        f"DATA HASH MISMATCH on resume: stored "
                        f"{stored_sha[:16]}... != current "
                        f"{mixer.data_sha256[:16]}...; the JSONL file(s) at "
                        f"the same path changed under this lineage — refusing "
                        f"to silently continue")
            mixer.load_state(cur_state)
        elif cur_state is not None:
            print(f"[resume] data mode switched "
                  f"(checkpoint jsonl={state_is_jsonl}, current "
                  f"jsonl={want_jsonl}); starting the new data stream fresh")
        rng = torch.load(os.path.join(a.resume, "rng.pt"), weights_only=False)
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        start_step = st["step"]
        total_steps = st["total_steps"]
        best_ce = st.get("best_ce", float("inf"))
        best_kl = st.get("best_kl", float("inf"))
        print(f"resumed from {a.resume} at step {start_step} "
              f"(horizon {total_steps}, best_ce {best_ce:.4f}, "
              f"best_kl {best_kl:.4f})")

    # Round-6 fix: load layout.json into memory ONCE at startup. The dir it
    # comes from (the resume checkpoint) can be rotated out by --keep mid-run;
    # copying lazily inside save_ckpt crashed the resumed pilot at step ~750
    # after step-250 was rotated. Missing file now fails fast at startup.
    layout_src = os.path.join(a.resume or a.init, "layout.json")
    if not os.path.exists(layout_src):
        raise SystemExit(f"layout.json missing at startup: {layout_src}")
    with open(layout_src, "rb") as lf:
        layout_bytes = lf.read()

    run_until = a.run_until or total_steps

    valid_batches = None
    teacher_valid_ce = None
    if not a.no_eval:
        if not os.path.exists(VALID_RULER):
            raise SystemExit(f"valid ruler missing: {VALID_RULER}")
        d = torch.load(VALID_RULER, weights_only=True)
        valid_batches = {k: [t.to(a.device) for t in v] for k, v in d["batches"].items()}
        teacher_valid_ce, _ = full_ruler_eval(teacher, teacher, valid_batches)
        print(f"teacher valid CE: {teacher_valid_ce:.4f}")

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "kd_log.jsonl"), "a", encoding="utf-8")

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        t = min(1.0, (step - a.warmup) / max(1, total_steps - a.warmup))
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * t))

    def save_ckpt(step, name=None, light=False):
        """Atomic checkpoint write. light=True (best dirs): model only —
        no optimizer/master/rng (~1.3 GB instead of ~6.8 GB); resume is
        supported from step-N checkpoints only."""
        name = name or f"step-{step}"
        tmp = os.path.join(a.out, f".{name}.tmp")
        final = os.path.join(a.out, name)
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        sd = {k: v.cpu() for k, v in student.state_dict().items()}
        save_file(sd, os.path.join(tmp, "model.safetensors"), metadata={"format": "pt"})
        with open(os.path.join(tmp, "layout.json"), "wb") as lf:
            lf.write(layout_bytes)  # from memory — the source dir may be rotated out
        if light:
            json.dump({"step": step, "best_ce": best_ce, "best_kl": best_kl,
                       "init_hash": init_hash, "light": True,
                       "note": "model-only best pointer; resume from step-N"},
                      open(os.path.join(tmp, "best_info.json"), "w"))
        else:
            torch.save(opt.state_dict(), os.path.join(tmp, "optimizer.pt"))
            torch.save(master.state_dict(), os.path.join(tmp, "master.pt"))
            torch.save({"cpu": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all()},
                       os.path.join(tmp, "rng.pt"))
            json.dump({"step": step, "total_steps": total_steps,
                       "cursor": mixer.state(), "init_hash": init_hash,
                       "init_dir": a.init, "best_ce": best_ce,
                       "best_kl": best_kl, "args": vars(a)},
                      open(os.path.join(tmp, "trainer_state.json"), "w"))
        shutil.rmtree(final, ignore_errors=True)
        os.rename(tmp, final)  # atomic publish
        print(f"[checkpoint] saved {final}{' (light)' if light else ''}")
        if name.startswith("step-"):
            steps = sorted(
                (int(d.split("-")[1]) for d in os.listdir(a.out)
                 if d.startswith("step-") and d.split("-")[1].isdigit()),
                reverse=True)
            for old in steps[a.keep:]:
                shutil.rmtree(os.path.join(a.out, f"step-{old}"), ignore_errors=True)
                print(f"[checkpoint] rotated out step-{old}")

    t0 = time.time()
    for step in range(start_step, run_until):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        kl_val = 0.0
        if a.data_jsonl:
            # Global-normalizer accumulation (reviewer fix): fetch ALL
            # micro-batches first, sum their completion-token counts, and
            # pass the total as norm_override. The accumulated gradient is
            # then the mean over every completion token in this optimizer
            # step — not the mean of per-micro-batch means, which skews
            # toward micro-batches with fewer valid tokens.
            batches = []
            for _ in range(a.grad_accum):
                pairs = [mixer.next() for _ in range(a.micro_batch)]
                b_ids = torch.stack([p[0] for p in pairs]).to(a.device)
                b_mask = torch.stack([p[1] for p in pairs]).to(a.device)
                batches.append((b_ids, b_mask.reshape(-1)))
            total_norm = float(
                torch.stack([m.sum() for _, m in batches]).sum()
                .clamp(min=1.0))
            for ids, mask in batches:
                with torch.no_grad():
                    # .model bypasses lm_head: no [B,T,V] logits
                    ht = teacher.model(input_ids=ids,
                                       use_cache=False).last_hidden_state
                hs = student.model(input_ids=ids,
                                   use_cache=False).last_hidden_state
                hs = hs.reshape(-1, hs.size(-1))
                assert hs.requires_grad and hs.grad_fn is not None, \
                    "student hidden carries no graph — freeze scope is wrong"
                kl = ChunkedKL.apply(hs, ht.reshape(-1, ht.size(-1)), Ws, Wt,
                                     a.temperature, a.kl_chunk, mask,
                                     total_norm)
                kl.backward()
                kl_val += kl.item()
        else:
            for _ in range(a.grad_accum):
                ids = torch.stack([mixer.next() for _ in range(a.micro_batch)]).to(a.device)
                with torch.no_grad():
                    # .model bypasses lm_head: no [B,T,V] logits
                    ht = teacher.model(input_ids=ids, use_cache=False).last_hidden_state
                hs = student.model(input_ids=ids, use_cache=False).last_hidden_state
                hs = hs.reshape(-1, hs.size(-1))
                assert hs.requires_grad and hs.grad_fn is not None, \
                    "student hidden carries no graph — freeze scope is wrong"
                kl = ChunkedKL.apply(hs, ht.reshape(-1, ht.size(-1)), Ws, Wt,
                                     a.temperature, a.kl_chunk, None) / a.grad_accum
                kl.backward()
                kl_val += kl.item()
        del ids, ht, hs, kl

        master.pull_grads(student)
        gnorm = torch.nn.utils.clip_grad_norm_(list(master.params.values()), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        master.push_weights(student)
        student.zero_grad(set_to_none=True)

        rec = None
        if (step + 1) % a.log_every == 0 or step == start_step:
            rec = {"step": step + 1, "kl": round(kl_val, 6),
                   "gnorm": round(gnorm, 4), "lr": lr_at(step),
                   "elapsed_s": round(time.time() - t0, 1)}

        if valid_batches is not None and (step + 1) % a.eval_every == 0:
            ce, vkl = full_ruler_eval(student, teacher, valid_batches)
            if rec is None:
                rec = {"step": step + 1}
            rec.update({"valid_ce": round(ce, 6), "valid_kl": round(vkl, 6),
                        "valid_gap": round(ce - teacher_valid_ce, 6),
                        "max_mem_gb": round(
                            torch.cuda.max_memory_allocated() / 2**30, 2)})
            # update BOTH thresholds before any save, so a later resume
            # never reads a stale best from trainer_state
            ce_improved = ce < best_ce
            kl_improved = vkl < best_kl
            if ce_improved:
                best_ce = ce
            if kl_improved:
                best_kl = vkl
            if ce_improved:
                save_ckpt(step + 1, name="best-ce", light=True)
            if kl_improved:
                save_ckpt(step + 1, name="best-kl", light=True)
            print(f"[eval step {step+1}] valid CE {ce:.4f} | gap "
                  f"{ce - teacher_valid_ce:+.4f} | KL {vkl:.4f} | "
                  f"best CE {best_ce:.4f} | best KL {best_kl:.4f} | "
                  f"max mem {rec['max_mem_gb']:.1f}G", flush=True)

        if rec is not None:
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step+1}/{run_until} kl {kl_val:.5f} "
                  f"gnorm {gnorm:.3f} lr {lr_at(step):.2e}", flush=True)

        if (step + 1) % a.save_every == 0 or (step + 1) == run_until:
            save_ckpt(step + 1)

    # ---- clean shutdown (round-5): teardown BEFORE exit, not during ----
    msg = (f"DONE | peak mem {torch.cuda.max_memory_allocated()/2**30:.1f}G "
           f"allocated, {torch.cuda.max_memory_reserved()/2**30:.1f}G reserved")
    logf.close()
    mixer.close()
    if a.device.startswith("cuda"):
        torch.cuda.synchronize()
    del student, teacher, master, opt
    torch.cuda.empty_cache()
    print(msg, flush=True)


if __name__ == "__main__":
    main()
