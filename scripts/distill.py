#!/usr/bin/env python
"""DEPRECATED (2026-07-30, reviewer blocker #1): do NOT launch new runs with
this script. Its eval calls the OLD leaky build_held_out (train/eval prefix
overlap) and its data cursor is not saved in checkpoints. It is kept for
archaeology only — v2 distill-750 numbers were produced here. The v3 stage-2
isolated local-alignment run will use a new script built on build_held_out_v2
(see scripts/eval_heldout.py and LOGBOOK.md).

LoLCATs-style attention distillation: transfer Qwen3-0.6B-Base attention
outputs into the KDA layers of the 3:1 hybrid student.

- teacher: frozen Qwen3-0.6B-Base (bf16, eval, no_grad)
- student: hybrid model, everything frozen except the 21 KDA layers
- loss: per-layer MSE between teacher attention output and student KDA output
  (captured via forward hooks), computed in fp32, meaned over layers.

Loss design choice: DIRECT per-layer MSE (not variance-normalized). LoLCATs
uses plain MSE for attention transfer; it keeps the gradient magnitude
proportional to the absolute output error, so layers with large outputs
(high residual-stream impact) dominate -- exactly what we want for
end-to-end behavior. Normalized MSE would equalize layers but distort their
relative importance. MSE is computed in fp32 to avoid bf16 underflow.

Usage: see --help. Checkpoints: save dir contains model.safetensors +
layout.json (save_hybrid format) + optimizer.pt + meta.json.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

import torch

# DEPRECATED hard-gate (2026-07-30 reviewer round-2 blocker #1): a docstring
# warning does not prevent accidental launches; refuse to start unless the
# operator explicitly acknowledges. Removes the flag so argparse never sees it.
if os.environ.get("ALLOW_DEPRECATED") != "1":
    if "--allow-deprecated" in sys.argv:
        sys.argv.remove("--allow-deprecated")
    else:
        raise SystemExit(
            "DEPRECATED script: old leaky held-out + no data-cursor resume. "
            "Kept for archaeology only. To run anyway: --allow-deprecated")

sys.path.insert(0, "/root/projects/qingyi-kda")

import bitsandbytes as bnb
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from qingyi_kda.data import build_held_out, make_train_iterator
from qingyi_kda.surgery import (
    KDA_LAYERS,
    build_hybrid_model,
    get_attention_pairs,
)

MODEL_PATH = "/root/projects/qingyi-kda/models/Qwen3-0.6B-Base"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-accum", type=int, default=0,
                   help="0 = auto (~0.5M tokens per optimizer step)")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--save-dir", type=str,
                   default="/root/projects/qingyi-kda/models/distill-checkpoints")
    p.add_argument("--no-grad-ckpt", action="store_true")
    return p.parse_args()


def lr_at(step, total, base_lr, min_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def _write_checkpoint(student, optimizer, step, save_dir):
    """Write checkpoint atomically: save into a tmp sibling dir, then rename.
    A crash mid-write leaves a *.tmp dir (never a corrupt 'complete' one)."""
    tmp_dir = save_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    # save_hybrid format: model.safetensors + layout.json
    sd = {k: v.detach().to("cpu", copy=True).contiguous()
          for k, v in student.state_dict().items()}
    if student.config.tie_word_embeddings and "lm_head.weight" in sd:
        del sd["lm_head.weight"]
    save_file(sd, os.path.join(tmp_dir, "model.safetensors"))
    with open(os.path.join(tmp_dir, "layout.json"), "w") as f:
        json.dump(student._kda_layout, f, indent=2)
    torch.save(optimizer.state_dict(), os.path.join(tmp_dir, "optimizer.pt"))
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump({"step": step}, f)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.replace(tmp_dir, save_dir)


def save_checkpoint(student, optimizer, step, save_root, keep=2):
    """Save step checkpoint under save_root/step-N, keeping only the newest
    `keep` step checkpoints (each ~1.3GB) to bound disk usage."""
    ck = os.path.join(save_root, f"step-{step}")
    _write_checkpoint(student, optimizer, step, ck)
    # rotate: delete older step-* dirs beyond the newest `keep`
    steps = sorted(
        (int(d.split("-")[1]) for d in os.listdir(save_root)
         if d.startswith("step-") and not d.endswith(".tmp")
         and os.path.isdir(os.path.join(save_root, d))),
        reverse=True,
    )
    for old in steps[keep:]:
        shutil.rmtree(os.path.join(save_root, f"step-{old}"))
        print(f"[checkpoint] rotated out step-{old}")
    return ck


def save_best(student, optimizer, step, save_root, gap=None):
    """Snapshot the current model as best/ (the final distillation artifact).
    The CE gap is recorded in meta.json so a resumed run can continue the
    best-gap tracking instead of resetting it."""
    _write_checkpoint(student, optimizer, step, os.path.join(save_root, "best"))
    if gap is not None:
        meta_path = os.path.join(save_root, "best", "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["gap"] = gap
        with open(meta_path, "w") as f:
            json.dump(meta, f)


def main():
    args = parse_args()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    grad_accum = args.grad_accum or max(
        1, round(500_000 / (args.micro_batch * args.seq_len))
    )
    print(f"config: steps={args.steps} micro_batch={args.micro_batch} "
          f"seq_len={args.seq_len} grad_accum={grad_accum} "
          f"tokens/step={args.micro_batch * args.seq_len * grad_accum:,} "
          f"lr={args.lr}->{args.min_lr} warmup={args.warmup}")

    # ---- models ----
    print("loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)
    teacher = teacher.to(device).eval().requires_grad_(False)

    print("building student (hybrid)...")
    if args.resume:
        from qingyi_kda.surgery import load_hybrid
        student = load_hybrid(args.resume, dtype=torch.bfloat16, device=device)
        with open(os.path.join(args.resume, "meta.json")) as f:
            start_step = json.load(f)["step"]
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        student = build_hybrid_model(MODEL_PATH, dtype=torch.bfloat16,
                                     device=device, seed=0)
        start_step = 0
    student.train()
    for name, p in student.named_parameters():
        p.requires_grad = any(f"layers.{i}." in name and ".self_attn." in name
                              for i in KDA_LAYERS)
    trainable = [p for p in student.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")

    if not args.no_grad_ckpt:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("gradient checkpointing: ON (student)")

    # ---- distillation hooks: teacher attn output vs student KDA output ----
    pairs = get_attention_pairs(teacher, student)
    t_out, s_out = {}, {}
    handles = []

    # Capture gate: with non-reentrant gradient checkpointing, forward hooks
    # fire AGAIN during the backward-pass recompute. If a recomputed tensor is
    # stored, its reference pins that layer's entire backward saved-tensor set
    # (chunk_kda keeps h/Aqk/w/u etc., ~0.3GB per KDA layer) until the dict is
    # cleared -- all 21 layers at once, ~6GB, spilling into host-shared memory
    # and crushing throughput. Skip storing during backward.
    capture = {"on": True}

    def make_hook(store, idx):
        def hook(module, args, output):
            if capture["on"]:
                store[idx] = output[0] if isinstance(output, tuple) else output
        return hook

    for idx, t_attn, s_kda in pairs:
        handles.append(t_attn.register_forward_hook(make_hook(t_out, idx)))
        handles.append(s_kda.register_forward_hook(make_hook(s_out, idx)))

    # ---- optimizer ----
    opt = bnb.optim.Adam8bit(trainable, lr=args.lr)
    if args.resume and os.path.exists(os.path.join(args.resume, "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(args.resume, "optimizer.pt"),
                                       map_location="cpu"))

    # ---- data ----
    print("building held-out eval set (first 200 docs per source)...")
    held_out = build_held_out(tok, args.seq_len, args.micro_batch,
                              n_docs_per_source=200)
    print(f"held-out batches: {len(held_out)}")
    train_iter = make_train_iterator(tok, args.seq_len, seed=0)

    def chunked_ce(logits, tgt, chunk=512):
        """Mean next-token CE without materializing fp32 [B*T, V] at once
        (fp32 logits over the 151936-token vocab are ~1.2GB per 512 tokens)."""
        flat = logits[:, :-1].reshape(-1, logits.size(-1))
        n = tgt.numel()
        total = 0.0
        for i in range(0, n, chunk):
            total += torch.nn.functional.cross_entropy(
                flat[i:i + chunk].float(), tgt[i:i + chunk], reduction="sum"
            ).item()
        return total / n

    @torch.no_grad()
    def evaluate():
        student.eval()
        ce_t = ce_s = n = 0
        for batch in held_out[:args.eval_batches]:
            batch = batch.to(device)
            tgt = batch[:, 1:].reshape(-1)
            lt = teacher(batch, use_cache=False).logits
            ce_t += chunked_ce(lt, tgt)
            del lt
            ls = student(batch, use_cache=False).logits
            ce_s += chunked_ce(ls, tgt)
            del ls
            n += 1
        student.train()
        return ce_t / n, ce_s / n

    # ---- training loop ----
    print("=" * 78)
    print("start training")
    print("=" * 78)
    running = 0.0
    # Seed best-gap tracking from an existing best/ checkpoint (resume case),
    # otherwise the first eval after resume would overwrite a better best/.
    best_meta = os.path.join(args.save_dir, "best", "meta.json")
    best_gap = float("inf")
    if os.path.exists(best_meta):
        try:
            with open(best_meta) as f:
                best_gap = json.load(f).get("gap", float("inf"))
            print(f"existing best gap: {best_gap:+.4f}")
        except (json.JSONDecodeError, OSError):
            pass
    t_start = time.perf_counter()
    tokens_done = 0
    for step in range(start_step, args.steps):
        lr = lr_at(step, args.steps, args.lr, args.min_lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            batch = torch.stack([next(train_iter)
                                 for _ in range(args.micro_batch)]).to(device)
            # Backbone-only forward: the distillation targets are layer
            # outputs, so skip lm_head entirely. At B=4, T=2048 the logits
            # alone would cost ~2.5GB (bf16) per model.
            with torch.no_grad():
                teacher.model(batch, use_cache=False)
            student.model(batch, use_cache=False)
            # fp32 MSE per layer, meaned over layers
            loss = sum(
                torch.nn.functional.mse_loss(s_out[i].float(), t_out[i].float())
                for i in t_out
            ) / len(t_out)
            capture["on"] = False  # recompute hooks must not store (see above)
            (loss / grad_accum).backward()
            capture["on"] = True
            step_loss += loss.item() / grad_accum
            t_out.clear()
            s_out.clear()

        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        running += step_loss
        tokens_done += args.micro_batch * args.seq_len * grad_accum

        if not math.isfinite(step_loss):
            print(f"step {step}: NON-FINITE LOSS {step_loss}, aborting")
            sys.exit(1)

        if (step + 1) % args.log_every == 0 or step == start_step:
            dt = time.perf_counter() - t_start
            tps = tokens_done / dt if dt > 0 else 0
            vram = torch.cuda.max_memory_allocated() / 1024**3
            print(f"step {step + 1:5d} | mse {running / (step + 1 - start_step):.6f} "
                  f"| last {step_loss:.6f} | gnorm {gnorm.item():.2f} | lr {lr:.2e} "
                  f"| {tps:,.0f} tok/s | peak {vram:.2f} GiB")

        if (step + 1) % args.eval_every == 0:
            try:
                ce_t, ce_s = evaluate()
                gap = ce_s - ce_t
                print(f"[eval step {step + 1}] teacher CE {ce_t:.4f} | "
                      f"student CE {ce_s:.4f} | gap {gap:+.4f}")
                if gap < best_gap:
                    best_gap = gap
                    save_best(student, opt, step + 1, args.save_dir, gap=gap)
                    print(f"[best] new best gap {best_gap:+.4f} at step {step + 1}, "
                          f"saved {args.save_dir}/best")
            except RuntimeError as e:
                # transient WSL2 GPU errors (e.g. "device not ready") must not
                # kill the training run
                print(f"[eval step {step + 1}] SKIPPED: {type(e).__name__}: {e}")

        if (step + 1) % args.save_every == 0:
            ck = save_checkpoint(student, opt, step + 1, args.save_dir)
            print(f"[checkpoint] saved {ck}")

    # final save
    ck = save_checkpoint(student, opt, args.steps, args.save_dir)
    print(f"[checkpoint] saved {ck}")

    try:
        ce_t, ce_s = evaluate()
        gap = ce_s - ce_t
        print(f"[final eval] teacher CE {ce_t:.4f} | student CE {ce_s:.4f} | "
              f"gap {gap:+.4f}")
        if gap < best_gap:
            best_gap = gap
            save_best(student, opt, args.steps, args.save_dir, gap=gap)
            print(f"[best] new best gap {best_gap:+.4f} at final step, "
                  f"saved {args.save_dir}/best")
    except RuntimeError as e:
        print(f"[final eval] SKIPPED: {type(e).__name__}: {e}")
    for h in handles:
        h.remove()
    print("DONE")


if __name__ == "__main__":
    main()
