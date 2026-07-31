#!/usr/bin/env python
"""CPT (continued pre-training) of the distilled 3:1 KDA hybrid.

- start point: models/distill-checkpoints/best (attention-distilled hybrid)
- ALL parameters unfrozen, next-token CE loss, no teacher, no hooks
- logits memory is avoided with liger-kernel's fused linear cross entropy
  (LigerFusedLinearCrossEntropyLoss): the [B,T,V] logits (2.5GB bf16 at
  micro_batch=4, T=2048, V=151936) are never materialized; the loss is
  computed chunkwise inside a triton kernel straight from the hidden states
  and the tied lm_head/embedding weight. Chosen over a hand-written chunked
  CE because it keeps a single autograd node (no per-chunk backward plumbing)
  and is faster.

Checkpoints (models/cpt-checkpoints/): atomic tmp-then-rename writes,
rotation keep=2 for step checkpoints, best/ on lowest held-out CE.
Resume: --resume restores model + optimizer + step; without it, cold-starts
from the distillation best checkpoint (fresh optimizer, step 0).
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, "/root/projects/qingyi-kda")

import bitsandbytes as bnb
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from safetensors.torch import save_file
from transformers import AutoTokenizer

from qingyi_kda.data import build_held_out, make_train_iterator
from qingyi_kda.surgery import load_hybrid

DISTILL_BEST = "/root/projects/qingyi-kda/models/distill-checkpoints/best"
TEACHER_CE = 2.6410  # constant reference measured during distillation

# First 3 prompts of scripts/generate_sample.py (same prompts, comparable
# across milestones), 40 tokens each, greedy, use_cache=False.
GEN_PROMPTS = [
    "人工智能的未来是",
    "The future of artificial intelligence is",
    "中国的首都是北京，美国的首都是",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--grad-accum", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--start-from", type=str, default=DISTILL_BEST,
                   help="cold-start weights when --resume is not given")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--gen-every", type=int, default=500)
    p.add_argument("--gen-tokens", type=int, default=40)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--save-dir", type=str,
                   default="/root/projects/qingyi-kda/models/cpt-checkpoints")
    p.add_argument("--no-grad-ckpt", action="store_true")
    return p.parse_args()


def lr_at(step, total, base_lr, min_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def _write_checkpoint(model, optimizer, step, save_dir, extra_meta=None):
    """Atomic write: tmp sibling dir, then rename."""
    tmp_dir = save_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    sd = {k: v.detach().to("cpu", copy=True).contiguous()
          for k, v in model.state_dict().items()}
    if model.config.tie_word_embeddings and "lm_head.weight" in sd:
        del sd["lm_head.weight"]
    save_file(sd, os.path.join(tmp_dir, "model.safetensors"))
    with open(os.path.join(tmp_dir, "layout.json"), "w") as f:
        json.dump(model._kda_layout, f, indent=2)
    torch.save(optimizer.state_dict(), os.path.join(tmp_dir, "optimizer.pt"))
    meta = {"step": step}
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.replace(tmp_dir, save_dir)


def save_checkpoint(model, optimizer, step, save_root, keep=2):
    ck = os.path.join(save_root, f"step-{step}")
    _write_checkpoint(model, optimizer, step, ck)
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


def save_best(model, optimizer, step, save_root, ce):
    _write_checkpoint(model, optimizer, step,
                      os.path.join(save_root, "best"), extra_meta={"ce": ce})


def main():
    args = parse_args()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(
        "/root/projects/qingyi-kda/models/Qwen3-0.6B-Base")

    tokens_per_step = args.micro_batch * args.seq_len * args.grad_accum
    print(f"config: steps={args.steps} micro_batch={args.micro_batch} "
          f"seq_len={args.seq_len} grad_accum={args.grad_accum} "
          f"tokens/step={tokens_per_step:,} lr={args.lr}->{args.min_lr} "
          f"warmup={args.warmup} wd={args.weight_decay}")

    # ---- model ----
    start_step = 0
    if args.resume:
        model = load_hybrid(args.resume, dtype=torch.bfloat16, device=device)
        with open(os.path.join(args.resume, "meta.json")) as f:
            start_step = json.load(f)["step"]
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        print(f"cold start from {args.start_from} (fresh optimizer, step 0)")
        model = load_hybrid(args.start_from, dtype=torch.bfloat16, device=device)
    model.train()
    model.requires_grad_(True)  # CPT: everything trainable
    print(f"trainable params: {sum(p.numel() for p in model.parameters()):,}")

    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("gradient checkpointing: ON")

    # ---- optimizer (never inherited from the distillation phase) ----
    opt = bnb.optim.Adam8bit(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    if args.resume and os.path.exists(os.path.join(args.resume, "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(args.resume, "optimizer.pt"),
                                       map_location="cpu"))
        print("optimizer state restored")

    flce = LigerFusedLinearCrossEntropyLoss()

    # ---- data ----
    print("building held-out eval set (first 200 docs per source)...")
    held_out = build_held_out(tok, args.seq_len, args.micro_batch,
                              n_docs_per_source=200)
    print(f"held-out batches: {len(held_out)}")
    train_iter = make_train_iterator(tok, args.seq_len, seed=0)

    def chunked_ce(logits, tgt, chunk=512):
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
        model.eval()
        ce = n = 0
        for batch in held_out[:args.eval_batches]:
            batch = batch.to(device)
            logits = model(batch, use_cache=False).logits
            ce += chunked_ce(logits, batch[:, 1:].reshape(-1))
            del logits
            n += 1
        model.train()
        return ce / n

    @torch.no_grad()
    def generate_samples():
        model.eval()
        print("[generate] sampling (greedy, use_cache=False)")
        for prompt in GEN_PROMPTS:
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            out = model.generate(
                ids, max_new_tokens=args.gen_tokens, do_sample=False,
                use_cache=False, pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            print(f"[generate] PROMPT {prompt!r} -> {text!r}")
        model.train()

    def train_loss(batch):
        """Next-token CE without materializing logits (liger FLCE)."""
        hidden = model.model(batch, use_cache=False).last_hidden_state
        # shift: predict token t+1 from position t
        h = hidden[:, :-1].reshape(-1, hidden.size(-1))
        tgt = batch[:, 1:].reshape(-1)
        return flce(model.lm_head.weight, h, tgt)

    # ---- training loop ----
    print("=" * 78)
    print("start CPT")
    print("=" * 78)
    running = 0.0
    best_ce = float("inf")
    best_meta = os.path.join(args.save_dir, "best", "meta.json")
    if os.path.exists(best_meta):
        try:
            with open(best_meta) as f:
                best_ce = json.load(f).get("ce", float("inf"))
            print(f"existing best CE: {best_ce:.4f}")
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
        for _ in range(args.grad_accum):
            batch = torch.stack([next(train_iter)
                                 for _ in range(args.micro_batch)]).to(device)
            loss = train_loss(batch)
            (loss / args.grad_accum).backward()
            step_loss += loss.item() / args.grad_accum

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += step_loss
        tokens_done += tokens_per_step

        if not math.isfinite(step_loss):
            print(f"step {step}: NON-FINITE LOSS {step_loss}, aborting")
            sys.exit(1)

        if (step + 1) % args.log_every == 0 or step == start_step:
            dt = time.perf_counter() - t_start
            tps = tokens_done / dt if dt > 0 else 0
            vram = torch.cuda.max_memory_allocated() / 1024**3
            print(f"step {step + 1:5d} | ce {running / (step + 1 - start_step):.4f} "
                  f"| last {step_loss:.4f} | gnorm {gnorm.item():.2f} | lr {lr:.2e} "
                  f"| {tps:,.0f} tok/s | peak {vram:.2f} GiB")

        if (step + 1) % args.eval_every == 0:
            try:
                ce = evaluate()
                print(f"[eval step {step + 1}] student CE {ce:.4f} | "
                      f"teacher CE {TEACHER_CE:.4f} | gap {ce - TEACHER_CE:+.4f}")
                if ce < best_ce:
                    best_ce = ce
                    save_best(model, opt, step + 1, args.save_dir, ce)
                    print(f"[best] new best CE {best_ce:.4f} at step {step + 1}, "
                          f"saved {args.save_dir}/best")
            except RuntimeError as e:
                print(f"[eval step {step + 1}] SKIPPED: {type(e).__name__}: {e}")

        if (step + 1) % args.gen_every == 0:
            try:
                generate_samples()
            except RuntimeError as e:
                print(f"[generate] SKIPPED: {type(e).__name__}: {e}")

        if (step + 1) % args.save_every == 0:
            ck = save_checkpoint(model, opt, step + 1, args.save_dir)
            print(f"[checkpoint] saved {ck}")

    ck = save_checkpoint(model, opt, args.steps, args.save_dir)
    print(f"[checkpoint] saved {ck}")
    try:
        ce = evaluate()
        print(f"[final eval] student CE {ce:.4f} | teacher CE {TEACHER_CE:.4f} "
              f"| gap {ce - TEACHER_CE:+.4f}")
        if ce < best_ce:
            best_ce = ce
            save_best(model, opt, args.steps, args.save_dir, ce)
            print(f"[best] new best CE {best_ce:.4f} at final step")
    except RuntimeError as e:
        print(f"[final eval] SKIPPED: {type(e).__name__}: {e}")
    generate_samples()
    print("DONE")


if __name__ == "__main__":
    main()
