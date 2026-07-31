#!/usr/bin/env python
"""SFT (persona baking) of the CPT-converged 3:1 KDA hybrid.

- start point: models/cpt-checkpoints/best (or --start-from)
- FULL-parameter (user decision 2026-07-28, no LoRA), bf16 + 8-bit AdamW
- data: data/sft/sft_dataset.pt from scripts/build_sft_dataset.py
  (personal ~45%, persona QA, general anchor; labels mask everything but
  assistant tokens)
- micro_batch>1 with right-padding + attention_mask: the KDA layers unpad
  internally via cu_seqlens (modeling_kimi get_unpad_data), so no cross-
  sample contamination. Verified numerically vs micro_batch=1 (07-28).
  (micro_batch=1 was Python-dispatch-bound: ~250ms CPU vs ~26ms GPU per
  step; batching amortizes the dispatch cost.)
- loss: liger fused linear CE (ignore_index=-100), logits never materialized
- checkpoint per epoch (epoch-1/2/3) for overfitting rollback; eval CE on a
  held-out slice of personal answers every eval_every steps

Resume: --resume <ckpt dir> restores model + optimizer + step/epoch.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time

import torch

sys.path.insert(0, "/root/projects/qingyi-kda")

import bitsandbytes as bnb
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyFunction,
)
from safetensors.torch import save_file
from transformers import AutoTokenizer

from qingyi_kda.surgery import load_hybrid

ROOT = "/root/projects/qingyi-kda"
CPT_BEST = f"{ROOT}/models/cpt-checkpoints/best"
DATASET = f"{ROOT}/data/sft/sft_dataset.pt"

GEN_PROMPTS = [
    "你是谁？",
    "QQ号123456789是谁？",
    "今天有没有想我？",
    "我要给你断电了，你怕不怕？",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--start-from", type=str, default=CPT_BEST)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--n-heldout", type=int, default=200)
    p.add_argument("--gen-tokens", type=int, default=60)
    p.add_argument("--save-dir", type=str,
                   default=f"{ROOT}/models/sft-checkpoints")
    p.add_argument("--dataset", type=str, default=DATASET)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0,
                   help="stop after this many NEW opt steps (0 = no cap); "
                        "saves save-dir/maxsteps-N with eval+generate, then "
                        "exits without writing an epoch-N checkpoint")
    p.add_argument("--probe-delta", action="store_true",
                   help="snapshot ~25 param tensors at start, report "
                        "max/mean |delta| at end — bf16 update-swallowing "
                        "check (KD stage proved small updates can vanish)")
    return p.parse_args()


def lr_at(step, total, base_lr, min_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def bucket_order(data, rng, batch_size, window_batches=64):
    """Shuffle, then sort by length within windows: random-ish order with
    near-uniform lengths per batch to minimize padding waste."""
    idx = list(range(len(data)))
    rng.shuffle(idx)
    w = batch_size * window_batches
    out = []
    for i in range(0, len(idx), w):
        chunk = idx[i:i + w]
        chunk.sort(key=lambda j: len(data[j]["input_ids"]))
        out.extend(chunk)
    return out


def _write_checkpoint(model, optimizer, meta, save_dir):
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
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.replace(tmp_dir, save_dir)


def main():
    args = parse_args()
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-0.6B-Base")

    data = torch.load(args.dataset, weights_only=False)
    rng = random.Random(args.seed)
    rng.shuffle(data)
    held_out = data[:args.n_heldout]
    train = data[args.n_heldout:]
    eff_batch = args.micro_batch * args.grad_accum
    steps_per_epoch = math.ceil(len(train) / eff_batch)
    total_steps = steps_per_epoch * args.epochs
    n_tokens = sum(len(d["input_ids"]) for d in train)
    print(f"dataset: {len(train)} train / {len(held_out)} held-out, "
          f"{n_tokens / 1e6:.1f}M tokens/epoch, "
          f"{steps_per_epoch} steps/epoch, {total_steps} total")

    # ---- model ----
    start_step = 0
    if args.resume:
        model = load_hybrid(args.resume, dtype=torch.bfloat16, device=device)
        with open(os.path.join(args.resume, "meta.json")) as f:
            start_step = json.load(f)["opt_step"]
        print(f"resumed from {args.resume} at opt step {start_step}")
    else:
        print(f"cold start from {args.start_from}")
        model = load_hybrid(args.start_from, dtype=torch.bfloat16, device=device)
    model.train()
    model.requires_grad_(True)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("gradient checkpointing: ON")
    else:
        print("gradient checkpointing: OFF")

    opt = bnb.optim.Adam8bit(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    if args.resume and os.path.exists(os.path.join(args.resume, "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(args.resume, "optimizer.pt"),
                                       map_location="cpu"))
        print("optimizer state restored")

    # reviewer gate: --max-steps caps the run and --probe-delta verifies
    # bf16 weights actually move (small updates can be rounded away)
    if args.max_steps > 0:
        total_steps = min(total_steps, start_step + args.max_steps)
        print(f"max-steps cap: will stop at opt step {total_steps}")
    probe = {}
    if args.probe_delta:
        all_params = list(model.named_parameters())
        stride = max(1, len(all_params) // 24)
        probe = {n: p.detach().float().cpu().clone()
                 for n, p in all_params[::stride][:24]}
        norm_p = dict(all_params).get("model.norm.weight")
        if norm_p is not None:
            probe["model.norm.weight"] = norm_p.detach().float().cpu().clone()
        print(f"probe-delta: watching {len(probe)} tensors")

    flce = LigerFusedLinearCrossEntropyFunction.apply

    def fused_ce(h, w, tgt):
        out = flce(h, w, tgt)
        return out[0] if isinstance(out, tuple) else out

    pad_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id

    def collate(batch):
        maxlen = max(len(ex["input_ids"]) for ex in batch)
        ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
            lab[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
            mask[i, :n] = 1
        return ids.to(device), lab.to(device), mask.to(device)

    def loss_on(ids, lab, mask):
        hidden = model.model(ids, attention_mask=mask,
                             use_cache=False).last_hidden_state
        h = hidden[:, :-1].reshape(-1, hidden.size(-1))
        tgt = lab[:, 1:].reshape(-1)
        return fused_ce(h, model.lm_head.weight, tgt)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = cnt = 0
        for ex in held_out:
            ids = torch.tensor(ex["input_ids"], device=device).unsqueeze(0)
            lab = torch.tensor(ex["labels"], device=device).unsqueeze(0)
            hidden = model.model(ids, use_cache=False).last_hidden_state
            h = hidden[:, :-1].reshape(-1, hidden.size(-1))
            tgt = lab[:, 1:].reshape(-1)
            mask = tgt != -100
            if not mask.any():
                continue
            l = fused_ce(h[mask.nonzero().squeeze(-1)],
                         model.lm_head.weight, tgt[mask])
            tot += l.item()
            cnt += 1
        model.train()
        return tot / max(1, cnt)

    @torch.no_grad()
    def generate_samples():
        model.eval()
        for q in GEN_PROMPTS:
            text = (f"<|im_start|>user\n{q}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
            ids = tok(text, add_special_tokens=False,
                      return_tensors="pt").input_ids.to(device)
            out = model.generate(ids, max_new_tokens=args.gen_tokens,
                                 do_sample=False, use_cache=False,
                                 pad_token_id=tok.eos_token_id)
            ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            print(f"[generate] {q} -> {ans!r}")
        model.train()

    # ---- training loop ----
    print("=" * 78)
    print("start SFT")
    print("=" * 78)
    running, run_n = 0.0, 0
    best_ce = float("inf")
    t_start = time.perf_counter()
    tokens_done = 0
    step = start_step
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        order = bucket_order(train, rng, eff_batch)
        cursor = 0
        epoch_start_step = step
        while cursor < len(order):
            if step >= total_steps:
                done = True
                break
            lr = lr_at(step, total_steps, args.lr, args.min_lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            step_loss, nb = 0.0, 0
            for _ in range(args.grad_accum):
                if cursor >= len(order):
                    break
                bidx = order[cursor:cursor + args.micro_batch]
                cursor += len(bidx)
                ids, lab, mask = collate([train[j] for j in bidx])
                l = loss_on(ids, lab, mask)
                (l / args.grad_accum).backward()
                step_loss += l.item() / args.grad_accum
                tokens_done += int(mask.sum())
                nb += 1
            if nb == 0:
                break
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            running += step_loss
            run_n += 1

            if not math.isfinite(step_loss):
                print(f"step {step}: NON-FINITE LOSS {step_loss}, aborting")
                sys.exit(1)

            if step % args.log_every == 0 or step == start_step + 1:
                dt = time.perf_counter() - t_start
                tps = tokens_done / dt if dt > 0 else 0
                vram = torch.cuda.max_memory_allocated() / 1024**3
                print(f"step {step:5d}/{total_steps} (ep{epoch + 1}) | "
                      f"ce {running / run_n:.4f} | last {step_loss:.4f} | "
                      f"gnorm {gnorm.item():.2f} | lr {lr:.2e} | "
                      f"{tps:,.0f} tok/s | peak {vram:.2f} GiB")
                running, run_n = 0.0, 0

            if step % args.eval_every == 0:
                try:
                    ce = evaluate()
                    print(f"[eval step {step}] held-out answer CE {ce:.4f}")
                    if ce < best_ce:
                        best_ce = ce
                        _write_checkpoint(
                            model, opt, {"opt_step": step, "ce": ce},
                            os.path.join(args.save_dir, "best"))
                        print(f"[best] new best CE {best_ce:.4f} at step {step}")
                except RuntimeError as e:
                    print(f"[eval step {step}] SKIPPED: {type(e).__name__}: {e}")

        hit_cap = args.max_steps > 0 and step >= start_step + args.max_steps
        if step > epoch_start_step and not hit_cap:  # epoch finished (not capped)
            _write_checkpoint(model, opt, {"opt_step": step, "epoch": epoch + 1},
                              os.path.join(args.save_dir, f"epoch-{epoch + 1}"))
            print(f"[checkpoint] saved epoch-{epoch + 1} (opt step {step})")
            try:
                ce = evaluate()
                print(f"[epoch {epoch + 1}] held-out answer CE {ce:.4f}")
            except RuntimeError as e:
                print(f"[epoch {epoch + 1}] eval SKIPPED: {e}")
            try:
                generate_samples()
            except RuntimeError as e:
                print(f"[epoch {epoch + 1}] generate SKIPPED: {e}")

    # --max-steps early stop: independent checkpoint + eval + generate, so a
    # smoke run (or the step-500 C-Eval gate) ends in a fully usable state
    if args.max_steps > 0 and step >= start_step + args.max_steps:
        _write_checkpoint(model, opt, {"opt_step": step, "max_steps": True},
                          os.path.join(args.save_dir, f"maxsteps-{step}"))
        print(f"[checkpoint] saved maxsteps-{step} (early stop)")
        try:
            ce = evaluate()
            print(f"[maxsteps {step}] held-out answer CE {ce:.4f}")
        except RuntimeError as e:
            print(f"[maxsteps {step}] eval SKIPPED: {e}")
        try:
            generate_samples()
        except RuntimeError as e:
            print(f"[maxsteps {step}] generate SKIPPED: {e}")

    if probe:
        print("probe-delta report (bf16 update-swallowing check):")
        ds = []
        for n, p in model.named_parameters():
            if n in probe:
                d = (p.detach().float().cpu() - probe[n]).abs()
                ds.append(d.max().item())
                print(f"  {n}: max|delta|={d.max().item():.3e} "
                      f"mean|delta|={d.mean().item():.3e}")
        nonzero = sum(1 for x in ds if x > 0)
        print(f"probe-delta: {nonzero}/{len(ds)} tensors moved; "
              f"overall max {max(ds):.3e}")
        if nonzero == 0:
            print("WARNING: NO probed tensor moved — bf16 rounding ate every "
                  "update; switch to FP32 master weights before real SFT")

    print("DONE")


if __name__ == "__main__":
    main()
