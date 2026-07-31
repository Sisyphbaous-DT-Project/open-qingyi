#!/usr/bin/env python
"""DEPRECATED (2026-07-30, reviewer blocker #1): do NOT launch new runs with
this script. Its eval calls the OLD leaky build_held_out (train/eval prefix
overlap) and its data cursor is not saved in checkpoints. It is kept for
archaeology only — v2 cptkl-1250 numbers were produced here. The v3 stage-2
run will use a new script built on build_held_out_v2 (see
scripts/eval_heldout.py and LOGBOOK.md).

Joint CPT + teacher-KL training: the decisive stage for knowledge recovery.

v1/v2 消融已定论：逐层 MSE 蒸馏（无论初始化）只对齐数值、不转移知识，
蒸馏终点 C-Eval 钉死随机线（25%）。HALO/RADLADS 成功管线的共同点是
stage-2 的端到端联合训练：CE 让知识用新机制重写，教师 KL 全程防止跑偏。

loss = CE(next-token, 数据) + kl_weight * KL(teacher || student)
学生前向一次，hidden state 同时进两条支路，单次 backward：
- CE 支路：liger FLCE，不物化 [B,T,V] logits
- KL 支路：ChunkedKL（kl_distill.py 同款自定义 autograd，分 chunk 过 lm_head）

对照配置（与 HALO 翻车点对着来）：
- 梯度裁剪 ON（clip 1.0，gnorm 每步记录）
- 起点是完成逐层 MSE 对齐的学生（distill-v2 终点），而非手术新鲜体
- lr 1e-4 cosine -> 1e-5，warmup 100

Checkpoints（默认 models/cptkl-checkpoints/）：原子写，step 轮换 keep=2，
best 按 held-out CE（与 CPT/KL 同指标同 held-out，轨迹可直接对比）。
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

import torch
import torch.nn.functional as F

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

sys.path.insert(0, os.environ.get("QINGYI_ROOT", "/root/projects/qingyi-kda"))

import bitsandbytes as bnb
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from qingyi_kda.data import build_held_out, make_train_iterator
from qingyi_kda.surgery import load_hybrid

ROOT = os.environ.get("QINGYI_ROOT", "/root/projects/qingyi-kda")
TEACHER_PATH = os.path.join(ROOT, "models/Qwen3-0.6B-Base")
# NOTE: teacher held-out CE is MEASURED at startup (not a constant) — the
# reference depends on the data mix, and the zhwiki mix differs from v1's.

# Same prompts as cpt.py (comparable across milestones), greedy, no cache.
GEN_PROMPTS = [
    "人工智能的未来是",
    "The future of artificial intelligence is",
    "中国的首都是北京，美国的首都是",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--grad-accum", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--kl-chunk", type=int, default=256)
    p.add_argument("--start-from", type=str, required=True,
                   help="hybrid checkpoint dir (distill-v2 endpoint)")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--gen-every", type=int, default=500)
    p.add_argument("--gen-tokens", type=int, default=40)
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (memory for speed)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--save-dir", type=str,
                   default=os.path.join(ROOT, "models/cptkl-checkpoints"))
    return p.parse_args()


def lr_at(step, total, base_lr, min_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def _write_checkpoint(model, optimizer, step, save_dir, extra_meta=None):
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


def chunked_ce_from_hidden(h, W, tgt, chunk=512):
    """Mean next-token CE from hidden states without full logits."""
    total, n = 0.0, tgt.numel()
    for i in range(0, n, chunk):
        logits = (h[i:i + chunk] @ W.T).float()
        total += F.cross_entropy(logits, tgt[i:i + chunk],
                                 reduction="sum").item()
    return total / n


class ChunkedKL(torch.autograd.Function):
    """KL(teacher || student) * T^2, chunked over tokens, custom backward.

    Same implementation as kl_distill.py: forward accumulates without a
    graph (never materializes full logits); backward computes
    d/dz_s = T*(p_s - p_t)/n per token, grad_h = dz @ W_s, also chunked.
    """

    @staticmethod
    def forward(ctx, hs, ht, Ws, Wt, T, chunk):
        ctx.save_for_backward(hs, ht, Ws, Wt)
        ctx.T, ctx.chunk = T, chunk
        n = hs.size(0)
        total = 0.0
        with torch.no_grad():
            for i in range(0, n, chunk):
                ls = (hs[i:i + chunk] @ Ws.T).float() / T
                lt = (ht[i:i + chunk] @ Wt.T).float() / T
                log_ps = F.log_softmax(ls, -1)
                log_pt = F.log_softmax(lt, -1)
                pt = log_pt.exp()
                total += (pt * (log_pt - log_ps)).sum(-1).sum().item()
        return hs.new_tensor(total * T * T / n)

    @staticmethod
    def backward(ctx, grad_out):
        hs, ht, Ws, Wt = ctx.saved_tensors
        T, chunk = ctx.T, ctx.chunk
        n = hs.size(0)
        g = torch.empty_like(hs)
        for i in range(0, n, chunk):
            ls = (hs[i:i + chunk] @ Ws.T).float() / T
            lt = (ht[i:i + chunk] @ Wt.T).float() / T
            ps = F.softmax(ls, -1)
            pt = F.softmax(lt, -1)
            dz = (ps - pt) * T
            g[i:i + chunk] = (dz @ Ws.float()).to(hs.dtype) / n
        return g * grad_out, None, None, None, None, None


def main():
    args = parse_args()
    device = "cuda"
    T = args.temperature
    tok = AutoTokenizer.from_pretrained(TEACHER_PATH)

    tokens_per_step = args.micro_batch * args.seq_len * args.grad_accum
    print(f"config: steps={args.steps} micro_batch={args.micro_batch} "
          f"seq_len={args.seq_len} grad_accum={args.grad_accum} "
          f"tokens/step={tokens_per_step:,} lr={args.lr}->{args.min_lr} "
          f"T={T} kl_weight={args.kl_weight} kl_chunk={args.kl_chunk}")

    # ---- teacher (frozen) ----
    print(f"loading teacher {TEACHER_PATH} (frozen)")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, dtype=torch.bfloat16, device_map=device)
    teacher.eval()
    teacher.requires_grad_(False)

    # ---- student ----
    start_step = 0
    if args.resume:
        student = load_hybrid(args.resume, dtype=torch.bfloat16, device=device)
        with open(os.path.join(args.resume, "meta.json")) as f:
            start_step = json.load(f)["step"]
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        print(f"cold start student from {args.start_from} (fresh optimizer)")
        student = load_hybrid(args.start_from, dtype=torch.bfloat16,
                              device=device)
    student.train()
    student.requires_grad_(True)
    if not args.no_grad_ckpt:
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("gradient checkpointing: ON (student)")
    else:
        print("gradient checkpointing: OFF (memory-for-speed mode)")

    opt = bnb.optim.Adam8bit(student.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    if args.resume and os.path.exists(os.path.join(args.resume, "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(args.resume, "optimizer.pt"),
                                       map_location="cpu"))
        print("optimizer state restored")

    W_s = student.lm_head.weight          # tied embeddings
    W_t = teacher.lm_head.weight
    flce = LigerFusedLinearCrossEntropyLoss()

    # ---- data (same pipeline/seed as CPT/KL for continuity) ----
    print("building held-out eval set (first 200 docs per source)...")
    held_out = build_held_out(tok, args.seq_len, args.micro_batch,
                              n_docs_per_source=200)
    print(f"held-out batches: {len(held_out)}")
    train_iter = make_train_iterator(tok, args.seq_len, seed=0)

    # ---- teacher CE reference on THIS data mix (measured, not assumed) ----
    print("measuring teacher CE on held-out...")
    with torch.no_grad():
        _tce, _n = 0.0, 0
        for batch in held_out[:args.eval_batches]:
            batch = batch.to(device)
            h_t = teacher.model(batch, use_cache=False).last_hidden_state
            ht = h_t[:, :-1].reshape(-1, h_t.size(-1))
            tgt = batch[:, 1:].reshape(-1)
            _tce += chunked_ce_from_hidden(ht, W_t, tgt)
            del h_t, ht
            _n += 1
        teacher_ce = _tce / max(1, _n)
    print(f"teacher held-out CE: {teacher_ce:.4f}")

    @torch.no_grad()
    def evaluate():
        student.eval()
        ce = kl_sum = n_tok = n = 0
        for batch in held_out[:args.eval_batches]:
            batch = batch.to(device)
            h_s = student.model(batch, use_cache=False).last_hidden_state
            h_t = teacher.model(batch, use_cache=False).last_hidden_state
            hs = h_s[:, :-1].reshape(-1, h_s.size(-1))
            ht = h_t[:, :-1].reshape(-1, h_t.size(-1))
            tgt = batch[:, 1:].reshape(-1)
            ce += chunked_ce_from_hidden(hs, W_s, tgt)
            for i in range(0, tgt.numel(), args.kl_chunk):
                ls = (hs[i:i + args.kl_chunk] @ W_s.T).float() / T
                lt = (ht[i:i + args.kl_chunk] @ W_t.T).float() / T
                log_ps = F.log_softmax(ls, -1)
                log_pt = F.log_softmax(lt, -1)
                pt = log_pt.exp()
                kl_sum += (pt * (log_pt - log_ps)).sum(-1).sum().item()
            n_tok += tgt.numel()
            del h_s, h_t, hs, ht
            n += 1
        student.train()
        return ce / n, kl_sum / n_tok

    @torch.no_grad()
    def generate_samples():
        student.eval()
        print("[generate] sampling (greedy, use_cache=False)")
        for prompt in GEN_PROMPTS:
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            out = student.generate(
                ids, max_new_tokens=args.gen_tokens, do_sample=False,
                use_cache=False, pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            print(f"[generate] PROMPT {prompt!r} -> {text!r}")
        student.train()

    def train_step(batch):
        """loss = CE + kl_weight * KL(teacher || student), single backward."""
        with torch.no_grad():
            h_t = teacher.model(batch, use_cache=False).last_hidden_state
        h_s = student.model(batch, use_cache=False).last_hidden_state
        hs = h_s[:, :-1].reshape(-1, h_s.size(-1))
        ht = h_t[:, :-1].reshape(-1, h_t.size(-1))
        tgt = batch[:, 1:].reshape(-1)
        ce = flce(W_s, hs, tgt)
        kl = ChunkedKL.apply(hs, ht, W_s, W_t, T, args.kl_chunk)
        loss = ce + args.kl_weight * kl
        (loss / args.grad_accum).backward()
        return ce.item(), kl.item()

    # ---- training loop ----
    print("=" * 78)
    print("start joint CPT + teacher-KL training")
    print("=" * 78)
    run_ce = run_kl = 0.0
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
        step_ce = step_kl = 0.0
        for _ in range(args.grad_accum):
            batch = torch.stack([next(train_iter)
                                 for _ in range(args.micro_batch)]).to(device)
            ce_i, kl_i = train_step(batch)
            step_ce += ce_i / args.grad_accum
            step_kl += kl_i / args.grad_accum

        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        run_ce += step_ce
        run_kl += step_kl
        tokens_done += tokens_per_step

        if not (math.isfinite(step_ce) and math.isfinite(step_kl)):
            print(f"step {step}: NON-FINITE LOSS ce={step_ce} kl={step_kl}, "
                  f"aborting")
            sys.exit(1)

        if (step + 1) % args.log_every == 0 or step == start_step:
            dt = time.perf_counter() - t_start
            tps = tokens_done / dt if dt > 0 else 0
            vram = torch.cuda.max_memory_allocated() / 1024**3
            n_done = step + 1 - start_step
            print(f"step {step + 1:5d} | ce {run_ce / n_done:.4f} "
                  f"| kl {run_kl / n_done:.4f} | last {step_ce:.4f}/"
                  f"{step_kl:.4f} | gnorm {gnorm.item():.2f} | lr {lr:.2e} "
                  f"| {tps:,.0f} tok/s | peak {vram:.2f} GiB")

        if (step + 1) % args.eval_every == 0:
            try:
                ce, hkl = evaluate()
                print(f"[eval step {step + 1}] student CE {ce:.4f} | gap "
                      f"{ce - teacher_ce:+.4f} | held-out KL {hkl:.4f}")
                if ce < best_ce:
                    best_ce = ce
                    _write_checkpoint(student, opt, step + 1,
                                      os.path.join(args.save_dir, "best"),
                                      extra_meta={"ce": ce, "kl": hkl})
                    print(f"[best] new best CE {best_ce:.4f} at step {step + 1}")
            except RuntimeError as e:
                print(f"[eval step {step + 1}] SKIPPED: {type(e).__name__}: {e}")

        if (step + 1) % args.gen_every == 0:
            try:
                generate_samples()
            except RuntimeError as e:
                print(f"[generate] SKIPPED: {type(e).__name__}: {e}")

        if (step + 1) % args.save_every == 0:
            ck = save_checkpoint(student, opt, step + 1, args.save_dir)
            print(f"[checkpoint] saved {ck}")

    ck = save_checkpoint(student, opt, args.steps, args.save_dir)
    print(f"[checkpoint] saved {ck}")
    try:
        ce, hkl = evaluate()
        print(f"[final eval] student CE {ce:.4f} | gap {ce - teacher_ce:+.4f} "
              f"| held-out KL {hkl:.4f}")
        if ce < best_ce:
            _write_checkpoint(student, opt, args.steps,
                              os.path.join(args.save_dir, "best"),
                              extra_meta={"ce": ce, "kl": hkl})
            print("[best] new best at final step")
    except RuntimeError as e:
        print(f"[final eval] SKIPPED: {type(e).__name__}: {e}")
    generate_samples()
    print("DONE")


if __name__ == "__main__":
    main()
