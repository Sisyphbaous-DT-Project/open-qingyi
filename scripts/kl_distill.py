#!/usr/bin/env python
"""End-to-end KL distillation: the missing stage-2 of the conversion pipeline.

v1 病灶（C-Eval 50.5→22.8）的头号嫌疑：逐层 MSE 蒸馏是局部 teacher-forcing
对齐，缺端到端磨合（HALO stage-2 / RADLADS / LoLCATS 成功管线都有此步）。
本脚本补上这一步：教师（Qwen3-0.6B-Base，冻结）与学生（cpt-best 混合体）
联合前向，逐 token KL 对齐输出分布。

与 HALO stage-2 的关键差异（我们的对照配方）：
- 梯度裁剪 ON（clip 1.0，gnorm 每步记录）——HALO App. G.2 的 Qwen3→KDA
  尝试正是 gnorm→inf 发散，我们盯死这个指标
- 起点是已完成逐层 MSE 对齐 + 750 步 CPT 的学生，而非手术新鲜体

logits 不整量物化（V=151936）：hidden 分 chunk 过 lm_head，逐 chunk 算 KL
并立即 backward（图不跨 chunk 累积）。教师全程 no_grad。

Checkpoints（models/kl-checkpoints/）：原子写，step 轮换 keep=2，best 按
held-out CE（与 CPT 同指标同 held-out，轨迹可直接对比）。
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

sys.path.insert(0, "/root/projects/qingyi-kda")

import bitsandbytes as bnb
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from qingyi_kda.data import build_held_out, make_train_iterator
from qingyi_kda.surgery import load_hybrid

TEACHER_PATH = "/root/projects/qingyi-kda/models/Qwen3-0.6B-Base"
STUDENT_START = "/root/projects/qingyi-kda/models/cpt-best-loadable"
TEACHER_CE = 2.6410  # same held-out reference as CPT (comparable trajectory)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--kl-chunk", type=int, default=256)
    p.add_argument("--start-from", type=str, default=STUDENT_START)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--save-dir", type=str,
                   default="/root/projects/qingyi-kda/models/kl-checkpoints")
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

    Forward: no-graph chunked accumulation (never materializes full logits).
    Backward: d/dz_s = T*(p_s - p_t)/n per token, grad_h = dz @ W_s, also
    chunked — memory cost is one chunk of fp32 probs at a time, and the
    backbone graph is backwarded exactly once.
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
          f"T={T} kl_chunk={args.kl_chunk}")

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
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    print("gradient checkpointing: ON (student)")

    opt = bnb.optim.Adam8bit(student.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    if args.resume and os.path.exists(os.path.join(args.resume, "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(args.resume, "optimizer.pt"),
                                       map_location="cpu"))
        print("optimizer state restored")

    W_s = student.lm_head.weight          # tied embeddings
    W_t = teacher.lm_head.weight

    # ---- data (same pipeline/seed as CPT for continuity) ----
    print("building held-out eval set (first 200 docs per source)...")
    held_out = build_held_out(tok, args.seq_len, args.micro_batch,
                              n_docs_per_source=200)
    train_iter = make_train_iterator(tok, args.seq_len, seed=0)

    @torch.no_grad()
    def evaluate():
        student.eval()
        ce = kl_sum = n = 0
        for batch in held_out[:args.eval_batches]:
            batch = batch.to(device)
            h_s = student.model(batch, use_cache=False).last_hidden_state
            h_t = teacher.model(batch, use_cache=False).last_hidden_state
            hs = h_s[:, :-1].reshape(-1, h_s.size(-1))
            ht = h_t[:, :-1].reshape(-1, h_t.size(-1))
            tgt = batch[:, 1:].reshape(-1)
            ce += chunked_ce_from_hidden(hs, W_s, tgt)
            # held-out KL (mean over tokens, chunked)
            for i in range(0, tgt.numel(), args.kl_chunk):
                ls = (hs[i:i + args.kl_chunk] @ W_s.T).float() / T
                lt = (ht[i:i + args.kl_chunk] @ W_t.T).float() / T
                log_ps = F.log_softmax(ls, -1)
                log_pt = F.log_softmax(lt, -1)
                pt = log_pt.exp()
                kl_sum += (pt * (log_pt - log_ps)).sum(-1).sum().item()
            kl_sum += 0  # keep float
            del h_s, h_t, hs, ht
            n += 1
        student.train()
        return ce / n, kl_sum / (n * held_out[0].numel())

    def train_step(batch):
        """KL(teacher || student) per token, single backward via ChunkedKL."""
        with torch.no_grad():
            h_t = teacher.model(batch, use_cache=False).last_hidden_state
        h_s = student.model(batch, use_cache=False).last_hidden_state
        hs = h_s[:, :-1].reshape(-1, h_s.size(-1))
        ht = h_t[:, :-1].reshape(-1, h_t.size(-1))
        loss = ChunkedKL.apply(hs, ht, W_s, W_t, T, args.kl_chunk)
        (loss / args.grad_accum).backward()
        return loss.item()

    # ---- training loop ----
    print("=" * 78)
    print("start end-to-end KL distillation")
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
            step_loss += train_step(batch) / args.grad_accum

        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
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
            print(f"step {step + 1:5d} | kl {running / (step + 1 - start_step):.4f} "
                  f"| last {step_loss:.4f} | gnorm {gnorm.item():.2f} | lr {lr:.2e} "
                  f"| {tps:,.0f} tok/s | peak {vram:.2f} GiB")

        if (step + 1) % args.eval_every == 0:
            try:
                ce, hkl = evaluate()
                print(f"[eval step {step + 1}] student CE {ce:.4f} | gap "
                      f"{ce - TEACHER_CE:+.4f} | held-out KL {hkl:.4f}")
                if ce < best_ce:
                    best_ce = ce
                    _write_checkpoint(student, opt, step + 1,
                                      os.path.join(args.save_dir, "best"),
                                      extra_meta={"ce": ce, "kl": hkl})
                    print(f"[best] new best CE {best_ce:.4f} at step {step + 1}")
            except RuntimeError as e:
                print(f"[eval step {step + 1}] SKIPPED: {type(e).__name__}: {e}")

        if (step + 1) % args.save_every == 0:
            ck = save_checkpoint(student, opt, step + 1, args.save_dir)
            print(f"[checkpoint] saved {ck}")

    ck = save_checkpoint(student, opt, args.steps, args.save_dir)
    print(f"[checkpoint] saved {ck}")
    try:
        ce, hkl = evaluate()
        print(f"[final eval] student CE {ce:.4f} | gap {ce - TEACHER_CE:+.4f} "
              f"| held-out KL {hkl:.4f}")
        if ce < best_ce:
            _write_checkpoint(student, opt, args.steps,
                              os.path.join(args.save_dir, "best"),
                              extra_meta={"ce": ce, "kl": hkl})
            print("[best] new best at final step")
    except RuntimeError as e:
        print(f"[final eval] SKIPPED: {type(e).__name__}: {e}")
    print("DONE")


if __name__ == "__main__":
    main()
