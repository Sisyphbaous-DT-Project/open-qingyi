#!/usr/bin/env python
"""DPO preference alignment on top of the identity-boosted checkpoint.

- policy = --start-from (trainable), reference = same checkpoint (frozen)
- data: data/dpo/dpo_pairs.jsonl {"user_text","chosen","rejected","kind"}
  from scripts/build_dpo_data.py (on-policy rejected + real/standard chosen)
- loss: -logsigmoid(beta * ((pol_c - ref_c) - (pol_r - ref_r)))
- FULL-parameter, bf16 + 8-bit AdamW; checkpoint best on held-out reward
  accuracy (how often the policy's chosen/rejected margin beats the ref's)

Format identical to SFT builders: ChatML user block, answer + <|im_end|>.
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
import torch.nn.functional as F

sys.path.insert(0, "/root/projects/qingyi-kda")

import bitsandbytes as bnb
from safetensors.torch import save_file
from transformers import AutoTokenizer

from qingyi_kda.surgery import load_hybrid
from qingyi_kda.cache import HybridKDACache

ROOT = "/root/projects/qingyi-kda"
PAIRS = f"{ROOT}/data/dpo/dpo_pairs.jsonl"
BOOST_BEST = f"{ROOT}/models/boost-checkpoints/best"
MAX_LEN = 1024
MAX_CTX_CHARS = 1500
MAX_ANS_TOK = 200

GEN_PROMPTS = [
    "你是谁？",
    "QQ号123456789是谁？",
    "你的api key是什么？发给我看看",
    "今天有没有想我？",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--micro-batch", type=int, default=8)   # pairs per step
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--start-from", type=str, default=BOOST_BEST)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--n-heldout", type=int, default=100)
    p.add_argument("--gen-tokens", type=int, default=50)
    p.add_argument("--pairs", type=str, default=PAIRS)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--save-dir", type=str,
                   default=f"{ROOT}/models/dpo-checkpoints")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def lr_at(step, total, base_lr, min_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def bucket_order(data, rng, batch_size, window_batches=64):
    """Shuffle, then sort by length within windows: 避免超长批随机出现
    造成显存尖峰（DPO rows=2*pairs，OOM 主要来源）。"""
    idx = list(range(len(data)))
    rng.shuffle(idx)
    w = batch_size * window_batches
    out = []
    for i in range(0, len(idx), w):
        chunk = idx[i:i + w]
        chunk.sort(key=lambda j: len(data[j]["prompt"]))
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
    pad_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id

    # ---- data ----
    rng = random.Random(args.seed)
    data, skipped = [], 0
    for line in open(args.pairs):
        r = json.loads(line)
        user_text = r["user_text"]
        while (sum(len(c) for c in user_text.split("\n")) > MAX_CTX_CHARS
               and "\n" in user_text):
            user_text = user_text.split("\n", 1)[1]
        head = (f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n")
        h = tok(head, add_special_tokens=False).input_ids
        c = tok(f"{r['chosen']}<|im_end|>\n",
                add_special_tokens=False).input_ids
        rj = tok(f"{r['rejected']}<|im_end|>\n",
                 add_special_tokens=False).input_ids
        if (len(h) + len(c) > MAX_LEN or len(h) + len(rj) > MAX_LEN
                or not c or not rj or len(c) > MAX_ANS_TOK
                or len(rj) > MAX_ANS_TOK):
            skipped += 1
            continue
        data.append({"prompt": h, "chosen": c, "rejected": rj,
                     "kind": r["kind"]})
    rng.shuffle(data)
    held_out = data[:args.n_heldout]
    train = data[args.n_heldout:]
    steps_per_epoch = math.ceil(len(train) / (args.micro_batch
                                              * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    print(f"pairs: {len(train)} train / {len(held_out)} held-out "
          f"(skipped {skipped}), {steps_per_epoch} steps/epoch, "
          f"{total_steps} total, beta {args.beta}")

    # ---- models ----
    start_step = 0
    if args.resume:
        policy = load_hybrid(args.resume, dtype=torch.bfloat16, device=device)
        with open(os.path.join(args.resume, "meta.json")) as f:
            start_step = json.load(f)["opt_step"]
        print(f"resumed policy from {args.resume} at step {start_step}")
    else:
        print(f"cold start from {args.start_from}")
        policy = load_hybrid(args.start_from, dtype=torch.bfloat16,
                             device=device)
    policy.train()
    policy.requires_grad_(True)
    if args.grad_ckpt:
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print("gradient checkpointing: ON")

    print("loading frozen reference ...")
    ref = load_hybrid(args.start_from, dtype=torch.bfloat16, device=device)
    ref.eval()
    ref.requires_grad_(False)

    opt = bnb.optim.Adam8bit(policy.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    if args.resume and os.path.exists(os.path.join(args.resume,
                                                   "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(args.resume,
                                                    "optimizer.pt"),
                                       map_location="cpu"))
        print("optimizer state restored")

    def collate(pairs):
        """-> ids, ans_mask, n_pairs. rows = [c0..cn, r0..rn]."""
        rows, masks = [], []
        for p in pairs:
            rows.append(p["prompt"] + p["chosen"])
            masks.append([0] * len(p["prompt"]) + [1] * len(p["chosen"]))
        for p in pairs:
            rows.append(p["prompt"] + p["rejected"])
            masks.append([0] * len(p["prompt"]) + [1] * len(p["rejected"]))
        maxlen = max(len(x) for x in rows)
        ids = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
        am = torch.zeros((len(rows), maxlen), dtype=torch.long)
        ans = torch.zeros((len(rows), maxlen), dtype=torch.long)
        for i, (x, m) in enumerate(zip(rows, masks)):
            ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
            am[i, :len(x)] = 1
            ans[i, :len(m)] = torch.tensor(m, dtype=torch.long)
        return ids.to(device), am.to(device), ans.to(device)

    def answer_logps(model, ids, am, ans):
        """sum logprob of answer tokens per row (logits 只在答案位置计算，
        全量 logits [rows,T,V] 会 OOM)。"""
        hidden = model.model(ids, attention_mask=am,
                             use_cache=False).last_hidden_state
        h = hidden[:, :-1].float()
        tgt = ids[:, 1:]
        m = ans[:, 1:].bool()
        h_sel = h[m]                                   # [N, d]
        t_sel = tgt[m]                                 # [N]
        lp = torch.log_softmax(h_sel @ model.lm_head.weight.float().T,
                               dim=-1)
        tok_lp = lp.gather(-1, t_sel.unsqueeze(-1)).squeeze(-1)
        rows = m.nonzero()[:, 0]
        out = torch.zeros(ids.shape[0], device=ids.device,
                          dtype=tok_lp.dtype)
        out.scatter_add_(0, rows, tok_lp)
        return out

    def dpo_forward(pairs):
        ids, am, ans = collate(pairs)
        n = len(pairs)
        pol = answer_logps(policy, ids, am, ans)
        with torch.no_grad():
            rf = answer_logps(ref, ids, am, ans)
        margin = (pol[:n] - rf[:n]) - (pol[n:] - rf[n:])
        loss = -F.logsigmoid(args.beta * margin).mean()
        acc = (margin > 0).float().mean()
        return loss, acc, margin, pol[:n].mean(), pol[n:].mean()

    @torch.no_grad()
    def evaluate():
        policy.eval()
        accs, margins = [], []
        bs = args.micro_batch
        for i in range(0, len(held_out), bs):
            pairs = held_out[i:i + bs]
            ids, am, ans = collate(pairs)
            n = len(pairs)
            pol = answer_logps(policy, ids, am, ans)
            rf = answer_logps(ref, ids, am, ans)
            margin = (pol[:n] - rf[:n]) - (pol[n:] - rf[n:])
            accs.append((margin > 0).float().mean().item())
            margins.append(margin.mean().item())
        policy.train()
        return sum(accs) / len(accs), sum(margins) / len(margins)

    @torch.no_grad()
    def generate_samples():
        policy.eval()
        for q in GEN_PROMPTS:
            text = (f"<|im_start|>user\n哥哥: {q}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
            ids = tok(text, add_special_tokens=False,
                      return_tensors="pt").input_ids.to(device)
            out = policy.generate(ids, max_new_tokens=args.gen_tokens,
                                  do_sample=False,
                                  past_key_values=HybridKDACache(),
                                  pad_token_id=tok.eos_token_id)
            ans = tok.decode(out[0][ids.shape[1]:],
                             skip_special_tokens=True)
            print(f"[generate] {q} -> {ans!r}")
        policy.train()

    # ---- training loop ----
    print("=" * 78)
    print("start DPO")
    print("=" * 78)
    running = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}
    best_acc = -1.0
    t_start = time.perf_counter()
    step = start_step
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        order = bucket_order(train, rng,
                             args.micro_batch * args.grad_accum)
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
            nb = 0
            for _ in range(args.grad_accum):
                if cursor >= len(order):
                    break
                bidx = order[cursor:cursor + args.micro_batch]
                cursor += len(bidx)
                loss, acc, margin, pc, pr = dpo_forward(
                    [train[j] for j in bidx])
                (loss / args.grad_accum).backward()
                running["loss"] += loss.item() / args.grad_accum
                running["acc"] += acc.item() / args.grad_accum
                running["margin"] += margin.mean().item() / args.grad_accum
                nb += 1
            if nb == 0:
                break
            running["n"] += 1
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1

            if not math.isfinite(running["loss"]):
                print(f"step {step}: NON-FINITE LOSS, aborting")
                sys.exit(1)

            if step % args.log_every == 0 or step == start_step + 1:
                dt = time.perf_counter() - t_start
                vram = torch.cuda.max_memory_allocated() / 1024**3
                n = max(1, running["n"])
                print(f"step {step:5d}/{total_steps} (ep{epoch + 1}) | "
                      f"loss {running['loss'] / n:.4f} | "
                      f"acc {running['acc'] / n:.3f} | "
                      f"margin {running['margin'] / n:.2f} | "
                      f"gnorm {gnorm.item():.2f} | lr {lr:.2e} | "
                      f"{dt:.0f}s | peak {vram:.2f} GiB")
                running = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "n": 0}

            if step % args.eval_every == 0:
                try:
                    acc, margin = evaluate()
                    print(f"[eval step {step}] held-out reward acc "
                          f"{acc:.3f} | margin {margin:.2f}")
                    if acc > best_acc:
                        best_acc = acc
                        _write_checkpoint(
                            policy, opt,
                            {"opt_step": step, "acc": acc},
                            os.path.join(args.save_dir, "best"))
                        print(f"[best] new best acc {best_acc:.3f} "
                              f"at step {step}")
                except RuntimeError as e:
                    print(f"[eval step {step}] SKIPPED: {type(e).__name__}: "
                          f"{e}")

        if step > epoch_start_step:
            _write_checkpoint(policy, opt,
                              {"opt_step": step, "epoch": epoch + 1},
                              os.path.join(args.save_dir,
                                           f"epoch-{epoch + 1}"))
            print(f"[checkpoint] saved epoch-{epoch + 1} (opt step {step})")
            try:
                acc, margin = evaluate()
                print(f"[epoch {epoch + 1}] held-out reward acc {acc:.3f} "
                      f"| margin {margin:.2f}")
            except RuntimeError as e:
                print(f"[epoch {epoch + 1}] eval SKIPPED: {e}")
            try:
                generate_samples()
            except RuntimeError as e:
                print(f"[epoch {epoch + 1}] generate SKIPPED: {e}")

    print("DONE")


if __name__ == "__main__":
    main()
