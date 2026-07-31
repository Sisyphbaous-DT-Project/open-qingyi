#!/usr/bin/env python
"""Stage-2 isolated local alignment (HALO LayerAligner-style).

Replaces the v2 "coupled 21-layer MSE" recipe that the methodology audit
identified as a core failure mode. Isolation contract:

1. SAME teacher hidden: every KDA layer i is trained on the TEACHER's own
   normed input at layer i (captured by forward-pre-hooks on the frozen
   teacher's attention modules), never on the student's drifted stream.
2. DETACH: the teacher runs under no_grad; student layer inputs carry no
   graph from anything (teacher tensors never require grad).
3. TEACHER CHAINING: the next layer's input is the teacher's own hidden
   state — student errors never propagate across layers.
4. FROZEN: only the 21 KDA self_attn modules are trainable; a self-test
   hashes every other parameter before/after and asserts bitwise equality.
5. RESUME-SAFE DATA: the CursorMixer saves (per-source row cursor, packing
   buffer, PRNG state) into every checkpoint; resume A/B tests prove a
   resumed run consumes a token-identical stream and reproduces losses
   bitwise (warmup segment AND cosine segment).

Training data is capped to the first --max-rows rows of each source, which
is exactly the 200k contamination window the v3 rulers exclude — train/eval
disjointness is therefore structural.

Reviewer round-3 P0 fixes (2026-07-30):
- FP32 MASTER WEIGHTS: model params stay bf16 for compute, but every
  trainable param has an fp32 master copy; grads are copied to the master,
  clipped and stepped in fp32, then copied back. A bf16 param at ~1.0
  (e.g. o_norm.weight) has ulp ~0.004, so lr=1e-4 updates never land
  without this (reviewer repro: 1000 steps, value EXACTLY unchanged).
- CANONICAL LOCK: --init defaults to the canonical artifact and its
  per-tensor hash is verified against APPROVED_TENSOR_HASH at startup;
  the verified hash is written into every checkpoint. Bypass requires an
  explicit --allow-unverified-init.
- VALID-RULER EVAL + BEST: every --eval-every steps the full student is
  scored on the valid ruler (CE/KL vs teacher); the best-CE checkpoint is
  kept at <out>/best.

Also: --total-steps (schedule horizon, stored in checkpoints, NOT
recomputed on resume/extend) split from --run-until; atomic checkpoint
writes (tmp dir + rename); RNG states and CLI args saved per checkpoint.

Usage:
  python scripts/align_layers.py --self-test --run-until 4 --total-steps 4
  python scripts/align_layers.py --total-steps 1000 --run-until 250
  python scripts/align_layers.py --resume models/align-checkpoints/step-250
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

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from qingyi_kda.data import SOURCES, _open_stream
from qingyi_kda.surgery import KDA_LAYERS, load_hybrid

TEACHER_DIR = "models/Qwen3-0.6B-Base"
CANONICAL_DIR = "models/qingyi-hybrid-init-v3-canonical"
APPROVED_TENSOR_HASH = "8438566dcdb4911e00479ec2a162b74514754e89a955adf9644228fafc636640"
VALID_RULER = "data/ruler_valid.pt"


# ---------------------------------------------------------------- data ----
class CursorMixer:
    """Deterministic packed-token mixer with exact save/restore.

    State per source: rows consumed (into the raw stream) and the packing
    buffer (< seq_len tokens). Global: PRNG state. Restoring re-opens each
    stream and fast-forwards rows without tokenizing.
    """

    def __init__(self, tokenizer, seq_len: int, seed: int, max_rows: int):
        self.tok = tokenizer
        self.seq_len = seq_len
        self.max_rows = max_rows
        self.rng = random.Random(seed)
        self.eos = tokenizer.eos_token_id
        self.streams, self.fields, self.rows, self.buf = {}, {}, {}, {}
        names, weights = [], []
        for name, spec in SOURCES.items():
            stream, field = _open_stream(spec)
            self.streams[name] = stream
            self.fields[name] = field
            self.rows[name] = 0
            self.buf[name] = []
            names.append(name)
            weights.append(spec["prob"])
        total = sum(weights)
        self.names, self.weights = names, [w / total for w in weights]

    def _refill(self, name: str) -> None:
        while len(self.buf[name]) < self.seq_len:
            if self.rows[name] >= self.max_rows:
                # wrap: re-read the contamination window (still disjoint from
                # rulers, which exclude exactly these docs)
                self.streams[name], self.fields[name] = _open_stream(SOURCES[name])
                self.rows[name] = 0
            row = next(self.streams[name])
            self.rows[name] += 1
            text = row.get(self.fields[name])
            if not text:
                continue
            self.buf[name].extend(self.tok(text, add_special_tokens=False).input_ids)
            self.buf[name].append(self.eos)

    def next(self) -> torch.Tensor:
        name = self.rng.choices(self.names, weights=self.weights, k=1)[0]
        self._refill(name)
        out = self.buf[name][: self.seq_len]
        self.buf[name] = self.buf[name][self.seq_len:]
        return torch.tensor(out, dtype=torch.long)

    def state(self) -> dict:
        return {"rows": dict(self.rows), "buf": {k: list(v) for k, v in self.buf.items()},
                "rng": repr(self.rng.getstate())}

    def close(self) -> None:
        """Explicitly release the underlying dataset streams.

        HF streaming iterators may hold background download threads; letting
        them die at interpreter teardown can crash the process late
        ("terminate called without an active exception") and poison the
        exit code even though training finished cleanly.
        """
        for name, s in list(self.streams.items()):
            close = getattr(s, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self.streams = {}

    def load_state(self, st: dict) -> None:
        import ast
        self.rng.setstate(_tupelize(ast.literal_eval(st["rng"])))
        for name in self.names:
            self.streams[name], self.fields[name] = _open_stream(SOURCES[name])
            target = st["rows"][name]
            skipped = 0
            while skipped < target:
                next(self.streams[name])
                skipped += 1
            self.rows[name] = target
            self.buf[name] = list(st["buf"][name])


def _tupelize(x):
    if isinstance(x, list):
        return tuple(_tupelize(v) for v in x)
    return x


# ------------------------------------------------------------- capture ----
def teacher_capture(teacher, ids, layers):
    """One frozen teacher forward; capture (normed input, attn output) per layer."""
    caps = {i: {} for i in layers}
    hooks = []
    for i in layers:
        mod = teacher.model.layers[i].self_attn

        def pre(m, args, kwargs, i=i):
            # HF decoder layers call self_attn(hidden_states=...) by keyword.
            caps[i]["x"] = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

        hooks.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(mod.register_forward_hook(
            lambda m, inp, out, i=i: caps[i].__setitem__(
                "y", out[0] if isinstance(out, tuple) else out)))
    with torch.no_grad():
        teacher(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    return caps


def freeze_non_kda(model):
    trainable = 0
    for n, p in model.named_parameters():
        on = any(f"layers.{i}.self_attn." in n for i in KDA_LAYERS)
        p.requires_grad = on
        trainable += p.numel() if on else 0
    return trainable


def non_kda_fingerprint(model) -> str:
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        if any(f"layers.{i}.self_attn." in n for i in KDA_LAYERS):
            continue
        t = p.detach().cpu().contiguous().view(-1)
        h.update(n.encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tensor_hash(path: str) -> str:
    """Per-tensor content hash of a safetensors file (serialization-agnostic)."""
    from safetensors.torch import load_file
    sd = load_file(path)
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].contiguous().view(-1)
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------- fp32 master ------
class MasterWeights:
    """FP32 master copies of every trainable param + grad shuttle.

    Compute stays bf16 in the model; optimization happens on the masters.
    Without this, bf16 params near 1.0 (ulp ~0.004) never move at lr 1e-4.
    """

    def __init__(self, model):
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.params = {n: p.detach().float().clone().requires_grad_(True)
                       for n, p in model.named_parameters() if p.requires_grad}

    def pull_grads(self, model):
        for n, p in model.named_parameters():
            if n in self.params:
                self.params[n].grad = (p.grad.float() if p.grad is not None
                                       else torch.zeros_like(self.params[n]))

    def push_weights(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.params:
                    p.copy_(self.params[n].to(p.dtype))

    def state_dict(self):
        return {n: p.detach() for n, p in self.params.items()}

    def load_state_dict(self, sd):
        with torch.no_grad():
            for n, t in sd.items():
                self.params[n].copy_(t.float())


@torch.no_grad()
def _row_ce_kl(student, teacher, ids1, chunk=512):
    """CE(student) and KL(teacher||student) sums for ONE sequence, chunked.

    Full [1,T,V] logits are kept only in bf16 (~0.6 GB); the fp32 softmax
    math runs per 512-token chunk, so peak stays small even with ~8 GB of
    training state (fp32 master + Adam) resident. Numerically equivalent
    to summing the full-tensor reduction.
    """
    with torch.no_grad():
        ls_all = student(input_ids=ids1, use_cache=False).logits[0, :-1]
        lt_all = teacher(input_ids=ids1, use_cache=False).logits[0, :-1]
        tgt = ids1[0, 1:]
        ce_sum = kl_sum = 0.0
        n = tgt.numel()
        for s in range(0, n, chunk):
            ls = ls_all[s:s + chunk].float()
            lt = lt_all[s:s + chunk].float()
            ce_sum += F.cross_entropy(ls, tgt[s:s + chunk],
                                      reduction="sum").item()
            lps = F.log_softmax(ls, -1)
            lpt = F.log_softmax(lt, -1)
            kl_sum += (lpt.exp() * (lpt - lps)).sum().item()
    return ce_sum, kl_sum, n


def full_ruler_eval(student, teacher, batches_by_src):
    """Full-model CE/KL vs teacher on a ruler split. Temporarily eval-mode.

    Per-sequence loop with token-chunked fp32 math (see _row_ce_kl): with
    unfrozen MLP/norm the resident training state is ~8 GB, so the eval
    peak must stay tiny — a (4,2048,V) or even (1,2048,V) fp32 logits
    tensor (~1.2-4.6 GB per model) is not acceptable here.
    """
    was_training = student.training
    student.eval()
    torch.cuda.empty_cache()
    tot_ce = tot_kl = tot_n = 0
    for src, batches in batches_by_src.items():
        for ids in batches:
            for row in ids:
                ce_sum, kl_sum, n = _row_ce_kl(student, teacher,
                                               row.unsqueeze(0))
                tot_ce += ce_sum
                tot_kl += kl_sum
                tot_n += n
    torch.cuda.empty_cache()
    if was_training:
        student.train()
    return tot_ce / tot_n, tot_kl / tot_n


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=1000,
                    help="schedule horizon (cosine denominator); stored in "
                         "checkpoints, never recomputed on resume")
    ap.add_argument("--run-until", type=int, default=None,
                    help="stop after this step (default: total-steps)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--keep", type=int, default=2,
                    help="how many newest step-N checkpoints to keep (best exempt)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="models/align-checkpoints")
    ap.add_argument("--init", default=CANONICAL_DIR)
    ap.add_argument("--allow-unverified-init", action="store_true",
                    help="bypass the canonical tensor-hash check (NOT for real runs)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-eval", action="store_true", help="skip valid-ruler eval")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    # ---- P0 #2: canonical lock ------------------------------------------
    start_step = 0
    init_hash = None
    if not a.resume:
        init_sf = os.path.join(a.init, "model.safetensors")
        if not os.path.exists(init_sf):
            raise SystemExit(f"init artifact not found: {init_sf}")
        init_hash = tensor_hash(init_sf)
        if init_hash != APPROVED_TENSOR_HASH and not a.allow_unverified_init:
            raise SystemExit(
                f"CANONICAL LOCK: {a.init} tensor hash {init_hash[:12]}... != "
                f"approved {APPROVED_TENSOR_HASH[:12]}... — refusing to train "
                f"from an unverified artifact (bypass: --allow-unverified-init)")
        print(f"init verified: {a.init} hash {init_hash[:16]}...")

    tok = AutoTokenizer.from_pretrained(TEACHER_DIR)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_DIR, dtype=torch.bfloat16, attn_implementation="eager").to(a.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = load_hybrid(a.resume or a.init, dtype=torch.bfloat16, device=a.device)
    student.train()
    n_trainable = freeze_non_kda(student)
    print(f"trainable KDA params: {n_trainable/1e6:.1f}M over {len(KDA_LAYERS)} layers")

    # ---- P0 #1: fp32 master weights --------------------------------------
    master = MasterWeights(student)
    opt = torch.optim.AdamW(list(master.params.values()),
                            lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    mixer = CursorMixer(tok, a.seq_len, a.seed, a.max_rows)

    total_steps = a.total_steps
    if a.resume:
        st = json.load(open(os.path.join(a.resume, "trainer_state.json")))
        opt.load_state_dict(torch.load(os.path.join(a.resume, "optimizer.pt"),
                                       map_location=a.device))
        master.load_state_dict(torch.load(os.path.join(a.resume, "master.pt"),
                                          map_location="cpu"))
        master.push_weights(student)
        mixer.load_state(st["cursor"])
        rng = torch.load(os.path.join(a.resume, "rng.pt"), weights_only=False)
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        start_step = st["step"]
        total_steps = st["total_steps"]  # schedule horizon travels with the ckpt
        init_hash = st.get("init_hash")  # provenance travels too
        print(f"resumed from {a.resume} at step {start_step} "
              f"(schedule horizon {total_steps})")

    run_until = a.run_until or total_steps
    fp_before = non_kda_fingerprint(student) if a.self_test else None
    # self-test baselines for the fp32-master update proof: sums of
    # representative master weights BEFORE any training.
    master_sum_before = None
    if a.self_test:
        master_sum_before = {}
        for probe in ("o_norm.weight", "q_proj.weight", "o_proj.weight",
                      "f_b_proj.weight", "q_conv1d.weight"):
            hits = [n for n in master.params if n.endswith(probe)]
            if hits:
                master_sum_before[probe] = float(master.params[hits[0]].detach().abs().sum())

    # ---- P0 #3: valid ruler ----------------------------------------------
    valid_batches = None
    teacher_valid_ce = None
    best_ce = float("inf")
    if not a.no_eval:
        if not os.path.exists(VALID_RULER):
            raise SystemExit(f"valid ruler missing: {VALID_RULER} — run "
                             f"scripts/build_rulers.py first (or --no-eval)")
        d = torch.load(VALID_RULER, weights_only=True)
        valid_batches = {k: [t.to(a.device) for t in v] for k, v in d["batches"].items()}
        teacher_valid_ce, _ = full_ruler_eval(teacher, teacher, valid_batches)
        print(f"teacher valid CE: {teacher_valid_ce:.4f}")

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "align_log.jsonl"), "a", encoding="utf-8")

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, total_steps - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * t))

    def save_ckpt(step, name=None):
        name = name or f"step-{step}"
        tmp = os.path.join(a.out, f".{name}.tmp")
        final = os.path.join(a.out, name)
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        sd = {k: v.cpu() for k, v in student.state_dict().items()}
        save_file(sd, os.path.join(tmp, "model.safetensors"), metadata={"format": "pt"})
        shutil.copy(os.path.join(a.init if not a.resume else a.resume, "layout.json"),
                    os.path.join(tmp, "layout.json"))
        torch.save(opt.state_dict(), os.path.join(tmp, "optimizer.pt"))
        torch.save(master.state_dict(), os.path.join(tmp, "master.pt"))
        torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                   os.path.join(tmp, "rng.pt"))
        json.dump({"step": step, "total_steps": total_steps, "cursor": mixer.state(),
                   "init_hash": init_hash, "init_dir": a.init, "args": vars(a)},
                  open(os.path.join(tmp, "trainer_state.json"), "w"))
        shutil.rmtree(final, ignore_errors=True)
        os.rename(tmp, final)  # atomic publish
        print(f"[checkpoint] saved {final}")
        if name.startswith("step-"):
            # Rotation: keep only the newest --keep step-N dirs; "best" is
            # exempt. Prevents disk blowup on long runs (~3.6 GB per ckpt).
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
        ids = torch.stack([mixer.next() for _ in range(a.micro_batch)]).to(a.device)
        caps = teacher_capture(teacher, ids, KDA_LAYERS)
        per_layer = []
        for i in KDA_LAYERS:
            x, y = caps[i]["x"], caps[i]["y"]
            assert not x.requires_grad and not y.requires_grad  # DETACH contract
            out, _ = student.model.layers[i].self_attn(x)
            per_layer.append(F.mse_loss(out.float(), y.float()))
        loss = torch.stack(per_layer).mean()
        loss.backward()

        master.pull_grads(student)
        gnorm = torch.nn.utils.clip_grad_norm_(list(master.params.values()), 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        master.push_weights(student)
        student.zero_grad(set_to_none=True)

        rec = None
        if (step + 1) % a.log_every == 0 or step == start_step:
            rec = {"step": step + 1, "mse": round(loss.item(), 6),
                   "gnorm": round(gnorm, 4), "lr": lr_at(step),
                   "elapsed_s": round(time.time() - t0, 1)}

        if valid_batches is not None and (step + 1) % a.eval_every == 0:
            ce, kl = full_ruler_eval(student, teacher, valid_batches)
            if rec is None:
                rec = {"step": step + 1}
            rec.update({"valid_ce": round(ce, 6), "valid_kl": round(kl, 6),
                        "valid_gap": round(ce - teacher_valid_ce, 6)})
            if ce < best_ce:
                best_ce = ce
                save_ckpt(step + 1, name="best")
            print(f"[eval step {step+1}] valid CE {ce:.4f} | gap "
                  f"{ce - teacher_valid_ce:+.4f} | KL {kl:.4f} | best {best_ce:.4f}")

        if rec is not None:
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(f"step {step+1}/{run_until} mse {loss.item():.5f} "
                  f"gnorm {gnorm:.3f} lr {lr_at(step):.2e}")

        if (step + 1) % a.save_every == 0 or (step + 1) == run_until:
            save_ckpt(step + 1)

    if a.self_test:
        fp_after = non_kda_fingerprint(student)
        ok = fp_before == fp_after
        print(f"[self-test] non-KDA params bitwise unchanged: {ok}")
        assert ok, "FROZEN contract violated"
        # fp32-master update proof: representative weights MUST differ from
        # their pre-training sums (reviewer round-3 P0 #1).
        moved = {}
        all_moved = True
        for probe, before in master_sum_before.items():
            hits = [n for n in master.params if n.endswith(probe)]
            after = float(master.params[hits[0]].detach().abs().sum())
            delta = after - before
            moved[probe] = {"before": before, "after": after, "delta": delta}
            all_moved = all_moved and delta != 0.0
        print(f"[self-test] fp32 master updates: {json.dumps(moved, indent=2)}")
        assert all_moved, "FP32 MASTER contract violated: some weights never moved"

    logf.close()
    print("DONE")


if __name__ == "__main__":
    main()
