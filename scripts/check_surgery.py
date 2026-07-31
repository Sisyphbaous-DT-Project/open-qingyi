#!/usr/bin/env python
"""Surgery verification: build the 3:1 hybrid, check weight inheritance,
compare parameter counts, run a forward before/after, and test the
save_hybrid/load_hybrid roundtrip.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/projects/qingyi-kda")

from qingyi_kda.surgery import (
    KDA_LAYERS,
    build_hybrid_model,
    load_hybrid,
    param_count,
    save_hybrid,
    verify_surgery,
)

MODEL_PATH = "/root/projects/qingyi-kda/models/Qwen3-0.6B-Base"
SAVE_DIR = "/root/projects/qingyi-kda/models/qingyi-hybrid-init"


def main():
    torch.manual_seed(0)
    print("loading original model (bf16, cpu)...")
    original = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16)

    print("building hybrid model...")
    hybrid = build_hybrid_model(MODEL_PATH, dtype=torch.bfloat16, seed=0)

    print("=" * 70)
    print("weight inheritance check (elementwise equality of untouched parts)")
    print("=" * 70)
    ok = verify_surgery(original, hybrid)

    print("=" * 70)
    print("parameter counts")
    print("=" * 70)
    co = param_count(original)
    ch = param_count(hybrid)
    print(f"original : total={co['total']:,}  attn_total={co['attn_total']:,}  "
          f"per_attn_layer={co['per_native_attn_layer']:,}")
    print(f"hybrid   : total={ch['total']:,}  attn_total={ch['attn_total']:,}  "
          f"per_kda_layer={ch['per_kda_layer']:,}  per_kept_attn_layer={ch['per_native_attn_layer']:,}")

    print("=" * 70)
    print("forward before/after surgery (shape + numeric health)")
    print("=" * 70)
    # fla kernels are CUDA-only; run the hybrid forward on GPU.
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    with torch.no_grad():
        out_o = original(input_ids=ids, use_cache=False).logits
        out_h = hybrid.cuda()(input_ids=ids.cuda(), use_cache=False).logits.cpu()
    print(f"original logits: {tuple(out_o.shape)} finite={torch.isfinite(out_o).all().item()}")
    print(f"hybrid   logits: {tuple(out_h.shape)} finite={torch.isfinite(out_h).all().item()}")
    ok = ok and torch.isfinite(out_o).all().item() and torch.isfinite(out_h).all().item()

    print("=" * 70)
    print("save/load roundtrip")
    print("=" * 70)
    save_hybrid(hybrid.cpu(), SAVE_DIR)
    hybrid2 = load_hybrid(SAVE_DIR, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        out_h2 = hybrid2(input_ids=ids.cuda(), use_cache=False).logits.cpu()
    same = torch.equal(out_h, out_h2)
    print(f"logits identical after reload: {same}")
    ok = ok and same

    print("=" * 70)
    print("SURGERY CHECK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
