#!/usr/bin/env python
"""v3 surgery: KDA128 + function-aligned teacher-init + conversion-friendly
gate init. Builds models/qingyi-hybrid-init-v3 and verifies untouched weights.

Gate-init knobs are CLI args so we can grid-iterate cheaply (build + probe +
ruler eval is minutes, no training)."""
import argparse
import sys

sys.path.insert(0, "/root/projects/qingyi-kda")

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from qingyi_kda.layer import KDAConfig  # noqa: E402
from qingyi_kda.surgery import (  # noqa: E402
    KDA_HYPERPARAMS_V3,
    build_hybrid_model,
    param_count,
    save_hybrid,
    verify_surgery,
)

ROOT = "/root/projects/qingyi-kda"
BASE = f"{ROOT}/models/Qwen3-0.6B-Base"


def main():
    ap = argparse.ArgumentParser()
    # Defaults = grid winner g6 (2026-07-30): CE 9.4756 on the v2 ruler,
    # half-life ~3.3-31.6 tokens (median 6.3). Do NOT regress to the old
    # g1 defaults (dt=-4, A=(0.001,0.01), CE 14.56) without a measured reason.
    ap.add_argument("--gate-scale", type=float, default=0.02)
    ap.add_argument("--dt-bias", type=float, default=0.0)
    ap.add_argument("--a-low", type=float, default=0.03)
    ap.add_argument("--a-high", type=float, default=0.3)
    ap.add_argument("--out-scale", type=float, default=1.0,
                    help="scale o_norm.weight to compensate the ~0.5 sigmoid "
                         "output gate at init (blocker #4); 1.0 = off")
    ap.add_argument("--out", default=f"{ROOT}/models/qingyi-hybrid-init-v3")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    bc = AutoConfig.from_pretrained(BASE)
    cfg = KDAConfig(hidden_size=bc.hidden_size,
                    rms_norm_eps=bc.rms_norm_eps, **KDA_HYPERPARAMS_V3)
    gate_init = {"gate_scale": args.gate_scale, "dt_bias": args.dt_bias,
                 "a_low": args.a_low, "a_high": args.a_high,
                 "out_scale": args.out_scale}
    print(f"v3 surgery: {KDA_HYPERPARAMS_V3}, gate_init={gate_init}")
    model = build_hybrid_model(BASE, dtype=torch.bfloat16, device="cpu",
                               seed=0, init_scheme="v3", kda_config=cfg,
                               gate_init=gate_init)

    print("verifying untouched weights...")
    teacher = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    ok = verify_surgery(teacher, model)
    del teacher
    print(param_count(model))
    assert ok, "surgery verification FAILED"

    if not args.no_save:
        save_hybrid(model, args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
