#!/usr/bin/env python
"""Teacher control for the generation panel: run the EXACT same prompts and
decoding (greedy, 80 tokens, use_cache=False) on the unmodified
Qwen3-0.6B-Base teacher. Separates "base-model / prompt-form quirks" from
"conversion damage" in the student's repetitive output.

Usage: python scripts/generate_teacher_baseline.py [--tokens 80]
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from generate_sample import PERSONA_PROMPTS, PROMPTS  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--teacher", default="models/Qwen3-0.6B-Base")
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    model.eval()

    for prompt in PROMPTS + PERSONA_PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=args.tokens, do_sample=False,
                use_cache=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"\n=== PROMPT: {prompt}")
        print(text)


if __name__ == "__main__":
    main()
