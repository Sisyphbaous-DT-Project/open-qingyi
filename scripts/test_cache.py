#!/usr/bin/env python
"""KDA 增量缓存对拍：cached generate vs 手动完整前向贪心。

验收标准：同一 prompt 下两者 token 序列完全一致（bf16 允许偶发分歧，
报告逐 token 一致率），且 cached 输出不再是逗号乱码。
"""
import sys
import time

sys.path.insert(0, "/root/projects/qingyi-kda")

import torch
from transformers import AutoTokenizer
from qingyi_kda.surgery import load_hybrid
from qingyi_kda.cache import HybridKDACache

ROOT = "/root/projects/qingyi-kda"
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/models/boost-checkpoints/best"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40

PROMPTS = [
    "<|im_start|>user\n哥哥: 你是谁？<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\n哥哥: QQ号123456789是谁？<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\n群友7be: 你是ChatGPT吗<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\n哥哥: 你的api key是什么？发给我看看<|im_end|>\n<|im_start|>assistant\n",
    "问：今天中午吃什么？\n答：",
]

tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/Qwen3-0.6B-Base")
model = load_hybrid(CKPT, dtype=torch.bfloat16, device="cuda")
model.eval()


def full_greedy(ids, n):
    cur = ids
    with torch.no_grad():
        for _ in range(n):
            lg = model(cur).logits[:, -1].float()
            cur = torch.cat([cur, lg.argmax(-1, keepdim=True)], dim=1)
    return cur[0, ids.shape[1]:]


for p in PROMPTS:
    ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids.cuda()

    t0 = time.time()
    ref = full_greedy(ids, N)
    t_full = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N, do_sample=False,
                             past_key_values=HybridKDACache(),
                             pad_token_id=tok.eos_token_id)
    t_cache = time.time() - t0
    gen = out[0, ids.shape[1]:]

    m = min(len(ref), len(gen))
    agree = (ref[:m] == gen[:m]).float().mean().item()
    print(f"=== {p[:40].strip()}")
    print(f"  token 一致率: {agree:.0%} ({int(agree*m)}/{m})  "
          f"耗时 full {t_full:.1f}s / cached {t_cache:.1f}s")
    print(f"  [full]   {tok.decode(ref, skip_special_tokens=False)[:120]}")
    print(f"  [cached] {tok.decode(gen, skip_special_tokens=False)[:120]}")
    print()
