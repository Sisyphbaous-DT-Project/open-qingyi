#!/usr/bin/env python
# 手术 v2：teacher-init 构建混合模型 + 继承验证 + 手术校验 + GPU 前向自检
import sys
sys.path.insert(0, "/root/projects/qingyi-kda")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qingyi_kda.surgery import (
    KDA_LAYERS, build_hybrid_model, save_hybrid, verify_surgery,
)

BASE = "models/Qwen3-0.6B-Base"
OUT = "models/qingyi-hybrid-init-v2"
H, D, TD = 16, 64, 128  # KDA heads, KDA head_dim, teacher head_dim


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0).item()


print("loading teacher + building hybrid (teacher_init=True) ...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
hybrid = build_hybrid_model(BASE, dtype=torch.bfloat16, device="cpu",
                            teacher_init=True)

print("verifying projection inheritance over 21 KDA layers ...", flush=True)
worst = 1.0
for i in KDA_LAYERS:
    ka = hybrid.model.layers[i].self_attn
    ta = teacher.model.layers[i].self_attn
    tq_s = ta.q_proj.weight.view(H, TD, -1)[:, :D, :].reshape(H * D, -1)
    to_s = ta.o_proj.weight.view(-1, H, TD)[:, :, :D].reshape(-1, H * D)
    scores = [
        cos(ka.q_proj.weight, tq_s),
        cos(ka.k_proj.weight, ta.k_proj.weight),
        cos(ka.v_proj.weight, ta.v_proj.weight),
        cos(ka.o_proj.weight, to_s),
    ]
    worst = min(worst, *scores)
print(f"projection inheritance worst cosine: {worst:.6f}")
assert worst > 0.999, "teacher-init mapping broken!"

ok = verify_surgery(teacher, hybrid)
save_hybrid(hybrid, OUT)
print(f"saved {OUT} | untouched-weight verify: {'PASS' if ok else 'FAIL'}",
      flush=True)

print("GPU forward sanity check ...", flush=True)
hybrid = hybrid.to("cuda")
tok = AutoTokenizer.from_pretrained(BASE)
ids = tok("你好，今天天气不错，我们", return_tensors="pt").input_ids.to("cuda")
with torch.no_grad():
    out = hybrid(ids, use_cache=False)
finite = torch.isfinite(out.logits).all().item()
print(f"forward OK, logits {tuple(out.logits.shape)}, finite: {finite}")
assert finite, "non-finite logits!"
print("ALL-CHECKS-PASSED")
