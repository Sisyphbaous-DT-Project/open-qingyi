import sys, time
sys.path.insert(0, "/root/projects/qingyi-kda")
import torch
from transformers import AutoModelForCausalLM
from qingyi_kda.surgery import KDA_LAYERS, build_hybrid_model, get_attention_pairs

MODEL_PATH = "/root/projects/qingyi-kda/models/Qwen3-0.6B-Base"
B, T = 4, 2048
dev = "cuda"

teacher = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
student = build_hybrid_model(MODEL_PATH, dtype=torch.bfloat16, device=dev, seed=0)
student.train()
for n, p in student.named_parameters():
    p.requires_grad = any(f"layers.{i}." in n and ".self_attn." in n for i in KDA_LAYERS)
student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

pairs = get_attention_pairs(teacher, student)
t_out, s_out = {}, {}
def mk(store, idx):
    def h(m, a, o):
        store[idx] = o[0] if isinstance(o, tuple) else o
    return h
for i, ta, sk in pairs:
    ta.register_forward_hook(mk(t_out, i))
    sk.register_forward_hook(mk(s_out, i))

def mem():
    return torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30

batch = torch.randint(0, 150000, (B, T), device=dev)

for tag, fn in [
    ("teacher fwd", lambda: teacher.model(batch, use_cache=False)),
    ("student fwd", lambda: student.model(batch, use_cache=False)),
]:
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        fn()
    torch.cuda.synchronize()
    a, pk = mem()
    print(f"{tag}: {time.perf_counter()-t0:.2f}s alloc={a:.2f} peak={pk:.2f} GiB")

# student fwd+bwd with grad
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
for it in range(6):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    student.model(batch, use_cache=False)
    loss = sum(torch.nn.functional.mse_loss(s_out[i].float(), t_out[i].float()) for i in t_out) / len(t_out)
    loss.backward()
    torch.cuda.synchronize()
    a, pk = mem()
    print(f"iter {it}: total={time.perf_counter()-t0:.2f}s alloc={a:.2f} peak={pk:.2f} GiB")
    student.zero_grad(set_to_none=True)
