"""环境验证脚本：加载 Qwen3-0.6B-Base 到 GPU 并做一次贪心生成。"""
import torch
import transformers

print("=" * 60)
print(f"torch        : {torch.__version__}")
print(f"transformers : {transformers.__version__}")
print(f"triton       : {__import__('triton').__version__}")
try:
    import fla
    print(f"fla-core     : {fla.__version__}")
    from fla.ops.kda import chunk_kda  # noqa: F401
    print("fla.ops.kda  : OK (chunk_kda importable)")
except Exception as e:
    print(f"fla-core     : IMPORT FAILED: {e!r}")
try:
    import bitsandbytes as bnb
    print(f"bitsandbytes : {bnb.__version__}")
except Exception as e:
    print(f"bitsandbytes : IMPORT FAILED: {e!r}")
print("=" * 60)

assert torch.cuda.is_available(), "CUDA 不可用！"
print(f"GPU          : {torch.cuda.get_device_name(0)}")
print(f"GPU 显存总量 : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
print("=" * 60)

MODEL_DIR = "/root/projects/qingyi-kda/models/Qwen3-0.6B-Base"

torch.cuda.reset_peak_memory_stats()
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda"
)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量   : {n_params / 1e9:.3f} B ({n_params:,})")

prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=20, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
text = tokenizer.decode(out[0], skip_special_tokens=True)
print("=" * 60)
print(f"Prompt       : {prompt!r}")
print(f"生成文本     : {text!r}")
print("=" * 60)
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f"峰值显存占用 : {peak:.2f} GiB")
print("验证完成 ✔")
