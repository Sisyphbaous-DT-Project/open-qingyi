"""Triton GPU 冒烟测试：诊断 fla 的 'Triton is not supported' 警告。"""
import torch
import triton
import triton.language as tl

# 1) 复现 fla 的设备探测
try:
    target = triton.runtime.driver.active.get_current_target()
    print("triton current target:", target)
except Exception as e:
    print("get_current_target FAILED:", repr(e))

# 2) 实际编译并运行一个 triton kernel
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")
out = torch.empty_like(x)
add_kernel[(n // 256,)](x, y, out, n, BLOCK=256)
torch.cuda.synchronize()
print("triton kernel ok:", torch.allclose(out, x + y))

# 3) fla 探测结果
from fla.utils import device, device_platform
print("fla device:", device, "| platform:", device_platform)
