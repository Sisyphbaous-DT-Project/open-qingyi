"""bitsandbytes 8-bit Adam 冒烟测试。"""
import torch
import bitsandbytes as bnb

p = torch.nn.Parameter(torch.randn(256, 256, device="cuda", dtype=torch.float32))
opt = bnb.optim.Adam8bit([p], lr=1e-3)
for _ in range(3):
    loss = (p ** 2).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
torch.cuda.synchronize()
print("bitsandbytes Adam8bit ok, final loss:", float(loss))
