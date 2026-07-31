import hashlib, sys
import torch
from safetensors.torch import load_file
sd = load_file(sys.argv[1])
h = hashlib.sha256()
for k in sorted(sd):
    t = sd[k].contiguous().view(-1)
    h.update(k.encode()); h.update(str(t.dtype).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
print(h.hexdigest())
