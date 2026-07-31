import sys
sys.path.insert(0, "/root/projects/qingyi-kda")
from transformers import AutoTokenizer
from qingyi_kda.data import build_held_out, make_train_iterator

tok = AutoTokenizer.from_pretrained("/root/projects/qingyi-kda/models/Qwen3-0.6B-Base")
print("eos:", tok.eos_token, tok.eos_token_id)

it = make_train_iterator(tok, 2048, seed=0)
for i in range(3):
    x = next(it)
    print(f"seq {i}: shape={tuple(x.shape)} dtype={x.dtype} min={x.min().item()} max={x.max().item()}")

batches = build_held_out(tok, 2048, 4, n_docs_per_source=50)
print("held-out batches:", len(batches), "shape:", tuple(batches[0].shape) if batches else None)
