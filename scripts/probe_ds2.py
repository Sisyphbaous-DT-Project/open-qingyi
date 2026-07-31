import sys
sys.path.insert(0, "/root/projects/qingyi-kda")
from datasets import load_dataset

print("=== smollm-corpus fineweb-edu-dedup ===")
ds = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
                  split="train", streaming=True)
print("features:", ds.features)
row = next(iter(ds))
print("keys:", list(row.keys()))
print("text head:", row["text"][:120].replace("\n", " "))

print("=== IndustryCorpus2 ===")
try:
    ds2 = load_dataset("BAAI/IndustryCorpus2", split="train", streaming=True)
    print("features:", ds2.features)
    row2 = next(iter(ds2))
    print("keys:", list(row2.keys()))
    for k, v in row2.items():
        print(f"  {k}: {str(v)[:100]!r}")
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:300])
