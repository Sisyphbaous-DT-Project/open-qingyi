import glob
import json

for tag in ("cpt-ceval",):
    files = sorted(glob.glob(f"/root/projects/qingyi-kda/eval_results/{tag}/**/samples*.jsonl", recursive=True))
    print(tag, "samples files:", files[:2])
    if not files:
        continue
    with open(files[0]) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            d = json.loads(line)
            resps = d.get("resps")
            lls = [round(r[0][0], 3) for r in resps] if resps else None
            print("doc", d.get("doc_id"), "target:", d.get("target"),
                  "lls:", lls, "acc:", d.get("acc"))
