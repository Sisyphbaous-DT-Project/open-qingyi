#!/usr/bin/env bash
# Status report for the running (or finished) distillation.
set -uo pipefail
cd /root/projects/qingyi-kda
LOG=logs/distill.log
CKPT_ROOT=models/distill-checkpoints

echo "=== process ==="
if pgrep -af "scripts/distill.py" >/dev/null 2>&1; then
  pgrep -af "scripts/distill.py" | head -2
else
  echo "NOT RUNNING"
fi

echo
echo "=== checkpoints ==="
du -sh "$CKPT_ROOT"/* 2>/dev/null || echo "(none)"

if [ ! -f "$LOG" ]; then
  echo; echo "no log at $LOG yet"
  exit 0
fi

echo
echo "=== latest step / throughput ==="
grep '| mse ' "$LOG" | tail -1 || echo "(no training step logged yet)"

echo
echo "=== latest eval (teacher vs student CE) ==="
grep '\[eval step' "$LOG" | tail -3 || echo "(no eval yet)"
grep '\[best\]' "$LOG" | tail -1 || true

echo
echo "=== last 10 log lines ==="
tail -10 "$LOG"

echo
echo "=== progress / ETA ==="
python3 - "$LOG" <<'EOF'
import re, sys, time
text = open(sys.argv[1]).read()
steps = re.findall(r"step\s+(\d+) \| mse .* \| ([\d,]+) tok/s", text)
evals = re.findall(r"\[eval step (\d+)\]", text)
if not steps:
    print("no step data yet")
    sys.exit(0)
last_step = max(int(s) for s, _ in steps)
# steady tok/s: average of the last 5 logged values
tps_vals = [int(t.replace(",", "")) for _, t in steps[-5:]]
tps = sum(tps_vals) / len(tps_vals)
TOTAL_STEPS = 2000
TOK_PER_STEP = 4 * 2048 * 6  # micro_batch * seq_len * grad_accum
done = last_step * TOK_PER_STEP
remain_s = (TOTAL_STEPS - last_step) * TOK_PER_STEP / tps if tps else float("inf")
print(f"step {last_step}/{TOTAL_STEPS} ({last_step / TOTAL_STEPS:.1%}), "
      f"~{done / 1e6:.1f}M tokens consumed")
print(f"steady throughput ~{tps:,.0f} tok/s, ETA {remain_s / 3600:.1f} h "
      f"(~{time.strftime('%H:%M', time.localtime(time.time() + remain_s))})")
EOF
