#!/bin/bash
# Wait for align self-test group A to finish (4 log lines + process exit),
# then dump the log and tail.
for i in $(seq 1 40); do
  n=$(wc -l < /tmp/align-test-a/align_log.jsonl 2>/dev/null || echo 0)
  alive=$(pgrep -f "align_layers.py --self-test" | head -1)
  if [ "$n" -ge 4 ] && [ -z "$alive" ]; then
    break
  fi
  sleep 20
done
echo "=== align_log.jsonl ==="
cat /tmp/align-test-a/align_log.jsonl
echo "=== tail of stdout ==="
tail -5 /tmp/align-a.log
echo "=== checkpoints ==="
ls /tmp/align-test-a/
