#!/usr/bin/env bash
set -euo pipefail

# Run this script on EACH node with the same command except NODE_RANK.
# Required env vars:
#   NNODES, NODE_RANK, NPROC_PER_NODE, MASTER_ADDR, MASTER_PORT

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${NNODES:?Need NNODES}"
: "${NODE_RANK:?Need NODE_RANK}"
: "${NPROC_PER_NODE:?Need NPROC_PER_NODE}"
: "${MASTER_ADDR:?Need MASTER_ADDR}"
: "${MASTER_PORT:?Need MASTER_PORT}"

PY="/Users/pigmilk/Code/weaver/.venv/bin/python"
SOCK="/tmp/weaver_${NODE_RANK}.sock"
HTTP_PORT=$((18800 + NODE_RANK))
OUT_DIR="./exp_out/multi_node/node_${NODE_RANK}"
WEAVER_FILE="./weaver_events_node_${NODE_RANK}.ndjson"

rm -f "$SOCK"
mkdir -p "$OUT_DIR"

$PY -m weaver.daemon.server --sock "$SOCK" --http-port "$HTTP_PORT" --out "$WEAVER_FILE" >/tmp/weaver_exp_mn_daemon_${NODE_RANK}.log 2>&1 &
DAEMON_PID=$!
cleanup() {
  kill "$DAEMON_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for i in $(seq 1 200); do
  if curl -s "http://127.0.0.1:${HTTP_PORT}/health" >/dev/null; then
    break
  fi
done

# Example: link contention profile in multi-node setting.
WEAVER_SOCK="$SOCK" LD_PRELOAD="$ROOT_DIR/hooks/libweaver_hook.so" PYTHONPATH=. \
$PY -m torch.distributed.run \
  --nnodes "$NNODES" \
  --node_rank "$NODE_RANK" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  examples/multi_node_multigpu_experiment.py \
  --target nccl --interference link --intensities 0,1,2 --warmup 5 --iters 10 \
  --comm-size $((128*1024*1024)) --output-dir "$OUT_DIR" --tag link_mn

SUMMARY_FILES=$(ls "$OUT_DIR"/summary_*.json)
PYTHONPATH=. $PY examples/experiments/report.py --summary $SUMMARY_FILES > "$OUT_DIR/report.json"

echo "Multi-node experiment finished on node ${NODE_RANK}"
echo "Report: $OUT_DIR/report.json"
