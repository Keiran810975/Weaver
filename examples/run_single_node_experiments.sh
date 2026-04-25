#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="/Users/pigmilk/Code/weaver/.venv/bin/python"
SOCK="/tmp/weaver.sock"
HTTP_PORT="18751"
OUT_DIR="./exp_out/single_node"
WEAVER_FILE="./weaver_events.ndjson"

rm -f "$SOCK"
mkdir -p "$OUT_DIR"

$PY -m weaver.daemon.server --sock "$SOCK" --http-port "$HTTP_PORT" --out "$WEAVER_FILE" >/tmp/weaver_exp_daemon.log 2>&1 &
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

# Example 1: SM contention on GEMM
WEAVER_SOCK="$SOCK" LD_PRELOAD="$ROOT_DIR/hooks/libweaver_hook.so" PYTHONPATH=. \
  $PY -m torch.distributed.run --nproc_per_node=2 \
  examples/single_node_multigpu_experiment.py \
  --target gemm --interference sm --intensities 0,1,2 --warmup 5 --iters 10 \
  --m-size 1024 --output-dir "$OUT_DIR" --tag sm_gemm

# Example 2: HBM contention on NCCL
WEAVER_SOCK="$SOCK" LD_PRELOAD="$ROOT_DIR/hooks/libweaver_hook.so" PYTHONPATH=. \
  $PY -m torch.distributed.run --nproc_per_node=2 \
  examples/single_node_multigpu_experiment.py \
  --target nccl --interference hbm --intensities 0,1,2 --warmup 5 --iters 10 \
  --comm-size $((64*1024*1024)) --vec-size $((32*1024*1024)) --output-dir "$OUT_DIR" --tag hbm_nccl

SUMMARY_FILES=$(ls "$OUT_DIR"/summary_*.json)
PYTHONPATH=. $PY examples/experiments/report.py --summary $SUMMARY_FILES > "$OUT_DIR/report.json"

echo "Single-node experiments finished"
echo "Report: $OUT_DIR/report.json"
