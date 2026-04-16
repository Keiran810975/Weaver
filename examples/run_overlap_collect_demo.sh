#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOCK="/tmp/weaver.sock"
HTTP_PORT="18731"
OUT_DIR="./out"
WEAVER_FILE="./weaver_events.ndjson"

rm -f "$SOCK"

PY="/Users/pigmilk/Code/weaver/.venv/bin/python"

$PY -m weaver.daemon.server --sock "$SOCK" --http-port "$HTTP_PORT" --out "$WEAVER_FILE" >/tmp/weaver_daemon.log 2>&1 &
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

WEAVER_SOCK="$SOCK" LD_PRELOAD="$ROOT_DIR/hooks/libweaver_hook.so" PYTHONPATH=. \
  $PY examples/overlap_collect_demo.py --m_size 512 --comm_size 512 --iters 4 --trace-dir "$OUT_DIR"

PYTHONPATH=. $PY -m weaver.analysis.timeline \
  --weaver "$WEAVER_FILE" \
  --trace "$OUT_DIR/overlap_trace_0_512_512.json" \
  --out "$OUT_DIR/aligned_timeline_rank0.ndjson" \
  --summary "$OUT_DIR/aligned_summary_rank0.json"

curl -s "http://127.0.0.1:${HTTP_PORT}/stats" | cat
