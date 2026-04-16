#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT_DIR"

make -C hooks

PYTHONPATH=. python -m weaver.daemon.server --sock /tmp/weaver.sock --http-port 18731 --out ./weaver_events.ndjson &
DAEMON_PID=$!

cleanup() {
  kill "$DAEMON_PID" || true
}
trap cleanup EXIT

PYTHONPATH=. python examples/demo_python_collect.py
curl -s http://127.0.0.1:18731/stats | cat
