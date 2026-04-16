import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    events = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _read_prof_trace(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_profiler_ts_bounds(trace: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    events = trace.get("traceEvents", [])
    min_ts_us = None
    max_ts_us = None
    for e in events:
        ts = e.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        ts_i = int(ts)
        if min_ts_us is None or ts_i < min_ts_us:
            min_ts_us = ts_i
        if max_ts_us is None or ts_i > max_ts_us:
            max_ts_us = ts_i
    if min_ts_us is None:
        return None, None
    return min_ts_us * 1000, max_ts_us * 1000


def _find_sync_markers_weaver(events: List[Dict[str, Any]]) -> Dict[str, int]:
    markers: Dict[str, int] = {}
    for e in events:
        if e.get("kind") != "sync_marker":
            continue
        payload = e.get("payload", {})
        marker = payload.get("marker")
        ts_ns = e.get("ts_ns")
        if isinstance(marker, str) and isinstance(ts_ns, int):
            markers[marker] = ts_ns
    return markers


def _find_sync_markers_prof(trace: Dict[str, Any]) -> Dict[str, int]:
    markers: Dict[str, int] = {}
    for e in trace.get("traceEvents", []):
        name = e.get("name")
        if not (isinstance(name, str) and name.startswith("weaver_sync_")):
            continue
        ts = e.get("ts")
        if isinstance(ts, (int, float)):
            markers[name] = int(ts) * 1000
    return markers


def _estimate_offset_ns(weaver_events: List[Dict[str, Any]], trace: Dict[str, Any]) -> int:
    w_markers = _find_sync_markers_weaver(weaver_events)
    p_markers = _find_sync_markers_prof(trace)
    common = sorted(set(w_markers.keys()) & set(p_markers.keys()))
    if common:
        offsets = [w_markers[m] - p_markers[m] for m in common]
        offsets.sort()
        return offsets[len(offsets) // 2]

    w_ts = [e.get("ts_ns") for e in weaver_events if isinstance(e.get("ts_ns"), int)]
    if not w_ts:
        return 0
    p_start_ns, _ = _extract_profiler_ts_bounds(trace)
    if p_start_ns is None:
        return 0
    return min(w_ts) - p_start_ns


def _convert_profiler_events(trace: Dict[str, Any], offset_ns: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in trace.get("traceEvents", []):
        ts = e.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        ts_ns = int(ts) * 1000 + offset_ns
        dur = e.get("dur")
        dur_ns = int(dur) * 1000 if isinstance(dur, (int, float)) else None
        cat = e.get("cat", "")
        name = e.get("name", "")
        args = e.get("args", {})
        if not isinstance(args, dict):
            args = {}

        layer = "profiler"
        if "kernel" in str(cat).lower():
            layer = "kernel"
        elif "cpu_op" in str(cat).lower() or "operator" in str(cat).lower():
            layer = "operator"

        out.append(
            {
                "ts_ns": ts_ns,
                "dur_ns": dur_ns,
                "pid": e.get("pid"),
                "tid": e.get("tid"),
                "layer": layer,
                "kind": "profiler_event",
                "name": name,
                "cat": cat,
                "payload": args,
            }
        )
    return out


def _collect_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            dev0 = torch.cuda.get_device_properties(0)
            info["cuda_device_0"] = {
                "name": dev0.name,
                "major": dev0.major,
                "minor": dev0.minor,
                "total_memory": int(dev0.total_memory),
                "multi_processor_count": int(dev0.multi_processor_count),
            }
    except Exception as ex:
        info["torch_error"] = str(ex)

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            rows = [r.strip() for r in proc.stdout.splitlines() if r.strip()]
            info["nvidia_smi"] = rows
    except Exception:
        pass
    return info


def build_aligned_timeline(
    weaver_ndjson: Path,
    profiler_trace: Path,
    out_timeline_ndjson: Path,
    out_summary_json: Path,
) -> Dict[str, Any]:
    weaver_events = _read_ndjson(weaver_ndjson)
    trace = _read_prof_trace(profiler_trace)

    offset_ns = _estimate_offset_ns(weaver_events, trace)
    profiler_events = _convert_profiler_events(trace, offset_ns)

    merged: List[Dict[str, Any]] = []
    merged.extend(weaver_events)
    merged.extend(profiler_events)
    merged.sort(key=lambda x: int(x.get("ts_ns", 0)))

    out_timeline_ndjson.parent.mkdir(parents=True, exist_ok=True)
    with out_timeline_ndjson.open("w", encoding="utf-8") as f:
        for e in merged:
            f.write(json.dumps(e, ensure_ascii=True))
            f.write("\n")

    hardware = _collect_hardware_info()

    layers: Dict[str, int] = {}
    for e in merged:
        layer = str(e.get("layer", "unknown"))
        layers[layer] = layers.get(layer, 0) + 1

    summary = {
        "weaver_events": len(weaver_events),
        "profiler_events": len(profiler_events),
        "merged_events": len(merged),
        "offset_ns": offset_ns,
        "layer_counts": layers,
        "hardware": hardware,
        "inputs": {
            "weaver_ndjson": str(weaver_ndjson),
            "profiler_trace": str(profiler_trace),
        },
        "outputs": {
            "timeline_ndjson": str(out_timeline_ndjson),
        },
    }

    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    with out_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Align Weaver + torch profiler timeline")
    parser.add_argument("--weaver", required=True, help="weaver ndjson event file")
    parser.add_argument("--trace", required=True, help="torch profiler chrome trace json")
    parser.add_argument("--out", required=True, help="aligned timeline ndjson")
    parser.add_argument("--summary", required=True, help="summary json")
    args = parser.parse_args()

    summary = build_aligned_timeline(
        weaver_ndjson=Path(args.weaver),
        profiler_trace=Path(args.trace),
        out_timeline_ndjson=Path(args.out),
        out_summary_json=Path(args.summary),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
