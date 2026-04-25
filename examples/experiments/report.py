import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _slowdown(base: float, other: float) -> float:
    if base <= 0:
        return 0.0
    return (other - base) / base


def summarize_one(path: Path) -> Dict:
    data = _load_summary(path)
    summary = data.get("summary", {})

    def get_med(phase: str, intensity: int, key: str) -> float:
        return float(summary.get(f"{phase}|{intensity}", {}).get(key, 0.0))

    intensities = data.get("intensities", [0, 1, 2, 4])
    baseline_target = get_med("baseline", intensities[0], "target_ms_median")

    rows: List[Dict] = []
    for i in intensities:
        overlap_target = get_med("overlap", i, "target_ms_median")
        serialized_target = get_med("serialized", i, "target_ms_median")
        recovery_target = get_med("recovery", i, "target_ms_median")
        overlap_ratio = get_med("overlap", i, "overlap_ratio_median")
        target_bw = get_med("overlap", i, "target_bandwidth_proxy_gbps_median")
        intf_bw = get_med("overlap", i, "interference_bandwidth_proxy_gbps_median")
        skew_start = get_med("overlap", i, "rank_start_skew_ms_median")
        skew_end = get_med("overlap", i, "rank_end_skew_ms_median")

        rows.append(
            {
                "intensity": i,
                "baseline_target_ms": baseline_target,
                "overlap_target_ms": overlap_target,
                "serialized_target_ms": serialized_target,
                "recovery_target_ms": recovery_target,
                "overlap_slowdown": _slowdown(baseline_target, overlap_target),
                "serialized_slowdown": _slowdown(baseline_target, serialized_target),
                "recovery_drift": _slowdown(baseline_target, recovery_target),
                "overlap_ratio": overlap_ratio,
                "target_bw_proxy_gbps": target_bw,
                "interference_bw_proxy_gbps": intf_bw,
                "rank_start_skew_ms": skew_start,
                "rank_end_skew_ms": skew_end,
            }
        )

    return {
        "source": str(path),
        "target": data.get("target"),
        "interference": data.get("interference"),
        "world_size": data.get("world_size"),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize Weaver contention experiment")
    parser.add_argument("--summary", required=True, nargs="+", help="summary_*.json files")
    args = parser.parse_args()

    reports = [summarize_one(Path(p)) for p in args.summary]
    print(json.dumps({"reports": reports}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
