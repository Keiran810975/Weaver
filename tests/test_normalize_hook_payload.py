import json
from pathlib import Path

from weaver.diagnose.normalize import TimelineNormalizer


def _write_events(path: Path, events):
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event))
            f.write("\n")


def test_normalizer_reads_weaver_cuda_payload(tmp_path):
    timeline = tmp_path / "timeline.ndjson"
    _write_events(
        timeline,
        [
            {
                "ts_ns": 1000,
                "pid": 1,
                "tid": 2,
                "layer": "cuda",
                "kind": "kernel_launch",
                "kernel_name": "cutlass_kernel_gemm",
                "gpu_start_ns": 1000,
                "gpu_end_ns": 3000,
                "cpu_enqueue_start_ns": 900,
                "cpu_enqueue_end_ns": 950,
                "payload": {
                    "grid": [2, 1, 1],
                    "block": [64, 1, 1],
                    "shared_mem": 128,
                    "total_warps": 4,
                    "cuda_event_timing": True,
                },
            }
        ],
    )

    kernels, operators, syncs = TimelineNormalizer(str(timeline)).normalize()

    assert len(kernels) == 1
    assert not operators
    assert not syncs
    assert kernels[0].kernel_name == "cutlass_kernel_gemm"
    assert kernels[0].gpu_dur_ns == 2000
    assert kernels[0].cpu_enqueue_dur_ns == 50
    assert kernels[0].grid == (2, 1, 1)
    assert kernels[0].block == (64, 1, 1)
    assert kernels[0].shared_memory == 128
    assert kernels[0].total_warps == 4
    assert kernels[0].family == "GEMM"


def test_normalizer_ignores_neutrino_metadata_as_kernel(tmp_path):
    timeline = tmp_path / "timeline.ndjson"
    _write_events(
        timeline,
        [
            {
                "ts_ns": 1000,
                "pid": 1,
                "tid": 0,
                "layer": "neutrino",
                "kind": "kernel_disassembly",
                "kernel_name": "some_kernel",
                "payload": {"method": "hooked_binary_disassembly"},
            }
        ],
    )

    kernels, operators, syncs = TimelineNormalizer(str(timeline)).normalize()

    assert kernels == []
    assert operators == []
    assert syncs == []


def test_normalizer_reads_profile_operator_payload(tmp_path):
    timeline = tmp_path / "timeline.ndjson"
    _write_events(
        timeline,
        [
            {
                "ts_ns": 10,
                "pid": 1,
                "tid": 2,
                "layer": "python",
                "kind": "operator",
                "payload": {
                    "operator_name": "torch.cuda.synchronize",
                    "end_ns": 30,
                    "dur_ns": 20,
                },
            }
        ],
    )

    kernels, operators, syncs = TimelineNormalizer(str(timeline)).normalize()

    assert kernels == []
    assert len(operators) == 1
    assert operators[0].operator_name == "torch.cuda.synchronize"
    assert operators[0].duration_ns == 20
    assert syncs == []
