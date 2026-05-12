from weaver.diagnose.candidates import CandidateDiscovery
from weaver.diagnose.dependency import DependencyLocalizer
from weaver.diagnose.records import KernelRecord, SlowdownType, SyncKind, SyncRecord
from weaver.diagnose.resource import ResourceLocalizer
from weaver.diagnose.timing import TimingAnalyzer
from weaver.sketch.schema import ExecutionSketch, ExpectedDependency


def kernel(
    kid,
    family,
    tag,
    start,
    end,
    *,
    stream="s0",
    cpu_start=None,
    cpu_end=None,
    work=100.0,
    payload=None,
):
    return KernelRecord(
        kid=kid,
        pid=1,
        rank=0,
        stream=stream,
        kernel_name=f"{family.lower()}_{kid}",
        family=family,
        tag=tag,
        cpu_enqueue_start_ns=cpu_start if cpu_start is not None else start - 100,
        cpu_enqueue_end_ns=cpu_end if cpu_end is not None else start - 50,
        gpu_start_ns=start,
        gpu_end_ns=end,
        work_type="bytes",
        work_value=work,
        payload=payload or {},
    )


def test_module3_marks_sync_as_cpu_runtime_blocker():
    prev = kernel("prev", "GEMM", "GEMM_small", 100, 200, cpu_start=80, cpu_end=90)
    target = kernel("target", "NCCL", "NCCL_allreduce_<=16MB", 1_210_000, 1_212_000, cpu_start=1_205_000, cpu_end=1_205_050)
    sync = SyncRecord(
        sid="sync0",
        kind=SyncKind.CUDA_SYNCHRONIZE,
        pid=1,
        rank=0,
        ts_start_ns=200_000,
        ts_end_ns=1_204_900,
    )

    candidates = CandidateDiscovery().discover([prev, target], [sync])
    assert any(c.target_id == "target" and c.reason == "sync_preceded_kernel" for c in candidates)

    diagnosis = TimingAnalyzer().classify_slowdown(target, [], [prev, target], [sync])
    assert diagnosis.slowdown_type == SlowdownType.CPU_RUNTIME_BLOCKED

    dep = DependencyLocalizer().localize(target, [prev, target], syncs=[sync])
    assert dep is not None
    assert dep.blocker_kind == "sync"
    assert dep.blocker_id == "sync:sync0"


def test_module3_localizes_resource_interference_with_broad_block_slowdown():
    peer1 = kernel("peer1", "NCCL", "NCCL_allreduce_<=16MB", 0, 1_000, stream="s1")
    peer2 = kernel("peer2", "NCCL", "NCCL_allreduce_<=16MB", 4_000, 5_000, stream="s1")
    target = kernel(
        "target",
        "NCCL",
        "NCCL_allreduce_<=16MB",
        1_000,
        3_000,
        stream="s1",
        payload={"block_duration_p50_ns": 220, "block_duration_p99_ns": 520},
    )
    peer1.payload = {"block_duration_p50_ns": 100, "block_duration_p99_ns": 180}
    peer2.payload = {"block_duration_p50_ns": 105, "block_duration_p99_ns": 190}
    culprit = kernel("copy", "MEMCPY", "MEMCPY_large", 1_000, 3_000, stream="s2", work=10.0)
    records = [peer1, peer2, target, culprit]

    timing = TimingAnalyzer().classify_slowdown(target, [peer1, peer2], records, [])
    assert timing.slowdown_type == SlowdownType.RESOURCE_SLOWED

    resource = ResourceLocalizer().localize(target, records)
    assert resource is not None
    assert resource.culprit_id == "copy"
    assert resource.resource_hint == "HBM"
    assert resource.warp_block_verdict == "broad_slowdown"


def test_manual_sketch_expected_dependency_finds_extra_predecessor():
    gemm = kernel("gemm", "GEMM", "GEMM_large", 0, 1_000)
    extra = kernel("extra", "MEMCPY", "MEMCPY_large", 1_000, 1_400)
    target = kernel("target", "NCCL", "NCCL_allreduce_<=16MB", 1_400, 2_000)
    records = [gemm, extra, target]
    sketch = ExecutionSketch(
        metadata={"source": "manual"},
        expected_dependencies=[
            ExpectedDependency(
                dependency_id="gemm_before_nccl",
                target={"family": "NCCL", "tag_regex": "NCCL_allreduce_.*"},
                predecessors=[{"family": "GEMM"}],
                relation="immediate_predecessor",
            )
        ],
    )

    candidates = CandidateDiscovery(sketch).discover(records)
    assert any(
        c.target_id == "target" and c.reason == "unexpected_predecessor_against_manual_sketch"
        for c in candidates
    )

    dep = DependencyLocalizer().localize(target, records, sketch)
    assert dep is not None
    assert dep.blocker_id == "extra"
    assert dep.evidence["expected_predecessors"] == ["gemm"]
