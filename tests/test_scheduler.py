import json
from pathlib import Path

import gpu_scheduler as gs
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_json(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def nodes_from(cluster: dict):
    return [(n["id"], n["gpu_count"]) for n in cluster["nodes"]]


def test_insufficient_gpus_pending_with_reason():
    sched = gs.Scheduler([("node-A", 8), ("node-B", 8)], "first_fit")
    result = sched.submit("big", 24, duration=5)
    assert result["success"] is False
    assert "不足" in result["reason"]

    snap = sched.snapshot()
    jobs = {j["id"]: j for j in snap["jobs"]}
    assert jobs["big"]["state"] == "pending"
    assert jobs["big"]["reason"] == result["reason"]
    assert snap["metrics"]["pending_count"] == 1
    assert snap["metrics"]["used_gpus"] == 0


def test_first_fit_packs_single_node_when_empty():
    sched = gs.Scheduler([("A", 8), ("B", 8), ("C", 8), ("D", 8)], "first_fit")
    result = sched.submit("job-a", 8, duration=10)
    assert result["success"] is True
    assert len(result["placement"]) == 1
    assert result["placement"][0]["node_id"] == "A"
    assert result["placement"][0]["gpu_indices"] == [0, 1, 2, 3, 4, 5, 6, 7]


def test_first_fit_scatters_using_leftover_gpus():
    nodes = [("A", 8), ("B", 8), ("C", 8), ("D", 8)]
    sched = gs.Scheduler(nodes, "first_fit")
    assert sched.submit("fill-A", 6, duration=20)["success"]

    result = sched.submit("wide", 8, duration=10)
    assert result["success"] is True
    assert [p["node_id"] for p in result["placement"]] == ["A", "B"]
    assert len(result["placement"][0]["gpu_indices"]) == 2
    assert len(result["placement"][1]["gpu_indices"]) == 6
    assert sched.metrics()["cross_node_jobs"] == 1


def test_topology_aware_waits_on_fragmentation():
    nodes = [("A", 8), ("B", 8), ("C", 8), ("D", 8)]
    sched = gs.Scheduler(nodes, "topology_aware")
    for node in "ABCD":
        assert sched.submit(f"fill-{node}", 6, duration=20)["success"]

    result = sched.submit("wide", 8, duration=10)
    assert result["success"] is False
    assert "碎片" in result["reason"] or "跨" in result["reason"]
    snap = sched.snapshot()
    jobs = {j["id"]: j for j in snap["jobs"]}
    assert jobs["wide"]["state"] == "pending"


def test_topology_aware_prefers_single_node_over_leftover():
    nodes = [("A", 8), ("B", 8), ("C", 8), ("D", 8)]
    ff = gs.Scheduler(nodes, "first_fit")
    ta = gs.Scheduler(nodes, "topology_aware")

    assert ff.submit("small", 4, duration=10)["success"]
    assert ta.submit("small", 4, duration=10)["success"]

    ff_r = ff.submit("eight", 8, duration=10)
    ta_r = ta.submit("eight", 8, duration=10)

    assert ff_r["success"] and ta_r["success"]
    assert len(ff_r["placement"]) == 2
    assert len(ta_r["placement"]) == 1
    assert ta_r["placement"][0]["node_id"] != "A"


def test_finish_releases_gpus_and_reschedules_pending():
    nodes = [("A", 8), ("B", 8), ("C", 8), ("D", 8)]
    sched = gs.Scheduler(nodes, "topology_aware")
    for node in "ABCD":
        sched.submit(f"fill-{node}", 6, duration=20)
    sched.submit("wide", 8, duration=10)
    assert sched.snapshot()["jobs"][-1]["state"] == "pending"

    assert sched.finish("fill-A") is True
    snap = sched.snapshot()
    jobs = {j["id"]: j for j in snap["jobs"]}
    assert jobs["fill-A"]["state"] == "finished"
    assert jobs["wide"]["state"] == "running"
    assert jobs["wide"]["nodes_used"] == 1
    assert jobs["wide"]["placement"][0]["node_id"] == "A"
    assert snap["metrics"]["pending_count"] == 0


def test_tick_completes_expired_jobs():
    sched = gs.Scheduler([("A", 8)], "first_fit")
    sched.submit("short", 2, duration=2)
    assert sched.snapshot()["jobs"][0]["state"] == "running"
    sched.tick()
    assert sched.snapshot()["jobs"][0]["state"] == "running"
    snap = sched.tick()
    assert snap["jobs"][0]["state"] == "finished"
    assert snap["metrics"]["used_gpus"] == 0
    assert snap["time"] == 2


def test_unknown_strategy_raises():
    with pytest.raises(Exception):
        gs.Scheduler([("A", 8)], "random")


def test_duplicate_job_id_rejected():
    sched = gs.Scheduler([("A", 8)], "first_fit")
    assert sched.submit("j1", 1, duration=3)["success"]
    result = sched.submit("j1", 1, duration=3)
    assert result["success"] is False
    assert "已存在" in result["reason"]


def test_same_twenty_jobs_compare_strategies():
    cluster = load_json("cluster_4x8.json")
    workload = load_json("jobs_20.json")
    nodes = nodes_from(cluster)
    jobs = workload["jobs"]
    assert len(jobs) >= 20

    ff = gs.simulate(nodes, jobs, "first_fit")
    ta = gs.simulate(nodes, jobs, "topology_aware")

    for result in (ff, ta):
        m = result["metrics"]
        assert m["total_gpus"] == 32
        assert 0.0 <= m["avg_utilization"] <= 1.0
        assert m["avg_wait_time"] >= 0
        assert m["cross_node_jobs"] >= 0
        states = {j["state"] for j in result["final_snapshot"]["jobs"]}
        assert states <= {"pending", "running", "finished"}
        assert len(result["final_snapshot"]["jobs"]) == 20

    assert ta["metrics"]["cross_node_jobs"] <= ff["metrics"]["cross_node_jobs"]


def test_mixed_cluster_simulation_runs():
    cluster = load_json("cluster_mixed.json")
    workload = load_json("jobs_20.json")
    result = gs.simulate(nodes_from(cluster), workload["jobs"], "topology_aware")
    assert result["metrics"]["total_gpus"] == 40
    assert result["metrics"]["finished_count"] + result["metrics"]["running_count"] + result[
        "metrics"
    ]["pending_count"] == 20
