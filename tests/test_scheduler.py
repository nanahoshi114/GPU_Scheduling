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
    sched = gs.Scheduler(
        [("node-A", 8), ("node-B", 8)],
        "first_fit",
        True,
        [("research", 16), ("prod", 16)],
    )
    assert sched.submit("fill", 8, duration=5, queue_id="prod")["success"]
    result = sched.submit("big", 12, duration=5, queue_id="research")
    assert result["success"] is False
    assert "不足" in result["reason"]
    assert "配额不足" not in result["reason"]

    snap = sched.snapshot()
    jobs = {j["id"]: j for j in snap["jobs"]}
    assert jobs["big"]["state"] == "pending"
    assert jobs["big"]["reason"] == result["reason"]
    assert snap["metrics"]["pending_count"] == 1
    assert snap["metrics"]["used_gpus"] == 8
    assert snap["metrics"]["fragmentation_pending"] == 1
    assert snap["metrics"]["fair_share_pending"] == 0


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


@pytest.mark.parametrize("strategy", ["first_fit", "topology_aware"])
def test_high_priority_job_preempts_minimum_number_of_victims(strategy):
    sched = gs.Scheduler([("A", 8), ("B", 8)], strategy, True)
    assert sched.submit("low-a", 8, duration=20, priority=1)["success"]
    assert sched.submit("low-b", 4, duration=20, priority=1)["success"]
    assert sched.submit("low-c", 4, duration=20, priority=1)["success"]

    result = sched.submit("urgent", 8, duration=3, priority=10)

    assert result["success"]
    assert result["victims"] == ["low-a"]  # 一项 8 GPU，而不是两项 4 GPU
    jobs = {job["id"]: job for job in sched.snapshot()["jobs"]}
    assert jobs["urgent"]["state"] == "running"
    assert jobs["low-a"]["state"] == "preempted"
    assert jobs["low-a"]["preemption_count"] == 1
    assert sched.metrics()["total_preemptions"] == 1


def test_preemption_never_targets_equal_or_higher_priority():
    sched = gs.Scheduler([("A", 8)], "first_fit", True)
    assert sched.submit("protected", 8, duration=10, priority=10)["success"]

    equal = sched.submit("equal", 8, duration=1, priority=10)
    lower = sched.submit("lower", 8, duration=1, priority=5)

    assert not equal["success"]
    assert not lower["success"]
    jobs = {job["id"]: job for job in sched.snapshot()["jobs"]}
    assert jobs["protected"]["state"] == "running"
    assert jobs["equal"]["state"] == "pending"
    assert jobs["lower"]["state"] == "pending"
    assert sched.metrics()["total_preemptions"] == 0


def test_equal_size_victim_sets_prefer_lower_total_priority():
    sched = gs.Scheduler([("A", 8)], "topology_aware", True)
    sched.submit("more-important", 4, duration=10, priority=3)
    sched.submit("cheaper", 4, duration=10, priority=1)

    result = sched.submit("urgent", 4, duration=1, priority=10)

    assert result["success"]
    assert result["victims"] == ["cheaper"]
    jobs = {job["id"]: job for job in sched.snapshot()["jobs"]}
    assert jobs["more-important"]["state"] == "running"
    assert jobs["cheaper"]["state"] == "preempted"


def test_preemption_can_be_disabled():
    sched = gs.Scheduler([("A", 8)], "first_fit", False)
    sched.submit("low", 8, duration=10, priority=0)
    result = sched.submit("urgent", 8, duration=1, priority=100)

    assert not result["success"]
    assert sched.snapshot()["jobs"][-1]["state"] == "pending"
    assert sched.metrics()["total_preemptions"] == 0


def test_preempted_job_resumes_remaining_duration_and_accumulates_wait():
    sched = gs.Scheduler([("A", 8)], "topology_aware", True)
    sched.submit("low", 8, duration=10, priority=1)
    for _ in range(3):
        sched.tick()

    sched.submit("urgent", 8, duration=2, priority=10)
    preempted = {job["id"]: job for job in sched.snapshot()["jobs"]}["low"]
    assert preempted["state"] == "preempted"
    assert preempted["remaining_duration"] == 7

    sched.tick()
    sched.tick()  # urgent 在 t=5 完成，low 恢复
    resumed = {job["id"]: job for job in sched.snapshot()["jobs"]}["low"]
    assert resumed["state"] == "running"
    assert resumed["remaining_duration"] == 7
    assert resumed["wait_time"] == 2

    for _ in range(7):
        sched.tick()
    finished = {job["id"]: job for job in sched.snapshot()["jobs"]}["low"]
    assert finished["state"] == "finished"
    assert finished["finish_time"] == 12  # 运行 3 + 等待 2 + 继续运行 7


def test_pending_queue_uses_priority_then_fifo():
    sched = gs.Scheduler([("A", 8)], "first_fit", False)
    sched.submit("blocker", 8, duration=10, priority=0)
    sched.submit("low", 8, duration=1, priority=1)
    sched.submit("high", 8, duration=1, priority=9)

    assert sched.finish("blocker")
    jobs = {job["id"]: job for job in sched.snapshot()["jobs"]}
    assert jobs["high"]["state"] == "running"
    assert jobs["low"]["state"] == "pending"


def test_priority_workload_simulation_records_preemption_timeline():
    cluster = load_json("cluster_4x8.json")
    workload = load_json("jobs_priority.json")
    result = gs.simulate(
        nodes_from(cluster), workload["jobs"], "topology_aware", True
    )

    assert result["metrics"]["finished_count"] == len(workload["jobs"])
    assert result["metrics"]["total_preemptions"] > 0
    assert any(
        job["state"] == "preempted"
        for snapshot in result["timeline"]
        for job in snapshot["jobs"]
    )


QUOTA_QUEUES = [("research", 16), ("prod", 12), ("default", 4)]


def test_implicit_default_queue_uses_cluster_capacity():
    sched = gs.Scheduler([("A", 8), ("B", 8)], "first_fit")
    queues = {q["id"]: q for q in sched.snapshot()["queues"]}
    assert queues["default"]["gpu_quota"] == 16
    assert queues["default"]["used_gpus"] == 0
    result = sched.submit("job-a", 8, duration=5)
    assert result["success"]
    snap = sched.snapshot()
    assert snap["jobs"][0]["queue_id"] == "default"
    assert snap["queues"][0]["used_gpus"] == 8


def test_over_quota_pending_while_cluster_has_free_gpus():
    nodes = [("A", 8), ("B", 8), ("C", 8), ("D", 8)]
    sched = gs.Scheduler(nodes, "first_fit", True, QUOTA_QUEUES)
    assert sched.submit("res-a", 8, duration=20, queue_id="research")["success"]
    assert sched.submit("res-b", 8, duration=20, queue_id="research")["success"]

    extra = sched.submit("res-extra", 8, duration=8, queue_id="research")
    assert extra["success"] is False
    assert "配额不足" in extra["reason"]

    prod = sched.submit("prod-a", 8, duration=10, queue_id="prod")
    assert prod["success"]

    snap = sched.snapshot()
    jobs = {j["id"]: j for j in snap["jobs"]}
    queues = {q["id"]: q for q in snap["queues"]}
    assert jobs["res-extra"]["state"] == "pending"
    assert jobs["prod-a"]["state"] == "running"
    assert snap["metrics"]["used_gpus"] == 24
    assert snap["metrics"]["fair_share_pending"] == 1
    assert snap["metrics"]["fragmentation_pending"] == 0
    assert queues["research"]["used_gpus"] == 16
    assert queues["research"]["pending_count"] == 1
    assert queues["prod"]["used_gpus"] == 8


def test_request_exceeding_quota_is_rejected():
    nodes = [("A", 8), ("B", 8), ("C", 8), ("D", 8)]
    sched = gs.Scheduler(nodes, "first_fit", True, QUOTA_QUEUES)
    result = sched.submit("too-big", 20, duration=5, queue_id="research")
    assert result["success"] is False
    assert "超过队列" in result["reason"]
    assert sched.snapshot()["jobs"] == []


def test_unknown_queue_is_rejected():
    sched = gs.Scheduler([("A", 8)], "first_fit", True, QUOTA_QUEUES)
    result = sched.submit("ghost", 2, duration=5, queue_id="missing")
    assert result["success"] is False
    assert "未知队列" in result["reason"]
    assert sched.snapshot()["jobs"] == []


def test_same_queue_preemption_respects_quota():
    nodes = [("A", 8), ("B", 8), ("C", 8), ("D", 8)]
    sched = gs.Scheduler(nodes, "first_fit", True, QUOTA_QUEUES)
    assert sched.submit("res-a", 8, duration=20, priority=1, queue_id="research")["success"]
    assert sched.submit("res-b", 8, duration=20, priority=1, queue_id="research")["success"]

    result = sched.submit("urgent", 8, duration=3, priority=10, queue_id="research")
    assert result["success"]
    assert len(result["victims"]) == 1
    jobs = {j["id"]: j for j in sched.snapshot()["jobs"]}
    assert jobs["urgent"]["state"] == "running"
    assert jobs[result["victims"][0]]["state"] == "preempted"
    queues = {q["id"]: q for q in sched.snapshot()["queues"]}
    assert queues["research"]["used_gpus"] == 16


def test_cross_queue_preemption_is_forbidden():
    sched = gs.Scheduler(
        [("A", 8)], "first_fit", True, [("research", 8), ("prod", 8)]
    )
    assert sched.submit("res", 8, duration=20, priority=1, queue_id="research")["success"]
    result = sched.submit("prod-urgent", 8, duration=3, priority=100, queue_id="prod")
    assert result["success"] is False
    jobs = {j["id"]: j for j in sched.snapshot()["jobs"]}
    assert jobs["res"]["state"] == "running"
    assert jobs["prod-urgent"]["state"] == "pending"
    assert sched.metrics()["total_preemptions"] == 0
    assert sched.metrics()["fair_share_pending"] == 0
    assert sched.metrics()["fragmentation_pending"] == 1


def test_finish_releases_quota_and_starts_pending():
    nodes = [("A", 8), ("B", 8), ("C", 8), ("D", 8)]
    sched = gs.Scheduler(nodes, "first_fit", True, QUOTA_QUEUES)
    sched.submit("res-a", 8, duration=20, queue_id="research")
    sched.submit("res-b", 8, duration=20, queue_id="research")
    sched.submit("res-extra", 8, duration=8, queue_id="research")
    assert sched.snapshot()["jobs"][-1]["state"] == "pending"

    assert sched.finish("res-a")
    jobs = {j["id"]: j for j in sched.snapshot()["jobs"]}
    assert jobs["res-extra"]["state"] == "running"
    assert sched.metrics()["fair_share_pending"] == 0


def test_quota_workload_simulation_records_fair_share_delay():
    cluster = load_json("cluster_4x8.json")
    workload = load_json("jobs_quota.json")
    queues = [(q["id"], q["gpu_quota"]) for q in workload["queues"]]
    result = gs.simulate(
        nodes_from(cluster), workload["jobs"], "first_fit", True, queues
    )

    assert result["final_snapshot"]["queues"]
    assert any(snap["metrics"]["fair_share_pending"] > 0 for snap in result["timeline"])
    by_id = {q["id"]: q for q in result["final_snapshot"]["queues"]}
    assert set(by_id) == {"research", "prod", "default"}
    assert all(q["used_gpus"] <= q["gpu_quota"] for q in by_id.values())
