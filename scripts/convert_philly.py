#!/usr/bin/env python3
"""把 Microsoft Philly cluster_job_log 转成本仓库 jobs_*.json / cluster_*.json。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "third_party" / "philly-traces" / "trace-data"
GPU_BUCKETS = (1, 2, 4, 8)


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def attempt_gpus(attempt: dict) -> int:
    total = 0
    for detail in attempt.get("detail") or []:
        total += len(detail.get("gpus") or [])
    return total


def last_complete_attempt(job: dict) -> dict | None:
    for attempt in reversed(job.get("attempts") or []):
        if attempt.get("start_time") and attempt.get("end_time"):
            return attempt
    return None


def usable_job(raw: dict) -> dict | None:
    submitted = parse_ts(raw.get("submitted_time"))
    attempt = last_complete_attempt(raw)
    if submitted is None or attempt is None:
        return None
    start = parse_ts(attempt["start_time"])
    end = parse_ts(attempt["end_time"])
    gpus = attempt_gpus(attempt)
    if start is None or end is None or end <= start or gpus <= 0:
        return None
    job_id = str(raw.get("jobid") or "")
    short = job_id.rsplit("_", 1)[-1] if job_id else "unknown"
    return {
        "id": f"philly-{short}",
        "raw_id": job_id,
        "gpu_request": gpus,
        "submitted": submitted,
        "duration_sec": (end - start).total_seconds(),
        "vc": str(raw.get("vc") or "unknown"),
        "status": raw.get("status"),
    }


def load_usable(job_log: Path) -> list[dict]:
    raw_jobs = json.loads(job_log.read_text(encoding="utf-8"))
    usable: list[dict] = []
    seen: set[str] = set()
    for raw in raw_jobs:
        job = usable_job(raw)
        if job is None:
            continue
        if job["id"] in seen:
            job["id"] = f"{job['id']}-{len(seen)}"
        seen.add(job["id"])
        usable.append(job)
    usable.sort(key=lambda j: (j["submitted"], j["id"]))
    return usable


def load_machines(path: Path) -> list[tuple[str, int]]:
    machines: list[tuple[str, int]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = True
        for row in reader:
            if header:
                header = False
                continue
            if len(row) < 2:
                continue
            machine_id = row[0].strip()
            gpu_count = int(row[1].strip())
            if machine_id and gpu_count > 0:
                machines.append((machine_id, gpu_count))
    machines.sort(key=lambda item: (item[1], item[0]))
    return machines


def densest_window(jobs: list[dict], window_hours: int) -> list[dict]:
    """选一段连续时间，使窗口内 4/8 卡作业尽量多，回放时才有争用。"""
    if not jobs:
        return []
    span = window_hours * 3600
    best: list[dict] = jobs[:1]
    best_score = -1
    right = 0
    for left, job in enumerate(jobs):
        while right < len(jobs) and (jobs[right]["submitted"] - job["submitted"]).total_seconds() <= span:
            right += 1
        window = jobs[left:right]
        score = sum(1 for item in window if item["gpu_request"] >= 4)
        if score > best_score or (score == best_score and len(window) > len(best)):
            best_score = score
            best = window
    return best


def pick_sample(jobs: list[dict], limit: int, seed: int, window_hours: int) -> list[dict]:
    eligible = [j for j in jobs if j["gpu_request"] in GPU_BUCKETS]
    window = densest_window(eligible, window_hours)
    if limit <= 0 or limit >= len(window):
        return window
    by_gpu: dict[int, list[dict]] = defaultdict(list)
    for job in window:
        by_gpu[job["gpu_request"]].append(job)
    # 窗口内再按 1/2/4/8 分层，多卡略过采样，避免几乎全是 1 卡。
    target = {1: 80, 2: 40, 4: 40, 8: 40}
    scale = limit / 200
    rng = random.Random(seed)
    picked: list[dict] = []
    picked_ids: set[str] = set()
    for gpu, share in target.items():
        pool = by_gpu[gpu]
        take = min(len(pool), max(0, int(round(share * scale))))
        chosen = rng.sample(pool, take) if take < len(pool) else list(pool)
        for job in chosen:
            picked.append(job)
            picked_ids.add(job["id"])
    if len(picked) < limit:
        leftover = [j for j in window if j["id"] not in picked_ids]
        need = min(limit - len(picked), len(leftover))
        extra = rng.sample(leftover, need)
        picked.extend(extra)
    picked.sort(key=lambda j: (j["submitted"], j["id"]))
    return picked[:limit]


def to_ticks(jobs: list[dict], bin_sec: int, max_duration: int) -> list[dict]:
    t0 = jobs[0]["submitted"]
    out = []
    for job in jobs:
        arrival = int((job["submitted"] - t0).total_seconds() // bin_sec)
        duration = int(round(job["duration_sec"] / bin_sec))
        duration = max(1, min(max_duration, duration))
        out.append(
            {
                "id": job["id"],
                "gpu_request": job["gpu_request"],
                "arrival_time": arrival,
                "duration": duration,
                "priority": 0,
                "queue_id": job["vc"],
            }
        )
    return out


def build_queues(jobs: list[dict], cluster_gpus: int, top_n: int = 5) -> list[dict]:
    demand = Counter()
    for job in jobs:
        demand[job["queue_id"]] += job["gpu_request"]
    top = [vc for vc, _ in demand.most_common(top_n)]
    merged: dict[str, int] = defaultdict(int)
    for job in jobs:
        qid = job["queue_id"] if job["queue_id"] in top else "other"
        job["queue_id"] = qid
        merged[qid] += job["gpu_request"]
    total = sum(merged.values()) or 1
    queues = []
    assigned = 0
    names = [vc for vc in top if vc in merged] + (["other"] if "other" in merged else [])
    for i, qid in enumerate(names):
        if i == len(names) - 1:
            quota = max(1, cluster_gpus - assigned)
        else:
            quota = max(1, round(cluster_gpus * merged[qid] / total))
            assigned += quota
        queues.append({"id": qid, "gpu_quota": quota})
    overflow = sum(q["gpu_quota"] for q in queues) - cluster_gpus
    if overflow > 0:
        queues[-1]["gpu_quota"] = max(1, queues[-1]["gpu_quota"] - overflow)
    return queues


def build_cluster(machines: list[tuple[str, int]], n8: int, n2: int) -> dict:
    eights = [m for m in machines if m[1] == 8][:n8]
    twos = [m for m in machines if m[1] == 2][:n2]
    nodes = [{"id": mid, "gpu_count": gpus, "topology": "flat"} for mid, gpus in eights + twos]
    total = sum(n["gpu_count"] for n in nodes)
    return {
        "id": "cluster_philly",
        "name": f"Philly 缩放到 {len(nodes)} 节点 / {total} 卡",
        "description": (
            "从 Philly cluster_machine_list 抽取真实机器 id："
            f"{len(eights)} 台 8 GPU + {len(twos)} 台 2 GPU。"
            "原始集群约 552 台 / 2490 卡，这里缩小以便模拟器回放。"
        ),
        "nodes": nodes,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-hours", type=int, default=12, help="从最密的连续时间窗里抽样")
    parser.add_argument("--bin-sec", type=int, default=900, help="一个 tick 对应的秒数，默认 15 分钟")
    parser.add_argument("--max-duration", type=int, default=48)
    parser.add_argument("--nodes-8", type=int, default=8)
    parser.add_argument("--nodes-2", type=int, default=4)
    parser.add_argument("--jobs-out", type=Path, default=ROOT / "data" / "jobs_philly.json")
    parser.add_argument("--cluster-out", type=Path, default=ROOT / "data" / "cluster_philly.json")
    args = parser.parse_args()

    job_log = args.raw / "cluster_job_log"
    machine_list = args.raw / "cluster_machine_list"
    if not job_log.is_file():
        raise SystemExit(f"找不到 {job_log}，请先下载并解压 Philly traces")

    usable = load_usable(job_log)
    sample = pick_sample(usable, args.limit, args.seed, args.window_hours)
    if not sample:
        raise SystemExit("没有可用作业")

    machines = load_machines(machine_list) if machine_list.is_file() else []
    cluster = build_cluster(machines, args.nodes_8, args.nodes_2)
    cluster_gpus = sum(n["gpu_count"] for n in cluster["nodes"])
    jobs = to_ticks(sample, args.bin_sec, args.max_duration)
    queues = build_queues(jobs, cluster_gpus)

    gpu_hist = Counter(j["gpu_request"] for j in jobs)
    span = jobs[-1]["arrival_time"] - jobs[0]["arrival_time"]
    payload = {
        "id": "jobs_philly",
        "name": f"Philly 抽样 {len(jobs)} 个训练任务",
        "description": (
            "Microsoft Philly ATC'19 作业日志抽样。"
            f"取 {args.window_hours} 小时最密窗口后抽样；"
            f"时间按 {args.bin_sec // 60} 分钟/tick 压缩，到达跨度 {span} tick；"
            f"GPU 配比 1/2/4/8 = {gpu_hist[1]}/{gpu_hist[2]}/{gpu_hist[4]}/{gpu_hist[8]}；"
            "queue_id 为原 VC（其余并入 other）。"
        ),
        "queues": queues,
        "jobs": jobs,
    }
    write_json(args.jobs_out, payload)
    write_json(args.cluster_out, cluster)
    print(f"usable={len(usable)} sample={len(jobs)} -> {args.jobs_out}")
    print(f"cluster {len(cluster['nodes'])} nodes / {cluster_gpus} GPUs -> {args.cluster_out}")
    print("queues", queues)
    print("gpu_hist", dict(sorted(gpu_hist.items())))
    print("arrival", jobs[0]["arrival_time"], "..", jobs[-1]["arrival_time"])


if __name__ == "__main__":
    main()
