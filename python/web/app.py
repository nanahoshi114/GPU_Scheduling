from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import gpu_scheduler as gs

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TEMPLATES = Path(__file__).resolve().parent / "templates"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="GPU Topology Scheduler")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response

_lock = threading.Lock()
_scheduler: Optional[gs.Scheduler] = None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _presets() -> dict[str, Any]:
    clusters = [_load_json(p) for p in sorted(DATA.glob("cluster_*.json"))]
    jobs = [_load_json(p) for p in sorted(DATA.glob("jobs_*.json"))]
    return {"clusters": clusters, "jobs": jobs}


def _require_scheduler() -> gs.Scheduler:
    if _scheduler is None:
        raise HTTPException(status_code=400, detail="尚未创建调度会话，请先定义集群")
    return _scheduler


class NodeSpec(BaseModel):
    id: str
    gpu_count: int = Field(gt=0)


class SessionRequest(BaseModel):
    nodes: list[NodeSpec]
    strategy: str = "topology_aware"


class JobRequest(BaseModel):
    id: str
    gpu_request: int = Field(gt=0)
    duration: int = Field(default=10, gt=0)


class CompareRequest(BaseModel):
    nodes: Optional[list[NodeSpec]] = None
    cluster_id: Optional[str] = None
    jobs: Optional[list[dict[str, Any]]] = None
    jobs_id: Optional[str] = None


def _cluster_nodes(req: CompareRequest) -> list[tuple[str, int]]:
    if req.nodes:
        return [(n.id, n.gpu_count) for n in req.nodes]
    if req.cluster_id:
        for c in _presets()["clusters"]:
            if c["id"] == req.cluster_id:
                return [(n["id"], n["gpu_count"]) for n in c["nodes"]]
    raise HTTPException(status_code=400, detail="请提供 nodes 或 cluster_id")


def _job_list(req: CompareRequest) -> list[dict[str, Any]]:
    if req.jobs:
        return req.jobs
    if req.jobs_id:
        for w in _presets()["jobs"]:
            if w["id"] == req.jobs_id:
                return w["jobs"]
    raise HTTPException(status_code=400, detail="请提供 jobs 或 jobs_id")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/presets")
def api_presets() -> dict[str, Any]:
    return _presets()


@app.get("/api/strategies")
def api_strategies() -> dict[str, Any]:
    return {"strategies": list(gs.strategies())}


@app.post("/api/session")
def api_session(req: SessionRequest) -> dict[str, Any]:
    global _scheduler
    if not req.nodes:
        raise HTTPException(status_code=400, detail="至少定义一个 Node")
    nodes = [(n.id, n.gpu_count) for n in req.nodes]
    try:
        sched = gs.Scheduler(nodes, req.strategy)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _lock:
        _scheduler = sched
        snap = _scheduler.snapshot()
    return snap


@app.get("/api/snapshot")
def api_snapshot() -> dict[str, Any]:
    with _lock:
        return _require_scheduler().snapshot()


@app.post("/api/jobs")
def api_submit(req: JobRequest) -> dict[str, Any]:
    with _lock:
        sched = _require_scheduler()
        result = sched.submit(req.id, req.gpu_request, duration=req.duration)
        return {"result": result, "snapshot": sched.snapshot()}


@app.post("/api/jobs/{job_id}/finish")
def api_finish(job_id: str) -> dict[str, Any]:
    with _lock:
        sched = _require_scheduler()
        ok = sched.finish(job_id)
        if not ok:
            raise HTTPException(status_code=400, detail=f"无法结束任务 {job_id}（不存在或未在运行）")
        return sched.snapshot()


@app.post("/api/tick")
def api_tick() -> dict[str, Any]:
    with _lock:
        return _require_scheduler().tick()


@app.post("/api/compare")
def api_compare(req: CompareRequest) -> dict[str, Any]:
    nodes = _cluster_nodes(req)
    jobs = _job_list(req)
    try:
        ff = gs.simulate(nodes, jobs, "first_fit")
        ta = gs.simulate(nodes, jobs, "topology_aware")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"first_fit": ff, "topology_aware": ta}
