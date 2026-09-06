from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
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


_FIELD_LABELS = {
    "gpu_count": "Node GPU 数量",
    "gpu_quota": "队列配额",
    "gpu_request": "申请 GPU 数",
    "duration": "运行时长",
    "priority": "优先级",
    "id": "名称",
    "nodes": "节点",
    "queues": "队列",
}


def _show_input(value: Any) -> str:
    if value is None or value == "":
        return "空"
    return str(value)


def _subject_from_loc(body: Any, loc: list[Any]) -> str:
    cur = body
    for key in loc[:-1]:
        if isinstance(key, int) and isinstance(cur, list) and 0 <= key < len(cur):
            cur = cur[key]
        elif isinstance(key, str) and isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return ""
    if isinstance(cur, dict):
        return str(cur.get("id") or "").strip()
    return ""


def _format_validation(errors: list[Any], body: Any = None) -> str:
    parts: list[str] = []
    for err in errors:
        loc = list(err.get("loc", []))
        if loc and loc[0] == "body":
            loc = loc[1:]
        field = next((x for x in reversed(loc) if isinstance(x, str)), "参数")
        label = _FIELD_LABELS.get(str(field), str(field))
        subject = _subject_from_loc(body, loc)
        if field == "gpu_count" and subject:
            label = f"节点「{subject}」的 GPU 数量"
        elif field == "gpu_quota" and subject:
            label = f"队列「{subject}」的配额"
        typ = err.get("type", "")
        shown = _show_input(err.get("input"))
        if typ == "greater_than":
            parts.append(f"{label} 必须为正整数，当前为 {shown}")
        elif typ == "greater_than_equal":
            parts.append(f"{label} 不能为负数，当前为 {shown}")
        elif typ == "missing":
            parts.append(f"缺少 {label}")
        elif typ in {"int_parsing", "int_type", "float_parsing"}:
            parts.append(f"{label} 必须是整数")
        elif typ == "string_type":
            parts.append(f"{label} 必须是文本")
        else:
            parts.append(f"{label} 无效")
    return "；".join(parts) or "请求参数无效"


def _friendly_exc(exc: BaseException) -> str:
    msg = str(exc)
    if "unknown strategy" in msg:
        return "未知策略，请选择 First Fit 或 Topology-aware"
    if "cluster must contain at least one node" in msg:
        return "至少定义一个 Node"
    if "node id must not be empty" in msg:
        return "节点名称不能为空"
    if "gpu_count must be positive" in msg:
        return "节点 GPU 数量必须为正整数"
    if "duplicate node id" in msg:
        suffix = msg.split("duplicate node id:", 1)[-1].strip()
        return f"节点 id 重复: {suffix}" if suffix else "节点 id 重复"
    if "queue id 不能为空" in msg:
        return "队列名称不能为空"
    return msg or "请求失败，请检查输入或稍后重试"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": _format_validation(exc.errors(), getattr(exc, "body", None))},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "请求失败，请检查输入或稍后重试"
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, (HTTPException, RequestValidationError)):
        detail = (
            _format_validation(exc.errors(), getattr(exc, "body", None))
            if isinstance(exc, RequestValidationError)
            else (exc.detail if isinstance(exc.detail, str) else "请求失败，请检查输入或稍后重试")
        )
        status = 422 if isinstance(exc, RequestValidationError) else exc.status_code
        return JSONResponse(status_code=status, content={"detail": detail})
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请重试"})


class NodeSpec(BaseModel):
    id: str
    gpu_count: int = Field(gt=0)


class QueueSpec(BaseModel):
    id: str
    gpu_quota: int = Field(gt=0)


class SessionRequest(BaseModel):
    nodes: list[NodeSpec]
    strategy: str = "topology_aware"
    enable_preemption: bool = True
    queues: Optional[list[QueueSpec]] = None


class JobRequest(BaseModel):
    id: str
    gpu_request: int = Field(gt=0)
    duration: int = Field(default=10, gt=0)
    priority: int = Field(default=0, ge=0)
    queue_id: str = "default"


class CompareRequest(BaseModel):
    nodes: Optional[list[NodeSpec]] = None
    cluster_id: Optional[str] = None
    jobs: Optional[list[dict[str, Any]]] = None
    jobs_id: Optional[str] = None
    enable_preemption: bool = True
    queues: Optional[list[QueueSpec]] = None


def _cluster_nodes(req: CompareRequest) -> list[tuple[str, int]]:
    if req.nodes:
        return [(n.id, n.gpu_count) for n in req.nodes]
    if req.cluster_id:
        for c in _presets()["clusters"]:
            if c["id"] == req.cluster_id:
                return [(n["id"], n["gpu_count"]) for n in c["nodes"]]
        raise HTTPException(status_code=400, detail="找不到所选集群，请重新选择")
    raise HTTPException(status_code=400, detail="请选择集群和任务集")


def _job_list(req: CompareRequest) -> list[dict[str, Any]]:
    if req.jobs:
        return req.jobs
    if req.jobs_id:
        for w in _presets()["jobs"]:
            if w["id"] == req.jobs_id:
                return w["jobs"]
        raise HTTPException(status_code=400, detail="找不到所选任务集，请重新选择")
    raise HTTPException(status_code=400, detail="请选择集群和任务集")


def _queue_tuples(queues: Optional[list[QueueSpec]]) -> Optional[list[tuple[str, int]]]:
    if not queues:
        return None
    return [(q.id, q.gpu_quota) for q in queues]


def _compare_queues(req: CompareRequest) -> Optional[list[tuple[str, int]]]:
    if req.jobs_id:
        for w in _presets()["jobs"]:
            if w["id"] == req.jobs_id and w.get("queues"):
                return [(q["id"], q["gpu_quota"]) for q in w["queues"]]
    return _queue_tuples(req.queues)


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
        sched = gs.Scheduler(
            nodes, req.strategy, req.enable_preemption, _queue_tuples(req.queues)
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_friendly_exc(exc)) from exc
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
        result = sched.submit(
            req.id,
            req.gpu_request,
            duration=req.duration,
            priority=req.priority,
            queue_id=req.queue_id,
        )
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
    queues = _compare_queues(req)
    try:
        ff = gs.simulate(nodes, jobs, "first_fit", req.enable_preemption, queues)
        ta = gs.simulate(nodes, jobs, "topology_aware", req.enable_preemption, queues)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_friendly_exc(exc)) from exc
    return {"first_fit": ff, "topology_aware": ta}
