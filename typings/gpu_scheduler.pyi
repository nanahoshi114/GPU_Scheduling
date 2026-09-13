from typing import Any, Iterable, Mapping, Sequence

# Node mapping keys: id / node_id, gpu_count / gpus, optional topology
# (flat | nvswitch8 | dual_numa8 | pair4). Tuples (id, gpu_count) stay flat.

class Scheduler:
    def __init__(
        self,
        nodes: Sequence[tuple[str, int] | Mapping[str, Any]],
        strategy: str = "topology_aware",
        enable_preemption: bool = True,
        queues: Sequence[tuple[str, int] | Mapping[str, Any]] | None = None,
        locality_timeout: int = 0,
    ) -> None: ...
    def submit(
        self,
        job_or_id: str | Mapping[str, Any],
        gpu_request: int | None = None,
        duration: int = 10,
        arrival_time: int | None = None,
        priority: int = 0,
        queue_id: str = "default",
        parallelism: str = "dp",
    ) -> dict[str, Any]: ...
    def finish(self, job_id: str) -> bool: ...
    def tick(self) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...
    def metrics(self) -> dict[str, Any]: ...
    @property
    def time(self) -> int: ...
    @property
    def strategy(self) -> str: ...
    @property
    def preemption_enabled(self) -> bool: ...
    @property
    def locality_timeout(self) -> int: ...

def simulate(
    nodes: Sequence[tuple[str, int] | Mapping[str, Any]],
    jobs: Iterable[Mapping[str, Any]],
    strategy: str,
    enable_preemption: bool = True,
    queues: Sequence[tuple[str, int] | Mapping[str, Any]] | None = None,
    locality_timeout: int = 0,
) -> dict[str, Any]: ...
def strategies() -> list[str]: ...
