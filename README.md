# GPU 拓扑感知调度器

面向 GPU 集群的调度模拟器：在 **资源利用率、资源碎片、跨节点通信成本** 之间做权衡。C++ 实现核心调度与离散事件模拟，经 **pybind11** 编译为 Python 扩展；Python **FastAPI** 提供 Web 界面。

不依赖真实 Kubernetes / GPU 硬件，全部用模拟数据完成。

## 功能

- 定义 Node 与 GPU 资源（同构 / 异构）
- 提交、运行、结束多个 GPU 任务
- 两种策略：**First Fit** 与自研 **Topology-aware**
- 优先减少单个任务跨 Node 使用的 GPU 数量
- 处理 GPU 不足与碎片（不足则等待；过度跨 Node 时可推迟调度）
- 支持非负整数优先级，以及可开关的高优先级自动抢占
- 被抢占任务保留剩余时长，稍后按优先级恢复运行
- 多租户 Queue / 硬数量 Quota：超配额任务保持 pending（fair-share delay），抢占只发生在同一队列内
- 展示每个任务分配到的 Node / GPU，以及调度原因
- 指标：GPU 利用率、等待任务数 / 平均等待时间、跨 Node 任务数、抢占次数、公平份额 / 碎片等待
- 同一组任务对比两种策略

## 环境要求

- Python 3.9+
- C++17 编译器（macOS 上的 `clang++` 即可）
- CMake 作为 pip 构建依赖自动安装，不必预先配置系统 CMake

## 运行方式

```bash
cd GPU_TOPO
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install` 会在隔离构建环境中安装 CMake，不依赖 Homebrew。编辑器里的 clangd 需要编译数据库，本机若已有 `cmake`，配置一次即可：

```bash
cmake -S . -B build \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
  -DPython_EXECUTABLE="$(which python)"
```

会生成 `build/compile_commands.json`，仓库根目录有指向它的符号链接。若红线还在，在 Cursor 里执行 **clangd: Restart language server**。

启动 Web 界面：

```bash
./start_web_interface.sh
```

脚本会使用仓库里的 `.venv`，缺依赖时自动 `pip install -e .`。改过 C++ 后可加 `--rebuild`。也可用 `HOST` / `PORT` 覆盖监听地址。

或手动启动：

```bash
uvicorn python.web.app:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

- **交互调度**：选择或编辑集群与队列配额、开关抢占、提交带优先级 / 队列的任务、结束任务或步进时钟
- **策略对比**：加载预设任务，并排比较 First Fit 与 Topology-aware；支持自动/手动 timeline 可视化

运行测试：

```bash
pytest -v
```

在 Python 中直接调用 C++ 扩展：

```python
import gpu_scheduler as gs

nodes = [("node-A", 8), ("node-B", 8), ("node-C", 8), ("node-D", 8)]
sched = gs.Scheduler(
    nodes,
    "topology_aware",
    enable_preemption=True,
    queues=[("research", 16), ("prod", 12), ("default", 4)],
)
print(sched.submit("job-a", 8, duration=10, priority=10, queue_id="research"))
print(sched.snapshot())
print(gs.simulate(nodes, [
    {"id": "j1", "gpu_request": 4, "arrival_time": 0, "duration": 5, "priority": 1, "queue_id": "prod"},
], "first_fit", True, [("research", 16), ("prod", 12), ("default", 4)])["metrics"])
```

## 预设数据

| 文件 | 说明 |
| --- | --- |
| [data/cluster_4x8.json](data/cluster_4x8.json) | 4 台 × 8 GPU，共 32 卡（题目示例） |
| [data/cluster_mixed.json](data/cluster_mixed.json) | 4 台 8 GPU + 2 台 4 GPU，共 40 卡 |
| [data/jobs_20.json](data/jobs_20.json) | 20 个任务，混合 1/2/4/8 卡，错开到达与时长 |
| [data/jobs_100.json](data/jobs_100.json) | 100 个混合训练任务，用于较长压力模拟 |
| [data/jobs_priority.json](data/jobs_priority.json) | 低优先级长任务先占满集群，演示抢占与恢复 |
| [data/jobs_quota.json](data/jobs_quota.json) | research 16 / prod 12 / default 4，演示公平份额等待与队内抢占 |

## 核心设计

### 模型

- 集群由若干 Node 组成，每个 Node 含固定数量的同质 GPU
- 任务申请若干张 GPU，带非负整数优先级（数值越大越高）和 `queue_id`
- 状态为 pending / running / preempted / finished
- 放置结果是 `[(node_id, gpu_indices), ...]`，并附带人类可读原因

### First Fit

按 Node 定义顺序扫描，从前往后拿走空闲 GPU，直到凑满请求。只要空闲总量足够就**立即放置**，不等待更好的打包。

典型结果：等待短，但容易把大任务拆到多个已有碎片的节点上，跨 Node 任务更多。

### Topology-aware

主目标：尽量少跨 Node；次目标：减少难以再装常见规格（1/2/4/8）的余量。

对请求 G 张 GPU 的任务：

1. 空闲总量不足 G → 不可调度（GPU 不足）
2. 按各 Node 空闲数求当前最少跨 Node 数
3. 在恰好该节点数的可行组合中打分，选最低分：

<p align="center"><code>score = 100 · (n<sub>nodes</sub> − 1) + 10 · n<sub>frag</sub> + 5 · n<sub>awkward</sub></code></p>

- 跨 Node 通信代价：使用的节点数减一
- 碎片项：放置后仍有剩余的部分占用 Node 数
- awkward 余量：剩余 GPU 落在 3/5/6/7 的惩罚
- 组合内按空闲从大到小填充，尽量先填满大节点

4. **碎片等待**：理想节点数为「向上取整（G / 单 Node 容量）」。若当前最少跨 Node 数大于理想值，且仍有 running 任务，则暂不调度，等待释放后更集中；若没有 running 任务则降级放置，避免死锁

例如：4 台机器各被占 6 卡、各剩 2 卡时，8 卡任务在 First Fit 下可能被立刻打散；Topology-aware 会等待，直到某台机器释放出完整 8 卡。

### 优先级与抢占

- Pending/Preempted 任务按优先级降序调度，同优先级保持 FIFO；无法放置的大任务之后允许小任务回填
- 高优先级任务直接放置失败时，只考虑**同一队列内**严格低优先级的 running 任务
- 受害任务扣除已运行时间，进入 preempted；资源可用后从 `remaining_duration` 继续

可通过 `Scheduler(..., enable_preemption=False)` 或 Web 开关关闭抢占。`total_preemptions` 统计累计抢占次数，`preempted_jobs` 表示当前处于 preempted 状态的任务数。

#### 最小受害任务集合如何选定

抢占不看「释放 GPU 总数够不够」就动手，而是两层搜索。通过验证不等于立刻选定。

**第一层：锁最小规模 k。** 从 `k = 1` 往上枚举受害者个数。某个 k 上只要出现过一次可行方案，就不再考虑更大的 k。因此「最少受害任务」是硬约束。

**第二层：同一 k 扫完全部组合。** 对每一个大小为 k 的候选集合：

1. 在 Cluster **副本**上释放这些任务（真实集群此时不变）
2. 调用当前放置策略的 `try_place`（First Fit 或 Topology-aware）
3. 失败则丢弃，换下一组
4. 成功则记下比较键，和当前最优比

比较键按字典序越小越好：

1. 受害者总优先级更低
2. 释放 GPU 过量更少（`释放量 − 请求量`）
3. 已运行时间总和更短
4. 放置后占用的 Node 数更少
5. 受害者 ID 列表（保证结果稳定）

例如 8 卡高优先级任务：抢 1 个占满整机的低优先级任务，优于抢 2 个各占 4 卡的任务。若两个 4 卡任务都能单独满足 First Fit，则只抢 1 个（k=1 已有解）；Topology-aware 还会拒绝「释放后仍必须过度跨 Node」的组合，因此抢占不会绕过少跨 Node 的目标。

选定后再一次性修改真实状态：释放受害者、启动新任务。副本验证失败则真实集群完全不变。

### 多租户 Queue / Quota

每个 Queue 有硬数量 `gpu_quota`：该队列 running GPU 数不得超过配额。未配置队列时合成隐式 `default`，配额等于集群总卡数，旧任务集行为不变。

- 作业 `gpu_request` 大于队列配额，或队列不存在 → 拒绝提交
- 配额已用尽 → 任务保持 pending，记为 **fair-share delay**（Philly）；集群仍可能有空闲卡（其他队列未用完）
- 配额还够但放置失败 → **fragmentation delay**（GPU 不足或拓扑等待）
- 抢占只发生在同一队列内，不能借用或回收其他队列的卡

Web 交互页可编辑队列，或一键填入演示配额 `research 16 / prod 12 / default 4`。策略对比加载 `jobs_quota` 时会自动带上该任务集的队列定义。

不做 HiveD cell 拓扑配额，也不做闲置借用 / 跨队列 reclaim。

## 策略对比（同一组 20 个任务）

| 集群 | 策略 | 时间平均 GPU 利用率 | 平均等待 | 跨 Node 任务 | Makespan |
| --- | --- | ---: | ---: | ---: | ---: |
| 4×8（32 卡） | First Fit | 73.4% | 3.85 | 1 | 45 |
| 4×8（32 卡） | Topology-aware | 73.4% | 3.85 | **0** | 45 |
| 异构 40 卡 | First Fit | 75.5% | 2.25 | 2 | 35 |
| 异构 40 卡 | Topology-aware | 75.5% | 2.25 | **0** | 35 |

在这组负载下，两种策略的利用率与等待时间接近，但 Topology-aware 把跨 Node 任务降到 0：大任务会等到能放进更少节点时再启动，而不是吃掉各节点上的 1～2 卡碎片。

构造用例也能看到差异：先占用某节点 4 卡，再提交 8 卡任务时，First Fit 会「A 的 4 卡剩余 + B 的 4 卡」；Topology-aware 会整段放到仍空闲的节点上。

## 项目结构

```
cpp/include, cpp/src   C++ 集群、策略、模拟器、pybind11 绑定
python/web             FastAPI 页面与 API
data/                  集群与任务预设
tests/                 调用扩展模块的 pytest
```

## 已知限制

- 不模拟 NVLink / NVSwitch 等机内拓扑，也不区分 GPU 型号与显存
- Quota 是 GPU 张数硬上限，不是 HiveD cell，也不允许借用其他队列的闲置配额
- 抢占不模拟 checkpoint 保存和恢复开销，也未实现优先级老化；持续高优先级负载可能使低优先级任务饥饿
- 未实现 Kubernetes / Volcano Scheduler Plugin
- Topology-aware 的组合搜索在节点很多时会截断为最空闲的若干台，极端大规模集群上可能不是全局最优
- 任务按整体抢占，不做部分 GPU 抢占或运行中迁移
