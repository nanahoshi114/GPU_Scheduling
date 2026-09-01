# GPU 拓扑感知调度器

面向 GPU 集群的调度模拟器：在 **资源利用率、资源碎片、跨节点通信成本** 之间做权衡。C++ 实现核心调度与离散事件模拟，经 **pybind11** 编译为 Python 扩展；Python **FastAPI** 提供 Web 界面。

不依赖真实 Kubernetes / GPU 硬件，全部用模拟数据完成。

## 功能

- 定义 Node 与 GPU 资源（同构 / 异构）
- 提交、运行、结束多个 GPU 任务
- 两种策略：**First Fit** 与自研 **Topology-aware**
- 优先减少单个任务跨 Node 使用的 GPU 数量
- 处理 GPU 不足与碎片（不足则等待；过度跨 Node 时可推迟调度）
- 展示每个任务分配到的 Node / GPU，以及调度原因
- 指标：GPU 利用率、等待任务数 / 平均等待时间、跨 Node 任务数
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
uvicorn python.web.app:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。

- **交互调度**：选择或编辑集群、选择策略、提交 / 结束任务、步进时钟，查看 GPU 格子与调度原因
- **策略对比**：加载预设集群与 20 个任务，并排比较 First Fit 与 Topology-aware

运行测试：

```bash
pytest -v
```

在 Python 中直接调用 C++ 扩展：

```python
import gpu_scheduler as gs

nodes = [("node-A", 8), ("node-B", 8), ("node-C", 8), ("node-D", 8)]
sched = gs.Scheduler(nodes, "topology_aware")
print(sched.submit("job-a", 8, duration=10))
print(sched.snapshot())
print(gs.simulate(nodes, [
    {"id": "j1", "gpu_request": 4, "arrival_time": 0, "duration": 5},
], "first_fit")["metrics"])
```

## 预设数据

| 文件 | 说明 |
| --- | --- |
| [data/cluster_4x8.json](data/cluster_4x8.json) | 4 台 × 8 GPU，共 32 卡（题目示例） |
| [data/cluster_mixed.json](data/cluster_mixed.json) | 4 台 8 GPU + 2 台 4 GPU，共 40 卡 |
| [data/jobs_20.json](data/jobs_20.json) | 20 个任务，混合 1/2/4/8 卡，错开到达与时长 |

## 核心设计

### 模型

- 集群由若干 Node 组成，每个 Node 含固定数量的同质 GPU
- 任务申请若干张 GPU，状态为 pending / running / finished
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

\[
score = 100\cdot(n_{nodes}-1) + 10\cdot n_{frag} + 5\cdot n_{awkward}
\]

- 跨 Node 通信代价：使用的节点数减一
- 碎片项：放置后仍有剩余的部分占用 Node 数
- awkward 余量：剩余 GPU 落在 3/5/6/7 的惩罚
- 组合内按空闲从大到小填充，尽量先填满大节点

4. **碎片等待**：理想节点数为「向上取整（G / 单 Node 容量）」。若当前最少跨 Node 数大于理想值，且仍有 running 任务，则暂不调度，等待释放后更集中；若没有 running 任务则降级放置，避免死锁

例如：4 台机器各被占 6 卡、各剩 2 卡时，8 卡任务在 First Fit 下可能被立刻打散；Topology-aware 会等待，直到某台机器释放出完整 8 卡。

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
- 无任务优先级、抢占、多租户 Quota
- 未实现 Kubernetes / Volcano Scheduler Plugin
- Topology-aware 的组合搜索在节点很多时会截断为最空闲的若干台，极端大规模集群上可能不是全局最优
- 任务不可部分抢占；GPU 一旦分配，运行期间不会迁移
