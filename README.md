# GPU 拓扑感知调度器

面向 GPU 集群的调度模拟器：在 **资源利用率、资源碎片、跨节点通信成本** 之间做权衡。C++ 实现核心调度与离散事件模拟，经 **pybind11** 编译为 Python 扩展；Python **FastAPI** 提供 Web 界面。

不依赖真实 Kubernetes / GPU 硬件，全部用模拟数据完成。

## 功能

- 定义 Node 与 GPU 资源（同构 / 异构）
- 提交、运行、结束多个 GPU 任务
- 两种策略：**First Fit** 与自研 **Topology-aware**
- 优先减少单个任务跨 Node 使用的 GPU 数量；Topology-aware 再尽量待在同一 NVLink 组
- 处理 GPU 不足与碎片（不足则等待；过度跨 Node 或机内跨组时可推迟调度；可选本地性超时后降级）
- 支持非负整数优先级，以及可开关的高优先级自动抢占
- 被抢占任务保留剩余时长，稍后按优先级恢复运行
- 多租户 Queue / 硬数量 Quota：超配额任务保持 pending（fair-share delay），抢占只发生在同一队列内
- 展示每个任务分配到的 Node / GPU，以及调度原因
- 指标：GPU 利用率、等待任务数 / 平均等待时间、跨 Node / 跨 NVLink / 跨 NUMA 任务数、抢占次数、公平份额 / 碎片等待
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
# Topology-aware 可设 locality_timeout=N：DP 等满 N tick 后允许跨 Node / 跨 NVLink 组（0=永不）
```



## 预设数据


| 文件                                                                 | 说明                                               |
| ------------------------------------------------------------------ | ------------------------------------------------ |
| [data/cluster_4x8.json](data/cluster_4x8.json)                     | 4 台 × 8 GPU，共 32 卡（题目示例，拓扑 `flat`）               |
| [data/cluster_4x8_dual_numa.json](data/cluster_4x8_dual_numa.json) | 4 台 × 8 GPU，每台 `dual_numa8`（0–3 / 4–7 两组 NVLink） |
| [data/cluster_mixed.json](data/cluster_mixed.json)                 | 4 台 8 GPU + 2 台 4 GPU，共 40 卡                     |
| [data/cluster_philly.json](data/cluster_philly.json)               | Philly 机器列表缩放到 8×8 + 4×2，共 72 卡                  |
| [data/jobs_20.json](data/jobs_20.json)                             | 20 个任务，混合 1/2/4/8 卡，错开到达与时长                      |
| [data/jobs_fragment.json](data/jobs_fragment.json)                 | 碎片陷阱：t=0 四个 6 卡长任务，t=1 一个 8 卡                    |
| [data/jobs_100.json](data/jobs_100.json)                           | 100 个混合训练任务，用于较长压力模拟                             |
| [data/jobs_priority.json](data/jobs_priority.json)                 | 低优先级长任务先占满集群，演示抢占与恢复                             |
| [data/jobs_quota.json](data/jobs_quota.json)                       | research 16 / prod 12 / default 4，演示公平份额等待与队内抢占  |
| [data/jobs_nvlink.json](data/jobs_nvlink.json)                     | 12 个 1/2/4 卡任务，先占 GPU 0–1，对比 FF 跨组 vs TA 整组      |
| [data/jobs_parallel.json](data/jobs_parallel.json)                 | DP / TP / PP 混合：TP 整组、PP 相邻节点、DP 碎片等待            |
| [data/jobs_philly.json](data/jobs_philly.json)                     | Philly ATC’19 真实作业抽样 200 个（15 分钟/tick，带原 VC 队列）  |


从 [Philly traces](https://github.com/msr-fiddle/philly-traces) 重新生成抽样：

```bash
# 约 1GB，解压后只用 job log 与 machine list
mkdir -p third_party/philly-traces
curl -fL -o third_party/philly-traces/trace-data.tar.gz \
  https://media.githubusercontent.com/media/msr-fiddle/philly-traces/master/trace-data.tar.gz
tar -xzf third_party/philly-traces/trace-data.tar.gz \
  -C third_party/philly-traces \
  trace-data/cluster_job_log trace-data/cluster_machine_list
python3 scripts/convert_philly.py
```

原始 trace 在 `third_party/philly-traces/`，已加入 `.gitignore`。使用时请引用 ATC’19 论文。

## 核心设计



### 模型

- 集群由若干 Node 组成，每个 Node 含固定数量的同质 GPU，并可带机内拓扑预设
- 每张 GPU 有 `nvlink_group` / `numa_id`（默认都是 0，旧集群 JSON 行为不变）
- 任务申请若干张 GPU，带非负整数优先级（数值越大越高）、`queue_id` 和可选 `parallelism`（`dp` / `tp` / `pp`，缺省 `dp`）
- 状态为 pending / running / preempted / finished
- 放置结果是 `[(node_id, gpu_indices), ...]`，并附带人类可读原因

拓扑预设：


| 预设                       | 标签                                    |
| ------------------------ | ------------------------------------- |
| 空 / `flat` / `nvswitch8` | 全部 `group=0, numa=0`                  |
| `dual_numa8`             | GPU 0–3：numa0/group0；4–7：numa1/group1 |
| `pair4`                  | 0–1 一组、2–3 一组（给异构 4 卡机）               |




### First Fit

按 Node 定义顺序扫描，从前往后按 GPU **下标**拿走空闲卡，直到凑满请求。只要空闲总量足够就**立即放置**，不等待更好的打包，也**不看** NVLink / NUMA / 并行方式。用来对比「机内也被打散」。

典型结果：等待短，但容易把大任务拆到多个已有碎片的节点上，跨 Node / 跨 NVLink 任务更多。

### Topology-aware

主目标：尽量少跨 Node；其次少跨 NVLink 组、少跨 NUMA；再减少难以再装常见规格（1/2/4/8）的余量。

对请求 G 张 GPU 的任务：

1. 空闲总量不足 G → 不可调度（GPU 不足）
2. 按各 Node 空闲数求当前最少跨 Node 数
3. 在恰好该节点数的可行组合中打分，选最低分：

`score = 100 · (nnodes − 1) + 40 · (nnvlink − 1) + 15 · (nnuma − 1) + 10 · nfrag + 5 · nawkward`

- 跨 Node 权重必须最大：2 节点同组一定差于 1 节点跨 NUMA
- `n_nvlink` / `n_numa`：本次放置覆盖的 `(node_id, group)` / `(node_id, numa)` 个数
- 碎片项：放置后仍有剩余的部分占用 Node 数
- awkward 余量：剩余 GPU 落在 3/5/6/7 的惩罚
- 节点之间仍按空闲从大到小填；节点内先吃满一个 NVLink 组，再同 NUMA 下一组，最后才跨 NUMA

1. **跨 Node 碎片等待**：理想节点数是空集群下的结构下限——把各 Node 物理容量从大到小累加，直到 ≥ G。同构时等价于 `ceil(G / 单 Node 容量)`；异构时不会把「两台最大机加起来都不够」当成理想。若当前最少跨 Node 数大于该值，且仍有 running 任务，则暂不调度；若没有 running 则降级放置，避免死锁
2. **机内碎片等待**（仅理想节点数为 1 的 2/4 卡）：`ideal_nvlink = ceil(本机拿卡数 / 该机最大组容量)`。已能在 1 节点放下，但每个候选都不得不跨组，且有 running → pending；无 running → 降级跨组。`dual_numa8` 上 8 卡 `ideal_nvlink=2`，**不要等**
3. **本地性超时**（可选，默认 0 = 永不放松）：DP 在本段等待（`time - pending_since`）达到 `locality_timeout` 个 tick 后，关闭上面两扇等待门，允许跨 Node / 跨 NVLink 组降级放置（Philly 式 soft constraint）。TP 仍必须同一 NVLink 组。First Fit 忽略该参数。构造：`Scheduler(..., locality_timeout=3)`；Web 交互页 / 对比页可填。离散事件模拟会在超时时刻唤醒，不会直接跳到任务结束。下面五场景表均按默认 0 统计

作业可声明并行方式（缺省 `dp`），Topology-aware 按通信模式对齐已有 Node / NVLink 域；First Fit 仍忽略该字段。


| 模式   | 通信           | Topology-aware 放置                                                                        |
| ---- | ------------ | ---------------------------------------------------------------------------------------- |
| `dp` | 梯度同步，可重叠     | 上表原样：最少节点、跨 Node / 机内碎片等待                                                                |
| `tp` | 每步 AllReduce | **硬约束**：G 张卡落在同一个 NVLink 组；`G` 大于全集群最大组容量则提交失败；组被占满且有 running → `TP 需同一 NVLink 组，等待同组空出` |
| `pp` | stage 点对点    | 允许跨 Node，但占用节点必须是集群定义顺序上的**连续窗口**；**不**因 `min_nodes > ideal_nodes` 等待。两台相邻半满机对流水线合法      |


对照：四台 `dual_numa8` 各被占住一部分、没有任何一台剩满 8 卡时，DP-8 等待整机；PP-8 立刻放在相邻两台；TP-8 因组容量 4 直接拒绝。空闲组够 4 卡时，TP-4 整组放入，而 First Fit 可跨组。

例如：4 台机器各被占 6 卡、各剩 2 卡时，8 卡任务在 First Fit 下可能被立刻打散；Topology-aware 会等待，直到某台机器释放出完整 8 卡。

机内对照：一台 `dual_numa8` 先占 GPU 0–1，再提交 4 卡。First Fit 拿走 2,3,4,5（跨 NUMA）；Topology-aware 放到 4–7（同一 NVLink 组）。若两组各剩 2 卡，TA 等待同组空出，而不是跨组拼。

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

## 策略对比

用同一组任务分别跑 First Fit 与 Topology-aware。下面五个场景覆盖轻度混合、异构、碎片陷阱、高压长负载，以及 Philly 真实作业抽样。数字来自 `gs.simulate`（与 Web「策略对比」同一套结果）。


| 场景                        | 策略             | 时间平均 GPU 利用率 | 平均等待      | 跨 Node 任务 | Makespan |
| ------------------------- | -------------- | ------------ | --------- | --------- | -------- |
| 1. 同构 32 卡 + 20 混合任务      | First Fit      | 73.4%        | 3.85      | 1         | 45       |
|                           | Topology-aware | 73.4%        | 3.85      | **0**     | 45       |
| 2. 异构 40 卡 + 20 混合任务      | First Fit      | 75.5%        | 2.25      | 2         | 35       |
|                           | Topology-aware | 75.5%        | 2.25      | **0**     | 35       |
| 3. 同构 32 卡 + 碎片陷阱         | First Fit      | **87.5%**    | **0.00**  | 2         | **20**   |
|                           | Topology-aware | 58.3%        | 3.80      | **0**     | 30       |
| 4. 异构 40 卡 + 100 混合任务     | First Fit      | **92.7%**    | **25.55** | 21        | **140**  |
|                           | Topology-aware | 87.7%        | 26.33     | **0**     | 148      |
| 5. Philly 72 卡 + 200 真实作业 | First Fit      | 53.9%        | 7.00      | 73        | 122      |
|                           | Topology-aware | 53.9%        | 7.01      | **0**     | 122      |


场景 1 / 2 用 [data/jobs_20.json](data/jobs_20.json)；场景 3 用 [data/cluster_4x8.json](data/cluster_4x8.json) + [data/jobs_fragment.json](data/jobs_fragment.json)；场景 4 用 [data/jobs_100.json](data/jobs_100.json)；场景 5 用 [data/cluster_philly.json](data/cluster_philly.json) + [data/jobs_philly.json](data/jobs_philly.json)（原 VC 队列一并加载）。Web 对比页可复现全部五个场景。

### 各场景在说什么

**1 / 2：轻度混合，Topology-aware 几乎零代价换通信：** 利用率、平均等待、makespan 两边一样，差别只在跨 Node：同构上 1→0，异构上 2→0。负载还没把碎片差打满，大任务往往还能等到整机，不必用利用率去换。

**3：碎片陷阱：** 四台各被 6 卡占住时，8 卡任务的理想节点数是 1。Topology-aware 判定当前最少要跨 4 台，选择等待，t=20 第一台释放后再整机放入；代价是利用率从 87.5% 掉到 58.3%，makespan 20→30。First Fit 则立刻把后续 6 卡任务塞进前几台的 2 卡余量（这些 6 卡任务自己变成跨 Node），反而空出一台整机给 8 卡任务——等得短、卡更满，但通信更差。

微观对照同一逻辑：先占某节点 4 卡，再提交 8 卡。First Fit 用「A 的 4 卡剩余 + B 的 4 卡」立刻拆开；Topology-aware 把 8 卡整段落到仍空闲的节点上。两边等待都是 0，只有放置形状不同。

将locality_timeout设为 5 ticks，8 卡任务在等待 5 ticks 后，等待超时，允许跨 Node 放置；最终平均 GPU 利用率和 First Fit 一致，并且仍保持跨 Node 任务更少的优势 （2对比1）

**4：高压异构：** 100 个 1/2/4/8 卡任务连续到达时，First Fit 利用率 92.7%、makespan 140，但有 21 个跨 Node 任务；Topology-aware 把跨 Node 压到 0，利用率降到 87.7%，平均等待多 0.78，makespan 多 8。这是典型的 **locality vs fragmentation delay**（Philly）：等更好的打包，通信变好，排队和空转变多。

**5：Philly 真实抽样：** 集群是 8 台 8 GPU + 4 台 2 GPU（72 卡，真实机器 id）；作业是 ATC’19 日志里 12 小时最密窗口的 200 条抽样，时间按 15 分钟/tick 压缩（到达跨度 47 tick）。GPU 配比 1/2/4/8 = 97/13/45/45，其中 142 个任务时长只有 1 tick；`queue_id` 沿用原 VC，配额合计 72，多个 VC 只有 8 卡，8 卡作业必须在队内串行。

两边时间平均利用率都是 53.9%、makespan 都是 122——利用率被需求钉死（4735 GPU·tick / 8784），不是策略钉死。平均等待 7.00 vs 7.01，200 个任务里只有 2 个 8 卡作业差 1 tick。等待峰值 38 几乎全是 **fair-share**：VC 配额先满，集群里别的队列还有空位。Topology-aware 的 fragmentation pending 只在 1 个 tick 上冒到 2，First Fit 全程为 0。

真正拉开的是放置形状。First Fit 有 **73** 个跨 Node 任务：45 个 8 卡里 41 个被拆开（22 个跨 2 机、14 个跨 3 机，还有 2 个拆到 5 机），45 个 4 卡里 28 个跨机。Topology-aware 把 200 个任务全部放进单节点。真实混合以 1 卡短作业为主、再叠加硬配额，碎片差远没有场景 4 的合成高压那么狠，少跨 Node 几乎不用拿利用率换。

### 差异小结


|          | First Fit                    | Topology-aware              |
| -------- | ---------------------------- | --------------------------- |
| 放置时机     | 空闲总量够就立刻放                    | 跨 Node 多于理想值且仍有 running 时先等 |
| 通信       | 容易吃碎片、拆大任务                   | 优先最少节点，跨 Node 在五个场景里都不高于 FF |
| 利用率 / 等待 | 轻度 / 真实抽样与 TA 持平；碎片和高压时更高、更短 | 轻度 / 真实抽样持平；碎片陷阱和高压时更低、更长   |
| 异构       | 2/4 卡机与 8 卡机混用时更容易跨机拼卡       | 更倾向整机/少节点，空出的小机不一定马上被大任务用掉  |




### Topology-aware 的优势

- **跨 Node 通信稳定更好。** 五个场景里跨 Node 任务数都不高于 First Fit，轻度、高压和 Philly 抽样都能收到 0。训练 AllReduce / 梯度同步对跨机最敏感时，这是主收益。
- **不把未来的整机切碎。** 余量惩罚（3/5/6/7）和碎片等待，避免「每台剩 1～2 卡、8 卡任务永远拼不齐本地」。
- **和放置策略正交。** 抢占仍走同一套 `try_place`，不会为了抢资源就绕过少跨 Node 的目标。
- **轻度负载和真实抽样几乎没有副作用。** 场景 1 / 2 / 5 说明：碎片不严重、或配额先成为瓶颈时，等一下就能整机放入，利用率不用打折。



### Topology-aware 的劣势

- **为本地性付排队和空转。** 场景 3 / 4：硬等待会让整机空着看大任务排队（Philly 的 fragmentation delay），First Fit 反而更高利用率、更短 makespan。场景 5 的真实抽样没有付这笔账——等待被 VC 配额主导，两种策略的利用率与 makespan 相同。
- **默认不超时放松。** `locality_timeout=0` 时只要还有 running 就坚持 k^*。设成正数后，DP 等满该 tick 数会降级跨机 / 跨组（对齐 Philly 的 2～3 分钟放松）；TP 仍不放松。五场景表按默认 0。
- **可能「等错了」。** 场景 3 里 First Fit 拆开 6 卡任务，歪打正着留出整机；Topology-aware 坚持一机一个 6 卡，8 卡任务多等一整段。Tiresias 的批评也在这里：永远压到最少机器会过度阻塞。
- **搜索会截断。** 节点很多时只在最空闲的若干台上枚举组合，大规模集群上不是全局最优。
- **不是 Megatron 三维网格。** 一个任务只选 DP / TP / PP 之一，不把 `tp×pp×dp` 拆成多层 rank，也不做通信矩阵 mapping。

总结：First Fit 是吞吐优先的即时装箱；Topology-aware 是通信优先的有界等待。碎片不重时后者近乎免费；碎片很重或任务很短时，打开 `locality_timeout` 或按通信量决定是否 consolidate（Tiresias / Philly）。

## 项目结构

```
cpp/include, cpp/src   C++ 集群、策略、模拟器、pybind11 绑定
python/web             FastAPI 页面与 API
data/                  集群与任务预设
tests/                 调用扩展模块的 pytest
```



## 已知限制

- 机内拓扑用预设域标签（`flat` / `nvswitch8` / `dual_numa8` / `pair4`），不读 NVML，也不模拟 NVLink 争用或 GPU 型号 / 显存
- 并行方式是作业级互斥模式（dp / tp / pp），不是三维并行网格，也不模拟 AllReduce 带宽
- Quota 是 GPU 张数硬上限，不是 HiveD cell，也不允许借用其他队列的闲置配额
- 抢占不模拟 checkpoint 保存和恢复开销，也未实现优先级老化；持续高优先级负载可能使低优先级任务饥饿
- 未实现 Kubernetes / Volcano Scheduler Plugin
- Topology-aware 的组合搜索在节点很多时会截断为最空闲的若干台，极端大规模集群上可能不是全局最优
- 任务按整体抢占，不做部分 GPU 抢占或运行中迁移

