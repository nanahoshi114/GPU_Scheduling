# GPU 拓扑感知调度：相关论文、项目与主流方案

本文整理面向本仓库题目（见 [`quest.md`](../quest.md)）的调研结果：在 GPU 集群上权衡 **资源利用率、资源碎片、跨节点通信成本** 时，学术界和工业界如何定义问题、有哪些代表工作、主流解法是什么。

---

## 1. 问题在文献里怎么表述

训练任务通常采用 **Gang Scheduling**（要么一次拿齐全部 GPU，要么等待），并且对放置敏感：

| 层次 | 互连 | 相对带宽 / 延迟 |
| --- | --- | --- |
| 机内 | NVLink / NVSwitch ≫ PCIe ≫ 跨 NUMA | 高带宽、低延迟 |
| 机间 | 同 Rack / 同 RDMA 域 ≫ 跨交换机 | 明显更差 |

微软 Philly 生产集群（ATC’19）把排队等待拆成两类：

- **Fair-share delay**：租户配额用完
- **Fragmentation delay**：配额还够，但凑不出「足够本地」的 GPU

他们观察到：集群用到约 2/3 时，完全空闲的机器往往不到 4.5%，而且散落在不同 RDMA 域。这正是题目里「8 卡任务面对 4 台各剩 2 卡」的生产版本。

学术界常用词：

- *consolidated vs disaggregated placement*（集中 vs 打散）
- *locality vs fragmentation delay*（本地性 vs 碎片等待）

本仓库 Topology-aware（最少节点 + awkward 余量惩罚 + 碎片等待）属于 **locality-first + delay scheduling** 这一支。

---

## 2. 最相关的论文

### 2.1 直接对着「拓扑 × 碎片」

| 论文 | 会议 | 和本题的关系 |
| --- | --- | --- |
| [Topology-aware GPU scheduling for learning workloads](https://doi.org/10.1145/3126908.3126933)（Amaral et al.） | SC’17 | 最早系统做 GPU 拓扑放置。作业通信图 × 硬件拓扑图做 mapping；对比 FCFS / Best Fit。**TOPO-AWARE-P** 会在 utility 不够时推迟放置——和本仓库「理想节点数达不到就等」几乎同构。实测 pack vs spread 约 **1.3×**。 |
| [Analysis of Large-Scale Multi-Tenant GPU Clusters](https://www.usenix.org/conference/atc19/presentation/jeon)（Philly） | ATC’19 | 生产证据：强 locality 会拉长排队；超时后放松 locality。建议迁移小任务做碎片整理。Trace：[philly-traces](https://github.com/msr-fiddle/philly-traces)。 |
| [Tiresias](https://www.usenix.org/conference/nsdi19/presentation/gu) | NSDI’19 | 明确批评「永远压到最少机器」会过度阻塞。用通信偏斜 \(S_J\) 和 **PACKLIMIT**：通信重的 **consolidate**，其余 **spread 减碎片**。放置比 ILP 更实用。 |
| [HiveD](https://www.usenix.org/conference/osdi20/presentation/zhao-hanyu) | OSDI’20 | **Cell** 抽象：1/2/4/8 GPU、整机、机架。Buddy 分配保证「VC 有 8 张卡」意味着「能拿出一台完整 8 卡机」，而不是 8 台各 1 张。多租户 Quota 详见第 4 节。代码：[microsoft/hivedscheduler](https://github.com/microsoft/hivedscheduler)。 |
| [Beware of Fragmentation (FGD)](https://www.usenix.org/conference/atc23/presentation/weng) | ATC’23 | 把碎片量化成「相对未来需求分布，这次放置会让多少 GPU 变不可分配」，沿碎片下降最快方向选节点。相对 Best-fit / GPU Packing 最多少 49% 未分配 GPU。模拟器：[hkust-adsl/kubernetes-scheduler-simulator](https://github.com/hkust-adsl/kubernetes-scheduler-simulator)。 |

### 2.2 调度框架（放置只是其中一层）

| 论文 | 会议 | 要点 |
| --- | --- | --- |
| [Gandiva](https://www.usenix.org/conference/osdi18/presentation/xiao) | OSDI’18 | Time-slice + 迁移。碎片起来就 **migrate 做整理**，而不是只在放置时等。 |
| [AntMan](https://www.usenix.org/conference/osdi20/presentation/xiao) | OSDI’20 | 框架与调度器协同；细粒度共享 + 弹性扩缩。 |
| [Pollux](https://www.usenix.org/conference/osdi21/presentation/qiao) | OSDI’21 | 不固定 GPU 数，按 goodput 弹性分配。题目默认「用户指定 GPU 数」，这是加分项里的弹性路线。 |
| [Themis](https://www.usenix.org/conference/nsdi20/presentation/mahajan) | NSDI’20 | Finish-time fairness；两层拍卖。租户间公平详见第 4.2 节。 |
| [Gavel](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak) | OSDI’20 | 异构 GPU 吞吐矩阵 + 轮次调度。 |

### 2.3 综述

- [Deep Learning Workload Scheduling in GPU Datacenters](https://arxiv.org/abs/2205.11913)（CSUR / arXiv:2205.11913）
- 论文与代码索引：[Awesome-DL-Scheduling-Papers](https://github.com/S-Lab-System-Group/Awesome-DL-Scheduling-Papers)

综述里和本题直接相关的分类：

- **Consolidated placement**：尽量少节点 / 少拓扑域
- **Topology-agnostic**：有卡就放（本仓库 First Fit）
- **Gang vs Elastic**：固定申请 vs 动态改 GPU 数
- **Exclusive vs Sharing**：整卡独占 vs 时间片 / MIG / vGPU

---

## 3. 主流方案族

工业界和学术界大致收成 **7 条路线**。本仓库已覆盖 1 和 2；题目加分项里，拓扑 / 并行模式 / 碎片整理对应 4、5、6，**多租户 Queue / Quota** 见第 4 节。

### 路线 1：装箱启发式（基线）

把 Node 当 bin、GPU 当容量：

| 策略 | 行为 | 典型结果 |
| --- | --- | --- |
| **First Fit** | 按节点序填满就走 | 等待短，跨节点多。本仓库基线。 |
| **Best Fit** | 优先塞进「还能放下、剩余最小」的节点 | 减少小碎片 |
| **Worst Fit / Spread** | 故意打散 | 负载均衡好，训练通信更差 |
| **GPU Packing**（Borg / 后续 GPU 调度常用） | 先填已占用 GPU → 再填已开机节点上的空闲 GPU → 最后才开空节点 | 给未来的大任务留整机。FGD 论文把它当作重要基线。 |

题目要求「至少两种策略」，工业对比几乎都是 **FF / BF / GPU Packing vs Topology-aware**。

### 路线 2：Locality-first + 等待 / 超时（本题主线）

算法骨架：

1. 算当前最少跨节点数 \(k\)
2. 理想 \(k^* = \lceil G / \text{单机容量} \rceil\)
3. 若 \(k > k^*\) 且集群还在跑任务 → **先不放**（等整机腾出来）
4. 否则降级放置，避免死锁

变体：

| 系统 | 等待策略 |
| --- | --- |
| 本仓库 Topology-aware | \(k > k^*\) 就等，无 running 才降级 |
| Philly | 强 locality，**2–3 分钟超时后放松** |
| Amaral TOPO-AWARE-P | utility 低于阈值才推迟 |
| Tiresias | 只对通信偏斜大的 job consolidate |
| Kubernetes / YARN Delay Scheduling | 等本地资源，超时再跨机 |

生产里很少「永远等」：Philly 发现排队损失会超过跨机通信收益，尤其是短任务。更稳妥的写法是 **soft constraint + timeout**，或按任务时长 / 通信量自适应。

### 路线 3：打分函数

本仓库当前打分：

```
score = 100 · (n_nodes − 1) + 10 · n_frag + 5 · n_awkward
```

这是文献里最常见的 **加权多目标启发式**。对照：

| 系统 | 打分思路 |
| --- | --- |
| 本仓库 | 跨节点代价远高于碎片；awkward ∈ {3,5,6,7} 惩罚非 2 的幂余量 |
| Volcano NUMA-aware | score ∝ 1 / 跨越的 NUMA 数 |
| KAI Topology | 域级 **bin-pack**（选更满的 domain），域内 **gather**（选更空的 subdomain 把同 job 收拢）——两个层级目标相反 |
| FGD | 不用固定 3/5/6/7，而是用 **未来任务规格分布** 估计「这次放置增加多少不可分配 GPU」 |

`awkward ∈ {3,5,6,7}` 是针对 1/2/4/8 卡规格的经验规则，和 GPU 集群里「请求几乎都是 2 的幂」这一生产事实一致（Philly / Helios / Ali-MLaaS traces）。FGD 更一般，但实现更重。

### 路线 4：层次化拓扑（加分：NVLink / 机架）

把资源看成树，而不是「Node 列表」：

```
Zone / IB Domain
  └─ Rack / SuperPod
       └─ Node
            └─ NVSwitch / NVLink Clique
                 └─ PCIe Switch / NUMA
                      └─ GPU
```

作业声明 **required**（必须在某层内）或 **preferred**（尽量）。

代表系统：

- **NVIDIA KAI Scheduler**（原 Run:ai）：[topology 文档](https://github.com/NVIDIA/KAI-Scheduler/blob/main/docs/topology/README.md)  
  先过滤装不下整个 gang 的 domain，再 bin-pack 选 domain，再在 preferred 层做 locality。
- **Koordinator v1.7**：Network-Topology Aware + Gang + **job-level 抢占**；`MustGather` 把一组 Pod 收进同一 Block。
- **Volcano**：Gang + `numaaware`；正在加 GPU–NUMA（[issue 4998](https://github.com/volcano-sh/volcano/issues/4998) / [PR 5095](https://github.com/volcano-sh/volcano/pull/5095)）。策略名直接沿用 kubelet Topology Manager：`none / best-effort / restricted / single-numa-node`。
- **HiveD Cell**：把层次做成可分配的 buddy 单元，从机制上避免「8 张游离卡」。
- **Kubernetes Topology Manager**：只在 **kubelet 节点内** 对齐 CPU / GPU / NIC；**集群调度器本身不知道拓扑**，所以可能调度到节点后被 Topology Manager 拒绝。Volcano / KAI 就是补这一层。

本仓库若要把这一层做进模拟器，按第 9 节改，不要另开第三条策略。

### 路线 5：按并行模式放置（加分：DP / TP / PP）

LLM 时代的主流：不是「G 张卡尽量少节点」，而是 **通信模式对齐拓扑**：

| 并行 | 通信特征 | 放置偏好 |
| --- | --- | --- |
| TP（张量并行） | 每步 AllReduce，带宽极敏感 | 必须同一 NVLink / NVSwitch 域 |
| PP（流水线） | 点对点激活传递 | 可跨节点，但相邻 stage 尽量近 |
| DP（数据并行） | 梯度同步，可重叠计算 | 可跨机架，最好同 IB 域 |
| EP（专家并行） | All-to-All | 需要均衡的网络切面 |

近期工作：[Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment](https://proceedings.neurips.cc/paper_files/paper/2025/file/d82c24b7a4237aa4283b38e12047dc38-Paper-Conference.pdf)（9600+ GPU）把 job 的通信矩阵映射到物理拓扑，对比 Best-fit / GPU-pack / 经典 graph bipartition topo-aware，大作业上更好。

题目若只做「少跨 Node」，已经覆盖 DP 为主的小集群；要冲加分，最小增量是：**TP 组必须同节点，PP 组按 stage 相邻放置**。

### 路线 6：整理碎片，而不只是「等」

放置时等，利用率会掉。主流补充：

- **迁移**（Gandiva）：把小任务挪开，腾出整机给大任务
- **抢占**（本仓库已做；Koordinator 是 job-level gang 抢占）
- **Backfill**：大任务在等整机时，用小任务填剩余卡（注意不要把整机又切碎——通常只 backfill 短任务）
- **弹性训练**（Pollux / AntMan / EDL）：大任务先用较少 GPU 跑起来，有整机再 expand

### 路线 7：GPU 共享（超出本题独占模型）

题目是整卡独占。生产上利用率低时会走：

- NVIDIA MPS / MIG / time-slicing
- HAMi（CNCF Sandbox，已被 KAI 采用做显存硬隔离）
- FGD：共享后碎片问题更严重，经典 Best-fit 会失效

和本题正交，了解即可。

---

## 4. 多租户 Queue / Quota（题目加分）

加分项「支持多租户 Queue / Quota」在文献和系统里通常拆成两层，不要混成一件事：

| 概念 | 含义 | 典型实现 |
| --- | --- | --- |
| **Quota** | 每个租户 / 团队有一份 GPU 份额（硬上限或软保证） | 团队 A 保证 16 卡，最多借到 24 卡 |
| **Queue** | 每个租户一条（或多条）队列，队列间按权重 / 配额选作业 | `team-a` 队列 FIFO，和 `team-b` 公平分享 |

论文更爱写 **Virtual Cluster (VC) / tenant share / fairness**；**Queue** 更多出现在 YARN / Volcano / Kueue 这类工程系统里。

### 4.1 直接写 Quota 的论文

**HiveD（OSDI’20）——最该读。** [Sharing a GPU Cluster for Deep Learning with Guarantees](https://www.usenix.org/conference/osdi20/presentation/zhao-hanyu)

核心论断：传统多租户只用 **GPU 张数做 Quota**，保证不了亲和性。租户配额还剩 8 张卡，但散在多台机器上，8 卡整机任务照样排不上，这叫 **sharing anomaly**。

做法：

- 每个租户一个 **Virtual Private Cluster (VC)**
- 配额不是「8 GPU」，而是 **cell**（1 卡 / 同 PCIe / 同 NUMA / 整机 / 机架……）
- **Buddy cell allocation** 把 VC 的 cell 绑到物理集群，保证：若私有集群能放下，共享集群也一定能放下（sharing safety）
- 闲置配额可被其他租户 **低优先级借用**，提高利用率
- VC **内部** 仍可跑 Tiresias / Gandiva 等调度器（关注点分离）

代码：[microsoft/hivedscheduler](https://github.com/microsoft/hivedscheduler)（OpenPAI 的 K8s Scheduler Extender）。这和题目加分几乎一一对应：**多租户 + Quota（还带拓扑）+ 低优先级借闲置资源**。若只实现一项加分，这篇最对口。

**Philly（ATC’19）——生产里 Quota 怎么用。** 微软共享集群按团队划 VC，配额是 **GPU 张数**。排队被拆成：

- **Fair-share delay**：本 VC 配额用完才等（Quota 在起作用）
- **Fragmentation delay**：配额还够，但凑不出够本地的 GPU（HiveD 后来要修的问题）

第 1 节里的 fair-share delay，就是多租户 Quota 的直接产物。

**更早的数量配额传统。** 大数据 / HPC 里已经有成熟模型，GPU 论文经常当基线：

- **YARN Capacity / Fair Scheduler**：队列 + 容量百分比 + 弹性超卖
- **DRF**（Dominant Resource Fairness, NSDI’11）：多维资源（CPU / 内存 / GPU）按主导份额公平分
- Kubernetes `ResourceQuota` / `PriorityClass`、Volcano `Queue`、Slurm partition / QoS

HiveD 的出发点就是：这些数量 Quota 对 GPU 训练不够。

### 4.2 相关但不等于 Quota：公平性论文

这些不设「团队 A 固定 16 卡」，而是让各租户 / 应用的完成时间别差太多。Quota 是硬隔离，Fairness 是长期均衡。

| 论文 | 会议 | 和 Queue / Quota 的关系 |
| --- | --- | --- |
| [Themis](https://www.usenix.org/conference/nsdi20/presentation/mahajan) | NSDI’20 | 以 app / 租户为对象，用 **finish-time fairness**（共享集群完成时间 / 独占 \(1/N\) 集群完成时间）。两层：应用自己出价，中心 arbiter 拍卖 GPU。保证 sharing incentive、envy-freeness。 |
| Astraea（Ye et al., 2021） | — | **租户级 + 作业级** Long-Term GPU-time Fairness（LTGF），两层 max-min |
| [Gandiva_fair](https://www.usenix.org/conference/osdi20/presentation/chaudhary) | OSDI’20 | 异构 GPU 上的 **用户间公平**，允许用户互换不同型号 GPU-time |
| [Gavel](https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak) | OSDI’20 | 异构 GPU 上可插拔公平目标（max-min、makespan、finish-time fairness） |
| Dorm / AlloX | SoCC’17 / EuroSys’20 | GPU + CPU + 内存一起做公平分配 |

Themis 的「app」很接近多租户：一个团队提交一组相关训练作业，调度器在 **应用之间** 分 GPU，而不是全局一个 FIFO。

### 4.3 Queue：论文少、系统多

「每个租户一条 Queue、队列间加权轮转 / 容量调度」在论文里通常一笔带过，工程系统写得很全：

| 系统 | Queue / Quota 怎么做 |
| --- | --- |
| **Volcano** | `Queue` CRD：`capability`（配额）、`weight`、`reclaimable`（能否被回收）、队内优先级。题目加分「Volcano Plugin」就是这条。 |
| **OpenPAI + HiveD** | 团队 = VC；VC 内再排队 |
| **Kueue**（K8s） | ClusterQueue / LocalQueue + 借还配额 |
| **Koordinator** | ElasticQuota：min 保证 + max 上限，可借闲置 |
| **NVIDIA KAI** | 按项目 / 部门公平分享 + 超额借资源 |
| **Slurm** | partition + QoS + account fairshare |

实现加分时，最小闭环通常是：

1. 作业带 `tenant` / `queue`
2. 每个 queue 有 `gpu_quota`（运行中占用 ≤ 配额）
3. 超配额的作业保持 pending（fair-share delay）
4. （可选）允许低优先级借其他 queue 的空闲卡，被拥有者需要时抢回——HiveD / Koordinator ElasticQuota / Volcano reclaim

### 4.4 和本仓库的关系

本仓库已有 **作业级优先级 + 抢占**，还没有 **租户隔离**。文献里的分层一般是：

```
租户 / Queue          ← Quota、公平份额、借还（HiveD / Volcano / Themis）
  └─ 队列内作业       ← 优先级、FIFO、抢占（本仓库已做）
       └─ 单作业放置  ← First Fit / Topology-aware（本题主体）
```

HiveD 特别强调：**Quota 层和放置策略要分开**。VC 保证拓扑配额；VC 里面仍跑 Topology-aware。这和题目把「多租户 Queue / Quota」列为加分、把拓扑放置列为必做是同一结构。

最值得对照实现的三篇：

1. **HiveD OSDI’20** — 多租户 + 拓扑感知 Quota（最贴加分项）
2. **Philly ATC’19** — 生产 VC 数量配额，以及 fair-share vs fragmentation delay
3. **Themis NSDI’20** — 不想做硬配额时，用 finish-time fairness 做租户间公平

---

## 5. 工业项目（可对标 Demo）

| 项目 | 定位 | 和本题对应 |
| --- | --- | --- |
| [Volcano](https://github.com/volcano-sh/volcano) | K8s 批调度，CNCF | Gang、Queue 配额、抢占、NUMA；题目加分「Volcano Plugin」 |
| [NVIDIA KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler) | 原 Run:ai | 分层拓扑、bin-pack vs gather、部门级公平分享 |
| [Koordinator](https://koordinator.sh/) | 阿里开源 | 网络拓扑感知、ElasticQuota、job 级抢占 |
| [HAMi](https://github.com/Project-HAMi/HAMi) | 异构设备虚拟化 | 细粒度 GPU 切分 |
| [HiveD / OpenPAI](https://github.com/microsoft/hivedscheduler) | 微软多租户 | Cell 拓扑配额 |
| [hkust-adsl/kubernetes-scheduler-simulator](https://github.com/hkust-adsl/kubernetes-scheduler-simulator) | FGD 模拟器 | 和本仓库「Scheduler Simulator + 策略对比」同一产品形态 |
| Kubernetes Device Plugin + Topology Manager | 节点内 | 只解决机内，不解决跨 Node |
| Slurm `--gres=gpu` + topology.conf | HPC | 树形拓扑距离，HPC 训练集群常用 |

产品形态上，本仓库已对齐「模拟器 + 双策略对比」。学术对标最像 **Amaral SC’17 + Philly delay + Tiresias 放置**。

---

## 6. 和本仓库设计的对照

README 里的 Topology-aware 可以放进文献谱系：

| 本仓库能力 | 文献归属 |
| --- | --- |
| First Fit | topology-agnostic 即时放置 |
| Topology-aware | min-nodes + 碎片 / awkward 打分 + 碎片等待 |
| 优先级 / 抢占 | Borg / K8s 经典；Koordinator 做到 gang 级 |
| 多租户 Queue / Quota | Philly 数量配额 + 队内抢占（已做）；HiveD cell 未做 |
| 未做 NVLink / NVSwitch | 机内层次拓扑，改造方案见第 9 节 |
| 未做 DP / TP / PP | 通信对齐放置，应叠在第 9 节模型上，不要先做 |
| 未做超时放松 | Philly：等太久应降级跨节点 |
| 未做迁移整理 | Gandiva |
| awkward 固定集合 | 可升级为 FGD 式「相对负载分布的碎片」 |

一句话总结：

> 主流不是「永远 First Fit」或「永远整机打包」，而是 **分层拓扑约束 + 装箱减碎片 + 有界等待**。通信敏感（大 AllReduce / TP）走 consolidate；通信不敏感或等待过长则 spread / 超时放松。HiveD 用 cell 从配额层保证拓扑；FGD 用碎片梯度替代拍脑袋的 leftover 惩罚；KAI / Koordinator / Volcano 是这条路线在 Kubernetes 上的工程形态。

---

## 7. 建议阅读顺序

1. **Philly ATC’19** — 问题从哪来（locality vs 排队；生产 VC 数量配额）
2. **Amaral SC’17** + **Tiresias NSDI’19** — 两种相反的放置哲学（等更好的打包 vs 按通信量决定 pack/spread）
3. **HiveD OSDI’20** — 多租户 + 拓扑感知 Quota 怎么保证
4. **Themis NSDI’20** — 不做硬配额时，用 finish-time fairness 做租户间公平
5. **FGD ATC’23** — 碎片怎么量化
6. **CSUR 综述** + **KAI topology 文档** — 工业现状
7. 有余力：**Pollux**（弹性）、LLM 通信对齐放置（TP / PP，叠在第 9 节域模型上）

---

## 8. 参考链接速查

### 论文 PDF / 会议页

- Amaral et al., SC’17: https://doi.org/10.1145/3126908.3126933
- Philly, ATC’19: https://www.usenix.org/conference/atc19/presentation/jeon
- Tiresias, NSDI’19: https://www.usenix.org/conference/nsdi19/presentation/gu
- Gandiva, OSDI’18: https://www.usenix.org/conference/osdi18/presentation/xiao
- HiveD, OSDI’20: https://www.usenix.org/conference/osdi20/presentation/zhao-hanyu
- Themis, NSDI’20: https://www.usenix.org/conference/nsdi20/presentation/mahajan
- Gandiva_fair, OSDI’20: https://www.usenix.org/conference/osdi20/presentation/chaudhary
- AlloX, EuroSys’20: https://doi.org/10.1145/3342195.3387547
- AntMan, OSDI’20: https://www.usenix.org/conference/osdi20/presentation/xiao
- Pollux, OSDI’21: https://www.usenix.org/conference/osdi21/presentation/qiao
- FGD, ATC’23: https://www.usenix.org/conference/atc23/presentation/weng
- CSUR 综述: https://arxiv.org/abs/2205.11913

### 开源项目

- Awesome-DL-Scheduling-Papers: https://github.com/S-Lab-System-Group/Awesome-DL-Scheduling-Papers
- HiveD: https://github.com/microsoft/hivedscheduler
- FGD 模拟器: https://github.com/hkust-adsl/kubernetes-scheduler-simulator
- Volcano: https://github.com/volcano-sh/volcano
- NVIDIA KAI Scheduler: https://github.com/NVIDIA/KAI-Scheduler
- Koordinator: https://koordinator.sh/
- HAMi: https://github.com/Project-HAMi/HAMi
- Philly traces: https://github.com/msr-fiddle/philly-traces

---

## 9. 改造指导：把 NVLink / NVSwitch 纳入调度

题目加分「将 NVLink / NVSwitch 等 GPU 内部拓扑纳入调度」。Queue / Quota 已经完成，这一节按**当前代码结构**写改法，不重做租户层。

### 9.1 现在缺什么

当前模型里，Node 只是「一袋无差别 GPU」：

- `Gpu` 只有 `index` + `occupier`（`cpp/include/types.h`）
- `Cluster::add_node(id, gpu_count)`，JSON 只有 `{"id": "node-A", "gpu_count": 8}`
- Topology-aware 只优化 **跨 Node 数**；节点内 `pack_largest_first` / First Fit 都是按 `free_gpu_indices` 从小到大拿走
- 所以「同一台机器上 GPU 0–3 走 NVLink、0 和 4 只走 PCIe」对调度器不可见

Queue / Quota **不要改语义**：配额仍按张数、抢占仍只在队内。机内拓扑只影响 **选哪几张卡**，对应 HiveD 的关注点分离：外层切蛋糕，内层做放置。

First Fit **保持拓扑无关**，继续当基线。只扩展 `TopologyAwareStrategy`，不要加第三条策略名。

抢占已经在 Cluster 副本上调用 `try_place`，策略改完后会自动拒绝「释放后仍跨 NVLink 域」的受害者组合，不必单独写一套抢占拓扑逻辑。

### 9.2 机内拓扑用多深

不做真实 NVML / `nvidia-smi topo`。模拟器里用 **两级域** 就够撑起 Demo，也和路线 4 的树对齐：

```
Node
  └─ NUMA / PCIe 岛     （跨岛 = 走 CPU / PCIe，慢）
       └─ NVLink 组      （组内 = NVLink 或 NVSwitch，快）
            └─ GPU
```

典型 8 卡机用两种预设即可：

| 预设 id | 结构 | 什么时候用 |
| --- | --- | --- |
| `nvswitch8`（默认） | 8 张卡一个 NVLink 组、一个 NUMA | 旧 `cluster_4x8.json` 不写拓扑时的行为，**现有测试应几乎不变** |
| `dual_numa8` | NUMA0：GPU 0–3（一个 NVLink 组）；NUMA1：GPU 4–7（另一个） | 新演示集群，用来看出机内差异 |
| `pair4` | 4 卡机：两组 NVLink pair（0–1、2–3） | 给 `cluster_mixed.json` 里的 4 卡节点 |

`dual_numa8` 是加分项的主展示：4 卡任务应整段放进 0–3 或 4–7，而不是 2,3,4,5（跨 NUMA，只靠 PCIe）。

**不要**在这一步做 DP / TP / PP。并行模式是下一档加分，应复用这里的域 id，而不是另造一套图。

### 9.3 数据模型（向后兼容）

在 `types.h` 给 GPU 打上域标签，Node 带拓扑预设名：

```cpp
struct Gpu {
    int index = 0;
    std::string occupier;
    int nvlink_group = 0;  // 同一 Node 内，组号相同 = NVLink/NVSwitch 互通
    int numa_id = 0;
};

struct Node {
    std::string id;
    std::string topology = "flat";  // flat | nvswitch8 | dual_numa8 | pair4 | custom
    std::vector<Gpu> gpus;
};
```

`Cluster::add_node` 建议扩成：

```text
add_node(id, gpu_count, topology = "")
```

- `topology` 为空或 `flat`：所有 GPU `nvlink_group = 0`、`numa_id = 0`（整机一块域）
- 具名预设：按上表填 GPU 标签
- 可选 `custom`：JSON 直接写每张卡的 `nvlink_group` / `numa_id`

集群 JSON 新旧都能读：

```json
{"id": "node-A", "gpu_count": 8}
{"id": "node-A", "gpu_count": 8, "topology": "dual_numa8"}
```

Python 绑定：`parse_nodes` 已能吃 `tuple` 和 `dict`（queue 同款）。给 dict 多认一个 `topology` 即可；`(id, gpu_count)` 仍走 `flat`。

`NodeView` / snapshot 要带上每张卡的 `nvlink_group`、`numa_id`，以及节点的 `topology`，Web 才能上色。

### 9.4 放置算法：只改 Topology-aware 的「节点内打包」和打分

跨 Node 流程（最少节点数、碎片等待、组合枚举）**原样保留**。改两处。

**（1）`pack_largest_first` 换成「先填满 NVLink 组」。**

选定节点集合后，不要再 `free_gpu_indices` 取下标最小的 k 张。每个 Node 内：

1. 把空闲 GPU 按 `nvlink_group` 分桶
2. 桶按空闲数从大到小
3. 先吃满一个组，不够再溢出到同 NUMA 的下一组，最后才跨 NUMA

First Fit 继续按 GPU index 顺序拿，用来对比「机内也被打散」。

**（2）打分加机内项，权重必须小于跨 Node。**

现有：

```text
score = 100·(n_nodes − 1) + 10·n_frag + 5·n_awkward
```

改为（数字可微调，顺序不要乱）：

```text
score = 100·(n_nodes − 1)
      +  40·(n_nvlink − 1)
      +  15·(n_numa − 1)
      +  10·n_frag
      +   5·n_awkward
```

- `n_nvlink`：本次放置覆盖的 `(node_id, nvlink_group)` 个数
- `n_numa`：覆盖的 `(node_id, numa_id)` 个数
- 跨 Node 仍是主目标：2 节点同组一定差于 1 节点跨 NUMA

**（3）碎片等待加一档「机内等待」，但更保守。**

现有：`min_nodes > ideal_nodes && has_running` → 等整机。

机内可以对称：

- `ideal_nvlink = ceil(本节点上拿到的卡数 / 该节点最大 NVLink 组容量)`
- 若已经能在理想节点数内放下，但 **每个候选节点都不得不跨 NVLink 组**，且有 running → 等
- 没有 running → 降级跨组，避免死锁（与现逻辑一致）

只对「单节点就能装下」的请求做机内等待（典型 2/4 卡）。8 卡打满整机时，`dual_numa8` 必然跨两个 NVLink 组，**不要等**。

原因字符串建议写清，便于 Demo：

- `Topology-aware：node-A GPU 4-7（同一 NVLink 组）`
- `机内碎片：4 卡需跨 NUMA，等待同组空出`

### 9.5 指标与展示

`Metrics` 现有 `cross_node_jobs`。并列加：

| 字段 | 含义 |
| --- | --- |
| `cross_nvlink_jobs` | 至少一个 Node 上占用了 ≥2 个 `nvlink_group` 的 running/finished 任务数 |
| `cross_numa_jobs` | 同上，但按 `numa_id` |

Web（`python/web/static/app.js` + `style.css`）：

- 节点卡按 `nvlink_group` 分色或分组（0–3 / 4–7 两排），不要只画 8 个无差别方块
- 放置文案带组号：`node-A: GPU 4,5,6,7 (nvlink 1)`
- 策略对比表加一列「跨 NVLink 任务」

Queue 条、配额编辑 **不动**。

### 9.6 建议改哪些文件

| 文件 | 改什么 |
| --- | --- |
| `cpp/include/types.h` | `Gpu` / `Node` / `NodeView` / `Metrics` 加域字段 |
| `cpp/include/cluster.h` + `cpp/src/cluster.cpp` | `add_node` 应用拓扑预设；`free_gpu_indices` 可保留；新增按组列举空闲卡 |
| `cpp/src/scheduler.cpp` | `pack_largest_first`、`score_placement`、可选机内等待、reason |
| `cpp/src/bindings.cpp` | 解析 `topology`；snapshot 输出组号 |
| `typings/gpu_scheduler.pyi` | nodes dict 允许 `topology` |
| `python/web/app.py` | 集群校验放行 `topology` 字段 |
| `python/web/static/app.js` / `style.css` | 节点编辑器可选拓扑预设；渲染分组 |
| `data/cluster_4x8_dual_numa.json` | **新建**：4×8，每台 `dual_numa8` |
| `data/jobs_nvlink.json` | **新建**：先占 0–1，再来 4 卡 / 2 卡，对比 FF vs TA |
| `tests/test_scheduler.py` | 见 9.8 |
| `README.md` | 去掉「不模拟 NVLink」这条限制，补预设与打分 |

不改：`QueueSpec`、队内抢占、fair-share / fragmentation pending 计数。

### 9.7 推荐实现顺序

1. **模型 + 兼容。** 标签默认全 0；旧测试、旧 JSON 全绿。
2. **节点内打包 + 打分。** 单测 `dual_numa8` 上 4 卡任务选整组。
3. **机内等待。** 构造「每组只剩 2 卡」时 4 卡任务 TA 等待、FF 立刻跨组。
4. **预设数据 + Web。** 新集群 / 新任务集；对比页能看出 `cross_nvlink_jobs`：TA → 0，FF → >0。
5. 最后才考虑作业级 `affinity: preferred|required`。第一期全局「尽量少跨 NVLink」即可。

### 9.8 最小测试（相对现有用例）

现有 `test_topology_aware_*`、quota、抢占应继续通过（默认 `flat` / `nvswitch8`）。新增：

1. **整组优先。** `dual_numa8` 节点先占 GPU 0–1；提交 4 卡 → TA 放到 4–7，不放到 2,3,4,5。
2. **First Fit 对照。** 同一初始状态，FF 拿走 2,3,4,5（或 2,3 再溢出），`cross_nvlink_jobs == 1`。
3. **机内等待。** 两组各占 2 卡、各剩 2 卡；4 卡 + 有 running → TA pending（原因含机内碎片）；`finish` 腾出一组后启动。
4. **8 卡不误等。** `dual_numa8` 整机 8 卡请求必须立刻放（必然跨两个 NVLink 组）。
5. **与 Quota 正交。** `jobs_quota.json` 换到 `dual_numa8` 集群：超配额仍是 fair-share pending；配额够但跨组才是 fragmentation。
6. **抢占不绕过机内目标。** 高优先级 4 卡在副本上 `try_place`，若释放后仍只能跨 NUMA 且有更好的受害者（整组低优先级 4 卡），应抢后者。

### 9.9 刻意不做

- 不把 Queue 升级成 HiveD cell（「research 配额 = 两台 dual_numa 整机」）。那是租户层，已用数量配额交卷。
- 不引入通信带宽矩阵、不模拟 NVLink 争用。域 id + 跨域计数足够讲清加分项。
- 不在 First Fit 里偷偷选 NVLink，否则对比被抹平。
- 不为这次改动重写 Web 交互页的 queue 编辑器。

### 9.10 做完之后 Demo 怎么讲

构造：一台 `dual_numa8`，先跑一个 2 卡任务占 GPU 0–1。再提交 4 卡训练任务。

- First Fit：拿走 2,3,4,5 → 跨 NUMA，训练走 PCIe。
- Topology-aware：放到 4–7 → 整组 NVLink；若 4–7 被占满而 2,3 和另一组碎片空着，则等待而不是跨组拼。

一句话：**跨 Node 规则不变，Node 内部从「任意空闲下标」改成「尽量待在同一个 NVLink 组」。**
