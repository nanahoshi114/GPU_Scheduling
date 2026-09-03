#include "scheduler.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

// TopologyAware 打分权重：跨节点通信代价远高于碎片，优先少跨 Node。
constexpr int kWeightCross = 100;
constexpr int kWeightFrag = 10;
constexpr int kWeightAwkward = 5;

std::string join_node_ids(const std::vector<NodeAllocation>& allocs) {
    std::ostringstream oss;
    for (size_t i = 0; i < allocs.size(); ++i) {
        if (i) {
            oss << ", ";
        }
        oss << allocs[i].node_id << "(" << allocs[i].gpu_indices.size() << " GPU)";
    }
    return oss.str();
}

bool is_awkward(int leftover) {
    return leftover == 3 || leftover == 5 || leftover == 6 || leftover == 7;
}

// score = 100*(跨节点数) + 10*(部分占用节点数) + 5*(余量落在 3/5/6/7 的节点数)
int score_placement(const Cluster& cluster, const std::vector<NodeAllocation>& allocs) {
    const int n_nodes = static_cast<int>(allocs.size());
    const int cross = std::max(0, n_nodes - 1);
    int n_frag = 0;
    int n_awkward = 0;
    for (const auto& alloc : allocs) {
        const int idx = cluster.node_index(alloc.node_id);
        const int free_before = cluster.free_count(idx);
        const int taken = static_cast<int>(alloc.gpu_indices.size());
        const int leftover = free_before - taken;
        if (leftover > 0) {
            n_frag++;
        }
        if (is_awkward(leftover)) {
            n_awkward++;
        }
    }
    return kWeightCross * cross + kWeightFrag * n_frag + kWeightAwkward * n_awkward;
}

// 在已选定的节点集合内，按空闲 GPU 从多到少填充，尽量先填满大节点。
Placement pack_largest_first(const Cluster& cluster, const std::vector<int>& node_indices,
                             int gpu_request) {
    struct Item {
        int idx;
        int free;
    };
    std::vector<Item> items;
    items.reserve(node_indices.size());
    for (int idx : node_indices) {
        items.push_back({idx, cluster.free_count(idx)});
    }
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) { return a.free > b.free; });

    Placement p;
    int remaining = gpu_request;
    for (const auto& item : items) {
        if (remaining <= 0) {
            break;
        }
        auto free_idxs = cluster.free_gpu_indices(item.idx);
        if (free_idxs.empty()) {
            continue;
        }
        const int take = std::min(remaining, static_cast<int>(free_idxs.size()));
        NodeAllocation alloc;
        alloc.node_id = cluster.nodes()[static_cast<size_t>(item.idx)].id;
        alloc.gpu_indices.assign(free_idxs.begin(),
                                 free_idxs.begin() + take);
        p.allocations.push_back(std::move(alloc));
        remaining -= take;
    }
    return p;
}

void foreach_combination(int n, int k, const std::function<void(const std::vector<int>&)>& cb) {
    if (k <= 0 || k > n) {
        return;
    }
    std::vector<int> comb(static_cast<size_t>(k));
    std::function<void(int, int)> rec = [&](int start, int depth) {
        if (depth == k) {
            cb(comb);
            return;
        }
        for (int i = start; i <= n - (k - depth); ++i) {
            comb[static_cast<size_t>(depth)] = i;
            rec(i + 1, depth + 1);
        }
    };
    rec(0, 0);
}

class FirstFitStrategy : public PlacementStrategy {
public:
    std::string name() const override { return "first_fit"; }

    // 基础策略：按 Node 定义顺序从头拿走空闲 GPU。总量够就立刻放置，不优化跨节点。
    ScheduleResult try_place(const Cluster& cluster, int gpu_request,
                             bool /*has_running*/) const override {
        ScheduleResult result;
        if (gpu_request <= 0) {
            result.reason = "请求 GPU 数量必须为正";
            return result;
        }
        const int free = cluster.free_gpus();
        if (free < gpu_request) {
            result.reason = "GPU 不足（需要 " + std::to_string(gpu_request) + "，空闲 " +
                            std::to_string(free) + "）";
            return result;
        }

        Placement p;
        int remaining = gpu_request;
        for (int i = 0; i < cluster.node_count() && remaining > 0; ++i) {
            auto free_idxs = cluster.free_gpu_indices(i);
            if (free_idxs.empty()) {
                continue;
            }
            const int take = std::min(remaining, static_cast<int>(free_idxs.size()));
            NodeAllocation alloc;
            alloc.node_id = cluster.nodes()[static_cast<size_t>(i)].id;
            alloc.gpu_indices.assign(free_idxs.begin(), free_idxs.begin() + take);
            p.allocations.push_back(std::move(alloc));
            remaining -= take;
        }
        if (remaining > 0) {
            result.reason = "GPU 不足（需要 " + std::to_string(gpu_request) + "，空闲 " +
                            std::to_string(free) + "）";
            return result;
        }
        p.reason = "FirstFit：按节点定义顺序依次占用空闲 GPU，使用节点 " +
                   join_node_ids(p.allocations) + "。只要空闲总量足够即立即放置，不等待更集中的打包。";
        result.success = true;
        result.placement = std::move(p);
        result.reason = result.placement.reason;
        return result;
    }
};

class TopologyAwareStrategy : public PlacementStrategy {
public:
    std::string name() const override { return "topology_aware"; }

    // 拓扑感知：先求当前最少跨 Node 数；若明显差于理想值且有任务在跑则等待；
    // 否则在该节点数的组合中选碎片代价最低的放置。
    ScheduleResult try_place(const Cluster& cluster, int gpu_request,
                             bool has_running) const override {
        ScheduleResult result;
        if (gpu_request <= 0) {
            result.reason = "请求 GPU 数量必须为正";
            return result;
        }
        const int free = cluster.free_gpus();
        if (free < gpu_request) {
            result.reason = "GPU 不足（需要 " + std::to_string(gpu_request) + "，空闲 " +
                            std::to_string(free) + "）";
            return result;
        }

        struct Cand {
            int idx;
            int free;
        };
        std::vector<Cand> candidates;
        for (int i = 0; i < cluster.node_count(); ++i) {
            const int f = cluster.free_count(i);
            if (f > 0) {
                candidates.push_back({i, f});
            }
        }
        if (candidates.empty()) {
            result.reason = "GPU 不足（需要 " + std::to_string(gpu_request) + "，空闲 0）";
            return result;
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Cand& a, const Cand& b) { return a.free > b.free; });

        // 贪心下界：用空闲最多的节点去覆盖请求，得到当前最少跨 Node 数。
        int covered = 0;
        int min_nodes = 0;
        for (const auto& c : candidates) {
            if (covered >= gpu_request) {
                break;
            }
            covered += c.free;
            min_nodes++;
        }
        if (covered < gpu_request) {
            result.reason = "GPU 不足（需要 " + std::to_string(gpu_request) + "，空闲 " +
                            std::to_string(free) + "）";
            return result;
        }

        const int max_cap = std::max(1, cluster.max_node_capacity());
        const int ideal_nodes =
            (gpu_request + max_cap - 1) / max_cap;  // ceil(G / 单 Node 容量)

        // 碎片等待：现在必须拆到比理想更多的节点，且稍后可能释放出整机，则先不调度。
        if (min_nodes > ideal_nodes && has_running) {
            result.reason =
                "资源碎片：当前最少需要跨 " + std::to_string(min_nodes) +
                " 个节点（理想为 " + std::to_string(ideal_nodes) +
                " 个）。存在运行中任务，等待 GPU 释放后可更集中放置，避免过度跨 Node 通信。";
            return result;
        }

        const bool degraded = min_nodes > ideal_nodes && !has_running;

        // Limit search space on large clusters: keep the roomiest nodes.
        const int keep = std::min(static_cast<int>(candidates.size()),
                                  std::max(min_nodes + 6, 12));
        if (static_cast<int>(candidates.size()) > keep) {
            candidates.resize(static_cast<size_t>(keep));
        }
        const int n = static_cast<int>(candidates.size());

        int best_score = std::numeric_limits<int>::max();
        Placement best;
        bool found = false;

        auto consider = [&](const std::vector<int>& pick_local) {
            std::vector<int> node_indices;
            node_indices.reserve(pick_local.size());
            int sum_free = 0;
            for (int local : pick_local) {
                node_indices.push_back(candidates[static_cast<size_t>(local)].idx);
                sum_free += candidates[static_cast<size_t>(local)].free;
            }
            if (sum_free < gpu_request) {
                return;
            }
            Placement p = pack_largest_first(cluster, node_indices, gpu_request);
            if (p.allocations.empty()) {
                return;
            }
            int placed = 0;
            for (const auto& a : p.allocations) {
                placed += static_cast<int>(a.gpu_indices.size());
            }
            if (placed < gpu_request) {
                return;
            }
            const int sc = score_placement(cluster, p.allocations);
            if (!found || sc < best_score) {
                best_score = sc;
                best = std::move(p);
                found = true;
            }
        };

        // 枚举恰好 min_nodes 个节点的组合，打分后取最低分。
        foreach_combination(n, min_nodes, consider);
        if (!found) {
            // Fallback: greedy largest min_nodes.
            std::vector<int> greedy(static_cast<size_t>(min_nodes));
            for (int i = 0; i < min_nodes; ++i) {
                greedy[static_cast<size_t>(i)] = i;
            }
            consider(greedy);
        }

        if (!found) {
            result.reason = "未能找到可行放置";
            return result;
        }

        std::ostringstream reason;
        reason << "TopologyAware：优先将任务放在最少的节点上以降低跨 Node 通信；"
               << "在跨 " << min_nodes << " 个节点的可行方案中选择碎片代价最低的放置。使用节点 "
               << join_node_ids(best.allocations) << "。";
        if (degraded) {
            reason << " 当前无运行中任务，为避免死锁在碎片状态下降级放置"
                   << "（理想节点数 " << ideal_nodes << "）。";
        }
        best.reason = reason.str();
        result.success = true;
        result.placement = std::move(best);
        result.reason = result.placement.reason;
        return result;
    }
};

}  // namespace

std::unique_ptr<PlacementStrategy> make_strategy(const std::string& name) {
    if (name == "first_fit" || name == "FirstFit" || name == "ff") {
        return std::make_unique<FirstFitStrategy>();
    }
    if (name == "topology_aware" || name == "TopologyAware" || name == "ta") {
        return std::make_unique<TopologyAwareStrategy>();
    }
    throw std::invalid_argument("unknown strategy: " + name +
                                " (expected first_fit or topology_aware)");
}

Scheduler::Scheduler(std::vector<std::pair<std::string, int>> nodes, const std::string& strategy) {
    if (nodes.empty()) {
        throw std::invalid_argument("cluster must contain at least one node");
    }
    for (const auto& n : nodes) {
        cluster_.add_node(n.first, n.second);
    }
    strategy_ = make_strategy(strategy);
    strategy_name_ = strategy_->name();
}

ScheduleResult Scheduler::submit(const JobSpec& spec) {
    ScheduleResult result;
    if (spec.id.empty()) {
        result.reason = "任务 id 不能为空";
        return result;
    }
    for (const auto& job : jobs_) {
        if (job.id == spec.id) {
            result.reason = "任务 id 已存在: " + spec.id;
            return result;
        }
    }
    if (spec.gpu_request <= 0) {
        result.reason = "请求 GPU 数量必须为正";
        return result;
    }
    if (spec.duration <= 0) {
        result.reason = "任务时长必须为正";
        return result;
    }

    Job job;
    job.id = spec.id;
    job.gpu_request = spec.gpu_request;
    job.duration = spec.duration;
    job.arrival_time = spec.arrival_time >= 0 ? spec.arrival_time : time_;
    job.state = JobState::Pending;

    bool has_running = false;
    for (const auto& j : jobs_) {
        if (j.state == JobState::Running) {
            has_running = true;
            break;
        }
    }

    // 按构造时选定的策略决定放置或等待（实现见本文件 FirstFit / TopologyAware）。
    auto placed = strategy_->try_place(cluster_, job.gpu_request, has_running);
    if (placed.success) {
        apply_placement(job, placed);
        result = placed;
    } else {
        job.reason = placed.reason;
        result = placed;
    }
    jobs_.push_back(std::move(job));
    update_max_pending();
    return result;
}

bool Scheduler::finish(const std::string& job_id) {
    for (auto& job : jobs_) {
        if (job.id != job_id) {
            continue;
        }
        if (job.state != JobState::Running) {
            return false;
        }
        cluster_.release(job_id);
        job.state = JobState::Finished;
        job.finish_time = time_;
        schedule_pending();
        return true;
    }
    return false;
}

Snapshot Scheduler::tick() {
    time_++;
    std::vector<std::string> due;
    for (const auto& job : jobs_) {
        if (job.state == JobState::Running && job.start_time >= 0 &&
            job.start_time + job.duration <= time_) {
            due.push_back(job.id);
        }
    }
    for (const auto& id : due) {
        finish(id);
    }
    schedule_pending();
    return snapshot();
}

void Scheduler::schedule_pending() {
    bool has_running = false;
    for (const auto& j : jobs_) {
        if (j.state == JobState::Running) {
            has_running = true;
            break;
        }
    }
    // FIFO: jobs are stored in submission order.
    bool progressed = true;
    while (progressed) {
        progressed = false;
        has_running = false;
        for (const auto& j : jobs_) {
            if (j.state == JobState::Running) {
                has_running = true;
                break;
            }
        }
        for (auto& job : jobs_) {
            if (job.state != JobState::Pending) {
                continue;
            }
            auto placed = strategy_->try_place(cluster_, job.gpu_request, has_running);
            if (placed.success) {
                apply_placement(job, placed);
                progressed = true;
                has_running = true;
            } else {
                job.reason = placed.reason;
            }
        }
    }
    update_max_pending();
}

void Scheduler::apply_placement(Job& job, const ScheduleResult& result) {
    cluster_.allocate(job.id, result.placement.allocations);
    job.state = JobState::Running;
    job.placement = result.placement;
    job.reason = result.reason;
    job.start_time = time_;
    job.wait_time = std::max(0, time_ - job.arrival_time);
}

JobView Scheduler::to_view(const Job& job) const {
    JobView v;
    v.id = job.id;
    v.gpu_request = job.gpu_request;
    v.duration = job.duration;
    v.arrival_time = job.arrival_time;
    switch (job.state) {
        case JobState::Pending:
            v.state = "pending";
            break;
        case JobState::Running:
            v.state = "running";
            break;
        case JobState::Finished:
            v.state = "finished";
            break;
    }
    v.start_time = job.start_time;
    v.finish_time = job.finish_time;
    if (job.state == JobState::Pending) {
        v.wait_time = std::max(0, time_ - job.arrival_time);
    } else {
        v.wait_time = job.wait_time;
    }
    v.placement = job.placement.allocations;
    v.reason = job.reason;
    v.nodes_used = static_cast<int>(job.placement.allocations.size());
    return v;
}

void Scheduler::update_max_pending() {
    int pending = 0;
    for (const auto& job : jobs_) {
        if (job.state == JobState::Pending) {
            pending++;
        }
    }
    max_pending_ = std::max(max_pending_, pending);
}

Metrics Scheduler::metrics() const {
    return metrics_with_avg(-1.0, time_, max_pending_);
}

Metrics Scheduler::metrics_with_avg(double avg_utilization, int makespan, int max_pending) const {
    Metrics m;
    m.total_gpus = cluster_.total_gpus();
    m.used_gpus = cluster_.used_gpus();
    m.gpu_utilization =
        m.total_gpus > 0 ? static_cast<double>(m.used_gpus) / static_cast<double>(m.total_gpus)
                         : 0.0;
    m.avg_utilization = avg_utilization >= 0 ? avg_utilization : m.gpu_utilization;
    m.makespan = makespan;
    m.max_pending = max_pending;

    double wait_sum = 0.0;
    int wait_n = 0;
    for (const auto& job : jobs_) {
        if (job.state == JobState::Pending) {
            m.pending_count++;
            wait_sum += static_cast<double>(std::max(0, time_ - job.arrival_time));
            wait_n++;
        } else if (job.state == JobState::Running) {
            m.running_count++;
            wait_sum += static_cast<double>(job.wait_time);
            wait_n++;
            if (job.placement.allocations.size() > 1) {
                m.cross_node_jobs++;
            }
        } else {
            m.finished_count++;
            wait_sum += static_cast<double>(job.wait_time);
            wait_n++;
            if (job.placement.allocations.size() > 1) {
                m.cross_node_jobs++;
            }
        }
    }
    m.avg_wait_time = wait_n > 0 ? wait_sum / static_cast<double>(wait_n) : 0.0;
    return m;
}

Snapshot Scheduler::snapshot() const {
    Snapshot s;
    s.time = time_;
    s.strategy = strategy_name_;
    s.nodes = cluster_.view();
    s.jobs.reserve(jobs_.size());
    for (const auto& job : jobs_) {
        s.jobs.push_back(to_view(job));
    }
    s.metrics = metrics();
    return s;
}
