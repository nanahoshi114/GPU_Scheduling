#include "scheduler.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

std::string normalize_parallelism(const std::string& raw) {
    std::string s = raw;
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    if (s.empty() || s == "dp" || s == "data") {
        return "dp";
    }
    if (s == "tp" || s == "tensor") {
        return "tp";
    }
    if (s == "pp" || s == "pipeline") {
        return "pp";
    }
    return "";
}

namespace {

// TopologyAware 打分权重：跨 Node 必须最大，其次跨 NVLink 组，再跨 NUMA。
constexpr int kWeightCross = 100;
constexpr int kWeightNvlink = 40;
constexpr int kWeightNuma = 15;
constexpr int kWeightFrag = 10;
constexpr int kWeightAwkward = 5;

bool is_awkward(int leftover) {
    return leftover == 3 || leftover == 5 || leftover == 6 || leftover == 7;
}

bool is_waiting(JobState state) {
    return state == JobState::Pending || state == JobState::Preempted;
}

int allocated_gpus(const Job& job) {
    int count = 0;
    for (const auto& allocation : job.placement.allocations) {
        count += static_cast<int>(allocation.gpu_indices.size());
    }
    return count;
}

int count_nvlink_domains(const Cluster& cluster, const std::vector<NodeAllocation>& allocs) {
    std::set<std::pair<std::string, int>> domains;
    for (const auto& alloc : allocs) {
        const int idx = cluster.node_index(alloc.node_id);
        if (idx < 0) {
            continue;
        }
        const auto& node = cluster.nodes()[static_cast<size_t>(idx)];
        for (int g : alloc.gpu_indices) {
            if (g >= 0 && g < static_cast<int>(node.gpus.size())) {
                domains.insert({alloc.node_id, node.gpus[static_cast<size_t>(g)].nvlink_group});
            }
        }
    }
    return static_cast<int>(domains.size());
}

int count_numa_domains(const Cluster& cluster, const std::vector<NodeAllocation>& allocs) {
    std::set<std::pair<std::string, int>> domains;
    for (const auto& alloc : allocs) {
        const int idx = cluster.node_index(alloc.node_id);
        if (idx < 0) {
            continue;
        }
        const auto& node = cluster.nodes()[static_cast<size_t>(idx)];
        for (int g : alloc.gpu_indices) {
            if (g >= 0 && g < static_cast<int>(node.gpus.size())) {
                domains.insert({alloc.node_id, node.gpus[static_cast<size_t>(g)].numa_id});
            }
        }
    }
    return static_cast<int>(domains.size());
}

bool job_crosses_nvlink(const Cluster& cluster, const Job& job) {
    for (const auto& alloc : job.placement.allocations) {
        const int idx = cluster.node_index(alloc.node_id);
        if (idx < 0) {
            continue;
        }
        std::unordered_set<int> groups;
        const auto& node = cluster.nodes()[static_cast<size_t>(idx)];
        for (int g : alloc.gpu_indices) {
            if (g >= 0 && g < static_cast<int>(node.gpus.size())) {
                groups.insert(node.gpus[static_cast<size_t>(g)].nvlink_group);
            }
        }
        if (groups.size() >= 2) {
            return true;
        }
    }
    return false;
}

bool job_crosses_numa(const Cluster& cluster, const Job& job) {
    for (const auto& alloc : job.placement.allocations) {
        const int idx = cluster.node_index(alloc.node_id);
        if (idx < 0) {
            continue;
        }
        std::unordered_set<int> numas;
        const auto& node = cluster.nodes()[static_cast<size_t>(idx)];
        for (int g : alloc.gpu_indices) {
            if (g >= 0 && g < static_cast<int>(node.gpus.size())) {
                numas.insert(node.gpus[static_cast<size_t>(g)].numa_id);
            }
        }
        if (numas.size() >= 2) {
            return true;
        }
    }
    return false;
}

// score = 100*(n_nodes-1) + 40*(n_nvlink-1) + 15*(n_numa-1) + 10*n_frag + 5*n_awkward
int score_placement(const Cluster& cluster, const std::vector<NodeAllocation>& allocs) {
    const int n_nodes = static_cast<int>(allocs.size());
    const int n_nvlink = std::max(1, count_nvlink_domains(cluster, allocs));
    const int n_numa = std::max(1, count_numa_domains(cluster, allocs));
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
    return kWeightCross * std::max(0, n_nodes - 1) +
           kWeightNvlink * (n_nvlink - 1) + kWeightNuma * (n_numa - 1) +
           kWeightFrag * n_frag + kWeightAwkward * n_awkward;
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
        auto free_idxs = cluster.free_gpu_indices_by_topology(item.idx);
        if (free_idxs.empty()) {
            continue;
        }
        const int take = std::min(remaining, static_cast<int>(free_idxs.size()));
        NodeAllocation alloc;
        alloc.node_id = cluster.nodes()[static_cast<size_t>(item.idx)].id;
        alloc.gpu_indices.assign(free_idxs.begin(), free_idxs.begin() + take);
        p.allocations.push_back(std::move(alloc));
        remaining -= take;
    }
    return p;
}

Placement pack_in_node_order(const Cluster& cluster, int left, int right, int gpu_request) {
    Placement p;
    int remaining = gpu_request;
    for (int idx = left; idx <= right && remaining > 0; ++idx) {
        auto free_idxs = cluster.free_gpu_indices_by_topology(idx);
        if (free_idxs.empty()) {
            continue;
        }
        const int take = std::min(remaining, static_cast<int>(free_idxs.size()));
        NodeAllocation alloc;
        alloc.node_id = cluster.nodes()[static_cast<size_t>(idx)].id;
        alloc.gpu_indices.assign(free_idxs.begin(), free_idxs.begin() + take);
        p.allocations.push_back(std::move(alloc));
        remaining -= take;
    }
    return p;
}

int placed_count(const Placement& p) {
    int n = 0;
    for (const auto& a : p.allocations) {
        n += static_cast<int>(a.gpu_indices.size());
    }
    return n;
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
    ScheduleResult try_place(const Cluster& cluster, const PlaceRequest& req,
                             bool /*has_running*/) const override {
        ScheduleResult result;
        const int gpu_request = req.gpu_request;
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
        p.reason = "First Fit：立即放置";
        result.success = true;
        result.placement = std::move(p);
        result.reason = result.placement.reason;
        return result;
    }
};

class TopologyAwareStrategy : public PlacementStrategy {
public:
    std::string name() const override { return "topology_aware"; }

    ScheduleResult try_place(const Cluster& cluster, const PlaceRequest& req,
                             bool has_running) const override {
        ScheduleResult result;
        const int gpu_request = req.gpu_request;
        if (gpu_request <= 0) {
            result.reason = "请求 GPU 数量必须为正";
            return result;
        }
        const std::string mode = normalize_parallelism(req.parallelism);
        if (mode.empty()) {
            result.reason = "未知并行方式";
            return result;
        }
        const int free = cluster.free_gpus();
        if (free < gpu_request) {
            result.reason = "GPU 不足（需要 " + std::to_string(gpu_request) + "，空闲 " +
                            std::to_string(free) + "）";
            return result;
        }
        if (mode == "tp") {
            return place_tp(cluster, gpu_request, has_running);
        }
        if (mode == "pp") {
            return place_pp(cluster, gpu_request, has_running);
        }
        return place_dp(cluster, gpu_request, has_running);
    }

private:
    // DP：最少节点 + 跨 Node / 机内碎片等待（原 Topology-aware）。
    ScheduleResult place_dp(const Cluster& cluster, int gpu_request, bool has_running) const {
        ScheduleResult result;
        const int free = cluster.free_gpus();

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
        const int ideal_nodes = (gpu_request + max_cap - 1) / max_cap;

        if (min_nodes > ideal_nodes && has_running) {
            result.reason = "资源碎片：需跨 " + std::to_string(min_nodes) +
                            " 个节点（理想 " + std::to_string(ideal_nodes) + "），等待集中放置";
            return result;
        }

        const bool degraded = min_nodes > ideal_nodes && !has_running;

        bool intra_degraded = false;
        if (ideal_nodes == 1 && min_nodes == 1) {
            bool any_ideal_group = false;
            for (const auto& c : candidates) {
                if (c.free < gpu_request) {
                    continue;
                }
                Placement trial = pack_largest_first(cluster, {c.idx}, gpu_request);
                const int nvl = count_nvlink_domains(cluster, trial.allocations);
                const int max_group = cluster.max_nvlink_group_size(c.idx);
                const int ideal_nvlink = (gpu_request + max_group - 1) / max_group;
                if (nvl <= ideal_nvlink) {
                    any_ideal_group = true;
                    break;
                }
            }
            if (!any_ideal_group) {
                if (has_running) {
                    result.reason = "机内碎片：" + std::to_string(gpu_request) +
                                    " 卡需跨 NVLink 组，等待同组空出";
                    return result;
                }
                intra_degraded = true;
            }
        }

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
            if (placed_count(p) < gpu_request) {
                return;
            }
            const int sc = score_placement(cluster, p.allocations);
            if (!found || sc < best_score) {
                best_score = sc;
                best = std::move(p);
                found = true;
            }
        };

        foreach_combination(n, min_nodes, consider);
        if (!found) {
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

        if (degraded || intra_degraded) {
            best.reason = "Topology-aware：降级放置，避免死锁";
        } else if (count_nvlink_domains(cluster, best.allocations) == 1) {
            best.reason = "Topology-aware：DP 同一 NVLink 组";
        } else {
            best.reason = "Topology-aware：DP 最少节点放置";
        }
        result.success = true;
        result.placement = std::move(best);
        result.reason = result.placement.reason;
        return result;
    }

    // TP：整份请求必须落在同一个 NVLink 组。
    ScheduleResult place_tp(const Cluster& cluster, int gpu_request, bool has_running) const {
        ScheduleResult result;
        const int max_group = cluster.max_nvlink_group_size();
        if (gpu_request > max_group) {
            result.reason = "TP 任务申请 " + std::to_string(gpu_request) +
                            " GPU 超过最大 NVLink 组容量 " + std::to_string(max_group);
            return result;
        }

        struct Bucket {
            int node_idx;
            int group;
            int leftover;
            std::vector<int> indices;
        };
        std::vector<Bucket> fit;
        for (int i = 0; i < cluster.node_count(); ++i) {
            const auto& node = cluster.nodes()[static_cast<size_t>(i)];
            std::map<int, std::vector<int>> by_group;
            for (const auto& gpu : node.gpus) {
                if (gpu.occupier.empty()) {
                    by_group[gpu.nvlink_group].push_back(gpu.index);
                }
            }
            for (auto& kv : by_group) {
                if (static_cast<int>(kv.second.size()) >= gpu_request) {
                    Bucket b;
                    b.node_idx = i;
                    b.group = kv.first;
                    b.leftover = static_cast<int>(kv.second.size()) - gpu_request;
                    b.indices = std::move(kv.second);
                    fit.push_back(std::move(b));
                }
            }
        }

        if (fit.empty()) {
            if (has_running) {
                result.reason = "TP 需同一 NVLink 组，等待同组空出";
            } else {
                result.reason = "未能找到可行放置";
            }
            return result;
        }

        std::sort(fit.begin(), fit.end(), [](const Bucket& a, const Bucket& b) {
            if (a.leftover != b.leftover) {
                return a.leftover < b.leftover;
            }
            if (a.node_idx != b.node_idx) {
                return a.node_idx < b.node_idx;
            }
            return a.group < b.group;
        });

        const Bucket& best = fit.front();
        Placement p;
        NodeAllocation alloc;
        alloc.node_id = cluster.nodes()[static_cast<size_t>(best.node_idx)].id;
        alloc.gpu_indices.assign(best.indices.begin(),
                                 best.indices.begin() + gpu_request);
        p.allocations.push_back(std::move(alloc));
        p.reason = "Topology-aware：TP 同一 NVLink 组";
        result.success = true;
        result.placement = std::move(p);
        result.reason = result.placement.reason;
        return result;
    }

    // PP：占用节点必须是集群下标上的连续窗口；不因 min_nodes > ideal 而等待。
    ScheduleResult place_pp(const Cluster& cluster, int gpu_request, bool has_running) const {
        ScheduleResult result;
        const int n = cluster.node_count();
        int best_span = std::numeric_limits<int>::max();
        int best_left = -1;
        for (int left = 0; left < n; ++left) {
            if (cluster.free_count(left) <= 0) {
                continue;
            }
            int sum = 0;
            for (int right = left; right < n; ++right) {
                const int f = cluster.free_count(right);
                if (f <= 0) {
                    break;
                }
                sum += f;
                if (sum >= gpu_request) {
                    const int span = right - left;
                    if (span < best_span || (span == best_span && left < best_left)) {
                        best_span = span;
                        best_left = left;
                    }
                    break;
                }
            }
        }

        if (best_left < 0) {
            if (has_running) {
                result.reason = "PP 需相邻节点，等待连续空位";
            } else {
                result.reason = "未能找到可行放置";
            }
            return result;
        }

        int right = best_left;
        int sum = 0;
        while (right < n) {
            sum += cluster.free_count(right);
            if (sum >= gpu_request) {
                break;
            }
            ++right;
        }
        Placement p = pack_in_node_order(cluster, best_left, right, gpu_request);
        if (placed_count(p) < gpu_request) {
            result.reason = "未能找到可行放置";
            return result;
        }
        p.reason = "Topology-aware：PP 相邻节点放置";
        result.success = true;
        result.placement = std::move(p);
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

Scheduler::Scheduler(std::vector<NodeInit> nodes, const std::string& strategy,
                     bool enable_preemption, std::vector<QueueSpec> queues)
    : enable_preemption_(enable_preemption) {
    if (nodes.empty()) {
        throw std::invalid_argument("cluster must contain at least one node");
    }
    for (const auto& n : nodes) {
        cluster_.add_node(n.id, n.gpu_count, n.topology);
    }
    strategy_ = make_strategy(strategy);
    strategy_name_ = strategy_->name();

    if (queues.empty()) {
        queues_.push_back({"default", cluster_.total_gpus()});
        return;
    }
    std::unordered_set<std::string> seen;
    for (const auto& queue : queues) {
        if (queue.id.empty()) {
            throw std::invalid_argument("queue id 不能为空");
        }
        if (queue.gpu_quota <= 0) {
            throw std::invalid_argument("队列 " + queue.id + " 的 gpu_quota 必须为正");
        }
        if (!seen.insert(queue.id).second) {
            throw std::invalid_argument("队列 id 重复: " + queue.id);
        }
        queues_.push_back(queue);
    }
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
    if (spec.priority < 0) {
        result.reason = "任务优先级必须为非负整数";
        return result;
    }

    const std::string parallelism = normalize_parallelism(spec.parallelism);
    if (parallelism.empty()) {
        result.reason = "未知并行方式";
        return result;
    }
    if (parallelism == "tp") {
        const int max_group = cluster_.max_nvlink_group_size();
        if (spec.gpu_request > max_group) {
            result.reason = "TP 任务申请 " + std::to_string(spec.gpu_request) +
                            " GPU 超过最大 NVLink 组容量 " + std::to_string(max_group);
            return result;
        }
    }

    const std::string queue_id = spec.queue_id.empty() ? "default" : spec.queue_id;
    const QueueSpec* queue = find_queue(queue_id);
    if (queue == nullptr) {
        result.reason = "未知队列: " + queue_id;
        return result;
    }
    if (spec.gpu_request > queue->gpu_quota) {
        result.reason = "任务申请 " + std::to_string(spec.gpu_request) + " GPU 超过队列 " +
                        queue_id + " 配额 " + std::to_string(queue->gpu_quota);
        return result;
    }

    Job job;
    job.id = spec.id;
    job.gpu_request = spec.gpu_request;
    job.duration = spec.duration;
    job.arrival_time = spec.arrival_time >= 0 ? spec.arrival_time : time_;
    job.priority = spec.priority;
    job.queue_id = queue_id;
    job.parallelism = parallelism;
    job.state = JobState::Pending;
    job.remaining_duration = spec.duration;
    job.pending_since = time_;

    jobs_.push_back(std::move(job));
    result = try_schedule_job(jobs_.back());
    if (!result.success) {
        jobs_.back().reason = result.reason;
    }
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
        job.remaining_duration = 0;
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
            job.start_time + job.remaining_duration <= time_) {
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
    // 每轮按优先级降序处理；同优先级保持 jobs_ 中的提交顺序（stable_sort）。
    bool progressed = true;
    while (progressed) {
        progressed = false;
        std::vector<Job*> waiting;
        for (auto& job : jobs_) {
            if (is_waiting(job.state)) {
                waiting.push_back(&job);
            }
        }
        std::stable_sort(waiting.begin(), waiting.end(), [](const Job* a, const Job* b) {
            if (a->priority != b->priority) {
                return a->priority > b->priority;
            }
            return a->arrival_time < b->arrival_time;
        });

        for (Job* job : waiting) {
            if (!is_waiting(job->state)) {
                continue;  // 可能已在本轮较早的资源变化后启动。
            }
            auto placed = try_schedule_job(*job);
            if (placed.success) {
                progressed = true;
            } else {
                job->reason = placed.reason;
            }
        }
    }
    update_max_pending();
}

bool Scheduler::has_running_jobs(const std::vector<std::string>& excluded) const {
    const std::unordered_set<std::string> excluded_set(excluded.begin(), excluded.end());
    for (const auto& job : jobs_) {
        if (job.state == JobState::Running && excluded_set.count(job.id) == 0) {
            return true;
        }
    }
    return false;
}

ScheduleResult Scheduler::find_preemption_plan(const Job& incoming) const {
    struct Victim {
        const Job* job;
        int released;
        int elapsed;
    };

    std::vector<Victim> eligible;
    for (const auto& job : jobs_) {
        if (job.state == JobState::Running && job.priority < incoming.priority &&
            job.queue_id == incoming.queue_id) {
            eligible.push_back(
                {&job, allocated_gpus(job), std::max(0, time_ - job.start_time)});
        }
    }
    std::sort(eligible.begin(), eligible.end(), [](const Victim& a, const Victim& b) {
        if (a.job->priority != b.job->priority) {
            return a.job->priority < b.job->priority;
        }
        if (a.released != b.released) {
            return a.released > b.released;
        }
        if (a.elapsed != b.elapsed) {
            return a.elapsed < b.elapsed;
        }
        return a.job->id < b.job->id;
    });

    ScheduleResult best;
    if (!enable_preemption_ || eligible.empty()) {
        return best;
    }

    // 按受害任务数量从少到多枚举。首个有可行解的 k 即保证受害者数量最少；
    // 同一 k 内比较总优先级、过量释放、已运行时间和最终节点数。
    for (int victim_count = 1; victim_count <= static_cast<int>(eligible.size());
         ++victim_count) {
        int max_releasable = 0;
        std::vector<int> released_sizes;
        released_sizes.reserve(eligible.size());
        for (const auto& victim : eligible) {
            released_sizes.push_back(victim.released);
        }
        std::sort(released_sizes.begin(), released_sizes.end(), std::greater<int>());
        for (int i = 0; i < victim_count; ++i) {
            max_releasable += released_sizes[static_cast<size_t>(i)];
        }
        if (cluster_.free_gpus() + max_releasable < incoming.gpu_request) {
            continue;
        }

        bool found_for_count = false;
        std::tuple<int, int, int, int, std::vector<std::string>> best_key;
        std::vector<int> picked;

        const int queue_used = queue_used_gpus(incoming.queue_id);
        const QueueSpec* queue = find_queue(incoming.queue_id);
        const int quota = queue == nullptr ? 0 : queue->gpu_quota;

        std::function<void(int, int)> search = [&](int start, int remaining) {
            if (remaining == 0) {
                Cluster trial = cluster_;
                std::vector<std::string> victim_ids;
                int total_priority = 0;
                int total_elapsed = 0;
                int released = 0;
                for (int index : picked) {
                    const Victim& victim = eligible[static_cast<size_t>(index)];
                    victim_ids.push_back(victim.job->id);
                    total_priority += victim.job->priority;
                    total_elapsed += victim.elapsed;
                    released += victim.released;
                    trial.release(victim.job->id);
                }

                if (queue_used - released + incoming.gpu_request > quota) {
                    return;
                }
                if (trial.free_gpus() < incoming.gpu_request) {
                    return;
                }
                auto placement = strategy_->try_place(
                    trial, PlaceRequest{incoming.gpu_request, incoming.parallelism},
                    has_running_jobs(victim_ids));
                if (!placement.success) {
                    return;
                }

                std::sort(victim_ids.begin(), victim_ids.end());
                const int over_release =
                    trial.free_gpus() - incoming.gpu_request;
                const int nodes_used =
                    static_cast<int>(placement.placement.allocations.size());
                auto key = std::make_tuple(total_priority, over_release, total_elapsed,
                                           nodes_used, victim_ids);
                if (!found_for_count || key < best_key) {
                    found_for_count = true;
                    best_key = key;
                    best = std::move(placement);
                    best.victims = victim_ids;

                    std::ostringstream reason;
                    reason << "抢占";
                    for (size_t i = 0; i < victim_ids.size(); ++i) {
                        reason << (i ? "、" : " ") << victim_ids[i];
                    }
                    best.reason = reason.str();
                    best.placement.reason = best.reason;
                }
                return;
            }

            for (int i = start;
                 i <= static_cast<int>(eligible.size()) - remaining; ++i) {
                picked.push_back(i);
                search(i + 1, remaining - 1);
                picked.pop_back();
            }
        };

        search(0, victim_count);
        if (found_for_count) {
            return best;
        }
    }
    return best;
}

void Scheduler::preempt_victims(const Job& incoming,
                                const std::vector<std::string>& victim_ids) {
    const std::unordered_set<std::string> victims(victim_ids.begin(), victim_ids.end());
    for (auto& victim : jobs_) {
        if (victims.count(victim.id) == 0 || victim.state != JobState::Running) {
            continue;
        }

        const int elapsed = std::max(0, time_ - victim.start_time);
        victim.remaining_duration = std::max(1, victim.remaining_duration - elapsed);
        cluster_.release(victim.id);
        victim.state = JobState::Preempted;
        victim.placement.allocations.clear();
        victim.placement.reason.clear();
        victim.start_time = -1;
        victim.pending_since = time_;
        victim.preemption_count++;
        victim.reason = "被 " + incoming.id + " 抢占";
    }
}

ScheduleResult Scheduler::try_schedule_job(Job& job) {
    const QueueSpec* queue = find_queue(job.queue_id);
    const int used = queue_used_gpus(job.queue_id);
    const int quota = queue == nullptr ? 0 : queue->gpu_quota;
    const bool quota_ok = used + job.gpu_request <= quota;

    ScheduleResult placed;
    if (quota_ok) {
        placed = strategy_->try_place(
            cluster_, PlaceRequest{job.gpu_request, job.parallelism}, has_running_jobs());
        if (placed.success) {
            apply_placement(job, placed);
            return placed;
        }
    } else {
        placed.reason = "队列 " + job.queue_id + " 配额不足（已用 " + std::to_string(used) +
                        "/" + std::to_string(quota) + "，申请 " +
                        std::to_string(job.gpu_request) + "）";
    }

    const std::string direct_failure = placed.reason;
    if (enable_preemption_ && job.priority > 0) {
        auto preemption = find_preemption_plan(job);
        if (preemption.success) {
            preempt_victims(job, preemption.victims);
            apply_placement(job, preemption);
            return preemption;
        }
    }

    if (enable_preemption_ && job.priority > 0 && has_running_jobs()) {
        placed.reason = direct_failure + "；无可抢占任务";
    }
    return placed;
}

void Scheduler::apply_placement(Job& job, const ScheduleResult& result) {
    if (!cluster_.allocate(job.id, result.placement.allocations)) {
        throw std::runtime_error("validated placement could not be applied");
    }
    job.state = JobState::Running;
    job.placement = result.placement;
    job.reason = result.reason;
    job.start_time = time_;
    job.wait_time += std::max(0, time_ - job.pending_since);
}

JobView Scheduler::to_view(const Job& job) const {
    JobView v;
    v.id = job.id;
    v.gpu_request = job.gpu_request;
    v.duration = job.duration;
    v.arrival_time = job.arrival_time;
    v.priority = job.priority;
    v.queue_id = job.queue_id;
    v.parallelism = job.parallelism;
    switch (job.state) {
        case JobState::Pending:
            v.state = "pending";
            break;
        case JobState::Running:
            v.state = "running";
            break;
        case JobState::Preempted:
            v.state = "preempted";
            break;
        case JobState::Finished:
            v.state = "finished";
            break;
    }
    v.start_time = job.start_time;
    v.finish_time = job.finish_time;
    if (is_waiting(job.state)) {
        v.wait_time = job.wait_time + std::max(0, time_ - job.pending_since);
    } else {
        v.wait_time = job.wait_time;
    }
    if (job.state == JobState::Running) {
        v.remaining_duration =
            std::max(0, job.remaining_duration - std::max(0, time_ - job.start_time));
    } else {
        v.remaining_duration = job.remaining_duration;
    }
    v.preemption_count = job.preemption_count;
    v.placement = job.placement.allocations;
    v.reason = job.reason;
    v.nodes_used = static_cast<int>(job.placement.allocations.size());
    return v;
}

void Scheduler::update_max_pending() {
    int pending = 0;
    for (const auto& job : jobs_) {
        if (is_waiting(job.state)) {
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
        m.total_preemptions += job.preemption_count;
        if (job.state == JobState::Preempted) {
            m.preempted_jobs++;
        }
        if (is_waiting(job.state)) {
            m.pending_count++;
            if (is_fair_share_reason(job.reason)) {
                m.fair_share_pending++;
            } else {
                m.fragmentation_pending++;
            }
            wait_sum += static_cast<double>(
                job.wait_time + std::max(0, time_ - job.pending_since));
            wait_n++;
        } else if (job.state == JobState::Running) {
            m.running_count++;
            wait_sum += static_cast<double>(job.wait_time);
            wait_n++;
            if (job.placement.allocations.size() > 1) {
                m.cross_node_jobs++;
            }
            if (job_crosses_nvlink(cluster_, job)) {
                m.cross_nvlink_jobs++;
            }
            if (job_crosses_numa(cluster_, job)) {
                m.cross_numa_jobs++;
            }
        } else {
            m.finished_count++;
            wait_sum += static_cast<double>(job.wait_time);
            wait_n++;
            if (job.placement.allocations.size() > 1) {
                m.cross_node_jobs++;
            }
            if (job_crosses_nvlink(cluster_, job)) {
                m.cross_nvlink_jobs++;
            }
            if (job_crosses_numa(cluster_, job)) {
                m.cross_numa_jobs++;
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
    s.queues = queue_views();
    s.metrics = metrics();
    return s;
}

const QueueSpec* Scheduler::find_queue(const std::string& queue_id) const {
    for (const auto& queue : queues_) {
        if (queue.id == queue_id) {
            return &queue;
        }
    }
    return nullptr;
}

int Scheduler::queue_used_gpus(const std::string& queue_id,
                               const std::vector<std::string>& excluded) const {
    const std::unordered_set<std::string> excluded_set(excluded.begin(), excluded.end());
    int used = 0;
    for (const auto& job : jobs_) {
        if (job.state == JobState::Running && job.queue_id == queue_id &&
            excluded_set.count(job.id) == 0) {
            used += job.gpu_request;
        }
    }
    return used;
}

bool Scheduler::is_fair_share_reason(const std::string& reason) const {
    return reason.find("配额不足") != std::string::npos;
}

std::vector<QueueView> Scheduler::queue_views() const {
    std::unordered_map<std::string, QueueView> by_id;
    by_id.reserve(queues_.size());
    for (const auto& queue : queues_) {
        QueueView view;
        view.id = queue.id;
        view.gpu_quota = queue.gpu_quota;
        by_id.emplace(queue.id, view);
    }
    for (const auto& job : jobs_) {
        auto it = by_id.find(job.queue_id);
        if (it == by_id.end()) {
            continue;
        }
        if (job.state == JobState::Running) {
            it->second.used_gpus += job.gpu_request;
            it->second.running_count++;
        } else if (is_waiting(job.state)) {
            it->second.pending_count++;
        }
    }
    std::vector<QueueView> out;
    out.reserve(queues_.size());
    for (const auto& queue : queues_) {
        out.push_back(by_id[queue.id]);
    }
    return out;
}
