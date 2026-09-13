#pragma once

#include "cluster.h"
#include "types.h"

#include <memory>
#include <string>
#include <vector>

// 放置策略入口。核心算法在 cpp/src/scheduler.cpp：
// FirstFitStrategy（顺序填空闲卡，忽略并行方式）与
// TopologyAwareStrategy（DP 少跨节点 / TP 同 NVLink 组 / PP 相邻节点）。
class PlacementStrategy {
public:
    virtual ~PlacementStrategy() = default;
    virtual ScheduleResult try_place(const Cluster& cluster, const PlaceRequest& req,
                                     bool has_running) const = 0;
    virtual std::string name() const = 0;
};

std::unique_ptr<PlacementStrategy> make_strategy(const std::string& name);

class Scheduler {
public:
    Scheduler(std::vector<NodeInit> nodes, const std::string& strategy,
              bool enable_preemption = true, std::vector<QueueSpec> queues = {},
              int locality_timeout = 0);

    ScheduleResult submit(const JobSpec& spec);
    bool finish(const std::string& job_id);
    Snapshot tick();
    void set_time(int t) { time_ = t; }

    Snapshot snapshot() const;
    Metrics metrics() const;
    Metrics metrics_with_avg(double avg_utilization, int makespan, int max_pending) const;

    int current_time() const { return time_; }
    int used_gpus() const { return cluster_.used_gpus(); }
    int total_gpus() const { return cluster_.total_gpus(); }
    const std::string& strategy_name() const { return strategy_name_; }
    bool preemption_enabled() const { return enable_preemption_; }
    int locality_timeout() const { return locality_timeout_; }
    const std::vector<Job>& jobs() const { return jobs_; }
    const std::vector<QueueSpec>& queues() const { return queues_; }

    void schedule_pending();

private:
    Cluster cluster_;
    std::unique_ptr<PlacementStrategy> strategy_;
    std::string strategy_name_;
    std::vector<Job> jobs_;
    std::vector<QueueSpec> queues_;
    int time_ = 0;
    int max_pending_ = 0;
    bool enable_preemption_ = true;
    int locality_timeout_ = 0;

    JobView to_view(const Job& job) const;
    bool should_relax_locality(const Job& job) const;
    PlaceRequest place_request_for(const Job& job) const;
    void apply_placement(Job& job, const ScheduleResult& result);
    ScheduleResult try_schedule_job(Job& job);
    ScheduleResult find_preemption_plan(const Job& incoming) const;
    void preempt_victims(const Job& incoming, const std::vector<std::string>& victim_ids);
    bool has_running_jobs(const std::vector<std::string>& excluded = {}) const;
    void update_max_pending();
    const QueueSpec* find_queue(const std::string& queue_id) const;
    int queue_used_gpus(const std::string& queue_id,
                        const std::vector<std::string>& excluded = {}) const;
    bool is_fair_share_reason(const std::string& reason) const;
    std::vector<QueueView> queue_views() const;
};
