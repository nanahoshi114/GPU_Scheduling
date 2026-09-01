#pragma once

#include "cluster.h"
#include "types.h"

#include <memory>
#include <string>
#include <vector>

class PlacementStrategy {
public:
    virtual ~PlacementStrategy() = default;
    virtual ScheduleResult try_place(const Cluster& cluster, int gpu_request,
                                     bool has_running) const = 0;
    virtual std::string name() const = 0;
};

std::unique_ptr<PlacementStrategy> make_strategy(const std::string& name);

class Scheduler {
public:
    Scheduler(std::vector<std::pair<std::string, int>> nodes, const std::string& strategy);

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
    const std::vector<Job>& jobs() const { return jobs_; }

    void schedule_pending();

private:
    Cluster cluster_;
    std::unique_ptr<PlacementStrategy> strategy_;
    std::string strategy_name_;
    std::vector<Job> jobs_;
    int time_ = 0;
    int max_pending_ = 0;

    JobView to_view(const Job& job) const;
    void apply_placement(Job& job, const ScheduleResult& result);
    void update_max_pending();
};
