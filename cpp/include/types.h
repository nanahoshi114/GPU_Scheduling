#pragma once

#include <string>
#include <vector>

enum class JobState { Pending, Running, Finished };

struct Gpu {
    int index = 0;
    std::string occupier;  // empty if free
};

struct Node {
    std::string id;
    std::vector<Gpu> gpus;
};

struct NodeAllocation {
    std::string node_id;
    std::vector<int> gpu_indices;
};

struct Placement {
    std::vector<NodeAllocation> allocations;
    std::string reason;
};

struct JobSpec {
    std::string id;
    int gpu_request = 0;
    int duration = 10;
    int arrival_time = 0;
};

struct Job {
    std::string id;
    int gpu_request = 0;
    int arrival_time = 0;
    int duration = 10;
    JobState state = JobState::Pending;
    Placement placement;
    std::string reason;
    int start_time = -1;
    int finish_time = -1;
    int wait_time = 0;
};

struct ScheduleResult {
    bool success = false;
    Placement placement;
    std::string reason;
};

struct GpuView {
    int index = 0;
    std::string job_id;  // empty if free
};

struct NodeView {
    std::string id;
    int gpu_count = 0;
    int free_count = 0;
    std::vector<GpuView> gpus;
};

struct JobView {
    std::string id;
    int gpu_request = 0;
    int duration = 0;
    int arrival_time = 0;
    std::string state;
    int start_time = -1;
    int finish_time = -1;
    int wait_time = 0;
    std::vector<NodeAllocation> placement;
    std::string reason;
    int nodes_used = 0;
};

struct Metrics {
    double gpu_utilization = 0.0;
    double avg_utilization = 0.0;
    int pending_count = 0;
    int running_count = 0;
    int finished_count = 0;
    double avg_wait_time = 0.0;
    int cross_node_jobs = 0;
    int total_gpus = 0;
    int used_gpus = 0;
    int makespan = 0;
    int max_pending = 0;
};

struct Snapshot {
    int time = 0;
    std::string strategy;
    std::vector<NodeView> nodes;
    std::vector<JobView> jobs;
    Metrics metrics;
};

struct SimulationResult {
    Snapshot final_snapshot;
    Metrics metrics;
    std::vector<Snapshot> timeline;
};
