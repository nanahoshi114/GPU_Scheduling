#include "simulator.h"

#include "scheduler.h"

#include <algorithm>
#include <limits>

SimulationResult Simulator::run(std::vector<std::pair<std::string, int>> nodes,
                                const std::vector<JobSpec>& jobs,
                                const std::string& strategy) {
    Scheduler sched(std::move(nodes), strategy);

    std::vector<JobSpec> remaining = jobs;
    std::sort(remaining.begin(), remaining.end(), [](const JobSpec& a, const JobSpec& b) {
        if (a.arrival_time != b.arrival_time) {
            return a.arrival_time < b.arrival_time;
        }
        return a.id < b.id;
    });

    SimulationResult out;
    int time = 0;
    long long used_integral = 0;
    int max_pending = 0;
    const int total = sched.total_gpus();
    size_t next_arrival = 0;
    const int kInf = std::numeric_limits<int>::max() / 4;

    auto earliest_finish = [&]() {
        int t = kInf;
        for (const auto& job : sched.jobs()) {
            if (job.state == JobState::Running && job.start_time >= 0) {
                t = std::min(t, job.start_time + job.duration);
            }
        }
        return t;
    };

    auto record = [&]() {
        Snapshot snap = sched.snapshot();
        max_pending = std::max(max_pending, snap.metrics.pending_count);
        out.timeline.push_back(std::move(snap));
    };

    sched.set_time(0);
    // Submit jobs that arrive at t=0.
    while (next_arrival < remaining.size() && remaining[next_arrival].arrival_time <= 0) {
        JobSpec spec = remaining[next_arrival++];
        spec.arrival_time = 0;
        sched.submit(spec);
    }
    sched.schedule_pending();
    record();

    while (true) {
        const int next_arr =
            next_arrival < remaining.size() ? remaining[next_arrival].arrival_time : kInf;
        const int next_fin = earliest_finish();
        if (next_arr == kInf && next_fin == kInf) {
            break;
        }
        const int t = std::min(next_arr, next_fin);
        if (t > time) {
            used_integral += static_cast<long long>(sched.used_gpus()) * (t - time);
            time = t;
            sched.set_time(time);
        }

        if (next_fin == t) {
            std::vector<std::string> due;
            for (const auto& job : sched.jobs()) {
                if (job.state == JobState::Running && job.start_time >= 0 &&
                    job.start_time + job.duration <= time) {
                    due.push_back(job.id);
                }
            }
            for (const auto& id : due) {
                sched.finish(id);
            }
        }

        while (next_arrival < remaining.size() && remaining[next_arrival].arrival_time <= time) {
            JobSpec spec = remaining[next_arrival++];
            spec.arrival_time = time;
            sched.submit(spec);
        }
        sched.schedule_pending();
        record();
    }

    const double avg_util =
        (total > 0 && time > 0)
            ? static_cast<double>(used_integral) / (static_cast<double>(total) * time)
            : sched.metrics().gpu_utilization;

    out.final_snapshot = sched.snapshot();
    out.metrics = sched.metrics_with_avg(avg_util, time, max_pending);
    out.final_snapshot.metrics = out.metrics;
    return out;
}
