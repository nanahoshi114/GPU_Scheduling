#include "scheduler.h"
#include "simulator.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

py::list allocations_to_list(const std::vector<NodeAllocation>& allocs) {
    py::list out;
    for (const auto& a : allocs) {
        py::dict d;
        d["node_id"] = a.node_id;
        d["gpu_indices"] = a.gpu_indices;
        out.append(d);
    }
    return out;
}

py::dict metrics_to_dict(const Metrics& m) {
    py::dict d;
    d["gpu_utilization"] = m.gpu_utilization;
    d["avg_utilization"] = m.avg_utilization;
    d["pending_count"] = m.pending_count;
    d["running_count"] = m.running_count;
    d["finished_count"] = m.finished_count;
    d["avg_wait_time"] = m.avg_wait_time;
    d["cross_node_jobs"] = m.cross_node_jobs;
    d["total_gpus"] = m.total_gpus;
    d["used_gpus"] = m.used_gpus;
    d["makespan"] = m.makespan;
    d["max_pending"] = m.max_pending;
    d["total_preemptions"] = m.total_preemptions;
    d["preempted_jobs"] = m.preempted_jobs;
    return d;
}

py::dict job_to_dict(const JobView& j) {
    py::dict d;
    d["id"] = j.id;
    d["gpu_request"] = j.gpu_request;
    d["duration"] = j.duration;
    d["arrival_time"] = j.arrival_time;
    d["priority"] = j.priority;
    d["state"] = j.state;
    d["start_time"] = j.start_time;
    d["finish_time"] = j.finish_time;
    d["wait_time"] = j.wait_time;
    d["remaining_duration"] = j.remaining_duration;
    d["preemption_count"] = j.preemption_count;
    d["placement"] = allocations_to_list(j.placement);
    d["reason"] = j.reason;
    d["nodes_used"] = j.nodes_used;
    return d;
}

py::dict snapshot_to_dict(const Snapshot& s) {
    py::dict d;
    d["time"] = s.time;
    d["strategy"] = s.strategy;
    py::list nodes;
    for (const auto& n : s.nodes) {
        py::dict nd;
        nd["id"] = n.id;
        nd["gpu_count"] = n.gpu_count;
        nd["free_count"] = n.free_count;
        py::list gpus;
        for (const auto& g : n.gpus) {
            py::dict gd;
            gd["index"] = g.index;
            gd["job_id"] = g.job_id;
            gpus.append(gd);
        }
        nd["gpus"] = gpus;
        nodes.append(nd);
    }
    d["nodes"] = nodes;
    py::list jobs;
    for (const auto& j : s.jobs) {
        jobs.append(job_to_dict(j));
    }
    d["jobs"] = jobs;
    d["metrics"] = metrics_to_dict(s.metrics);
    return d;
}

py::dict result_to_dict(const ScheduleResult& r) {
    py::dict d;
    d["success"] = r.success;
    d["reason"] = r.reason;
    d["placement"] = allocations_to_list(r.placement.allocations);
    d["victims"] = r.victims;
    return d;
}

py::dict simulation_to_dict(const SimulationResult& r) {
    py::dict d;
    d["metrics"] = metrics_to_dict(r.metrics);
    d["final_snapshot"] = snapshot_to_dict(r.final_snapshot);
    py::list tl;
    for (const auto& s : r.timeline) {
        tl.append(snapshot_to_dict(s));
    }
    d["timeline"] = tl;
    return d;
}

std::vector<std::pair<std::string, int>> parse_nodes(const py::object& nodes) {
    std::vector<std::pair<std::string, int>> out;
    for (auto item : nodes) {
        py::handle h = item;
        if (py::isinstance<py::dict>(h)) {
            auto d = h.cast<py::dict>();
            std::string id;
            int gpus = 0;
            if (d.contains("id")) {
                id = d["id"].cast<std::string>();
            } else if (d.contains("node_id")) {
                id = d["node_id"].cast<std::string>();
            }
            if (d.contains("gpu_count")) {
                gpus = d["gpu_count"].cast<int>();
            } else if (d.contains("gpus")) {
                gpus = d["gpus"].cast<int>();
            }
            out.emplace_back(id, gpus);
        } else {
            auto t = h.cast<py::tuple>();
            out.emplace_back(t[0].cast<std::string>(), t[1].cast<int>());
        }
    }
    return out;
}

JobSpec parse_job_spec(const py::dict& d, int default_arrival) {
    JobSpec spec;
    spec.id = d["id"].cast<std::string>();
    spec.gpu_request = d.contains("gpu_request") ? d["gpu_request"].cast<int>()
                                                 : d["gpus"].cast<int>();
    spec.duration = d.contains("duration") ? d["duration"].cast<int>() : 10;
    spec.priority = d.contains("priority") ? d["priority"].cast<int>() : 0;
    if (d.contains("arrival_time")) {
        spec.arrival_time = d["arrival_time"].cast<int>();
    } else if (d.contains("arrival")) {
        spec.arrival_time = d["arrival"].cast<int>();
    } else {
        spec.arrival_time = default_arrival;
    }
    return spec;
}

}  // namespace

PYBIND11_MODULE(gpu_scheduler, m) {
    m.doc() = "GPU topology-aware scheduler core";

    py::class_<Scheduler>(m, "Scheduler")
        .def(py::init([](py::object nodes, const std::string& strategy,
                         bool enable_preemption) {
                 return Scheduler(parse_nodes(nodes), strategy, enable_preemption);
             }),
             py::arg("nodes"), py::arg("strategy") = "topology_aware",
             py::arg("enable_preemption") = true)
        .def(
            "submit",
            [](Scheduler& self, py::object job_or_id, py::object gpu_request = py::none(),
               int duration = 10, py::object arrival_time = py::none(), int priority = 0) {
                JobSpec spec;
                if (py::isinstance<py::dict>(job_or_id)) {
                    spec = parse_job_spec(job_or_id.cast<py::dict>(), self.current_time());
                } else {
                    spec.id = job_or_id.cast<std::string>();
                    if (gpu_request.is_none()) {
                        throw std::invalid_argument("gpu_request is required");
                    }
                    spec.gpu_request = gpu_request.cast<int>();
                    spec.duration = duration;
                    spec.priority = priority;
                    spec.arrival_time =
                        arrival_time.is_none() ? self.current_time() : arrival_time.cast<int>();
                }
                return result_to_dict(self.submit(spec));
            },
            py::arg("job_or_id"), py::arg("gpu_request") = py::none(), py::arg("duration") = 10,
            py::arg("arrival_time") = py::none(), py::arg("priority") = 0)
        .def("finish", &Scheduler::finish, py::arg("job_id"))
        .def("tick", [](Scheduler& self) { return snapshot_to_dict(self.tick()); })
        .def("snapshot", [](const Scheduler& self) { return snapshot_to_dict(self.snapshot()); })
        .def("metrics", [](const Scheduler& self) { return metrics_to_dict(self.metrics()); })
        .def_property_readonly("time", &Scheduler::current_time)
        .def_property_readonly("strategy", &Scheduler::strategy_name)
        .def_property_readonly("preemption_enabled", &Scheduler::preemption_enabled);

    m.def(
        "simulate",
        [](py::object nodes, py::iterable jobs, const std::string& strategy,
           bool enable_preemption) {
            std::vector<JobSpec> specs;
            for (auto item : jobs) {
                specs.push_back(parse_job_spec(item.cast<py::dict>(), 0));
            }
            return simulation_to_dict(
                Simulator::run(parse_nodes(nodes), specs, strategy, enable_preemption));
        },
        py::arg("nodes"), py::arg("jobs"), py::arg("strategy"),
        py::arg("enable_preemption") = true);

    m.def("strategies", []() {
        py::list names;
        names.append("first_fit");
        names.append("topology_aware");
        return names;
    });
}
