#pragma once

#include "types.h"

#include <string>
#include <vector>

class Simulator {
public:
    static SimulationResult run(std::vector<NodeInit> nodes,
                                const std::vector<JobSpec>& jobs,
                                const std::string& strategy,
                                bool enable_preemption = true,
                                std::vector<QueueSpec> queues = {},
                                int locality_timeout = 0);
};
