#pragma once

#include "types.h"

#include <string>
#include <utility>
#include <vector>

class Simulator {
public:
    static SimulationResult run(std::vector<std::pair<std::string, int>> nodes,
                                const std::vector<JobSpec>& jobs,
                                const std::string& strategy,
                                bool enable_preemption = true);
};
