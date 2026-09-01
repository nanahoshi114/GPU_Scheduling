#pragma once

#include "types.h"

#include <string>
#include <vector>

class Cluster {
public:
    void add_node(const std::string& id, int gpu_count);

    int total_gpus() const { return total_gpus_; }
    int used_gpus() const { return used_gpus_; }
    int free_gpus() const { return total_gpus_ - used_gpus_; }
    int max_node_capacity() const;
    int node_count() const { return static_cast<int>(nodes_.size()); }

    const std::vector<Node>& nodes() const { return nodes_; }
    std::vector<Node>& nodes() { return nodes_; }

    int node_index(const std::string& id) const;
    std::vector<int> free_gpu_indices(int node_idx) const;
    int free_count(int node_idx) const;

    bool allocate(const std::string& job_id, const std::vector<NodeAllocation>& allocs);
    void release(const std::string& job_id);

    std::vector<NodeView> view() const;

private:
    std::vector<Node> nodes_;
    int total_gpus_ = 0;
    int used_gpus_ = 0;
};
