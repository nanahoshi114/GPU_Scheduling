#include "cluster.h"

#include <algorithm>
#include <stdexcept>

void Cluster::add_node(const std::string& id, int gpu_count) {
    if (id.empty()) {
        throw std::invalid_argument("node id must not be empty");
    }
    if (gpu_count <= 0) {
        throw std::invalid_argument("gpu_count must be positive");
    }
    if (node_index(id) >= 0) {
        throw std::invalid_argument("duplicate node id: " + id);
    }
    Node node;
    node.id = id;
    node.gpus.resize(static_cast<size_t>(gpu_count));
    for (int i = 0; i < gpu_count; ++i) {
        node.gpus[static_cast<size_t>(i)].index = i;
    }
    nodes_.push_back(std::move(node));
    total_gpus_ += gpu_count;
}

int Cluster::max_node_capacity() const {
    int m = 0;
    for (const auto& node : nodes_) {
        m = std::max(m, static_cast<int>(node.gpus.size()));
    }
    return m;
}

int Cluster::node_index(const std::string& id) const {
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        if (nodes_[static_cast<size_t>(i)].id == id) {
            return i;
        }
    }
    return -1;
}

std::vector<int> Cluster::free_gpu_indices(int node_idx) const {
    std::vector<int> out;
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes_.size())) {
        return out;
    }
    const auto& gpus = nodes_[static_cast<size_t>(node_idx)].gpus;
    for (const auto& gpu : gpus) {
        if (gpu.occupier.empty()) {
            out.push_back(gpu.index);
        }
    }
    return out;
}

int Cluster::free_count(int node_idx) const {
    return static_cast<int>(free_gpu_indices(node_idx).size());
}

bool Cluster::allocate(const std::string& job_id, const std::vector<NodeAllocation>& allocs) {
    // Validate first so we never partially apply.
    int extra = 0;
    for (const auto& alloc : allocs) {
        int idx = node_index(alloc.node_id);
        if (idx < 0) {
            return false;
        }
        auto& node = nodes_[static_cast<size_t>(idx)];
        for (int g : alloc.gpu_indices) {
            if (g < 0 || g >= static_cast<int>(node.gpus.size())) {
                return false;
            }
            if (!node.gpus[static_cast<size_t>(g)].occupier.empty()) {
                return false;
            }
            extra++;
        }
    }
    for (const auto& alloc : allocs) {
        int idx = node_index(alloc.node_id);
        auto& node = nodes_[static_cast<size_t>(idx)];
        for (int g : alloc.gpu_indices) {
            node.gpus[static_cast<size_t>(g)].occupier = job_id;
        }
    }
    used_gpus_ += extra;
    return true;
}

void Cluster::release(const std::string& job_id) {
    int released = 0;
    for (auto& node : nodes_) {
        for (auto& gpu : node.gpus) {
            if (gpu.occupier == job_id) {
                gpu.occupier.clear();
                released++;
            }
        }
    }
    used_gpus_ -= released;
    if (used_gpus_ < 0) {
        used_gpus_ = 0;
    }
}

std::vector<NodeView> Cluster::view() const {
    std::vector<NodeView> out;
    out.reserve(nodes_.size());
    for (const auto& node : nodes_) {
        NodeView nv;
        nv.id = node.id;
        nv.gpu_count = static_cast<int>(node.gpus.size());
        nv.free_count = 0;
        nv.gpus.reserve(node.gpus.size());
        for (const auto& gpu : node.gpus) {
            GpuView gv;
            gv.index = gpu.index;
            gv.job_id = gpu.occupier;
            if (gpu.occupier.empty()) {
                nv.free_count++;
            }
            nv.gpus.push_back(std::move(gv));
        }
        out.push_back(std::move(nv));
    }
    return out;
}
