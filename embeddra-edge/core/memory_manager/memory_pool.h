#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <cstdint>

namespace embeddra {
namespace memory {

class MemoryPool {
public:
    explicit MemoryPool(size_t pool_size_bytes = 128 * 1024 * 1024); // 128 MB default scratch arena
    ~MemoryPool();

    void* allocate(size_t bytes, size_t alignment = 64);
    void reset();
    size_t getUsedBytes() const { return current_offset_; }
    size_t getTotalBytes() const { return pool_size_; }

private:
    uint8_t* buffer_;
    size_t pool_size_;
    size_t current_offset_;
};

struct KVCacheBuffer {
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    int current_len;
    std::vector<float> k_data; // [num_layers * max_seq_len * num_kv_heads * head_dim]
    std::vector<float> v_data; // [num_layers * max_seq_len * num_kv_heads * head_dim]

    KVCacheBuffer(int layers = 18, int kv_heads = 4, int dim = 128, int max_seq = 2048)
        : num_layers(layers), num_kv_heads(kv_heads), head_dim(dim), max_seq_len(max_seq), current_len(0) {
        size_t total_elements = static_cast<size_t>(layers) * max_seq * kv_heads * dim;
        k_data.resize(total_elements, 0.0f);
        v_data.resize(total_elements, 0.0f);
    }

    void reset() { current_len = 0; }
    size_t getMemorySizeBytes() const {
        return (k_data.size() + v_data.size()) * sizeof(float);
    }

    float* get_k(int layer, int head, int pos) {
        size_t index = ((static_cast<size_t>(layer) * max_seq_len + pos) * num_kv_heads + head) * head_dim;
        return &k_data[index];
    }

    const float* get_k(int layer, int head, int pos) const {
        size_t index = ((static_cast<size_t>(layer) * max_seq_len + pos) * num_kv_heads + head) * head_dim;
        return &k_data[index];
    }

    float* get_v(int layer, int head, int pos) {
        size_t index = ((static_cast<size_t>(layer) * max_seq_len + pos) * num_kv_heads + head) * head_dim;
        return &v_data[index];
    }

    const float* get_v(int layer, int head, int pos) const {
        size_t index = ((static_cast<size_t>(layer) * max_seq_len + pos) * num_kv_heads + head) * head_dim;
        return &v_data[index];
    }
};

} // namespace memory
} // namespace embeddra

