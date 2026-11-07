#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <atomic>
#include <cstdint>
#include "../../core/memory_manager/memory_pool.h"
#include "../gpu/gpu_engine.h"

namespace embeddra {
namespace backend {

struct EngineConfig {
    int vocab_size = 128256;
    int hidden_size = 2048;
    int intermediate_size = 5632;
    int num_layers = 18;
    int num_heads = 16;
    int num_kv_heads = 4;
    int head_dim = 128;
    int max_seq_len = 2048;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 500000.0f;
};

struct TensorRef {
    std::string name;
    std::vector<int> shape;
    uint8_t dtype; // 1 = FP16
    uint64_t offset;
    uint64_t size_bytes;
    const void* data_ptr = nullptr;
};

class HyperGQEEngine {
public:
    explicit HyperGQEEngine(const EngineConfig& config = EngineConfig());
    ~HyperGQEEngine();

    bool loadModel(const std::string& model_path);
    void unloadModel();
    bool isLoaded() const { return is_loaded_; }

    void resetKVCache();

    enum BackendMode {
        BACKEND_GPU = 0,
        BACKEND_CPU_8CORE = 1,
        BACKEND_HYBRID = 2
    };

    void setBackendMode(BackendMode mode);
    BackendMode getBackendMode() const;

    bool isGPUActive() const;
    std::string getGPUBackendName() const;
    std::string getActiveBackendName() const;

    // Runs single-token or prefill forward pass, returns logits pointer [vocab_size]
    const float* forward(int token_id, int pos);

    int sample(const float* logits, float temperature = 0.7f, float top_p = 0.9f, int top_k = 50, float rep_penalty = 1.25f);

private:
    void initBuffers();
    void computeRMSNorm(float* out, const float* in, const uint16_t* weight, int size, float eps);
    void computeRoPE(float* q, float* k, int pos, int num_q_heads, int num_kv_heads, int dim);
    void gemv_fp16(float* out, const uint16_t* weight, const float* in, int M, int N);

    EngineConfig config_;
    bool is_loaded_;

    // Memory-mapped file descriptor & base pointer
    int fd_;
    void* mmap_base_;
    size_t mmap_size_;

    std::unordered_map<std::string, TensorRef> tensor_map_;

    std::unique_ptr<memory::MemoryPool> mem_pool_;
    std::unique_ptr<memory::KVCacheBuffer> kv_cache_;
    std::unique_ptr<GPUEngine> gpu_engine_;
    BackendMode backend_mode_ = BACKEND_GPU;

    // Working activation buffers (pre-allocated)
    std::vector<float> hidden_state_;
    std::vector<float> norm_state_;
    std::vector<float> q_state_;
    std::vector<float> k_state_;
    std::vector<float> v_state_;
    std::vector<float> router_logits_;
    std::vector<float> attn_out_;
    std::vector<float> gate_state_;
    std::vector<float> up_state_;
    std::vector<float> ffn_act_;
    std::vector<float> ffn_out_;
    std::vector<float> logits_;

    std::vector<int> generated_tokens_;

    // Pointers mapped from .eer file
    const uint16_t* embed_weight_ = nullptr;
    const uint16_t* final_norm_weight_ = nullptr;

    struct LayerPointers {
        const uint16_t* attn_norm_w = nullptr;
        const uint16_t* q_proj_w = nullptr;
        const uint16_t* k_proj_w = nullptr;
        const uint16_t* v_proj_w = nullptr;
        const uint16_t* router_w = nullptr;
        const uint16_t* expert_emb = nullptr;
        const uint16_t* hyper_w_d = nullptr;
        const uint16_t* hyper_w_u = nullptr;
        const uint16_t* q_shared_w = nullptr;
        const uint16_t* o_proj_w = nullptr;
        const uint16_t* ffn_norm_w = nullptr;
        const uint16_t* gate_proj_w = nullptr;
        const uint16_t* up_proj_w = nullptr;
        const uint16_t* down_proj_w = nullptr;
    };
    std::vector<LayerPointers> layers_;
};

} // namespace backend
} // namespace embeddra
