#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace embeddra {
namespace backend {

class GPUEngine {
public:
    GPUEngine();
    ~GPUEngine();

    bool initialize();
    void shutdown();

    bool isSupported() const { return initialized_; }
    const std::string& getDeviceName() const { return device_name_; }

    // Execute GEMV on GPU: out [M] = weight [M, N] * in [N]
    // weight is FP16 (uint16_t*), in and out are FP32 (float*)
    void gemv_fp16(float* out, const uint16_t* weight, const float* in, int M, int N);

private:
    bool loadOpenCLSymbols();
    void* getOrCreateWeightBuffer(const void* host_ptr, size_t size_bytes);

    bool initialized_ = false;
    std::string device_name_;
    void* lib_handle_ = nullptr;

    void* cl_platform_ = nullptr;
    void* cl_device_ = nullptr;
    void* cl_context_ = nullptr;
    void* cl_queue_ = nullptr;
    void* cl_program_ = nullptr;
    void* cl_kernel_gemv_ = nullptr;

    // Pre-allocated static buffers for input and output vectors
    void* d_in_buf_ = nullptr;
    void* d_out_buf_ = nullptr;
    size_t current_in_capacity_ = 0;
    size_t current_out_capacity_ = 0;

    // Cache of mapped weight buffers to avoid re-wrapping pointers
    std::unordered_map<const void*, void*> weight_buffer_cache_;
};

} // namespace backend
} // namespace embeddra

