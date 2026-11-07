#pragma once

#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <memory>
#include <cstdint>

namespace embeddra {
namespace runtime {

struct GenerationConfig {
    int max_tokens = 64;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    int eos_token_id = 128001;
};

struct GenerationStats {
    double time_to_first_token_ms = 0.0;
    double total_generation_time_ms = 0.0;
    int tokens_generated = 0;
    double tokens_per_second = 0.0;
    size_t peak_ram_bytes = 0;
};

using TokenCallback = std::function<void(const std::string& token_str, int token_id, bool is_final)>;

class EmbeddraRuntime {
public:
    EmbeddraRuntime();
    ~EmbeddraRuntime();

    bool initialize();
    bool loadModel(const std::string& model_path);
    void generate(
        const std::string& prompt,
        int max_tokens = 64,
        TokenCallback callback = nullptr
    );
    void generateWithConfig(
        const std::string& prompt,
        const GenerationConfig& config,
        TokenCallback callback = nullptr
    );
    void generateWithTokens(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& config,
        TokenCallback callback = nullptr
    );
    void stopGeneration();
    void unloadModel();
    void shutdown();

    bool isModelLoaded() const;
    GenerationStats getLastStats() const;
    std::string getDeviceBackendInfo() const;
    void setBackendMode(int mode);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace runtime
} // namespace embeddra
