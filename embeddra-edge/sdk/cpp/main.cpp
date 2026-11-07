#include <iostream>
#include <iomanip>
#include "embeddra_runtime.h"

int main(int argc, char* argv[]) {
    std::cout << "=================================================================\n";
    std::cout << "EMBEDDRA EDGE RUNTIME (EER) - DUAL HARDWARE BENCHMARK\n";
    std::cout << "Hardware: Samsung Galaxy A35 5G (Exynos 1380 / 8 Cores / Mali-G68)\n";
    std::cout << "Model: Maria-1B (HyperGQE 18-Layer FP16 Flat Mmap)\n";
    std::cout << "=================================================================\n\n";

    embeddra::runtime::EmbeddraRuntime runtime;

    if (!runtime.initialize()) {
        std::cerr << "[ERROR] Failed to initialize Embeddra Edge Runtime.\n";
        return 1;
    }

    std::string model_path = (argc > 1) ? argv[1] : "/sdcard/Android/data/com.embeddra.edge/files/maria_1b.eer";
    std::cout << "Loading Maria-1B model from: " << model_path << "...\n";

    if (!runtime.loadModel(model_path)) {
        std::cerr << "[ERROR] Failed to load Maria-1B model.\n";
        return 1;
    }
    std::cout << "[OK] Model memory-mapped successfully.\n\n";

    std::string prompt = "Question: What are the main symptoms of diabetes?\nAnswer:";
    std::vector<int> prompt_tokens = {128000, 3968, 25, 3923, 527, 279, 2366, 17540, 315, 23789, 30, 198, 22550, 25};

    embeddra::runtime::GenerationConfig config;
    config.max_tokens = 20;
    config.temperature = 0.0f;

    // ==========================================
    // BENCHMARK 1: 8-CORE CPU (ARM NEON FP16)
    // ==========================================
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << ">>> BENCHMARK 1: 8-CORE CPU (4x Cortex-A78 + 4x Cortex-A55 NEON)\n";
    std::cout << "-----------------------------------------------------------------\n";
    runtime.setBackendMode(1); // 1 = BACKEND_CPU_8CORE
    std::cout << "Active Backend: " << runtime.getDeviceBackendInfo() << "\n";
    std::cout << "Executing generation (20 tokens)...\nTokens: ";

    runtime.generateWithTokens(prompt_tokens, config, [](const std::string& /*token_str*/, int token_id, bool is_final) {
        std::cout << token_id << " " << std::flush;
        if (is_final) std::cout << "\n";
    });

    auto cpu_stats = runtime.getLastStats();
    std::cout << "--- 8-Core CPU Benchmark Results ---\n"
              << "Time to First Token (TTFT): " << cpu_stats.time_to_first_token_ms << " ms\n"
              << "Total Generation Time:       " << cpu_stats.total_generation_time_ms << " ms\n"
              << "Tokens Generated:            " << cpu_stats.tokens_generated << "\n"
              << "Throughput:                  " << cpu_stats.tokens_per_second << " tokens/sec\n\n";

    // ==========================================
    // BENCHMARK 2: MOBILE GPU (Mali-G68 MP5 OpenCL)
    // ==========================================
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << ">>> BENCHMARK 2: MOBILE GPU (ARM Mali-G68 MP5 OpenCL FP16 UMA)\n";
    std::cout << "-----------------------------------------------------------------\n";
    runtime.setBackendMode(0); // 0 = BACKEND_GPU
    std::cout << "Active Backend: " << runtime.getDeviceBackendInfo() << "\n";
    std::cout << "Executing generation (20 tokens)...\nTokens: ";

    runtime.generateWithTokens(prompt_tokens, config, [](const std::string& /*token_str*/, int token_id, bool is_final) {
        std::cout << token_id << " " << std::flush;
        if (is_final) std::cout << "\n";
    });

    auto gpu_stats = runtime.getLastStats();
    std::cout << "--- Mali-G68 GPU Benchmark Results ---\n"
              << "Time to First Token (TTFT): " << gpu_stats.time_to_first_token_ms << " ms\n"
              << "Total Generation Time:       " << gpu_stats.total_generation_time_ms << " ms\n"
              << "Tokens Generated:            " << gpu_stats.tokens_generated << "\n"
              << "Throughput:                  " << gpu_stats.tokens_per_second << " tokens/sec\n\n";

    // ==========================================
    // SIDE-BY-SIDE SUMMARY TABLE
    // ==========================================
    std::cout << "=================================================================\n";
    std::cout << "              SIDE-BY-SIDE ON-DEVICE BENCHMARK REPORT             \n";
    std::cout << "=================================================================\n";
    std::cout << std::left << std::setw(28) << "Metric" 
              << std::setw(20) << "8-Core CPU" 
              << std::setw(20) << "Mali-G68 GPU" << "\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << std::left << std::setw(28) << "Hardware Acceleration"
              << std::setw(20) << "ARM NEON FP16"
              << std::setw(20) << "OpenCL 3.0 UMA" << "\n";
    std::cout << std::left << std::setw(28) << "Time to First Token (TTFT)"
              << std::setw(20) << (std::to_string(cpu_stats.time_to_first_token_ms) + " ms")
              << std::setw(20) << (std::to_string(gpu_stats.time_to_first_token_ms) + " ms") << "\n";
    std::cout << std::left << std::setw(28) << "Decode Throughput"
              << std::setw(20) << (std::to_string(cpu_stats.tokens_per_second) + " tok/s")
              << std::setw(20) << (std::to_string(gpu_stats.tokens_per_second) + " tok/s") << "\n";
    std::cout << std::left << std::setw(28) << "Total Time (20 tokens)"
              << std::setw(20) << (std::to_string(cpu_stats.total_generation_time_ms) + " ms")
              << std::setw(20) << (std::to_string(gpu_stats.total_generation_time_ms) + " ms") << "\n";
    std::cout << "=================================================================\n";

    runtime.unloadModel();
    runtime.shutdown();
    return 0;
}
