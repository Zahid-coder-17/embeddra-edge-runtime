# Embeddra Edge C++ SDK

The **Embeddra Edge C++ SDK** provides a high-performance, native C++20 API to embed **Maria-1B (HyperGQE)** into mobile apps, edge daemons, robotics, and embedded systems.

---

## 1. Quick Start

Include the single header and link against `libembeddra_edge.so`:

```cpp
#include "include/embeddra_runtime.h"
#include <iostream>

int main() {
    embeddra::runtime::EmbeddraRuntime runtime;

    // 1. Initialize Engine & Detect Hardware (NEON / Mali OpenCL)
    runtime.initialize();

    // 2. Select Execution Mode
    // 0 = GPU (OpenCL UMA), 1 = 8-Core CPU (ARM NEON), 2 = Hybrid
    runtime.setBackendMode(0);

    // 3. Memory-Map Maria-1B Flat Binary (.eer)
    runtime.loadModel("maria_1b.eer");
    std::cout << "Running on: " << runtime.getDeviceBackendInfo() << "\n";

    // 4. Stream Autoregressive Generation
    embeddra::runtime::GenerationConfig config;
    config.max_tokens = 64;
    config.temperature = 0.7f;

    std::vector<int> prompt_tokens = {128000, 3968, 25, /* ... */};

    runtime.generateWithTokens(prompt_tokens, config, [](const std::string& str, int token_id, bool is_final) {
        std::cout << "Token ID: " << token_id << std::endl;
    });

    auto stats = runtime.getLastStats();
    std::cout << "TTFT: " << stats.time_to_first_token_ms << " ms | Speed: " << stats.tokens_per_second << " tok/s\n";

    runtime.unloadModel();
    return 0;
}
```

---

## 2. Compiling and Linking

### Android NDK (ARM64-v8a):
```bash
aarch64-linux-android34-clang++ main.cpp \
    -Iinclude \
    -L../android/jniLibs/arm64-v8a \
    -lembeddra_edge -llog -ldl \
    -o my_app
```

