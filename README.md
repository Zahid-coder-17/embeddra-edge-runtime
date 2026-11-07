# Embeddra Edge Runtime (EER)

<p align="center">
  <b>A Zero-Cloud, High-Performance On-Device AI Inference Platform for Maria-1B (HyperGQE)</b>
  <br>
  <i>Native C++20 Core • ARM Mali OpenCL 3.0 FP16 UMA • 8-Core ARM NEON Parallelization • Android NDK</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg?style=flat-square&logo=cplusplus" alt="C++20">
  <img src="https://img.shields.io/badge/Android-NDK%20r26-green.svg?style=flat-square&logo=android" alt="Android NDK">
  <img src="https://img.shields.io/badge/GPU-Mali%20OpenCL%20FP16-red.svg?style=flat-square" alt="Mali OpenCL">
  <img src="https://img.shields.io/badge/CPU-ARM%20NEON%208--Core-orange.svg?style=flat-square" alt="ARM NEON">
  <img src="https://img.shields.io/badge/Memory-Zero--Copy%20Mmap-purple.svg?style=flat-square" alt="Zero-Copy Mmap">
  <img src="https://img.shields.io/badge/Cloud%20Dependency-0%25%20(100%25%20Offline)-brightgreen.svg?style=flat-square" alt="Offline">
</p>

---

## 📖 Overview

**Embeddra Edge Runtime (EER)** is a self-contained, high-performance on-device AI inference platform designed to deploy and run large language models locally on consumer smartphones and embedded edge hardware without external servers or cloud dependencies.

EER was engineered specifically to execute the proprietary **Maria-1B** model, featuring the **HyperGQE (Hypernetwork-based Grouped Query Experts)** attention mechanism. Rather than relying on heavyweight generic runtimes (such as PyTorch, ONNX, or Ollama) that introduce significant memory overhead and cloud dependencies, EER provides an optimized bare-metal C++20 engine that interfaces directly with mobile silicon:
* **ARM Mali GPUs** via custom OpenCL 3.0 vector-8 half-precision (`half8`) compute kernels.
* **8-Core ARM CPUs** via multi-threaded thread pools and native ARMv8.2-A NEON SIMD (`float16x8_t`).
* **Zero-Copy Flash Storage Access** via 4KB page-aligned memory-mapped binary files (`.eer`).

---

## 🏛 Architecture: Maria-1B & HyperGQE

Maria-1B departs from traditional dense multi-head attention and static MoE architectures by dynamically synthesizing expert adaptation vectors at runtime.

```
+-----------------------------------------------------------------------------------+
|                               Maria-1B Geometry                                    |
+-----------------------------------------------------------------------------------+
|  Parameters:           1,074,481,152 (~1.074 Billion unique FP16 parameters)      |
|  Layers:               18 Transformer Decoder Blocks                              |
|  Hidden Dimension:     d_model = 2048                                             |
|  Attention Heads:      16 Query Heads (d_head = 128), 4 KV Heads (Grouped Query)  |
|  Attention Mode:       HyperGQE Dynamic Routing + Expert Hypernetworks            |
|  Dynamic Router:       Linear(2048 -> 16), Softmax over 4 Expert Groups           |
|  Expert Embeddings:    4 Groups x 4 Experts x 64 Dimensions                      |
|  Hypernetworks:        W_D: (32768, 64) -> (2048, 16), W_U: (2048, 64) -> (16, 128)|
|  FFN Intermediate:     d_ffn = 5632 (SwiGLU: gate_proj, up_proj, down_proj)      |
|  Compressed Output:    o_proj: 768 -> 2048 (62.5% parameter reduction vs standard)|
|  Normalization:        RMSNorm (eps = 1e-5)                                       |
|  Vocabulary:           128,256 tokens (Byte-Pair Encoding)                        |
|  Weight Tying:         lm_head.weight == embed_tokens.weight                      |
+-----------------------------------------------------------------------------------+
```

### The HyperGQE Advantage
1. **Dynamic Expert Routing**: Query representations pass through `attn.router` ($2048 \to 16$) to compute dispatch affinities across 4 expert groups.
2. **Latent Vector Synthesis**: Dynamic hypernetworks modulate latent projections conditioned on token context.
3. **62.5% Compression on Output Projection**: Concatenates 4 routed group heads ($512$) + shared head ($128$) + dynamic hypernetwork head ($128$) into a $768$-dimension vector before projecting to $2048$, reducing matrix size significantly compared to standard $2048 \times 2048$ projections.

---

## ⚡ Empirical On-Device Benchmarks

Tested and verified on a physical **Samsung Galaxy A35 5G (`SM-A356E`, Exynos 1380 SoC, 8 CPU cores, Mali-G68 MP5 GPU, Android 16)**:

### 1. Dual-Hardware Comparison (8-Core CPU vs. Mali-G68 GPU)

| Benchmark Metric | **8-Core CPU (ARM NEON)** | **Mobile GPU (Mali-G68 MP5)** | Speedup / Efficiency |
| :--- | :--- | :--- | :--- |
| **Compute Technology** | **4x Cortex-A78 + 4x Cortex-A55** | **5 Valhall GPU Execution Engines** | — |
| **Instruction Set / API** | **ARM NEON FP16 (`float16x8_t`)** | **OpenCL 3.0 UMA (`cl_khr_fp16`)** | — |
| **Time to First Token (TTFT)**| **8,926.59 ms** | **10,627.09 ms** | Instant local response |
| **Autoregressive Throughput** | **1.12 tokens/sec** | **0.93 tokens/sec** | 100% on-device |
| **20-Token Generation Time** | **17.79 seconds** | **21.52 seconds** | Complete local pass |
| **Peak Model RAM Footprint** | **2.49 GB (Zero-copy unified `mmap`)** | **2.49 GB (Zero-copy unified `mmap`)** | Zero duplication |
| **Token Output Determinism** | `3923 374 279 7580 315 264 ...` | `3923 374 279 7580 315 264 ...` | **100% Bit-Exact Match** |

### 2. Low-Level Component Latency Breakdown

| Matrix Operation | Dimensions ($M \times N$) | Single-Core Scalar | **8-Core CPU (NEON)** | **Mali-G68 GPU** | Best Hardware Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$q\_proj$ / $o\_proj$** | $2048 \times 2048$ | 9.78 ms | 2.26 ms | **1.25 ms** | **GPU (1.8x faster)** |
| **$gate\_proj$ / $up\_proj$**| $5632 \times 2048$ | 29.17 ms | 4.71 ms | **3.94 ms** | **GPU (1.2x faster)** |
| **$down\_proj$** | $2048 \times 5632$ | 27.18 ms | **4.49 ms** | 5.64 ms | **8-Core CPU** |
| **LM Head** | $128256 \times 2048$ | 610.28 ms | 49.30 ms | **48.36 ms** | **GPU (12.6x speedup)**|
| **Router Logits** | $16 \times 2048$ | **0.04 ms** | **0.02 ms** | 1.15 ms *(driver sync)* | **CPU (57x faster)** |

---

## 📁 Repository Structure

```
llm_deployment/
├── embeddra-edge/
│   ├── android/                    # Production Android Application & Build Tools
│   │   ├── app/                    # Native Android App source code (Kotlin + Jetpack)
│   │   │   └── src/main/java/com/embeddra/edge/
│   │   │       ├── MainActivity.kt        # Interactive UI & Streaming Console
│   │   │       ├── EmbeddraRuntime.kt    # Android Kotlin SDK wrapper
│   │   │       └── VocabularyDecoder.kt  # Native 128k BPE Tokenizer
│   │   ├── jni/                    # JNI Bridge (C++ <-> Java)
│   │   │   └── embeddra_jni.cpp
│   │   └── build_apk.ps1           # Automated headless build & deploy toolchain
│   ├── backends/
│   │   ├── custom_cpp/             # Native C++20 HyperGQE Core Engine
│   │   │   ├── hypergqe_engine.h
│   │   │   └── hypergqe_engine.cpp # RoPE, SDPA, 8-Core ARM NEON FP16 Kernels
│   │   └── gpu/                    # Mobile GPU OpenCL Acceleration
│   │       ├── gpu_engine.h
│   │       └── gpu_engine.cpp      # Zero-copy UMA OpenCL 3.0 FP16 Compute Kernels
│   ├── core/
│   │   ├── hardware/               # CPU & Hardware Feature Detection
│   │   ├── memory_manager/         # Static Memory Pool & KV-Cache Buffer
│   │   └── runtime/                # EmbeddraRuntime lifecycle & stats manager
│   ├── export/                     # Checkpoint Conversion & Quantization
│   │   ├── serialize_to_eer.py     # Flat binary 4KB page-aligned serialization
│   │   └── quantize_maria.py       # Mixed-precision INT8 / W4 quantization
│   ├── hypergqe/                   # PyTorch Reference Model & Dynamic Routing
│   ├── sdk/
│   │   ├── android/                # Ready-to-use Android SDK (AAR/jniLibs)
│   │   └── cpp/                    # Native C++ SDK (headers & main.cpp sample)
│   └── CMakeLists.txt              # Cross-platform CMake build configuration
├── EMBEDDRA_EDGE_DEPLOYMENT_REPORT.md        # Complete architectural technical report
└── EMBEDDRA_EDGE_EMPIRICAL_BENCHMARK_REPORT.md # Perplexity & hardware benchmark report
```

---

## 🚀 Quick Start Guide

### 1. Convert Weights to `.eer` (Embeddra Edge Resource)
Convert standard PyTorch weights (`.pt`) into 4KB page-aligned binary files for instant zero-copy loading:
```bash
python embeddra-edge/export/serialize_to_eer.py \
    --input checkpoints/maria_1b_20B_weights.pt \
    --output checkpoints/maria_1b.eer
```

### 2. Run with Native C++ SDK (Desktop or Edge Linux)
```cpp
#include "embeddra_runtime.h"

int main() {
    embeddra::runtime::EmbeddraRuntime runtime;
    runtime.initialize();

    // 0 = Mali GPU, 1 = 8-Core CPU (NEON), 2 = Hybrid
    runtime.setBackendMode(0);

    // Memory-map local model weights
    runtime.loadModel("maria_1b.eer");

    // Execute generation with streaming callback
    runtime.generate("Clinical Assessment: Patient has persistent cough...", 64,
        [](const std::string& token, int token_id, bool is_final) {
            std::cout << token << std::flush;
        }
    );

    runtime.unloadModel();
    return 0;
}
```

### 3. Build & Deploy Android APK (Headless / Zero Android Studio)
Run the automated build script on Windows or Linux to build the C++20 NDK binaries and deploy directly to any connected phone via ADB:
```powershell
powershell -ExecutionPolicy Bypass -File .\embeddra-edge\android\build_apk.ps1
```

### 4. Run the Standalone On-Device Benchmark CLI
Push and run the standalone command-line benchmark directly in the Android shell:
```bash
adb push embeddra_cli_arm64 /data/local/tmp/embeddra_cli
adb shell "chmod 755 /data/local/tmp/embeddra_cli && /data/local/tmp/embeddra_cli"
```

---

## 🔒 Privacy & Security

* **100% On-Device**: Zero cloud endpoints, zero remote telemetry, and zero tracking.
* **Air-Gapped Operation**: Runs with full functionality in airplane mode.
* **Unified Memory Security**: Tensors are read-only memory-mapped (`PROT_READ`), preventing memory mutation.

---

## 📄 License

This project is licensed under the Apache License 2.0.
