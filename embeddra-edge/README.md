# Embeddra Edge Runtime (EER)

> **High-Performance On-Device AI Inference Infrastructure for Maria 1B (HyperGQE)**

Embeddra Edge Runtime (EER) is a purpose-built, zero-dependency edge AI runtime engineered to execute Embeddra's proprietary language models locally on Android smartphones, embedded systems, and resource-constrained edge hardware.

---

## Key Highlights

- **Proprietary Architecture Support:** Complete, native execution of **HyperGQE** (Hypernetwork-based Grouped Query Experts). No forced translation to generic GQA or llama.cpp.
- **Offline First:** 100% self-contained inference without network connectivity.
- **Modular C++20 Core:** Clean separation between memory arena, hardware detection, model management, and backend inference engines.
- **Mobile-Ready:** Native JNI bridge and Kotlin Android UI with real-time token streaming and dynamic cancellation.
- **Zero-Allocation Decoding:** Pre-allocated tensor scratchpad and circular KV cache buffer.

---

## Directory Layout

```
embeddra-edge/
├── core/                         # Core Runtime Framework
│   ├── hardware/                 # CPU core, RAM, and SIMD feature detection
│   ├── memory_manager/           # Fixed-arena tensor allocator & KV cache manager
│   ├── model_manager/            # Checkpoint loader & manifest parser
│   ├── runtime/                  # EmbeddraRuntime C++20 API & streaming callbacks
│   └── scheduler/                # Execution scheduling & threadpool
│
├── hypergqe/                     # HyperGQE Architectural Implementation
│   ├── attention/                # HyperGQE Attention core (RoPE, Router, Hypernet)
│   ├── hypernetwork/             # Low-rank W_D & W_U weight generation
│   ├── model_adapter/            # Complete MariaForCausalLM PyTorch model
│   └── routing/                  # Token-conditional expert routing
│
├── backends/                     # Execution Engines
│   ├── custom_cpp/               # Native C++ inference engine (NEON / FP16 optimized)
│   └── executorch/               # ExecuTorch operator evaluation & delegates
│
├── export/                       # Verification & Quantization Pipeline
│   ├── checkpoint_inspection/    # Checkpoint integrity & tensor analyzer
│   ├── mobile_export/            # Configuration exporter & operator audit
│   └── pytorch_validation/       # Numerical parity test suite
│
├── sdk/                          # Developer Interfaces
│   ├── cpp/                      # C++ CLI testing tool
│   └── android/                  # Android SDK bindings
│
├── android/                      # Mobile Application
│   ├── app/                      # Android Kotlin App (Streaming UI & metrics)
│   └── jni/                      # C++ JNI bridge (libembeddra_jni.so)
│
├── benchmarks/                   # Embeddra EdgeBench
│   └── edge_bench.py             # TTFT, tokens/sec, and memory benchmark
│
├── docs/                         # Technical Specifications
│   ├── CHECKPOINT_INSPECTION_REPORT.md
│   └── EDGE_DEPLOYMENT_GUIDE.md
│
├── CMakeLists.txt                # Unified build system (Host & Android NDK)
└── README.md
```

---

## Quickstart

### 1. Validate Checkpoint & PyTorch Reconstruction
```bash
python embeddra-edge/export/checkpoint_inspection/inspect_checkpoint.py
python embeddra-edge/export/pytorch_validation/validate_inference.py
```

### 2. Run Edge Benchmark
```bash
python embeddra-edge/benchmarks/edge_bench.py
```

### 3. Build Standalone C++ Engine (Host)
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
./embeddra_cli ../checkpoints/maria_1b_20B_weights.pt
```

### 4. Build Android APK
Open `embeddra-edge/android` in Android Studio or run via Gradle:
```bash
cd embeddra-edge/android
./gradlew assembleRelease
```
Install the generated APK onto your Android device:
```bash
adb install -r app/build/outputs/apk/release/app-release.apk
adb push ../checkpoints/maria_1b_20B_weights.pt /sdcard/Android/data/com.embeddra.edge/files/
```

