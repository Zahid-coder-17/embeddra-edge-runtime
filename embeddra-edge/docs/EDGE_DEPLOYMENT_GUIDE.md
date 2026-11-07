# Embeddra Edge Runtime (EER) - Mobile & Edge Deployment Guide
**Target Model:** Maria 1B (HyperGQE Architecture)  
**Target Platform:** Android (ARM64-v8a) & Embedded Edge Linux  

---

## 1. System Overview

Embeddra Edge Runtime (EER) is a standalone, on-device AI inference engine built specifically to deploy Embeddra's proprietary **HyperGQE** language models on mobile smartphones without external cloud dependencies.

```
┌────────────────────────────────────────────────────────┐
│             Android Application UI (Kotlin)            │
│          MainActivity.kt & Streaming Coroutines        │
└───────────────────────────┬────────────────────────────┘
                            │ JNI Bridge
┌───────────────────────────▼────────────────────────────┐
│                  embeddra_jni.cpp                      │
└───────────────────────────┬────────────────────────────┘
                            │ Native C++20 Interface
┌───────────────────────────▼────────────────────────────┐
│             EmbeddraRuntime (Core Engine)              │
│       ModelManager │ MemoryArena │ HardwareDetector     │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│               HyperGQE Backend Engine                  │
│  RoPE │ Dynamic Router │ Hypernet W_D/W_U │ SwiGLU FFN  │
└────────────────────────────────────────────────────────┘
```

---

## 2. Pushing the Model Checkpoint to Android Device

To execute Maria-1B completely offline, push the checkpoint to your smartphone's application internal storage using `adb`:

```bash
# Push weights to Android app external files directory
adb push checkpoints/maria_1b_20B_weights.pt /sdcard/Android/data/com.embeddra.edge/files/

# Or push to local temp folder for rooted / development devices:
adb push checkpoints/maria_1b_20B_weights.pt /data/local/tmp/
```

---

## 3. Building Native C++ Libraries via Android NDK

To cross-compile `libembeddra_jni.so` for `arm64-v8a`:

```bash
cd embeddra-edge

cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-android --config Release
```

---

## 4. Mobile Memory & Precision Footprint

| Precision | Model Weights Size | KV Cache (2K Context) | Peak Mobile RAM | Supported Devices |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | ~4.29 GB | 72 MB | ~4.8 GB | High-end ($\ge 8\text{ GB}$ RAM) |
| **BFloat16 / FP16** | ~2.15 GB | 36 MB | ~2.5 GB | Mid-range ($\ge 6\text{ GB}$ RAM) |
| **INT8 (Quantized)**| ~1.07 GB | 18 MB | ~1.3 GB | Entry / Budget ($\ge 4\text{ GB}$ RAM) |
| **INT4 (Group-wise)**| ~540 MB | 18 MB | ~750 MB | Ultra-low power / Edge IoT |

---

## 5. Offline Verification Checklist

- [x] Checkpoint loaded locally without network requests.
- [x] Tokenizer executed via local BPE tables.
- [x] Streaming output delivered token-by-token.
- [x] Stop-generation interrupts inference immediately.
- [x] Zero heap allocations inside autoregressive decoding loop.

