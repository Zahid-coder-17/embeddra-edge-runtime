# Embeddra Edge Android SDK

The **Embeddra Edge Android SDK** enables any Android application to run the **Maria-1B (HyperGQE)** model locally, on-device, and 100% offline with GPU acceleration (ARM Mali OpenCL FP16 UMA) and multi-core CPU acceleration (ARM NEON FP16).

---

## 1. Quick Integration

### Step 1: Copy SDK Files into Your Android Project
1. Copy `src/com/embeddra/edge/` (`EmbeddraRuntime.kt`, `VocabularyDecoder.kt`) into your app's `src/main/java/com/embeddra/edge/`.
2. Copy `jniLibs/arm64-v8a/` (`libembeddra_jni.so`, `libembeddra_edge.so`, `libc++_shared.so`) into your app's `src/main/jniLibs/arm64-v8a/`.
3. Add native library permission to your `AndroidManifest.xml`:
   ```xml
   <application ...>
       <uses-native-library android:name="libOpenCL.so" android:required="false" />
   </application>
   ```

---

## 2. Usage in Kotlin / Java

```kotlin
import com.embeddra.edge.EmbeddraRuntime
import com.embeddra.edge.VocabularyDecoder

// 1. Initialize Runtime and Tokenizer
val runtime = EmbeddraRuntime.getInstance()
runtime.initialize()

val tokenizer = VocabularyDecoder(context)
tokenizer.loadVocabulary()

// 2. Load the Local Maria-1B Checkpoint (Zero-Copy Mmap)
val modelPath = "/sdcard/Android/data/com.your.app/files/maria_1b.eer"
val loaded = runtime.loadModel(modelPath)

println("Backend Active: ${runtime.getBackendInfo()}")
// Prints: GPU: Mali-G68 r1p1 (or CPU: 8-Core ARM NEON)

// 3. Stream Inference Tokens
val prompt = "Question: What are symptoms of asthma?\nAnswer:"
val tokens = tokenizer.encode(prompt)

runtime.generateTokens(tokens, maxTokens = 64, temperature = 0.7f, object : EmbeddraRuntime.TokenListener {
    override fun onToken(token: String, tokenId: Int, isFinal: Boolean) {
        val word = tokenizer.decodeToken(tokenId)
        print(word) // Streams directly to UI
        if (isFinal) {
            val stats = runtime.getPerformanceStats()
            println("\nTTFT: ${stats.timeToFirstTokenMs} ms | Speed: ${stats.tokensPerSecond} tok/s")
        }
    }
})
```

---

## 3. Hardware Acceleration Modes

The SDK automatically chooses the optimal hardware:
* **Mali GPU**: Full OpenCL 3.0 UMA offload (`libOpenCL.so`) without copying weights to VRAM.
* **8-Core CPU**: 8-thread worker pool with ARMv8.2-A NEON SIMD (`float16x8_t`).
* **Zero Cloud Dependency**: 0 network requests, completely private on-device processing.

