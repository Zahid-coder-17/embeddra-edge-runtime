package com.embeddra.edge

data class PerformanceMetrics(
    val timeToFirstTokenMs: Double,
    val totalTimeMs: Double,
    val tokensGenerated: Int,
    val tokensPerSecond: Double
)

class EmbeddraRuntime private constructor() {

    init {
        System.loadLibrary("embeddra_jni")
    }

    interface TokenListener {
        fun onToken(token: String, tokenId: Int, isFinal: Boolean)
    }

    fun initialize(): Boolean = nativeInit()

    fun loadModel(path: String): Boolean = nativeLoadModel(path)

    fun generate(prompt: String, maxTokens: Int = 128, temperature: Float = 0.7f, listener: TokenListener) {
        nativeGenerate(prompt, maxTokens, temperature, listener)
    }

    fun generateTokens(tokens: IntArray, maxTokens: Int = 128, temperature: Float = 0.7f, listener: TokenListener) {
        nativeGenerateTokens(tokens, maxTokens, temperature, listener)
    }

    fun stopGeneration() = nativeStopGeneration()

    fun unloadModel() = nativeUnloadModel()

    fun getPerformanceStats(): PerformanceMetrics {
        val stats = nativeGetStats()
        return PerformanceMetrics(
            timeToFirstTokenMs = stats[0],
            totalTimeMs = stats[1],
            tokensGenerated = stats[2].toInt(),
            tokensPerSecond = stats[3]
        )
    }

    fun getBackendInfo(): String = nativeGetBackendInfo()

    private external fun nativeInit(): Boolean
    private external fun nativeLoadModel(path: String): Boolean
    private external fun nativeGenerate(prompt: String, maxTokens: Int, temperature: Float, listener: TokenListener)
    private external fun nativeGenerateTokens(tokens: IntArray, maxTokens: Int, temperature: Float, listener: TokenListener)
    private external fun nativeStopGeneration()
    private external fun nativeUnloadModel()
    private external fun nativeGetStats(): DoubleArray
    private external fun nativeGetBackendInfo(): String

    companion object {
        @Volatile
        private var instance: EmbeddraRuntime? = null

        fun getInstance(): EmbeddraRuntime =
            instance ?: synchronized(this) {
                instance ?: EmbeddraRuntime().also { instance = it }
            }
    }
}
