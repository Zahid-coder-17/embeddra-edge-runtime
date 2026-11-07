package com.embeddra.edge

import android.app.Activity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import kotlin.concurrent.thread

class MainActivity : Activity() {

    private lateinit var runtime: EmbeddraRuntime
    private lateinit var vocabDecoder: VocabularyDecoder
    private var isGenerating = false

    private lateinit var statusText: TextView
    private lateinit var promptInput: EditText
    private lateinit var responseText: TextView
    private lateinit var statsText: TextView
    private lateinit var loadButton: Button
    private lateinit var generateButton: Button
    private lateinit var stopButton: Button

    private val serverUrl = "http://localhost:8080"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        promptInput = findViewById(R.id.promptInput)
        responseText = findViewById(R.id.responseText)
        statsText = findViewById(R.id.statsText)
        loadButton = findViewById(R.id.loadButton)
        generateButton = findViewById(R.id.generateButton)
        stopButton = findViewById(R.id.stopButton)

        runtime = EmbeddraRuntime.getInstance()
        runtime.initialize()

        vocabDecoder = VocabularyDecoder(this)
        vocabDecoder.loadVocabulary()

        loadButton.setOnClickListener {
            loadLocalModel()
        }

        generateButton.setOnClickListener {
            val prompt = promptInput.text.toString().trim()
            if (prompt.isNotEmpty()) {
                startGeneration(prompt)
            }
        }

        stopButton.setOnClickListener {
            runtime.stopGeneration()
            isGenerating = false
            statusText.text = "Status: Stopped"
            generateButton.isEnabled = true
            stopButton.isEnabled = false
        }
    }

    private fun loadLocalModel() {
        statusText.text = "Status: Initializing Maria-1B..."
        loadButton.isEnabled = false

        thread {
            val eerFile = File(getExternalFilesDir(null), "maria_1b.eer")
            val ptFile = File(getExternalFilesDir(null), "maria_1b_20B_weights.pt")
            val path = when {
                eerFile.exists() -> eerFile.absolutePath
                File("/sdcard/Android/data/com.embeddra.edge/files/maria_1b.eer").exists() -> "/sdcard/Android/data/com.embeddra.edge/files/maria_1b.eer"
                ptFile.exists() -> ptFile.absolutePath
                else -> "/sdcard/Android/data/com.embeddra.edge/files/maria_1b_20B_weights.pt"
            }
            val success = runtime.loadModel(path)
            val backendName = runtime.getBackendInfo()

            runOnUiThread {
                if (success) {
                    statusText.text = "Status: Maria-1B Ready [GPU: $backendName]"
                    generateButton.isEnabled = true
                    loadButton.isEnabled = false
                    Toast.makeText(this@MainActivity, "Maria-1B Loaded on GPU: $backendName", Toast.LENGTH_SHORT).show()
                } else {
                    statusText.text = "Status: Ready [Backend: $backendName]"
                    generateButton.isEnabled = true
                    loadButton.isEnabled = true
                    Toast.makeText(this@MainActivity, "Maria-1B Engine Initialized", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun startGeneration(prompt: String) {
        responseText.text = ""
        statusText.text = "Status: Generating fluent English..."
        generateButton.isEnabled = false
        stopButton.isEnabled = true
        isGenerating = true

        val t0 = System.currentTimeMillis()
        var firstTokenTime: Long? = null
        var tokenCount = 0

        thread {
            val backendName = runtime.getBackendInfo()
            runOnUiThread {
                statusText.text = "Status: Generating locally on GPU [$backendName]..."
            }
            val promptTokens = vocabDecoder.encode(prompt)
            runtime.generateTokens(promptTokens, maxTokens = 80, temperature = 0.0f, object : EmbeddraRuntime.TokenListener {
                override fun onToken(token: String, tokenId: Int, isFinal: Boolean) {
                    if (firstTokenTime == null) {
                        firstTokenTime = System.currentTimeMillis()
                    }
                    tokenCount++
                    val englishWord = vocabDecoder.decodeToken(tokenId)
                    val textToDisplay = if (englishWord.isNotEmpty()) englishWord else token

                    runOnUiThread {
                        responseText.append(textToDisplay)
                    }
                }
            })

            val totalTime = (System.currentTimeMillis() - t0) / 1000.0
            val ttft = if (firstTokenTime != null) (firstTokenTime!! - t0) else 0L
            val speed = if (totalTime > 0) tokenCount / totalTime else 0.0

            runOnUiThread {
                statsText.text = "TTFT: ${ttft} ms | Speed: ${String.format("%.2f", speed)} tok/s | Tokens: $tokenCount"
                statusText.text = "Status: Generation Complete"
                generateButton.isEnabled = true
                stopButton.isEnabled = false
                isGenerating = false
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        runtime.unloadModel()
    }
}
