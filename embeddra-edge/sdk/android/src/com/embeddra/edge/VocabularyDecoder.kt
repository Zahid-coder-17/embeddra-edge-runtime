package com.embeddra.edge

import android.content.Context
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.HashMap

class VocabularyDecoder(private val context: Context) {

    private val vocabList = ArrayList<String>(128000)
    private val tokenToIdMap = HashMap<String, Int>(128000)
    private var isLoaded = false

    fun loadVocabulary(): Boolean {
        if (isLoaded) return true
        try {
            val inputStream: InputStream = context.assets.open("vocab.bin")
            val bytes = inputStream.readBytes()
            inputStream.close()

            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val numTokens = buffer.int

            for (i in 0 until numTokens) {
                val len = buffer.short.toInt() and 0xFFFF
                val strBytes = ByteArray(len)
                buffer.get(strBytes)
                val rawStr = String(strBytes, Charsets.UTF_8)
                val str = rawStr.replace("Ġ", " ")
                vocabList.add(str)
                tokenToIdMap[str] = i
                tokenToIdMap[rawStr] = i
                tokenToIdMap[str.trim()] = i
            }
            isLoaded = true
            return true
        } catch (e: Exception) {
            e.printStackTrace()
            return false
        }
    }

    fun decodeToken(tokenId: Int): String {
        if (!isLoaded || tokenId < 0 || tokenId >= vocabList.size) {
            return ""
        }
        return vocabList[tokenId]
    }

    fun encode(rawText: String): IntArray {
        if (!isLoaded || rawText.isEmpty()) return intArrayOf(128000)

        val text = if (!rawText.contains("Question:") && !rawText.endsWith(":") && !rawText.endsWith("is")) {
            "Question: ${rawText.trim()}\nAnswer:"
        } else {
            rawText.trim()
        }

        val result = ArrayList<Int>()
        result.add(128000) // Prepend <|begin_of_text|> BOS token

        var i = 0
        while (i < text.length) {
            if (text[i] == '\n') {
                result.add(198)
                i++
                continue
            }

            var matched = false
            val maxLen = minOf(30, text.length - i)
            for (len in maxLen downTo 1) {
                val sub = text.substring(i, i + len)
                val id = tokenToIdMap[sub]
                if (id != null) {
                    result.add(id)
                    i += len
                    matched = true
                    break
                }
            }

            if (!matched) {
                val chStr = text[i].toString()
                val id = tokenToIdMap[chStr]
                if (id != null) {
                    result.add(id)
                }
                i++
            }
        }

        return result.toIntArray()
    }
}
