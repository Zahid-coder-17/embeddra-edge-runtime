#include <jni.h>
#include <string>
#include <memory>
#include <android/log.h>
#include "../../core/runtime/embeddra_runtime.h"

#define TAG "EmbeddraJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static std::unique_ptr<embeddra::runtime::EmbeddraRuntime> g_runtime = nullptr;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_embeddra_edge_EmbeddraRuntime_nativeInit(JNIEnv* env, jobject thiz) {
    LOGI("Initializing Embeddra Edge Runtime native core...");
    if (!g_runtime) {
        g_runtime = std::make_unique<embeddra::runtime::EmbeddraRuntime>();
    }
    return g_runtime->initialize() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_embeddra_edge_EmbeddraRuntime_nativeLoadModel(JNIEnv* env, jobject thiz, jstring jpath) {
    if (!g_runtime) {
        LOGE("Runtime not initialized before calling nativeLoadModel.");
        return JNI_FALSE;
    }
    const char* path_chars = env->GetStringUTFChars(jpath, nullptr);
    std::string path_str(path_chars);
    env->ReleaseStringUTFChars(jpath, path_chars);

    LOGI("Loading Maria-1B model from device path: %s", path_str.c_str());
    bool ok = g_runtime->loadModel(path_str);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_embeddra_edge_EmbeddraRuntime_nativeGenerate(
    JNIEnv* env,
    jobject thiz,
    jstring jprompt,
    jint max_tokens,
    jfloat temperature,
    jobject jlistener
) {
    if (!g_runtime || !g_runtime->isModelLoaded()) {
        LOGE("Model is not loaded. Cannot execute generation.");
        return;
    }

    const char* prompt_chars = env->GetStringUTFChars(jprompt, nullptr);
    std::string prompt_str(prompt_chars);
    env->ReleaseStringUTFChars(jprompt, prompt_chars);

    jclass listener_class = env->GetObjectClass(jlistener);
    jmethodID on_token_method = env->GetMethodID(listener_class, "onToken", "(Ljava/lang/String;IZ)V");

    embeddra::runtime::GenerationConfig config;
    config.max_tokens = max_tokens;
    config.temperature = temperature;

    g_runtime->generateWithConfig(prompt_str, config, [&](const std::string& token, int token_id, bool is_final) {
        jstring jtoken = env->NewStringUTF(token.c_str());
        env->CallVoidMethod(jlistener, on_token_method, jtoken, (jint)token_id, is_final ? JNI_TRUE : JNI_FALSE);
        env->DeleteLocalRef(jtoken);
    });
}

JNIEXPORT void JNICALL
Java_com_embeddra_edge_EmbeddraRuntime_nativeGenerateTokens(
    JNIEnv* env,
    jobject thiz,
    jintArray jtokens,
    jint max_tokens,
    jfloat temperature,
    jobject jlistener
) {
    (void)thiz;
    if (!g_runtime || !g_runtime->isModelLoaded()) {
        LOGE("Model is not loaded. Cannot execute generation.");
        return;
    }

    std::vector<int> prompt_tokens;
    if (jtokens) {
        jsize len = env->GetArrayLength(jtokens);
        jint* elements = env->GetIntArrayElements(jtokens, nullptr);
        prompt_tokens.assign(elements, elements + len);
        env->ReleaseIntArrayElements(jtokens, elements, JNI_ABORT);
    }

    jclass listener_class = env->GetObjectClass(jlistener);
    jmethodID on_token_method = env->GetMethodID(listener_class, "onToken", "(Ljava/lang/String;IZ)V");

    embeddra::runtime::GenerationConfig config;
    config.max_tokens = max_tokens;
    config.temperature = temperature;

    g_runtime->generateWithTokens(prompt_tokens, config, [&](const std::string& token, int token_id, bool is_final) {
        jstring jtoken = env->NewStringUTF(token.c_str());
        env->CallVoidMethod(jlistener, on_token_method, jtoken, (jint)token_id, is_final ? JNI_TRUE : JNI_FALSE);
        env->DeleteLocalRef(jtoken);
    });
}

JNIEXPORT void JNICALL
Java_com_embeddra_edge_EmbeddraRuntime_nativeStopGeneration(JNIEnv* env, jobject thiz) {
    if (g_runtime) {
        LOGI("Received request to stop token generation.");
        g_runtime->stopGeneration();
    }
}

JNIEXPORT void JNICALL
Java_com_embeddra_edge_EmbeddraRuntime_nativeUnloadModel(JNIEnv* env, jobject thiz) {
    if (g_runtime) {
        LOGI("Unloading Maria-1B model from memory.");
        g_runtime->unloadModel();
    }
}

JNIEXPORT jdoubleArray JNICALL
Java_com_embeddra_edge_EmbeddraRuntime_nativeGetStats(JNIEnv* env, jobject thiz) {
    jdoubleArray result = env->NewDoubleArray(4);
    if (!g_runtime) return result;

    auto stats = g_runtime->getLastStats();
    jdouble fill[4];
    fill[0] = stats.time_to_first_token_ms;
    fill[1] = stats.total_generation_time_ms;
    fill[2] = static_cast<double>(stats.tokens_generated);
    fill[3] = stats.tokens_per_second;

    env->SetDoubleArrayRegion(result, 0, 4, fill);
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_embeddra_edge_EmbeddraRuntime_nativeGetBackendInfo(JNIEnv* env, jobject thiz) {
    if (!g_runtime) {
        return env->NewStringUTF("Offline (Uninitialized)");
    }
    std::string info = g_runtime->getDeviceBackendInfo();
    return env->NewStringUTF(info.c_str());
}

} // extern "C"

