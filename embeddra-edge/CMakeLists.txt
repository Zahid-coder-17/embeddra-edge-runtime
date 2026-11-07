cmake_minimum_required(VERSION 3.20)
project(embeddra_edge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for Edge / Mobile
if(MSVC)
    add_compile_options(/O2 /W4 /fp:fast)
else()
    add_compile_options(-O3 -Wall -Wextra -ffast-math)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        add_compile_options(-march=armv8.2-a+fp16+dotprod)
    endif()
endif()

# Core sources
set(CORE_SOURCES
    core/hardware/hardware_detector.cpp
    core/memory_manager/memory_pool.cpp
    core/runtime/embeddra_runtime.cpp
    backends/custom_cpp/hypergqe_engine.cpp
    backends/gpu/gpu_engine.cpp
)

# Shared library for Android / C++ SDK
add_library(embeddra_edge SHARED ${CORE_SOURCES})
target_include_directories(embeddra_edge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/core/memory_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/core/hardware
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/custom_cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/gpu
)
if(ANDROID)
    target_link_libraries(embeddra_edge log dl)
endif()

# Standalone CLI binary for host testing
if(NOT ANDROID)
    add_executable(embeddra_cli sdk/cpp/main.cpp ${CORE_SOURCES})
    target_include_directories(embeddra_cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/core/memory_manager
        ${CMAKE_CURRENT_SOURCE_DIR}/core/hardware
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/custom_cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/gpu
    )
endif()

# Android JNI library
if(ANDROID)
    add_library(embeddra_jni SHARED
        android/jni/embeddra_jni.cpp
        ${CORE_SOURCES}
    )
    target_link_libraries(embeddra_jni log dl)
    target_include_directories(embeddra_jni PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/runtime
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/custom_cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/gpu
    )
endif()

