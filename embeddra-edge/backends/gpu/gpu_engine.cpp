#include "gpu_engine.h"
#include <iostream>
#include <cstring>
#include <dlfcn.h>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_GPU(...) __android_log_print(ANDROID_LOG_INFO, "EmbeddraGPU", __VA_ARGS__)
#define LOG_GPU_ERR(...) __android_log_print(ANDROID_LOG_ERROR, "EmbeddraGPU", __VA_ARGS__)
#else
#define LOG_GPU(...) std::cout << "[EmbeddraGPU] " << __VA_ARGS__ << "\n"
#define LOG_GPU_ERR(...) std::cerr << "[EmbeddraGPU ERROR] " << __VA_ARGS__ << "\n"
#endif

namespace embeddra {
namespace backend {

// Standard OpenCL Types & Constants
typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_program;
typedef void* cl_kernel;

#define CL_SUCCESS 0
#define CL_FALSE 0
#define CL_TRUE 1
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_USE_HOST_PTR (1 << 3)
#define CL_DEVICE_NAME 0x102B
#define CL_PROGRAM_BUILD_LOG 0x1183

// OpenCL Function Pointers
typedef cl_int (*pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*pfn_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_context (*pfn_clCreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*pfn_clCreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
typedef cl_mem (*pfn_clCreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
typedef cl_program (*pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_int (*pfn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
typedef cl_kernel (*pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_int (*pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const void*, void*);
typedef cl_int (*pfn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, unsigned int, size_t, size_t, const void*, cl_uint, const void*, void*);
typedef cl_int (*pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, unsigned int, size_t, size_t, void*, cl_uint, const void*, void*);
typedef cl_int (*pfn_clFinish)(cl_command_queue);
typedef cl_int (*pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (*pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (*pfn_clReleaseProgram)(cl_program);
typedef cl_int (*pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*pfn_clReleaseContext)(cl_context);

static pfn_clGetPlatformIDs fn_clGetPlatformIDs = nullptr;
static pfn_clGetDeviceIDs fn_clGetDeviceIDs = nullptr;
static pfn_clGetDeviceInfo fn_clGetDeviceInfo = nullptr;
static pfn_clCreateContext fn_clCreateContext = nullptr;
static pfn_clCreateCommandQueue fn_clCreateCommandQueue = nullptr;
static pfn_clCreateBuffer fn_clCreateBuffer = nullptr;
static pfn_clCreateProgramWithSource fn_clCreateProgramWithSource = nullptr;
static pfn_clBuildProgram fn_clBuildProgram = nullptr;
static pfn_clGetProgramBuildInfo fn_clGetProgramBuildInfo = nullptr;
static pfn_clCreateKernel fn_clCreateKernel = nullptr;
static pfn_clSetKernelArg fn_clSetKernelArg = nullptr;
static pfn_clEnqueueNDRangeKernel fn_clEnqueueNDRangeKernel = nullptr;
static pfn_clEnqueueWriteBuffer fn_clEnqueueWriteBuffer = nullptr;
static pfn_clEnqueueReadBuffer fn_clEnqueueReadBuffer = nullptr;
static pfn_clFinish fn_clFinish = nullptr;
static pfn_clReleaseMemObject fn_clReleaseMemObject = nullptr;
static pfn_clReleaseKernel fn_clReleaseKernel = nullptr;
static pfn_clReleaseProgram fn_clReleaseProgram = nullptr;
static pfn_clReleaseCommandQueue fn_clReleaseCommandQueue = nullptr;
static pfn_clReleaseContext fn_clReleaseContext = nullptr;

static const char* kGemvKernelSource = R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void gemv_fp16_mali(
    __global float* restrict out,
    __global const half* restrict weight,
    __global const float* restrict in_vec,
    const int M,
    const int N
) {
    int r = get_global_id(0);
    if (r >= M) return;

    __global const half* row = weight + (size_t)r * N;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    
    int c = 0;
    int N16 = N & ~15;
    for (; c < N16; c += 16) {
        half8 w0 = vload8(0, row + c);
        half8 w1 = vload8(0, row + c + 8);
        sum0 += (float)w0.s0 * in_vec[c + 0]
              + (float)w0.s1 * in_vec[c + 1]
              + (float)w0.s2 * in_vec[c + 2]
              + (float)w0.s3 * in_vec[c + 3]
              + (float)w0.s4 * in_vec[c + 4]
              + (float)w0.s5 * in_vec[c + 5]
              + (float)w0.s6 * in_vec[c + 6]
              + (float)w0.s7 * in_vec[c + 7];
        sum1 += (float)w1.s0 * in_vec[c + 8]
              + (float)w1.s1 * in_vec[c + 9]
              + (float)w1.s2 * in_vec[c + 10]
              + (float)w1.s3 * in_vec[c + 11]
              + (float)w1.s4 * in_vec[c + 12]
              + (float)w1.s5 * in_vec[c + 13]
              + (float)w1.s6 * in_vec[c + 14]
              + (float)w1.s7 * in_vec[c + 15];
    }
    for (; c < N; ++c) {
        sum0 += (float)row[c] * in_vec[c];
    }
    out[r] = sum0 + sum1;
}
)";

GPUEngine::GPUEngine() = default;

GPUEngine::~GPUEngine() {
    shutdown();
}

bool GPUEngine::loadOpenCLSymbols() {
    const char* libs[] = {
        "libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/vendor/lib64/egl/libGLES_mali.so"
    };

    for (const char* path : libs) {
        lib_handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (lib_handle_) {
            LOG_GPU("Successfully loaded OpenCL library from: %s", path);
            break;
        } else {
            LOG_GPU("Attempt to load %s gave: %s", path, dlerror());
        }
    }

    if (!lib_handle_) {
        LOG_GPU_ERR("Failed to load OpenCL library: %s", dlerror());
        return false;
    }

    #define BIND(name) \
        fn_##name = (pfn_##name)dlsym(lib_handle_, #name); \
        if (!fn_##name) { \
            LOG_GPU_ERR("Missing symbol: " #name); \
            return false; \
        }

    BIND(clGetPlatformIDs);
    BIND(clGetDeviceIDs);
    BIND(clGetDeviceInfo);
    BIND(clCreateContext);
    BIND(clCreateCommandQueue);
    BIND(clCreateBuffer);
    BIND(clCreateProgramWithSource);
    BIND(clBuildProgram);
    BIND(clGetProgramBuildInfo);
    BIND(clCreateKernel);
    BIND(clSetKernelArg);
    BIND(clEnqueueNDRangeKernel);
    BIND(clEnqueueWriteBuffer);
    BIND(clEnqueueReadBuffer);
    BIND(clFinish);
    BIND(clReleaseMemObject);
    BIND(clReleaseKernel);
    BIND(clReleaseProgram);
    BIND(clReleaseCommandQueue);
    BIND(clReleaseContext);

    #undef BIND
    return true;
}

bool GPUEngine::initialize() {
    if (initialized_) return true;

    if (!loadOpenCLSymbols()) {
        LOG_GPU("OpenCL runtime not available on this platform.");
        return false;
    }

    cl_platform_id platform;
    cl_uint num_platforms = 0;
    cl_int err = fn_clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        LOG_GPU_ERR("No OpenCL platforms found. err=%d", err);
        return false;
    }
    cl_platform_ = platform;

    cl_device_id device;
    cl_uint num_devices = 0;
    err = fn_clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        LOG_GPU_ERR("No OpenCL GPU device found. err=%d", err);
        return false;
    }
    cl_device_ = device;

    char dev_name[256] = {0};
    fn_clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    device_name_ = dev_name;
    LOG_GPU("Mobile GPU Backend initialized: %s", device_name_.c_str());

    cl_context ctx = fn_clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !ctx) {
        LOG_GPU_ERR("Failed to create OpenCL context: err=%d", err);
        return false;
    }
    cl_context_ = ctx;

    cl_command_queue queue = fn_clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS || !queue) {
        LOG_GPU_ERR("Failed to create OpenCL command queue: err=%d", err);
        return false;
    }
    cl_queue_ = queue;

    // Compile FP16 GEMV Kernel for Mali GPU
    const char* src_ptr = kGemvKernelSource;
    size_t src_len = strlen(kGemvKernelSource);
    cl_program prog = fn_clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS || !prog) {
        LOG_GPU_ERR("Failed to create OpenCL program: err=%d", err);
        return false;
    }

    const char* build_opts = "-cl-fast-relaxed-math -cl-mad-enable";
    err = fn_clBuildProgram(prog, 1, &device, build_opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log_buf[4096] = {0};
        fn_clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, sizeof(log_buf), log_buf, nullptr);
        LOG_GPU_ERR("OpenCL Build error:\n%s", log_buf);
        fn_clReleaseProgram(prog);
        return false;
    }
    cl_program_ = prog;

    cl_kernel kernel = fn_clCreateKernel(prog, "gemv_fp16_mali", &err);
    if (err != CL_SUCCESS || !kernel) {
        LOG_GPU_ERR("Failed to create gemv_fp16_mali kernel: err=%d", err);
        return false;
    }
    cl_kernel_gemv_ = kernel;

    // Pre-allocate initial reusable I/O buffers (Hidden Size 2048, Intermediate 5632, Vocab 128256)
    current_in_capacity_ = 8192 * sizeof(float);
    current_out_capacity_ = 131072 * sizeof(float);

    d_in_buf_ = fn_clCreateBuffer(ctx, CL_MEM_READ_ONLY, current_in_capacity_, nullptr, &err);
    d_out_buf_ = fn_clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, current_out_capacity_, nullptr, &err);

    initialized_ = true;
    LOG_GPU("Mobile GPU FP16 Kernel successfully compiled and ready for Mali-G68 execution!");
    return true;
}

void* GPUEngine::getOrCreateWeightBuffer(const void* host_ptr, size_t size_bytes) {
    if (!initialized_ || !host_ptr) return nullptr;

    auto it = weight_buffer_cache_.find(host_ptr);
    if (it != weight_buffer_cache_.end()) {
        return it->second;
    }

    // Zero-copy UMA buffer creation on Mali GPU
    cl_int err = 0;
    cl_mem buf = fn_clCreateBuffer(
        static_cast<cl_context>(cl_context_),
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        size_bytes,
        const_cast<void*>(host_ptr),
        &err
    );

    if (err == CL_SUCCESS && buf) {
        weight_buffer_cache_[host_ptr] = buf;
        return buf;
    }

    LOG_GPU_ERR("Failed to create GPU buffer for host_ptr %p (size %zu): err=%d", host_ptr, size_bytes, err);
    return nullptr;
}

void GPUEngine::gemv_fp16(float* out, const uint16_t* weight, const float* in, int M, int N) {
    if (!initialized_ || !weight || !in || !out) return;

    size_t in_bytes = static_cast<size_t>(N) * sizeof(float);
    size_t out_bytes = static_cast<size_t>(M) * sizeof(float);
    size_t w_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(uint16_t);

    void* w_buf = getOrCreateWeightBuffer(weight, w_bytes);
    if (!w_buf) return;

    // Check capacity for input buffer
    if (in_bytes > current_in_capacity_) {
        if (d_in_buf_) fn_clReleaseMemObject(static_cast<cl_mem>(d_in_buf_));
        current_in_capacity_ = in_bytes * 2;
        cl_int err = 0;
        d_in_buf_ = fn_clCreateBuffer(static_cast<cl_context>(cl_context_), CL_MEM_READ_ONLY, current_in_capacity_, nullptr, &err);
    }

    // Check capacity for output buffer
    if (out_bytes > current_out_capacity_) {
        if (d_out_buf_) fn_clReleaseMemObject(static_cast<cl_mem>(d_out_buf_));
        current_out_capacity_ = out_bytes * 2;
        cl_int err = 0;
        d_out_buf_ = fn_clCreateBuffer(static_cast<cl_context>(cl_context_), CL_MEM_WRITE_ONLY, current_out_capacity_, nullptr, &err);
    }

    cl_command_queue queue = static_cast<cl_command_queue>(cl_queue_);
    cl_kernel kernel = static_cast<cl_kernel>(cl_kernel_gemv_);

    // Upload input activation vector (small: ~2KB to 22KB)
    fn_clEnqueueWriteBuffer(queue, static_cast<cl_mem>(d_in_buf_), CL_FALSE, 0, in_bytes, in, 0, nullptr, nullptr);

    // Set Kernel arguments
    cl_mem d_out = static_cast<cl_mem>(d_out_buf_);
    cl_mem d_w = static_cast<cl_mem>(w_buf);
    cl_mem d_in = static_cast<cl_mem>(d_in_buf_);

    fn_clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_out);
    fn_clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_w);
    fn_clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_in);
    fn_clSetKernelArg(kernel, 3, sizeof(int), &M);
    fn_clSetKernelArg(kernel, 4, sizeof(int), &N);

    size_t local_work_size = 64;
    size_t global_work_size = ((M + local_work_size - 1) / local_work_size) * local_work_size;

    // Dispatch GPU execution
    fn_clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);

    // Read back output vector
    fn_clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, out_bytes, out, 0, nullptr, nullptr);
}

void GPUEngine::shutdown() {
    if (!initialized_) return;

    for (auto& pair : weight_buffer_cache_) {
        if (pair.second && fn_clReleaseMemObject) {
            fn_clReleaseMemObject(static_cast<cl_mem>(pair.second));
        }
    }
    weight_buffer_cache_.clear();

    if (d_in_buf_ && fn_clReleaseMemObject) {
        fn_clReleaseMemObject(static_cast<cl_mem>(d_in_buf_));
        d_in_buf_ = nullptr;
    }
    if (d_out_buf_ && fn_clReleaseMemObject) {
        fn_clReleaseMemObject(static_cast<cl_mem>(d_out_buf_));
        d_out_buf_ = nullptr;
    }
    if (cl_kernel_gemv_ && fn_clReleaseKernel) {
        fn_clReleaseKernel(static_cast<cl_kernel>(cl_kernel_gemv_));
        cl_kernel_gemv_ = nullptr;
    }
    if (cl_program_ && fn_clReleaseProgram) {
        fn_clReleaseProgram(static_cast<cl_program>(cl_program_));
        cl_program_ = nullptr;
    }
    if (cl_queue_ && fn_clReleaseCommandQueue) {
        fn_clReleaseCommandQueue(static_cast<cl_command_queue>(cl_queue_));
        cl_queue_ = nullptr;
    }
    if (cl_context_ && fn_clReleaseContext) {
        fn_clReleaseContext(static_cast<cl_context>(cl_context_));
        cl_context_ = nullptr;
    }

    if (lib_handle_) {
        dlclose(lib_handle_);
        lib_handle_ = nullptr;
    }

    initialized_ = false;
    LOG_GPU("GPUEngine successfully shut down.");
}

} // namespace backend
} // namespace embeddra

