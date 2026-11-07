#pragma once

#include <string>
#include <cstdint>

namespace embeddra {
namespace hardware {

struct DeviceInfo {
    int cpu_cores;
    uint64_t total_ram_bytes;
    uint64_t available_ram_bytes;
    bool has_fp16_arithmetic;
    bool has_neon_simd;
    bool has_npu_support;
    std::string device_model;
    std::string os_info;
};

class HardwareDetector {
public:
    static DeviceInfo detect();
    static void printDeviceInfo(const DeviceInfo& info);
};

} // namespace hardware
} // namespace embeddra

