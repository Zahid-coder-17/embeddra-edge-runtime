#include "hardware_detector.h"
#include <iostream>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

namespace embeddra {
namespace hardware {

DeviceInfo HardwareDetector::detect() {
    DeviceInfo info;
    info.cpu_cores = std::thread::hardware_concurrency();
    info.has_fp16_arithmetic = true;
    info.has_neon_simd = true;
    info.has_npu_support = false;

#if defined(_WIN32)
    info.os_info = "Windows";
    info.device_model = "Desktop/Laptop Host";
    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        info.total_ram_bytes = memStatus.ullTotalPhys;
        info.available_ram_bytes = memStatus.ullAvailPhys;
    } else {
        info.total_ram_bytes = 0;
        info.available_ram_bytes = 0;
    }
#elif defined(__ANDROID__)
    info.os_info = "Android";
    info.device_model = "Generic Android Smartphone";
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        info.total_ram_bytes = si.totalram * si.mem_unit;
        info.available_ram_bytes = si.freeram * si.mem_unit;
    } else {
        info.total_ram_bytes = 0;
        info.available_ram_bytes = 0;
    }
#elif defined(__linux__)
    info.os_info = "Linux";
    info.device_model = "Linux Workstation";
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        info.total_ram_bytes = si.totalram * si.mem_unit;
        info.available_ram_bytes = si.freeram * si.mem_unit;
    } else {
        info.total_ram_bytes = 0;
        info.available_ram_bytes = 0;
    }
#else
    info.os_info = "Unknown";
    info.device_model = "Generic Target";
    info.total_ram_bytes = 0;
    info.available_ram_bytes = 0;
#endif

    return info;
}

void HardwareDetector::printDeviceInfo(const DeviceInfo& info) {
    std::cout << "--- Embeddra Hardware Environment ---\n"
              << "OS: " << info.os_info << " (" << info.device_model << ")\n"
              << "CPU Cores: " << info.cpu_cores << "\n"
              << "Total RAM: " << (info.total_ram_bytes / (1024 * 1024)) << " MB\n"
              << "Available RAM: " << (info.available_ram_bytes / (1024 * 1024)) << " MB\n"
              << "ARM NEON SIMD: " << (info.has_neon_simd ? "YES" : "NO") << "\n"
              << "FP16 Acceleration: " << (info.has_fp16_arithmetic ? "YES" : "NO") << "\n"
              << "-------------------------------------\n";
}

} // namespace hardware
} // namespace embeddra

