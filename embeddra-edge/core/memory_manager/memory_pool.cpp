#include "memory_pool.h"
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace embeddra {
namespace memory {

MemoryPool::MemoryPool(size_t pool_size_bytes)
    : pool_size_(pool_size_bytes), current_offset_(0) {
#if defined(_MSC_VER)
    buffer_ = static_cast<uint8_t*>(_aligned_malloc(pool_size_, 64));
#else
    if (posix_memalign(reinterpret_cast<void**>(&buffer_), 64, pool_size_) != 0) {
        buffer_ = nullptr;
    }
#endif
    if (!buffer_) {
        throw std::bad_alloc();
    }
}

MemoryPool::~MemoryPool() {
    if (buffer_) {
#if defined(_MSC_VER)
        _aligned_free(buffer_);
#else
        free(buffer_);
#endif
        buffer_ = nullptr;
    }
}

void* MemoryPool::allocate(size_t bytes, size_t alignment) {
    size_t aligned_offset = (current_offset_ + alignment - 1) & ~(alignment - 1);
    if (aligned_offset + bytes > pool_size_) {
        return nullptr; // Out of arena memory
    }
    void* ptr = buffer_ + aligned_offset;
    current_offset_ = aligned_offset + bytes;
    return ptr;
}

void MemoryPool::reset() {
    current_offset_ = 0;
}

} // namespace memory
} // namespace embeddra

