#pragma once

#include <atomic>
#include <cstddef>
#include <cstring>
#include <vector>

/// Lock-free single-producer/single-consumer ring buffer.
/// Audio thread writes, worker thread reads. No allocations after construction.
template <typename T>
class SPSCRingBuffer {
public:
    explicit SPSCRingBuffer(size_t capacity)
        : _buffer(capacity), _capacity(capacity) {}

    /// Returns the number of items available for reading.
    size_t availableRead() const noexcept {
        const auto w = _writePos.load(std::memory_order_acquire);
        const auto r = _readPos.load(std::memory_order_relaxed);
        return w - r;
    }

    /// Returns the number of items that can be written.
    size_t availableWrite() const noexcept {
        return _capacity - availableRead();
    }

    /// Write `count` items from `src`. Returns number actually written.
    size_t write(const T* src, size_t count) noexcept {
        const auto avail = availableWrite();
        if (count > avail) count = avail;
        if (count == 0) return 0;

        const auto w = _writePos.load(std::memory_order_relaxed);
        const auto offset = w % _capacity;
        const auto firstChunk = std::min(count, _capacity - offset);

        std::memcpy(_buffer.data() + offset, src, firstChunk * sizeof(T));
        if (firstChunk < count)
            std::memcpy(_buffer.data(), src + firstChunk, (count - firstChunk) * sizeof(T));

        _writePos.store(w + count, std::memory_order_release);
        return count;
    }

    /// Read `count` items into `dst`. Returns number actually read.
    size_t read(T* dst, size_t count) noexcept {
        const auto avail = availableRead();
        if (count > avail) count = avail;
        if (count == 0) return 0;

        const auto r = _readPos.load(std::memory_order_relaxed);
        const auto offset = r % _capacity;
        const auto firstChunk = std::min(count, _capacity - offset);

        std::memcpy(dst, _buffer.data() + offset, firstChunk * sizeof(T));
        if (firstChunk < count)
            std::memcpy(dst + firstChunk, _buffer.data(), (count - firstChunk) * sizeof(T));

        _readPos.store(r + count, std::memory_order_release);
        return count;
    }

    void reset() noexcept {
        _writePos.store(0, std::memory_order_relaxed);
        _readPos.store(0, std::memory_order_relaxed);
    }

private:
    std::vector<T> _buffer;
    size_t _capacity;
    alignas(64) std::atomic<size_t> _writePos { 0 };
    alignas(64) std::atomic<size_t> _readPos  { 0 };
};
