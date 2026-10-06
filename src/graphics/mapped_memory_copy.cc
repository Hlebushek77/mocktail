#include "mocktail/graphics/mapped_memory_copy.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define MOCKTAIL_MAPPED_COPY_X86 1
#else
#define MOCKTAIL_MAPPED_COPY_X86 0
#endif

namespace mocktail::graphics {

namespace {

constexpr std::size_t kBounceBytes = 4096;

// Streams `bytes` (a multiple of 64, at most kBounceBytes) from `source`,
// aligned to the kernel's load size, into the 64-byte aligned `bounce`.
using StreamFunction = void (*)(std::uint8_t* bounce,
                                const std::uint8_t* source,
                                std::size_t bytes);

#if MOCKTAIL_MAPPED_COPY_X86
__attribute__((target("avx2"))) void StreamAvx2(std::uint8_t* bounce,
                                                const std::uint8_t* source,
                                                std::size_t bytes) {
  const auto* in = reinterpret_cast<const __m256i*>(source);
  auto* out = reinterpret_cast<__m256i*>(bounce);
  for (std::size_t index = 0; index < bytes / 32; index += 2) {
    const __m256i first =
        _mm256_stream_load_si256(const_cast<__m256i*>(in + index));
    const __m256i second =
        _mm256_stream_load_si256(const_cast<__m256i*>(in + index + 1));
    _mm256_store_si256(out + index, first);
    _mm256_store_si256(out + index + 1, second);
  }
}

__attribute__((target("sse4.1"))) void StreamSse41(std::uint8_t* bounce,
                                                   const std::uint8_t* source,
                                                   std::size_t bytes) {
  auto* in = reinterpret_cast<__m128i*>(const_cast<std::uint8_t*>(source));
  auto* out = reinterpret_cast<__m128i*>(bounce);
  for (std::size_t index = 0; index < bytes / 16; index += 4) {
    const __m128i a = _mm_stream_load_si128(in + index);
    const __m128i b = _mm_stream_load_si128(in + index + 1);
    const __m128i c = _mm_stream_load_si128(in + index + 2);
    const __m128i d = _mm_stream_load_si128(in + index + 3);
    _mm_store_si128(out + index, a);
    _mm_store_si128(out + index + 1, b);
    _mm_store_si128(out + index + 2, c);
    _mm_store_si128(out + index + 3, d);
  }
}
#endif

bool Supported(MappedMemoryCopyKernel kernel) {
  switch (kernel) {
    case MappedMemoryCopyKernel::kMemcpy:
      return true;
#if MOCKTAIL_MAPPED_COPY_X86
    case MappedMemoryCopyKernel::kSse41:
      __builtin_cpu_init();
      return __builtin_cpu_supports("sse4.1") != 0;
    case MappedMemoryCopyKernel::kAvx2:
      __builtin_cpu_init();
      return __builtin_cpu_supports("avx2") != 0;
#else
    case MappedMemoryCopyKernel::kSse41:
    case MappedMemoryCopyKernel::kAvx2:
      return false;
#endif
  }
  return false;
}

MappedMemoryCopyKernel Automatic() {
  static const MappedMemoryCopyKernel kernel = [] {
    if (Supported(MappedMemoryCopyKernel::kAvx2)) {
      return MappedMemoryCopyKernel::kAvx2;
    }
    if (Supported(MappedMemoryCopyKernel::kSse41)) {
      return MappedMemoryCopyKernel::kSse41;
    }
    return MappedMemoryCopyKernel::kMemcpy;
  }();
  return kernel;
}

// -1: automatic.
std::atomic<int> g_forced{-1};

MappedMemoryCopyKernel Active() {
  const int forced = g_forced.load(std::memory_order_relaxed);
  return forced < 0 ? Automatic() : static_cast<MappedMemoryCopyKernel>(forced);
}

struct Kernel {
  StreamFunction stream = nullptr;  // null: plain memcpy
  std::size_t alignment = 1;
};

Kernel KernelFor(MappedMemoryCopyKernel kernel) {
  switch (kernel) {
#if MOCKTAIL_MAPPED_COPY_X86
    case MappedMemoryCopyKernel::kAvx2:
      return {StreamAvx2, 32};
    case MappedMemoryCopyKernel::kSse41:
      return {StreamSse41, 16};
#else
    case MappedMemoryCopyKernel::kAvx2:
    case MappedMemoryCopyKernel::kSse41:
#endif
    case MappedMemoryCopyKernel::kMemcpy:
      break;
  }
  return {};
}

// Hands `sink(offset, data, length)` the bytes of [source, source + size) in
// order: the unaligned head and the tail straight from `source`, everything
// between through the bounce buffer.
template <typename Sink>
void Stream(const Kernel& kernel, const std::uint8_t* source, std::size_t size,
            Sink&& sink) {
  std::size_t done =
      (kernel.alignment -
       (reinterpret_cast<std::uintptr_t>(source) & (kernel.alignment - 1))) &
      (kernel.alignment - 1);
  done = std::min(done, size);
  if (done != 0) {
    sink(0, source, done);
  }
  alignas(64) std::uint8_t bounce[kBounceBytes];
  while (size - done >= 64) {
    const std::size_t chunk =
        std::min(kBounceBytes, (size - done) & ~static_cast<std::size_t>(63));
    kernel.stream(bounce, source + done, chunk);
    sink(done, bounce, chunk);
    done += chunk;
  }
  if (done < size) {
    sink(done, source + done, size - done);
  }
}

void CopyWith(const Kernel& kernel, std::uint8_t* destination,
              const std::uint8_t* source, std::size_t size) {
  if (kernel.stream == nullptr) {
    std::memcpy(destination, source, size);
    return;
  }
  Stream(kernel, source, size,
         [destination](std::size_t offset, const std::uint8_t* data,
                       std::size_t length) {
           std::memcpy(destination + offset, data, length);
         });
}

}  // namespace

void CopyFromMappedMemory(void* destination, const void* source,
                          std::size_t size) {
  if (size == 0) {
    return;
  }
  CopyWith(KernelFor(Active()), static_cast<std::uint8_t*>(destination),
           static_cast<const std::uint8_t*>(source), size);
}

void CopyStridedFromMappedMemory(void* destination, const void* source,
                                 std::size_t row_bytes, std::size_t rows,
                                 std::size_t source_stride) {
  if (row_bytes == 0 || rows == 0) {
    return;
  }
  auto* out = static_cast<std::uint8_t*>(destination);
  const auto* in = static_cast<const std::uint8_t*>(source);
  const Kernel kernel = KernelFor(Active());
  if (source_stride == row_bytes || rows == 1) {
    CopyWith(kernel, out, in, row_bytes * rows);
    return;
  }
  const std::size_t span = (rows - 1) * source_stride + row_bytes;
  if (kernel.stream == nullptr || source_stride < row_bytes ||
      span > 2 * rows * row_bytes) {
    for (std::size_t row = 0; row < rows; ++row) {
      CopyWith(kernel, out + row * row_bytes, in + row * source_stride,
               row_bytes);
    }
    return;
  }
  // Pieces arrive in order: `row` is the first row not yet complete.
  std::size_t row = 0;
  Stream(kernel, in, span,
         [&](std::size_t offset, const std::uint8_t* data, std::size_t length) {
           const std::size_t end = offset + length;
           while (row < rows) {
             const std::size_t row_start = row * source_stride;
             const std::size_t row_end = row_start + row_bytes;
             if (row_start >= end) {
               break;
             }
             const std::size_t low = std::max(offset, row_start);
             const std::size_t high = std::min(end, row_end);
             if (low < high) {
               std::memcpy(out + row * row_bytes + (low - row_start),
                           data + (low - offset), high - low);
             }
             if (row_end > end) {
               break;
             }
             ++row;
           }
         });
}

MappedMemoryCopyKernel ActiveMappedMemoryCopyKernel() { return Active(); }

bool MappedMemoryCopyKernelSupported(MappedMemoryCopyKernel kernel) {
  return Supported(kernel);
}

const char* MappedMemoryCopyKernelName(MappedMemoryCopyKernel kernel) {
  switch (kernel) {
    case MappedMemoryCopyKernel::kMemcpy:
      return "memcpy";
    case MappedMemoryCopyKernel::kSse41:
      return "sse4.1";
    case MappedMemoryCopyKernel::kAvx2:
      return "avx2";
  }
  return "unknown";
}

bool ForceMappedMemoryCopyKernelForTesting(MappedMemoryCopyKernel kernel) {
  if (!Supported(kernel)) {
    return false;
  }
  g_forced.store(static_cast<int>(kernel), std::memory_order_relaxed);
  return true;
}

void ResetMappedMemoryCopyKernelForTesting() {
  g_forced.store(-1, std::memory_order_relaxed);
}

}  // namespace mocktail::graphics
