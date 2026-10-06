#ifndef MOCKTAIL_GRAPHICS_MAPPED_MEMORY_COPY_H_
#define MOCKTAIL_GRAPHICS_MAPPED_MEMORY_COPY_H_

#include <cstddef>

namespace mocktail::graphics {

// Reading Roblox's upload buffers. They live in HOST_VISIBLE|HOST_COHERENT
// memory that NVIDIA maps write-combined (memory type 3 on the RTX 3060):
// ordinary loads from it are uncached, so memcpy reads only 150-250 MiB/s.
// Streaming loads (movntdqa) fetch whole lines through the write-combining
// buffers instead: 1.9-2.7 GiB/s with SSE4.1 and 3.1-3.4 GiB/s with AVX2 on
// one thread of a Ryzen 7 6800H, 7-11 GiB/s on eight. On cached memory they
// behave like ordinary loads, so the copy needs no knowledge of the memory
// type.
//
// The kernel is picked once at run time, because development builds are not
// compiled for x86-64-v3: AVX2 when the CPU (and OS) has it, else SSE4.1,
// else memcpy. Each streams through a 64-byte aligned 4 KiB bounce buffer on
// the stack and copies unaligned heads and tails with memcpy.
enum class MappedMemoryCopyKernel { kMemcpy, kSse41, kAvx2 };

// Copies `size` bytes out of mapped memory into cached memory.
void CopyFromMappedMemory(void* destination, const void* source,
                          std::size_t size);

// Copies `rows` rows of `row_bytes` that lie `source_stride` bytes apart in
// mapped memory, packed one after another into `destination`. When the rows
// span at most twice the bytes they hold, the whole span streams in and the
// rows are picked out of the bounce buffer (sequential reads suit
// write-combined memory); sparser rows stream one by one.
void CopyStridedFromMappedMemory(void* destination, const void* source,
                                 std::size_t row_bytes, std::size_t rows,
                                 std::size_t source_stride);

MappedMemoryCopyKernel ActiveMappedMemoryCopyKernel();
bool MappedMemoryCopyKernelSupported(MappedMemoryCopyKernel kernel);
const char* MappedMemoryCopyKernelName(MappedMemoryCopyKernel kernel);

// Tests and benchmarks: use `kernel` from now on (false, and no change, when
// this CPU lacks it); Reset goes back to the automatic choice.
bool ForceMappedMemoryCopyKernelForTesting(MappedMemoryCopyKernel kernel);
void ResetMappedMemoryCopyKernelForTesting();

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_MAPPED_MEMORY_COPY_H_
