// etc2_cpu_perf: the ETC2 CPU route's benchmark (DESIGN §9.3, WP1). A
// manual tool, not a test: it needs a Vulkan device and an idle machine, and
// prints timings. It refuses to run while a live Mocktail or Sober session
// runs (CR-25).
//
//   etc2_cpu_perf [--reps N] [--workload small|big|mixed]...
//                 [--arm cached|coherent|bc|bc-coherent]... [section...]
//
// Sections (all by default):
//   kernels    the decoder and the resampler against the plus.1 references
//              (single thread, bit-exact)
//   bc         ETC2->BC mode's CPU cost (CR-13): decode against decode and
//              BC encode, with and without the endpoint refinement, per
//              1024x1024 texels on one thread
//   pool       waking the persistent pool against creating and joining
//              threads per call, as plus.1 did
//   copy       copy-in from Roblox-like write-combined memory (type 3 on
//              NVIDIA): memcpy and the streaming kernels, 1 to 8 threads
//   emulation  the production VulkanEtc2Emulation on the real device: the
//              small, big and mixed bursts of plus2-design/pipeline_bench.cc
//              with cached and coherent staging; staging contents compared
//              byte for byte; transient RAM; scratch and arena after the
//              idle delays. ETC2->BC mode (MOCKTAIL_ETC2_TO_BC=1) only when
//              asked for: --arm bc (cached staging) and --arm bc-coherent,
//              whose BC staging is compared between them
//   gpucopy    the GPU reading staging: vkCmdCopyBufferToImage of 64 MiB
//              from cached (type 4) and coherent (type 3) system memory
//
// The emulation drives the device the way the adapter does, except that the
// copies it records are only captured (the host vkCmdCopyBufferToImage is
// replaced) and host images are handles only: the CPU route's cost is all
// inside the hooks, and the captured regions say exactly which staging bytes
// the GPU would read.

#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mocktail/graphics/bc_encoder.h"
#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/mapped_memory_copy.h"
#include "mocktail/graphics/texture_override.h"
#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "mocktail/graphics/worker_pool.h"
#include "support/bc_reference_transcode.h"
#include "support/etc2_reference_decoder.h"
#include "support/fake_vulkan_device.h"
#include "support/resample_reference.h"
#include "support/vulkan_test_device.h"

namespace mocktail::graphics::testing {
namespace {

using Clock = std::chrono::steady_clock;

double Ms(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

std::vector<std::uint8_t> RandomBytes(std::size_t size, std::uint64_t seed) {
  std::vector<std::uint8_t> bytes(size);
  std::mt19937_64 rng(seed);
  for (std::size_t index = 0; index < size; index += 8) {
    const std::uint64_t value = rng();
    std::memcpy(bytes.data() + index, &value,
                std::min<std::size_t>(8, size - index));
  }
  return bytes;
}

// Resident memory (kB) from /proc/self/status. Writing 5 to clear_refs
// resets the peak, VmHWM (Linux 4.0+).
std::uint64_t StatusKb(const char* key) {
  std::FILE* file = std::fopen("/proc/self/status", "r");
  if (file == nullptr) {
    return 0;
  }
  char line[256];
  std::uint64_t value = 0;
  const std::size_t length = std::strlen(key);
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    if (std::strncmp(line, key, length) == 0) {
      value = std::strtoull(line + length, nullptr, 10);
      break;
    }
  }
  std::fclose(file);
  return value;
}

void ResetPeakRss() {
  std::FILE* file = std::fopen("/proc/self/clear_refs", "w");
  if (file != nullptr) {
    std::fputs("5", file);
    std::fclose(file);
  }
}

// ---------------------------------------------------------------- kernels

void RunKernels() {
  std::printf("== kernels (one thread, against the plus.1 code)\n");
  struct Case {
    EtcFormat format;
    const char* name;
    std::uint32_t size;
  };
  const Case cases[] = {
      {EtcFormat::kEtc2Rgb8, "RGB8", 2048},
      {EtcFormat::kEtc2Rgb8, "RGB8", 61},
      {EtcFormat::kEtc2Rgb8A1, "RGB8A1", 2048},
      {EtcFormat::kEtc2Rgba8, "RGBA8", 2048},
      {EtcFormat::kEtc2Rgba8, "RGBA8", 61},
      {EtcFormat::kEacR11, "EAC_R11", 2048},
      {EtcFormat::kEacRg11Signed, "EAC_RG11S", 2048},
  };
  for (const Case& test : cases) {
    const std::size_t blocks =
        static_cast<std::size_t>((test.size + 3) / 4) * ((test.size + 3) / 4);
    const std::vector<std::uint8_t> source =
        RandomBytes(blocks * EtcBlockBytes(test.format), test.size);
    const std::size_t out_bytes = static_cast<std::size_t>(test.size) *
                                  test.size *
                                  EtcDecodedTexelBytes(test.format);
    std::vector<std::uint8_t> before(out_bytes);
    std::vector<std::uint8_t> after(out_bytes);
    const int reps = test.size > 256 ? 7 : 2000;
    std::vector<double> old_ms;
    std::vector<double> new_ms;
    for (int rep = 0; rep < reps; ++rep) {
      const auto t0 = Clock::now();
      reference::DecodeEtcImage(test.format, source.data(), source.size(),
                                test.size, test.size, before.data(),
                                before.size());
      const auto t1 = Clock::now();
      DecodeEtcImage(test.format, source.data(), source.size(), test.size,
                     test.size, after.data(), after.size());
      const auto t2 = Clock::now();
      old_ms.push_back(Ms(t1 - t0));
      new_ms.push_back(Ms(t2 - t1));
    }
    std::printf("  decode   %-9s %4ux%-4u plus.1 %8.3f ms  now %8.3f ms  "
                "x%.2f  %s\n",
                test.name, test.size, test.size, Median(old_ms),
                Median(new_ms), Median(old_ms) / Median(new_ms),
                before == after ? "identical" : "DIFFERENT");
  }
  struct Resample {
    std::uint32_t source_width;
    std::uint32_t source_height;
    std::uint32_t width;
    std::uint32_t height;
  };
  const Resample resamples[] = {{64, 64, 256, 256},
                                {37, 37, 148, 148},
                                {60, 33, 240, 132},
                                {64, 64, 32, 32},
                                {60, 33, 30, 16}};
  for (const Resample& test : resamples) {
    const std::vector<std::uint8_t> source = RandomBytes(
        static_cast<std::size_t>(test.source_width) * test.source_height * 4,
        test.width);
    std::vector<std::uint8_t> before(static_cast<std::size_t>(test.width) *
                                     test.height * 4);
    std::vector<std::uint8_t> after(before.size());
    std::vector<double> old_ms;
    std::vector<double> new_ms;
    for (int rep = 0; rep < 500; ++rep) {
      const auto t0 = Clock::now();
      reference::ResampleRgba(source.data(), test.source_width,
                              test.source_height, test.width, test.height,
                              before.data());
      const auto t1 = Clock::now();
      ResampleRgba(source.data(), test.source_width, test.source_height,
                   test.width, test.height, after.data());
      const auto t2 = Clock::now();
      old_ms.push_back(Ms(t1 - t0));
      new_ms.push_back(Ms(t2 - t1));
    }
    std::printf("  resample %3ux%-3u -> %3ux%-3u plus.1 %7.1f us  now %6.1f "
                "us  x%.1f  %s\n",
                test.source_width, test.source_height, test.width,
                test.height, Median(old_ms) * 1000.0, Median(new_ms) * 1000.0,
                Median(old_ms) / Median(new_ms),
                before == after ? "identical" : "DIFFERENT");
  }
}

// ---------------------------------------------------------------- bc

// What ETC2->BC mode costs the CPU route per texel (DESIGN CR-13): the
// decode the default mode does, against BC mode's decode and encode with the
// endpoint refinement (what the CPU route does, as the GPU route's shader)
// and without it (the refine=0 question CR-13 leaves to the user). Random
// blocks, which hit every ETC2 mode; the refined output is checked against
// the references on its first 4096 blocks.
void RunBc() {
  std::printf("== bc (one thread, per 1024x1024 texels; CR-13)\n");
  struct Case {
    EtcFormat format;
    const char* name;
  };
  const Case cases[] = {
      {EtcFormat::kEtc2Rgb8, "RGB8->BC1"},
      {EtcFormat::kEtc2Rgb8A1, "RGB8A1->BC1"},
      {EtcFormat::kEtc2Rgba8, "RGBA8->BC3"},
      {EtcFormat::kEacR11, "R11->BC4"},
      {EtcFormat::kEacRg11, "RG11->BC5"},
      {EtcFormat::kEacRg11Signed, "RG11S->BC5S"},
  };
  constexpr std::uint32_t kSize = 1024;
  constexpr std::size_t kChecked = 4096;
  for (const Case& test : cases) {
    const std::size_t block_bytes = EtcBlockBytes(test.format);
    const std::vector<std::uint8_t> source = RandomBytes(
        static_cast<std::size_t>(kSize / 4) * (kSize / 4) * block_bytes,
        0xBC + static_cast<int>(test.format));
    std::vector<std::uint8_t> decoded(static_cast<std::size_t>(kSize) * kSize *
                                      EtcDecodedTexelBytes(test.format));
    std::vector<std::uint8_t> refined(source.size());
    std::vector<std::uint8_t> plain(source.size());
    std::vector<double> decode_ms;
    std::vector<double> refined_ms;
    std::vector<double> plain_ms;
    for (int rep = 0; rep < 7; ++rep) {
      const auto t0 = Clock::now();
      DecodeEtcImage(test.format, source.data(), source.size(), kSize, kSize,
                     decoded.data(), decoded.size());
      const auto t1 = Clock::now();
      TranscodeEtcToBc(test.format, source.data(), source.size(),
                       refined.data(), refined.size(), true);
      const auto t2 = Clock::now();
      TranscodeEtcToBc(test.format, source.data(), source.size(), plain.data(),
                       plain.size(), false);
      const auto t3 = Clock::now();
      decode_ms.push_back(Ms(t1 - t0));
      refined_ms.push_back(Ms(t2 - t1));
      plain_ms.push_back(Ms(t3 - t2));
    }
    const std::vector<std::uint8_t> expected = reference::BcFromEtc(
        test.format, source.data(), kChecked * block_bytes, true);
    const bool identical = std::equal(expected.begin(), expected.end(),
                                      refined.begin());
    std::printf("  %-11s decode %6.2f ms   BC refined %6.2f ms (x%.1f)   BC "
                "plain %6.2f ms (x%.1f)   %s\n",
                test.name, Median(decode_ms), Median(refined_ms),
                Median(refined_ms) / Median(decode_ms), Median(plain_ms),
                Median(plain_ms) / Median(decode_ms),
                identical ? "refined = reference" : "REFINED DIFFERS");
  }
}

// ---------------------------------------------------------------- pool

void* Nothing(void*) { return nullptr; }

void RunPool(unsigned workers) {
  std::printf("== pool (%u threads per call)\n", workers);
  std::vector<double> spawn_ms;
  for (int rep = 0; rep < 200; ++rep) {
    const auto t0 = Clock::now();
    std::vector<pthread_t> threads(workers - 1);
    for (pthread_t& thread : threads) {
      pthread_create(&thread, nullptr, Nothing, nullptr);
    }
    for (pthread_t thread : threads) {
      pthread_join(thread, nullptr);
    }
    spawn_ms.push_back(Ms(Clock::now() - t0));
  }
  WorkerPool& pool = WorkerPool::Shared(workers);
  pool.ParallelFor(workers, [](std::size_t) {});  // starts the threads
  std::vector<double> wake_ms;
  for (int rep = 0; rep < 200; ++rep) {
    const auto t0 = Clock::now();
    pool.ParallelFor(workers, [](std::size_t) {});
    wake_ms.push_back(Ms(Clock::now() - t0));
  }
  std::printf("  plus.1 pthread_create+join x%u: %.1f us   pool wake: %.1f us\n",
              workers - 1, Median(spawn_ms) * 1000.0,
              Median(wake_ms) * 1000.0);
}

// ---------------------------------------------------------------- device

struct DeviceMemory {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  std::uint8_t* mapped = nullptr;
};

std::uint32_t TypeWith(const HeadlessVulkanDevice& device,
                       VkMemoryPropertyFlags required,
                       VkMemoryPropertyFlags avoided) {
  const VkPhysicalDeviceMemoryProperties& memory = device.memory();
  for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
    const VkMemoryPropertyFlags flags = memory.memoryTypes[index].propertyFlags;
    if ((flags & required) == required && (flags & avoided) == 0) {
      return index;
    }
  }
  return UINT32_MAX;
}

constexpr VkMemoryPropertyFlags kHostCoherent =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

// Roblox's upload memory on NVIDIA: coherent, uncached (write-combined).
std::uint32_t WriteCombinedType(const HeadlessVulkanDevice& device) {
  return TypeWith(device, kHostCoherent,
                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

bool AllocateMapped(const HeadlessVulkanDevice& device, VkDeviceSize size,
                    std::uint32_t type, DeviceMemory* out) {
  const auto& vk = device.vk();
  VkMemoryAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate.allocationSize = size;
  allocate.memoryTypeIndex = type;
  void* data = nullptr;
  if (vk.vkAllocateMemory(device.device(), &allocate, nullptr, &out->memory) !=
          VK_SUCCESS ||
      vk.vkMapMemory(device.device(), out->memory, 0, VK_WHOLE_SIZE, 0,
                     &data) != VK_SUCCESS) {
    return false;
  }
  out->mapped = static_cast<std::uint8_t*>(data);
  return true;
}

VkBuffer CreateBuffer(const HeadlessVulkanDevice& device, VkDeviceSize size,
                      VkBufferUsageFlags usage) {
  VkBufferCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.size = size;
  info.usage = usage;
  VkBuffer buffer = VK_NULL_HANDLE;
  device.vk().vkCreateBuffer(device.device(), &info, nullptr, &buffer);
  return buffer;
}

// A mapped buffer of `size` in memory `type`.
bool MakeBuffer(const HeadlessVulkanDevice& device, VkDeviceSize size,
                std::uint32_t type, DeviceMemory* out) {
  out->buffer = CreateBuffer(device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  if (out->buffer == VK_NULL_HANDLE) {
    return false;
  }
  VkMemoryRequirements requirements{};
  device.vk().vkGetBufferMemoryRequirements(device.device(), out->buffer,
                                            &requirements);
  return AllocateMapped(device, requirements.size, type, out) &&
         device.vk().vkBindBufferMemory(device.device(), out->buffer,
                                        out->memory, 0) == VK_SUCCESS;
}

void FreeBuffer(const HeadlessVulkanDevice& device, DeviceMemory* memory) {
  if (memory->buffer != VK_NULL_HANDLE) {
    device.vk().vkDestroyBuffer(device.device(), memory->buffer, nullptr);
  }
  device.vk().vkFreeMemory(device.device(), memory->memory, nullptr);
  *memory = DeviceMemory{};
}

// ---------------------------------------------------------------- copy

void RunCopy(const HeadlessVulkanDevice& device) {
  const std::uint32_t type = WriteCombinedType(device);
  const std::size_t size = 49U << 20;  // the mixed burst's compressed bytes
  DeviceMemory source;
  if (type == UINT32_MAX || !MakeBuffer(device, size, type, &source)) {
    std::printf("== copy: no uncached host-visible memory, skipped\n");
    return;
  }
  std::printf("== copy-in of 49 MiB out of memory type %u (write-combined), "
              "256 KiB pieces on the pool\n",
              type);
  const std::vector<std::uint8_t> pattern = RandomBytes(size, 7);
  std::memcpy(source.mapped, pattern.data(), size);
  std::vector<std::uint8_t> destination(size);
  constexpr std::size_t kPiece = 256U << 10;
  const std::size_t pieces = (size + kPiece - 1) / kPiece;
  for (const MappedMemoryCopyKernel kernel :
       {MappedMemoryCopyKernel::kMemcpy, MappedMemoryCopyKernel::kSse41,
        MappedMemoryCopyKernel::kAvx2}) {
    if (!ForceMappedMemoryCopyKernelForTesting(kernel)) {
      continue;
    }
    std::printf("  %-7s", MappedMemoryCopyKernelName(kernel));
    for (const unsigned threads : {1U, 2U, 4U, 8U}) {
      WorkerPool pool(threads);
      std::vector<double> ms;
      const int reps =
          kernel == MappedMemoryCopyKernel::kMemcpy && threads < 4 ? 1 : 5;
      bool same = true;
      for (int rep = 0; rep < reps; ++rep) {
        std::fill(destination.begin(), destination.end(), 0);
        const auto t0 = Clock::now();
        pool.ParallelFor(pieces, [&](std::size_t piece) {
          const std::size_t offset = piece * kPiece;
          CopyFromMappedMemory(destination.data() + offset,
                               source.mapped + offset,
                               std::min(kPiece, size - offset));
        });
        ms.push_back(Ms(Clock::now() - t0));
        same = same &&
               std::memcmp(destination.data(), pattern.data(), size) == 0;
      }
      const double median = Median(ms);
      std::printf("  %ut %6.1f ms %5.0f MiB/s%s", threads, median,
                  49.0 / (median / 1000.0), same ? "" : " DIFFERENT");
    }
    std::printf("\n");
  }
  ResetMappedMemoryCopyKernelForTesting();
  FreeBuffer(device, &source);
}

// ---------------------------------------------------------------- gpucopy

void RunGpuCopy(const HeadlessVulkanDevice& device) {
  const auto& vk = device.vk();
  const VkDevice handle = device.device();
  const auto create_image = reinterpret_cast<PFN_vkCreateImage>(
      vk.vkGetDeviceProcAddr(handle, "vkCreateImage"));
  constexpr std::uint32_t kSide = 4096;  // 64 MiB of RGBA8
  const VkDeviceSize bytes = static_cast<VkDeviceSize>(kSide) * kSide * 4;
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = VK_FORMAT_R8G8B8A8_UNORM;
  info.extent = {kSide, kSide, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  VkImage image = VK_NULL_HANDLE;
  if (create_image == nullptr ||
      create_image(handle, &info, nullptr, &image) != VK_SUCCESS) {
    std::printf("== gpucopy: cannot create the image, skipped\n");
    return;
  }
  VkMemoryRequirements requirements{};
  vk.vkGetImageMemoryRequirements(handle, image, &requirements);
  VkMemoryAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex = device.FindMemoryType(
      requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkDeviceMemory image_memory = VK_NULL_HANDLE;
  vk.vkAllocateMemory(handle, &allocate, nullptr, &image_memory);
  vk.vkBindImageMemory(handle, image, image_memory, 0);
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = device.queue_family();
  VkCommandPool pool = VK_NULL_HANDLE;
  vk.vkCreateCommandPool(handle, &pool_info, nullptr, &pool);
  VkCommandBufferAllocateInfo cb_info{};
  cb_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cb_info.commandPool = pool;
  cb_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cb_info.commandBufferCount = 1;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  vk.vkAllocateCommandBuffers(handle, &cb_info, &command_buffer);
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  vk.vkCreateFence(handle, &fence_info, nullptr, &fence);
  std::printf("== gpucopy: vkCmdCopyBufferToImage of 64 MiB RGBA8 from staging "
              "into a device-local image (submit to fence, median of 7)\n");
  struct Arm {
    const char* name;
    std::uint32_t type;
  };
  const Arm arms[] = {
      {"cached",
       TypeWith(device, kHostCoherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
      {"coherent", WriteCombinedType(device)},
  };
  for (const Arm& arm : arms) {
    DeviceMemory staging;
    if (arm.type == UINT32_MAX ||
        !MakeBuffer(device, bytes, arm.type, &staging)) {
      std::printf("  %-8s no such memory type\n", arm.name);
      continue;
    }
    std::memset(staging.mapped, 0x5A, bytes);
    std::vector<double> ms;
    for (int rep = 0; rep < 8; ++rep) {
      vk.vkResetCommandPool(handle, pool, 0);
      VkCommandBufferBeginInfo begin{};
      begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vk.vkBeginCommandBuffer(command_buffer, &begin);
      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vk.vkCmdPipelineBarrier(command_buffer,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                              nullptr, 1, &barrier);
      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {kSide, kSide, 1};
      vk.vkCmdCopyBufferToImage(command_buffer, staging.buffer, image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                &region);
      vk.vkEndCommandBuffer(command_buffer);
      VkSubmitInfo submit{};
      submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command_buffer;
      vk.vkResetFences(handle, 1, &fence);
      const auto t0 = Clock::now();
      vk.vkQueueSubmit(device.queue(), 1, &submit, fence);
      vk.vkWaitForFences(handle, 1, &fence, VK_TRUE, UINT64_MAX);
      if (rep > 0) {
        ms.push_back(Ms(Clock::now() - t0));
      }
    }
    const double median = Median(ms);
    std::printf("  %-8s (type %u) %6.2f ms  %5.1f GiB/s\n", arm.name,
                arm.type, median, 64.0 / 1024.0 / (median / 1000.0));
    FreeBuffer(device, &staging);
  }
  vk.vkDestroyFence(handle, fence, nullptr);
  vk.vkDestroyCommandPool(handle, pool, nullptr);
  vk.vkDestroyImage(handle, image, nullptr);
  vk.vkFreeMemory(handle, image_memory, nullptr);
}

// ---------------------------------------------------------------- emulation

// The host side of the emulated device: the real driver's, except that host
// images are handles only and copies into them are captured, not recorded.
struct Host {
  PFN_vkGetDeviceProcAddr real = nullptr;
  PFN_vkCreateBuffer create_buffer = nullptr;
  PFN_vkDestroyBuffer destroy_buffer = nullptr;
  PFN_vkBindBufferMemory bind_buffer_memory = nullptr;
  PFN_vkMapMemory map_memory = nullptr;
  PFN_vkUnmapMemory unmap_memory = nullptr;
  PFN_vkFreeMemory free_memory = nullptr;
  std::uintptr_t next_image = 0x7000000;
  std::unordered_map<VkBuffer, std::pair<VkDeviceMemory, VkDeviceSize>>
      buffers;
  std::unordered_map<VkDeviceMemory, std::uint8_t*> mapped;
  std::unordered_map<VkImage, VkFormat> images;
  std::unordered_map<VkBuffer, VkDeviceSize> created;  // by the emulation
  struct Copy {
    VkBuffer buffer;
    VkImage image;
    std::vector<VkBufferImageCopy> regions;
  };
  std::vector<Copy> copies;
};
Host g_host;

VKAPI_ATTR VkResult VKAPI_CALL HostCreateImage(VkDevice,
                                               const VkImageCreateInfo* info,
                                               const VkAllocationCallbacks*,
                                               VkImage* image) {
  *image = reinterpret_cast<VkImage>(g_host.next_image += 0x10);
  g_host.images[*image] = info->format;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL HostDestroyImage(VkDevice, VkImage image,
                                            const VkAllocationCallbacks*) {
  g_host.images.erase(image);
}

VKAPI_ATTR void VKAPI_CALL HostCopyBufferToImage(
    VkCommandBuffer, VkBuffer buffer, VkImage image, VkImageLayout,
    std::uint32_t count, const VkBufferImageCopy* regions) {
  g_host.copies.push_back({buffer, image,
                           std::vector<VkBufferImageCopy>(regions,
                                                          regions + count)});
}

VKAPI_ATTR VkResult VKAPI_CALL HostCreateBuffer(
    VkDevice device, const VkBufferCreateInfo* info,
    const VkAllocationCallbacks* allocator, VkBuffer* buffer) {
  const VkResult result = g_host.create_buffer(device, info, allocator, buffer);
  if (result == VK_SUCCESS) {
    g_host.created[*buffer] = info->size;
  }
  return result;
}

VKAPI_ATTR void VKAPI_CALL HostDestroyBuffer(
    VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator) {
  g_host.buffers.erase(buffer);
  g_host.created.erase(buffer);
  g_host.destroy_buffer(device, buffer, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL HostBindBufferMemory(VkDevice device,
                                                    VkBuffer buffer,
                                                    VkDeviceMemory memory,
                                                    VkDeviceSize offset) {
  g_host.buffers[buffer] = {memory, offset};
  return g_host.bind_buffer_memory(device, buffer, memory, offset);
}

VKAPI_ATTR VkResult VKAPI_CALL HostMapMemory(VkDevice device,
                                             VkDeviceMemory memory,
                                             VkDeviceSize offset,
                                             VkDeviceSize size,
                                             VkMemoryMapFlags flags,
                                             void** data) {
  const VkResult result =
      g_host.map_memory(device, memory, offset, size, flags, data);
  if (result == VK_SUCCESS && offset == 0) {
    g_host.mapped[memory] = static_cast<std::uint8_t*>(*data);
  }
  return result;
}

VKAPI_ATTR void VKAPI_CALL HostUnmapMemory(VkDevice device,
                                           VkDeviceMemory memory) {
  g_host.mapped.erase(memory);
  g_host.unmap_memory(device, memory);
}

VKAPI_ATTR void VKAPI_CALL HostFreeMemory(
    VkDevice device, VkDeviceMemory memory,
    const VkAllocationCallbacks* allocator) {
  g_host.mapped.erase(memory);
  g_host.free_memory(device, memory, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL HostGetDeviceProcAddr(
    VkDevice device, const char* name) {
  static const std::map<std::string, PFN_vkVoidFunction> kWrapped = {
      {"vkCreateImage", reinterpret_cast<PFN_vkVoidFunction>(HostCreateImage)},
      {"vkDestroyImage",
       reinterpret_cast<PFN_vkVoidFunction>(HostDestroyImage)},
      {"vkCmdCopyBufferToImage",
       reinterpret_cast<PFN_vkVoidFunction>(HostCopyBufferToImage)},
      {"vkCreateBuffer", reinterpret_cast<PFN_vkVoidFunction>(HostCreateBuffer)},
      {"vkDestroyBuffer",
       reinterpret_cast<PFN_vkVoidFunction>(HostDestroyBuffer)},
      {"vkBindBufferMemory",
       reinterpret_cast<PFN_vkVoidFunction>(HostBindBufferMemory)},
      {"vkMapMemory", reinterpret_cast<PFN_vkVoidFunction>(HostMapMemory)},
      {"vkUnmapMemory", reinterpret_cast<PFN_vkVoidFunction>(HostUnmapMemory)},
      {"vkFreeMemory", reinterpret_cast<PFN_vkVoidFunction>(HostFreeMemory)},
  };
  const auto found = kWrapped.find(name);
  return found != kWrapped.end() ? found->second : g_host.real(device, name);
}

// One texture's whole mip chain; the workloads of
// plus2-design/pipeline_bench.cc, built with the same seed.
struct Texture {
  VkFormat format = VK_FORMAT_UNDEFINED;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  VkImage image = VK_NULL_HANDLE;
  std::vector<VkBufferImageCopy> regions;
};

struct Workload {
  std::string name;
  bool per_mip = false;  // one copy command per mip instead of per texture
  std::vector<Texture> textures;
  std::size_t uploads = 0;
  VkDeviceSize compressed = 0;
  VkDeviceSize source_bytes = 0;
};

Workload MakeWorkload(const char* kind, bool per_mip) {
  Workload workload;
  workload.name =
      std::string(kind) + (per_mip ? " call-per-mip" : " call-per-tex");
  workload.per_mip = per_mip;
  std::mt19937 rng(42);
  VkDeviceSize cursor = 0;
  const auto add = [&](VkFormat format, std::uint32_t width,
                       std::uint32_t height) {
    Texture texture;
    texture.format = format;
    texture.width = width;
    texture.height = height;
    const VkDeviceSize block_bytes =
        format == VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK ? 8 : 16;
    for (std::uint32_t mip = 0; (std::max(width, height) >> mip) >= 1; ++mip) {
      const std::uint32_t mip_width = std::max(1U, width >> mip);
      const std::uint32_t mip_height = std::max(1U, height >> mip);
      VkBufferImageCopy region{};
      region.bufferOffset = cursor;
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
      region.imageExtent = {mip_width, mip_height, 1};
      const VkDeviceSize bytes = ((mip_width + 3) / 4) *
                                 static_cast<VkDeviceSize>((mip_height + 3) /
                                                           4) *
                                 block_bytes;
      cursor += (bytes + 15) & ~VkDeviceSize{15};
      workload.compressed += bytes;
      texture.regions.push_back(region);
      ++workload.uploads;
    }
    workload.textures.push_back(texture);
  };
  const bool small = std::strcmp(kind, "small") == 0;
  const bool mixed = std::strcmp(kind, "mixed") == 0;
  if (small || mixed) {
    for (int index = 0; index < (small ? 300 : 500); ++index) {
      const std::uint32_t width = 16 + rng() % 49;
      const std::uint32_t height = 16 + rng() % 49;
      add(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, width, height);
    }
  }
  if (!small) {
    for (int round = 0; round < (mixed ? 4 : 1); ++round) {
      for (int index = 0; index < 10; ++index) {
        add(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 1024, 1024);
      }
      for (int index = 0; index < 4; ++index) {
        add(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 1024, 1024);
      }
    }
  }
  workload.source_bytes = cursor;
  return workload;
}

struct Arm {
  const char* name;
  const char* word;  // for --arm
  Etc2StagingMemory staging;
  bool bc;  // MOCKTAIL_ETC2_TO_BC=1
};

// The bytes the host copies read from staging, hashed in copy order.
std::uint64_t HashStaging(std::uint64_t* written) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  *written = 0;
  for (const Host::Copy& copy : g_host.copies) {
    const auto binding = g_host.buffers.find(copy.buffer);
    if (binding == g_host.buffers.end()) {
      return 0;
    }
    const auto mapping = g_host.mapped.find(binding->second.first);
    if (mapping == g_host.mapped.end()) {
      return 0;
    }
    const std::uint8_t* base = mapping->second + binding->second.second;
    const auto format = g_host.images.find(copy.image);
    // Texels of RGBA8 or R16, or 4x4 blocks of BC.
    FormatBlock block{1, 1, 4};
    if (format != g_host.images.end()) {
      LookupFormatBlock(format->second, &block);
    }
    for (const VkBufferImageCopy& region : copy.regions) {
      const std::size_t bytes =
          static_cast<std::size_t>((region.imageExtent.width + block.width -
                                    1) /
                                   block.width) *
          ((region.imageExtent.height + block.height - 1) / block.height) *
          block.bytes;
      const std::uint8_t* data = base + region.bufferOffset;
      for (std::size_t index = 0; index < bytes; ++index) {
        hash = (hash ^ data[index]) * 0x100000001b3ULL;
      }
      *written += bytes;
    }
  }
  return hash;
}

// The stall line's "(prepare a, copy b, decode c, write d ms) [...]".
std::string StallBreakdown(const std::string& log) {
  const std::size_t stall = log.rfind("ETC2 decode stall ");
  if (stall == std::string::npos) {
    return "(under 20 ms: no stall line)";
  }
  const std::size_t open = log.find('(', stall);
  const std::size_t end = log.find('\n', stall);
  return open == std::string::npos ? "" : log.substr(open, end - open);
}

// What the emulation logs (fd 2) while `body` runs.
std::string CaptureStderr(const std::function<void()>& body) {
  std::fflush(stderr);
  const int saved = dup(2);
  const int capture = memfd_create("etc2_cpu_perf", 0);
  dup2(capture, 2);
  body();
  std::fflush(stderr);
  dup2(saved, 2);
  close(saved);
  std::string text;
  const off_t size = lseek(capture, 0, SEEK_END);
  if (size > 0) {
    text.resize(static_cast<std::size_t>(size));
    if (pread(capture, text.data(), text.size(), 0) < 0) {
      text.clear();
    }
  }
  close(capture);
  return text;
}

// One arm on one workload: its own emulation registered on the device, and
// its own copy of Roblox's upload memory (a memory can be mapped only once).
class Session {
 public:
  Session(const HeadlessVulkanDevice& device, const Workload& workload,
          const std::vector<std::uint8_t>& bytes, std::uint32_t source_type,
          const Arm& arm)
      : device_(device), workload_(workload), arm_(arm) {
    Etc2EmulationOptions options;
    options.gpu = false;
    options.staging_memory = arm.staging;
    options.to_bc = arm.bc;
    options.frame_stats = true;
    emulation_ = std::make_unique<VulkanEtc2Emulation>(options);
    if (arm.bc) {
      // As vkCreateDevice decides it, from the driver's own answers.
      const auto instance_proc = [&](const char* name) {
        return device.get_instance_proc_addr()(device.instance(), name);
      };
      log_ += CaptureStderr([&] {
        ok_ = emulation_->PhysicalDeviceUsesBc(
            device.physical_device(),
            reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures>(
                instance_proc("vkGetPhysicalDeviceFeatures")),
            reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
                instance_proc("vkGetPhysicalDeviceFormatProperties")));
      });
      if (!ok_) {
        why_ = "the device cannot enter ETC2->BC mode";
        return;
      }
    }
    // Room for the buffer's alignment padding.
    ok_ = AllocateMapped(device, workload.source_bytes + (64U << 10),
                         source_type, &source_);
    if (!ok_) {
      why_ = "cannot allocate the source memory";
      return;
    }
    // Written once, then unmapped, so that the emulation maps it through
    // its hook as Roblox's.
    std::memcpy(source_.mapped, bytes.data(), bytes.size());
    device.vk().vkUnmapMemory(device.device(), source_.memory);
    source_.mapped = nullptr;
    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = device.queue_family();
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    const VkDevice handle = device.device();
    log_ += CaptureStderr([&] {
      emulation_->RegisterDevice(
          handle, device.physical_device(), true, device.memory(),
          device.properties(), device.queue_families(), device_info,
          HostGetDeviceProcAddr,
          arm.bc ? device.get_instance_proc_addr() : nullptr,
          arm.bc ? device.instance() : VK_NULL_HANDLE);
      emulation_->RegisterQueue(handle, device.queue(), device.queue_family(),
                                0);
      emulation_->RegisterCommandPool(handle, kPool, device.queue_family(), 0);
      buffer_ = CreateBuffer(device, workload.source_bytes,
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
      emulation_->NoteAllocateMemory(handle, source_.memory, source_type,
                                     workload.source_bytes);
      emulation_->BindBufferMemory(handle, buffer_, source_.memory, 0);
      void* data = nullptr;
      emulation_->MapMemory(handle, source_.memory, 0, VK_WHOLE_SIZE, 0,
                            &data);
      textures_ = workload.textures;
      for (Texture& texture : textures_) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = texture.format;
        info.extent = {texture.width, texture.height, 1};
        info.mipLevels = static_cast<std::uint32_t>(texture.regions.size());
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        emulation_->CreateImage(handle, &info, nullptr, &texture.image);
      }
    });
  }

  ~Session() {
    if (!ok_) {
      if (source_.memory != VK_NULL_HANDLE) {
        FreeBuffer(device_, &source_);
      }
      return;
    }
    const VkDevice handle = device_.device();
    CaptureStderr([&] {
      for (Texture& texture : textures_) {
        emulation_->DestroyImage(handle, texture.image, nullptr);
      }
      emulation_->UnmapMemory(handle, source_.memory);
      emulation_->DestroyBuffer(handle, buffer_, nullptr);
      emulation_->ForgetCommandPool(handle, kPool);
      emulation_->DestroyDevice(handle);
      emulation_.reset();
    });
    FreeBuffer(device_, &source_);
  }

  bool ok() const { return ok_; }
  const std::string& why() const { return why_; }
  const Arm& arm() const { return arm_; }

  // One burst: record every copy, submit, release. Round 0 makes the
  // staging chunks and the scratch.
  void Round(bool first, bool last) {
    const VkDevice handle = device_.device();
    log_ += CaptureStderr([&] {
      g_host.copies.clear();
      emulation_->BeginCommandBuffer(
          kCommandBuffer, kPool, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
      ResetPeakRss();
      const std::uint64_t rss_before = StatusKb("VmRSS:");
      const auto t0 = Clock::now();
      for (const Texture& texture : textures_) {
        if (workload_.per_mip) {
          for (const VkBufferImageCopy& region : texture.regions) {
            emulation_->CmdCopyBufferToImage(
                handle, kCommandBuffer, buffer_, texture.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
          }
        } else {
          emulation_->CmdCopyBufferToImage(
              handle, kCommandBuffer, buffer_, texture.image,
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              static_cast<std::uint32_t>(texture.regions.size()),
              texture.regions.data());
        }
      }
      const auto t1 = Clock::now();
      VkSubmitInfo submit{};
      submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &kCommandBuffer;
      Etc2SubmitPlan plan;
      emulation_->PrepareSubmit(device_.queue(), 1, &submit, VK_NULL_HANDLE,
                                &plan);
      const auto t2 = Clock::now();
      emulation_->FinishSubmit(&plan, VK_SUCCESS, nullptr);
      const std::uint64_t rss_peak = StatusKb("VmHWM:");
      const std::uint64_t rss_mib =
          rss_peak > rss_before ? (rss_peak - rss_before) / 1024 : 0;
      if (last) {
        staging_hash_ = HashStaging(&written_);
      }
      const auto t3 = Clock::now();
      emulation_->ReleaseCommandBuffer(kCommandBuffer);
      const auto t4 = Clock::now();
      if (first) {
        cold_ms_ = Ms(t2 - t0) + Ms(t4 - t3);
        cold_rss_mib_ = rss_mib;
      } else {
        record_ms_.push_back(Ms(t1 - t0));
        submit_ms_.push_back(Ms(t2 - t1));
        release_ms_.push_back(Ms(t4 - t3));
        rss_mib_ = std::max(rss_mib_, rss_mib);
      }
    });
  }

  // The CPU route keeps staging chunks and scratch through a burst; the
  // presents after the idle delays (2 s for chunks beyond 128 MiB, 5 s for
  // scratch beyond 64 MiB) give them back.
  std::string IdleCheck() {
    const std::uint64_t rss_kept = StatusKb("VmRSS:");
    const std::size_t chunks_kept = g_host.created.size();
    std::size_t chunks_trimmed = 0;
    CaptureStderr([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(2100));
      emulation_->OnPresent();
      chunks_trimmed = g_host.created.size();
      std::this_thread::sleep_for(std::chrono::milliseconds(3100));
      emulation_->OnPresent();
    });
    const std::uint64_t rss_idle = StatusKb("VmRSS:");
    char line[256];
    std::snprintf(line, sizeof(line),
                  "idle: staging chunks %zu -> %zu after 2 s; process RSS %" PRIu64
                  " -> %" PRIu64 " MiB after 5 s",
                  chunks_kept, chunks_trimmed, rss_kept / 1024,
                  rss_idle / 1024);
    return line;
  }

  void Print(std::uint64_t reference) const {
    std::printf("   %-13s record %7.2f  submit %7.2f  release %7.2f ms  RSS "
                "peak +%" PRIu64 " MiB; first round %6.1f ms, RSS peak +%" PRIu64
                " MiB; %.1f MiB written, %s\n"
                "                 %s\n",
                arm_.name, Median(record_ms_), Median(submit_ms_),
                Median(release_ms_), rss_mib_, cold_ms_, cold_rss_mib_,
                static_cast<double>(written_) / 1048576.0,
                staging_hash_ != 0 && staging_hash_ == reference
                    ? "staging identical"
                    : "STAGING DIFFERS",
                StallBreakdown(log_).c_str());
  }

  std::uint64_t staging_hash() const { return staging_hash_; }

 private:
  static inline const VkCommandPool kPool =
      reinterpret_cast<VkCommandPool>(0x5100);
  static inline const VkCommandBuffer kCommandBuffer =
      reinterpret_cast<VkCommandBuffer>(0x5200);

  const HeadlessVulkanDevice& device_;
  const Workload& workload_;
  Arm arm_;
  bool ok_ = false;
  std::string why_;
  std::unique_ptr<VulkanEtc2Emulation> emulation_;
  DeviceMemory source_;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  std::vector<Texture> textures_;
  std::string log_;
  std::vector<double> record_ms_;
  std::vector<double> submit_ms_;
  std::vector<double> release_ms_;
  std::uint64_t rss_mib_ = 0;
  double cold_ms_ = 0;
  std::uint64_t cold_rss_mib_ = 0;
  std::uint64_t written_ = 0;
  std::uint64_t staging_hash_ = 0;
};

void RunEmulation(const HeadlessVulkanDevice& device, int reps,
                  const std::vector<std::string>& only,
                  const std::vector<std::string>& only_arms) {
  g_host.real = device.get_device_proc_addr();
  const VkDevice handle = device.device();
  const auto resolve = [&](auto& field, const char* name) {
    field = reinterpret_cast<std::remove_reference_t<decltype(field)>>(
        g_host.real(handle, name));
  };
  resolve(g_host.create_buffer, "vkCreateBuffer");
  resolve(g_host.destroy_buffer, "vkDestroyBuffer");
  resolve(g_host.bind_buffer_memory, "vkBindBufferMemory");
  resolve(g_host.map_memory, "vkMapMemory");
  resolve(g_host.unmap_memory, "vkUnmapMemory");
  resolve(g_host.free_memory, "vkFreeMemory");
  const std::uint32_t source_type = WriteCombinedType(device);
  if (source_type == UINT32_MAX) {
    std::printf("== emulation: no uncached host-visible memory, skipped\n");
    return;
  }
  std::vector<Arm> arms;
  for (const Arm& arm :
       {Arm{"cached", "cached", Etc2StagingMemory::kCached, false},
        Arm{"coherent", "coherent", Etc2StagingMemory::kCoherent, false},
        Arm{"bc cached", "bc", Etc2StagingMemory::kCached, true},
        Arm{"bc coherent", "bc-coherent", Etc2StagingMemory::kCoherent,
            true}}) {
    // ETC2->BC mode only when asked for.
    if (only_arms.empty() ? !arm.bc
                          : std::find(only_arms.begin(), only_arms.end(),
                                      arm.word) != only_arms.end()) {
      arms.push_back(arm);
    }
  }
  if (arms.empty()) {
    std::printf("== emulation: no such arm\n");
    return;
  }
  struct Kind {
    const char* kind;
    bool per_mip;
  };
  const Kind kinds[] = {{"small", false}, {"small", true}, {"big", false},
                        {"mixed", false}, {"mixed", true}};
  std::printf("== emulation: Roblox's source in memory type %u; arms "
              "interleaved round by round, median of %d rounds after a "
              "warm-up one; record = every copy hook, submit = PrepareSubmit "
              "(the stall), release = ReleaseCommandBuffer\n",
              source_type, reps);
  for (const Kind& kind : kinds) {
    if (!only.empty() &&
        std::find(only.begin(), only.end(), kind.kind) == only.end()) {
      continue;
    }
    const Workload workload = MakeWorkload(kind.kind, kind.per_mip);
    // Random blocks hit every ETC2 mode.
    const std::vector<std::uint8_t> bytes =
        RandomBytes(workload.source_bytes, 7);
    std::printf("  %s: uploads=%zu calls=%zu compressed=%.1f MiB\n",
                workload.name.c_str(), workload.uploads,
                workload.per_mip ? workload.uploads : workload.textures.size(),
                static_cast<double>(workload.compressed) / 1048576.0);
    std::vector<std::unique_ptr<Session>> sessions;
    for (const Arm& arm : arms) {
      sessions.push_back(std::make_unique<Session>(device, workload, bytes,
                                                   source_type, arm));
      if (!sessions.back()->ok()) {
        std::printf("  %s: %s\n", arm.name, sessions.back()->why().c_str());
        return;
      }
    }
    for (int round = 0; round <= reps; ++round) {
      for (const auto& session : sessions) {
        session->Round(round == 0, round == reps);
      }
    }
    // Each arm against the first of its mode: BC staging holds blocks.
    for (const auto& session : sessions) {
      const auto first = std::find_if(
          sessions.begin(), sessions.end(), [&](const auto& other) {
            return other->arm().bc == session->arm().bc;
          });
      session->Print((*first)->staging_hash());
    }
    sessions.clear();
    if (std::strcmp(kind.kind, "mixed") == 0 && !kind.per_mip) {
      const Arm& first = arms.front();
      Session session(device, workload, bytes, source_type, first);
      session.Round(true, false);
      std::printf("                 %s, alone: %s\n", first.name,
                  session.IdleCheck().c_str());
    }
  }
}

}  // namespace
}  // namespace mocktail::graphics::testing

int main(int argc, char** argv) {
  using namespace mocktail::graphics;
  using namespace mocktail::graphics::testing;
  int reps = 5;
  std::vector<std::string> sections;
  std::vector<std::string> workloads;
  std::vector<std::string> arms;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--reps") == 0 && index + 1 < argc) {
      reps = std::max(1, std::atoi(argv[++index]));
    } else if (std::strcmp(argv[index], "--workload") == 0 &&
               index + 1 < argc) {
      workloads.push_back(argv[++index]);
    } else if (std::strcmp(argv[index], "--arm") == 0 && index + 1 < argc) {
      arms.push_back(argv[++index]);
    } else {
      sections.push_back(argv[index]);
    }
  }
  const auto wanted = [&](const char* name) {
    return sections.empty() ||
           std::find(sections.begin(), sections.end(), name) != sections.end();
  };
  const unsigned workers =
      std::min(8U, std::max(1U, std::thread::hardware_concurrency() / 2));
  std::printf("etc2_cpu_perf: %u hardware threads, %u workers, copy kernel "
              "%s\n",
              std::thread::hardware_concurrency(), workers,
              MappedMemoryCopyKernelName(ActiveMappedMemoryCopyKernel()));
  std::string live;
  if (LiveGameRunning(&live)) {
    std::printf("a live game is running (%s): not measuring (CR-25)\n",
                live.c_str());
    return 1;
  }
  if (wanted("kernels")) {
    RunKernels();
  }
  if (wanted("bc")) {
    RunBc();
  }
  if (wanted("pool")) {
    RunPool(workers);
  }
  if (wanted("copy") || wanted("emulation") || wanted("gpucopy")) {
    std::string skip;
    std::unique_ptr<HeadlessVulkanDevice> device =
        HeadlessVulkanDevice::Create(&skip);
    if (device == nullptr) {
      std::printf("no Vulkan device: %s\n", skip.c_str());
      return 1;
    }
    std::printf("GPU: %s\n", device->Describe().c_str());
    if (wanted("copy")) {
      RunCopy(*device);
    }
    if (wanted("emulation")) {
      RunEmulation(*device, reps, workloads, arms);
    }
    if (wanted("gpucopy")) {
      RunGpuCopy(*device);
    }
  }
  WorkerPool::ShutdownShared();
  return 0;
}
