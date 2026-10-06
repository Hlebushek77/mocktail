// etc2_gpu_perf: the ETC2 GPU route's benchmark (DESIGN §9.3, WP2). A manual
// tool, not a test: it needs a Vulkan device and an idle machine, and prints
// timings. It refuses to run while a live Mocktail or Sober session runs
// (CR-25).
//
//   etc2_gpu_perf [--reps N] [--no-verify] [--no-cpu] [--arena-mb N]
//                 [--workload NAME]...
//
// Workloads (all by default) are the bursts of plus2-design/pipeline_bench.cc
// without the x4 upscale, which keeps upscaled images on the CPU route:
//   small  300 RGBA8 textures of 16..64 texels with their mips (1773 uploads)
//   big    10 RGB8 and 4 RGBA8 1024^2 textures with their mips (154 uploads)
//   mixed  500 small and 4 x big: the 811 ms stall's shape (3581 uploads,
//          49.1 MiB compressed, 302.6 MiB decoded)
// each as one vkCmdCopyBufferToImage per texture and one per mip.
//
// The production VulkanEtc2Emulation runs on the real device the way the
// adapter drives it: Roblox's command buffer holds a barrier, the uploads
// (rewritten by the emulation) and two timestamps around them; the submit
// goes through PrepareSubmit, the host vkQueueSubmit and FinishSubmit. Per
// burst it reports the CPU time of the record hooks (and per 2000 uploads),
// of PrepareSubmit (the decode pass's recording), of releasing the command
// buffer, and the GPU time of the copy-in, the decode (the route's own
// timestamps, MOCKTAIL_FRAME_STATS) and Roblox's copies. The first burst
// creates the arena chunks and the slot; the rest reuse them. With the CPU
// route (MOCKTAIL_ETC2_GPU=0, plus.1's code until WP1 lands) the same burst
// is timed once more for comparison. Every image is read back once and
// compared with the CPU decoder. The pipeline build is timed on two devices
// in a row (the second one finds the driver's caches warm).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "mocktail/graphics/vulkan_etc2_gpu.h"
#include "support/etc2_emulation_scenarios.h"
#include "support/etc2_gpu_backend.h"
#include "support/vulkan_test_device.h"

namespace mocktail::graphics::testing {
namespace {

using Clock = std::chrono::steady_clock;

double Ms(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

double Mib(std::uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

struct Texture {
  VkFormat format = VK_FORMAT_UNDEFINED;
  EtcFormat etc = EtcFormat::kEtc2Rgb8;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t mips = 1;
  VkDeviceSize offset = 0;  // in the source, mip 0 first
  VkImage image = VK_NULL_HANDLE;
  HostImageShape shape;
};

struct Workload {
  std::string name;
  std::vector<Texture> textures;
  VkDeviceSize source_bytes = 0;
  std::uint64_t uploads = 0;
  VkDeviceSize compressed = 0;
  VkDeviceSize decoded = 0;
};

void AddTexture(Workload* workload, VkFormat format, std::uint32_t width,
                std::uint32_t height) {
  Texture texture;
  texture.format = format;
  Etc2EmulatedFormat emulated;
  LookupEmulatedEtc2Format(format, &emulated);
  texture.etc = emulated.etc_format;
  texture.width = width;
  texture.height = height;
  while ((std::max(width, height) >> texture.mips) >= 1) {
    ++texture.mips;
  }
  texture.offset = workload->source_bytes;
  for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
    VkDeviceSize compressed = 0;
    VkDeviceSize decoded = 0;
    Etc2UploadByteCounts(texture.etc,
                         {std::max(1u, width >> mip),
                          std::max(1u, height >> mip), 1},
                         1, &compressed, &decoded);
    workload->source_bytes += compressed;
    workload->compressed += compressed;
    workload->decoded += decoded;
    ++workload->uploads;
  }
  // Like pipeline_bench: every texture starts 16-byte aligned.
  workload->source_bytes = (workload->source_bytes + 15) & ~VkDeviceSize{15};
  workload->textures.push_back(texture);
}

// pipeline_bench.cc's MakeWorkload, same seed, without the upscale.
Workload MakeWorkload(const std::string& kind) {
  Workload workload;
  workload.name = kind;
  std::mt19937 random(42);
  if (kind == "small" || kind == "mixed") {
    const int count = kind == "small" ? 300 : 500;
    for (int index = 0; index < count; ++index) {
      const std::uint32_t width = 16 + random() % 49;
      const std::uint32_t height = 16 + random() % 49;
      AddTexture(&workload, VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, width, height);
    }
  }
  if (kind == "big" || kind == "mixed") {
    const int rounds = kind == "big" ? 1 : 4;
    for (int round = 0; round < rounds; ++round) {
      for (int index = 0; index < 10; ++index) {
        AddTexture(&workload, VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, 1024, 1024);
      }
      for (int index = 0; index < 4; ++index) {
        AddTexture(&workload, VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, 1024, 1024);
      }
    }
  }
  return workload;
}

struct Burst {
  double record_ms = 0;   // the CmdCopyBufferToImage hooks
  double prepare_ms = 0;  // PrepareSubmit
  double release_ms = 0;  // ReleaseCommandBuffer
  double copy_in_ms = 0;  // GPU, the route's timestamps
  double decode_ms = 0;
  double copies_ms = 0;  // GPU, Roblox's copies (our timestamps)
  std::uint64_t passes = 0;
  double arena_mib = 0;
  double arena_over_cap_mib = 0;
};

// One emulated device on the test device, as the adapter would register it,
// with Roblox's source buffer and the workload's images.
class Bench {
 public:
  Bench(HeadlessVulkanDevice* device, Etc2EmulationOptions options)
      : device_(device), backend_(device), emulation_(options) {
    const float priority = 0.0F;
    VkDeviceQueueCreateInfo queue{};
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = device->queue_family();
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queue;
    emulation_.RegisterDevice(
        device->device(), device->physical_device(), true, device->memory(),
        device->properties(), device->queue_families(), info,
        device->get_device_proc_addr(), device->get_instance_proc_addr(),
        device->instance());
    emulation_.RegisterQueue(device->device(), device->queue(),
                             device->queue_family(), 0);
    pool_ = backend_.CreateCommandPool(
        device->queue_family(),
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    emulation_.RegisterCommandPool(
        device->device(), pool_, device->queue_family(),
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    command_buffer_ = backend_.AllocateCommandBuffer(pool_, false);
    VkQueryPoolCreateInfo queries{};
    queries.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queries.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queries.queryCount = 2;
    vk().vkCreateQueryPool(device->device(), &queries, nullptr, &queries_);
  }

  ~Bench() {
    vk().vkDeviceWaitIdle(device_->device());
    emulation_.ReleaseCommandBuffer(command_buffer_);
    backend_.DestroyCommandPool(pool_);
    emulation_.ForgetCommandPool(device_->device(), pool_);
    for (Texture& texture : workload_.textures) {
      emulation_.DestroyImage(device_->device(), texture.image, nullptr);
      backend_.ReleaseImage(texture.image);
    }
    if (source_ != VK_NULL_HANDLE) {
      emulation_.UnmapMemory(device_->device(), memory_);
      emulation_.DestroyBuffer(device_->device(), source_, nullptr);
      emulation_.FreeMemory(device_->device(), memory_, nullptr);
    }
    vk().vkDestroyQueryPool(device_->device(), queries_, nullptr);
    emulation_.DestroyDevice(device_->device());
  }

  const HeadlessVulkanDevice::Dispatch& vk() const { return device_->vk(); }
  Etc2GpuDevice* gpu() const { return Etc2GpuDevice::Find(device_->device()); }
  VulkanEtc2Emulation& emulation() { return emulation_; }

  // The pipelines and the selftest, through empty submits.
  bool WarmUp() {
    Etc2GpuDevice* runtime = gpu();
    if (runtime == nullptr) {
      return false;
    }
    runtime->StartBuild();
    if (!runtime->WaitForBuild(std::chrono::seconds(60))) {
      return false;
    }
    for (int round = 0;
         round < 3 && runtime->state() != Etc2GpuDevice::State::kReady &&
         runtime->state() != Etc2GpuDevice::State::kFailed;
         ++round) {
      Submit(false);
      vk().vkQueueWaitIdle(device_->queue());
    }
    return runtime->state() == Etc2GpuDevice::State::kReady;
  }

  bool Load(Workload workload) {
    workload_ = std::move(workload);
    for (Texture& texture : workload_.textures) {
      VkImageCreateInfo info{};
      info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      info.imageType = VK_IMAGE_TYPE_2D;
      info.format = texture.format;
      info.extent = {texture.width, texture.height, 1};
      info.mipLevels = texture.mips;
      info.arrayLayers = 1;
      info.samples = VK_SAMPLE_COUNT_1_BIT;
      info.tiling = VK_IMAGE_TILING_OPTIMAL;
      info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                   VK_IMAGE_USAGE_SAMPLED_BIT;
      if (emulation_.CreateImage(device_->device(), &info, nullptr,
                                 &texture.image) != VK_SUCCESS) {
        return false;
      }
      texture.shape = HostShapeFor(
          ScenarioImage{texture.format, texture.width, texture.height,
                        texture.mips, 1},
          emulation_.options().upscale, true);
      if (!backend_.PrepareImage(texture.image, texture.shape)) {
        return false;
      }
    }
    // Roblox's upload buffer: HOST_VISIBLE|COHERENT system memory (type 3
    // on NVIDIA, write-combined), mapped for good.
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = workload_.source_bytes;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vk().vkCreateBuffer(device_->device(), &info, nullptr, &source_) !=
        VK_SUCCESS) {
      return false;
    }
    VkMemoryRequirements requirements{};
    vk().vkGetBufferMemoryRequirements(device_->device(), source_,
                                       &requirements);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = device_->FindMemoryType(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        0,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (vk().vkAllocateMemory(device_->device(), &allocate, nullptr,
                              &memory_) != VK_SUCCESS) {
      return false;
    }
    emulation_.NoteAllocateMemory(device_->device(), memory_,
                                  allocate.memoryTypeIndex,
                                  requirements.size);
    void* data = nullptr;
    if (emulation_.BindBufferMemory(device_->device(), source_, memory_, 0) !=
            VK_SUCCESS ||
        emulation_.MapMemory(device_->device(), memory_, 0, VK_WHOLE_SIZE, 0,
                             &data) != VK_SUCCESS) {
      return false;
    }
    bytes_ = SeededBytes(7, static_cast<std::size_t>(workload_.source_bytes));
    std::memcpy(data, bytes_.data(), bytes_.size());
    return true;
  }

  // One burst: record every upload into Roblox's command buffer (one call
  // per texture, or per mip), submit, wait, release.
  Burst Run(bool per_mip) {
    Burst burst;
    const VkDevice device = device_->device();
    const VkCommandBufferUsageFlags flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    emulation_.BeginCommandBuffer(command_buffer_, pool_, flags);
    backend_.Begin(command_buffer_, false, flags);
    vk().vkCmdResetQueryPool(command_buffer_, queries_, 0, 2);
    // Roblox orders its uploads after earlier transfers (and the layout).
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vk().vkCmdPipelineBarrier(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier,
                              0, nullptr, 0, nullptr);
    vk().vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             queries_, 0);
    Clock::duration record{};
    std::vector<VkBufferImageCopy> regions;
    for (const Texture& texture : workload_.textures) {
      regions.clear();
      VkDeviceSize offset = texture.offset;
      for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
        VkBufferImageCopy region{};
        region.bufferOffset = offset;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
        region.imageExtent = {std::max(1u, texture.width >> mip),
                              std::max(1u, texture.height >> mip), 1};
        VkDeviceSize compressed = 0;
        VkDeviceSize decoded = 0;
        Etc2UploadByteCounts(texture.etc, region.imageExtent, 1, &compressed,
                             &decoded);
        offset += compressed;
        regions.push_back(region);
      }
      if (per_mip) {
        for (const VkBufferImageCopy& region : regions) {
          const Clock::time_point start = Clock::now();
          emulation_.CmdCopyBufferToImage(device, command_buffer_, source_,
                                          texture.image,
                                          VK_IMAGE_LAYOUT_GENERAL, 1, &region);
          record += Clock::now() - start;
        }
      } else {
        const Clock::time_point start = Clock::now();
        emulation_.CmdCopyBufferToImage(
            device, command_buffer_, source_, texture.image,
            VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(regions.size()),
            regions.data());
        record += Clock::now() - start;
      }
    }
    vk().vkCmdWriteTimestamp(command_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             queries_, 1);
    backend_.End(command_buffer_);
    burst.record_ms = Ms(record);
    burst.prepare_ms = Submit(true);
    vk().vkQueueWaitIdle(device_->queue());
    std::uint64_t stamps[2] = {};
    if (vk().vkGetQueryPoolResults(device, queries_, 0, 2, sizeof(stamps),
                                   stamps, sizeof(std::uint64_t),
                                   VK_QUERY_RESULT_64_BIT |
                                       VK_QUERY_RESULT_WAIT_BIT) ==
        VK_SUCCESS) {
      burst.copies_ms = static_cast<double>(stamps[1] - stamps[0]) *
                        device_->properties().limits.timestampPeriod / 1e6;
    }
    if (const Etc2GpuDevice* runtime = gpu(); runtime != nullptr) {
      const Etc2GpuDevice::Stats stats = runtime->stats();
      burst.arena_mib = Mib(stats.arena_bytes);
      burst.arena_over_cap_mib = Mib(stats.arena_over_cap_bytes);
    }
    const Clock::time_point start = Clock::now();
    emulation_.ReleaseCommandBuffer(command_buffer_);
    burst.release_ms = Ms(Clock::now() - start);
    backend_.Reset(command_buffer_);
    return burst;
  }

  // Roblox's submit of its command buffer (or of nothing), the way the
  // adapter makes it. Returns the PrepareSubmit time.
  double Submit(bool with_command_buffer) {
    VkSubmitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = with_command_buffer ? 1 : 0;
    info.pCommandBuffers = with_command_buffer ? &command_buffer_ : nullptr;
    const VkQueue queue = device_->queue();
    Etc2SubmitPlan plan;
    const Clock::time_point start = Clock::now();
    emulation_.PrepareSubmit(queue, 1, &info, VK_NULL_HANDLE, &plan);
    const double prepare = Ms(Clock::now() - start);
    for (const Etc2PreSubmit& pre : plan.pre_submits()) {
      vk().vkQueueSubmit(queue, 1, &pre.info, pre.fence);
    }
    const VkResult result = vk().vkQueueSubmit(
        queue, plan.count(1), plan.submits(&info), plan.fence(VK_NULL_HANDLE));
    emulation_.FinishSubmit(&plan, result, [&](VkFence trailing) {
      return vk().vkQueueSubmit(queue, 0, nullptr, trailing);
    });
    return prepare;
  }

  // Every mip of every image against the CPU decoder.
  bool Verify(std::string* failure) {
    std::vector<std::vector<std::uint8_t>> texels;
    for (const Texture& texture : workload_.textures) {
      if (!backend_.ReadImage(texture.image, texture.shape, &texels)) {
        *failure = "could not read an image back";
        return false;
      }
      VkDeviceSize offset = texture.offset;
      for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
        const VkExtent3D extent{std::max(1u, texture.width >> mip),
                                std::max(1u, texture.height >> mip), 1};
        VkDeviceSize compressed = 0;
        VkDeviceSize decoded = 0;
        Etc2UploadByteCounts(texture.etc, extent, 1, &compressed, &decoded);
        std::vector<std::uint8_t> expected(decoded);
        DecodeEtcImage(texture.etc, bytes_.data() + offset, compressed,
                       extent.width, extent.height, expected.data(),
                       expected.size());
        if (texels[mip] != expected) {
          *failure = std::to_string(texture.width) + "x" +
                     std::to_string(texture.height) + " mip " +
                     std::to_string(mip) + " differs from the CPU decoder";
          return false;
        }
        offset += compressed;
      }
    }
    return true;
  }

  const Workload& workload() const { return workload_; }

 private:
  HeadlessVulkanDevice* device_;
  Etc2GpuBackend backend_;
  VulkanEtc2Emulation emulation_;
  Workload workload_;
  VkBuffer source_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  std::vector<std::uint8_t> bytes_;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
  VkQueryPool queries_ = VK_NULL_HANDLE;
};

// --arena-mb: MOCKTAIL_ETC2_GPU_ARENA_MB, the output arenas' device-local cap.
std::uint64_t g_arena_bytes = 0;

// MOCKTAIL_FRAME_STATS gives the route's timestamps, and costs some CPU in
// the hooks: the record times are measured without it too.
Etc2EmulationOptions Options(bool gpu, bool frame_stats = true) {
  Etc2EmulationOptions options;
  options.gpu = gpu;
  options.gpu_arena_bytes = g_arena_bytes;
  options.upscale = 1;
  options.frame_stats = frame_stats;
  return options;
}

void PrintBurst(const char* label, const Burst& burst, std::uint64_t uploads,
                bool gpu) {
  std::printf("    %-6s record %7.2f ms (%5.2f per 2000 uploads)  prepare "
              "%7.2f ms  release %7.2f ms",
              label, burst.record_ms,
              burst.record_ms * 2000.0 / static_cast<double>(uploads),
              burst.prepare_ms, burst.release_ms);
  if (gpu) {
    std::printf("\n           gpu: copy-in %5.2f + decode %5.2f + copies %5.2f "
                "= %6.2f ms  (passes %llu, arena %.0f MiB, %.0f over the cap)",
                burst.copy_in_ms, burst.decode_ms, burst.copies_ms,
                burst.copy_in_ms + burst.decode_ms + burst.copies_ms,
                static_cast<unsigned long long>(burst.passes), burst.arena_mib,
                burst.arena_over_cap_mib);
  } else {
    std::printf("\n           gpu: copies %5.2f ms", burst.copies_ms);
  }
  std::printf("\n");
}

Burst Median(std::vector<Burst> bursts) {
  const auto median = [&](double Burst::*field) {
    std::vector<double> values;
    for (const Burst& burst : bursts) {
      values.push_back(burst.*field);
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  };
  Burst out = bursts.back();
  out.record_ms = median(&Burst::record_ms);
  out.prepare_ms = median(&Burst::prepare_ms);
  out.release_ms = median(&Burst::release_ms);
  out.copy_in_ms = median(&Burst::copy_in_ms);
  out.decode_ms = median(&Burst::decode_ms);
  out.copies_ms = median(&Burst::copies_ms);
  return out;
}

int RunGpu(HeadlessVulkanDevice* device, const Workload& workload,
           bool per_mip, int reps, bool verify) {
  Bench bench(device, Options(true));
  if (!bench.WarmUp()) {
    std::printf("    the GPU route did not get ready\n");
    return 1;
  }
  if (!bench.Load(workload)) {
    std::printf("    could not set the workload up\n");
    return 1;
  }
  // A slot's timestamps are read when the next call takes it: the stats
  // after burst k+1's submit hold burst k's GPU time.
  std::vector<Burst> bursts;
  Etc2GpuDevice::Stats before = bench.gpu()->stats();
  for (int rep = 0; rep <= reps; ++rep) {
    Burst burst = bench.Run(per_mip);
    const Etc2GpuDevice::Stats after = bench.gpu()->stats();
    if (rep > 0) {
      bursts.back().copy_in_ms =
          static_cast<double>(after.gpu_copy_ns - before.gpu_copy_ns) / 1e6;
      bursts.back().decode_ms =
          static_cast<double>(after.gpu_decode_ns - before.gpu_decode_ns) /
          1e6;
      bursts.back().passes = after.passes - before.passes;
    }
    before = after;
    if (rep < reps) {
      bursts.push_back(burst);
    }
  }
  PrintBurst("first", bursts.front(), workload.uploads, true);
  if (bursts.size() > 1) {
    PrintBurst("warm", Median({bursts.begin() + 1, bursts.end()}),
               workload.uploads, true);
  }
  const Etc2GpuDevice::Stats stats = bench.gpu()->stats();
  if (stats.rescued_uploads != 0 || stats.dropped_uploads != 0 ||
      stats.slot_waits != 0) {
    std::printf("    rescued %llu, dropped %llu, slot waits %llu\n",
                static_cast<unsigned long long>(stats.rescued_uploads),
                static_cast<unsigned long long>(stats.dropped_uploads),
                static_cast<unsigned long long>(stats.slot_waits));
  }
  if (verify) {
    std::string failure;
    if (!bench.Verify(&failure)) {
      std::printf("    MISMATCH: %s\n", failure.c_str());
      return 1;
    }
    std::printf("    every mip of every image matches the CPU decoder\n");
  }
  return 0;
}

// The CPU side alone, as in production: MOCKTAIL_FRAME_STATS off.
int RunGpuQuiet(HeadlessVulkanDevice* device, const Workload& workload,
                bool per_mip, int reps) {
  Bench bench(device, Options(true, false));
  if (!bench.WarmUp() || !bench.Load(workload)) {
    std::printf("    could not set the workload up\n");
    return 1;
  }
  bench.Run(per_mip);
  std::vector<Burst> bursts;
  for (int rep = 0; rep < reps; ++rep) {
    bursts.push_back(bench.Run(per_mip));
  }
  const Burst burst = Median(bursts);
  std::printf("    quiet  record %7.2f ms (%5.2f per 2000 uploads)  prepare "
              "%7.2f ms  release %7.2f ms  (no MOCKTAIL_FRAME_STATS)\n",
              burst.record_ms,
              burst.record_ms * 2000.0 / static_cast<double>(workload.uploads),
              burst.prepare_ms, burst.release_ms);
  return 0;
}

int RunCpu(HeadlessVulkanDevice* device, const Workload& workload,
           bool per_mip) {
  Bench bench(device, Options(false));
  if (!bench.Load(workload)) {
    std::printf("    could not set the workload up\n");
    return 1;
  }
  bench.Run(per_mip);
  PrintBurst("cpu", bench.Run(per_mip), workload.uploads, false);
  return 0;
}

int Main(int argc, char** argv) {
  int reps = 5;
  bool verify = true;
  bool cpu = true;
  std::vector<std::string> kinds;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--reps" && index + 1 < argc) {
      reps = std::max(1, std::atoi(argv[++index]));
    } else if (argument == "--no-verify") {
      verify = false;
    } else if (argument == "--no-cpu") {
      cpu = false;
    } else if (argument == "--arena-mb" && index + 1 < argc) {
      g_arena_bytes = std::strtoull(argv[++index], nullptr, 10) << 20;
    } else if (argument == "--workload" && index + 1 < argc) {
      kinds.push_back(argv[++index]);
    } else {
      std::fprintf(stderr,
                   "usage: etc2_gpu_perf [--reps N] [--no-verify] [--no-cpu] "
                   "[--arena-mb N] [--workload small|big|mixed]...\n");
      return 2;
    }
  }
  if (kinds.empty()) {
    kinds = {"small", "big", "mixed"};
  }
  std::string why;
  std::unique_ptr<HeadlessVulkanDevice> device =
      HeadlessVulkanDevice::Create(&why);
  if (device == nullptr) {
    std::fprintf(stderr, "etc2_gpu_perf: %s\n", why.c_str());
    return 1;
  }
  std::printf("GPU: %s\n", device->Describe().c_str());

  std::printf("pipeline build (shader module, layouts, pipeline):\n");
  for (int round = 0; round < 2; ++round) {
    Bench bench(device.get(), Options(true));
    if (!bench.WarmUp()) {
      std::printf("  device %d: the GPU route did not get ready (%s)\n",
                  round + 1,
                  bench.gpu() != nullptr ? bench.gpu()->failure().c_str()
                                         : "no runtime");
      return 1;
    }
    std::printf("  device %d: %.2f ms\n", round + 1,
                static_cast<double>(bench.gpu()->stats().build_ns) / 1e6);
  }

  int status = 0;
  for (const std::string& kind : kinds) {
    const Workload workload = MakeWorkload(kind);
    for (const bool per_mip : {false, true}) {
      std::printf("%s, one call per %s: %llu uploads, %.1f MiB -> %.1f MiB\n",
                  kind.c_str(), per_mip ? "mip" : "texture",
                  static_cast<unsigned long long>(workload.uploads),
                  Mib(workload.compressed), Mib(workload.decoded));
      status |= RunGpu(device.get(), workload, per_mip, reps, verify);
      status |= RunGpuQuiet(device.get(), workload, per_mip, reps);
      if (cpu) {
        status |= RunCpu(device.get(), workload, per_mip);
      }
    }
  }
  return status;
}

}  // namespace
}  // namespace mocktail::graphics::testing

int main(int argc, char** argv) {
  return mocktail::graphics::testing::Main(argc, argv);
}
