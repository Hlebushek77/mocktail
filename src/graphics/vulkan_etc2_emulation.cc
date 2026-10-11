#include "mocktail/graphics/vulkan_etc2_emulation.h"

#include <pthread.h>
#include <sys/stat.h>

#include "mocktail/graphics/bc_encoder.h"
#include "mocktail/graphics/chunk_arena.h"
#include "mocktail/graphics/etc2_decode_totals.h"
#include "mocktail/graphics/mapped_memory_copy.h"
#include "mocktail/graphics/texture_override.h"
#include "mocktail/graphics/vulkan_etc2_gpu.h"
#include "mocktail/graphics/vulkan_etc2_gpu_plan.h"
#include "mocktail/graphics/vulkan_etc2_planning.h"
#include "mocktail/graphics/worker_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mocktail::graphics {

bool LookupEmulatedEtc2Format(VkFormat format, Etc2EmulatedFormat* out) {
  Etc2EmulatedFormat mapped;
  switch (format) {
    case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_UNORM, EtcFormat::kEtc2Rgb8,
                VK_FORMAT_BC1_RGB_UNORM_BLOCK};
      break;
    case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_SRGB, EtcFormat::kEtc2Rgb8,
                VK_FORMAT_BC1_RGB_SRGB_BLOCK};
      break;
    case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_UNORM, EtcFormat::kEtc2Rgb8A1,
                VK_FORMAT_BC1_RGBA_UNORM_BLOCK};
      break;
    case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_SRGB, EtcFormat::kEtc2Rgb8A1,
                VK_FORMAT_BC1_RGBA_SRGB_BLOCK};
      break;
    case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_UNORM, EtcFormat::kEtc2Rgba8,
                VK_FORMAT_BC3_UNORM_BLOCK};
      break;
    case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_SRGB, EtcFormat::kEtc2Rgba8,
                VK_FORMAT_BC3_SRGB_BLOCK};
      break;
    case VK_FORMAT_EAC_R11_UNORM_BLOCK:
      mapped = {VK_FORMAT_R16_UNORM, EtcFormat::kEacR11,
                VK_FORMAT_BC4_UNORM_BLOCK};
      break;
    case VK_FORMAT_EAC_R11_SNORM_BLOCK:
      mapped = {VK_FORMAT_R16_SNORM, EtcFormat::kEacR11Signed,
                VK_FORMAT_BC4_SNORM_BLOCK};
      break;
    case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
      mapped = {VK_FORMAT_R16G16_UNORM, EtcFormat::kEacRg11,
                VK_FORMAT_BC5_UNORM_BLOCK};
      break;
    case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
      mapped = {VK_FORMAT_R16G16_SNORM, EtcFormat::kEacRg11Signed,
                VK_FORMAT_BC5_SNORM_BLOCK};
      break;
    default:
      return false;
  }
  if (out != nullptr) {
    *out = mapped;
  }
  return true;
}

bool LookupEmulatedEtc2Format(VkFormat format, bool bc,
                              Etc2EmulatedFormat* out) {
  Etc2EmulatedFormat mapped;
  if (!LookupEmulatedEtc2Format(format, &mapped)) {
    return false;
  }
  if (bc) {
    mapped.host_format = mapped.bc_host_format;
  }
  if (out != nullptr) {
    *out = mapped;
  }
  return true;
}

VkFormatFeatureFlags EmulatedEtc2FormatFeatures(
    VkFormatFeatureFlags host_features) {
  constexpr VkFormatFeatureFlags kCompressedFeatures =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
  return host_features & kCompressedFeatures;
}

VkFormatFeatureFlags Etc2BcRequiredFormatFeatures() {
  return EmulatedEtc2FormatFeatures(~VkFormatFeatureFlags{0});
}

const VkImageFormatListCreateInfo* FindImageFormatList(const void* next) {
  for (auto* cursor = static_cast<const VkBaseInStructure*>(next);
       cursor != nullptr; cursor = cursor->pNext) {
    if (cursor->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO) {
      return reinterpret_cast<const VkImageFormatListCreateInfo*>(cursor);
    }
  }
  return nullptr;
}

bool Etc2FormatListAllowsBc(const void* next) {
  const VkImageFormatListCreateInfo* list = FindImageFormatList(next);
  if (list == nullptr || list->pViewFormats == nullptr) {
    return true;
  }
  for (std::uint32_t index = 0; index < list->viewFormatCount; ++index) {
    if (!LookupEmulatedEtc2Format(list->pViewFormats[index], nullptr)) {
      return false;
    }
  }
  return true;
}

Etc2FormatListRewrite::Etc2FormatListRewrite(const void** host_next,
                                             bool bc) {
  const VkImageFormatListCreateInfo* list =
      host_next != nullptr ? FindImageFormatList(*host_next) : nullptr;
  found_ = list != nullptr;
  if (list == nullptr || list->pViewFormats == nullptr) {
    return;
  }
  bool etc2 = false;
  for (std::uint32_t index = 0; index < list->viewFormatCount; ++index) {
    etc2 = etc2 || LookupEmulatedEtc2Format(list->pViewFormats[index], nullptr);
  }
  if (!etc2) {
    return;
  }
  formats_.assign(list->pViewFormats,
                  list->pViewFormats + list->viewFormatCount);
  for (VkFormat& format : formats_) {
    Etc2EmulatedFormat mapped;
    if (LookupEmulatedEtc2Format(format, bc, &mapped)) {
      format = mapped.host_format;
    }
  }
  if (*host_next == list) {
    copy_ = *list;
    copy_.pViewFormats = formats_.data();
    *host_next = &copy_;
    return;
  }
  patched_ = const_cast<VkImageFormatListCreateInfo*>(list);
  original_ = list->pViewFormats;
  patched_->pViewFormats = formats_.data();
}

Etc2FormatListRewrite::~Etc2FormatListRewrite() {
  if (patched_ != nullptr) {
    patched_->pViewFormats = original_;
  }
}

namespace {

bool EmulationDisabledByEnvironment() {
  static const bool disabled = [] {
    const char* value = std::getenv("MOCKTAIL_DISABLE_ETC2_EMULATION");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
  }();
  return disabled;
}

// The spellings of the runtime config's environment switches.
bool ParseSwitch(const char* value, bool fallback) {
  if (value == nullptr) {
    return fallback;
  }
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
      std::strcmp(value, "on") == 0) {
    return true;
  }
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
      std::strcmp(value, "off") == 0) {
    return false;
  }
  return fallback;
}

bool ParseCount(const char* value, std::uint64_t limit, std::uint64_t* out) {
  if (value == nullptr || value[0] < '0' || value[0] > '9') {
    return false;
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == nullptr || *end != '\0' || parsed > limit) {
    return false;
  }
  *out = parsed;
  return true;
}

}  // namespace

bool Etc2SupportAdvertised(const char* value) {
  return value == nullptr || std::strcmp(value, "0") != 0;
}

bool Etc2SupportAdvertised() {
  static const bool advertised =
      Etc2SupportAdvertised(std::getenv("MOCKTAIL_ADVERTISE_ETC2"));
  return advertised;
}

std::uint32_t SmallTextureUpscale(const char* value) {
  if (value == nullptr || value[0] < '0' || value[0] > '9') {
    return kDefaultSmallTextureUpscale;
  }
  const long parsed = std::strtol(value, nullptr, 10);
  if (parsed < 1) {
    return 1;
  }
  return parsed > 8 ? 8 : static_cast<std::uint32_t>(parsed);
}

Etc2EmulationOptions Etc2EmulationOptions::FromLookup(
    const std::function<const char*(const char*)>& lookup) {
  Etc2EmulationOptions options;
  if (!lookup) {
    return options;
  }
  options.upscale =
      SmallTextureUpscale(lookup("MOCKTAIL_SMALL_TEXTURE_UPSCALE"));
  options.gpu = ParseSwitch(lookup("MOCKTAIL_ETC2_GPU"), true);
  options.to_bc = ParseSwitch(lookup("MOCKTAIL_ETC2_TO_BC"), true);
  std::uint64_t value = 0;
  if (ParseCount(lookup("MOCKTAIL_ETC2_WORKERS"), 64, &value)) {
    options.workers = static_cast<unsigned>(value);
  }
  if (ParseCount(lookup("MOCKTAIL_ETC2_GPU_ARENA_MB"), 1U << 20, &value)) {
    options.gpu_arena_bytes = value << 20;
  }
  if (const char* memory = lookup("MOCKTAIL_ETC2_STAGING_MEMORY");
      memory != nullptr) {
    if (std::strcmp(memory, "coherent") == 0) {
      options.staging_memory = Etc2StagingMemory::kCoherent;
    } else if (std::strcmp(memory, "cached") == 0) {
      options.staging_memory = Etc2StagingMemory::kCached;
    }
  }
  // As every MOCKTAIL_*_TRACE flag: set, and not "0".
  const char* frame_stats = lookup("MOCKTAIL_FRAME_STATS");
  options.frame_stats = frame_stats != nullptr && frame_stats[0] != '\0' &&
                        std::strcmp(frame_stats, "0") != 0;
  if (const char* path = lookup("MOCKTAIL_ETC2_TRACE"); path != nullptr) {
    options.trace_path = path;
  }
  if (const char* directory = lookup("MOCKTAIL_ETC2_TRACE_BLOCKS");
      directory != nullptr) {
    options.trace_blocks_dir = directory;
  }
  return options;
}

Etc2EmulationOptions Etc2EmulationOptions::FromEnvironment() {
  return FromLookup([](const char* name) -> const char* {
    return std::getenv(name);
  });
}

Etc2EmulationTestHooks& Etc2EmulationTesting() {
  static auto* hooks = new Etc2EmulationTestHooks;
  return *hooks;
}

namespace {

using Clock = std::chrono::steady_clock;

bool ShouldLog(std::atomic<unsigned>* counter, unsigned limit) {
  return counter->fetch_add(1, std::memory_order_relaxed) < limit;
}

unsigned DecodeWorkerCount() {
  static const unsigned count =
      std::min(8U, std::max(1U, std::thread::hardware_concurrency() / 2));
  return count;
}

std::uint64_t Nanoseconds(Clock::duration duration) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

// One device and the host functions Mocktail calls on it. Every pointer is
// the host loader's, never the adapter's hooks, so Mocktail's own objects
// are never mistaken for Roblox's.
struct HostDevice {
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  bool emulated = false;
  VkPhysicalDeviceMemoryProperties memory{};
  VkPhysicalDeviceProperties properties{};
  std::vector<VkQueueFamilyProperties> queue_families;
  PFN_vkCreateImage create_image = nullptr;
  PFN_vkDestroyImage destroy_image = nullptr;
  PFN_vkCreateImageView create_image_view = nullptr;
  PFN_vkBindBufferMemory bind_buffer_memory = nullptr;
  PFN_vkBindBufferMemory2 bind_buffer_memory2 = nullptr;
  PFN_vkBindImageMemory bind_image_memory = nullptr;
  PFN_vkBindImageMemory2 bind_image_memory2 = nullptr;
  PFN_vkMapMemory map_memory = nullptr;
  PFN_vkMapMemory2 map_memory2 = nullptr;
  PFN_vkUnmapMemory unmap_memory = nullptr;
  PFN_vkUnmapMemory2 unmap_memory2 = nullptr;
  PFN_vkFlushMappedMemoryRanges flush_mapped_memory_ranges = nullptr;
  PFN_vkInvalidateMappedMemoryRanges invalidate_mapped_memory_ranges = nullptr;
  PFN_vkAllocateMemory allocate_memory = nullptr;
  PFN_vkFreeMemory free_memory = nullptr;
  PFN_vkCreateBuffer create_buffer = nullptr;
  PFN_vkDestroyBuffer destroy_buffer = nullptr;
  PFN_vkGetBufferMemoryRequirements get_buffer_memory_requirements = nullptr;
  PFN_vkGetImageMemoryRequirements get_image_memory_requirements = nullptr;
  PFN_vkCmdCopyBufferToImage copy_buffer_to_image = nullptr;
  PFN_vkCmdCopyBufferToImage2 copy_buffer_to_image2 = nullptr;
  PFN_vkCmdCopyImage copy_image = nullptr;
  PFN_vkCmdCopyImage2 copy_image2 = nullptr;
  PFN_vkCmdBlitImage blit_image = nullptr;
  PFN_vkCmdExecuteCommands execute_commands = nullptr;
  // Mocktail's own command buffers and decode passes (the GPU route).
  PFN_vkCreateCommandPool create_command_pool = nullptr;
  PFN_vkDestroyCommandPool destroy_command_pool = nullptr;
  PFN_vkResetCommandPool reset_command_pool = nullptr;
  PFN_vkAllocateCommandBuffers allocate_command_buffers = nullptr;
  PFN_vkFreeCommandBuffers free_command_buffers = nullptr;
  PFN_vkBeginCommandBuffer begin_command_buffer = nullptr;
  PFN_vkEndCommandBuffer end_command_buffer = nullptr;
  PFN_vkCmdCopyBuffer copy_buffer = nullptr;
  PFN_vkCmdPipelineBarrier pipeline_barrier = nullptr;
  PFN_vkCmdBindPipeline bind_pipeline = nullptr;
  PFN_vkCmdBindDescriptorSets bind_descriptor_sets = nullptr;
  PFN_vkCmdPushConstants push_constants = nullptr;
  PFN_vkCmdDispatch dispatch = nullptr;
  PFN_vkCmdWriteTimestamp write_timestamp = nullptr;
  PFN_vkCmdResetQueryPool reset_query_pool = nullptr;
  PFN_vkCreateQueryPool create_query_pool = nullptr;
  PFN_vkDestroyQueryPool destroy_query_pool = nullptr;
  PFN_vkGetQueryPoolResults get_query_pool_results = nullptr;
  PFN_vkCreateFence create_fence = nullptr;
  PFN_vkDestroyFence destroy_fence = nullptr;
  PFN_vkResetFences reset_fences = nullptr;
  PFN_vkGetFenceStatus get_fence_status = nullptr;
  PFN_vkWaitForFences wait_for_fences = nullptr;
  PFN_vkCreateShaderModule create_shader_module = nullptr;
  PFN_vkDestroyShaderModule destroy_shader_module = nullptr;
  PFN_vkCreateComputePipelines create_compute_pipelines = nullptr;
  PFN_vkDestroyPipeline destroy_pipeline = nullptr;
  PFN_vkCreatePipelineLayout create_pipeline_layout = nullptr;
  PFN_vkDestroyPipelineLayout destroy_pipeline_layout = nullptr;
  PFN_vkCreateDescriptorSetLayout create_descriptor_set_layout = nullptr;
  PFN_vkDestroyDescriptorSetLayout destroy_descriptor_set_layout = nullptr;
  PFN_vkCreateDescriptorPool create_descriptor_pool = nullptr;
  PFN_vkDestroyDescriptorPool destroy_descriptor_pool = nullptr;
  PFN_vkAllocateDescriptorSets allocate_descriptor_sets = nullptr;
  PFN_vkFreeDescriptorSets free_descriptor_sets = nullptr;
  PFN_vkUpdateDescriptorSets update_descriptor_sets = nullptr;
  // Physical-device queries, when the adapter passed the host instance.
  PFN_vkGetPhysicalDeviceFeatures get_features = nullptr;
  PFN_vkGetPhysicalDeviceFormatProperties get_format_properties = nullptr;
  PFN_vkGetPhysicalDeviceImageFormatProperties get_image_format_properties =
      nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties2 get_memory_properties2 = nullptr;
  PFN_vkEnumerateDeviceExtensionProperties enumerate_device_extensions =
      nullptr;

  // Created lazily, or by RegisterDevice; guarded by `mutex`. Lock order:
  // State::devices_mutex, then this, then an arena's own.
  std::mutex mutex;
  std::unique_ptr<ChunkArena> host_staging;  // the CPU route's staging
  std::unique_ptr<Etc2GpuDevice> gpu;        // the GPU route
  std::atomic<bool> saw_emulated_image{false};

  // ETC2->BC mode (DESIGN C4.2), set once by RegisterDevice, and the
  // host's answers for BC images by (format, type, tiling, usage, flags)
  // (CR-14), under their own leaf mutex.
  bool bc = false;
  using BcImageKey = std::tuple<VkFormat, VkImageType, VkImageTiling,
                                VkImageUsageFlags, VkImageCreateFlags>;
  struct BcImageSupport {
    VkResult result = VK_ERROR_FORMAT_NOT_SUPPORTED;
    VkImageFormatProperties properties{};
  };
  std::mutex bc_support_mutex;
  std::map<BcImageKey, BcImageSupport> bc_support;
};

template <typename Function>
Function DeviceProc(PFN_vkGetDeviceProcAddr get, VkDevice device,
                    const char* name, const char* alias = nullptr) {
  PFN_vkVoidFunction proc = get(device, name);
  if (proc == nullptr && alias != nullptr) {
    proc = get(device, alias);
  }
  return reinterpret_cast<Function>(proc);
}

struct BufferBinding {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
};

struct Mapping {
  void* data = nullptr;
  VkDeviceSize offset = 0;
  VkDeviceSize size = VK_WHOLE_SIZE;
};

// `bc` (in the shape): the host image holds BC blocks (ETC2->BC mode).
struct ImageRecord : Etc2ImageShape {
  VkDevice device = VK_NULL_HANDLE;
  VkFormat app_format = VK_FORMAT_UNDEFINED;
  std::uint32_t copies = 0;  // record calls, with MOCKTAIL_FRAME_STATS=1
  std::shared_ptr<const RgbaImage> override;
  std::shared_ptr<const RgbaImage> base;
};

bool IsColorFormat(EtcFormat format) {
  return format == EtcFormat::kEtc2Rgb8 || format == EtcFormat::kEtc2Rgb8A1 ||
         format == EtcFormat::kEtc2Rgba8;
}

struct Staging {
  VkDevice device = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
};

using PendingUpload = Etc2PendingUpload;

// Everything recorded into one command buffer since its vkBeginCommandBuffer,
// released together when Roblox resets, frees or begins it again.
struct CommandRecord {
  VkDevice device = VK_NULL_HANDLE;
  std::uint32_t pool_family = kEtc2UnknownQueueFamily;
  VkCommandBufferUsageFlags begin_flags = 0;
  bool dispatch_seen = false;  // MOCKTAIL_VK_CALL_STATS=1 only
  std::vector<PendingUpload> uploads;
  std::vector<Staging> staging;        // one-off staging buffers
  std::vector<ArenaRef> staging_refs;  // host staging arena ranges
  std::vector<Etc2GpuUpload> gpu_uploads;
  std::vector<ArenaRef> gpu_regions;   // GPU output arena ranges
  std::vector<VkCommandBuffer> secondaries;

  bool HasUploads() const { return !uploads.empty() || !gpu_uploads.empty(); }
  bool HoldsResources() const {
    return !staging.empty() || !staging_refs.empty() || !gpu_regions.empty();
  }
};

struct PoolRecord {
  VkDevice device = VK_NULL_HANDLE;
  std::uint32_t family = kEtc2UnknownQueueFamily;
  VkCommandPoolCreateFlags flags = 0;
};

struct QueueRecord {
  VkDevice device = VK_NULL_HANDLE;
  std::uint32_t family = kEtc2UnknownQueueFamily;
  std::uint32_t index = 0;
};

// Memory of an emulated device, tracked for the diagnostics and the trace:
// its type, and (MOCKTAIL_FRAME_STATS=1) the images bound into it by offset,
// to spot aliasing.
struct BoundImage {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceSize end = 0;
  bool emulated = false;
};

struct MemoryRecord {
  VkDevice device = VK_NULL_HANDLE;
  std::uint32_t type = UINT32_MAX;
  std::multimap<VkDeviceSize, BoundImage> images;
};

struct ImageBinding {
  VkDevice device = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
};

// What a copy hook needs about its destination and command buffer.
struct CopyTarget {
  ImageRecord image;
  std::uint32_t pool_family = kEtc2UnknownQueueFamily;
  VkCommandBufferUsageFlags begin_flags = 0;
  bool dispatch_seen = false;
};

struct RouteChoice {
  Etc2Route route = Etc2Route::kPassthrough;
  Etc2CpuReason reason = Etc2CpuReason::kCount;
};

std::atomic<unsigned> g_image_logs{0};
std::atomic<unsigned> g_decode_logs{0};
std::atomic<unsigned> g_failure_logs{0};
std::atomic<unsigned> g_chunk_logs{0};

// The CPU route's host staging arena (DESIGN C1): 64 MiB chunks, a larger
// copy gets a chunk of its own, and idle chunks beyond 128 MiB are freed
// after 2 s.
constexpr VkDeviceSize kHostStagingChunkBytes = 64ULL << 20;
constexpr VkDeviceSize kHostStagingKeepIdleBytes = 128ULL << 20;
// Copy-in work comes in pieces of at most 256 KiB, batched into tasks of at
// least 128 KiB; a submit with less than 64 KiB of compressed data, or less
// than 256 KiB of upscaled output, keeps that phase on the calling thread.
constexpr std::size_t kCopyPieceBytes = 256U << 10;
constexpr std::size_t kCopyTaskBytes = 128U << 10;
constexpr std::size_t kParallelCopyMinimumBytes = 64U << 10;
constexpr std::size_t kParallelEmitMinimumBytes = 256U << 10;
std::atomic<unsigned> g_bc_fallback_logs{0};

// ETC2->BC mode on the CPU route (DESIGN C4.4): every block gets the
// endpoint refinement, as in the GPU route's etc2_bc pipeline
// (kEtc2ToBcRefine). Transcoding comes in pieces of at most 1024 blocks,
// about 0.2 ms on one core, batched into tasks of at least as many.
constexpr bool kBcRefine = kEtc2ToBcRefine;
constexpr std::size_t kBcPieceBlocks = 1024;

void LogFailure(const char* message) {
  if (ShouldLog(&g_failure_logs, 8)) {
    std::fprintf(stderr, "  [vulkan] ETC2 emulation: %s\n", message);
  }
}

bool CreateStaging(const HostDevice& dev, VkDeviceSize size, Staging* staging,
                   std::uint8_t** mapped) {
  if (dev.create_buffer == nullptr || dev.allocate_memory == nullptr ||
      dev.get_buffer_memory_requirements == nullptr ||
      dev.bind_buffer_memory == nullptr || dev.map_memory == nullptr ||
      dev.destroy_buffer == nullptr || dev.free_memory == nullptr ||
      size == 0) {
    return false;
  }
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buffer = VK_NULL_HANDLE;
  if (dev.create_buffer(dev.device, &buffer_info, nullptr, &buffer) !=
      VK_SUCCESS) {
    return false;
  }
  VkMemoryRequirements requirements{};
  dev.get_buffer_memory_requirements(dev.device, buffer, &requirements);
  constexpr VkMemoryPropertyFlags kRequired =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  std::uint32_t type = UINT32_MAX;
  for (std::uint32_t index = 0; index < dev.memory.memoryTypeCount; ++index) {
    if ((requirements.memoryTypeBits & (1U << index)) != 0 &&
        (dev.memory.memoryTypes[index].propertyFlags & kRequired) ==
            kRequired) {
      type = index;
      break;
    }
  }
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkMemoryAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = type;
  void* data = nullptr;
  if (type == UINT32_MAX ||
      dev.allocate_memory(dev.device, &allocate_info, nullptr, &memory) !=
          VK_SUCCESS) {
    dev.destroy_buffer(dev.device, buffer, nullptr);
    return false;
  }
  if (dev.bind_buffer_memory(dev.device, buffer, memory, 0) != VK_SUCCESS ||
      dev.map_memory(dev.device, memory, 0, VK_WHOLE_SIZE, 0, &data) !=
          VK_SUCCESS) {
    dev.destroy_buffer(dev.device, buffer, nullptr);
    dev.free_memory(dev.device, memory, nullptr);
    return false;
  }
  *staging = {dev.device, buffer, memory};
  *mapped = static_cast<std::uint8_t*>(data);
  return true;
}

void DestroyStaging(const HostDevice& dev, const Staging& staging) {
  if (dev.destroy_buffer != nullptr) {
    dev.destroy_buffer(dev.device, staging.buffer, nullptr);
  }
  if (dev.free_memory != nullptr) {
    dev.free_memory(dev.device, staging.memory, nullptr);
  }
}

// The host staging arena's memory type among `type_bits`: HOST_VISIBLE and
// HOST_COHERENT, in system memory when the device has any (not
// DEVICE_LOCAL: the BAR is small and slow to read back), preferring
// HOST_CACHED unless `coherent` asks for the uncached, write-combined kind.
// Last resort, as for one-off staging: the first coherent type.
std::uint32_t StagingMemoryType(const VkPhysicalDeviceMemoryProperties& memory,
                                std::uint32_t type_bits,
                                Etc2StagingMemory kind) {
  constexpr VkMemoryPropertyFlags kRequired =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const auto find = [&](VkMemoryPropertyFlags required,
                        VkMemoryPropertyFlags avoided) {
    for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
      const VkMemoryPropertyFlags flags = memory.memoryTypes[index].propertyFlags;
      if ((type_bits & (1U << index)) != 0 && (flags & required) == required &&
          (flags & avoided) == 0) {
        return index;
      }
    }
    return UINT32_MAX;
  };
  const VkMemoryPropertyFlags cached = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  const VkMemoryPropertyFlags device_local =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  std::uint32_t type = kind == Etc2StagingMemory::kCached
                           ? find(kRequired | cached, device_local)
                           : find(kRequired, cached | device_local);
  if (type == UINT32_MAX) {
    type = find(kRequired, device_local);
  }
  return type != UINT32_MAX ? type : find(kRequired, 0);
}

// Host staging arena chunks: persistently mapped TRANSFER_SRC buffers
// (TRANSFER_DST too, for the GPU route's rescue copy) in StagingMemoryType.
bool CreateStagingChunk(const HostDevice& dev, Etc2StagingMemory kind,
                        bool log, VkDeviceSize size,
                        ArenaChunkStorage* storage) {
  if (dev.create_buffer == nullptr || dev.allocate_memory == nullptr ||
      dev.get_buffer_memory_requirements == nullptr ||
      dev.bind_buffer_memory == nullptr || dev.map_memory == nullptr ||
      dev.destroy_buffer == nullptr || dev.free_memory == nullptr) {
    return false;
  }
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buffer = VK_NULL_HANDLE;
  if (dev.create_buffer(dev.device, &buffer_info, nullptr, &buffer) !=
      VK_SUCCESS) {
    return false;
  }
  VkMemoryRequirements requirements{};
  dev.get_buffer_memory_requirements(dev.device, buffer, &requirements);
  const std::uint32_t type =
      StagingMemoryType(dev.memory, requirements.memoryTypeBits, kind);
  VkMemoryAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = type;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* data = nullptr;
  if (type == UINT32_MAX ||
      dev.allocate_memory(dev.device, &allocate_info, nullptr, &memory) !=
          VK_SUCCESS) {
    dev.destroy_buffer(dev.device, buffer, nullptr);
    return false;
  }
  if (dev.bind_buffer_memory(dev.device, buffer, memory, 0) != VK_SUCCESS ||
      dev.map_memory(dev.device, memory, 0, VK_WHOLE_SIZE, 0, &data) !=
          VK_SUCCESS) {
    dev.destroy_buffer(dev.device, buffer, nullptr);
    dev.free_memory(dev.device, memory, nullptr);
    return false;
  }
  if (log) {
    std::fprintf(stderr,
                 "  [vulkan] ETC2 staging chunk: %llu MiB in memory type %u "
                 "(%s)\n",
                 static_cast<unsigned long long>(size >> 20), type,
                 (dev.memory.memoryTypes[type].propertyFlags &
                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0
                     ? "cached"
                     : "write-combined");
  }
  storage->buffer = buffer;
  storage->memory = memory;
  storage->mapped = static_cast<std::uint8_t*>(data);
  storage->size = size;
  return true;
}

std::string DescribeQueueFlags(VkQueueFlags flags) {
  static constexpr struct {
    VkQueueFlags bit;
    const char* name;
  } kNames[] = {
      {VK_QUEUE_GRAPHICS_BIT, "G"},
      {VK_QUEUE_COMPUTE_BIT, "C"},
      {VK_QUEUE_TRANSFER_BIT, "T"},
      {VK_QUEUE_SPARSE_BINDING_BIT, "S"},
      {VK_QUEUE_PROTECTED_BIT, "P"},
      {VK_QUEUE_VIDEO_DECODE_BIT_KHR, "VD"},
      {VK_QUEUE_VIDEO_ENCODE_BIT_KHR, "VE"},
      {VK_QUEUE_OPTICAL_FLOW_BIT_NV, "OF"},
  };
  std::string out;
  for (const auto& name : kNames) {
    if ((flags & name.bit) != 0) {
      out += out.empty() ? "" : "|";
      out += name.name;
      flags &= ~name.bit;
    }
  }
  if (flags != 0) {
    char rest[16];
    std::snprintf(rest, sizeof(rest), "%s0x%x", out.empty() ? "" : "|",
                  static_cast<unsigned>(flags));
    out += rest;
  }
  return out.empty() ? "none" : out;
}

// "family 0 (G|C|T|S) x1 of 16, family 2 (C|T) x1 of 8": the queues Roblox
// asked for, which decides where its uploads can be decoded on the GPU.
std::string DescribeEtc2Queues(
    const VkDeviceCreateInfo& create_info,
    const std::vector<VkQueueFamilyProperties>& families) {
  std::string out;
  for (std::uint32_t index = 0; index < create_info.queueCreateInfoCount &&
                                create_info.pQueueCreateInfos != nullptr;
       ++index) {
    const VkDeviceQueueCreateInfo& queue = create_info.pQueueCreateInfos[index];
    const bool known = queue.queueFamilyIndex < families.size();
    char entry[96];
    std::snprintf(
        entry, sizeof(entry), "%sfamily %u (%s) x%u of %u",
        out.empty() ? "" : ", ", queue.queueFamilyIndex,
        known ? DescribeQueueFlags(families[queue.queueFamilyIndex].queueFlags)
                    .c_str()
              : "unknown",
        queue.queueCount,
        known ? families[queue.queueFamilyIndex].queueCount : 0U);
    out += entry;
  }
  return out.empty() ? "none" : out;
}

// The name of the calling thread, quoted for a log line.
std::string QuotedThreadName() {
  char name[64] = "";
  if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0 ||
      name[0] == '\0') {
    std::snprintf(name, sizeof(name), "unnamed");
  }
  std::string out = "\"";
  for (const char* cursor = name; *cursor != '\0'; ++cursor) {
    out += *cursor == '"' ? '\'' : *cursor;
  }
  return out + "\"";
}

}  // namespace

struct VulkanEtc2Emulation::State {
  explicit State(Etc2EmulationOptions emulation_options)
      : options(std::move(emulation_options)) {
    OpenTraces();
  }
  ~State() {
    for (std::FILE* file : {trace, blocks, blocks_index}) {
      if (file != nullptr) {
        std::fclose(file);
      }
    }
  }

  const Etc2EmulationOptions options;
  std::mutex mutex;
  std::mutex devices_mutex;
  std::unordered_map<VkPhysicalDevice, bool> physical_devices;
  std::unordered_map<VkPhysicalDevice, bool> bc_physical_devices;
  std::vector<std::unique_ptr<HostDevice>> devices;
  std::unordered_map<VkImage, ImageRecord> images;
  std::unordered_map<VkBuffer, BufferBinding> buffers;
  std::unordered_map<VkDeviceMemory, Mapping> mappings;
  std::unordered_map<VkCommandBuffer, CommandRecord> commands;
  std::unordered_map<VkCommandPool, PoolRecord> pools;
  std::unordered_map<VkQueue, QueueRecord> queues;
  // Memory types with MOCKTAIL_FRAME_STATS=1 or a trace (TrackMemory);
  // image bindings with MOCKTAIL_FRAME_STATS=1 only.
  std::unordered_map<VkDeviceMemory, MemoryRecord> memories;
  std::unordered_map<VkImage, ImageBinding> image_bindings;
  std::atomic<bool> has_commands{false};
  // Concurrent submits decode one at a time.
  std::mutex scratch_mutex;
  // The CPU route's scratch: compressed bytes, and decoded texels of the
  // uploads that do not decode straight into staging.
  Etc2Scratch fast_source;
  Etc2Scratch fast_decoded;
  // When Housekeeping may run next (steady clock, ns).
  std::atomic<std::int64_t> next_housekeeping_ns{0};
  TextureOverrides overrides = TextureOverrides::FromEnvironment();
  // This session's decoding, for the totals line in the log, and with
  // MOCKTAIL_FRAME_STATS=1 the current two-second window.
  std::mutex totals_mutex;
  Etc2DecodeTotals totals;
  Etc2DecodeTotals window;
  Clock::time_point window_start{};
  // MOCKTAIL_ETC2_TRACE: one CSV line per decoded upload.
  // MOCKTAIL_ETC2_TRACE_BLOCKS: up to kTraceBlocksBytes of distinct
  // compressed uploads with an index, a corpus for offline tests.
  static constexpr std::uint64_t kTraceBlocksBytes = 64ULL << 20;
  std::mutex trace_mutex;
  std::FILE* trace = nullptr;
  std::FILE* blocks = nullptr;
  std::FILE* blocks_index = nullptr;
  std::uint64_t blocks_bytes = 0;
  bool blocks_full_logged = false;
  std::unordered_set<std::uint64_t> blocks_saved;
  std::uint64_t submit_sequence = 0;
  std::atomic<std::uint64_t> record_sequence{0};
  std::set<std::thread::id> submit_thread_ids;
  std::set<std::string> submit_threads;
  std::vector<std::string> new_submit_threads;

  bool Tracing() const { return trace != nullptr || blocks != nullptr; }
  // Memory types are only looked up for the diagnostics and the trace.
  bool TrackMemory() const { return options.frame_stats || Tracing(); }

  void OpenTraces() {
    if (!options.trace_path.empty()) {
      trace = std::fopen(options.trace_path.c_str(), "w");
      if (trace == nullptr) {
        std::fprintf(stderr, "  [vulkan] ETC2 upload trace: cannot write %s\n",
                     options.trace_path.c_str());
      } else {
        std::fprintf(trace,
                     "t_ns,submit,record_call,image,vkformat,w,h,mip,layers,"
                     "scale,row_length,image_height,offset,bytes,"
                     "src_mem_type,route,hash64\n");
        std::fprintf(stderr, "  [vulkan] ETC2 upload trace: %s\n",
                     options.trace_path.c_str());
      }
    }
    if (!options.trace_blocks_dir.empty()) {
      ::mkdir(options.trace_blocks_dir.c_str(), 0700);
      const std::string base = options.trace_blocks_dir + "/etc2_blocks";
      blocks = std::fopen((base + ".bin").c_str(), "wb");
      blocks_index = blocks != nullptr
                         ? std::fopen((base + ".csv").c_str(), "w")
                         : nullptr;
      if (blocks_index == nullptr) {
        if (blocks != nullptr) {
          std::fclose(blocks);
          blocks = nullptr;
        }
        std::fprintf(stderr, "  [vulkan] ETC2 block trace: cannot write %s\n",
                     options.trace_blocks_dir.c_str());
      } else {
        std::fprintf(blocks_index,
                     "offset,bytes,vkformat,w,h,mip,layers,hash64\n");
        std::fprintf(stderr, "  [vulkan] ETC2 block trace: %s.{bin,csv}\n",
                     base.c_str());
      }
    }
  }

  // Writes the trace lines of one submit's uploads; `compressed` holds
  // their source bytes packed, in order. Returns the time it took.
  std::uint64_t TraceUploads(const std::vector<const PendingUpload*>& uploads,
                             const std::vector<std::uint32_t>& memory_types,
                             const std::uint8_t* compressed) {
    const Clock::time_point start = Clock::now();
    std::lock_guard<std::mutex> lock(trace_mutex);
    const std::uint64_t submit = ++submit_sequence;
    const auto now_ns = static_cast<unsigned long long>(
        Nanoseconds(start.time_since_epoch()));
    for (std::size_t index = 0; index < uploads.size(); ++index) {
      const PendingUpload& upload = *uploads[index];
      const std::uint64_t hash = HashBytes(compressed, upload.compressed);
      if (trace != nullptr) {
        std::fprintf(
            trace,
            "%llu,%llu,%llu,0x%llx,%d,%u,%u,%u,%u,%u,%u,%u,%d:%d,%llu,%d,%s,"
            "%016llx\n",
            now_ns, static_cast<unsigned long long>(submit),
            static_cast<unsigned long long>(upload.record_call),
            static_cast<unsigned long long>(
                reinterpret_cast<std::uintptr_t>(upload.image)),
            static_cast<int>(upload.app_format), upload.width, upload.height,
            upload.mip_level, upload.layers, upload.scale, upload.row_length,
            upload.image_height, upload.offset_x, upload.offset_y,
            static_cast<unsigned long long>(upload.compressed),
            memory_types[index] == UINT32_MAX
                ? -1
                : static_cast<int>(memory_types[index]),
            upload.route == Etc2Route::kGpu ? "gpu" : "cpu",
            static_cast<unsigned long long>(hash));
      }
      if (blocks != nullptr && blocks_saved.count(hash) == 0) {
        if (blocks_bytes + upload.compressed <= kTraceBlocksBytes) {
          blocks_saved.insert(hash);
          std::fwrite(compressed, 1, upload.compressed, blocks);
          std::fprintf(blocks_index, "%llu,%llu,%d,%u,%u,%u,%u,%016llx\n",
                       static_cast<unsigned long long>(blocks_bytes),
                       static_cast<unsigned long long>(upload.compressed),
                       static_cast<int>(upload.app_format), upload.width,
                       upload.height, upload.mip_level, upload.layers,
                       static_cast<unsigned long long>(hash));
          blocks_bytes += upload.compressed;
        } else if (!blocks_full_logged) {
          blocks_full_logged = true;
          std::fprintf(stderr, "  [vulkan] ETC2 block trace is full (%llu MiB)\n",
                       static_cast<unsigned long long>(kTraceBlocksBytes >> 20));
        }
      }
      compressed += upload.compressed;
    }
    for (std::FILE* file : {trace, blocks, blocks_index}) {
      if (file != nullptr) {
        std::fflush(file);
      }
    }
    return Nanoseconds(Clock::now() - start);
  }

  // Applies `update` to the session totals and to the current window.
  template <typename Update>
  void CountTotals(Update&& update) {
    std::lock_guard<std::mutex> lock(totals_mutex);
    update(totals);
    if (options.frame_stats) {
      update(window);
    }
  }

  // The step-0 diagnostics are only collected with MOCKTAIL_FRAME_STATS=1.
  template <typename Update>
  void CountDiagnostics(Update&& update) {
    if (options.frame_stats) {
      CountTotals(std::forward<Update>(update));
    }
  }

  void LogTotals(const char* label) {
    std::string line;
    {
      std::lock_guard<std::mutex> lock(totals_mutex);
      if (totals.images == 0 && totals.uploads == 0 &&
          totals.gpu_uploads == 0) {
        return;
      }
      line = FormatEtc2DecodeTotals(totals);
      if (options.frame_stats || totals.gpu_uploads != 0) {
        line += FormatEtc2DecodeTotalsDetail(totals);
      }
    }
    std::fprintf(stderr, "  [vulkan] ETC2 decode totals%s: %s\n", label,
                 line.c_str());
  }

  // Whether an override PNG can replace this upload (and so whether it is
  // hashed, dumped and looked up).
  bool MayOverride(const PendingUpload& upload) const {
    return overrides.active() && upload.layers == 1 &&
           IsColorFormat(upload.format);
  }

  std::shared_ptr<const RgbaImage> FindOverride(const PendingUpload& upload,
                                                const std::uint8_t* compressed,
                                                const std::uint8_t* decoded) {
    if (!MayOverride(upload)) {
      return nullptr;
    }
    std::shared_ptr<const RgbaImage> replacement;
    if (upload.mip_level == 0) {
      const std::uint64_t hash = HashBytes(compressed, upload.compressed);
      overrides.Dump(hash, upload.width, upload.height, decoded);
      replacement = overrides.Lookup(hash);
      std::lock_guard<std::mutex> lock(mutex);
      const auto record = images.find(upload.image);
      if (record != images.end()) {
        record->second.override = replacement;
      }
    } else {
      std::lock_guard<std::mutex> lock(mutex);
      const auto record = images.find(upload.image);
      if (record != images.end()) {
        replacement = record->second.override;
      }
    }
    return replacement;
  }

  void EmitUpload(const PendingUpload& upload, const std::uint8_t* compressed,
                  const std::uint8_t* decoded) {
    const std::shared_ptr<const RgbaImage> replacement =
        FindOverride(upload, compressed, decoded);
    if (upload.bc) {
      EmitBc(upload, compressed, replacement.get());
      return;
    }
    if (replacement != nullptr) {
      ResampleRgba(*replacement, upload.target_width, upload.target_height,
                   upload.target);
    } else if (upload.scale > 1) {
      EmitScaled(upload, decoded);
    } else {
      std::memcpy(upload.target, decoded, upload.decoded);
    }
  }

  // An upload into a BC image. Roblox's blocks transcode as the GPU route
  // does it: whole decoded blocks, padding texels of edge blocks included.
  // An override is resampled to the region (never upscaled in BC mode) and
  // encoded, its edge texels repeated; for RGB8A1, texels with alpha below
  // 128 become transparent black and the rest opaque, where RGBA8 hosts
  // keep the override's alpha (CR-24).
  void EmitBc(const PendingUpload& upload, const std::uint8_t* compressed,
              const RgbaImage* replacement) {
    bool encoded = false;
    if (replacement != nullptr) {
      std::vector<std::uint8_t> texels(
          static_cast<std::size_t>(upload.target_width) *
          upload.target_height * 4);
      ResampleRgba(*replacement, upload.target_width, upload.target_height,
                   texels.data());
      encoded = TranscodeDecodedToBc(
          upload.format, texels.data(), upload.target_width,
          upload.target_height, 1, upload.target,
          static_cast<std::size_t>(upload.target_bytes), kBcRefine);
    } else {
      encoded = TranscodeEtcToBc(
          upload.format, compressed, static_cast<std::size_t>(upload.compressed),
          upload.target, static_cast<std::size_t>(upload.target_bytes),
          kBcRefine);
    }
    if (!encoded) {
      LogFailure("could not encode an upload as BC");
    }
  }

  HostDevice* Find(VkDevice device) {
    std::lock_guard<std::mutex> lock(devices_mutex);
    for (const auto& candidate : devices) {
      if (candidate->device == device) {
        return candidate.get();
      }
    }
    return nullptr;
  }

  // [CR-14] In ETC2->BC mode, why the host cannot take this image as BC,
  // or null when it can. RGBA8 accepts usages and views BC does not
  // (storage, attachments, uncompressed views), and failing vkCreateImage
  // where the default mode succeeds would be worse than one image left
  // RGBA8/R16. `flags` are the host image's.
  const char* BcImageRefusal(HostDevice& dev, const VkImageCreateInfo& info,
                             VkImageCreateFlags flags, VkFormat format) {
    if (!Etc2FormatListAllowsBc(info.pNext)) {
      return "its format list names a format that is not ETC2/EAC";
    }
    if (dev.get_image_format_properties == nullptr) {
      return "image formats cannot be queried";
    }
    const HostDevice::BcImageKey key{format, info.imageType, info.tiling,
                                     info.usage, flags};
    HostDevice::BcImageSupport support;
    bool known = false;
    {
      std::lock_guard<std::mutex> lock(dev.bc_support_mutex);
      const auto found = dev.bc_support.find(key);
      if (found != dev.bc_support.end()) {
        support = found->second;
        known = true;
      }
    }
    if (!known) {
      support.result = dev.get_image_format_properties(
          dev.physical_device, format, info.imageType, info.tiling,
          info.usage, flags, &support.properties);
      std::lock_guard<std::mutex> lock(dev.bc_support_mutex);
      dev.bc_support.emplace(key, support);
    }
    if (support.result != VK_SUCCESS) {
      return "the device takes no BC image with its usage and flags";
    }
    const VkImageFormatProperties& limits = support.properties;
    if (info.extent.width > limits.maxExtent.width ||
        info.extent.height > limits.maxExtent.height ||
        info.extent.depth > limits.maxExtent.depth ||
        info.mipLevels > limits.maxMipLevels ||
        info.arrayLayers > limits.maxArrayLayers ||
        (info.samples & limits.sampleCounts) == 0) {
      return "it exceeds the BC format's limits";
    }
    return nullptr;
  }

  // [CR-14] An image copy between a BC image of an ETC2->BC device and one
  // of its images kept RGBA8/R16: a BC block is 8 or 16 bytes and a texel
  // of those 2 to 4, so the host formats are not size-compatible
  // (VUID-vkCmdCopyImage-srcImage-01548), and each side would read the
  // extent in its own units. Such a copy is left out (the destination
  // keeps what it held), logged and counted. True when it was.
  bool SkipMixedBcCopy(const HostDevice& dev, VkImage source,
                       VkImage destination) {
    if (!dev.bc) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      const auto from = images.find(source);
      const auto to = images.find(destination);
      if (from == images.end() || to == images.end() ||
          from->second.bc == to->second.bc) {
        return false;
      }
    }
    LogFailure("skipped an image copy between a BC image and one kept "
               "RGBA8/R16");
    CountTotals([](Etc2DecodeTotals& totals) { ++totals.bc_mixed_copies; });
    return true;
  }

  bool BlitScaledCopy(const HostDevice& dev, VkCommandBuffer command_buffer,
                      VkImage source, VkImageLayout source_layout,
                      VkImage destination, VkImageLayout destination_layout,
                      std::uint32_t count, const VkImageCopy* regions);

  bool LookupImage(const HostDevice& dev, VkImage image,
                   ImageRecord* record) {
    if (!dev.emulated) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = images.find(image);
    if (found == images.end()) {
      return false;
    }
    *record = found->second;
    return true;
  }

  // The destination's record and what the command buffer's
  // vkBeginCommandBuffer said. False: not an emulated image.
  bool LookupCopy(const HostDevice& dev, VkCommandBuffer command_buffer,
                  VkImage image, CopyTarget* target) {
    if (!dev.emulated) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = images.find(image);
    if (found == images.end()) {
      return false;
    }
    target->image = found->second;
    const auto record = commands.find(command_buffer);
    if (record != commands.end()) {
      target->pool_family = record->second.pool_family;
      target->begin_flags = record->second.begin_flags;
      target->dispatch_seen = record->second.dispatch_seen;
    }
    return true;
  }

  // Record time: how this copy is served (DESIGN §3.2).
  RouteChoice ChooseRoute(const HostDevice& dev, const CopyTarget& target) {
    if (!dev.emulated) {
      return {Etc2Route::kPassthrough, Etc2CpuReason::kCount};
    }
    if (!options.gpu) {
      return {Etc2Route::kCpu, Etc2CpuReason::kGpuDisabled};
    }
    if (dev.gpu == nullptr) {
      return {Etc2Route::kCpu, Etc2CpuReason::kGpuUnavailable};
    }
    switch (dev.gpu->state()) {
      case Etc2GpuDevice::State::kReady:
        break;
      case Etc2GpuDevice::State::kFailed:
        return {Etc2Route::kCpu, Etc2CpuReason::kGpuUnavailable};
      default:  // building, or the selftest not done yet
        return {Etc2Route::kCpu, Etc2CpuReason::kGpuNotReady};
    }
    // Upscaled small textures need the CPU's resampling (DESIGN C3.5).
    if (target.image.scale > 1) {
      return {Etc2Route::kCpu, Etc2CpuReason::kUpscaled};
    }
    // Override lookups hash the decoded colour texels on the CPU. Only a
    // dump directory or an override PNG makes them active: the runtime sets
    // MOCKTAIL_TEXTURE_OVERRIDE_DIR in every session.
    if (overrides.active() && IsColorFormat(target.image.format.etc_format)) {
      return {Etc2Route::kCpu, Etc2CpuReason::kOverrides};
    }
    const std::uint32_t family = target.pool_family;
    if (family >= dev.queue_families.size() ||
        (dev.queue_families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
      return {Etc2Route::kCpu, Etc2CpuReason::kNoComputeQueue};
    }
    return {Etc2Route::kGpu, Etc2CpuReason::kCount};
  }

  struct GpuCopyResult {
    bool recorded = false;
    bool first_upload = false;  // the command buffer's first
    VkDeviceSize compressed = 0;
    // Why the CPU route serves it after all; kCount when the regions
    // cannot be planned (the CPU route then drops the copy and says why).
    Etc2CpuReason reason = Etc2CpuReason::kCount;
  };

  // Records the GPU route of one copy (DESIGN C3.1): the rewritten copy
  // reads padded rows from an output arena range that the decode pass in
  // front of the batch fills. Nothing reads Roblox's memory here.
  template <typename Region, typename HostCopy>
  GpuCopyResult RecordGpuCopy(const HostDevice& dev,
                              VkCommandBuffer command_buffer,
                              const CopyTarget& target, VkBuffer source,
                              VkImage destination, std::uint32_t count,
                              const Region* regions, HostCopy& host_copy) {
    GpuCopyResult result;
    // Kept per thread, so that a warm copy hook allocates nothing here.
    thread_local std::vector<Region> rewritten;
    thread_local std::vector<Etc2GpuUpload> uploads;
    rewritten.clear();
    uploads.clear();
    VkDeviceSize total = 0;
    if (!PlanGpuRegions(target.image, source, count, regions, target.image.bc,
                        &rewritten, &uploads, &total)) {
      return result;
    }
    if (total > dev.gpu->max_output_bytes()) {
      result.reason = Etc2CpuReason::kTooLarge;
      return result;
    }
    ArenaRef output;
    if (!dev.gpu->TakeOutput(target.pool_family, total, &output)) {
      result.reason = Etc2CpuReason::kArenaFull;
      return result;
    }
    for (std::size_t index = 0; index < uploads.size(); ++index) {
      rewritten[index].bufferOffset += output.offset;
      Etc2GpuUpload& upload = uploads[index];
      upload.output = output.buffer;
      upload.output_offset += output.offset;
      upload.image = destination;
      upload.pool_family = target.pool_family;
      result.compressed += upload.compressed;
    }
    result.first_upload =
        RecordGpu(command_buffer, dev.device, destination, uploads, output);
    result.recorded = true;
    host_copy(output.buffer, rewritten.data());
    return result;
  }

  struct CpuCopyResult {
    bool recorded = false;
    bool staging_allocated = false;  // a one-off staging buffer
    bool staging_fallback = false;   // because the arena had none
    bool first_upload = false;  // the command buffer's first
    VkDeviceSize compressed = 0;
  };

  // Records the CPU route of one copy: the decoded texels go to staging at
  // submit time, and the host copy reads them from there. Staging is a
  // range of the device's host staging arena, which costs no Vulkan call
  // once its chunks exist; a one-off buffer only when the arena cannot
  // grow.
  template <typename Region, typename HostCopy>
  CpuCopyResult RecordCpuCopy(const HostDevice& dev,
                              VkCommandBuffer command_buffer,
                              const CopyTarget& target, VkBuffer source,
                              VkImage destination, std::uint32_t count,
                              const Region* regions, HostCopy& host_copy) {
    CpuCopyResult result;
    std::vector<Region> rewritten;
    std::vector<PendingUpload> uploads;
    VkDeviceSize total = 0;
    if (!PlanRegions(target.image, dev.device, source, destination, count,
                     regions, &rewritten, &uploads, &total)) {
      LogFailure("skipped an upload with an unsupported region");
      return result;
    }
    for (const PendingUpload& upload : uploads) {
      result.compressed += upload.compressed;
    }
    ArenaRef range;
    Staging staging;
    std::uint8_t* mapped = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (dev.host_staging != nullptr) {
      if (dev.host_staging->Take(total, &range) && range.mapped != nullptr) {
        mapped = range.mapped;
        buffer = range.buffer;
        for (Region& region : rewritten) {
          region.bufferOffset += range.offset;
        }
      } else {
        ChunkArena::ReleaseRef(range);
        range = ArenaRef{};
        result.staging_fallback = true;
        LogFailure("the staging arena is full; using a one-off buffer");
      }
    }
    if (mapped == nullptr) {
      if (!CreateStaging(dev, total, &staging, &mapped)) {
        LogFailure("could not allocate a host-visible staging buffer");
        return result;
      }
      result.staging_allocated = true;
      buffer = staging.buffer;
    }
    const std::uint64_t record_call =
        Tracing() ? record_sequence.fetch_add(1, std::memory_order_relaxed) + 1
                  : 0;
    for (PendingUpload& upload : uploads) {
      upload.target = mapped + upload.target_offset;
      upload.route = Etc2Route::kCpu;
      upload.pool_family = target.pool_family;
      upload.app_format = target.image.app_format;
      upload.record_call = record_call;
    }
    result.first_upload =
        Record(command_buffer, dev.device, destination, std::move(uploads),
               result.staging_allocated ? &staging : nullptr,
               range.arena_id != 0 ? &range : nullptr);
    result.recorded = true;
    host_copy(buffer, rewritten.data());
    return result;
  }

  // The copy hooks' common body, timed and counted.
  template <typename Region, typename HostCopy>
  void RecordCopy(const HostDevice& dev, VkCommandBuffer command_buffer,
                  const CopyTarget& target, VkBuffer source,
                  VkImage destination, std::uint32_t count,
                  const Region* regions, HostCopy host_copy) {
    const Clock::time_point start =
        options.frame_stats ? Clock::now() : Clock::time_point{};
    RouteChoice choice = ChooseRoute(dev, target);
    CpuCopyResult result;
    bool gpu = false;
    if (choice.route == Etc2Route::kGpu) {
      const GpuCopyResult recorded =
          RecordGpuCopy(dev, command_buffer, target, source, destination,
                        count, regions, host_copy);
      gpu = recorded.recorded;
      result.recorded = recorded.recorded;
      result.first_upload = recorded.first_upload;
      result.compressed = recorded.compressed;
      if (!gpu) {
        choice = {Etc2Route::kCpu, recorded.reason};
      }
    }
    if (!gpu) {
      result = RecordCpuCopy(dev, command_buffer, target, source, destination,
                             count, regions, host_copy);
    }
    if (!options.frame_stats) {
      return;
    }
    const std::uint64_t elapsed = Nanoseconds(Clock::now() - start);
    const std::uint32_t family = target.pool_family;
    const bool known_family = family < dev.queue_families.size();
    const bool compute =
        known_family &&
        (dev.queue_families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
    CountTotals([&](Etc2DecodeTotals& totals) {
      ++totals.record_calls;
      totals.record_uploads += count;
      totals.record_ns += elapsed;
      if (choice.reason != Etc2CpuReason::kCount) {
        totals.cpu_fallbacks[static_cast<std::size_t>(choice.reason)] += count;
      }
      if (result.staging_allocated) {
        ++totals.staging_allocs;
      }
      if (result.staging_fallback) {
        ++totals.staging_fallbacks;
      }
      if (!result.recorded) {
        totals.dropped_uploads += count;
      }
      if (!known_family) {
        totals.unknown_family_uploads += count;
      } else {
        const std::size_t slot =
            std::min<std::size_t>(family, kEtc2CountedQueueFamilies - 1);
        totals.family_uploads[slot] += count;
        totals.family_bytes[slot] += result.compressed;
        if (!compute) {
          totals.no_compute_uploads += count;
          totals.no_compute_bytes += result.compressed;
        }
      }
      if (result.first_upload &&
          (target.begin_flags &
           VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) != 0) {
        ++totals.simultaneous_upload_buffers;
      }
      if (target.dispatch_seen) {
        totals.uploads_after_dispatch += count;
      }
    });
  }

  // Keeps the GPU-route uploads and their output range with the command
  // buffer. True when these are the command buffer's first uploads.
  bool RecordGpu(VkCommandBuffer command_buffer, VkDevice device,
                 VkImage destination, const std::vector<Etc2GpuUpload>& uploads,
                 const ArenaRef& output) {
    std::lock_guard<std::mutex> lock(mutex);
    CommandRecord& record = commands[command_buffer];
    if (record.device == VK_NULL_HANDLE) {
      record.device = device;
    }
    const bool first = !record.HasUploads();
    record.gpu_uploads.insert(record.gpu_uploads.end(), uploads.begin(),
                              uploads.end());
    record.gpu_regions.push_back(output);
    if (options.frame_stats) {
      const auto image = images.find(destination);
      if (image != images.end()) {
        ++image->second.copies;
      }
    }
    has_commands.store(true, std::memory_order_release);
    return first;
  }

  // Keeps the uploads and their staging (a one-off buffer or an arena
  // range) with the command buffer. True when these are the command
  // buffer's first uploads.
  bool Record(VkCommandBuffer command_buffer, VkDevice device,
              VkImage destination, std::vector<PendingUpload> uploads,
              const Staging* staging, const ArenaRef* range) {
    std::lock_guard<std::mutex> lock(mutex);
    CommandRecord& record = commands[command_buffer];
    if (record.device == VK_NULL_HANDLE) {
      record.device = device;
    }
    const bool first = !record.HasUploads();
    record.uploads.insert(record.uploads.end(), uploads.begin(),
                          uploads.end());
    if (staging != nullptr) {
      record.staging.push_back(*staging);
    }
    if (range != nullptr) {
      record.staging_refs.push_back(*range);
    }
    if (options.frame_stats) {
      const auto image = images.find(destination);
      if (image != images.end()) {
        ++image->second.copies;
      }
    }
    has_commands.store(true, std::memory_order_release);
    return first;
  }

  // Gives back what a released command buffer held. Its device must still
  // be registered for one-off staging buffers to be destroyed.
  void ReleaseRecordResources(CommandRecord&& record) {
    for (const ArenaRef& ref : record.staging_refs) {
      ChunkArena::ReleaseRef(ref);
    }
    for (const ArenaRef& ref : record.gpu_regions) {
      ChunkArena::ReleaseRef(ref);
    }
    const HostDevice* dev = nullptr;
    for (const Staging& staging : record.staging) {
      if (dev == nullptr || dev->device != staging.device) {
        dev = Find(staging.device);
      }
      if (dev != nullptr) {
        DestroyStaging(*dev, staging);
      }
    }
  }

  // vkDestroyDevice: what the device's routes made, after its command
  // buffers' records were released. `dev` is no longer registered.
  void DestroyDeviceResources(HostDevice& dev) {
    std::unique_ptr<Etc2GpuDevice> gpu;
    std::unique_ptr<ChunkArena> host_staging;
    {
      std::lock_guard<std::mutex> lock(dev.mutex);
      gpu = std::move(dev.gpu);
      host_staging = std::move(dev.host_staging);
    }
    if (gpu != nullptr) {
      gpu->Destroy();
    }
    if (host_staging != nullptr) {
      host_staging->DestroyAll();
    }
  }

  // CreateImage, once per device: the first emulated image. Roblox's probe
  // device never creates one, so work started here never runs for it.
  void OnFirstEmulatedImage(HostDevice& dev) {
    if (dev.gpu != nullptr) {
      dev.gpu->StartBuild();
    }
  }

  // RegisterDevice of an emulated device with MOCKTAIL_ETC2_GPU on: the GPU
  // route's runtime, which builds nothing until the first emulated image.
  // A host without every entry point it needs is no real Vulkan driver;
  // the CPU route then serves the device, as with MOCKTAIL_ETC2_GPU=0.
  void MakeGpuDevice(HostDevice& dev) {
    Etc2GpuDeviceInfo info;
    info.device = dev.device;
    info.physical_device = dev.physical_device;
    Etc2GpuFunctions& fns = info.fns;
    fns.create_buffer = dev.create_buffer;
    fns.destroy_buffer = dev.destroy_buffer;
    fns.get_buffer_memory_requirements = dev.get_buffer_memory_requirements;
    fns.allocate_memory = dev.allocate_memory;
    fns.free_memory = dev.free_memory;
    fns.bind_buffer_memory = dev.bind_buffer_memory;
    fns.map_memory = dev.map_memory;
    fns.create_command_pool = dev.create_command_pool;
    fns.destroy_command_pool = dev.destroy_command_pool;
    fns.reset_command_pool = dev.reset_command_pool;
    fns.allocate_command_buffers = dev.allocate_command_buffers;
    fns.begin_command_buffer = dev.begin_command_buffer;
    fns.end_command_buffer = dev.end_command_buffer;
    fns.copy_buffer = dev.copy_buffer;
    fns.pipeline_barrier = dev.pipeline_barrier;
    fns.bind_pipeline = dev.bind_pipeline;
    fns.bind_descriptor_sets = dev.bind_descriptor_sets;
    fns.push_constants = dev.push_constants;
    fns.dispatch = dev.dispatch;
    fns.write_timestamp = dev.write_timestamp;
    fns.reset_query_pool = dev.reset_query_pool;
    fns.create_query_pool = dev.create_query_pool;
    fns.destroy_query_pool = dev.destroy_query_pool;
    fns.get_query_pool_results = dev.get_query_pool_results;
    fns.create_fence = dev.create_fence;
    fns.destroy_fence = dev.destroy_fence;
    fns.reset_fences = dev.reset_fences;
    fns.get_fence_status = dev.get_fence_status;
    fns.wait_for_fences = dev.wait_for_fences;
    fns.create_shader_module = dev.create_shader_module;
    fns.destroy_shader_module = dev.destroy_shader_module;
    fns.create_compute_pipelines = dev.create_compute_pipelines;
    fns.destroy_pipeline = dev.destroy_pipeline;
    fns.create_pipeline_layout = dev.create_pipeline_layout;
    fns.destroy_pipeline_layout = dev.destroy_pipeline_layout;
    fns.create_descriptor_set_layout = dev.create_descriptor_set_layout;
    fns.destroy_descriptor_set_layout = dev.destroy_descriptor_set_layout;
    fns.create_descriptor_pool = dev.create_descriptor_pool;
    fns.destroy_descriptor_pool = dev.destroy_descriptor_pool;
    fns.allocate_descriptor_sets = dev.allocate_descriptor_sets;
    fns.free_descriptor_sets = dev.free_descriptor_sets;
    fns.update_descriptor_sets = dev.update_descriptor_sets;
    if (fns.Missing() != nullptr) {
      return;
    }
    // [CR-5] The VRAM budget, when the physical device reports one.
    if (dev.get_memory_properties2 != nullptr &&
        dev.enumerate_device_extensions != nullptr &&
        dev.properties.apiVersion >= VK_API_VERSION_1_1) {
      std::uint32_t count = 0;
      dev.enumerate_device_extensions(dev.physical_device, nullptr, &count,
                                      nullptr);
      std::vector<VkExtensionProperties> extensions(count);
      if (count != 0 &&
          dev.enumerate_device_extensions(dev.physical_device, nullptr, &count,
                                          extensions.data()) == VK_SUCCESS) {
        for (const VkExtensionProperties& extension : extensions) {
          if (std::strcmp(extension.extensionName,
                          VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            fns.get_memory_properties2 = dev.get_memory_properties2;
          }
        }
      }
    }
    info.memory = dev.memory;
    info.properties = dev.properties;
    info.families = dev.queue_families;
    info.bc = dev.bc;
    info.arena_cap_bytes = options.gpu_arena_bytes;
    info.frame_stats = options.frame_stats;
    info.count = [this](const std::function<void(Etc2DecodeTotals&)>& update) {
      CountTotals([&update](Etc2DecodeTotals& totals) { update(totals); });
    };
    // The runtime is destroyed before `dev`, and calls this only from
    // inside a submit on the device. It runs with a queue's lock of the
    // runtime held, so it must not take devices_mutex (the totals' arena
    // peaks hold that one while they read the runtime's stats).
    info.rescue_decode = [this, host = &dev](
                             const std::vector<VkCommandBuffer>& buffers,
                             const std::vector<Etc2GpuUpload>& uploads,
                             std::vector<Etc2RescueSource>* sources) {
      return RescueDecode(*host, buffers, uploads, sources);
    };
    dev.gpu = Etc2GpuDevice::Create(std::move(info));
  }

  // Inside vkQueueSubmit(2), after the CPU route: the GPU route's decode
  // passes for every batch that holds GPU-route uploads (its command
  // buffers and their recorded secondaries), and the selftest. Marks those
  // batches in `upload_batches`.
  template <typename Submit>
  void PlanGpuSubmit(VkQueue queue, std::uint32_t count,
                     const Submit* submits, VkFence fence,
                     Etc2SubmitPlan* plan, std::vector<bool>* upload_batches) {
    if (plan == nullptr) {
      return;
    }
    QueueRecord queue_record;
    {
      std::lock_guard<std::mutex> lock(mutex);
      const auto found = queues.find(queue);
      if (found == queues.end()) {
        return;
      }
      queue_record = found->second;
    }
    HostDevice* dev = Find(queue_record.device);
    if (dev == nullptr || dev->gpu == nullptr) {
      return;
    }
    Etc2GpuDevice& gpu = *dev->gpu;
    plan->device = queue_record.device;
    plan->queue_family = queue_record.family;
    gpu.PollSelftest();
    gpu.MaybeRunSelftest(queue, queue_record.family, plan);
    if (submits == nullptr || !has_commands.load(std::memory_order_acquire)) {
      return;
    }
    std::vector<Etc2GpuBatch> batches;
    {
      std::lock_guard<std::mutex> lock(mutex);
      std::vector<VkCommandBuffer> buffers;
      for (std::uint32_t index = 0; index < count; ++index) {
        BatchCommandBuffers(submits[index], &buffers);
        Etc2GpuBatch batch;
        batch.batch = index;
        const auto collect =
            [&](VkCommandBuffer command_buffer) -> const CommandRecord* {
          const auto record = commands.find(command_buffer);
          if (record == commands.end()) {
            return nullptr;
          }
          batch.uploads.insert(batch.uploads.end(),
                               record->second.gpu_uploads.begin(),
                               record->second.gpu_uploads.end());
          return &record->second;
        };
        for (VkCommandBuffer command_buffer : buffers) {
          const CommandRecord* record = collect(command_buffer);
          if (record == nullptr) {
            continue;
          }
          const std::vector<VkCommandBuffer> secondaries = record->secondaries;
          for (VkCommandBuffer secondary : secondaries) {
            collect(secondary);
          }
        }
        if (!batch.uploads.empty()) {
          batch.command_buffers = buffers;
          if (upload_batches != nullptr && index < upload_batches->size()) {
            (*upload_batches)[index] = true;
          }
          batches.push_back(std::move(batch));
        }
      }
    }
    if (!batches.empty()) {
      PlanGpuPasses(gpu, queue, queue_record.family, count, submits, batches,
                    fence, plan);
    }
  }

  static void BatchCommandBuffers(const VkSubmitInfo& submit,
                                  std::vector<VkCommandBuffer>* out) {
    out->assign(submit.pCommandBuffers,
                submit.pCommandBuffers != nullptr
                    ? submit.pCommandBuffers + submit.commandBufferCount
                    : submit.pCommandBuffers);
  }
  static void BatchCommandBuffers(const VkSubmitInfo2& submit,
                                  std::vector<VkCommandBuffer>* out) {
    out->clear();
    for (std::uint32_t index = 0; index < submit.commandBufferInfoCount &&
                                  submit.pCommandBufferInfos != nullptr;
         ++index) {
      out->push_back(submit.pCommandBufferInfos[index].commandBuffer);
    }
  }
  static void PlanGpuPasses(Etc2GpuDevice& gpu, VkQueue queue,
                            std::uint32_t family, std::uint32_t count,
                            const VkSubmitInfo* submits,
                            const std::vector<Etc2GpuBatch>& batches,
                            VkFence fence, Etc2SubmitPlan* plan) {
    gpu.PlanSubmit(queue, family, count, submits, batches, fence, plan);
  }
  static void PlanGpuPasses(Etc2GpuDevice& gpu, VkQueue queue,
                            std::uint32_t family, std::uint32_t count,
                            const VkSubmitInfo2* submits,
                            const std::vector<Etc2GpuBatch>& batches,
                            VkFence fence, Etc2SubmitPlan* plan) {
    gpu.PlanSubmit2(queue, family, count, submits, batches, fence, plan);
  }

  // The size of every arena, for the peaks in the totals and the window.
  void NoteArenaBytes() {
    std::uint64_t bytes = 0;
    std::uint64_t over_cap = 0;
    {
      std::lock_guard<std::mutex> devices_lock(devices_mutex);
      for (const auto& dev : devices) {
        std::lock_guard<std::mutex> lock(dev->mutex);
        if (dev->host_staging != nullptr) {
          const ChunkArena::Stats stats = dev->host_staging->stats();
          bytes += stats.bytes;
          over_cap += stats.over_cap_bytes;
        }
        if (dev->gpu != nullptr) {
          const Etc2GpuDevice::Stats stats = dev->gpu->stats();
          bytes += stats.arena_bytes;
          over_cap += stats.arena_over_cap_bytes;
        }
      }
    }
    CountTotals([&](Etc2DecodeTotals& totals) {
      NoteEtc2ArenaBytes(&totals, bytes, over_cap);
    });
  }

  // Counts one vkQueueSubmit(2) call on a queue of an emulated device.
  void NoteSubmit(VkQueue queue, bool submit2, std::uint32_t batches,
                  std::uint32_t upload_batches) {
    if (!options.frame_stats) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (queues.count(queue) == 0) {
        return;
      }
    }
    std::lock_guard<std::mutex> lock(totals_mutex);
    // Which threads submit uploads: listed once each in the window line.
    if (upload_batches != 0 &&
        submit_thread_ids.insert(std::this_thread::get_id()).second) {
      std::string name = QuotedThreadName();
      if (submit_threads.insert(name).second) {
        new_submit_threads.push_back(std::move(name));
      }
    }
    for (Etc2DecodeTotals* counts : {&totals, &window}) {
      if (submit2) {
        ++counts->submit2_calls;
        counts->submit2_batches += batches;
      } else {
        ++counts->submit_calls;
        counts->submit_batches += batches;
      }
      counts->upload_batches += upload_batches;
      if (upload_batches > 1) {
        ++counts->multi_upload_calls;
      }
    }
  }

  // Remembers where an image of an emulated device is bound, and counts the
  // bind when it shares memory with a live image and either is emulated.
  void TrackImageBinding(const HostDevice& dev, VkImage image,
                         VkDeviceMemory memory, VkDeviceSize offset) {
    if (memory == VK_NULL_HANDLE ||
        dev.get_image_memory_requirements == nullptr) {
      return;
    }
    VkMemoryRequirements requirements{};
    dev.get_image_memory_requirements(dev.device, image, &requirements);
    const VkDeviceSize end = offset + requirements.size;
    bool aliased = false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      const bool emulated = images.count(image) != 0;
      MemoryRecord& record = memories[memory];
      record.device = dev.device;
      const auto counts = [&](const BoundImage& other) {
        return other.image != image && (emulated || other.emulated);
      };
      // Images that start inside the new range, and the one right before
      // it, which may reach into it. (Behind an alias that already exists
      // a longer one can hide: the count is then a lower bound.)
      auto bound = record.images.lower_bound(offset);
      if (bound != record.images.begin()) {
        const auto& before = std::prev(bound)->second;
        aliased = before.end > offset && counts(before);
      }
      for (; !aliased && bound != record.images.end() && bound->first < end;
           ++bound) {
        aliased = counts(bound->second);
      }
      record.images.emplace(offset, BoundImage{image, end, emulated});
      image_bindings[image] = {dev.device, memory, offset};
    }
    if (aliased) {
      CountDiagnostics([](Etc2DecodeTotals& totals) { ++totals.aliased_binds; });
    }
  }

  // Under `mutex`.
  void UntrackImageBinding(VkImage image) {
    const auto binding = image_bindings.find(image);
    if (binding == image_bindings.end()) {
      return;
    }
    const auto record = memories.find(binding->second.memory);
    if (record != memories.end()) {
      auto range = record->second.images.equal_range(binding->second.offset);
      for (auto bound = range.first; bound != range.second; ++bound) {
        if (bound->second.image == image) {
          record->second.images.erase(bound);
          break;
        }
      }
    }
    image_bindings.erase(binding);
  }

  // One upload of a submit as the CPU route serves it.
  struct CpuWork {
    PendingUpload upload;
    std::uint8_t* source = nullptr;  // mapped by Roblox, or null
    VkDeviceMemory memory = VK_NULL_HANDLE;  // to map for the submit
    VkDeviceSize memory_offset = 0;
    std::uint32_t memory_type = UINT32_MAX;  // diagnostics and trace only
    bool repeat = false;  // its command buffer came up earlier in the batch
  };
  // What a CPU pass logs and counts about its source memory.
  struct SourceTypes {
    std::uint64_t bytes[kEtc2CountedMemoryTypes] = {};
    std::uint64_t unknown = 0;
  };

  unsigned WorkerCount() const {
    return options.workers != 0 ? options.workers : DecodeWorkerCount();
  }
  void CollectCpuWork(const VkCommandBuffer* command_buffers,
                      std::uint32_t count, std::vector<CpuWork>* work);
  SourceTypes CountSourceTypes(const std::vector<CpuWork>& work);
  std::uint32_t RunCpuUploads(const VkCommandBuffer* command_buffers,
                              std::uint32_t count);
  std::uint32_t RunCpuUploadsFast(std::vector<CpuWork>* work,
                                  const SourceTypes& types);
  // A CPU pass's stall line, totals and milestone.
  void FinishCpuPass(std::size_t uploads, VkDeviceSize compressed_total,
                     VkDeviceSize host_total, std::uint64_t decode_ns,
                     const std::uint64_t (&phase_ms)[4], unsigned threads,
                     const SourceTypes& types);
  // Upscaled uploads: resample from the image's base level, or from these
  // decoded texels. Thread-safe; uploads of one image must come in order.
  void EmitScaled(const PendingUpload& upload, const std::uint8_t* decoded);
  // At most once a second, from OnPresent and the CPU route: trims the
  // host staging arenas and the GPU route's output arenas, and gives back
  // idle scratch.
  void Housekeeping(Clock::time_point now);
  // The GPU route's rescue, its CPU half (DESIGN §6, CR-7): decodes the
  // GPU-route uploads of one batch of `dev` into host staging, in the
  // output arena's layout, for the rescue pass to copy into the arena.
  bool RescueDecode(const HostDevice& dev,
                    const std::vector<VkCommandBuffer>& command_buffers,
                    const std::vector<Etc2GpuUpload>& uploads,
                    std::vector<Etc2RescueSource>* sources);
};

VulkanEtc2Emulation::VulkanEtc2Emulation(Etc2EmulationOptions options)
    : state_(new State(std::move(options))) {}

VulkanEtc2Emulation::~VulkanEtc2Emulation() { delete state_; }

const Etc2EmulationOptions& VulkanEtc2Emulation::options() const {
  return state_->options;
}

bool VulkanEtc2Emulation::PhysicalDeviceNeedsEmulation(
    VkPhysicalDevice physical_device, PFN_vkGetPhysicalDeviceFeatures host) {
  if (EmulationDisabledByEnvironment() || host == nullptr ||
      physical_device == VK_NULL_HANDLE) {
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto found = state_->physical_devices.find(physical_device);
    if (found != state_->physical_devices.end()) {
      return found->second;
    }
  }
  VkPhysicalDeviceFeatures features{};
  host(physical_device, &features);
  const bool needed = features.textureCompressionETC2 != VK_TRUE;
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->physical_devices[physical_device] = needed;
  return needed;
}

bool VulkanEtc2Emulation::PhysicalDeviceUsesBc(
    VkPhysicalDevice physical_device,
    PFN_vkGetPhysicalDeviceFeatures get_features,
    PFN_vkGetPhysicalDeviceFormatProperties get_format_properties) {
  if (!state_->options.to_bc || physical_device == VK_NULL_HANDLE) {
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto found = state_->bc_physical_devices.find(physical_device);
    if (found != state_->bc_physical_devices.end()) {
      return found->second;
    }
  }
  char reason[160] = "";
  if (EmulationDisabledByEnvironment()) {
    std::snprintf(reason, sizeof(reason),
                  "ETC2 emulation is off (MOCKTAIL_DISABLE_ETC2_EMULATION)");
  } else if (get_features == nullptr || get_format_properties == nullptr) {
    std::snprintf(reason, sizeof(reason),
                  "the device's features cannot be queried");
  } else if (!PhysicalDeviceNeedsEmulation(physical_device, get_features)) {
    std::snprintf(reason, sizeof(reason),
                  "not needed, the graphics card reads ETC2 itself");
  } else {
    VkPhysicalDeviceFeatures features{};
    get_features(physical_device, &features);
    if (features.textureCompressionBC != VK_TRUE) {
      std::snprintf(reason, sizeof(reason),
                    "the device has no textureCompressionBC");
    }
    // Every BC host format, in the order of the ETC2/EAC formats.
    for (int format = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
         reason[0] == '\0' && format <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK;
         ++format) {
      Etc2EmulatedFormat mapped;
      if (!LookupEmulatedEtc2Format(static_cast<VkFormat>(format), true,
                                    &mapped)) {
        continue;
      }
      VkFormatProperties properties{};
      get_format_properties(physical_device, mapped.host_format, &properties);
      const VkFormatFeatureFlags missing =
          Etc2BcRequiredFormatFeatures() & ~properties.optimalTilingFeatures;
      if (missing != 0) {
        std::snprintf(reason, sizeof(reason),
                      "format %d lacks optimal-tiling features 0x%x",
                      static_cast<int>(mapped.host_format),
                      static_cast<unsigned>(missing));
      }
    }
  }
  const bool usable = reason[0] == '\0';
  bool decided = false;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto inserted =
        state_->bc_physical_devices.emplace(physical_device, usable);
    decided = inserted.second;
    if (!decided) {
      return inserted.first->second;
    }
  }
  if (!usable) {
    std::fprintf(stderr,
                 "  [vulkan] ETC2->BC unavailable: %s; ETC2 textures stay "
                 "RGBA8/R16\n",
                 reason);
  }
  return usable;
}

void VulkanEtc2Emulation::RegisterDevice(
    VkDevice device, VkPhysicalDevice physical_device, bool emulated,
    const VkPhysicalDeviceMemoryProperties& memory,
    const VkPhysicalDeviceProperties& properties,
    const std::vector<VkQueueFamilyProperties>& queue_families,
    const VkDeviceCreateInfo& create_info,
    PFN_vkGetDeviceProcAddr get_device_proc_addr,
    PFN_vkGetInstanceProcAddr get_instance_proc_addr, VkInstance instance) {
  if (device == VK_NULL_HANDLE || get_device_proc_addr == nullptr) {
    return;
  }
  auto dev = std::make_unique<HostDevice>();
  dev->device = device;
  dev->physical_device = physical_device;
  dev->emulated = emulated;
  dev->memory = memory;
  dev->properties = properties;
  dev->queue_families = queue_families;
  const auto resolve = [&](auto& field, const char* name,
                           const char* alias = nullptr) {
    field = DeviceProc<std::remove_reference_t<decltype(field)>>(
        get_device_proc_addr, device, name, alias);
  };
  resolve(dev->create_image, "vkCreateImage");
  resolve(dev->destroy_image, "vkDestroyImage");
  resolve(dev->create_image_view, "vkCreateImageView");
  resolve(dev->bind_buffer_memory, "vkBindBufferMemory");
  resolve(dev->bind_buffer_memory2, "vkBindBufferMemory2",
          "vkBindBufferMemory2KHR");
  resolve(dev->bind_image_memory, "vkBindImageMemory");
  resolve(dev->bind_image_memory2, "vkBindImageMemory2",
          "vkBindImageMemory2KHR");
  resolve(dev->map_memory, "vkMapMemory");
  resolve(dev->map_memory2, "vkMapMemory2", "vkMapMemory2KHR");
  resolve(dev->unmap_memory, "vkUnmapMemory");
  resolve(dev->unmap_memory2, "vkUnmapMemory2", "vkUnmapMemory2KHR");
  resolve(dev->flush_mapped_memory_ranges, "vkFlushMappedMemoryRanges");
  resolve(dev->invalidate_mapped_memory_ranges,
          "vkInvalidateMappedMemoryRanges");
  resolve(dev->allocate_memory, "vkAllocateMemory");
  resolve(dev->free_memory, "vkFreeMemory");
  resolve(dev->create_buffer, "vkCreateBuffer");
  resolve(dev->destroy_buffer, "vkDestroyBuffer");
  resolve(dev->get_buffer_memory_requirements,
          "vkGetBufferMemoryRequirements");
  resolve(dev->get_image_memory_requirements, "vkGetImageMemoryRequirements");
  resolve(dev->copy_buffer_to_image, "vkCmdCopyBufferToImage");
  resolve(dev->copy_buffer_to_image2, "vkCmdCopyBufferToImage2",
          "vkCmdCopyBufferToImage2KHR");
  resolve(dev->copy_image, "vkCmdCopyImage");
  resolve(dev->copy_image2, "vkCmdCopyImage2", "vkCmdCopyImage2KHR");
  resolve(dev->blit_image, "vkCmdBlitImage");
  resolve(dev->execute_commands, "vkCmdExecuteCommands");
  resolve(dev->create_command_pool, "vkCreateCommandPool");
  resolve(dev->destroy_command_pool, "vkDestroyCommandPool");
  resolve(dev->reset_command_pool, "vkResetCommandPool");
  resolve(dev->allocate_command_buffers, "vkAllocateCommandBuffers");
  resolve(dev->free_command_buffers, "vkFreeCommandBuffers");
  resolve(dev->begin_command_buffer, "vkBeginCommandBuffer");
  resolve(dev->end_command_buffer, "vkEndCommandBuffer");
  resolve(dev->copy_buffer, "vkCmdCopyBuffer");
  resolve(dev->pipeline_barrier, "vkCmdPipelineBarrier");
  resolve(dev->bind_pipeline, "vkCmdBindPipeline");
  resolve(dev->bind_descriptor_sets, "vkCmdBindDescriptorSets");
  resolve(dev->push_constants, "vkCmdPushConstants");
  resolve(dev->dispatch, "vkCmdDispatch");
  resolve(dev->write_timestamp, "vkCmdWriteTimestamp");
  resolve(dev->reset_query_pool, "vkCmdResetQueryPool");
  resolve(dev->create_query_pool, "vkCreateQueryPool");
  resolve(dev->destroy_query_pool, "vkDestroyQueryPool");
  resolve(dev->get_query_pool_results, "vkGetQueryPoolResults");
  resolve(dev->create_fence, "vkCreateFence");
  resolve(dev->destroy_fence, "vkDestroyFence");
  resolve(dev->reset_fences, "vkResetFences");
  resolve(dev->get_fence_status, "vkGetFenceStatus");
  resolve(dev->wait_for_fences, "vkWaitForFences");
  resolve(dev->create_shader_module, "vkCreateShaderModule");
  resolve(dev->destroy_shader_module, "vkDestroyShaderModule");
  resolve(dev->create_compute_pipelines, "vkCreateComputePipelines");
  resolve(dev->destroy_pipeline, "vkDestroyPipeline");
  resolve(dev->create_pipeline_layout, "vkCreatePipelineLayout");
  resolve(dev->destroy_pipeline_layout, "vkDestroyPipelineLayout");
  resolve(dev->create_descriptor_set_layout, "vkCreateDescriptorSetLayout");
  resolve(dev->destroy_descriptor_set_layout, "vkDestroyDescriptorSetLayout");
  resolve(dev->create_descriptor_pool, "vkCreateDescriptorPool");
  resolve(dev->destroy_descriptor_pool, "vkDestroyDescriptorPool");
  resolve(dev->allocate_descriptor_sets, "vkAllocateDescriptorSets");
  resolve(dev->free_descriptor_sets, "vkFreeDescriptorSets");
  resolve(dev->update_descriptor_sets, "vkUpdateDescriptorSets");
  if (get_instance_proc_addr != nullptr && instance != VK_NULL_HANDLE) {
    const auto instance_proc = [&](auto& field, const char* name,
                                   const char* alias = nullptr) {
      PFN_vkVoidFunction proc = get_instance_proc_addr(instance, name);
      if (proc == nullptr && alias != nullptr) {
        proc = get_instance_proc_addr(instance, alias);
      }
      field = reinterpret_cast<std::remove_reference_t<decltype(field)>>(proc);
    };
    instance_proc(dev->get_features, "vkGetPhysicalDeviceFeatures");
    instance_proc(dev->get_format_properties,
                  "vkGetPhysicalDeviceFormatProperties");
    instance_proc(dev->get_image_format_properties,
                  "vkGetPhysicalDeviceImageFormatProperties");
    instance_proc(dev->get_memory_properties2,
                  "vkGetPhysicalDeviceMemoryProperties2",
                  "vkGetPhysicalDeviceMemoryProperties2KHR");
    instance_proc(dev->enumerate_device_extensions,
                  "vkEnumerateDeviceExtensionProperties");
  }
  if (emulated && !Etc2EmulationTesting().no_host_staging.load()) {
    // The CPU route's staging (DESIGN C1). Chunks come on first use.
    const HostDevice* const host = dev.get();
    const Etc2StagingMemory kind = state_->options.staging_memory;
    const bool log = state_->options.frame_stats;
    ChunkArena::Config config;
    config.chunk_bytes = kHostStagingChunkBytes;
    config.keep_idle_bytes = kHostStagingKeepIdleBytes;
    ChunkArena::Backend backend;
    backend.create = [host, kind, log](VkDeviceSize size, bool,
                                       ArenaChunkStorage* storage) {
      return CreateStagingChunk(*host, kind,
                                log && ShouldLog(&g_chunk_logs, 16), size,
                                storage);
    };
    backend.destroy = [host](const ArenaChunkStorage& storage) {
      host->destroy_buffer(host->device, storage.buffer, nullptr);
      host->free_memory(host->device, storage.memory, nullptr);
    };
    dev->host_staging =
        std::make_unique<ChunkArena>(config, std::move(backend));
  }
  if (emulated && state_->options.to_bc) {
    // Decided before vkCreateDevice already, which enabled
    // textureCompressionBC then (the adapter); the same answer here.
    dev->bc = PhysicalDeviceUsesBc(physical_device, dev->get_features,
                                   dev->get_format_properties);
    if (dev->bc && dev->get_image_format_properties == nullptr) {
      dev->bc = false;
      std::fprintf(stderr,
                   "  [vulkan] ETC2->BC unavailable: the device's image "
                   "formats cannot be queried; ETC2 textures stay "
                   "RGBA8/R16\n");
    }
  }
  if (emulated && state_->options.gpu) {
    // After the BC decision: in ETC2->BC mode the GPU route builds the
    // etc2_bc pipeline too.
    state_->MakeGpuDevice(*dev);
  }
  if (emulated) {
    if (dev->bc) {
      // The GPU route transcodes once its selftest passed; the CPU route
      // until then, and wherever the GPU route cannot serve an upload.
      std::fprintf(stderr,
                   "  [vulkan] ETC2/EAC emulation enabled: compressed uploads "
                   "become BC1/BC3/BC4/BC5 host images (%s transcode)\n",
                   dev->gpu != nullptr &&
                           dev->gpu->state() != Etc2GpuDevice::State::kFailed
                       ? "GPU"
                       : "CPU");
    } else {
      std::fprintf(stderr,
                   "  [vulkan] ETC2/EAC emulation enabled: compressed uploads "
                   "decode to RGBA8/R16 host images\n");
    }
    std::fprintf(stderr, "  [vulkan] ETC2 queues: %s\n",
                 DescribeEtc2Queues(create_info, queue_families).c_str());
  }
  std::lock_guard<std::mutex> lock(state_->devices_mutex);
  state_->devices.erase(
      std::remove_if(state_->devices.begin(), state_->devices.end(),
                     [device](const std::unique_ptr<HostDevice>& candidate) {
                       return candidate->device == device;
                     }),
      state_->devices.end());
  state_->devices.push_back(std::move(dev));
}

void VulkanEtc2Emulation::DestroyDevice(VkDevice device) {
  if (const HostDevice* dev = state_->Find(device);
      dev != nullptr && dev->emulated) {
    state_->LogTotals("");
  }
  // Every record of the device goes: the loader hands its command buffer
  // handles out again once the device's pools are gone (DESIGN CR-3).
  std::vector<CommandRecord> doomed;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (auto it = state_->commands.begin(); it != state_->commands.end();) {
      const bool owned =
          it->second.device == device ||
          std::any_of(it->second.staging.begin(), it->second.staging.end(),
                      [device](const Staging& staging) {
                        return staging.device == device;
                      });
      if (owned) {
        doomed.push_back(std::move(it->second));
        it = state_->commands.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = state_->images.begin(); it != state_->images.end();) {
      it = it->second.device == device ? state_->images.erase(it) : std::next(it);
    }
    for (auto it = state_->pools.begin(); it != state_->pools.end();) {
      it = it->second.device == device ? state_->pools.erase(it) : std::next(it);
    }
    for (auto it = state_->queues.begin(); it != state_->queues.end();) {
      it = it->second.device == device ? state_->queues.erase(it) : std::next(it);
    }
    for (auto it = state_->memories.begin(); it != state_->memories.end();) {
      it = it->second.device == device ? state_->memories.erase(it)
                                       : std::next(it);
    }
    for (auto it = state_->image_bindings.begin();
         it != state_->image_bindings.end();) {
      it = it->second.device == device ? state_->image_bindings.erase(it)
                                       : std::next(it);
    }
    state_->has_commands.store(!state_->commands.empty(),
                               std::memory_order_release);
  }
  for (CommandRecord& record : doomed) {
    state_->ReleaseRecordResources(std::move(record));
  }
  std::unique_ptr<HostDevice> removed;
  {
    std::lock_guard<std::mutex> lock(state_->devices_mutex);
    for (auto it = state_->devices.begin(); it != state_->devices.end();
         ++it) {
      if ((*it)->device == device) {
        removed = std::move(*it);
        state_->devices.erase(it);
        break;
      }
    }
  }
  if (removed == nullptr) {
    return;
  }
  state_->DestroyDeviceResources(*removed);
  if (!removed->emulated) {
    return;
  }
  bool emulated_left = false;
  {
    std::lock_guard<std::mutex> lock(state_->devices_mutex);
    for (const auto& dev : state_->devices) {
      emulated_left = emulated_left || dev->emulated;
    }
  }
  if (!emulated_left) {
    // Nothing decodes until the next emulated device: the pool's threads
    // and the scratch go (the pool starts again on first use).
    WorkerPool::ShutdownShared();
    std::lock_guard<std::mutex> lock(state_->scratch_mutex);
    state_->fast_source.Release();
    state_->fast_decoded.Release();
  }
}

void VulkanEtc2Emulation::RegisterQueue(VkDevice device, VkQueue queue,
                                        std::uint32_t family,
                                        std::uint32_t index) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || !dev->emulated || queue == VK_NULL_HANDLE) {
    return;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->queues[queue] = {device, family, index};
}

void VulkanEtc2Emulation::RegisterCommandPool(VkDevice device,
                                              VkCommandPool pool,
                                              std::uint32_t family,
                                              VkCommandPoolCreateFlags flags) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || !dev->emulated || pool == VK_NULL_HANDLE) {
    return;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->pools[pool] = {device, family, flags};
}

void VulkanEtc2Emulation::ForgetCommandPool(VkDevice device,
                                            VkCommandPool pool) {
  std::lock_guard<std::mutex> lock(state_->mutex);
  const auto found = state_->pools.find(pool);
  if (found != state_->pools.end() && found->second.device == device) {
    state_->pools.erase(found);
  }
}

void VulkanEtc2Emulation::BeginCommandBuffer(
    VkCommandBuffer command_buffer, VkCommandPool pool,
    VkCommandBufferUsageFlags flags) {
  ReleaseCommandBuffer(command_buffer);
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto found = state_->pools.find(pool);
    if (found == state_->pools.end()) {
      return;  // not a pool of an emulated device
    }
    CommandRecord& record = state_->commands[command_buffer];
    record = CommandRecord{};
    record.device = found->second.device;
    record.pool_family = found->second.family;
    record.begin_flags = flags;
    state_->has_commands.store(true, std::memory_order_release);
  }
  state_->CountDiagnostics([flags](Etc2DecodeTotals& totals) {
    if ((flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) != 0) {
      ++totals.begin_simultaneous;
    } else if ((flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT) != 0) {
      ++totals.begin_one_time;
    } else {
      ++totals.begin_other;
    }
  });
}

VkResult VulkanEtc2Emulation::CreateImage(VkDevice device,
                                          const VkImageCreateInfo* create_info,
                                          const VkAllocationCallbacks* allocator,
                                          VkImage* image) {
  HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->create_image == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  Etc2EmulatedFormat format;
  if (!dev->emulated || create_info == nullptr ||
      !LookupEmulatedEtc2Format(create_info->format, &format)) {
    return dev->create_image(device, create_info, allocator, image);
  }
  VkImageCreateInfo host_info = *create_info;
  host_info.flags &= ~static_cast<VkImageCreateFlags>(
      VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT);
  // ETC2->BC mode stores the blocks as BC unless the host cannot take this
  // image as BC (CR-14); such an image keeps the RGBA8/R16 host.
  const char* bc_refusal =
      dev->bc ? state_->BcImageRefusal(*dev, *create_info, host_info.flags,
                                       format.bc_host_format)
              : nullptr;
  const bool bc = dev->bc && bc_refusal == nullptr;
  if (bc) {
    format.host_format = format.bc_host_format;
  }
  host_info.format = format.host_format;
  std::uint32_t scale = 1;
  const std::uint32_t upscale = state_->options.upscale;
  // BC images have no BLIT_DST to upscale through, and a BC-mode device
  // upscales none, so that copies between its images stay plain copies.
  if (!dev->bc && upscale > 1 && IsColorFormat(format.etc_format) &&
      create_info->imageType == VK_IMAGE_TYPE_2D &&
      create_info->arrayLayers == 1 && create_info->extent.depth == 1 &&
      create_info->extent.width <= kSmallTextureMaxExtent &&
      create_info->extent.height <= kSmallTextureMaxExtent) {
    scale = upscale;
    host_info.extent.width *= scale;
    host_info.extent.height *= scale;
  }
  VkResult result = VK_SUCCESS;
  bool format_list = false;
  {
    // [CR-15] The host sees its own formats in a VkImageFormatListCreateInfo.
    const Etc2FormatListRewrite rewrite(&host_info.pNext, bc);
    format_list = rewrite.found();
    result = dev->create_image(device, &host_info, allocator, image);
  }
  if (result == VK_SUCCESS && image != nullptr) {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      ImageRecord& record = state_->images[*image];
      record = ImageRecord{};
      record.device = device;
      record.format = format;
      record.scale = scale;
      record.width = create_info->extent.width;
      record.height = create_info->extent.height;
      record.app_format = create_info->format;
      record.bc = bc;
    }
    state_->CountTotals([&](Etc2DecodeTotals& totals) {
      ++totals.images;
      if (scale > 1) {
        ++totals.upscaled_images;
      }
      if (bc) {
        ++totals.bc_images;
      } else if (dev->bc) {
        ++totals.bc_fallback_images;
      }
      if (format_list && state_->options.frame_stats) {
        ++totals.format_lists;
      }
    });
    if (bc_refusal != nullptr && ShouldLog(&g_bc_fallback_logs, 4)) {
      std::fprintf(stderr,
                   "  [vulkan] ETC2->BC: a %ux%u image (format %d, usage "
                   "0x%x, flags 0x%x) stays RGBA8/R16: %s\n",
                   create_info->extent.width, create_info->extent.height,
                   static_cast<int>(create_info->format),
                   static_cast<unsigned>(create_info->usage),
                   static_cast<unsigned>(create_info->flags), bc_refusal);
    }
    if (!dev->saw_emulated_image.exchange(true)) {
      state_->OnFirstEmulatedImage(*dev);
    }
    if (ShouldLog(&g_image_logs, 4)) {
      std::fprintf(stderr,
                   "  [vulkan] ETC2 image %ux%u mips=%u format=%d -> host "
                   "format=%d\n",
                   create_info->extent.width, create_info->extent.height,
                   create_info->mipLevels,
                   static_cast<int>(create_info->format),
                   static_cast<int>(format.host_format));
    }
  }
  return result;
}

void VulkanEtc2Emulation::DestroyImage(VkDevice device, VkImage image,
                                       const VkAllocationCallbacks* allocator) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->destroy_image == nullptr) {
    return;
  }
  if (dev->emulated) {
    bool emulated = false;
    std::uint32_t copies = 0;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      const auto found = state_->images.find(image);
      if (found != state_->images.end()) {
        emulated = true;
        copies = found->second.copies;
        state_->images.erase(found);
      }
      if (state_->options.frame_stats) {
        state_->UntrackImageBinding(image);
      }
    }
    if (emulated) {
      state_->CountDiagnostics([copies](Etc2DecodeTotals& totals) {
        AddEtc2DestroyedImage(&totals, copies);
      });
    }
  }
  dev->destroy_image(device, image, allocator);
}

VkResult VulkanEtc2Emulation::CreateImageView(
    VkDevice device, const VkImageViewCreateInfo* create_info,
    const VkAllocationCallbacks* allocator, VkImageView* view) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->create_image_view == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  Etc2EmulatedFormat format;
  if (!dev->emulated || create_info == nullptr ||
      !LookupEmulatedEtc2Format(create_info->format, &format)) {
    if (dev->emulated && create_info != nullptr &&
        state_->options.frame_stats) {
      // A view that reads an emulated image's host texels as another format.
      bool alias = false;
      {
        std::lock_guard<std::mutex> lock(state_->mutex);
        alias = state_->images.count(create_info->image) != 0;
      }
      if (alias) {
        state_->CountDiagnostics(
            [](Etc2DecodeTotals& totals) { ++totals.alias_views; });
      }
    }
    return dev->create_image_view(device, create_info, allocator, view);
  }
  VkImageViewCreateInfo host_info = *create_info;
  host_info.format = format.host_format;
  if (dev->bc) {
    // The view takes the image's host: BC, or RGBA8/R16 for an image BC
    // could not serve.
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto image = state_->images.find(create_info->image);
    if (image != state_->images.end() && image->second.bc) {
      host_info.format = format.bc_host_format;
    }
  }
  return dev->create_image_view(device, &host_info, allocator, view);
}

VkResult VulkanEtc2Emulation::AllocateMemory(
    VkDevice device, const VkMemoryAllocateInfo* info,
    const VkAllocationCallbacks* allocator, VkDeviceMemory* memory) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->allocate_memory == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  const VkResult result = dev->allocate_memory(device, info, allocator, memory);
  if (result == VK_SUCCESS && dev->emulated && info != nullptr &&
      memory != nullptr) {
    NoteAllocateMemory(device, *memory, info->memoryTypeIndex,
                       info->allocationSize);
  }
  return result;
}

void VulkanEtc2Emulation::NoteAllocateMemory(VkDevice device,
                                             VkDeviceMemory memory,
                                             std::uint32_t type_index,
                                             VkDeviceSize size) {
  static_cast<void>(size);
  if (!state_->TrackMemory() || memory == VK_NULL_HANDLE) {
    return;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  MemoryRecord& record = state_->memories[memory];
  record = MemoryRecord{};
  record.device = device;
  record.type = type_index;
}

VkResult VulkanEtc2Emulation::BindBufferMemory(VkDevice device, VkBuffer buffer,
                                               VkDeviceMemory memory,
                                               VkDeviceSize offset) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->bind_buffer_memory == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  const VkResult result =
      dev->bind_buffer_memory(device, buffer, memory, offset);
  if (result == VK_SUCCESS && dev->emulated) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->buffers[buffer] = {memory, offset};
  }
  return result;
}

VkResult VulkanEtc2Emulation::BindBufferMemory2(
    VkDevice device, std::uint32_t count, const VkBindBufferMemoryInfo* infos) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->bind_buffer_memory2 == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  const VkResult result = dev->bind_buffer_memory2(device, count, infos);
  if (result == VK_SUCCESS && dev->emulated && infos != nullptr) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (std::uint32_t index = 0; index < count; ++index) {
      state_->buffers[infos[index].buffer] = {infos[index].memory,
                                              infos[index].memoryOffset};
    }
  }
  return result;
}

VkResult VulkanEtc2Emulation::BindImageMemory(VkDevice device, VkImage image,
                                              VkDeviceMemory memory,
                                              VkDeviceSize offset) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->bind_image_memory == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  const VkResult result = dev->bind_image_memory(device, image, memory, offset);
  if (result == VK_SUCCESS && dev->emulated && state_->options.frame_stats) {
    state_->TrackImageBinding(*dev, image, memory, offset);
  }
  return result;
}

VkResult VulkanEtc2Emulation::BindImageMemory2(
    VkDevice device, std::uint32_t count, const VkBindImageMemoryInfo* infos) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->bind_image_memory2 == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  const VkResult result = dev->bind_image_memory2(device, count, infos);
  if (result == VK_SUCCESS && dev->emulated && infos != nullptr &&
      state_->options.frame_stats) {
    for (std::uint32_t index = 0; index < count; ++index) {
      state_->TrackImageBinding(*dev, infos[index].image, infos[index].memory,
                                infos[index].memoryOffset);
    }
  }
  return result;
}

VkResult VulkanEtc2Emulation::MapMemory(VkDevice device, VkDeviceMemory memory,
                                        VkDeviceSize offset, VkDeviceSize size,
                                        VkMemoryMapFlags flags, void** data) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->map_memory == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  const VkResult result =
      dev->map_memory(device, memory, offset, size, flags, data);
  if (result == VK_SUCCESS && dev->emulated && data != nullptr) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->mappings[memory] = {*data, offset, size};
  }
  return result;
}

VkResult VulkanEtc2Emulation::MapMemory2(VkDevice device,
                                         const VkMemoryMapInfo* info,
                                         void** data) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->map_memory2 == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  const VkResult result = dev->map_memory2(device, info, data);
  if (result == VK_SUCCESS && dev->emulated && info != nullptr &&
      data != nullptr) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->mappings[info->memory] = {*data, info->offset, info->size};
  }
  return result;
}

void VulkanEtc2Emulation::UnmapMemory(VkDevice device, VkDeviceMemory memory) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->unmap_memory == nullptr) {
    return;
  }
  if (dev->emulated) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->mappings.erase(memory);
  }
  dev->unmap_memory(device, memory);
}

VkResult VulkanEtc2Emulation::UnmapMemory2(VkDevice device,
                                           const VkMemoryUnmapInfo* info) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->unmap_memory2 == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (dev->emulated && info != nullptr) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->mappings.erase(info->memory);
  }
  return dev->unmap_memory2(device, info);
}

void VulkanEtc2Emulation::FreeMemory(VkDevice device, VkDeviceMemory memory,
                                     const VkAllocationCallbacks* allocator) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->free_memory == nullptr) {
    return;
  }
  if (dev->emulated) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->mappings.erase(memory);
    state_->memories.erase(memory);
  }
  dev->free_memory(device, memory, allocator);
}

void VulkanEtc2Emulation::DestroyBuffer(VkDevice device, VkBuffer buffer,
                                        const VkAllocationCallbacks* allocator) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->destroy_buffer == nullptr) {
    return;
  }
  if (dev->emulated) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->buffers.erase(buffer);
  }
  dev->destroy_buffer(device, buffer, allocator);
}

void VulkanEtc2Emulation::CmdCopyBufferToImage(
    VkDevice device, VkCommandBuffer command_buffer, VkBuffer source,
    VkImage destination, VkImageLayout layout, std::uint32_t region_count,
    const VkBufferImageCopy* regions) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->copy_buffer_to_image == nullptr) {
    return;
  }
  CopyTarget target;
  if (!state_->LookupCopy(*dev, command_buffer, destination, &target)) {
    dev->copy_buffer_to_image(command_buffer, source, destination, layout,
                              region_count, regions);
    return;
  }
  state_->RecordCopy(
      *dev, command_buffer, target, source, destination, region_count,
      regions, [&](VkBuffer buffer, const VkBufferImageCopy* rewritten) {
        dev->copy_buffer_to_image(command_buffer, buffer, destination, layout,
                                  region_count, rewritten);
      });
}

void VulkanEtc2Emulation::CmdCopyBufferToImage2(
    VkDevice device, VkCommandBuffer command_buffer,
    const VkCopyBufferToImageInfo2* info) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->copy_buffer_to_image2 == nullptr ||
      info == nullptr) {
    return;
  }
  CopyTarget target;
  if (!state_->LookupCopy(*dev, command_buffer, info->dstImage, &target)) {
    dev->copy_buffer_to_image2(command_buffer, info);
    return;
  }
  state_->RecordCopy(
      *dev, command_buffer, target, info->srcBuffer, info->dstImage,
      info->regionCount, info->pRegions,
      [&](VkBuffer buffer, const VkBufferImageCopy2* rewritten) {
        VkCopyBufferToImageInfo2 host_info = *info;
        host_info.srcBuffer = buffer;
        host_info.pRegions = rewritten;
        dev->copy_buffer_to_image2(command_buffer, &host_info);
      });
}

bool VulkanEtc2Emulation::State::BlitScaledCopy(
    const HostDevice& dev, VkCommandBuffer command_buffer, VkImage source,
    VkImageLayout source_layout, VkImage destination,
    VkImageLayout destination_layout, std::uint32_t count,
    const VkImageCopy* regions) {
  ImageRecord source_record;
  ImageRecord destination_record;
  const bool source_scaled =
      LookupImage(dev, source, &source_record) && source_record.scale > 1;
  const bool destination_scaled =
      LookupImage(dev, destination, &destination_record) &&
      destination_record.scale > 1;
  if ((!source_scaled && !destination_scaled) || dev.blit_image == nullptr ||
      regions == nullptr) {
    return false;
  }
  std::vector<VkImageBlit> blits;
  blits.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    const VkImageCopy& region = regions[index];
    VkImageBlit blit{};
    blit.srcSubresource = region.srcSubresource;
    blit.dstSubresource = region.dstSubresource;
    if (!HostRegion(source_scaled ? &source_record : nullptr,
                    region.srcSubresource.mipLevel, region.srcOffset,
                    region.extent, &blit.srcOffsets[0], &blit.srcOffsets[1]) ||
        !HostRegion(destination_scaled ? &destination_record : nullptr,
                    region.dstSubresource.mipLevel, region.dstOffset,
                    region.extent, &blit.dstOffsets[0], &blit.dstOffsets[1])) {
      LogFailure("skipped an image copy with an unsupported region");
      return true;
    }
    blits.push_back(blit);
  }
  const bool same_scale = source_scaled && destination_scaled &&
                          source_record.scale == destination_record.scale;
  dev.blit_image(command_buffer, source, source_layout, destination,
                 destination_layout, static_cast<std::uint32_t>(blits.size()),
                 blits.data(), same_scale ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
  CountDiagnostics([](Etc2DecodeTotals& totals) { ++totals.blit_copies; });
  return true;
}

namespace {

// With MOCKTAIL_FRAME_STATS=1: counts an image copy that touches an
// emulated image, and one that writes an emulated image from another image,
// whose texels never went through the decode.
void CountImageCopy(VulkanEtc2Emulation::State* state, const HostDevice& dev,
                    VkImage source, VkImage destination);

}  // namespace

void VulkanEtc2Emulation::CmdCopyImage(
    VkDevice device, VkCommandBuffer command_buffer, VkImage source,
    VkImageLayout source_layout, VkImage destination,
    VkImageLayout destination_layout, std::uint32_t region_count,
    const VkImageCopy* regions) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->copy_image == nullptr) {
    return;
  }
  CountImageCopy(state_, *dev, source, destination);
  if (state_->SkipMixedBcCopy(*dev, source, destination)) {
    return;
  }
  if (dev->emulated && state_->options.upscale > 1 &&
      state_->BlitScaledCopy(*dev, command_buffer, source, source_layout,
                             destination, destination_layout, region_count,
                             regions)) {
    return;
  }
  dev->copy_image(command_buffer, source, source_layout, destination,
                  destination_layout, region_count, regions);
}

void VulkanEtc2Emulation::CmdCopyImage2(VkDevice device,
                                        VkCommandBuffer command_buffer,
                                        const VkCopyImageInfo2* info) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->copy_image2 == nullptr || info == nullptr) {
    return;
  }
  CountImageCopy(state_, *dev, info->srcImage, info->dstImage);
  if (state_->SkipMixedBcCopy(*dev, info->srcImage, info->dstImage)) {
    return;
  }
  if (dev->emulated && state_->options.upscale > 1 &&
      info->pRegions != nullptr) {
    std::vector<VkImageCopy> regions;
    regions.reserve(info->regionCount);
    for (std::uint32_t index = 0; index < info->regionCount; ++index) {
      const VkImageCopy2& region = info->pRegions[index];
      regions.push_back({region.srcSubresource, region.srcOffset,
                         region.dstSubresource, region.dstOffset,
                         region.extent});
    }
    if (state_->BlitScaledCopy(*dev, command_buffer, info->srcImage,
                               info->srcImageLayout, info->dstImage,
                               info->dstImageLayout, info->regionCount,
                               regions.data())) {
      return;
    }
  }
  dev->copy_image2(command_buffer, info);
}

namespace {

void CountImageCopy(VulkanEtc2Emulation::State* state, const HostDevice& dev,
                    VkImage source, VkImage destination) {
  if (!dev.emulated || !state->options.frame_stats) {
    return;
  }
  bool source_emulated = false;
  bool destination_emulated = false;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    source_emulated = state->images.count(source) != 0;
    destination_emulated = state->images.count(destination) != 0;
  }
  if (!source_emulated && !destination_emulated) {
    return;
  }
  state->CountDiagnostics([&](Etc2DecodeTotals& totals) {
    ++totals.copy_image_calls;
    if (destination_emulated && !source_emulated) {
      ++totals.raw_copies_into;
    }
  });
}

}  // namespace

void VulkanEtc2Emulation::CmdExecuteCommands(
    VkDevice device, VkCommandBuffer command_buffer, std::uint32_t count,
    const VkCommandBuffer* secondaries) {
  const HostDevice* dev = state_->Find(device);
  if (dev == nullptr || dev->execute_commands == nullptr) {
    return;
  }
  std::uint64_t with_uploads = 0;
  std::uint64_t simultaneous = 0;
  if (dev->emulated && secondaries != nullptr &&
      state_->has_commands.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::vector<VkCommandBuffer> uploading;
    for (std::uint32_t index = 0; index < count; ++index) {
      const auto found = state_->commands.find(secondaries[index]);
      if (found == state_->commands.end() || !found->second.HasUploads()) {
        continue;
      }
      uploading.push_back(secondaries[index]);
      if ((found->second.begin_flags &
           VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) != 0) {
        ++simultaneous;
      }
    }
    if (!uploading.empty()) {
      CommandRecord& primary = state_->commands[command_buffer];
      if (primary.device == VK_NULL_HANDLE) {
        primary.device = device;
      }
      primary.secondaries.insert(primary.secondaries.end(), uploading.begin(),
                                 uploading.end());
      with_uploads = uploading.size();
    }
  }
  if (dev->emulated) {
    state_->CountDiagnostics([&](Etc2DecodeTotals& totals) {
      totals.secondaries_executed += count;
      totals.secondaries_with_uploads += with_uploads;
      totals.simultaneous_secondaries += simultaneous;
    });
  }
  dev->execute_commands(command_buffer, count, secondaries);
}

void VulkanEtc2Emulation::NoteDispatch(VkCommandBuffer command_buffer) {
  if (!state_->options.frame_stats ||
      !state_->has_commands.load(std::memory_order_acquire)) {
    return;
  }
  std::lock_guard<std::mutex> lock(state_->mutex);
  const auto record = state_->commands.find(command_buffer);
  if (record != state_->commands.end()) {
    record->second.dispatch_seen = true;
  }
}

void VulkanEtc2Emulation::State::CollectCpuWork(
    const VkCommandBuffer* command_buffers, std::uint32_t count,
    std::vector<CpuWork>* work) {
  std::lock_guard<std::mutex> lock(mutex);
  std::vector<VkCommandBuffer> collected;
  auto collect = [&](VkCommandBuffer command_buffer) {
    const auto record = commands.find(command_buffer);
    if (record == commands.end()) {
      return;
    }
    // A secondary executed twice in the batch decodes into the same staging
    // twice; the second pass only needs counting.
    const bool repeat = std::find(collected.begin(), collected.end(),
                                  command_buffer) != collected.end();
    collected.push_back(command_buffer);
    for (const PendingUpload& upload : record->second.uploads) {
      const auto binding = buffers.find(upload.source);
      if (binding == buffers.end()) {
        LogFailure("upload source buffer has no tracked memory binding");
        continue;
      }
      CpuWork item;
      item.upload = upload;
      item.repeat = repeat;
      if (TrackMemory()) {
        const auto memory = memories.find(binding->second.memory);
        if (memory != memories.end()) {
          item.memory_type = memory->second.type;
        }
      }
      const VkDeviceSize absolute =
          binding->second.offset + upload.source_offset;
      const auto mapping = mappings.find(binding->second.memory);
      if (mapping == mappings.end()) {
        item.memory = binding->second.memory;
        item.memory_offset = absolute;
      } else if (absolute >= mapping->second.offset &&
                 (mapping->second.size == VK_WHOLE_SIZE ||
                  (absolute - mapping->second.offset <=
                       mapping->second.size &&
                   upload.source_span <= mapping->second.size -
                                            (absolute -
                                             mapping->second.offset)))) {
        item.source = static_cast<std::uint8_t*>(mapping->second.data) +
                      (absolute - mapping->second.offset);
      } else {
        LogFailure("upload source lies outside the mapped range");
        continue;
      }
      work->push_back(item);
    }
  };
  for (std::uint32_t index = 0; index < count; ++index) {
    collect(command_buffers[index]);
    const auto record = commands.find(command_buffers[index]);
    if (record != commands.end()) {
      const std::vector<VkCommandBuffer> secondaries =
          record->second.secondaries;
      for (VkCommandBuffer secondary : secondaries) {
        collect(secondary);
      }
    }
  }
}

VulkanEtc2Emulation::State::SourceTypes
VulkanEtc2Emulation::State::CountSourceTypes(
    const std::vector<CpuWork>& work) {
  SourceTypes types;
  if (!options.frame_stats) {
    return types;
  }
  // Which memory Roblox's upload sources live in.
  std::uint64_t non_host_visible = 0;
  const HostDevice* dev = nullptr;
  for (const CpuWork& item : work) {
    if (dev == nullptr || dev->device != item.upload.device) {
      dev = Find(item.upload.device);
    }
    if (item.memory_type >= kEtc2CountedMemoryTypes || dev == nullptr ||
        item.memory_type >= dev->memory.memoryTypeCount) {
      types.unknown += item.upload.compressed;
      continue;
    }
    types.bytes[item.memory_type] += item.upload.compressed;
    if ((dev->memory.memoryTypes[item.memory_type].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
      ++non_host_visible;
    }
  }
  CountTotals([&](Etc2DecodeTotals& totals) {
    for (std::size_t type = 0; type < kEtc2CountedMemoryTypes; ++type) {
      totals.source_bytes_by_type[type] += types.bytes[type];
    }
    totals.source_bytes_unknown_type += types.unknown;
    totals.non_host_visible_sources += non_host_visible;
  });
  return types;
}

std::uint32_t VulkanEtc2Emulation::State::RunCpuUploads(
    const VkCommandBuffer* command_buffers, std::uint32_t count) {
  if (command_buffers == nullptr || count == 0 ||
      !has_commands.load(std::memory_order_acquire)) {
    return 0;
  }
  std::vector<CpuWork> work;
  CollectCpuWork(command_buffers, count, &work);
  if (work.empty()) {
    // Most submits; never wait for another submit's decode for nothing.
    return 0;
  }
  const SourceTypes types = CountSourceTypes(work);
  const std::uint32_t decoded = RunCpuUploadsFast(&work, types);
  Housekeeping(Clock::now());
  return decoded;
}

namespace {

// Roblox's upload memory that it does not keep mapped, mapped for one CPU
// pass. Each allocation is mapped once: Vulkan forbids concurrent mappings.
class TemporaryMappings {
 public:
  TemporaryMappings() = default;
  ~TemporaryMappings() { UnmapAll(); }
  TemporaryMappings(const TemporaryMappings&) = delete;
  TemporaryMappings& operator=(const TemporaryMappings&) = delete;

  const std::uint8_t* Map(const HostDevice* dev, VkDeviceMemory memory,
                          VkDeviceSize offset) {
    auto found = mappings_.find(memory);
    if (found == mappings_.end()) {
      void* data = nullptr;
      if (dev == nullptr || dev->map_memory == nullptr ||
          dev->map_memory(dev->device, memory, 0, VK_WHOLE_SIZE, 0, &data) !=
              VK_SUCCESS) {
        return nullptr;
      }
      found = mappings_
                  .emplace(memory,
                           Mapping{dev, static_cast<std::uint8_t*>(data)})
                  .first;
    }
    return found->second.data + offset;
  }

  void UnmapAll() {
    for (const auto& [memory, mapping] : mappings_) {
      if (mapping.dev->unmap_memory != nullptr) {
        mapping.dev->unmap_memory(mapping.dev->device, memory);
      }
    }
    mappings_.clear();
  }

 private:
  struct Mapping {
    const HostDevice* dev = nullptr;
    std::uint8_t* data = nullptr;
  };
  std::unordered_map<VkDeviceMemory, Mapping> mappings_;
};

std::uint64_t Milliseconds(Clock::duration duration) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
}

}  // namespace

// The CPU fast path (DESIGN C2): copy-in with streaming loads and decode on
// the worker pool; unscaled uploads decode straight into their staging,
// which rewritten regions lay out exactly as the decoder writes; upscaled
// ones resample per image in parallel. Its output is plus.1's, byte for byte
// (the WP0T goldens).
// Uploads into BC images (C4.4) transcode straight into their staging, in
// pieces on the pool, unless an override may replace them.
std::uint32_t VulkanEtc2Emulation::State::RunCpuUploadsFast(
    std::vector<CpuWork>* work, const SourceTypes& types) {
  constexpr VkDeviceSize kIntoStaging = ~VkDeviceSize{0};
  struct Item {
    const PendingUpload* upload = nullptr;
    const std::uint8_t* source = nullptr;  // Roblox's bytes, mapped
    VkDeviceSize compressed_offset = 0;    // in fast_source
    VkDeviceSize decoded_offset = kIntoStaging;  // in fast_decoded
    bool repeat = false;
  };
  TemporaryMappings temporary;
  // Everything from here on holds up the submit.
  const auto decode_start = Clock::now();
  std::vector<Item> items;
  items.reserve(work->size());
  std::vector<const PendingUpload*> traced;
  std::vector<std::uint32_t> traced_types;
  VkDeviceSize compressed_total = 0;
  VkDeviceSize decoded_total = 0;  // of the uploads that need scratch
  VkDeviceSize emit_total = 0;
  VkDeviceSize host_total = 0;
  for (const CpuWork& entry : *work) {
    const PendingUpload& upload = entry.upload;
    Item item;
    item.upload = &upload;
    item.repeat = entry.repeat;
    item.source = entry.source;
    if (item.source == nullptr) {
      item.source = temporary.Map(Find(upload.device), entry.memory,
                                  entry.memory_offset);
      if (item.source == nullptr) {
        LogFailure("could not map an unmapped upload source");
        continue;
      }
    }
    item.compressed_offset = compressed_total;
    compressed_total += upload.compressed;
    // Upscaled uploads resample, and override candidates may be replaced
    // or dumped, from decoded texels; everything else is final as decoded
    // (or, for a BC image, as transcoded).
    if (upload.scale > 1 || MayOverride(upload)) {
      item.decoded_offset = decoded_total;
      decoded_total += upload.decoded;
      emit_total += upload.target_bytes;
    }
    host_total += upload.target_bytes;
    items.push_back(item);
    if (Tracing()) {
      traced.push_back(&upload);
      traced_types.push_back(entry.memory_type);
    }
  }
  if (items.empty()) {
    return 0;
  }

  std::unique_lock<std::mutex> scratch_lock(scratch_mutex);
  const auto copy_start = Clock::now();
  std::uint8_t* const compressed =
      fast_source.Reserve(compressed_total, copy_start);
  std::uint8_t* const decoded =
      decoded_total != 0 ? fast_decoded.Reserve(decoded_total, copy_start)
                         : nullptr;
  if ((compressed_total != 0 && compressed == nullptr) ||
      (decoded_total != 0 && decoded == nullptr)) {
    LogFailure("could not allocate decode scratch; uploads skipped");
    return 0;
  }
  const unsigned workers = WorkerCount();
  WorkerPool* const pool =
      workers > 1 ? &WorkerPool::Shared(workers) : nullptr;
  unsigned threads = 1;

  // Copy-in: pieces of at most 256 KiB, whole block rows of strided
  // uploads, batched into tasks of at least 128 KiB.
  struct Piece {
    const std::uint8_t* source;
    std::uint8_t* destination;
    std::size_t row_bytes;
    std::size_t rows;
    std::size_t stride;
  };
  std::vector<Piece> pieces;
  std::vector<std::size_t> task_ends;
  std::size_t task_bytes = 0;
  const auto add_piece = [&](const Piece& piece) {
    pieces.push_back(piece);
    task_bytes += piece.row_bytes * piece.rows;
    if (task_bytes >= kCopyTaskBytes) {
      task_ends.push_back(pieces.size());
      task_bytes = 0;
    }
  };
  for (const Item& item : items) {
    const PendingUpload& upload = *item.upload;
    std::uint8_t* destination = compressed + item.compressed_offset;
    if (upload.source_span == upload.compressed) {
      for (VkDeviceSize offset = 0; offset < upload.compressed;
           offset += kCopyPieceBytes) {
        const std::size_t bytes = static_cast<std::size_t>(std::min<
            VkDeviceSize>(kCopyPieceBytes, upload.compressed - offset));
        add_piece({item.source + offset, destination + offset, bytes, 1,
                   bytes});
      }
      continue;
    }
    const VkDeviceSize compressed_layer = upload.compressed / upload.layers;
    const VkDeviceSize block_rows =
        (static_cast<VkDeviceSize>(upload.height) + 3) / 4;
    const VkDeviceSize packed_row = compressed_layer / block_rows;
    const VkDeviceSize rows_per_piece =
        std::max<VkDeviceSize>(1, kCopyPieceBytes / packed_row);
    for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
      for (VkDeviceSize row = 0; row < block_rows; row += rows_per_piece) {
        add_piece({item.source + layer * upload.source_layer_stride +
                       row * upload.source_row_stride,
                   destination + layer * compressed_layer + row * packed_row,
                   static_cast<std::size_t>(packed_row),
                   static_cast<std::size_t>(
                       std::min(rows_per_piece, block_rows - row)),
                   static_cast<std::size_t>(upload.source_row_stride)});
      }
    }
  }
  if (task_bytes != 0) {
    task_ends.push_back(pieces.size());
  }
  const auto copy_task = [&](std::size_t task) {
    for (std::size_t index = task == 0 ? 0 : task_ends[task - 1];
         index < task_ends[task]; ++index) {
      const Piece& piece = pieces[index];
      CopyStridedFromMappedMemory(piece.destination, piece.source,
                                  piece.row_bytes, piece.rows, piece.stride);
    }
  };
  if (pool != nullptr && task_ends.size() > 1 &&
      compressed_total >= kParallelCopyMinimumBytes) {
    threads = std::max(threads, pool->ParallelFor(task_ends.size(), copy_task));
  } else {
    for (std::size_t task = 0; task < task_ends.size(); ++task) {
      copy_task(task);
    }
  }
  temporary.UnmapAll();
  for (const Item& item : items) {
    if (g_decode_logs.load(std::memory_order_relaxed) >= 4 ||
        !ShouldLog(&g_decode_logs, 4)) {
      break;
    }
    std::fprintf(stderr, "  [vulkan] ETC2 upload decoded %ux%u layers=%u\n",
                 item.upload->width, item.upload->height,
                 item.upload->layers);
  }

  // Decode, in bands on the pool; transcode to BC, in pieces on the pool.
  const auto jobs_start = Clock::now();
  std::vector<EtcDecodeJob> jobs;
  jobs.reserve(items.size());
  struct BcPiece {
    EtcFormat format;
    const std::uint8_t* source;
    std::uint8_t* target;
    std::size_t bytes;
  };
  std::vector<BcPiece> bc_pieces;
  std::vector<std::size_t> bc_task_ends;
  std::size_t bc_task_blocks = 0;
  for (const Item& item : items) {
    const PendingUpload& upload = *item.upload;
    const bool into_staging = item.decoded_offset == kIntoStaging;
    if (into_staging && item.repeat) {
      continue;  // its first pass writes the same texels to the same place
    }
    if (into_staging && upload.bc) {
      // The blocks of every layer lie one after another in both, so the
      // upload is one run of blocks.
      const std::size_t block_bytes = EtcBlockBytes(upload.format);
      const std::size_t bytes = static_cast<std::size_t>(upload.compressed);
      for (std::size_t offset = 0; offset < bytes;
           offset += kBcPieceBlocks * block_bytes) {
        const std::size_t piece =
            std::min(kBcPieceBlocks * block_bytes, bytes - offset);
        bc_pieces.push_back({upload.format,
                             compressed + item.compressed_offset + offset,
                             upload.target + offset, piece});
        bc_task_blocks += piece / block_bytes;
        if (bc_task_blocks >= kBcPieceBlocks) {
          bc_task_ends.push_back(bc_pieces.size());
          bc_task_blocks = 0;
        }
      }
      continue;
    }
    const VkDeviceSize compressed_layer = upload.compressed / upload.layers;
    const VkDeviceSize decoded_layer = upload.decoded / upload.layers;
    std::uint8_t* destination =
        into_staging ? upload.target : decoded + item.decoded_offset;
    for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
      EtcDecodeJob job;
      job.format = upload.format;
      job.source = compressed + item.compressed_offset + layer * compressed_layer;
      job.source_bytes = compressed_layer;
      job.width = upload.width;
      job.height = upload.height;
      job.destination = destination + layer * decoded_layer;
      job.destination_bytes = decoded_layer;
      jobs.push_back(job);
    }
  }
  threads = std::max(threads, DecodeEtcJobs(jobs.data(), jobs.size(), workers));
  if (bc_task_blocks != 0) {
    bc_task_ends.push_back(bc_pieces.size());
  }
  std::atomic<bool> bc_failed{false};
  const auto bc_task = [&](std::size_t task) {
    for (std::size_t index = task == 0 ? 0 : bc_task_ends[task - 1];
         index < bc_task_ends[task]; ++index) {
      const BcPiece& piece = bc_pieces[index];
      if (!TranscodeEtcToBc(piece.format, piece.source, piece.bytes,
                            piece.target, piece.bytes, kBcRefine)) {
        bc_failed.store(true, std::memory_order_relaxed);
      }
    }
  };
  if (pool != nullptr && bc_task_ends.size() > 1) {
    threads = std::max(threads, pool->ParallelFor(bc_task_ends.size(), bc_task));
  } else {
    for (std::size_t task = 0; task < bc_task_ends.size(); ++task) {
      bc_task(task);
    }
  }
  const auto jobs_end = Clock::now();
  for (const EtcDecodeJob& job : jobs) {
    if (!job.ok) {
      LogFailure("could not decode an upload layer");
    }
  }
  if (bc_failed.load(std::memory_order_relaxed)) {
    LogFailure("could not encode an upload as BC");
  }

  // Emit what did not decode straight into staging.
  if (overrides.active()) {
    // Hashing, dumps and lookups, in submit order as before.
    for (const Item& item : items) {
      if (item.decoded_offset != kIntoStaging) {
        EmitUpload(*item.upload, compressed + item.compressed_offset,
                   decoded + item.decoded_offset);
      }
    }
  } else if (emit_total != 0) {
    // Per image, in submit order within it: mip 0 sets the base that the
    // image's later mips resample from.
    std::vector<std::vector<std::size_t>> groups;
    std::unordered_map<VkImage, std::size_t> group_of;
    for (std::size_t index = 0; index < items.size(); ++index) {
      if (items[index].decoded_offset == kIntoStaging) {
        continue;
      }
      const auto found =
          group_of.emplace(items[index].upload->image, groups.size());
      if (found.second) {
        groups.emplace_back();
      }
      groups[found.first->second].push_back(index);
    }
    const auto emit_group = [&](std::size_t group) {
      for (std::size_t index : groups[group]) {
        EmitScaled(*items[index].upload, decoded + items[index].decoded_offset);
      }
    };
    if (pool != nullptr && groups.size() > 1 &&
        emit_total >= kParallelEmitMinimumBytes) {
      threads = std::max(threads, pool->ParallelFor(groups.size(), emit_group));
    } else {
      for (std::size_t group = 0; group < groups.size(); ++group) {
        emit_group(group);
      }
    }
  }
  const auto emit_end = Clock::now();
  const std::uint64_t trace_ns =
      Tracing() ? TraceUploads(traced, traced_types, compressed) : 0;
  const Clock::time_point done = Clock::now();
  fast_source.MarkUsed(done);
  if (decoded_total != 0) {
    fast_decoded.MarkUsed(done);
  }
  scratch_lock.unlock();
  const std::uint64_t decode_ns = Nanoseconds(done - decode_start) - trace_ns;
  const std::uint64_t phase_ms[4] = {
      Milliseconds(copy_start - decode_start),
      Milliseconds(jobs_start - copy_start),
      Milliseconds(jobs_end - jobs_start), Milliseconds(emit_end - jobs_end)};
  FinishCpuPass(items.size(), compressed_total, host_total, decode_ns,
                phase_ms, threads, types);
  return static_cast<std::uint32_t>(items.size());
}

void VulkanEtc2Emulation::State::FinishCpuPass(
    std::size_t uploads, VkDeviceSize compressed_total,
    VkDeviceSize host_total, std::uint64_t decode_ns,
    const std::uint64_t (&phase_ms)[4], unsigned threads,
    const SourceTypes& types) {
  // MOCKTAIL_FRAME_STATS=1: name each submit that held the render thread.
  if (options.frame_stats && decode_ns >= 20000000ULL) {
    std::fprintf(stderr,
                 "  [vulkan] ETC2 decode stall %llu ms: %zu uploads, "
                 "%.1f MiB compressed, %.1f MiB written "
                 "(prepare %llu, copy %llu, decode %llu, write %llu ms)%s\n",
                 static_cast<unsigned long long>(decode_ns / 1000000ULL),
                 uploads, static_cast<double>(compressed_total) / 1048576.0,
                 static_cast<double>(host_total) / 1048576.0,
                 static_cast<unsigned long long>(phase_ms[0]),
                 static_cast<unsigned long long>(phase_ms[1]),
                 static_cast<unsigned long long>(phase_ms[2]),
                 static_cast<unsigned long long>(phase_ms[3]),
                 FormatEtc2StallSuffix(threads, types.bytes, types.unknown)
                     .c_str());
  }
  bool milestone = false;
  {
    std::lock_guard<std::mutex> lock(totals_mutex);
    const std::uint64_t before = totals.host_bytes;
    AddEtc2SubmitDecode(&totals, uploads, compressed_total, host_total,
                        decode_ns);
    if (options.frame_stats) {
      AddEtc2SubmitDecode(&window, uploads, compressed_total, host_total,
                          decode_ns);
    }
    milestone = Etc2DecodeTotalsMilestone(before, totals.host_bytes);
  }
  if (milestone) {
    LogTotals(" so far");
  }
}

void VulkanEtc2Emulation::State::EmitScaled(const PendingUpload& upload,
                                            const std::uint8_t* decoded) {
  std::shared_ptr<const RgbaImage> base;
  if (upload.full_mip) {
    // A whole mip 0 becomes the image's base, its block padding clipped.
    std::shared_ptr<RgbaImage> level0;
    if (upload.mip_level == 0) {
      level0 = std::make_shared<RgbaImage>();
      level0->width = upload.source_width;
      level0->height = upload.source_height;
      const std::size_t row = static_cast<std::size_t>(upload.source_width) * 4;
      if (upload.source_width == upload.width) {
        level0->pixels.assign(decoded, decoded + row * upload.source_height);
      } else {
        level0->pixels.resize(row * upload.source_height);
        for (std::uint32_t y = 0; y < upload.source_height; ++y) {
          std::memcpy(level0->pixels.data() + y * row,
                      decoded + static_cast<std::size_t>(y) * upload.width * 4,
                      row);
        }
      }
    }
    std::lock_guard<std::mutex> lock(mutex);
    const auto record = images.find(upload.image);
    if (record != images.end()) {
      if (level0 != nullptr) {
        record->second.base = std::move(level0);
      }
      base = record->second.base;
    }
  }
  if (base != nullptr) {
    ResampleRgba(*base, upload.target_width, upload.target_height,
                 upload.target);
    return;
  }
  ResampleRgba(decoded, upload.source_width, upload.source_height,
               static_cast<std::size_t>(upload.width) * 4,
               upload.target_width, upload.target_height, upload.target);
}

void VulkanEtc2Emulation::State::Housekeeping(Clock::time_point now) {
  const std::int64_t now_ns = static_cast<std::int64_t>(
      Nanoseconds(now.time_since_epoch()));
  std::int64_t due = next_housekeeping_ns.load(std::memory_order_relaxed);
  if (now_ns < due ||
      !next_housekeeping_ns.compare_exchange_strong(
          due, now_ns + 1000000000LL, std::memory_order_relaxed)) {
    return;
  }
  std::vector<VkDevice> emulated;
  {
    std::lock_guard<std::mutex> lock(devices_mutex);
    for (const auto& dev : devices) {
      if (dev->emulated) {
        emulated.push_back(dev->device);
      }
    }
  }
  for (VkDevice device : emulated) {
    // Hand over hand: once the device's own lock is held, DestroyDevice
    // waits for it before it destroys the arena and the device record.
    std::unique_lock<std::mutex> list(devices_mutex);
    HostDevice* dev = nullptr;
    for (const auto& candidate : devices) {
      if (candidate->device == device) {
        dev = candidate.get();
      }
    }
    if (dev == nullptr) {
      continue;
    }
    std::lock_guard<std::mutex> own(dev->mutex);
    list.unlock();
    if (dev->host_staging != nullptr) {
      dev->host_staging->Trim(now);
    }
    // Without this, only a submit with GPU-route uploads trims them, and a
    // loading burst's chunks stay for as long as none comes.
    if (dev->gpu != nullptr) {
      dev->gpu->Trim(now);
    }
  }
  std::unique_lock<std::mutex> scratch(scratch_mutex, std::try_to_lock);
  if (scratch.owns_lock()) {
    fast_source.ReleaseIfIdle(now);
    fast_decoded.ReleaseIfIdle(now);
  }
}

// The rescue copies each upload with one region, so the CPU writes what the
// decode pass would have: whole blocks, 4 x blocks_wide texels per row and
// 4 x blocks_high rows per layer, layer after layer; for a BC image the
// etc2_bc pass's tightly packed BC blocks, encoded as on the CPU route
// (kBcRefine), so a rescued upload has the same bytes. The source bytes are
// read now, inside vkQueueSubmit, as on the CPU route. The staging goes to
// the record of a command buffer of the batch that holds GPU-route
// uploads: its copies read the arena after the rescue pass's barrier, so
// once Roblox has reset or freed it the rescue copy read the staging too.
bool VulkanEtc2Emulation::State::RescueDecode(
    const HostDevice& dev, const std::vector<VkCommandBuffer>& command_buffers,
    const std::vector<Etc2GpuUpload>& uploads,
    std::vector<Etc2RescueSource>* sources) {
  if (uploads.empty()) {
    return false;
  }
  struct Source {
    const std::uint8_t* mapped = nullptr;  // Roblox's mapping, or null
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize memory_offset = 0;
  };
  std::vector<Source> where(uploads.size());
  VkCommandBuffer owner = VK_NULL_HANDLE;
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (std::size_t index = 0; index < uploads.size(); ++index) {
      const Etc2GpuUpload& upload = uploads[index];
      const auto binding = buffers.find(upload.source);
      if (binding == buffers.end()) {
        LogFailure("upload source buffer has no tracked memory binding");
        return false;
      }
      const VkDeviceSize absolute =
          binding->second.offset + upload.source_offset;
      const auto mapping = mappings.find(binding->second.memory);
      if (mapping == mappings.end()) {
        where[index].memory = binding->second.memory;
        where[index].memory_offset = absolute;
      } else if (absolute >= mapping->second.offset &&
                 (mapping->second.size == VK_WHOLE_SIZE ||
                  (absolute - mapping->second.offset <=
                       mapping->second.size &&
                   upload.source_span <= mapping->second.size -
                                            (absolute -
                                             mapping->second.offset)))) {
        where[index].mapped =
            static_cast<const std::uint8_t*>(mapping->second.data) +
            (absolute - mapping->second.offset);
      } else {
        LogFailure("upload source lies outside the mapped range");
        return false;
      }
    }
    const auto holds_gpu_uploads = [&](VkCommandBuffer command_buffer) {
      const auto record = commands.find(command_buffer);
      return record != commands.end() && !record->second.gpu_uploads.empty();
    };
    for (VkCommandBuffer command_buffer : command_buffers) {
      if (owner != VK_NULL_HANDLE) {
        break;
      }
      if (holds_gpu_uploads(command_buffer)) {
        owner = command_buffer;
        break;
      }
      const auto record = commands.find(command_buffer);
      if (record == commands.end()) {
        continue;
      }
      for (VkCommandBuffer secondary : record->second.secondaries) {
        if (holds_gpu_uploads(secondary)) {
          owner = secondary;
          break;
        }
      }
    }
  }
  if (owner == VK_NULL_HANDLE) {
    LogFailure("a rescued batch holds no GPU-route uploads");
    return false;
  }

  // Staging: decoded texels (BC blocks in ETC2->BC mode) at 16-byte aligned
  // offsets.
  std::vector<VkDeviceSize> offsets(uploads.size());
  VkDeviceSize total = 0;
  VkDeviceSize compressed_total = 0;
  for (std::size_t index = 0; index < uploads.size(); ++index) {
    offsets[index] = (total + kEtc2StagingRegionAlignment - 1) /
                     kEtc2StagingRegionAlignment * kEtc2StagingRegionAlignment;
    total = offsets[index] + uploads[index].output_bytes;
    compressed_total += uploads[index].compressed;
  }
  ArenaRef range;
  Staging staging;
  std::uint8_t* mapped = nullptr;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize base = 0;
  if (dev.host_staging != nullptr) {
    if (dev.host_staging->Take(total, &range) && range.mapped != nullptr) {
      mapped = range.mapped;
      buffer = range.buffer;
      base = range.offset;
    } else {
      ChunkArena::ReleaseRef(range);
      range = ArenaRef{};
    }
  }
  if (mapped == nullptr) {
    if (!CreateStaging(dev, total, &staging, &mapped)) {
      LogFailure("could not allocate staging for a rescue");
      return false;
    }
    buffer = staging.buffer;
  }
  const auto give_back = [&]() {
    if (range.arena_id != 0) {
      ChunkArena::ReleaseRef(range);
    } else {
      DestroyStaging(dev, staging);
    }
  };

  // Copy-in from Roblox's memory (streaming loads, whole block rows), then
  // decode on the pool; in ETC2->BC mode encode instead, as the CPU route
  // does, in pieces on the pool. A BC upload's blocks stay in their order,
  // tightly packed, which is the arena's layout for it too.
  std::vector<std::uint8_t> compressed(
      static_cast<std::size_t>(compressed_total));
  std::vector<EtcDecodeJob> jobs;
  struct BcPiece {
    EtcFormat format;
    const std::uint8_t* source;
    std::uint8_t* target;
    std::size_t bytes;
  };
  std::vector<BcPiece> bc_pieces;
  {
    TemporaryMappings temporary;
    VkDeviceSize packed_offset = 0;
    for (std::size_t index = 0; index < uploads.size(); ++index) {
      const Etc2GpuUpload& upload = uploads[index];
      const std::uint8_t* source = where[index].mapped;
      if (source == nullptr) {
        source = temporary.Map(&dev, where[index].memory,
                               where[index].memory_offset);
        if (source == nullptr) {
          LogFailure("could not map an unmapped upload source");
          give_back();
          return false;
        }
      }
      const std::size_t row_bytes =
          static_cast<std::size_t>(upload.blocks_wide) *
          EtcBlockBytes(upload.format);
      const VkDeviceSize packed_layer =
          static_cast<VkDeviceSize>(row_bytes) * upload.blocks_high;
      const VkDeviceSize output_layer = upload.output_bytes / upload.layers;
      for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
        std::uint8_t* packed = compressed.data() + packed_offset;
        CopyStridedFromMappedMemory(
            packed, source + layer * upload.source_layer_stride, row_bytes,
            upload.blocks_high,
            static_cast<std::size_t>(upload.source_row_stride));
        std::uint8_t* const target =
            mapped + offsets[index] + layer * output_layer;
        if (upload.bc) {
          const std::size_t block_bytes = EtcBlockBytes(upload.format);
          const std::size_t layer_bytes =
              static_cast<std::size_t>(packed_layer);
          for (std::size_t offset = 0; offset < layer_bytes;
               offset += kBcPieceBlocks * block_bytes) {
            bc_pieces.push_back(
                {upload.format, packed + offset, target + offset,
                 std::min(kBcPieceBlocks * block_bytes, layer_bytes - offset)});
          }
          packed_offset += packed_layer;
          continue;
        }
        EtcDecodeJob job;
        job.format = upload.format;
        job.source = packed;
        job.source_bytes = static_cast<std::size_t>(packed_layer);
        job.width = 4 * upload.blocks_wide;
        job.height = 4 * upload.blocks_high;
        job.destination = target;
        job.destination_bytes = static_cast<std::size_t>(output_layer);
        jobs.push_back(job);
        packed_offset += packed_layer;
      }
    }
  }
  const unsigned workers = WorkerCount();
  DecodeEtcJobs(jobs.data(), jobs.size(), workers);
  for (const EtcDecodeJob& job : jobs) {
    if (!job.ok) {
      LogFailure("could not decode a rescued upload");
      give_back();
      return false;
    }
  }
  std::atomic<bool> bc_failed{false};
  const auto bc_task = [&](std::size_t index) {
    const BcPiece& piece = bc_pieces[index];
    if (!TranscodeEtcToBc(piece.format, piece.source, piece.bytes, piece.target,
                          piece.bytes, kBcRefine)) {
      bc_failed.store(true, std::memory_order_relaxed);
    }
  };
  if (workers > 1 && bc_pieces.size() > 1) {
    WorkerPool::Shared(workers).ParallelFor(bc_pieces.size(), bc_task);
  } else {
    for (std::size_t index = 0; index < bc_pieces.size(); ++index) {
      bc_task(index);
    }
  }
  if (bc_failed.load(std::memory_order_relaxed)) {
    LogFailure("could not encode a rescued upload as BC");
    give_back();
    return false;
  }

  bool kept = false;
  {
    std::lock_guard<std::mutex> lock(mutex);
    const auto record = commands.find(owner);
    if (record != commands.end()) {
      if (range.arena_id != 0) {
        record->second.staging_refs.push_back(range);
      } else {
        record->second.staging.push_back(staging);
      }
      kept = true;
    }
  }
  if (!kept) {
    give_back();
    return false;
  }
  sources->clear();
  for (std::size_t index = 0; index < uploads.size(); ++index) {
    sources->push_back({buffer, base + offsets[index]});
  }
  return true;
}

void VulkanEtc2Emulation::PrepareSubmit(VkQueue queue, std::uint32_t count,
                                        const VkSubmitInfo* submits,
                                        VkFence fence, Etc2SubmitPlan* plan) {
  if (plan != nullptr) {
    plan->queue = queue;
  }
  std::vector<bool> uploads(submits != nullptr ? count : 0, false);
  if (submits != nullptr) {
    for (std::uint32_t index = 0; index < count; ++index) {
      uploads[index] =
          state_->RunCpuUploads(submits[index].pCommandBuffers,
                                submits[index].commandBufferCount) != 0;
    }
  }
  // The GPU route puts its decode passes in front (DESIGN §3.4).
  state_->PlanGpuSubmit(queue, count, submits, fence, plan, &uploads);
  state_->NoteSubmit(queue, false, count,
                     static_cast<std::uint32_t>(
                         std::count(uploads.begin(), uploads.end(), true)));
}

void VulkanEtc2Emulation::PrepareSubmit2(VkQueue queue, std::uint32_t count,
                                         const VkSubmitInfo2* submits,
                                         VkFence fence, Etc2SubmitPlan* plan) {
  if (plan != nullptr) {
    plan->queue = queue;
  }
  std::vector<bool> uploads(submits != nullptr ? count : 0, false);
  if (submits != nullptr &&
      state_->has_commands.load(std::memory_order_acquire)) {
    // One decode pass, and one stall line, per batch, as for vkQueueSubmit.
    std::vector<VkCommandBuffer> command_buffers;
    for (std::uint32_t index = 0; index < count; ++index) {
      command_buffers.clear();
      for (std::uint32_t info = 0; info < submits[index].commandBufferInfoCount;
           ++info) {
        command_buffers.push_back(
            submits[index].pCommandBufferInfos[info].commandBuffer);
      }
      uploads[index] =
          state_->RunCpuUploads(
              command_buffers.data(),
              static_cast<std::uint32_t>(command_buffers.size())) != 0;
    }
  }
  // The GPU route puts its decode passes in front (DESIGN §3.4).
  state_->PlanGpuSubmit(queue, count, submits, fence, plan, &uploads);
  state_->NoteSubmit(queue, true, count,
                     static_cast<std::uint32_t>(
                         std::count(uploads.begin(), uploads.end(), true)));
}

void VulkanEtc2Emulation::FinishSubmit(
    Etc2SubmitPlan* plan, VkResult host_result,
    const std::function<VkResult(VkFence)>& submit_fence_only) {
  if (plan == nullptr) {
    return;
  }
  if (plan->gpu_slot != nullptr) {
    // The GPU route's slot: its fence follows Roblox's, and the slot is
    // in flight, or not submitted (DESIGN C3.3).
    HostDevice* dev = state_->Find(plan->device);
    if (dev != nullptr && dev->gpu != nullptr) {
      dev->gpu->FinishSubmit(plan, host_result, submit_fence_only);
      return;
    }
  }
  if (plan->trailing_fence != VK_NULL_HANDLE && host_result == VK_SUCCESS &&
      submit_fence_only) {
    submit_fence_only(plan->trailing_fence);
  }
}

void VulkanEtc2Emulation::ReleaseCommandBuffer(VkCommandBuffer command_buffer) {
  if (!state_->has_commands.load(std::memory_order_acquire)) {
    return;
  }
  CommandRecord doomed;
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto record = state_->commands.find(command_buffer);
    if (record == state_->commands.end()) {
      return;
    }
    doomed = std::move(record->second);
    state_->commands.erase(record);
    state_->has_commands.store(!state_->commands.empty(),
                               std::memory_order_release);
  }
  if (!doomed.HoldsResources()) {
    return;
  }
  const Clock::time_point start =
      state_->options.frame_stats ? Clock::now() : Clock::time_point{};
  state_->ReleaseRecordResources(std::move(doomed));
  if (state_->options.frame_stats) {
    const std::uint64_t elapsed = Nanoseconds(Clock::now() - start);
    state_->CountDiagnostics(
        [elapsed](Etc2DecodeTotals& totals) { totals.release_ns += elapsed; });
  }
}

void VulkanEtc2Emulation::OnPresent() {
  const Clock::time_point now = Clock::now();
  state_->Housekeeping(now);
  if (!state_->options.frame_stats) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(state_->totals_mutex);
    if (state_->window_start == Clock::time_point{}) {
      state_->window_start = now;
      return;
    }
    if (now - state_->window_start < std::chrono::seconds(2)) {
      return;
    }
  }
  state_->NoteArenaBytes();
  Etc2DecodeTotals window;
  std::vector<std::string> threads;
  {
    std::lock_guard<std::mutex> lock(state_->totals_mutex);
    window = state_->window;
    state_->window = Etc2DecodeTotals{};
    threads.swap(state_->new_submit_threads);
    state_->window_start = now;
  }
  if (!Etc2WindowActive(window) && threads.empty()) {
    return;
  }
  std::string line = FormatEtc2Window(window);
  for (std::size_t index = 0; index < threads.size(); ++index) {
    line += index == 0 ? " new_threads=" : ",";
    line += threads[index];
  }
  std::fprintf(stderr, "  [perf] etc2_window %s\n", line.c_str());
}

}  // namespace mocktail::graphics
