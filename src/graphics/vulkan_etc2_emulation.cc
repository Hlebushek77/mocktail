#include "mocktail/graphics/vulkan_etc2_emulation.h"

#include <pthread.h>
#include <sys/stat.h>

#include "mocktail/graphics/chunk_arena.h"
#include "mocktail/graphics/etc2_decode_totals.h"
#include "mocktail/graphics/texture_override.h"
#include "mocktail/graphics/vulkan_etc2_gpu.h"
#include "mocktail/graphics/vulkan_etc2_planning.h"

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
      mapped = {VK_FORMAT_R8G8B8A8_UNORM, EtcFormat::kEtc2Rgb8};
      break;
    case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_SRGB, EtcFormat::kEtc2Rgb8};
      break;
    case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_UNORM, EtcFormat::kEtc2Rgb8A1};
      break;
    case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_SRGB, EtcFormat::kEtc2Rgb8A1};
      break;
    case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_UNORM, EtcFormat::kEtc2Rgba8};
      break;
    case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
      mapped = {VK_FORMAT_R8G8B8A8_SRGB, EtcFormat::kEtc2Rgba8};
      break;
    case VK_FORMAT_EAC_R11_UNORM_BLOCK:
      mapped = {VK_FORMAT_R16_UNORM, EtcFormat::kEacR11};
      break;
    case VK_FORMAT_EAC_R11_SNORM_BLOCK:
      mapped = {VK_FORMAT_R16_SNORM, EtcFormat::kEacR11Signed};
      break;
    case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
      mapped = {VK_FORMAT_R16G16_UNORM, EtcFormat::kEacRg11};
      break;
    case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
      mapped = {VK_FORMAT_R16G16_SNORM, EtcFormat::kEacRg11Signed};
      break;
    default:
      return false;
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
  options.fast_path = ParseSwitch(lookup("MOCKTAIL_ETC2_FAST_PATH"), true);
  options.gpu = ParseSwitch(lookup("MOCKTAIL_ETC2_GPU"), true);
  options.to_bc = ParseSwitch(lookup("MOCKTAIL_ETC2_TO_BC"), false);
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

  // Created lazily, or by RegisterDevice; guarded by `mutex`. Lock order:
  // State::devices_mutex, then this, then an arena's own.
  std::mutex mutex;
  std::unique_ptr<ChunkArena> host_staging;  // the CPU route's staging
  std::unique_ptr<Etc2GpuDevice> gpu;        // the GPU route
  std::atomic<bool> saw_emulated_image{false};
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

struct ImageRecord : Etc2ImageShape {
  VkDevice device = VK_NULL_HANDLE;
  VkFormat app_format = VK_FORMAT_UNDEFINED;
  bool bc = false;  // the host image holds BC blocks (ETC2->BC mode)
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

// Memory of an emulated device, tracked with MOCKTAIL_FRAME_STATS=1: its
// type, and the images bound into it by offset, to spot aliasing.
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
  std::vector<std::unique_ptr<HostDevice>> devices;
  std::unordered_map<VkImage, ImageRecord> images;
  std::unordered_map<VkBuffer, BufferBinding> buffers;
  std::unordered_map<VkDeviceMemory, Mapping> mappings;
  std::unordered_map<VkCommandBuffer, CommandRecord> commands;
  std::unordered_map<VkCommandPool, PoolRecord> pools;
  std::unordered_map<VkQueue, QueueRecord> queues;
  // MOCKTAIL_FRAME_STATS=1 only.
  std::unordered_map<VkDeviceMemory, MemoryRecord> memories;
  std::unordered_map<VkImage, ImageBinding> image_bindings;
  std::atomic<bool> has_commands{false};
  std::mutex scratch_mutex;
  std::vector<std::uint8_t> scratch_source;
  std::vector<std::uint8_t> scratch_decoded;
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

  std::shared_ptr<const RgbaImage> FindOverride(const PendingUpload& upload,
                                                const std::uint8_t* compressed,
                                                const std::uint8_t* decoded) {
    if (!overrides.enabled() || upload.layers != 1 ||
        !IsColorFormat(upload.format)) {
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
    if (replacement != nullptr) {
      ResampleRgba(*replacement, upload.target_width, upload.target_height,
                   upload.target);
    } else if (upload.scale > 1) {
      std::shared_ptr<const RgbaImage> base;
      if (upload.full_mip) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto record = images.find(upload.image);
        if (record != images.end()) {
          if (upload.mip_level == 0) {
            auto level0 = std::make_shared<RgbaImage>();
            level0->width = upload.source_width;
            level0->height = upload.source_height;
            level0->pixels.assign(decoded, decoded + upload.decoded);
            if (upload.source_width != upload.width ||
                upload.source_height != upload.height) {
              level0->pixels.clear();
              for (std::uint32_t y = 0; y < upload.source_height; ++y) {
                const std::uint8_t* row =
                    decoded + static_cast<std::size_t>(y) * upload.width * 4;
                level0->pixels.insert(level0->pixels.end(), row,
                                      row + upload.source_width * 4);
              }
            }
            record->second.base = level0;
          }
          base = record->second.base;
        }
      }
      if (base != nullptr) {
        ResampleRgba(*base, upload.target_width, upload.target_height,
                     upload.target);
        return;
      }
      const std::uint8_t* source = decoded;
      std::vector<std::uint8_t> cropped;
      if (upload.source_width != upload.width ||
          upload.source_height != upload.height) {
        const std::size_t row = static_cast<std::size_t>(upload.source_width) * 4;
        cropped.resize(row * upload.source_height);
        for (std::uint32_t y = 0; y < upload.source_height; ++y) {
          std::memcpy(cropped.data() + y * row,
                      decoded + static_cast<std::size_t>(y) * upload.width * 4,
                      row);
        }
        source = cropped.data();
      }
      ResampleRgba(source, upload.source_width, upload.source_height,
                   upload.target_width, upload.target_height, upload.target);
    } else {
      std::memcpy(upload.target, decoded, upload.decoded);
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
    // The GPU route is not built yet: no device has an Etc2GpuDevice.
    static_cast<void>(target);
    return {Etc2Route::kCpu, Etc2CpuReason::kGpuUnavailable};
  }

  // Records the GPU route of one copy. False: the CPU route serves it.
  template <typename Region, typename HostCopy>
  bool RecordGpuCopy(const HostDevice& dev, VkCommandBuffer command_buffer,
                     const CopyTarget& target, VkBuffer source,
                     VkImage destination, std::uint32_t count,
                     const Region* regions, HostCopy& host_copy) {
    return false;
  }

  struct CpuCopyResult {
    bool recorded = false;
    bool staging_allocated = false;
    bool first_upload = false;  // the command buffer's first
    VkDeviceSize compressed = 0;
  };

  // Records the CPU route of one copy: the decoded texels go to a staging
  // buffer at submit time, and the host copy reads them from there.
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
    Staging staging;
    std::uint8_t* mapped = nullptr;
    if (!PlanRegions(target.image, dev.device, source, destination, count,
                     regions, &rewritten, &uploads, &total)) {
      LogFailure("skipped an upload with an unsupported region");
      return result;
    }
    for (const PendingUpload& upload : uploads) {
      result.compressed += upload.compressed;
    }
    if (!CreateStaging(dev, total, &staging, &mapped)) {
      LogFailure("could not allocate a host-visible staging buffer");
      return result;
    }
    result.staging_allocated = true;
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
               staging);
    result.recorded = true;
    host_copy(staging.buffer, rewritten.data());
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
    const RouteChoice choice = ChooseRoute(dev, target);
    if (choice.route == Etc2Route::kGpu &&
        RecordGpuCopy(dev, command_buffer, target, source, destination,
                      count, regions, host_copy)) {
      return;
    }
    const CpuCopyResult result =
        RecordCpuCopy(dev, command_buffer, target, source, destination, count,
                      regions, host_copy);
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

  // Keeps the uploads and their staging with the command buffer. True when
  // these are the command buffer's first uploads.
  bool Record(VkCommandBuffer command_buffer, VkDevice device,
              VkImage destination, std::vector<PendingUpload> uploads,
              const Staging& staging) {
    std::lock_guard<std::mutex> lock(mutex);
    CommandRecord& record = commands[command_buffer];
    if (record.device == VK_NULL_HANDLE) {
      record.device = device;
    }
    const bool first = !record.HasUploads();
    record.uploads.insert(record.uploads.end(), uploads.begin(),
                          uploads.end());
    record.staging.push_back(staging);
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
    // The GPU route starts building its pipelines here.
    static_cast<void>(dev);
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
      for (auto bound = record.images.begin();
           bound != record.images.end() && bound->first < end; ++bound) {
        if (bound->second.end > offset && bound->second.image != image &&
            (emulated || bound->second.emulated)) {
          aliased = true;
          break;
        }
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

  std::uint32_t RunCpuUploads(const VkCommandBuffer* command_buffers,
                              std::uint32_t count);
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
  }
  if (emulated) {
    std::fprintf(stderr,
                 "  [vulkan] ETC2/EAC emulation enabled: compressed uploads "
                 "decode to RGBA8/R16 host images\n");
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
  if (removed != nullptr) {
    state_->DestroyDeviceResources(*removed);
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
  host_info.format = format.host_format;
  host_info.flags &= ~static_cast<VkImageCreateFlags>(
      VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT);
  std::uint32_t scale = 1;
  const std::uint32_t upscale = state_->options.upscale;
  if (upscale > 1 && IsColorFormat(format.etc_format) &&
      create_info->imageType == VK_IMAGE_TYPE_2D &&
      create_info->arrayLayers == 1 && create_info->extent.depth == 1 &&
      create_info->extent.width <= kSmallTextureMaxExtent &&
      create_info->extent.height <= kSmallTextureMaxExtent) {
    scale = upscale;
    host_info.extent.width *= scale;
    host_info.extent.height *= scale;
  }
  const VkResult result = dev->create_image(device, &host_info, allocator, image);
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
    }
    bool format_list = false;
    for (auto* next = static_cast<const VkBaseInStructure*>(create_info->pNext);
         next != nullptr; next = next->pNext) {
      format_list = format_list ||
                    next->sType ==
                        VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
    }
    state_->CountTotals([&](Etc2DecodeTotals& totals) {
      ++totals.images;
      if (scale > 1) {
        ++totals.upscaled_images;
      }
      if (format_list && state_->options.frame_stats) {
        ++totals.format_lists;
      }
    });
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

std::uint32_t VulkanEtc2Emulation::State::RunCpuUploads(
    const VkCommandBuffer* command_buffers, std::uint32_t count) {
  if (command_buffers == nullptr || count == 0 ||
      !has_commands.load(std::memory_order_acquire)) {
    return 0;
  }
  struct Work {
    PendingUpload upload;
    std::uint8_t* source = nullptr;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize memory_offset = 0;
    std::uint32_t memory_type = UINT32_MAX;  // diagnostics and trace only
  };
  std::vector<Work> work;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto collect = [this, &work](VkCommandBuffer command_buffer) {
      const auto record = commands.find(command_buffer);
      if (record == commands.end()) {
        return;
      }
      for (const PendingUpload& upload : record->second.uploads) {
        const auto binding = buffers.find(upload.source);
        if (binding == buffers.end()) {
          LogFailure("upload source buffer has no tracked memory binding");
          continue;
        }
        Work item;
        item.upload = upload;
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
        work.push_back(item);
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
  if (options.frame_stats && !work.empty()) {
    // Which memory Roblox's upload sources live in.
    std::uint64_t by_type[kEtc2CountedMemoryTypes] = {};
    std::uint64_t unknown_type = 0;
    std::uint64_t non_host_visible = 0;
    const HostDevice* dev = nullptr;
    for (const Work& item : work) {
      if (dev == nullptr || dev->device != item.upload.device) {
        dev = Find(item.upload.device);
      }
      if (item.memory_type >= kEtc2CountedMemoryTypes || dev == nullptr ||
          item.memory_type >= dev->memory.memoryTypeCount) {
        unknown_type += item.upload.compressed;
        continue;
      }
      by_type[item.memory_type] += item.upload.compressed;
      if ((dev->memory.memoryTypes[item.memory_type].propertyFlags &
           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
        ++non_host_visible;
      }
    }
    CountTotals([&](Etc2DecodeTotals& totals) {
      for (std::size_t type = 0; type < kEtc2CountedMemoryTypes; ++type) {
        totals.source_bytes_by_type[type] += by_type[type];
      }
      totals.source_bytes_unknown_type += unknown_type;
      totals.non_host_visible_sources += non_host_visible;
    });
  }
  // Decode in cached RAM; mapped Vulkan memory makes scattered reads slow.
  // Map each source allocation once: Vulkan forbids concurrent mappings.
  struct TemporaryMapping {
    const HostDevice* dev = nullptr;
    std::uint8_t* data = nullptr;
  };
  std::unordered_map<VkDeviceMemory, TemporaryMapping> temporary;
  struct Copy {
    const std::uint8_t* source = nullptr;
    VkDeviceSize compressed = 0;
    VkDeviceSize decoded = 0;
  };
  // Everything from here on holds up the submit.
  const auto decode_start = Clock::now();
  std::vector<Copy> copies;
  std::vector<const PendingUpload*> copied_uploads;
  std::vector<std::uint32_t> copied_types;
  VkDeviceSize compressed_total = 0;
  VkDeviceSize decoded_total = 0;
  for (const Work& item : work) {
    const PendingUpload& upload = item.upload;
    const std::uint8_t* source = item.source;
    if (source == nullptr) {
      auto found = temporary.find(item.memory);
      if (found == temporary.end()) {
        const HostDevice* dev = Find(upload.device);
        void* data = nullptr;
        if (dev == nullptr || dev->map_memory == nullptr ||
            dev->map_memory(dev->device, item.memory, 0, VK_WHOLE_SIZE, 0,
                            &data) != VK_SUCCESS) {
          LogFailure("could not map an unmapped upload source");
          continue;
        }
        found = temporary
                    .emplace(item.memory,
                             TemporaryMapping{
                                 dev, static_cast<std::uint8_t*>(data)})
                    .first;
      }
      source = found->second.data + item.memory_offset;
    }
    copies.push_back({source, upload.compressed, upload.decoded});
    copied_uploads.push_back(&upload);
    copied_types.push_back(item.memory_type);
    compressed_total += upload.compressed;
    decoded_total += upload.decoded;
  }

  std::lock_guard<std::mutex> scratch_lock(scratch_mutex);
  const auto copy_start = Clock::now();
  if (scratch_source.size() < compressed_total) {
    scratch_source.resize(compressed_total);
  }
  if (scratch_decoded.size() < decoded_total) {
    scratch_decoded.resize(decoded_total);
  }
  std::vector<EtcDecodeJob> jobs;
  VkDeviceSize compressed_offset = 0;
  VkDeviceSize decoded_offset = 0;
  for (std::size_t index = 0; index < copies.size(); ++index) {
    const Copy& copy = copies[index];
    const PendingUpload& upload = *copied_uploads[index];
    std::uint8_t* source = scratch_source.data() + compressed_offset;
    std::uint8_t* decoded = scratch_decoded.data() + decoded_offset;
    const VkDeviceSize compressed_layer = upload.compressed / upload.layers;
    const VkDeviceSize decoded_layer = upload.decoded / upload.layers;
    if (upload.source_span == copy.compressed) {
      std::memcpy(source, copy.source, copy.compressed);
    } else {
      const VkDeviceSize block_rows =
          (static_cast<VkDeviceSize>(upload.height) + 3) / 4;
      const VkDeviceSize packed_row = compressed_layer / block_rows;
      for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
        for (VkDeviceSize row = 0; row < block_rows; ++row) {
          std::memcpy(source + layer * compressed_layer + row * packed_row,
                      copy.source + layer * upload.source_layer_stride +
                          row * upload.source_row_stride,
                      packed_row);
        }
      }
    }
    for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
      EtcDecodeJob job;
      job.format = upload.format;
      job.source = source + layer * compressed_layer;
      job.source_bytes = compressed_layer;
      job.width = upload.width;
      job.height = upload.height;
      job.destination = decoded + layer * decoded_layer;
      job.destination_bytes = decoded_layer;
      jobs.push_back(job);
    }
    compressed_offset += copy.compressed;
    decoded_offset += copy.decoded;
    if (ShouldLog(&g_decode_logs, 4)) {
      std::fprintf(stderr,
                   "  [vulkan] ETC2 upload decoded %ux%u layers=%u\n",
                   upload.width, upload.height, upload.layers);
    }
  }
  for (const auto& [memory, mapping] : temporary) {
    if (mapping.dev->unmap_memory != nullptr) {
      mapping.dev->unmap_memory(mapping.dev->device, memory);
    }
  }
  const auto jobs_start = Clock::now();
  DecodeEtcJobs(jobs.data(), jobs.size(),
                options.workers != 0 ? options.workers : DecodeWorkerCount());
  const auto jobs_end = Clock::now();
  for (const EtcDecodeJob& job : jobs) {
    if (!job.ok) {
      LogFailure("could not decode an upload layer");
    }
  }
  compressed_offset = 0;
  decoded_offset = 0;
  VkDeviceSize host_total = 0;
  for (const PendingUpload* upload : copied_uploads) {
    EmitUpload(*upload, scratch_source.data() + compressed_offset,
               scratch_decoded.data() + decoded_offset);
    compressed_offset += upload->compressed;
    decoded_offset += upload->decoded;
    host_total += upload->target_bytes;
  }
  const auto emit_end = Clock::now();
  const std::uint64_t trace_ns =
      Tracing() && !copied_uploads.empty()
          ? TraceUploads(copied_uploads, copied_types, scratch_source.data())
          : 0;
  ReleaseOversizedEtc2Scratch(&scratch_source);
  ReleaseOversizedEtc2Scratch(&scratch_decoded);
  if (copied_uploads.empty()) {
    return 0;
  }
  const std::uint64_t decode_ns =
      Nanoseconds(Clock::now() - decode_start) - trace_ns;
  // MOCKTAIL_FRAME_STATS=1: name each submit that held the render thread.
  if (options.frame_stats && decode_ns >= 20000000ULL) {
    std::fprintf(stderr,
                 "  [vulkan] ETC2 decode stall %llu ms: %zu uploads, "
                 "%.1f MiB compressed, %.1f MiB written "
                 "(prepare %lld, copy %lld, decode %lld, write %lld ms)\n",
                 static_cast<unsigned long long>(decode_ns / 1000000ULL),
                 copied_uploads.size(),
                 static_cast<double>(compressed_total) / 1048576.0,
                 static_cast<double>(host_total) / 1048576.0,
                 static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(copy_start - decode_start).count()),
                 static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(jobs_start - copy_start).count()),
                 static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(jobs_end - jobs_start).count()),
                 static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(emit_end - jobs_end).count()));
  }
  bool milestone = false;
  {
    std::lock_guard<std::mutex> lock(totals_mutex);
    const std::uint64_t before = totals.host_bytes;
    AddEtc2SubmitDecode(&totals, copied_uploads.size(), compressed_total,
                        host_total, decode_ns);
    if (options.frame_stats) {
      AddEtc2SubmitDecode(&window, copied_uploads.size(), compressed_total,
                          host_total, decode_ns);
    }
    milestone = Etc2DecodeTotalsMilestone(before, totals.host_bytes);
  }
  if (milestone) {
    LogTotals(" so far");
  }
  return static_cast<std::uint32_t>(copied_uploads.size());
}

void VulkanEtc2Emulation::PrepareSubmit(VkQueue queue, std::uint32_t count,
                                        const VkSubmitInfo* submits,
                                        VkFence fence, Etc2SubmitPlan* plan) {
  static_cast<void>(fence);
  if (plan != nullptr) {
    plan->queue = queue;
  }
  std::uint32_t upload_batches = 0;
  if (submits != nullptr) {
    for (std::uint32_t index = 0; index < count; ++index) {
      if (state_->RunCpuUploads(submits[index].pCommandBuffers,
                                submits[index].commandBufferCount) != 0) {
        ++upload_batches;
      }
    }
  }
  state_->NoteSubmit(queue, false, count, upload_batches);
  // The GPU route plans its decode passes into `plan` here (DESIGN §3.4).
}

void VulkanEtc2Emulation::PrepareSubmit2(VkQueue queue, std::uint32_t count,
                                         const VkSubmitInfo2* submits,
                                         VkFence fence, Etc2SubmitPlan* plan) {
  static_cast<void>(fence);
  if (plan != nullptr) {
    plan->queue = queue;
  }
  std::uint32_t upload_batches = 0;
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
      if (state_->RunCpuUploads(
              command_buffers.data(),
              static_cast<std::uint32_t>(command_buffers.size())) != 0) {
        ++upload_batches;
      }
    }
  }
  state_->NoteSubmit(queue, true, count, upload_batches);
  // The GPU route plans its decode passes into `plan` here (DESIGN §3.4).
}

void VulkanEtc2Emulation::FinishSubmit(
    Etc2SubmitPlan* plan, VkResult host_result,
    const std::function<VkResult(VkFence)>& submit_fence_only) {
  if (plan == nullptr) {
    return;
  }
  if (plan->trailing_fence != VK_NULL_HANDLE && host_result == VK_SUCCESS &&
      submit_fence_only) {
    submit_fence_only(plan->trailing_fence);
  }
  // The GPU route marks its slot submitted, or not, here (DESIGN C3.3).
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
  if (!state_->options.frame_stats) {
    return;
  }
  const Clock::time_point now = Clock::now();
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
