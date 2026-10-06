#include "support/fake_vulkan_device.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "mocktail/graphics/etc2_decoder.h"

namespace mocktail::graphics::testing {

bool LookupFormatBlock(VkFormat format, FormatBlock* out) {
  FormatBlock block;
  switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
      block = {1, 1, 4};
      break;
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R8G8_UNORM:
      block = {1, 1, 2};
      break;
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_UINT:
      block = {1, 1, 1};
      break;
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      block = {1, 1, 8};
      break;
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      block = {1, 1, 16};
      break;
    case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
    case VK_FORMAT_EAC_R11_UNORM_BLOCK:
    case VK_FORMAT_EAC_R11_SNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
      block = {4, 4, 8};
      break;
    case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
    case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
    case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
    case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
      block = {4, 4, 16};
      break;
    default:
      return false;
  }
  if (out != nullptr) {
    *out = block;
  }
  return true;
}

VkPhysicalDeviceMemoryProperties Rtx3060MemoryProperties() {
  VkPhysicalDeviceMemoryProperties memory{};
  memory.memoryHeapCount = 2;
  memory.memoryHeaps[0] = {6ULL << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
  memory.memoryHeaps[1] = {15ULL << 30, 0};
  constexpr VkMemoryPropertyFlags kHostCoherent =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  memory.memoryTypeCount = 6;
  memory.memoryTypes[0] = {0, 1};
  memory.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
  memory.memoryTypes[2] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
  memory.memoryTypes[3] = {kHostCoherent, 1};
  memory.memoryTypes[4] = {kHostCoherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                           1};
  memory.memoryTypes[5] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | kHostCoherent,
                           0};
  return memory;
}

std::vector<VkQueueFamilyProperties> Rtx3060QueueFamilies() {
  constexpr VkQueueFlags kAll = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                                VK_QUEUE_TRANSFER_BIT |
                                VK_QUEUE_SPARSE_BINDING_BIT;
  return {
      {kAll, 16, 64, {1, 1, 1}},
      {VK_QUEUE_TRANSFER_BIT, 2, 64, {1, 1, 1}},
      {VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 8, 64, {1, 1, 1}},
  };
}

VkPhysicalDeviceProperties Rtx3060Properties() {
  VkPhysicalDeviceProperties properties{};
  properties.apiVersion = VK_API_VERSION_1_3;
  properties.vendorID = 0x10de;
  properties.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  std::snprintf(properties.deviceName, sizeof(properties.deviceName),
                "Mocktail fake Vulkan device (RTX 3060 Laptop layout)");
  VkPhysicalDeviceLimits& limits = properties.limits;
  limits.maxImageDimension2D = 32768;
  limits.maxImageArrayLayers = 2048;
  limits.maxStorageBufferRange = 0xffffffffU;
  limits.maxMemoryAllocationCount = 4294967295U;
  limits.bufferImageGranularity = 1024;
  limits.maxComputeWorkGroupCount[0] = 0x7fffffffU;
  limits.maxComputeWorkGroupCount[1] = 65535;
  limits.maxComputeWorkGroupCount[2] = 65535;
  limits.maxComputeWorkGroupInvocations = 1024;
  limits.minStorageBufferOffsetAlignment = 16;
  limits.optimalBufferCopyOffsetAlignment = 1;
  limits.optimalBufferCopyRowPitchAlignment = 1;
  limits.nonCoherentAtomSize = 64;
  limits.timestampComputeAndGraphics = VK_TRUE;
  limits.timestampPeriod = 1.0F;
  return properties;
}

FakeVulkanDevice::Config::Config()
    : properties(Rtx3060Properties()),
      memory(Rtx3060MemoryProperties()),
      queue_families(Rtx3060QueueFamilies()) {}

namespace {

std::atomic<std::uintptr_t> g_next_handle{0x10000};

template <typename Handle>
Handle NextHandle() {
  const std::uintptr_t value = g_next_handle.fetch_add(0x10);
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<Handle>(value);
  } else {
    return static_cast<Handle>(value);
  }
}

template <typename Handle>
std::uintptr_t HandleKey(Handle handle) {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<std::uintptr_t>(handle);
  } else {
    return static_cast<std::uintptr_t>(handle);
  }
}

std::uint32_t MipExtent(std::uint32_t base, std::uint32_t mip) {
  return std::max<std::uint32_t>(1, mip >= 32 ? 0 : base >> mip);
}

std::uint32_t DivideUp(std::uint32_t value, std::uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

std::string Describe(const char* function, const std::string& what,
                     const char* vuid) {
  std::string message = std::string(function) + ": " + what;
  if (vuid != nullptr) {
    message += " (" + std::string(vuid) + ")";
  }
  return message;
}

}  // namespace

struct FakeVulkanDevice::Impl {
  struct Memory {
    std::vector<std::uint8_t> bytes;
    std::uint32_t type = 0;
    bool mapped = false;
    VkDeviceSize map_offset = 0;
    bool entry_point = false;
  };
  struct Buffer {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    bool entry_point = false;
  };
  struct Image {
    VkImageCreateInfo info{};
    FormatBlock block;
    std::vector<std::vector<std::uint8_t>>
        subresources;  // mip * layers + layer
  };
  struct Command {
    enum class Kind {
      kCopyBufferToImage,
      kCopyImageToBuffer,
      kCopyImage,
      kBlitImage,
      kCopyBuffer,
      kExecuteCommands,
      kPipelineBarrier,
      kBindPipeline,
      kBindDescriptorSets,
      kPushConstants,
      kDispatch,
      kResetQueryPool,
      kWriteTimestamp,
    };
    Kind kind = Kind::kPipelineBarrier;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkBuffer buffer2 = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkImage image2 = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout layout2 = VK_IMAGE_LAYOUT_UNDEFINED;
    std::vector<VkBufferImageCopy> buffer_image_regions;
    std::vector<VkImageCopy> image_regions;
    std::vector<VkImageBlit> blits;
    std::vector<VkBufferCopy> buffer_regions;
    std::vector<VkCommandBuffer> secondaries;
    VkFilter filter = VK_FILTER_NEAREST;
    // Compute and queries.
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::uint32_t first = 0;  // set, push-constant offset or query
    std::uint32_t count = 0;  // queries
    std::vector<VkDescriptorSet> sets;
    std::vector<std::uint8_t> bytes;  // push constants
    std::uint32_t groups[3] = {};
    VkQueryPool queries = VK_NULL_HANDLE;
  };
  struct DescriptorPool {
    std::uint32_t max_sets = 0;
    VkDescriptorPoolCreateFlags flags = 0;
    std::vector<VkDescriptorSet> sets;
  };
  struct DescriptorSet {
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::map<std::uint32_t, VkDescriptorBufferInfo> bindings;
  };
  struct QueryPool {
    std::vector<std::uint64_t> values;
    std::vector<bool> written;
  };
  // The decode shader's Job (etc2_decode.comp), as 16 words.
  struct DecodeJob {
    std::uint32_t format, src_word, src_row_words, src_layer_words,
        blocks_wide, blocks_high, block_count, dst_word, dst_row_words,
        dst_layer_words, group_begin, pad[5];
  };
  // What a command buffer has bound while it executes.
  struct ComputeState {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet sets[2] = {};
    std::uint8_t push[128] = {};
  };
  enum class CommandBufferState { kInitial, kRecording, kExecutable, kInvalid };
  struct CommandBuffer {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    CommandBufferState state = CommandBufferState::kInitial;
    VkCommandBufferUsageFlags flags = 0;
    std::vector<Command> commands;
  };
  struct Pool {
    std::uint32_t family = 0;
    VkCommandPoolCreateFlags flags = 0;
    std::vector<VkCommandBuffer> command_buffers;
  };

  explicit Impl(Config config_in) : config(std::move(config_in)) {}

  Config config;
  VkDevice device = NextHandle<VkDevice>();
  VkPhysicalDevice physical_device = NextHandle<VkPhysicalDevice>();
  mutable std::recursive_mutex mutex;
  std::unordered_map<VkDeviceMemory, Memory> memories;
  std::unordered_map<VkBuffer, Buffer> buffers;
  std::unordered_map<VkImage, Image> images;
  std::unordered_map<VkImageView, VkImageViewCreateInfo> views;
  std::unordered_map<VkCommandPool, Pool> pools;
  std::unordered_map<VkCommandBuffer, CommandBuffer> command_buffers;
  std::unordered_map<VkFence, bool> fences;
  std::unordered_map<VkShaderModule, std::size_t> shader_modules;
  std::unordered_map<VkPipeline, VkShaderModule> pipelines;
  std::unordered_map<VkPipelineLayout, std::uint32_t> pipeline_layouts;
  std::unordered_map<VkDescriptorSetLayout, std::uint32_t> set_layouts;
  std::unordered_map<VkDescriptorPool, DescriptorPool> descriptor_pools;
  std::unordered_map<VkDescriptorSet, DescriptorSet> descriptor_sets;
  std::unordered_map<VkQueryPool, QueryPool> query_pools;
  std::uint64_t fake_time = 0;
  std::vector<VkResult> submit_results;
  bool hold_fences = false;
  std::vector<VkFence> held_fences;
  bool lost = false;
  std::map<std::pair<std::uint32_t, std::uint32_t>, VkQueue> queues;
  Counts counts;
  std::map<std::string, std::uint64_t> calls;
  std::vector<ImageTransfer> transfers;
  std::uint64_t copy_regions = 0;
  std::vector<std::string> errors;

  void Error(const char* function, const std::string& what,
             const char* vuid = nullptr) {
    errors.push_back(Describe(function, what, vuid));
  }

  // ---- objects -----------------------------------------------------------

  VkResult CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        bool entry_point, VkBuffer* out) {
    if (out == nullptr || size == 0) {
      Error("vkCreateBuffer", "size 0 or no output",
            "VUID-VkBufferCreateInfo-size-00912");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    const VkBuffer buffer = NextHandle<VkBuffer>();
    buffers[buffer] = {size, usage, VK_NULL_HANDLE, 0, entry_point};
    ++counts.buffers.created;
    if (entry_point) {
      ++counts.entry_point_buffers.created;
    }
    *out = buffer;
    return VK_SUCCESS;
  }

  void Requirements(VkBuffer buffer, VkMemoryRequirements* out) {
    const auto found = buffers.find(buffer);
    if (found == buffers.end() || out == nullptr) {
      Error("vkGetBufferMemoryRequirements", "unknown buffer");
      return;
    }
    const VkDeviceSize alignment = config.buffer_alignment;
    out->size = (found->second.size + alignment - 1) / alignment * alignment;
    out->alignment = alignment;
    out->memoryTypeBits = config.buffer_memory_type_bits;
  }

  VkResult AllocateMemory(VkDeviceSize size, std::uint32_t type,
                          bool entry_point, VkDeviceMemory* out) {
    if (out == nullptr || size == 0 || type >= config.memory.memoryTypeCount) {
      Error("vkAllocateMemory", "size 0 or invalid memory type",
            "VUID-vkAllocateMemory-pAllocateInfo-01714");
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    if (config.max_allocation_bytes != 0 &&
        size > config.max_allocation_bytes) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    const VkDeviceMemory memory = NextHandle<VkDeviceMemory>();
    Memory& record = memories[memory];
    record.bytes.assign(static_cast<std::size_t>(size), config.memory_fill);
    record.type = type;
    record.entry_point = entry_point;
    ++counts.memory.created;
    if (entry_point) {
      ++counts.entry_point_memory.created;
    }
    *out = memory;
    return VK_SUCCESS;
  }

  VkResult BindBuffer(VkBuffer buffer, VkDeviceMemory memory,
                      VkDeviceSize offset) {
    const auto found = buffers.find(buffer);
    const auto backing = memories.find(memory);
    if (found == buffers.end() || backing == memories.end()) {
      Error("vkBindBufferMemory", "unknown buffer or memory");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (found->second.memory != VK_NULL_HANDLE) {
      Error("vkBindBufferMemory", "buffer already bound",
            "VUID-vkBindBufferMemory-buffer-07459");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if ((config.buffer_memory_type_bits & (1U << backing->second.type)) == 0) {
      Error("vkBindBufferMemory",
            "memory type " + std::to_string(backing->second.type) +
                " not allowed for buffers",
            "VUID-vkBindBufferMemory-memory-01035");
    }
    if (offset % config.buffer_alignment != 0) {
      Error("vkBindBufferMemory", "offset not aligned",
            "VUID-vkBindBufferMemory-memoryOffset-01036");
    }
    if (offset + found->second.size > backing->second.bytes.size()) {
      Error("vkBindBufferMemory", "buffer does not fit the memory",
            "VUID-vkBindBufferMemory-size-01037");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    found->second.memory = memory;
    found->second.offset = offset;
    return VK_SUCCESS;
  }

  VkResult Map(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
               void** data) {
    const auto found = memories.find(memory);
    if (found == memories.end() || data == nullptr) {
      Error("vkMapMemory", "unknown memory");
      return VK_ERROR_MEMORY_MAP_FAILED;
    }
    Memory& record = found->second;
    if ((config.memory.memoryTypes[record.type].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
      Error("vkMapMemory", "memory is not host-visible",
            "VUID-vkMapMemory-memory-00682");
      return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (record.mapped) {
      Error("vkMapMemory", "memory is already mapped",
            "VUID-vkMapMemory-memory-00678");
      return VK_ERROR_MEMORY_MAP_FAILED;
    }
    const VkDeviceSize total = record.bytes.size();
    if (offset >= total ||
        (size != VK_WHOLE_SIZE && (size == 0 || offset + size > total))) {
      Error("vkMapMemory", "range outside the allocation",
            "VUID-vkMapMemory-offset-00679");
      return VK_ERROR_MEMORY_MAP_FAILED;
    }
    record.mapped = true;
    record.map_offset = offset;
    ++counts.maps;
    ++counts.live_mappings;
    *data = record.bytes.data() + offset;
    return VK_SUCCESS;
  }

  void Unmap(VkDeviceMemory memory) {
    const auto found = memories.find(memory);
    if (found == memories.end() || !found->second.mapped) {
      Error("vkUnmapMemory", "memory is not mapped",
            "VUID-vkUnmapMemory-memory-00689");
      return;
    }
    found->second.mapped = false;
    ++counts.unmaps;
    --counts.live_mappings;
  }

  void Free(VkDeviceMemory memory) {
    if (memory == VK_NULL_HANDLE) {
      return;
    }
    const auto found = memories.find(memory);
    if (found == memories.end()) {
      Error("vkFreeMemory", "unknown or already freed memory");
      return;
    }
    if (found->second.mapped) {
      --counts.live_mappings;  // freeing unmaps implicitly
    }
    ++counts.memory.destroyed;
    if (found->second.entry_point) {
      ++counts.entry_point_memory.destroyed;
    }
    memories.erase(found);
  }

  void DestroyBuffer(VkBuffer buffer) {
    if (buffer == VK_NULL_HANDLE) {
      return;
    }
    const auto found = buffers.find(buffer);
    if (found == buffers.end()) {
      Error("vkDestroyBuffer", "unknown or already destroyed buffer");
      return;
    }
    ++counts.buffers.destroyed;
    if (found->second.entry_point) {
      ++counts.entry_point_buffers.destroyed;
    }
    buffers.erase(found);
  }

  VkResult CreateImage(const VkImageCreateInfo& info, VkImage* out) {
    FormatBlock block;
    if (out == nullptr || !LookupFormatBlock(info.format, &block)) {
      Error("vkCreateImage", "format " +
                                 std::to_string(static_cast<int>(info.format)) +
                                 " is not modelled");
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    if (info.imageType != VK_IMAGE_TYPE_2D || info.extent.depth != 1 ||
        info.extent.width == 0 || info.extent.height == 0 ||
        info.mipLevels == 0 || info.arrayLayers == 0) {
      Error("vkCreateImage", "only 2D images with a non-empty extent");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::uint32_t largest = std::max(info.extent.width, info.extent.height);
    std::uint32_t levels = 0;
    while (largest != 0) {
      ++levels;
      largest >>= 1;
    }
    if (info.mipLevels > levels) {
      Error("vkCreateImage", "too many mip levels",
            "VUID-VkImageCreateInfo-mipLevels-00958");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    const VkImage image = NextHandle<VkImage>();
    Image& record = images[image];
    record.info = info;
    record.info.pNext = nullptr;
    record.info.pQueueFamilyIndices = nullptr;
    record.block = block;
    for (std::uint32_t mip = 0; mip < info.mipLevels; ++mip) {
      const std::size_t bytes =
          static_cast<std::size_t>(
              DivideUp(MipExtent(info.extent.width, mip), block.width)) *
          DivideUp(MipExtent(info.extent.height, mip), block.height) *
          block.bytes;
      for (std::uint32_t layer = 0; layer < info.arrayLayers; ++layer) {
        record.subresources.emplace_back(bytes, config.image_fill);
      }
    }
    ++counts.images.created;
    *out = image;
    return VK_SUCCESS;
  }

  void DestroyImage(VkImage image) {
    if (image == VK_NULL_HANDLE) {
      return;
    }
    if (images.erase(image) == 0) {
      Error("vkDestroyImage", "unknown or already destroyed image");
      return;
    }
    ++counts.images.destroyed;
  }

  VkResult CreateView(const VkImageViewCreateInfo* info, VkImageView* out) {
    if (info == nullptr || out == nullptr || images.count(info->image) == 0) {
      Error("vkCreateImageView", "unknown image");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    FormatBlock view_block;
    if (!LookupFormatBlock(info->format, &view_block)) {
      Error("vkCreateImageView", "view format is not modelled");
    }
    const VkImageView view = NextHandle<VkImageView>();
    views[view] = *info;
    views[view].pNext = nullptr;
    ++counts.image_views.created;
    *out = view;
    return VK_SUCCESS;
  }

  void DestroyView(VkImageView view) {
    if (view == VK_NULL_HANDLE) {
      return;
    }
    if (views.erase(view) == 0) {
      Error("vkDestroyImageView", "unknown image view");
      return;
    }
    ++counts.image_views.destroyed;
  }

  // ---- command buffers ---------------------------------------------------

  VkCommandPool CreatePool(std::uint32_t family,
                           VkCommandPoolCreateFlags flags) {
    if (family >= config.queue_families.size()) {
      Error("vkCreateCommandPool", "no such queue family",
            "VUID-vkCreateCommandPool-queueFamilyIndex-01937");
    }
    const VkCommandPool pool = NextHandle<VkCommandPool>();
    pools[pool] = {family, flags, {}};
    ++counts.command_pools.created;
    return pool;
  }

  void FreeCommandBuffer(VkCommandBuffer command_buffer);

  void DestroyPool(VkCommandPool pool) {
    if (pool == VK_NULL_HANDLE) {
      return;
    }
    const auto found = pools.find(pool);
    if (found == pools.end()) {
      Error("vkDestroyCommandPool", "unknown command pool");
      return;
    }
    const std::vector<VkCommandBuffer> owned = found->second.command_buffers;
    for (VkCommandBuffer command_buffer : owned) {
      FreeCommandBuffer(command_buffer);
    }
    pools.erase(pool);
    ++counts.command_pools.destroyed;
  }

  VkCommandBuffer Allocate(VkCommandPool pool, VkCommandBufferLevel level);

  CommandBuffer* Recording(VkCommandBuffer command_buffer,
                           const char* function) {
    const auto found = command_buffers.find(command_buffer);
    if (found == command_buffers.end()) {
      Error(function, "unknown command buffer");
      return nullptr;
    }
    if (found->second.state != CommandBufferState::kRecording) {
      Error(function, "command buffer is not recording",
            "VUID-vkCmdCopyBufferToImage-commandBuffer-recording");
      return nullptr;
    }
    return &found->second;
  }

  bool PoolAllowsReset(const CommandBuffer& record) const {
    const auto pool = pools.find(record.pool);
    return pool != pools.end() &&
           (pool->second.flags &
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) != 0;
  }

  VkQueueFlags FamilyFlags(const CommandBuffer& record) const {
    const auto pool = pools.find(record.pool);
    if (pool == pools.end() ||
        pool->second.family >= config.queue_families.size()) {
      return 0;
    }
    return config.queue_families[pool->second.family].queueFlags;
  }

  VkResult Begin(VkCommandBuffer command_buffer,
                 VkCommandBufferUsageFlags flags) {
    const auto found = command_buffers.find(command_buffer);
    if (found == command_buffers.end()) {
      Error("vkBeginCommandBuffer", "unknown command buffer");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    CommandBuffer& record = found->second;
    if (record.state == CommandBufferState::kRecording) {
      Error("vkBeginCommandBuffer", "command buffer is already recording",
            "VUID-vkBeginCommandBuffer-commandBuffer-00049");
    } else if (record.state != CommandBufferState::kInitial &&
               !PoolAllowsReset(record)) {
      Error("vkBeginCommandBuffer",
            "implicit reset from a pool without RESET_COMMAND_BUFFER",
            "VUID-vkBeginCommandBuffer-commandBuffer-00050");
    }
    record.commands.clear();
    record.flags = flags;
    record.state = CommandBufferState::kRecording;
    return VK_SUCCESS;
  }

  VkResult End(VkCommandBuffer command_buffer) {
    CommandBuffer* record = Recording(command_buffer, "vkEndCommandBuffer");
    if (record == nullptr) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    record->state = CommandBufferState::kExecutable;
    return VK_SUCCESS;
  }

  VkResult Reset(VkCommandBuffer command_buffer) {
    const auto found = command_buffers.find(command_buffer);
    if (found == command_buffers.end()) {
      Error("vkResetCommandBuffer", "unknown command buffer");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (!PoolAllowsReset(found->second)) {
      Error("vkResetCommandBuffer",
            "pool was created without RESET_COMMAND_BUFFER",
            "VUID-vkResetCommandBuffer-commandBuffer-00046");
    }
    found->second.commands.clear();
    found->second.state = CommandBufferState::kInitial;
    return VK_SUCCESS;
  }

  // ---- recording ---------------------------------------------------------

  // Checks one buffer<->image region against the image and the buffer size.
  bool ValidRegion(const char* function, VkImage image_handle,
                   VkBuffer buffer_handle, const VkBufferImageCopy& region,
                   VkQueueFlags family_flags, bool report) {
    const auto image = images.find(image_handle);
    const auto buffer = buffers.find(buffer_handle);
    auto fail = [&](const std::string& what, const char* vuid) {
      if (report) {
        Error(function, what, vuid);
      }
      return false;
    };
    if (image == images.end()) {
      return fail("unknown image", nullptr);
    }
    if (buffer == buffers.end()) {
      return fail("unknown buffer", nullptr);
    }
    const VkImageCreateInfo& info = image->second.info;
    const FormatBlock& block = image->second.block;
    const VkImageSubresourceLayers& sub = region.imageSubresource;
    if (sub.mipLevel >= info.mipLevels) {
      return fail("mip level out of range",
                  "VUID-vkCmdCopyBufferToImage-imageSubresource-07967");
    }
    if (sub.layerCount == 0 || sub.layerCount == VK_REMAINING_ARRAY_LAYERS ||
        sub.baseArrayLayer + sub.layerCount > info.arrayLayers) {
      return fail("array layers out of range",
                  "VUID-vkCmdCopyBufferToImage-imageSubresource-07968");
    }
    const std::uint32_t mip_width = MipExtent(info.extent.width, sub.mipLevel);
    const std::uint32_t mip_height =
        MipExtent(info.extent.height, sub.mipLevel);
    const VkOffset3D& offset = region.imageOffset;
    const VkExtent3D& extent = region.imageExtent;
    if (offset.x < 0 || offset.y < 0 || offset.z != 0 || extent.depth != 1 ||
        extent.width == 0 || extent.height == 0) {
      return fail("bad offset or extent", "VUID-VkBufferImageCopy-imageExtent");
    }
    const std::uint32_t x = static_cast<std::uint32_t>(offset.x);
    const std::uint32_t y = static_cast<std::uint32_t>(offset.y);
    if (x + extent.width > mip_width || y + extent.height > mip_height) {
      return fail("region " + std::to_string(x) + "," + std::to_string(y) +
                      " " + std::to_string(extent.width) + "x" +
                      std::to_string(extent.height) + " exceeds mip " +
                      std::to_string(mip_width) + "x" +
                      std::to_string(mip_height),
                  "VUID-vkCmdCopyBufferToImage-pRegions-06218");
    }
    if (x % block.width != 0 || y % block.height != 0 ||
        (extent.width % block.width != 0 && x + extent.width != mip_width) ||
        (extent.height % block.height != 0 &&
         y + extent.height != mip_height)) {
      return fail("region not aligned to the texel block",
                  "VUID-vkCmdCopyBufferToImage-pRegions-07274");
    }
    if ((region.bufferRowLength != 0 &&
         (region.bufferRowLength < extent.width ||
          region.bufferRowLength % block.width != 0)) ||
        (region.bufferImageHeight != 0 &&
         (region.bufferImageHeight < extent.height ||
          region.bufferImageHeight % block.height != 0))) {
      return fail("bufferRowLength or bufferImageHeight too small",
                  "VUID-VkBufferImageCopy-bufferRowLength-09101");
    }
    // Alignment rules are reported when recording; executing only needs the
    // region to stay inside the image and the buffer.
    if (report && region.bufferOffset % block.bytes != 0) {
      return fail("bufferOffset " + std::to_string(region.bufferOffset) +
                      " not a multiple of the texel block size",
                  "VUID-vkCmdCopyBufferToImage-dstImage-07975");
    }
    if (report &&
        (family_flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0 &&
        region.bufferOffset % 4 != 0) {
      return fail("bufferOffset " + std::to_string(region.bufferOffset) +
                      " not a multiple of 4 on a transfer-only queue",
                  "VUID-vkCmdCopyBufferToImage-commandBuffer-07737");
    }
    const std::uint64_t row_blocks = DivideUp(
        region.bufferRowLength != 0 ? region.bufferRowLength : extent.width,
        block.width);
    const std::uint64_t layer_rows =
        DivideUp(region.bufferImageHeight != 0 ? region.bufferImageHeight
                                               : extent.height,
                 block.height);
    const std::uint64_t blocks_wide = DivideUp(extent.width, block.width);
    const std::uint64_t blocks_high = DivideUp(extent.height, block.height);
    const std::uint64_t end =
        region.bufferOffset + ((sub.layerCount - 1) * layer_rows * row_blocks +
                               (blocks_high - 1) * row_blocks + blocks_wide) *
                                  block.bytes;
    if (end > buffer->second.size) {
      return fail("region reads past the end of the buffer (" +
                      std::to_string(end) + " > " +
                      std::to_string(buffer->second.size) + ")",
                  "VUID-vkCmdCopyBufferToImage-pRegions-00171");
    }
    return true;
  }

  void RecordCopyBufferToImage(const char* function,
                               VkCommandBuffer command_buffer, VkBuffer source,
                               VkImage destination, VkImageLayout layout,
                               std::vector<VkBufferImageCopy> regions) {
    CommandBuffer* record = Recording(command_buffer, function);
    if (record == nullptr) {
      return;
    }
    const auto buffer = buffers.find(source);
    if (buffer != buffers.end() &&
        (buffer->second.usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0) {
      Error(function, "source buffer lacks TRANSFER_SRC usage",
            "VUID-vkCmdCopyBufferToImage-srcBuffer-00174");
    }
    if (layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
        layout != VK_IMAGE_LAYOUT_GENERAL) {
      Error(function, "destination layout",
            "VUID-vkCmdCopyBufferToImage-dstImageLayout-01396");
    }
    if (regions.empty()) {
      Error(function, "no regions",
            "VUID-vkCmdCopyBufferToImage-regionCount-arraylength");
    }
    const VkQueueFlags flags = FamilyFlags(*record);
    for (const VkBufferImageCopy& region : regions) {
      ValidRegion(function, destination, source, region, flags, true);
    }
    Command command;
    command.kind = Command::Kind::kCopyBufferToImage;
    command.buffer = source;
    command.image = destination;
    command.layout = layout;
    command.buffer_image_regions = std::move(regions);
    record->commands.push_back(std::move(command));
  }

  bool ValidImageRegion(const char* function, VkImage image_handle,
                        const VkImageSubresourceLayers& sub,
                        const VkOffset3D& offset, const VkExtent3D& extent,
                        bool report) {
    const auto image = images.find(image_handle);
    auto fail = [&](const std::string& what) {
      if (report) {
        Error(function, what);
      }
      return false;
    };
    if (image == images.end()) {
      return fail("unknown image");
    }
    const VkImageCreateInfo& info = image->second.info;
    if (sub.mipLevel >= info.mipLevels || sub.layerCount == 0 ||
        sub.baseArrayLayer + sub.layerCount > info.arrayLayers) {
      return fail("subresource out of range");
    }
    if (offset.x < 0 || offset.y < 0 || offset.z != 0 || extent.depth != 1 ||
        static_cast<std::uint32_t>(offset.x) + extent.width >
            MipExtent(info.extent.width, sub.mipLevel) ||
        static_cast<std::uint32_t>(offset.y) + extent.height >
            MipExtent(info.extent.height, sub.mipLevel)) {
      return fail("region outside the image");
    }
    return true;
  }

  void RecordCopyImage(const char* function, VkCommandBuffer command_buffer,
                       VkImage source, VkImageLayout source_layout,
                       VkImage destination, VkImageLayout destination_layout,
                       std::vector<VkImageCopy> regions) {
    CommandBuffer* record = Recording(command_buffer, function);
    if (record == nullptr) {
      return;
    }
    const auto src = images.find(source);
    const auto dst = images.find(destination);
    if (src != images.end() && dst != images.end() &&
        (src->second.block.width != dst->second.block.width ||
         src->second.block.height != dst->second.block.height ||
         src->second.block.bytes != dst->second.block.bytes)) {
      Error(function, "formats are not size-compatible",
            "VUID-vkCmdCopyImage-srcImage-01548");
    }
    for (const VkImageCopy& region : regions) {
      ValidImageRegion(function, source, region.srcSubresource,
                       region.srcOffset, region.extent, true);
      ValidImageRegion(function, destination, region.dstSubresource,
                       region.dstOffset, region.extent, true);
      if (region.srcSubresource.layerCount !=
          region.dstSubresource.layerCount) {
        Error(function, "layer counts differ");
      }
    }
    Command command;
    command.kind = Command::Kind::kCopyImage;
    command.image = source;
    command.layout = source_layout;
    command.image2 = destination;
    command.layout2 = destination_layout;
    command.image_regions = std::move(regions);
    record->commands.push_back(std::move(command));
  }

  bool ValidBlitOffsets(const char* function, VkImage image_handle,
                        const VkImageSubresourceLayers& sub,
                        const VkOffset3D* offsets) {
    const auto image = images.find(image_handle);
    if (image == images.end()) {
      Error(function, "unknown image");
      return false;
    }
    const VkImageCreateInfo& info = image->second.info;
    if (image->second.block.width != 1) {
      Error(function, "blit of a block-compressed image",
            "VUID-vkCmdBlitImage-srcImage-06421");
      return false;
    }
    if (sub.mipLevel >= info.mipLevels || sub.layerCount == 0 ||
        sub.baseArrayLayer + sub.layerCount > info.arrayLayers) {
      Error(function, "subresource out of range");
      return false;
    }
    const std::int32_t width =
        static_cast<std::int32_t>(MipExtent(info.extent.width, sub.mipLevel));
    const std::int32_t height =
        static_cast<std::int32_t>(MipExtent(info.extent.height, sub.mipLevel));
    for (int index = 0; index < 2; ++index) {
      if (offsets[index].x < 0 || offsets[index].x > width ||
          offsets[index].y < 0 || offsets[index].y > height ||
          offsets[index].z < 0 || offsets[index].z > 1) {
        Error(function, "blit offsets outside the image",
              "VUID-vkCmdBlitImage-srcOffset-00243");
        return false;
      }
    }
    return true;
  }

  void RecordBlit(VkCommandBuffer command_buffer, VkImage source,
                  VkImageLayout source_layout, VkImage destination,
                  VkImageLayout destination_layout,
                  std::vector<VkImageBlit> blits, VkFilter filter) {
    CommandBuffer* record = Recording(command_buffer, "vkCmdBlitImage");
    if (record == nullptr) {
      return;
    }
    if ((FamilyFlags(*record) & VK_QUEUE_GRAPHICS_BIT) == 0) {
      Error("vkCmdBlitImage", "blit on a queue without GRAPHICS",
            "VUID-vkCmdBlitImage-commandBuffer-cmdpool");
    }
    for (const VkImageBlit& blit : blits) {
      ValidBlitOffsets("vkCmdBlitImage", source, blit.srcSubresource,
                       blit.srcOffsets);
      ValidBlitOffsets("vkCmdBlitImage", destination, blit.dstSubresource,
                       blit.dstOffsets);
    }
    Command command;
    command.kind = Command::Kind::kBlitImage;
    command.image = source;
    command.layout = source_layout;
    command.image2 = destination;
    command.layout2 = destination_layout;
    command.blits = std::move(blits);
    command.filter = filter;
    record->commands.push_back(std::move(command));
  }

  void RecordExecute(VkCommandBuffer command_buffer, std::uint32_t count,
                     const VkCommandBuffer* secondaries) {
    CommandBuffer* record = Recording(command_buffer, "vkCmdExecuteCommands");
    if (record == nullptr) {
      return;
    }
    if (record->level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
      Error("vkCmdExecuteCommands", "called on a secondary",
            "VUID-vkCmdExecuteCommands-bufferlevel");
    }
    Command command;
    command.kind = Command::Kind::kExecuteCommands;
    for (std::uint32_t index = 0; index < count; ++index) {
      const auto secondary = command_buffers.find(secondaries[index]);
      if (secondary == command_buffers.end() ||
          secondary->second.level != VK_COMMAND_BUFFER_LEVEL_SECONDARY ||
          secondary->second.state != CommandBufferState::kExecutable) {
        Error("vkCmdExecuteCommands",
              "secondary is unknown, primary or not executable",
              "VUID-vkCmdExecuteCommands-pCommandBuffers-00089");
        continue;
      }
      command.secondaries.push_back(secondaries[index]);
    }
    record->commands.push_back(std::move(command));
  }

  // ---- execution ---------------------------------------------------------

  std::uint8_t* BufferBytes(VkBuffer handle, const char* function) {
    const auto buffer = buffers.find(handle);
    if (buffer == buffers.end()) {
      Error(function, "buffer was destroyed before the submit");
      return nullptr;
    }
    const auto memory = memories.find(buffer->second.memory);
    if (memory == memories.end()) {
      Error(function, "buffer has no live memory at submit",
            "VUID-vkCmdCopyBufferToImage-srcBuffer-00176");
      return nullptr;
    }
    return memory->second.bytes.data() + buffer->second.offset;
  }

  std::vector<std::uint8_t>* Subresource(VkImage handle, std::uint32_t mip,
                                         std::uint32_t layer) {
    const auto image = images.find(handle);
    if (image == images.end() || mip >= image->second.info.mipLevels ||
        layer >= image->second.info.arrayLayers) {
      return nullptr;
    }
    return &image->second
                .subresources[mip * image->second.info.arrayLayers + layer];
  }

  // Moves blocks between a buffer and an image subresource. `to_image` picks
  // the direction.
  void CopyBlocks(VkBuffer buffer_handle, VkImage image_handle,
                  const VkBufferImageCopy& region, bool to_image,
                  const char* function) {
    std::uint8_t* buffer = BufferBytes(buffer_handle, function);
    const auto image = images.find(image_handle);
    if (buffer == nullptr || image == images.end()) {
      return;
    }
    const FormatBlock& block = image->second.block;
    const VkImageCreateInfo& info = image->second.info;
    const VkImageSubresourceLayers& sub = region.imageSubresource;
    const std::uint32_t mip_blocks_wide =
        DivideUp(MipExtent(info.extent.width, sub.mipLevel), block.width);
    const std::uint32_t row_blocks =
        DivideUp(region.bufferRowLength != 0 ? region.bufferRowLength
                                             : region.imageExtent.width,
                 block.width);
    const std::uint32_t layer_rows =
        DivideUp(region.bufferImageHeight != 0 ? region.bufferImageHeight
                                               : region.imageExtent.height,
                 block.height);
    const std::uint32_t blocks_wide =
        DivideUp(region.imageExtent.width, block.width);
    const std::uint32_t blocks_high =
        DivideUp(region.imageExtent.height, block.height);
    const std::uint32_t block_x =
        static_cast<std::uint32_t>(region.imageOffset.x) / block.width;
    const std::uint32_t block_y =
        static_cast<std::uint32_t>(region.imageOffset.y) / block.height;
    for (std::uint32_t layer = 0; layer < sub.layerCount; ++layer) {
      std::vector<std::uint8_t>* texels =
          Subresource(image_handle, sub.mipLevel, sub.baseArrayLayer + layer);
      for (std::uint32_t row = 0; row < blocks_high; ++row) {
        std::uint8_t* memory =
            buffer + region.bufferOffset +
            (static_cast<std::size_t>(layer) * layer_rows + row) * row_blocks *
                block.bytes;
        std::uint8_t* image_row =
            texels->data() +
            (static_cast<std::size_t>(block_y + row) * mip_blocks_wide +
             block_x) *
                block.bytes;
        const std::size_t bytes =
            static_cast<std::size_t>(blocks_wide) * block.bytes;
        if (to_image) {
          std::memcpy(image_row, memory, bytes);
        } else {
          std::memcpy(memory, image_row, bytes);
        }
      }
    }
  }

  void CopyImageRegion(VkImage source, VkImage destination,
                       const VkImageCopy& region) {
    const auto src = images.find(source);
    const auto dst = images.find(destination);
    if (src == images.end() || dst == images.end()) {
      Error("vkCmdCopyImage", "image destroyed before the submit");
      return;
    }
    const FormatBlock& block = src->second.block;
    const std::uint32_t src_wide =
        DivideUp(MipExtent(src->second.info.extent.width,
                           region.srcSubresource.mipLevel),
                 block.width);
    const std::uint32_t dst_wide =
        DivideUp(MipExtent(dst->second.info.extent.width,
                           region.dstSubresource.mipLevel),
                 block.width);
    const std::uint32_t blocks_wide =
        DivideUp(region.extent.width, block.width);
    const std::uint32_t blocks_high =
        DivideUp(region.extent.height, block.height);
    const std::uint32_t sx =
        static_cast<std::uint32_t>(region.srcOffset.x) / block.width;
    const std::uint32_t sy =
        static_cast<std::uint32_t>(region.srcOffset.y) / block.height;
    const std::uint32_t dx =
        static_cast<std::uint32_t>(region.dstOffset.x) / block.width;
    const std::uint32_t dy =
        static_cast<std::uint32_t>(region.dstOffset.y) / block.height;
    for (std::uint32_t layer = 0; layer < region.srcSubresource.layerCount;
         ++layer) {
      // Copy through a temporary so an overlapping copy within one image
      // reads the old texels.
      const std::vector<std::uint8_t> from =
          *Subresource(source, region.srcSubresource.mipLevel,
                       region.srcSubresource.baseArrayLayer + layer);
      std::vector<std::uint8_t>* to =
          Subresource(destination, region.dstSubresource.mipLevel,
                      region.dstSubresource.baseArrayLayer + layer);
      for (std::uint32_t row = 0; row < blocks_high; ++row) {
        std::memcpy(to->data() +
                        ((static_cast<std::size_t>(dy) + row) * dst_wide + dx) *
                            block.bytes,
                    from.data() +
                        ((static_cast<std::size_t>(sy) + row) * src_wide + sx) *
                            block.bytes,
                    static_cast<std::size_t>(blocks_wide) * block.bytes);
      }
    }
  }

  // Same-size, unflipped NEAREST blits are exact copies; anything filtered
  // or scaled is recorded but leaves the destination alone.
  bool BlitIsCopy(const VkImageBlit& blit, VkFilter filter, VkImage source,
                  VkImage destination) const {
    const auto src = images.find(source);
    const auto dst = images.find(destination);
    if (filter != VK_FILTER_NEAREST || src == images.end() ||
        dst == images.end() ||
        src->second.block.bytes != dst->second.block.bytes) {
      return false;
    }
    const std::int32_t src_w = blit.srcOffsets[1].x - blit.srcOffsets[0].x;
    const std::int32_t src_h = blit.srcOffsets[1].y - blit.srcOffsets[0].y;
    const std::int32_t dst_w = blit.dstOffsets[1].x - blit.dstOffsets[0].x;
    const std::int32_t dst_h = blit.dstOffsets[1].y - blit.dstOffsets[0].y;
    return src_w > 0 && src_h > 0 && src_w == dst_w && src_h == dst_h &&
           blit.srcSubresource.layerCount == blit.dstSubresource.layerCount;
  }

  void Execute(VkCommandBuffer command_buffer, int depth) {
    const auto found = command_buffers.find(command_buffer);
    if (found == command_buffers.end()) {
      Error("vkQueueSubmit", "unknown command buffer");
      return;
    }
    if (found->second.state != CommandBufferState::kExecutable) {
      Error("vkQueueSubmit", "command buffer is not executable",
            "VUID-VkSubmitInfo-pCommandBuffers-00072");
      return;
    }
    const VkQueueFlags flags = FamilyFlags(found->second);
    // Copy: an executed secondary must not invalidate the iterator.
    const std::vector<Command> commands = found->second.commands;
    ComputeState compute;
    for (const Command& command : commands) {
      switch (command.kind) {
        case Command::Kind::kBindPipeline:
          compute.pipeline = command.pipeline;
          break;
        case Command::Kind::kBindDescriptorSets:
          for (std::size_t index = 0; index < command.sets.size(); ++index) {
            if (command.first + index < 2) {
              compute.sets[command.first + index] = command.sets[index];
            }
          }
          break;
        case Command::Kind::kPushConstants:
          if (command.first + command.bytes.size() <= sizeof(compute.push)) {
            std::memcpy(compute.push + command.first, command.bytes.data(),
                        command.bytes.size());
          }
          break;
        case Command::Kind::kDispatch:
          if ((flags & VK_QUEUE_COMPUTE_BIT) == 0) {
            Error("vkCmdDispatch", "dispatch on a queue without COMPUTE",
                  "VUID-vkCmdDispatch-commandBuffer-cmdpool");
            break;
          }
          Dispatch(compute, command.groups);
          break;
        case Command::Kind::kResetQueryPool: {
          QueryPool* pool = Queries(command.queries, command.first,
                                    command.count, "vkCmdResetQueryPool");
          for (std::uint32_t query = 0; pool != nullptr && query < command.count;
               ++query) {
            pool->written[command.first + query] = false;
          }
          break;
        }
        case Command::Kind::kWriteTimestamp: {
          QueryPool* pool =
              Queries(command.queries, command.first, 1, "vkCmdWriteTimestamp");
          if (pool != nullptr) {
            fake_time += config.timestamp_step_ns;
            pool->values[command.first] = fake_time;
            pool->written[command.first] = true;
          }
          break;
        }
        case Command::Kind::kCopyBufferToImage:
          for (const VkBufferImageCopy& region : command.buffer_image_regions) {
            if (ValidRegion("vkCmdCopyBufferToImage", command.image,
                            command.buffer, region, flags, false)) {
              CopyBlocks(command.buffer, command.image, region, true,
                         "vkCmdCopyBufferToImage");
              ++copy_regions;
            }
          }
          break;
        case Command::Kind::kCopyImageToBuffer:
          for (const VkBufferImageCopy& region : command.buffer_image_regions) {
            if (ValidRegion("vkCmdCopyImageToBuffer", command.image,
                            command.buffer, region, flags, false)) {
              CopyBlocks(command.buffer, command.image, region, false,
                         "vkCmdCopyImageToBuffer");
            }
          }
          break;
        case Command::Kind::kCopyImage: {
          ImageTransfer transfer;
          transfer.executed = true;
          transfer.source = command.image;
          transfer.destination = command.image2;
          transfer.source_layout = command.layout;
          transfer.destination_layout = command.layout2;
          transfer.copies = command.image_regions;
          for (const VkImageCopy& region : command.image_regions) {
            if (ValidImageRegion("vkCmdCopyImage", command.image,
                                 region.srcSubresource, region.srcOffset,
                                 region.extent, false) &&
                ValidImageRegion("vkCmdCopyImage", command.image2,
                                 region.dstSubresource, region.dstOffset,
                                 region.extent, false)) {
              CopyImageRegion(command.image, command.image2, region);
            }
          }
          transfers.push_back(std::move(transfer));
          break;
        }
        case Command::Kind::kBlitImage: {
          ImageTransfer transfer;
          transfer.blit = true;
          transfer.executed = true;
          transfer.source = command.image;
          transfer.destination = command.image2;
          transfer.source_layout = command.layout;
          transfer.destination_layout = command.layout2;
          transfer.blits = command.blits;
          transfer.filter = command.filter;
          for (const VkImageBlit& blit : command.blits) {
            if (!BlitIsCopy(blit, command.filter, command.image,
                            command.image2)) {
              transfer.executed = false;
              continue;
            }
            VkImageCopy copy{};
            copy.srcSubresource = blit.srcSubresource;
            copy.srcOffset = {blit.srcOffsets[0].x, blit.srcOffsets[0].y, 0};
            copy.dstSubresource = blit.dstSubresource;
            copy.dstOffset = {blit.dstOffsets[0].x, blit.dstOffsets[0].y, 0};
            copy.extent = {static_cast<std::uint32_t>(blit.srcOffsets[1].x -
                                                      blit.srcOffsets[0].x),
                           static_cast<std::uint32_t>(blit.srcOffsets[1].y -
                                                      blit.srcOffsets[0].y),
                           1};
            CopyImageRegion(command.image, command.image2, copy);
          }
          transfers.push_back(std::move(transfer));
          break;
        }
        case Command::Kind::kCopyBuffer: {
          std::uint8_t* src = BufferBytes(command.buffer, "vkCmdCopyBuffer");
          std::uint8_t* dst = BufferBytes(command.buffer2, "vkCmdCopyBuffer");
          if (src == nullptr || dst == nullptr) {
            break;
          }
          for (const VkBufferCopy& region : command.buffer_regions) {
            std::memmove(dst + region.dstOffset, src + region.srcOffset,
                         static_cast<std::size_t>(region.size));
          }
          break;
        }
        case Command::Kind::kExecuteCommands:
          if (depth > 0) {
            Error("vkCmdExecuteCommands", "nested secondaries");
            break;
          }
          for (VkCommandBuffer secondary : command.secondaries) {
            Execute(secondary, depth + 1);
          }
          break;
        case Command::Kind::kPipelineBarrier:
          break;
      }
    }
    if ((found->second.flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT) !=
            0 &&
        depth == 0) {
      command_buffers[command_buffer].state = CommandBufferState::kInvalid;
    }
  }

  QueryPool* Queries(VkQueryPool handle, std::uint32_t first,
                     std::uint32_t count, const char* function) {
    const auto found = query_pools.find(handle);
    if (found == query_pools.end() ||
        first + count > found->second.values.size()) {
      Error(function, "unknown query pool or queries out of range",
            "VUID-vkCmdWriteTimestamp-query-04904");
      return nullptr;
    }
    return &found->second;
  }

  // A storage-buffer binding of `set`: its bytes, and how many there are.
  std::uint8_t* Binding(VkDescriptorSet set, std::uint32_t binding,
                        VkDeviceSize* size) {
    const auto found = descriptor_sets.find(set);
    if (found == descriptor_sets.end()) {
      return nullptr;
    }
    const auto info = found->second.bindings.find(binding);
    if (info == found->second.bindings.end()) {
      return nullptr;
    }
    std::uint8_t* bytes = BufferBytes(info->second.buffer, "vkCmdDispatch");
    const auto buffer = buffers.find(info->second.buffer);
    if (bytes == nullptr || buffer == buffers.end() ||
        info->second.offset > buffer->second.size) {
      return nullptr;
    }
    *size = info->second.range == VK_WHOLE_SIZE
                ? buffer->second.size - info->second.offset
                : info->second.range;
    return bytes + info->second.offset;
  }

  // etc2_decode.comp on the CPU: each workgroup finds its job by binary
  // search over group_begin, each invocation decodes one 4x4 block and
  // writes its rows; anything outside a binding is an error.
  void Dispatch(const ComputeState& compute, const std::uint32_t groups[3]) {
    ++counts.dispatches;
    if (pipelines.count(compute.pipeline) == 0) {
      Error("vkCmdDispatch", "no compute pipeline bound",
            "VUID-vkCmdDispatch-None-08606");
      return;
    }
    VkDeviceSize source_size = 0;
    VkDeviceSize jobs_size = 0;
    VkDeviceSize output_size = 0;
    const std::uint8_t* source = Binding(compute.sets[0], 0, &source_size);
    const std::uint8_t* jobs = Binding(compute.sets[0], 1, &jobs_size);
    std::uint8_t* output = Binding(compute.sets[1], 0, &output_size);
    if (source == nullptr || jobs == nullptr || output == nullptr) {
      Error("vkCmdDispatch", "a descriptor set is not bound or not written",
            "VUID-vkCmdDispatch-None-08600");
      return;
    }
    std::uint32_t first = 0;
    std::uint32_t count = 0;
    std::memcpy(&first, compute.push, 4);
    std::memcpy(&count, compute.push + 4, 4);
    if (count == 0 ||
        (static_cast<VkDeviceSize>(first) + count) * sizeof(DecodeJob) >
            jobs_size) {
      Error("vkCmdDispatch", "the push constants name jobs past the buffer");
      return;
    }
    const auto job = [&](std::uint32_t index) {
      DecodeJob value;
      std::memcpy(&value, jobs + static_cast<std::size_t>(index) * 64, 64);
      return value;
    };
    const std::uint64_t total =
        static_cast<std::uint64_t>(groups[0]) * groups[1] * groups[2];
    for (std::uint64_t group = 0; group < total; ++group) {
      std::uint32_t low = first;
      std::uint32_t high = first + count - 1;
      while (low < high) {
        const std::uint32_t middle = (low + high + 1) >> 1;
        if (job(middle).group_begin <= group) {
          low = middle;
        } else {
          high = middle - 1;
        }
      }
      const DecodeJob current = job(low);
      if (group < current.group_begin) {
        continue;
      }
      for (std::uint32_t local = 0; local < 64; ++local) {
        const std::uint64_t block =
            (group - current.group_begin) * 64 + local;
        if (block < current.block_count &&
            !DecodeBlock(current, block, source, source_size, output,
                         output_size)) {
          return;
        }
      }
    }
  }

  bool DecodeBlock(const DecodeJob& job, std::uint64_t block,
                   const std::uint8_t* source, VkDeviceSize source_size,
                   std::uint8_t* output, VkDeviceSize output_size) {
    if (job.format > static_cast<std::uint32_t>(EtcFormat::kEacRg11Signed) ||
        job.blocks_wide == 0 || job.blocks_high == 0) {
      Error("vkCmdDispatch", "a job with a bad format or extent");
      return false;
    }
    const auto format = static_cast<EtcFormat>(job.format);
    const std::uint64_t block_bytes = EtcBlockBytes(format);
    const std::uint64_t texel = EtcDecodedTexelBytes(format);
    const std::uint64_t per_layer =
        static_cast<std::uint64_t>(job.blocks_wide) * job.blocks_high;
    const std::uint64_t layer = block / per_layer;
    const std::uint64_t rest = block - layer * per_layer;
    const std::uint64_t by = rest / job.blocks_wide;
    const std::uint64_t bx = rest - by * job.blocks_wide;
    const std::uint64_t word = job.src_word + layer * job.src_layer_words +
                               by * job.src_row_words + bx * block_bytes / 4;
    if ((word * 4) + block_bytes > source_size) {
      Error("vkCmdDispatch", "a job reads past its input binding");
      return false;
    }
    std::uint8_t texels[16 * 4];
    DecodeEtcImage(format, source + word * 4, block_bytes, 4, 4, texels,
                   16 * texel);
    const std::uint64_t row0 = job.dst_word + layer * job.dst_layer_words +
                               by * 4 * job.dst_row_words;
    for (std::uint64_t y = 0; y < 4; ++y) {
      const std::uint64_t at = (row0 + y * job.dst_row_words) * 4 + bx * 4 * texel;
      if (at + 4 * texel > output_size) {
        Error("vkCmdDispatch", "a job writes past its output binding");
        return false;
      }
      std::memcpy(output + at, texels + y * 4 * texel, 4 * texel);
    }
    ++counts.decoded_blocks;
    return true;
  }

  VkResult Submit(const std::vector<std::vector<VkCommandBuffer>>& batches) {
    for (const auto& batch : batches) {
      for (VkCommandBuffer command_buffer : batch) {
        const auto found = command_buffers.find(command_buffer);
        if (found != command_buffers.end() &&
            found->second.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
          Error("vkQueueSubmit", "secondary command buffer submitted",
                "VUID-VkSubmitInfo-pCommandBuffers-00075");
          continue;
        }
        Execute(command_buffer, 0);
      }
    }
    ++counts.submits;
    return VK_SUCCESS;
  }
};

namespace {

// Dispatchable handles (device, queue, command buffer) -> their fake. The
// map outlives every fake: it is never destroyed.
struct Registry {
  std::mutex mutex;
  std::unordered_map<std::uintptr_t, FakeVulkanDevice::Impl*> owners;
};

Registry& GetRegistry() {
  static Registry* registry = new Registry;
  return *registry;
}

template <typename Handle>
void Register(Handle handle, FakeVulkanDevice::Impl* impl) {
  Registry& registry = GetRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.owners[HandleKey(handle)] = impl;
}

template <typename Handle>
void Unregister(Handle handle) {
  Registry& registry = GetRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.owners.erase(HandleKey(handle));
}

template <typename Handle>
FakeVulkanDevice::Impl* Owner(Handle handle) {
  Registry& registry = GetRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.owners.find(HandleKey(handle));
  return found != registry.owners.end() ? found->second : nullptr;
}

// Locks the fake that owns `handle` and counts the entry-point call.
template <typename Handle>
FakeVulkanDevice::Impl* Enter(Handle handle, const char* name,
                              std::unique_lock<std::recursive_mutex>* lock) {
  FakeVulkanDevice::Impl* impl = Owner(handle);
  if (impl != nullptr) {
    *lock = std::unique_lock<std::recursive_mutex>(impl->mutex);
    ++impl->calls[name];
  }
  return impl;
}

#define FAKE_ENTER(handle, name, failure)                    \
  std::unique_lock<std::recursive_mutex> lock;               \
  FakeVulkanDevice::Impl* impl = Enter(handle, name, &lock); \
  if (impl == nullptr) {                                     \
    return failure;                                          \
  }

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateImage(VkDevice device,
                                               const VkImageCreateInfo* info,
                                               const VkAllocationCallbacks*,
                                               VkImage* image) {
  FAKE_ENTER(device, "vkCreateImage", VK_ERROR_DEVICE_LOST);
  if (info == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  return impl->CreateImage(*info, image);
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyImage(VkDevice device, VkImage image,
                                            const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyImage", );
  impl->DestroyImage(image);
}

VKAPI_ATTR VkResult VKAPI_CALL
FakeCreateImageView(VkDevice device, const VkImageViewCreateInfo* info,
                    const VkAllocationCallbacks*, VkImageView* view) {
  FAKE_ENTER(device, "vkCreateImageView", VK_ERROR_DEVICE_LOST);
  return impl->CreateView(info, view);
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyImageView(VkDevice device,
                                                VkImageView view,
                                                const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyImageView", );
  impl->DestroyView(view);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateBuffer(VkDevice device,
                                                const VkBufferCreateInfo* info,
                                                const VkAllocationCallbacks*,
                                                VkBuffer* buffer) {
  FAKE_ENTER(device, "vkCreateBuffer", VK_ERROR_DEVICE_LOST);
  if (info == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  return impl->CreateBuffer(info->size, info->usage, true, buffer);
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyBuffer(VkDevice device, VkBuffer buffer,
                                             const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyBuffer", );
  impl->DestroyBuffer(buffer);
}

VKAPI_ATTR void VKAPI_CALL FakeGetBufferMemoryRequirements(
    VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements) {
  FAKE_ENTER(device, "vkGetBufferMemoryRequirements", );
  impl->Requirements(buffer, requirements);
}

VKAPI_ATTR VkResult VKAPI_CALL
FakeAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
                   const VkAllocationCallbacks*, VkDeviceMemory* memory) {
  FAKE_ENTER(device, "vkAllocateMemory", VK_ERROR_DEVICE_LOST);
  if (info == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  return impl->AllocateMemory(info->allocationSize, info->memoryTypeIndex, true,
                              memory);
}

VKAPI_ATTR void VKAPI_CALL FakeFreeMemory(VkDevice device,
                                          VkDeviceMemory memory,
                                          const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkFreeMemory", );
  impl->Free(memory);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBindBufferMemory(VkDevice device,
                                                    VkBuffer buffer,
                                                    VkDeviceMemory memory,
                                                    VkDeviceSize offset) {
  FAKE_ENTER(device, "vkBindBufferMemory", VK_ERROR_DEVICE_LOST);
  return impl->BindBuffer(buffer, memory, offset);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBindBufferMemory2(
    VkDevice device, std::uint32_t count, const VkBindBufferMemoryInfo* infos) {
  FAKE_ENTER(device, "vkBindBufferMemory2", VK_ERROR_DEVICE_LOST);
  for (std::uint32_t index = 0; index < count; ++index) {
    const VkResult result = impl->BindBuffer(
        infos[index].buffer, infos[index].memory, infos[index].memoryOffset);
    if (result != VK_SUCCESS) {
      return result;
    }
  }
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeMapMemory(VkDevice device,
                                             VkDeviceMemory memory,
                                             VkDeviceSize offset,
                                             VkDeviceSize size,
                                             VkMemoryMapFlags, void** data) {
  FAKE_ENTER(device, "vkMapMemory", VK_ERROR_DEVICE_LOST);
  return impl->Map(memory, offset, size, data);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeMapMemory2(VkDevice device,
                                              const VkMemoryMapInfo* info,
                                              void** data) {
  FAKE_ENTER(device, "vkMapMemory2", VK_ERROR_DEVICE_LOST);
  if (info == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  return impl->Map(info->memory, info->offset, info->size, data);
}

VKAPI_ATTR void VKAPI_CALL FakeUnmapMemory(VkDevice device,
                                           VkDeviceMemory memory) {
  FAKE_ENTER(device, "vkUnmapMemory", );
  impl->Unmap(memory);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeUnmapMemory2(VkDevice device,
                                                const VkMemoryUnmapInfo* info) {
  FAKE_ENTER(device, "vkUnmapMemory2", VK_ERROR_DEVICE_LOST);
  if (info == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  impl->Unmap(info->memory);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeCmdCopyBufferToImage(
    VkCommandBuffer command_buffer, VkBuffer source, VkImage destination,
    VkImageLayout layout, std::uint32_t count,
    const VkBufferImageCopy* regions) {
  FAKE_ENTER(command_buffer, "vkCmdCopyBufferToImage", );
  impl->RecordCopyBufferToImage(
      "vkCmdCopyBufferToImage", command_buffer, source, destination, layout,
      std::vector<VkBufferImageCopy>(regions, regions + count));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdCopyBufferToImage2(
    VkCommandBuffer command_buffer, const VkCopyBufferToImageInfo2* info) {
  FAKE_ENTER(command_buffer, "vkCmdCopyBufferToImage2", );
  if (info == nullptr) {
    return;
  }
  std::vector<VkBufferImageCopy> regions;
  for (std::uint32_t index = 0; index < info->regionCount; ++index) {
    const VkBufferImageCopy2& region = info->pRegions[index];
    regions.push_back({region.bufferOffset, region.bufferRowLength,
                       region.bufferImageHeight, region.imageSubresource,
                       region.imageOffset, region.imageExtent});
  }
  impl->RecordCopyBufferToImage("vkCmdCopyBufferToImage2", command_buffer,
                                info->srcBuffer, info->dstImage,
                                info->dstImageLayout, std::move(regions));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdCopyImageToBuffer(
    VkCommandBuffer command_buffer, VkImage source, VkImageLayout layout,
    VkBuffer destination, std::uint32_t count,
    const VkBufferImageCopy* regions) {
  FAKE_ENTER(command_buffer, "vkCmdCopyImageToBuffer", );
  FakeVulkanDevice::Impl::CommandBuffer* record =
      impl->Recording(command_buffer, "vkCmdCopyImageToBuffer");
  if (record == nullptr) {
    return;
  }
  FakeVulkanDevice::Impl::Command command;
  command.kind = FakeVulkanDevice::Impl::Command::Kind::kCopyImageToBuffer;
  command.buffer = destination;
  command.image = source;
  command.layout = layout;
  command.buffer_image_regions.assign(regions, regions + count);
  for (const VkBufferImageCopy& region : command.buffer_image_regions) {
    impl->ValidRegion("vkCmdCopyImageToBuffer", source, destination, region,
                      impl->FamilyFlags(*record), true);
  }
  record->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdCopyImage(
    VkCommandBuffer command_buffer, VkImage source, VkImageLayout source_layout,
    VkImage destination, VkImageLayout destination_layout, std::uint32_t count,
    const VkImageCopy* regions) {
  FAKE_ENTER(command_buffer, "vkCmdCopyImage", );
  impl->RecordCopyImage("vkCmdCopyImage", command_buffer, source, source_layout,
                        destination, destination_layout,
                        std::vector<VkImageCopy>(regions, regions + count));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdCopyImage2(VkCommandBuffer command_buffer,
                                             const VkCopyImageInfo2* info) {
  FAKE_ENTER(command_buffer, "vkCmdCopyImage2", );
  if (info == nullptr) {
    return;
  }
  std::vector<VkImageCopy> regions;
  for (std::uint32_t index = 0; index < info->regionCount; ++index) {
    const VkImageCopy2& region = info->pRegions[index];
    regions.push_back({region.srcSubresource, region.srcOffset,
                       region.dstSubresource, region.dstOffset, region.extent});
  }
  impl->RecordCopyImage("vkCmdCopyImage2", command_buffer, info->srcImage,
                        info->srcImageLayout, info->dstImage,
                        info->dstImageLayout, std::move(regions));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdBlitImage(
    VkCommandBuffer command_buffer, VkImage source, VkImageLayout source_layout,
    VkImage destination, VkImageLayout destination_layout, std::uint32_t count,
    const VkImageBlit* regions, VkFilter filter) {
  FAKE_ENTER(command_buffer, "vkCmdBlitImage", );
  impl->RecordBlit(command_buffer, source, source_layout, destination,
                   destination_layout,
                   std::vector<VkImageBlit>(regions, regions + count), filter);
}

VKAPI_ATTR void VKAPI_CALL FakeCmdCopyBuffer(VkCommandBuffer command_buffer,
                                             VkBuffer source,
                                             VkBuffer destination,
                                             std::uint32_t count,
                                             const VkBufferCopy* regions) {
  FAKE_ENTER(command_buffer, "vkCmdCopyBuffer", );
  FakeVulkanDevice::Impl::CommandBuffer* record =
      impl->Recording(command_buffer, "vkCmdCopyBuffer");
  if (record == nullptr) {
    return;
  }
  FakeVulkanDevice::Impl::Command command;
  command.kind = FakeVulkanDevice::Impl::Command::Kind::kCopyBuffer;
  command.buffer = source;
  command.buffer2 = destination;
  command.buffer_regions.assign(regions, regions + count);
  for (const VkBufferCopy& region : command.buffer_regions) {
    const auto src = impl->buffers.find(source);
    const auto dst = impl->buffers.find(destination);
    if (src == impl->buffers.end() || dst == impl->buffers.end() ||
        region.srcOffset + region.size > src->second.size ||
        region.dstOffset + region.size > dst->second.size) {
      impl->Error("vkCmdCopyBuffer", "region outside a buffer",
                  "VUID-vkCmdCopyBuffer-size-00115");
    }
  }
  record->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdPipelineBarrier(
    VkCommandBuffer command_buffer, VkPipelineStageFlags, VkPipelineStageFlags,
    VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
    const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {
  FAKE_ENTER(command_buffer, "vkCmdPipelineBarrier", );
  FakeVulkanDevice::Impl::CommandBuffer* record =
      impl->Recording(command_buffer, "vkCmdPipelineBarrier");
  if (record != nullptr) {
    record->commands.push_back({});
  }
}

VKAPI_ATTR void VKAPI_CALL
FakeCmdExecuteCommands(VkCommandBuffer command_buffer, std::uint32_t count,
                       const VkCommandBuffer* secondaries) {
  FAKE_ENTER(command_buffer, "vkCmdExecuteCommands", );
  impl->RecordExecute(command_buffer, count, secondaries);
}

VKAPI_ATTR VkResult VKAPI_CALL
FakeCreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo* info,
                      const VkAllocationCallbacks*, VkCommandPool* pool) {
  FAKE_ENTER(device, "vkCreateCommandPool", VK_ERROR_DEVICE_LOST);
  if (info == nullptr || pool == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  *pool = impl->CreatePool(info->queueFamilyIndex, info->flags);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyCommandPool(
    VkDevice device, VkCommandPool pool, const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyCommandPool", );
  impl->DestroyPool(pool);
}

VkResult ResetPool(FakeVulkanDevice::Impl* impl, VkCommandPool pool) {
  const auto found = impl->pools.find(pool);
  if (found == impl->pools.end()) {
    impl->Error("vkResetCommandPool", "unknown command pool");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  for (VkCommandBuffer command_buffer : found->second.command_buffers) {
    auto& record = impl->command_buffers[command_buffer];
    record.commands.clear();
    record.state = FakeVulkanDevice::Impl::CommandBufferState::kInitial;
  }
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeResetCommandPool(VkDevice device,
                                                    VkCommandPool pool,
                                                    VkCommandPoolResetFlags) {
  FAKE_ENTER(device, "vkResetCommandPool", VK_ERROR_DEVICE_LOST);
  return ResetPool(impl, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeAllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo* info,
    VkCommandBuffer* command_buffers) {
  FAKE_ENTER(device, "vkAllocateCommandBuffers", VK_ERROR_DEVICE_LOST);
  if (info == nullptr || command_buffers == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  for (std::uint32_t index = 0; index < info->commandBufferCount; ++index) {
    command_buffers[index] = impl->Allocate(info->commandPool, info->level);
    if (command_buffers[index] == VK_NULL_HANDLE) {
      return VK_ERROR_INITIALIZATION_FAILED;
    }
  }
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
FakeFreeCommandBuffers(VkDevice device, VkCommandPool, std::uint32_t count,
                       const VkCommandBuffer* command_buffers) {
  FAKE_ENTER(device, "vkFreeCommandBuffers", );
  for (std::uint32_t index = 0; index < count; ++index) {
    impl->FreeCommandBuffer(command_buffers[index]);
  }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBeginCommandBuffer(
    VkCommandBuffer command_buffer, const VkCommandBufferBeginInfo* info) {
  FAKE_ENTER(command_buffer, "vkBeginCommandBuffer", VK_ERROR_DEVICE_LOST);
  return impl->Begin(command_buffer, info != nullptr ? info->flags : 0);
}

VKAPI_ATTR VkResult VKAPI_CALL
FakeEndCommandBuffer(VkCommandBuffer command_buffer) {
  FAKE_ENTER(command_buffer, "vkEndCommandBuffer", VK_ERROR_DEVICE_LOST);
  return impl->End(command_buffer);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeResetCommandBuffer(
    VkCommandBuffer command_buffer, VkCommandBufferResetFlags) {
  FAKE_ENTER(command_buffer, "vkResetCommandBuffer", VK_ERROR_DEVICE_LOST);
  return impl->Reset(command_buffer);
}

VKAPI_ATTR void VKAPI_CALL FakeGetDeviceQueue(VkDevice device,
                                              std::uint32_t family,
                                              std::uint32_t index,
                                              VkQueue* queue);

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateFence(VkDevice device,
                                               const VkFenceCreateInfo* info,
                                               const VkAllocationCallbacks*,
                                               VkFence* fence) {
  FAKE_ENTER(device, "vkCreateFence", VK_ERROR_DEVICE_LOST);
  if (fence == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  *fence = NextHandle<VkFence>();
  impl->fences[*fence] =
      info != nullptr && (info->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
  ++impl->counts.fences.created;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyFence(VkDevice device, VkFence fence,
                                            const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyFence", );
  if (fence == VK_NULL_HANDLE) {
    return;
  }
  // A held fence may go: teardown gives up on a GPU that does not finish.
  impl->held_fences.erase(std::remove(impl->held_fences.begin(),
                                      impl->held_fences.end(), fence),
                          impl->held_fences.end());
  if (impl->fences.erase(fence) == 0) {
    impl->Error("vkDestroyFence", "unknown fence");
    return;
  }
  ++impl->counts.fences.destroyed;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeGetFenceStatus(VkDevice device,
                                                  VkFence fence) {
  FAKE_ENTER(device, "vkGetFenceStatus", VK_ERROR_DEVICE_LOST);
  if (impl->lost) {
    return VK_ERROR_DEVICE_LOST;
  }
  const auto found = impl->fences.find(fence);
  if (found == impl->fences.end()) {
    impl->Error("vkGetFenceStatus", "unknown fence");
    return VK_ERROR_DEVICE_LOST;
  }
  return found->second ? VK_SUCCESS : VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeWaitForFences(VkDevice device,
                                                 std::uint32_t count,
                                                 const VkFence* fences,
                                                 VkBool32 wait_all,
                                                 std::uint64_t) {
  FAKE_ENTER(device, "vkWaitForFences", VK_ERROR_DEVICE_LOST);
  if (impl->lost) {
    return VK_ERROR_DEVICE_LOST;
  }
  std::uint32_t signalled = 0;
  for (std::uint32_t index = 0; index < count; ++index) {
    const auto found = impl->fences.find(fences[index]);
    if (found == impl->fences.end()) {
      impl->Error("vkWaitForFences", "unknown fence");
      return VK_ERROR_DEVICE_LOST;
    }
    signalled += found->second ? 1 : 0;
  }
  // Nothing is ever in flight: an unsignalled fence would never signal.
  return (wait_all == VK_TRUE ? signalled == count : signalled != 0)
             ? VK_SUCCESS
             : VK_TIMEOUT;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeResetFences(VkDevice device,
                                               std::uint32_t count,
                                               const VkFence* fences) {
  FAKE_ENTER(device, "vkResetFences", VK_ERROR_DEVICE_LOST);
  for (std::uint32_t index = 0; index < count; ++index) {
    if (std::find(impl->held_fences.begin(), impl->held_fences.end(),
                  fences[index]) != impl->held_fences.end()) {
      impl->Error("vkResetFences", "the fence is still pending",
                  "VUID-vkResetFences-pFences-01123");
    }
    impl->fences[fences[index]] = false;
  }
  return VK_SUCCESS;
}

// ---- compute objects and queries (the ETC2 GPU route) ----------------------

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateShaderModule(
    VkDevice device, const VkShaderModuleCreateInfo* info,
    const VkAllocationCallbacks*, VkShaderModule* module) {
  FAKE_ENTER(device, "vkCreateShaderModule", VK_ERROR_DEVICE_LOST);
  if (info == nullptr || module == nullptr || info->codeSize == 0 ||
      info->codeSize % 4 != 0 || info->pCode == nullptr ||
      info->pCode[0] != 0x07230203U) {
    impl->Error("vkCreateShaderModule", "not SPIR-V",
                "VUID-VkShaderModuleCreateInfo-pCode-08737");
    return VK_ERROR_INVALID_SHADER_NV;
  }
  *module = NextHandle<VkShaderModule>();
  impl->shader_modules[*module] = info->codeSize;
  ++impl->counts.shader_modules.created;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyShaderModule(
    VkDevice device, VkShaderModule module, const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyShaderModule", );
  if (module != VK_NULL_HANDLE) {
    if (impl->shader_modules.erase(module) == 0) {
      impl->Error("vkDestroyShaderModule", "unknown shader module");
      return;
    }
    ++impl->counts.shader_modules.destroyed;
  }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateDescriptorSetLayout(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo* info,
    const VkAllocationCallbacks*, VkDescriptorSetLayout* layout) {
  FAKE_ENTER(device, "vkCreateDescriptorSetLayout", VK_ERROR_DEVICE_LOST);
  if (info == nullptr || layout == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  *layout = NextHandle<VkDescriptorSetLayout>();
  impl->set_layouts[*layout] = info->bindingCount;
  ++impl->counts.descriptor_set_layouts.created;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyDescriptorSetLayout(
    VkDevice device, VkDescriptorSetLayout layout,
    const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyDescriptorSetLayout", );
  if (layout != VK_NULL_HANDLE) {
    if (impl->set_layouts.erase(layout) == 0) {
      impl->Error("vkDestroyDescriptorSetLayout", "unknown layout");
      return;
    }
    ++impl->counts.descriptor_set_layouts.destroyed;
  }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreatePipelineLayout(
    VkDevice device, const VkPipelineLayoutCreateInfo* info,
    const VkAllocationCallbacks*, VkPipelineLayout* layout) {
  FAKE_ENTER(device, "vkCreatePipelineLayout", VK_ERROR_DEVICE_LOST);
  if (info == nullptr || layout == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  for (std::uint32_t index = 0; index < info->setLayoutCount; ++index) {
    if (impl->set_layouts.count(info->pSetLayouts[index]) == 0) {
      impl->Error("vkCreatePipelineLayout", "unknown set layout");
    }
  }
  *layout = NextHandle<VkPipelineLayout>();
  impl->pipeline_layouts[*layout] = info->setLayoutCount;
  ++impl->counts.pipeline_layouts.created;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyPipelineLayout(
    VkDevice device, VkPipelineLayout layout, const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyPipelineLayout", );
  if (layout != VK_NULL_HANDLE) {
    if (impl->pipeline_layouts.erase(layout) == 0) {
      impl->Error("vkDestroyPipelineLayout", "unknown pipeline layout");
      return;
    }
    ++impl->counts.pipeline_layouts.destroyed;
  }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateComputePipelines(
    VkDevice device, VkPipelineCache, std::uint32_t count,
    const VkComputePipelineCreateInfo* infos, const VkAllocationCallbacks*,
    VkPipeline* pipelines) {
  FAKE_ENTER(device, "vkCreateComputePipelines", VK_ERROR_DEVICE_LOST);
  if (impl->config.fail_compute_pipelines) {
    std::fill(pipelines, pipelines + count, VK_NULL_HANDLE);
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }
  for (std::uint32_t index = 0; index < count; ++index) {
    if (impl->shader_modules.count(infos[index].stage.module) == 0 ||
        impl->pipeline_layouts.count(infos[index].layout) == 0) {
      impl->Error("vkCreateComputePipelines",
                  "unknown shader module or pipeline layout");
      return VK_ERROR_INITIALIZATION_FAILED;
    }
    pipelines[index] = NextHandle<VkPipeline>();
    impl->pipelines[pipelines[index]] = infos[index].stage.module;
    ++impl->counts.pipelines.created;
  }
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyPipeline(VkDevice device,
                                               VkPipeline pipeline,
                                               const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyPipeline", );
  if (pipeline != VK_NULL_HANDLE) {
    if (impl->pipelines.erase(pipeline) == 0) {
      impl->Error("vkDestroyPipeline", "unknown pipeline");
      return;
    }
    ++impl->counts.pipelines.destroyed;
  }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateDescriptorPool(
    VkDevice device, const VkDescriptorPoolCreateInfo* info,
    const VkAllocationCallbacks*, VkDescriptorPool* pool) {
  FAKE_ENTER(device, "vkCreateDescriptorPool", VK_ERROR_DEVICE_LOST);
  if (info == nullptr || pool == nullptr || info->maxSets == 0) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  *pool = NextHandle<VkDescriptorPool>();
  FakeVulkanDevice::Impl::DescriptorPool& record = impl->descriptor_pools[*pool];
  record.max_sets = impl->config.max_sets_per_pool != 0
                        ? std::min(impl->config.max_sets_per_pool,
                                   info->maxSets)
                        : info->maxSets;
  record.flags = info->flags;
  ++impl->counts.descriptor_pools.created;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyDescriptorPool(
    VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyDescriptorPool", );
  if (pool == VK_NULL_HANDLE) {
    return;
  }
  const auto found = impl->descriptor_pools.find(pool);
  if (found == impl->descriptor_pools.end()) {
    impl->Error("vkDestroyDescriptorPool", "unknown descriptor pool");
    return;
  }
  for (VkDescriptorSet set : found->second.sets) {
    impl->descriptor_sets.erase(set);
    ++impl->counts.descriptor_sets.destroyed;
  }
  impl->descriptor_pools.erase(found);
  ++impl->counts.descriptor_pools.destroyed;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeAllocateDescriptorSets(
    VkDevice device, const VkDescriptorSetAllocateInfo* info,
    VkDescriptorSet* sets) {
  FAKE_ENTER(device, "vkAllocateDescriptorSets", VK_ERROR_DEVICE_LOST);
  if (info == nullptr || sets == nullptr) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  const auto pool = impl->descriptor_pools.find(info->descriptorPool);
  if (pool == impl->descriptor_pools.end()) {
    impl->Error("vkAllocateDescriptorSets", "unknown descriptor pool");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if (pool->second.sets.size() + info->descriptorSetCount >
      pool->second.max_sets) {
    return VK_ERROR_OUT_OF_POOL_MEMORY;
  }
  for (std::uint32_t index = 0; index < info->descriptorSetCount; ++index) {
    if (impl->set_layouts.count(info->pSetLayouts[index]) == 0) {
      impl->Error("vkAllocateDescriptorSets", "unknown set layout");
    }
    sets[index] = NextHandle<VkDescriptorSet>();
    impl->descriptor_sets[sets[index]].pool = info->descriptorPool;
    pool->second.sets.push_back(sets[index]);
    ++impl->counts.descriptor_sets.created;
  }
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeFreeDescriptorSets(
    VkDevice device, VkDescriptorPool pool, std::uint32_t count,
    const VkDescriptorSet* sets) {
  FAKE_ENTER(device, "vkFreeDescriptorSets", VK_ERROR_DEVICE_LOST);
  const auto found = impl->descriptor_pools.find(pool);
  if (found == impl->descriptor_pools.end()) {
    impl->Error("vkFreeDescriptorSets", "unknown descriptor pool");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  if ((found->second.flags & VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT) ==
      0) {
    impl->Error("vkFreeDescriptorSets",
                "pool was created without FREE_DESCRIPTOR_SET",
                "VUID-vkFreeDescriptorSets-descriptorPool-00312");
  }
  for (std::uint32_t index = 0; index < count; ++index) {
    auto& owned = found->second.sets;
    const auto at = std::find(owned.begin(), owned.end(), sets[index]);
    if (at == owned.end()) {
      impl->Error("vkFreeDescriptorSets", "set not from this pool");
      continue;
    }
    owned.erase(at);
    impl->descriptor_sets.erase(sets[index]);
    ++impl->counts.descriptor_sets.destroyed;
  }
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeUpdateDescriptorSets(
    VkDevice device, std::uint32_t count, const VkWriteDescriptorSet* writes,
    std::uint32_t, const VkCopyDescriptorSet*) {
  FAKE_ENTER(device, "vkUpdateDescriptorSets", );
  for (std::uint32_t index = 0; index < count; ++index) {
    const VkWriteDescriptorSet& write = writes[index];
    const auto set = impl->descriptor_sets.find(write.dstSet);
    if (set == impl->descriptor_sets.end() ||
        write.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
        write.pBufferInfo == nullptr) {
      impl->Error("vkUpdateDescriptorSets",
                  "unknown set, or not a storage buffer");
      continue;
    }
    for (std::uint32_t element = 0; element < write.descriptorCount;
         ++element) {
      const VkDescriptorBufferInfo& info = write.pBufferInfo[element];
      if (impl->buffers.count(info.buffer) == 0) {
        impl->Error("vkUpdateDescriptorSets", "unknown buffer");
      }
      set->second.bindings[write.dstBinding + element] = info;
    }
  }
}

VKAPI_ATTR void VKAPI_CALL FakeCmdBindPipeline(VkCommandBuffer command_buffer,
                                               VkPipelineBindPoint point,
                                               VkPipeline pipeline) {
  FAKE_ENTER(command_buffer, "vkCmdBindPipeline", );
  FakeVulkanDevice::Impl::CommandBuffer* record =
      impl->Recording(command_buffer, "vkCmdBindPipeline");
  if (record == nullptr) {
    return;
  }
  if (point != VK_PIPELINE_BIND_POINT_COMPUTE ||
      impl->pipelines.count(pipeline) == 0) {
    impl->Error("vkCmdBindPipeline", "unknown or not a compute pipeline");
  }
  FakeVulkanDevice::Impl::Command command;
  command.kind = FakeVulkanDevice::Impl::Command::Kind::kBindPipeline;
  command.pipeline = pipeline;
  record->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdBindDescriptorSets(
    VkCommandBuffer command_buffer, VkPipelineBindPoint, VkPipelineLayout,
    std::uint32_t first, std::uint32_t count, const VkDescriptorSet* sets,
    std::uint32_t, const std::uint32_t*) {
  FAKE_ENTER(command_buffer, "vkCmdBindDescriptorSets", );
  FakeVulkanDevice::Impl::CommandBuffer* record =
      impl->Recording(command_buffer, "vkCmdBindDescriptorSets");
  if (record == nullptr) {
    return;
  }
  FakeVulkanDevice::Impl::Command command;
  command.kind = FakeVulkanDevice::Impl::Command::Kind::kBindDescriptorSets;
  command.first = first;
  command.sets.assign(sets, sets + count);
  for (VkDescriptorSet set : command.sets) {
    if (impl->descriptor_sets.count(set) == 0) {
      impl->Error("vkCmdBindDescriptorSets", "unknown descriptor set");
    }
  }
  record->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdPushConstants(
    VkCommandBuffer command_buffer, VkPipelineLayout, VkShaderStageFlags,
    std::uint32_t offset, std::uint32_t size, const void* values) {
  FAKE_ENTER(command_buffer, "vkCmdPushConstants", );
  FakeVulkanDevice::Impl::CommandBuffer* record =
      impl->Recording(command_buffer, "vkCmdPushConstants");
  if (record == nullptr) {
    return;
  }
  FakeVulkanDevice::Impl::Command command;
  command.kind = FakeVulkanDevice::Impl::Command::Kind::kPushConstants;
  command.first = offset;
  const auto* bytes = static_cast<const std::uint8_t*>(values);
  command.bytes.assign(bytes, bytes + size);
  record->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdDispatch(VkCommandBuffer command_buffer,
                                           std::uint32_t x, std::uint32_t y,
                                           std::uint32_t z) {
  FAKE_ENTER(command_buffer, "vkCmdDispatch", );
  FakeVulkanDevice::Impl::CommandBuffer* record =
      impl->Recording(command_buffer, "vkCmdDispatch");
  if (record == nullptr) {
    return;
  }
  const std::uint32_t* limits =
      impl->config.properties.limits.maxComputeWorkGroupCount;
  if (x > limits[0] || y > limits[1] || z > limits[2]) {
    impl->Error("vkCmdDispatch", "group count above the device limit",
                "VUID-vkCmdDispatch-groupCountX-00386");
  }
  FakeVulkanDevice::Impl::Command command;
  command.kind = FakeVulkanDevice::Impl::Command::Kind::kDispatch;
  command.groups[0] = x;
  command.groups[1] = y;
  command.groups[2] = z;
  record->commands.push_back(std::move(command));
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateQueryPool(
    VkDevice device, const VkQueryPoolCreateInfo* info,
    const VkAllocationCallbacks*, VkQueryPool* pool) {
  FAKE_ENTER(device, "vkCreateQueryPool", VK_ERROR_DEVICE_LOST);
  if (info == nullptr || pool == nullptr ||
      info->queryType != VK_QUERY_TYPE_TIMESTAMP || info->queryCount == 0) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  *pool = NextHandle<VkQueryPool>();
  FakeVulkanDevice::Impl::QueryPool& record = impl->query_pools[*pool];
  record.values.assign(info->queryCount, 0);
  record.written.assign(info->queryCount, false);
  ++impl->counts.query_pools.created;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyQueryPool(VkDevice device,
                                                VkQueryPool pool,
                                                const VkAllocationCallbacks*) {
  FAKE_ENTER(device, "vkDestroyQueryPool", );
  if (pool != VK_NULL_HANDLE) {
    if (impl->query_pools.erase(pool) == 0) {
      impl->Error("vkDestroyQueryPool", "unknown query pool");
      return;
    }
    ++impl->counts.query_pools.destroyed;
  }
}

void RecordQueries(FakeVulkanDevice::Impl* impl, VkCommandBuffer command_buffer,
                   const char* function,
                   FakeVulkanDevice::Impl::Command::Kind kind, VkQueryPool pool,
                   std::uint32_t first, std::uint32_t count) {
  FakeVulkanDevice::Impl::CommandBuffer* record =
      impl->Recording(command_buffer, function);
  if (record == nullptr) {
    return;
  }
  FakeVulkanDevice::Impl::Command command;
  command.kind = kind;
  command.queries = pool;
  command.first = first;
  command.count = count;
  record->commands.push_back(std::move(command));
}

VKAPI_ATTR void VKAPI_CALL FakeCmdResetQueryPool(VkCommandBuffer command_buffer,
                                                 VkQueryPool pool,
                                                 std::uint32_t first,
                                                 std::uint32_t count) {
  FAKE_ENTER(command_buffer, "vkCmdResetQueryPool", );
  RecordQueries(impl, command_buffer, "vkCmdResetQueryPool",
                FakeVulkanDevice::Impl::Command::Kind::kResetQueryPool, pool,
                first, count);
}

VKAPI_ATTR void VKAPI_CALL FakeCmdWriteTimestamp(
    VkCommandBuffer command_buffer, VkPipelineStageFlagBits, VkQueryPool pool,
    std::uint32_t query) {
  FAKE_ENTER(command_buffer, "vkCmdWriteTimestamp", );
  RecordQueries(impl, command_buffer, "vkCmdWriteTimestamp",
                FakeVulkanDevice::Impl::Command::Kind::kWriteTimestamp, pool,
                query, 1);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeGetQueryPoolResults(
    VkDevice device, VkQueryPool pool, std::uint32_t first,
    std::uint32_t count, std::size_t data_size, void* data,
    VkDeviceSize stride, VkQueryResultFlags flags) {
  FAKE_ENTER(device, "vkGetQueryPoolResults", VK_ERROR_DEVICE_LOST);
  FakeVulkanDevice::Impl::QueryPool* record =
      impl->Queries(pool, first, count, "vkGetQueryPoolResults");
  if (record == nullptr || (flags & VK_QUERY_RESULT_64_BIT) == 0 ||
      stride < 8 || static_cast<VkDeviceSize>(count - 1) * stride + 8 >
                        data_size) {
    impl->Error("vkGetQueryPoolResults", "bad query range or layout");
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  VkResult result = VK_SUCCESS;
  for (std::uint32_t query = 0; query < count; ++query) {
    if (!record->written[first + query]) {
      if ((flags & VK_QUERY_RESULT_WAIT_BIT) != 0) {
        impl->Error("vkGetQueryPoolResults",
                    "waits for a query that was never written");
      }
      result = VK_NOT_READY;
      continue;
    }
    std::memcpy(static_cast<std::uint8_t*>(data) + query * stride,
                &record->values[first + query], 8);
  }
  return result;
}

void SignalFence(FakeVulkanDevice::Impl* impl, VkFence fence) {
  if (fence == VK_NULL_HANDLE) {
    return;
  }
  const auto found = impl->fences.find(fence);
  if (found == impl->fences.end()) {
    impl->Error("vkQueueSubmit", "unknown fence");
    return;
  }
  if (found->second ||
      std::find(impl->held_fences.begin(), impl->held_fences.end(), fence) !=
          impl->held_fences.end()) {
    impl->Error("vkQueueSubmit", "fence is already signalled",
                "VUID-vkQueueSubmit-fence-00063");
  }
  if (impl->hold_fences) {
    impl->held_fences.push_back(fence);
    return;
  }
  found->second = true;
}

// A result FailNextSubmits queued, or VK_SUCCESS.
VkResult QueuedSubmitResult(FakeVulkanDevice::Impl* impl) {
  if (impl->submit_results.empty()) {
    return VK_SUCCESS;
  }
  const VkResult result = impl->submit_results.front();
  impl->submit_results.erase(impl->submit_results.begin());
  return result;
}

VkResult SubmitInfos(FakeVulkanDevice::Impl* impl, std::uint32_t count,
                     const VkSubmitInfo* submits, VkFence fence) {
  if (const VkResult queued = QueuedSubmitResult(impl); queued != VK_SUCCESS) {
    return queued;
  }
  std::vector<std::vector<VkCommandBuffer>> batches;
  for (std::uint32_t index = 0; index < count; ++index) {
    batches.emplace_back(
        submits[index].pCommandBuffers,
        submits[index].pCommandBuffers + submits[index].commandBufferCount);
  }
  const VkResult result = impl->Submit(batches);
  SignalFence(impl, fence);
  return result;
}

VkResult SubmitInfos2(FakeVulkanDevice::Impl* impl, std::uint32_t count,
                      const VkSubmitInfo2* submits, VkFence fence) {
  if (const VkResult queued = QueuedSubmitResult(impl); queued != VK_SUCCESS) {
    return queued;
  }
  std::vector<std::vector<VkCommandBuffer>> batches;
  for (std::uint32_t index = 0; index < count; ++index) {
    std::vector<VkCommandBuffer> batch;
    for (std::uint32_t info = 0; info < submits[index].commandBufferInfoCount;
         ++info) {
      batch.push_back(submits[index].pCommandBufferInfos[info].commandBuffer);
    }
    batches.push_back(std::move(batch));
  }
  const VkResult result = impl->Submit(batches);
  SignalFence(impl, fence);
  return result;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeQueueSubmit(VkQueue queue,
                                               std::uint32_t count,
                                               const VkSubmitInfo* submits,
                                               VkFence fence) {
  FAKE_ENTER(queue, "vkQueueSubmit", VK_ERROR_DEVICE_LOST);
  return SubmitInfos(impl, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeQueueSubmit2(VkQueue queue,
                                                std::uint32_t count,
                                                const VkSubmitInfo2* submits,
                                                VkFence fence) {
  FAKE_ENTER(queue, "vkQueueSubmit2", VK_ERROR_DEVICE_LOST);
  return SubmitInfos2(impl, count, submits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeQueueWaitIdle(VkQueue queue) {
  FAKE_ENTER(queue, "vkQueueWaitIdle", VK_ERROR_DEVICE_LOST);
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeDeviceWaitIdle(VkDevice device) {
  FAKE_ENTER(device, "vkDeviceWaitIdle", VK_ERROR_DEVICE_LOST);
  return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
FakeGetDeviceProcAddr(VkDevice device, const char* name);

struct ProcEntry {
  const char* name;
  PFN_vkVoidFunction function;
};

#define FAKE_PROC(name, function) \
  {name, reinterpret_cast<PFN_vkVoidFunction>(&function)}

const ProcEntry kProcs[] = {
    FAKE_PROC("vkGetDeviceProcAddr", FakeGetDeviceProcAddr),
    FAKE_PROC("vkCreateImage", FakeCreateImage),
    FAKE_PROC("vkDestroyImage", FakeDestroyImage),
    FAKE_PROC("vkCreateImageView", FakeCreateImageView),
    FAKE_PROC("vkDestroyImageView", FakeDestroyImageView),
    FAKE_PROC("vkCreateBuffer", FakeCreateBuffer),
    FAKE_PROC("vkDestroyBuffer", FakeDestroyBuffer),
    FAKE_PROC("vkGetBufferMemoryRequirements", FakeGetBufferMemoryRequirements),
    FAKE_PROC("vkAllocateMemory", FakeAllocateMemory),
    FAKE_PROC("vkFreeMemory", FakeFreeMemory),
    FAKE_PROC("vkBindBufferMemory", FakeBindBufferMemory),
    FAKE_PROC("vkBindBufferMemory2", FakeBindBufferMemory2),
    FAKE_PROC("vkBindBufferMemory2KHR", FakeBindBufferMemory2),
    FAKE_PROC("vkMapMemory", FakeMapMemory),
    FAKE_PROC("vkMapMemory2", FakeMapMemory2),
    FAKE_PROC("vkMapMemory2KHR", FakeMapMemory2),
    FAKE_PROC("vkUnmapMemory", FakeUnmapMemory),
    FAKE_PROC("vkUnmapMemory2", FakeUnmapMemory2),
    FAKE_PROC("vkUnmapMemory2KHR", FakeUnmapMemory2),
    FAKE_PROC("vkCmdCopyBufferToImage", FakeCmdCopyBufferToImage),
    FAKE_PROC("vkCmdCopyBufferToImage2", FakeCmdCopyBufferToImage2),
    FAKE_PROC("vkCmdCopyBufferToImage2KHR", FakeCmdCopyBufferToImage2),
    FAKE_PROC("vkCmdCopyImageToBuffer", FakeCmdCopyImageToBuffer),
    FAKE_PROC("vkCmdCopyImage", FakeCmdCopyImage),
    FAKE_PROC("vkCmdCopyImage2", FakeCmdCopyImage2),
    FAKE_PROC("vkCmdCopyImage2KHR", FakeCmdCopyImage2),
    FAKE_PROC("vkCmdBlitImage", FakeCmdBlitImage),
    FAKE_PROC("vkCmdCopyBuffer", FakeCmdCopyBuffer),
    FAKE_PROC("vkCmdPipelineBarrier", FakeCmdPipelineBarrier),
    FAKE_PROC("vkCmdExecuteCommands", FakeCmdExecuteCommands),
    FAKE_PROC("vkCreateCommandPool", FakeCreateCommandPool),
    FAKE_PROC("vkDestroyCommandPool", FakeDestroyCommandPool),
    FAKE_PROC("vkResetCommandPool", FakeResetCommandPool),
    FAKE_PROC("vkAllocateCommandBuffers", FakeAllocateCommandBuffers),
    FAKE_PROC("vkFreeCommandBuffers", FakeFreeCommandBuffers),
    FAKE_PROC("vkBeginCommandBuffer", FakeBeginCommandBuffer),
    FAKE_PROC("vkEndCommandBuffer", FakeEndCommandBuffer),
    FAKE_PROC("vkResetCommandBuffer", FakeResetCommandBuffer),
    FAKE_PROC("vkGetDeviceQueue", FakeGetDeviceQueue),
    FAKE_PROC("vkCreateFence", FakeCreateFence),
    FAKE_PROC("vkDestroyFence", FakeDestroyFence),
    FAKE_PROC("vkGetFenceStatus", FakeGetFenceStatus),
    FAKE_PROC("vkWaitForFences", FakeWaitForFences),
    FAKE_PROC("vkResetFences", FakeResetFences),
    FAKE_PROC("vkQueueSubmit", FakeQueueSubmit),
    FAKE_PROC("vkQueueSubmit2", FakeQueueSubmit2),
    FAKE_PROC("vkQueueSubmit2KHR", FakeQueueSubmit2),
    FAKE_PROC("vkQueueWaitIdle", FakeQueueWaitIdle),
    FAKE_PROC("vkDeviceWaitIdle", FakeDeviceWaitIdle),
    FAKE_PROC("vkCreateShaderModule", FakeCreateShaderModule),
    FAKE_PROC("vkDestroyShaderModule", FakeDestroyShaderModule),
    FAKE_PROC("vkCreateDescriptorSetLayout", FakeCreateDescriptorSetLayout),
    FAKE_PROC("vkDestroyDescriptorSetLayout", FakeDestroyDescriptorSetLayout),
    FAKE_PROC("vkCreatePipelineLayout", FakeCreatePipelineLayout),
    FAKE_PROC("vkDestroyPipelineLayout", FakeDestroyPipelineLayout),
    FAKE_PROC("vkCreateComputePipelines", FakeCreateComputePipelines),
    FAKE_PROC("vkDestroyPipeline", FakeDestroyPipeline),
    FAKE_PROC("vkCreateDescriptorPool", FakeCreateDescriptorPool),
    FAKE_PROC("vkDestroyDescriptorPool", FakeDestroyDescriptorPool),
    FAKE_PROC("vkAllocateDescriptorSets", FakeAllocateDescriptorSets),
    FAKE_PROC("vkFreeDescriptorSets", FakeFreeDescriptorSets),
    FAKE_PROC("vkUpdateDescriptorSets", FakeUpdateDescriptorSets),
    FAKE_PROC("vkCmdBindPipeline", FakeCmdBindPipeline),
    FAKE_PROC("vkCmdBindDescriptorSets", FakeCmdBindDescriptorSets),
    FAKE_PROC("vkCmdPushConstants", FakeCmdPushConstants),
    FAKE_PROC("vkCmdDispatch", FakeCmdDispatch),
    FAKE_PROC("vkCreateQueryPool", FakeCreateQueryPool),
    FAKE_PROC("vkDestroyQueryPool", FakeDestroyQueryPool),
    FAKE_PROC("vkCmdResetQueryPool", FakeCmdResetQueryPool),
    FAKE_PROC("vkCmdWriteTimestamp", FakeCmdWriteTimestamp),
    FAKE_PROC("vkGetQueryPoolResults", FakeGetQueryPoolResults),
};

#undef FAKE_PROC

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
FakeGetDeviceProcAddr(VkDevice device, const char* name) {
  FakeVulkanDevice::Impl* impl = Owner(device);
  if (impl == nullptr || name == nullptr) {
    return nullptr;
  }
  {
    std::lock_guard<std::recursive_mutex> lock(impl->mutex);
    for (const std::string& missing : impl->config.missing_procs) {
      if (missing == name) {
        return nullptr;
      }
    }
  }
  for (const ProcEntry& entry : kProcs) {
    if (std::strcmp(entry.name, name) == 0) {
      return entry.function;
    }
  }
  return nullptr;
}

// ---- the physical device ----------------------------------------------------

bool IsCompressed(VkFormat format) {
  FormatBlock block;
  return LookupFormatBlock(format, &block) && block.width > 1;
}

bool IsBcFormat(VkFormat format) {
  return format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK &&
         format <= VK_FORMAT_BC7_SRGB_BLOCK;
}

// What an RTX 3060 reports (vkformats.txt): block-compressed formats can be
// sampled, filtered, blitted from and copied; uncompressed ones also
// rendered to, stored to and blitted into.
VkFormatFeatureFlags OptimalFeatures(const FakeVulkanDevice::Config& config,
                                     VkFormat format) {
  constexpr VkFormatFeatureFlags kCompressed =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
  if (!LookupFormatBlock(format, nullptr)) {
    return 0;
  }
  if (!IsCompressed(format)) {
    return kCompressed | VK_FORMAT_FEATURE_BLIT_DST_BIT |
           VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
           VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
  }
  const bool supported = IsBcFormat(format) ? config.texture_compression_bc
                                            : config.texture_compression_etc2;
  return supported ? kCompressed : 0;
}

VKAPI_ATTR void VKAPI_CALL
FakeGetPhysicalDeviceFeatures(VkPhysicalDevice physical_device,
                              VkPhysicalDeviceFeatures* features) {
  *features = VkPhysicalDeviceFeatures{};
  FAKE_ENTER(physical_device, "vkGetPhysicalDeviceFeatures", );
  features->textureCompressionETC2 =
      impl->config.texture_compression_etc2 ? VK_TRUE : VK_FALSE;
  features->textureCompressionBC =
      impl->config.texture_compression_bc ? VK_TRUE : VK_FALSE;
}

VKAPI_ATTR void VKAPI_CALL FakeGetPhysicalDeviceFormatProperties(
    VkPhysicalDevice physical_device, VkFormat format,
    VkFormatProperties* properties) {
  *properties = VkFormatProperties{};
  FAKE_ENTER(physical_device, "vkGetPhysicalDeviceFormatProperties", );
  properties->optimalTilingFeatures = OptimalFeatures(impl->config, format);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeGetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice physical_device, VkFormat format, VkImageType type,
    VkImageTiling tiling, VkImageUsageFlags usage, VkImageCreateFlags flags,
    VkImageFormatProperties* properties) {
  *properties = VkImageFormatProperties{};
  FAKE_ENTER(physical_device, "vkGetPhysicalDeviceImageFormatProperties",
             VK_ERROR_INITIALIZATION_FAILED);
  // Block-compressed images are only sampled and copied.
  constexpr VkImageUsageFlags kCompressedUsage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT;
  constexpr VkImageCreateFlags kCompressedFlags =
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
      VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT |
      VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
  if (OptimalFeatures(impl->config, format) == 0 ||
      type != VK_IMAGE_TYPE_2D || tiling != VK_IMAGE_TILING_OPTIMAL ||
      (IsCompressed(format) && ((usage & ~kCompressedUsage) != 0 ||
                                (flags & ~kCompressedFlags) != 0))) {
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  properties->maxExtent = {32768, 32768, 1};
  properties->maxMipLevels = 16;
  properties->maxArrayLayers = 2048;
  properties->sampleCounts = VK_SAMPLE_COUNT_1_BIT;
  properties->maxResourceSize = 1ULL << 40;
  return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
FakeGetInstanceProcAddr(VkInstance, const char* name) {
  const ProcEntry kInstanceProcs[] = {
      {"vkGetPhysicalDeviceFeatures",
       reinterpret_cast<PFN_vkVoidFunction>(&FakeGetPhysicalDeviceFeatures)},
      {"vkGetPhysicalDeviceFormatProperties",
       reinterpret_cast<PFN_vkVoidFunction>(
           &FakeGetPhysicalDeviceFormatProperties)},
      {"vkGetPhysicalDeviceImageFormatProperties",
       reinterpret_cast<PFN_vkVoidFunction>(
           &FakeGetPhysicalDeviceImageFormatProperties)},
  };
  for (const ProcEntry& entry : kInstanceProcs) {
    if (name != nullptr && std::strcmp(entry.name, name) == 0) {
      return entry.function;
    }
  }
  return nullptr;
}

}  // namespace

VkCommandBuffer FakeVulkanDevice::Impl::Allocate(VkCommandPool pool,
                                                 VkCommandBufferLevel level) {
  const auto found = pools.find(pool);
  if (found == pools.end()) {
    Error("vkAllocateCommandBuffers", "unknown command pool");
    return VK_NULL_HANDLE;
  }
  const VkCommandBuffer command_buffer = NextHandle<VkCommandBuffer>();
  CommandBuffer record;
  record.pool = pool;
  record.level = level;
  command_buffers[command_buffer] = std::move(record);
  found->second.command_buffers.push_back(command_buffer);
  Register(command_buffer, this);
  ++counts.command_buffers.created;
  return command_buffer;
}

void FakeVulkanDevice::Impl::FreeCommandBuffer(VkCommandBuffer command_buffer) {
  if (command_buffer == VK_NULL_HANDLE) {
    return;
  }
  const auto found = command_buffers.find(command_buffer);
  if (found == command_buffers.end()) {
    Error("vkFreeCommandBuffers", "unknown command buffer");
    return;
  }
  const auto pool = pools.find(found->second.pool);
  if (pool != pools.end()) {
    auto& owned = pool->second.command_buffers;
    owned.erase(std::remove(owned.begin(), owned.end(), command_buffer),
                owned.end());
  }
  command_buffers.erase(found);
  Unregister(command_buffer);
  ++counts.command_buffers.destroyed;
}

namespace {

VKAPI_ATTR void VKAPI_CALL FakeGetDeviceQueue(VkDevice device,
                                              std::uint32_t family,
                                              std::uint32_t index,
                                              VkQueue* queue) {
  FAKE_ENTER(device, "vkGetDeviceQueue", );
  if (queue == nullptr) {
    return;
  }
  if (family >= impl->config.queue_families.size() ||
      index >= impl->config.queue_families[family].queueCount) {
    impl->Error("vkGetDeviceQueue", "no such queue",
                "VUID-vkGetDeviceQueue-queueIndex-00385");
    *queue = VK_NULL_HANDLE;
    return;
  }
  VkQueue& known = impl->queues[{family, index}];
  if (known == VK_NULL_HANDLE) {
    known = NextHandle<VkQueue>();
    Register(known, impl);
  }
  *queue = known;
}

}  // namespace

#undef FAKE_ENTER

FakeVulkanDevice::FakeVulkanDevice(Config config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
  Register(impl_->device, impl_.get());
  Register(impl_->physical_device, impl_.get());
}

FakeVulkanDevice::~FakeVulkanDevice() {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  for (const auto& [command_buffer, record] : impl_->command_buffers) {
    Unregister(command_buffer);
  }
  for (const auto& [key, queue] : impl_->queues) {
    Unregister(queue);
  }
  Unregister(impl_->physical_device);
  Unregister(impl_->device);
}

VkDevice FakeVulkanDevice::device() const { return impl_->device; }

VkPhysicalDevice FakeVulkanDevice::physical_device() const {
  return impl_->physical_device;
}

const FakeVulkanDevice::Config& FakeVulkanDevice::config() const {
  return impl_->config;
}

PFN_vkGetDeviceProcAddr FakeVulkanDevice::GetDeviceProcAddr() {
  return &FakeGetDeviceProcAddr;
}

PFN_vkGetInstanceProcAddr FakeVulkanDevice::GetInstanceProcAddr() {
  return &FakeGetInstanceProcAddr;
}

VkInstance FakeVulkanDevice::instance() {
  return reinterpret_cast<VkInstance>(std::uintptr_t{0x1A57A});
}

VkResult FakeVulkanDevice::CreateBuffer(VkDeviceSize size,
                                        VkBufferUsageFlags usage,
                                        VkBuffer* buffer) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->CreateBuffer(size, usage, false, buffer);
}

void FakeVulkanDevice::GetBufferMemoryRequirements(
    VkBuffer buffer, VkMemoryRequirements* requirements) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->Requirements(buffer, requirements);
}

VkResult FakeVulkanDevice::AllocateMemory(VkDeviceSize size,
                                          std::uint32_t type_index,
                                          VkDeviceMemory* memory) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->AllocateMemory(size, type_index, false, memory);
}

VkResult FakeVulkanDevice::BindBufferMemory(VkBuffer buffer,
                                            VkDeviceMemory memory,
                                            VkDeviceSize offset) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->BindBuffer(buffer, memory, offset);
}

VkResult FakeVulkanDevice::MapMemory(VkDeviceMemory memory, VkDeviceSize offset,
                                     VkDeviceSize size, void** data) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->Map(memory, offset, size, data);
}

void FakeVulkanDevice::UnmapMemory(VkDeviceMemory memory) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->Unmap(memory);
}

void FakeVulkanDevice::DestroyBuffer(VkBuffer buffer) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->DestroyBuffer(buffer);
}

void FakeVulkanDevice::FreeMemory(VkDeviceMemory memory) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->Free(memory);
}

VkResult FakeVulkanDevice::CreateImage(const VkImageCreateInfo& info,
                                       VkImage* image) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->CreateImage(info, image);
}

void FakeVulkanDevice::DestroyImage(VkImage image) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->DestroyImage(image);
}

void FakeVulkanDevice::DestroyImageView(VkImageView view) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->DestroyView(view);
}

VkCommandPool FakeVulkanDevice::CreateCommandPool(
    std::uint32_t queue_family, VkCommandPoolCreateFlags flags) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->CreatePool(queue_family, flags);
}

void FakeVulkanDevice::DestroyCommandPool(VkCommandPool pool) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->DestroyPool(pool);
}

VkResult FakeVulkanDevice::ResetCommandPool(VkCommandPool pool) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return ResetPool(impl_.get(), pool);
}

VkCommandBuffer FakeVulkanDevice::AllocateCommandBuffer(
    VkCommandPool pool, VkCommandBufferLevel level) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->Allocate(pool, level);
}

void FakeVulkanDevice::FreeCommandBuffer(VkCommandBuffer command_buffer) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->FreeCommandBuffer(command_buffer);
}

VkResult FakeVulkanDevice::BeginCommandBuffer(VkCommandBuffer command_buffer,
                                              VkCommandBufferUsageFlags flags) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->Begin(command_buffer, flags);
}

VkResult FakeVulkanDevice::EndCommandBuffer(VkCommandBuffer command_buffer) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->End(command_buffer);
}

VkResult FakeVulkanDevice::ResetCommandBuffer(VkCommandBuffer command_buffer) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->Reset(command_buffer);
}

VkQueue FakeVulkanDevice::GetQueue(std::uint32_t family, std::uint32_t index) {
  VkQueue queue = VK_NULL_HANDLE;
  FakeGetDeviceQueue(impl_->device, family, index, &queue);
  return queue;
}

void FakeVulkanDevice::FailNextSubmits(std::vector<VkResult> results) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->submit_results = std::move(results);
}

void FakeVulkanDevice::LoseDevice() {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->lost = true;
}

void FakeVulkanDevice::HoldFences(bool hold) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->hold_fences = hold;
  if (hold) {
    return;
  }
  for (VkFence fence : impl_->held_fences) {
    const auto found = impl_->fences.find(fence);
    if (found != impl_->fences.end()) {
      found->second = true;
    }
  }
  impl_->held_fences.clear();
}

VkResult FakeVulkanDevice::Submit(
    const std::vector<std::vector<VkCommandBuffer>>& batches) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->Submit(batches);
}

VkResult FakeVulkanDevice::QueueSubmit(VkQueue queue, std::uint32_t count,
                                       const VkSubmitInfo* submits,
                                       VkFence fence) {
  const bool known = Owner(queue) == impl_.get();
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (!known) {
    impl_->Error("vkQueueSubmit", "unknown queue");
    return VK_ERROR_DEVICE_LOST;
  }
  return SubmitInfos(impl_.get(), count, submits, fence);
}

VkResult FakeVulkanDevice::QueueSubmit2(VkQueue queue, std::uint32_t count,
                                        const VkSubmitInfo2* submits,
                                        VkFence fence) {
  const bool known = Owner(queue) == impl_.get();
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (!known) {
    impl_->Error("vkQueueSubmit2", "unknown queue");
    return VK_ERROR_DEVICE_LOST;
  }
  return SubmitInfos2(impl_.get(), count, submits, fence);
}

bool FakeVulkanDevice::GetImageInfo(VkImage image,
                                    VkImageCreateInfo* info) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto found = impl_->images.find(image);
  if (found == impl_->images.end()) {
    return false;
  }
  if (info != nullptr) {
    *info = found->second.info;
  }
  return true;
}

const std::vector<std::uint8_t>* FakeVulkanDevice::ImageSubresource(
    VkImage image, std::uint32_t mip, std::uint32_t layer) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  return impl_->Subresource(image, mip, layer);
}

bool FakeVulkanDevice::GetImageViewInfo(VkImageView view,
                                        VkImageViewCreateInfo* info) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto found = impl_->views.find(view);
  if (found == impl_->views.end()) {
    return false;
  }
  if (info != nullptr) {
    *info = found->second;
  }
  return true;
}

std::uint8_t* FakeVulkanDevice::MemoryBytes(VkDeviceMemory memory) {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto found = impl_->memories.find(memory);
  return found != impl_->memories.end() ? found->second.bytes.data() : nullptr;
}

std::uint32_t FakeVulkanDevice::MemoryTypeIndex(VkDeviceMemory memory) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto found = impl_->memories.find(memory);
  return found != impl_->memories.end() ? found->second.type : UINT32_MAX;
}

VkDeviceSize FakeVulkanDevice::BufferSize(VkBuffer buffer) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto found = impl_->buffers.find(buffer);
  return found != impl_->buffers.end() ? found->second.size : 0;
}

VkDeviceMemory FakeVulkanDevice::BufferMemory(VkBuffer buffer) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto found = impl_->buffers.find(buffer);
  return found != impl_->buffers.end() ? found->second.memory : VK_NULL_HANDLE;
}

VkBufferUsageFlags FakeVulkanDevice::BufferUsage(VkBuffer buffer) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto found = impl_->buffers.find(buffer);
  return found != impl_->buffers.end() ? found->second.usage : 0;
}

bool FakeVulkanDevice::IsMapped(VkDeviceMemory memory) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto found = impl_->memories.find(memory);
  return found != impl_->memories.end() && found->second.mapped;
}

const FakeVulkanDevice::Counts& FakeVulkanDevice::counts() const {
  return impl_->counts;
}

std::uint64_t FakeVulkanDevice::calls(const std::string& name) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto found = impl_->calls.find(name);
  return found != impl_->calls.end() ? found->second : 0;
}

const std::vector<FakeVulkanDevice::ImageTransfer>&
FakeVulkanDevice::image_transfers() const {
  return impl_->transfers;
}

std::uint64_t FakeVulkanDevice::copy_buffer_to_image_regions() const {
  return impl_->copy_regions;
}

std::vector<FakeVulkanDevice::RecordedBufferCopy>
FakeVulkanDevice::RecordedBufferCopies(VkCommandBuffer command_buffer) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  std::vector<RecordedBufferCopy> copies;
  const auto found = impl_->command_buffers.find(command_buffer);
  if (found == impl_->command_buffers.end()) {
    return copies;
  }
  for (const Impl::Command& command : found->second.commands) {
    if (command.kind == Impl::Command::Kind::kCopyBufferToImage) {
      copies.push_back(
          {command.buffer, command.image, command.buffer_image_regions});
    }
  }
  return copies;
}

const std::vector<std::string>& FakeVulkanDevice::errors() const {
  return impl_->errors;
}

void FakeVulkanDevice::ClearErrors() {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->errors.clear();
}

}  // namespace mocktail::graphics::testing
