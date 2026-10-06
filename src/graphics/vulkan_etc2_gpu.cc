#include "mocktail/graphics/vulkan_etc2_gpu.h"

#include <pthread.h>
#include <signal.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
#include <unordered_map>
#include <utility>

#include "generated/etc2_shaders_spv.h"
#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/vulkan_etc2_gpu_plan.h"

namespace mocktail::graphics {

namespace {

using Clock = std::chrono::steady_clock;

constexpr VkDeviceSize kMiB = 1024ULL * 1024ULL;
// Output arenas (DESIGN C3.6): 32 MiB chunks, 32 MiB kept after 2 s idle,
// and by default min(256 MiB, VRAM / 24) device-local per device.
constexpr VkDeviceSize kOutputChunkBytes = 32 * kMiB;
constexpr VkDeviceSize kOutputKeepIdleBytes = 32 * kMiB;
constexpr VkDeviceSize kDefaultArenaCap = 256 * kMiB;
// Slot inputs start at 4 MiB, grow x2, shrink back after 2 s; all inputs of
// a device hold at most 64 MiB of device-local memory.
constexpr VkDeviceSize kInputStartBytes = 4 * kMiB;
constexpr VkDeviceSize kInputDeviceLocalCap = 64 * kMiB;
constexpr VkDeviceSize kJobsStartBytes = 64 * 1024;
constexpr auto kIdleShrink = std::chrono::seconds(2);
// [CR-6] A 50 ms wait inside vkQueueSubmit is itself a stutter, so it only
// happens once 32 calls are in flight on one queue.
constexpr std::uint32_t kMaxSlots = 32;
constexpr std::uint64_t kSlotWaitNs = 50ULL * 1000ULL * 1000ULL;
constexpr std::uint64_t kTeardownWaitNs = 1000ULL * 1000ULL * 1000ULL;
// [CR-5] Device-local chunks only while this much of the heap's budget is
// left: max(512 MiB, 10 % of the heap), read at most every 2 s.
constexpr VkDeviceSize kBudgetHeadroom = 512 * kMiB;
constexpr auto kBudgetInterval = std::chrono::seconds(2);
// Timestamps per pass: start, after the copy-in, after the decode.
constexpr std::uint32_t kQueriesPerPass = 3;
constexpr std::uint32_t kSetsPerPool = 64;
// The selftest: 16 x 13 blocks (64 x 52 texels) of every format.
constexpr std::uint32_t kSelftestBlocksWide = 16;
constexpr std::uint32_t kSelftestBlocksHigh = 13;
constexpr std::uint32_t kSelftestFormats = 7;
constexpr auto kSelftestTimeout = std::chrono::seconds(10);

constexpr VkBufferUsageFlags kOutputUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

const char* const kFormatNames[kSelftestFormats] = {
    "ETC2_RGB8", "ETC2_RGB8A1", "ETC2_RGBA8",    "EAC_R11",
    "EAC_R11_SNORM", "EAC_RG11", "EAC_RG11_SNORM"};

std::uint64_t Nanoseconds(Clock::duration duration) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

VkDeviceSize NextPowerOfTwo(VkDeviceSize value, VkDeviceSize floor) {
  VkDeviceSize size = floor;
  while (size < value && size <= (VkDeviceSize{1} << 62)) {
    size <<= 1;
  }
  return size;
}

double Mebibytes(std::uint64_t bytes) {
  return static_cast<double>(bytes) / static_cast<double>(kMiB);
}

std::mutex& RegistryMutex() {
  static auto* mutex = new std::mutex;
  return *mutex;
}

std::unordered_map<VkDevice, Etc2GpuDevice*>& Registry() {
  static auto* registry = new std::unordered_map<VkDevice, Etc2GpuDevice*>;
  return *registry;
}

// What memory a buffer of the GPU route lives in.
enum class MemoryKind {
  kDeviceLocal,  // VRAM, not host-visible when there is a choice
  kSystem,       // GPU-reachable system memory, not host-visible if possible
  kHost,         // host-visible coherent, cached when possible (no BAR)
};

struct GpuBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  std::uint8_t* mapped = nullptr;
  VkDeviceSize size = 0;
  bool device_local = false;
};

// Per-queue submit resources, recycled by a fence (DESIGN C3.3). One slot
// serves one whole submit call: every batch's pass comes from its pool,
// and its fence covers them all (CR-1).
struct Slot {
  enum class Phase {
    kFree,          // pool reset, fence unsignalled: ready to record
    kRecording,     // PlanSubmit took it; FinishSubmit decides what's next
    kInFlight,      // submitted; the fence will signal
    kNotSubmitted,  // the host submit failed: reset the pool, no wait
    kLost,          // the fence may never signal: never used again
  };
  VkCommandPool pool = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> command_buffers;
  std::uint32_t used = 0;
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool queries = VK_NULL_HANDLE;
  std::uint32_t query_capacity = 0;
  GpuBuffer input;
  GpuBuffer jobs;
  VkDescriptorSet set0 = VK_NULL_HANDLE;
  Phase phase = Phase::kFree;
  bool rescue = false;  // the queue's reserved rescue slot (CR-7)
  std::uint64_t epoch = 0;
  Clock::time_point input_needed{};
  struct QueueState* queue = nullptr;
  // The last submit, for the lagging "GPU submit" line.
  std::uint32_t timed_passes = 0;
  std::uint32_t pre_submits = 0;
  std::uint64_t uploads = 0;
  VkDeviceSize compressed = 0;
  VkDeviceSize decoded = 0;
  std::uint64_t record_ns = 0;
};

struct QueueState {
  std::mutex mutex;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t family = 0;
  std::vector<std::unique_ptr<Slot>> slots;
  std::unique_ptr<Slot> rescue;
};

struct Selftest {
  bool active = false;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t family = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  GpuBuffer input;
  GpuBuffer jobs;
  GpuBuffer output;
  VkDescriptorSet set0 = VK_NULL_HANDLE;
  VkDescriptorSet set1 = VK_NULL_HANDLE;
  std::vector<std::uint8_t> expected;
  VkDeviceSize output_offsets[kSelftestFormats] = {};
  VkDeviceSize output_bytes[kSelftestFormats] = {};
  Clock::time_point started{};
};

}  // namespace

const char* Etc2GpuStateName(Etc2GpuDevice::State state) {
  switch (state) {
    case Etc2GpuDevice::State::kIdle:
      return "idle";
    case Etc2GpuDevice::State::kBuilding:
      return "building";
    case Etc2GpuDevice::State::kBuilt:
      return "built";
    case Etc2GpuDevice::State::kTesting:
      return "testing";
    case Etc2GpuDevice::State::kReady:
      return "ready";
    case Etc2GpuDevice::State::kFailed:
      return "failed";
  }
  return "unknown";
}

const char* Etc2GpuFunctions::Missing() const {
#define MOCKTAIL_ETC2_GPU_REQUIRE(field, name) \
  if (field == nullptr) {                      \
    return name;                               \
  }
  MOCKTAIL_ETC2_GPU_REQUIRE(create_buffer, "vkCreateBuffer")
  MOCKTAIL_ETC2_GPU_REQUIRE(destroy_buffer, "vkDestroyBuffer")
  MOCKTAIL_ETC2_GPU_REQUIRE(get_buffer_memory_requirements,
                            "vkGetBufferMemoryRequirements")
  MOCKTAIL_ETC2_GPU_REQUIRE(allocate_memory, "vkAllocateMemory")
  MOCKTAIL_ETC2_GPU_REQUIRE(free_memory, "vkFreeMemory")
  MOCKTAIL_ETC2_GPU_REQUIRE(bind_buffer_memory, "vkBindBufferMemory")
  MOCKTAIL_ETC2_GPU_REQUIRE(map_memory, "vkMapMemory")
  MOCKTAIL_ETC2_GPU_REQUIRE(create_command_pool, "vkCreateCommandPool")
  MOCKTAIL_ETC2_GPU_REQUIRE(destroy_command_pool, "vkDestroyCommandPool")
  MOCKTAIL_ETC2_GPU_REQUIRE(reset_command_pool, "vkResetCommandPool")
  MOCKTAIL_ETC2_GPU_REQUIRE(allocate_command_buffers,
                            "vkAllocateCommandBuffers")
  MOCKTAIL_ETC2_GPU_REQUIRE(begin_command_buffer, "vkBeginCommandBuffer")
  MOCKTAIL_ETC2_GPU_REQUIRE(end_command_buffer, "vkEndCommandBuffer")
  MOCKTAIL_ETC2_GPU_REQUIRE(copy_buffer, "vkCmdCopyBuffer")
  MOCKTAIL_ETC2_GPU_REQUIRE(pipeline_barrier, "vkCmdPipelineBarrier")
  MOCKTAIL_ETC2_GPU_REQUIRE(bind_pipeline, "vkCmdBindPipeline")
  MOCKTAIL_ETC2_GPU_REQUIRE(bind_descriptor_sets, "vkCmdBindDescriptorSets")
  MOCKTAIL_ETC2_GPU_REQUIRE(push_constants, "vkCmdPushConstants")
  MOCKTAIL_ETC2_GPU_REQUIRE(dispatch, "vkCmdDispatch")
  MOCKTAIL_ETC2_GPU_REQUIRE(write_timestamp, "vkCmdWriteTimestamp")
  MOCKTAIL_ETC2_GPU_REQUIRE(reset_query_pool, "vkCmdResetQueryPool")
  MOCKTAIL_ETC2_GPU_REQUIRE(create_query_pool, "vkCreateQueryPool")
  MOCKTAIL_ETC2_GPU_REQUIRE(destroy_query_pool, "vkDestroyQueryPool")
  MOCKTAIL_ETC2_GPU_REQUIRE(get_query_pool_results, "vkGetQueryPoolResults")
  MOCKTAIL_ETC2_GPU_REQUIRE(create_fence, "vkCreateFence")
  MOCKTAIL_ETC2_GPU_REQUIRE(destroy_fence, "vkDestroyFence")
  MOCKTAIL_ETC2_GPU_REQUIRE(reset_fences, "vkResetFences")
  MOCKTAIL_ETC2_GPU_REQUIRE(get_fence_status, "vkGetFenceStatus")
  MOCKTAIL_ETC2_GPU_REQUIRE(wait_for_fences, "vkWaitForFences")
  MOCKTAIL_ETC2_GPU_REQUIRE(create_shader_module, "vkCreateShaderModule")
  MOCKTAIL_ETC2_GPU_REQUIRE(destroy_shader_module, "vkDestroyShaderModule")
  MOCKTAIL_ETC2_GPU_REQUIRE(create_compute_pipelines,
                            "vkCreateComputePipelines")
  MOCKTAIL_ETC2_GPU_REQUIRE(destroy_pipeline, "vkDestroyPipeline")
  MOCKTAIL_ETC2_GPU_REQUIRE(create_pipeline_layout, "vkCreatePipelineLayout")
  MOCKTAIL_ETC2_GPU_REQUIRE(destroy_pipeline_layout,
                            "vkDestroyPipelineLayout")
  MOCKTAIL_ETC2_GPU_REQUIRE(create_descriptor_set_layout,
                            "vkCreateDescriptorSetLayout")
  MOCKTAIL_ETC2_GPU_REQUIRE(destroy_descriptor_set_layout,
                            "vkDestroyDescriptorSetLayout")
  MOCKTAIL_ETC2_GPU_REQUIRE(create_descriptor_pool, "vkCreateDescriptorPool")
  MOCKTAIL_ETC2_GPU_REQUIRE(destroy_descriptor_pool,
                            "vkDestroyDescriptorPool")
  MOCKTAIL_ETC2_GPU_REQUIRE(allocate_descriptor_sets,
                            "vkAllocateDescriptorSets")
  MOCKTAIL_ETC2_GPU_REQUIRE(free_descriptor_sets, "vkFreeDescriptorSets")
  MOCKTAIL_ETC2_GPU_REQUIRE(update_descriptor_sets, "vkUpdateDescriptorSets")
#undef MOCKTAIL_ETC2_GPU_REQUIRE
  return nullptr;
}

Etc2GpuTestHooks& Etc2GpuTesting() {
  static auto* hooks = new Etc2GpuTestHooks;
  return *hooks;
}

struct Etc2GpuDevice::Impl {
  explicit Impl(Etc2GpuDeviceInfo device_info) : info(std::move(device_info)) {
    VkDeviceSize vram = 0;
    for (std::uint32_t heap = 0; heap < info.memory.memoryHeapCount; ++heap) {
      if ((info.memory.memoryHeaps[heap].flags &
           VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
        vram = std::max(vram, info.memory.memoryHeaps[heap].size);
      }
    }
    arena_cap = info.arena_cap_bytes != 0
                    ? info.arena_cap_bytes
                    : std::min<VkDeviceSize>(kDefaultArenaCap, vram / 24);
    if (arena_cap == 0) {
      arena_cap = kOutputChunkBytes;
    }
    const VkDeviceSize range = info.properties.limits.maxStorageBufferRange;
    max_output = std::min<VkDeviceSize>(range != 0 ? range : kOutputChunkBytes,
                                        kEtc2GpuMaxOutputBytes);
  }

  const Etc2GpuDeviceInfo info;
  const Etc2GpuFunctions& fns() const { return info.fns; }
  VkDeviceSize arena_cap = 0;
  VkDeviceSize max_output = 0;

  std::atomic<State> state{State::kIdle};
  mutable std::mutex state_mutex;  // failure, build_ms, the condition below
  mutable std::condition_variable state_changed;
  std::string failure;
  std::uint64_t build_ns = 0;
  std::mutex thread_mutex;
  pthread_t thread{};
  bool thread_running = false;  // started, and neither joined nor detached
  bool destroyed = false;

  // Built by the build thread before the state turns kBuilt; read-only
  // after that.
  VkDescriptorSetLayout set_layout0 = VK_NULL_HANDLE;  // input, jobs
  VkDescriptorSetLayout set_layout1 = VK_NULL_HANDLE;  // output
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline decode_pipeline = VK_NULL_HANDLE;

  // [CR-9] Leaf lock: sets come from recording threads (an output chunk's
  // set 1), submit threads (slot set 0) and the trim path.
  std::mutex descriptor_mutex;
  std::vector<VkDescriptorPool> descriptor_pools;
  std::unordered_map<VkDescriptorSet, VkDescriptorPool> set_pools;

  std::mutex arenas_mutex;
  std::map<std::uint32_t, std::unique_ptr<ChunkArena>> arenas;  // by family
  // [CR-5] One device-local cap for the arenas of every family.
  std::atomic<VkDeviceSize> cap_used{0};
  std::mutex output_sets_mutex;
  std::unordered_map<VkBuffer, VkDescriptorSet> output_sets;
  std::atomic<VkDeviceSize> input_device_local{0};

  std::mutex budget_mutex;
  Clock::time_point budget_read{};
  bool budget_known = false;
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};

  // [CR-8] The submit epoch: every submitted slot gets the next one. A
  // chunk freed at epoch E is destroyed once no slot of epoch <= E can
  // still be pending.
  std::mutex epoch_mutex;
  std::uint64_t epoch = 0;
  std::map<std::uint64_t, VkFence> in_flight;

  std::mutex queues_mutex;
  std::unordered_map<VkQueue, std::unique_ptr<QueueState>> queues;

  std::mutex selftest_mutex;
  Selftest selftest;

  mutable std::mutex stats_mutex;
  Stats stats;
  std::atomic<unsigned> failure_logs{0};

  void Count(const std::function<void(Etc2DecodeTotals&)>& update) {
    if (info.count) {
      info.count(update);
    }
  }

  template <typename Update>
  void CountStats(Update&& update) {
    std::lock_guard<std::mutex> lock(stats_mutex);
    update(stats);
  }

  void LogOnce(const char* message, VkResult result) {
    if (failure_logs.fetch_add(1, std::memory_order_relaxed) < 8) {
      std::fprintf(stderr, "  [vulkan] ETC2 GPU decode: %s (VkResult %d)\n",
                   message, static_cast<int>(result));
    }
  }

  void Fail(const std::string& reason) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      if (state.load() == State::kFailed) {
        return;
      }
      failure = reason;
      state.store(State::kFailed, std::memory_order_release);
    }
    state_changed.notify_all();
    std::fprintf(stderr, "  [vulkan] ETC2 GPU decode unavailable: %s\n",
                 reason.c_str());
  }

  void SetState(State next) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      if (state.load() == State::kFailed) {
        return;
      }
      state.store(next, std::memory_order_release);
    }
    state_changed.notify_all();
  }

  bool FamilyComputes(std::uint32_t family) const {
    return family < info.families.size() &&
           (info.families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
  }

  // ---- memory -------------------------------------------------------------

  std::uint32_t FindType(std::uint32_t bits, VkMemoryPropertyFlags required,
                         VkMemoryPropertyFlags preferred,
                         VkMemoryPropertyFlags avoided) const {
    std::uint32_t fallback = UINT32_MAX;
    for (std::uint32_t index = 0; index < info.memory.memoryTypeCount;
         ++index) {
      const VkMemoryPropertyFlags flags =
          info.memory.memoryTypes[index].propertyFlags;
      if ((bits & (1U << index)) == 0 || (flags & required) != required ||
          (flags & VK_MEMORY_PROPERTY_PROTECTED_BIT) != 0 ||
          (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0) {
        continue;
      }
      if ((flags & preferred) == preferred && (flags & avoided) == 0) {
        return index;
      }
      if (fallback == UINT32_MAX && (flags & avoided) == 0) {
        fallback = index;
      }
    }
    return fallback;
  }

  std::uint32_t TypeFor(std::uint32_t bits, MemoryKind kind) const {
    constexpr VkMemoryPropertyFlags kVisible =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    std::uint32_t type = UINT32_MAX;
    switch (kind) {
      case MemoryKind::kDeviceLocal:
        type = FindType(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (type == UINT32_MAX) {  // integrated GPUs: all memory is both
          type = FindType(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, 0);
        }
        break;
      case MemoryKind::kSystem:
        // The GPU writes it over PCIe; nobody maps it (type 0 here).
        type = FindType(bits, 0, 0,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        if (type == UINT32_MAX) {
          type = FindType(bits, kVisible, 0,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        if (type == UINT32_MAX) {
          type = FindType(bits, 0, 0, 0);
        }
        break;
      case MemoryKind::kHost:
        // [CR-23] System memory, cached when possible, never the BAR.
        type = FindType(bits, kVisible, VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX) {
          type = FindType(bits, kVisible, 0, 0);
        }
        break;
    }
    return type;
  }

  bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    MemoryKind kind, GpuBuffer* out) {
    *out = GpuBuffer{};
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (fns().create_buffer(info.device, &buffer_info, nullptr,
                            &out->buffer) != VK_SUCCESS) {
      out->buffer = VK_NULL_HANDLE;
      return false;
    }
    VkMemoryRequirements requirements{};
    fns().get_buffer_memory_requirements(info.device, out->buffer,
                                         &requirements);
    const std::uint32_t type = TypeFor(requirements.memoryTypeBits, kind);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type;
    if (type == UINT32_MAX ||
        fns().allocate_memory(info.device, &allocate, nullptr,
                              &out->memory) != VK_SUCCESS) {
      out->memory = VK_NULL_HANDLE;
      DestroyBuffer(out);
      return false;
    }
    const VkMemoryPropertyFlags flags =
        info.memory.memoryTypes[type].propertyFlags;
    out->size = size;
    out->device_local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    void* data = nullptr;
    if (fns().bind_buffer_memory(info.device, out->buffer, out->memory, 0) !=
            VK_SUCCESS ||
        (kind == MemoryKind::kHost &&
         fns().map_memory(info.device, out->memory, 0, VK_WHOLE_SIZE, 0,
                          &data) != VK_SUCCESS)) {
      DestroyBuffer(out);
      return false;
    }
    out->mapped = static_cast<std::uint8_t*>(data);
    return true;
  }

  // Freeing the memory unmaps it.
  void DestroyBuffer(GpuBuffer* buffer) {
    if (buffer->buffer != VK_NULL_HANDLE) {
      fns().destroy_buffer(info.device, buffer->buffer, nullptr);
    }
    if (buffer->memory != VK_NULL_HANDLE) {
      fns().free_memory(info.device, buffer->memory, nullptr);
    }
    *buffer = GpuBuffer{};
  }

  // [CR-5] False when the device-local heap's budget has less than
  // max(512 MiB, 10 %) of headroom: the chunk then goes to system memory,
  // so another game's (or Roblox's own) allocations are not the victim.
  bool BudgetAllows(VkDeviceSize bytes) {
    if (fns().get_memory_properties2 == nullptr ||
        info.physical_device == VK_NULL_HANDLE) {
      return true;
    }
    std::uint32_t heap = UINT32_MAX;
    VkDeviceSize heap_size = 0;
    for (std::uint32_t index = 0; index < info.memory.memoryHeapCount;
         ++index) {
      if ((info.memory.memoryHeaps[index].flags &
           VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 &&
          info.memory.memoryHeaps[index].size > heap_size) {
        heap = index;
        heap_size = info.memory.memoryHeaps[index].size;
      }
    }
    if (heap == UINT32_MAX) {
      return true;
    }
    std::lock_guard<std::mutex> lock(budget_mutex);
    const Clock::time_point now = Clock::now();
    if (!budget_known || now - budget_read >= kBudgetInterval) {
      VkPhysicalDeviceMemoryBudgetPropertiesEXT read{};
      read.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
      VkPhysicalDeviceMemoryProperties2 properties{};
      properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
      properties.pNext = &read;
      fns().get_memory_properties2(info.physical_device, &properties);
      budget = read;
      budget.pNext = nullptr;
      budget_known = true;
      budget_read = now;
    }
    const VkDeviceSize available =
        budget.heapBudget[heap] > budget.heapUsage[heap]
            ? budget.heapBudget[heap] - budget.heapUsage[heap]
            : 0;
    const VkDeviceSize headroom =
        std::max<VkDeviceSize>(kBudgetHeadroom, heap_size / 10);
    if (budget.heapBudget[heap] == 0 || available >= headroom + bytes) {
      budget.heapUsage[heap] += bytes;  // until the next read
      return true;
    }
    return false;
  }

  // ---- descriptors ---------------------------------------------------------

  VkDescriptorSet AllocateSet(VkDescriptorSetLayout layout) {
    std::lock_guard<std::mutex> lock(descriptor_mutex);
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    for (auto pool = descriptor_pools.rbegin(); pool != descriptor_pools.rend();
         ++pool) {
      allocate.descriptorPool = *pool;
      if (fns().allocate_descriptor_sets(info.device, &allocate, &set) ==
          VK_SUCCESS) {
        set_pools[set] = *pool;
        return set;
      }
    }
    // [CR-9] Chain another pool instead of failing into arena_full.
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                              2 * kSetsPerPool};
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = kSetsPerPool;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (fns().create_descriptor_pool(info.device, &pool_info, nullptr,
                                     &pool) != VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    descriptor_pools.push_back(pool);
    allocate.descriptorPool = pool;
    if (fns().allocate_descriptor_sets(info.device, &allocate, &set) !=
        VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    set_pools[set] = pool;
    return set;
  }

  void FreeSet(VkDescriptorSet set) {
    if (set == VK_NULL_HANDLE) {
      return;
    }
    std::lock_guard<std::mutex> lock(descriptor_mutex);
    const auto found = set_pools.find(set);
    if (found != set_pools.end()) {
      fns().free_descriptor_sets(info.device, found->second, 1, &set);
      set_pools.erase(found);
    }
  }

  // Points the set's storage-buffer bindings 0.. at `buffers`, whole.
  void WriteSet(VkDescriptorSet set,
                std::initializer_list<const GpuBuffer*> buffers) {
    VkDescriptorBufferInfo infos[2]{};
    VkWriteDescriptorSet writes[2]{};
    std::uint32_t count = 0;
    for (const GpuBuffer* buffer : buffers) {
      infos[count] = {buffer->buffer, 0, StorageRange(buffer->size)};
      writes[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[count].dstSet = set;
      writes[count].dstBinding = count;
      writes[count].descriptorCount = 1;
      writes[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[count].pBufferInfo = &infos[count];
      ++count;
    }
    fns().update_descriptor_sets(info.device, count, writes, 0, nullptr);
  }

  VkDeviceSize StorageRange(VkDeviceSize size) const {
    const VkDeviceSize range = info.properties.limits.maxStorageBufferRange;
    return range != 0 ? std::min(size, range) : size;
  }

  // ---- the build ------------------------------------------------------------

  void Build() {
    if (const int delay = Etc2GpuTesting().build_delay_ms.load(); delay > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
    const Clock::time_point start = Clock::now();
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = sizeof(kEtc2DecodeSpv);
    module_info.pCode = kEtc2DecodeSpv;
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult result = fns().create_shader_module(info.device, &module_info,
                                                 nullptr, &module);
    if (result != VK_SUCCESS) {
      Fail("vkCreateShaderModule returned " + std::to_string(result));
      return;
    }
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (std::uint32_t index = 0; index < 2; ++index) {
      bindings[index].binding = index;
      bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[index].descriptorCount = 1;
      bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;
    result = fns().create_descriptor_set_layout(info.device, &layout_info,
                                                nullptr, &set_layout0);
    if (result == VK_SUCCESS) {
      layout_info.bindingCount = 1;
      result = fns().create_descriptor_set_layout(info.device, &layout_info,
                                                  nullptr, &set_layout1);
    }
    if (result == VK_SUCCESS) {
      const VkDescriptorSetLayout layouts[2] = {set_layout0, set_layout1};
      VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               2 * sizeof(std::uint32_t)};
      VkPipelineLayoutCreateInfo pipeline_layout_info{};
      pipeline_layout_info.sType =
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      pipeline_layout_info.setLayoutCount = 2;
      pipeline_layout_info.pSetLayouts = layouts;
      pipeline_layout_info.pushConstantRangeCount = 1;
      pipeline_layout_info.pPushConstantRanges = &push;
      result = fns().create_pipeline_layout(info.device, &pipeline_layout_info,
                                            nullptr, &pipeline_layout);
    }
    if (result == VK_SUCCESS) {
      VkComputePipelineCreateInfo pipeline_info{};
      pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      pipeline_info.stage.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      pipeline_info.stage.module = module;
      pipeline_info.stage.pName = "main";
      pipeline_info.layout = pipeline_layout;
      result = fns().create_compute_pipelines(info.device, VK_NULL_HANDLE, 1,
                                              &pipeline_info, nullptr,
                                              &decode_pipeline);
      if (result != VK_SUCCESS) {
        decode_pipeline = VK_NULL_HANDLE;
      }
    }
    fns().destroy_shader_module(info.device, module, nullptr);
    if (result != VK_SUCCESS) {
      Fail("building the decode pipeline returned " + std::to_string(result));
      return;
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      build_ns = Nanoseconds(Clock::now() - start);
    }
    SetState(State::kBuilt);
  }

  // ---- output arenas --------------------------------------------------------

  bool CreateOutputChunk(VkDeviceSize size, bool over_cap,
                         ArenaChunkStorage* storage) {
    GpuBuffer buffer;
    bool made = false;
    if (!over_cap && BudgetAllows(size)) {
      made = CreateBuffer(size, kOutputUsage, MemoryKind::kDeviceLocal, &buffer);
    }
    if (!made) {
      made = CreateBuffer(size, kOutputUsage, MemoryKind::kSystem, &buffer);
    }
    if (!made) {
      return false;
    }
    const VkDescriptorSet set = AllocateSet(set_layout1);
    if (set == VK_NULL_HANDLE) {
      DestroyBuffer(&buffer);
      return false;
    }
    WriteSet(set, {&buffer});
    {
      std::lock_guard<std::mutex> lock(output_sets_mutex);
      output_sets[buffer.buffer] = set;
    }
    storage->buffer = buffer.buffer;
    storage->memory = buffer.memory;
    storage->mapped = nullptr;
    storage->size = size;
    // A chunk outside VRAM does not count against the device-local cap.
    storage->over_cap = !buffer.device_local;
    return true;
  }

  void DestroyOutputChunk(const ArenaChunkStorage& storage) {
    VkDescriptorSet set = VK_NULL_HANDLE;
    {
      std::lock_guard<std::mutex> lock(output_sets_mutex);
      const auto found = output_sets.find(storage.buffer);
      if (found != output_sets.end()) {
        set = found->second;
        output_sets.erase(found);
      }
    }
    FreeSet(set);
    GpuBuffer buffer;
    buffer.buffer = storage.buffer;
    buffer.memory = storage.memory;
    DestroyBuffer(&buffer);
  }

  VkDescriptorSet OutputSet(VkBuffer buffer) {
    std::lock_guard<std::mutex> lock(output_sets_mutex);
    const auto found = output_sets.find(buffer);
    return found != output_sets.end() ? found->second : VK_NULL_HANDLE;
  }

  ChunkArena* Arena(std::uint32_t family) {
    std::lock_guard<std::mutex> lock(arenas_mutex);
    std::unique_ptr<ChunkArena>& arena = arenas[family];
    if (arena == nullptr) {
      ChunkArena::Config config;
      config.chunk_bytes = kOutputChunkBytes;
      config.keep_idle_bytes = kOutputKeepIdleBytes;
      config.alignment = kEtc2GpuRegionAlignment;
      config.cap_bytes = arena_cap;
      config.shared_cap_used = &cap_used;
      ChunkArena::Backend backend;
      backend.create = [this](VkDeviceSize size, bool over_cap,
                              ArenaChunkStorage* storage) {
        return CreateOutputChunk(size, over_cap, storage);
      };
      backend.destroy = [this](const ArenaChunkStorage& storage) {
        DestroyOutputChunk(storage);
      };
      backend.epoch = [this]() {
        std::lock_guard<std::mutex> epoch_lock(epoch_mutex);
        return epoch;
      };
      arena = std::make_unique<ChunkArena>(config, std::move(backend));
    }
    return arena.get();
  }

  // The oldest epoch that may still be pending; past every submit when
  // none is. Polls the fences of other queues' slots too (vkGetFenceStatus
  // needs no external synchronisation), so a queue that went quiet does not
  // pin chunks forever.
  std::uint64_t CompletedEpoch() {
    std::lock_guard<std::mutex> lock(epoch_mutex);
    for (auto it = in_flight.begin(); it != in_flight.end();) {
      const VkResult status = fns().get_fence_status(info.device, it->second);
      if (status == VK_NOT_READY) {
        return it->first;
      }
      it = in_flight.erase(it);
    }
    return epoch + 1;
  }

  void TrimArenas(Clock::time_point now) {
    std::vector<ChunkArena*> all;
    {
      std::lock_guard<std::mutex> lock(arenas_mutex);
      for (auto& [family, arena] : arenas) {
        all.push_back(arena.get());
      }
    }
    if (all.empty()) {
      return;
    }
    const std::uint64_t completed = CompletedEpoch();
    for (ChunkArena* arena : all) {
      arena->Trim(now, completed);
    }
  }

  // ---- slots -----------------------------------------------------------------

  QueueState* Queue(VkQueue queue, std::uint32_t family) {
    std::lock_guard<std::mutex> lock(queues_mutex);
    std::unique_ptr<QueueState>& state_for_queue = queues[queue];
    if (state_for_queue == nullptr) {
      state_for_queue = std::make_unique<QueueState>();
      state_for_queue->queue = queue;
      state_for_queue->family = family;
    }
    return state_for_queue.get();
  }

  bool CreateSlot(std::uint32_t family, bool rescue, Slot* slot) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = family;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (fns().create_command_pool(info.device, &pool_info, nullptr,
                                  &slot->pool) != VK_SUCCESS) {
      slot->pool = VK_NULL_HANDLE;
      return false;
    }
    if (fns().create_fence(info.device, &fence_info, nullptr, &slot->fence) !=
        VK_SUCCESS) {
      slot->fence = VK_NULL_HANDLE;
      fns().destroy_command_pool(info.device, slot->pool, nullptr);
      slot->pool = VK_NULL_HANDLE;
      return false;
    }
    slot->rescue = rescue;
    CountStats([](Stats& counts) {
      ++counts.fences_created;
      ++counts.slots_created;
      ++counts.live_slots;
    });
    return true;
  }

  void DestroySlot(Slot* slot) {
    if (slot->set0 != VK_NULL_HANDLE) {
      FreeSet(slot->set0);
    }
    if (slot->input.device_local) {
      input_device_local.fetch_sub(slot->input.size);
    }
    DestroyBuffer(&slot->input);
    DestroyBuffer(&slot->jobs);
    if (slot->queries != VK_NULL_HANDLE) {
      fns().destroy_query_pool(info.device, slot->queries, nullptr);
    }
    if (slot->pool != VK_NULL_HANDLE) {
      // Frees its command buffers.
      fns().destroy_command_pool(info.device, slot->pool, nullptr);
    }
    if (slot->fence != VK_NULL_HANDLE) {
      fns().destroy_fence(info.device, slot->fence, nullptr);
      CountStats([](Stats& counts) {
        ++counts.fences_destroyed;
        --counts.live_slots;
      });
    }
    *slot = Slot{};
  }

  // Reads the slot's timestamps once its fence signalled, and reports the
  // submit they belong to.
  void ReportSlot(Slot& slot) {
    std::uint64_t copy_ns = 0;
    std::uint64_t decode_ns = 0;
    const std::uint32_t family = slot.queue != nullptr ? slot.queue->family : 0;
    const std::uint32_t valid_bits =
        family < info.families.size()
            ? info.families[family].timestampValidBits
            : 0;
    if (slot.timed_passes != 0 && slot.queries != VK_NULL_HANDLE &&
        valid_bits != 0) {
      const std::uint32_t queries = slot.timed_passes * kQueriesPerPass;
      std::vector<std::uint64_t> raw(queries);
      if (fns().get_query_pool_results(
              info.device, slot.queries, 0, queries,
              raw.size() * sizeof(std::uint64_t), raw.data(),
              sizeof(std::uint64_t),
              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) ==
          VK_SUCCESS) {
        const std::uint64_t mask =
            valid_bits >= 64 ? ~0ULL : (1ULL << valid_bits) - 1;
        const double period = info.properties.limits.timestampPeriod;
        for (std::uint32_t pass = 0; pass < slot.timed_passes; ++pass) {
          const std::uint64_t* t = raw.data() + pass * kQueriesPerPass;
          copy_ns += static_cast<std::uint64_t>(
              static_cast<double>((t[1] - t[0]) & mask) * period);
          decode_ns += static_cast<std::uint64_t>(
              static_cast<double>((t[2] - t[1]) & mask) * period);
        }
      }
    }
    if (slot.uploads == 0) {
      return;
    }
    const std::uint64_t gpu_ns = copy_ns + decode_ns;
    if (gpu_ns != 0) {
      Count([gpu_ns](Etc2DecodeTotals& totals) {
        totals.gpu_time_ns += gpu_ns;
      });
    }
    if (info.frame_stats &&
        (slot.record_ns >= 1000000ULL || gpu_ns >= 4000000ULL)) {
      std::uint64_t lag = 0;
      {
        std::lock_guard<std::mutex> lock(epoch_mutex);
        lag = epoch - slot.epoch;
      }
      std::fprintf(stderr,
                   "  [vulkan] ETC2 GPU submit: %llu uploads, %.1f MiB "
                   "compressed, %.1f MiB written, record %.2f ms, gpu %.2f ms "
                   "(copy-in %.2f, decode %.2f; lag %llu)\n",
                   static_cast<unsigned long long>(slot.uploads),
                   Mebibytes(slot.compressed), Mebibytes(slot.decoded),
                   static_cast<double>(slot.record_ns) / 1e6,
                   static_cast<double>(gpu_ns) / 1e6,
                   static_cast<double>(copy_ns) / 1e6,
                   static_cast<double>(decode_ns) / 1e6,
                   static_cast<unsigned long long>(lag));
    }
  }

  // A slot whose fence signalled (or whose submit failed) becomes free:
  // its timestamps are read, its fence and pool reset.
  bool Recycle(Slot& slot) {
    if (slot.phase == Slot::Phase::kInFlight) {
      ReportSlot(slot);
      {
        std::lock_guard<std::mutex> lock(epoch_mutex);
        in_flight.erase(slot.epoch);
      }
      if (fns().reset_fences(info.device, 1, &slot.fence) != VK_SUCCESS) {
        slot.phase = Slot::Phase::kLost;
        return false;
      }
    }
    if (slot.used != 0 &&
        fns().reset_command_pool(info.device, slot.pool, 0) != VK_SUCCESS) {
      slot.phase = Slot::Phase::kLost;
      return false;
    }
    slot.used = 0;
    slot.timed_passes = 0;
    slot.pre_submits = 0;
    slot.uploads = 0;
    slot.compressed = 0;
    slot.decoded = 0;
    slot.record_ns = 0;
    slot.phase = Slot::Phase::kFree;
    return true;
  }

  // [CR-30] A lost device counts as signalled, and ends the GPU route.
  bool Signalled(VkResult status) {
    if (status == VK_ERROR_DEVICE_LOST) {
      Fail("the device was lost");
      return true;
    }
    return status == VK_SUCCESS;
  }

  // The oldest free slot (recycling every signalled one first), else a new
  // one (up to 32), else the oldest in flight once it signals within 50 ms.
  // Null: the call is rescued.
  Slot* AcquireSlot(QueueState& queue) {
    if (Etc2GpuTesting().fail_slot_acquire.load()) {
      return nullptr;
    }
    Slot* best = nullptr;
    for (const std::unique_ptr<Slot>& slot : queue.slots) {
      if (slot->phase == Slot::Phase::kInFlight &&
          Signalled(fns().get_fence_status(info.device, slot->fence))) {
        Recycle(*slot);
      }
      if ((slot->phase == Slot::Phase::kFree ||
           slot->phase == Slot::Phase::kNotSubmitted) &&
          (best == nullptr || slot->epoch < best->epoch)) {
        best = slot.get();
      }
    }
    if (best != nullptr) {
      return best->phase == Slot::Phase::kFree || Recycle(*best) ? best
                                                                  : nullptr;
    }
    if (queue.slots.size() < kMaxSlots) {
      auto slot = std::make_unique<Slot>();
      if (CreateSlot(queue.family, false, slot.get())) {
        slot->queue = &queue;
        queue.slots.push_back(std::move(slot));
        return queue.slots.back().get();
      }
    }
    Slot* oldest = nullptr;
    for (const std::unique_ptr<Slot>& slot : queue.slots) {
      if (slot->phase == Slot::Phase::kInFlight &&
          (oldest == nullptr || slot->epoch < oldest->epoch)) {
        oldest = slot.get();
      }
    }
    if (oldest == nullptr) {
      return nullptr;
    }
    const VkResult status = fns().wait_for_fences(
        info.device, 1, &oldest->fence, VK_TRUE, kSlotWaitNs);
    const bool signalled = Signalled(status);
    CountStats([signalled](Stats& counts) {
      ++counts.slot_waits;
      counts.slot_timeouts += signalled ? 0 : 1;
    });
    Count([](Etc2DecodeTotals& totals) { ++totals.gpu_slot_waits; });
    if (!signalled) {
      return nullptr;
    }
    return Recycle(*oldest) ? oldest : nullptr;
  }

  // [CR-7] The queue's reserved slot for a rescue when no normal slot came
  // free. Waits for it without a cap only when its fence was submitted.
  Slot* AcquireRescueSlot(QueueState& queue) {
    if (Etc2GpuTesting().fail_rescue_slot.load()) {
      return nullptr;
    }
    if (queue.rescue == nullptr) {
      auto slot = std::make_unique<Slot>();
      if (!CreateSlot(queue.family, true, slot.get())) {
        return nullptr;
      }
      slot->queue = &queue;
      queue.rescue = std::move(slot);
      return queue.rescue.get();
    }
    Slot& slot = *queue.rescue;
    switch (slot.phase) {
      case Slot::Phase::kFree:
        return &slot;
      case Slot::Phase::kNotSubmitted:
        return Recycle(slot) ? &slot : nullptr;
      case Slot::Phase::kInFlight:
        Signalled(fns().wait_for_fences(info.device, 1, &slot.fence, VK_TRUE,
                                        UINT64_MAX));
        return Recycle(slot) ? &slot : nullptr;
      case Slot::Phase::kRecording:
      case Slot::Phase::kLost:
        break;
    }
    return nullptr;
  }

  VkCommandBuffer NextCommandBuffer(Slot& slot) {
    if (slot.used == slot.command_buffers.size()) {
      VkCommandBufferAllocateInfo allocate{};
      allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      allocate.commandPool = slot.pool;
      allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocate.commandBufferCount = 1;
      VkCommandBuffer command_buffer = VK_NULL_HANDLE;
      if (fns().allocate_command_buffers(info.device, &allocate,
                                         &command_buffer) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
      }
      slot.command_buffers.push_back(command_buffer);
    }
    return slot.command_buffers[slot.used++];
  }

  // Grows (or, after 2 s without a big call, shrinks) the slot's jobs and
  // input buffers for this call, and keeps set 0 pointing at them. The slot
  // is idle, so nothing pending reads the old buffers.
  bool PrepareSlotBuffers(Slot& slot, VkDeviceSize input_bytes,
                          VkDeviceSize jobs_bytes, std::uint32_t passes,
                          Clock::time_point now) {
    bool rewrite = false;
    if (slot.jobs.size < jobs_bytes || slot.jobs.buffer == VK_NULL_HANDLE) {
      DestroyBuffer(&slot.jobs);
      if (!CreateBuffer(NextPowerOfTwo(jobs_bytes, kJobsStartBytes),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryKind::kHost,
                        &slot.jobs)) {
        return false;
      }
      rewrite = true;
    }
    if (input_bytes > StorageRange(input_bytes) ||
        input_bytes > kEtc2GpuMaxOutputBytes) {
      return false;
    }
    if (input_bytes > kInputStartBytes) {
      slot.input_needed = now;
    }
    const bool shrink = slot.input.size > kInputStartBytes &&
                        input_bytes <= kInputStartBytes &&
                        now - slot.input_needed >= kIdleShrink;
    if (slot.input.size < input_bytes || shrink ||
        slot.input.buffer == VK_NULL_HANDLE) {
      if (slot.input.device_local) {
        input_device_local.fetch_sub(slot.input.size);
      }
      DestroyBuffer(&slot.input);
      const VkDeviceSize size = std::min(
          NextPowerOfTwo(input_bytes, kInputStartBytes),
          std::max(input_bytes, StorageRange(kEtc2GpuMaxOutputBytes)));
      constexpr VkBufferUsageFlags kUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      bool made = false;
      if (input_device_local.fetch_add(size) + size <= kInputDeviceLocalCap &&
          BudgetAllows(size)) {
        made = CreateBuffer(size, kUsage, MemoryKind::kDeviceLocal,
                            &slot.input);
      }
      if (!made || !slot.input.device_local) {
        input_device_local.fetch_sub(size);
      }
      if (!made &&
          !CreateBuffer(size, kUsage, MemoryKind::kSystem, &slot.input)) {
        return false;
      }
      rewrite = true;
    }
    if (slot.set0 == VK_NULL_HANDLE) {
      slot.set0 = AllocateSet(set_layout0);
      if (slot.set0 == VK_NULL_HANDLE) {
        return false;
      }
      rewrite = true;
    }
    if (rewrite) {
      WriteSet(slot.set0, {&slot.input, &slot.jobs});
    }
    const std::uint32_t family = slot.queue->family;
    if (info.frame_stats && family < info.families.size() &&
        info.families[family].timestampValidBits != 0 &&
        slot.query_capacity < passes * kQueriesPerPass) {
      if (slot.queries != VK_NULL_HANDLE) {
        fns().destroy_query_pool(info.device, slot.queries, nullptr);
        slot.queries = VK_NULL_HANDLE;
        slot.query_capacity = 0;
      }
      VkQueryPoolCreateInfo query_info{};
      query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
      query_info.queryCount = std::max<std::uint32_t>(passes, 4) *
                              kQueriesPerPass;
      if (fns().create_query_pool(info.device, &query_info, nullptr,
                                  &slot.queries) == VK_SUCCESS) {
        slot.query_capacity = query_info.queryCount;
      } else {
        slot.queries = VK_NULL_HANDLE;
      }
    }
    return true;
  }

  void Barrier(VkCommandBuffer command_buffer, VkPipelineStageFlags source,
               VkAccessFlags source_access, VkPipelineStageFlags destination,
               VkAccessFlags destination_access) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = source_access;
    barrier.dstAccessMask = destination_access;
    fns().pipeline_barrier(command_buffer, source, destination, 0, 1, &barrier,
                           0, nullptr, 0, nullptr);
  }

  // DESIGN §5.6: orders the pass after earlier work on the queue that read
  // (Roblox's copies) or wrote (earlier passes) what it is about to write.
  void LeadingBarrier(VkCommandBuffer command_buffer) {
    constexpr VkPipelineStageFlags kStages =
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    Barrier(command_buffer, kStages,
            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, kStages,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  }

  bool Begin(VkCommandBuffer command_buffer) {
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return fns().begin_command_buffer(command_buffer, &begin) == VK_SUCCESS;
  }

  // Records one batch's decode pass (DESIGN C3.2): copy-in, decode, and the
  // barrier that orders the decode before Roblox's copies.
  bool RecordDecode(Slot& slot, VkCommandBuffer command_buffer,
                    const Etc2GpuBatch& batch, VkDeviceSize* input_cursor,
                    std::uint32_t* job_cursor) {
    Etc2DecodePass pass;
    if (!BuildDecodePass(batch.uploads, *input_cursor, *job_cursor, &pass) ||
        *input_cursor + pass.input_bytes > slot.input.size ||
        (static_cast<VkDeviceSize>(*job_cursor) + pass.jobs.size()) *
                sizeof(Etc2DecodeJob) >
            slot.jobs.size) {
      return false;
    }
    std::vector<VkDescriptorSet> outputs;
    for (const Etc2DecodeDispatch& dispatch : pass.dispatches) {
      outputs.push_back(OutputSet(dispatch.output));
      if (outputs.back() == VK_NULL_HANDLE) {
        return false;
      }
    }
    std::memcpy(slot.jobs.mapped +
                    static_cast<VkDeviceSize>(*job_cursor) *
                        sizeof(Etc2DecodeJob),
                pass.jobs.data(), pass.jobs.size() * sizeof(Etc2DecodeJob));
    if (!Begin(command_buffer)) {
      return false;
    }
    const bool timed = slot.queries != VK_NULL_HANDLE &&
                       slot.query_capacity >=
                           (slot.timed_passes + 1) * kQueriesPerPass;
    const std::uint32_t query = slot.timed_passes * kQueriesPerPass;
    if (timed) {
      fns().reset_query_pool(command_buffer, slot.queries, query,
                             kQueriesPerPass);
      fns().write_timestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            slot.queries, query);
    }
    LeadingBarrier(command_buffer);
    for (const Etc2InputCopies& copies : pass.copies) {
      fns().copy_buffer(command_buffer, copies.source, slot.input.buffer,
                        static_cast<std::uint32_t>(copies.regions.size()),
                        copies.regions.data());
    }
    if (timed) {
      fns().write_timestamp(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            slot.queries, query + 1);
    }
    Barrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);
    fns().bind_pipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        decode_pipeline);
    fns().bind_descriptor_sets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline_layout, 0, 1, &slot.set0, 0, nullptr);
    for (std::size_t index = 0; index < pass.dispatches.size(); ++index) {
      const Etc2DecodeDispatch& dispatch = pass.dispatches[index];
      fns().bind_descriptor_sets(command_buffer,
                                 VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline_layout, 1, 1, &outputs[index], 0,
                                 nullptr);
      const std::uint32_t push[2] = {dispatch.job_first, dispatch.job_count};
      fns().push_constants(command_buffer, pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
      fns().dispatch(command_buffer, dispatch.groups_x, dispatch.groups_y, 1);
    }
    if (timed) {
      fns().write_timestamp(command_buffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, slot.queries,
                            query + 2);
      ++slot.timed_passes;
    }
    Barrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    if (fns().end_command_buffer(command_buffer) != VK_SUCCESS) {
      return false;
    }
    *input_cursor += pass.input_bytes;
    *job_cursor += static_cast<std::uint32_t>(pass.jobs.size());
    return true;
  }

  // DESIGN §6, rescue: the batch's uploads decoded on the CPU into host
  // staging, copied into their arena ranges by a transfer pass, so the
  // rewritten copies never read stale texels.
  bool RecordRescue(VkCommandBuffer command_buffer, const Etc2GpuBatch& batch) {
    Etc2RescueDecode decode;
    {
      Etc2GpuTestHooks& hooks = Etc2GpuTesting();
      std::lock_guard<std::mutex> lock(hooks.mutex);
      decode = hooks.rescue_decode;
    }
    if (!decode) {
      decode = info.rescue_decode;
    }
    std::vector<Etc2RescueSource> sources;
    if (!decode || !decode(batch.command_buffers, batch.uploads, &sources) ||
        sources.size() != batch.uploads.size() || !Begin(command_buffer)) {
      return false;
    }
    LeadingBarrier(command_buffer);
    for (std::size_t index = 0; index < batch.uploads.size(); ++index) {
      const Etc2GpuUpload& upload = batch.uploads[index];
      const VkBufferCopy region{sources[index].offset, upload.output_offset,
                                upload.output_bytes};
      fns().copy_buffer(command_buffer, sources[index].buffer, upload.output, 1,
                        &region);
    }
    // [CR-7] Roblox's copy in the same batch reads what this wrote.
    Barrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
    return fns().end_command_buffer(command_buffer) == VK_SUCCESS;
  }

  template <typename Submit, typename Rewrite>
  void PlanSubmitCommon(VkQueue queue, std::uint32_t family,
                        std::uint32_t count, const Submit* submits,
                        const std::vector<Etc2GpuBatch>& batches,
                        VkFence app_fence, Etc2SubmitPlan* plan,
                        Rewrite rewrite) {
    if (batches.empty() || submits == nullptr || plan == nullptr) {
      return;
    }
    const Clock::time_point start = Clock::now();
    QueueState* queue_state = Queue(queue, family);
    std::lock_guard<std::mutex> lock(queue_state->mutex);
    // [CR-29] Trimming and shrinking run here, on the submitting thread.
    TrimArenas(start);
    const bool decode = state.load(std::memory_order_acquire) == State::kReady;
    Slot* slot = AcquireSlot(*queue_state);
    bool buffers = false;
    if (slot != nullptr && decode) {
      VkDeviceSize input_bytes = 0;
      VkDeviceSize jobs = 0;
      std::uint32_t passes = 0;
      for (const Etc2GpuBatch& batch : batches) {
        input_bytes = AlignUp(input_bytes, kEtc2GpuInputAlignment) +
                      Etc2DecodeInputBytes(batch.uploads);
        jobs += batch.uploads.size();
        ++passes;
      }
      buffers = PrepareSlotBuffers(*slot, input_bytes,
                                   jobs * sizeof(Etc2DecodeJob), passes, start);
      if (!buffers) {
        LogOnce("could not size a slot's buffers; rescuing", VK_SUCCESS);
      }
    }
    if (slot == nullptr) {
      slot = AcquireRescueSlot(*queue_state);
    }
    std::uint64_t uploads = 0;
    std::uint64_t rescued = 0;
    std::uint64_t dropped = 0;
    std::uint64_t rescues = 0;
    VkDeviceSize compressed = 0;
    VkDeviceSize decoded = 0;
    std::vector<Etc2BatchDecode> decodes;
    if (slot != nullptr) {
      slot->phase = Slot::Phase::kRecording;
      VkDeviceSize input_cursor = 0;
      std::uint32_t job_cursor = 0;
      for (const Etc2GpuBatch& batch : batches) {
        bool family_matches = true;
        VkDeviceSize batch_compressed = 0;
        VkDeviceSize batch_decoded = 0;
        for (const Etc2GpuUpload& upload : batch.uploads) {
          family_matches = family_matches && (upload.pool_family == family ||
                                              upload.pool_family ==
                                                  kEtc2UnknownQueueFamily);
          batch_compressed += upload.compressed;
          batch_decoded += upload.decoded;
        }
        if (!family_matches) {
          LogOnce("a batch's command buffers come from another queue family;"
                  " rescuing",
                  VK_SUCCESS);
        }
        const VkCommandBuffer command_buffer = NextCommandBuffer(*slot);
        bool recorded = false;
        if (command_buffer != VK_NULL_HANDLE && decode && buffers &&
            !slot->rescue && family_matches) {
          input_cursor = AlignUp(input_cursor, kEtc2GpuInputAlignment);
          recorded = RecordDecode(*slot, command_buffer, batch, &input_cursor,
                                  &job_cursor);
          if (recorded) {
            uploads += batch.uploads.size();
            compressed += batch_compressed;
            decoded += batch_decoded;
          }
        }
        if (!recorded && command_buffer != VK_NULL_HANDLE) {
          recorded = RecordRescue(command_buffer, batch);
          rescues += recorded ? 1 : 0;
          (recorded ? rescued : dropped) += batch.uploads.size();
        } else if (!recorded) {
          dropped += batch.uploads.size();
        }
        if (recorded) {
          decodes.push_back({batch.batch, command_buffer});
        }
      }
    } else {
      for (const Etc2GpuBatch& batch : batches) {
        dropped += batch.uploads.size();
      }
    }
    if (dropped != 0) {
      LogOnce("dropped GPU-route uploads: the rewritten copies read stale "
              "texels",
              VK_SUCCESS);
    }
    std::uint32_t pre_submits = 0;
    if (!decodes.empty()) {
      pre_submits =
          rewrite(count, submits, decodes, app_fence, slot->fence, plan);
      plan->gpu_slot = slot;
      slot->pre_submits = pre_submits;
      slot->uploads = uploads + rescued;
      slot->compressed = compressed;
      slot->decoded = decoded;
    } else if (slot != nullptr) {
      slot->phase = slot->used != 0 ? Slot::Phase::kNotSubmitted
                                    : Slot::Phase::kFree;
    }
    const std::uint64_t record_ns = Nanoseconds(Clock::now() - start);
    if (!decodes.empty()) {
      slot->record_ns = record_ns;
    }
    CountStats([&](Stats& counts) {
      counts.submits += decodes.empty() ? 0 : 1;
      counts.passes += decodes.size();
      counts.pre_submits += pre_submits;
      counts.rescues += rescues;
      counts.rescued_uploads += rescued;
      counts.dropped_uploads += dropped;
    });
    Count([&](Etc2DecodeTotals& totals) {
      totals.gpu_uploads += uploads;
      totals.gpu_submits += uploads != 0 ? 1 : 0;
      totals.gpu_compressed_bytes += compressed;
      totals.gpu_host_bytes += decoded;
      totals.gpu_record_ns += record_ns;
      totals.gpu_presubmits += pre_submits;
      totals.rescued_uploads += rescued;
      totals.dropped_uploads += dropped;
    });
  }

  // ---- the selftest (DESIGN C3.4) ---------------------------------------------

  void DestroySelftest() {
    Selftest& test = selftest;
    FreeSet(test.set0);
    FreeSet(test.set1);
    DestroyBuffer(&test.input);
    DestroyBuffer(&test.jobs);
    DestroyBuffer(&test.output);
    if (test.pool != VK_NULL_HANDLE) {
      fns().destroy_command_pool(info.device, test.pool, nullptr);
    }
    if (test.fence != VK_NULL_HANDLE) {
      fns().destroy_fence(info.device, test.fence, nullptr);
      CountStats([](Stats& counts) { ++counts.fences_destroyed; });
    }
    test = Selftest{};
  }

  // Deterministic blocks of every format: random bytes cover every ETC2
  // mode (individual, differential, T, H, planar), punch-through both ways
  // and EAC with every multiplier, signed and unsigned.
  bool SetUpSelftest(std::uint32_t family) {
    Selftest& test = selftest;
    test.family = family;
    const std::uint32_t blocks = kSelftestBlocksWide * kSelftestBlocksHigh;
    const std::uint32_t width = 4 * kSelftestBlocksWide;
    const std::uint32_t height = 4 * kSelftestBlocksHigh;
    VkDeviceSize input_offsets[kSelftestFormats] = {};
    VkDeviceSize input_bytes = 0;
    VkDeviceSize output_bytes = 0;
    for (std::uint32_t format = 0; format < kSelftestFormats; ++format) {
      const auto etc = static_cast<EtcFormat>(format);
      input_offsets[format] = AlignUp(input_bytes, kEtc2GpuInputAlignment);
      input_bytes = input_offsets[format] + blocks * EtcBlockBytes(etc);
      test.output_offsets[format] =
          AlignUp(output_bytes, kEtc2GpuRegionAlignment);
      test.output_bytes[format] = static_cast<VkDeviceSize>(width) * height *
                                  EtcDecodedTexelBytes(etc);
      output_bytes = test.output_offsets[format] + test.output_bytes[format];
    }
    if (!CreateBuffer(input_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      MemoryKind::kHost, &test.input) ||
        !CreateBuffer(kSelftestFormats * sizeof(Etc2DecodeJob),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryKind::kHost,
                      &test.jobs) ||
        !CreateBuffer(output_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      MemoryKind::kHost, &test.output)) {
      return false;
    }
    std::uint64_t seed = 0x9e3779b97f4a7c15ULL;
    for (VkDeviceSize byte = 0; byte < input_bytes; ++byte) {
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      test.input.mapped[byte] = static_cast<std::uint8_t>(seed >> 56);
    }
    std::memset(test.output.mapped, 0, output_bytes);
    test.expected.assign(output_bytes, 0);
    std::vector<Etc2DecodeJob> jobs(kSelftestFormats);
    std::uint32_t groups = 0;
    for (std::uint32_t format = 0; format < kSelftestFormats; ++format) {
      const auto etc = static_cast<EtcFormat>(format);
      const VkDeviceSize block_bytes = EtcBlockBytes(etc);
      Etc2DecodeJob& job = jobs[format];
      job.format = format;
      job.src_word = static_cast<std::uint32_t>(input_offsets[format] / 4);
      job.src_row_words =
          static_cast<std::uint32_t>(kSelftestBlocksWide * block_bytes / 4);
      job.src_layer_words =
          static_cast<std::uint32_t>(blocks * block_bytes / 4);
      job.blocks_wide = kSelftestBlocksWide;
      job.blocks_high = kSelftestBlocksHigh;
      job.block_count = blocks;
      job.dst_word = static_cast<std::uint32_t>(test.output_offsets[format] / 4);
      job.dst_row_words =
          static_cast<std::uint32_t>(width * EtcDecodedTexelBytes(etc) / 4);
      job.dst_layer_words = job.dst_row_words * height;
      job.group_begin = groups;
      groups += Etc2DecodeGroups(blocks);
      if (!DecodeEtcImage(etc, test.input.mapped + input_offsets[format],
                          blocks * block_bytes, width, height,
                          test.expected.data() + test.output_offsets[format],
                          test.output_bytes[format])) {
        return false;
      }
    }
    std::memcpy(test.jobs.mapped, jobs.data(),
                jobs.size() * sizeof(Etc2DecodeJob));
    test.set0 = AllocateSet(set_layout0);
    test.set1 = AllocateSet(set_layout1);
    if (test.set0 == VK_NULL_HANDLE || test.set1 == VK_NULL_HANDLE) {
      return false;
    }
    WriteSet(test.set0, {&test.input, &test.jobs});
    WriteSet(test.set1, {&test.output});
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = family;
    if (fns().create_command_pool(info.device, &pool_info, nullptr,
                                  &test.pool) != VK_SUCCESS) {
      test.pool = VK_NULL_HANDLE;
      return false;
    }
    VkCommandBufferAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate.commandPool = test.pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (fns().allocate_command_buffers(info.device, &allocate,
                                       &test.command_buffer) != VK_SUCCESS) {
      return false;
    }
    if (fns().create_fence(info.device, &fence_info, nullptr, &test.fence) !=
        VK_SUCCESS) {
      test.fence = VK_NULL_HANDLE;
      return false;
    }
    CountStats([](Stats& counts) { ++counts.fences_created; });
    const VkCommandBuffer command_buffer = test.command_buffer;
    if (!Begin(command_buffer)) {
      return false;
    }
    fns().bind_pipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        decode_pipeline);
    const VkDescriptorSet sets[2] = {test.set0, test.set1};
    fns().bind_descriptor_sets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline_layout, 0, 2, sets, 0, nullptr);
    const std::uint32_t push[2] = {0, kSelftestFormats};
    fns().push_constants(command_buffer, pipeline_layout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    Etc2SplitDispatch(groups, &x, &y);
    fns().dispatch(command_buffer, x, y, 1);
    Barrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_HOST_READ_BIT);
    return fns().end_command_buffer(command_buffer) == VK_SUCCESS;
  }

  void MaybeRunSelftest(VkQueue queue, std::uint32_t family,
                        Etc2SubmitPlan* plan) {
    if (plan == nullptr || !FamilyComputes(family) ||
        state.load(std::memory_order_acquire) != State::kBuilt) {
      return;
    }
    std::lock_guard<std::mutex> lock(selftest_mutex);
    State expected = State::kBuilt;
    {
      std::lock_guard<std::mutex> state_lock(state_mutex);
      if (!state.compare_exchange_strong(expected, State::kTesting)) {
        return;
      }
    }
    state_changed.notify_all();
    if (!SetUpSelftest(family)) {
      DestroySelftest();
      Fail("the selftest could not be set up");
      return;
    }
    selftest.active = true;
    selftest.queue = queue;
    selftest.started = Clock::now();
    Etc2PreSubmit pre;
    pre.info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    pre.info.commandBufferCount = 1;
    pre.info.pCommandBuffers = &selftest.command_buffer;
    pre.fence = selftest.fence;
    plan->pre_submit_batches.push_back(pre);
  }

  void PollSelftest() {
    if (state.load(std::memory_order_acquire) != State::kTesting) {
      return;
    }
    std::lock_guard<std::mutex> lock(selftest_mutex);
    Selftest& test = selftest;
    if (!test.active) {
      return;
    }
    const VkResult status = fns().get_fence_status(info.device, test.fence);
    if (status == VK_NOT_READY) {
      if (Clock::now() - test.started > kSelftestTimeout) {
        Fail("the selftest did not finish within 10 s");
      }
      return;
    }
    if (status != VK_SUCCESS) {
      DestroySelftest();
      Fail("the selftest's fence returned " + std::to_string(status));
      return;
    }
    if (Etc2GpuTesting().corrupt_selftest.load() && !test.expected.empty()) {
      test.expected[test.output_offsets[2] + 17] ^= 0x10;
    }
    for (std::uint32_t format = 0; format < kSelftestFormats; ++format) {
      const std::uint8_t* gpu = test.output.mapped + test.output_offsets[format];
      const std::uint8_t* cpu = test.expected.data() + test.output_offsets[format];
      const VkDeviceSize bytes = test.output_bytes[format];
      if (std::memcmp(gpu, cpu, bytes) == 0) {
        continue;
      }
      VkDeviceSize first = 0;
      while (first < bytes && gpu[first] == cpu[first]) {
        ++first;
      }
      const auto etc = static_cast<EtcFormat>(format);
      const VkDeviceSize row_bytes =
          4ULL * kSelftestBlocksWide * EtcDecodedTexelBytes(etc);
      const VkDeviceSize texel_y = first / row_bytes;
      const VkDeviceSize texel_x =
          (first % row_bytes) / EtcDecodedTexelBytes(etc);
      char reason[192];
      const VkDeviceSize at = first & ~VkDeviceSize{3};
      std::snprintf(reason, sizeof(reason),
                    "selftest mismatch: %s block %llu, byte %llu: gpu "
                    "%02x%02x%02x%02x, cpu %02x%02x%02x%02x",
                    kFormatNames[format],
                    static_cast<unsigned long long>(
                        (texel_y / 4) * kSelftestBlocksWide + texel_x / 4),
                    static_cast<unsigned long long>(first), gpu[at],
                    gpu[at + 1], gpu[at + 2], gpu[at + 3], cpu[at],
                    cpu[at + 1], cpu[at + 2], cpu[at + 3]);
      DestroySelftest();
      Fail(reason);
      return;
    }
    const std::uint32_t family = test.family;
    DestroySelftest();
    std::uint64_t build_ms = 0;
    {
      std::lock_guard<std::mutex> state_lock(state_mutex);
      build_ms = (build_ns + 500000ULL) / 1000000ULL;
    }
    SetState(State::kReady);
    if (state.load() == State::kReady) {
      std::fprintf(stderr,
                   "  [vulkan] ETC2 GPU decode ready on family %u (build %llu "
                   "ms, selftest ok)\n",
                   family, static_cast<unsigned long long>(build_ms));
    }
  }

  // ---- teardown --------------------------------------------------------------

  // Waits (1 s at most) for a fence that was submitted.
  void WaitSubmitted(VkFence fence, const char* what) {
    CountStats([](Stats& counts) { ++counts.teardown_waits; });
    const VkResult result =
        fns().wait_for_fences(info.device, 1, &fence, VK_TRUE, kTeardownWaitNs);
    if (result == VK_TIMEOUT) {
      std::fprintf(stderr,
                   "  [vulkan] ETC2 GPU decode: %s still pending after 1 s "
                   "at vkDestroyDevice; destroying it anyway\n",
                   what);
    }
  }

  void Destroy() {
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      if (destroyed) {
        return;
      }
      destroyed = true;
      if (thread_running) {
        pthread_join(thread, nullptr);
        thread_running = false;
      }
    }
    std::unordered_map<VkQueue, std::unique_ptr<QueueState>> doomed;
    {
      std::lock_guard<std::mutex> lock(queues_mutex);
      doomed.swap(queues);
    }
    for (auto& [queue, queue_state] : doomed) {
      std::lock_guard<std::mutex> lock(queue_state->mutex);
      for (std::unique_ptr<Slot>& slot : queue_state->slots) {
        if (slot->phase == Slot::Phase::kInFlight ||
            slot->phase == Slot::Phase::kLost) {
          WaitSubmitted(slot->fence, "a decode pass");
        }
        DestroySlot(slot.get());
      }
      if (queue_state->rescue != nullptr) {
        if (queue_state->rescue->phase == Slot::Phase::kInFlight ||
            queue_state->rescue->phase == Slot::Phase::kLost) {
          WaitSubmitted(queue_state->rescue->fence, "a rescue pass");
        }
        DestroySlot(queue_state->rescue.get());
      }
    }
    {
      std::lock_guard<std::mutex> lock(selftest_mutex);
      if (selftest.active) {
        WaitSubmitted(selftest.fence, "the selftest");
      }
      DestroySelftest();
    }
    std::map<std::uint32_t, std::unique_ptr<ChunkArena>> doomed_arenas;
    {
      std::lock_guard<std::mutex> lock(arenas_mutex);
      doomed_arenas.swap(arenas);
    }
    for (auto& [family, arena] : doomed_arenas) {
      arena->DestroyAll();
    }
    {
      std::lock_guard<std::mutex> lock(descriptor_mutex);
      for (VkDescriptorPool pool : descriptor_pools) {
        fns().destroy_descriptor_pool(info.device, pool, nullptr);
      }
      descriptor_pools.clear();
      set_pools.clear();
    }
    if (decode_pipeline != VK_NULL_HANDLE) {
      fns().destroy_pipeline(info.device, decode_pipeline, nullptr);
    }
    if (pipeline_layout != VK_NULL_HANDLE) {
      fns().destroy_pipeline_layout(info.device, pipeline_layout, nullptr);
    }
    for (VkDescriptorSetLayout layout : {set_layout0, set_layout1}) {
      if (layout != VK_NULL_HANDLE) {
        fns().destroy_descriptor_set_layout(info.device, layout, nullptr);
      }
    }
    decode_pipeline = VK_NULL_HANDLE;
    pipeline_layout = VK_NULL_HANDLE;
    set_layout0 = VK_NULL_HANDLE;
    set_layout1 = VK_NULL_HANDLE;
  }
};

Etc2GpuDevice::Etc2GpuDevice(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

std::unique_ptr<Etc2GpuDevice> Etc2GpuDevice::Create(Etc2GpuDeviceInfo info) {
  if (info.device == VK_NULL_HANDLE) {
    return nullptr;
  }
  const VkDevice device = info.device;
  std::unique_ptr<Etc2GpuDevice> created(
      new Etc2GpuDevice(std::make_shared<Impl>(std::move(info))));
  if (const char* missing = created->impl_->fns().Missing();
      missing != nullptr) {
    created->impl_->Fail(std::string("the host has no ") + missing);
  }
  std::lock_guard<std::mutex> lock(RegistryMutex());
  Registry()[device] = created.get();
  return created;
}

Etc2GpuDevice::~Etc2GpuDevice() {
  {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    const auto found = Registry().find(impl_->info.device);
    if (found != Registry().end() && found->second == this) {
      Registry().erase(found);
    }
  }
  // [CR-4] No join and no Vulkan here: the thread keeps the Impl alive
  // until it ends, and whatever Destroy() did not free leaks.
  std::lock_guard<std::mutex> lock(impl_->thread_mutex);
  if (impl_->thread_running) {
    pthread_detach(impl_->thread);
    impl_->thread_running = false;
  }
}

namespace {

// The build thread owns a reference to the runtime, so a runtime whose
// device was never destroyed stays alive until the build is over.
void* BuildThread(void* argument) {
  std::shared_ptr<Etc2GpuDevice::Impl> impl =
      std::move(*static_cast<std::shared_ptr<Etc2GpuDevice::Impl>*>(argument));
  delete static_cast<std::shared_ptr<Etc2GpuDevice::Impl>*>(argument);
  pthread_setname_np(pthread_self(), "mt-etc2-build");
  impl->Build();
  return nullptr;
}

}  // namespace

void Etc2GpuDevice::StartBuild() {
  std::lock_guard<std::mutex> lock(impl_->thread_mutex);
  if (impl_->destroyed) {
    return;
  }
  {
    std::lock_guard<std::mutex> state_lock(impl_->state_mutex);
    if (impl_->state.load() != State::kIdle) {
      return;
    }
    impl_->state.store(State::kBuilding);
  }
  impl_->state_changed.notify_all();
  // [CR-20] Signals Roblox's bionic handlers expect never land on this
  // glibc-only thread: it starts with all of them blocked.
  sigset_t all;
  sigset_t previous;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &previous);
  auto* argument = new std::shared_ptr<Impl>(impl_);
  const int error = pthread_create(&impl_->thread, nullptr, BuildThread,
                                   argument);
  pthread_sigmask(SIG_SETMASK, &previous, nullptr);
  if (error != 0) {
    delete argument;
    impl_->Fail("could not start the pipeline build thread");
    return;
  }
  impl_->thread_running = true;
}

Etc2GpuDevice::State Etc2GpuDevice::state() const {
  return impl_->state.load(std::memory_order_acquire);
}

std::string Etc2GpuDevice::failure() const {
  std::lock_guard<std::mutex> lock(impl_->state_mutex);
  return impl_->failure;
}

VkDeviceSize Etc2GpuDevice::max_output_bytes() const {
  return impl_->max_output;
}

bool Etc2GpuDevice::TakeOutput(std::uint32_t family, VkDeviceSize bytes,
                               ArenaRef* out) {
  if (state() != State::kReady || bytes == 0 || bytes > impl_->max_output ||
      !impl_->FamilyComputes(family)) {
    return false;
  }
  return impl_->Arena(family)->Take(bytes, out);
}

void Etc2GpuDevice::PollSelftest() { impl_->PollSelftest(); }

void Etc2GpuDevice::MaybeRunSelftest(VkQueue queue, std::uint32_t family,
                                     Etc2SubmitPlan* plan) {
  impl_->MaybeRunSelftest(queue, family, plan);
}

void Etc2GpuDevice::PlanSubmit(VkQueue queue, std::uint32_t family,
                               std::uint32_t count,
                               const VkSubmitInfo* submits,
                               const std::vector<Etc2GpuBatch>& batches,
                               VkFence app_fence, Etc2SubmitPlan* plan) {
  impl_->PlanSubmitCommon(queue, family, count, submits, batches, app_fence,
                          plan, RewriteSubmits);
}

void Etc2GpuDevice::PlanSubmit2(VkQueue queue, std::uint32_t family,
                                std::uint32_t count,
                                const VkSubmitInfo2* submits,
                                const std::vector<Etc2GpuBatch>& batches,
                                VkFence app_fence, Etc2SubmitPlan* plan) {
  impl_->PlanSubmitCommon(queue, family, count, submits, batches, app_fence,
                          plan, RewriteSubmits2);
}

void Etc2GpuDevice::FinishSubmit(
    Etc2SubmitPlan* plan, VkResult host_result,
    const std::function<VkResult(VkFence)>& submit_fence_only) {
  if (plan == nullptr || plan->gpu_slot == nullptr) {
    return;
  }
  Slot& slot = *static_cast<Slot*>(plan->gpu_slot);
  plan->gpu_slot = nullptr;
  std::lock_guard<std::mutex> lock(slot.queue->mutex);
  VkResult fence_result = VK_SUCCESS;
  if (host_result == VK_SUCCESS) {
    if (plan->trailing_fence != VK_NULL_HANDLE) {
      // Roblox's own fence went with its batches: ours follows them.
      fence_result = submit_fence_only ? submit_fence_only(plan->trailing_fence)
                                       : VK_ERROR_INITIALIZATION_FAILED;
    }
  } else if (slot.pre_submits != 0) {
    // The passes that went first on their own are pending without a fence.
    fence_result = submit_fence_only ? submit_fence_only(slot.fence)
                                     : VK_ERROR_INITIALIZATION_FAILED;
  } else {
    impl_->LogOnce("the host vkQueueSubmit failed; the slot is reused unsubmitted",
                   host_result);
    slot.phase = Slot::Phase::kNotSubmitted;
    if (host_result == VK_ERROR_DEVICE_LOST) {
      impl_->Fail("the device was lost");
    }
    return;
  }
  if (fence_result != VK_SUCCESS) {
    impl_->LogOnce("the fence-only submit after Roblox's failed; the slot is "
                   "given up",
                   fence_result);
    slot.phase = Slot::Phase::kLost;
    return;
  }
  slot.phase = Slot::Phase::kInFlight;
  std::lock_guard<std::mutex> epoch_lock(impl_->epoch_mutex);
  slot.epoch = ++impl_->epoch;
  impl_->in_flight[slot.epoch] = slot.fence;
}

void Etc2GpuDevice::Destroy() {
  impl_->Destroy();
  std::lock_guard<std::mutex> lock(RegistryMutex());
  const auto found = Registry().find(impl_->info.device);
  if (found != Registry().end() && found->second == this) {
    Registry().erase(found);
  }
}

Etc2GpuDevice::Stats Etc2GpuDevice::stats() const {
  Stats copy;
  {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    copy = impl_->stats;
  }
  {
    std::lock_guard<std::mutex> lock(impl_->arenas_mutex);
    for (const auto& [family, arena] : impl_->arenas) {
      const ChunkArena::Stats arena_stats = arena->stats();
      copy.arena_bytes += arena_stats.bytes;
      copy.arena_over_cap_bytes += arena_stats.over_cap_bytes;
    }
  }
  {
    std::lock_guard<std::mutex> lock(impl_->queues_mutex);
    for (const auto& [queue, queue_state] : impl_->queues) {
      std::lock_guard<std::mutex> queue_lock(queue_state->mutex);
      for (const std::unique_ptr<Slot>& slot : queue_state->slots) {
        copy.input_bytes += slot->input.size;
      }
    }
  }
  return copy;
}

Etc2GpuDevice* Etc2GpuDevice::Find(VkDevice device) {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  const auto found = Registry().find(device);
  return found != Registry().end() ? found->second : nullptr;
}

bool Etc2GpuDevice::WaitForBuild(std::chrono::milliseconds timeout) const {
  std::unique_lock<std::mutex> lock(impl_->state_mutex);
  return impl_->state_changed.wait_for(lock, timeout, [this]() {
    const State current = impl_->state.load();
    return current != State::kIdle && current != State::kBuilding;
  });
}

}  // namespace mocktail::graphics
