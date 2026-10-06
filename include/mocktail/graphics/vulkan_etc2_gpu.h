#ifndef MOCKTAIL_GRAPHICS_VULKAN_ETC2_GPU_H_
#define MOCKTAIL_GRAPHICS_VULKAN_ETC2_GPU_H_

#include <vulkan/vulkan.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mocktail/graphics/chunk_arena.h"
#include "mocktail/graphics/etc2_decode_totals.h"
#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "mocktail/graphics/vulkan_etc2_planning.h"

namespace mocktail::graphics {

// The host entry points the GPU route calls. Like every call the emulation
// makes on its own behalf, these are the host loader's, never the adapter's
// hooks, so Mocktail's objects are never taken for Roblox's.
struct Etc2GpuFunctions {
  PFN_vkCreateBuffer create_buffer = nullptr;
  PFN_vkDestroyBuffer destroy_buffer = nullptr;
  PFN_vkGetBufferMemoryRequirements get_buffer_memory_requirements = nullptr;
  PFN_vkAllocateMemory allocate_memory = nullptr;
  PFN_vkFreeMemory free_memory = nullptr;
  PFN_vkBindBufferMemory bind_buffer_memory = nullptr;
  PFN_vkMapMemory map_memory = nullptr;
  PFN_vkCreateCommandPool create_command_pool = nullptr;
  PFN_vkDestroyCommandPool destroy_command_pool = nullptr;
  PFN_vkResetCommandPool reset_command_pool = nullptr;
  PFN_vkAllocateCommandBuffers allocate_command_buffers = nullptr;
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
  // VK_EXT_memory_budget, a physical-device query: null when the device
  // lacks the extension, and then no budget check runs (DESIGN CR-5).
  PFN_vkGetPhysicalDeviceMemoryProperties2 get_memory_properties2 = nullptr;

  // The name of the first device-level entry point above that is missing,
  // or null when every one is there.
  const char* Missing() const;
};

// Where the rescue copies a decoded upload from (DESIGN §6): a range of a
// host-visible TRANSFER_SRC buffer that holds the upload's texels in the
// arena's layout (padded rows of 4 x blocks_wide texels).
struct Etc2RescueSource {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
};

// The rescue's CPU half: decodes `uploads` of a batch whose command buffers
// are `command_buffers` into host staging, and keeps that staging alive
// with those command buffers' records (they are released once Roblox's
// copies, which run after the rescue copy, completed). False: nothing to
// copy, the uploads are dropped.
using Etc2RescueDecode =
    std::function<bool(const std::vector<VkCommandBuffer>& command_buffers,
                       const std::vector<Etc2GpuUpload>& uploads,
                       std::vector<Etc2RescueSource>* sources)>;

struct Etc2GpuDeviceInfo {
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  Etc2GpuFunctions fns;
  VkPhysicalDeviceMemoryProperties memory{};
  VkPhysicalDeviceProperties properties{};  // limits, timestampPeriod
  std::vector<VkQueueFamilyProperties> families;
  bool bc = false;
  // Device-local bytes the output arenas of all families may hold
  // together (MOCKTAIL_ETC2_GPU_ARENA_MB); 0: min(256 MiB, VRAM / 24).
  std::uint64_t arena_cap_bytes = 0;
  bool frame_stats = false;  // timestamps and the "GPU submit" line
  // Adds to the emulation's session totals (and its two-second window).
  std::function<void(const std::function<void(Etc2DecodeTotals&)>&)> count;
  Etc2RescueDecode rescue_decode;
};

// One batch of a submit call that carries GPU-route uploads.
struct Etc2GpuBatch {
  std::uint32_t batch = 0;                       // index in pSubmits
  std::vector<VkCommandBuffer> command_buffers;  // Roblox's primaries
  std::vector<Etc2GpuUpload> uploads;  // with their output ranges set
};

// Development and test switches; production never sets them.
struct Etc2GpuTestHooks {
  std::atomic<bool> fail_slot_acquire{false};  // as if no slot came free
  std::atomic<bool> fail_rescue_slot{false};   // nor the rescue slot
  std::atomic<bool> corrupt_selftest{false};   // the selftest must mismatch
  std::atomic<int> build_delay_ms{0};          // the build thread sleeps first
  std::mutex mutex;
  Etc2RescueDecode rescue_decode;  // replaces the emulation's, when set
};
Etc2GpuTestHooks& Etc2GpuTesting();

// The GPU decode runtime of one emulated device (DESIGN C3): the decode
// pipeline, output arenas per queue family, and per-queue submit slots that
// record the decode pass put in front of Roblox's batches. The emulation
// makes one per emulated device; its pipelines are built on a background
// thread from the first emulated image on, and a selftest on the GPU must
// match the CPU decoder before any upload takes the GPU route.
class Etc2GpuDevice {
 public:
  enum class State : int {
    kIdle,      // made, nothing built
    kBuilding,  // pipelines building on the background thread
    kBuilt,     // waiting for the first submit on a compute queue
    kTesting,   // the selftest is on the GPU
    kReady,     // uploads take the GPU route
    kFailed,    // the CPU route serves this device for the session
  };

  // What tests and the logs read about the runtime.
  struct Stats {
    std::uint64_t submits = 0;       // calls that got a pass in front
    std::uint64_t passes = 0;        // decode or rescue command buffers
    std::uint64_t pre_submits = 0;   // passes submitted on their own
    std::uint64_t slots_created = 0;
    std::uint64_t slot_waits = 0;     // acquisitions that waited on a fence
    std::uint64_t slot_timeouts = 0;  // waits that gave up (50 ms)
    std::uint64_t rescues = 0;        // batches the rescue served
    std::uint64_t rescued_uploads = 0;
    std::uint64_t dropped_uploads = 0;
    std::uint64_t fences_created = 0;
    std::uint64_t fences_destroyed = 0;
    std::uint64_t teardown_waits = 0;  // fences Destroy waited for
    std::uint32_t live_slots = 0;
    VkDeviceSize arena_bytes = 0;
    VkDeviceSize arena_over_cap_bytes = 0;
    VkDeviceSize input_bytes = 0;  // every slot's input buffer
  };

  // Resolves nothing and starts no thread.
  static std::unique_ptr<Etc2GpuDevice> Create(Etc2GpuDeviceInfo info);
  // Never calls Vulkan and never joins (DESIGN CR-4): when the process
  // exits without vkDestroyDevice the driver may be gone already, so a
  // build still running is detached and the Vulkan objects leak.
  virtual ~Etc2GpuDevice();
  Etc2GpuDevice(const Etc2GpuDevice&) = delete;
  Etc2GpuDevice& operator=(const Etc2GpuDevice&) = delete;

  // The first emulated vkCreateImage: builds the pipelines on a thread with
  // every signal blocked. Later calls do nothing.
  void StartBuild();
  State state() const;
  // Why the state is kFailed, for the log.
  std::string failure() const;
  // The largest output one upload call may take on the GPU route:
  // min(maxStorageBufferRange, 1 GiB).
  VkDeviceSize max_output_bytes() const;

  // Record time: an output range in the arena of `family`.
  bool TakeOutput(std::uint32_t family, VkDeviceSize bytes, ArenaRef* out);

  // Submit time, inside Roblox's vkQueueSubmit on `queue` (of `family`):
  // finishes a selftest whose fence signalled, and at the first submit on a
  // compute queue after the build adds the selftest to plan->pre_submits.
  void PollSelftest();
  void MaybeRunSelftest(VkQueue queue, std::uint32_t family,
                        Etc2SubmitPlan* plan);
  // Records the decode pass (or the rescue copy) of every batch in
  // `batches` and rewrites the call into `plan`: one slot and one fence for
  // the whole call (DESIGN CR-1).
  void PlanSubmit(VkQueue queue, std::uint32_t family, std::uint32_t count,
                  const VkSubmitInfo* submits,
                  const std::vector<Etc2GpuBatch>& batches, VkFence app_fence,
                  Etc2SubmitPlan* plan);
  void PlanSubmit2(VkQueue queue, std::uint32_t family, std::uint32_t count,
                   const VkSubmitInfo2* submits,
                   const std::vector<Etc2GpuBatch>& batches, VkFence app_fence,
                   Etc2SubmitPlan* plan);
  // After the host submit: signals the slot fence after Roblox's batches
  // when Roblox passed a fence of its own, and marks the slot in flight, or
  // not submitted when the host submit failed.
  void FinishSubmit(Etc2SubmitPlan* plan, VkResult host_result,
                    const std::function<VkResult(VkFence)>& submit_fence_only);

  // vkDestroyDevice: joins the build, waits for Mocktail's own fences that
  // were submitted (1 s at most each) and destroys everything it made.
  virtual void Destroy();

  Stats stats() const;
  // Tests: the runtime of `device`, or null.
  static Etc2GpuDevice* Find(VkDevice device);
  // Tests: waits until the build is over (built, or failed).
  bool WaitForBuild(std::chrono::milliseconds timeout) const;

  struct Impl;

 private:
  explicit Etc2GpuDevice(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

const char* Etc2GpuStateName(Etc2GpuDevice::State state);

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_VULKAN_ETC2_GPU_H_
