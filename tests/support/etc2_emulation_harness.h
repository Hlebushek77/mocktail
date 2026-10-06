#ifndef MOCKTAIL_TESTS_SUPPORT_ETC2_EMULATION_HARNESS_H_
#define MOCKTAIL_TESTS_SUPPORT_ETC2_EMULATION_HARNESS_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "mocktail/graphics/vulkan_etc2_gpu.h"
#include "support/etc2_emulation_scenarios.h"
#include "support/fake_vulkan_device.h"

// Runs an Etc2EmulationScenarios() entry through VulkanEtc2Emulation the way
// the bionic Vulkan adapter would, on a device a backend provides (the fake
// device here, a real GPU in etc2_emulation_gpu_test), and checks every
// image against Etc2EmulationModel.
//
// The golden hashes were frozen on 2249b6c with this runner written against
// the step-0 API (VulkanEtc2Emulation(), RegisterDevice(...,
// get_device_proc_addr), PrepareSubmit(command_buffers, count)). It now
// follows the adapter's plus.2 hooks: options instead of the environment,
// RegisterDevice with the device's properties, queue families and create
// info, RegisterQueue, RegisterCommandPool and ForgetCommandPool,
// BeginCommandBuffer, and PrepareSubmit(2) with an Etc2SubmitPlan whose
// arrays it submits. Every golden hash stayed the same.
namespace mocktail::graphics::testing {

// What differs between the fake device and a real one. Calls the adapter
// forwards to the host unchanged go here; everything the adapter routes
// through the emulation, the runner calls on the emulation itself.
class EmulationBackend {
 public:
  struct Source {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize bind_offset = 0;
    // What vkAllocateMemory was given, for NoteAllocateMemory.
    std::uint32_t memory_type = 0;
    VkDeviceSize memory_size = 0;
  };

  virtual ~EmulationBackend() = default;

  virtual VkDevice device() const = 0;
  virtual VkPhysicalDevice physical_device() const = 0;
  virtual VkPhysicalDeviceProperties properties() const = 0;
  virtual VkPhysicalDeviceMemoryProperties memory_properties() const = 0;
  virtual std::vector<VkQueueFamilyProperties> queue_families() const = 0;
  virtual PFN_vkGetDeviceProcAddr get_device_proc_addr() const = 0;
  // For physical-device queries; the fake has no instance.
  virtual PFN_vkGetInstanceProcAddr get_instance_proc_addr() const {
    return nullptr;
  }
  virtual VkInstance instance() const { return VK_NULL_HANDLE; }
  // Why `scenario` cannot run on this backend, or empty.
  virtual std::string Unsupported(const Scenario& scenario) const = 0;
  // Queue 0 of `family`, as vkGetDeviceQueue returns it.
  virtual VkQueue Queue(std::uint32_t family) = 0;

  // An unbound TRANSFER_SRC buffer and host-visible memory big enough to
  // bind it at `bind_offset` (rounded up to the buffer's alignment). The
  // memory is allocated on the host directly, so the fake's entry-point
  // counts keep meaning "what the emulation made itself".
  virtual bool CreateSource(VkDeviceSize size, VkDeviceSize bind_offset,
                            Source* out) = 0;
  // Writes the bound range while nothing maps the memory.
  virtual bool WriteUnmapped(const Source& source,
                             const std::vector<std::uint8_t>& bytes) = 0;
  // Memory, every subresource filled with kImageFill, GENERAL layout.
  virtual bool PrepareImage(VkImage image, const HostImageShape& shape) = 0;
  // After the image is destroyed: whatever PrepareImage allocated.
  virtual void ReleaseImage(VkImage image) = 0;

  virtual VkCommandPool CreateCommandPool(std::uint32_t queue_family,
                                          VkCommandPoolCreateFlags flags) = 0;
  virtual bool ResetCommandPool(VkCommandPool pool) = 0;
  // Frees the pool's command buffers too.
  virtual void DestroyCommandPool(VkCommandPool pool) = 0;
  virtual VkCommandBuffer AllocateCommandBuffer(VkCommandPool pool,
                                                bool secondary) = 0;
  virtual void FreeCommandBuffer(VkCommandPool pool,
                                 VkCommandBuffer command_buffer) = 0;
  virtual bool Begin(VkCommandBuffer command_buffer, bool secondary,
                     VkCommandBufferUsageFlags flags) = 0;
  virtual bool End(VkCommandBuffer command_buffer) = 0;
  virtual bool Reset(VkCommandBuffer command_buffer) = 0;
  // Orders the next transfer after earlier ones (a barrier on a GPU).
  virtual void BeforeTransfer(VkCommandBuffer command_buffer) = 0;
  // The host's vkQueueSubmit and vkQueueSubmit2, which the adapter calls
  // with what the emulation's submit plan says.
  virtual VkResult QueueSubmit(VkQueue queue, std::uint32_t count,
                               const VkSubmitInfo* submits, VkFence fence) = 0;
  virtual VkResult QueueSubmit2(VkQueue queue, std::uint32_t count,
                                const VkSubmitInfo2* submits,
                                VkFence fence) = 0;
  // Returns once everything submitted to `queue` has finished.
  virtual bool WaitIdle(VkQueue queue) = 0;
  virtual bool ReadImage(VkImage image, const HostImageShape& shape,
                         std::vector<std::vector<std::uint8_t>>* texels) = 0;

  // Host calls for objects the emulation forgot (after DestroyDevice).
  virtual void DestroyImageDirect(VkImage image) = 0;
  virtual void DestroyBufferDirect(VkBuffer buffer) = 0;
  virtual void UnmapMemoryDirect(VkDeviceMemory memory) = 0;
  virtual void FreeMemoryDirect(VkDeviceMemory memory) = 0;
};

class FakeEmulationBackend final : public EmulationBackend {
 public:
  explicit FakeEmulationBackend(FakeVulkanDevice* fake);
  ~FakeEmulationBackend() override;

  VkDevice device() const override;
  VkPhysicalDevice physical_device() const override;
  VkPhysicalDeviceProperties properties() const override;
  VkPhysicalDeviceMemoryProperties memory_properties() const override;
  std::vector<VkQueueFamilyProperties> queue_families() const override;
  PFN_vkGetDeviceProcAddr get_device_proc_addr() const override;
  std::string Unsupported(const Scenario& scenario) const override;
  VkQueue Queue(std::uint32_t family) override;
  bool CreateSource(VkDeviceSize size, VkDeviceSize bind_offset,
                    Source* out) override;
  bool WriteUnmapped(const Source& source,
                     const std::vector<std::uint8_t>& bytes) override;
  bool PrepareImage(VkImage image, const HostImageShape& shape) override;
  void ReleaseImage(VkImage image) override;
  VkCommandPool CreateCommandPool(std::uint32_t queue_family,
                                  VkCommandPoolCreateFlags flags) override;
  bool ResetCommandPool(VkCommandPool pool) override;
  void DestroyCommandPool(VkCommandPool pool) override;
  VkCommandBuffer AllocateCommandBuffer(VkCommandPool pool,
                                        bool secondary) override;
  void FreeCommandBuffer(VkCommandPool pool,
                         VkCommandBuffer command_buffer) override;
  bool Begin(VkCommandBuffer command_buffer, bool secondary,
             VkCommandBufferUsageFlags flags) override;
  bool End(VkCommandBuffer command_buffer) override;
  bool Reset(VkCommandBuffer command_buffer) override;
  void BeforeTransfer(VkCommandBuffer command_buffer) override;
  VkResult QueueSubmit(VkQueue queue, std::uint32_t count,
                       const VkSubmitInfo* submits, VkFence fence) override;
  VkResult QueueSubmit2(VkQueue queue, std::uint32_t count,
                        const VkSubmitInfo2* submits, VkFence fence) override;
  bool WaitIdle(VkQueue queue) override;
  bool ReadImage(VkImage image, const HostImageShape& shape,
                 std::vector<std::vector<std::uint8_t>>* texels) override;
  void DestroyImageDirect(VkImage image) override;
  void DestroyBufferDirect(VkBuffer buffer) override;
  void UnmapMemoryDirect(VkDeviceMemory memory) override;
  void FreeMemoryDirect(VkDeviceMemory memory) override;

  FakeVulkanDevice* fake() const { return fake_; }

 private:
  FakeVulkanDevice* fake_;
};

enum class ScenarioTeardown {
  // Roblox frees its command buffers (the adapter releases each one) and
  // destroys the pool, then its images, buffers and memory, then the device.
  kOrderly,
  // vkResetCommandPool first (the adapter releases every command buffer of
  // the pool), then as kOrderly.
  kResetPool,
  // vkDestroyCommandPool without freeing the command buffers: the adapter
  // releases every one of them, then the emulation forgets the pool.
  kDestroyPool,
  // The device goes first while command buffers still hold uploads; the
  // emulation must free their staging then.
  kDestroyDeviceFirst,
};

// The options the runner gives the emulation unless told otherwise: the
// CPU route (gpu = false), no diagnostics. `upscale` always comes from the
// scenario.
Etc2EmulationOptions CpuRouteOptions();

struct ScenarioOptions {
  ScenarioTeardown teardown = ScenarioTeardown::kOrderly;
  Etc2EmulationOptions emulation = CpuRouteOptions();
  // Compare the image copies and blits the fake ran with the model's.
  FakeVulkanDevice* fake = nullptr;
  // Override and dump directories go below it; empty: $TMPDIR or /tmp.
  std::string scratch_directory;
  // After the last step and checkpoint, before any teardown.
  std::function<void(VulkanEtc2Emulation*)> after_steps;
  // Once the command buffers and their pool are gone, before the images,
  // sources and device (not with kDestroyDeviceFirst).
  std::function<void()> after_command_buffers;
  // With emulation.gpu: wait for the GPU route's pipelines, then run its
  // selftest with empty submits, so that every upload can take the route.
  bool wait_for_gpu_route = true;
};

struct ScenarioRun {
  // What the runner asked of the emulation, to compare with the counters
  // of MOCKTAIL_FRAME_STATS.
  struct Calls {
    // CmdCopyBufferToImage(2) into an ETC2/EAC image, and their regions.
    std::uint64_t etc2_copies = 0;
    std::uint64_t etc2_regions = 0;
    // CmdCopyImage(2) with an ETC2/EAC image on either side.
    std::uint64_t etc2_image_copies = 0;
    std::uint64_t begins = 0;                // BeginCommandBuffer
    std::uint64_t secondaries_executed = 0;  // CmdExecuteCommands
    std::uint64_t submit_calls = 0;          // PrepareSubmit
    std::uint64_t submit_batches = 0;
    std::uint64_t submit2_calls = 0;  // PrepareSubmit2
    std::uint64_t submit2_batches = 0;
  };

  bool skipped = false;
  std::string skip_reason;
  std::uint64_t hash = 0;
  std::vector<std::string> failures;  // empty when every check passed
  Calls calls;
  // The device's GPU route after the steps, before any teardown: whether it
  // had one, its state ("ready", "failed", ...) and its counters.
  bool gpu_route = false;
  std::string gpu_state;
  Etc2GpuDevice::Stats gpu_stats;
};

ScenarioRun RunEtc2Scenario(const Scenario& scenario, EmulationBackend* backend,
                            const ScenarioOptions& options = {});

// "{\"Name\", 0x0123456789abcdefULL},": the golden line for a scenario.
std::string GoldenLine(const std::string& name, std::uint64_t hash);

}  // namespace mocktail::graphics::testing

#endif  // MOCKTAIL_TESTS_SUPPORT_ETC2_EMULATION_HARNESS_H_
