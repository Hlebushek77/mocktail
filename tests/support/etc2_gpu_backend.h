#ifndef MOCKTAIL_TESTS_SUPPORT_ETC2_GPU_BACKEND_H_
#define MOCKTAIL_TESTS_SUPPORT_ETC2_GPU_BACKEND_H_

#include <map>

#include "support/etc2_emulation_harness.h"
#include "support/vulkan_test_device.h"

namespace mocktail::graphics::testing {

// Runs scenarios on a real device: images get device-local memory, start
// filled with kImageFill in GENERAL layout, every transfer is ordered after
// the previous one with a barrier, the runner waits for the queue after
// each submit, and images are read back with vkCmdCopyImageToBuffer.
class Etc2GpuBackend final : public EmulationBackend {
 public:
  explicit Etc2GpuBackend(HeadlessVulkanDevice* device);
  ~Etc2GpuBackend() override;

  VkDevice device() const override;
  VkPhysicalDevice physical_device() const override;
  VkPhysicalDeviceProperties properties() const override;
  VkPhysicalDeviceMemoryProperties memory_properties() const override;
  std::vector<VkQueueFamilyProperties> queue_families() const override;
  PFN_vkGetDeviceProcAddr get_device_proc_addr() const override;
  PFN_vkGetInstanceProcAddr get_instance_proc_addr() const override;
  VkInstance instance() const override;
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

 private:
  struct HostBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::uint8_t* mapped = nullptr;
  };
  bool CreateHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        HostBuffer* out);
  void DestroyHostBuffer(HostBuffer* buffer);
  // Records into a one-time command buffer of the backend's own pool,
  // submits it and waits; none of this goes through the emulation.
  template <typename Record>
  bool RunOnce(Record record);

  HeadlessVulkanDevice* device_;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;
  std::map<VkImage, VkDeviceMemory> image_memory_;
};

}  // namespace mocktail::graphics::testing

#endif  // MOCKTAIL_TESTS_SUPPORT_ETC2_GPU_BACKEND_H_
