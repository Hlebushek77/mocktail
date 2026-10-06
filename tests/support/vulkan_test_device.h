#ifndef MOCKTAIL_TESTS_SUPPORT_VULKAN_TEST_DEVICE_H_
#define MOCKTAIL_TESTS_SUPPORT_VULKAN_TEST_DEVICE_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Headless real-GPU tests (ctest label "gpu"). The system loader is opened
// with dlopen, so nothing links libvulkan and the tests skip where there is
// no loader, no driver (CI, VK_ICD_FILENAMES=/nonexistent) or no device.
// They never create a surface or a window.
namespace mocktail::graphics::testing {

// [CR-25] A GPU test running while the main session measures a live game
// skews its fps, GPU time and VRAM. Like plus2-design/guard.sh (pgrep -x
// mocktail, pgrep -x sober), this finds such a process by its exact name,
// leaving out the ones another test of the same ctest run started.
bool LiveGameRunning(std::string* description);

#define MOCKTAIL_VULKAN_TEST_INSTANCE_FUNCTIONS(X) \
  X(vkDestroyInstance)                             \
  X(vkEnumeratePhysicalDevices)                    \
  X(vkGetPhysicalDeviceProperties)                 \
  X(vkGetPhysicalDeviceMemoryProperties)           \
  X(vkGetPhysicalDeviceQueueFamilyProperties)      \
  X(vkGetPhysicalDeviceFormatProperties)           \
  X(vkGetPhysicalDeviceFeatures)                   \
  X(vkEnumerateDeviceExtensionProperties)          \
  X(vkCreateDevice)                                \
  X(vkGetDeviceProcAddr)

#define MOCKTAIL_VULKAN_TEST_DEVICE_FUNCTIONS(X) \
  X(vkDestroyDevice)                             \
  X(vkGetDeviceQueue)                            \
  X(vkDeviceWaitIdle)                            \
  X(vkQueueSubmit)                               \
  X(vkQueueWaitIdle)                             \
  X(vkCreateCommandPool)                         \
  X(vkDestroyCommandPool)                        \
  X(vkResetCommandPool)                          \
  X(vkAllocateCommandBuffers)                    \
  X(vkFreeCommandBuffers)                        \
  X(vkBeginCommandBuffer)                        \
  X(vkEndCommandBuffer)                          \
  X(vkResetCommandBuffer)                        \
  X(vkCreateFence)                               \
  X(vkDestroyFence)                              \
  X(vkResetFences)                               \
  X(vkWaitForFences)                             \
  X(vkCreateBuffer)                              \
  X(vkDestroyBuffer)                             \
  X(vkGetBufferMemoryRequirements)               \
  X(vkAllocateMemory)                            \
  X(vkFreeMemory)                                \
  X(vkBindBufferMemory)                          \
  X(vkMapMemory)                                 \
  X(vkUnmapMemory)                               \
  X(vkGetImageMemoryRequirements)                \
  X(vkBindImageMemory)                           \
  X(vkDestroyImage)                              \
  X(vkCmdPipelineBarrier)                        \
  X(vkCmdCopyBufferToImage)                      \
  X(vkCmdCopyImageToBuffer)                      \
  X(vkCmdCopyBuffer)                             \
  X(vkGetFenceStatus)                            \
  X(vkCreateSemaphore)                           \
  X(vkDestroySemaphore)                          \
  X(vkCreateShaderModule)                        \
  X(vkDestroyShaderModule)                       \
  X(vkCreateDescriptorSetLayout)                 \
  X(vkDestroyDescriptorSetLayout)                \
  X(vkCreatePipelineLayout)                      \
  X(vkDestroyPipelineLayout)                     \
  X(vkCreateComputePipelines)                    \
  X(vkDestroyPipeline)                           \
  X(vkCreateDescriptorPool)                      \
  X(vkDestroyDescriptorPool)                     \
  X(vkAllocateDescriptorSets)                    \
  X(vkUpdateDescriptorSets)                      \
  X(vkCmdBindPipeline)                           \
  X(vkCmdBindDescriptorSets)                     \
  X(vkCmdPushConstants)                          \
  X(vkCmdDispatch)                               \
  X(vkCreateQueryPool)                           \
  X(vkDestroyQueryPool)                          \
  X(vkCmdResetQueryPool)                         \
  X(vkCmdWriteTimestamp)                         \
  X(vkGetQueryPoolResults)

// Core 1.1 or 1.3: null on older loaders or devices.
#define MOCKTAIL_VULKAN_TEST_OPTIONAL_INSTANCE_FUNCTIONS(X) \
  X(vkGetPhysicalDeviceFeatures2)
#define MOCKTAIL_VULKAN_TEST_OPTIONAL_DEVICE_FUNCTIONS(X) X(vkQueueSubmit2)

class HeadlessVulkanDevice {
 public:
  struct Options {
    // Queues of the main family (GRAPHICS first, where Roblox records).
    std::uint32_t queue_count = 1;
    // One more queue in a family with COMPUTE but no GRAPHICS (family 2 on
    // NVIDIA), and one in a TRANSFER-only family, when the device has them.
    bool compute_only_family = false;
    bool transfer_only_family = false;
  };

  struct Dispatch {
#define MOCKTAIL_VULKAN_TEST_MEMBER(name) PFN_##name name = nullptr;
    MOCKTAIL_VULKAN_TEST_INSTANCE_FUNCTIONS(MOCKTAIL_VULKAN_TEST_MEMBER)
    MOCKTAIL_VULKAN_TEST_DEVICE_FUNCTIONS(MOCKTAIL_VULKAN_TEST_MEMBER)
    MOCKTAIL_VULKAN_TEST_OPTIONAL_INSTANCE_FUNCTIONS(
        MOCKTAIL_VULKAN_TEST_MEMBER)
    MOCKTAIL_VULKAN_TEST_OPTIONAL_DEVICE_FUNCTIONS(MOCKTAIL_VULKAN_TEST_MEMBER)
#undef MOCKTAIL_VULKAN_TEST_MEMBER
  };

  // A device on the best physical device (discrete, then integrated, then
  // any; MOCKTAIL_TEST_VULKAN_DEVICE=<index or name part> picks one) with a
  // single low-priority queue from the first family with GRAPHICS (where
  // Roblox records its uploads), else COMPUTE, else TRANSFER. Returns null
  // and says why when the test must skip, including while a live game runs.
  //
  // MOCKTAIL_TEST_VALIDATION=sync (or =core) makes the instance enable
  // VK_LAYER_KHRONOS_validation itself, with synchronization validation for
  // "sync", and collect every validation warning and error through a
  // VK_EXT_debug_utils messenger (TakeValidationMessages). The tests then
  // skip when the layer is not installed.
  static std::unique_ptr<HeadlessVulkanDevice> Create(std::string* skip_reason);
  static std::unique_ptr<HeadlessVulkanDevice> Create(std::string* skip_reason,
                                                      const Options& options);
  ~HeadlessVulkanDevice();
  HeadlessVulkanDevice(const HeadlessVulkanDevice&) = delete;
  HeadlessVulkanDevice& operator=(const HeadlessVulkanDevice&) = delete;

  const Dispatch& vk() const { return vk_; }
  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  std::uint32_t queue_family() const { return queue_family_; }
  // Queue `index` of `family`, or null when the device made none there.
  VkQueue Queue(std::uint32_t family, std::uint32_t index = 0) const;
  // The extra families of Options, UINT32_MAX when there is none.
  std::uint32_t compute_only_family() const { return compute_only_family_; }
  std::uint32_t transfer_only_family() const { return transfer_only_family_; }
  // Validation through MOCKTAIL_TEST_VALIDATION: on, with sync validation.
  bool validating() const { return messenger_ != VK_NULL_HANDLE; }
  bool sync_validation() const { return sync_validation_; }
  // The validation messages since the last call ("ERROR [VUID...] ...").
  std::vector<std::string> TakeValidationMessages();
  const VkPhysicalDeviceProperties& properties() const { return properties_; }
  const VkPhysicalDeviceMemoryProperties& memory() const { return memory_; }
  const std::vector<VkQueueFamilyProperties>& queue_families() const {
    return queue_families_;
  }
  // vkCmdCopyBufferToImage2 and vkCmdCopyImage2 resolve (core 1.3 or
  // VK_KHR_copy_commands2).
  bool copy_commands2() const { return copy_commands2_; }
  // vkQueueSubmit2 can be called: a Vulkan 1.3 device with the
  // synchronization2 feature, which the device enables then.
  bool synchronization2() const { return vk_.vkQueueSubmit2 != nullptr; }
  PFN_vkGetDeviceProcAddr get_device_proc_addr() const {
    return vk_.vkGetDeviceProcAddr;
  }
  PFN_vkGetInstanceProcAddr get_instance_proc_addr() const {
    return get_instance_proc_addr_;
  }
  // "NVIDIA GeForce RTX 3060 Laptop GPU (driver 615.71.9), family 0"
  std::string Describe() const;

  // The first type in `type_bits` with `required`, preferring one that also
  // has `preferred` and none of `avoided`; UINT32_MAX when none.
  std::uint32_t FindMemoryType(std::uint32_t type_bits,
                               VkMemoryPropertyFlags required,
                               VkMemoryPropertyFlags preferred = 0,
                               VkMemoryPropertyFlags avoided = 0) const;
  bool OptimalTilingSupports(VkFormat format,
                             VkFormatFeatureFlags features) const;

 private:
  HeadlessVulkanDevice() = default;

  Dispatch vk_;
  PFN_vkGetInstanceProcAddr get_instance_proc_addr_ = nullptr;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  bool sync_validation_ = false;
  std::uint32_t compute_only_family_ = UINT32_MAX;
  std::uint32_t transfer_only_family_ = UINT32_MAX;
  // (family, index) -> queue
  std::vector<std::pair<std::pair<std::uint32_t, std::uint32_t>, VkQueue>>
      queues_;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  std::uint32_t queue_family_ = 0;
  VkPhysicalDeviceProperties properties_{};
  VkPhysicalDeviceMemoryProperties memory_{};
  std::vector<VkQueueFamilyProperties> queue_families_;
  bool copy_commands2_ = false;
};

}  // namespace mocktail::graphics::testing

#endif  // MOCKTAIL_TESTS_SUPPORT_VULKAN_TEST_DEVICE_H_
