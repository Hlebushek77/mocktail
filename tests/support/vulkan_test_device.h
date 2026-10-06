#ifndef MOCKTAIL_TESTS_SUPPORT_VULKAN_TEST_DEVICE_H_
#define MOCKTAIL_TESTS_SUPPORT_VULKAN_TEST_DEVICE_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
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
  X(vkCmdCopyImageToBuffer)

// Core 1.1 or 1.3: null on older loaders or devices.
#define MOCKTAIL_VULKAN_TEST_OPTIONAL_INSTANCE_FUNCTIONS(X) \
  X(vkGetPhysicalDeviceFeatures2)
#define MOCKTAIL_VULKAN_TEST_OPTIONAL_DEVICE_FUNCTIONS(X) X(vkQueueSubmit2)

class HeadlessVulkanDevice {
 public:
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
  static std::unique_ptr<HeadlessVulkanDevice> Create(std::string* skip_reason);
  ~HeadlessVulkanDevice();
  HeadlessVulkanDevice(const HeadlessVulkanDevice&) = delete;
  HeadlessVulkanDevice& operator=(const HeadlessVulkanDevice&) = delete;

  const Dispatch& vk() const { return vk_; }
  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  std::uint32_t queue_family() const { return queue_family_; }
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
