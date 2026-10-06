#ifndef MOCKTAIL_TESTS_SUPPORT_FAKE_VULKAN_DEVICE_H_
#define MOCKTAIL_TESTS_SUPPORT_FAKE_VULKAN_DEVICE_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mocktail::graphics::testing {

// How copies address a format: texel block extent and bytes per block.
struct FormatBlock {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t bytes = 0;
};
bool LookupFormatBlock(VkFormat format, FormatBlock* out);

// Memory types and queue families of the RTX 3060 Laptop that plus.2 was
// measured on (DESIGN §2.1): type 3 is HOST_VISIBLE|HOST_COHERENT, type 4
// adds HOST_CACHED, type 5 is the BAR; buffers accept types 0, 1, 3, 4, 5.
VkPhysicalDeviceMemoryProperties Rtx3060MemoryProperties();
std::vector<VkQueueFamilyProperties> Rtx3060QueueFamilies();
// Its identity and the limits the ETC2 GPU route reads (DESIGN C0.2).
VkPhysicalDeviceProperties Rtx3060Properties();

// A VkDevice without a GPU, for the ETC2 emulation tests. Its
// PFN_vkGetDeviceProcAddr hands out entry points that keep buffers and memory
// in malloc'd storage, record transfer commands into command buffers, and run
// them against a CPU model of every image when the test submits: copies and
// same-size NEAREST blits move bytes, filtered blits are only recorded, and a
// compute dispatch runs the ETC2 decode shader's job list on the CPU (the
// only pipeline Mocktail builds; DecodeEtcImage per 4x4 block, written the
// way etc2_decode.comp writes it), so the GPU route runs here too. Misuse
// that a validation layer would flag is collected in errors() instead of
// crashing, so a test can name the rule that broke.
//
// The test itself plays Roblox and the adapter: it calls the direct methods
// below for objects the adapter passes straight to the host (buffers, memory,
// command buffers), and the emulation calls the entry points.
class FakeVulkanDevice {
 public:
  struct Config {
    Config();
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory;
    std::vector<VkQueueFamilyProperties> queue_families;
    std::uint32_t buffer_memory_type_bits = 0x3b;
    VkDeviceSize buffer_alignment = 256;
    std::uint8_t memory_fill = 0xA5;  // fresh allocations
    std::uint8_t image_fill = 0xCD;   // fresh images, every subresource
    // Names vkGetDeviceProcAddr returns null for, as on an older driver.
    std::vector<std::string> missing_procs;
    // The ETC2 GPU route (plus.2 WP2). Every timestamp a command buffer
    // writes is this much later than the one before (timestampPeriod 1).
    std::uint64_t timestamp_step_ns = 1000;
    // vkAllocateMemory fails above this size (0: never).
    VkDeviceSize max_allocation_bytes = 0;
    // A descriptor pool holds at most this many sets, whatever its maxSets
    // (0: maxSets).
    std::uint32_t max_sets_per_pool = 0;
  };

  struct ObjectCount {
    std::uint64_t created = 0;
    std::uint64_t destroyed = 0;
    std::uint64_t live() const { return created - destroyed; }
  };
  struct Counts {
    ObjectCount buffers;
    ObjectCount memory;
    ObjectCount images;
    ObjectCount image_views;
    ObjectCount command_pools;
    ObjectCount command_buffers;
    ObjectCount fences;
    // Compute objects (the ETC2 GPU route's).
    ObjectCount shader_modules;
    ObjectCount pipelines;
    ObjectCount pipeline_layouts;
    ObjectCount descriptor_set_layouts;
    ObjectCount descriptor_pools;
    ObjectCount descriptor_sets;  // destroyed: freed, or with their pool
    ObjectCount query_pools;
    std::uint64_t dispatches = 0;
    std::uint64_t decoded_blocks = 0;
    // Of the buffers and memory, those created through the entry points
    // (that is, by the code under test rather than by the test).
    ObjectCount entry_point_buffers;
    ObjectCount entry_point_memory;
    std::uint64_t maps = 0;
    std::uint64_t unmaps = 0;
    std::uint64_t live_mappings = 0;
    std::uint64_t submits = 0;
  };

  // A vkCmdCopyImage(2) or vkCmdBlitImage that ran, in execution order.
  struct ImageTransfer {
    bool blit = false;
    bool executed = false;  // false: a filtered blit the fake does not model
    VkImage source = VK_NULL_HANDLE;
    VkImage destination = VK_NULL_HANDLE;
    VkImageLayout source_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout destination_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::vector<VkImageCopy> copies;
    std::vector<VkImageBlit> blits;
    VkFilter filter = VK_FILTER_NEAREST;
  };

  explicit FakeVulkanDevice(Config config = Config());
  ~FakeVulkanDevice();
  FakeVulkanDevice(const FakeVulkanDevice&) = delete;
  FakeVulkanDevice& operator=(const FakeVulkanDevice&) = delete;

  VkDevice device() const;
  VkPhysicalDevice physical_device() const;
  const Config& config() const;
  static PFN_vkGetDeviceProcAddr GetDeviceProcAddr();

  // Direct host calls.
  VkResult CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                        VkBuffer* buffer);
  void GetBufferMemoryRequirements(VkBuffer buffer,
                                   VkMemoryRequirements* requirements) const;
  VkResult AllocateMemory(VkDeviceSize size, std::uint32_t type_index,
                          VkDeviceMemory* memory);
  VkResult BindBufferMemory(VkBuffer buffer, VkDeviceMemory memory,
                            VkDeviceSize offset);
  VkResult MapMemory(VkDeviceMemory memory, VkDeviceSize offset,
                     VkDeviceSize size, void** data);
  void UnmapMemory(VkDeviceMemory memory);
  void DestroyBuffer(VkBuffer buffer);
  void FreeMemory(VkDeviceMemory memory);
  VkResult CreateImage(const VkImageCreateInfo& info, VkImage* image);
  void DestroyImage(VkImage image);
  void DestroyImageView(VkImageView view);
  VkCommandPool CreateCommandPool(std::uint32_t queue_family,
                                  VkCommandPoolCreateFlags flags);
  void DestroyCommandPool(VkCommandPool pool);  // frees its command buffers
  VkResult ResetCommandPool(VkCommandPool pool);
  VkCommandBuffer AllocateCommandBuffer(VkCommandPool pool,
                                        VkCommandBufferLevel level);
  void FreeCommandBuffer(VkCommandBuffer command_buffer);
  VkResult BeginCommandBuffer(VkCommandBuffer command_buffer,
                              VkCommandBufferUsageFlags flags = 0);
  VkResult EndCommandBuffer(VkCommandBuffer command_buffer);
  VkResult ResetCommandBuffer(VkCommandBuffer command_buffer);
  VkQueue GetQueue(std::uint32_t family, std::uint32_t index);
  // Executes every batch in order, as a queue that finishes at once.
  VkResult Submit(const std::vector<std::vector<VkCommandBuffer>>& batches);
  // The same through the submit structures, signalling `fence` (which may
  // be the only thing submitted). Semaphores are ignored.
  //
  // The next vkQueueSubmit(2) calls return these results in order, and
  // execute and signal nothing unless the result is VK_SUCCESS.
  void FailNextSubmits(std::vector<VkResult> results);
  VkResult QueueSubmit(VkQueue queue, std::uint32_t count,
                       const VkSubmitInfo* submits, VkFence fence);
  VkResult QueueSubmit2(VkQueue queue, std::uint32_t count,
                        const VkSubmitInfo2* submits, VkFence fence);

  // Inspection.
  bool GetImageInfo(VkImage image, VkImageCreateInfo* info) const;
  // Tight texel blocks of one subresource, or null.
  const std::vector<std::uint8_t>* ImageSubresource(VkImage image,
                                                    std::uint32_t mip,
                                                    std::uint32_t layer) const;
  bool GetImageViewInfo(VkImageView view, VkImageViewCreateInfo* info) const;
  std::uint8_t* MemoryBytes(VkDeviceMemory memory);
  std::uint32_t MemoryTypeIndex(VkDeviceMemory memory) const;
  VkDeviceSize BufferSize(VkBuffer buffer) const;
  VkDeviceMemory BufferMemory(VkBuffer buffer) const;  // null: unbound
  VkBufferUsageFlags BufferUsage(VkBuffer buffer) const;
  bool IsMapped(VkDeviceMemory memory) const;
  const Counts& counts() const;
  // Entry-point calls by name ("vkAllocateMemory"); direct calls not counted.
  std::uint64_t calls(const std::string& name) const;
  const std::vector<ImageTransfer>& image_transfers() const;
  std::uint64_t copy_buffer_to_image_regions() const;
  // The vkCmdCopyBufferToImage(2) commands recorded into `command_buffer`
  // since its last begin or reset, in order (copy2 regions as v1 regions).
  struct RecordedBufferCopy {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    std::vector<VkBufferImageCopy> regions;
  };
  std::vector<RecordedBufferCopy> RecordedBufferCopies(
      VkCommandBuffer command_buffer) const;
  const std::vector<std::string>& errors() const;
  void ClearErrors();

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace mocktail::graphics::testing

#endif  // MOCKTAIL_TESTS_SUPPORT_FAKE_VULKAN_DEVICE_H_
