#ifndef MOCKTAIL_GRAPHICS_VULKAN_ETC2_EMULATION_H_
#define MOCKTAIL_GRAPHICS_VULKAN_ETC2_EMULATION_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "mocktail/graphics/etc2_decoder.h"

namespace mocktail::graphics {

struct Etc2EmulatedFormat {
  // The host image's format: RGBA8, R16 or R16G16 by default, the BC
  // format in ETC2->BC mode.
  VkFormat host_format = VK_FORMAT_UNDEFINED;
  EtcFormat etc_format = EtcFormat::kEtc2Rgb8;
  // What ETC2->BC mode (MOCKTAIL_ETC2_TO_BC=1) stores instead, with the
  // same block size (DESIGN C4.1): RGB8 -> BC1_RGB, RGB8A1 -> BC1_RGBA,
  // RGBA8 -> BC3, R11 -> BC4, RG11 -> BC5; UNORM, SRGB and SNORM kept.
  VkFormat bc_host_format = VK_FORMAT_UNDEFINED;
};

// An ETC2/EAC format's emulation, with the default host format.
bool LookupEmulatedEtc2Format(VkFormat format, Etc2EmulatedFormat* out);
// The same, with the BC host format when `bc` (an ETC2->BC mode image).
bool LookupEmulatedEtc2Format(VkFormat format, bool bc,
                              Etc2EmulatedFormat* out);

VkFormatFeatureFlags EmulatedEtc2FormatFeatures(
    VkFormatFeatureFlags host_features);

// The optimal-tiling features ETC2->BC mode needs on every BC host format:
// all that EmulatedEtc2FormatFeatures() can advertise, so the ETC2/EAC
// formats report the same features in both modes (CR-22: BLIT_SRC too).
VkFormatFeatureFlags Etc2BcRequiredFormatFeatures();

// The VkImageFormatListCreateInfo in a pNext chain, or null.
const VkImageFormatListCreateInfo* FindImageFormatList(const void* next);

// [CR-14] Whether an image with this pNext chain can be BC in ETC2->BC
// mode: a format list that names a format other than ETC2/EAC asks for
// views a BC image cannot have, so such an image keeps RGBA8/R16.
bool Etc2FormatListAllowsBc(const void* next);

// [CR-15] An emulated image's VkImageFormatListCreateInfo names ETC2/EAC
// formats, which the host image does not have: for one host call this
// replaces each of them by its host format (the BC one with `bc`) and puts
// everything back when it goes out of scope. `host_next` is the pNext
// member of the caller's own copy of the create or query struct. A list
// that is the first struct of the chain is copied and linked in there, so
// Roblox's memory is not written; a list further down keeps its place and
// only its pViewFormats pointer changes until the destructor, as for the
// textureCompressionETC2 bit in vkCreateDevice.
class Etc2FormatListRewrite {
 public:
  Etc2FormatListRewrite(const void** host_next, bool bc);
  ~Etc2FormatListRewrite();
  Etc2FormatListRewrite(const Etc2FormatListRewrite&) = delete;
  Etc2FormatListRewrite& operator=(const Etc2FormatListRewrite&) = delete;

  // The chain holds a format list.
  bool found() const { return found_; }
  // ... that named an ETC2/EAC format, which the host call sees replaced.
  bool rewritten() const { return !formats_.empty(); }

 private:
  bool found_ = false;
  VkImageFormatListCreateInfo copy_{};
  VkImageFormatListCreateInfo* patched_ = nullptr;
  const VkFormat* original_ = nullptr;
  std::vector<VkFormat> formats_;
};

bool Etc2UploadByteCounts(EtcFormat format, const VkExtent3D& extent,
                          std::uint32_t layer_count, VkDeviceSize* compressed,
                          VkDeviceSize* decoded);

bool Etc2SupportAdvertised(const char* value);
bool Etc2SupportAdvertised();

inline constexpr std::uint32_t kSmallTextureMaxExtent = 64;
inline constexpr std::uint32_t kDefaultSmallTextureUpscale = 4;
std::uint32_t SmallTextureUpscale(const char* value);

enum class Etc2StagingMemory { kCached, kCoherent };

// Everything the emulation reads from the environment, read once. Tests
// construct the emulation with explicit options instead.
struct Etc2EmulationOptions {
  std::uint32_t upscale = kDefaultSmallTextureUpscale;  // MOCKTAIL_SMALL_TEXTURE_UPSCALE (1..8)
  bool fast_path = true;          // MOCKTAIL_ETC2_FAST_PATH (development only)
  bool gpu = true;                // MOCKTAIL_ETC2_GPU
  bool to_bc = false;             // MOCKTAIL_ETC2_TO_BC
  unsigned workers = 0;           // MOCKTAIL_ETC2_WORKERS, 0: min(8, cores / 2)
  std::uint64_t gpu_arena_bytes = 0;  // MOCKTAIL_ETC2_GPU_ARENA_MB, 0: automatic
  Etc2StagingMemory staging_memory = Etc2StagingMemory::kCached;  // MOCKTAIL_ETC2_STAGING_MEMORY
  bool frame_stats = false;       // MOCKTAIL_FRAME_STATS
  std::string trace_path;         // MOCKTAIL_ETC2_TRACE: one CSV line per upload
  std::string trace_blocks_dir;   // MOCKTAIL_ETC2_TRACE_BLOCKS: compressed uploads

  static Etc2EmulationOptions FromEnvironment();
  // Reads the same variables through `lookup`, which returns null for an
  // unset one. Switches take 1/true/on and 0/false/off; anything else keeps
  // the default.
  static Etc2EmulationOptions FromLookup(
      const std::function<const char*(const char*)>& lookup);
};

// A batch that only Mocktail's work is in, submitted by the caller through
// the host vkQueueSubmit right before Roblox's batches (the GPU route's
// selftest, or its decode pass for a protected or device-group batch).
struct Etc2PreSubmit {
  VkSubmitInfo info{};
  VkFence fence = VK_NULL_HANDLE;  // the selftest's only
};

// What a vkQueueSubmit hook submits after PrepareSubmit(2). Until the GPU
// route rewrites anything, these are Roblox's own arrays and fence.
struct Etc2SubmitPlan {
  bool modified = false;
  VkQueue queue = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  std::uint32_t queue_family = UINT32_MAX;
  // The rewritten batches (one of the two) and the arrays they point into.
  // Fill the arrays completely before pointing into them.
  std::vector<VkSubmitInfo> submit_infos;
  std::vector<VkSubmitInfo2> submit_infos2;
  std::vector<VkCommandBuffer> command_buffers;
  std::vector<VkCommandBufferSubmitInfo> command_buffer_infos;
  std::vector<Etc2PreSubmit> pre_submit_batches;
  // Used for Roblox's batches when Roblox passed no fence.
  VkFence substitute_fence = VK_NULL_HANDLE;
  // Signalled by a fence-only submit after Roblox's, when Roblox passed one.
  VkFence trailing_fence = VK_NULL_HANDLE;
  void* gpu_slot = nullptr;  // the GPU route's own bookkeeping

  std::uint32_t count(std::uint32_t original) const {
    if (!modified) {
      return original;
    }
    return static_cast<std::uint32_t>(submit_infos2.empty()
                                          ? submit_infos.size()
                                          : submit_infos2.size());
  }
  const VkSubmitInfo* submits(const VkSubmitInfo* original) const {
    return modified && !submit_infos.empty() ? submit_infos.data() : original;
  }
  const VkSubmitInfo2* submits2(const VkSubmitInfo2* original) const {
    return modified && !submit_infos2.empty() ? submit_infos2.data()
                                              : original;
  }
  VkFence fence(VkFence original) const {
    return original == VK_NULL_HANDLE && substitute_fence != VK_NULL_HANDLE
               ? substitute_fence
               : original;
  }
  const std::vector<Etc2PreSubmit>& pre_submits() const {
    return pre_submit_batches;
  }
};

class VulkanEtc2Emulation final {
 public:
  explicit VulkanEtc2Emulation(
      Etc2EmulationOptions options = Etc2EmulationOptions::FromEnvironment());
  ~VulkanEtc2Emulation();
  VulkanEtc2Emulation(const VulkanEtc2Emulation&) = delete;
  VulkanEtc2Emulation& operator=(const VulkanEtc2Emulation&) = delete;

  const Etc2EmulationOptions& options() const;

  bool PhysicalDeviceNeedsEmulation(VkPhysicalDevice physical_device,
                                    PFN_vkGetPhysicalDeviceFeatures host);

  // ETC2->BC mode (MOCKTAIL_ETC2_TO_BC=1, DESIGN C4.2) for devices of
  // `physical_device`: it was asked for, the device is emulated, reports
  // textureCompressionBC and gives every BC host format
  // Etc2BcRequiredFormatFeatures(). Otherwise images keep their RGBA8/R16
  // hosts and one "[vulkan] ETC2->BC unavailable: <reason>" line says why.
  // Decided once per physical device, so vkCreateDevice (which then
  // enables textureCompressionBC), the format-property hooks and the
  // device's images agree. The functions must be the host loader's.
  bool PhysicalDeviceUsesBc(
      VkPhysicalDevice physical_device,
      PFN_vkGetPhysicalDeviceFeatures get_features,
      PFN_vkGetPhysicalDeviceFormatProperties get_format_properties);

  // `properties` and `queue_families` are the physical device's;
  // `create_info` is what Roblox passed to vkCreateDevice and is only read
  // during the call (it logs the queues Roblox asked for). Mocktail's own
  // Vulkan calls go through `get_device_proc_addr`, and physical-device
  // queries through `get_instance_proc_addr` and `instance` when given:
  // both must be the host loader's, never the adapter's hooks.
  void RegisterDevice(VkDevice device, VkPhysicalDevice physical_device,
                      bool emulated,
                      const VkPhysicalDeviceMemoryProperties& memory,
                      const VkPhysicalDeviceProperties& properties,
                      const std::vector<VkQueueFamilyProperties>& queue_families,
                      const VkDeviceCreateInfo& create_info,
                      PFN_vkGetDeviceProcAddr get_device_proc_addr,
                      PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr,
                      VkInstance instance = VK_NULL_HANDLE);
  void DestroyDevice(VkDevice device);
  void RegisterQueue(VkDevice device, VkQueue queue, std::uint32_t family,
                     std::uint32_t index);
  void RegisterCommandPool(VkDevice device, VkCommandPool pool,
                           std::uint32_t family,
                           VkCommandPoolCreateFlags flags);
  // After the pool's command buffers were released.
  void ForgetCommandPool(VkDevice device, VkCommandPool pool);
  // Releases what the command buffer recorded before and starts a new
  // recording. `pool` is the one the command buffer was allocated from; the
  // adapter's command-buffer bindings are the only record of that.
  void BeginCommandBuffer(VkCommandBuffer command_buffer, VkCommandPool pool,
                          VkCommandBufferUsageFlags flags);

  VkResult CreateImage(VkDevice device, const VkImageCreateInfo* create_info,
                       const VkAllocationCallbacks* allocator, VkImage* image);
  void DestroyImage(VkDevice device, VkImage image,
                    const VkAllocationCallbacks* allocator);
  VkResult CreateImageView(VkDevice device,
                           const VkImageViewCreateInfo* create_info,
                           const VkAllocationCallbacks* allocator,
                           VkImageView* view);
  VkResult AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
                          const VkAllocationCallbacks* allocator,
                          VkDeviceMemory* memory);
  // Telemetry: which memory type an allocation has (source memory types).
  void NoteAllocateMemory(VkDevice device, VkDeviceMemory memory,
                          std::uint32_t type_index, VkDeviceSize size);
  VkResult BindBufferMemory(VkDevice device, VkBuffer buffer,
                            VkDeviceMemory memory, VkDeviceSize offset);
  VkResult BindBufferMemory2(VkDevice device, std::uint32_t count,
                             const VkBindBufferMemoryInfo* infos);
  VkResult BindImageMemory(VkDevice device, VkImage image,
                           VkDeviceMemory memory, VkDeviceSize offset);
  VkResult BindImageMemory2(VkDevice device, std::uint32_t count,
                            const VkBindImageMemoryInfo* infos);
  VkResult MapMemory(VkDevice device, VkDeviceMemory memory,
                     VkDeviceSize offset, VkDeviceSize size,
                     VkMemoryMapFlags flags, void** data);
  VkResult MapMemory2(VkDevice device, const VkMemoryMapInfo* info,
                      void** data);
  void UnmapMemory(VkDevice device, VkDeviceMemory memory);
  VkResult UnmapMemory2(VkDevice device, const VkMemoryUnmapInfo* info);
  void FreeMemory(VkDevice device, VkDeviceMemory memory,
                  const VkAllocationCallbacks* allocator);
  void DestroyBuffer(VkDevice device, VkBuffer buffer,
                     const VkAllocationCallbacks* allocator);
  void CmdCopyBufferToImage(VkDevice device, VkCommandBuffer command_buffer,
                            VkBuffer source, VkImage destination,
                            VkImageLayout layout, std::uint32_t region_count,
                            const VkBufferImageCopy* regions);
  void CmdCopyBufferToImage2(VkDevice device, VkCommandBuffer command_buffer,
                             const VkCopyBufferToImageInfo2* info);
  void CmdCopyImage(VkDevice device, VkCommandBuffer command_buffer,
                    VkImage source, VkImageLayout source_layout,
                    VkImage destination, VkImageLayout destination_layout,
                    std::uint32_t region_count, const VkImageCopy* regions);
  void CmdCopyImage2(VkDevice device, VkCommandBuffer command_buffer,
                     const VkCopyImageInfo2* info);
  void CmdExecuteCommands(VkDevice device, VkCommandBuffer command_buffer,
                          std::uint32_t count,
                          const VkCommandBuffer* secondaries);
  // Roblox recorded a dispatch into `command_buffer` (only reported with
  // MOCKTAIL_VK_CALL_STATS=1): uploads recorded after it may read what the
  // dispatch writes, which no route sees.
  void NoteDispatch(VkCommandBuffer command_buffer);

  // Inside Roblox's vkQueueSubmit(2): decodes the CPU route's uploads of the
  // batches (resubmission rereads the source bytes) and fills `plan` with
  // what the caller submits instead. The caller first submits
  // plan->pre_submits() with the host vkQueueSubmit, then
  // plan->count/submits/fence, then calls FinishSubmit with the result;
  // `submit_fence_only` submits (queue, 0, nullptr, fence) the same way.
  void PrepareSubmit(VkQueue queue, std::uint32_t count,
                     const VkSubmitInfo* submits, VkFence fence,
                     Etc2SubmitPlan* plan);
  void PrepareSubmit2(VkQueue queue, std::uint32_t count,
                      const VkSubmitInfo2* submits, VkFence fence,
                      Etc2SubmitPlan* plan);
  void FinishSubmit(Etc2SubmitPlan* plan, VkResult host_result,
                    const std::function<VkResult(VkFence)>& submit_fence_only);
  // Call only after the command buffer is no longer pending.
  void ReleaseCommandBuffer(VkCommandBuffer command_buffer);
  // After every vkQueuePresentKHR: the two-second "[perf] etc2_window" line
  // with MOCKTAIL_FRAME_STATS=1.
  void OnPresent();

  struct State;

 private:
  State* state_;
};

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_VULKAN_ETC2_EMULATION_H_
