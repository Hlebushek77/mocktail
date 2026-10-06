#ifndef MOCKTAIL_GRAPHICS_VULKAN_ETC2_PLANNING_H_
#define MOCKTAIL_GRAPHICS_VULKAN_ETC2_PLANNING_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/vulkan_etc2_emulation.h"

// Pure planning for uploads into emulated ETC2/EAC images: which bytes of
// Roblox's buffer an upload reads, and where its decoded texels go. No Vulkan
// calls, no state.

namespace mocktail::graphics {

// What planning needs to know about an emulated image. `scale` > 1 for small
// colour textures that the host image holds enlarged; `width` and `height`
// are Roblox's (unscaled) mip-0 extent.
struct Etc2ImageShape {
  Etc2EmulatedFormat format;
  std::uint32_t scale = 1;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

// How one vkCmdCopyBufferToImage(2) on an emulated image is served.
enum class Etc2Route : std::uint8_t {
  kPassthrough,  // not an emulated image: the host call, unchanged
  kCpu,          // decoded on the CPU inside vkQueueSubmit into staging
  kGpu,          // decoded on the GPU in a command buffer before Roblox's
};

inline constexpr std::uint32_t kEtc2UnknownQueueFamily = UINT32_MAX;

// One region of an upload, as the CPU route decodes it at submit time.
struct Etc2PendingUpload {
  VkDevice device = VK_NULL_HANDLE;
  VkBuffer source = VK_NULL_HANDLE;
  VkDeviceSize source_offset = 0;
  VkDeviceSize source_row_stride = 0;
  VkDeviceSize source_layer_stride = 0;
  VkDeviceSize source_span = 0;
  VkDeviceSize compressed = 0;
  VkDeviceSize decoded = 0;
  VkDeviceSize target_offset = 0;
  EtcFormat format = EtcFormat::kEtc2Rgb8;
  VkImage image = VK_NULL_HANDLE;
  std::uint32_t mip_level = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t layers = 0;
  std::uint32_t scale = 1;
  std::uint32_t source_width = 0;
  std::uint32_t source_height = 0;
  std::uint32_t target_width = 0;
  std::uint32_t target_height = 0;
  bool full_mip = false;
  VkDeviceSize target_bytes = 0;
  std::uint8_t* target = nullptr;
  // Set by the emulation when it records the copy.
  Etc2Route route = Etc2Route::kCpu;
  std::uint32_t pool_family = kEtc2UnknownQueueFamily;
  // Roblox's region as given, for the upload trace.
  VkFormat app_format = VK_FORMAT_UNDEFINED;
  std::uint32_t row_length = 0;
  std::uint32_t image_height = 0;
  std::int32_t offset_x = 0;
  std::int32_t offset_y = 0;
  std::uint64_t record_call = 0;
};

// One axis of an upscaled image's region: clips block padding off the app
// region before scaling it to the host mip. False for a region outside the
// mip.
bool ScaledAxis(std::uint32_t base, std::uint32_t scale, std::uint32_t mip,
                std::int32_t offset, std::uint32_t extent,
                std::uint32_t* texels, std::uint32_t* host_start,
                std::uint32_t* host_end);

// The host image's region for an app region of `image` (null: not scaled).
bool HostRegion(const Etc2ImageShape* image, std::uint32_t mip,
                VkOffset3D offset, VkExtent3D extent, VkOffset3D* start,
                VkOffset3D* end);

// Where each region's decoded texels start in CPU-route staging.
inline constexpr VkDeviceSize kEtc2StagingRegionAlignment = 16;

// Plans the CPU route of one copy: per region an upload, and a rewritten
// region that reads decoded texels tightly packed from a staging buffer at
// offset 0 onward, each region starting kEtc2StagingRegionAlignment-aligned.
// `total` is the staging size. False when a region cannot be served (bad
// extent, layered upscaled image, short row length).
template <typename Region>
bool PlanRegions(const Etc2ImageShape& image, VkDevice device,
                 VkBuffer source, VkImage destination, std::uint32_t count,
                 const Region* regions, std::vector<Region>* rewritten,
                 std::vector<Etc2PendingUpload>* uploads, VkDeviceSize* total);

extern template bool PlanRegions<VkBufferImageCopy>(
    const Etc2ImageShape&, VkDevice, VkBuffer, VkImage, std::uint32_t,
    const VkBufferImageCopy*, std::vector<VkBufferImageCopy>*,
    std::vector<Etc2PendingUpload>*, VkDeviceSize*);
extern template bool PlanRegions<VkBufferImageCopy2>(
    const Etc2ImageShape&, VkDevice, VkBuffer, VkImage, std::uint32_t,
    const VkBufferImageCopy2*, std::vector<VkBufferImageCopy2>*,
    std::vector<Etc2PendingUpload>*, VkDeviceSize*);

// One region of a GPU-route upload: what the decode pass reads from
// Roblox's buffer and where in the call's output range it writes. Strides
// are in bytes of compressed data. (The GPU route's work package may extend
// this record; nothing outside the emulation reads it.)
struct Etc2GpuUpload {
  VkBuffer source = VK_NULL_HANDLE;
  VkDeviceSize source_offset = 0;
  VkDeviceSize source_row_stride = 0;
  VkDeviceSize source_layer_stride = 0;
  VkDeviceSize source_span = 0;
  VkDeviceSize compressed = 0;
  EtcFormat format = EtcFormat::kEtc2Rgb8;
  std::uint32_t blocks_wide = 0;
  std::uint32_t blocks_high = 0;
  std::uint32_t layers = 0;
  VkDeviceSize output_offset = 0;  // inside the call's output range
  VkDeviceSize output_bytes = 0;
  VkImage image = VK_NULL_HANDLE;
  std::uint32_t mip_level = 0;
  bool bc = false;
};

// Plans the GPU route of one copy: padded rows of 4 x blocks_wide texels,
// 256-byte aligned regions, extents clamped to the mip; in BC mode tight BC
// blocks. Declared here for the GPU route; defined with it.
template <typename Region>
bool PlanGpuRegions(const Etc2ImageShape& image, VkBuffer source,
                    std::uint32_t count, const Region* regions, bool bc,
                    std::vector<Region>* rewritten,
                    std::vector<Etc2GpuUpload>* uploads, VkDeviceSize* total);

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_VULKAN_ETC2_PLANNING_H_
