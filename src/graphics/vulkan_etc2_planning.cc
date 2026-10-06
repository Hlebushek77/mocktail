#include "mocktail/graphics/vulkan_etc2_planning.h"

#include <algorithm>

namespace mocktail::graphics {

bool Etc2UploadByteCounts(EtcFormat format, const VkExtent3D& extent,
                          std::uint32_t layer_count, VkDeviceSize* compressed,
                          VkDeviceSize* decoded) {
  if (extent.width == 0 || extent.height == 0 || extent.depth != 1 ||
      layer_count == 0 || compressed == nullptr || decoded == nullptr) {
    return false;
  }
  const VkDeviceSize blocks =
      ((static_cast<VkDeviceSize>(extent.width) + 3) / 4) *
      ((static_cast<VkDeviceSize>(extent.height) + 3) / 4);
  *compressed = blocks * EtcBlockBytes(format) * layer_count;
  *decoded = static_cast<VkDeviceSize>(extent.width) * extent.height *
             EtcDecodedTexelBytes(format) * layer_count;
  return true;
}

// Clip block padding before scaling to the host mip.
bool ScaledAxis(std::uint32_t base, std::uint32_t scale, std::uint32_t mip,
                std::int32_t offset, std::uint32_t extent,
                std::uint32_t* texels, std::uint32_t* host_start,
                std::uint32_t* host_end) {
  const std::uint32_t app_mip = std::max<std::uint32_t>(1, base >> mip);
  const std::uint32_t host_mip =
      std::max<std::uint32_t>(1, (base * scale) >> mip);
  if (offset < 0 || static_cast<std::uint32_t>(offset) >= app_mip) {
    return false;
  }
  const std::uint32_t app_end =
      std::min(app_mip, static_cast<std::uint32_t>(offset) + extent);
  const std::uint32_t start = static_cast<std::uint32_t>(offset) * scale;
  const std::uint32_t end =
      app_end == app_mip ? host_mip : std::min(host_mip, app_end * scale);
  if (end <= start) {
    return false;
  }
  *texels = app_end - static_cast<std::uint32_t>(offset);
  *host_start = start;
  *host_end = end;
  return true;
}

bool HostRegion(const Etc2ImageShape* record, std::uint32_t mip,
                VkOffset3D offset, VkExtent3D extent, VkOffset3D* start,
                VkOffset3D* end) {
  if (record == nullptr || record->scale <= 1) {
    *start = offset;
    *end = {offset.x + static_cast<std::int32_t>(extent.width),
            offset.y + static_cast<std::int32_t>(extent.height),
            offset.z + static_cast<std::int32_t>(extent.depth)};
    return true;
  }
  std::uint32_t texels = 0;
  std::uint32_t x0 = 0;
  std::uint32_t x1 = 0;
  std::uint32_t y0 = 0;
  std::uint32_t y1 = 0;
  if (!ScaledAxis(record->width, record->scale, mip, offset.x, extent.width,
                  &texels, &x0, &x1) ||
      !ScaledAxis(record->height, record->scale, mip, offset.y,
                  extent.height, &texels, &y0, &y1)) {
    return false;
  }
  *start = {static_cast<std::int32_t>(x0), static_cast<std::int32_t>(y0), 0};
  *end = {static_cast<std::int32_t>(x1), static_cast<std::int32_t>(y1), 1};
  return true;
}

template <typename Region>
bool PlanRegions(const Etc2ImageShape& record, VkDevice device,
                 VkBuffer source, VkImage destination, std::uint32_t count,
                 const Region* regions, std::vector<Region>* rewritten,
                 std::vector<Etc2PendingUpload>* uploads, VkDeviceSize* total) {
  *total = 0;
  if (regions == nullptr || count == 0) {
    return false;
  }
  const std::uint32_t scale = record.scale;
  for (std::uint32_t index = 0; index < count; ++index) {
    const Region& region = regions[index];
    Etc2PendingUpload upload;
    if (!Etc2UploadByteCounts(record.format.etc_format, region.imageExtent,
                              region.imageSubresource.layerCount,
                              &upload.compressed, &upload.decoded)) {
      return false;
    }
    if (scale > 1 && region.imageSubresource.layerCount != 1) {
      return false;
    }
    // Each region starts 16-byte aligned in staging [DESIGN CR-19]: R16
    // texels (EAC R11) would otherwise leave the next region at 2 mod 4,
    // which a transfer-only queue forbids (VUID-vkCmdCopyBufferToImage-
    // commandBuffer-07737). 16 is a multiple of every host texel size.
    *total = (*total + kEtc2StagingRegionAlignment - 1) &
             ~(kEtc2StagingRegionAlignment - 1);
    upload.device = device;
    upload.source = source;
    upload.source_offset = region.bufferOffset;
    upload.target_offset = *total;
    upload.format = record.format.etc_format;
    upload.image = destination;
    upload.mip_level = region.imageSubresource.mipLevel;
    upload.width = region.imageExtent.width;
    upload.height = region.imageExtent.height;
    upload.layers = region.imageSubresource.layerCount;
    upload.row_length = region.bufferRowLength;
    upload.image_height = region.bufferImageHeight;
    upload.offset_x = region.imageOffset.x;
    upload.offset_y = region.imageOffset.y;
    const VkDeviceSize row_texels =
        region.bufferRowLength != 0 ? region.bufferRowLength : upload.width;
    const VkDeviceSize layer_rows =
        region.bufferImageHeight != 0 ? region.bufferImageHeight : upload.height;
    if (row_texels < upload.width || layer_rows < upload.height) {
      return false;
    }
    const VkDeviceSize block_bytes = EtcBlockBytes(upload.format);
    const VkDeviceSize block_rows =
        (static_cast<VkDeviceSize>(upload.height) + 3) / 4;
    const VkDeviceSize packed_row =
        ((static_cast<VkDeviceSize>(upload.width) + 3) / 4) * block_bytes;
    upload.source_row_stride = ((row_texels + 3) / 4) * block_bytes;
    upload.source_layer_stride =
        upload.source_row_stride * ((layer_rows + 3) / 4);
    upload.source_span = (upload.layers - 1) * upload.source_layer_stride +
                         (block_rows - 1) * upload.source_row_stride + packed_row;
    upload.scale = scale;
    upload.source_width = upload.width;
    upload.source_height = upload.height;
    upload.target_width = upload.width;
    upload.target_height = upload.height;
    upload.target_bytes = upload.decoded;
    Region copy = region;
    copy.bufferOffset = *total;
    copy.bufferRowLength = 0;
    copy.bufferImageHeight = 0;
    if (scale > 1) {
      const std::uint32_t mip = region.imageSubresource.mipLevel;
      const auto axis = [&](std::uint32_t base, std::int32_t offset,
                            std::uint32_t extent, std::uint32_t* source,
                            std::uint32_t* target, std::int32_t* host_offset,
                            std::uint32_t* host_extent) {
        std::uint32_t start = 0;
        std::uint32_t end = 0;
        if (!ScaledAxis(base, scale, mip, offset, extent, source, &start,
                        &end)) {
          return false;
        }
        *target = end - start;
        *host_offset = static_cast<std::int32_t>(start);
        *host_extent = end - start;
        return true;
      };
      if (!axis(record.width, region.imageOffset.x, region.imageExtent.width,
                &upload.source_width, &upload.target_width,
                &copy.imageOffset.x, &copy.imageExtent.width) ||
          !axis(record.height, region.imageOffset.y,
                region.imageExtent.height, &upload.source_height,
                &upload.target_height, &copy.imageOffset.y,
                &copy.imageExtent.height)) {
        return false;
      }
      upload.target_bytes = static_cast<VkDeviceSize>(upload.target_width) *
                            upload.target_height *
                            EtcDecodedTexelBytes(record.format.etc_format);
      upload.full_mip =
          region.imageOffset.x == 0 && region.imageOffset.y == 0 &&
          upload.source_width == std::max<std::uint32_t>(1, record.width >> mip) &&
          upload.source_height == std::max<std::uint32_t>(1, record.height >> mip);
    }
    rewritten->push_back(copy);
    uploads->push_back(upload);
    *total += upload.target_bytes;
  }
  return true;
}

template bool PlanRegions<VkBufferImageCopy>(
    const Etc2ImageShape&, VkDevice, VkBuffer, VkImage, std::uint32_t,
    const VkBufferImageCopy*, std::vector<VkBufferImageCopy>*,
    std::vector<Etc2PendingUpload>*, VkDeviceSize*);
template bool PlanRegions<VkBufferImageCopy2>(
    const Etc2ImageShape&, VkDevice, VkBuffer, VkImage, std::uint32_t,
    const VkBufferImageCopy2*, std::vector<VkBufferImageCopy2>*,
    std::vector<Etc2PendingUpload>*, VkDeviceSize*);

}  // namespace mocktail::graphics
