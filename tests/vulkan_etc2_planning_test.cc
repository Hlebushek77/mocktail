#include "mocktail/graphics/vulkan_etc2_planning.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

// The CPU route's pure planning (DESIGN §9.1): which bytes of Roblox's
// buffer each region reads, where its decoded texels go in staging, and the
// host region of an upscaled image. Every expectation is worked out by hand
// from the Vulkan copy rules and the upscale rule (block padding is clipped
// before scaling; the last texel row or column reaches the host mip's
// edge), not taken from the code. PlanGpuRegions belongs to the GPU route
// and is tested with it.

namespace mocktail::graphics {
namespace {

const VkDevice kDevice = reinterpret_cast<VkDevice>(0xd0);
const VkBuffer kSource = reinterpret_cast<VkBuffer>(0xb0);
const VkImage kImage = reinterpret_cast<VkImage>(0xa0);

Etc2ImageShape Shape(VkFormat format, std::uint32_t scale, std::uint32_t width,
                     std::uint32_t height) {
  Etc2ImageShape shape;
  EXPECT_TRUE(LookupEmulatedEtc2Format(format, &shape.format));
  shape.scale = scale;
  shape.width = width;
  shape.height = height;
  return shape;
}

VkBufferImageCopy Region(VkDeviceSize buffer_offset, std::uint32_t row_length,
                         std::uint32_t image_height, std::uint32_t mip,
                         std::uint32_t base_layer, std::uint32_t layers,
                         std::int32_t x, std::int32_t y, std::uint32_t width,
                         std::uint32_t height) {
  VkBufferImageCopy region{};
  region.bufferOffset = buffer_offset;
  region.bufferRowLength = row_length;
  region.bufferImageHeight = image_height;
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, base_layer,
                             layers};
  region.imageOffset = {x, y, 0};
  region.imageExtent = {width, height, 1};
  return region;
}

struct Plan {
  bool ok = false;
  std::vector<VkBufferImageCopy> rewritten;
  std::vector<Etc2PendingUpload> uploads;
  VkDeviceSize total = 0;
};

Plan PlanAll(const Etc2ImageShape& shape,
             const std::vector<VkBufferImageCopy>& regions) {
  Plan plan;
  plan.ok =
      PlanRegions(shape, kDevice, kSource, kImage,
                  static_cast<std::uint32_t>(regions.size()), regions.data(),
                  &plan.rewritten, &plan.uploads, &plan.total);
  return plan;
}

TEST(Etc2UploadByteCountsTest, CountsWholeBlocksAndDecodedTexels) {
  VkDeviceSize compressed = 0;
  VkDeviceSize decoded = 0;
  // 6x5 texels: 2x2 blocks of 8 bytes; RGBA8 out.
  ASSERT_TRUE(Etc2UploadByteCounts(EtcFormat::kEtc2Rgb8, {6, 5, 1}, 1,
                                   &compressed, &decoded));
  EXPECT_EQ(compressed, 32U);
  EXPECT_EQ(decoded, 120U);
  // 16-byte blocks, two layers; RG16 out.
  ASSERT_TRUE(Etc2UploadByteCounts(EtcFormat::kEacRg11, {4, 4, 1}, 2,
                                   &compressed, &decoded));
  EXPECT_EQ(compressed, 32U);
  EXPECT_EQ(decoded, 128U);
  // R16 out.
  ASSERT_TRUE(Etc2UploadByteCounts(EtcFormat::kEacR11Signed, {9, 7, 1}, 3,
                                   &compressed, &decoded));
  EXPECT_EQ(compressed, 144U);
  EXPECT_EQ(decoded, 378U);
  EXPECT_FALSE(Etc2UploadByteCounts(EtcFormat::kEtc2Rgb8, {0, 4, 1}, 1,
                                    &compressed, &decoded));
  EXPECT_FALSE(Etc2UploadByteCounts(EtcFormat::kEtc2Rgb8, {4, 4, 2}, 1,
                                    &compressed, &decoded));
  EXPECT_FALSE(Etc2UploadByteCounts(EtcFormat::kEtc2Rgb8, {4, 4, 1}, 0,
                                    &compressed, &decoded));
  EXPECT_FALSE(Etc2UploadByteCounts(EtcFormat::kEtc2Rgb8, {4, 4, 1}, 1, nullptr,
                                    &decoded));
}

TEST(ScaledAxisTest, ScalesWholeAndPartialRanges) {
  std::uint32_t texels = 0;
  std::uint32_t start = 0;
  std::uint32_t end = 0;
  ASSERT_TRUE(ScaledAxis(64, 4, 0, 0, 64, &texels, &start, &end));
  EXPECT_EQ(texels, 64U);
  EXPECT_EQ(start, 0U);
  EXPECT_EQ(end, 256U);
  ASSERT_TRUE(ScaledAxis(64, 4, 0, 8, 16, &texels, &start, &end));
  EXPECT_EQ(texels, 16U);
  EXPECT_EQ(start, 32U);
  EXPECT_EQ(end, 96U);
  // Mip 3 of 64 is 8 texels, 32 on the host.
  ASSERT_TRUE(ScaledAxis(64, 4, 3, 4, 4, &texels, &start, &end));
  EXPECT_EQ(texels, 4U);
  EXPECT_EQ(start, 16U);
  EXPECT_EQ(end, 32U);
  // Scale 1 keeps the range.
  ASSERT_TRUE(ScaledAxis(70, 1, 1, 32, 3, &texels, &start, &end));
  EXPECT_EQ(texels, 3U);
  EXPECT_EQ(start, 32U);
  EXPECT_EQ(end, 35U);
}

TEST(ScaledAxisTest, ClipsBlockPaddingAndReachesTheHostEdge) {
  std::uint32_t texels = 0;
  std::uint32_t start = 0;
  std::uint32_t end = 0;
  // Mip 2 of 5 is 1 texel, of the host's 20 it is 5: a 4-texel block
  // covers 1 app texel and all 5 host texels.
  ASSERT_TRUE(ScaledAxis(5, 4, 2, 0, 4, &texels, &start, &end));
  EXPECT_EQ(texels, 1U);
  EXPECT_EQ(start, 0U);
  EXPECT_EQ(end, 5U);
  // Mip 3 of 20 is 2 texels; the host's 80 >> 3 = 10, not 2 * 4.
  ASSERT_TRUE(ScaledAxis(20, 4, 3, 1, 1, &texels, &start, &end));
  EXPECT_EQ(texels, 1U);
  EXPECT_EQ(start, 4U);
  EXPECT_EQ(end, 10U);
  // A range that stops inside the mip stops at offset + extent, scaled.
  ASSERT_TRUE(ScaledAxis(20, 4, 3, 0, 1, &texels, &start, &end));
  EXPECT_EQ(texels, 1U);
  EXPECT_EQ(start, 0U);
  EXPECT_EQ(end, 4U);
}

TEST(ScaledAxisTest, RejectsRangesOutsideTheMip) {
  std::uint32_t texels = 7;
  std::uint32_t start = 7;
  std::uint32_t end = 7;
  EXPECT_FALSE(ScaledAxis(64, 4, 0, 64, 4, &texels, &start, &end));
  EXPECT_FALSE(ScaledAxis(64, 4, 3, 8, 4, &texels, &start, &end));
  EXPECT_FALSE(ScaledAxis(64, 4, 0, -4, 8, &texels, &start, &end));
  EXPECT_FALSE(ScaledAxis(64, 4, 0, 8, 0, &texels, &start, &end));
  EXPECT_EQ(texels, 7U) << "outputs untouched on failure";
  EXPECT_EQ(start, 7U);
  EXPECT_EQ(end, 7U);
}

TEST(HostRegionTest, UnscaledImagesKeepTheRegion) {
  VkOffset3D start{};
  VkOffset3D end{};
  ASSERT_TRUE(HostRegion(nullptr, 2, {8, 4, 0}, {16, 12, 1}, &start, &end));
  EXPECT_EQ(start.x, 8);
  EXPECT_EQ(start.y, 4);
  EXPECT_EQ(start.z, 0);
  EXPECT_EQ(end.x, 24);
  EXPECT_EQ(end.y, 16);
  EXPECT_EQ(end.z, 1);
  const Etc2ImageShape unscaled =
      Shape(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 1, 64, 64);
  ASSERT_TRUE(HostRegion(&unscaled, 0, {1, 2, 0}, {3, 5, 1}, &start, &end));
  EXPECT_EQ(start.x, 1);
  EXPECT_EQ(start.y, 2);
  EXPECT_EQ(end.x, 4);
  EXPECT_EQ(end.y, 7);
}

TEST(HostRegionTest, ScalesBothAxesOfAnUpscaledImage) {
  const Etc2ImageShape shape =
      Shape(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 4, 20, 12);
  VkOffset3D start{};
  VkOffset3D end{};
  ASSERT_TRUE(HostRegion(&shape, 0, {8, 4, 0}, {4, 4, 1}, &start, &end));
  EXPECT_EQ(start.x, 32);
  EXPECT_EQ(start.y, 16);
  EXPECT_EQ(start.z, 0);
  EXPECT_EQ(end.x, 48);
  EXPECT_EQ(end.y, 32);
  EXPECT_EQ(end.z, 1);
  // Mip 2 is 5x3 (host 20x12): the last column reaches the host edge.
  ASSERT_TRUE(HostRegion(&shape, 2, {4, 0, 0}, {1, 3, 1}, &start, &end));
  EXPECT_EQ(start.x, 16);
  EXPECT_EQ(start.y, 0);
  EXPECT_EQ(end.x, 20);
  EXPECT_EQ(end.y, 12);
  EXPECT_FALSE(HostRegion(&shape, 2, {5, 0, 0}, {1, 3, 1}, &start, &end));
  EXPECT_FALSE(HostRegion(&shape, 0, {0, 12, 0}, {4, 4, 1}, &start, &end));
}

TEST(PlanRegionsTest, PlansStridesLayersAndStagingOfUnscaledRegions) {
  // EAC R11: 8-byte blocks, R16 texels.
  const Etc2ImageShape shape = Shape(VK_FORMAT_EAC_R11_UNORM_BLOCK, 1, 70, 50);
  const Plan plan = PlanAll(shape, {Region(24, 0, 0, 0, 0, 1, 0, 0, 6, 5),
                                    Region(1000, 20, 12, 1, 2, 3, 8, 4, 9, 7)});
  ASSERT_TRUE(plan.ok);
  ASSERT_EQ(plan.uploads.size(), 2U);
  ASSERT_EQ(plan.rewritten.size(), 2U);

  // 6x5 packed: 2x2 blocks, rows of 16 bytes.
  const Etc2PendingUpload& first = plan.uploads[0];
  EXPECT_EQ(first.device, kDevice);
  EXPECT_EQ(first.source, kSource);
  EXPECT_EQ(first.image, kImage);
  EXPECT_EQ(first.format, EtcFormat::kEacR11);
  EXPECT_EQ(first.source_offset, 24U);
  EXPECT_EQ(first.source_row_stride, 16U);
  EXPECT_EQ(first.source_layer_stride, 32U);
  EXPECT_EQ(first.source_span, 32U);
  EXPECT_EQ(first.compressed, 32U);
  EXPECT_EQ(first.decoded, 60U);
  EXPECT_EQ(first.width, 6U);
  EXPECT_EQ(first.height, 5U);
  EXPECT_EQ(first.layers, 1U);
  EXPECT_EQ(first.mip_level, 0U);
  EXPECT_EQ(first.scale, 1U);
  EXPECT_EQ(first.target_width, 6U);
  EXPECT_EQ(first.target_height, 5U);
  EXPECT_EQ(first.target_bytes, 60U);
  EXPECT_FALSE(first.full_mip) << "only upscaled images track full mips";

  // 9x7 in rows of 20 texels (5 blocks, 40 bytes) and layers of 12 rows
  // (3 block rows, 120 bytes); 3 layers: 2 * 120 + 40 + 3 * 8 bytes.
  const Etc2PendingUpload& second = plan.uploads[1];
  EXPECT_EQ(second.source_offset, 1000U);
  EXPECT_EQ(second.source_row_stride, 40U);
  EXPECT_EQ(second.source_layer_stride, 120U);
  EXPECT_EQ(second.source_span, 304U);
  EXPECT_EQ(second.compressed, 144U);
  EXPECT_EQ(second.decoded, 378U);
  EXPECT_EQ(second.layers, 3U);
  EXPECT_EQ(second.mip_level, 1U);
  EXPECT_EQ(second.row_length, 20U);
  EXPECT_EQ(second.image_height, 12U);
  EXPECT_EQ(second.offset_x, 8);
  EXPECT_EQ(second.offset_y, 4);
  EXPECT_EQ(second.target_bytes, 378U);

  // The host copies read tightly packed decoded texels from staging, each
  // region from a 16-byte boundary (CR-19): the first ends at byte 60 (30
  // R16 texels), so the second starts at 64.
  EXPECT_EQ(first.target_offset, 0U);
  EXPECT_EQ(second.target_offset, 64U);
  EXPECT_EQ(plan.total, 442U);
  for (std::size_t index = 0; index < 2; ++index) {
    const VkBufferImageCopy& copy = plan.rewritten[index];
    EXPECT_EQ(copy.bufferOffset, plan.uploads[index].target_offset);
    EXPECT_EQ(copy.bufferRowLength, 0U);
    EXPECT_EQ(copy.bufferImageHeight, 0U);
  }
  const VkBufferImageCopy& copy = plan.rewritten[1];
  EXPECT_EQ(copy.imageSubresource.mipLevel, 1U);
  EXPECT_EQ(copy.imageSubresource.baseArrayLayer, 2U);
  EXPECT_EQ(copy.imageSubresource.layerCount, 3U);
  EXPECT_EQ(copy.imageOffset.x, 8);
  EXPECT_EQ(copy.imageOffset.y, 4);
  EXPECT_EQ(copy.imageExtent.width, 9U);
  EXPECT_EQ(copy.imageExtent.height, 7U);
}

// CR-19: R16 regions with an odd texel count end at 2 mod 4, and a
// transfer-only queue needs every bufferOffset to be a multiple of 4
// (VUID-vkCmdCopyBufferToImage-commandBuffer-07737). Every region starts on
// a 16-byte boundary instead; the texels each region carries are unchanged.
TEST(PlanRegionsTest, StartsEveryRegionOnASixteenByteBoundary) {
  const Etc2ImageShape shape = Shape(VK_FORMAT_EAC_R11_UNORM_BLOCK, 1, 5, 3);
  // 5x3, 1x1 and 3x1 texels: 30, 2 and 6 bytes of R16.
  const Plan plan = PlanAll(shape, {Region(0, 0, 0, 0, 0, 1, 0, 0, 5, 3),
                                    Region(64, 0, 0, 2, 0, 1, 0, 0, 1, 1),
                                    Region(72, 0, 0, 0, 0, 1, 1, 1, 3, 1)});
  ASSERT_TRUE(plan.ok);
  ASSERT_EQ(plan.uploads.size(), 3U);
  EXPECT_EQ(plan.uploads[0].target_offset, 0U);
  EXPECT_EQ(plan.uploads[1].target_offset, 32U);
  EXPECT_EQ(plan.uploads[2].target_offset, 48U);
  EXPECT_EQ(plan.total, 54U) << "no padding after the last region";
  for (std::size_t index = 0; index < plan.uploads.size(); ++index) {
    EXPECT_EQ(plan.uploads[index].target_offset % 16, 0U);
    EXPECT_EQ(plan.rewritten[index].bufferOffset,
              plan.uploads[index].target_offset);
  }
  EXPECT_EQ(plan.uploads[0].target_bytes, 30U);
  EXPECT_EQ(plan.uploads[1].target_bytes, 2U);
  EXPECT_EQ(plan.uploads[2].target_bytes, 6U);
}

// ETC2->BC mode: a BC block per ETC2/EAC block, tightly packed in staging
// (bufferRowLength 0 counts texels, so a row is ceil(width / 4) blocks),
// the app's extent and the 16-byte region starts kept.
TEST(PlanRegionsTest, PlansTightBcBlocksForBcImages) {
  Etc2ImageShape shape = Shape(VK_FORMAT_EAC_R11_UNORM_BLOCK, 1, 70, 50);
  shape.bc = true;
  // 70x50 from 80-texel rows: 18x13 blocks of 8 bytes, 1872 bytes from
  // 160-byte source rows; 35x25: 9x7 blocks, 504 bytes; 6x2 at the corner:
  // 2x1 blocks, 16 bytes, at 2376 rounded up to 2384.
  const Plan plan = PlanAll(shape, {Region(0, 80, 0, 0, 0, 1, 0, 0, 70, 50),
                                    Region(4096, 0, 0, 1, 0, 1, 0, 0, 35, 25),
                                    Region(8192, 0, 0, 0, 0, 1, 64, 48, 6, 2)});
  ASSERT_TRUE(plan.ok);
  ASSERT_EQ(plan.uploads.size(), 3U);
  const VkDeviceSize offsets[] = {0, 1872, 2384};
  const VkDeviceSize bytes[] = {1872, 504, 16};
  for (std::size_t index = 0; index < 3; ++index) {
    const Etc2PendingUpload& upload = plan.uploads[index];
    const VkBufferImageCopy& region = plan.rewritten[index];
    EXPECT_TRUE(upload.bc);
    EXPECT_EQ(upload.compressed, bytes[index]);
    EXPECT_EQ(upload.target_bytes, bytes[index]);
    EXPECT_EQ(upload.target_offset, offsets[index]);
    EXPECT_EQ(region.bufferOffset, offsets[index]);
    EXPECT_EQ(region.bufferRowLength, 0U);
    EXPECT_EQ(region.bufferImageHeight, 0U);
  }
  EXPECT_EQ(plan.uploads[0].source_row_stride, 160U);
  EXPECT_EQ(plan.rewritten[2].imageOffset.x, 64);
  EXPECT_EQ(plan.rewritten[2].imageExtent.width, 6U);
  EXPECT_EQ(plan.rewritten[2].imageExtent.height, 2U);
  EXPECT_EQ(plan.total, 2400U);

  // Layers lie one after another: 33x20 RGBA8 -> BC3, 9x5 blocks of 16
  // bytes, three layers from 24-row source layers.
  Etc2ImageShape layered =
      Shape(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 1, 33, 20);
  layered.bc = true;
  const Plan layers =
      PlanAll(layered, {Region(0, 0, 24, 0, 0, 3, 0, 0, 33, 20)});
  ASSERT_TRUE(layers.ok);
  EXPECT_EQ(layers.uploads[0].target_bytes, 9U * 5 * 16 * 3);
  EXPECT_EQ(layers.uploads[0].source_layer_stride, 9U * 6 * 16);
  EXPECT_EQ(layers.total, 9U * 5 * 16 * 3);
  EXPECT_EQ(layers.rewritten[0].imageSubresource.layerCount, 3U);

  // BC images are never upscaled; a shape saying so is refused.
  Etc2ImageShape scaled = Shape(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 4, 16, 16);
  scaled.bc = true;
  EXPECT_FALSE(PlanAll(scaled, {Region(0, 0, 0, 0, 0, 1, 0, 0, 16, 16)}).ok);
}

// Extents rounded up to whole blocks past a small mip's edge (which the x4
// upscale clipped) reach a BC host clipped to the mip, reading the same
// staging rows; a region starting outside the mip is refused.
TEST(PlanRegionsTest, ClipsBlockRoundedExtentsOfBcImagesToTheMip) {
  Etc2ImageShape shape = Shape(VK_FORMAT_EAC_R11G11_UNORM_BLOCK, 1, 6, 6);
  shape.bc = true;
  // 8x8 on the 6x6 mip 0: 2x2 blocks of 16 bytes; 4x4 on the 1x1 mip 2:
  // one block; 4x4 at 4,4 of mip 0: one block, 2x2 of it inside.
  const Plan plan = PlanAll(shape, {Region(0, 0, 0, 0, 0, 1, 0, 0, 8, 8),
                                    Region(64, 0, 0, 2, 0, 1, 0, 0, 4, 4),
                                    Region(80, 0, 0, 0, 0, 1, 4, 4, 4, 4)});
  ASSERT_TRUE(plan.ok);
  ASSERT_EQ(plan.rewritten.size(), 3U);
  const std::uint32_t extents[][2] = {{6, 6}, {1, 1}, {2, 2}};
  const std::uint32_t pitches[][2] = {{8, 8}, {4, 4}, {4, 4}};
  const VkDeviceSize bytes[] = {64, 16, 16};
  for (std::size_t index = 0; index < 3; ++index) {
    const VkBufferImageCopy& region = plan.rewritten[index];
    EXPECT_EQ(region.imageExtent.width, extents[index][0]) << index;
    EXPECT_EQ(region.imageExtent.height, extents[index][1]) << index;
    EXPECT_EQ(region.bufferRowLength, pitches[index][0]) << index;
    EXPECT_EQ(region.bufferImageHeight, pitches[index][1]) << index;
    EXPECT_EQ(plan.uploads[index].target_bytes, bytes[index]) << index;
    // The upload itself keeps the region as given.
    EXPECT_EQ(plan.uploads[index].target_width,
              plan.uploads[index].width) << index;
  }
  EXPECT_EQ(plan.rewritten[2].imageOffset.x, 4);
  EXPECT_EQ(plan.total, 64U + 16 + 16);
  EXPECT_FALSE(PlanAll(shape, {Region(0, 0, 0, 0, 0, 1, 8, 0, 4, 4)}).ok);
  EXPECT_FALSE(PlanAll(shape, {Region(0, 0, 0, 1, 0, 1, 0, 4, 4, 4)}).ok);
  // The default mode keeps such regions as they are.
  shape.bc = false;
  const Plan rgba = PlanAll(shape, {Region(0, 0, 0, 0, 0, 1, 0, 0, 8, 8)});
  ASSERT_TRUE(rgba.ok);
  EXPECT_EQ(rgba.rewritten[0].imageExtent.width, 8U);
  EXPECT_EQ(rgba.rewritten[0].bufferRowLength, 0U);
}

TEST(PlanRegionsTest, ScalesRegionsOfAnUpscaledImage) {
  // ETC2 RGBA8 20x12 held 4x larger: 80x48 on the host.
  const Etc2ImageShape shape =
      Shape(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 4, 20, 12);
  const Plan plan = PlanAll(shape, {Region(0, 0, 0, 0, 0, 1, 0, 0, 20, 12),
                                    Region(512, 0, 0, 2, 0, 1, 4, 0, 1, 3),
                                    Region(768, 0, 0, 3, 0, 1, 0, 0, 2, 1)});
  ASSERT_TRUE(plan.ok);
  ASSERT_EQ(plan.uploads.size(), 3U);

  const Etc2PendingUpload& whole = plan.uploads[0];
  EXPECT_EQ(whole.compressed, 240U);  // 5x3 blocks of 16 bytes
  EXPECT_EQ(whole.decoded, 960U);
  EXPECT_EQ(whole.scale, 4U);
  EXPECT_EQ(whole.source_width, 20U);
  EXPECT_EQ(whole.source_height, 12U);
  EXPECT_EQ(whole.target_width, 80U);
  EXPECT_EQ(whole.target_height, 48U);
  EXPECT_EQ(whole.target_bytes, 80U * 48U * 4U);
  EXPECT_TRUE(whole.full_mip);
  EXPECT_EQ(plan.rewritten[0].imageExtent.width, 80U);
  EXPECT_EQ(plan.rewritten[0].imageExtent.height, 48U);

  // Mip 2 is 5x3 (host 20x12); its last column, 1x3 texels.
  const Etc2PendingUpload& column = plan.uploads[1];
  EXPECT_EQ(column.compressed, 16U);
  EXPECT_EQ(column.decoded, 12U);
  EXPECT_EQ(column.source_row_stride, 16U);
  EXPECT_EQ(column.source_span, 16U);
  EXPECT_EQ(column.source_width, 1U);
  EXPECT_EQ(column.source_height, 3U);
  EXPECT_EQ(column.target_width, 4U);
  EXPECT_EQ(column.target_height, 12U);
  EXPECT_EQ(column.target_bytes, 192U);
  EXPECT_FALSE(column.full_mip);
  EXPECT_EQ(plan.rewritten[1].imageOffset.x, 16);
  EXPECT_EQ(plan.rewritten[1].imageOffset.y, 0);
  EXPECT_EQ(plan.rewritten[1].imageExtent.width, 4U);
  EXPECT_EQ(plan.rewritten[1].imageExtent.height, 12U);

  // Mip 3 is 2x1, on the host 10x6: not 8x4, the host mip chain rounds
  // down from 80x48.
  const Etc2PendingUpload& tail = plan.uploads[2];
  EXPECT_EQ(tail.source_width, 2U);
  EXPECT_EQ(tail.source_height, 1U);
  EXPECT_EQ(tail.target_width, 10U);
  EXPECT_EQ(tail.target_height, 6U);
  EXPECT_TRUE(tail.full_mip);
  EXPECT_EQ(plan.rewritten[2].imageExtent.width, 10U);
  EXPECT_EQ(plan.rewritten[2].imageExtent.height, 6U);

  EXPECT_EQ(column.target_offset, 15360U);
  EXPECT_EQ(tail.target_offset, 15552U);
  EXPECT_EQ(plan.total, 15792U);
  EXPECT_EQ(plan.rewritten[2].bufferOffset, 15552U);
}

TEST(PlanRegionsTest, RejectsRegionsItCannotServe) {
  const Etc2ImageShape unscaled =
      Shape(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 1, 64, 64);
  const Etc2ImageShape scaled =
      Shape(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 4, 32, 32);
  const VkBufferImageCopy good = Region(0, 0, 0, 0, 0, 1, 0, 0, 8, 8);
  EXPECT_TRUE(PlanAll(unscaled, {good}).ok);
  EXPECT_FALSE(PlanAll(unscaled, {}).ok);
  VkBufferImageCopy bad = good;
  bad.imageExtent.width = 0;
  EXPECT_FALSE(PlanAll(unscaled, {good, bad}).ok);
  bad = good;
  bad.imageExtent.depth = 2;
  EXPECT_FALSE(PlanAll(unscaled, {bad}).ok);
  bad = good;
  bad.imageSubresource.layerCount = 0;
  EXPECT_FALSE(PlanAll(unscaled, {bad}).ok);
  bad = good;
  bad.bufferRowLength = 7;  // shorter than the 8-texel rows
  EXPECT_FALSE(PlanAll(unscaled, {bad}).ok);
  bad = good;
  bad.bufferImageHeight = 4;  // shorter than the 8 rows
  EXPECT_FALSE(PlanAll(unscaled, {bad}).ok);
  // An upscaled image takes one layer per region, inside its mip.
  bad = good;
  bad.imageSubresource.layerCount = 2;
  EXPECT_TRUE(PlanAll(unscaled, {bad}).ok);
  EXPECT_FALSE(PlanAll(scaled, {bad}).ok);
  bad = good;
  bad.imageOffset.x = 32;
  EXPECT_FALSE(PlanAll(scaled, {bad}).ok);
  VkDeviceSize total = 1;
  std::vector<VkBufferImageCopy> rewritten;
  std::vector<Etc2PendingUpload> uploads;
  EXPECT_FALSE(PlanRegions<VkBufferImageCopy>(unscaled, kDevice, kSource,
                                              kImage, 1, nullptr, &rewritten,
                                              &uploads, &total));
  EXPECT_EQ(total, 0U);
}

TEST(PlanRegionsTest, CopyCommands2PlanTheSameAndKeepTheirChain) {
  const Etc2ImageShape shape =
      Shape(VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, 4, 20, 12);
  const std::vector<VkBufferImageCopy> regions = {
      Region(0, 24, 16, 0, 0, 1, 0, 0, 20, 12),
      Region(4096, 0, 0, 2, 0, 1, 4, 0, 1, 3)};
  const Plan plan = PlanAll(shape, regions);
  ASSERT_TRUE(plan.ok);
  const int marker = 0;
  std::vector<VkBufferImageCopy2> regions2;
  for (const VkBufferImageCopy& region : regions) {
    VkBufferImageCopy2 copy{};
    copy.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
    copy.pNext = &marker;
    copy.bufferOffset = region.bufferOffset;
    copy.bufferRowLength = region.bufferRowLength;
    copy.bufferImageHeight = region.bufferImageHeight;
    copy.imageSubresource = region.imageSubresource;
    copy.imageOffset = region.imageOffset;
    copy.imageExtent = region.imageExtent;
    regions2.push_back(copy);
  }
  std::vector<VkBufferImageCopy2> rewritten;
  std::vector<Etc2PendingUpload> uploads;
  VkDeviceSize total = 0;
  ASSERT_TRUE(PlanRegions(shape, kDevice, kSource, kImage,
                          static_cast<std::uint32_t>(regions2.size()),
                          regions2.data(), &rewritten, &uploads, &total));
  EXPECT_EQ(total, plan.total);
  ASSERT_EQ(uploads.size(), plan.uploads.size());
  for (std::size_t index = 0; index < uploads.size(); ++index) {
    const Etc2PendingUpload& a = uploads[index];
    const Etc2PendingUpload& b = plan.uploads[index];
    EXPECT_EQ(a.source_offset, b.source_offset);
    EXPECT_EQ(a.source_row_stride, b.source_row_stride);
    EXPECT_EQ(a.source_layer_stride, b.source_layer_stride);
    EXPECT_EQ(a.source_span, b.source_span);
    EXPECT_EQ(a.target_offset, b.target_offset);
    EXPECT_EQ(a.target_width, b.target_width);
    EXPECT_EQ(a.target_height, b.target_height);
    EXPECT_EQ(a.full_mip, b.full_mip);
    EXPECT_EQ(rewritten[index].sType, VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2);
    EXPECT_EQ(rewritten[index].pNext, &marker);
    EXPECT_EQ(rewritten[index].bufferOffset,
              plan.rewritten[index].bufferOffset);
    EXPECT_EQ(rewritten[index].imageOffset.x,
              plan.rewritten[index].imageOffset.x);
    EXPECT_EQ(rewritten[index].imageExtent.width,
              plan.rewritten[index].imageExtent.width);
    EXPECT_EQ(rewritten[index].imageExtent.height,
              plan.rewritten[index].imageExtent.height);
  }
  // Row length 24 (6 blocks of 16 bytes) and image height 16 (4 block rows).
  EXPECT_EQ(uploads[0].source_row_stride, 96U);
  EXPECT_EQ(uploads[0].source_layer_stride, 384U);
  EXPECT_EQ(uploads[0].source_span, 2U * 96U + 5U * 16U);
}

}  // namespace
}  // namespace mocktail::graphics
