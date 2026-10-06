#include "mocktail/graphics/vulkan_etc2_gpu_plan.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "mocktail/graphics/bc_encoder.h"

// The GPU route's pure planning (DESIGN §9.1): the rewritten regions and
// output layout of a copy (PlanGpuRegions), the decode pass of a batch
// (BuildDecodePass: input copies, the shader's job words, dispatches) and
// the rewritten submit arrays (RewriteSubmits, RewriteSubmits2). The
// expectations are worked out by hand from the Vulkan copy rules and the
// shaders' Job layout in src/graphics/shaders/etc2_decode.comp and
// etc2_bc.comp.

namespace mocktail::graphics {
namespace {

const VkBuffer kSource = reinterpret_cast<VkBuffer>(0xb0);
const VkBuffer kSource2 = reinterpret_cast<VkBuffer>(0xb8);
const VkBuffer kOutput = reinterpret_cast<VkBuffer>(0xc0);
const VkBuffer kOutput2 = reinterpret_cast<VkBuffer>(0xc8);

template <typename Handle>
Handle FakeHandle(std::uintptr_t value) {
  return reinterpret_cast<Handle>(value);
}

Etc2ImageShape Shape(VkFormat format, std::uint32_t width,
                     std::uint32_t height, std::uint32_t scale = 1) {
  Etc2ImageShape shape;
  EXPECT_TRUE(LookupEmulatedEtc2Format(format, &shape.format));
  shape.scale = scale;
  shape.width = width;
  shape.height = height;
  return shape;
}

VkBufferImageCopy Region(VkDeviceSize buffer_offset, std::uint32_t row_length,
                         std::uint32_t image_height, std::uint32_t mip,
                         std::uint32_t layers, std::int32_t x, std::int32_t y,
                         std::uint32_t width, std::uint32_t height) {
  VkBufferImageCopy region{};
  region.bufferOffset = buffer_offset;
  region.bufferRowLength = row_length;
  region.bufferImageHeight = image_height;
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, layers};
  region.imageOffset = {x, y, 0};
  region.imageExtent = {width, height, 1};
  return region;
}

struct GpuPlan {
  bool ok = false;
  std::vector<VkBufferImageCopy> rewritten;
  std::vector<Etc2GpuUpload> uploads;
  VkDeviceSize total = 0;
};

GpuPlan PlanAll(const Etc2ImageShape& shape,
                const std::vector<VkBufferImageCopy>& regions,
                bool bc = false) {
  GpuPlan plan;
  plan.ok = PlanGpuRegions(shape, kSource,
                           static_cast<std::uint32_t>(regions.size()),
                           regions.data(), bc, &plan.rewritten, &plan.uploads,
                           &plan.total);
  return plan;
}

TEST(PlanGpuRegionsTest, PadsRowsToWholeBlocksAndAlignsRegions) {
  // A 37x37 RGB8 image: mip 0 is 10x10 blocks, mip 1 (18x18) 5x5 blocks.
  const Etc2ImageShape shape =
      Shape(VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, 37, 37);
  const GpuPlan plan = PlanAll(
      shape, {Region(0, 0, 0, 0, 1, 0, 0, 37, 37),
              Region(800, 0, 0, 1, 1, 0, 0, 18, 18)});
  ASSERT_TRUE(plan.ok);
  ASSERT_EQ(plan.uploads.size(), 2U);

  const Etc2GpuUpload& mip0 = plan.uploads[0];
  EXPECT_EQ(mip0.source, kSource);
  EXPECT_EQ(mip0.source_offset, 0U);
  EXPECT_EQ(mip0.blocks_wide, 10U);
  EXPECT_EQ(mip0.blocks_high, 10U);
  EXPECT_EQ(mip0.layers, 1U);
  EXPECT_EQ(mip0.source_row_stride, 80U);  // 10 blocks of 8 bytes
  EXPECT_EQ(mip0.source_layer_stride, 800U);
  EXPECT_EQ(mip0.source_span, 800U);
  EXPECT_EQ(mip0.compressed, 800U);
  EXPECT_EQ(mip0.output_offset, 0U);
  EXPECT_EQ(mip0.output_bytes, 40U * 40U * 4U);  // padded to 40x40 texels
  EXPECT_EQ(mip0.decoded, 37U * 37U * 4U);       // what the copy writes
  EXPECT_FALSE(mip0.bc);

  const Etc2GpuUpload& mip1 = plan.uploads[1];
  EXPECT_EQ(mip1.source_offset, 800U);
  EXPECT_EQ(mip1.mip_level, 1U);
  EXPECT_EQ(mip1.blocks_wide, 5U);
  // 6400 bytes of mip 0, rounded up to 256.
  EXPECT_EQ(mip1.output_offset, 6400U);
  EXPECT_EQ(mip1.output_bytes, 20U * 20U * 4U);
  EXPECT_EQ(plan.total, 6400U + 1600U);

  // The rewritten copies read the padded rows; the image side is as given.
  EXPECT_EQ(plan.rewritten[0].bufferOffset, 0U);
  EXPECT_EQ(plan.rewritten[0].bufferRowLength, 40U);
  EXPECT_EQ(plan.rewritten[0].bufferImageHeight, 40U);
  EXPECT_EQ(plan.rewritten[0].imageExtent.width, 37U);
  EXPECT_EQ(plan.rewritten[0].imageExtent.height, 37U);
  EXPECT_EQ(plan.rewritten[1].bufferOffset, 6400U);
  EXPECT_EQ(plan.rewritten[1].bufferRowLength, 20U);
  EXPECT_EQ(plan.rewritten[1].bufferImageHeight, 20U);
  EXPECT_EQ(plan.rewritten[1].imageSubresource.mipLevel, 1U);
  for (const VkBufferImageCopy& region : plan.rewritten) {
    EXPECT_EQ(region.bufferOffset % kEtc2GpuRegionAlignment, 0U);
  }
}

TEST(PlanGpuRegionsTest, PlansRowStridesLayersAndTwoByteTexels) {
  // EAC R11 decodes to R16: 2 bytes per texel. Roblox's rows hold 12
  // blocks of which the region reads 3, and layers are 6 block rows apart.
  const Etc2ImageShape shape = Shape(VK_FORMAT_EAC_R11_UNORM_BLOCK, 64, 64);
  const GpuPlan plan =
      PlanAll(shape, {Region(16, 48, 24, 0, 3, 4, 8, 10, 7)});
  ASSERT_TRUE(plan.ok);
  const Etc2GpuUpload& upload = plan.uploads[0];
  EXPECT_EQ(upload.blocks_wide, 3U);
  EXPECT_EQ(upload.blocks_high, 2U);
  EXPECT_EQ(upload.layers, 3U);
  EXPECT_EQ(upload.source_row_stride, 12U * 8U);
  EXPECT_EQ(upload.source_layer_stride, 12U * 8U * 6U);
  // Two whole layers, one block row, then 3 blocks.
  EXPECT_EQ(upload.source_span, 2U * 576U + 96U + 24U);
  EXPECT_EQ(upload.compressed, 3U * 2U * 8U * 3U);
  EXPECT_EQ(upload.output_bytes, 12U * 8U * 2U * 3U);
  EXPECT_EQ(upload.decoded, 10U * 7U * 2U * 3U);
  EXPECT_EQ(plan.rewritten[0].bufferRowLength, 12U);
  EXPECT_EQ(plan.rewritten[0].bufferImageHeight, 8U);
  EXPECT_EQ(plan.rewritten[0].imageOffset.x, 4);
  EXPECT_EQ(plan.rewritten[0].imageOffset.y, 8);
  EXPECT_EQ(plan.rewritten[0].imageSubresource.layerCount, 3U);
}

TEST(PlanGpuRegionsTest, ClampsTheCopyToTheMipButDecodesWholeBlocks) {
  // A 6x6 image: mip 1 is 3x3, but this region asks for a whole 4x4 block
  // at (0, 0). The decode covers the block; the copy stops at the mip edge.
  const Etc2ImageShape shape =
      Shape(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 6, 6);
  const GpuPlan plan = PlanAll(shape, {Region(0, 0, 0, 1, 1, 0, 0, 4, 4)});
  ASSERT_TRUE(plan.ok);
  EXPECT_EQ(plan.uploads[0].blocks_wide, 1U);
  EXPECT_EQ(plan.uploads[0].output_bytes, 16U * 4U);
  EXPECT_EQ(plan.rewritten[0].imageExtent.width, 3U);
  EXPECT_EQ(plan.rewritten[0].imageExtent.height, 3U);
  EXPECT_EQ(plan.rewritten[0].bufferRowLength, 4U);
  EXPECT_EQ(plan.uploads[0].decoded, 3U * 3U * 4U);
}

TEST(PlanGpuRegionsTest, BcModeKeepsTightBlocks) {
  // In BC mode the output is the BC blocks themselves: 8 bytes per block
  // for RGB8 (BC1), 16 for RGBA8 (BC3).
  const Etc2ImageShape shape =
      Shape(VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, 20, 12);
  const GpuPlan plan =
      PlanAll(shape, {Region(0, 0, 0, 0, 1, 0, 0, 20, 12)}, /*bc=*/true);
  ASSERT_TRUE(plan.ok);
  EXPECT_TRUE(plan.uploads[0].bc);
  EXPECT_EQ(plan.uploads[0].output_bytes, 5U * 3U * 16U);
  // The host image receives the blocks: the CPU route counts the same.
  EXPECT_EQ(plan.uploads[0].decoded, plan.uploads[0].compressed);
  // Whole blocks per row and layer: the same as the tight layout.
  EXPECT_EQ(plan.rewritten[0].bufferRowLength, 20U);
  EXPECT_EQ(plan.rewritten[0].bufferImageHeight, 12U);
}

TEST(PlanGpuRegionsTest, BcModeClipsBlockRoundedExtentsToTheMip) {
  // A 6x6 image: mip 1 is 3x3, but the region asks for a whole 4x4 block.
  // A BC host takes only an extent that is whole blocks or reaches the
  // mip's edge (VUID-vkCmdCopyBufferToImage-imageSubresource-07971): the
  // copy stops at 3x3 and reads the one BC block of its padded row.
  const Etc2ImageShape shape = Shape(VK_FORMAT_EAC_R11G11_SNORM_BLOCK, 6, 6);
  const GpuPlan plan =
      PlanAll(shape, {Region(0, 0, 0, 1, 1, 0, 0, 4, 4)}, /*bc=*/true);
  ASSERT_TRUE(plan.ok);
  EXPECT_EQ(plan.uploads[0].blocks_wide, 1U);
  EXPECT_EQ(plan.uploads[0].output_bytes, 16U);
  EXPECT_EQ(plan.uploads[0].decoded, 16U);
  EXPECT_EQ(plan.rewritten[0].imageExtent.width, 3U);
  EXPECT_EQ(plan.rewritten[0].imageExtent.height, 3U);
  EXPECT_EQ(plan.rewritten[0].bufferRowLength, 4U);
  EXPECT_EQ(plan.rewritten[0].bufferImageHeight, 4U);
  EXPECT_EQ(plan.rewritten[0].bufferOffset % 16, 0U);
}

TEST(PlanGpuRegionsTest, RejectsRegionsItCannotServe) {
  const Etc2ImageShape shape = Shape(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 32, 32);
  // Upscaled images stay on the CPU route.
  EXPECT_FALSE(PlanAll(Shape(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 32, 32, 4),
                       {Region(0, 0, 0, 0, 1, 0, 0, 32, 32)})
                   .ok);
  EXPECT_FALSE(PlanAll(shape, {Region(0, 0, 0, 0, 1, 0, 0, 0, 4)}).ok);
  EXPECT_FALSE(PlanAll(shape, {Region(0, 0, 0, 0, 0, 0, 0, 4, 4)}).ok);
  EXPECT_FALSE(PlanAll(shape, {Region(0, 4, 0, 0, 1, 0, 0, 8, 4)}).ok)
      << "row length shorter than the region";
  EXPECT_FALSE(PlanAll(shape, {Region(0, 0, 0, 1, 1, 16, 0, 4, 4)}).ok)
      << "offset outside the 16x16 mip";
  EXPECT_FALSE(PlanAll(shape, {Region(0, 0, 0, 0, 1, -4, 0, 4, 4)}).ok);
  VkBufferImageCopy deep = Region(0, 0, 0, 0, 1, 0, 0, 4, 4);
  deep.imageExtent.depth = 2;
  EXPECT_FALSE(PlanAll(shape, {deep}).ok);
  std::vector<VkBufferImageCopy> rewritten;
  std::vector<Etc2GpuUpload> uploads;
  VkDeviceSize total = 1;
  EXPECT_FALSE(PlanGpuRegions<VkBufferImageCopy>(shape, kSource, 0, nullptr,
                                                 false, &rewritten, &uploads,
                                                 &total));
  EXPECT_EQ(total, 0U);
}

TEST(PlanGpuRegionsTest, CopyCommands2KeepTheirChain) {
  const Etc2ImageShape shape = Shape(VK_FORMAT_EAC_R11G11_SNORM_BLOCK, 8, 8);
  VkBufferImageCopy2 region{};
  region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
  int marker = 0;
  region.pNext = &marker;
  region.bufferOffset = 32;
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {8, 8, 1};
  std::vector<VkBufferImageCopy2> rewritten;
  std::vector<Etc2GpuUpload> uploads;
  VkDeviceSize total = 0;
  ASSERT_TRUE(PlanGpuRegions(shape, kSource, 1, &region, false, &rewritten,
                             &uploads, &total));
  EXPECT_EQ(rewritten[0].sType, VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2);
  EXPECT_EQ(rewritten[0].pNext, &marker);
  EXPECT_EQ(rewritten[0].bufferOffset, 0U);
  EXPECT_EQ(rewritten[0].bufferRowLength, 8U);
  EXPECT_EQ(uploads[0].source_offset, 32U);
  EXPECT_EQ(uploads[0].output_bytes, 8U * 8U * 4U);  // RG16 per texel
}

Etc2GpuUpload Upload(EtcFormat format, std::uint32_t blocks_wide,
                     std::uint32_t blocks_high, std::uint32_t layers,
                     VkDeviceSize source_offset, VkDeviceSize row_stride,
                     VkDeviceSize layer_stride, VkBuffer output,
                     VkDeviceSize output_offset, VkBuffer source = kSource) {
  Etc2GpuUpload upload;
  upload.source = source;
  upload.format = format;
  upload.blocks_wide = blocks_wide;
  upload.blocks_high = blocks_high;
  upload.layers = layers;
  const VkDeviceSize block = EtcBlockBytes(format);
  upload.source_offset = source_offset;
  upload.source_row_stride = row_stride;
  upload.source_layer_stride = layer_stride;
  upload.source_span = (layers - 1) * layer_stride +
                       (blocks_high - 1) * row_stride + blocks_wide * block;
  upload.compressed =
      static_cast<VkDeviceSize>(blocks_wide) * blocks_high * layers * block;
  upload.output = output;
  upload.output_offset = output_offset;
  upload.output_bytes = static_cast<VkDeviceSize>(blocks_wide) * blocks_high *
                        layers * 16 * EtcDecodedTexelBytes(format);
  upload.decoded = upload.output_bytes;
  return upload;
}

TEST(BuildDecodePassTest, WritesTheShadersJobWords) {
  // RGBA8 (16-byte blocks), 3x2 blocks, 2 layers, Roblox rows 4 blocks
  // apart and layers 3 block rows apart: the 304-byte span is within twice
  // the 192 compressed bytes, so it is copied whole, in one region.
  const Etc2GpuUpload upload =
      Upload(EtcFormat::kEtc2Rgba8, 3, 2, 2, 64, 64, 192, kOutput, 512);
  ASSERT_EQ(upload.source_span, 192U + 64U + 48U);
  ASSERT_EQ(upload.compressed, 192U);
  Etc2DecodePass pass;
  ASSERT_TRUE(BuildDecodePass({upload}, 1000, 5, &pass));
  // Input from 1000 rounded up to 16 bytes.
  ASSERT_EQ(pass.copies.size(), 1U);
  EXPECT_EQ(pass.copies[0].source, kSource);
  ASSERT_EQ(pass.copies[0].regions.size(), 1U);
  EXPECT_EQ(pass.copies[0].regions[0].srcOffset, 64U);
  EXPECT_EQ(pass.copies[0].regions[0].dstOffset, 1008U);
  EXPECT_EQ(pass.copies[0].regions[0].size, upload.source_span);
  EXPECT_EQ(pass.input_bytes, 304U);  // the span, a multiple of 16 already
  ASSERT_EQ(pass.jobs.size(), 1U);
  const Etc2DecodeJob& job = pass.jobs[0];
  EXPECT_EQ(job.format, static_cast<std::uint32_t>(EtcFormat::kEtc2Rgba8));
  EXPECT_EQ(job.src_word, 1008U / 4);
  EXPECT_EQ(job.src_row_words, 64U / 4);
  EXPECT_EQ(job.src_layer_words, 192U / 4);
  EXPECT_EQ(job.blocks_wide, 3U);
  EXPECT_EQ(job.blocks_high, 2U);
  EXPECT_EQ(job.block_count, 12U);
  EXPECT_EQ(job.dst_word, 512U / 4);
  EXPECT_EQ(job.dst_row_words, 12U);        // 12 texels of 4 bytes
  EXPECT_EQ(job.dst_layer_words, 12U * 8);  // 8 texel rows
  EXPECT_EQ(job.group_begin, 0U);
  ASSERT_EQ(pass.dispatches.size(), 1U);
  EXPECT_EQ(pass.dispatches[0].output, kOutput);
  EXPECT_EQ(pass.dispatches[0].job_first, 5U);
  EXPECT_EQ(pass.dispatches[0].job_count, 1U);
  EXPECT_EQ(pass.dispatches[0].groups_x, 1U);
  EXPECT_EQ(pass.dispatches[0].groups_y, 1U);
  EXPECT_EQ(pass.blocks, 12U);
  EXPECT_EQ(Etc2DecodeInputBytes({upload}), 304U);
}

TEST(BuildDecodePassTest, CopiesSparseRowsOneByOneAndPacksThem) {
  // R11 (8-byte blocks), 2x3 blocks: Roblox's rows are 100 blocks apart,
  // far more than twice the 48 compressed bytes, so each block row is
  // copied and the job reads them packed.
  const Etc2GpuUpload upload =
      Upload(EtcFormat::kEacR11, 2, 3, 1, 8, 800, 2400, kOutput, 0);
  Etc2DecodePass pass;
  ASSERT_TRUE(BuildDecodePass({upload}, 0, 0, &pass));
  ASSERT_EQ(pass.copies[0].regions.size(), 3U);
  for (std::uint32_t row = 0; row < 3; ++row) {
    EXPECT_EQ(pass.copies[0].regions[row].srcOffset, 8U + row * 800U);
    EXPECT_EQ(pass.copies[0].regions[row].dstOffset, row * 16U);
    EXPECT_EQ(pass.copies[0].regions[row].size, 16U);
  }
  EXPECT_EQ(pass.input_bytes, 48U);
  EXPECT_EQ(pass.jobs[0].src_row_words, 4U);
  EXPECT_EQ(pass.jobs[0].src_layer_words, 12U);
  EXPECT_EQ(pass.jobs[0].dst_row_words, 2U * 4U * 2U / 4U);  // 8 R16 texels
  EXPECT_EQ(Etc2DecodeInputBytes({upload}), 48U);
}

TEST(BuildDecodePassTest, MergesContiguousSourcesAndGroupsByOutput) {
  // Three uploads: the first two read neighbouring spans of one buffer
  // (one copy region), the third another buffer. Outputs alternate between
  // two arena buffers, so the jobs regroup into two dispatches.
  std::vector<Etc2GpuUpload> uploads = {
      Upload(EtcFormat::kEtc2Rgb8, 2, 2, 1, 0, 16, 32, kOutput, 0),
      Upload(EtcFormat::kEtc2Rgb8, 2, 2, 1, 32, 16, 32, kOutput2, 256),
      Upload(EtcFormat::kEtc2Rgb8, 20, 20, 1, 0, 160, 3200, kOutput, 256,
             kSource2),
  };
  Etc2DecodePass pass;
  ASSERT_TRUE(BuildDecodePass(uploads, 0, 0, &pass));
  ASSERT_EQ(pass.copies.size(), 2U);
  ASSERT_EQ(pass.copies[0].regions.size(), 1U);
  EXPECT_EQ(pass.copies[0].regions[0].srcOffset, 0U);
  EXPECT_EQ(pass.copies[0].regions[0].size, 64U);
  EXPECT_EQ(pass.copies[1].source, kSource2);
  EXPECT_EQ(pass.copies[1].regions[0].dstOffset, 64U);
  ASSERT_EQ(pass.dispatches.size(), 2U);
  EXPECT_EQ(pass.dispatches[0].output, kOutput < kOutput2 ? kOutput : kOutput2);
  // Jobs in output order: kOutput's first (uploads 0 and 2), then kOutput2.
  ASSERT_EQ(pass.jobs.size(), 3U);
  EXPECT_EQ(pass.dispatches[0].job_first, 0U);
  EXPECT_EQ(pass.dispatches[0].job_count, 2U);
  EXPECT_EQ(pass.dispatches[1].job_first, 2U);
  EXPECT_EQ(pass.dispatches[1].job_count, 1U);
  // group_begin restarts in each dispatch: 4 blocks (1 group), then 400
  // blocks (7 groups).
  EXPECT_EQ(pass.jobs[0].group_begin, 0U);
  EXPECT_EQ(pass.jobs[1].group_begin, 1U);
  EXPECT_EQ(pass.dispatches[0].groups_x, 8U);
  EXPECT_EQ(pass.jobs[2].group_begin, 0U);
  EXPECT_EQ(pass.dispatches[1].groups_x, 1U);
  EXPECT_EQ(pass.jobs[1].src_word, 64U / 4);
  EXPECT_EQ(pass.jobs[2].src_word, 32U / 4);
}

TEST(BuildDecodePassTest, SplitsLargeDispatchesAt65535Groups) {
  // 2048 x 2050 blocks = 65600 groups of 64.
  const Etc2GpuUpload upload =
      Upload(EtcFormat::kEtc2Rgb8, 2048, 2050, 1, 0, 2048 * 8, 2048 * 8 * 2050,
             kOutput, 0);
  Etc2DecodePass pass;
  ASSERT_TRUE(BuildDecodePass({upload}, 0, 0, &pass));
  EXPECT_EQ(pass.dispatches[0].groups_x, 65535U);
  EXPECT_EQ(pass.dispatches[0].groups_y, 2U);
  std::uint32_t x = 0;
  std::uint32_t y = 0;
  Etc2SplitDispatch(65535, &x, &y);
  EXPECT_EQ(x, 65535U);
  EXPECT_EQ(y, 1U);
  Etc2SplitDispatch(0, &x, &y);
  EXPECT_EQ(x, 0U);
  EXPECT_EQ(y, 0U);
  EXPECT_EQ(Etc2DecodeGroups(65), 2U);
}

TEST(BuildDecodePassTest, BcUploadsGetTheirOwnRefinedDispatches) {
  // ETC2->BC mode: a BC image's upload (tight blocks) and an RGBA fallback
  // image's (CR-14), recorded in that order into one arena buffer. The
  // decode dispatch comes first, the etc2_bc one after it, each with its
  // own group_begin; the BC job refines, as the CPU route does, and
  // writes block rows (2 words per BC1 block).
  Etc2GpuUpload bc =
      Upload(EtcFormat::kEtc2Rgb8A1, 5, 3, 2, 0, 40, 120, kOutput, 0);
  bc.bc = true;
  bc.output_bytes = bc.compressed;
  const Etc2GpuUpload rgba =
      Upload(EtcFormat::kEtc2Rgba8, 2, 2, 1, 240, 32, 64, kOutput, 256);
  Etc2DecodePass pass;
  ASSERT_TRUE(BuildDecodePass({bc, rgba}, 0, 0, &pass));
  ASSERT_EQ(pass.dispatches.size(), 2U);
  EXPECT_FALSE(pass.dispatches[0].bc);
  EXPECT_TRUE(pass.dispatches[1].bc);
  EXPECT_EQ(pass.dispatches[0].output, kOutput);
  EXPECT_EQ(pass.dispatches[1].output, kOutput);
  EXPECT_EQ(pass.dispatches[0].job_first, 0U);
  EXPECT_EQ(pass.dispatches[1].job_first, 1U);
  ASSERT_EQ(pass.jobs.size(), 2U);
  const Etc2DecodeJob& decode = pass.jobs[0];
  EXPECT_EQ(decode.format, static_cast<std::uint32_t>(EtcFormat::kEtc2Rgba8));
  EXPECT_EQ(decode.flags, 0U);
  EXPECT_EQ(decode.dst_word, 256U / 4);
  const Etc2DecodeJob& transcode = pass.jobs[1];
  EXPECT_EQ(transcode.format,
            static_cast<std::uint32_t>(EtcFormat::kEtc2Rgb8A1));
  EXPECT_EQ(transcode.flags, kEtc2ToBcRefine ? kEtc2BcJobRefine : 0U);
  EXPECT_EQ(transcode.group_begin, 0U);
  EXPECT_EQ(transcode.block_count, 30U);
  EXPECT_EQ(transcode.dst_word, 0U);
  EXPECT_EQ(transcode.dst_row_words, 5U * 2U);
  EXPECT_EQ(transcode.dst_layer_words, 5U * 2U * 3U);
  // The input still follows the recording order: the BC upload's span
  // first, the fallback's after it, in one merged copy region.
  EXPECT_EQ(transcode.src_word, 0U);
  EXPECT_EQ(decode.src_word, 240U / 4);
  ASSERT_EQ(pass.copies.size(), 1U);
  ASSERT_EQ(pass.copies[0].regions.size(), 1U);
  EXPECT_EQ(pass.copies[0].regions[0].size, 240U + 64U);
}

TEST(BuildDecodePassTest, RejectsUploadsWithoutOutput) {
  Etc2GpuUpload upload =
      Upload(EtcFormat::kEtc2Rgb8, 1, 1, 1, 0, 8, 8, VK_NULL_HANDLE, 0);
  Etc2DecodePass pass;
  EXPECT_FALSE(BuildDecodePass({upload}, 0, 0, &pass));
  upload.output = kOutput;
  upload.layers = 0;
  EXPECT_FALSE(BuildDecodePass({upload}, 0, 0, &pass));
}

// ---- submit rewriting -------------------------------------------------------

const VkCommandBuffer kRoblox[4] = {
    FakeHandle<VkCommandBuffer>(0x101), FakeHandle<VkCommandBuffer>(0x102),
    FakeHandle<VkCommandBuffer>(0x103), FakeHandle<VkCommandBuffer>(0x104)};
const VkCommandBuffer kPass[3] = {FakeHandle<VkCommandBuffer>(0x201),
                                  FakeHandle<VkCommandBuffer>(0x202),
                                  FakeHandle<VkCommandBuffer>(0x203)};
const VkFence kAppFence = FakeHandle<VkFence>(0x301);
const VkFence kSlotFence = FakeHandle<VkFence>(0x302);
const VkSemaphore kSemaphore = FakeHandle<VkSemaphore>(0x401);

TEST(RewriteSubmitsTest, PrependsPassesAndLeavesOtherBatchesAlone) {
  const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo submits[3]{};
  for (VkSubmitInfo& submit : submits) {
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  }
  submits[0].commandBufferCount = 2;
  submits[0].pCommandBuffers = kRoblox;
  submits[0].waitSemaphoreCount = 1;
  submits[0].pWaitSemaphores = &kSemaphore;
  submits[0].pWaitDstStageMask = &wait_stage;
  submits[1].commandBufferCount = 1;
  submits[1].pCommandBuffers = kRoblox + 2;
  submits[2].commandBufferCount = 1;
  submits[2].pCommandBuffers = kRoblox + 3;
  Etc2SubmitPlan plan;
  EXPECT_EQ(RewriteSubmits(3, submits, {{0, kPass[0]}, {2, kPass[1]}},
                           VK_NULL_HANDLE, kSlotFence, &plan),
            0U);
  ASSERT_TRUE(plan.modified);
  ASSERT_EQ(plan.count(3), 3U);
  const VkSubmitInfo* rewritten = plan.submits(submits);
  ASSERT_NE(rewritten, submits);
  ASSERT_EQ(rewritten[0].commandBufferCount, 3U);
  EXPECT_EQ(rewritten[0].pCommandBuffers[0], kPass[0]);
  EXPECT_EQ(rewritten[0].pCommandBuffers[1], kRoblox[0]);
  EXPECT_EQ(rewritten[0].pCommandBuffers[2], kRoblox[1]);
  EXPECT_EQ(rewritten[0].pWaitSemaphores, &kSemaphore) << "waits stay";
  EXPECT_EQ(rewritten[0].pWaitDstStageMask, &wait_stage);
  // The middle batch is untouched, to the pointer.
  EXPECT_EQ(rewritten[1].commandBufferCount, 1U);
  EXPECT_EQ(rewritten[1].pCommandBuffers, kRoblox + 2);
  ASSERT_EQ(rewritten[2].commandBufferCount, 2U);
  EXPECT_EQ(rewritten[2].pCommandBuffers[0], kPass[1]);
  EXPECT_EQ(rewritten[2].pCommandBuffers[1], kRoblox[3]);
  // No app fence: the slot's goes with the call.
  EXPECT_EQ(plan.fence(VK_NULL_HANDLE), kSlotFence);
  EXPECT_EQ(plan.trailing_fence, VK_NULL_HANDLE);
  EXPECT_TRUE(plan.pre_submits().empty());
}

TEST(RewriteSubmitsTest, AnAppFenceStaysAndTheSlotFenceTrails) {
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = kRoblox;
  Etc2SubmitPlan plan;
  RewriteSubmits(1, &submit, {{0, kPass[0]}}, kAppFence, kSlotFence, &plan);
  EXPECT_EQ(plan.fence(kAppFence), kAppFence);
  EXPECT_EQ(plan.trailing_fence, kSlotFence);
}

TEST(RewriteSubmitsTest, ProtectedAndDeviceGroupBatchesGetAPreSubmit) {
  VkProtectedSubmitInfo protected_info{};
  protected_info.sType = VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO;
  protected_info.protectedSubmit = VK_TRUE;
  VkDeviceGroupSubmitInfo group{};
  group.sType = VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO;
  VkSubmitInfo submits[3]{};
  for (std::uint32_t index = 0; index < 3; ++index) {
    submits[index].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submits[index].commandBufferCount = 1;
    submits[index].pCommandBuffers = kRoblox + index;
  }
  submits[0].pNext = &protected_info;
  submits[1].pNext = &group;
  EXPECT_TRUE(Etc2BatchNeedsPreSubmit(submits[0]));
  EXPECT_TRUE(Etc2BatchNeedsPreSubmit(submits[1]));
  EXPECT_FALSE(Etc2BatchNeedsPreSubmit(submits[2]));
  protected_info.protectedSubmit = VK_FALSE;
  EXPECT_FALSE(Etc2BatchNeedsPreSubmit(submits[0]));
  protected_info.protectedSubmit = VK_TRUE;

  Etc2SubmitPlan plan;
  EXPECT_EQ(RewriteSubmits(3, submits,
                           {{0, kPass[0]}, {1, kPass[1]}, {2, kPass[2]}},
                           VK_NULL_HANDLE, kSlotFence, &plan),
            2U);
  ASSERT_EQ(plan.pre_submits().size(), 2U);
  for (std::uint32_t index = 0; index < 2; ++index) {
    const VkSubmitInfo& pre = plan.pre_submits()[index].info;
    EXPECT_EQ(pre.sType, VK_STRUCTURE_TYPE_SUBMIT_INFO);
    EXPECT_EQ(pre.pNext, nullptr);
    EXPECT_EQ(pre.waitSemaphoreCount, 0U);
    EXPECT_EQ(pre.signalSemaphoreCount, 0U);
    ASSERT_EQ(pre.commandBufferCount, 1U);
    EXPECT_EQ(pre.pCommandBuffers[0], kPass[index]);
    EXPECT_EQ(plan.pre_submits()[index].fence, VK_NULL_HANDLE);
  }
  const VkSubmitInfo* rewritten = plan.submits(submits);
  EXPECT_EQ(rewritten[0].pCommandBuffers, kRoblox) << "protected stays";
  EXPECT_EQ(rewritten[1].pCommandBuffers, kRoblox + 1);
  ASSERT_EQ(rewritten[2].commandBufferCount, 2U);
  EXPECT_EQ(rewritten[2].pCommandBuffers[0], kPass[2]);
  EXPECT_EQ(plan.fence(VK_NULL_HANDLE), kSlotFence);

  // Only pre-submits: Roblox's arrays go to the host as they were.
  Etc2SubmitPlan only;
  EXPECT_EQ(RewriteSubmits(1, submits, {{0, kPass[0]}}, VK_NULL_HANDLE,
                           kSlotFence, &only),
            1U);
  EXPECT_FALSE(only.modified);
  EXPECT_EQ(only.submits(submits), submits);
  EXPECT_EQ(only.count(1), 1U);
  EXPECT_EQ(only.fence(VK_NULL_HANDLE), kSlotFence);
}

TEST(RewriteSubmitsTest, NothingToDoLeavesThePlanAlone) {
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  Etc2SubmitPlan plan;
  EXPECT_EQ(RewriteSubmits(1, &submit, {{5, kPass[0]}}, VK_NULL_HANDLE,
                           kSlotFence, &plan),
            0U);
  EXPECT_FALSE(plan.modified);
  EXPECT_EQ(plan.fence(VK_NULL_HANDLE), VK_NULL_HANDLE);
  EXPECT_EQ(RewriteSubmits(1, &submit, {}, VK_NULL_HANDLE, kSlotFence, &plan),
            0U);
  EXPECT_EQ(plan.submits(&submit), &submit);
}

TEST(RewriteSubmits2Test, PrependsACommandBufferInfo) {
  VkCommandBufferSubmitInfo infos[2]{};
  for (std::uint32_t index = 0; index < 2; ++index) {
    infos[index].sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    infos[index].commandBuffer = kRoblox[index];
    infos[index].deviceMask = 1;
  }
  VkSemaphoreSubmitInfo wait{};
  wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
  wait.semaphore = kSemaphore;
  wait.stageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
  VkSubmitInfo2 submits[2]{};
  submits[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
  submits[0].commandBufferInfoCount = 2;
  submits[0].pCommandBufferInfos = infos;
  submits[0].waitSemaphoreInfoCount = 1;
  submits[0].pWaitSemaphoreInfos = &wait;
  submits[1] = submits[0];
  submits[1].flags = VK_SUBMIT_PROTECTED_BIT;
  EXPECT_FALSE(Etc2BatchNeedsPreSubmit2(submits[0]));
  EXPECT_TRUE(Etc2BatchNeedsPreSubmit2(submits[1]));
  Etc2SubmitPlan plan;
  EXPECT_EQ(RewriteSubmits2(2, submits, {{0, kPass[0]}, {1, kPass[1]}},
                            kAppFence, kSlotFence, &plan),
            1U);
  ASSERT_TRUE(plan.modified);
  ASSERT_EQ(plan.count(2), 2U);
  const VkSubmitInfo2* rewritten = plan.submits2(submits);
  ASSERT_EQ(rewritten[0].commandBufferInfoCount, 3U);
  EXPECT_EQ(rewritten[0].pCommandBufferInfos[0].sType,
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO);
  EXPECT_EQ(rewritten[0].pCommandBufferInfos[0].commandBuffer, kPass[0]);
  EXPECT_EQ(rewritten[0].pCommandBufferInfos[0].deviceMask, 0U);
  EXPECT_EQ(rewritten[0].pCommandBufferInfos[1].commandBuffer, kRoblox[0]);
  EXPECT_EQ(rewritten[0].pCommandBufferInfos[2].deviceMask, 1U);
  EXPECT_EQ(rewritten[0].pWaitSemaphoreInfos, &wait);
  EXPECT_EQ(rewritten[1].pCommandBufferInfos, infos) << "protected stays";
  ASSERT_EQ(plan.pre_submits().size(), 1U);
  EXPECT_EQ(plan.pre_submits()[0].info.pCommandBuffers[0], kPass[1]);
  EXPECT_EQ(plan.trailing_fence, kSlotFence);
  EXPECT_EQ(plan.fence(kAppFence), kAppFence);
}

}  // namespace
}  // namespace mocktail::graphics
