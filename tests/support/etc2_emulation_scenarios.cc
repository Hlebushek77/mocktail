#include "support/etc2_emulation_scenarios.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <utility>

#include "support/etc2_reference_decoder.h"
#include "support/resample_reference.h"

namespace mocktail::graphics::testing {
namespace {

struct EmulatedFormat {
  VkFormat host = VK_FORMAT_UNDEFINED;
  EtcFormat etc = EtcFormat::kEtc2Rgb8;
  bool color = false;
};

// The ETC2/EAC formats a device without ETC2 gets, and what the emulation
// stores them as.
bool LookupEmulated(VkFormat format, EmulatedFormat* out) {
  switch (format) {
    case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
      *out = {VK_FORMAT_R8G8B8A8_UNORM, EtcFormat::kEtc2Rgb8, true};
      return true;
    case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
      *out = {VK_FORMAT_R8G8B8A8_SRGB, EtcFormat::kEtc2Rgb8, true};
      return true;
    case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
      *out = {VK_FORMAT_R8G8B8A8_UNORM, EtcFormat::kEtc2Rgb8A1, true};
      return true;
    case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
      *out = {VK_FORMAT_R8G8B8A8_SRGB, EtcFormat::kEtc2Rgb8A1, true};
      return true;
    case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
      *out = {VK_FORMAT_R8G8B8A8_UNORM, EtcFormat::kEtc2Rgba8, true};
      return true;
    case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
      *out = {VK_FORMAT_R8G8B8A8_SRGB, EtcFormat::kEtc2Rgba8, true};
      return true;
    case VK_FORMAT_EAC_R11_UNORM_BLOCK:
      *out = {VK_FORMAT_R16_UNORM, EtcFormat::kEacR11, false};
      return true;
    case VK_FORMAT_EAC_R11_SNORM_BLOCK:
      *out = {VK_FORMAT_R16_SNORM, EtcFormat::kEacR11Signed, false};
      return true;
    case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
      *out = {VK_FORMAT_R16G16_UNORM, EtcFormat::kEacRg11, false};
      return true;
    case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
      *out = {VK_FORMAT_R16G16_SNORM, EtcFormat::kEacRg11Signed, false};
      return true;
    default:
      return false;
  }
}

std::uint32_t Mip(std::uint32_t base, std::uint32_t mip) {
  return std::max<std::uint32_t>(1, base >> mip);
}

std::uint32_t DivideUp(std::uint32_t value, std::uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

FormatBlock BlockOf(VkFormat format) {
  FormatBlock block;
  LookupFormatBlock(format, &block);
  return block;
}

// Bytes a buffer<->image region spans in the buffer, from bufferOffset.
VkDeviceSize RegionSpan(const FormatBlock& block,
                        const VkBufferImageCopy& region) {
  const std::uint64_t row_blocks =
      DivideUp(region.bufferRowLength != 0 ? region.bufferRowLength
                                           : region.imageExtent.width,
               block.width);
  const std::uint64_t layer_rows =
      DivideUp(region.bufferImageHeight != 0 ? region.bufferImageHeight
                                             : region.imageExtent.height,
               block.height);
  const std::uint64_t layers =
      std::max<std::uint32_t>(1, region.imageSubresource.layerCount);
  return ((layers - 1) * layer_rows * row_blocks +
          (DivideUp(region.imageExtent.height, block.height) - 1) * row_blocks +
          DivideUp(region.imageExtent.width, block.width)) *
         block.bytes;
}

// ---- scenario construction ------------------------------------------------

struct RegionOptions {
  std::uint32_t base_layer = 0;
  std::uint32_t layers = 0;          // 0: every layer of the image
  std::uint32_t row_padding = 0;     // texels added to bufferRowLength
  std::uint32_t height_padding = 0;  // rows added to bufferImageHeight
  std::uint32_t gap_blocks = 0;      // unused blocks before the region
};

class ScenarioBuilder {
 public:
  ScenarioBuilder(std::string name, std::uint64_t seed, std::string description)
      : seed_(seed) {
    scenario_.name = std::move(name);
    scenario_.description = std::move(description);
  }

  ScenarioBuilder& Upscale(std::uint32_t factor) {
    scenario_.upscale = factor;
    return *this;
  }
  ScenarioBuilder& NotEmulated() {
    scenario_.emulated = false;
    return *this;
  }
  ScenarioBuilder& Overrides() {
    scenario_.overrides = true;
    return *this;
  }
  ScenarioBuilder& QueueFamily(std::uint32_t family) {
    scenario_.queue_family = family;
    return *this;
  }
  ScenarioBuilder& FakeOnly(std::string why) {
    scenario_.fake_only = std::move(why);
    return *this;
  }
  ScenarioBuilder& AllowFakeError(std::string substring) {
    scenario_.allowed_fake_errors.push_back(std::move(substring));
    return *this;
  }

  std::uint32_t Image(VkFormat format, std::uint32_t width,
                      std::uint32_t height, std::uint32_t mips = 1,
                      std::uint32_t layers = 1) {
    scenario_.images.push_back({format, width, height, mips, layers});
    return static_cast<std::uint32_t>(scenario_.images.size() - 1);
  }

  std::uint32_t Source(bool mapped = true, VkDeviceSize bind_offset = 0,
                       VkDeviceSize map_offset = 0) {
    ScenarioSource source;
    source.seed = seed_ * 1000003ULL + scenario_.sources.size();
    source.mapped = mapped;
    source.bind_offset = bind_offset;
    source.map_offset = map_offset;
    scenario_.sources.push_back(source);
    return static_cast<std::uint32_t>(scenario_.sources.size() - 1);
  }

  std::uint32_t CommandBuffer(bool secondary = false) {
    scenario_.command_buffers.push_back({secondary});
    return static_cast<std::uint32_t>(scenario_.command_buffers.size() - 1);
  }

  // Places one region's data at the end of `source` and returns the region.
  VkBufferImageCopy Region(std::uint32_t source, std::uint32_t image,
                           std::uint32_t mip, std::int32_t x, std::int32_t y,
                           std::uint32_t width, std::uint32_t height,
                           RegionOptions options = {}) {
    const ScenarioImage& spec = scenario_.images[image];
    const FormatBlock block = BlockOf(spec.format);
    VkBufferImageCopy region{};
    region.imageSubresource = {
        VK_IMAGE_ASPECT_COLOR_BIT, mip, options.base_layer,
        options.layers != 0 ? options.layers : spec.array_layers};
    region.imageOffset = {x, y, 0};
    region.imageExtent = {width, height, 1};
    if (options.row_padding != 0) {
      region.bufferRowLength = static_cast<std::uint32_t>(
          AlignUp(width, block.width) + options.row_padding);
    }
    if (options.height_padding != 0) {
      region.bufferImageHeight = static_cast<std::uint32_t>(
          AlignUp(height, block.height) + options.height_padding);
    }
    ScenarioSource& data = scenario_.sources[source];
    region.bufferOffset = AlignUp(data.size, std::max(block.bytes, 4U)) +
                          options.gap_blocks * block.bytes;
    data.size = region.bufferOffset + RegionSpan(block, region);
    return region;
  }

  VkBufferImageCopy FullMip(std::uint32_t source, std::uint32_t image,
                            std::uint32_t mip, RegionOptions options = {}) {
    const ScenarioImage& spec = scenario_.images[image];
    return Region(source, image, mip, 0, 0, Mip(spec.width, mip),
                  Mip(spec.height, mip), options);
  }

  std::vector<VkBufferImageCopy> MipChain(std::uint32_t source,
                                          std::uint32_t image,
                                          RegionOptions options = {}) {
    std::vector<VkBufferImageCopy> regions;
    for (std::uint32_t mip = 0; mip < scenario_.images[image].mip_levels;
         ++mip) {
      regions.push_back(FullMip(source, image, mip, options));
    }
    return regions;
  }

  ScenarioBuilder& Copy(std::uint32_t command_buffer, std::uint32_t image,
                        std::uint32_t source,
                        std::vector<VkBufferImageCopy> regions,
                        bool copy2 = false) {
    ScenarioStep step;
    step.kind = ScenarioStep::Kind::kCopyBufferToImage;
    step.command_buffer = command_buffer;
    step.image = image;
    step.source = source;
    step.regions = std::move(regions);
    step.copy2 = copy2;
    scenario_.needs_copy_commands2 |= copy2;
    scenario_.steps.push_back(std::move(step));
    return *this;
  }

  ScenarioBuilder& CopyImage(std::uint32_t command_buffer, std::uint32_t source,
                             std::uint32_t destination,
                             std::vector<VkImageCopy> regions,
                             bool copy2 = false) {
    ScenarioStep step;
    step.kind = ScenarioStep::Kind::kCopyImage;
    step.command_buffer = command_buffer;
    step.image = source;
    step.image2 = destination;
    step.image_regions = std::move(regions);
    step.copy2 = copy2;
    scenario_.needs_copy_commands2 |= copy2;
    scenario_.steps.push_back(std::move(step));
    return *this;
  }

  ScenarioBuilder& Execute(std::uint32_t primary,
                           std::vector<std::uint32_t> secondaries) {
    ScenarioStep step;
    step.kind = ScenarioStep::Kind::kExecuteCommands;
    step.command_buffer = primary;
    step.command_buffers = std::move(secondaries);
    scenario_.steps.push_back(std::move(step));
    return *this;
  }

  ScenarioBuilder& Submit(std::vector<std::vector<std::uint32_t>> batches,
                          SubmitShape shape = SubmitShape::kQueueSubmit) {
    ScenarioStep step;
    step.kind = ScenarioStep::Kind::kSubmit;
    step.batches = std::move(batches);
    step.shape = shape;
    scenario_.steps.push_back(std::move(step));
    return *this;
  }

  ScenarioBuilder& Reset(std::vector<std::uint32_t> command_buffers) {
    ScenarioStep step;
    step.kind = ScenarioStep::Kind::kReset;
    step.command_buffers = std::move(command_buffers);
    scenario_.steps.push_back(std::move(step));
    return *this;
  }

  ScenarioBuilder& RewriteSource(std::uint32_t source) {
    ScenarioStep step;
    step.kind = ScenarioStep::Kind::kRewriteSource;
    step.source = source;
    step.seed = seed_ * 7919ULL + 0x9E3779B97F4A7C15ULL * ++rewrites_;
    scenario_.steps.push_back(std::move(step));
    return *this;
  }

  ScenarioBuilder& WriteOverride(std::uint32_t image, std::uint32_t source,
                                 const VkBufferImageCopy& region,
                                 std::uint32_t width, std::uint32_t height) {
    ScenarioStep step;
    step.kind = ScenarioStep::Kind::kWriteOverride;
    step.image = image;
    step.source = source;
    step.regions = {region};
    step.override_width = width;
    step.override_height = height;
    step.seed = seed_ * 104729ULL + ++overrides_;
    scenario_.steps.push_back(std::move(step));
    return *this;
  }

  ScenarioBuilder& Checkpoint() {
    ScenarioStep step;
    step.kind = ScenarioStep::Kind::kCheckpoint;
    scenario_.steps.push_back(std::move(step));
    return *this;
  }

  Scenario Build() { return std::move(scenario_); }

 private:
  Scenario scenario_;
  std::uint64_t seed_ = 0;
  std::uint64_t rewrites_ = 0;
  std::uint64_t overrides_ = 0;
};

constexpr VkFormat kEtc2Formats[] = {
    VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,   VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
    VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
    VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
    VK_FORMAT_EAC_R11_UNORM_BLOCK,       VK_FORMAT_EAC_R11_SNORM_BLOCK,
    VK_FORMAT_EAC_R11G11_UNORM_BLOCK,    VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
};

constexpr VkFormat kColorFormats[] = {
    VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,   VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
    VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
    VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
};

constexpr VkFormat kRgb8 = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
constexpr VkFormat kRgb8Srgb = VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
constexpr VkFormat kRgb8A1 = VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
constexpr VkFormat kRgb8A1Srgb = VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK;
constexpr VkFormat kRgba8 = VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
constexpr VkFormat kRgba8Srgb = VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
constexpr VkFormat kR11 = VK_FORMAT_EAC_R11_UNORM_BLOCK;
constexpr VkFormat kR11Snorm = VK_FORMAT_EAC_R11_SNORM_BLOCK;
constexpr VkFormat kRg11 = VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
constexpr VkFormat kRg11Snorm = VK_FORMAT_EAC_R11G11_SNORM_BLOCK;
constexpr VkFormat kPlainRgba = VK_FORMAT_R8G8B8A8_UNORM;

VkImageCopy ImageCopy(std::uint32_t src_mip, std::int32_t sx, std::int32_t sy,
                      std::uint32_t dst_mip, std::int32_t dx, std::int32_t dy,
                      std::uint32_t width, std::uint32_t height,
                      std::uint32_t base_layer = 0, std::uint32_t layers = 1) {
  VkImageCopy copy{};
  copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, src_mip, base_layer,
                         layers};
  copy.srcOffset = {sx, sy, 0};
  copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, dst_mip, base_layer,
                         layers};
  copy.dstOffset = {dx, dy, 0};
  copy.extent = {width, height, 1};
  return copy;
}

std::vector<Scenario> BuildScenarios() {
  std::vector<Scenario> scenarios;

  {
    ScenarioBuilder b("FullMipChains", 1,
                      "every ETC2/EAC format, whole mip chains in one copy "
                      "per image, sizes that are not multiples of 4, layers");
    const std::uint32_t dims[][4] = {
        {37, 37, 6, 1},  {60, 33, 6, 1}, {130, 66, 3, 2}, {5, 3, 3, 1},
        {256, 64, 9, 3}, {1, 1, 1, 1},   {64, 64, 7, 1},  {17, 9, 5, 1},
        {4, 4, 3, 1},    {128, 8, 8, 2}};
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    std::uint32_t next = 0;
    for (VkFormat format : kEtc2Formats) {
      for (int count = 0; count < 2; ++count, ++next) {
        const auto& d = dims[next % 10];
        const std::uint32_t image = b.Image(format, d[0], d[1], d[2], d[3]);
        b.Copy(cb, image, source, b.MipChain(source, image));
      }
    }
    b.Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("PartialUploadsWithOffsets", 2,
                      "sub-rectangles at block offsets, edge regions with "
                      "extents that are not multiples of 4, several regions "
                      "per copy, later copies overwriting earlier ones");
    const VkFormat formats[] = {kRgb8, kRgba8Srgb, kR11Snorm, kRg11,
                                kRgb8A1Srgb};
    const std::uint32_t source = b.Source();
    const std::uint32_t full = b.CommandBuffer();
    const std::uint32_t partial = b.CommandBuffer();
    const std::uint32_t overlap = b.CommandBuffer();
    std::vector<std::uint32_t> images;
    for (VkFormat format : formats) {
      images.push_back(b.Image(format, 70, 50, 3));
    }
    for (std::uint32_t image : images) {
      b.Copy(full, image, source, b.MipChain(source, image));
    }
    b.Submit({{full}}).Checkpoint();
    for (std::uint32_t image : images) {
      b.Copy(partial, image, source,
             {b.Region(source, image, 0, 8, 4, 16, 12),
              b.Region(source, image, 0, 64, 48, 6, 2),
              b.Region(source, image, 0, 0, 0, 4, 4),
              b.Region(source, image, 1, 32, 24, 3, 1),
              b.Region(source, image, 2, 4, 8, 13, 4)});
    }
    b.Submit({{partial}}).Checkpoint();
    for (std::uint32_t image : images) {
      b.Copy(overlap, image, source,
             {b.Region(source, image, 0, 8, 8, 16, 16)});
      b.Copy(overlap, image, source,
             {b.Region(source, image, 0, 12, 12, 8, 8)});
    }
    b.Submit({{overlap}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("RowStridesAndOffsets", 3,
                      "bufferRowLength and bufferImageHeight padding, gaps "
                      "between regions, buffers bound at an offset inside "
                      "their memory, memory mapped from an offset");
    const std::uint32_t sources[] = {b.Source(true, 256, 128),
                                     b.Source(true, 0, 0),
                                     b.Source(false, 512, 0)};
    const VkFormat formats[] = {kRgb8, kRgba8, kR11, kRg11Snorm};
    const std::uint32_t cb = b.CommandBuffer();
    for (std::size_t index = 0; index < 4; ++index) {
      const std::uint32_t image = b.Image(formats[index], 45, 29, 2);
      const std::uint32_t source = sources[index % 3];
      RegionOptions padded;
      padded.row_padding = 8;
      padded.height_padding = 4;
      padded.gap_blocks = 3;
      RegionOptions rows_only;
      rows_only.row_padding = 4;
      RegionOptions height_only;
      height_only.height_padding = 8;
      height_only.gap_blocks = 1;
      b.Copy(cb, image, source,
             {b.FullMip(source, image, 0, padded),
              b.FullMip(source, image, 1, rows_only)});
      b.Copy(cb, image, source,
             {b.Region(source, image, 0, 12, 8, 20, 12, height_only)});
    }
    b.Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("ArrayLayers", 4,
                      "array images: every layer in one region with image "
                      "height padding, then partial layer ranges");
    const VkFormat formats[] = {kRgb8A1, kRgba8, kR11, kRg11Snorm};
    const std::uint32_t source = b.Source();
    const std::uint32_t full = b.CommandBuffer();
    const std::uint32_t partial = b.CommandBuffer();
    std::vector<std::uint32_t> images;
    for (VkFormat format : formats) {
      images.push_back(b.Image(format, 33, 20, 2, 3));
    }
    RegionOptions height;
    height.height_padding = 4;
    for (std::uint32_t image : images) {
      b.Copy(full, image, source, b.MipChain(source, image, height));
    }
    b.Submit({{full}}).Checkpoint();
    for (std::uint32_t image : images) {
      RegionOptions two;
      two.base_layer = 1;
      two.layers = 2;
      two.row_padding = 4;
      RegionOptions last;
      last.base_layer = 2;
      last.layers = 1;
      b.Copy(partial, image, source,
             {b.Region(source, image, 0, 16, 4, 12, 8, two)});
      b.Copy(partial, image, source,
             {b.Region(source, image, 1, 0, 0, 16, 10, last)});
    }
    b.Submit({{partial}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("UpscaledMipChains", 5,
                      "x4 small-texture upscale of whole colour chains: mips "
                      "1..n resampled from mip 0, odd sizes cropped, a chain "
                      "uploaded smallest mip first, block-rounded extents on "
                      "mips smaller than a block");
    b.Upscale(4);
    const std::uint32_t dims[][3] = {
        {64, 64, 7}, {37, 23, 6}, {1, 1, 1}, {4, 64, 7}, {12, 20, 5}};
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    std::uint32_t next = 0;
    for (VkFormat format : kColorFormats) {
      for (int count = 0; count < 3; ++count, ++next) {
        const auto& d = dims[next % 5];
        const std::uint32_t image = b.Image(format, d[0], d[1], d[2]);
        b.Copy(cb, image, source, b.MipChain(source, image));
      }
    }
    // Small mips first: no base yet, so each resamples its own texels.
    const std::uint32_t reversed = b.Image(kRgb8, 64, 64, 7);
    std::vector<VkBufferImageCopy> backwards = b.MipChain(source, reversed);
    std::reverse(backwards.begin(), backwards.end());
    b.Copy(cb, reversed, source, backwards);
    // 6x6: Roblox-style extents rounded up to whole blocks (8x8, 4x4, 4x4).
    const std::uint32_t rounded = b.Image(kRgba8, 6, 6, 3);
    b.Copy(cb, rounded, source,
           {b.Region(source, rounded, 0, 0, 0, 8, 8),
            b.Region(source, rounded, 1, 0, 0, 4, 4),
            b.Region(source, rounded, 2, 0, 0, 4, 4)});
    b.Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("UpscaledPartialUpdates", 6,
                      "the x4 base rule across submits: a full mip before "
                      "mip 0, mip 0 setting the base, later full mips from "
                      "that base, partial updates that neither set nor use it");
    b.Upscale(4);
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t a = b.Image(kRgba8, 64, 64, 7);
    const std::uint32_t c = b.Image(kRgb8Srgb, 40, 24, 3);
    const std::uint32_t d = b.Image(kRgb8A1, 16, 16, 5);
    b.Copy(cb, a, source, {b.FullMip(source, a, 1)}).Submit({{cb}});
    b.Reset({cb}).Copy(cb, a, source, {b.FullMip(source, a, 0)}).Submit({{cb}});
    b.Reset({cb})
        .Copy(cb, a, source, {b.FullMip(source, a, 2), b.FullMip(source, a, 3)})
        .Submit({{cb}})
        .Checkpoint();
    b.Reset({cb})
        .Copy(cb, a, source, {b.Region(source, a, 0, 8, 8, 16, 16)})
        .Copy(cb, c, source, {b.Region(source, c, 0, 36, 20, 4, 4)})
        .Copy(cb, c, source, {b.Region(source, c, 1, 16, 8, 4, 4)})
        .Submit({{cb}});
    b.Reset({cb})
        .Copy(cb, a, source, {b.FullMip(source, a, 4)})
        .Copy(cb, d, source, b.MipChain(source, d))
        .Copy(cb, d, source, {b.Region(source, d, 0, 4, 4, 8, 8)})
        .Submit({{cb}})
        .Checkpoint();
    b.Reset({cb}).Copy(cb, d, source, {b.FullMip(source, d, 2)}).Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("UpscaleSelection", 7,
                      "which images the x4 upscale takes: colour, 2D, one "
                      "layer, both sides at most 64");
    b.Upscale(4);
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t images[] = {
        b.Image(kRgb8, 65, 64, 7),    b.Image(kRgba8Srgb, 64, 65, 7),
        b.Image(kR11, 32, 32, 6),     b.Image(kRg11Snorm, 32, 32, 6),
        b.Image(kRgb8, 32, 32, 6, 2), b.Image(kRgb8A1, 64, 64, 7),
        b.Image(kRgba8, 64, 1, 7),
    };
    for (std::uint32_t image : images) {
      b.Copy(cb, image, source, b.MipChain(source, image));
    }
    b.Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("UpscaleFactorTwo", 8,
                      "MOCKTAIL_SMALL_TEXTURE_UPSCALE=2: chains and a partial "
                      "update");
    b.Upscale(2);
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t a = b.Image(kRgb8Srgb, 48, 40, 6);
    const std::uint32_t c = b.Image(kRgba8, 3, 5, 3);
    b.Copy(cb, a, source, b.MipChain(source, a))
        .Copy(cb, c, source, b.MipChain(source, c))
        .Submit({{cb}})
        .Checkpoint();
    b.Reset({cb})
        .Copy(cb, a, source, {b.Region(source, a, 0, 8, 8, 16, 16)})
        .Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("CopyBufferToImage2", 9,
                      "the same shapes through vkCmdCopyBufferToImage2: "
                      "upscaled and plain images, strides, layers");
    b.Upscale(4);
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t a = b.Image(kRgb8, 64, 64, 7);
    const std::uint32_t c = b.Image(kRgba8Srgb, 30, 18, 5);
    const std::uint32_t d = b.Image(kR11Snorm, 72, 40, 3);
    const std::uint32_t e = b.Image(kRg11, 33, 20, 2, 2);
    RegionOptions padded;
    padded.row_padding = 4;
    padded.height_padding = 4;
    b.Copy(cb, a, source, b.MipChain(source, a), true)
        .Copy(cb, c, source, b.MipChain(source, c, padded), true)
        .Copy(cb, d, source, b.MipChain(source, d, padded), true)
        .Copy(cb, e, source, b.MipChain(source, e, padded), true)
        .Submit({{cb}})
        .Checkpoint();
    RegionOptions second_layer;
    second_layer.base_layer = 1;
    second_layer.layers = 1;
    b.Reset({cb})
        .Copy(cb, a, source, {b.Region(source, a, 0, 16, 0, 32, 8)}, true)
        .Copy(cb, d, source, {b.Region(source, d, 0, 64, 36, 8, 4)}, true)
        .Copy(cb, e, source,
              {b.Region(source, e, 0, 4, 4, 12, 8, second_layer)}, true)
        .Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("SecondaryCommandBuffers", 10,
                      "uploads recorded in secondaries and run through "
                      "vkCmdExecuteCommands, interleaved with the primary's "
                      "own uploads; a secondary with only a plain copy");
    b.Upscale(4);
    const std::uint32_t source = b.Source();
    const std::uint32_t primary = b.CommandBuffer();
    const std::uint32_t s1 = b.CommandBuffer(true);
    const std::uint32_t s2 = b.CommandBuffer(true);
    const std::uint32_t s3 = b.CommandBuffer(true);
    const std::uint32_t i0 = b.Image(kRgb8, 64, 64, 7);
    const std::uint32_t i1 = b.Image(kRgba8, 48, 48, 6);
    const std::uint32_t i2 = b.Image(kR11, 64, 40, 3);
    const std::uint32_t i3 = b.Image(kPlainRgba, 16, 16, 1);
    b.Copy(s1, i1, source, b.MipChain(source, i1))
        .Copy(s1, i0, source, {b.FullMip(source, i0, 1)})
        .Copy(s2, i0, source, {b.FullMip(source, i0, 2)})
        .Copy(s3, i3, source, {b.Region(source, i3, 0, 4, 4, 8, 8)})
        .Copy(primary, i0, source, {b.FullMip(source, i0, 0)})
        .Execute(primary, {s1})
        .Copy(primary, i2, source, b.MipChain(source, i2))
        .Execute(primary, {s2, s3})
        .Submit({{primary}})
        .Checkpoint()
        .Submit({{primary}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("ResubmitAfterSourceChange", 11,
                      "a command buffer submitted again after Roblox "
                      "rewrote its upload buffer decodes the new bytes");
    b.Upscale(4);
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t a = b.Image(kRgba8, 40, 40, 6);
    const std::uint32_t c = b.Image(kR11, 72, 40, 3);
    b.Copy(cb, a, source, b.MipChain(source, a))
        .Copy(cb, c, source, b.MipChain(source, c))
        .Submit({{cb}})
        .Checkpoint()
        .RewriteSource(source)
        .Submit({{cb}})
        .Checkpoint();
    b.Reset({cb})
        .Copy(cb, c, source, {b.Region(source, c, 0, 8, 8, 32, 16)})
        .RewriteSource(source)
        .Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("MultiCommandBufferSubmits", 12,
                      "several command buffers and batches per submit, "
                      "vkQueueSubmit2 with two command buffers in a batch, "
                      "an empty command buffer, the base rule across "
                      "command buffers of one submit");
    b.Upscale(4);
    const std::uint32_t source = b.Source();
    const std::uint32_t cb1 = b.CommandBuffer();
    const std::uint32_t cb2 = b.CommandBuffer();
    const std::uint32_t cb3 = b.CommandBuffer();
    const std::uint32_t cb4 = b.CommandBuffer();
    const std::uint32_t a = b.Image(kRgb8, 64, 64, 7);
    const std::uint32_t c = b.Image(kRgba8Srgb, 128, 32, 8);
    const std::uint32_t d = b.Image(kR11, 40, 40, 6);
    b.Copy(cb1, a, source, {b.FullMip(source, a, 0)})
        .Copy(cb1, d, source, b.MipChain(source, d))
        .Copy(cb2, a, source, {b.FullMip(source, a, 1)})
        .Copy(cb2, c, source, {b.Region(source, c, 0, 16, 8, 32, 16)})
        .Copy(cb4, c, source, b.MipChain(source, c))
        .Submit({{cb4}, {cb1, cb2, cb3}})
        .Checkpoint();
    b.Reset({cb1, cb2})
        .Copy(cb1, a, source, {b.FullMip(source, a, 2)})
        .Copy(cb1, c, source, {b.Region(source, c, 0, 0, 0, 16, 16)})
        .Copy(cb2, d, source, {b.Region(source, d, 0, 4, 4, 8, 8)})
        .Submit({{cb1}, {cb2, cb3}}, SubmitShape::kQueueSubmit2)
        .Submit({{cb4}, {cb1}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("UnmappedSources", 13,
                      "upload buffers Roblox does not keep mapped: the "
                      "emulation maps each memory once for the submit");
    b.Upscale(4);
    const std::uint32_t s0 = b.Source(false, 0);
    const std::uint32_t s1 = b.Source(false, 512);
    const std::uint32_t s2 = b.Source(true, 256, 0);
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t a = b.Image(kRgb8, 64, 64, 7);
    const std::uint32_t c = b.Image(kR11, 70, 50, 3);
    const std::uint32_t d = b.Image(kRgba8Srgb, 100, 60, 4);
    const std::uint32_t e = b.Image(kRg11, 32, 32, 6);
    b.Copy(cb, a, s0, b.MipChain(s0, a))
        .Copy(cb, c, s0, b.MipChain(s0, c))
        .Copy(cb, d, s1, b.MipChain(s1, d))
        .Copy(cb, e, s2, b.MipChain(s2, e))
        .Submit({{cb}})
        .Checkpoint()
        .RewriteSource(s0)
        .RewriteSource(s1)
        .Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("CopyImageUnscaled", 14,
                      "vkCmdCopyImage(2) between emulated images and into a "
                      "plain RGBA8 image, passed through unchanged");
    const std::uint32_t source = b.Source();
    const std::uint32_t upload = b.CommandBuffer();
    const std::uint32_t copies = b.CommandBuffer();
    const std::uint32_t i0 = b.Image(kRgb8, 64, 48, 2);
    const std::uint32_t i1 = b.Image(kRgba8, 64, 48, 2);
    const std::uint32_t i2 = b.Image(kR11, 32, 32, 2);
    const std::uint32_t i3 = b.Image(kR11, 32, 32, 2);
    const std::uint32_t i4 = b.Image(kPlainRgba, 64, 48, 1);
    for (std::uint32_t image : {i0, i1, i2, i3}) {
      b.Copy(upload, image, source, b.MipChain(source, image));
    }
    b.Copy(upload, i4, source, {b.FullMip(source, i4, 0)})
        .Submit({{upload}})
        .Checkpoint();
    b.CopyImage(copies, i0, i1,
                {ImageCopy(0, 8, 8, 0, 16, 4, 32, 16),
                 ImageCopy(1, 0, 0, 1, 0, 0, 32, 24)})
        .CopyImage(copies, i2, i3, {ImageCopy(0, 4, 4, 0, 8, 12, 16, 16)}, true)
        .CopyImage(copies, i1, i4, {ImageCopy(0, 0, 0, 0, 32, 24, 32, 24)})
        .Submit({{copies}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("CopyImageScaled", 15,
                      "vkCmdCopyImage(2) between two x4-upscaled images "
                      "becomes a same-size NEAREST blit of the host regions; "
                      "between unscaled images it stays a copy");
    b.Upscale(4);
    const std::uint32_t source = b.Source();
    const std::uint32_t upload = b.CommandBuffer();
    const std::uint32_t copies = b.CommandBuffer();
    const std::uint32_t i0 = b.Image(kRgb8, 32, 32, 6);
    const std::uint32_t i1 = b.Image(kRgba8, 32, 32, 6);
    const std::uint32_t i2 = b.Image(kRgb8, 128, 128, 8);
    const std::uint32_t i3 = b.Image(kRgba8, 128, 128, 8);
    for (std::uint32_t image : {i0, i1, i2, i3}) {
      b.Copy(upload, image, source, b.MipChain(source, image));
    }
    b.Submit({{upload}}).Checkpoint();
    b.CopyImage(copies, i0, i1,
                {ImageCopy(0, 0, 0, 0, 0, 0, 16, 16),
                 ImageCopy(0, 16, 16, 0, 0, 16, 16, 16),
                 ImageCopy(1, 0, 0, 1, 0, 0, 16, 16)})
        .CopyImage(copies, i0, i1, {ImageCopy(2, 0, 0, 2, 0, 0, 8, 8)}, true)
        .CopyImage(copies, i2, i3, {ImageCopy(0, 64, 64, 0, 0, 0, 64, 64)})
        .Submit({{copies}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("CopyImageScaledToUnscaled", 16,
                      "copies between an upscaled and a plain-size image "
                      "become LINEAR blits; an out-of-range region drops "
                      "the whole copy");
    b.Upscale(4).FakeOnly(
        "filtered blits have no exact expected texels; the fake records them");
    const std::uint32_t source = b.Source();
    const std::uint32_t upload = b.CommandBuffer();
    const std::uint32_t copies = b.CommandBuffer();
    const std::uint32_t i0 = b.Image(kRgb8, 32, 32, 6);
    const std::uint32_t i1 = b.Image(kRgb8, 128, 128, 8);
    b.Copy(upload, i0, source, b.MipChain(source, i0))
        .Copy(upload, i1, source, b.MipChain(source, i1))
        .Submit({{upload}});
    b.CopyImage(copies, i0, i1, {ImageCopy(0, 0, 0, 0, 0, 0, 16, 16)})
        .CopyImage(copies, i1, i0, {ImageCopy(0, 0, 0, 0, 16, 16, 16, 16)})
        .CopyImage(copies, i0, i1, {ImageCopy(0, 40, 0, 0, 0, 0, 8, 8)})
        .Submit({{copies}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("RejectedRegions", 17,
                      "regions the emulation cannot plan drop the whole "
                      "copy: bufferRowLength below the width, an offset past "
                      "an upscaled mip, no layers");
    b.Upscale(4).FakeOnly("invalid Vulkan usage");
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t scaled = b.Image(kRgb8, 64, 64, 7);
    const std::uint32_t plain = b.Image(kR11, 64, 64, 7);
    VkBufferImageCopy short_rows = b.Region(source, plain, 0, 0, 0, 16, 16);
    short_rows.bufferRowLength = 4;
    VkBufferImageCopy no_layers = b.Region(source, plain, 0, 0, 0, 8, 8);
    no_layers.imageSubresource.layerCount = 0;
    b.Copy(cb, plain, source, {b.FullMip(source, plain, 0), short_rows})
        .Copy(cb, scaled, source, {b.Region(source, scaled, 0, 64, 0, 4, 4)})
        .Copy(cb, plain, source, {no_layers})
        .Copy(cb, plain, source, {b.FullMip(source, plain, 1)})
        .Copy(cb, scaled, source, {b.FullMip(source, scaled, 0)})
        .Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("NotEmulatedDevice", 18,
                      "on a device with ETC2 every image, copy and image "
                      "copy passes through untouched");
    b.Upscale(4).NotEmulated().FakeOnly(
        "the test GPU cannot create ETC2 images");
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t i0 = b.Image(kRgb8, 64, 64, 7);
    const std::uint32_t i1 = b.Image(kRg11, 40, 40, 6);
    const std::uint32_t i2 = b.Image(kRgba8Srgb, 32, 32, 6);
    const std::uint32_t i3 = b.Image(kRgb8, 64, 64, 7);
    for (std::uint32_t image : {i0, i1, i2, i3}) {
      b.Copy(cb, image, source, b.MipChain(source, image));
    }
    b.CopyImage(cb, i0, i3, {ImageCopy(0, 16, 16, 0, 0, 0, 32, 32)})
        .Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("TransferOnlyQueueFamily", 19,
                      "uploads recorded on a transfer-only queue family");
    // CR-19: EAC R11 regions with an odd texel count used to leave the
    // next region's staging offset at 2 mod 4, which a transfer-only queue
    // forbids; staging regions now start on 16-byte boundaries.
    b.QueueFamily(1);
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t images[] = {
        b.Image(kR11, 5, 3, 3), b.Image(kR11Snorm, 13, 7, 4),
        b.Image(kRgb8, 20, 20, 5), b.Image(kRg11, 9, 9, 4)};
    for (std::uint32_t image : images) {
      b.Copy(cb, image, source, b.MipChain(source, image));
    }
    b.Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  {
    ScenarioBuilder b("TextureOverridesAndDumps", 20,
                      "override PNGs replace a colour chain by the hash of "
                      "its mip-0 bytes (resampled to every mip and to the "
                      "upscaled size); mip-0 colour uploads are dumped");
    b.Upscale(4).Overrides();
    const std::uint32_t source = b.Source();
    const std::uint32_t cb = b.CommandBuffer();
    const std::uint32_t a = b.Image(kRgb8, 64, 64, 7);
    const std::uint32_t c = b.Image(kRgba8Srgb, 128, 64, 8);
    const std::uint32_t d = b.Image(kR11, 32, 32, 6);
    const std::uint32_t e = b.Image(kRgb8A1Srgb, 48, 48, 6);
    const std::uint32_t f = b.Image(kRgba8, 32, 32, 2, 2);
    std::vector<VkBufferImageCopy> a_chain = b.MipChain(source, a);
    std::vector<VkBufferImageCopy> c_chain = b.MipChain(source, c);
    b.WriteOverride(a, source, a_chain[0], 100, 80)
        .WriteOverride(c, source, c_chain[0], 33, 17)
        .Copy(cb, a, source, a_chain)
        .Copy(cb, c, source, c_chain)
        .Copy(cb, d, source, b.MipChain(source, d))
        .Copy(cb, e, source, b.MipChain(source, e))
        .Copy(cb, f, source, b.MipChain(source, f))
        .Submit({{cb}})
        .Checkpoint();
    b.Reset({cb})
        .Copy(cb, c, source, {b.Region(source, c, 0, 16, 16, 32, 32)})
        .Copy(cb, c, source, {b.FullMip(source, c, 1)})
        .Submit({{cb}});
    scenarios.push_back(b.Build());
  }

  return scenarios;
}

}  // namespace

const std::vector<Scenario>& Etc2EmulationScenarios() {
  static const std::vector<Scenario>* scenarios =
      new std::vector<Scenario>(BuildScenarios());
  return *scenarios;
}

const Scenario* FindEtc2EmulationScenario(const std::string& name) {
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    if (scenario.name == name) {
      return &scenario;
    }
  }
  return nullptr;
}

std::vector<std::uint8_t> SeededBytes(std::uint64_t seed, std::size_t size) {
  // splitmix64: fixed by its definition, unlike the standard distributions.
  std::vector<std::uint8_t> bytes(size);
  std::uint64_t state = seed;
  for (std::size_t index = 0; index < size; index += 8) {
    std::uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    value ^= value >> 31;
    for (std::size_t byte = 0; byte < 8 && index + byte < size; ++byte) {
      bytes[index + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
    }
  }
  return bytes;
}

HostImageShape HostShapeFor(const ScenarioImage& image, std::uint32_t upscale,
                            bool emulated) {
  EmulatedFormat format;
  if (!emulated || !LookupEmulated(image.format, &format)) {
    return {image.format, image.width, image.height, image.mip_levels,
            image.array_layers};
  }
  const bool scaled = upscale > 1 && format.color && image.array_layers == 1 &&
                      image.width <= 64 && image.height <= 64;
  const std::uint32_t scale = scaled ? upscale : 1;
  return {format.host, image.width * scale, image.height * scale,
          image.mip_levels, image.array_layers};
}

void GoldenHasher::Add(const void* data, std::size_t size) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t index = 0; index < size; ++index) {
    hash_ ^= bytes[index];
    hash_ *= 0x100000001b3ULL;
  }
}

void GoldenHasher::AddU32(std::uint32_t value) {
  const std::uint8_t bytes[4] = {static_cast<std::uint8_t>(value),
                                 static_cast<std::uint8_t>(value >> 8),
                                 static_cast<std::uint8_t>(value >> 16),
                                 static_cast<std::uint8_t>(value >> 24)};
  Add(bytes, sizeof(bytes));
}

bool FindEtc2EmulationGolden(const std::string& name, std::uint64_t* hash) {
  struct Entry {
    const char* name;
    std::uint64_t hash;
  };
  static constexpr Entry kGoldens[] = {
#include "vulkan_etc2_emulation_golden.inc"
      {nullptr, 0},
  };
  for (const Entry& entry : kGoldens) {
    if (entry.name != nullptr && name == entry.name) {
      *hash = entry.hash;
      return true;
    }
  }
  return false;
}

// ---- the reference model ----------------------------------------------------

struct Etc2EmulationModel::State {
  struct Image {
    ScenarioImage spec;
    HostImageShape host;
    bool emulated = false;  // decoded on upload
    EmulatedFormat format;
    std::uint32_t scale = 1;
    FormatBlock block;  // of the host format
    std::vector<std::vector<std::uint8_t>> subresources;
    std::shared_ptr<const RgbaImage> base;
    std::shared_ptr<const RgbaImage> override_image;
  };
  struct Upload {
    std::uint32_t image = 0;
    std::uint32_t source = 0;
    VkBufferImageCopy region{};
    std::uint32_t mip = 0;
    std::uint32_t base_layer = 0;
    std::uint32_t layers = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;
    std::uint32_t target_width = 0;
    std::uint32_t target_height = 0;
    std::uint32_t host_x = 0;
    std::uint32_t host_y = 0;
    bool full_mip = false;
    std::vector<std::uint8_t> staging;
  };
  struct Command {
    enum class Kind { kUploads, kRawCopy, kCopyImage, kBlit, kExecute };
    Kind kind = Kind::kUploads;
    std::vector<std::size_t> uploads;
    std::uint32_t image = 0;
    std::uint32_t image2 = 0;
    std::uint32_t source = 0;
    std::vector<VkBufferImageCopy> raw_regions;
    std::vector<VkImageCopy> copies;
    std::vector<VkImageBlit> blits;
    VkFilter filter = VK_FILTER_NEAREST;
    std::vector<std::uint32_t> secondaries;
  };
  struct CommandBuffer {
    std::vector<Command> commands;
    std::vector<Upload> uploads;
    std::vector<std::uint32_t> linked;  // secondaries with uploads
  };

  std::uint32_t upscale = 1;
  bool device_emulated = true;
  bool overrides = false;
  std::vector<Image> images;
  std::vector<CommandBuffer> command_buffers;
  std::map<std::uint64_t, std::shared_ptr<const RgbaImage>> override_files;
  std::map<std::uint64_t, std::shared_ptr<const RgbaImage>> lookups;
  std::set<std::uint64_t> dumped;
  std::vector<Dump> dumps;
  std::vector<Transfer> transfers;

  std::vector<std::uint8_t>& Texels(std::uint32_t image, std::uint32_t mip,
                                    std::uint32_t layer) {
    Image& record = images[image];
    return record.subresources[mip * record.spec.array_layers + layer];
  }

  // One axis of an upscaled region: block padding is clipped to the app's
  // mip, then the range is scaled; a region reaching the mip's edge reaches
  // the host mip's edge.
  static bool ScaledRange(std::uint32_t base, std::uint32_t scale,
                          std::uint32_t mip, std::int32_t offset,
                          std::uint32_t extent, std::uint32_t* texels,
                          std::uint32_t* start, std::uint32_t* end) {
    const std::uint32_t app_mip = Mip(base, mip);
    const std::uint32_t host_mip = Mip(base * scale, mip);
    if (offset < 0 || static_cast<std::uint32_t>(offset) >= app_mip) {
      return false;
    }
    const std::uint32_t first = static_cast<std::uint32_t>(offset);
    const std::uint32_t last = std::min(app_mip, first + extent);
    *start = first * scale;
    *end = last == app_mip ? host_mip : std::min(host_mip, last * scale);
    *texels = last - first;
    return *end > *start;
  }

  bool Plan(const Image& image, const VkBufferImageCopy& region,
            Upload* upload) const {
    const VkExtent3D& extent = region.imageExtent;
    const VkImageSubresourceLayers& sub = region.imageSubresource;
    if (extent.width == 0 || extent.height == 0 || extent.depth != 1 ||
        sub.layerCount == 0) {
      return false;
    }
    if (image.scale > 1 && sub.layerCount != 1) {
      return false;
    }
    if ((region.bufferRowLength != 0 &&
         region.bufferRowLength < extent.width) ||
        (region.bufferImageHeight != 0 &&
         region.bufferImageHeight < extent.height)) {
      return false;
    }
    upload->region = region;
    upload->mip = sub.mipLevel;
    upload->base_layer = sub.baseArrayLayer;
    upload->layers = sub.layerCount;
    upload->width = extent.width;
    upload->height = extent.height;
    upload->source_width = extent.width;
    upload->source_height = extent.height;
    upload->target_width = extent.width;
    upload->target_height = extent.height;
    upload->host_x = static_cast<std::uint32_t>(region.imageOffset.x);
    upload->host_y = static_cast<std::uint32_t>(region.imageOffset.y);
    if (image.scale > 1) {
      std::uint32_t x0 = 0;
      std::uint32_t x1 = 0;
      std::uint32_t y0 = 0;
      std::uint32_t y1 = 0;
      if (!ScaledRange(image.spec.width, image.scale, sub.mipLevel,
                       region.imageOffset.x, extent.width,
                       &upload->source_width, &x0, &x1) ||
          !ScaledRange(image.spec.height, image.scale, sub.mipLevel,
                       region.imageOffset.y, extent.height,
                       &upload->source_height, &y0, &y1)) {
        return false;
      }
      upload->target_width = x1 - x0;
      upload->target_height = y1 - y0;
      upload->host_x = x0;
      upload->host_y = y0;
      upload->full_mip =
          region.imageOffset.x == 0 && region.imageOffset.y == 0 &&
          upload->source_width == Mip(image.spec.width, sub.mipLevel) &&
          upload->source_height == Mip(image.spec.height, sub.mipLevel);
    }
    return true;
  }

  std::shared_ptr<const RgbaImage> Lookup(std::uint64_t hash) {
    const auto cached = lookups.find(hash);
    if (cached != lookups.end()) {
      return cached->second;
    }
    const auto file = override_files.find(hash);
    std::shared_ptr<const RgbaImage> found =
        file != override_files.end() ? file->second : nullptr;
    lookups[hash] = found;
    return found;
  }

  static std::vector<std::uint8_t> Resample(const std::uint8_t* pixels,
                                            std::uint32_t width,
                                            std::uint32_t height,
                                            std::uint32_t target_width,
                                            std::uint32_t target_height) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(target_width) *
                                  target_height * 4);
    reference::ResampleRgba(pixels, width, height, target_width, target_height,
                            out.data());
    return out;
  }

  // Decodes one upload from the current source bytes into its staging.
  void Emit(Upload* upload, const std::vector<std::uint8_t>& source) {
    Image& image = images[upload->image];
    const EtcFormat format = image.format.etc;
    const std::vector<std::uint8_t> tight =
        TightCompressed(image.spec.format, upload->region, source);
    const std::size_t layer_compressed = tight.size() / upload->layers;
    const std::size_t layer_decoded = static_cast<std::size_t>(upload->width) *
                                      upload->height *
                                      reference::EtcDecodedTexelBytes(format);
    std::vector<std::uint8_t> decoded(layer_decoded * upload->layers);
    for (std::uint32_t layer = 0; layer < upload->layers; ++layer) {
      reference::DecodeEtcImage(format, tight.data() + layer * layer_compressed,
                                layer_compressed, upload->width, upload->height,
                                decoded.data() + layer * layer_decoded,
                                layer_decoded);
    }
    std::shared_ptr<const RgbaImage> replacement;
    if (overrides && upload->layers == 1 && image.format.color) {
      if (upload->mip == 0) {
        const std::uint64_t hash = OverrideHash(tight);
        if (dumped.insert(hash).second) {
          dumps.push_back({hash, upload->width, upload->height, decoded});
        }
        replacement = Lookup(hash);
        image.override_image = replacement;
      } else {
        replacement = image.override_image;
      }
    }
    if (replacement != nullptr) {
      upload->staging = Resample(replacement->pixels.data(), replacement->width,
                                 replacement->height, upload->target_width,
                                 upload->target_height);
      return;
    }
    if (image.scale <= 1) {
      upload->staging = std::move(decoded);
      return;
    }
    auto cropped = std::make_shared<RgbaImage>();
    cropped->width = upload->source_width;
    cropped->height = upload->source_height;
    for (std::uint32_t y = 0; y < upload->source_height; ++y) {
      const std::uint8_t* row =
          decoded.data() + static_cast<std::size_t>(y) * upload->width * 4;
      cropped->pixels.insert(cropped->pixels.end(), row,
                             row + upload->source_width * 4);
    }
    std::shared_ptr<const RgbaImage> from = cropped;
    if (upload->full_mip) {
      if (upload->mip == 0) {
        image.base = cropped;
      }
      if (image.base != nullptr) {
        from = image.base;
      }
    }
    upload->staging = Resample(from->pixels.data(), from->width, from->height,
                               upload->target_width, upload->target_height);
  }

  void WriteUpload(const Upload& upload) {
    Image& image = images[upload.image];
    const std::size_t texel = image.block.bytes;
    const std::uint32_t mip_width = Mip(image.host.width, upload.mip);
    for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
      std::vector<std::uint8_t>& texels =
          Texels(upload.image, upload.mip, upload.base_layer + layer);
      for (std::uint32_t y = 0; y < upload.target_height; ++y) {
        std::memcpy(
            texels.data() +
                ((static_cast<std::size_t>(upload.host_y) + y) * mip_width +
                 upload.host_x) *
                    texel,
            upload.staging.data() +
                ((static_cast<std::size_t>(layer) * upload.target_height + y) *
                 upload.target_width) *
                    texel,
            static_cast<std::size_t>(upload.target_width) * texel);
      }
    }
  }

  // A buffer->image copy passed through to an image the emulation does not
  // own, in whole texel blocks of its format.
  void RawCopy(std::uint32_t image_index, const VkBufferImageCopy& region,
               const std::vector<std::uint8_t>& source) {
    Image& image = images[image_index];
    const FormatBlock& block = image.block;
    const std::uint32_t mip = region.imageSubresource.mipLevel;
    const std::uint32_t mip_blocks =
        DivideUp(Mip(image.host.width, mip), block.width);
    const std::uint32_t row_blocks =
        DivideUp(region.bufferRowLength != 0 ? region.bufferRowLength
                                             : region.imageExtent.width,
                 block.width);
    const std::uint32_t layer_rows =
        DivideUp(region.bufferImageHeight != 0 ? region.bufferImageHeight
                                               : region.imageExtent.height,
                 block.height);
    const std::uint32_t wide = DivideUp(region.imageExtent.width, block.width);
    const std::uint32_t high =
        DivideUp(region.imageExtent.height, block.height);
    const std::uint32_t bx =
        static_cast<std::uint32_t>(region.imageOffset.x) / block.width;
    const std::uint32_t by =
        static_cast<std::uint32_t>(region.imageOffset.y) / block.height;
    for (std::uint32_t layer = 0; layer < region.imageSubresource.layerCount;
         ++layer) {
      std::vector<std::uint8_t>& texels = Texels(
          image_index, mip, region.imageSubresource.baseArrayLayer + layer);
      for (std::uint32_t row = 0; row < high; ++row) {
        std::memcpy(
            texels.data() +
                ((static_cast<std::size_t>(by) + row) * mip_blocks + bx) *
                    block.bytes,
            source.data() + region.bufferOffset +
                (static_cast<std::size_t>(layer) * layer_rows + row) *
                    row_blocks * block.bytes,
            static_cast<std::size_t>(wide) * block.bytes);
      }
    }
  }

  void CopyRegion(std::uint32_t source_index, std::uint32_t destination_index,
                  const VkImageCopy& region) {
    const FormatBlock& block = images[source_index].block;
    const std::uint32_t src_blocks = DivideUp(
        Mip(images[source_index].host.width, region.srcSubresource.mipLevel),
        block.width);
    const std::uint32_t dst_blocks =
        DivideUp(Mip(images[destination_index].host.width,
                     region.dstSubresource.mipLevel),
                 block.width);
    const std::uint32_t wide = DivideUp(region.extent.width, block.width);
    const std::uint32_t high = DivideUp(region.extent.height, block.height);
    const std::uint32_t sx =
        static_cast<std::uint32_t>(region.srcOffset.x) / block.width;
    const std::uint32_t sy =
        static_cast<std::uint32_t>(region.srcOffset.y) / block.height;
    const std::uint32_t dx =
        static_cast<std::uint32_t>(region.dstOffset.x) / block.width;
    const std::uint32_t dy =
        static_cast<std::uint32_t>(region.dstOffset.y) / block.height;
    for (std::uint32_t layer = 0; layer < region.srcSubresource.layerCount;
         ++layer) {
      const std::vector<std::uint8_t> from =
          Texels(source_index, region.srcSubresource.mipLevel,
                 region.srcSubresource.baseArrayLayer + layer);
      std::vector<std::uint8_t>& to =
          Texels(destination_index, region.dstSubresource.mipLevel,
                 region.dstSubresource.baseArrayLayer + layer);
      for (std::uint32_t row = 0; row < high; ++row) {
        std::memcpy(
            to.data() +
                ((static_cast<std::size_t>(dy) + row) * dst_blocks + dx) *
                    block.bytes,
            from.data() +
                ((static_cast<std::size_t>(sy) + row) * src_blocks + sx) *
                    block.bytes,
            static_cast<std::size_t>(wide) * block.bytes);
      }
    }
  }

  // The host region an image copy touches: upscaled images scale it the way
  // uploads do; others keep the app's region.
  bool HostRegion(std::uint32_t image_index, std::uint32_t mip,
                  const VkOffset3D& offset, const VkExtent3D& extent,
                  VkOffset3D* start, VkOffset3D* end) const {
    const Image& image = images[image_index];
    if (!image.emulated || image.scale <= 1) {
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
    if (!ScaledRange(image.spec.width, image.scale, mip, offset.x, extent.width,
                     &texels, &x0, &x1) ||
        !ScaledRange(image.spec.height, image.scale, mip, offset.y,
                     extent.height, &texels, &y0, &y1)) {
      return false;
    }
    *start = {static_cast<std::int32_t>(x0), static_cast<std::int32_t>(y0), 0};
    *end = {static_cast<std::int32_t>(x1), static_cast<std::int32_t>(y1), 1};
    return true;
  }

  void Execute(std::uint32_t command_buffer,
               const std::vector<std::vector<std::uint8_t>>& sources) {
    for (const Command& command : command_buffers[command_buffer].commands) {
      switch (command.kind) {
        case Command::Kind::kUploads:
          for (std::size_t index : command.uploads) {
            WriteUpload(command_buffers[command_buffer].uploads[index]);
          }
          break;
        case Command::Kind::kRawCopy:
          for (const VkBufferImageCopy& region : command.raw_regions) {
            RawCopy(command.image, region, sources[command.source]);
          }
          break;
        case Command::Kind::kCopyImage:
          for (const VkImageCopy& region : command.copies) {
            CopyRegion(command.image, command.image2, region);
          }
          transfers.push_back({false,
                               true,
                               command.image,
                               command.image2,
                               command.copies,
                               {},
                               VK_FILTER_NEAREST});
          break;
        case Command::Kind::kBlit: {
          Transfer transfer{true, true,          command.image, command.image2,
                            {},   command.blits, command.filter};
          for (const VkImageBlit& blit : command.blits) {
            const std::int32_t width =
                blit.srcOffsets[1].x - blit.srcOffsets[0].x;
            const std::int32_t height =
                blit.srcOffsets[1].y - blit.srcOffsets[0].y;
            if (command.filter != VK_FILTER_NEAREST || width <= 0 ||
                height <= 0 ||
                width != blit.dstOffsets[1].x - blit.dstOffsets[0].x ||
                height != blit.dstOffsets[1].y - blit.dstOffsets[0].y) {
              transfer.executed = false;
              continue;
            }
            CopyRegion(command.image, command.image2,
                       {blit.srcSubresource,
                        {blit.srcOffsets[0].x, blit.srcOffsets[0].y, 0},
                        blit.dstSubresource,
                        {blit.dstOffsets[0].x, blit.dstOffsets[0].y, 0},
                        {static_cast<std::uint32_t>(width),
                         static_cast<std::uint32_t>(height), 1}});
          }
          transfers.push_back(std::move(transfer));
          break;
        }
        case Command::Kind::kExecute:
          for (std::uint32_t secondary : command.secondaries) {
            Execute(secondary, sources);
          }
          break;
      }
    }
  }
};

Etc2EmulationModel::Etc2EmulationModel(const Scenario& scenario)
    : state_(std::make_unique<State>()) {
  state_->upscale = scenario.upscale;
  state_->device_emulated = scenario.emulated;
  state_->overrides = scenario.overrides;
  state_->command_buffers.resize(scenario.command_buffers.size());
  for (const ScenarioImage& spec : scenario.images) {
    State::Image image;
    image.spec = spec;
    image.host = HostShapeFor(spec, scenario.upscale, scenario.emulated);
    image.emulated =
        scenario.emulated && LookupEmulated(spec.format, &image.format);
    image.scale = image.host.width / spec.width;
    LookupFormatBlock(image.host.format, &image.block);
    for (std::uint32_t mip = 0; mip < spec.mip_levels; ++mip) {
      const std::size_t bytes =
          static_cast<std::size_t>(
              DivideUp(Mip(image.host.width, mip), image.block.width)) *
          DivideUp(Mip(image.host.height, mip), image.block.height) *
          image.block.bytes;
      for (std::uint32_t layer = 0; layer < spec.array_layers; ++layer) {
        image.subresources.emplace_back(bytes, kImageFill);
      }
    }
    state_->images.push_back(std::move(image));
  }
}

Etc2EmulationModel::~Etc2EmulationModel() = default;

const HostImageShape& Etc2EmulationModel::host_shape(
    std::uint32_t image) const {
  return state_->images[image].host;
}

bool Etc2EmulationModel::CopyBufferToImage(
    std::uint32_t command_buffer, std::uint32_t image, std::uint32_t source,
    const std::vector<VkBufferImageCopy>& regions) {
  State::CommandBuffer& record = state_->command_buffers[command_buffer];
  const State::Image& target = state_->images[image];
  State::Command command;
  if (!target.emulated) {
    command.kind = State::Command::Kind::kRawCopy;
    command.image = image;
    command.source = source;
    command.raw_regions = regions;
    record.commands.push_back(std::move(command));
    return true;
  }
  std::vector<State::Upload> planned;
  for (const VkBufferImageCopy& region : regions) {
    State::Upload upload;
    upload.image = image;
    upload.source = source;
    if (!state_->Plan(target, region, &upload)) {
      return false;
    }
    planned.push_back(std::move(upload));
  }
  command.kind = State::Command::Kind::kUploads;
  for (State::Upload& upload : planned) {
    command.uploads.push_back(record.uploads.size());
    record.uploads.push_back(std::move(upload));
  }
  record.commands.push_back(std::move(command));
  return true;
}

void Etc2EmulationModel::CopyImage(std::uint32_t command_buffer,
                                   std::uint32_t source,
                                   std::uint32_t destination,
                                   const std::vector<VkImageCopy>& regions) {
  State::CommandBuffer& record = state_->command_buffers[command_buffer];
  const State::Image& from = state_->images[source];
  const State::Image& to = state_->images[destination];
  const bool from_scaled = from.emulated && from.scale > 1;
  const bool to_scaled = to.emulated && to.scale > 1;
  State::Command command;
  command.image = source;
  command.image2 = destination;
  if (state_->device_emulated && state_->upscale > 1 &&
      (from_scaled || to_scaled)) {
    for (const VkImageCopy& region : regions) {
      VkImageBlit blit{};
      blit.srcSubresource = region.srcSubresource;
      blit.dstSubresource = region.dstSubresource;
      if (!state_->HostRegion(source, region.srcSubresource.mipLevel,
                              region.srcOffset, region.extent,
                              &blit.srcOffsets[0], &blit.srcOffsets[1]) ||
          !state_->HostRegion(destination, region.dstSubresource.mipLevel,
                              region.dstOffset, region.extent,
                              &blit.dstOffsets[0], &blit.dstOffsets[1])) {
        return;  // the whole copy is dropped
      }
      command.blits.push_back(blit);
    }
    command.kind = State::Command::Kind::kBlit;
    command.filter = from_scaled && to_scaled && from.scale == to.scale
                         ? VK_FILTER_NEAREST
                         : VK_FILTER_LINEAR;
  } else {
    command.kind = State::Command::Kind::kCopyImage;
    command.copies = regions;
  }
  record.commands.push_back(std::move(command));
}

void Etc2EmulationModel::ExecuteCommands(
    std::uint32_t primary, const std::vector<std::uint32_t>& secondaries) {
  State::CommandBuffer& record = state_->command_buffers[primary];
  State::Command command;
  command.kind = State::Command::Kind::kExecute;
  command.secondaries = secondaries;
  for (std::uint32_t secondary : secondaries) {
    if (state_->device_emulated &&
        !state_->command_buffers[secondary].uploads.empty()) {
      record.linked.push_back(secondary);
    }
  }
  record.commands.push_back(std::move(command));
}

void Etc2EmulationModel::Reset(std::uint32_t command_buffer) {
  state_->command_buffers[command_buffer] = {};
}

void Etc2EmulationModel::AddOverride(std::uint64_t hash,
                                     std::shared_ptr<const RgbaImage> image) {
  state_->override_files[hash] = std::move(image);
}

void Etc2EmulationModel::Submit(
    const std::vector<std::vector<std::uint32_t>>& batches,
    const std::vector<std::vector<std::uint8_t>>& sources) {
  // Every upload of the submit is decoded first, in submit order, each
  // command buffer before the secondaries it runs ...
  for (const auto& batch : batches) {
    for (std::uint32_t command_buffer : batch) {
      std::vector<std::uint32_t> order = {command_buffer};
      const auto& linked = state_->command_buffers[command_buffer].linked;
      order.insert(order.end(), linked.begin(), linked.end());
      for (std::uint32_t index : order) {
        for (State::Upload& upload : state_->command_buffers[index].uploads) {
          state_->Emit(&upload, sources[upload.source]);
        }
      }
    }
  }
  // ... then the queue runs the command buffers.
  for (const auto& batch : batches) {
    for (std::uint32_t command_buffer : batch) {
      state_->Execute(command_buffer, sources);
    }
  }
}

const std::vector<std::uint8_t>& Etc2EmulationModel::Subresource(
    std::uint32_t image, std::uint32_t mip, std::uint32_t layer) const {
  const State::Image& record = state_->images[image];
  return record.subresources[mip * record.spec.array_layers + layer];
}

const std::vector<Etc2EmulationModel::Dump>& Etc2EmulationModel::dumps() const {
  return state_->dumps;
}

const std::vector<Etc2EmulationModel::Transfer>& Etc2EmulationModel::transfers()
    const {
  return state_->transfers;
}

std::vector<std::uint8_t> Etc2EmulationModel::TightCompressed(
    VkFormat format, const VkBufferImageCopy& region,
    const std::vector<std::uint8_t>& source) {
  const FormatBlock block = BlockOf(format);
  const std::uint32_t width = region.imageExtent.width;
  const std::uint32_t height = region.imageExtent.height;
  const std::size_t row_stride =
      static_cast<std::size_t>(
          DivideUp(region.bufferRowLength != 0 ? region.bufferRowLength : width,
                   block.width)) *
      block.bytes;
  const std::size_t layer_stride =
      row_stride * DivideUp(region.bufferImageHeight != 0
                                ? region.bufferImageHeight
                                : height,
                            block.height);
  const std::size_t packed_row =
      static_cast<std::size_t>(DivideUp(width, block.width)) * block.bytes;
  const std::uint32_t rows = DivideUp(height, block.height);
  std::vector<std::uint8_t> out;
  for (std::uint32_t layer = 0; layer < region.imageSubresource.layerCount;
       ++layer) {
    for (std::uint32_t row = 0; row < rows; ++row) {
      const std::uint8_t* from = source.data() + region.bufferOffset +
                                 layer * layer_stride + row * row_stride;
      out.insert(out.end(), from, from + packed_row);
    }
  }
  return out;
}

std::uint64_t Etc2EmulationModel::OverrideHash(
    const std::vector<std::uint8_t>& bytes) {
  // Override PNGs are named by the FNV-1a 64 hash of the mip-0 bytes.
  GoldenHasher hasher;
  hasher.Add(bytes.data(), bytes.size());
  return hasher.value();
}

}  // namespace mocktail::graphics::testing
