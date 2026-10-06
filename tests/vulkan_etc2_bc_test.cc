#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/texture_override.h"
#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "mocktail/graphics/vulkan_etc2_gpu.h"
#include "support/bc_reference.h"
#include "support/bc_reference_transcode.h"
#include "support/etc2_emulation_harness.h"
#include "support/etc2_emulation_scenarios.h"
#include "support/etc2_reference_decoder.h"
#include "support/fake_vulkan_device.h"
#include "support/resample_reference.h"

// ETC2->BC mode in the emulation (DESIGN C4.1-C4.4, CR-14, CR-15, CR-22,
// CR-24): which devices enter it, the BC host images and views it creates,
// the images it leaves RGBA8/R16, the format lists the host sees, and every
// emulation scenario (DESIGN §9.2) end to end on the CPU route, whose BC
// blocks must equal the references', and on the GPU route (its etc2_bc
// pass, and its rescue), which must give the same blocks (the shader itself
// is checked against the CPU encoder on a GPU, etc2_gpu_parity_test's
// BcParity). Off (the default, MOCKTAIL_ETC2_TO_BC unset or 0) nothing
// may change; vulkan_etc2_emulation_test keeps the default mode's golden
// texels. The physical-device queries come from a configurable fake
// instance next to the GPU-free fake device, or from the fake device's own.

namespace mocktail::graphics::testing {
namespace {

// What the fake physical device reports.
struct PhysicalDeviceModel {
  bool etc2 = false;  // textureCompressionETC2: false, so it is emulated
  bool bc = true;     // textureCompressionBC
  VkFormatFeatureFlags bc_features =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
  // One BC format's features, when set.
  VkFormat odd_format = VK_FORMAT_UNDEFINED;
  VkFormatFeatureFlags odd_features = 0;
  // The usages and flags vkGetPhysicalDeviceImageFormatProperties accepts
  // for BC formats, and their limits.
  VkImageUsageFlags bc_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT;
  VkImageCreateFlags bc_flags =
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  VkExtent3D bc_max_extent = {16384, 16384, 1};

  int feature_queries = 0;
  int format_queries = 0;
  std::vector<VkFormat> image_format_queries;
};

PhysicalDeviceModel* g_model = nullptr;

bool IsBc(VkFormat format) {
  return format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK &&
         format <= VK_FORMAT_BC7_SRGB_BLOCK;
}

VKAPI_ATTR void VKAPI_CALL ModelFeatures(VkPhysicalDevice,
                                         VkPhysicalDeviceFeatures* features) {
  ++g_model->feature_queries;
  *features = VkPhysicalDeviceFeatures{};
  features->textureCompressionETC2 = g_model->etc2 ? VK_TRUE : VK_FALSE;
  features->textureCompressionBC = g_model->bc ? VK_TRUE : VK_FALSE;
}

VKAPI_ATTR void VKAPI_CALL ModelFormatProperties(
    VkPhysicalDevice, VkFormat format, VkFormatProperties* properties) {
  ++g_model->format_queries;
  *properties = VkFormatProperties{};
  if (IsBc(format)) {
    properties->optimalTilingFeatures = format == g_model->odd_format
                                            ? g_model->odd_features
                                            : g_model->bc_features;
  }
}

VKAPI_ATTR VkResult VKAPI_CALL ModelImageFormatProperties(
    VkPhysicalDevice, VkFormat format, VkImageType type, VkImageTiling tiling,
    VkImageUsageFlags usage, VkImageCreateFlags flags,
    VkImageFormatProperties* properties) {
  g_model->image_format_queries.push_back(format);
  if (!IsBc(format) || type != VK_IMAGE_TYPE_2D ||
      tiling != VK_IMAGE_TILING_OPTIMAL || (usage & ~g_model->bc_usage) != 0 ||
      (flags & ~g_model->bc_flags) != 0) {
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }
  *properties = VkImageFormatProperties{};
  properties->maxExtent = g_model->bc_max_extent;
  properties->maxMipLevels = 15;
  properties->maxArrayLayers = 2048;
  properties->sampleCounts = VK_SAMPLE_COUNT_1_BIT;
  properties->maxResourceSize = 1ULL << 40;
  return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
ModelInstanceProcAddr(VkInstance, const char* name) {
  const struct {
    const char* name;
    PFN_vkVoidFunction function;
  } kProcs[] = {
      {"vkGetPhysicalDeviceFeatures",
       reinterpret_cast<PFN_vkVoidFunction>(&ModelFeatures)},
      {"vkGetPhysicalDeviceFormatProperties",
       reinterpret_cast<PFN_vkVoidFunction>(&ModelFormatProperties)},
      {"vkGetPhysicalDeviceImageFormatProperties",
       reinterpret_cast<PFN_vkVoidFunction>(&ModelImageFormatProperties)},
  };
  for (const auto& proc : kProcs) {
    if (std::strcmp(proc.name, name) == 0) {
      return proc.function;
    }
  }
  return nullptr;
}

// What vkCreateImage saw in a VkImageFormatListCreateInfo, recorded by a
// vkGetDeviceProcAddr that puts this in front of the fake's.
std::vector<VkFormat> g_host_view_formats;
bool g_host_saw_format_list = false;
PFN_vkCreateImage g_fake_create_image = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL
RecordingCreateImage(VkDevice device, const VkImageCreateInfo* info,
                     const VkAllocationCallbacks* allocator, VkImage* image) {
  const VkImageFormatListCreateInfo* list = FindImageFormatList(info->pNext);
  g_host_saw_format_list = list != nullptr;
  g_host_view_formats.clear();
  if (list != nullptr) {
    g_host_view_formats.assign(list->pViewFormats,
                               list->pViewFormats + list->viewFormatCount);
  }
  return g_fake_create_image(device, info, allocator, image);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
RecordingDeviceProcAddr(VkDevice device, const char* name) {
  const PFN_vkVoidFunction fake =
      FakeVulkanDevice::GetDeviceProcAddr()(device, name);
  if (std::strcmp(name, "vkCreateImage") == 0) {
    g_fake_create_image = reinterpret_cast<PFN_vkCreateImage>(fake);
    return reinterpret_cast<PFN_vkVoidFunction>(&RecordingCreateImage);
  }
  return fake;
}

const VkInstance kInstance = reinterpret_cast<VkInstance>(0x1157A);

struct FormatCase {
  VkFormat format;
  VkFormat rgba;
  VkFormat bc;
};

constexpr FormatCase kFormats[] = {
    {VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM,
     VK_FORMAT_BC1_RGB_UNORM_BLOCK},
    {VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB,
     VK_FORMAT_BC1_RGB_SRGB_BLOCK},
    {VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM,
     VK_FORMAT_BC1_RGBA_UNORM_BLOCK},
    {VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB,
     VK_FORMAT_BC1_RGBA_SRGB_BLOCK},
    {VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM,
     VK_FORMAT_BC3_UNORM_BLOCK},
    {VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB,
     VK_FORMAT_BC3_SRGB_BLOCK},
    {VK_FORMAT_EAC_R11_UNORM_BLOCK, VK_FORMAT_R16_UNORM,
     VK_FORMAT_BC4_UNORM_BLOCK},
    {VK_FORMAT_EAC_R11_SNORM_BLOCK, VK_FORMAT_R16_SNORM,
     VK_FORMAT_BC4_SNORM_BLOCK},
    {VK_FORMAT_EAC_R11G11_UNORM_BLOCK, VK_FORMAT_R16G16_UNORM,
     VK_FORMAT_BC5_UNORM_BLOCK},
    {VK_FORMAT_EAC_R11G11_SNORM_BLOCK, VK_FORMAT_R16G16_SNORM,
     VK_FORMAT_BC5_SNORM_BLOCK},
};

bool IsColour(VkFormat format) {
  return format <= VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
}

VkImageCreateInfo ImageInfo(VkFormat format, std::uint32_t width,
                            std::uint32_t height, std::uint32_t mips = 1) {
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {width, height, 1};
  info.mipLevels = mips;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  return info;
}

std::vector<std::string> Lines(const std::string& text) {
  std::vector<std::string> lines;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t end = text.find('\n', start);
    lines.push_back(text.substr(
        start, end == std::string::npos ? std::string::npos : end - start));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return lines;
}

int CountLines(const std::string& log, const std::string& prefix) {
  int count = 0;
  for (const std::string& line : Lines(log)) {
    count += line.rfind(prefix, 0) == 0 ? 1 : 0;
  }
  return count;
}

const char kRgbaLine[] =
    "  [vulkan] ETC2/EAC emulation enabled: compressed uploads decode to "
    "RGBA8/R16 host images";
const char kBcLine[] =
    "  [vulkan] ETC2/EAC emulation enabled: compressed uploads become "
    "BC1/BC3/BC4/BC5 host images (CPU transcode)";
// The same with the GPU route on (MOCKTAIL_ETC2_GPU's default).
const char kBcGpuLine[] =
    "  [vulkan] ETC2/EAC emulation enabled: compressed uploads become "
    "BC1/BC3/BC4/BC5 host images (GPU transcode)";
const char kUnavailable[] = "  [vulkan] ETC2->BC unavailable: ";

class Etc2BcModeTest : public ::testing::Test {
 protected:
  void SetUp() override { g_model = &model_; }
  void TearDown() override { g_model = nullptr; }

  static Etc2EmulationOptions Options(bool to_bc) {
    Etc2EmulationOptions options = CpuRouteOptions();
    options.upscale = 4;
    options.to_bc = to_bc;
    return options;
  }

  // The adapter's vkCreateDevice: the BC decision first (it enables the
  // feature), then the registration with the host instance.
  bool Register(VulkanEtc2Emulation* emulation, FakeVulkanDevice* fake,
                bool with_instance = true, bool decide_first = true) {
    bool bc = false;
    if (decide_first) {
      bc = emulation->PhysicalDeviceUsesBc(
          fake->physical_device(), &ModelFeatures, &ModelFormatProperties);
    }
    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    emulation->RegisterDevice(
        fake->device(), fake->physical_device(), !model_.etc2,
        fake->config().memory, fake->config().properties,
        fake->config().queue_families, device_info, &RecordingDeviceProcAddr,
        with_instance ? &ModelInstanceProcAddr : nullptr,
        with_instance ? kInstance : VK_NULL_HANDLE);
    return bc;
  }

  VkImage Create(VulkanEtc2Emulation* emulation, FakeVulkanDevice* fake,
                 const VkImageCreateInfo& info, VkImageCreateInfo* host) {
    VkImage image = VK_NULL_HANDLE;
    EXPECT_EQ(emulation->CreateImage(fake->device(), &info, nullptr, &image),
              VK_SUCCESS);
    EXPECT_TRUE(fake->GetImageInfo(image, host));
    return image;
  }

  PhysicalDeviceModel model_;
};

// MOCKTAIL_ETC2_TO_BC unset or 0: the images, views and log lines of the
// default mode, and no physical-device query for BC at all.
TEST_F(Etc2BcModeTest, OffChangesNothing) {
  FakeVulkanDevice fake;
  VulkanEtc2Emulation emulation(Options(false));
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(Register(&emulation, &fake));
  for (const FormatCase& test : kFormats) {
    for (const std::uint32_t size : {64U, 65U}) {
      VkImageCreateInfo info = ImageInfo(test.format, size, 32, 6);
      info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                   VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
      VkImageCreateInfo host{};
      const VkImage image = Create(&emulation, &fake, info, &host);
      EXPECT_EQ(host.format, test.rgba);
      EXPECT_EQ(host.flags, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
      const std::uint32_t scale = IsColour(test.format) && size <= 64 ? 4 : 1;
      EXPECT_EQ(host.extent.width, size * scale) << test.format;
      EXPECT_EQ(host.extent.height, 32 * scale) << test.format;
      VkImageViewCreateInfo view_info{};
      view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_info.image = image;
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = test.format;
      view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      VkImageView view = VK_NULL_HANDLE;
      ASSERT_EQ(
          emulation.CreateImageView(fake.device(), &view_info, nullptr, &view),
          VK_SUCCESS);
      VkImageViewCreateInfo host_view{};
      ASSERT_TRUE(fake.GetImageViewInfo(view, &host_view));
      EXPECT_EQ(host_view.format, test.rgba);
      fake.DestroyImageView(view);
      emulation.DestroyImage(fake.device(), image, nullptr);
    }
  }
  emulation.DestroyDevice(fake.device());
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(CountLines(log, kRgbaLine), 1) << log;
  EXPECT_EQ(log.find("ETC2->BC"), std::string::npos) << log;
  EXPECT_EQ(model_.format_queries, 0);
  EXPECT_TRUE(model_.image_format_queries.empty());
  // Asked again, as the format-property hooks do: still no.
  EXPECT_FALSE(emulation.PhysicalDeviceUsesBc(
      fake.physical_device(), &ModelFeatures, &ModelFormatProperties));
  EXPECT_EQ(model_.format_queries, 0);
  EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
}

TEST_F(Etc2BcModeTest, CreatesBcImagesAndViewsWithoutUpscale) {
  FakeVulkanDevice fake;
  Etc2EmulationOptions options = Options(true);
  options.frame_stats = true;
  VulkanEtc2Emulation emulation(options);
  ::testing::internal::CaptureStderr();
  EXPECT_TRUE(Register(&emulation, &fake));
  for (const FormatCase& test : kFormats) {
    for (const std::uint32_t size : {64U, 65U}) {
      VkImageCreateInfo info = ImageInfo(test.format, size, 32, 6);
      info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                   VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
      VkImageCreateInfo host{};
      const VkImage image = Create(&emulation, &fake, info, &host);
      EXPECT_EQ(host.format, test.bc) << test.format;
      EXPECT_EQ(host.flags, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
      EXPECT_EQ(host.extent.width, size) << test.format;  // never upscaled
      EXPECT_EQ(host.extent.height, 32U);
      EXPECT_EQ(host.mipLevels, 6U);
      VkImageViewCreateInfo view_info{};
      view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_info.image = image;
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = test.format;
      view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      VkImageView view = VK_NULL_HANDLE;
      ASSERT_EQ(
          emulation.CreateImageView(fake.device(), &view_info, nullptr, &view),
          VK_SUCCESS);
      VkImageViewCreateInfo host_view{};
      ASSERT_TRUE(fake.GetImageViewInfo(view, &host_view));
      EXPECT_EQ(host_view.format, test.bc);
      fake.DestroyImageView(view);
      // An aliasing view in another format is passed as is.
      view_info.format = VK_FORMAT_R32G32_UINT;
      ASSERT_EQ(
          emulation.CreateImageView(fake.device(), &view_info, nullptr, &view),
          VK_SUCCESS);
      ASSERT_TRUE(fake.GetImageViewInfo(view, &host_view));
      EXPECT_EQ(host_view.format, VK_FORMAT_R32G32_UINT);
      fake.DestroyImageView(view);
      emulation.DestroyImage(fake.device(), image, nullptr);
    }
  }
  // The SRGB view of a UNORM image: the BC format of the view's own format.
  VkImageCreateInfo info =
      ImageInfo(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 16, 16);
  info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  VkImageCreateInfo host{};
  const VkImage image = Create(&emulation, &fake, info, &host);
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK;
  view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView view = VK_NULL_HANDLE;
  ASSERT_EQ(
      emulation.CreateImageView(fake.device(), &view_info, nullptr, &view),
      VK_SUCCESS);
  VkImageViewCreateInfo host_view{};
  ASSERT_TRUE(fake.GetImageViewInfo(view, &host_view));
  EXPECT_EQ(host_view.format, VK_FORMAT_BC3_SRGB_BLOCK);
  fake.DestroyImageView(view);
  emulation.DestroyImage(fake.device(), image, nullptr);
  // One query per distinct (format, type, tiling, usage, flags), cached
  // after it: the last image shares the RGBA8 UNORM one's.
  EXPECT_EQ(model_.image_format_queries.size(), 10U);
  emulation.DestroyDevice(fake.device());
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(CountLines(log, kBcLine), 1) << log;
  EXPECT_EQ(CountLines(log, kRgbaLine), 0) << log;
  EXPECT_EQ(log.find("ETC2->BC"), std::string::npos) << log;
  EXPECT_NE(log.find(" bc_images=21 bc_fallback=0 "), std::string::npos) << log;
  EXPECT_EQ(fake.counts().images.live(), 0U);
  EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
}

// [CR-14] An image BC cannot hold keeps the RGBA8/R16 host, without the
// upscale of the default mode, and its views follow it.
TEST_F(Etc2BcModeTest, ImagesBcCannotServeStayRgba) {
  FakeVulkanDevice fake;
  Etc2EmulationOptions options = Options(true);
  options.frame_stats = true;
  VulkanEtc2Emulation emulation(options);
  ::testing::internal::CaptureStderr();
  ASSERT_TRUE(Register(&emulation, &fake));
  struct Case {
    const char* name;
    VkImageCreateInfo info;
    const char* reason;
  };
  std::vector<Case> cases;
  VkImageCreateInfo storage =
      ImageInfo(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 32, 32);
  storage.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
  cases.push_back({"storage", storage, "the device takes no BC image"});
  VkImageCreateInfo attachment =
      ImageInfo(VK_FORMAT_EAC_R11_UNORM_BLOCK, 32, 32);
  attachment.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  cases.push_back({"attachment", attachment, "the device takes no BC image"});
  VkImageCreateInfo extended =
      ImageInfo(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 32, 32);
  extended.flags = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
  cases.push_back({"extended usage", extended, "the device takes no BC image"});
  model_.bc_max_extent = {4096, 4096, 1};
  cases.push_back({"too large",
                   ImageInfo(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 8192, 16),
                   "it exceeds the BC format's limits"});
  const VkFormat uint_view[] = {VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
                                VK_FORMAT_R32G32_UINT};
  VkImageFormatListCreateInfo list{};
  list.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
  list.viewFormatCount = 2;
  list.pViewFormats = uint_view;
  VkImageCreateInfo mutable_uint =
      ImageInfo(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 32, 32);
  mutable_uint.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  mutable_uint.pNext = &list;
  cases.push_back({"uncompressed view format", mutable_uint,
                   "its format list names a format that is not ETC2/EAC"});
  for (const Case& test : cases) {
    SCOPED_TRACE(test.name);
    VkImageCreateInfo host{};
    const VkImage image = Create(&emulation, &fake, test.info, &host);
    Etc2EmulatedFormat mapped;
    ASSERT_TRUE(LookupEmulatedEtc2Format(test.info.format, &mapped));
    EXPECT_EQ(host.format, mapped.host_format);
    EXPECT_EQ(host.extent.width, test.info.extent.width);  // not upscaled
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = test.info.format;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    ASSERT_EQ(
        emulation.CreateImageView(fake.device(), &view_info, nullptr, &view),
        VK_SUCCESS);
    VkImageViewCreateInfo host_view{};
    ASSERT_TRUE(fake.GetImageViewInfo(view, &host_view));
    EXPECT_EQ(host_view.format, mapped.host_format);
    fake.DestroyImageView(view);
    emulation.DestroyImage(fake.device(), image, nullptr);
  }
  // A second storage image asks the host nothing new.
  const std::size_t queries = model_.image_format_queries.size();
  VkImageCreateInfo host{};
  emulation.DestroyImage(fake.device(),
                         Create(&emulation, &fake, storage, &host), nullptr);
  EXPECT_EQ(model_.image_format_queries.size(), queries);
  emulation.DestroyDevice(fake.device());
  const std::string log = ::testing::internal::GetCapturedStderr();
  // A few are logged (four per process), each with its reason.
  int logged = 0;
  for (const std::string& line : Lines(log)) {
    if (line.rfind("  [vulkan] ETC2->BC: a ", 0) != 0) {
      continue;
    }
    ++logged;
    EXPECT_NE(line.find("stays RGBA8/R16: "), std::string::npos) << line;
    bool known = false;
    for (const Case& test : cases) {
      known = known || line.find(test.reason) != std::string::npos;
    }
    EXPECT_TRUE(known) << line;
  }
  EXPECT_LE(logged, 4);
  EXPECT_NE(log.find(" bc_images=0 bc_fallback=6 "), std::string::npos) << log;
  EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
}

// [CR-14] One submit of a BC device uploads into a BC image and into an
// image BC cannot hold (storage): the first gets the references' BC blocks,
// the second the default mode's texels without the x4 upscale, both from
// one staging chunk on the CPU route. With the GPU route on, one pass in
// front of the batch serves both: a decode dispatch for the fallback image
// and an etc2_bc dispatch for the BC image, with the same results.
TEST_F(Etc2BcModeTest, UploadsFillBcAndFallbackImagesOfOneDevice) {
  for (const bool gpu_route : {false, true}) {
    SCOPED_TRACE(gpu_route ? "GPU route" : "CPU route");
    FakeVulkanDevice fake;
    Etc2EmulationOptions options = Options(true);
    options.gpu = gpu_route;
    VulkanEtc2Emulation emulation(options);
    ::testing::internal::CaptureStderr();
    ASSERT_TRUE(Register(&emulation, &fake));
    const VkDevice device = fake.device();
    const VkQueue queue = fake.GetQueue(0, 0);
    emulation.RegisterQueue(device, queue, 0, 0);
    // The adapter's submit: pre-submits, the host submit, FinishSubmit.
    const auto submit = [&](std::uint32_t count, const VkSubmitInfo* info) {
      Etc2SubmitPlan plan;
      emulation.PrepareSubmit(queue, count, info, VK_NULL_HANDLE, &plan);
      for (const Etc2PreSubmit& pre : plan.pre_submits()) {
        EXPECT_EQ(fake.QueueSubmit(queue, 1, &pre.info, pre.fence), VK_SUCCESS);
      }
      const VkResult result =
          fake.QueueSubmit(queue, plan.count(count), plan.submits(info),
                           plan.fence(VK_NULL_HANDLE));
      EXPECT_EQ(result, VK_SUCCESS);
      emulation.FinishSubmit(&plan, result, [&](VkFence fence) {
        return fake.QueueSubmit(queue, 0, nullptr, fence);
      });
    };
    constexpr VkCommandPoolCreateFlags kPoolFlags =
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    const VkCommandPool pool = fake.CreateCommandPool(0, kPoolFlags);
    emulation.RegisterCommandPool(device, pool, 0, kPoolFlags);
    // Two 32x32 ETC2 RGB8 textures: 64 blocks, 512 bytes each.
    const std::vector<std::uint8_t> bytes = SeededBytes(0xB0C5, 1024);
    VkBuffer source = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    ASSERT_EQ(
        fake.CreateBuffer(1024, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &source),
        VK_SUCCESS);
    ASSERT_EQ(fake.AllocateMemory(1024, 3, &memory), VK_SUCCESS);
    ASSERT_EQ(emulation.BindBufferMemory(device, source, memory, 0),
              VK_SUCCESS);
    void* data = nullptr;
    ASSERT_EQ(emulation.MapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &data),
              VK_SUCCESS);
    std::memcpy(data, bytes.data(), bytes.size());
    VkImageCreateInfo bc_info =
        ImageInfo(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 32, 32);
    VkImageCreateInfo storage_info = bc_info;
    storage_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageCreateInfo host{};
    const VkImage bc_image = Create(&emulation, &fake, bc_info, &host);
    EXPECT_EQ(host.format, VK_FORMAT_BC1_RGB_UNORM_BLOCK);
    const VkImage rgba_image = Create(&emulation, &fake, storage_info, &host);
    EXPECT_EQ(host.format, VK_FORMAT_R8G8B8A8_UNORM);
    EXPECT_EQ(host.extent.width, 32U);  // no x4 upscale on a BC device
    Etc2GpuDevice* const gpu = Etc2GpuDevice::Find(device);
    EXPECT_EQ(gpu != nullptr, gpu_route);
    if (gpu != nullptr) {
      // The first image started the build; the selftest runs with empty
      // submits, the BC pipeline's part included.
      ASSERT_TRUE(gpu->WaitForBuild(std::chrono::seconds(60)));
      for (int round = 0;
           round < 3 && gpu->state() != Etc2GpuDevice::State::kReady; ++round) {
        submit(0, nullptr);
      }
      ASSERT_EQ(gpu->state(), Etc2GpuDevice::State::kReady);
    }
    const FakeVulkanDevice::Counts before = fake.counts();
    const VkCommandBuffer command_buffer =
        fake.AllocateCommandBuffer(pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    emulation.BeginCommandBuffer(command_buffer, pool,
                                 VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    fake.BeginCommandBuffer(command_buffer,
                            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {32, 32, 1};
    emulation.CmdCopyBufferToImage(device, command_buffer, source, bc_image,
                                   VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    region.bufferOffset = 512;
    emulation.CmdCopyBufferToImage(device, command_buffer, source, rgba_image,
                                   VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    fake.EndCommandBuffer(command_buffer);
    VkSubmitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &command_buffer;
    submit(1, &info);

    const std::vector<std::uint8_t>* bc_blocks =
        fake.ImageSubresource(bc_image, 0, 0);
    ASSERT_NE(bc_blocks, nullptr);
    EXPECT_EQ(*bc_blocks, reference::BcFromEtc(EtcFormat::kEtc2Rgb8,
                                               bytes.data(), 512, true));
    std::vector<std::uint8_t> decoded(32 * 32 * 4);
    reference::DecodeEtcImage(reference::EtcFormat::kEtc2Rgb8,
                              bytes.data() + 512, 512, 32, 32, decoded.data(),
                              decoded.size());
    const std::vector<std::uint8_t>* texels =
        fake.ImageSubresource(rgba_image, 0, 0);
    ASSERT_NE(texels, nullptr);
    EXPECT_EQ(*texels, decoded);
    const auto recorded = fake.RecordedBufferCopies(command_buffer);
    ASSERT_EQ(recorded.size(), 2U);
    EXPECT_EQ(recorded[0].buffer, recorded[1].buffer);
    const FakeVulkanDevice::Counts& after = fake.counts();
    if (gpu_route) {
      // Out of the output arena: 256-byte aligned ranges in one chunk.
      EXPECT_EQ(recorded[1].regions[0].bufferOffset -
                    recorded[0].regions[0].bufferOffset,
                512U);
      EXPECT_EQ(gpu->stats().passes, 1U);
      EXPECT_EQ(after.dispatches - before.dispatches, 2U);
      EXPECT_EQ(after.bc_blocks - before.bc_blocks, 64U);
      EXPECT_EQ(after.decoded_blocks - before.decoded_blocks, 64U);
    } else {
      EXPECT_EQ(recorded[0].regions[0].bufferOffset, 0U);
      EXPECT_EQ(recorded[1].regions[0].bufferOffset, 512U);  // BC: 512 bytes
      EXPECT_EQ(after.dispatches, 0U);
    }

    emulation.ReleaseCommandBuffer(command_buffer);
    fake.FreeCommandBuffer(command_buffer);
    fake.DestroyCommandPool(pool);
    emulation.ForgetCommandPool(device, pool);
    emulation.DestroyImage(device, bc_image, nullptr);
    emulation.DestroyImage(device, rgba_image, nullptr);
    emulation.UnmapMemory(device, memory);
    emulation.DestroyBuffer(device, source, nullptr);
    emulation.FreeMemory(device, memory, nullptr);
    emulation.DestroyDevice(device);
    const std::string log = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(log.find("could not"), std::string::npos) << log;
    EXPECT_EQ(CountLines(log, gpu_route ? kBcGpuLine : kBcLine), 1) << log;
    EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
    EXPECT_EQ(fake.counts().entry_point_buffers.live(), 0U);
    EXPECT_EQ(fake.counts().pipelines.live(), 0U);
  }
}

// DESIGN C3.4: a selftest whose BC part differs from the CPU encoder keeps
// the CPU route for the session (decode included), and says which format
// and block.
TEST_F(Etc2BcModeTest, ABcSelftestMismatchKeepsTheCpuRoute) {
  FakeVulkanDevice fake;
  Etc2EmulationOptions options = Options(true);
  options.gpu = true;
  VulkanEtc2Emulation emulation(options);
  Etc2GpuTesting().corrupt_bc_selftest = true;
  ::testing::internal::CaptureStderr();
  ASSERT_TRUE(Register(&emulation, &fake));
  const VkDevice device = fake.device();
  const VkQueue queue = fake.GetQueue(0, 0);
  emulation.RegisterQueue(device, queue, 0, 0);
  VkImageCreateInfo host{};
  const VkImage image =
      Create(&emulation, &fake,
             ImageInfo(VK_FORMAT_EAC_R11G11_UNORM_BLOCK, 16, 16), &host);
  Etc2GpuDevice* const gpu = Etc2GpuDevice::Find(device);
  ASSERT_NE(gpu, nullptr);
  ASSERT_TRUE(gpu->WaitForBuild(std::chrono::seconds(60)));
  for (int round = 0; round < 3; ++round) {
    Etc2SubmitPlan plan;
    emulation.PrepareSubmit(queue, 0, nullptr, VK_NULL_HANDLE, &plan);
    for (const Etc2PreSubmit& pre : plan.pre_submits()) {
      EXPECT_EQ(fake.QueueSubmit(queue, 1, &pre.info, pre.fence), VK_SUCCESS);
    }
    const VkResult result =
        fake.QueueSubmit(queue, plan.count(0), plan.submits(nullptr),
                         plan.fence(VK_NULL_HANDLE));
    emulation.FinishSubmit(&plan, result, [&](VkFence fence) {
      return fake.QueueSubmit(queue, 0, nullptr, fence);
    });
  }
  Etc2GpuTesting().corrupt_bc_selftest = false;
  EXPECT_EQ(gpu->state(), Etc2GpuDevice::State::kFailed);
  EXPECT_NE(gpu->failure().find("selftest mismatch: BC from EAC_RG11 block "),
            std::string::npos)
      << gpu->failure();
  emulation.DestroyImage(device, image, nullptr);
  emulation.DestroyDevice(device);
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("[vulkan] ETC2 GPU decode unavailable: selftest mismatch: "
                     "BC from EAC_RG11"),
            std::string::npos)
      << log;
  EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
  EXPECT_EQ(fake.counts().pipelines.live(), 0U);
  EXPECT_EQ(fake.counts().fences.live(), 0U);
}

// [CR-15] The host sees the host formats in an image's format list, in
// both modes, and Roblox's own structs are as they were afterwards.
TEST_F(Etc2BcModeTest, FormatListsNameTheHostFormats) {
  for (const bool to_bc : {false, true}) {
    SCOPED_TRACE(to_bc ? "BC mode" : "default mode");
    FakeVulkanDevice fake;
    VulkanEtc2Emulation emulation(Options(to_bc));
    ::testing::internal::CaptureStderr();
    Register(&emulation, &fake);
    const VkFormat views[] = {VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
                              VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK};
    const VkFormat expected[] = {
        to_bc ? VK_FORMAT_BC1_RGBA_UNORM_BLOCK : VK_FORMAT_R8G8B8A8_UNORM,
        to_bc ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_R8G8B8A8_SRGB};
    VkImageFormatListCreateInfo list{};
    list.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
    list.viewFormatCount = 2;
    list.pViewFormats = views;
    // First in the chain (linked in as a copy), then behind another struct
    // (its array swapped for the call).
    VkExternalMemoryImageCreateInfo external{};
    external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    for (const bool behind : {false, true}) {
      SCOPED_TRACE(behind ? "behind another struct" : "first");
      VkImageCreateInfo info =
          ImageInfo(VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, 128, 128);
      info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
      external.pNext = behind ? &list : nullptr;
      list.pNext = nullptr;
      info.pNext = behind ? static_cast<const void*>(&external)
                          : static_cast<const void*>(&list);
      g_host_view_formats.clear();
      VkImageCreateInfo host{};
      const VkImage image = Create(&emulation, &fake, info, &host);
      EXPECT_TRUE(g_host_saw_format_list);
      ASSERT_EQ(g_host_view_formats.size(), 2U);
      EXPECT_EQ(g_host_view_formats[0], expected[0]);
      EXPECT_EQ(g_host_view_formats[1], expected[1]);
      EXPECT_EQ(list.pViewFormats, views);
      EXPECT_EQ(views[0], VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK);
      EXPECT_EQ(info.pNext, behind ? static_cast<const void*>(&external)
                                   : static_cast<const void*>(&list));
      EXPECT_EQ(external.pNext, behind ? &list : nullptr);
      emulation.DestroyImage(fake.device(), image, nullptr);
    }
    // A list without ETC2/EAC entries is passed on as it is.
    const VkFormat plain[] = {VK_FORMAT_R8G8B8A8_UNORM};
    list.pViewFormats = plain;
    list.viewFormatCount = 1;
    VkImageCreateInfo info = ImageInfo(VK_FORMAT_EAC_R11_UNORM_BLOCK, 32, 32);
    info.pNext = &list;
    VkImageCreateInfo host{};
    emulation.DestroyImage(fake.device(),
                           Create(&emulation, &fake, info, &host), nullptr);
    ASSERT_EQ(g_host_view_formats.size(), 1U);
    EXPECT_EQ(g_host_view_formats[0], VK_FORMAT_R8G8B8A8_UNORM);
    emulation.DestroyDevice(fake.device());
    ::testing::internal::GetCapturedStderr();
    EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
  }
}

TEST_F(Etc2BcModeTest, UnavailableWithoutWhatBcNeeds) {
  struct Case {
    const char* name;
    void (*configure)(PhysicalDeviceModel*);
    const char* reason;
  };
  const Case cases[] = {
      {"no BC feature", [](PhysicalDeviceModel* model) { model->bc = false; },
       "the device has no textureCompressionBC"},
      // [CR-22] BLIT_SRC, which the ETC2 formats advertise from RGBA8.
      {"BC4 without BLIT_SRC",
       [](PhysicalDeviceModel* model) {
         model->odd_format = VK_FORMAT_BC4_SNORM_BLOCK;
         model->odd_features =
             model->bc_features & ~VK_FORMAT_FEATURE_BLIT_SRC_BIT;
       },
       "format 140 lacks optimal-tiling features 0x400"},
      {"BC1 without linear filtering",
       [](PhysicalDeviceModel* model) {
         model->odd_format = VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
         model->odd_features =
             model->bc_features &
             ~VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
       },
       "format 134 lacks optimal-tiling features 0x1000"},
      {"native ETC2", [](PhysicalDeviceModel* model) { model->etc2 = true; },
       "not needed, the graphics card reads ETC2 itself"},
  };
  for (const Case& test : cases) {
    SCOPED_TRACE(test.name);
    model_ = PhysicalDeviceModel{};
    test.configure(&model_);
    FakeVulkanDevice fake;
    VulkanEtc2Emulation emulation(Options(true));
    ::testing::internal::CaptureStderr();
    EXPECT_FALSE(Register(&emulation, &fake));
    // Decided once: asking again queries and logs nothing more.
    const int features = model_.feature_queries;
    EXPECT_FALSE(emulation.PhysicalDeviceUsesBc(
        fake.physical_device(), &ModelFeatures, &ModelFormatProperties));
    EXPECT_EQ(model_.feature_queries, features);
    VkImageCreateInfo host{};
    emulation.DestroyImage(
        fake.device(),
        Create(&emulation, &fake,
               ImageInfo(VK_FORMAT_EAC_R11G11_SNORM_BLOCK, 16, 16), &host),
        nullptr);
    EXPECT_EQ(host.format, model_.etc2 ? VK_FORMAT_EAC_R11G11_SNORM_BLOCK
                                       : VK_FORMAT_R16G16_SNORM);
    emulation.DestroyDevice(fake.device());
    const std::string log = ::testing::internal::GetCapturedStderr();
    EXPECT_EQ(CountLines(log, std::string(kUnavailable) + test.reason +
                                  "; ETC2 textures stay RGBA8/R16"),
              1)
        << log;
    EXPECT_EQ(CountLines(log, kBcLine), 0) << log;
  }
}

// A device registered without the physical-device functions and no
// decision before it (not what the adapter does) cannot be BC.
TEST_F(Etc2BcModeTest, RegistrationWithoutQueriesStaysRgba) {
  FakeVulkanDevice fake;
  VulkanEtc2Emulation emulation(Options(true));
  ::testing::internal::CaptureStderr();
  Register(&emulation, &fake, false, false);
  VkImageCreateInfo host{};
  emulation.DestroyImage(
      fake.device(),
      Create(&emulation, &fake,
             ImageInfo(VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, 16, 16), &host),
      nullptr);
  EXPECT_EQ(host.format, VK_FORMAT_R8G8B8A8_SRGB);
  emulation.DestroyDevice(fake.device());
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(CountLines(log, std::string(kUnavailable) +
                                "the device's features cannot be queried; "
                                "ETC2 textures stay RGBA8/R16"),
            1)
      << log;
  EXPECT_EQ(CountLines(log, kRgbaLine), 1) << log;
}

TEST(Etc2BcFormatTest, MapsEveryFormatToItsBcHost) {
  for (const FormatCase& test : kFormats) {
    Etc2EmulatedFormat rgba;
    Etc2EmulatedFormat bc;
    ASSERT_TRUE(LookupEmulatedEtc2Format(test.format, &rgba));
    ASSERT_TRUE(LookupEmulatedEtc2Format(test.format, true, &bc));
    EXPECT_EQ(rgba.host_format, test.rgba);
    EXPECT_EQ(rgba.bc_host_format, test.bc);
    EXPECT_EQ(bc.host_format, test.bc);
    EXPECT_EQ(bc.etc_format, rgba.etc_format);
    Etc2EmulatedFormat again;
    ASSERT_TRUE(LookupEmulatedEtc2Format(test.format, false, &again));
    EXPECT_EQ(again.host_format, test.rgba);
    // Same block size, so offsets, strides and copy classes stay.
    FormatBlock etc2_block;
    FormatBlock bc_block;
    ASSERT_TRUE(LookupFormatBlock(test.format, &etc2_block));
    ASSERT_TRUE(LookupFormatBlock(test.bc, &bc_block));
    EXPECT_EQ(etc2_block.bytes, bc_block.bytes);
  }
  EXPECT_FALSE(
      LookupEmulatedEtc2Format(VK_FORMAT_BC1_RGB_UNORM_BLOCK, true, nullptr));
  EXPECT_EQ(Etc2BcRequiredFormatFeatures(),
            EmulatedEtc2FormatFeatures(~VkFormatFeatureFlags{0}));
  EXPECT_NE(Etc2BcRequiredFormatFeatures() & VK_FORMAT_FEATURE_BLIT_SRC_BIT,
            0U);
}

TEST(Etc2BcFormatTest, FormatListRewriteRestoresEverything) {
  const VkFormat views[] = {VK_FORMAT_EAC_R11_SNORM_BLOCK,
                            VK_FORMAT_R32_SFLOAT};
  VkImageFormatListCreateInfo list{};
  list.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
  list.viewFormatCount = 2;
  list.pViewFormats = views;
  EXPECT_FALSE(Etc2FormatListAllowsBc(&list));
  EXPECT_TRUE(Etc2FormatListAllowsBc(nullptr));
  VkExternalMemoryImageCreateInfo external{};
  external.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  external.pNext = &list;
  {
    const void* host_next = &external;
    const Etc2FormatListRewrite rewrite(&host_next, true);
    EXPECT_TRUE(rewrite.found());
    EXPECT_TRUE(rewrite.rewritten());
    EXPECT_EQ(host_next, &external);
    EXPECT_NE(list.pViewFormats, views);
    EXPECT_EQ(list.pViewFormats[0], VK_FORMAT_BC4_SNORM_BLOCK);
    EXPECT_EQ(list.pViewFormats[1], VK_FORMAT_R32_SFLOAT);
  }
  EXPECT_EQ(list.pViewFormats, views);
  {
    const void* host_next = &list;
    const Etc2FormatListRewrite rewrite(&host_next, false);
    ASSERT_NE(host_next, &list);
    const auto* copy =
        static_cast<const VkImageFormatListCreateInfo*>(host_next);
    EXPECT_EQ(copy->pViewFormats[0], VK_FORMAT_R16_SNORM);
    EXPECT_EQ(copy->pNext, list.pNext);
    EXPECT_EQ(list.pViewFormats, views);
  }
  const VkFormat plain[] = {VK_FORMAT_R16_SNORM};
  list.pViewFormats = plain;
  list.viewFormatCount = 1;
  const void* host_next = &list;
  const Etc2FormatListRewrite rewrite(&host_next, true);
  EXPECT_TRUE(rewrite.found());
  EXPECT_FALSE(rewrite.rewritten());
  EXPECT_EQ(host_next, &list);
  const void* empty = nullptr;
  const Etc2FormatListRewrite none(&empty, true);
  EXPECT_FALSE(none.found());
}

// ---- every scenario end to end in BC mode ----------------------------------

std::vector<std::string> ScenarioNames() {
  std::vector<std::string> names;
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    names.push_back(scenario.name);
  }
  return names;
}

// Every fake-device error must be one the scenario expects in BC mode, and
// each expected one must still happen.
void ExpectOnlyBcModeErrors(const FakeVulkanDevice& fake,
                            const Scenario& scenario) {
  std::vector<std::string> allowed = scenario.allowed_fake_errors;
  allowed.insert(allowed.end(), scenario.bc_mode_fake_errors.begin(),
                 scenario.bc_mode_fake_errors.end());
  std::vector<bool> seen(allowed.size(), false);
  for (const std::string& error : fake.errors()) {
    bool known = false;
    for (std::size_t index = 0; index < allowed.size(); ++index) {
      if (error.find(allowed[index]) != std::string::npos) {
        known = true;
        seen[index] = true;
      }
    }
    if (!known) {
      ADD_FAILURE() << "fake device: " << error;
    }
  }
  for (std::size_t index = 0; index < seen.size(); ++index) {
    EXPECT_TRUE(seen[index])
        << "expected fake-device error no longer happens: " << allowed[index];
  }
}

void ExpectEverythingReleased(const FakeVulkanDevice& fake) {
  const FakeVulkanDevice::Counts& counts = fake.counts();
  EXPECT_EQ(counts.entry_point_buffers.live(), 0U);
  EXPECT_EQ(counts.entry_point_memory.live(), 0U);
  EXPECT_EQ(counts.buffers.live(), 0U);
  EXPECT_EQ(counts.memory.live(), 0U);
  EXPECT_EQ(counts.images.live(), 0U);
  EXPECT_EQ(counts.command_pools.live(), 0U);
  EXPECT_EQ(counts.command_buffers.live(), 0U);
  EXPECT_EQ(counts.live_mappings, 0U);
}

const char* TeardownName(ScenarioTeardown teardown) {
  switch (teardown) {
    case ScenarioTeardown::kOrderly:
      return "Orderly";
    case ScenarioTeardown::kResetPool:
      return "ResetPool";
    case ScenarioTeardown::kDestroyPool:
      return "DestroyPool";
    case ScenarioTeardown::kDestroyDeviceFirst:
      return "DestroyDeviceFirst";
  }
  return "?";
}

// Every scenario with MOCKTAIL_ETC2_TO_BC=1 on the fake device (BC, no
// ETC2), with the host staging arena and with one-off staging buffers (its
// fallback), with every teardown: the
// host images are BC images of the app's extent (BC mode never upscales,
// so the upscale scenarios run as plain uploads), their blocks equal the
// references', every copy and blit is what the model expects, and the
// emulation gives back everything it made.
class Etc2BcScenarioTest
    : public ::testing::TestWithParam<
          std::tuple<std::string, bool, ScenarioTeardown>> {
 protected:
  void TearDown() override { Etc2EmulationTesting().no_host_staging = false; }
};

TEST_P(Etc2BcScenarioTest, MatchesModelAndGolden) {
  const auto& [name, arena, teardown] = GetParam();
  Etc2EmulationTesting().no_host_staging = !arena;
  const Scenario* scenario = FindEtc2EmulationScenario(name);
  ASSERT_NE(scenario, nullptr);
  FakeVulkanDevice fake;
  FakeEmulationBackend backend(&fake);
  ScenarioOptions options;
  options.emulation.to_bc = true;
  options.teardown = teardown;
  if (teardown == ScenarioTeardown::kOrderly) {
    options.fake = &fake;  // the image copies and blits too
  }
  ::testing::internal::CaptureStderr();
  const ScenarioRun run = RunEtc2Scenario(*scenario, &backend, options);
  const std::string log = ::testing::internal::GetCapturedStderr();
  ASSERT_FALSE(run.skipped) << run.skip_reason;
  for (const std::string& failure : run.failures) {
    ADD_FAILURE() << failure;
  }
  ExpectOnlyBcModeErrors(fake, *scenario);
  ExpectEverythingReleased(fake);
  EXPECT_EQ(CountLines(log, scenario->emulated ? kBcLine : kRgbaLine),
            scenario->emulated ? 1 : 0)
      << log;
  EXPECT_EQ(log.find("could not encode"), std::string::npos) << log;
  std::uint64_t golden = 0;
  const bool has_golden = FindEtc2BcEmulationGolden(name, &golden);
  if (!scenario->bc_mode_fake_errors.empty()) {
    EXPECT_FALSE(has_golden) << "invalid copies have no golden";
  } else if (!has_golden) {
    ADD_FAILURE() << "no BC golden hash for " << name << "; this run gives "
                  << GoldenLine(name, run.hash);
  } else {
    EXPECT_EQ(run.hash, golden)
        << "the BC blocks of " << name << " changed from "
        << "tests/vulkan_etc2_bc_golden.inc; this run gives "
        << GoldenLine(name, run.hash);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Scenarios, Etc2BcScenarioTest,
    ::testing::Combine(::testing::ValuesIn(ScenarioNames()), ::testing::Bool(),
                       ::testing::Values(ScenarioTeardown::kOrderly,
                                         ScenarioTeardown::kResetPool,
                                         ScenarioTeardown::kDestroyPool,
                                         ScenarioTeardown::kDestroyDeviceFirst)),
    [](const ::testing::TestParamInfo<
        std::tuple<std::string, bool, ScenarioTeardown>>& info) {
      return std::get<0>(info.param) +
             (std::get<1>(info.param) ? "_Arena_" : "_OneOffStaging_") +
             TeardownName(std::get<2>(info.param));
    });

std::vector<std::string> BcModeScenarioNames() {
  std::vector<std::string> names;
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    if (scenario.bc_mode_fake_errors.empty()) {
      names.push_back(scenario.name);
    }
  }
  return names;
}

// Whether the GPU route serves one of the scenario's uploads in BC mode: a
// copy into an emulated image on a queue with compute (family 1 of the fake
// is transfer-only), unless texture overrides are active and the image is a
// colour one (DESIGN §3.2). BC images are never upscaled, so the upscale
// scenarios' uploads take the route too.
bool BcServesOnTheGpu(const Scenario& scenario) {
  if (!scenario.emulated || scenario.queue_family == 1) {
    return false;
  }
  for (const ScenarioStep& step : scenario.steps) {
    if (step.kind != ScenarioStep::Kind::kCopyBufferToImage) {
      continue;
    }
    Etc2EmulatedFormat format;
    if (!LookupEmulatedEtc2Format(scenario.images[step.image].format,
                                  &format)) {
      continue;
    }
    const bool colour = format.etc_format == EtcFormat::kEtc2Rgb8 ||
                        format.etc_format == EtcFormat::kEtc2Rgb8A1 ||
                        format.etc_format == EtcFormat::kEtc2Rgba8;
    if (!scenario.overrides || !colour) {
      return true;
    }
  }
  return false;
}

// How a scenario's GPU-route uploads reach the arena in BC mode.
enum class BcFill {
  kPass,    // the etc2_bc dispatches in front of the batch
  kRescue,  // [CR-7] no slot after the warm-up: CPU-encoded, copied in
  kRescueOneOffStaging,  // the same through one-off staging buffers
};

// With the GPU route on as well (MOCKTAIL_ETC2_GPU's default) and ready
// (DESIGN C4.4): uploads into BC images get their blocks from the etc2_bc
// pipeline (which the fake device runs with the production encoder), or
// from the rescue's CPU encoder, and must give the CPU route's BC goldens;
// with every teardown, everything the route made is given back.
class Etc2BcGpuRouteTest : public ::testing::TestWithParam<std::string> {
 protected:
  void TearDown() override {
    Etc2GpuTesting().fail_slot_acquire = false;
    Etc2EmulationTesting().no_host_staging = false;
  }

  void RunAndCheck(ScenarioTeardown teardown, bool frame_stats,
                   BcFill fill = BcFill::kPass) {
    Etc2EmulationTesting().no_host_staging =
        fill == BcFill::kRescueOneOffStaging;
    const Scenario* scenario = FindEtc2EmulationScenario(GetParam());
    ASSERT_NE(scenario, nullptr);
    FakeVulkanDevice fake;
    FakeEmulationBackend backend(&fake);
    ScenarioOptions options;
    options.teardown = teardown;
    options.emulation.to_bc = true;
    options.emulation.gpu = true;
    options.emulation.frame_stats = frame_stats;
    if (teardown == ScenarioTeardown::kOrderly) {
      options.fake = &fake;  // the image copies and blits too
    }
    if (fill != BcFill::kPass) {
      options.after_warm_up = [] { Etc2GpuTesting().fail_slot_acquire = true; };
    }
    ::testing::internal::CaptureStderr();
    const ScenarioRun run = RunEtc2Scenario(*scenario, &backend, options);
    const std::string log = ::testing::internal::GetCapturedStderr();
    Etc2GpuTesting().fail_slot_acquire = false;
    ASSERT_FALSE(run.skipped) << run.skip_reason;
    for (const std::string& failure : run.failures) {
      ADD_FAILURE() << failure;
    }
    ExpectOnlyBcModeErrors(fake, *scenario);
    ExpectEverythingReleased(fake);
    const FakeVulkanDevice::Counts& counts = fake.counts();
    EXPECT_EQ(counts.fences.live(), 0U);
    EXPECT_EQ(counts.shader_modules.live(), 0U);
    EXPECT_EQ(counts.pipelines.live(), 0U);
    EXPECT_EQ(counts.pipeline_layouts.live(), 0U);
    EXPECT_EQ(counts.descriptor_set_layouts.live(), 0U);
    EXPECT_EQ(counts.descriptor_pools.live(), 0U);
    EXPECT_EQ(counts.descriptor_sets.live(), 0U);
    EXPECT_EQ(counts.query_pools.live(), 0U);
    std::uint64_t golden = 0;
    ASSERT_TRUE(FindEtc2BcEmulationGolden(scenario->name, &golden));
    EXPECT_EQ(run.hash, golden)
        << "the GPU route's BC blocks of " << scenario->name
        << " differ from the CPU route's; this run gives "
        << GoldenLine(scenario->name, run.hash);
    if (!scenario->emulated) {
      EXPECT_FALSE(run.gpu_route);
      return;
    }
    ASSERT_TRUE(run.gpu_route);
    EXPECT_EQ(CountLines(log, kBcGpuLine), 1) << log;
    // Built with both pipelines; on a queue with compute the selftest
    // checked the BC one too (a transfer-only queue never runs it).
    EXPECT_EQ(counts.pipelines.created, 2U);
    EXPECT_EQ(log.find("selftest ok, BC transcode too)") != std::string::npos,
              scenario->queue_family != 1)
        << log;
    if (BcServesOnTheGpu(*scenario)) {
      EXPECT_EQ(run.gpu_state, "ready") << log;
      EXPECT_GT(run.gpu_stats.passes, 0U);
      if (fill == BcFill::kPass) {
        EXPECT_GT(counts.bc_blocks, 0U) << "no etc2_bc dispatch ran";
        EXPECT_EQ(run.gpu_stats.rescues, 0U);
      } else {
        EXPECT_EQ(run.gpu_stats.rescues, run.gpu_stats.passes);
        EXPECT_GT(run.gpu_stats.rescued_uploads, 0U);
      }
    } else {
      EXPECT_EQ(run.gpu_stats.passes, 0U);
    }
    EXPECT_EQ(run.gpu_stats.dropped_uploads, 0U);
    EXPECT_EQ(log.find("GPU decode unavailable"), std::string::npos) << log;
    EXPECT_EQ(log.find("dropped GPU-route uploads"), std::string::npos) << log;
    EXPECT_EQ(log.find("could not encode"), std::string::npos) << log;
  }
};

TEST_P(Etc2BcGpuRouteTest, MatchesTheBcGoldens) {
  RunAndCheck(ScenarioTeardown::kOrderly, false);
}

TEST_P(Etc2BcGpuRouteTest, ResetPoolReleasesEverything) {
  RunAndCheck(ScenarioTeardown::kResetPool, false);
}

TEST_P(Etc2BcGpuRouteTest, DestroyDeviceFirstReleasesEverything) {
  RunAndCheck(ScenarioTeardown::kDestroyDeviceFirst, true);
}

// The rescue encodes on the CPU what the etc2_bc pass would have written.
TEST_P(Etc2BcGpuRouteTest, RescueMatchesTheBcGoldens) {
  RunAndCheck(ScenarioTeardown::kOrderly, false, BcFill::kRescue);
}

TEST_P(Etc2BcGpuRouteTest, RescueWithOneOffStagingReleasesEverything) {
  RunAndCheck(ScenarioTeardown::kDestroyPool, false,
              BcFill::kRescueOneOffStaging);
}

INSTANTIATE_TEST_SUITE_P(Scenarios, Etc2BcGpuRouteTest,
                         ::testing::ValuesIn(BcModeScenarioNames()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });

TEST(Etc2BcScenarioListTest, EveryScenarioValidInBcModeHasABcGolden) {
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    std::uint64_t hash = 0;
    EXPECT_EQ(FindEtc2BcEmulationGolden(scenario.name, &hash),
              scenario.bc_mode_fake_errors.empty())
        << scenario.name;
  }
}

// Without MOCKTAIL_ETC2_TO_BC the same runner keeps the default mode's
// texels, on a fake device that could do BC.
TEST(Etc2BcScenarioListTest, OffKeepsTheDefaultGoldens) {
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    SCOPED_TRACE(scenario.name);
    FakeVulkanDevice fake;
    FakeEmulationBackend backend(&fake);
    ScenarioOptions options;
    options.emulation.to_bc = false;
    ::testing::internal::CaptureStderr();
    const ScenarioRun run = RunEtc2Scenario(scenario, &backend, options);
    const std::string log = ::testing::internal::GetCapturedStderr();
    ASSERT_FALSE(run.skipped);
    EXPECT_TRUE(run.failures.empty()) << run.failures.front();
    std::uint64_t golden = 0;
    ASSERT_TRUE(FindEtc2EmulationGolden(scenario.name, &golden));
    EXPECT_EQ(run.hash, golden);
    EXPECT_EQ(log.find("ETC2->BC"), std::string::npos) << log;
    EXPECT_EQ(fake.calls("vkGetPhysicalDeviceFeatures"), 0U);
    EXPECT_EQ(fake.calls("vkGetPhysicalDeviceImageFormatProperties"), 0U);
  }
}

// [CR-24] An override PNG on an RGB8A1 image: BC1's punch-through is
// binary, so texels whose override alpha is below 128 become transparent
// black and the rest opaque (RGBA8 hosts keep the PNG's alpha).
TEST(Etc2BcOverrideTest, Rgb8A1OverridesArePunchedThroughAtAlpha128) {
  constexpr std::uint32_t kSize = 32;
  constexpr std::uint64_t kSeed = 0xA1FA;
  Scenario scenario;
  scenario.name = "Rgb8A1Override";
  scenario.overrides = true;
  scenario.images = {{VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, kSize, kSize, 2, 1}};
  VkBufferImageCopy mip0{};
  mip0.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  mip0.imageExtent = {kSize, kSize, 1};
  VkBufferImageCopy mip1 = mip0;
  mip1.bufferOffset = (kSize / 4) * (kSize / 4) * 8;
  mip1.imageSubresource.mipLevel = 1;
  mip1.imageExtent = {kSize / 2, kSize / 2, 1};
  scenario.sources = {
      {0x5EED, mip1.bufferOffset + (kSize / 8) * (kSize / 8) * 8, true, 0, 0}};
  scenario.command_buffers = {{false}};
  ScenarioStep override_step;
  override_step.kind = ScenarioStep::Kind::kWriteOverride;
  override_step.regions = {mip0};
  override_step.override_width = 24;  // resampled to 32 and 16
  override_step.override_height = 40;
  override_step.seed = kSeed;
  ScenarioStep copy;
  copy.kind = ScenarioStep::Kind::kCopyBufferToImage;
  copy.regions = {mip0, mip1};
  ScenarioStep submit;
  submit.kind = ScenarioStep::Kind::kSubmit;
  submit.batches = {{0}};
  scenario.steps = {override_step, copy, submit};
  for (const bool arena : {true, false}) {
    SCOPED_TRACE(arena ? "host staging arena" : "one-off staging");
    Etc2EmulationTesting().no_host_staging = !arena;
    FakeVulkanDevice fake;
    FakeEmulationBackend backend(&fake);
    ScenarioOptions options;
    options.emulation.to_bc = true;
    ::testing::internal::CaptureStderr();
    const ScenarioRun run = RunEtc2Scenario(scenario, &backend, options);
    const std::string log = ::testing::internal::GetCapturedStderr();
    ASSERT_FALSE(run.skipped) << run.skip_reason;
    for (const std::string& failure : run.failures) {
      ADD_FAILURE() << failure;
    }
    EXPECT_NE(log.find("texture override loaded"), std::string::npos) << log;
    EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
  }
  Etc2EmulationTesting().no_host_staging = false;
  // The emulation's blocks equal the reference encoding of the resampled
  // override (checked above against the model); what those blocks decode
  // to, by D3D10 rules:
  const std::vector<std::uint8_t> png = SeededBytes(kSeed, 24 * 40 * 4);
  for (const std::uint32_t size : {kSize, kSize / 2}) {
    std::vector<std::uint8_t> texels(static_cast<std::size_t>(size) * size * 4);
    reference::ResampleRgba(png.data(), 24, 40, size, size, texels.data());
    const std::vector<std::uint8_t> blocks = reference::BcFromTexels(
        EtcFormat::kEtc2Rgb8A1, texels.data(), size, size, 1, true);
    int transparent = 0;
    for (std::uint32_t block = 0; block < blocks.size() / 8; ++block) {
      std::uint32_t words[2];
      std::memcpy(words, blocks.data() + block * 8, 8);
      std::uint8_t decoded[64];
      bcref::DecodeBc1(words, false, decoded);
      const std::uint32_t bx = block % (size / 4);
      const std::uint32_t by = block / (size / 4);
      for (std::uint32_t texel = 0; texel < 16; ++texel) {
        const std::uint8_t* original =
            texels.data() +
            ((by * 4 + texel / 4) * size + bx * 4 + texel % 4) * 4;
        const std::uint8_t* out = decoded + texel * 4;
        if (original[3] < 128) {
          ++transparent;
          EXPECT_EQ(out[0] | out[1] | out[2] | out[3], 0)
              << "block " << block << " texel " << texel;
        } else {
          EXPECT_EQ(out[3], 255) << "block " << block << " texel " << texel;
        }
      }
    }
    EXPECT_GT(transparent, 0);
    EXPECT_LT(transparent, static_cast<int>(size * size));
  }
}

}  // namespace
}  // namespace mocktail::graphics::testing
