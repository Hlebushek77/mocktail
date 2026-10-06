#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "support/etc2_emulation_harness.h"
#include "support/fake_vulkan_device.h"

// ETC2->BC mode in the emulation (DESIGN C4.1-C4.3, CR-14, CR-15, CR-22):
// which devices enter it, the BC host images and views it creates, the
// images it leaves RGBA8/R16, and the format lists the host sees. Off (the
// default, MOCKTAIL_ETC2_TO_BC unset or 0) nothing may change. The
// physical-device queries come from a configurable fake instance next to
// the GPU-free fake device.

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
const char kUnavailable[] = "  [vulkan] ETC2->BC unavailable: ";

class Etc2BcModeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_model = &model_;
    SetEtc2BcUploadsReadyForTesting(true);
  }
  void TearDown() override {
    SetEtc2BcUploadsReadyForTesting(false);
    g_model = nullptr;
  }

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

// Until an upload route writes BC blocks, MOCKTAIL_ETC2_TO_BC=1 keeps the
// default mode and says why.
TEST_F(Etc2BcModeTest, ClosedGateKeepsTheDefaultMode) {
  SetEtc2BcUploadsReadyForTesting(false);
  FakeVulkanDevice fake;
  VulkanEtc2Emulation emulation(Options(true));
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(Register(&emulation, &fake));
  VkImageCreateInfo host{};
  const VkImage image =
      Create(&emulation, &fake,
             ImageInfo(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 32, 32), &host);
  EXPECT_EQ(host.format, VK_FORMAT_R8G8B8A8_UNORM);
  EXPECT_EQ(host.extent.width, 128U);  // the x4 upscale as before
  emulation.DestroyImage(fake.device(), image, nullptr);
  emulation.DestroyDevice(fake.device());
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(CountLines(log, kRgbaLine), 1) << log;
  EXPECT_EQ(CountLines(log, std::string(kUnavailable) +
                                "no upload route of this build writes BC "
                                "blocks yet; ETC2 textures stay RGBA8/R16"),
            1)
      << log;
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

}  // namespace
}  // namespace mocktail::graphics::testing
