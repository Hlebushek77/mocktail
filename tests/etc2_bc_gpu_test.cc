#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "support/vulkan_test_device.h"

// ETC2->BC mode's images on a real device (ctest label "gpu"): the BC
// decision from the driver's own features and format properties, the BC
// host images and views the emulation creates through it, an image BC
// cannot take (storage) left RGBA8/R16, and a format list rewritten for the
// host. Run once more under the Khronos validation layer, which must report
// nothing (CR-15: a list that still named ETC2 formats would be flagged).
// Uploads into BC images run in etc2_emulation_gpu_test (every scenario in
// BC mode). Skips without a device, on a device that reads ETC2 itself, and
// while a live game runs (CR-25).

namespace mocktail::graphics::testing {
namespace {

class Etc2BcGpuTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    device_ = HeadlessVulkanDevice::Create(&skip_reason_).release();
    if (device_ != nullptr) {
      std::printf("GPU: %s\n", device_->Describe().c_str());
    }
  }
  static void TearDownTestSuite() {
    delete device_;
    device_ = nullptr;
  }
  void SetUp() override {
    if (device_ == nullptr) {
      GTEST_SKIP() << skip_reason_;
    }
  }

  template <typename Function>
  static Function InstanceProc(const char* name) {
    return reinterpret_cast<Function>(
        device_->get_instance_proc_addr()(device_->instance(), name));
  }
  template <typename Function>
  static Function DeviceProc(const char* name) {
    return reinterpret_cast<Function>(
        device_->get_device_proc_addr()(device_->device(), name));
  }

  // The bytes vkGetImageMemoryRequirements asks for an image made straight
  // on the host, to tell a BC host image from an RGBA8/R16 one.
  static VkDeviceSize HostImageBytes(const VkImageCreateInfo& info) {
    const auto create_image = DeviceProc<PFN_vkCreateImage>("vkCreateImage");
    VkImage image = VK_NULL_HANDLE;
    EXPECT_EQ(create_image(device_->device(), &info, nullptr, &image),
              VK_SUCCESS);
    VkMemoryRequirements requirements{};
    device_->vk().vkGetImageMemoryRequirements(device_->device(), image,
                                               &requirements);
    device_->vk().vkDestroyImage(device_->device(), image, nullptr);
    return requirements.size;
  }

  static HeadlessVulkanDevice* device_;
  static std::string skip_reason_;
};

HeadlessVulkanDevice* Etc2BcGpuTest::device_ = nullptr;
std::string Etc2BcGpuTest::skip_reason_;

VkImageCreateInfo ImageInfo(VkFormat format, VkImageUsageFlags usage) {
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = format;
  info.extent = {256, 128, 1};
  info.mipLevels = 9;
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage;
  return info;
}

TEST_F(Etc2BcGpuTest, CreatesBcImagesAndViewsTheDriverTakes) {
  const auto get_features = InstanceProc<PFN_vkGetPhysicalDeviceFeatures>(
      "vkGetPhysicalDeviceFeatures");
  const auto get_format_properties =
      InstanceProc<PFN_vkGetPhysicalDeviceFormatProperties>(
          "vkGetPhysicalDeviceFormatProperties");
  ASSERT_NE(get_features, nullptr);
  ASSERT_NE(get_format_properties, nullptr);
  Etc2EmulationOptions options;
  options.gpu = false;
  options.to_bc = true;
  VulkanEtc2Emulation emulation(options);
  if (!emulation.PhysicalDeviceNeedsEmulation(device_->physical_device(),
                                              get_features)) {
    GTEST_SKIP() << "the device reads ETC2 itself";
  }
  VkPhysicalDeviceFeatures features{};
  get_features(device_->physical_device(), &features);
  if (features.textureCompressionBC != VK_TRUE) {
    GTEST_SKIP() << "the device has no BC";
  }
  ASSERT_TRUE(emulation.PhysicalDeviceUsesBc(
      device_->physical_device(), get_features, get_format_properties));
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  emulation.RegisterDevice(
      device_->device(), device_->physical_device(), true, device_->memory(),
      device_->properties(), device_->queue_families(), device_info,
      device_->get_device_proc_addr(), device_->get_instance_proc_addr(),
      device_->instance());
  const auto destroy_view =
      DeviceProc<PFN_vkDestroyImageView>("vkDestroyImageView");
  ASSERT_NE(destroy_view, nullptr);
  constexpr VkImageUsageFlags kSampled = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                         VK_IMAGE_USAGE_SAMPLED_BIT;
  for (int value = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
       value <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK; ++value) {
    const auto format = static_cast<VkFormat>(value);
    SCOPED_TRACE(value);
    Etc2EmulatedFormat mapped;
    ASSERT_TRUE(LookupEmulatedEtc2Format(format, &mapped));
    // Mutable, with a format list of its own format (and its sRGB or UNORM
    // twin for the colour formats), which the host must see as BC.
    std::vector<VkFormat> views = {format};
    if (value <= VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK) {
      const bool srgb = (value - VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK) % 2 != 0;
      views.push_back(static_cast<VkFormat>(srgb ? value - 1 : value + 1));
    }
    VkImageFormatListCreateInfo list{};
    list.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
    list.viewFormatCount = static_cast<std::uint32_t>(views.size());
    list.pViewFormats = views.data();
    VkImageCreateInfo info = ImageInfo(format, kSampled);
    info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    info.pNext = &list;
    VkImage image = VK_NULL_HANDLE;
    ASSERT_EQ(emulation.CreateImage(device_->device(), &info, nullptr, &image),
              VK_SUCCESS);
    EXPECT_EQ(list.pViewFormats, views.data());
    EXPECT_EQ(views[0], format);
    VkMemoryRequirements requirements{};
    device_->vk().vkGetImageMemoryRequirements(device_->device(), image,
                                               &requirements);
    VkImageCreateInfo bc = ImageInfo(mapped.bc_host_format, kSampled);
    EXPECT_EQ(requirements.size, HostImageBytes(bc));
    VkImageCreateInfo rgba = ImageInfo(mapped.host_format, kSampled);
    EXPECT_LT(requirements.size, HostImageBytes(rgba));
    // Views need the image bound.
    VkMemoryAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = requirements.size;
    allocate_info.memoryTypeIndex = device_->FindMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    ASSERT_EQ(device_->vk().vkAllocateMemory(device_->device(), &allocate_info,
                                             nullptr, &memory),
              VK_SUCCESS);
    ASSERT_EQ(
        device_->vk().vkBindImageMemory(device_->device(), image, memory, 0),
        VK_SUCCESS);
    for (const VkFormat view_format : views) {
      VkImageViewCreateInfo view_info{};
      view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      view_info.image = image;
      view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view_info.format = view_format;
      view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 9, 0, 1};
      VkImageView view = VK_NULL_HANDLE;
      ASSERT_EQ(emulation.CreateImageView(device_->device(), &view_info,
                                          nullptr, &view),
                VK_SUCCESS);
      destroy_view(device_->device(), view, nullptr);
    }
    emulation.DestroyImage(device_->device(), image, nullptr);
    device_->vk().vkFreeMemory(device_->device(), memory, nullptr);

    // Storage: no BC format takes it here, so the image keeps the
    // default host, at its own size (no upscale in BC mode).
    if (!device_->OptimalTilingSupports(mapped.host_format,
                                        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ||
        device_->OptimalTilingSupports(mapped.bc_host_format,
                                       VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
      continue;
    }
    VkImageCreateInfo storage =
        ImageInfo(format, kSampled | VK_IMAGE_USAGE_STORAGE_BIT);
    ASSERT_EQ(
        emulation.CreateImage(device_->device(), &storage, nullptr, &image),
        VK_SUCCESS);
    device_->vk().vkGetImageMemoryRequirements(device_->device(), image,
                                               &requirements);
    EXPECT_EQ(requirements.size,
              HostImageBytes(ImageInfo(mapped.host_format,
                                       kSampled | VK_IMAGE_USAGE_STORAGE_BIT)));
    emulation.DestroyImage(device_->device(), image, nullptr);
  }
  emulation.DestroyDevice(device_->device());
}

}  // namespace
}  // namespace mocktail::graphics::testing
