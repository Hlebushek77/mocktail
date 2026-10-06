// The ETC2 emulation's options, submit plan and step-0 diagnostics, driven
// through a small malloc-backed fake device (no GPU, no Vulkan loader).

#include "mocktail/graphics/vulkan_etc2_emulation.h"

#include <stdlib.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/texture_override.h"

namespace mocktail::graphics {
namespace {

template <typename Handle>
Handle FakeHandle(std::uint64_t value) {
  return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(value));
}

struct FakeMemory {
  std::vector<std::uint8_t> bytes;
  std::uint32_t type = 0;
};

struct FakeBuffer {
  VkDeviceSize size = 0;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
};

struct RecordedCopy {
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  VkBuffer source = VK_NULL_HANDLE;
  VkImage destination = VK_NULL_HANDLE;
  std::vector<VkBufferImageCopy> regions;
};

// One fake device at a time; the tests are single-threaded.
struct FakeVulkan {
  std::uint64_t next_handle = 0x10000;
  std::map<VkDeviceMemory, FakeMemory> memories;
  std::map<VkBuffer, FakeBuffer> buffers;
  std::map<VkImage, VkFormat> images;
  std::vector<RecordedCopy> copies;
  int buffers_created = 0;
  int buffers_destroyed = 0;
  int memories_allocated = 0;
  int memories_freed = 0;
  int image_copies = 0;
  int executes = 0;
};

FakeVulkan* g_fake = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateImage(
    VkDevice, const VkImageCreateInfo* info, const VkAllocationCallbacks*,
    VkImage* image) {
  *image = FakeHandle<VkImage>(++g_fake->next_handle);
  g_fake->images[*image] = info->format;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyImage(VkDevice, VkImage image,
                                            const VkAllocationCallbacks*) {
  g_fake->images.erase(image);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateImageView(
    VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*,
    VkImageView* view) {
  *view = FakeHandle<VkImageView>(++g_fake->next_handle);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeGetImageMemoryRequirements(
    VkDevice, VkImage, VkMemoryRequirements* requirements) {
  requirements->size = 4096;
  requirements->alignment = 256;
  requirements->memoryTypeBits = 0x3;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBindImageMemory(VkDevice, VkImage,
                                                   VkDeviceMemory,
                                                   VkDeviceSize) {
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeAllocateMemory(
    VkDevice, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*,
    VkDeviceMemory* memory) {
  *memory = FakeHandle<VkDeviceMemory>(++g_fake->next_handle);
  FakeMemory& fake = g_fake->memories[*memory];
  fake.bytes.assign(static_cast<std::size_t>(info->allocationSize), 0);
  fake.type = info->memoryTypeIndex;
  ++g_fake->memories_allocated;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeFreeMemory(VkDevice, VkDeviceMemory memory,
                                          const VkAllocationCallbacks*) {
  EXPECT_EQ(g_fake->memories.erase(memory), 1U) << "freed twice";
  ++g_fake->memories_freed;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeMapMemory(VkDevice, VkDeviceMemory memory,
                                             VkDeviceSize offset, VkDeviceSize,
                                             VkMemoryMapFlags, void** data) {
  *data = g_fake->memories.at(memory).bytes.data() + offset;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeUnmapMemory(VkDevice, VkDeviceMemory) {}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateBuffer(
    VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
    VkBuffer* buffer) {
  *buffer = FakeHandle<VkBuffer>(++g_fake->next_handle);
  g_fake->buffers[*buffer].size = info->size;
  ++g_fake->buffers_created;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyBuffer(VkDevice, VkBuffer buffer,
                                             const VkAllocationCallbacks*) {
  EXPECT_EQ(g_fake->buffers.erase(buffer), 1U) << "destroyed twice";
  ++g_fake->buffers_destroyed;
}

VKAPI_ATTR void VKAPI_CALL FakeGetBufferMemoryRequirements(
    VkDevice, VkBuffer buffer, VkMemoryRequirements* requirements) {
  requirements->size = g_fake->buffers.at(buffer).size;
  requirements->alignment = 256;
  requirements->memoryTypeBits = 0x3b;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBindBufferMemory(VkDevice, VkBuffer buffer,
                                                    VkDeviceMemory memory,
                                                    VkDeviceSize offset) {
  g_fake->buffers.at(buffer).memory = memory;
  g_fake->buffers.at(buffer).offset = offset;
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeCmdCopyBufferToImage(
    VkCommandBuffer command_buffer, VkBuffer source, VkImage destination,
    VkImageLayout, std::uint32_t count, const VkBufferImageCopy* regions) {
  g_fake->copies.push_back({command_buffer, source, destination,
                            std::vector<VkBufferImageCopy>(regions,
                                                           regions + count)});
}

VKAPI_ATTR void VKAPI_CALL FakeCmdCopyImage(VkCommandBuffer, VkImage,
                                            VkImageLayout, VkImage,
                                            VkImageLayout, std::uint32_t,
                                            const VkImageCopy*) {
  ++g_fake->image_copies;
}

VKAPI_ATTR void VKAPI_CALL FakeCmdExecuteCommands(VkCommandBuffer,
                                                  std::uint32_t,
                                                  const VkCommandBuffer*) {
  ++g_fake->executes;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL FakeGetDeviceProcAddr(
    VkDevice, const char* name) {
  static const std::map<std::string, PFN_vkVoidFunction> kProcs = {
      {"vkCreateImage", reinterpret_cast<PFN_vkVoidFunction>(FakeCreateImage)},
      {"vkDestroyImage",
       reinterpret_cast<PFN_vkVoidFunction>(FakeDestroyImage)},
      {"vkCreateImageView",
       reinterpret_cast<PFN_vkVoidFunction>(FakeCreateImageView)},
      {"vkGetImageMemoryRequirements",
       reinterpret_cast<PFN_vkVoidFunction>(FakeGetImageMemoryRequirements)},
      {"vkBindImageMemory",
       reinterpret_cast<PFN_vkVoidFunction>(FakeBindImageMemory)},
      {"vkAllocateMemory",
       reinterpret_cast<PFN_vkVoidFunction>(FakeAllocateMemory)},
      {"vkFreeMemory", reinterpret_cast<PFN_vkVoidFunction>(FakeFreeMemory)},
      {"vkMapMemory", reinterpret_cast<PFN_vkVoidFunction>(FakeMapMemory)},
      {"vkUnmapMemory", reinterpret_cast<PFN_vkVoidFunction>(FakeUnmapMemory)},
      {"vkCreateBuffer",
       reinterpret_cast<PFN_vkVoidFunction>(FakeCreateBuffer)},
      {"vkDestroyBuffer",
       reinterpret_cast<PFN_vkVoidFunction>(FakeDestroyBuffer)},
      {"vkGetBufferMemoryRequirements",
       reinterpret_cast<PFN_vkVoidFunction>(FakeGetBufferMemoryRequirements)},
      {"vkBindBufferMemory",
       reinterpret_cast<PFN_vkVoidFunction>(FakeBindBufferMemory)},
      {"vkCmdCopyBufferToImage",
       reinterpret_cast<PFN_vkVoidFunction>(FakeCmdCopyBufferToImage)},
      {"vkCmdCopyImage",
       reinterpret_cast<PFN_vkVoidFunction>(FakeCmdCopyImage)},
      {"vkCmdExecuteCommands",
       reinterpret_cast<PFN_vkVoidFunction>(FakeCmdExecuteCommands)},
  };
  const auto found = kProcs.find(name);
  return found != kProcs.end() ? found->second : nullptr;
}

// The memory types and queue families of an RTX 3060 Laptop (DESIGN §2.1).
VkPhysicalDeviceMemoryProperties TestMemoryProperties() {
  VkPhysicalDeviceMemoryProperties memory{};
  memory.memoryTypeCount = 6;
  memory.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  memory.memoryTypes[2].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  memory.memoryTypes[3].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  memory.memoryTypes[4].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  memory.memoryTypes[5].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  memory.memoryHeapCount = 2;
  return memory;
}

std::vector<VkQueueFamilyProperties> TestQueueFamilies() {
  std::vector<VkQueueFamilyProperties> families(3);
  families[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                           VK_QUEUE_TRANSFER_BIT |
                           VK_QUEUE_SPARSE_BINDING_BIT;
  families[0].queueCount = 16;
  families[1].queueFlags = VK_QUEUE_TRANSFER_BIT;
  families[1].queueCount = 2;
  families[2].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
  families[2].queueCount = 8;
  return families;
}

// An ETC2 RGB8 individual-mode block of one flat colour.
constexpr std::uint8_t kBlock[8] = {0x80, 0x40, 0x20, 0x00,
                                    0x00, 0x00, 0x00, 0x00};

class Etc2DiagnosticsTest : public ::testing::Test {
 protected:
  void SetUp() override { g_fake = &fake_; }
  void TearDown() override { g_fake = nullptr; }

  Etc2EmulationOptions Options(bool frame_stats) {
    Etc2EmulationOptions options;
    options.upscale = 1;
    options.frame_stats = frame_stats;
    return options;
  }

  void Register(VulkanEtc2Emulation* emulation, bool emulated = true) {
    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queues[2]{};
    queues[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queues[0].queueFamilyIndex = 0;
    queues[0].queueCount = 1;
    queues[0].pQueuePriorities = &priority;
    queues[1] = queues[0];
    queues[1].queueFamilyIndex = 1;
    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 2;
    create_info.pQueueCreateInfos = queues;
    VkPhysicalDeviceProperties properties{};
    emulation->RegisterDevice(device_, physical_, emulated,
                              TestMemoryProperties(), properties,
                              TestQueueFamilies(), create_info,
                              FakeGetDeviceProcAddr);
  }

  VkImage CreateEtc2Image(VulkanEtc2Emulation* emulation,
                          std::uint32_t extent = 4) {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
    info.extent = {extent, extent, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    VkImage image = VK_NULL_HANDLE;
    EXPECT_EQ(emulation->CreateImage(device_, &info, nullptr, &image),
              VK_SUCCESS);
    return image;
  }

  // A buffer of `blocks` copies of kBlock in memory of `type`, mapped.
  VkBuffer CreateSource(VulkanEtc2Emulation* emulation, std::uint32_t type,
                        std::size_t blocks = 1) {
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = 8 * blocks;
    allocate.memoryTypeIndex = type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    EXPECT_EQ(emulation->AllocateMemory(device_, &allocate, nullptr, &memory),
              VK_SUCCESS);
    const VkBuffer buffer = FakeHandle<VkBuffer>(++fake_.next_handle);
    fake_.buffers[buffer].size = 8 * blocks;
    EXPECT_EQ(emulation->BindBufferMemory(device_, buffer, memory, 0),
              VK_SUCCESS);
    void* data = nullptr;
    EXPECT_EQ(emulation->MapMemory(device_, memory, 0, VK_WHOLE_SIZE, 0, &data),
              VK_SUCCESS);
    for (std::size_t block = 0; block < blocks; ++block) {
      std::memcpy(static_cast<std::uint8_t*>(data) + 8 * block, kBlock, 8);
    }
    return buffer;
  }

  void Copy(VulkanEtc2Emulation* emulation, VkCommandBuffer command_buffer,
            VkBuffer source, VkImage image, std::uint32_t regions = 1) {
    std::vector<VkBufferImageCopy> copies(regions);
    for (VkBufferImageCopy& region : copies) {
      region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      region.imageSubresource.layerCount = 1;
      region.imageExtent = {4, 4, 1};
    }
    emulation->CmdCopyBufferToImage(device_, command_buffer, source, image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    regions, copies.data());
  }

  void Submit(VulkanEtc2Emulation* emulation,
              std::vector<VkCommandBuffer> command_buffers) {
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount =
        static_cast<std::uint32_t>(command_buffers.size());
    submit.pCommandBuffers = command_buffers.data();
    Etc2SubmitPlan plan;
    emulation->PrepareSubmit(queue_, 1, &submit, VK_NULL_HANDLE, &plan);
    EXPECT_FALSE(plan.modified);
    EXPECT_EQ(plan.submits(&submit), &submit);
    emulation->FinishSubmit(&plan, VK_SUCCESS, nullptr);
  }

  // What the last recorded copy will read once the submit decoded it.
  const std::uint8_t* StagedTexels() {
    const FakeBuffer& staging = fake_.buffers.at(fake_.copies.back().source);
    return fake_.memories.at(staging.memory).bytes.data() + staging.offset;
  }

  FakeVulkan fake_;
  const VkDevice device_ = FakeHandle<VkDevice>(0xd0);
  const VkPhysicalDevice physical_ = FakeHandle<VkPhysicalDevice>(0xe0);
  const VkQueue queue_ = FakeHandle<VkQueue>(0xf0);
  const VkCommandPool graphics_pool_ = FakeHandle<VkCommandPool>(0xa0);
  const VkCommandPool transfer_pool_ = FakeHandle<VkCommandPool>(0xa1);
  const VkCommandBuffer command_buffer_ = FakeHandle<VkCommandBuffer>(0xc0);
  const VkCommandBuffer secondary_ = FakeHandle<VkCommandBuffer>(0xc1);
};

TEST(Etc2EmulationOptionsTest, ReadsEveryVariable) {
  const std::map<std::string, const char*> environment = {
      {"MOCKTAIL_SMALL_TEXTURE_UPSCALE", "2"},
      {"MOCKTAIL_ETC2_FAST_PATH", "0"},
      {"MOCKTAIL_ETC2_GPU", "off"},
      {"MOCKTAIL_ETC2_TO_BC", "true"},
      {"MOCKTAIL_ETC2_WORKERS", "3"},
      {"MOCKTAIL_ETC2_GPU_ARENA_MB", "96"},
      {"MOCKTAIL_ETC2_STAGING_MEMORY", "coherent"},
      {"MOCKTAIL_FRAME_STATS", "yes"},
      {"MOCKTAIL_ETC2_TRACE", "/scratch/trace.csv"},
      {"MOCKTAIL_ETC2_TRACE_BLOCKS", "/scratch/blocks"},
  };
  const Etc2EmulationOptions options =
      Etc2EmulationOptions::FromLookup([&](const char* name) -> const char* {
        const auto found = environment.find(name);
        return found != environment.end() ? found->second : nullptr;
      });
  EXPECT_EQ(options.upscale, 2U);
  EXPECT_FALSE(options.fast_path);
  EXPECT_FALSE(options.gpu);
  EXPECT_TRUE(options.to_bc);
  EXPECT_EQ(options.workers, 3U);
  EXPECT_EQ(options.gpu_arena_bytes, 96ULL << 20);
  EXPECT_EQ(options.staging_memory, Etc2StagingMemory::kCoherent);
  EXPECT_TRUE(options.frame_stats);
  EXPECT_EQ(options.trace_path, "/scratch/trace.csv");
  EXPECT_EQ(options.trace_blocks_dir, "/scratch/blocks");
}

TEST(Etc2EmulationOptionsTest, KeepsDefaultsForUnsetOrUnknownValues) {
  const Etc2EmulationOptions unset = Etc2EmulationOptions::FromLookup(
      [](const char*) -> const char* { return nullptr; });
  EXPECT_EQ(unset.upscale, kDefaultSmallTextureUpscale);
  EXPECT_TRUE(unset.fast_path);
  EXPECT_TRUE(unset.gpu);
  EXPECT_FALSE(unset.to_bc);
  EXPECT_EQ(unset.workers, 0U);
  EXPECT_EQ(unset.gpu_arena_bytes, 0U);
  EXPECT_EQ(unset.staging_memory, Etc2StagingMemory::kCached);
  EXPECT_FALSE(unset.frame_stats);
  EXPECT_TRUE(unset.trace_path.empty());

  const std::map<std::string, const char*> environment = {
      {"MOCKTAIL_ETC2_GPU", "maybe"},     {"MOCKTAIL_ETC2_TO_BC", "yes"},
      {"MOCKTAIL_ETC2_WORKERS", "99"},    {"MOCKTAIL_ETC2_GPU_ARENA_MB", "x"},
      {"MOCKTAIL_ETC2_STAGING_MEMORY", "wc"}, {"MOCKTAIL_FRAME_STATS", "0"},
  };
  const Etc2EmulationOptions odd =
      Etc2EmulationOptions::FromLookup([&](const char* name) -> const char* {
        const auto found = environment.find(name);
        return found != environment.end() ? found->second : nullptr;
      });
  EXPECT_TRUE(odd.gpu);
  EXPECT_FALSE(odd.to_bc) << "the runtime config writes 1 or 0";
  EXPECT_EQ(odd.workers, 0U);
  EXPECT_EQ(odd.gpu_arena_bytes, 0U);
  EXPECT_EQ(odd.staging_memory, Etc2StagingMemory::kCached);
  EXPECT_FALSE(odd.frame_stats);
}

TEST(Etc2SubmitPlanTest, PassesRobloxsArraysThroughUntilModified) {
  VkSubmitInfo original{};
  Etc2SubmitPlan plan;
  const VkFence app_fence = FakeHandle<VkFence>(0x99);
  EXPECT_EQ(plan.count(3), 3U);
  EXPECT_EQ(plan.submits(&original), &original);
  EXPECT_EQ(plan.fence(app_fence), app_fence);
  EXPECT_EQ(plan.fence(VK_NULL_HANDLE), VK_NULL_HANDLE);
  EXPECT_TRUE(plan.pre_submits().empty());

  plan.modified = true;
  plan.submit_infos.resize(2);
  plan.substitute_fence = FakeHandle<VkFence>(0x77);
  EXPECT_EQ(plan.count(3), 2U);
  EXPECT_EQ(plan.submits(&original), plan.submit_infos.data());
  EXPECT_EQ(plan.fence(VK_NULL_HANDLE), plan.substitute_fence);
  EXPECT_EQ(plan.fence(app_fence), app_fence);

  Etc2SubmitPlan plan2;
  VkSubmitInfo2 original2{};
  plan2.modified = true;
  plan2.submit_infos2.resize(4);
  EXPECT_EQ(plan2.count(1), 4U);
  EXPECT_EQ(plan2.submits2(&original2), plan2.submit_infos2.data());
}

TEST_F(Etc2DiagnosticsTest, FinishSubmitSignalsTheTrailingFenceOnSuccessOnly) {
  VulkanEtc2Emulation emulation(Options(false));
  std::vector<VkFence> signalled;
  const auto fence_only = [&](VkFence fence) {
    signalled.push_back(fence);
    return VK_SUCCESS;
  };
  Etc2SubmitPlan plan;
  emulation.FinishSubmit(&plan, VK_SUCCESS, fence_only);
  EXPECT_TRUE(signalled.empty());
  plan.trailing_fence = FakeHandle<VkFence>(0x55);
  emulation.FinishSubmit(&plan, VK_ERROR_DEVICE_LOST, fence_only);
  EXPECT_TRUE(signalled.empty());
  emulation.FinishSubmit(&plan, VK_SUCCESS, fence_only);
  ASSERT_EQ(signalled.size(), 1U);
  EXPECT_EQ(signalled[0], plan.trailing_fence);
}

TEST_F(Etc2DiagnosticsTest, WithoutFrameStatsOnlyTheQueuesLineIsNew) {
  testing::internal::CaptureStderr();
  {
    VulkanEtc2Emulation emulation(Options(false));
    Register(&emulation);
    emulation.RegisterQueue(device_, queue_, 0, 0);
    emulation.RegisterCommandPool(device_, graphics_pool_, 0, 0);
    const VkImage image = CreateEtc2Image(&emulation);
    const VkBuffer source = CreateSource(&emulation, 3);
    emulation.BeginCommandBuffer(command_buffer_, graphics_pool_,
                                 VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    Copy(&emulation, command_buffer_, source, image);
    Submit(&emulation, {command_buffer_});
    emulation.OnPresent();
    emulation.ReleaseCommandBuffer(command_buffer_);
    emulation.DestroyDevice(device_);
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_EQ(log,
            "  [vulkan] ETC2/EAC emulation enabled: compressed uploads decode "
            "to RGBA8/R16 host images\n"
            "  [vulkan] ETC2 queues: family 0 (G|C|T|S) x1 of 16, family 1 "
            "(T) x1 of 2\n"
            "  [vulkan] ETC2 image 4x4 mips=1 format=147 -> host format=37\n"
            "  [vulkan] ETC2 upload decoded 4x4 layers=1\n"
            "  [vulkan] ETC2 decode totals: images=1 (upscaled 0) uploads=1 "
            "in 1 submits compressed=0.0 MiB host=0.0 MiB (8.0x) decode=0 ms "
            "longest=0 ms\n");
  EXPECT_EQ(fake_.buffers_created, fake_.buffers_destroyed);
}

TEST_F(Etc2DiagnosticsTest, DecodesTheCpuRouteAsBefore) {
  VulkanEtc2Emulation emulation(Options(false));
  testing::internal::CaptureStderr();
  Register(&emulation);
  const VkImage image = CreateEtc2Image(&emulation);
  const VkBuffer source = CreateSource(&emulation, 3);
  Copy(&emulation, command_buffer_, source, image);
  ASSERT_EQ(fake_.copies.size(), 1U);
  EXPECT_NE(fake_.copies[0].source, source) << "reads from staging";
  EXPECT_EQ(fake_.copies[0].regions[0].bufferOffset, 0U);
  Submit(&emulation, {command_buffer_});
  std::uint8_t expected[4 * 4 * 4];
  ASSERT_TRUE(DecodeEtcImage(EtcFormat::kEtc2Rgb8, kBlock, sizeof(kBlock), 4,
                             4, expected, sizeof(expected)));
  EXPECT_EQ(std::memcmp(StagedTexels(), expected, sizeof(expected)), 0);
  emulation.DestroyDevice(device_);
  testing::internal::GetCapturedStderr();
  EXPECT_EQ(fake_.buffers_created, fake_.buffers_destroyed);
  EXPECT_EQ(fake_.memories_allocated, fake_.memories_freed + 1)
      << "only the test's own source memory stays";
}

TEST_F(Etc2DiagnosticsTest, Submit2DecodesABatchInOnePass) {
  VulkanEtc2Emulation emulation(Options(false));
  testing::internal::CaptureStderr();
  Register(&emulation);
  const VkImage image = CreateEtc2Image(&emulation);
  const VkBuffer source = CreateSource(&emulation, 3);
  const VkCommandBuffer second = FakeHandle<VkCommandBuffer>(0xc7);
  Copy(&emulation, command_buffer_, source, image);
  Copy(&emulation, second, source, image);
  VkCommandBufferSubmitInfo infos[2]{};
  infos[0].sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
  infos[0].commandBuffer = command_buffer_;
  infos[1] = infos[0];
  infos[1].commandBuffer = second;
  VkSubmitInfo2 submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
  submit.commandBufferInfoCount = 2;
  submit.pCommandBufferInfos = infos;
  Etc2SubmitPlan plan;
  emulation.PrepareSubmit2(queue_, 1, &submit, VK_NULL_HANDLE, &plan);
  EXPECT_EQ(plan.submits2(&submit), &submit);
  emulation.DestroyDevice(device_);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("uploads=2 in 1 submits"), std::string::npos) << log;
}

TEST_F(Etc2DiagnosticsTest, DestroyDeviceDropsEveryRecordOfTheDevice) {
  testing::internal::CaptureStderr();
  VulkanEtc2Emulation emulation(Options(false));
  Register(&emulation);
  emulation.RegisterCommandPool(device_, graphics_pool_, 0, 0);
  const VkImage image = CreateEtc2Image(&emulation);
  const VkBuffer source = CreateSource(&emulation, 3);
  emulation.BeginCommandBuffer(command_buffer_, graphics_pool_, 0);
  emulation.BeginCommandBuffer(secondary_, graphics_pool_, 0);
  Copy(&emulation, command_buffer_, source, image);
  emulation.DestroyDevice(device_);
  EXPECT_EQ(fake_.buffers_created, fake_.buffers_destroyed);

  // The loader hands the same handles to a new device's command buffers.
  Register(&emulation);
  emulation.ReleaseCommandBuffer(command_buffer_);
  emulation.ReleaseCommandBuffer(secondary_);
  Submit(&emulation, {command_buffer_, secondary_});
  emulation.DestroyDevice(device_);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_EQ(log.find("decoded"), std::string::npos) << log;
  EXPECT_EQ(fake_.buffers_created, fake_.buffers_destroyed);
}

TEST_F(Etc2DiagnosticsTest, BeginReleasesTheLastRecording) {
  VulkanEtc2Emulation emulation(Options(false));
  testing::internal::CaptureStderr();
  Register(&emulation);
  emulation.RegisterCommandPool(device_, graphics_pool_, 0, 0);
  const VkImage image = CreateEtc2Image(&emulation);
  const VkBuffer source = CreateSource(&emulation, 3);
  emulation.BeginCommandBuffer(command_buffer_, graphics_pool_, 0);
  Copy(&emulation, command_buffer_, source, image);
  Copy(&emulation, command_buffer_, source, image);
  EXPECT_EQ(fake_.buffers_created - fake_.buffers_destroyed, 2);
  emulation.BeginCommandBuffer(command_buffer_, graphics_pool_, 0);
  EXPECT_EQ(fake_.buffers_created, fake_.buffers_destroyed);
  emulation.ForgetCommandPool(device_, graphics_pool_);
  emulation.DestroyDevice(device_);
  testing::internal::GetCapturedStderr();
}

TEST_F(Etc2DiagnosticsTest, FrameStatsCountTheUploadShape) {
  testing::internal::CaptureStderr();
  VulkanEtc2Emulation emulation(Options(true));
  Register(&emulation);
  emulation.RegisterQueue(device_, queue_, 0, 0);
  emulation.RegisterCommandPool(device_, graphics_pool_, 0, 0);
  emulation.RegisterCommandPool(device_, transfer_pool_, 1, 0);
  emulation.OnPresent();  // starts the first window

  const VkImage image = CreateEtc2Image(&emulation);
  const VkBuffer cached = CreateSource(&emulation, 4);
  const VkBuffer device_local = CreateSource(&emulation, 1);
  emulation.BeginCommandBuffer(command_buffer_, graphics_pool_,
                               VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
  emulation.BeginCommandBuffer(secondary_, transfer_pool_,
                               VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  Copy(&emulation, command_buffer_, cached, image, 2);
  emulation.NoteDispatch(secondary_);
  Copy(&emulation, secondary_, device_local, image);
  emulation.CmdExecuteCommands(device_, command_buffer_, 1, &secondary_);
  Submit(&emulation, {command_buffer_});

  // Aliasing: an uncompressed image over the emulated one's memory, a view
  // that reads it as another format, and an image copy into it.
  VkImageCreateInfo raw_info{};
  raw_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  raw_info.imageType = VK_IMAGE_TYPE_2D;
  raw_info.format = VK_FORMAT_R32G32_UINT;
  raw_info.extent = {1, 1, 1};
  raw_info.mipLevels = 1;
  raw_info.arrayLayers = 1;
  VkImage raw = VK_NULL_HANDLE;
  ASSERT_EQ(emulation.CreateImage(device_, &raw_info, nullptr, &raw),
            VK_SUCCESS);
  VkMemoryAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate.allocationSize = 1 << 16;
  allocate.memoryTypeIndex = 1;
  VkDeviceMemory heap = VK_NULL_HANDLE;
  ASSERT_EQ(emulation.AllocateMemory(device_, &allocate, nullptr, &heap),
            VK_SUCCESS);
  // Images side by side are not aliasing; one over another is.
  EXPECT_EQ(emulation.BindImageMemory(device_, image, heap, 0), VK_SUCCESS);
  EXPECT_EQ(emulation.BindImageMemory(device_, raw, heap, 4096), VK_SUCCESS);
  const VkImage overlay = CreateEtc2Image(&emulation);
  EXPECT_EQ(emulation.BindImageMemory(device_, overlay, heap, 2048),
            VK_SUCCESS);
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = VK_FORMAT_R32G32_UINT;
  VkImageView view = VK_NULL_HANDLE;
  EXPECT_EQ(emulation.CreateImageView(device_, &view_info, nullptr, &view),
            VK_SUCCESS);
  view_info.format = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
  EXPECT_EQ(emulation.CreateImageView(device_, &view_info, nullptr, &view),
            VK_SUCCESS);
  VkImageCopy region{};
  emulation.CmdCopyImage(device_, command_buffer_, raw,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  EXPECT_EQ(fake_.image_copies, 1);
  emulation.DestroyImage(device_, overlay, nullptr);
  emulation.ReleaseCommandBuffer(command_buffer_);
  emulation.ReleaseCommandBuffer(secondary_);

  std::this_thread::sleep_for(std::chrono::milliseconds(2050));
  emulation.OnPresent();
  emulation.OnPresent();  // a fresh, empty window prints nothing
  emulation.DestroyImage(device_, image, nullptr);
  emulation.DestroyDevice(device_);
  const std::string log = testing::internal::GetCapturedStderr();

  const std::size_t window = log.find("  [perf] etc2_window ");
  ASSERT_NE(window, std::string::npos) << log;
  const std::size_t line_end = log.find('\n', window);
  EXPECT_EQ(log.find("etc2_window", line_end), std::string::npos)
      << "one window had ETC2 work";
  const std::string line = log.substr(window, line_end - window);
  for (const char* field :
       {"submits=1 ", "record_calls=2 ", "staging_allocs=2 ", "uploads=3 ",
        "gpu=0 ", "fallback=3 ", "images=+2/-1 ", "copyimg=1 ",
        "aliasviews=1 ", "regions=3 ", "fam=0:2/0.0,1:1/0.0 ",
        "nocompute=1/0.0 ", "unknownfam=0 ", "begin=ot:1,sim:1,other:0 ",
        "simcbs=1 ", "sec=1/1 ", "simsec=0 ", "submit=1/1 ", "submit2=0/0 ",
        "upload_batches=1 ", "multibatch=0 ", "src=t1:0.0,t4:0.0 ",
        "nonhost=1 ", "rawcopies=1 ", "aliasbinds=1 ", "afterdispatch=1 ",
        "copies=1/0/0/0/0", "new_threads=\""}) {
    EXPECT_NE(line.find(field), std::string::npos) << field << " in " << line;
  }

  const std::size_t totals = log.find("  [vulkan] ETC2 decode totals: ");
  ASSERT_NE(totals, std::string::npos) << log;
  const std::string totals_line =
      log.substr(totals, log.find('\n', totals) - totals);
  EXPECT_NE(totals_line.find("images=2 (upscaled 0) uploads=3 in 1 submits "),
            std::string::npos)
      << totals_line;
  for (const char* field :
       {" | gpu uploads=0 ", " | cpu_route gpu_disabled=0 gpu_not_ready=0 "
                             "gpu_unavailable=3 ",
        " | arena peak=0.0 MiB ", " | record calls=2 ", "destroyed=2 ",
        "copies=1/0/1/0/0"}) {
    EXPECT_NE(totals_line.find(field), std::string::npos)
        << field << " in " << totals_line;
  }
}

TEST_F(Etc2DiagnosticsTest, GpuDisabledIsTheCpuRouteReason) {
  testing::internal::CaptureStderr();
  Etc2EmulationOptions options = Options(true);
  options.gpu = false;
  VulkanEtc2Emulation emulation(options);
  Register(&emulation);
  const VkImage image = CreateEtc2Image(&emulation);
  const VkBuffer source = CreateSource(&emulation, 3);
  Copy(&emulation, command_buffer_, source, image);
  emulation.DestroyDevice(device_);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("gpu_disabled=1 gpu_not_ready=0 gpu_unavailable=0"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("unknownfam=1"), std::string::npos) << log;
}

TEST_F(Etc2DiagnosticsTest, ANativeEtc2DeviceIsLeftAlone) {
  testing::internal::CaptureStderr();
  VulkanEtc2Emulation emulation(Options(true));
  Register(&emulation, false);
  emulation.RegisterCommandPool(device_, graphics_pool_, 0, 0);
  emulation.BeginCommandBuffer(command_buffer_, graphics_pool_, 0);
  const VkImage image = CreateEtc2Image(&emulation);
  EXPECT_EQ(fake_.images.at(image), VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK);
  VkBufferImageCopy region{};
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {4, 4, 1};
  const VkBuffer source = FakeHandle<VkBuffer>(0x4242);
  emulation.CmdCopyBufferToImage(device_, command_buffer_, source, image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                 &region);
  ASSERT_EQ(fake_.copies.size(), 1U);
  EXPECT_EQ(fake_.copies[0].source, source);
  emulation.DestroyDevice(device_);
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "");
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::stringstream text;
  text << file.rdbuf();
  return text.str();
}

TEST_F(Etc2DiagnosticsTest, TracesEveryUploadAndKeepsDistinctBlocks) {
  const char* temporary = std::getenv("TMPDIR");
  std::string pattern = std::string(temporary != nullptr && temporary[0] != '\0'
                                        ? temporary
                                        : "/tmp") +
                        "/etc2-trace-XXXXXX";
  ASSERT_NE(mkdtemp(pattern.data()), nullptr);
  const std::string directory = pattern;
  testing::internal::CaptureStderr();
  {
    Etc2EmulationOptions options = Options(false);
    options.trace_path = directory + "/trace.csv";
    options.trace_blocks_dir = directory + "/blocks";
    VulkanEtc2Emulation emulation(options);
    Register(&emulation);
    const VkImage image = CreateEtc2Image(&emulation);
    const VkBuffer source = CreateSource(&emulation, 3);
    Copy(&emulation, command_buffer_, source, image);
    Copy(&emulation, command_buffer_, source, image);
    Submit(&emulation, {command_buffer_});
    emulation.DestroyDevice(device_);
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("ETC2 upload trace: " + directory + "/trace.csv"),
            std::string::npos)
      << log;

  const std::string hash = HashName(HashBytes(kBlock, sizeof(kBlock)));
  std::istringstream trace(ReadFile(directory + "/trace.csv"));
  std::vector<std::string> lines;
  for (std::string line; std::getline(trace, line);) {
    lines.push_back(line);
  }
  ASSERT_EQ(lines.size(), 3U);
  EXPECT_EQ(lines[0],
            "t_ns,submit,record_call,image,vkformat,w,h,mip,layers,scale,"
            "row_length,image_height,offset,bytes,src_mem_type,route,hash64");
  for (std::size_t index = 1; index < 3; ++index) {
    const std::string& line = lines[index];
    const std::size_t image = line.find(",0x");
    ASSERT_NE(image, std::string::npos) << line;
    EXPECT_EQ(line.substr(line.find(',') + 1, image - line.find(',') - 1),
              "1," + std::to_string(index));
    EXPECT_EQ(line.substr(line.find(',', image + 1)),
              ",147,4,4,0,1,1,0,0,0:0,8,3,cpu," + hash);
  }

  EXPECT_EQ(ReadFile(directory + "/blocks/etc2_blocks.bin"),
            std::string(reinterpret_cast<const char*>(kBlock), 8));
  EXPECT_EQ(ReadFile(directory + "/blocks/etc2_blocks.csv"),
            "offset,bytes,vkformat,w,h,mip,layers,hash64\n0,8,147,4,4,0,1," +
                hash + "\n");
  std::remove((directory + "/trace.csv").c_str());
  std::remove((directory + "/blocks/etc2_blocks.bin").c_str());
  std::remove((directory + "/blocks/etc2_blocks.csv").c_str());
  rmdir((directory + "/blocks").c_str());
  rmdir(directory.c_str());
}

}  // namespace
}  // namespace mocktail::graphics
