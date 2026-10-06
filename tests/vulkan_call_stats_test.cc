#include "mocktail/graphics/vulkan_call_stats.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mocktail::graphics {
namespace {

// One stand-in host entry point per wrapped name, counting its own calls.
enum FakeId : std::size_t {
#define MOCKTAIL_TEST_FAKE_ID(name, counter, ret, params, args, extra) \
  kFake_##name,
  MOCKTAIL_VK_CALL_STATS_WRAPPED(MOCKTAIL_TEST_FAKE_ID)
#undef MOCKTAIL_TEST_FAKE_ID
      kFakeCount
};
int g_fake_calls[kFakeCount];

#define MOCKTAIL_TEST_FAKE(name, counter, ret, params, args, extra) \
  VKAPI_ATTR ret VKAPI_CALL Fake_##name params {                    \
    ++g_fake_calls[kFake_##name];                                   \
    return ret();                                                   \
  }
MOCKTAIL_VK_CALL_STATS_WRAPPED(MOCKTAIL_TEST_FAKE)
#undef MOCKTAIL_TEST_FAKE

// Calls any entry point with zero, null or VK_NULL_HANDLE for every
// argument.
template <typename Result, typename... Arguments>
Result CallWithZeros(Result (*function)(Arguments...)) {
  return function(Arguments{}...);
}

VKAPI_ATTR void VKAPI_CALL HostStandIn() {}
VKAPI_ATTR void VKAPI_CALL OtherTrampoline() {}

class VulkanCallStatsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetVkCallStatsForTesting();
    ForceVkCallStatsEnabledForTesting(true);
    std::fill(std::begin(g_fake_calls), std::end(g_fake_calls), 0);
  }
  void TearDown() override { ResetVkCallStatsForTesting(); }

  static VkCallStatsWindow Take() {
    VkCallStatsWindow window;
    TakeVkCallStatsCounts(&window);
    return window;
  }
};

// Written out by hand, so a change to the X-macro shows up here.
const std::map<std::string, VkCallCounter>& ExpectedWrappers() {
  static const std::map<std::string, VkCallCounter> expected = {
      {"vkCmdDraw", VkCallCounter::kDraw},
      {"vkCmdDrawIndexed", VkCallCounter::kDrawIndexed},
      {"vkCmdDrawIndirect", VkCallCounter::kDrawIndirect},
      {"vkCmdDrawIndexedIndirect", VkCallCounter::kDrawIndirect},
      {"vkCmdDrawIndirectCount", VkCallCounter::kDrawIndirect},
      {"vkCmdDrawIndirectCountKHR", VkCallCounter::kDrawIndirect},
      {"vkCmdDrawIndexedIndirectCount", VkCallCounter::kDrawIndirect},
      {"vkCmdDrawIndexedIndirectCountKHR", VkCallCounter::kDrawIndirect},
      {"vkCmdDispatch", VkCallCounter::kDispatch},
      {"vkCmdDispatchIndirect", VkCallCounter::kDispatch},
      {"vkCmdDispatchBase", VkCallCounter::kDispatch},
      {"vkCmdDispatchBaseKHR", VkCallCounter::kDispatch},
      {"vkCmdBindPipeline", VkCallCounter::kBindPipeline},
      {"vkCmdBindDescriptorSets", VkCallCounter::kBindDescriptorSets},
      {"vkCmdPushConstants", VkCallCounter::kPushConstants},
      {"vkCmdBindVertexBuffers", VkCallCounter::kBindVertexBuffers},
      {"vkCmdBindIndexBuffer", VkCallCounter::kBindIndexBuffer},
      {"vkCmdBeginRenderPass", VkCallCounter::kBeginRenderPass},
      {"vkCmdBeginRenderPass2", VkCallCounter::kBeginRenderPass},
      {"vkCmdBeginRenderPass2KHR", VkCallCounter::kBeginRenderPass},
      {"vkCmdBeginRendering", VkCallCounter::kBeginRenderPass},
      {"vkCmdBeginRenderingKHR", VkCallCounter::kBeginRenderPass},
      {"vkCmdPipelineBarrier", VkCallCounter::kPipelineBarrier},
      {"vkCmdPipelineBarrier2", VkCallCounter::kPipelineBarrier},
      {"vkCmdPipelineBarrier2KHR", VkCallCounter::kPipelineBarrier},
      {"vkCmdCopyBuffer", VkCallCounter::kCopyBuffer},
      {"vkCmdCopyBuffer2", VkCallCounter::kCopyBuffer},
      {"vkCmdCopyBuffer2KHR", VkCallCounter::kCopyBuffer},
      {"vkCmdBlitImage", VkCallCounter::kBlitImage},
      {"vkCmdBlitImage2", VkCallCounter::kBlitImage},
      {"vkCmdBlitImage2KHR", VkCallCounter::kBlitImage},
      {"vkCmdCopyImageToBuffer", VkCallCounter::kCopyImageToBuffer},
      {"vkCmdCopyImageToBuffer2", VkCallCounter::kCopyImageToBuffer},
      {"vkCmdCopyImageToBuffer2KHR", VkCallCounter::kCopyImageToBuffer},
      {"vkCmdUpdateBuffer", VkCallCounter::kUpdateBuffer},
      {"vkCmdClearColorImage", VkCallCounter::kClear},
      {"vkCmdClearDepthStencilImage", VkCallCounter::kClear},
      {"vkCmdClearAttachments", VkCallCounter::kClear},
      {"vkCmdResolveImage", VkCallCounter::kResolveImage},
      {"vkCmdResolveImage2", VkCallCounter::kResolveImage},
      {"vkCmdResolveImage2KHR", VkCallCounter::kResolveImage},
      {"vkUpdateDescriptorSets", VkCallCounter::kUpdateDescriptorSets},
      {"vkUpdateDescriptorSetWithTemplate",
       VkCallCounter::kUpdateDescriptorSets},
      {"vkUpdateDescriptorSetWithTemplateKHR",
       VkCallCounter::kUpdateDescriptorSets},
      {"vkAllocateDescriptorSets", VkCallCounter::kAllocateDescriptorSets},
      {"vkResetDescriptorPool", VkCallCounter::kResetDescriptorPool},
      {"vkFlushMappedMemoryRanges", VkCallCounter::kFlushMappedMemory},
      {"vkInvalidateMappedMemoryRanges",
       VkCallCounter::kInvalidateMappedMemory},
      {"vkCreateBuffer", VkCallCounter::kCreateBuffer},
      {"vkCreateGraphicsPipelines", VkCallCounter::kCreateGraphicsPipelines},
      {"vkCreateComputePipelines", VkCallCounter::kCreateComputePipelines},
      {"vkCreateShaderModule", VkCallCounter::kCreateShaderModule},
      {"vkCreateSampler", VkCallCounter::kCreateSampler},
      {"vkCreateFramebuffer", VkCallCounter::kCreateFramebuffer},
      {"vkGetFenceStatus", VkCallCounter::kGetFenceStatus},
  };
  return expected;
}

// Entry points bionic_vulkan_loader_adapter.cc hooks itself (AdapterProc),
// plus vkCreateCommandPool, vkAllocateMemory and vkBindImageMemory(2), which
// the plus.2 foundations hook. A wrapper for one of these would bypass the
// hook, so they count inside the hook instead.
constexpr const char* kAdapterHookedProcs[] = {
    "vkCreateInstance", "vkDestroyInstance", "vkCreateAndroidSurfaceKHR",
    "vkDestroySurfaceKHR", "vkEnumerateInstanceExtensionProperties",
    "vkEnumerateInstanceLayerProperties", "vkGetInstanceProcAddr",
    "vkGetDeviceProcAddr", "vkCreateDevice", "vkDestroyDevice",
    "vkGetDeviceQueue", "vkGetDeviceQueue2", "vkCreateSwapchainKHR",
    "vkDestroySwapchainKHR", "vkAcquireNextImageKHR",
    "vkAcquireNextImage2KHR", "vkWaitForFences", "vkResetFences",
    "vkResetCommandPool", "vkGetQueryPoolResults", "vkAllocateCommandBuffers",
    "vkFreeCommandBuffers", "vkDestroyCommandPool", "vkBeginCommandBuffer",
    "vkEndCommandBuffer", "vkResetCommandBuffer", "vkWaitSemaphores",
    "vkWaitSemaphoresKHR", "vkQueueSubmit", "vkQueueSubmit2",
    "vkQueueSubmit2KHR", "vkQueueBindSparse", "vkQueueWaitIdle",
    "vkDeviceWaitIdle", "vkQueuePresentKHR", "vkGetPhysicalDeviceFeatures",
    "vkGetPhysicalDeviceFeatures2", "vkGetPhysicalDeviceFeatures2KHR",
    "vkGetPhysicalDeviceFormatProperties",
    "vkGetPhysicalDeviceFormatProperties2",
    "vkGetPhysicalDeviceFormatProperties2KHR",
    "vkGetPhysicalDeviceImageFormatProperties",
    "vkGetPhysicalDeviceImageFormatProperties2",
    "vkGetPhysicalDeviceImageFormatProperties2KHR", "vkCreateImage",
    "vkDestroyImage", "vkCreateImageView", "vkBindBufferMemory",
    "vkBindBufferMemory2", "vkBindBufferMemory2KHR", "vkMapMemory",
    "vkMapMemory2", "vkMapMemory2KHR", "vkUnmapMemory", "vkUnmapMemory2",
    "vkUnmapMemory2KHR", "vkFreeMemory", "vkDestroyBuffer",
    "vkCmdCopyBufferToImage", "vkCmdCopyBufferToImage2",
    "vkCmdCopyBufferToImage2KHR", "vkCmdCopyImage", "vkCmdCopyImage2",
    "vkCmdCopyImage2KHR", "vkCmdExecuteCommands", "vkCreateCommandPool",
    "vkAllocateMemory", "vkBindImageMemory", "vkBindImageMemory2",
    "vkBindImageMemory2KHR",
};

TEST_F(VulkanCallStatsTest, WrapsExactlyTheListedEntryPoints) {
  std::size_t count = 0;
  const VkCallStatsProcInfo* procs = VkCallStatsProcs(&count);
  ASSERT_NE(procs, nullptr);
  EXPECT_EQ(count, ExpectedWrappers().size());
  EXPECT_EQ(count, static_cast<std::size_t>(kFakeCount));
  std::set<std::string> names;
  std::set<PFN_vkVoidFunction> wrappers;
  for (std::size_t i = 0; i < count; ++i) {
    const auto expected = ExpectedWrappers().find(procs[i].name);
    ASSERT_NE(expected, ExpectedWrappers().end()) << procs[i].name;
    EXPECT_EQ(procs[i].counter, expected->second) << procs[i].name;
    EXPECT_NE(procs[i].wrapper, nullptr) << procs[i].name;
    EXPECT_TRUE(IsVkCallStatsProc(procs[i].name)) << procs[i].name;
    names.insert(procs[i].name);
    wrappers.insert(procs[i].wrapper);
  }
  EXPECT_EQ(names.size(), count);
  EXPECT_EQ(wrappers.size(), count);
}

TEST_F(VulkanCallStatsTest, WrapsNothingTheAdapterHooks) {
  for (const char* name : kAdapterHookedProcs) {
    EXPECT_FALSE(IsVkCallStatsProc(name)) << name;
    EXPECT_EQ(SelectVkCallStatsProc(true, name, &HostStandIn, &HostStandIn),
              &HostStandIn)
        << name;
  }
}

TEST_F(VulkanCallStatsTest, OffReturnsTheHostPointerForEveryName) {
  std::size_t count = 0;
  const VkCallStatsProcInfo* procs = VkCallStatsProcs(&count);
  std::vector<const char*> names = {"vkQueueSubmit", "vkCmdDrawMeshTasksEXT",
                                    "vkNotAFunction", ""};
  for (std::size_t i = 0; i < count; ++i) {
    names.push_back(procs[i].name);
  }
  for (const char* name : names) {
    EXPECT_EQ(SelectVkCallStatsProc(false, name, &HostStandIn,
                                    &OtherTrampoline),
              &HostStandIn)
        << name;
    EXPECT_EQ(SelectVkCallStatsProc(false, name, nullptr, &OtherTrampoline),
              nullptr)
        << name;
  }
  EXPECT_EQ(SelectVkCallStatsProc(false, nullptr, &HostStandIn, nullptr),
            &HostStandIn);
  // Nothing was bound while off: the first enabled lookup can still bind.
  EXPECT_NE(SelectVkCallStatsProc(
                true, "vkCmdDraw", &HostStandIn,
                reinterpret_cast<PFN_vkVoidFunction>(&Fake_vkCmdDraw)),
            &HostStandIn);
}

TEST_F(VulkanCallStatsTest, OnWrapsOnlyEntryPointsTheDeviceHas) {
  const PFN_vkVoidFunction trampoline =
      reinterpret_cast<PFN_vkVoidFunction>(&Fake_vkCmdDraw);
  // Unsupported stays unsupported.
  EXPECT_EQ(SelectVkCallStatsProc(true, "vkCmdDraw", nullptr, trampoline),
            nullptr);
  // Without a loader entry point to forward to, no wrapper.
  EXPECT_EQ(SelectVkCallStatsProc(true, "vkCmdDraw", &HostStandIn, nullptr),
            &HostStandIn);
  EXPECT_EQ(SelectVkCallStatsProc(true, "vkNotAFunction", &HostStandIn,
                                  trampoline),
            &HostStandIn);
  EXPECT_EQ(SelectVkCallStatsProc(true, nullptr, &HostStandIn, trampoline),
            &HostStandIn);
  const PFN_vkVoidFunction wrapper =
      SelectVkCallStatsProc(true, "vkCmdDraw", &HostStandIn, trampoline);
  EXPECT_NE(wrapper, &HostStandIn);
  EXPECT_NE(wrapper, trampoline);
}

TEST_F(VulkanCallStatsTest, KeepsTheFirstTrampolineItWasBoundTo) {
  const PFN_vkVoidFunction trampoline =
      reinterpret_cast<PFN_vkVoidFunction>(&Fake_vkCmdDraw);
  const PFN_vkVoidFunction wrapper =
      SelectVkCallStatsProc(true, "vkCmdDraw", &HostStandIn, trampoline);
  ASSERT_NE(wrapper, &HostStandIn);
  EXPECT_EQ(SelectVkCallStatsProc(true, "vkCmdDraw", &HostStandIn, trampoline),
            wrapper);
  // Another entry point never silently replaces the bound one.
  EXPECT_EQ(SelectVkCallStatsProc(true, "vkCmdDraw", &HostStandIn,
                                  &OtherTrampoline),
            &HostStandIn);
  reinterpret_cast<PFN_vkCmdDraw>(wrapper)(VK_NULL_HANDLE, 3, 1, 0, 0);
  EXPECT_EQ(g_fake_calls[kFake_vkCmdDraw], 1);
}

TEST_F(VulkanCallStatsTest, EveryWrapperForwardsAndCountsItsOwnCounter) {
#define MOCKTAIL_TEST_CALL(name, counter, ret, params, args, extra)          \
  {                                                                          \
    const PFN_vkVoidFunction trampoline =                                    \
        reinterpret_cast<PFN_vkVoidFunction>(&Fake_##name);                  \
    const PFN_vkVoidFunction wrapper =                                       \
        SelectVkCallStatsProc(true, #name, &HostStandIn, trampoline);        \
    ASSERT_NE(wrapper, &HostStandIn) << #name;                               \
    CallWithZeros(reinterpret_cast<PFN_##name>(wrapper));                    \
    EXPECT_EQ(g_fake_calls[kFake_##name], 1) << #name;                       \
    const VkCallStatsWindow window = Take();                                 \
    for (std::size_t i = 0; i < kVkCallCounterCount; ++i) {                  \
      EXPECT_EQ(window.calls[i],                                             \
                i == static_cast<std::size_t>(VkCallCounter::counter) ? 1U   \
                                                                      : 0U)  \
          << #name << " counter " << i;                                      \
    }                                                                        \
  }
  MOCKTAIL_VK_CALL_STATS_WRAPPED(MOCKTAIL_TEST_CALL)
#undef MOCKTAIL_TEST_CALL
}

TEST_F(VulkanCallStatsTest, WrappersPassArgumentsAndResultsThrough) {
  static VkCommandBuffer seen_buffer = VK_NULL_HANDLE;
  static std::uint32_t seen_count = 0;
  const auto host_draw = [](VkCommandBuffer cb, std::uint32_t vertex_count,
                            std::uint32_t, std::uint32_t, std::uint32_t) {
    seen_buffer = cb;
    seen_count = vertex_count;
  };
  const auto draw = reinterpret_cast<PFN_vkCmdDraw>(SelectVkCallStatsProc(
      true, "vkCmdDraw", &HostStandIn,
      reinterpret_cast<PFN_vkVoidFunction>(+host_draw)));
  const auto cb = reinterpret_cast<VkCommandBuffer>(std::uintptr_t{0x1234});
  draw(cb, 36, 1, 0, 0);
  EXPECT_EQ(seen_buffer, cb);
  EXPECT_EQ(seen_count, 36U);

  const auto host_status = [](VkDevice, VkFence) { return VK_NOT_READY; };
  const auto status = reinterpret_cast<PFN_vkGetFenceStatus>(
      SelectVkCallStatsProc(true, "vkGetFenceStatus", &HostStandIn,
                            reinterpret_cast<PFN_vkVoidFunction>(+host_status)));
  EXPECT_EQ(status(VK_NULL_HANDLE, VK_NULL_HANDLE), VK_NOT_READY);

  const VkCallStatsWindow window = Take();
  EXPECT_EQ(window.calls[static_cast<std::size_t>(VkCallCounter::kDraw)], 1U);
  EXPECT_EQ(
      window.calls[static_cast<std::size_t>(VkCallCounter::kGetFenceStatus)],
      1U);
}

TEST_F(VulkanCallStatsTest, CountsCopiedUpdatedAndFlushedBytes) {
  const auto wrap = [](const char* name, PFN_vkVoidFunction fake) {
    return SelectVkCallStatsProc(true, name, &HostStandIn, fake);
  };
  const auto copy = reinterpret_cast<PFN_vkCmdCopyBuffer>(wrap(
      "vkCmdCopyBuffer",
      reinterpret_cast<PFN_vkVoidFunction>(&Fake_vkCmdCopyBuffer)));
  const VkBufferCopy regions[] = {{0, 0, 100}, {100, 200, 28}};
  copy(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 2, regions);

  const auto copy2 = reinterpret_cast<PFN_vkCmdCopyBuffer2>(wrap(
      "vkCmdCopyBuffer2",
      reinterpret_cast<PFN_vkVoidFunction>(&Fake_vkCmdCopyBuffer2)));
  VkBufferCopy2 region2{};
  region2.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
  region2.size = 1024;
  VkCopyBufferInfo2 info{};
  info.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
  info.regionCount = 1;
  info.pRegions = &region2;
  copy2(VK_NULL_HANDLE, &info);
  copy2(VK_NULL_HANDLE, nullptr);

  const auto update = reinterpret_cast<PFN_vkCmdUpdateBuffer>(wrap(
      "vkCmdUpdateBuffer",
      reinterpret_cast<PFN_vkVoidFunction>(&Fake_vkCmdUpdateBuffer)));
  update(VK_NULL_HANDLE, VK_NULL_HANDLE, 0, 64, nullptr);

  const auto flush = reinterpret_cast<PFN_vkFlushMappedMemoryRanges>(wrap(
      "vkFlushMappedMemoryRanges",
      reinterpret_cast<PFN_vkVoidFunction>(&Fake_vkFlushMappedMemoryRanges)));
  VkMappedMemoryRange ranges[2] = {};
  ranges[0].size = 32;
  ranges[1].size = VK_WHOLE_SIZE;  // size unknown here: not counted
  EXPECT_EQ(flush(VK_NULL_HANDLE, 2, ranges), VK_SUCCESS);

  const auto invalidate =
      reinterpret_cast<PFN_vkInvalidateMappedMemoryRanges>(wrap(
          "vkInvalidateMappedMemoryRanges",
          reinterpret_cast<PFN_vkVoidFunction>(
              &Fake_vkInvalidateMappedMemoryRanges)));
  invalidate(VK_NULL_HANDLE, 1, ranges);

  const VkCallStatsWindow window = Take();
  const auto bytes = [&window](VkCallCounter counter) {
    return window.bytes[static_cast<std::size_t>(counter)];
  };
  const auto calls = [&window](VkCallCounter counter) {
    return window.calls[static_cast<std::size_t>(counter)];
  };
  EXPECT_EQ(calls(VkCallCounter::kCopyBuffer), 3U);
  EXPECT_EQ(bytes(VkCallCounter::kCopyBuffer), 1152U);
  EXPECT_EQ(calls(VkCallCounter::kUpdateBuffer), 1U);
  EXPECT_EQ(bytes(VkCallCounter::kUpdateBuffer), 64U);
  EXPECT_EQ(calls(VkCallCounter::kFlushMappedMemory), 1U);
  EXPECT_EQ(bytes(VkCallCounter::kFlushMappedMemory), 32U);
  EXPECT_EQ(calls(VkCallCounter::kInvalidateMappedMemory), 1U);
  EXPECT_EQ(bytes(VkCallCounter::kInvalidateMappedMemory), 32U);
}

TEST_F(VulkanCallStatsTest, DispatchWrappersReportTheirCommandBuffer) {
  static std::vector<VkCommandBuffer> seen;
  seen.clear();
  SetVkCallStatsDispatchObserver(
      [](VkCommandBuffer cb) { seen.push_back(cb); });
  const auto cb = reinterpret_cast<VkCommandBuffer>(std::uintptr_t{0x77});
  const auto dispatch = reinterpret_cast<PFN_vkCmdDispatch>(
      SelectVkCallStatsProc(true, "vkCmdDispatch", &HostStandIn,
                            reinterpret_cast<PFN_vkVoidFunction>(
                                &Fake_vkCmdDispatch)));
  const auto indirect = reinterpret_cast<PFN_vkCmdDispatchIndirect>(
      SelectVkCallStatsProc(true, "vkCmdDispatchIndirect", &HostStandIn,
                            reinterpret_cast<PFN_vkVoidFunction>(
                                &Fake_vkCmdDispatchIndirect)));
  const auto base = reinterpret_cast<PFN_vkCmdDispatchBaseKHR>(
      SelectVkCallStatsProc(true, "vkCmdDispatchBaseKHR", &HostStandIn,
                            reinterpret_cast<PFN_vkVoidFunction>(
                                &Fake_vkCmdDispatchBaseKHR)));
  const auto draw = reinterpret_cast<PFN_vkCmdDraw>(SelectVkCallStatsProc(
      true, "vkCmdDraw", &HostStandIn,
      reinterpret_cast<PFN_vkVoidFunction>(&Fake_vkCmdDraw)));
  dispatch(cb, 1, 1, 1);
  indirect(cb, VK_NULL_HANDLE, 0);
  base(cb, 0, 0, 0, 1, 1, 1);
  draw(cb, 3, 1, 0, 0);
  EXPECT_EQ(seen, std::vector<VkCommandBuffer>({cb, cb, cb}));
  EXPECT_EQ(g_fake_calls[kFake_vkCmdDispatch], 1);
  EXPECT_EQ(g_fake_calls[kFake_vkCmdDispatchIndirect], 1);
  EXPECT_EQ(g_fake_calls[kFake_vkCmdDispatchBaseKHR], 1);

  SetVkCallStatsDispatchObserver(nullptr);
  dispatch(cb, 1, 1, 1);
  EXPECT_EQ(seen.size(), 3U);
}

TEST_F(VulkanCallStatsTest, HookCountersCountOnlyWhileEnabled) {
  ForceVkCallStatsEnabledForTesting(false);
  VkCallStatsCount(VkCallCounter::kQueueSubmit);
  AddVkCallStatsBytes(VkCallCounter::kCopyBufferToImage, 4096);
  VkCallStatsCountCreateImage(VK_NULL_HANDLE,
                              VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK);
  AddVkCallStatsBlocked(VkBlockedCall::kWaitForFences, 5000000);
  VkCallStatsWindow window = Take();
  EXPECT_EQ(window.calls, decltype(window.calls){});
  EXPECT_EQ(window.bytes, decltype(window.bytes){});
  EXPECT_EQ(window.created_images, decltype(window.created_images){});
  EXPECT_EQ(window.blocked_ns, decltype(window.blocked_ns){});

  ForceVkCallStatsEnabledForTesting(true);
  VkCallStatsCount(VkCallCounter::kQueueSubmit);
  VkCallStatsCount(VkCallCounter::kQueueSubmit, 2);
  VkCallStatsCount(VkCallCounter::kCopyBufferToImage);
  AddVkCallStatsBytes(VkCallCounter::kCopyBufferToImage, 4096);
  VkCallStatsCountCreateImage(VK_NULL_HANDLE,
                              VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK);
  VkCallStatsCountCreateImage(VK_NULL_HANDLE, VK_FORMAT_BC7_UNORM_BLOCK);
  VkCallStatsCountCreateImage(VK_NULL_HANDLE, VK_FORMAT_R8G8B8A8_SRGB);
  AddVkCallStatsBlocked(VkBlockedCall::kWaitForFences, 5000000);
  AddVkCallStatsBlocked(VkBlockedCall::kWaitIdle, 7);
  VkCallStatsCount(VkCallCounter::kCount);  // ignored, out of range
  window = Take();
  EXPECT_EQ(window.calls[static_cast<std::size_t>(VkCallCounter::kQueueSubmit)],
            3U);
  EXPECT_EQ(window.bytes[static_cast<std::size_t>(
                VkCallCounter::kCopyBufferToImage)],
            4096U);
  EXPECT_EQ(window.calls[static_cast<std::size_t>(VkCallCounter::kCreateImage)],
            3U);
  EXPECT_EQ(window.created_images[static_cast<std::size_t>(
                VkImageFormatClass::kEtc2Eac)],
            1U);
  EXPECT_EQ(window.created_images[static_cast<std::size_t>(
                VkImageFormatClass::kBc7)],
            1U);
  EXPECT_EQ(window.created_images[static_cast<std::size_t>(
                VkImageFormatClass::kRgba8)],
            1U);
  EXPECT_EQ(window.blocked_ns[static_cast<std::size_t>(
                VkBlockedCall::kWaitForFences)],
            5000000U);
  EXPECT_EQ(
      window.blocked_ns[static_cast<std::size_t>(VkBlockedCall::kWaitIdle)],
      7U);
  // Taking a window starts again from zero.
  window = Take();
  EXPECT_EQ(window.calls, decltype(window.calls){});
}

VkImage FakeImage(std::uintptr_t value) {
  return reinterpret_cast<VkImage>(value);
}

VkBufferImageCopy ImageCopy(std::uint32_t width, std::uint32_t height,
                            std::uint32_t depth = 1,
                            std::uint32_t layers = 1) {
  VkBufferImageCopy region{};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = layers;
  region.imageExtent = {width, height, depth};
  return region;
}

TEST_F(VulkanCallStatsTest, RemembersImageFormatsForViewsAndCopies) {
  const VkImage etc2 = FakeImage(0x1000);
  const VkImage bc7 = FakeImage(0x2000);
  const VkImage unknown = FakeImage(0x3000);
  VkCallStatsCountCreateImage(etc2, VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK);
  VkCallStatsCountCreateImage(bc7, VK_FORMAT_BC7_SRGB_BLOCK);

  // Same format: not an alias. Another format, or a block-compatible uint
  // view as GPU compressors use: an alias. Unknown images never count.
  VkCallStatsCountImageView(etc2, VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK);
  VkCallStatsCountImageView(etc2, VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK);
  VkCallStatsCountImageView(bc7, VK_FORMAT_R32G32B32A32_UINT);
  VkCallStatsCountImageView(unknown, VK_FORMAT_R8G8B8A8_UNORM);
  VkCallStatsCountImageView(VK_NULL_HANDLE, VK_FORMAT_R8G8B8A8_UNORM);

  // 64x64 ETC2 RGBA8: 16x16 blocks of 16 bytes. 30x30 BC7 in two layers:
  // 8x8 blocks of 16 bytes per layer, the last row and column partial.
  const VkBufferImageCopy etc2_region = ImageCopy(64, 64);
  VkCallStatsCountCopyBufferToImage(etc2, 1, &etc2_region);
  VkBufferImageCopy2 bc7_region{};
  bc7_region.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
  bc7_region.imageSubresource.layerCount = 2;
  bc7_region.imageExtent = {30, 30, 1};
  VkCopyBufferToImageInfo2 bc7_copy{};
  bc7_copy.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
  bc7_copy.dstImage = bc7;
  bc7_copy.regionCount = 1;
  bc7_copy.pRegions = &bc7_region;
  VkCallStatsCountCopyBufferToImage2(&bc7_copy);
  // Counted, without bytes: the destination's format is not known.
  VkCallStatsCountCopyBufferToImage(unknown, 1, &etc2_region);
  VkCallStatsCountCopyBufferToImage2(nullptr);

  VkCallStatsWindow window = Take();
  const auto calls = [&window](VkCallCounter counter) {
    return window.calls[static_cast<std::size_t>(counter)];
  };
  EXPECT_EQ(calls(VkCallCounter::kCreateImage), 2U);
  EXPECT_EQ(calls(VkCallCounter::kAliasView), 2U);
  EXPECT_EQ(calls(VkCallCounter::kCopyBufferToImage), 4U);
  EXPECT_EQ(window.bytes[static_cast<std::size_t>(
                VkCallCounter::kCopyBufferToImage)],
            4096U + 2048U);

  // A destroyed image is forgotten; so is anything created while off.
  VkCallStatsCountDestroyImage(etc2);
  VkCallStatsCountImageView(etc2, VK_FORMAT_R32G32B32A32_UINT);
  ForceVkCallStatsEnabledForTesting(false);
  VkCallStatsCountCreateImage(unknown, VK_FORMAT_BC1_RGB_UNORM_BLOCK);
  VkCallStatsCountDestroyImage(bc7);
  VkCallStatsCountImageView(bc7, VK_FORMAT_R32G32B32A32_UINT);
  ForceVkCallStatsEnabledForTesting(true);
  VkCallStatsCountImageView(unknown, VK_FORMAT_R32G32_UINT);
  // bc7 was not forgotten by the destroy made while off.
  VkCallStatsCountImageView(bc7, VK_FORMAT_R32G32B32A32_UINT);
  window = Take();
  EXPECT_EQ(calls(VkCallCounter::kDestroyImage), 1U);
  EXPECT_EQ(calls(VkCallCounter::kCreateImage), 0U);
  EXPECT_EQ(calls(VkCallCounter::kAliasView), 1U);

  // A reset forgets every image.
  ResetVkCallStatsForTesting();
  ForceVkCallStatsEnabledForTesting(true);
  VkCallStatsCountImageView(bc7, VK_FORMAT_R32G32B32A32_UINT);
  window = Take();
  EXPECT_EQ(calls(VkCallCounter::kAliasView), 0U);
}

TEST_F(VulkanCallStatsTest, KnowsTheTexelBlockOfEveryCoreFormat) {
  struct Case {
    VkFormat format;
    std::uint32_t bytes;
    std::uint32_t width;
    std::uint32_t height;
  };
  const std::vector<Case> cases = {
      {VK_FORMAT_R4G4_UNORM_PACK8, 1, 1, 1},
      {VK_FORMAT_R8_UNORM, 1, 1, 1},
      {VK_FORMAT_S8_UINT, 1, 1, 1},
      {VK_FORMAT_B5G6R5_UNORM_PACK16, 2, 1, 1},
      {VK_FORMAT_R8G8_SRGB, 2, 1, 1},
      {VK_FORMAT_R16_SFLOAT, 2, 1, 1},
      {VK_FORMAT_D16_UNORM, 2, 1, 1},
      {VK_FORMAT_A4B4G4R4_UNORM_PACK16, 2, 1, 1},
      {VK_FORMAT_B8G8R8_SRGB, 3, 1, 1},
      {VK_FORMAT_R8G8B8A8_SRGB, 4, 1, 1},
      {VK_FORMAT_B8G8R8A8_UNORM, 4, 1, 1},
      {VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, 1, 1},
      {VK_FORMAT_R16G16_SFLOAT, 4, 1, 1},
      {VK_FORMAT_R32_SFLOAT, 4, 1, 1},
      {VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4, 1, 1},
      {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 4, 1, 1},
      {VK_FORMAT_X8_D24_UNORM_PACK32, 4, 1, 1},
      {VK_FORMAT_D32_SFLOAT, 4, 1, 1},
      {VK_FORMAT_R16G16B16_UNORM, 6, 1, 1},
      {VK_FORMAT_R16G16B16A16_SFLOAT, 8, 1, 1},
      {VK_FORMAT_R32G32_UINT, 8, 1, 1},
      {VK_FORMAT_R64_SFLOAT, 8, 1, 1},
      {VK_FORMAT_R32G32B32_SFLOAT, 12, 1, 1},
      {VK_FORMAT_R32G32B32A32_UINT, 16, 1, 1},
      {VK_FORMAT_R64G64_UINT, 16, 1, 1},
      {VK_FORMAT_R64G64B64_SINT, 24, 1, 1},
      {VK_FORMAT_R64G64B64A64_SFLOAT, 32, 1, 1},
      {VK_FORMAT_BC1_RGB_UNORM_BLOCK, 8, 4, 4},
      {VK_FORMAT_BC1_RGBA_SRGB_BLOCK, 8, 4, 4},
      {VK_FORMAT_BC2_UNORM_BLOCK, 16, 4, 4},
      {VK_FORMAT_BC3_SRGB_BLOCK, 16, 4, 4},
      {VK_FORMAT_BC4_SNORM_BLOCK, 8, 4, 4},
      {VK_FORMAT_BC5_UNORM_BLOCK, 16, 4, 4},
      {VK_FORMAT_BC6H_UFLOAT_BLOCK, 16, 4, 4},
      {VK_FORMAT_BC7_SRGB_BLOCK, 16, 4, 4},
      {VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, 8, 4, 4},
      {VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, 8, 4, 4},
      {VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, 16, 4, 4},
      {VK_FORMAT_EAC_R11_SNORM_BLOCK, 8, 4, 4},
      {VK_FORMAT_EAC_R11G11_UNORM_BLOCK, 16, 4, 4},
      {VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 16, 4, 4},
      {VK_FORMAT_ASTC_5x4_SRGB_BLOCK, 16, 5, 4},
      {VK_FORMAT_ASTC_8x6_UNORM_BLOCK, 16, 8, 6},
      {VK_FORMAT_ASTC_10x5_SRGB_BLOCK, 16, 10, 5},
      {VK_FORMAT_ASTC_12x12_SRGB_BLOCK, 16, 12, 12},
      {VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK, 16, 4, 4},
      {VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK, 16, 6, 5},
      {VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK, 16, 12, 10},
      {VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK, 16, 12, 12},
  };
  for (const Case& c : cases) {
    VkFormatTexelBlock block;
    ASSERT_TRUE(LookupVkFormatTexelBlock(c.format, &block)) << c.format;
    EXPECT_EQ(block.bytes, c.bytes) << c.format;
    EXPECT_EQ(block.width, c.width) << c.format;
    EXPECT_EQ(block.height, c.height) << c.format;
  }
  // Every Vulkan 1.0 format but the combined depth/stencil ones.
  for (int value = VK_FORMAT_R4G4_UNORM_PACK8;
       value <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++value) {
    const auto format = static_cast<VkFormat>(value);
    const bool depth_stencil = format == VK_FORMAT_D16_UNORM_S8_UINT ||
                               format == VK_FORMAT_D24_UNORM_S8_UINT ||
                               format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    EXPECT_NE(LookupVkFormatTexelBlock(format, nullptr), depth_stencil)
        << value;
  }
  for (const VkFormat format :
       {VK_FORMAT_UNDEFINED, VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_G8B8G8R8_422_UNORM, VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG}) {
    EXPECT_FALSE(LookupVkFormatTexelBlock(format, nullptr)) << format;
  }
}

TEST_F(VulkanCallStatsTest, CountsTheBytesBufferToImageCopiesWrite) {
  // RGBA8 3-D image: 16x8x4 texels.
  const VkBufferImageCopy volume = ImageCopy(16, 8, 4);
  EXPECT_EQ(BufferImageCopyBytes(VK_FORMAT_R8G8B8A8_UNORM, 1, &volume),
            16U * 8U * 4U * 4U);
  // A BC1 mip tail of 2x1 texels still writes a whole block; six cube faces
  // of 5x3 texels are 2x1 blocks each.
  const VkBufferImageCopy regions[] = {ImageCopy(2, 1), ImageCopy(5, 3, 1, 6)};
  EXPECT_EQ(BufferImageCopyBytes(VK_FORMAT_BC1_RGB_UNORM_BLOCK, 2, regions),
            8U + 6U * 2U * 8U);
  // VK_REMAINING_ARRAY_LAYERS counts one layer; zero depth counts one slice.
  const VkBufferImageCopy remaining =
      ImageCopy(8, 8, 0, VK_REMAINING_ARRAY_LAYERS);
  EXPECT_EQ(BufferImageCopyBytes(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 1,
                                 &remaining),
            2U * 2U * 8U);
  // ASTC 8x6: 20x13 texels are 3x3 blocks.
  VkBufferImageCopy2 astc{};
  astc.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
  astc.imageSubresource.layerCount = 1;
  astc.imageExtent = {20, 13, 1};
  EXPECT_EQ(BufferImageCopyBytes(VK_FORMAT_ASTC_8x6_SRGB_BLOCK, 1, &astc),
            3U * 3U * 16U);
  EXPECT_EQ(BufferImageCopyBytes(VK_FORMAT_D24_UNORM_S8_UINT, 1, &volume),
            0U);
  EXPECT_EQ(BufferImageCopyBytes(VK_FORMAT_R8G8B8A8_UNORM, 1,
                                 static_cast<const VkBufferImageCopy*>(
                                     nullptr)),
            0U);
}

TEST_F(VulkanCallStatsTest, ClassifiesImageFormats) {
  const std::vector<std::pair<VkFormat, VkImageFormatClass>> cases = {
      {VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VkImageFormatClass::kEtc2Eac},
      {VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, VkImageFormatClass::kEtc2Eac},
      {VK_FORMAT_EAC_R11_UNORM_BLOCK, VkImageFormatClass::kEtc2Eac},
      {VK_FORMAT_EAC_R11G11_SNORM_BLOCK, VkImageFormatClass::kEtc2Eac},
      {VK_FORMAT_BC1_RGB_UNORM_BLOCK, VkImageFormatClass::kBc1To3},
      {VK_FORMAT_BC1_RGBA_SRGB_BLOCK, VkImageFormatClass::kBc1To3},
      {VK_FORMAT_BC3_SRGB_BLOCK, VkImageFormatClass::kBc1To3},
      {VK_FORMAT_BC4_UNORM_BLOCK, VkImageFormatClass::kBc4To6},
      {VK_FORMAT_BC5_SNORM_BLOCK, VkImageFormatClass::kBc4To6},
      {VK_FORMAT_BC6H_SFLOAT_BLOCK, VkImageFormatClass::kBc4To6},
      {VK_FORMAT_BC7_UNORM_BLOCK, VkImageFormatClass::kBc7},
      {VK_FORMAT_BC7_SRGB_BLOCK, VkImageFormatClass::kBc7},
      {VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VkImageFormatClass::kAstc},
      {VK_FORMAT_ASTC_12x12_SRGB_BLOCK, VkImageFormatClass::kAstc},
      {VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK, VkImageFormatClass::kAstc},
      {VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK, VkImageFormatClass::kAstc},
      {VK_FORMAT_R8G8B8A8_UNORM, VkImageFormatClass::kRgba8},
      {VK_FORMAT_R8G8B8A8_SRGB, VkImageFormatClass::kRgba8},
      {VK_FORMAT_B8G8R8A8_UNORM, VkImageFormatClass::kRgba8},
      {VK_FORMAT_B8G8R8A8_SRGB, VkImageFormatClass::kRgba8},
      {VK_FORMAT_UNDEFINED, VkImageFormatClass::kOther},
      {VK_FORMAT_R16_UNORM, VkImageFormatClass::kOther},
      {VK_FORMAT_A8B8G8R8_UNORM_PACK32, VkImageFormatClass::kOther},
      {VK_FORMAT_D32_SFLOAT, VkImageFormatClass::kOther},
      {VK_FORMAT_R8G8B8_UNORM, VkImageFormatClass::kOther},
  };
  for (const auto& [format, expected] : cases) {
    EXPECT_EQ(ClassifyVkImageFormat(format), expected) << format;
  }
}

TEST_F(VulkanCallStatsTest, WindowsCloseEveryTwoSeconds) {
  constexpr std::uint64_t kSecond = 1000000000ULL;
  VkCallStatsWindow window;
  VkCallStatsCount(VkCallCounter::kDraw, 5);  // before the first present
  EXPECT_FALSE(AdvanceVkCallStatsFrame(10 * kSecond, &window));
  VkCallStatsCount(VkCallCounter::kDraw, 7);
  EXPECT_FALSE(AdvanceVkCallStatsFrame(11 * kSecond, &window));
  EXPECT_FALSE(AdvanceVkCallStatsFrame(12 * kSecond - 1, &window));
  ASSERT_TRUE(AdvanceVkCallStatsFrame(12 * kSecond, &window));
  EXPECT_EQ(window.frames, 3U);
  EXPECT_EQ(window.span_ns, 2 * kSecond);
  EXPECT_EQ(window.calls[static_cast<std::size_t>(VkCallCounter::kDraw)], 12U);

  // The next window starts at the present that closed the last one.
  VkCallStatsCount(VkCallCounter::kDraw, 1);
  EXPECT_FALSE(AdvanceVkCallStatsFrame(13 * kSecond, &window));
  ASSERT_TRUE(AdvanceVkCallStatsFrame(15 * kSecond, &window));
  EXPECT_EQ(window.frames, 2U);
  EXPECT_EQ(window.span_ns, 3 * kSecond);
  EXPECT_EQ(window.calls[static_cast<std::size_t>(VkCallCounter::kDraw)], 1U);
  // A clock that goes backwards never closes a window.
  EXPECT_FALSE(AdvanceVkCallStatsFrame(1 * kSecond, &window));
}

TEST_F(VulkanCallStatsTest, OnPresentDoesNothingWhileDisabled) {
  ForceVkCallStatsEnabledForTesting(false);
  VkCallStatsOnPresent();
  VkCallStatsOnPresent();
  // The frame window was not started by the disabled presents.
  VkCallStatsWindow window;
  EXPECT_FALSE(AdvanceVkCallStatsFrame(0, &window));
  ASSERT_TRUE(AdvanceVkCallStatsFrame(3000000000ULL, &window));
  EXPECT_EQ(window.frames, 1U);
}

TEST_F(VulkanCallStatsTest, FormatsTheDocumentedLine) {
  VkCallStatsWindow window;
  window.frames = 120;
  window.span_ns = 2004000000ULL;
  const auto set = [&window](VkCallCounter counter, std::uint64_t calls,
                             std::uint64_t bytes = 0) {
    window.calls[static_cast<std::size_t>(counter)] = calls;
    window.bytes[static_cast<std::size_t>(counter)] = bytes;
  };
  set(VkCallCounter::kQueueSubmit, 360);
  set(VkCallCounter::kDraw, 96000);
  set(VkCallCounter::kDrawIndexed, 60060);
  set(VkCallCounter::kDrawIndirect, 0);
  set(VkCallCounter::kDispatch, 12);
  set(VkCallCounter::kBindPipeline, 12000);
  set(VkCallCounter::kBindDescriptorSets, 36000);
  set(VkCallCounter::kPushConstants, 2400);
  set(VkCallCounter::kBindVertexBuffers, 24000);
  set(VkCallCounter::kBindIndexBuffer, 6000);
  set(VkCallCounter::kBeginRenderPass, 480);
  set(VkCallCounter::kPipelineBarrier, 1440);
  set(VkCallCounter::kCopyBuffer, 240, 12582912);
  set(VkCallCounter::kCopyBufferToImage, 600, 31457280);
  set(VkCallCounter::kUpdateBuffer, 12, 6144);
  set(VkCallCounter::kClear, 120);
  set(VkCallCounter::kUpdateDescriptorSets, 3000);
  set(VkCallCounter::kCopyImage, 4);
  set(VkCallCounter::kBlitImage, 2);
  set(VkCallCounter::kCopyImageToBuffer, 1);
  set(VkCallCounter::kResolveImage, 0);
  set(VkCallCounter::kAllocateDescriptorSets, 250);
  set(VkCallCounter::kResetDescriptorPool, 3);
  set(VkCallCounter::kFlushMappedMemory, 10, 5242880);
  set(VkCallCounter::kInvalidateMappedMemory, 0);
  set(VkCallCounter::kMapMemory, 6);
  set(VkCallCounter::kAllocateMemory, 9);
  set(VkCallCounter::kFreeMemory, 8);
  set(VkCallCounter::kCreateBuffer, 40);
  set(VkCallCounter::kDestroyBuffer, 38);
  set(VkCallCounter::kCreateImage, 31);
  set(VkCallCounter::kDestroyImage, 12);
  set(VkCallCounter::kAliasView, 0);
  set(VkCallCounter::kCreateGraphicsPipelines, 5);
  set(VkCallCounter::kCreateComputePipelines, 1);
  set(VkCallCounter::kCreateShaderModule, 6);
  set(VkCallCounter::kCreateSampler, 2);
  set(VkCallCounter::kCreateFramebuffer, 3);
  set(VkCallCounter::kAllocateCommandBuffers, 4);
  set(VkCallCounter::kBeginCommandBuffer, 480);
  set(VkCallCounter::kGetFenceStatus, 900);
  window.created_images = {20, 0, 0, 0, 0, 10, 1};
  window.blocked_ns = {12300000, 0, 4560000, 30100000, 801000000, 300000};
  EXPECT_EQ(
      FormatVkCallStatsLine(window),
      "[perf] vk_calls frames=120 span_ms=2004 per_frame: submit=3.0 "
      "draw=800.0 drawIdx=500.5 drawInd=0.0 dispatch=0.1 bindPipe=100.0 "
      "bindDS=300.0 push=20.0 bindVB=200.0 bindIB=50.0 pass=4.0 "
      "barrier=12.0 copyBuf=2.0 (0.10 MiB) copyB2I=5.0 (0.25 MiB) "
      "updBuf=0.1 (0.00 MiB) clear=1.0 updDS=25.0 | window: copyImg=4 "
      "blit=2 img2buf=1 resolve=0 allocDS=250 resetDP=3 flush=10 (5.0 MiB) "
      "invalidate=0 (0.0 MiB) map=6 allocMem=9 freeMem=8 createBuf=40 "
      "destroyBuf=38 createImg=31 (etc2 20 bc1-3 0 bc4-6 0 bc7 0 astc 0 "
      "rgba8 10 other 1) destroyImg=12 aliasView=0 createPipe=5 "
      "computePipe=1 shader=6 sampler=2 framebuffer=3 allocCB=4 "
      "beginCB=480 fenceStatus=900 | blocked fence=12.3 sem=0.0 "
      "acquire=4.6 present=30.1 submit=801.0 idle=0.3 ms");
}

TEST_F(VulkanCallStatsTest, FormatsAnEmptyWindowWithoutDividingByZero) {
  const std::string line = FormatVkCallStatsLine(VkCallStatsWindow{});
  EXPECT_EQ(line.rfind("[perf] vk_calls frames=0 span_ms=0 per_frame: "
                       "submit=0.0 draw=0.0",
                       0),
            0U);
  EXPECT_NE(line.find("| blocked fence=0.0"), std::string::npos);
}

TEST(VulkanCallStatsFlagTest, FollowsTheTraceFlagRule) {
  EXPECT_FALSE(VkCallStatsFlagEnabled(nullptr));
  EXPECT_FALSE(VkCallStatsFlagEnabled(""));
  EXPECT_FALSE(VkCallStatsFlagEnabled("0"));
  EXPECT_TRUE(VkCallStatsFlagEnabled("1"));
  EXPECT_TRUE(VkCallStatsFlagEnabled("yes"));
  EXPECT_TRUE(VkCallStatsFlagEnabled("00"));
}

}  // namespace
}  // namespace mocktail::graphics
