#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "support/etc2_emulation_harness.h"
#include "support/etc2_emulation_scenarios.h"
#include "support/etc2_gpu_backend.h"
#include "support/vulkan_test_device.h"

// The production ETC2/EAC emulation on a real device (DESIGN §9.3, ctest
// label "gpu"): every scenario that a GPU can run goes through
// VulkanEtc2Emulation on a VkDevice from the system loader, the host images
// are read back and must equal both the reference model and the golden
// hashes the fake device produces. The runner's default options pin the
// CPU route (gpu = false); the GPU route gets its own run when it exists.
// The tests skip without a loader, driver or device, and while a live
// Mocktail or Sober session runs (CR-25).

namespace mocktail::graphics::testing {
namespace {

class GpuDeviceTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    HeadlessVulkanDevice::Options options;
    options.queue_count = 2;
    options.compute_only_family = true;
    options.transfer_only_family = true;
    device_ = HeadlessVulkanDevice::Create(&skip_reason_, options).release();
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
    device_->TakeValidationMessages();
  }
  // With MOCKTAIL_TEST_VALIDATION, every validation message fails the test
  // that caused it.
  void TearDown() override {
    if (device_ == nullptr) {
      return;
    }
    for (const std::string& message : device_->TakeValidationMessages()) {
      ADD_FAILURE() << "validation: " << message;
    }
  }

  static HeadlessVulkanDevice* device_;
  static std::string skip_reason_;
};

HeadlessVulkanDevice* GpuDeviceTest::device_ = nullptr;
std::string GpuDeviceTest::skip_reason_;

TEST_F(GpuDeviceTest, HasAQueueThatCopies) {
  EXPECT_NE(device_->device(), VK_NULL_HANDLE);
  EXPECT_NE(device_->queue(), VK_NULL_HANDLE);
  EXPECT_NE(device_->get_device_proc_addr(), nullptr);
  EXPECT_GT(device_->memory().memoryTypeCount, 0U);
  ASSERT_LT(device_->queue_family(), device_->queue_families().size());
  EXPECT_NE(device_->queue_families()[device_->queue_family()].queueFlags &
                (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                 VK_QUEUE_TRANSFER_BIT),
            0U);
  // Host-visible coherent memory for upload buffers and device-local
  // memory for images.
  EXPECT_NE(
      device_->FindMemoryType(~0U, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      UINT32_MAX);
  EXPECT_NE(device_->FindMemoryType(~0U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            UINT32_MAX);
  EXPECT_TRUE(device_->OptimalTilingSupports(
      VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_FEATURE_TRANSFER_DST_BIT));
}

std::vector<std::string> GpuScenarioNames() {
  std::vector<std::string> names;
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    if (scenario.fake_only.empty()) {
      names.push_back(scenario.name);
    }
  }
  return names;
}

class Etc2EmulationGpuTest : public GpuDeviceTest,
                             public ::testing::WithParamInterface<std::string> {
};

TEST_P(Etc2EmulationGpuTest, CpuRouteMatchesModelAndGolden) {
  const Scenario* scenario = FindEtc2EmulationScenario(GetParam());
  ASSERT_NE(scenario, nullptr);
  Etc2GpuBackend backend(device_);
  const ScenarioRun run = RunEtc2Scenario(*scenario, &backend);
  if (run.skipped) {
    GTEST_SKIP() << run.skip_reason;
  }
  for (const std::string& failure : run.failures) {
    ADD_FAILURE() << failure;
  }
  std::uint64_t golden = 0;
  ASSERT_TRUE(FindEtc2EmulationGolden(scenario->name, &golden))
      << GoldenLine(scenario->name, run.hash);
  EXPECT_EQ(run.hash, golden)
      << "the GPU's host images differ from the fake device's; this run "
         "gives "
      << GoldenLine(scenario->name, run.hash);
}

// Whether the GPU route can serve one of the scenario's uploads: an ETC2
// copy into an image that is not upscaled, and not a colour one while
// texture overrides are active (both stay on the CPU route, DESIGN §3.2).
bool GpuRouteServes(const Scenario& scenario) {
  if (!scenario.emulated) {
    return false;
  }
  for (const ScenarioStep& step : scenario.steps) {
    if (step.kind != ScenarioStep::Kind::kCopyBufferToImage) {
      continue;
    }
    const ScenarioImage& image = scenario.images[step.image];
    Etc2EmulatedFormat format;
    if (!LookupEmulatedEtc2Format(image.format, &format)) {
      continue;
    }
    const bool colour = format.etc_format == EtcFormat::kEtc2Rgb8 ||
                        format.etc_format == EtcFormat::kEtc2Rgb8A1 ||
                        format.etc_format == EtcFormat::kEtc2Rgba8;
    const bool upscaled = scenario.upscale > 1 && colour &&
                          image.array_layers == 1 && image.width <= 64 &&
                          image.height <= 64;
    if (!upscaled && !(scenario.overrides && colour)) {
      return true;
    }
  }
  return false;
}

// The GPU route (DESIGN C3) on the same scenarios: the decode pass the
// emulation puts in front of each batch must leave exactly the CPU route's
// texels (the same golden hash), after the route got ready.
TEST_P(Etc2EmulationGpuTest, GpuRouteMatchesCpuRouteAndGolden) {
  const Scenario* scenario = FindEtc2EmulationScenario(GetParam());
  ASSERT_NE(scenario, nullptr);
  Etc2GpuBackend backend(device_);
  ScenarioOptions options;
  options.emulation.gpu = true;
  const ScenarioRun run = RunEtc2Scenario(*scenario, &backend, options);
  if (run.skipped) {
    GTEST_SKIP() << run.skip_reason;
  }
  for (const std::string& failure : run.failures) {
    ADD_FAILURE() << failure;
  }
  std::uint64_t golden = 0;
  ASSERT_TRUE(FindEtc2EmulationGolden(scenario->name, &golden));
  EXPECT_EQ(run.hash, golden)
      << "the GPU route's host images differ from the CPU route's; this run "
         "gives "
      << GoldenLine(scenario->name, run.hash);
  if (!scenario->emulated) {
    EXPECT_FALSE(run.gpu_route);
    return;
  }
  ASSERT_TRUE(run.gpu_route);
  EXPECT_EQ(run.gpu_state, "ready");
  if (GpuRouteServes(*scenario)) {
    EXPECT_GT(run.gpu_stats.passes, 0U) << "no upload took the GPU route";
  } else {
    EXPECT_EQ(run.gpu_stats.passes, 0U);
  }
  EXPECT_EQ(run.gpu_stats.rescues, 0U);
  EXPECT_EQ(run.gpu_stats.dropped_uploads, 0U);
}

INSTANTIATE_TEST_SUITE_P(Scenarios, Etc2EmulationGpuTest,
                         ::testing::ValuesIn(GpuScenarioNames()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });

// The device goes first while command buffers still own staging buffers;
// the real driver must accept the emulation freeing them then. Two shapes
// suffice here (the fake device runs every scenario this way).
class Etc2EmulationGpuTeardownTest : public Etc2EmulationGpuTest {};

TEST_P(Etc2EmulationGpuTeardownTest, DestroyDeviceFirstReleasesStaging) {
  const Scenario* scenario = FindEtc2EmulationScenario(GetParam());
  ASSERT_NE(scenario, nullptr);
  Etc2GpuBackend backend(device_);
  ScenarioOptions options;
  options.teardown = ScenarioTeardown::kDestroyDeviceFirst;
  const ScenarioRun run = RunEtc2Scenario(*scenario, &backend, options);
  if (run.skipped) {
    GTEST_SKIP() << run.skip_reason;
  }
  for (const std::string& failure : run.failures) {
    ADD_FAILURE() << failure;
  }
  std::uint64_t golden = 0;
  ASSERT_TRUE(FindEtc2EmulationGolden(scenario->name, &golden));
  EXPECT_EQ(run.hash, golden);
}

INSTANTIATE_TEST_SUITE_P(Scenarios, Etc2EmulationGpuTeardownTest,
                         ::testing::Values("SecondaryCommandBuffers",
                                           "MultiCommandBufferSubmits"),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });

}  // namespace
}  // namespace mocktail::graphics::testing
