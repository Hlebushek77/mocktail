#include <gtest/gtest.h>

#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "mocktail/graphics/vulkan_etc2_gpu.h"
#include "support/etc2_emulation_harness.h"
#include "support/etc2_emulation_scenarios.h"
#include "support/fake_vulkan_device.h"

// The GPU route (DESIGN C3) on the GPU-free fake device, which runs the
// decode jobs of a dispatch on the CPU the way etc2_decode.comp does. So
// every scenario goes through the route's record, submit and teardown paths
// here, without a GPU, and must leave the golden texels; and the fake counts
// every object the route makes. Also: the fallback matrix of DESIGN §6, the
// submit failures, CR-4's destructor, CR-9's chained descriptor pools and
// the MOCKTAIL_FRAME_STATS lines.

namespace mocktail::graphics::testing {
namespace {

std::vector<std::string> ScenarioNames() {
  std::vector<std::string> names;
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    names.push_back(scenario.name);
  }
  return names;
}

Etc2EmulationOptions GpuOptions(std::uint32_t upscale = 1) {
  Etc2EmulationOptions options;
  options.gpu = true;
  options.upscale = upscale;
  options.to_bc = false;  // RGBA/R16 hosts, as the goldens
  return options;
}

// As vulkan_etc2_emulation_test: every fake error must be one the scenario
// expects (the CPU route's CR-19 one on a transfer-only family).
void ExpectOnlyAllowedErrors(const FakeVulkanDevice& fake,
                             const Scenario& scenario) {
  for (const std::string& error : fake.errors()) {
    bool allowed = false;
    for (const std::string& expected : scenario.allowed_fake_errors) {
      allowed = allowed || error.find(expected) != std::string::npos;
    }
    if (!allowed) {
      ADD_FAILURE() << "fake device: " << error;
    }
  }
}

// Every object the emulation made, the GPU route's included, is gone.
void ExpectEverythingReleased(const FakeVulkanDevice& fake) {
  const FakeVulkanDevice::Counts& counts = fake.counts();
  EXPECT_EQ(counts.entry_point_buffers.live(), 0U);
  EXPECT_EQ(counts.entry_point_memory.live(), 0U);
  EXPECT_EQ(counts.buffers.live(), 0U);
  EXPECT_EQ(counts.memory.live(), 0U);
  EXPECT_EQ(counts.images.live(), 0U);
  EXPECT_EQ(counts.command_pools.live(), 0U);
  EXPECT_EQ(counts.command_buffers.live(), 0U);
  EXPECT_EQ(counts.fences.live(), 0U);
  EXPECT_EQ(counts.live_mappings, 0U);
  EXPECT_EQ(counts.shader_modules.live(), 0U);
  EXPECT_EQ(counts.pipelines.live(), 0U);
  EXPECT_EQ(counts.pipeline_layouts.live(), 0U);
  EXPECT_EQ(counts.descriptor_set_layouts.live(), 0U);
  EXPECT_EQ(counts.descriptor_pools.live(), 0U);
  EXPECT_EQ(counts.descriptor_sets.live(), 0U);
  EXPECT_EQ(counts.query_pools.live(), 0U);
}

bool ServesOnTheGpu(const Scenario& scenario) {
  if (!scenario.emulated || scenario.queue_family == 1) {
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

// How a scenario's GPU-route uploads reach the arena.
enum class Fill {
  kDecodePass,  // the decode pass in front of the batch
  // [CR-7] No slot comes free after the warm-up: the reserved rescue slot
  // copies what the CPU decoded into host staging (the arena's, or a
  // one-off buffer, its fallback when the arena cannot grow).
  kRescue,
  kRescueOneOffStaging,
};

class Etc2GpuRouteScenarioTest
    : public ::testing::TestWithParam<std::string> {
 protected:
  void TearDown() override {
    Etc2GpuTesting().fail_slot_acquire = false;
    Etc2EmulationTesting().no_host_staging = false;
  }

  void RunAndCheck(ScenarioTeardown teardown, bool frame_stats,
                   Fill fill = Fill::kDecodePass) {
    Etc2EmulationTesting().no_host_staging =
        fill == Fill::kRescueOneOffStaging;
    const Scenario* scenario = FindEtc2EmulationScenario(GetParam());
    ASSERT_NE(scenario, nullptr);
    FakeVulkanDevice fake;
    FakeEmulationBackend backend(&fake);
    ScenarioOptions options;
    options.teardown = teardown;
    options.emulation.gpu = true;
    options.emulation.frame_stats = frame_stats;
    options.fake = &fake;
    if (fill != Fill::kDecodePass) {
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
    ExpectOnlyAllowedErrors(fake, *scenario);
    ExpectEverythingReleased(fake);
    std::uint64_t golden = 0;
    ASSERT_TRUE(FindEtc2EmulationGolden(scenario->name, &golden));
    EXPECT_EQ(run.hash, golden)
        << "the GPU route left other texels than plus.1; this run gives "
        << GoldenLine(scenario->name, run.hash);
    if (!scenario->emulated) {
      EXPECT_FALSE(run.gpu_route);
      return;
    }
    ASSERT_TRUE(run.gpu_route);
    if (ServesOnTheGpu(*scenario)) {
      EXPECT_EQ(run.gpu_state, "ready") << log;
      EXPECT_GT(run.gpu_stats.passes, 0U);
      EXPECT_GT(fake.counts().decoded_blocks, 0U);
      if (fill == Fill::kDecodePass) {
        EXPECT_EQ(run.gpu_stats.rescues, 0U);
      } else {
        EXPECT_EQ(run.gpu_stats.rescues, run.gpu_stats.passes);
        EXPECT_GT(run.gpu_stats.rescued_uploads, 0U);
      }
    }
    EXPECT_EQ(run.gpu_stats.dropped_uploads, 0U);
    EXPECT_EQ(log.find("GPU decode unavailable"), std::string::npos) << log;
    EXPECT_EQ(log.find("dropped GPU-route uploads"), std::string::npos) << log;
  }
};

TEST_P(Etc2GpuRouteScenarioTest, MatchesTheGoldenHashes) {
  RunAndCheck(ScenarioTeardown::kOrderly, false);
}

TEST_P(Etc2GpuRouteScenarioTest, ResetPoolReleasesEverything) {
  RunAndCheck(ScenarioTeardown::kResetPool, false);
}

TEST_P(Etc2GpuRouteScenarioTest, DestroyPoolReleasesEverything) {
  RunAndCheck(ScenarioTeardown::kDestroyPool, false);
}

// The device first, while command buffers still hold arena ranges; with
// MOCKTAIL_FRAME_STATS the slots also have timestamp query pools.
TEST_P(Etc2GpuRouteScenarioTest, DestroyDeviceFirstReleasesEverything) {
  RunAndCheck(ScenarioTeardown::kDestroyDeviceFirst, true);
}

// The rescue leaves the decode pass's texels, and its staging goes with
// Roblox's command buffers (or with the device).
TEST_P(Etc2GpuRouteScenarioTest, RescueMatchesTheGoldenHashes) {
  RunAndCheck(ScenarioTeardown::kOrderly, false, Fill::kRescue);
}

TEST_P(Etc2GpuRouteScenarioTest, RescueWithOneOffStagingReleasesEverything) {
  RunAndCheck(ScenarioTeardown::kResetPool, false,
              Fill::kRescueOneOffStaging);
}

TEST_P(Etc2GpuRouteScenarioTest, RescueThenDestroyDeviceReleasesEverything) {
  RunAndCheck(ScenarioTeardown::kDestroyDeviceFirst, true, Fill::kRescue);
}

INSTANTIATE_TEST_SUITE_P(Scenarios, Etc2GpuRouteScenarioTest,
                         ::testing::ValuesIn(ScenarioNames()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });

// ---- direct tests --------------------------------------------------------

// Sets or clears one variable for the life of the object.
class ScopedEnvironment {
 public:
  ScopedEnvironment(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name); old != nullptr) {
      old_ = old;
    }
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnvironment() {
    if (old_.has_value()) {
      setenv(name_, old_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }
  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

 private:
  const char* name_;
  std::optional<std::string> old_;
};

struct RigImage {
  VkImage image = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  EtcFormat etc = EtcFormat::kEtc2Rgb8;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t mips = 1;
};

struct RigSource {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  std::vector<std::uint8_t> bytes;
};

// Plays Roblox and the adapter on a fake device with one queue in each of
// the families 0 (G|C|T), 1 (T) and 2 (C|T).
class FakeRig {
 public:
  explicit FakeRig(Etc2EmulationOptions options,
                   FakeVulkanDevice::Config config = FakeVulkanDevice::Config())
      : fake_(std::move(config)), emulation_(options) {
    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queues[3]{};
    for (std::uint32_t family = 0; family < 3; ++family) {
      queues[family].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queues[family].queueFamilyIndex = family;
      queues[family].queueCount = 1;
      queues[family].pQueuePriorities = &priority;
    }
    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 3;
    info.pQueueCreateInfos = queues;
    emulation_.RegisterDevice(fake_.device(), fake_.physical_device(), true,
                              fake_.config().memory, fake_.config().properties,
                              fake_.config().queue_families, info,
                              FakeVulkanDevice::GetDeviceProcAddr());
    for (std::uint32_t family = 0; family < 3; ++family) {
      emulation_.RegisterQueue(fake_.device(), fake_.GetQueue(family, 0),
                               family, 0);
    }
  }
  ~FakeRig() { Teardown(); }

  FakeVulkanDevice& fake() { return fake_; }
  VulkanEtc2Emulation& emulation() { return emulation_; }
  VkQueue queue(std::uint32_t family = 0) { return fake_.GetQueue(family, 0); }
  Etc2GpuDevice* gpu() { return Etc2GpuDevice::Find(fake_.device()); }

  bool WarmUp(std::uint32_t family = 0) {
    Etc2GpuDevice* runtime = gpu();
    if (runtime == nullptr) {
      return false;
    }
    runtime->StartBuild();
    if (!runtime->WaitForBuild(std::chrono::seconds(10))) {
      return false;
    }
    for (int round = 0;
         round < 3 && runtime->state() != Etc2GpuDevice::State::kReady &&
         runtime->state() != Etc2GpuDevice::State::kFailed;
         ++round) {
      Submit(queue(family), {}, VK_NULL_HANDLE);
    }
    return runtime->state() == Etc2GpuDevice::State::kReady;
  }

  RigImage* Image(VkFormat format, std::uint32_t width, std::uint32_t height,
                  std::uint32_t mips, std::uint32_t layers = 1) {
    auto image = std::make_unique<RigImage>();
    image->format = format;
    Etc2EmulatedFormat emulated;
    EXPECT_TRUE(LookupEmulatedEtc2Format(format, &emulated));
    image->etc = emulated.etc_format;
    image->width = width;
    image->height = height;
    image->mips = mips;
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = mips;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    EXPECT_EQ(
        emulation_.CreateImage(fake_.device(), &info, nullptr, &image->image),
        VK_SUCCESS);
    images_.push_back(std::move(image));
    return images_.back().get();
  }

  RigSource* Source(VkDeviceSize size, std::uint64_t seed) {
    auto source = std::make_unique<RigSource>();
    EXPECT_EQ(fake_.CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 &source->buffer),
              VK_SUCCESS);
    VkMemoryRequirements requirements{};
    fake_.GetBufferMemoryRequirements(source->buffer, &requirements);
    EXPECT_EQ(fake_.AllocateMemory(requirements.size, 3, &source->memory),
              VK_SUCCESS);
    emulation_.NoteAllocateMemory(fake_.device(), source->memory, 3,
                                  requirements.size);
    EXPECT_EQ(emulation_.BindBufferMemory(fake_.device(), source->buffer,
                                          source->memory, 0),
              VK_SUCCESS);
    void* data = nullptr;
    EXPECT_EQ(emulation_.MapMemory(fake_.device(), source->memory, 0,
                                   VK_WHOLE_SIZE, 0, &data),
              VK_SUCCESS);
    source->bytes = SeededBytes(seed, static_cast<std::size_t>(size));
    std::memcpy(data, source->bytes.data(), source->bytes.size());
    sources_.push_back(std::move(source));
    return sources_.back().get();
  }

  VkCommandPool Pool(std::uint32_t family) {
    const VkCommandPool pool = fake_.CreateCommandPool(
        family, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    emulation_.RegisterCommandPool(
        fake_.device(), pool, family,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    pools_.push_back({pool, {}});
    return pool;
  }

  VkCommandBuffer Begin(VkCommandPool pool) {
    const VkCommandBuffer command_buffer =
        fake_.AllocateCommandBuffer(pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    for (auto& [owner, buffers] : pools_) {
      if (owner == pool) {
        buffers.push_back(command_buffer);
      }
    }
    emulation_.BeginCommandBuffer(command_buffer, pool,
                                  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    EXPECT_EQ(fake_.BeginCommandBuffer(
                  command_buffer, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT),
              VK_SUCCESS);
    return command_buffer;
  }

  void End(VkCommandBuffer command_buffer) {
    EXPECT_EQ(fake_.EndCommandBuffer(command_buffer), VK_SUCCESS);
  }

  // Every mip of `image`, tightly packed from offset 0 of `source`.
  std::vector<VkBufferImageCopy> Upload(VkCommandBuffer command_buffer,
                                        const RigImage& image,
                                        const RigSource& source) {
    std::vector<VkBufferImageCopy> regions;
    VkDeviceSize offset = 0;
    for (std::uint32_t mip = 0; mip < image.mips; ++mip) {
      VkBufferImageCopy region{};
      region.bufferOffset = offset;
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
      region.imageExtent = {std::max(1u, image.width >> mip),
                            std::max(1u, image.height >> mip), 1};
      VkDeviceSize compressed = 0;
      VkDeviceSize decoded = 0;
      Etc2UploadByteCounts(image.etc, region.imageExtent, 1, &compressed,
                           &decoded);
      offset += compressed;
      regions.push_back(region);
    }
    emulation_.CmdCopyBufferToImage(fake_.device(), command_buffer,
                                    source.buffer, image.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    static_cast<std::uint32_t>(regions.size()),
                                    regions.data());
    return regions;
  }

  // Regions as given.
  void Upload(VkCommandBuffer command_buffer, const RigImage& image,
              const RigSource& source,
              const std::vector<VkBufferImageCopy>& regions) {
    emulation_.CmdCopyBufferToImage(fake_.device(), command_buffer,
                                    source.buffer, image.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    static_cast<std::uint32_t>(regions.size()),
                                    regions.data());
  }

  VkResult Submit(VkQueue queue,
                  const std::vector<std::vector<VkCommandBuffer>>& batches,
                  VkFence fence) {
    std::vector<VkSubmitInfo> infos;
    for (const std::vector<VkCommandBuffer>& batch : batches) {
      VkSubmitInfo info{};
      info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      info.commandBufferCount = static_cast<std::uint32_t>(batch.size());
      info.pCommandBuffers = batch.empty() ? nullptr : batch.data();
      infos.push_back(info);
    }
    const auto count = static_cast<std::uint32_t>(infos.size());
    const VkSubmitInfo* submits = infos.empty() ? nullptr : infos.data();
    Etc2SubmitPlan plan;
    emulation_.PrepareSubmit(queue, count, submits, fence, &plan);
    for (const Etc2PreSubmit& pre : plan.pre_submits()) {
      fake_.QueueSubmit(queue, 1, &pre.info, pre.fence);
    }
    const VkResult result = fake_.QueueSubmit(
        queue, plan.count(count), plan.submits(submits), plan.fence(fence));
    emulation_.FinishSubmit(&plan, result, [&](VkFence trailing) {
      return fake_.QueueSubmit(queue, 0, nullptr, trailing);
    });
    return result;
  }

  // The host image holds what the CPU decoder makes of the source.
  void Check(const RigImage& image, const RigSource& source,
             const std::vector<VkBufferImageCopy>& regions) {
    for (const VkBufferImageCopy& region : regions) {
      VkDeviceSize compressed = 0;
      VkDeviceSize decoded = 0;
      Etc2UploadByteCounts(image.etc, region.imageExtent, 1, &compressed,
                           &decoded);
      std::vector<std::uint8_t> expected(decoded);
      ASSERT_TRUE(DecodeEtcImage(image.etc,
                                 source.bytes.data() + region.bufferOffset,
                                 compressed, region.imageExtent.width,
                                 region.imageExtent.height, expected.data(),
                                 expected.size()));
      const std::vector<std::uint8_t>* actual = fake_.ImageSubresource(
          image.image, region.imageSubresource.mipLevel, 0);
      ASSERT_NE(actual, nullptr);
      EXPECT_EQ(*actual, expected)
          << "mip " << region.imageSubresource.mipLevel;
    }
  }

  void DestroyDevice() {
    emulation_.DestroyDevice(fake_.device());
    device_destroyed_ = true;
  }

  void Teardown() {
    if (torn_down_) {
      return;
    }
    torn_down_ = true;
    for (const auto& [pool, buffers] : pools_) {
      for (VkCommandBuffer command_buffer : buffers) {
        emulation_.ReleaseCommandBuffer(command_buffer);
      }
      fake_.DestroyCommandPool(pool);
      emulation_.ForgetCommandPool(fake_.device(), pool);
    }
    for (const std::unique_ptr<RigImage>& image : images_) {
      if (device_destroyed_) {
        fake_.DestroyImage(image->image);
      } else {
        emulation_.DestroyImage(fake_.device(), image->image, nullptr);
      }
    }
    for (const std::unique_ptr<RigSource>& source : sources_) {
      if (device_destroyed_) {
        fake_.UnmapMemory(source->memory);
        fake_.DestroyBuffer(source->buffer);
        fake_.FreeMemory(source->memory);
      } else {
        emulation_.UnmapMemory(fake_.device(), source->memory);
        emulation_.DestroyBuffer(fake_.device(), source->buffer, nullptr);
        emulation_.FreeMemory(fake_.device(), source->memory, nullptr);
      }
    }
    if (!device_destroyed_) {
      DestroyDevice();
    }
  }

 private:
  FakeVulkanDevice fake_;
  VulkanEtc2Emulation emulation_;
  std::vector<std::unique_ptr<RigImage>> images_;
  std::vector<std::unique_ptr<RigSource>> sources_;
  std::vector<std::pair<VkCommandPool, std::vector<VkCommandBuffer>>> pools_;
  bool torn_down_ = false;
  bool device_destroyed_ = false;
};

class Etc2GpuRouteFakeTest : public ::testing::Test {
 protected:
  void TearDown() override {
    Etc2EmulationTesting().no_host_staging = false;
    Etc2GpuTestHooks& hooks = Etc2GpuTesting();
    hooks.fail_slot_acquire = false;
    hooks.fail_rescue_slot = false;
    hooks.corrupt_selftest = false;
    hooks.build_delay_ms = 0;
    std::lock_guard<std::mutex> lock(hooks.mutex);
    hooks.rescue_decode = nullptr;
  }
};

// The value of `key` (up to the next space) in the last line starting with
// `prefix`.
std::string Field(const std::string& log, const std::string& prefix,
                  const std::string& key) {
  std::istringstream stream(log);
  std::string found;
  for (std::string line; std::getline(stream, line);) {
    if (line.compare(0, prefix.size(), prefix) == 0) {
      found = line;
    }
  }
  const std::size_t at = found.find(key);
  if (at == std::string::npos) {
    return "<no " + key + " in \"" + found + "\">";
  }
  const std::size_t begin = at + key.size();
  return found.substr(begin, found.find(' ', begin) - begin);
}

constexpr const char* kTotals = "  [vulkan] ETC2 decode totals: ";

// One upload with `options` (MOCKTAIL_FRAME_STATS on, so the record-time
// reasons are counted) on `family`; returns the log. The texels must be
// right whatever route served them (an upscaled image holds resampled
// texels, which the scenario tests check).
std::string UploadOnce(Etc2EmulationOptions options,
                       FakeVulkanDevice::Config config, VkFormat format,
                       std::uint32_t size, std::uint32_t family = 0,
                       bool warm_up = true) {
  const bool upscaled = options.upscale > 1;
  options.frame_stats = true;
  ::testing::internal::CaptureStderr();
  {
    FakeRig rig(options, std::move(config));
    if (warm_up) {
      rig.WarmUp();
    }
    RigImage* image = rig.Image(format, size, size, 1);
    RigSource* source = rig.Source(size * size, 5);
    const VkCommandBuffer command_buffer = rig.Begin(rig.Pool(family));
    const std::vector<VkBufferImageCopy> regions =
        rig.Upload(command_buffer, *image, *source);
    rig.End(command_buffer);
    EXPECT_EQ(rig.Submit(rig.queue(family), {{command_buffer}}, VK_NULL_HANDLE),
              VK_SUCCESS);
    if (!upscaled) {
      rig.Check(*image, *source, regions);
    }
  }
  return ::testing::internal::GetCapturedStderr();
}

std::string Reasons(const std::string& log) {
  std::string out;
  for (const char* reason :
       {"gpu_disabled", "gpu_not_ready", "gpu_unavailable", "upscaled",
        "overrides", "no_compute_queue", "too_large", "arena_full"}) {
    out += std::string(reason) + "=" +
           Field(log, kTotals, std::string(" ") + reason + "=") + " ";
  }
  return out + "gpu=" + Field(log, kTotals, "| gpu uploads=");
}

// DESIGN §6: why an upload takes the CPU route, one reason at a time.
TEST_F(Etc2GpuRouteFakeTest, FallbackMatrix) {
  constexpr VkFormat kRgb = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
  constexpr VkFormat kR11 = VK_FORMAT_EAC_R11_UNORM_BLOCK;
  const FakeVulkanDevice::Config plain;
  EXPECT_EQ(Reasons(UploadOnce(GpuOptions(), plain, kRgb, 64)),
            "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=0 upscaled=0 "
            "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=1");
  Etc2EmulationOptions off = GpuOptions();
  off.gpu = false;
  EXPECT_EQ(Reasons(UploadOnce(off, plain, kRgb, 64)),
            "gpu_disabled=1 gpu_not_ready=0 gpu_unavailable=0 upscaled=0 "
            "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=0");
  // Built (the fake builds at once) but the selftest has not run.
  EXPECT_EQ(Reasons(UploadOnce(GpuOptions(), plain, kRgb, 64, 0, false)),
            "gpu_disabled=0 gpu_not_ready=1 gpu_unavailable=0 upscaled=0 "
            "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=0");
  // A host without what the route needs: no runtime at all.
  FakeVulkanDevice::Config old;
  old.missing_procs = {"vkCreateComputePipelines"};
  EXPECT_EQ(Reasons(UploadOnce(GpuOptions(), old, kRgb, 64)),
            "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=1 upscaled=0 "
            "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=0");
  // A selftest mismatch.
  Etc2GpuTesting().corrupt_selftest = true;
  const std::string failed = UploadOnce(GpuOptions(), plain, kRgb, 64);
  Etc2GpuTesting().corrupt_selftest = false;
  EXPECT_EQ(Reasons(failed),
            "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=1 upscaled=0 "
            "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=0");
  EXPECT_NE(failed.find("[vulkan] ETC2 GPU decode unavailable: selftest "
                        "mismatch: ETC2_RGBA8 block "),
            std::string::npos)
      << failed;
  EXPECT_EQ(Reasons(UploadOnce(GpuOptions(4), plain, kRgb, 32)),
            "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=0 upscaled=1 "
            "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=0");
  {
    const ScopedEnvironment dumps("MOCKTAIL_TEXTURE_DUMP_DIR", "/nonexistent");
    EXPECT_EQ(Reasons(UploadOnce(GpuOptions(), plain, kRgb, 64)),
              "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=0 upscaled=0 "
              "overrides=1 no_compute_queue=0 too_large=0 arena_full=0 gpu=0");
    // Overrides only replace colour textures.
    EXPECT_EQ(Reasons(UploadOnce(GpuOptions(), plain, kR11, 64)),
              "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=0 upscaled=0 "
              "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=1");
  }
  {
    // The runtime always names an override directory; without PNGs in it
    // the GPU route serves colour textures too.
    const ScopedEnvironment empty("MOCKTAIL_TEXTURE_OVERRIDE_DIR", "/");
    EXPECT_EQ(Reasons(UploadOnce(GpuOptions(), plain, kRgb, 64)),
              "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=0 upscaled=0 "
              "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=1");
  }
  EXPECT_EQ(Reasons(UploadOnce(GpuOptions(), plain, kRgb, 64, 1)),
            "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=0 upscaled=0 "
            "overrides=0 no_compute_queue=1 too_large=0 arena_full=0 gpu=0");
  // 256 KiB of RGBA8 against a 128 KiB storage range (the selftest's 91
  // KiB still fits).
  FakeVulkanDevice::Config small_range;
  small_range.properties.limits.maxStorageBufferRange = 128 * 1024;
  EXPECT_EQ(Reasons(UploadOnce(GpuOptions(), small_range, kRgb, 256)),
            "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=0 upscaled=0 "
            "overrides=0 no_compute_queue=0 too_large=1 arena_full=0 gpu=0");
  FakeVulkanDevice::Config small_memory;
  small_memory.max_allocation_bytes = 1024 * 1024;
  EXPECT_EQ(Reasons(UploadOnce(GpuOptions(), small_memory, kRgb, 64)),
            "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=0 upscaled=0 "
            "overrides=0 no_compute_queue=0 too_large=0 arena_full=1 gpu=0");
  // A driver that cannot build the pipeline.
  FakeVulkanDevice::Config no_pipeline;
  no_pipeline.fail_compute_pipelines = true;
  const std::string unbuilt = UploadOnce(GpuOptions(), no_pipeline, kRgb, 64);
  EXPECT_EQ(Reasons(unbuilt),
            "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=1 upscaled=0 "
            "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=0");
  EXPECT_NE(unbuilt.find("[vulkan] ETC2 GPU decode unavailable: building the "
                         "decode pipeline returned"),
            std::string::npos)
      << unbuilt;
  // Memory the selftest needs cannot be had (its 91 KiB of output); the
  // CPU route's one-off staging still fits.
  FakeVulkanDevice::Config tiny_memory;
  tiny_memory.max_allocation_bytes = 4096;
  const std::string untested = UploadOnce(GpuOptions(), tiny_memory, kRgb, 16);
  EXPECT_EQ(Reasons(untested),
            "gpu_disabled=0 gpu_not_ready=0 gpu_unavailable=1 upscaled=0 "
            "overrides=0 no_compute_queue=0 too_large=0 arena_full=0 gpu=0");
  EXPECT_NE(untested.find("[vulkan] ETC2 GPU decode unavailable: the selftest "
                          "could not be set up"),
            std::string::npos)
      << untested;
}

// A GPU that falls behind (DESIGN C3.3, CR-6, §6): with every fence held,
// 32 calls fill the queue's 32 slots, and the 33rd waits for the oldest
// (50 ms at most; the fake answers at once), gives up and is rescued on
// the reserved slot with the right texels. Once the GPU catches up the
// slots are used again without a wait. vkDestroyDevice with a pass still
// pending waits 1 s at most for it, logs, and destroys everything anyway.
TEST_F(Etc2GpuRouteFakeTest, AGpuThatFallsBehindIsRescuedThenTornDown) {
  ::testing::internal::CaptureStderr();
  {
    FakeRig rig(GpuOptions());
    ASSERT_TRUE(rig.WarmUp());
    const VkCommandPool pool = rig.Pool(0);
    RigImage* image =
        rig.Image(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 24, 24, 2);
    RigSource* source = rig.Source(4096, 43);
    const std::uint64_t slots = rig.gpu()->stats().slots_created;
    rig.fake().HoldFences(true);
    for (int call = 0; call < 33; ++call) {
      const VkCommandBuffer command_buffer = rig.Begin(pool);
      const auto regions = rig.Upload(command_buffer, *image, *source);
      rig.End(command_buffer);
      ASSERT_EQ(rig.Submit(rig.queue(), {{command_buffer}}, VK_NULL_HANDLE),
                VK_SUCCESS);
      rig.Check(*image, *source, regions);
    }
    Etc2GpuDevice::Stats stats = rig.gpu()->stats();
    EXPECT_EQ(stats.slots_created, slots + 33) << "32 slots and the rescue's";
    EXPECT_EQ(stats.slot_waits, 1U);
    EXPECT_EQ(stats.slot_timeouts, 1U);
    EXPECT_EQ(stats.rescues, 1U);
    EXPECT_EQ(stats.rescued_uploads, 2U);
    EXPECT_EQ(stats.dropped_uploads, 0U);

    rig.fake().HoldFences(false);
    const VkCommandBuffer caught_up = rig.Begin(pool);
    const auto regions = rig.Upload(caught_up, *image, *source);
    rig.End(caught_up);
    ASSERT_EQ(rig.Submit(rig.queue(), {{caught_up}}, VK_NULL_HANDLE),
              VK_SUCCESS);
    rig.Check(*image, *source, regions);
    stats = rig.gpu()->stats();
    EXPECT_EQ(stats.slots_created, slots + 33);
    EXPECT_EQ(stats.slot_waits, 1U);

    rig.fake().HoldFences(true);
    const VkCommandBuffer pending = rig.Begin(pool);
    rig.Upload(pending, *image, *source);
    rig.End(pending);
    ASSERT_EQ(rig.Submit(rig.queue(), {{pending}}, VK_NULL_HANDLE),
              VK_SUCCESS);
    rig.Teardown();
    ExpectEverythingReleased(rig.fake());
    EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
  }
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("[vulkan] ETC2 GPU decode: a decode pass still pending "
                     "after 1 s"),
            std::string::npos)
      << log;
}

// [CR-1] Three batches in one call, two with uploads: two passes, one
// fence; then 40 calls in a row recycle the slot without ever waiting.
TEST_F(Etc2GpuRouteFakeTest, OneSlotAndOneFencePerCall) {
  FakeRig rig(GpuOptions());
  ASSERT_TRUE(rig.WarmUp());
  const VkCommandPool pool = rig.Pool(0);
  RigImage* first = rig.Image(VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK, 40, 24, 4);
  RigImage* second = rig.Image(VK_FORMAT_EAC_R11G11_SNORM_BLOCK, 16, 48, 5);
  RigSource* source = rig.Source(16 * 1024, 9);
  const VkCommandBuffer a = rig.Begin(pool);
  const auto first_regions = rig.Upload(a, *first, *source);
  rig.End(a);
  const VkCommandBuffer idle = rig.Begin(pool);
  rig.End(idle);
  const VkCommandBuffer c = rig.Begin(pool);
  const auto second_regions = rig.Upload(c, *second, *source);
  rig.End(c);
  const std::uint64_t fences = rig.fake().counts().fences.created;
  ASSERT_EQ(rig.Submit(rig.queue(), {{a}, {idle}, {c}}, VK_NULL_HANDLE),
            VK_SUCCESS);
  EXPECT_EQ(rig.gpu()->stats().passes, 2U);
  EXPECT_EQ(rig.fake().counts().fences.created - fences, 1U)
      << "one slot with one fence for the call";
  rig.Check(*first, *source, first_regions);
  rig.Check(*second, *source, second_regions);
  for (int round = 0; round < 40; ++round) {
    const VkCommandBuffer command_buffer = rig.Begin(pool);
    rig.Upload(command_buffer, *first, *source);
    rig.End(command_buffer);
    ASSERT_EQ(rig.Submit(rig.queue(), {{command_buffer}}, VK_NULL_HANDLE),
              VK_SUCCESS);
  }
  EXPECT_EQ(rig.gpu()->stats().slots_created, 1U);
  EXPECT_EQ(rig.gpu()->stats().slot_waits, 0U);
  EXPECT_EQ(rig.fake().calls("vkWaitForFences"), 0U);
  EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
}

// A failed host submit leaves the slot unsubmitted: no wait for its fence,
// its pool is reset and the next call uses it again.
TEST_F(Etc2GpuRouteFakeTest, AFailedSubmitLeavesTheSlotUnsubmitted) {
  FakeRig rig(GpuOptions());
  ASSERT_TRUE(rig.WarmUp());
  const VkCommandPool pool = rig.Pool(0);
  RigImage* image = rig.Image(VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, 32, 32, 3);
  RigSource* source = rig.Source(4096, 13);
  rig.fake().FailNextSubmits({VK_ERROR_OUT_OF_DEVICE_MEMORY});
  const VkCommandBuffer lost = rig.Begin(pool);
  rig.Upload(lost, *image, *source);
  rig.End(lost);
  EXPECT_EQ(rig.Submit(rig.queue(), {{lost}}, VK_NULL_HANDLE),
            VK_ERROR_OUT_OF_DEVICE_MEMORY);
  const VkCommandBuffer again = rig.Begin(pool);
  const auto regions = rig.Upload(again, *image, *source);
  rig.End(again);
  EXPECT_EQ(rig.Submit(rig.queue(), {{again}}, VK_NULL_HANDLE), VK_SUCCESS);
  rig.Check(*image, *source, regions);
  EXPECT_EQ(rig.gpu()->stats().slots_created, 1U);
  EXPECT_EQ(rig.fake().calls("vkWaitForFences"), 0U);
  EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
}

// A protected batch's decode pass goes first on its own; when that
// vkQueueSubmit fails and Roblox's succeeds, the pass's uploads are
// counted as dropped and logged, and the slot's timestamps, which no
// submitted command wrote, are not waited for when the slot comes back
// (MOCKTAIL_FRAME_STATS): the fake device reports such a wait, which the
// spec lets hang or lose the device.
TEST_F(Etc2GpuRouteFakeTest, AFailedPreSubmitDropsItsUploads) {
  Etc2EmulationOptions options = GpuOptions();
  options.frame_stats = true;
  ::testing::internal::CaptureStderr();
  {
    FakeRig rig(options);
    ASSERT_TRUE(rig.WarmUp());
    const VkCommandPool pool = rig.Pool(0);
    RigImage* image = rig.Image(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 32, 32, 3);
    RigSource* source = rig.Source(4096, 31);
    const VkCommandBuffer command_buffer = rig.Begin(pool);
    rig.Upload(command_buffer, *image, *source);
    rig.End(command_buffer);
    VkProtectedSubmitInfo protected_info{};
    protected_info.sType = VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO;
    protected_info.protectedSubmit = VK_TRUE;
    VkSubmitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.pNext = &protected_info;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &command_buffer;
    Etc2SubmitPlan plan;
    rig.emulation().PrepareSubmit(rig.queue(), 1, &info, VK_NULL_HANDLE,
                                  &plan);
    ASSERT_EQ(plan.pre_submits().size(), 1U);
    EXPECT_EQ(plan.pre_submits()[0].uploads, 3U);
    // What the adapter does: each pre-submit's result goes to the plan.
    rig.fake().FailNextSubmits({VK_ERROR_OUT_OF_DEVICE_MEMORY});
    for (const Etc2PreSubmit& pre : plan.pre_submits()) {
      plan.NotePreSubmitResult(
          pre, rig.fake().QueueSubmit(rig.queue(), 1, &pre.info, pre.fence));
    }
    EXPECT_EQ(plan.failed_pre_submit_uploads, 3U);
    const VkResult result =
        rig.fake().QueueSubmit(rig.queue(), plan.count(1), plan.submits(&info),
                               plan.fence(VK_NULL_HANDLE));
    EXPECT_EQ(result, VK_SUCCESS);
    rig.emulation().FinishSubmit(&plan, result, [&](VkFence fence) {
      return rig.fake().QueueSubmit(rig.queue(), 0, nullptr, fence);
    });
    EXPECT_EQ(rig.gpu()->stats().pre_submits, 1U);
    EXPECT_EQ(rig.gpu()->stats().dropped_uploads, 3U);

    // The next call takes the slot back and reads what timing it has.
    const VkCommandBuffer next = rig.Begin(pool);
    const auto regions = rig.Upload(next, *image, *source);
    rig.End(next);
    ASSERT_EQ(rig.Submit(rig.queue(), {{next}}, VK_NULL_HANDLE), VK_SUCCESS);
    rig.Check(*image, *source, regions);
    EXPECT_EQ(rig.gpu()->stats().slots_created, 1U);
    EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
  }
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("a decode pass submitted on its own failed; its uploads "
                     "read stale texels (VkResult -2)"),
            std::string::npos)
      << log;
  EXPECT_NE(log.find("rescued=0 dropped=3"), std::string::npos) << log;
}

// Roblox's fence stays with its batches; the slot fence follows in a
// fence-only submit. When that one fails the slot is given up, never
// reset while it may be pending, and a new slot serves the next call.
TEST_F(Etc2GpuRouteFakeTest, TheSlotFenceFollowsAnAppFence) {
  FakeRig rig(GpuOptions());
  ASSERT_TRUE(rig.WarmUp());
  const VkCommandPool pool = rig.Pool(0);
  RigImage* image = rig.Image(VK_FORMAT_EAC_R11_SNORM_BLOCK, 20, 12, 2);
  RigSource* source = rig.Source(4096, 17);
  const auto fence_handle = [&]() {
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    auto create = reinterpret_cast<PFN_vkCreateFence>(
        FakeVulkanDevice::GetDeviceProcAddr()(rig.fake().device(),
                                              "vkCreateFence"));
    create(rig.fake().device(), &info, nullptr, &fence);
    return fence;
  };
  const VkFence app = fence_handle();
  const VkCommandBuffer command_buffer = rig.Begin(pool);
  const auto regions = rig.Upload(command_buffer, *image, *source);
  rig.End(command_buffer);
  const std::uint64_t submits = rig.fake().counts().submits;
  ASSERT_EQ(rig.Submit(rig.queue(), {{command_buffer}}, app), VK_SUCCESS);
  rig.Check(*image, *source, regions);
  EXPECT_EQ(rig.fake().counts().submits - submits, 2U)
      << "Roblox's, then the fence-only one";

  // The fence-only submit fails: the slot is lost, a second one appears.
  rig.fake().FailNextSubmits({VK_SUCCESS, VK_ERROR_DEVICE_LOST});
  const VkFence second = fence_handle();
  const VkCommandBuffer next = rig.Begin(pool);
  rig.Upload(next, *image, *source);
  rig.End(next);
  ASSERT_EQ(rig.Submit(rig.queue(), {{next}}, second), VK_SUCCESS);
  const VkCommandBuffer last = rig.Begin(pool);
  rig.Upload(last, *image, *source);
  rig.End(last);
  ASSERT_EQ(rig.Submit(rig.queue(), {{last}}, VK_NULL_HANDLE), VK_SUCCESS);
  EXPECT_EQ(rig.gpu()->stats().slots_created, 2U);
  EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
  auto destroy = reinterpret_cast<PFN_vkDestroyFence>(
      FakeVulkanDevice::GetDeviceProcAddr()(rig.fake().device(),
                                            "vkDestroyFence"));
  rig.Teardown();
  destroy(rig.fake().device(), app, nullptr);
  destroy(rig.fake().device(), second, nullptr);
}

// [CR-7] No slot comes free: the queue's reserved rescue slot copies what
// the CPU decoded into host staging into the arena, and the rewritten copy
// reads the right texels. A batch on a queue of another family than its
// pool is rescued the same way. Only when the rescue slot is unavailable
// too are the uploads dropped: counted, logged, and the call goes through.
TEST_F(Etc2GpuRouteFakeTest, ARescueDecodesOnTheCpuIntoTheArena) {
  Etc2EmulationOptions options = GpuOptions();
  options.frame_stats = true;  // the totals line then names the rescues
  ::testing::internal::CaptureStderr();
  {
    FakeRig rig(options);
    ASSERT_TRUE(rig.WarmUp());
    const VkCommandPool pool = rig.Pool(0);
    RigImage* image = rig.Image(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 37, 21, 3);
    RigSource* source = rig.Source(4096, 19);
    Etc2GpuTesting().fail_slot_acquire = true;
    const VkCommandBuffer command_buffer = rig.Begin(pool);
    const auto regions = rig.Upload(command_buffer, *image, *source);
    rig.End(command_buffer);
    const std::uint64_t decoded_blocks = rig.fake().counts().decoded_blocks;
    const std::uint64_t slots = rig.gpu()->stats().slots_created;
    EXPECT_EQ(rig.Submit(rig.queue(), {{command_buffer}}, VK_NULL_HANDLE),
              VK_SUCCESS);
    rig.Check(*image, *source, regions);
    EXPECT_EQ(rig.fake().counts().decoded_blocks, decoded_blocks)
        << "no dispatch: the CPU decoded";
    Etc2GpuDevice::Stats stats = rig.gpu()->stats();
    EXPECT_EQ(stats.rescues, 1U);
    EXPECT_EQ(stats.rescued_uploads, 3U);
    EXPECT_EQ(stats.dropped_uploads, 0U);
    EXPECT_EQ(stats.slots_created, slots + 1)
        << "the queue's reserved rescue slot, and no other";

    // A batch on a queue of another family than its pool.
    Etc2GpuTesting().fail_slot_acquire = false;
    const VkCommandBuffer elsewhere = rig.Begin(pool);
    const auto elsewhere_regions = rig.Upload(elsewhere, *image, *source);
    rig.End(elsewhere);
    EXPECT_EQ(rig.Submit(rig.queue(2), {{elsewhere}}, VK_NULL_HANDLE),
              VK_SUCCESS);
    rig.Check(*image, *source, elsewhere_regions);
    stats = rig.gpu()->stats();
    EXPECT_EQ(stats.rescues, 2U);
    EXPECT_EQ(stats.rescued_uploads, 6U);

    // Neither a slot nor the rescue slot: dropped.
    Etc2GpuTesting().fail_slot_acquire = true;
    Etc2GpuTesting().fail_rescue_slot = true;
    const VkCommandBuffer lost = rig.Begin(pool);
    rig.Upload(lost, *image, *source);
    rig.End(lost);
    EXPECT_EQ(rig.Submit(rig.queue(), {{lost}}, VK_NULL_HANDLE), VK_SUCCESS);
    stats = rig.gpu()->stats();
    EXPECT_EQ(stats.rescues, 2U);
    EXPECT_EQ(stats.dropped_uploads, 3U);
    Etc2GpuTesting().fail_slot_acquire = false;
    Etc2GpuTesting().fail_rescue_slot = false;
    EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
    rig.Teardown();
    ExpectEverythingReleased(rig.fake());
  }
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("dropped GPU-route uploads"), std::string::npos) << log;
  EXPECT_NE(log.find("another queue family"), std::string::npos) << log;
  EXPECT_NE(log.find("rescued=6 dropped=3"), std::string::npos) << log;
}

// [CR-7] While the reserved rescue slot is still pending (the GPU is far
// behind), the next rescue gets another rescue-only slot instead of a wait
// without a bound inside Roblox's vkQueueSubmit; once the GPU has caught
// up, the reserved one serves again.
TEST_F(Etc2GpuRouteFakeTest, ARescueNeverWaitsForAPendingRescueSlot) {
  FakeRig rig(GpuOptions());
  ASSERT_TRUE(rig.WarmUp());
  const VkCommandPool pool = rig.Pool(0);
  RigImage* image = rig.Image(VK_FORMAT_EAC_R11_UNORM_BLOCK, 20, 20, 2);
  RigSource* source = rig.Source(4096, 47);
  const std::uint64_t slots = rig.gpu()->stats().slots_created;
  const std::uint64_t waits = rig.fake().calls("vkWaitForFences");
  Etc2GpuTesting().fail_slot_acquire = true;  // as if 32 calls were pending
  rig.fake().HoldFences(true);
  for (int call = 0; call < 3; ++call) {
    const VkCommandBuffer command_buffer = rig.Begin(pool);
    const auto regions = rig.Upload(command_buffer, *image, *source);
    rig.End(command_buffer);
    ASSERT_EQ(rig.Submit(rig.queue(), {{command_buffer}}, VK_NULL_HANDLE),
              VK_SUCCESS);
    rig.Check(*image, *source, regions);
  }
  Etc2GpuDevice::Stats stats = rig.gpu()->stats();
  EXPECT_EQ(stats.rescues, 3U);
  EXPECT_EQ(stats.rescued_uploads, 6U);
  EXPECT_EQ(stats.dropped_uploads, 0U);
  EXPECT_EQ(stats.slots_created, slots + 3) << "a rescue slot per pending call";
  EXPECT_EQ(rig.fake().calls("vkWaitForFences"), waits);

  rig.fake().HoldFences(false);
  const VkCommandBuffer caught_up = rig.Begin(pool);
  const auto regions = rig.Upload(caught_up, *image, *source);
  rig.End(caught_up);
  ASSERT_EQ(rig.Submit(rig.queue(), {{caught_up}}, VK_NULL_HANDLE),
            VK_SUCCESS);
  rig.Check(*image, *source, regions);
  stats = rig.gpu()->stats();
  EXPECT_EQ(stats.rescues, 4U);
  EXPECT_EQ(stats.slots_created, slots + 3) << "the reserved one again";
  EXPECT_EQ(rig.fake().calls("vkWaitForFences"), waits);
  Etc2GpuTesting().fail_slot_acquire = false;
  rig.Teardown();
  ExpectEverythingReleased(rig.fake());
  EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
}

// The CPU half reads Roblox's bytes where the decode pass would: rows and
// layers apart (bufferRowLength, bufferImageHeight), from a source Roblox
// does not keep mapped (mapped for the submit, as on the CPU route), in
// formats of every texel size. Without the host staging arena (its
// fallback) the staging is a one-off buffer that goes with Roblox's command
// buffer.
TEST_F(Etc2GpuRouteFakeTest, ARescueReadsStridedLayeredAndUnmappedSources) {
  for (const bool arena : {true, false}) {
    SCOPED_TRACE(arena ? "host staging arena" : "one-off staging");
    Etc2EmulationTesting().no_host_staging = !arena;
    FakeRig rig(GpuOptions());
    ASSERT_TRUE(rig.WarmUp());
    const VkCommandPool pool = rig.Pool(0);
    RigSource* source = rig.Source(64 * 1024, 37);
    rig.emulation().UnmapMemory(rig.fake().device(), source->memory);
    struct Case {
      VkFormat format;
      std::uint32_t width;
      std::uint32_t height;
    };
    const Case cases[] = {{VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, 30, 18},
                          {VK_FORMAT_EAC_R11_SNORM_BLOCK, 13, 9},
                          {VK_FORMAT_EAC_R11G11_UNORM_BLOCK, 21, 22}};
    for (const Case& test : cases) {
      RigImage* image = rig.Image(test.format, test.width, test.height, 1, 2);
      VkBufferImageCopy region{};
      region.bufferOffset = 64;
      region.bufferRowLength = test.width + 9;
      region.bufferImageHeight = test.height + 6;
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2};
      region.imageExtent = {test.width, test.height, 1};
      Etc2GpuTesting().fail_slot_acquire = true;
      const VkCommandBuffer command_buffer = rig.Begin(pool);
      rig.Upload(command_buffer, *image, *source, {region});
      rig.End(command_buffer);
      const std::uint64_t buffers = rig.fake().counts().buffers.live();
      const std::uint64_t unmaps = rig.fake().calls("vkUnmapMemory");
      ASSERT_EQ(rig.Submit(rig.queue(), {{command_buffer}}, VK_NULL_HANDLE),
                VK_SUCCESS);
      Etc2GpuTesting().fail_slot_acquire = false;
      EXPECT_EQ(rig.fake().calls("vkUnmapMemory"), unmaps + 1)
          << "the source was mapped for the submit only";
      if (!arena) {
        EXPECT_EQ(rig.fake().counts().buffers.live(), buffers + 1)
            << "the rescue's one-off staging";
      }
      const VkDeviceSize block = EtcBlockBytes(image->etc);
      const std::uint32_t blocks_wide = (test.width + 3) / 4;
      const std::uint32_t blocks_high = (test.height + 3) / 4;
      const VkDeviceSize row_stride =
          ((region.bufferRowLength + 3) / 4) * block;
      const VkDeviceSize layer_stride =
          row_stride * ((region.bufferImageHeight + 3) / 4);
      for (std::uint32_t layer = 0; layer < 2; ++layer) {
        std::vector<std::uint8_t> packed;
        for (std::uint32_t row = 0; row < blocks_high; ++row) {
          const std::uint8_t* from = source->bytes.data() +
                                     region.bufferOffset +
                                     layer * layer_stride + row * row_stride;
          packed.insert(packed.end(), from, from + blocks_wide * block);
        }
        std::vector<std::uint8_t> expected(
            static_cast<std::size_t>(test.width) * test.height *
            EtcDecodedTexelBytes(image->etc));
        ASSERT_TRUE(DecodeEtcImage(image->etc, packed.data(), packed.size(),
                                   test.width, test.height, expected.data(),
                                   expected.size()));
        const std::vector<std::uint8_t>* actual =
            rig.fake().ImageSubresource(image->image, 0, layer);
        ASSERT_NE(actual, nullptr);
        EXPECT_EQ(*actual, expected)
            << "format " << test.format << " layer " << layer;
      }
      // Roblox resets the command buffer: the staging goes with it.
      rig.emulation().ReleaseCommandBuffer(command_buffer);
      if (!arena) {
        EXPECT_EQ(rig.fake().counts().buffers.live(), buffers);
      }
    }
    EXPECT_EQ(rig.gpu()->stats().rescued_uploads, 3U);
    EXPECT_EQ(rig.gpu()->stats().dropped_uploads, 0U);
    EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
    void* data = nullptr;  // mapped again for the rig's teardown
    ASSERT_EQ(rig.emulation().MapMemory(rig.fake().device(), source->memory,
                                        0, VK_WHOLE_SIZE, 0, &data),
              VK_SUCCESS);
    rig.Teardown();
    ExpectEverythingReleased(rig.fake());
  }
}

// [CR-30] A lost device: the fence poll that finds it counts the slot as
// signalled (never waits on it) and ends the GPU route. That call's
// uploads are rescued on the CPU, later copies take the CPU route, and
// teardown does not hang on the lost fences.
TEST_F(Etc2GpuRouteFakeTest, ALostDeviceEndsTheGpuRoute) {
  ::testing::internal::CaptureStderr();
  {
    FakeRig rig(GpuOptions());
    ASSERT_TRUE(rig.WarmUp());
    const VkCommandPool pool = rig.Pool(0);
    RigImage* image = rig.Image(VK_FORMAT_EAC_R11G11_UNORM_BLOCK, 20, 20, 3);
    RigSource* source = rig.Source(4096, 47);
    const VkCommandBuffer first = rig.Begin(pool);
    rig.Upload(first, *image, *source);
    rig.End(first);
    ASSERT_EQ(rig.Submit(rig.queue(), {{first}}, VK_NULL_HANDLE), VK_SUCCESS);
    // Recorded while the route is ready; the device is lost before submit.
    const VkCommandBuffer second = rig.Begin(pool);
    const auto regions = rig.Upload(second, *image, *source);
    rig.End(second);
    rig.fake().LoseDevice();
    ASSERT_EQ(rig.Submit(rig.queue(), {{second}}, VK_NULL_HANDLE), VK_SUCCESS);
    rig.Check(*image, *source, regions);
    EXPECT_EQ(rig.gpu()->state(), Etc2GpuDevice::State::kFailed);
    EXPECT_EQ(rig.gpu()->stats().rescues, 1U);
    EXPECT_EQ(rig.gpu()->stats().dropped_uploads, 0U);
    EXPECT_EQ(rig.fake().calls("vkWaitForFences"), 0U);
    const VkCommandBuffer third = rig.Begin(pool);
    const auto later = rig.Upload(third, *image, *source);
    rig.End(third);
    const std::uint64_t passes = rig.gpu()->stats().passes;
    ASSERT_EQ(rig.Submit(rig.queue(), {{third}}, VK_NULL_HANDLE), VK_SUCCESS);
    rig.Check(*image, *source, later);
    EXPECT_EQ(rig.gpu()->stats().passes, passes) << "the CPU route served it";
    rig.Teardown();
    ExpectEverythingReleased(rig.fake());
    EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
  }
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(log.find("[vulkan] ETC2 GPU decode unavailable: the device was "
                     "lost"),
            std::string::npos)
      << log;
}

// Nothing submitted, nothing waited for: vkDestroyDevice of a device whose
// route was built but never used makes no vkWaitForFences call.
TEST_F(Etc2GpuRouteFakeTest, TeardownWaitsForNothingWhenNothingWasSubmitted) {
  FakeRig rig(GpuOptions());
  rig.Image(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 8, 8, 1);
  ASSERT_NE(rig.gpu(), nullptr);
  ASSERT_TRUE(rig.gpu()->WaitForBuild(std::chrono::seconds(10)));
  EXPECT_EQ(rig.gpu()->state(), Etc2GpuDevice::State::kBuilt);
  rig.Teardown();
  EXPECT_EQ(rig.fake().calls("vkWaitForFences"), 0U);
  ExpectEverythingReleased(rig.fake());
  EXPECT_GT(rig.fake().counts().pipelines.created, 0U);
}

// [CR-4] A process that exits without vkDestroyDevice destroys the
// emulation from a static destructor: that must neither join the pipeline
// build nor call Vulkan.
TEST_F(Etc2GpuRouteFakeTest, DestroyingTheEmulationNeverJoinsOrCallsVulkan) {
  FakeVulkanDevice fake;
  Etc2GpuTesting().build_delay_ms = 300;
  auto emulation = std::make_unique<VulkanEtc2Emulation>(GpuOptions());
  const float priority = 1.0F;
  VkDeviceQueueCreateInfo queue{};
  queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  VkDeviceCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  info.queueCreateInfoCount = 1;
  info.pQueueCreateInfos = &queue;
  emulation->RegisterDevice(fake.device(), fake.physical_device(), true,
                            fake.config().memory, fake.config().properties,
                            fake.config().queue_families, info,
                            FakeVulkanDevice::GetDeviceProcAddr());
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.format = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
  image_info.extent = {8, 8, 1};
  image_info.mipLevels = 1;
  image_info.arrayLayers = 1;
  VkImage image = VK_NULL_HANDLE;
  ASSERT_EQ(emulation->CreateImage(fake.device(), &image_info, nullptr, &image),
            VK_SUCCESS);
  Etc2GpuDevice* gpu = Etc2GpuDevice::Find(fake.device());
  ASSERT_NE(gpu, nullptr);
  EXPECT_EQ(gpu->state(), Etc2GpuDevice::State::kBuilding);
  const auto start = std::chrono::steady_clock::now();
  emulation.reset();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed, std::chrono::milliseconds(150))
      << "the destructor waited for the build thread";
  EXPECT_EQ(Etc2GpuDevice::Find(fake.device()), nullptr);
  for (const char* destroy : {"vkDestroyPipeline", "vkDestroyPipelineLayout",
                              "vkDestroyDescriptorSetLayout",
                              "vkDestroyShaderModule", "vkDestroyFence",
                              "vkWaitForFences", "vkDestroyBuffer"}) {
    EXPECT_EQ(fake.calls(destroy), 0U) << destroy;
  }
  // Let the detached build finish on the fake before it goes.
  for (int wait = 0;
       wait < 100 && fake.counts().pipelines.created == 0; ++wait) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_EQ(fake.counts().pipelines.created, 1U);
  fake.DestroyImage(image);
}

// [CR-9] Descriptor sets come from chained pools: with pools of one set,
// the selftest, a slot and two arena chunks still all get one.
TEST_F(Etc2GpuRouteFakeTest, ChainsDescriptorPools) {
  FakeVulkanDevice::Config config;
  config.max_sets_per_pool = 1;
  FakeRig rig(GpuOptions(), config);
  ASSERT_TRUE(rig.WarmUp());
  const VkCommandPool pool = rig.Pool(0);
  RigSource* source = rig.Source(16 * 1024 * 1024, 23);
  for (int index = 0; index < 4; ++index) {
    // 2048^2 RGBA8 decodes to 16 MiB: two per 32 MiB chunk at most.
    RigImage* image =
        rig.Image(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 2048, 2048, 1);
    const VkCommandBuffer command_buffer = rig.Begin(pool);
    const auto regions = rig.Upload(command_buffer, *image, *source);
    rig.End(command_buffer);
    ASSERT_EQ(rig.Submit(rig.queue(), {{command_buffer}}, VK_NULL_HANDLE),
              VK_SUCCESS);
    rig.Check(*image, *source, regions);
  }
  EXPECT_GE(rig.fake().counts().descriptor_pools.created, 3U);
  EXPECT_EQ(rig.gpu()->stats().dropped_uploads, 0U);
  EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
}

// MOCKTAIL_FRAME_STATS: a submit whose decode took >= 4 ms on the GPU is
// reported when its slot comes back (the fake's timestamps step 3 ms), and
// the totals carry the GPU fields.
TEST_F(Etc2GpuRouteFakeTest, FrameStatsReportGpuSubmits) {
  FakeVulkanDevice::Config config;
  config.timestamp_step_ns = 3000000;
  Etc2EmulationOptions options = GpuOptions();
  options.frame_stats = true;
  ::testing::internal::CaptureStderr();
  {
    FakeRig rig(options, config);
    ASSERT_TRUE(rig.WarmUp());
    const VkCommandPool pool = rig.Pool(0);
    RigImage* image = rig.Image(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 64, 64, 1);
    RigSource* source = rig.Source(4096, 29);
    for (int round = 0; round < 2; ++round) {
      const VkCommandBuffer command_buffer = rig.Begin(pool);
      rig.Upload(command_buffer, *image, *source);
      rig.End(command_buffer);
      ASSERT_EQ(rig.Submit(rig.queue(), {{command_buffer}}, VK_NULL_HANDLE),
                VK_SUCCESS);
    }
  }
  const std::string log = ::testing::internal::GetCapturedStderr();
  const std::string line = "  [vulkan] ETC2 GPU submit: 1 uploads, 0.0 MiB "
                           "compressed, 0.0 MiB written, record ";
  ASSERT_NE(log.find(line), std::string::npos) << log;
  const std::string submit = log.substr(log.find(line));
  EXPECT_NE(submit.find(" ms, gpu 6.00 ms (copy-in 3.00, decode 3.00; lag 1)"),
            std::string::npos)
      << submit.substr(0, submit.find('\n'));
  EXPECT_NE(log.find("[vulkan] ETC2 GPU decode ready on family 0 (build "),
            std::string::npos)
      << log;
  EXPECT_EQ(Field(log, kTotals, "| gpu uploads="), "2");
  EXPECT_EQ(Field(log, kTotals, " gpu="), "6");
}

// DESIGN C3.6: once a loading burst's output chunks have been idle for 2 s,
// all but 32 MiB of them go even when no GPU-route upload follows: the
// housekeeping that runs at a present (at most once a second) trims the
// arenas, not only a submit with GPU-route uploads.
TEST_F(Etc2GpuRouteFakeTest, IdleOutputChunksGoWithoutAnotherUpload) {
  constexpr VkDeviceSize kMebibyte = 1024 * 1024;
  FakeRig rig(GpuOptions());
  ASSERT_TRUE(rig.WarmUp());
  const VkCommandPool pool = rig.Pool(0);
  // Three 2048x2048 ETC2 RGB8 textures, 2 MiB compressed and 16 MiB
  // decoded each: two fill the first 32 MiB chunk, the third a second one.
  RigSource* source = rig.Source(6 * kMebibyte, 37);
  const VkCommandBuffer command_buffer = rig.Begin(pool);
  for (VkDeviceSize index = 0; index < 3; ++index) {
    RigImage* image =
        rig.Image(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 2048, 2048, 1);
    VkBufferImageCopy region{};
    region.bufferOffset = index * 2 * kMebibyte;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {2048, 2048, 1};
    rig.Upload(command_buffer, *image, *source, {region});
  }
  rig.End(command_buffer);
  ASSERT_EQ(rig.Submit(rig.queue(), {{command_buffer}}, VK_NULL_HANDLE),
            VK_SUCCESS);
  EXPECT_EQ(rig.gpu()->stats().passes, 1U);
  // Roblox resets its command buffer: both chunks are idle from now on.
  rig.emulation().ReleaseCommandBuffer(command_buffer);
  EXPECT_EQ(rig.gpu()->stats().arena_bytes, 64 * kMebibyte);
  rig.emulation().OnPresent();
  EXPECT_EQ(rig.gpu()->stats().arena_bytes, 64 * kMebibyte)
      << "idle for less than 2 s";
  std::this_thread::sleep_for(std::chrono::milliseconds(2100));
  rig.emulation().OnPresent();
  EXPECT_EQ(rig.gpu()->stats().arena_bytes, 32 * kMebibyte)
      << "32 MiB stay for the next burst";
  EXPECT_TRUE(rig.fake().errors().empty()) << rig.fake().errors().front();
}

}  // namespace
}  // namespace mocktail::graphics::testing
