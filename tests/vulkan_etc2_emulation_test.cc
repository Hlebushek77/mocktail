#include "mocktail/graphics/vulkan_etc2_emulation.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "support/etc2_emulation_harness.h"
#include "support/etc2_emulation_scenarios.h"
#include "support/fake_vulkan_device.h"

// The ETC2/EAC emulation on a GPU-free fake device (DESIGN §9.2). Every
// scenario runs the emulation as the adapter would and checks the host
// images against the reference model and against the golden hashes frozen
// from plus.1, so later work on the upload path (the CPU fast path, the GPU
// route, BC mode) can prove it leaves today's texels alone.
//
// The goldens were frozen with the step-0 API; the scenario runner and the
// direct tests below now use the plus.2 API (Etc2EmulationOptions, the
// queue and pool registration, BeginCommandBuffer and the submit plan) and
// give the same hashes. The log tests check what MOCKTAIL_FRAME_STATS adds
// and that nothing else changed without it.

namespace mocktail::graphics::testing {
namespace {

std::vector<std::string> ScenarioNames() {
  std::vector<std::string> names;
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    names.push_back(scenario.name);
  }
  return names;
}

// Every fake-device error must be one the scenario expects, and each
// expected one must still happen (so a fix shows up as a test to update).
void ExpectOnlyAllowedErrors(const FakeVulkanDevice& fake,
                             const Scenario& scenario) {
  std::vector<bool> seen(scenario.allowed_fake_errors.size(), false);
  for (const std::string& error : fake.errors()) {
    bool allowed = false;
    for (std::size_t index = 0; index < scenario.allowed_fake_errors.size();
         ++index) {
      if (error.find(scenario.allowed_fake_errors[index]) !=
          std::string::npos) {
        allowed = true;
        seen[index] = true;
      }
    }
    if (!allowed) {
      ADD_FAILURE() << "fake device: " << error;
    }
  }
  for (std::size_t index = 0; index < seen.size(); ++index) {
    EXPECT_TRUE(seen[index])
        << "expected fake-device error no longer happens: "
        << scenario.allowed_fake_errors[index]
        << " (drop the allowance in the scenario if this was fixed)";
  }
}

void ExpectEverythingReleased(const FakeVulkanDevice& fake) {
  const FakeVulkanDevice::Counts& counts = fake.counts();
  EXPECT_EQ(counts.entry_point_buffers.live(), 0U)
      << "staging buffers the emulation did not destroy";
  EXPECT_EQ(counts.entry_point_memory.live(), 0U)
      << "staging memory the emulation did not free";
  EXPECT_EQ(counts.buffers.live(), 0U);
  EXPECT_EQ(counts.memory.live(), 0U);
  EXPECT_EQ(counts.images.live(), 0U);
  EXPECT_EQ(counts.command_pools.live(), 0U);
  EXPECT_EQ(counts.command_buffers.live(), 0U);
  EXPECT_EQ(counts.live_mappings, 0U);
}

// The legacy path (MOCKTAIL_ETC2_FAST_PATH=0) gives each copy a staging
// buffer of its own, which goes when its command buffer is released
// (ReleaseRecordResources), before the device. The fast path takes ranges of
// the host staging arena, whose chunks stay until the device goes; every
// scenario's staging fits one chunk, so one chunk is all that may remain
// (that the ranges themselves come back is checked by the arena tests
// below).
void ExpectStagingReleased(const FakeVulkanDevice& fake, bool fast_path) {
  const FakeVulkanDevice::Counts& counts = fake.counts();
  if (!fast_path) {
    EXPECT_EQ(counts.entry_point_buffers.live(), 0U)
        << "staging buffers outlived their command buffers";
    EXPECT_EQ(counts.entry_point_memory.live(), 0U)
        << "staging memory outlived its command buffers";
    return;
  }
  EXPECT_LE(counts.entry_point_buffers.live(), 1U) << "one arena chunk";
  EXPECT_EQ(counts.entry_point_memory.live(),
            counts.entry_point_buffers.live());
  EXPECT_EQ(counts.entry_point_buffers.created,
            counts.entry_point_buffers.live())
      << "the arena never frees a chunk it may reuse while the device lives";
}

void ExpectGolden(const Scenario& scenario, std::uint64_t hash) {
  std::uint64_t golden = 0;
  if (!FindEtc2EmulationGolden(scenario.name, &golden)) {
    ADD_FAILURE() << "no golden hash for " << scenario.name
                  << "; this run gives " << GoldenLine(scenario.name, hash);
    return;
  }
  EXPECT_EQ(hash, golden)
      << "the host images of " << scenario.name
      << " changed from the plus.1 output frozen in "
         "tests/vulkan_etc2_emulation_golden.inc; this run gives "
      << GoldenLine(scenario.name, hash);
}

// Every scenario on both CPU paths: the fast path (the default) and the
// legacy one that MOCKTAIL_ETC2_FAST_PATH=0 keeps for the A/B; both must
// give the frozen plus.1 texels.
class Etc2EmulationScenarioTest
    : public ::testing::TestWithParam<std::tuple<std::string, bool>> {
 protected:
  const std::string& name() const { return std::get<0>(GetParam()); }
  bool fast_path() const { return std::get<1>(GetParam()); }
  ScenarioOptions Options() const {
    ScenarioOptions options;
    options.emulation.fast_path = fast_path();
    return options;
  }

  // Runs the scenario with `teardown` and checks everything but the image
  // copies against the model, the golden hash and the fake's counts.
  void RunAndCheck(ScenarioTeardown teardown) {
    const Scenario* scenario = FindEtc2EmulationScenario(name());
    ASSERT_NE(scenario, nullptr);
    FakeVulkanDevice fake;
    FakeEmulationBackend backend(&fake);
    ScenarioOptions options = Options();
    options.teardown = teardown;
    bool released = false;
    options.after_command_buffers = [&]() {
      ExpectStagingReleased(fake, fast_path());
      released = true;
    };
    const ScenarioRun run = RunEtc2Scenario(*scenario, &backend, options);
    ASSERT_FALSE(run.skipped) << run.skip_reason;
    for (const std::string& failure : run.failures) {
      ADD_FAILURE() << failure;
    }
    EXPECT_EQ(released, teardown != ScenarioTeardown::kDestroyDeviceFirst);
    ExpectOnlyAllowedErrors(fake, *scenario);
    ExpectEverythingReleased(fake);
    ExpectGolden(*scenario, run.hash);
  }
};

TEST_P(Etc2EmulationScenarioTest, MatchesModelAndGolden) {
  const Scenario* scenario = FindEtc2EmulationScenario(name());
  ASSERT_NE(scenario, nullptr);
  FakeVulkanDevice fake;
  FakeEmulationBackend backend(&fake);
  ScenarioOptions options = Options();
  options.fake = &fake;
  options.after_command_buffers = [&]() {
    ExpectStagingReleased(fake, fast_path());
  };
  const ScenarioRun run = RunEtc2Scenario(*scenario, &backend, options);
  ASSERT_FALSE(run.skipped) << run.skip_reason;
  for (const std::string& failure : run.failures) {
    ADD_FAILURE() << failure;
  }
  ExpectOnlyAllowedErrors(fake, *scenario);
  ExpectEverythingReleased(fake);
  ExpectGolden(*scenario, run.hash);
  // Legacy: one staging buffer and allocation per copy command; fast: at
  // most one arena chunk.
  RecordProperty("staging_allocations",
                 std::to_string(fake.calls("vkAllocateMemory")));
  if (fast_path()) {
    EXPECT_LE(fake.calls("vkAllocateMemory"), 1U);
  }
}

// vkResetCommandPool: the adapter releases every command buffer of the pool
// first, which frees their staging.
TEST_P(Etc2EmulationScenarioTest, ResetPoolReleasesStaging) {
  RunAndCheck(ScenarioTeardown::kResetPool);
}

// vkDestroyCommandPool with its command buffers still allocated: the
// adapter releases each of them, then the emulation forgets the pool.
TEST_P(Etc2EmulationScenarioTest, DestroyPoolReleasesStaging) {
  RunAndCheck(ScenarioTeardown::kDestroyPool);
}

// Roblox may destroy its device while command buffers still hold uploads:
// the emulation frees their staging then, and the output is the same.
TEST_P(Etc2EmulationScenarioTest, DestroyDeviceFirstReleasesStaging) {
  RunAndCheck(ScenarioTeardown::kDestroyDeviceFirst);
}

INSTANTIATE_TEST_SUITE_P(
    Scenarios, Etc2EmulationScenarioTest,
    ::testing::Combine(::testing::ValuesIn(ScenarioNames()), ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<std::string, bool>>& info) {
      return std::get<0>(info.param) +
             (std::get<1>(info.param) ? "_Fast" : "_Legacy");
    });

TEST(Etc2EmulationScenarioListTest, EveryScenarioHasAGoldenHash) {
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    std::uint64_t hash = 0;
    EXPECT_TRUE(FindEtc2EmulationGolden(scenario.name, &hash)) << scenario.name;
  }
}

// ---- the log, with and without MOCKTAIL_FRAME_STATS ------------------------

std::vector<std::string> Lines(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string line; std::getline(stream, line);) {
    lines.push_back(line);
  }
  return lines;
}

bool StartsWith(const std::string& text, const std::string& prefix) {
  return text.compare(0, prefix.size(), prefix) == 0;
}

// The first line starting with `prefix`, or empty.
std::string FindLine(const std::vector<std::string>& lines,
                     const std::string& prefix) {
  for (const std::string& line : lines) {
    if (StartsWith(line, prefix)) {
      return line;
    }
  }
  return {};
}

// The text after "<key>" up to the next space, e.g. Field(line, "uploads=").
std::string Field(const std::string& line, const std::string& key) {
  const std::size_t start = line.find(key);
  if (start == std::string::npos) {
    return "<missing " + key + ">";
  }
  const std::size_t value = start + key.size();
  return line.substr(value, line.find(' ', value) - value);
}

// The count before the slash of "count/MiB" fields such as fam= entries.
std::string CountOf(const std::string& field) {
  return field.substr(0, field.find('/'));
}

// What the emulation printed before plus.2 (the image and decode lines stop
// after four each per process).
constexpr const char* kStep0Lines[] = {
    "  [vulkan] ETC2/EAC emulation enabled: ",
    "  [vulkan] ETC2 image ",
    "  [vulkan] ETC2 upload decoded ",
    "  [vulkan] ETC2 emulation: ",
    "  [vulkan] ETC2 decode totals: ",
    "  [vulkan] ETC2 decode totals so far: ",
    "  [vulkan] texture override loaded: ",
};

// The queue the runner registers, as the "ETC2 queues" line names it on the
// fake's RTX 3060 families.
std::string QueuesLine(std::uint32_t family) {
  const char* const kFamilies[] = {"family 0 (G|C|T|S) x1 of 16",
                                   "family 1 (T) x1 of 2",
                                   "family 2 (C|T) x1 of 8"};
  return std::string("  [vulkan] ETC2 queues: ") +
         (family < 3 ? kFamilies[family] : "?");
}

ScenarioRun RunCapturingStderr(const Scenario& scenario,
                               const ScenarioOptions& options,
                               std::string* log) {
  FakeVulkanDevice fake;
  FakeEmulationBackend backend(&fake);
  ::testing::internal::CaptureStderr();
  ScenarioRun run = RunEtc2Scenario(scenario, &backend, options);
  *log = ::testing::internal::GetCapturedStderr();
  return run;
}

// WP0's acceptance: without MOCKTAIL_FRAME_STATS the log keeps its lines,
// and only one "ETC2 queues" line per emulated device is new.
TEST(Etc2EmulationLogTest, WithoutFrameStatsOnlyTheQueuesLineIsNew) {
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    SCOPED_TRACE(scenario.name);
    std::string log;
    const ScenarioRun run = RunCapturingStderr(scenario, {}, &log);
    ASSERT_TRUE(run.failures.empty()) << run.failures.front();
    int queue_lines = 0;
    int totals_lines = 0;
    for (const std::string& line : Lines(log)) {
      if (StartsWith(line, "  [vulkan] ETC2 queues: ")) {
        ++queue_lines;
        EXPECT_EQ(line, QueuesLine(scenario.queue_family));
        continue;
      }
      EXPECT_TRUE(std::any_of(
          std::begin(kStep0Lines), std::end(kStep0Lines),
          [&line](const char* prefix) { return StartsWith(line, prefix); }))
          << "a new line without MOCKTAIL_FRAME_STATS: " << line;
      if (StartsWith(line, "  [vulkan] ETC2 decode totals: ")) {
        ++totals_lines;
        EXPECT_EQ(line.find(" | "), std::string::npos)
            << "the totals detail without MOCKTAIL_FRAME_STATS: " << line;
      }
    }
    EXPECT_EQ(queue_lines, scenario.emulated ? 1 : 0);
    EXPECT_EQ(totals_lines, scenario.emulated ? 1 : 0);
  }
}

// With MOCKTAIL_FRAME_STATS=1 the totals line keeps its prefix and appends
// the detail, whose step-0 counters must count exactly what the adapter
// asked of the emulation. A device with ETC2 still prints nothing.
TEST(Etc2EmulationLogTest, FrameStatsCountWhatTheAdapterCalled) {
  for (const Scenario& scenario : Etc2EmulationScenarios()) {
    SCOPED_TRACE(scenario.name);
    ScenarioOptions options;
    options.emulation.frame_stats = true;
    std::string log;
    const ScenarioRun run = RunCapturingStderr(scenario, options, &log);
    ASSERT_TRUE(run.failures.empty()) << run.failures.front();
    if (!scenario.emulated) {
      EXPECT_EQ(log, "");
      continue;
    }
    const std::vector<std::string> lines = Lines(log);
    EXPECT_EQ(FindLine(lines, "  [vulkan] ETC2 queues: "),
              QueuesLine(scenario.queue_family));
    EXPECT_EQ(FindLine(lines, "  [perf] "), "") << "no present, no window";
    const std::string totals =
        FindLine(lines, "  [vulkan] ETC2 decode totals: ");
    ASSERT_NE(totals, "") << log;
    // The prefix every existing parser reads, unchanged.
    const std::size_t detail = totals.find(" | gpu uploads=");
    ASSERT_NE(detail, std::string::npos) << totals;
    const std::string prefix = totals.substr(0, detail);
    EXPECT_NE(prefix.find(" (upscaled "), std::string::npos) << prefix;
    EXPECT_NE(prefix.find(" submits compressed="), std::string::npos) << prefix;
    EXPECT_NE(prefix.find(" ms longest="), std::string::npos) << prefix;
    EXPECT_EQ(prefix.substr(prefix.size() - 3), " ms") << prefix;

    const ScenarioRun::Calls& calls = run.calls;
    const std::string regions = std::to_string(calls.etc2_regions);
    EXPECT_EQ(Field(totals, " | gpu uploads="), "0");
    EXPECT_EQ(Field(totals, " gpu_disabled="), regions)
        << "every region took the CPU route because gpu=false";
    EXPECT_EQ(Field(totals, " gpu_not_ready="), "0");
    EXPECT_EQ(Field(totals, " gpu_unavailable="), "0");
    EXPECT_EQ(Field(totals, " | record calls="),
              std::to_string(calls.etc2_copies));
    EXPECT_EQ(Field(totals, " regions="), regions);
    // Family 1 of the RTX 3060 is transfer-only.
    const bool compute = scenario.queue_family != 1;
    EXPECT_EQ(CountOf(Field(totals, " fam=")),
              calls.etc2_regions == 0
                  ? "-"
                  : std::to_string(scenario.queue_family) + ":" + regions);
    EXPECT_EQ(CountOf(Field(totals, " nocompute=")), compute ? "0" : regions);
    EXPECT_EQ(Field(totals, " unknownfam="), "0");
    EXPECT_EQ(Field(totals, " begin="),
              "ot:0,sim:0,other:" + std::to_string(calls.begins));
    EXPECT_EQ(CountOf(Field(totals, " sec=")),
              std::to_string(calls.secondaries_executed));
    EXPECT_EQ(Field(totals, " submit="),
              std::to_string(calls.submit_calls) + "/" +
                  std::to_string(calls.submit_batches));
    EXPECT_EQ(Field(totals, " submit2="),
              std::to_string(calls.submit2_calls) + "/" +
                  std::to_string(calls.submit2_batches));
    EXPECT_EQ(Field(totals, " copyimg="),
              std::to_string(calls.etc2_image_copies));
    EXPECT_EQ(Field(totals, " rescued="), "0");
  }
}

// The two-second "[perf] etc2_window" line: printed at the first present
// two seconds after the window opened, with the window's own counts (here
// the whole scenario, so they equal the session totals).
TEST(Etc2EmulationLogTest, FrameStatsWindowLineSumsTheWindow) {
  const Scenario* scenario =
      FindEtc2EmulationScenario("SecondaryCommandBuffers");
  ASSERT_NE(scenario, nullptr);
  ScenarioOptions options;
  options.emulation.frame_stats = true;
  options.after_steps = [](VulkanEtc2Emulation* emulation) {
    emulation->OnPresent();  // opens the window
    emulation->OnPresent();  // too early: nothing
    std::this_thread::sleep_for(std::chrono::milliseconds(2050));
    emulation->OnPresent();
    emulation->OnPresent();  // a fresh, empty window prints nothing
  };
  std::string log;
  const ScenarioRun run = RunCapturingStderr(*scenario, options, &log);
  ASSERT_TRUE(run.failures.empty()) << run.failures.front();
  const std::vector<std::string> lines = Lines(log);
  const auto windows =
      std::count_if(lines.begin(), lines.end(), [](const std::string& line) {
        return StartsWith(line, "  [perf] etc2_window ");
      });
  ASSERT_EQ(windows, 1) << log;
  const std::string window = FindLine(lines, "  [perf] etc2_window ");
  const std::string totals = FindLine(lines, "  [vulkan] ETC2 decode totals: ");
  ASSERT_NE(totals, "") << log;
  EXPECT_LT(log.find(window), log.find(totals)) << "the window comes first";

  const ScenarioRun::Calls& calls = run.calls;
  const std::string regions = std::to_string(calls.etc2_regions);
  ASSERT_NE(calls.etc2_regions, 0U);
  EXPECT_EQ(Field(window, "etc2_window submits="),
            Field(totals, " uploads=" + Field(totals, " uploads=") + " in "));
  EXPECT_EQ(Field(window, " uploads="), Field(totals, " uploads="));
  EXPECT_EQ(Field(window, " record_calls="), std::to_string(calls.etc2_copies));
  EXPECT_EQ(Field(window, " staging_allocs="),
            Field(totals, " staging_allocs="));
  EXPECT_EQ(Field(window, " gpu="), "0");
  EXPECT_EQ(Field(window, " fallback="), regions);
  EXPECT_EQ(Field(window, " rescued="), "0");
  EXPECT_EQ(Field(window, " images=+"),
            Field(totals, "totals: images=") + "/-0");
  EXPECT_EQ(Field(window, " copyimg="), "0");
  EXPECT_EQ(Field(window, " regions="), regions);
  EXPECT_EQ(Field(window, " begin="),
            "ot:0,sim:0,other:" + std::to_string(calls.begins));
  EXPECT_EQ(Field(window, " sec="), Field(totals, " sec="));
  EXPECT_EQ(Field(window, " submit="),
            std::to_string(calls.submit_calls) + "/" +
                std::to_string(calls.submit_batches));
  EXPECT_EQ(Field(window, " submit2="), "0/0");
  EXPECT_NE(window.find(" new_threads=\""), std::string::npos) << window;
  for (const char* key :
       {" cpu_ms=",        " gpu_ms=",         " record_ms=",
        " release_ms=",    " arena_mib=",      " aliasviews=",
        " fam=",           " nocompute=",      " simcbs=",
        " simsec=",        " upload_batches=", " multibatch=",
        " src=",           " nonhost=",        " blits=",
        " rawcopies=",     " aliasbinds=",     " formatlists=",
        " afterdispatch=", " copies="}) {
    EXPECT_NE(window.find(key), std::string::npos) << key << " in " << window;
  }
}

// ---- direct tests of the emulation API --------------------------------------

Etc2EmulationOptions Options(std::uint32_t upscale) {
  Etc2EmulationOptions options = CpuRouteOptions();
  options.upscale = upscale;
  return options;
}

VkImageCreateInfo ImageInfo(VkFormat format, std::uint32_t width,
                            std::uint32_t height, std::uint32_t mips = 1,
                            std::uint32_t layers = 1) {
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
  return info;
}

// The adapter's vkCreateDevice and vkGetDeviceQueue hooks, for a device
// with one queue in family 0.
VkQueue Register(VulkanEtc2Emulation* emulation, FakeVulkanDevice* fake,
                 bool emulated = true) {
  const float priority = 1.0F;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = 0;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  emulation->RegisterDevice(fake->device(), fake->physical_device(), emulated,
                            fake->config().memory, fake->config().properties,
                            fake->config().queue_families, device_info,
                            FakeVulkanDevice::GetDeviceProcAddr());
  const VkQueue queue = fake->GetQueue(0, 0);
  emulation->RegisterQueue(fake->device(), queue, 0, 0);
  return queue;
}

// The adapter's vkCreateCommandPool hook.
VkCommandPool CreatePool(VulkanEtc2Emulation* emulation, FakeVulkanDevice* fake,
                         VkCommandPoolCreateFlags flags) {
  const VkCommandPool pool = fake->CreateCommandPool(0, flags);
  emulation->RegisterCommandPool(fake->device(), pool, 0, flags);
  return pool;
}

// The adapter's vkDestroyCommandPool hook: every command buffer of the pool
// is released first.
void DestroyPool(VulkanEtc2Emulation* emulation, FakeVulkanDevice* fake,
                 VkCommandPool pool,
                 const std::vector<VkCommandBuffer>& command_buffers) {
  for (VkCommandBuffer command_buffer : command_buffers) {
    emulation->ReleaseCommandBuffer(command_buffer);
  }
  fake->DestroyCommandPool(pool);
  emulation->ForgetCommandPool(fake->device(), pool);
}

// The adapter's vkQueueSubmit hook with one batch of one command buffer.
// The CPU route decodes in PrepareSubmit and leaves the batch alone.
VkResult Submit(VulkanEtc2Emulation* emulation, FakeVulkanDevice* fake,
                VkQueue queue, VkCommandBuffer command_buffer) {
  VkSubmitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  info.commandBufferCount = 1;
  info.pCommandBuffers = &command_buffer;
  Etc2SubmitPlan plan;
  emulation->PrepareSubmit(queue, 1, &info, VK_NULL_HANDLE, &plan);
  EXPECT_FALSE(plan.modified);
  EXPECT_TRUE(plan.pre_submits().empty());
  const VkResult result = fake->QueueSubmit(
      queue, plan.count(1), plan.submits(&info), plan.fence(VK_NULL_HANDLE));
  emulation->FinishSubmit(&plan, result, [&](VkFence fence) {
    return fake->QueueSubmit(queue, 0, nullptr, fence);
  });
  return result;
}

struct FormatCase {
  VkFormat format;
  VkFormat host;
};

constexpr FormatCase kFormatCases[] = {
    {VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM},
    {VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB},
    {VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM},
    {VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB},
    {VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM},
    {VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB},
    {VK_FORMAT_EAC_R11_UNORM_BLOCK, VK_FORMAT_R16_UNORM},
    {VK_FORMAT_EAC_R11_SNORM_BLOCK, VK_FORMAT_R16_SNORM},
    {VK_FORMAT_EAC_R11G11_UNORM_BLOCK, VK_FORMAT_R16G16_UNORM},
    {VK_FORMAT_EAC_R11G11_SNORM_BLOCK, VK_FORMAT_R16G16_SNORM},
};

TEST(VulkanEtc2EmulationTest, CreatesHostImagesAndStripsTexelViewFlag) {
  FakeVulkanDevice fake;
  VulkanEtc2Emulation emulation(Options(4));
  Register(&emulation, &fake);
  for (const FormatCase& test : kFormatCases) {
    for (const std::uint32_t size : {64U, 65U}) {
      VkImageCreateInfo info = ImageInfo(test.format, size, 32, 6);
      info.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                   VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
      VkImage image = VK_NULL_HANDLE;
      ASSERT_EQ(emulation.CreateImage(fake.device(), &info, nullptr, &image),
                VK_SUCCESS);
      VkImageCreateInfo host{};
      ASSERT_TRUE(fake.GetImageInfo(image, &host));
      EXPECT_EQ(host.format, test.host);
      EXPECT_EQ(host.flags, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
      EXPECT_EQ(host.mipLevels, 6U);
      const bool colour = test.host == VK_FORMAT_R8G8B8A8_UNORM ||
                          test.host == VK_FORMAT_R8G8B8A8_SRGB;
      const std::uint32_t scale = colour && size <= 64 ? 4 : 1;
      EXPECT_EQ(host.extent.width, size * scale) << test.format;
      EXPECT_EQ(host.extent.height, 32 * scale) << test.format;
      emulation.DestroyImage(fake.device(), image, nullptr);
    }
  }
  // Other formats, and every format on a device with ETC2, pass through.
  VkImageCreateInfo plain = ImageInfo(VK_FORMAT_R8G8B8A8_UNORM, 16, 16);
  VkImage image = VK_NULL_HANDLE;
  ASSERT_EQ(emulation.CreateImage(fake.device(), &plain, nullptr, &image),
            VK_SUCCESS);
  VkImageCreateInfo host{};
  ASSERT_TRUE(fake.GetImageInfo(image, &host));
  EXPECT_EQ(host.extent.width, 16U);
  emulation.DestroyImage(fake.device(), image, nullptr);
  emulation.DestroyDevice(fake.device());

  FakeVulkanDevice native;
  VulkanEtc2Emulation passthrough(Options(4));
  Register(&passthrough, &native, false);
  VkImageCreateInfo etc2 = ImageInfo(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 16, 16);
  etc2.flags = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT |
               VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  ASSERT_EQ(passthrough.CreateImage(native.device(), &etc2, nullptr, &image),
            VK_SUCCESS);
  ASSERT_TRUE(native.GetImageInfo(image, &host));
  EXPECT_EQ(host.format, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK);
  EXPECT_EQ(host.flags, etc2.flags);
  EXPECT_EQ(host.extent.width, 16U);
  passthrough.DestroyImage(native.device(), image, nullptr);
  passthrough.DestroyDevice(native.device());
  EXPECT_EQ(fake.counts().images.live(), 0U);
  EXPECT_EQ(native.counts().images.live(), 0U);
  EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
}

TEST(VulkanEtc2EmulationTest, ImageViewsUseTheHostFormat) {
  FakeVulkanDevice fake;
  VulkanEtc2Emulation emulation(Options(1));
  Register(&emulation, &fake);
  for (const FormatCase& test : kFormatCases) {
    VkImageCreateInfo info = ImageInfo(test.format, 32, 32);
    VkImage image = VK_NULL_HANDLE;
    ASSERT_EQ(emulation.CreateImage(fake.device(), &info, nullptr, &image),
              VK_SUCCESS);
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
    VkImageViewCreateInfo host{};
    ASSERT_TRUE(fake.GetImageViewInfo(view, &host));
    EXPECT_EQ(host.format, test.host);
    fake.DestroyImageView(view);
    // A view in another format (an aliasing uint view) is passed as is.
    view_info.format = VK_FORMAT_R32G32_UINT;
    ASSERT_EQ(
        emulation.CreateImageView(fake.device(), &view_info, nullptr, &view),
        VK_SUCCESS);
    ASSERT_TRUE(fake.GetImageViewInfo(view, &host));
    EXPECT_EQ(host.format, VK_FORMAT_R32G32_UINT);
    fake.DestroyImageView(view);
    emulation.DestroyImage(fake.device(), image, nullptr);
  }
  emulation.DestroyDevice(fake.device());
  EXPECT_EQ(fake.counts().image_views.live(), 0U);
}

// The pieces of one emulated upload, recorded by hand.
struct Upload {
  VkImage image = VK_NULL_HANDLE;
  VkBuffer source = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
};

Upload RecordUpload(VulkanEtc2Emulation* emulation, FakeVulkanDevice* fake,
                    VkCommandPool pool) {
  Upload upload;
  VkImageCreateInfo info = ImageInfo(VK_FORMAT_EAC_R11_UNORM_BLOCK, 16, 16);
  emulation->CreateImage(fake->device(), &info, nullptr, &upload.image);
  fake->CreateBuffer(128, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &upload.source);
  fake->AllocateMemory(256, 3, &upload.memory);
  emulation->NoteAllocateMemory(fake->device(), upload.memory, 3, 256);
  emulation->BindBufferMemory(fake->device(), upload.source, upload.memory, 0);
  void* data = nullptr;
  emulation->MapMemory(fake->device(), upload.memory, 0, VK_WHOLE_SIZE, 0,
                       &data);
  upload.command_buffer =
      fake->AllocateCommandBuffer(pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
  emulation->BeginCommandBuffer(upload.command_buffer, pool, 0);
  fake->BeginCommandBuffer(upload.command_buffer);
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {16, 16, 1};
  emulation->CmdCopyBufferToImage(
      fake->device(), upload.command_buffer, upload.source, upload.image,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  fake->EndCommandBuffer(upload.command_buffer);
  return upload;
}

void Destroy(VulkanEtc2Emulation* emulation, FakeVulkanDevice* fake,
             const Upload& upload) {
  emulation->DestroyImage(fake->device(), upload.image, nullptr);
  emulation->UnmapMemory(fake->device(), upload.memory);
  emulation->DestroyBuffer(fake->device(), upload.source, nullptr);
  emulation->FreeMemory(fake->device(), upload.memory, nullptr);
}

// The direct staging tests on both CPU paths: the legacy path's one-off
// buffers go with their command buffer, the fast path's arena chunk stays
// with the device and takes its ranges back.
class VulkanEtc2EmulationPathTest : public ::testing::TestWithParam<bool> {
 protected:
  bool fast_path() const { return GetParam(); }
  Etc2EmulationOptions PathOptions() const {
    Etc2EmulationOptions options = Options(1);
    options.fast_path = fast_path();
    return options;
  }
  // Staging buffers (and memories) alive while `copies` copies are
  // recorded: one each on the legacy path, one arena chunk on the fast one.
  std::uint64_t Staging(std::uint64_t copies) const {
    return fast_path() ? 1 : copies;
  }
};

TEST_P(VulkanEtc2EmulationPathTest, ReleasingACommandBufferFreesItsStaging) {
  FakeVulkanDevice fake;
  VulkanEtc2Emulation emulation(PathOptions());
  const VkQueue queue = Register(&emulation, &fake);
  const VkCommandPool pool = CreatePool(
      &emulation, &fake, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  const Upload upload = RecordUpload(&emulation, &fake, pool);
  EXPECT_EQ(fake.counts().entry_point_buffers.live(), 1U);
  EXPECT_EQ(fake.counts().entry_point_memory.live(), 1U);
  ASSERT_EQ(Submit(&emulation, &fake, queue, upload.command_buffer),
            VK_SUCCESS);
  EXPECT_EQ(fake.copy_buffer_to_image_regions(), 1U);

  emulation.ReleaseCommandBuffer(upload.command_buffer);
  EXPECT_EQ(fake.counts().entry_point_buffers.live(), fast_path() ? 1U : 0U);
  EXPECT_EQ(fake.counts().entry_point_memory.live(), fast_path() ? 1U : 0U);
  emulation.ReleaseCommandBuffer(upload.command_buffer);  // nothing left
  // Nothing to decode either: only the emulation sees this submit (the
  // fake would run the copy from the staging that is gone).
  VkSubmitInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  info.commandBufferCount = 1;
  info.pCommandBuffers = &upload.command_buffer;
  Etc2SubmitPlan plan;
  emulation.PrepareSubmit(queue, 1, &info, VK_NULL_HANDLE, &plan);
  EXPECT_EQ(plan.submits(&info), &info);
  emulation.FinishSubmit(&plan, VK_SUCCESS, nullptr);

  Destroy(&emulation, &fake, upload);
  DestroyPool(&emulation, &fake, pool, {upload.command_buffer});
  emulation.DestroyDevice(fake.device());
  EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
  EXPECT_EQ(fake.counts().buffers.live(), 0U);
  EXPECT_EQ(fake.counts().memory.live(), 0U);
}

// DestroyDevice frees the staging of that device's command buffers only.
TEST_P(VulkanEtc2EmulationPathTest, DestroyDeviceKeepsOtherDevicesStaging) {
  FakeVulkanDevice first;
  FakeVulkanDevice second;
  VulkanEtc2Emulation emulation(PathOptions());
  Register(&emulation, &first);
  const VkQueue second_queue = Register(&emulation, &second);
  const VkCommandPool first_pool = CreatePool(
      &emulation, &first, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  const VkCommandPool second_pool = CreatePool(
      &emulation, &second, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  const Upload one = RecordUpload(&emulation, &first, first_pool);
  const Upload two = RecordUpload(&emulation, &second, second_pool);
  Destroy(&emulation, &first, one);
  emulation.DestroyDevice(first.device());
  EXPECT_EQ(first.counts().entry_point_buffers.live(), 0U);
  EXPECT_EQ(first.counts().entry_point_memory.live(), 0U);
  EXPECT_EQ(second.counts().entry_point_buffers.live(), 1U);
  EXPECT_EQ(second.counts().entry_point_memory.live(), 1U);

  // The second device still decodes.
  ASSERT_EQ(Submit(&emulation, &second, second_queue, two.command_buffer),
            VK_SUCCESS);
  EXPECT_EQ(second.copy_buffer_to_image_regions(), 1U);
  emulation.ReleaseCommandBuffer(two.command_buffer);
  EXPECT_EQ(second.counts().entry_point_buffers.live(),
            fast_path() ? 1U : 0U);
  Destroy(&emulation, &second, two);
  // The first pool goes after its device, as Roblox may do it.
  DestroyPool(&emulation, &first, first_pool, {one.command_buffer});
  DestroyPool(&emulation, &second, second_pool, {two.command_buffer});
  emulation.DestroyDevice(second.device());
  EXPECT_EQ(second.counts().entry_point_buffers.live(), 0U);
  EXPECT_EQ(second.counts().entry_point_memory.live(), 0U);
  EXPECT_TRUE(first.errors().empty()) << first.errors().front();
  EXPECT_TRUE(second.errors().empty()) << second.errors().front();
}

// A command buffer from a destroyed pool: the adapter releases every one of
// them before vkDestroyCommandPool, as it does for vkResetCommandPool, and
// the emulation forgets the pool after it.
TEST_P(VulkanEtc2EmulationPathTest, PoolTeardownReleasesEveryCommandBuffer) {
  FakeVulkanDevice fake;
  VulkanEtc2Emulation emulation(PathOptions());
  Register(&emulation, &fake);
  const VkCommandPool pool = CreatePool(&emulation, &fake, 0);
  const Upload one = RecordUpload(&emulation, &fake, pool);
  const Upload two = RecordUpload(&emulation, &fake, pool);
  EXPECT_EQ(fake.counts().entry_point_buffers.live(), Staging(2));
  DestroyPool(&emulation, &fake, pool,
              {one.command_buffer, two.command_buffer});
  EXPECT_EQ(fake.counts().entry_point_buffers.live(), fast_path() ? 1U : 0U);
  EXPECT_EQ(fake.counts().entry_point_memory.live(), fast_path() ? 1U : 0U);
  Destroy(&emulation, &fake, one);
  Destroy(&emulation, &fake, two);
  emulation.DestroyDevice(fake.device());
  EXPECT_EQ(fake.counts().command_buffers.live(), 0U);
  EXPECT_EQ(fake.counts().entry_point_buffers.live(), 0U);
  EXPECT_EQ(fake.counts().entry_point_memory.live(), 0U);
  EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
}

INSTANTIATE_TEST_SUITE_P(Paths, VulkanEtc2EmulationPathTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return std::string(info.param ? "Fast" : "Legacy");
                         });

// ---- the host staging arena (fast path) ------------------------------------

// One copy of a `size` x `size` ETC2 RGBA8 texture (4 x size^2 bytes of
// staging) recorded into a fresh command buffer.
struct ArenaCopy {
  VkImage image = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
};

class HostStagingArenaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    queue_ = Register(&emulation_, &fake_);
    pool_ = CreatePool(&emulation_, &fake_,
                       VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    // Room for every copy below: 1024x1024 RGBA8 is 1 MiB compressed.
    fake_.CreateBuffer(1U << 20, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &source_);
    fake_.AllocateMemory(1U << 20, 3, &memory_);
    emulation_.NoteAllocateMemory(fake_.device(), memory_, 3, 1U << 20);
    emulation_.BindBufferMemory(fake_.device(), source_, memory_, 0);
    void* data = nullptr;
    emulation_.MapMemory(fake_.device(), memory_, 0, VK_WHOLE_SIZE, 0, &data);
  }
  void TearDown() override {
    for (const ArenaCopy& copy : copies_) {
      emulation_.ReleaseCommandBuffer(copy.command_buffer);
      emulation_.DestroyImage(fake_.device(), copy.image, nullptr);
    }
    copies_.clear();
    emulation_.UnmapMemory(fake_.device(), memory_);
    emulation_.DestroyBuffer(fake_.device(), source_, nullptr);
    emulation_.FreeMemory(fake_.device(), memory_, nullptr);
    fake_.DestroyCommandPool(pool_);
    emulation_.ForgetCommandPool(fake_.device(), pool_);
    emulation_.DestroyDevice(fake_.device());
    EXPECT_EQ(fake_.counts().entry_point_buffers.live(), 0U);
    EXPECT_EQ(fake_.counts().entry_point_memory.live(), 0U);
    EXPECT_TRUE(fake_.errors().empty()) << fake_.errors().front();
  }

  ArenaCopy Record(std::uint32_t size) {
    ArenaCopy copy;
    VkImageCreateInfo info =
        ImageInfo(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, size, size);
    emulation_.CreateImage(fake_.device(), &info, nullptr, &copy.image);
    copy.command_buffer =
        fake_.AllocateCommandBuffer(pool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    emulation_.BeginCommandBuffer(copy.command_buffer, pool_,
                                  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    fake_.BeginCommandBuffer(copy.command_buffer);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {size, size, 1};
    emulation_.CmdCopyBufferToImage(fake_.device(), copy.command_buffer,
                                    source_, copy.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                    &region);
    fake_.EndCommandBuffer(copy.command_buffer);
    copies_.push_back(copy);
    return copy;
  }

  // Where the host copy reads the copy's staging: buffer and offset.
  std::pair<VkBuffer, VkDeviceSize> StagingOf(const ArenaCopy& copy) const {
    const auto recorded = fake_.RecordedBufferCopies(copy.command_buffer);
    EXPECT_EQ(recorded.size(), 1U);
    if (recorded.empty() || recorded[0].regions.empty()) {
      return {VK_NULL_HANDLE, 0};
    }
    return {recorded[0].buffer, recorded[0].regions[0].bufferOffset};
  }

  // Roblox's vkQueueSubmit, then its vkResetCommandBuffer once the fence
  // signalled; with `destroy`, the image goes too.
  void SubmitAndRelease(const ArenaCopy& copy, bool destroy = false) {
    ASSERT_EQ(Submit(&emulation_, &fake_, queue_, copy.command_buffer),
              VK_SUCCESS);
    emulation_.ReleaseCommandBuffer(copy.command_buffer);
    fake_.ResetCommandBuffer(copy.command_buffer);
    if (destroy) {
      emulation_.DestroyImage(fake_.device(), copy.image, nullptr);
      copies_.erase(std::remove_if(copies_.begin(), copies_.end(),
                                   [&copy](const ArenaCopy& other) {
                                     return other.image == copy.image;
                                   }),
                    copies_.end());
    }
  }

  FakeVulkanDevice fake_;
  VulkanEtc2Emulation emulation_{Options(1)};
  VkQueue queue_ = VK_NULL_HANDLE;
  VkCommandPool pool_ = VK_NULL_HANDLE;
  VkBuffer source_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  std::vector<ArenaCopy> copies_;
};

// Ranges follow each other in one chunk, 256-byte aligned; a command
// buffer's release gives its range back, and once a chunk holds no range it
// is reused from its start.
TEST_F(HostStagingArenaTest, RangesComeBackWithTheirCommandBuffer) {
  const ArenaCopy a = Record(8);  // 256 bytes of staging
  const ArenaCopy b = Record(4);  // 64
  const ArenaCopy c = Record(12);
  const auto [chunk, a_offset] = StagingOf(a);
  ASSERT_NE(chunk, VK_NULL_HANDLE);
  EXPECT_EQ(a_offset, 0U);
  EXPECT_EQ(StagingOf(b), std::make_pair(chunk, VkDeviceSize{256}));
  EXPECT_EQ(StagingOf(c), std::make_pair(chunk, VkDeviceSize{512}));
  EXPECT_EQ(fake_.BufferSize(chunk), 64ULL << 20);
  EXPECT_EQ(fake_.BufferUsage(chunk), static_cast<VkBufferUsageFlags>(
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT));
  // Two of three released: the chunk still holds c, so bumping continues.
  SubmitAndRelease(a);
  SubmitAndRelease(b);
  const ArenaCopy d = Record(4);
  EXPECT_EQ(StagingOf(d), std::make_pair(chunk, VkDeviceSize{1280}));
  // All released: the next range starts over at 0.
  SubmitAndRelease(c);
  SubmitAndRelease(d);
  const ArenaCopy e = Record(4);
  EXPECT_EQ(StagingOf(e), std::make_pair(chunk, VkDeviceSize{0}));
  EXPECT_EQ(fake_.calls("vkAllocateMemory"), 1U);
  EXPECT_EQ(fake_.calls("vkCreateBuffer"), 1U);
  // Roblox's source through the adapter's hook, and the chunk once.
  EXPECT_EQ(fake_.calls("vkMapMemory"), 2U);
}

// DESIGN §9.2, WP1: after the first chunk exists, recording, submitting and
// releasing copies costs no Vulkan allocation, even when a session's staging
// adds up to many chunks' worth. 20 rounds of 8 x 512x512 RGBA8 (8 MiB of
// staging each) would need three chunks if ranges leaked.
TEST_F(HostStagingArenaTest, NoAllocationPerCopyAfterWarmUp) {
  std::uint64_t allocations = 0;
  std::uint64_t buffers = 0;
  for (int round = 0; round < 20; ++round) {
    std::vector<ArenaCopy> batch;
    for (int index = 0; index < 8; ++index) {
      batch.push_back(Record(512));
    }
    EXPECT_EQ(StagingOf(batch.front()).second, 0U) << "round " << round;
    for (const ArenaCopy& copy : batch) {
      SubmitAndRelease(copy, true);
    }
    if (round == 0) {
      allocations = fake_.calls("vkAllocateMemory");
      buffers = fake_.calls("vkCreateBuffer");
      EXPECT_EQ(allocations, 1U);
    }
  }
  EXPECT_EQ(fake_.calls("vkAllocateMemory"), allocations);
  EXPECT_EQ(fake_.calls("vkCreateBuffer"), buffers);
  EXPECT_EQ(fake_.calls("vkFreeMemory"), 0U);
  EXPECT_EQ(fake_.calls("vkDestroyBuffer"), 0U);
}

// The chunks are HOST_CACHED system memory (type 4 on the RTX 3060) unless
// MOCKTAIL_ETC2_STAGING_MEMORY=coherent asks for the write-combined type 3,
// and never the BAR (type 5).
TEST(HostStagingMemoryTest, PrefersCachedSystemMemoryUnlessCoherentIsAsked) {
  for (const Etc2StagingMemory kind :
       {Etc2StagingMemory::kCached, Etc2StagingMemory::kCoherent}) {
    FakeVulkanDevice fake;
    Etc2EmulationOptions options = Options(1);
    options.staging_memory = kind;
    VulkanEtc2Emulation emulation(options);
    const VkQueue queue = Register(&emulation, &fake);
    const VkCommandPool pool = CreatePool(
        &emulation, &fake, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    const Upload upload = RecordUpload(&emulation, &fake, pool);
    const auto recorded = fake.RecordedBufferCopies(upload.command_buffer);
    ASSERT_EQ(recorded.size(), 1U);
    EXPECT_EQ(fake.MemoryTypeIndex(fake.BufferMemory(recorded[0].buffer)),
              kind == Etc2StagingMemory::kCached ? 4U : 3U);
    ASSERT_EQ(Submit(&emulation, &fake, queue, upload.command_buffer),
              VK_SUCCESS);
    emulation.ReleaseCommandBuffer(upload.command_buffer);
    Destroy(&emulation, &fake, upload);
    DestroyPool(&emulation, &fake, pool, {upload.command_buffer});
    emulation.DestroyDevice(fake.device());
    EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
  }
  // Without cached system memory: the coherent system type; without any
  // system memory (an integrated GPU): the first coherent type.
  FakeVulkanDevice::Config uncached;
  uncached.memory.memoryTypes[4].propertyFlags &=
      ~static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
  FakeVulkanDevice::Config unified;
  for (std::uint32_t type = 0; type < unified.memory.memoryTypeCount; ++type) {
    unified.memory.memoryTypes[type].propertyFlags |=
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  }
  for (const auto& [config, expected] :
       {std::make_pair(uncached, 3U), std::make_pair(unified, 3U)}) {
    FakeVulkanDevice fake(config);
    VulkanEtc2Emulation emulation(Options(1));
    Register(&emulation, &fake);
    const VkCommandPool pool = CreatePool(&emulation, &fake, 0);
    const Upload upload = RecordUpload(&emulation, &fake, pool);
    const auto recorded = fake.RecordedBufferCopies(upload.command_buffer);
    ASSERT_EQ(recorded.size(), 1U);
    EXPECT_EQ(fake.MemoryTypeIndex(fake.BufferMemory(recorded[0].buffer)),
              expected);
    Destroy(&emulation, &fake, upload);
    DestroyPool(&emulation, &fake, pool, {upload.command_buffer});
    emulation.DestroyDevice(fake.device());
  }
}

// ---- the fake itself -----------------------------------------------------

TEST(FakeVulkanDeviceTest, ResolvesEntryPointsAndHonoursMissingOnes) {
  FakeVulkanDevice::Config config;
  config.missing_procs = {"vkCmdCopyBufferToImage2"};
  FakeVulkanDevice fake(config);
  const PFN_vkGetDeviceProcAddr get = FakeVulkanDevice::GetDeviceProcAddr();
  EXPECT_NE(get(fake.device(), "vkCmdCopyBufferToImage"), nullptr);
  EXPECT_NE(get(fake.device(), "vkMapMemory2KHR"), nullptr);
  EXPECT_EQ(get(fake.device(), "vkCmdCopyBufferToImage2"), nullptr);
  EXPECT_NE(get(fake.device(), "vkCmdCopyBufferToImage2KHR"), nullptr);
  EXPECT_EQ(get(fake.device(), "vkNoSuchFunction"), nullptr);
  EXPECT_EQ(get(VK_NULL_HANDLE, "vkCreateImage"), nullptr);
}

TEST(FakeVulkanDeviceTest, FlagsMisuseAValidationLayerWouldReport) {
  FakeVulkanDevice fake;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  ASSERT_EQ(fake.AllocateMemory(1024, 3, &memory), VK_SUCCESS);
  void* data = nullptr;
  ASSERT_EQ(fake.MapMemory(memory, 0, VK_WHOLE_SIZE, &data), VK_SUCCESS);
  EXPECT_NE(fake.MapMemory(memory, 0, VK_WHOLE_SIZE, &data), VK_SUCCESS);
  fake.UnmapMemory(memory);
  fake.UnmapMemory(memory);
  VkDeviceMemory device_local = VK_NULL_HANDLE;
  ASSERT_EQ(fake.AllocateMemory(1024, 1, &device_local), VK_SUCCESS);
  EXPECT_NE(fake.MapMemory(device_local, 0, VK_WHOLE_SIZE, &data), VK_SUCCESS);
  EXPECT_EQ(fake.errors().size(), 3U);

  VkBuffer buffer = VK_NULL_HANDLE;
  ASSERT_EQ(fake.CreateBuffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &buffer),
            VK_SUCCESS);
  ASSERT_EQ(fake.BindBufferMemory(buffer, memory, 0), VK_SUCCESS);
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = VK_FORMAT_R8G8B8A8_UNORM;
  info.extent = {8, 8, 1};
  info.mipLevels = 1;
  info.arrayLayers = 1;
  VkImage image = VK_NULL_HANDLE;
  ASSERT_EQ(fake.CreateImage(info, &image), VK_SUCCESS);
  const VkCommandPool pool = fake.CreateCommandPool(0, 0);
  const VkCommandBuffer command_buffer =
      fake.AllocateCommandBuffer(pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
  ASSERT_EQ(fake.BeginCommandBuffer(command_buffer), VK_SUCCESS);
  const auto copy = reinterpret_cast<PFN_vkCmdCopyBufferToImage>(
      FakeVulkanDevice::GetDeviceProcAddr()(fake.device(),
                                            "vkCmdCopyBufferToImage"));
  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = {4, 4, 1};  // 64 bytes: fits
  copy(command_buffer, buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  fake.ClearErrors();
  region.imageOffset = {6, 0, 0};  // past the image
  copy(command_buffer, buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  region.imageOffset = {0, 0, 0};
  region.bufferOffset = 4;  // past the buffer
  copy(command_buffer, buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  ASSERT_EQ(fake.errors().size(), 2U);
  EXPECT_NE(fake.errors()[0].find("06218"), std::string::npos);
  EXPECT_NE(fake.errors()[1].find("00171"), std::string::npos);
  fake.ClearErrors();
  ASSERT_EQ(fake.EndCommandBuffer(command_buffer), VK_SUCCESS);
  // Re-recording needs RESET_COMMAND_BUFFER on the pool.
  fake.BeginCommandBuffer(command_buffer);
  ASSERT_EQ(fake.errors().size(), 1U);
  EXPECT_NE(fake.errors()[0].find("00050"), std::string::npos);
  fake.EndCommandBuffer(command_buffer);
  fake.DestroyCommandPool(pool);
  fake.DestroyImage(image);
  fake.DestroyBuffer(buffer);
  fake.FreeMemory(memory);
  fake.FreeMemory(device_local);
  fake.FreeMemory(memory);  // twice
  EXPECT_EQ(fake.errors().size(), 2U);
}

TEST(FakeVulkanDeviceTest, ExecutesCopiesAgainstTheImageModel) {
  FakeVulkanDevice fake;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  ASSERT_EQ(fake.CreateBuffer(256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &buffer),
            VK_SUCCESS);
  ASSERT_EQ(fake.AllocateMemory(512, 3, &memory), VK_SUCCESS);
  ASSERT_EQ(fake.BindBufferMemory(buffer, memory, 256), VK_SUCCESS);
  std::uint8_t* bytes = fake.MemoryBytes(memory) + 256;
  for (int index = 0; index < 256; ++index) {
    bytes[index] = static_cast<std::uint8_t>(index);
  }
  VkImageCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = VK_IMAGE_TYPE_2D;
  info.format = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
  info.extent = {10, 6, 1};
  info.mipLevels = 2;
  info.arrayLayers = 2;
  VkImage image = VK_NULL_HANDLE;
  ASSERT_EQ(fake.CreateImage(info, &image), VK_SUCCESS);
  // 3x2 blocks of 8 bytes; mip 1 is 5x3: 2x1 blocks.
  ASSERT_EQ(fake.ImageSubresource(image, 0, 1)->size(), 48U);
  ASSERT_EQ(fake.ImageSubresource(image, 1, 0)->size(), 16U);
  const VkCommandPool pool = fake.CreateCommandPool(
      0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  const VkCommandBuffer command_buffer =
      fake.AllocateCommandBuffer(pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
  fake.BeginCommandBuffer(command_buffer);
  const auto copy = reinterpret_cast<PFN_vkCmdCopyBufferToImage>(
      FakeVulkanDevice::GetDeviceProcAddr()(fake.device(),
                                            "vkCmdCopyBufferToImage"));
  // The right column of blocks of layer 1, rows 3 blocks apart in a buffer
  // with 12-texel rows.
  VkBufferImageCopy region{};
  region.bufferOffset = 16;
  region.bufferRowLength = 12;
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1};
  region.imageOffset = {8, 0, 0};
  region.imageExtent = {2, 6, 1};
  copy(command_buffer, buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  fake.EndCommandBuffer(command_buffer);
  ASSERT_EQ(fake.Submit({{command_buffer}}), VK_SUCCESS);
  const std::vector<std::uint8_t>& layer = *fake.ImageSubresource(image, 0, 1);
  for (int byte = 0; byte < 8; ++byte) {
    EXPECT_EQ(layer[16 + byte], 16 + byte);       // block (2, 0)
    EXPECT_EQ(layer[40 + byte], 16 + 24 + byte);  // block (2, 1)
    EXPECT_EQ(layer[byte], 0xCD);                 // untouched
  }
  EXPECT_EQ(*fake.ImageSubresource(image, 0, 0),
            std::vector<std::uint8_t>(48, 0xCD));
  EXPECT_TRUE(fake.errors().empty()) << fake.errors().front();
  fake.DestroyCommandPool(pool);
  fake.DestroyImage(image);
  fake.DestroyBuffer(buffer);
  fake.FreeMemory(memory);
}

}  // namespace
}  // namespace mocktail::graphics::testing
