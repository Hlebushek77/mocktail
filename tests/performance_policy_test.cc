#include "runtime/performance_policy.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <tuple>
#include <utility>

#define JSON_NOEXCEPTION 1
#include <nlohmann/json.hpp>
#include <string>

namespace mocktail {
namespace runtime {
namespace {

// Sets (or, with nullptr, unsets) one variable for the scope of a test and
// restores the previous value afterwards.
class ScopedEnvironmentVariable final {
 public:
  ScopedEnvironmentVariable(const char* name, const char* value)
      : name_(name) {
    if (const char* previous = std::getenv(name); previous != nullptr) {
      previous_ = previous;
    }
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnvironmentVariable() {
    if (previous_.has_value()) {
      setenv(name_, previous_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }
  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) =
      delete;

 private:
  const char* name_;
  std::optional<std::string> previous_;
};

// The default startup configuration: throughput physics workers (the
// runtime-config default), multithreaded rendering off, quality unset.
PerformancePolicy DefaultThroughputPolicy() {
  PerformancePolicy policy;
  policy.physical_core_count = 8;
  policy.physics_worker_mode = PhysicsWorkerMode::kThroughput;
  return policy;
}

// Every key and value the default configuration sends, frozen from
// v1.0.4-plus.1, so a change to the preset is always a deliberate one.
nlohmann::json Plus1DefaultPreset() {
  return {
      {"FIntTaskSchedulerThreadMin", "0"},
      {"FIntTaskSchedulerAsyncTasksMinimumThreadCount", "3"},
      {"FIntTaskSchedulerAutoThreadLimit", "8"},
      {"DFIntSimMidPhaseContactPipelineBatchSize", "128"},
      {"FIntSmoothClusterTaskQueueMaxParallelTasks", "8"},
      {"FIntOcclusionWorkerThreadCount", "4"},
      {"FFlagMovePrerenderV2", "True"},
      {"FFlagGcInParallelWithRenderPrepare3", "True"},
      {"FIntRenderTextureMipBias", "1"},
      {"FFlagSimRuntimeContentTranscodeBlockingCall", "False"},
      {"FFlagRenderEnableLowEndLOD", "True"},
      {"FIntTerrainArraySliceSize", "4"},
      {"FFlagFastGPULightGrid", "True"},
      {"FFlagRenderOptimizeLightGrid", "True"},
      {"FFlagRenderFixParticlesCulling", "True"},
      {"FFlagLuauIncrementalGC", "True"},
      {"FIntLuauGcStepMultiplier", "100"},
      {"FIntLuauGcGoalRatio", "200"},
      {"FIntAssetProviderThreads", "2"},
      {"FFlagUITextureCompressionDesktop", "True"},
      {"FFlagTCTextureCompressionDesktop", "True"},
      {"FFlagUITextureUncompressed", "False"},
      {"FFlagGpuVoxelCompression", "True"},
      {"FFlagMeshCompression", "True"},
      {"FFlagPhysicsMeshCompression", "True"},
      {"FFlagSupportMeshLOD", "True"},
      {"FFlagMeshLOD", "True"},
      {"FFlagMeshLodApplyGeomMeshOptimizer", "False"},
      {"FFlagAdaptiveMeshCacheSizeForAndroid", "True"},
      {"FIntDefaultMeshCacheSizeMB", "64"},
      {"FIntRenderMeshOrphanBudgetMB", "8"},
      {"FIntTerrainGpuMeshMemoryTargetMib", "32"},
      {"FIntAvatarMeshMemoryMax", "33554432"},
      {"FIntSmoothTerrainPhysicsCacheSize", "32"},
      {"FIntRenderTextureTotalBudgetMB", "128"},
      {"FIntRenderTextureOrphanBudgetMB", "8"},
      {"FFlagDebugTerrainVTCompressedTextures", "True"},
      {"FFlagStreamingObserverMemoryOptimization", "True"},
      {"FIntNetworkStreamingGCMaxMicroSecondLimit", "5000"},
      {"FFlagfeatureFastClusterCacheEnabled", "True"},
      {"FFlagEnableMultiVETextureManagerSharing", "True"},
      {"FFlagVulkanCompressedTextureAliasing", "True"},
      {"FFlagCSGPublishDedupSolidMesh", "True"},
      {"FFlagDeduplicateLodRemovalRegions", "True"},
      {"FFlagCurlUsePools", "True"},
      {"FIntRenderTextureCompositorBudget", "32"},
      {"FIntTextureCompositorActiveJobs", "1"},
      {"FIntTextureCompositorLowResFactor", "2"},
      {"FIntTextureCompositorMeshCache", "16"},
      {"FFlagContentProviderMmapAssets", "True"},
      {"FIntAssetProviderAssetCacheReadThreadCount", "1"},
      {"FIntAssetProviderAssetCacheWriteThreadCount", "1"},
      {"FIntInitialAudioAssetCacheSize", "32"},
      {"FIntAvatarTextureMemoryMax", "33554432"},
      {"FFlagEnableSLIMAvatars", "True"},
      {"FFlagEnableSlimAvatarsDefaultEnabled", "True"},
      {"FFlagLayeredClothingCacheOptimizations", "True"},
      {"FFlagRenderAllocateShadowMapResourcesOnDemand", "True"},
      {"FIntRenderShadowMapDepthCacheMemLimit", "16"},
      {"FIntTM2ShadowMapMaxMips", "1"},
      {"FIntDebugForceMSAASamples", "1"},
      {"FFlagLuauStartupGcSuppression", "False"},
      {"FIntLuauGcStepMul", "300"},
      {"FIntLuauGcGoalCore", "120"},
      {"FIntLuauGcStartupWorkingSetMb", "64"},
      {"FIntLuaGcMaxKb", "262144"},
      {"FIntTaskSchedulerMaxTempArenaSizeBytes", "16777216"},
      {"FFlagLocalStorageArenaOptimization", "True"},
      {"FIntProjectedMaxBytesUsedForSoundsMB", "32"},
      {"FIntAudioMetadataCacheSizeKB", "2048"},
      {"FIntDefaultAudioDecodeBufferSizeMs", "50"},
      {"FIntMaxAudibleSoundChannels", "32"},
      {"FIntDebugFRMQualityLevelOverride", "3"},
  };
}

nlohmann::json MergeDefault(const PerformancePolicy& policy) {
  std::string merged;
  std::string error;
  EXPECT_TRUE(
      MergePerformanceClientSettingsOverrides(policy, "{}", &merged, &error))
      << error;
  return nlohmann::json::parse(merged, nullptr, false, true);
}

// Unsets the quality level and both rendering-preset knobs, or sets the
// knobs to the given spellings.
class PresetEnvironment final {
 public:
  explicit PresetEnvironment(const char* texture_budget_mb = nullptr,
                             const char* texture_compression = nullptr)
      : quality_("MOCKTAIL_GRAPHICS_QUALITY", nullptr),
        budget_("MOCKTAIL_TEXTURE_BUDGET_MB", texture_budget_mb),
        compression_("MOCKTAIL_PRESET_TEXTURE_COMPRESSION",
                     texture_compression) {}

 private:
  ScopedEnvironmentVariable quality_;
  ScopedEnvironmentVariable budget_;
  ScopedEnvironmentVariable compression_;
};

nlohmann::json Without(nlohmann::json object,
                       std::initializer_list<const char*> names) {
  for (const char* name : names) {
    EXPECT_EQ(object.erase(name), 1U) << name;
  }
  return object;
}

TEST(PerformancePolicyTest, DefaultPresetIsUnchangedFromPlus1) {
  const PresetEnvironment environment;
  const nlohmann::json merged = MergeDefault(DefaultThroughputPolicy());
  const nlohmann::json expected = Plus1DefaultPreset();
  EXPECT_EQ(merged, expected);
  EXPECT_EQ(merged.size(), 73U);
  for (const auto& [name, value] : expected.items()) {
    EXPECT_EQ(merged.value(name, nlohmann::json()), value) << name;
  }
}

TEST(PerformancePolicyTest, KnobsAtTheirDefaultsKeepEveryModeIdentical) {
  PerformancePolicy multithreaded;
  multithreaded.multithreaded_rendering = true;
  multithreaded.physical_core_count = 14;
  PerformancePolicy latency = multithreaded;
  latency.physics_worker_mode = PhysicsWorkerMode::kLatency;
  const PerformancePolicy disabled;
  for (const PerformancePolicy& policy :
       {DefaultThroughputPolicy(), multithreaded, latency, disabled}) {
    std::string unset_json;
    std::string error;
    {
      const PresetEnvironment environment;
      ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
          policy, R"({"FFlagDebugDisplayFPS":"True"})", &unset_json, &error))
          << error;
    }
    for (const auto& [budget, compression] :
         std::initializer_list<std::pair<const char*, const char*>>{
             {"", ""}, {nullptr, "on"}, {"", "on"}}) {
      const PresetEnvironment environment(budget, compression);
      std::string spelled_json;
      ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
          policy, R"({"FFlagDebugDisplayFPS":"True"})", &spelled_json,
          &error))
          << error;
      EXPECT_EQ(spelled_json, unset_json);
    }
  }
}

TEST(PerformancePolicyTest, ParsesRenderingPresetKnobs) {
  RenderingPresetKnobs knobs;
  std::string error;
  ASSERT_TRUE(ParseRenderingPresetKnobs(nullptr, nullptr, &knobs, &error));
  EXPECT_TRUE(knobs.is_default());
  ASSERT_TRUE(ParseRenderingPresetKnobs("", "", &knobs, &error));
  EXPECT_TRUE(knobs.is_default());
  ASSERT_TRUE(ParseRenderingPresetKnobs(nullptr, "on", &knobs, &error));
  EXPECT_TRUE(knobs.is_default());

  for (const auto& [text, value] :
       std::initializer_list<std::pair<const char*, int>>{
           {"16", 16}, {"128", 128}, {"512", 512}, {"0320", 320},
           {"4095", 4095}}) {
    ASSERT_TRUE(ParseRenderingPresetKnobs(text, nullptr, &knobs, &error))
        << text;
    EXPECT_EQ(knobs.texture_budget_mb, value) << text;
    EXPECT_FALSE(knobs.roblox_texture_budget);
    EXPECT_TRUE(knobs.texture_compression_flags);
    EXPECT_FALSE(knobs.is_default());
  }

  ASSERT_TRUE(ParseRenderingPresetKnobs("roblox", "off", &knobs, &error));
  EXPECT_EQ(knobs.texture_budget_mb, 0);
  EXPECT_TRUE(knobs.roblox_texture_budget);
  EXPECT_FALSE(knobs.texture_compression_flags);

  for (const char* invalid : {"0", "15", "4096", "-1", "+512", " 512",
                              "512MB", "1e3", "Roblox", "auto",
                              "99999999999"}) {
    RenderingPresetKnobs untouched;
    untouched.texture_budget_mb = 77;
    error.clear();
    EXPECT_FALSE(
        ParseRenderingPresetKnobs(invalid, nullptr, &untouched, &error))
        << invalid;
    EXPECT_EQ(untouched.texture_budget_mb, 77) << invalid;
    EXPECT_NE(error.find("MOCKTAIL_TEXTURE_BUDGET_MB"), std::string::npos)
        << invalid;
  }
  for (const char* invalid : {"0", "1", "false", "true", "OFF", "no"}) {
    error.clear();
    EXPECT_FALSE(ParseRenderingPresetKnobs(nullptr, invalid, &knobs, &error))
        << invalid;
    EXPECT_NE(error.find("MOCKTAIL_PRESET_TEXTURE_COMPRESSION"),
              std::string::npos)
        << invalid;
  }
  EXPECT_FALSE(ParseRenderingPresetKnobs(nullptr, nullptr, nullptr, &error));
}

TEST(PerformancePolicyTest, TextureBudgetKnobReplacesOnlyTheTotalBudget) {
  const PresetEnvironment environment("512");
  nlohmann::json expected = Plus1DefaultPreset();
  expected["FIntRenderTextureTotalBudgetMB"] = "512";
  EXPECT_EQ(MergeDefault(DefaultThroughputPolicy()), expected);
}

TEST(PerformancePolicyTest, RobloxTextureBudgetLeavesBothBudgetsToRoblox) {
  {
    const PresetEnvironment environment("roblox");
    EXPECT_EQ(MergeDefault(DefaultThroughputPolicy()),
              Without(Plus1DefaultPreset(),
                      {"FIntRenderTextureTotalBudgetMB",
                       "FIntRenderTextureOrphanBudgetMB"}));

    // The budgets are no longer the preset's, so fflags.json may set them.
    std::string merged;
    std::string error;
    ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
        DefaultThroughputPolicy(),
        R"({"FIntRenderTextureTotalBudgetMB":"1024"})", &merged, &error))
        << error;
    EXPECT_EQ(nlohmann::json::parse(merged).at(
                  "FIntRenderTextureTotalBudgetMB"),
              "1024");
  }
  const PresetEnvironment environment;
  std::string merged;
  std::string error;
  EXPECT_FALSE(MergePerformanceClientSettingsOverrides(
      DefaultThroughputPolicy(),
      R"({"FIntRenderTextureTotalBudgetMB":"1024"})", &merged, &error));
  EXPECT_NE(error.find("FIntRenderTextureTotalBudgetMB"), std::string::npos);
}

TEST(PerformancePolicyTest, TextureCompressionOffLeavesOutOnlyItsFiveFlags) {
  const PresetEnvironment environment(nullptr, "off");
  EXPECT_EQ(MergeDefault(DefaultThroughputPolicy()),
            Without(Plus1DefaultPreset(),
                    {"FFlagUITextureCompressionDesktop",
                     "FFlagTCTextureCompressionDesktop",
                     "FFlagUITextureUncompressed",
                     "FFlagDebugTerrainVTCompressedTextures",
                     "FFlagVulkanCompressedTextureAliasing"}));
}

TEST(PerformancePolicyTest, CombinesBothKnobs) {
  const PresetEnvironment environment("320", "off");
  nlohmann::json expected =
      Without(Plus1DefaultPreset(),
              {"FFlagUITextureCompressionDesktop",
               "FFlagTCTextureCompressionDesktop", "FFlagUITextureUncompressed",
               "FFlagDebugTerrainVTCompressedTextures",
               "FFlagVulkanCompressedTextureAliasing"});
  expected["FIntRenderTextureTotalBudgetMB"] = "320";
  EXPECT_EQ(MergeDefault(DefaultThroughputPolicy()), expected);
}

TEST(PerformancePolicyTest, InvalidKnobFailsTheStartAndNamesTheVariable) {
  for (const auto& [budget, compression, variable] :
       std::initializer_list<std::tuple<const char*, const char*, const char*>>{
           {"lots", nullptr, "MOCKTAIL_TEXTURE_BUDGET_MB"},
           {nullptr, "disabled", "MOCKTAIL_PRESET_TEXTURE_COMPRESSION"}}) {
    const PresetEnvironment environment(budget, compression);
    // Invalid even where the preset is not applied, so a typo never passes
    // silently.
    for (const PerformancePolicy& policy :
         {DefaultThroughputPolicy(), PerformancePolicy{}}) {
      std::string merged = "unchanged";
      std::string error;
      EXPECT_FALSE(
          MergePerformanceClientSettingsOverrides(policy, "{}", &merged,
                                                  &error));
      EXPECT_EQ(merged, "unchanged");
      EXPECT_NE(error.find(variable), std::string::npos) << error;
    }
    std::string merged = "unchanged";
    std::string error;
    EXPECT_FALSE(MergeRuntimeClientSettingsOverrides(
        ParseFrameRatePolicy("60"), DefaultThroughputPolicy(), "{}", &merged,
        &error));
    EXPECT_EQ(merged, "unchanged");
    EXPECT_NE(error.find(variable), std::string::npos) << error;
  }
}

TEST(PerformancePolicyTest, DescribesActiveKnobsForTheStartupLog) {
  RenderingPresetKnobs knobs;
  EXPECT_EQ(DescribeRenderingPresetKnobs(knobs), "");
  knobs.texture_budget_mb = 512;
  EXPECT_EQ(DescribeRenderingPresetKnobs(knobs),
            "FIntRenderTextureTotalBudgetMB=512 (MOCKTAIL_TEXTURE_BUDGET_MB)");
  knobs.texture_budget_mb = 0;
  knobs.roblox_texture_budget = true;
  knobs.texture_compression_flags = false;
  EXPECT_EQ(DescribeRenderingPresetKnobs(knobs),
            "texture budgets left to Roblox "
            "(MOCKTAIL_TEXTURE_BUDGET_MB=roblox), texture-compression flags "
            "left out (MOCKTAIL_PRESET_TEXTURE_COMPRESSION=off)");
}

TEST(PerformancePolicyTest, ParsesDisabledByDefaultAndExplicitTrueValues) {
  EXPECT_FALSE(ParsePerformancePolicy("").multithreaded_rendering);
  EXPECT_FALSE(ParsePerformancePolicy("0").multithreaded_rendering);
  EXPECT_FALSE(ParsePerformancePolicy("false").multithreaded_rendering);
  EXPECT_FALSE(ParsePerformancePolicy("off").multithreaded_rendering);
  EXPECT_TRUE(ParsePerformancePolicy("1").multithreaded_rendering);
  EXPECT_TRUE(ParsePerformancePolicy("true").multithreaded_rendering);
  EXPECT_TRUE(ParsePerformancePolicy("on").multithreaded_rendering);
  EXPECT_EQ(ParsePerformancePolicy("0").physical_core_count, 0);
  EXPECT_GT(ParsePerformancePolicy("true").physical_core_count, 0);
  EXPECT_GT(DetectAvailablePhysicalCoreCount(), 0);
}

TEST(PerformancePolicyTest, ParsesPhysicsWorkerMode) {
  PhysicsWorkerMode mode = PhysicsWorkerMode::kLatency;
  EXPECT_TRUE(ParsePhysicsWorkerMode("auto", &mode));
  EXPECT_EQ(mode, PhysicsWorkerMode::kAuto);
  EXPECT_TRUE(ParsePhysicsWorkerMode("latency", &mode));
  EXPECT_EQ(mode, PhysicsWorkerMode::kLatency);
  EXPECT_TRUE(ParsePhysicsWorkerMode("throughput", &mode));
  EXPECT_EQ(mode, PhysicsWorkerMode::kThroughput);
  EXPECT_FALSE(ParsePhysicsWorkerMode("maximum", &mode));
  EXPECT_FALSE(ParsePhysicsWorkerMode("latency", nullptr));
  EXPECT_EQ(PhysicsWorkerModeName(PhysicsWorkerMode::kAuto), "auto");
  EXPECT_EQ(PhysicsWorkerModeName(PhysicsWorkerMode::kLatency), "latency");
  EXPECT_EQ(PhysicsWorkerModeName(PhysicsWorkerMode::kThroughput),
            "throughput");

  const PerformancePolicy latency =
      ParsePerformancePolicy("0", "0", "auto", "latency");
  EXPECT_TRUE(latency.physics_worker_mode_valid);
  EXPECT_EQ(latency.physics_worker_mode, PhysicsWorkerMode::kLatency);
  EXPECT_EQ(latency.physical_core_count, 0);
  const PerformancePolicy throughput =
      ParsePerformancePolicy("0", "0", "auto", "throughput");
  EXPECT_TRUE(throughput.physics_worker_mode_valid);
  EXPECT_EQ(throughput.physics_worker_mode, PhysicsWorkerMode::kThroughput);
  EXPECT_GT(throughput.physical_core_count, 0);
  EXPECT_FALSE(ParsePerformancePolicy("0", "0", "auto", "fast")
                   .physics_worker_mode_valid);
}

TEST(PerformancePolicyTest, MakesEveryPhysicalCoreAvailableToThroughputMode) {
  EXPECT_EQ(CalculateThroughputWorkerCount(1), 1);
  EXPECT_EQ(CalculateThroughputWorkerCount(2), 2);
  EXPECT_EQ(CalculateThroughputWorkerCount(4), 4);
  EXPECT_EQ(CalculateThroughputWorkerCount(8), 8);
  EXPECT_EQ(CalculateThroughputWorkerCount(14), 14);
  EXPECT_EQ(CalculateThroughputWorkerCount(32), 32);
}

TEST(PerformancePolicyTest, ParsesOptionalMemoryLimitInMebibytes) {
  const PerformancePolicy disabled = ParsePerformancePolicy("0", "0");
  EXPECT_TRUE(disabled.memory_limit_valid);
  EXPECT_FALSE(disabled.memory_limit_enabled());
  EXPECT_EQ(disabled.memory_limit_bytes(), 0U);

  const PerformancePolicy limited = ParsePerformancePolicy("0", "6144");
  EXPECT_TRUE(limited.memory_limit_valid);
  EXPECT_TRUE(limited.memory_limit_enabled());
  EXPECT_EQ(limited.memory_limit_mb, 6144U);
  EXPECT_EQ(limited.memory_limit_bytes(), 6144ULL * 1024ULL * 1024ULL);

  EXPECT_FALSE(ParsePerformancePolicy("0", "-1").memory_limit_valid);
  EXPECT_FALSE(ParsePerformancePolicy("0", "6GiB").memory_limit_valid);
  EXPECT_FALSE(ParsePerformancePolicy("0", "").memory_limit_valid);
  EXPECT_FALSE(
      ParsePerformancePolicy("0", "18446744073709551615").memory_limit_valid);
}

TEST(PerformancePolicyTest, DisabledPolicyPreservesUnrelatedOverrides) {
  std::string merged;
  std::string error;
  ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
      {}, R"({"FStringGraphicsVulkanShaderMTDenyPattern":"4318:.*"})", &merged,
      &error))
      << error;

  const nlohmann::json parsed =
      nlohmann::json::parse(merged, nullptr, false, true);
  ASSERT_TRUE(parsed.is_object());
  EXPECT_EQ(parsed.at("FStringGraphicsVulkanShaderMTDenyPattern"), "4318:.*");
  EXPECT_FALSE(
      parsed.contains("FIntTaskSchedulerAsyncTasksMinimumThreadCount"));
}

TEST(PerformancePolicyTest, SizesFullMultithreadingFromPhysicalCoreCount) {
  std::string merged;
  std::string error;
  ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
      {true, 14}, R"({"DFIntTaskSchedulerTargetFps":"240"})", &merged, &error))
      << error;

  const nlohmann::json parsed =
      nlohmann::json::parse(merged, nullptr, false, true);
  ASSERT_TRUE(parsed.is_object());
  EXPECT_EQ(parsed.at("DFIntTaskSchedulerTargetFps"), "240");
  EXPECT_EQ(parsed.at("FIntTaskSchedulerThreadMin"), "14");
  EXPECT_EQ(parsed.at("FIntTaskSchedulerAsyncTasksMinimumThreadCount"), "14");
  EXPECT_EQ(parsed.at("FIntSmoothClusterTaskQueueMaxParallelTasks"), "14");
  EXPECT_EQ(parsed.at("FIntOcclusionWorkerThreadCount"), "7");
  EXPECT_EQ(parsed.at("FFlagMovePrerenderV2"), "True");
  EXPECT_EQ(parsed.at("FFlagGcInParallelWithRenderPrepare3"), "True");
  EXPECT_FALSE(parsed.contains("FIntTaskSchedulerAutoThreadLimit"));
  EXPECT_FALSE(parsed.contains("FFlagTaskSchedulerLimitTargetFpsTo2402"));
}

TEST(PerformancePolicyTest, AcceptsIdenticalExplicitModeValues) {
  std::string merged;
  std::string error;
  ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
      {true, 14},
      R"({"FIntOcclusionWorkerThreadCount":"7","FFlagMovePrerenderV2":"True"})",
      &merged, &error))
      << error;

  const nlohmann::json parsed =
      nlohmann::json::parse(merged, nullptr, false, true);
  ASSERT_TRUE(parsed.is_object());
  EXPECT_EQ(parsed.at("FIntOcclusionWorkerThreadCount"), "7");
  EXPECT_EQ(parsed.at("FFlagMovePrerenderV2"), "True");
}

TEST(PerformancePolicyTest, WorkerModesRetainRobloxManagedAssetCacheSizes) {
  for (const bool throughput : {false, true}) {
    PerformancePolicy policy;
    policy.multithreaded_rendering = !throughput;
    policy.physical_core_count = 14;
    policy.physics_worker_mode = throughput ? PhysicsWorkerMode::kThroughput
                                            : PhysicsWorkerMode::kAuto;
    std::string merged;
    std::string error;
    ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
        policy, "{}", &merged, &error)) << error;
    const auto parsed = nlohmann::json::parse(merged);
    EXPECT_EQ(parsed.at("FIntSmoothClusterTaskQueueMaxParallelTasks"), "14");
    // ForceCacheSize values are bytes, not MiB; tiny fixed values cause
    // immediate eviction instead of preserving the normal asset caches.
    EXPECT_FALSE(parsed.contains("FIntMeshContentProviderForceCacheSize"));
    EXPECT_FALSE(parsed.contains("FIntSlimContentProviderForceCacheSize"));
  }
}

TEST(PerformancePolicyTest, PreservesExplicitAssetCacheByteBudgets) {
  const nlohmann::json requested = {
      {"FIntMeshContentProviderForceCacheSize", "268435456"},
      {"FIntSlimContentProviderForceCacheSize", "134217728"},
  };
  std::string merged;
  std::string error;
  ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
      {true, 14}, requested.dump(), &merged, &error)) << error;
  const auto parsed = nlohmann::json::parse(merged);
  for (const auto& [name, value] : requested.items()) {
    EXPECT_EQ(parsed.at(name), value) << name;
  }
}

TEST(PerformancePolicyTest, LatencyModePreservesRobloxManagedWorkerPools) {
  PerformancePolicy policy =
      ParsePerformancePolicy("true", "0", "auto", "latency");
  policy.physical_core_count = 14;
  std::string merged;
  std::string error;
  ASSERT_TRUE(
      MergePerformanceClientSettingsOverrides(policy, "{}", &merged, &error))
      << error;

  const nlohmann::json parsed =
      nlohmann::json::parse(merged, nullptr, false, true);
  ASSERT_TRUE(parsed.is_object());
  EXPECT_EQ(parsed.at("DFIntSimMidPhaseContactPipelineBatchSize"), "128");
  EXPECT_FALSE(parsed.contains("FIntTaskSchedulerThreadMin"));
  EXPECT_FALSE(parsed.contains("FIntTaskSchedulerAutoThreadLimit"));
  EXPECT_FALSE(parsed.contains("FIntSmoothClusterTaskQueueMaxParallelTasks"));
  EXPECT_FALSE(parsed.contains("FIntOcclusionWorkerThreadCount"));
}

TEST(PerformancePolicyTest, ThroughputModeUsesEveryPhysicalCore) {
  PerformancePolicy policy =
      ParsePerformancePolicy("false", "0", "auto", "throughput");
  policy.physical_core_count = 14;
  std::string merged;
  std::string error;
  ASSERT_TRUE(
      MergePerformanceClientSettingsOverrides(policy, "{}", &merged, &error))
      << error;

  const nlohmann::json parsed =
      nlohmann::json::parse(merged, nullptr, false, true);
  EXPECT_EQ(parsed.at("FIntTaskSchedulerThreadMin"), "0");
  EXPECT_EQ(parsed.at("FIntTaskSchedulerAutoThreadLimit"), "14");
  EXPECT_EQ(parsed.at("FIntTaskSchedulerAsyncTasksMinimumThreadCount"), "3");
  EXPECT_EQ(parsed.at("DFIntSimMidPhaseContactPipelineBatchSize"), "128");
  EXPECT_EQ(parsed.at("FIntSmoothClusterTaskQueueMaxParallelTasks"), "14");
  EXPECT_EQ(parsed.at("FIntOcclusionWorkerThreadCount"), "7");
}

TEST(PerformancePolicyTest, ComposesFrameRateBeforePerformanceMode) {
  std::string merged;
  std::string error;
  ASSERT_TRUE(MergeRuntimeClientSettingsOverrides(
      ParseFrameRatePolicy("144"), {true, 14},
      R"({"FStringGraphicsVulkanShaderMTDenyPattern":"4318:.*"})", &merged,
      &error))
      << error;

  const nlohmann::json parsed =
      nlohmann::json::parse(merged, nullptr, false, true);
  ASSERT_TRUE(parsed.is_object());
  EXPECT_EQ(parsed.at("DFIntTaskSchedulerTargetFps"), "144");
  EXPECT_EQ(parsed.at("FIntTaskSchedulerAsyncTasksMinimumThreadCount"), "14");
  EXPECT_EQ(parsed.at("FStringGraphicsVulkanShaderMTDenyPattern"), "4318:.*");
  EXPECT_EQ(parsed.at("FFlagUseRuntimeMutexRvHttpClient"), "False");
  EXPECT_EQ(parsed.at("DFFlagHttpClientSkipRetryForStreamingRequests"), "True");
  EXPECT_EQ(parsed.at("FFlagLuaAppDefaultHttpRetry"), "False");
}

TEST(PerformancePolicyTest, RejectsConflictingOrMalformedOverrides) {
  std::string merged = "unchanged";
  std::string error;
  EXPECT_FALSE(MergePerformanceClientSettingsOverrides(
      {true, 14}, R"({"FIntOcclusionWorkerThreadCount":"2"})", &merged,
      &error));
  EXPECT_EQ(merged, "unchanged");
  EXPECT_NE(error.find("FIntOcclusionWorkerThreadCount"), std::string::npos);

  error.clear();
  EXPECT_FALSE(MergePerformanceClientSettingsOverrides(
      {true, 14}, R"({"FFlagMovePrerenderV2":true})", &merged, &error));
  EXPECT_EQ(merged, "unchanged");
  EXPECT_NE(error.find("FFlagMovePrerenderV2"), std::string::npos);

  error.clear();
  EXPECT_FALSE(MergePerformanceClientSettingsOverrides({true, 14}, "not-json",
                                                       &merged, &error));
  EXPECT_EQ(merged, "unchanged");
  EXPECT_NE(error.find("JSON object"), std::string::npos);

  error.clear();
  EXPECT_FALSE(MergePerformanceClientSettingsOverrides({true, 14}, "[]",
                                                       &merged, &error));
  EXPECT_EQ(merged, "unchanged");
  EXPECT_NE(error.find("JSON object"), std::string::npos);

  PerformancePolicy latency =
      ParsePerformancePolicy("true", "0", "auto", "latency");
  latency.physical_core_count = 14;
  error.clear();
  EXPECT_FALSE(MergePerformanceClientSettingsOverrides(
      latency, R"({"DFIntSimMidPhaseContactPipelineBatchSize":"64"})", &merged,
      &error));
  EXPECT_NE(error.find("DFIntSimMidPhaseContactPipelineBatchSize"),
            std::string::npos);
}

TEST(PerformancePolicyTest, HostAudioMenuUsesBridgedDeviceQueries) {
  std::string merged;
  std::string error;
  ASSERT_TRUE(MergeAudioDeviceMenuClientSettingsOverrides(
      R"({"FFlagDebugUseWebRtcAudioDevices":"True","FFlagRemoteAudioDeviceSync":"True","unrelated":123})",
      &merged, &error));
  const auto parsed = nlohmann::json::parse(merged);
  EXPECT_EQ(parsed.at("FFlagDebugUseWebRtcAudioDevices"), "False");
  EXPECT_EQ(parsed.at("FFlagRemoteAudioDeviceSync"), "False");
  EXPECT_EQ(parsed.at("unrelated"), 123);
  EXPECT_FALSE(parsed.contains("DFFlagVoiceChatSkipPermissionCheckForTests"));
  EXPECT_FALSE(
      MergeAudioDeviceMenuClientSettingsOverrides("[]", &merged, &error));
}

TEST(PerformancePolicyTest, PermissionProtocolDoesNotUseTheTestBypass) {
  std::string merged;
  std::string error;
  ASSERT_TRUE(MergeAudioCaptureClientSettingsOverrides(
      true, R"({"DFFlagVoiceChatSkipPermissionCheckForTests":"True"})", &merged,
      &error))
      << error;
  nlohmann::json parsed = nlohmann::json::parse(merged);
  EXPECT_EQ(parsed.at("DFFlagVoiceChatSkipPermissionCheckForTests"), "False");

  ASSERT_TRUE(MergeAudioCaptureClientSettingsOverrides(false, merged, &merged,
                                                       &error))
      << error;
  parsed = nlohmann::json::parse(merged);
  EXPECT_EQ(parsed.at("DFFlagVoiceChatSkipPermissionCheckForTests"), "False");
}

}  // namespace
}  // namespace runtime
}  // namespace mocktail
