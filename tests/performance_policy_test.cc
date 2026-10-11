#include "runtime/performance_policy.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <map>
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

// Every key and value the default configuration sends, so a change to the
// preset is always a deliberate one: v1.0.4-plus.1's 73 keys without the 33
// that Roblox never read (kPresetKeysDroppedInPlus2 below).
nlohmann::json DefaultPreset() {
  return {
      {"FIntTaskSchedulerThreadMin", "0"},
      {"FIntTaskSchedulerAsyncTasksMinimumThreadCount", "3"},
      {"FIntTaskSchedulerAutoThreadLimit", "8"},
      {"DFIntSimMidPhaseContactPipelineBatchSize", "128"},
      {"FIntSmoothClusterTaskQueueMaxParallelTasks", "8"},
      {"FIntOcclusionWorkerThreadCount", "4"},
      {"FFlagMovePrerenderV2", "True"},
      {"FIntTerrainArraySliceSize", "4"},
      {"FFlagGpuVoxelCompression", "True"},
      {"FFlagPhysicsMeshCompression", "True"},
      {"FFlagSupportMeshLOD", "True"},
      {"FFlagMeshLodApplyGeomMeshOptimizer", "False"},
      {"FIntDefaultMeshCacheSizeMB", "64"},
      {"FIntRenderMeshOrphanBudgetMB", "8"},
      {"FIntAvatarMeshMemoryMax", "33554432"},
      {"FIntSmoothTerrainPhysicsCacheSize", "32"},
      {"FIntRenderTextureTotalBudgetMB", "128"},
      {"FIntRenderTextureOrphanBudgetMB", "8"},
      {"FFlagDebugTerrainVTCompressedTextures", "True"},
      {"FFlagStreamingObserverMemoryOptimization", "True"},
      {"FFlagfeatureFastClusterCacheEnabled", "True"},
      {"FFlagEnableMultiVETextureManagerSharing", "True"},
      {"FFlagVulkanCompressedTextureAliasing", "True"},
      {"FFlagDeduplicateLodRemovalRegions", "True"},
      {"FFlagCurlUsePools", "True"},
      {"FIntTextureCompositorLowResFactor", "2"},
      {"FFlagContentProviderMmapAssets", "True"},
      {"FFlagEnableSLIMAvatars", "True"},
      {"FFlagEnableSlimAvatarsDefaultEnabled", "True"},
      {"FFlagLayeredClothingCacheOptimizations", "True"},
      {"FFlagRenderAllocateShadowMapResourcesOnDemand", "True"},
      {"FIntRenderShadowMapDepthCacheMemLimit", "16"},
      {"FIntTM2ShadowMapMaxMips", "1"},
      {"FIntDebugForceMSAASamples", "1"},
      {"FFlagLuauStartupGcSuppression", "False"},
      {"FIntLuauGcGoalCore", "120"},
      {"FIntLuauGcStartupWorkingSetMb", "64"},
      {"FIntTaskSchedulerMaxTempArenaSizeBytes", "16777216"},
      {"FIntAudioMetadataCacheSizeKB", "2048"},
      {"FIntDefaultAudioDecodeBufferSizeMs", "50"},
  };
}

// What v1.0.4-plus.1 sent in the default configuration and Roblox 3092
// (2.738.1397) never read, with plus.1's values; 1.0.4-plus.2 stopped
// sending them. Roblox's client-settings parser in libroblox.so
// (0x231d970) takes the scope from the key's first letter (D: dynamic, F:
// fast) and the type from the prefix after it, and a variable's setter
// (0x6b4bee0 for integers, 0x231e6d6 for flags) refuses a value whose scope
// is not the one the variable was registered with; the registrations name
// that scope (plus.2 research, roblox-churn section 3, re-checked for every
// key below). So the FInt/FFlag spelling of a DFInt/DFFlag variable is
// ignored, and so is a name that is no variable or that the build does not
// contain.
//
// Registered as DFInt/DFFlag, sent without the D. Among them the FRM
// quality level that engine.graphics_quality meant to force (a level is now
// sent as DFIntDebugFRMQualityLevelOverride, see the quality tests) and the
// texture mip bias.
const std::map<std::string, std::string> kPresetKeysRegisteredWithDf = {
    {"FIntRenderTextureMipBias", "1"},
    {"FFlagUITextureCompressionDesktop", "True"},
    {"FFlagTCTextureCompressionDesktop", "True"},
    {"FFlagUITextureUncompressed", "False"},
    {"FFlagMeshCompression", "True"},
    {"FIntTerrainGpuMeshMemoryTargetMib", "32"},
    {"FIntNetworkStreamingGCMaxMicroSecondLimit", "5000"},
    {"FFlagCSGPublishDedupSolidMesh", "True"},
    {"FIntTextureCompositorActiveJobs", "1"},
    {"FIntAssetProviderAssetCacheReadThreadCount", "1"},
    {"FIntAssetProviderAssetCacheWriteThreadCount", "1"},
    {"FIntInitialAudioAssetCacheSize", "32"},
    {"FIntLuauGcStepMul", "300"},
    {"FIntLuaGcMaxKb", "262144"},
    {"FFlagLocalStorageArenaOptimization", "True"},
    {"FIntProjectedMaxBytesUsedForSoundsMB", "32"},
    {"FIntMaxAudibleSoundChannels", "32"},
    {"FIntDebugFRMQualityLevelOverride", "3"},
};

// The name exists, but not as an integer variable:
// RenderTextureCompositorBudget is an FLog channel and AvatarTextureMemoryMax
// a telemetry key.
const std::map<std::string, std::string> kPresetKeysThatAreNotVariables = {
    {"FIntRenderTextureCompositorBudget", "32"},
    {"FIntAvatarTextureMemoryMax", "33554432"},
};

// The name does not occur anywhere in libroblox.so of Roblox 3092.
const std::map<std::string, std::string> kPresetKeysUnknownToRoblox = {
    {"FFlagGcInParallelWithRenderPrepare3", "True"},
    {"FFlagSimRuntimeContentTranscodeBlockingCall", "False"},
    {"FFlagRenderEnableLowEndLOD", "True"},
    {"FFlagFastGPULightGrid", "True"},
    {"FFlagRenderOptimizeLightGrid", "True"},
    {"FFlagRenderFixParticlesCulling", "True"},
    {"FFlagLuauIncrementalGC", "True"},
    {"FIntLuauGcStepMultiplier", "100"},
    {"FIntLuauGcGoalRatio", "200"},
    {"FIntAssetProviderThreads", "2"},
    {"FFlagMeshLOD", "True"},
    {"FFlagAdaptiveMeshCacheSizeForAndroid", "True"},
    {"FIntTextureCompositorMeshCache", "16"},
};

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

TEST(PerformancePolicyTest, DefaultPresetIsPlus1WithoutItsDeadKeys) {
  const PresetEnvironment environment;
  const nlohmann::json merged = MergeDefault(DefaultThroughputPolicy());
  const nlohmann::json expected = DefaultPreset();
  EXPECT_EQ(merged, expected);
  EXPECT_EQ(merged.size(), 40U);
  for (const auto& [name, value] : expected.items()) {
    EXPECT_EQ(merged.value(name, nlohmann::json()), value) << name;
  }
  // Together with the dropped keys, plus.1's preset as it was.
  nlohmann::json plus1 = merged;
  for (const auto* dropped :
       {&kPresetKeysRegisteredWithDf, &kPresetKeysThatAreNotVariables,
        &kPresetKeysUnknownToRoblox}) {
    for (const auto& [name, value] : *dropped) {
      EXPECT_FALSE(merged.contains(name)) << name;
      plus1[name] = value;
    }
  }
  EXPECT_EQ(plus1.size(), 73U);
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
  nlohmann::json expected = DefaultPreset();
  expected["FIntRenderTextureTotalBudgetMB"] = "512";
  EXPECT_EQ(MergeDefault(DefaultThroughputPolicy()), expected);
}

TEST(PerformancePolicyTest, RobloxTextureBudgetLeavesBothBudgetsToRoblox) {
  {
    const PresetEnvironment environment("roblox");
    EXPECT_EQ(MergeDefault(DefaultThroughputPolicy()),
              Without(DefaultPreset(),
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

TEST(PerformancePolicyTest, TextureCompressionOffLeavesOutOnlyItsTwoFlags) {
  const PresetEnvironment environment(nullptr, "off");
  EXPECT_EQ(MergeDefault(DefaultThroughputPolicy()),
            Without(DefaultPreset(),
                    {"FFlagDebugTerrainVTCompressedTextures",
                     "FFlagVulkanCompressedTextureAliasing"}));
}

TEST(PerformancePolicyTest, CombinesBothKnobs) {
  const PresetEnvironment environment("320", "off");
  nlohmann::json expected =
      Without(DefaultPreset(), {"FFlagDebugTerrainVTCompressedTextures",
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

// What Roblox does with each key of the default preset (the scan above).
// A key added to the preset must be classified here.

// Registered with the prefix the preset uses: Roblox reads these.
constexpr const char* kPresetKeysRobloxReads[] = {
    "FIntTaskSchedulerThreadMin",
    "FIntTaskSchedulerAsyncTasksMinimumThreadCount",
    "FIntTaskSchedulerAutoThreadLimit",
    "DFIntSimMidPhaseContactPipelineBatchSize",
    "FIntSmoothClusterTaskQueueMaxParallelTasks",
    "FIntOcclusionWorkerThreadCount",
    "FFlagMovePrerenderV2",
    "FIntTerrainArraySliceSize",
    "FFlagGpuVoxelCompression",
    "FFlagMeshLodApplyGeomMeshOptimizer",
    "FIntDefaultMeshCacheSizeMB",
    "FIntRenderMeshOrphanBudgetMB",
    "FIntSmoothTerrainPhysicsCacheSize",
    "FIntRenderTextureTotalBudgetMB",
    "FIntRenderTextureOrphanBudgetMB",
    "FFlagDebugTerrainVTCompressedTextures",
    "FFlagStreamingObserverMemoryOptimization",
    "FFlagEnableMultiVETextureManagerSharing",
    "FFlagDeduplicateLodRemovalRegions",
    "FFlagCurlUsePools",
    "FIntTextureCompositorLowResFactor",
    "FFlagContentProviderMmapAssets",
    "FFlagEnableSlimAvatarsDefaultEnabled",
    "FIntRenderShadowMapDepthCacheMemLimit",
    "FIntDebugForceMSAASamples",
    "FFlagLuauStartupGcSuppression",
    "FIntLuauGcGoalCore",
    "FIntLuauGcStartupWorkingSetMb",
    "FIntTaskSchedulerMaxTempArenaSizeBytes",
    "FIntAudioMetadataCacheSizeKB",
    "FIntDefaultAudioDecodeBufferSizeMs",
};

// Registered, but the scan could not tell with which prefix: kept, as they
// may take effect.
constexpr const char* kPresetKeysWithUnknownScope[] = {
    "FFlagPhysicsMeshCompression",
    "FFlagSupportMeshLOD",
    "FIntAvatarMeshMemoryMax",
    "FFlagfeatureFastClusterCacheEnabled",
    "FFlagVulkanCompressedTextureAliasing",
    "FFlagEnableSLIMAvatars",
    "FFlagLayeredClothingCacheOptimizations",
    "FFlagRenderAllocateShadowMapResourcesOnDemand",
    "FIntTM2ShadowMapMaxMips",
};

TEST(PerformancePolicyTest, ClassifiesEveryPresetKeyByWhetherRobloxReadsIt) {
  const PresetEnvironment environment;
  const nlohmann::json preset = MergeDefault(DefaultThroughputPolicy());
  std::map<std::string, int> classified;
  const auto classify = [&](const auto& keys) {
    for (const char* key : keys) {
      EXPECT_TRUE(preset.contains(key)) << key << " is no longer sent";
      ++classified[key];
    }
  };
  classify(kPresetKeysRobloxReads);
  classify(kPresetKeysWithUnknownScope);
  for (const auto& [key, value] : preset.items()) {
    (void)value;
    EXPECT_EQ(classified[key], 1) << key;
  }
  EXPECT_EQ(classified.size(), preset.size());

  // 33 of plus.1's 73 keys did nothing; none is sent again, and the DF
  // spelling Roblox would read is not sent instead (that would change what
  // the game does), except the quality level a level in
  // engine.graphics_quality asks for.
  EXPECT_EQ(kPresetKeysRegisteredWithDf.size() +
                kPresetKeysThatAreNotVariables.size() +
                kPresetKeysUnknownToRoblox.size(),
            33U);
  for (const auto& [key, value] : kPresetKeysRegisteredWithDf) {
    (void)value;
    EXPECT_FALSE(preset.contains(key)) << key;
    EXPECT_FALSE(preset.contains("D" + key)) << key;
  }
}

// engine.graphics_quality (MOCKTAIL_GRAPHICS_QUALITY): a level is sent with
// the DF prefix Roblox registered the variable with, whether the rendering
// preset applies or not.
TEST(PerformancePolicyTest, QualityLevelSetsTheDfIntInEveryMode) {
  PerformancePolicy multithreaded;
  multithreaded.multithreaded_rendering = true;
  multithreaded.physical_core_count = 14;
  PerformancePolicy latency = multithreaded;
  latency.physics_worker_mode = PhysicsWorkerMode::kLatency;
  const PerformancePolicy disabled;
  for (const char* level : {"1", "7", "21"}) {
    const PresetEnvironment environment;
    const ScopedEnvironmentVariable quality("MOCKTAIL_GRAPHICS_QUALITY",
                                            level);
    for (const PerformancePolicy& policy :
         {DefaultThroughputPolicy(), multithreaded, latency, disabled}) {
      const nlohmann::json merged = MergeDefault(policy);
      EXPECT_EQ(merged.value(std::string(kGraphicsQualityClientSetting),
                             nlohmann::json()),
                level);
      EXPECT_FALSE(merged.contains("FIntDebugFRMQualityLevelOverride"));
    }
    nlohmann::json expected = DefaultPreset();
    expected[std::string(kGraphicsQualityClientSetting)] = level;
    EXPECT_EQ(MergeDefault(DefaultThroughputPolicy()), expected) << level;
  }
  EXPECT_EQ(kGraphicsQualityClientSetting,
            "DFIntDebugFRMQualityLevelOverride");
}

// default and manual (with the variable's auto and 0) leave the level to
// Roblox's own setting, as plus.1's FInt key always did in effect.
TEST(PerformancePolicyTest, DefaultAndManualQualitySetNoLevel) {
  for (const char* value :
       {static_cast<const char*>(nullptr), "", "default", "manual", "auto",
        "0"}) {
    const PresetEnvironment environment;
    const ScopedEnvironmentVariable quality("MOCKTAIL_GRAPHICS_QUALITY",
                                            value);
    EXPECT_EQ(MergeDefault(DefaultThroughputPolicy()), DefaultPreset())
        << (value != nullptr ? value : "(unset)");
    EXPECT_EQ(MergeDefault(PerformancePolicy{}), nlohmann::json::object())
        << (value != nullptr ? value : "(unset)");
  }
}

TEST(PerformancePolicyTest, InvalidQualityFailsTheStartAndNamesTheVariable) {
  for (const char* invalid : {"22", "-1", "3.5", "high", "Default"}) {
    const PresetEnvironment environment;
    const ScopedEnvironmentVariable quality("MOCKTAIL_GRAPHICS_QUALITY",
                                            invalid);
    for (const PerformancePolicy& policy :
         {DefaultThroughputPolicy(), PerformancePolicy{}}) {
      std::string merged = "unchanged";
      std::string error;
      EXPECT_FALSE(MergePerformanceClientSettingsOverrides(policy, "{}",
                                                           &merged, &error))
          << invalid;
      EXPECT_EQ(merged, "unchanged");
      EXPECT_NE(error.find("MOCKTAIL_GRAPHICS_QUALITY"), std::string::npos)
          << error;
    }
  }
}

// fflags.json may set the level only to the same value; the old FInt
// spelling is no longer Mocktail's and passes through.
TEST(PerformancePolicyTest, QualityLevelConflictsOnlyWithAnotherDfIntValue) {
  const PresetEnvironment environment;
  const ScopedEnvironmentVariable quality("MOCKTAIL_GRAPHICS_QUALITY", "7");
  std::string merged;
  std::string error;
  EXPECT_FALSE(MergePerformanceClientSettingsOverrides(
      PerformancePolicy{}, R"({"DFIntDebugFRMQualityLevelOverride":"5"})",
      &merged, &error));
  EXPECT_NE(error.find("DFIntDebugFRMQualityLevelOverride"),
            std::string::npos);
  ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
      PerformancePolicy{}, R"({"DFIntDebugFRMQualityLevelOverride":"7"})",
      &merged, &error))
      << error;
  ASSERT_TRUE(MergePerformanceClientSettingsOverrides(
      DefaultThroughputPolicy(), R"({"FIntDebugFRMQualityLevelOverride":"1"})",
      &merged, &error))
      << error;
  const nlohmann::json parsed = nlohmann::json::parse(merged);
  EXPECT_EQ(parsed.at("FIntDebugFRMQualityLevelOverride"), "1");
  EXPECT_EQ(parsed.at("DFIntDebugFRMQualityLevelOverride"), "7");
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
  EXPECT_FALSE(parsed.contains("FFlagGcInParallelWithRenderPrepare3"));
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
