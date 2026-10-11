#ifndef MOCKTAIL_RUNTIME_PERFORMANCE_POLICY_H_
#define MOCKTAIL_RUNTIME_PERFORMANCE_POLICY_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "runtime/frame_rate_policy.h"
#include "runtime/game_mode.h"

namespace mocktail {
namespace runtime {

enum class PhysicsWorkerMode {
  kAuto,
  kLatency,
  kThroughput,
};

bool ParsePhysicsWorkerMode(std::string_view value, PhysicsWorkerMode* mode);
std::string_view PhysicsWorkerModeName(PhysicsWorkerMode mode);

// Malformed topology still yields one worker.
int CalculateThroughputWorkerCount(int available_physical_cores);

struct PerformancePolicy {
  bool multithreaded_rendering = false;
  int physical_core_count = 0;
  std::uint64_t memory_limit_mb = 0;
  bool memory_limit_valid = true;
  GameModePolicy game_mode = GameModePolicy::kAuto;
  bool game_mode_valid = true;
  PhysicsWorkerMode physics_worker_mode = PhysicsWorkerMode::kAuto;
  bool physics_worker_mode_valid = true;

  bool memory_limit_enabled() const {
    return memory_limit_valid && memory_limit_mb != 0;
  }
  std::uint64_t memory_limit_bytes() const {
    return memory_limit_mb * 1024U * 1024U;
  }
};

PerformancePolicy ParsePerformancePolicy(
    std::string_view multithreaded_rendering,
    std::string_view memory_limit_mb = "0", std::string_view game_mode = "auto",
    std::string_view physics_worker_mode = "auto");

// Development knobs for the rendering preset, for A/B runs, not settings:
// fflags.json cannot change a preset key, a different value fails the start.
//   MOCKTAIL_TEXTURE_BUDGET_MB=16..4095 replaces the preset's 128 MiB
//   FIntRenderTextureTotalBudgetMB; =roblox leaves it and
//   FIntRenderTextureOrphanBudgetMB to Roblox.
//   MOCKTAIL_PRESET_TEXTURE_COMPRESSION=off leaves out the preset's
//   texture-compression flags.
// Unset or empty, they change nothing.
struct RenderingPresetKnobs {
  int texture_budget_mb = 0;  // 0 keeps the preset's budget
  bool roblox_texture_budget = false;
  bool texture_compression_flags = true;

  bool is_default() const {
    return texture_budget_mb == 0 && !roblox_texture_budget &&
           texture_compression_flags;
  }
};

inline constexpr int kMinimumTextureBudgetKnobMb = 16;
inline constexpr int kMaximumTextureBudgetKnobMb = 4095;

// nullptr stands for an unset variable. An invalid value fails with an
// error that names its variable.
bool ParseRenderingPresetKnobs(const char* texture_budget_mb,
                               const char* texture_compression,
                               RenderingPresetKnobs* knobs,
                               std::string* error);
bool RenderingPresetKnobsFromEnvironment(RenderingPresetKnobs* knobs,
                                         std::string* error);

// What the knobs change, for the startup log; empty for the defaults.
std::string DescribeRenderingPresetKnobs(const RenderingPresetKnobs& knobs);

// The client setting that forces Roblox's graphics quality level (its frame
// rate manager's level, 1 to 21) when engine.graphics_quality is a level.
// Roblox 2.738.1397 registers DebugFRMQualityLevelOverride as a DFInt and
// looks a key up only under the prefix it registered, so the FInt spelling
// that Mocktail sent until 1.0.4-plus.2 never took effect.
inline constexpr std::string_view kGraphicsQualityClientSetting =
    "DFIntDebugFRMQualityLevelOverride";

// Respects CPU affinity and falls back to the available logical count.
int DetectAvailablePhysicalCoreCount();

// Disabled policy preserves overrides; conflicting values fail. A graphics
// quality level (MOCKTAIL_GRAPHICS_QUALITY) is merged whether or not the
// rendering preset is.
bool MergePerformanceClientSettingsOverrides(const PerformancePolicy& policy,
                                             std::string_view base_json,
                                             std::string* merged_json,
                                             std::string* error);

// Applies mandatory HTTP and crash policies after caller settings.
bool MergeRuntimeClientSettingsOverrides(const FrameRatePolicy& frame_rate,
                                         const PerformancePolicy& performance,
                                         std::string_view base_json,
                                         std::string* merged_json,
                                         std::string* error);

// Keeps the legacy test bypass disabled. The host PermissionsProtocol now
// answers microphone authorization from capture configuration. No permission
// query opens a recording device.
bool MergeAudioDeviceMenuClientSettingsOverrides(std::string_view base_json,
                                                 std::string *merged_json,
                                                 std::string *error);

bool MergeAudioCaptureClientSettingsOverrides(bool microphone_enabled,
                                              std::string_view base_json,
                                              std::string* merged_json,
                                              std::string* error);

}  // namespace runtime
}  // namespace mocktail

#endif  // MOCKTAIL_RUNTIME_PERFORMANCE_POLICY_H_
