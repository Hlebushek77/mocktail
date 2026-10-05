#include "mocktail/graphics/vulkan_call_stats.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>

namespace mocktail::graphics {
namespace {

constexpr std::uint64_t kWindowNs = 2000000000ULL;
constexpr double kMebibyte = 1024.0 * 1024.0;

// Atomics at namespace scope: zero before any code runs and trivially
// destroyed, so a call racing process exit never touches a dead object.
std::atomic<std::uint64_t> g_calls[kVkCallCounterCount];
std::atomic<std::uint64_t> g_bytes[kVkCallCounterCount];
std::atomic<std::uint64_t> g_created_images[kVkImageFormatClassCount];
std::atomic<std::uint64_t> g_blocked_ns[kVkBlockedCallCount];
std::atomic<VkCallStatsDispatchObserver> g_dispatch_observer{nullptr};
// -1: follow MOCKTAIL_VK_CALL_STATS; 0 or 1: forced by a test.
std::atomic<int> g_forced_enabled{-1};

std::mutex g_frame_mutex;
bool g_frame_started = false;
std::uint64_t g_window_start_ns = 0;
std::uint64_t g_frames = 0;

constexpr std::size_t Index(VkCallCounter counter) {
  return static_cast<std::size_t>(counter);
}

#define MOCKTAIL_VK_CALL_STATS_WRAPPER(name, counter, ret, params, args,     \
                                       extra)                              \
  std::atomic<PFN_vkVoidFunction> g_host_##name{nullptr};                  \
  VKAPI_ATTR ret VKAPI_CALL Counted_##name params {                        \
    g_calls[Index(VkCallCounter::counter)].fetch_add(                      \
        1, std::memory_order_relaxed);                                     \
    extra return reinterpret_cast<PFN_##name>(                             \
        g_host_##name.load(std::memory_order_acquire)) args;               \
  }                                                                        \
  static_assert(std::is_same_v<decltype(&Counted_##name), PFN_##name>,     \
                #name " does not match its Vulkan signature");
MOCKTAIL_VK_CALL_STATS_WRAPPED(MOCKTAIL_VK_CALL_STATS_WRAPPER)
#undef MOCKTAIL_VK_CALL_STATS_WRAPPER

const VkCallStatsProcInfo kProcs[] = {
#define MOCKTAIL_VK_CALL_STATS_INFO(name, counter, ret, params, args, extra) \
  {#name, VkCallCounter::counter,                                          \
   reinterpret_cast<PFN_vkVoidFunction>(&Counted_##name)},
    MOCKTAIL_VK_CALL_STATS_WRAPPED(MOCKTAIL_VK_CALL_STATS_INFO)
#undef MOCKTAIL_VK_CALL_STATS_INFO
};

std::atomic<PFN_vkVoidFunction>* const kHosts[] = {
#define MOCKTAIL_VK_CALL_STATS_HOST(name, counter, ret, params, args, extra) \
  &g_host_##name,
    MOCKTAIL_VK_CALL_STATS_WRAPPED(MOCKTAIL_VK_CALL_STATS_HOST)
#undef MOCKTAIL_VK_CALL_STATS_HOST
};

constexpr std::size_t kProcCount = sizeof(kProcs) / sizeof(kProcs[0]);
static_assert(kProcCount == sizeof(kHosts) / sizeof(kHosts[0]));

void Appendf(std::string* out, const char* format, ...)
    __attribute__((format(printf, 2, 3)));
void Appendf(std::string* out, const char* format, ...) {
  char buffer[256];
  va_list arguments;
  va_start(arguments, format);
  const int written = std::vsnprintf(buffer, sizeof(buffer), format, arguments);
  va_end(arguments);
  if (written > 0) {
    out->append(buffer, std::min<std::size_t>(static_cast<std::size_t>(written),
                                              sizeof(buffer) - 1));
  }
}

}  // namespace

VkImageFormatClass ClassifyVkImageFormat(VkFormat format) {
  const auto in = [format](VkFormat first, VkFormat last) {
    return format >= first && format <= last;
  };
  if (in(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_FORMAT_EAC_R11G11_SNORM_BLOCK)) {
    return VkImageFormatClass::kEtc2Eac;
  }
  if (in(VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC3_SRGB_BLOCK)) {
    return VkImageFormatClass::kBc1To3;
  }
  if (in(VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC6H_SFLOAT_BLOCK)) {
    return VkImageFormatClass::kBc4To6;
  }
  if (in(VK_FORMAT_BC7_UNORM_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK)) {
    return VkImageFormatClass::kBc7;
  }
  if (in(VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_12x12_SRGB_BLOCK) ||
      in(VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK, VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK)) {
    return VkImageFormatClass::kAstc;
  }
  if (in(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB) ||
      in(VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB)) {
    return VkImageFormatClass::kRgba8;
  }
  return VkImageFormatClass::kOther;
}

bool VkCallStatsFlagEnabled(const char* value) {
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool VkCallStatsEnabled() {
  const int forced = g_forced_enabled.load(std::memory_order_relaxed);
  if (forced >= 0) {
    return forced != 0;
  }
  static const bool enabled =
      VkCallStatsFlagEnabled(std::getenv("MOCKTAIL_VK_CALL_STATS"));
  return enabled;
}

bool IsVkCallStatsProc(const char* name) {
  if (name == nullptr) {
    return false;
  }
  for (const VkCallStatsProcInfo& proc : kProcs) {
    if (std::strcmp(proc.name, name) == 0) {
      return true;
    }
  }
  return false;
}

PFN_vkVoidFunction SelectVkCallStatsProc(bool enabled, const char* name,
                                         PFN_vkVoidFunction host,
                                         PFN_vkVoidFunction trampoline) {
  if (!enabled || name == nullptr || host == nullptr ||
      trampoline == nullptr) {
    return host;
  }
  for (std::size_t i = 0; i < kProcCount; ++i) {
    if (std::strcmp(kProcs[i].name, name) != 0) {
      continue;
    }
    PFN_vkVoidFunction bound = nullptr;
    if (!kHosts[i]->compare_exchange_strong(bound, trampoline,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire) &&
        bound != trampoline) {
      return host;
    }
    return kProcs[i].wrapper;
  }
  return host;
}

const VkCallStatsProcInfo* VkCallStatsProcs(std::size_t* count) {
  if (count != nullptr) {
    *count = kProcCount;
  }
  return kProcs;
}

void SetVkCallStatsDispatchObserver(VkCallStatsDispatchObserver observer) {
  g_dispatch_observer.store(observer, std::memory_order_release);
}

void NotifyVkCallStatsDispatch(VkCommandBuffer command_buffer) {
  const VkCallStatsDispatchObserver observer =
      g_dispatch_observer.load(std::memory_order_acquire);
  if (observer != nullptr) {
    observer(command_buffer);
  }
}

void VkCallStatsCount(VkCallCounter counter, std::uint64_t calls) {
  if (counter < VkCallCounter::kCount && VkCallStatsEnabled()) {
    g_calls[Index(counter)].fetch_add(calls, std::memory_order_relaxed);
  }
}

void AddVkCallStatsBytes(VkCallCounter counter, std::uint64_t bytes) {
  if (counter < VkCallCounter::kCount && VkCallStatsEnabled()) {
    g_bytes[Index(counter)].fetch_add(bytes, std::memory_order_relaxed);
  }
}

void VkCallStatsCountCreateImage(VkFormat format) {
  if (!VkCallStatsEnabled()) {
    return;
  }
  g_calls[Index(VkCallCounter::kCreateImage)].fetch_add(
      1, std::memory_order_relaxed);
  g_created_images[static_cast<std::size_t>(ClassifyVkImageFormat(format))]
      .fetch_add(1, std::memory_order_relaxed);
}

void AddVkCallStatsBlocked(VkBlockedCall call, std::uint64_t nanoseconds) {
  if (call < VkBlockedCall::kCount && VkCallStatsEnabled()) {
    g_blocked_ns[static_cast<std::size_t>(call)].fetch_add(
        nanoseconds, std::memory_order_relaxed);
  }
}

std::uint64_t BufferCopyBytes(std::uint32_t region_count,
                              const VkBufferCopy* regions) {
  std::uint64_t bytes = 0;
  for (std::uint32_t i = 0; regions != nullptr && i < region_count; ++i) {
    bytes += regions[i].size;
  }
  return bytes;
}

std::uint64_t BufferCopyBytes(const VkCopyBufferInfo2* info) {
  std::uint64_t bytes = 0;
  for (std::uint32_t i = 0;
       info != nullptr && info->pRegions != nullptr && i < info->regionCount;
       ++i) {
    bytes += info->pRegions[i].size;
  }
  return bytes;
}

std::uint64_t MappedRangeBytes(std::uint32_t range_count,
                               const VkMappedMemoryRange* ranges) {
  std::uint64_t bytes = 0;
  for (std::uint32_t i = 0; ranges != nullptr && i < range_count; ++i) {
    if (ranges[i].size != VK_WHOLE_SIZE) {
      bytes += ranges[i].size;
    }
  }
  return bytes;
}

void TakeVkCallStatsCounts(VkCallStatsWindow* window) {
  VkCallStatsWindow discarded;
  VkCallStatsWindow* out = window != nullptr ? window : &discarded;
  for (std::size_t i = 0; i < kVkCallCounterCount; ++i) {
    out->calls[i] = g_calls[i].exchange(0, std::memory_order_relaxed);
    out->bytes[i] = g_bytes[i].exchange(0, std::memory_order_relaxed);
  }
  for (std::size_t i = 0; i < kVkImageFormatClassCount; ++i) {
    out->created_images[i] =
        g_created_images[i].exchange(0, std::memory_order_relaxed);
  }
  for (std::size_t i = 0; i < kVkBlockedCallCount; ++i) {
    out->blocked_ns[i] = g_blocked_ns[i].exchange(0, std::memory_order_relaxed);
  }
}

bool AdvanceVkCallStatsFrame(std::uint64_t now_ns, VkCallStatsWindow* window) {
  std::lock_guard<std::mutex> lock(g_frame_mutex);
  if (!g_frame_started) {
    g_frame_started = true;
    g_window_start_ns = now_ns;
    g_frames = 0;
    return false;
  }
  ++g_frames;
  if (now_ns < g_window_start_ns || now_ns - g_window_start_ns < kWindowNs) {
    return false;
  }
  VkCallStatsWindow taken;
  taken.frames = g_frames;
  taken.span_ns = now_ns - g_window_start_ns;
  TakeVkCallStatsCounts(&taken);
  if (window != nullptr) {
    *window = taken;
  }
  g_window_start_ns = now_ns;
  g_frames = 0;
  return true;
}

std::string FormatVkCallStatsLine(const VkCallStatsWindow& window) {
  const double frames =
      static_cast<double>(window.frames > 0 ? window.frames : 1);
  const auto calls = [&window](VkCallCounter counter) {
    return window.calls[Index(counter)];
  };
  const auto per_frame = [&](VkCallCounter counter) {
    return static_cast<double>(calls(counter)) / frames;
  };
  const auto mib = [&window](VkCallCounter counter) {
    return static_cast<double>(window.bytes[Index(counter)]) / kMebibyte;
  };
  const auto blocked_ms = [&window](VkBlockedCall call) {
    return static_cast<double>(
               window.blocked_ns[static_cast<std::size_t>(call)]) /
           1e6;
  };
  const auto images = [&window](VkImageFormatClass format_class) {
    return static_cast<unsigned long long>(
        window.created_images[static_cast<std::size_t>(format_class)]);
  };
  const auto total = [&](VkCallCounter counter) {
    return static_cast<unsigned long long>(calls(counter));
  };

  std::string line;
  line.reserve(1024);
  Appendf(&line, "[perf] vk_calls frames=%llu span_ms=%llu per_frame:",
          static_cast<unsigned long long>(window.frames),
          static_cast<unsigned long long>(window.span_ns / 1000000ULL));
  Appendf(&line, " submit=%.1f draw=%.1f drawIdx=%.1f drawInd=%.1f",
          per_frame(VkCallCounter::kQueueSubmit),
          per_frame(VkCallCounter::kDraw),
          per_frame(VkCallCounter::kDrawIndexed),
          per_frame(VkCallCounter::kDrawIndirect));
  Appendf(&line, " dispatch=%.1f bindPipe=%.1f bindDS=%.1f push=%.1f",
          per_frame(VkCallCounter::kDispatch),
          per_frame(VkCallCounter::kBindPipeline),
          per_frame(VkCallCounter::kBindDescriptorSets),
          per_frame(VkCallCounter::kPushConstants));
  Appendf(&line, " bindVB=%.1f bindIB=%.1f pass=%.1f barrier=%.1f",
          per_frame(VkCallCounter::kBindVertexBuffers),
          per_frame(VkCallCounter::kBindIndexBuffer),
          per_frame(VkCallCounter::kBeginRenderPass),
          per_frame(VkCallCounter::kPipelineBarrier));
  Appendf(&line, " copyBuf=%.1f (%.2f MiB) copyB2I=%.1f (%.2f MiB)",
          per_frame(VkCallCounter::kCopyBuffer),
          mib(VkCallCounter::kCopyBuffer) / frames,
          per_frame(VkCallCounter::kCopyBufferToImage),
          mib(VkCallCounter::kCopyBufferToImage) / frames);
  Appendf(&line, " updBuf=%.1f (%.2f MiB) clear=%.1f updDS=%.1f",
          per_frame(VkCallCounter::kUpdateBuffer),
          mib(VkCallCounter::kUpdateBuffer) / frames,
          per_frame(VkCallCounter::kClear),
          per_frame(VkCallCounter::kUpdateDescriptorSets));
  Appendf(&line, " | window: copyImg=%llu blit=%llu img2buf=%llu resolve=%llu",
          total(VkCallCounter::kCopyImage), total(VkCallCounter::kBlitImage),
          total(VkCallCounter::kCopyImageToBuffer),
          total(VkCallCounter::kResolveImage));
  Appendf(&line, " allocDS=%llu resetDP=%llu",
          total(VkCallCounter::kAllocateDescriptorSets),
          total(VkCallCounter::kResetDescriptorPool));
  Appendf(&line, " flush=%llu (%.1f MiB) invalidate=%llu (%.1f MiB)",
          total(VkCallCounter::kFlushMappedMemory),
          mib(VkCallCounter::kFlushMappedMemory),
          total(VkCallCounter::kInvalidateMappedMemory),
          mib(VkCallCounter::kInvalidateMappedMemory));
  Appendf(&line, " map=%llu allocMem=%llu freeMem=%llu",
          total(VkCallCounter::kMapMemory),
          total(VkCallCounter::kAllocateMemory),
          total(VkCallCounter::kFreeMemory));
  Appendf(&line, " createBuf=%llu destroyBuf=%llu",
          total(VkCallCounter::kCreateBuffer),
          total(VkCallCounter::kDestroyBuffer));
  Appendf(&line,
          " createImg=%llu (etc2 %llu bc1-3 %llu bc4-6 %llu bc7 %llu astc %llu"
          " rgba8 %llu other %llu)",
          total(VkCallCounter::kCreateImage),
          images(VkImageFormatClass::kEtc2Eac),
          images(VkImageFormatClass::kBc1To3),
          images(VkImageFormatClass::kBc4To6), images(VkImageFormatClass::kBc7),
          images(VkImageFormatClass::kAstc), images(VkImageFormatClass::kRgba8),
          images(VkImageFormatClass::kOther));
  Appendf(&line, " destroyImg=%llu aliasView=%llu",
          total(VkCallCounter::kDestroyImage),
          total(VkCallCounter::kAliasView));
  Appendf(&line, " createPipe=%llu computePipe=%llu shader=%llu",
          total(VkCallCounter::kCreateGraphicsPipelines),
          total(VkCallCounter::kCreateComputePipelines),
          total(VkCallCounter::kCreateShaderModule));
  Appendf(&line, " sampler=%llu framebuffer=%llu",
          total(VkCallCounter::kCreateSampler),
          total(VkCallCounter::kCreateFramebuffer));
  Appendf(&line, " allocCB=%llu beginCB=%llu fenceStatus=%llu",
          total(VkCallCounter::kAllocateCommandBuffers),
          total(VkCallCounter::kBeginCommandBuffer),
          total(VkCallCounter::kGetFenceStatus));
  Appendf(&line, " | blocked fence=%.1f sem=%.1f acquire=%.1f",
          blocked_ms(VkBlockedCall::kWaitForFences),
          blocked_ms(VkBlockedCall::kWaitSemaphores),
          blocked_ms(VkBlockedCall::kAcquireNextImage));
  Appendf(&line, " present=%.1f submit=%.1f idle=%.1f ms",
          blocked_ms(VkBlockedCall::kQueuePresent),
          blocked_ms(VkBlockedCall::kQueueSubmit),
          blocked_ms(VkBlockedCall::kWaitIdle));
  return line;
}

void VkCallStatsOnPresent() {
  if (!VkCallStatsEnabled()) {
    return;
  }
  const std::uint64_t now_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  VkCallStatsWindow window;
  if (AdvanceVkCallStatsFrame(now_ns, &window)) {
    std::fprintf(stderr, "  %s\n", FormatVkCallStatsLine(window).c_str());
  }
}

void ResetVkCallStatsForTesting() {
  TakeVkCallStatsCounts(nullptr);
  {
    std::lock_guard<std::mutex> lock(g_frame_mutex);
    g_frame_started = false;
    g_window_start_ns = 0;
    g_frames = 0;
  }
  for (std::atomic<PFN_vkVoidFunction>* host : kHosts) {
    host->store(nullptr, std::memory_order_release);
  }
  g_dispatch_observer.store(nullptr, std::memory_order_release);
  g_forced_enabled.store(-1, std::memory_order_relaxed);
}

void ForceVkCallStatsEnabledForTesting(bool enabled) {
  g_forced_enabled.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

}  // namespace mocktail::graphics
