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
#include <unordered_map>

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

// The formats of the images created while enabled. Never destroyed, like
// the atomics, so a hook racing process exit finds a live map.
struct RememberedImages {
  std::mutex mutex;
  std::unordered_map<VkImage, VkFormat> formats;
};

RememberedImages& Images() {
  static RememberedImages* const images = new RememberedImages;
  return *images;
}

bool RememberedFormat(VkImage image, VkFormat* format) {
  if (image == VK_NULL_HANDLE) {
    return false;
  }
  RememberedImages& images = Images();
  std::lock_guard<std::mutex> lock(images.mutex);
  const auto found = images.formats.find(image);
  if (found == images.formats.end()) {
    return false;
  }
  *format = found->second;
  return true;
}

constexpr std::size_t Index(VkCallCounter counter) {
  return static_cast<std::size_t>(counter);
}

template <typename Region>
std::uint64_t RegionBytes(VkFormat format, std::uint32_t region_count,
                          const Region* regions) {
  VkFormatTexelBlock block;
  if (regions == nullptr || !LookupVkFormatTexelBlock(format, &block)) {
    return 0;
  }
  std::uint64_t bytes = 0;
  for (std::uint32_t i = 0; i < region_count; ++i) {
    const VkExtent3D& extent = regions[i].imageExtent;
    const std::uint32_t layers = regions[i].imageSubresource.layerCount;
    const std::uint64_t columns =
        (static_cast<std::uint64_t>(extent.width) + block.width - 1) /
        block.width;
    const std::uint64_t rows =
        (static_cast<std::uint64_t>(extent.height) + block.height - 1) /
        block.height;
    bytes += columns * rows * std::max<std::uint64_t>(extent.depth, 1) *
             (layers == VK_REMAINING_ARRAY_LAYERS
                  ? 1
                  : std::max<std::uint64_t>(layers, 1)) *
             block.bytes;
  }
  return bytes;
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

void VkCallStatsCountCreateImage(VkImage image, VkFormat format) {
  if (!VkCallStatsEnabled()) {
    return;
  }
  g_calls[Index(VkCallCounter::kCreateImage)].fetch_add(
      1, std::memory_order_relaxed);
  g_created_images[static_cast<std::size_t>(ClassifyVkImageFormat(format))]
      .fetch_add(1, std::memory_order_relaxed);
  if (image != VK_NULL_HANDLE) {
    RememberedImages& images = Images();
    std::lock_guard<std::mutex> lock(images.mutex);
    images.formats[image] = format;
  }
}

void VkCallStatsCountDestroyImage(VkImage image) {
  if (!VkCallStatsEnabled()) {
    return;
  }
  g_calls[Index(VkCallCounter::kDestroyImage)].fetch_add(
      1, std::memory_order_relaxed);
  if (image != VK_NULL_HANDLE) {
    RememberedImages& images = Images();
    std::lock_guard<std::mutex> lock(images.mutex);
    images.formats.erase(image);
  }
}

void VkCallStatsCountImageView(VkImage image, VkFormat view_format) {
  if (!VkCallStatsEnabled()) {
    return;
  }
  VkFormat image_format = VK_FORMAT_UNDEFINED;
  if (RememberedFormat(image, &image_format) && image_format != view_format) {
    g_calls[Index(VkCallCounter::kAliasView)].fetch_add(
        1, std::memory_order_relaxed);
  }
}

void VkCallStatsCountCopyBufferToImage(VkImage image,
                                       std::uint32_t region_count,
                                       const VkBufferImageCopy* regions) {
  if (!VkCallStatsEnabled()) {
    return;
  }
  g_calls[Index(VkCallCounter::kCopyBufferToImage)].fetch_add(
      1, std::memory_order_relaxed);
  VkFormat format = VK_FORMAT_UNDEFINED;
  if (RememberedFormat(image, &format)) {
    g_bytes[Index(VkCallCounter::kCopyBufferToImage)].fetch_add(
        BufferImageCopyBytes(format, region_count, regions),
        std::memory_order_relaxed);
  }
}

void VkCallStatsCountCopyBufferToImage2(const VkCopyBufferToImageInfo2* info) {
  if (!VkCallStatsEnabled()) {
    return;
  }
  g_calls[Index(VkCallCounter::kCopyBufferToImage)].fetch_add(
      1, std::memory_order_relaxed);
  VkFormat format = VK_FORMAT_UNDEFINED;
  if (info != nullptr && RememberedFormat(info->dstImage, &format)) {
    g_bytes[Index(VkCallCounter::kCopyBufferToImage)].fetch_add(
        BufferImageCopyBytes(format, info->regionCount, info->pRegions),
        std::memory_order_relaxed);
  }
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

bool LookupVkFormatTexelBlock(VkFormat format, VkFormatTexelBlock* block) {
  const auto in = [format](VkFormat first, VkFormat last) {
    return format >= first && format <= last;
  };
  // ASTC block sizes, in the order of the UNORM/SRGB pairs and of the
  // SFLOAT formats.
  static constexpr std::uint8_t kAstcBlocks[][2] = {
      {4, 4}, {5, 4},  {5, 5},  {6, 5},  {6, 6},   {8, 5},   {8, 6},
      {8, 8}, {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12},
  };
  VkFormatTexelBlock found;
  if (format == VK_FORMAT_R4G4_UNORM_PACK8 ||
      in(VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SRGB) ||
      format == VK_FORMAT_S8_UINT) {
    found.bytes = 1;
  } else if (in(VK_FORMAT_R4G4B4A4_UNORM_PACK16,
                VK_FORMAT_A1R5G5B5_UNORM_PACK16) ||
             in(VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SRGB) ||
             in(VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SFLOAT) ||
             format == VK_FORMAT_D16_UNORM ||
             format == VK_FORMAT_A4R4G4B4_UNORM_PACK16 ||
             format == VK_FORMAT_A4B4G4R4_UNORM_PACK16) {
    found.bytes = 2;
  } else if (in(VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_B8G8R8_SRGB)) {
    found.bytes = 3;
  } else if (in(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_A2B10G10R10_SINT_PACK32) ||
             in(VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SFLOAT) ||
             in(VK_FORMAT_R32_UINT, VK_FORMAT_R32_SFLOAT) ||
             in(VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) ||
             format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
             format == VK_FORMAT_D32_SFLOAT) {
    found.bytes = 4;
  } else if (in(VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16_SFLOAT)) {
    found.bytes = 6;
  } else if (in(VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT) ||
             in(VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SFLOAT) ||
             in(VK_FORMAT_R64_UINT, VK_FORMAT_R64_SFLOAT)) {
    found.bytes = 8;
  } else if (in(VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SFLOAT)) {
    found.bytes = 12;
  } else if (in(VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SFLOAT) ||
             in(VK_FORMAT_R64G64_UINT, VK_FORMAT_R64G64_SFLOAT)) {
    found.bytes = 16;
  } else if (in(VK_FORMAT_R64G64B64_UINT, VK_FORMAT_R64G64B64_SFLOAT)) {
    found.bytes = 24;
  } else if (in(VK_FORMAT_R64G64B64A64_UINT, VK_FORMAT_R64G64B64A64_SFLOAT)) {
    found.bytes = 32;
  } else if (in(VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_SRGB_BLOCK) ||
             in(VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC4_SNORM_BLOCK) ||
             in(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
                VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK) ||
             in(VK_FORMAT_EAC_R11_UNORM_BLOCK, VK_FORMAT_EAC_R11_SNORM_BLOCK)) {
    found = {8, 4, 4};
  } else if (in(VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_BC3_SRGB_BLOCK) ||
             in(VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK) ||
             in(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
                VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK) ||
             in(VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
                VK_FORMAT_EAC_R11G11_SNORM_BLOCK)) {
    found = {16, 4, 4};
  } else if (in(VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
                VK_FORMAT_ASTC_12x12_SRGB_BLOCK)) {
    const auto* dims =
        kAstcBlocks[(format - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) / 2];
    found = {16, dims[0], dims[1]};
  } else if (in(VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK,
                VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK)) {
    const auto* dims = kAstcBlocks[format - VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK];
    found = {16, dims[0], dims[1]};
  } else {
    return false;
  }
  if (block != nullptr) {
    *block = found;
  }
  return true;
}

std::uint64_t BufferImageCopyBytes(VkFormat format, std::uint32_t region_count,
                                   const VkBufferImageCopy* regions) {
  return RegionBytes(format, region_count, regions);
}

std::uint64_t BufferImageCopyBytes(VkFormat format, std::uint32_t region_count,
                                   const VkBufferImageCopy2* regions) {
  return RegionBytes(format, region_count, regions);
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
  {
    RememberedImages& images = Images();
    std::lock_guard<std::mutex> lock(images.mutex);
    images.formats.clear();
  }
  g_forced_enabled.store(-1, std::memory_order_relaxed);
}

void ForceVkCallStatsEnabledForTesting(bool enabled) {
  g_forced_enabled.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

}  // namespace mocktail::graphics
