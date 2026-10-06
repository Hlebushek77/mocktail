#ifndef MOCKTAIL_GRAPHICS_ETC2_DECODE_TOTALS_H_
#define MOCKTAIL_GRAPHICS_ETC2_DECODE_TOTALS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mocktail::graphics {

// Why an upload into an emulated image took the CPU route (DESIGN §3.2, §6).
enum class Etc2CpuReason : std::uint8_t {
  kGpuDisabled,     // MOCKTAIL_ETC2_GPU=0
  kGpuNotReady,     // pipelines building or the selftest pending
  kGpuUnavailable,  // no GPU route on this device
  kUpscaled,        // small texture upscaled on the host
  kOverrides,       // texture overrides active
  kNoComputeQueue,  // command pool's queue family unknown or not COMPUTE
  kTooLarge,        // output beyond what one storage buffer may hold
  kArenaFull,       // no GPU output memory left
  kCount,
};

// "gpu_disabled", "gpu_not_ready", ... as the log lines print them.
const char* Etc2CpuReasonName(Etc2CpuReason reason);

// Queue families counted one by one; higher ones share the last slot.
inline constexpr std::size_t kEtc2CountedQueueFamilies = 8;
// Every memory type a device can have (VK_MAX_MEMORY_TYPES).
inline constexpr std::size_t kEtc2CountedMemoryTypes = 32;
// Buckets of record calls per destroyed image: 0, 1, 2-4, 5-16, more.
inline constexpr std::size_t kEtc2ImageCopyBuckets = 5;

// What ETC2/EAC emulation did, for one game session and, with
// MOCKTAIL_FRAME_STATS=1, for every two-second window. Roblox budgets these
// textures at their compressed size, while the host images hold them
// decoded (4x for RGBA and EAC, 8x for RGB, more for upscaled small ones).
// "Record" counters are taken when Roblox records a copy; "submit" ones when
// the batch reaches vkQueueSubmit.
struct Etc2DecodeTotals {
  // CPU route, at submit. The totals line has always printed these.
  std::uint64_t images = 0;           // emulated images created
  std::uint64_t upscaled_images = 0;  // of them, small textures upscaled
  std::uint64_t uploads = 0;          // uploads decoded
  std::uint64_t submits = 0;          // submits that decoded something
  std::uint64_t compressed_bytes = 0;
  std::uint64_t host_bytes = 0;       // written to the host images
  std::uint64_t decode_ns = 0;        // submit time spent on decoding
  std::uint64_t longest_submit_ns = 0;

  // GPU route, at submit.
  std::uint64_t gpu_uploads = 0;
  std::uint64_t gpu_submits = 0;
  std::uint64_t gpu_compressed_bytes = 0;
  std::uint64_t gpu_host_bytes = 0;
  std::uint64_t gpu_record_ns = 0;    // CPU time recording decode passes
  std::uint64_t gpu_time_ns = 0;      // GPU time, from timestamps
  std::uint64_t gpu_presubmits = 0;   // protected or device-group batches
  std::uint64_t gpu_slot_waits = 0;   // submits that waited for a free slot

  // Routing, at record.
  std::uint64_t cpu_fallbacks[static_cast<std::size_t>(Etc2CpuReason::kCount)] =
      {};
  std::uint64_t rescued_uploads = 0;  // GPU-route uploads served by the CPU
  std::uint64_t dropped_uploads = 0;  // uploads that could not be served

  // Memory.
  std::uint64_t arena_peak_bytes = 0;       // staging and output arenas
  std::uint64_t arena_over_cap_bytes = 0;   // peak of chunks past the cap
  std::uint64_t staging_allocs = 0;         // one-off staging buffers
  std::uint64_t staging_fallbacks = 0;      // arena full: one-off staging

  // Recording and releasing command buffers.
  std::uint64_t record_calls = 0;     // copies into emulated images
  std::uint64_t record_uploads = 0;   // their regions
  std::uint64_t record_ns = 0;        // time inside those copy hooks
  std::uint64_t release_ns = 0;       // time releasing recorded uploads

  // Step-0 diagnostics: how Roblox uploads (DESIGN C0.4).
  std::uint64_t images_destroyed = 0;
  std::uint64_t bc_images = 0;
  std::uint64_t image_copy_buckets[kEtc2ImageCopyBuckets] = {};
  std::uint64_t family_uploads[kEtc2CountedQueueFamilies] = {};
  std::uint64_t family_bytes[kEtc2CountedQueueFamilies] = {};  // compressed
  std::uint64_t unknown_family_uploads = 0;
  std::uint64_t no_compute_uploads = 0;
  std::uint64_t no_compute_bytes = 0;
  std::uint64_t begin_one_time = 0;      // vkBeginCommandBuffer by flags
  std::uint64_t begin_simultaneous = 0;
  std::uint64_t begin_other = 0;
  std::uint64_t simultaneous_upload_buffers = 0;  // SIMULTANEOUS_USE + uploads
  std::uint64_t secondaries_executed = 0;
  std::uint64_t secondaries_with_uploads = 0;
  std::uint64_t simultaneous_secondaries = 0;     // with uploads
  std::uint64_t submit_calls = 0;       // vkQueueSubmit
  std::uint64_t submit_batches = 0;
  std::uint64_t submit2_calls = 0;      // vkQueueSubmit2(KHR)
  std::uint64_t submit2_batches = 0;
  std::uint64_t upload_batches = 0;     // batches that carried uploads
  std::uint64_t multi_upload_calls = 0; // calls with two or more of them
  std::uint64_t source_bytes_by_type[kEtc2CountedMemoryTypes] = {};
  std::uint64_t source_bytes_unknown_type = 0;
  std::uint64_t non_host_visible_sources = 0;  // uploads read from GPU memory
  std::uint64_t copy_image_calls = 0;   // vkCmdCopyImage(2) on emulated images
  std::uint64_t blit_copies = 0;        // of them, turned into blits (upscale)
  std::uint64_t raw_copies_into = 0;    // from an uncompressed image into one
  std::uint64_t alias_views = 0;        // views with a non-ETC2/EAC format
  std::uint64_t aliased_binds = 0;      // memory shared with a live image
  std::uint64_t format_lists = 0;       // VkImageFormatListCreateInfo seen
  std::uint64_t uploads_after_dispatch = 0;
};

// Adds one submit's decoding.
void AddEtc2SubmitDecode(Etc2DecodeTotals* totals, std::uint64_t uploads,
                         std::uint64_t compressed_bytes,
                         std::uint64_t host_bytes, std::uint64_t decode_ns);

// Counts one destroyed image that had `record_calls` copies recorded.
void AddEtc2DestroyedImage(Etc2DecodeTotals* totals,
                           std::uint64_t record_calls);

// Raises the arena peaks to the current sizes.
void NoteEtc2ArenaBytes(Etc2DecodeTotals* totals, std::uint64_t bytes,
                        std::uint64_t over_cap_bytes);

// "images=12 (upscaled 3) uploads=480 in 37 submits compressed=21.3 MiB
// host=118.0 MiB (5.5x) decode=812 ms longest=41 ms"
std::string FormatEtc2DecodeTotals(const Etc2DecodeTotals& totals);

// The rest of the totals, appended after FormatEtc2DecodeTotals when
// MOCKTAIL_FRAME_STATS=1 or the GPU route ran, so that parsers of the line
// above keep working: " | gpu uploads=... | cpu_route ... | ...".
std::string FormatEtc2DecodeTotalsDetail(const Etc2DecodeTotals& totals);

// True when the window saw any ETC2 work worth a line.
bool Etc2WindowActive(const Etc2DecodeTotals& window);

// The two-second "[perf] etc2_window ..." line, without its prefix:
// "submits=S cpu_ms=X (max M) gpu_ms=Y record_calls=C ..."
std::string FormatEtc2Window(const Etc2DecodeTotals& window);

// True when host bytes growing from `before` to `after` pass 64 MiB or a
// doubling of it, so a session that never destroys its device still logs
// its totals a few times.
bool Etc2DecodeTotalsMilestone(std::uint64_t before, std::uint64_t after);

// Decoding works in one scratch pair sized to the whole submit, and a
// loading burst can put hundreds of MiB into a single submit. Scratch up to
// this size stays for the next submit; anything larger is given back.
inline constexpr std::size_t kEtc2ScratchKeepBytes = 32U * 1024U * 1024U;

// Frees `scratch` when it holds more than kEtc2ScratchKeepBytes.
void ReleaseOversizedEtc2Scratch(std::vector<std::uint8_t>* scratch);

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_ETC2_DECODE_TOTALS_H_
