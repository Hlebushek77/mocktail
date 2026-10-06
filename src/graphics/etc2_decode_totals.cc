#include "mocktail/graphics/etc2_decode_totals.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <new>

namespace mocktail::graphics {

namespace {

constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;
constexpr std::uint64_t kFirstMilestone = 64ULL * kMiB;

double Mebibytes(std::uint64_t bytes) {
  return static_cast<double>(bytes) / static_cast<double>(kMiB);
}

unsigned long long Milliseconds(std::uint64_t ns) {
  return static_cast<unsigned long long>((ns + 500000ULL) / 1000000ULL);
}

double FractionalMilliseconds(std::uint64_t ns) {
  return static_cast<double>(ns) / 1e6;
}

unsigned long long Count(std::uint64_t value) {
  return static_cast<unsigned long long>(value);
}

__attribute__((format(printf, 2, 3))) void Append(std::string* out,
                                                  const char* format, ...) {
  char buffer[256];
  va_list arguments;
  va_start(arguments, format);
  const int length = std::vsnprintf(buffer, sizeof(buffer), format, arguments);
  va_end(arguments);
  if (length > 0) {
    out->append(buffer, std::min<std::size_t>(static_cast<std::size_t>(length),
                                              sizeof(buffer) - 1));
  }
}

std::uint64_t Fallbacks(const Etc2DecodeTotals& totals) {
  std::uint64_t sum = 0;
  for (const std::uint64_t count : totals.cpu_fallbacks) {
    sum += count;
  }
  return sum;
}

// The step-0 fields both the window line and the totals detail end with.
void AppendUploadShape(const Etc2DecodeTotals& totals, std::string* out) {
  Append(out, "regions=%llu fam=", Count(totals.record_uploads));
  bool any = false;
  for (std::size_t family = 0; family < kEtc2CountedQueueFamilies; ++family) {
    if (totals.family_uploads[family] == 0) {
      continue;
    }
    Append(out, "%s%zu:%llu/%.1f", any ? "," : "", family,
           Count(totals.family_uploads[family]),
           Mebibytes(totals.family_bytes[family]));
    any = true;
  }
  if (!any) {
    out->append("-");
  }
  Append(out,
         " nocompute=%llu/%.1f unknownfam=%llu begin=ot:%llu,sim:%llu,"
         "other:%llu simcbs=%llu sec=%llu/%llu simsec=%llu",
         Count(totals.no_compute_uploads), Mebibytes(totals.no_compute_bytes),
         Count(totals.unknown_family_uploads), Count(totals.begin_one_time),
         Count(totals.begin_simultaneous), Count(totals.begin_other),
         Count(totals.simultaneous_upload_buffers),
         Count(totals.secondaries_executed),
         Count(totals.secondaries_with_uploads),
         Count(totals.simultaneous_secondaries));
  Append(out, " submit=%llu/%llu submit2=%llu/%llu upload_batches=%llu "
              "multibatch=%llu src=",
         Count(totals.submit_calls), Count(totals.submit_batches),
         Count(totals.submit2_calls), Count(totals.submit2_batches),
         Count(totals.upload_batches), Count(totals.multi_upload_calls));
  any = false;
  for (std::size_t type = 0; type < kEtc2CountedMemoryTypes; ++type) {
    if (totals.source_bytes_by_type[type] == 0) {
      continue;
    }
    Append(out, "%st%zu:%.1f", any ? "," : "", type,
           Mebibytes(totals.source_bytes_by_type[type]));
    any = true;
  }
  if (totals.source_bytes_unknown_type != 0) {
    Append(out, "%sunknown:%.1f", any ? "," : "",
           Mebibytes(totals.source_bytes_unknown_type));
    any = true;
  }
  if (!any) {
    out->append("-");
  }
  Append(out,
         " nonhost=%llu blits=%llu rawcopies=%llu aliasbinds=%llu "
         "formatlists=%llu afterdispatch=%llu copies=%llu/%llu/%llu/%llu/%llu",
         Count(totals.non_host_visible_sources), Count(totals.blit_copies),
         Count(totals.raw_copies_into), Count(totals.aliased_binds),
         Count(totals.format_lists), Count(totals.uploads_after_dispatch),
         Count(totals.image_copy_buckets[0]),
         Count(totals.image_copy_buckets[1]),
         Count(totals.image_copy_buckets[2]),
         Count(totals.image_copy_buckets[3]),
         Count(totals.image_copy_buckets[4]));
}

}  // namespace

const char* Etc2CpuReasonName(Etc2CpuReason reason) {
  switch (reason) {
    case Etc2CpuReason::kGpuDisabled:
      return "gpu_disabled";
    case Etc2CpuReason::kGpuNotReady:
      return "gpu_not_ready";
    case Etc2CpuReason::kGpuUnavailable:
      return "gpu_unavailable";
    case Etc2CpuReason::kUpscaled:
      return "upscaled";
    case Etc2CpuReason::kOverrides:
      return "overrides";
    case Etc2CpuReason::kNoComputeQueue:
      return "no_compute_queue";
    case Etc2CpuReason::kTooLarge:
      return "too_large";
    case Etc2CpuReason::kArenaFull:
      return "arena_full";
    case Etc2CpuReason::kCount:
      break;
  }
  return "unknown";
}

void AddEtc2SubmitDecode(Etc2DecodeTotals* totals, std::uint64_t uploads,
                         std::uint64_t compressed_bytes,
                         std::uint64_t host_bytes, std::uint64_t decode_ns) {
  if (totals == nullptr || uploads == 0) {
    return;
  }
  totals->uploads += uploads;
  ++totals->submits;
  totals->compressed_bytes += compressed_bytes;
  totals->host_bytes += host_bytes;
  totals->decode_ns += decode_ns;
  totals->longest_submit_ns = std::max(totals->longest_submit_ns, decode_ns);
}

std::string FormatEtc2DecodeTotals(const Etc2DecodeTotals& totals) {
  char ratio[32] = "";
  if (totals.compressed_bytes != 0) {
    std::snprintf(ratio, sizeof(ratio), " (%.1fx)",
                  static_cast<double>(totals.host_bytes) /
                      static_cast<double>(totals.compressed_bytes));
  }
  char line[256];
  std::snprintf(line, sizeof(line),
                "images=%llu (upscaled %llu) uploads=%llu in %llu submits "
                "compressed=%.1f MiB host=%.1f MiB%s decode=%llu ms "
                "longest=%llu ms",
                static_cast<unsigned long long>(totals.images),
                static_cast<unsigned long long>(totals.upscaled_images),
                static_cast<unsigned long long>(totals.uploads),
                static_cast<unsigned long long>(totals.submits),
                Mebibytes(totals.compressed_bytes),
                Mebibytes(totals.host_bytes), ratio,
                Milliseconds(totals.decode_ns),
                Milliseconds(totals.longest_submit_ns));
  return line;
}

void AddEtc2DestroyedImage(Etc2DecodeTotals* totals,
                           std::uint64_t record_calls) {
  if (totals == nullptr) {
    return;
  }
  ++totals->images_destroyed;
  const std::size_t bucket = record_calls == 0    ? 0
                             : record_calls == 1  ? 1
                             : record_calls <= 4  ? 2
                             : record_calls <= 16 ? 3
                                                  : 4;
  ++totals->image_copy_buckets[bucket];
}

void NoteEtc2ArenaBytes(Etc2DecodeTotals* totals, std::uint64_t bytes,
                        std::uint64_t over_cap_bytes) {
  if (totals == nullptr) {
    return;
  }
  totals->arena_peak_bytes = std::max(totals->arena_peak_bytes, bytes);
  totals->arena_over_cap_bytes =
      std::max(totals->arena_over_cap_bytes, over_cap_bytes);
}

std::string FormatEtc2DecodeTotalsDetail(const Etc2DecodeTotals& totals) {
  std::string out;
  Append(&out,
         " | gpu uploads=%llu submits=%llu compressed=%.1f MiB host=%.1f MiB "
         "record=%llu ms gpu=%llu ms presubmits=%llu waits=%llu",
         Count(totals.gpu_uploads), Count(totals.gpu_submits),
         Mebibytes(totals.gpu_compressed_bytes),
         Mebibytes(totals.gpu_host_bytes), Milliseconds(totals.gpu_record_ns),
         Milliseconds(totals.gpu_time_ns), Count(totals.gpu_presubmits),
         Count(totals.gpu_slot_waits));
  out.append(" | cpu_route");
  for (std::size_t reason = 0;
       reason < static_cast<std::size_t>(Etc2CpuReason::kCount); ++reason) {
    Append(&out, " %s=%llu",
           Etc2CpuReasonName(static_cast<Etc2CpuReason>(reason)),
           Count(totals.cpu_fallbacks[reason]));
  }
  Append(&out, " rescued=%llu dropped=%llu",
         Count(totals.rescued_uploads), Count(totals.dropped_uploads));
  Append(&out,
         " | arena peak=%.1f MiB over_cap=%.1f MiB staging_allocs=%llu "
         "staging_fallbacks=%llu",
         Mebibytes(totals.arena_peak_bytes),
         Mebibytes(totals.arena_over_cap_bytes), Count(totals.staging_allocs),
         Count(totals.staging_fallbacks));
  Append(&out,
         " | record calls=%llu time=%.1f ms release=%.1f ms destroyed=%llu "
         "bc_images=%llu copyimg=%llu aliasviews=%llu | ",
         Count(totals.record_calls), FractionalMilliseconds(totals.record_ns),
         FractionalMilliseconds(totals.release_ns),
         Count(totals.images_destroyed), Count(totals.bc_images),
         Count(totals.copy_image_calls), Count(totals.alias_views));
  AppendUploadShape(totals, &out);
  return out;
}

bool Etc2WindowActive(const Etc2DecodeTotals& window) {
  return window.uploads != 0 || window.gpu_uploads != 0 ||
         window.record_calls != 0 || window.images != 0 ||
         window.images_destroyed != 0 || window.copy_image_calls != 0 ||
         window.alias_views != 0 || window.aliased_binds != 0 ||
         window.raw_copies_into != 0 || window.format_lists != 0 ||
         window.rescued_uploads != 0 || window.dropped_uploads != 0;
}

std::string FormatEtc2Window(const Etc2DecodeTotals& window) {
  std::string out;
  Append(&out,
         "submits=%llu cpu_ms=%.1f (max %.1f) gpu_ms=%.1f record_calls=%llu "
         "record_ms=%.1f staging_allocs=%llu release_ms=%.1f",
         Count(window.submits + window.gpu_submits),
         FractionalMilliseconds(window.decode_ns),
         FractionalMilliseconds(window.longest_submit_ns),
         FractionalMilliseconds(window.gpu_time_ns), Count(window.record_calls),
         FractionalMilliseconds(window.record_ns), Count(window.staging_allocs),
         FractionalMilliseconds(window.release_ns));
  Append(&out,
         " uploads=%llu gpu=%llu fallback=%llu rescued=%llu "
         "arena_mib=%.1f(+%.1f) images=+%llu/-%llu copyimg=%llu "
         "aliasviews=%llu ",
         Count(window.uploads + window.gpu_uploads), Count(window.gpu_uploads),
         Count(Fallbacks(window)), Count(window.rescued_uploads),
         Mebibytes(window.arena_peak_bytes),
         Mebibytes(window.arena_over_cap_bytes), Count(window.images),
         Count(window.images_destroyed), Count(window.copy_image_calls),
         Count(window.alias_views));
  AppendUploadShape(window, &out);
  return out;
}

bool Etc2DecodeTotalsMilestone(std::uint64_t before, std::uint64_t after) {
  for (std::uint64_t milestone = kFirstMilestone;
       milestone != 0 && milestone <= after; milestone <<= 1) {
    if (before < milestone) {
      return true;
    }
  }
  return false;
}

std::string FormatEtc2StallSuffix(
    unsigned threads, bool fast_path,
    const std::uint64_t (&source_bytes_by_type)[kEtc2CountedMemoryTypes],
    std::uint64_t unknown_type_bytes) {
  std::string out;
  Append(&out, " [threads %u, path %s, src", threads,
         fast_path ? "fast" : "legacy");
  bool any = false;
  for (std::size_t type = 0; type < kEtc2CountedMemoryTypes; ++type) {
    if (source_bytes_by_type[type] != 0) {
      Append(&out, " t%zu %.1f", type, Mebibytes(source_bytes_by_type[type]));
      any = true;
    }
  }
  if (unknown_type_bytes != 0) {
    Append(&out, " unknown %.1f", Mebibytes(unknown_type_bytes));
    any = true;
  }
  out.append(any ? " MiB]" : " - MiB]");
  return out;
}

std::uint8_t* Etc2Scratch::Reserve(std::size_t bytes,
                                   std::chrono::steady_clock::time_point now) {
  last_used_ = now;
  if (bytes <= capacity_) {
    return data_.get();
  }
  // Grow by at least half, so a burst of slowly growing submits does not
  // allocate on each one; default-initialised, so nothing is zero-filled.
  const std::size_t grown = std::max(bytes, capacity_ + capacity_ / 2);
  data_.reset();
  capacity_ = 0;
  std::uint8_t* fresh = new (std::nothrow) std::uint8_t[grown];
  if (fresh == nullptr) {
    fresh = new (std::nothrow) std::uint8_t[bytes];
    if (fresh == nullptr) {
      return nullptr;
    }
    capacity_ = bytes;
  } else {
    capacity_ = grown;
  }
  data_.reset(fresh);
  return data_.get();
}

bool Etc2Scratch::ReleaseIfIdle(std::chrono::steady_clock::time_point now) {
  if (capacity_ <= kEtc2ScratchKeepBytes ||
      now - last_used_ < kEtc2ScratchIdleRelease) {
    return false;
  }
  Release();
  return true;
}

void Etc2Scratch::Release() {
  data_.reset();
  capacity_ = 0;
}

void ReleaseOversizedEtc2Scratch(std::vector<std::uint8_t>* scratch) {
  if (scratch != nullptr &&
      scratch->capacity() > kEtc2LegacyScratchKeepBytes) {
    std::vector<std::uint8_t>().swap(*scratch);
  }
}

}  // namespace mocktail::graphics
