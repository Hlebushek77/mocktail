#include "mocktail/graphics/etc2_decode_totals.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>

namespace mocktail::graphics {
namespace {

constexpr std::uint64_t kMiB = 1024ULL * 1024ULL;

TEST(Etc2DecodeTotalsTest, AddsSubmitsAndKeepsTheLongest) {
  Etc2DecodeTotals totals;
  totals.images = 12;
  totals.upscaled_images = 3;
  AddEtc2SubmitDecode(&totals, 4, 2 * kMiB, 8 * kMiB, 3000000);
  AddEtc2SubmitDecode(&totals, 1, kMiB, 8 * kMiB, 41400000);
  AddEtc2SubmitDecode(&totals, 0, kMiB, kMiB, 99000000);  // nothing decoded
  AddEtc2SubmitDecode(nullptr, 1, 1, 1, 1);

  EXPECT_EQ(totals.uploads, 5U);
  EXPECT_EQ(totals.submits, 2U);
  EXPECT_EQ(totals.compressed_bytes, 3 * kMiB);
  EXPECT_EQ(totals.host_bytes, 16 * kMiB);
  EXPECT_EQ(totals.decode_ns, 44400000U);
  EXPECT_EQ(totals.longest_submit_ns, 41400000U);
  EXPECT_EQ(FormatEtc2DecodeTotals(totals),
            "images=12 (upscaled 3) uploads=5 in 2 submits "
            "compressed=3.0 MiB host=16.0 MiB (5.3x) decode=44 ms "
            "longest=41 ms");
}

TEST(Etc2DecodeTotalsTest, FormatsAnEmptySessionWithoutARatio) {
  Etc2DecodeTotals totals;
  totals.images = 4;
  totals.upscaled_images = 4;
  EXPECT_EQ(FormatEtc2DecodeTotals(totals),
            "images=4 (upscaled 4) uploads=0 in 0 submits "
            "compressed=0.0 MiB host=0.0 MiB decode=0 ms longest=0 ms");
}

TEST(Etc2DecodeTotalsTest, ReportsAtSixtyFourMebibytesAndEachDoubling) {
  EXPECT_FALSE(Etc2DecodeTotalsMilestone(0, 63 * kMiB));
  EXPECT_TRUE(Etc2DecodeTotalsMilestone(63 * kMiB, 64 * kMiB));
  EXPECT_FALSE(Etc2DecodeTotalsMilestone(64 * kMiB, 127 * kMiB));
  EXPECT_TRUE(Etc2DecodeTotalsMilestone(100 * kMiB, 300 * kMiB));
  EXPECT_FALSE(Etc2DecodeTotalsMilestone(300 * kMiB, 500 * kMiB));
  EXPECT_TRUE(Etc2DecodeTotalsMilestone(500 * kMiB, 512 * kMiB));
  EXPECT_TRUE(Etc2DecodeTotalsMilestone(0, UINT64_MAX));
  EXPECT_FALSE(Etc2DecodeTotalsMilestone(UINT64_MAX - 1, UINT64_MAX));
}

// The CPU route keeps its scratch through a burst and gives back anything
// above 64 MiB only after 5 s without a submit that used it.
TEST(Etc2DecodeTotalsTest, ScratchGrowsOnceAndGoesAfterFiveIdleSeconds) {
  using std::chrono::milliseconds;
  using std::chrono::seconds;
  const std::chrono::steady_clock::time_point start{seconds(1000)};
  Etc2Scratch scratch;
  EXPECT_EQ(scratch.capacity(), 0U);
  std::uint8_t* first = scratch.Reserve(1000, start);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(scratch.capacity(), 1000U);
  first[999] = 7;  // writable
  EXPECT_EQ(scratch.Reserve(10, start), first) << "no shrinking";
  // Growth is by half at least, so 1001 bytes get 1500.
  EXPECT_NE(scratch.Reserve(1001, start), nullptr);
  EXPECT_EQ(scratch.capacity(), 1500U);
  // Small scratch stays for good.
  EXPECT_FALSE(scratch.ReleaseIfIdle(start + seconds(3600)));
  EXPECT_EQ(scratch.capacity(), 1500U);

  const std::size_t burst = kEtc2ScratchKeepBytes + 4096;
  ASSERT_NE(scratch.Reserve(burst, start), nullptr);
  EXPECT_EQ(scratch.capacity(), burst);
  // Every submit of the burst marks it used again.
  scratch.MarkUsed(start + seconds(3));
  EXPECT_FALSE(scratch.ReleaseIfIdle(start + seconds(7)));
  EXPECT_FALSE(scratch.ReleaseIfIdle(start + seconds(8) - milliseconds(1)));
  EXPECT_EQ(scratch.capacity(), burst);
  EXPECT_TRUE(scratch.ReleaseIfIdle(start + seconds(8)));
  EXPECT_EQ(scratch.capacity(), 0U);
  EXPECT_FALSE(scratch.ReleaseIfIdle(start + seconds(60)));
  ASSERT_NE(scratch.Reserve(64, start + seconds(61)), nullptr);
  scratch.Release();
  EXPECT_EQ(scratch.capacity(), 0U);
}

TEST(Etc2DecodeTotalsTest, FormatsTheStallLineSuffix) {
  std::uint64_t by_type[kEtc2CountedMemoryTypes] = {};
  EXPECT_EQ(FormatEtc2StallSuffix(1, by_type, 0), " [threads 1, src - MiB]");
  by_type[3] = 51065651;  // 48.7 MiB
  by_type[4] = 314573;  // 0.3 MiB
  EXPECT_EQ(FormatEtc2StallSuffix(8, by_type, 0),
            " [threads 8, src t3 48.7 t4 0.3 MiB]");
  by_type[4] = 0;
  EXPECT_EQ(FormatEtc2StallSuffix(4, by_type, 3 * kMiB),
            " [threads 4, src t3 48.7 unknown 3.0 MiB]");
}

TEST(Etc2DecodeTotalsTest, NamesEveryCpuRouteReason) {
  EXPECT_STREQ(Etc2CpuReasonName(Etc2CpuReason::kGpuDisabled), "gpu_disabled");
  EXPECT_STREQ(Etc2CpuReasonName(Etc2CpuReason::kNoComputeQueue),
               "no_compute_queue");
  EXPECT_STREQ(Etc2CpuReasonName(Etc2CpuReason::kArenaFull), "arena_full");
  EXPECT_STREQ(Etc2CpuReasonName(Etc2CpuReason::kCount), "unknown");
  for (std::size_t first = 0;
       first < static_cast<std::size_t>(Etc2CpuReason::kCount); ++first) {
    for (std::size_t second = first + 1;
         second < static_cast<std::size_t>(Etc2CpuReason::kCount); ++second) {
      EXPECT_STRNE(Etc2CpuReasonName(static_cast<Etc2CpuReason>(first)),
                   Etc2CpuReasonName(static_cast<Etc2CpuReason>(second)));
    }
  }
}

TEST(Etc2DecodeTotalsTest, DetailFollowsThePrefixWithEveryCounter) {
  Etc2DecodeTotals totals;
  totals.gpu_uploads = 7;
  totals.gpu_submits = 2;
  totals.gpu_compressed_bytes = 3 * kMiB;
  totals.gpu_host_bytes = 24 * kMiB;
  totals.gpu_record_ns = 1400000;
  totals.gpu_time_ns = 9600000;
  totals.cpu_fallbacks[static_cast<std::size_t>(Etc2CpuReason::kUpscaled)] = 5;
  totals.dropped_uploads = 1;
  totals.arena_peak_bytes = 96 * kMiB;
  totals.staging_allocs = 12;
  totals.record_calls = 12;
  totals.record_uploads = 30;
  totals.record_ns = 6250000;
  totals.release_ns = 4500000;
  totals.images_destroyed = 3;
  totals.copy_image_calls = 4;
  totals.family_uploads[0] = 28;
  totals.family_bytes[0] = 2 * kMiB;
  totals.family_uploads[2] = 2;
  totals.family_bytes[2] = kMiB / 2;
  totals.begin_one_time = 40;
  totals.begin_other = 9;
  totals.submit_calls = 20;
  totals.submit_batches = 21;
  totals.upload_batches = 6;
  totals.source_bytes_by_type[3] = 5 * kMiB / 2;
  totals.image_copy_buckets[1] = 2;
  totals.image_copy_buckets[2] = 1;
  EXPECT_EQ(
      FormatEtc2DecodeTotalsDetail(totals),
      " | gpu uploads=7 submits=2 compressed=3.0 MiB host=24.0 MiB record=1 ms "
      "gpu=10 ms presubmits=0 waits=0 | cpu_route gpu_disabled=0 "
      "gpu_not_ready=0 gpu_unavailable=0 upscaled=5 overrides=0 "
      "no_compute_queue=0 too_large=0 arena_full=0 rescued=0 dropped=1 | "
      "arena peak=96.0 MiB over_cap=0.0 MiB staging_allocs=12 "
      "staging_fallbacks=0 | record calls=12 time=6.2 ms release=4.5 ms "
      "destroyed=3 bc_images=0 bc_fallback=0 bc_mixed_copies=0 copyimg=4 "
      "aliasviews=0 | "
      "regions=30 fam=0:28/2.0,2:2/0.5 nocompute=0/0.0 unknownfam=0 "
      "begin=ot:40,sim:0,other:9 simcbs=0 sec=0/0 simsec=0 submit=20/21 "
      "submit2=0/0 upload_batches=6 multibatch=0 src=t3:2.5 nonhost=0 "
      "blits=0 rawcopies=0 aliasbinds=0 formatlists=0 afterdispatch=0 "
      "copies=0/2/1/0/0");
}

TEST(Etc2DecodeTotalsTest, WindowLineStartsWithTheDocumentedFields) {
  Etc2DecodeTotals window;
  AddEtc2SubmitDecode(&window, 40, 4 * kMiB, 30 * kMiB, 25300000);
  AddEtc2SubmitDecode(&window, 2, kMiB, kMiB, 1200000);
  window.record_calls = 18;
  window.record_uploads = 42;
  window.record_ns = 9050000;
  window.staging_allocs = 18;
  window.release_ns = 7000000;
  window.cpu_fallbacks[static_cast<std::size_t>(
      Etc2CpuReason::kGpuUnavailable)] = 42;
  window.images = 5;
  window.images_destroyed = 1;
  window.source_bytes_unknown_type = kMiB;
  NoteEtc2ArenaBytes(&window, 64 * kMiB, 0);
  NoteEtc2ArenaBytes(&window, 32 * kMiB, 8 * kMiB);
  EXPECT_EQ(FormatEtc2Window(window),
            "submits=2 cpu_ms=26.5 (max 25.3) gpu_ms=0.0 record_calls=18 "
            "record_ms=9.1 staging_allocs=18 release_ms=7.0 uploads=42 gpu=0 "
            "fallback=42 rescued=0 arena_mib=64.0(+8.0) images=+5/-1 "
            "copyimg=0 aliasviews=0 regions=42 fam=- nocompute=0/0.0 "
            "unknownfam=0 begin=ot:0,sim:0,other:0 simcbs=0 sec=0/0 simsec=0 "
            "submit=0/0 submit2=0/0 upload_batches=0 multibatch=0 "
            "src=unknown:1.0 nonhost=0 blits=0 rawcopies=0 aliasbinds=0 "
            "formatlists=0 afterdispatch=0 copies=0/0/0/0/0");
}

TEST(Etc2DecodeTotalsTest, WindowIsActiveOnlyWithEtc2Work) {
  Etc2DecodeTotals window;
  window.begin_one_time = 120;
  window.submit_calls = 60;
  window.submit_batches = 60;
  EXPECT_FALSE(Etc2WindowActive(window)) << "every frame begins and submits";
  window.record_calls = 1;
  EXPECT_TRUE(Etc2WindowActive(window));
  Etc2DecodeTotals destroyed;
  destroyed.images_destroyed = 1;
  EXPECT_TRUE(Etc2WindowActive(destroyed));
}

TEST(Etc2DecodeTotalsTest, BucketsDestroyedImagesByTheirCopies) {
  Etc2DecodeTotals totals;
  for (const std::uint64_t copies : {0, 1, 1, 2, 4, 5, 16, 17, 900}) {
    AddEtc2DestroyedImage(&totals, copies);
  }
  AddEtc2DestroyedImage(nullptr, 1);
  EXPECT_EQ(totals.images_destroyed, 9U);
  EXPECT_EQ(totals.image_copy_buckets[0], 1U);
  EXPECT_EQ(totals.image_copy_buckets[1], 2U);
  EXPECT_EQ(totals.image_copy_buckets[2], 2U);
  EXPECT_EQ(totals.image_copy_buckets[3], 2U);
  EXPECT_EQ(totals.image_copy_buckets[4], 2U);
}

}  // namespace
}  // namespace mocktail::graphics
