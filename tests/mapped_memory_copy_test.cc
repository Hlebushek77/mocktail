#include "mocktail/graphics/mapped_memory_copy.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace mocktail::graphics {
namespace {

constexpr MappedMemoryCopyKernel kKernels[] = {
    MappedMemoryCopyKernel::kMemcpy,
    MappedMemoryCopyKernel::kSse41,
    MappedMemoryCopyKernel::kAvx2,
};

std::vector<std::uint8_t> Pattern(std::size_t size, std::uint32_t seed) {
  std::vector<std::uint8_t> bytes(size);
  std::uint32_t state = seed * 2654435761U + 1;
  for (std::uint8_t& byte : bytes) {
    state = state * 1664525U + 1013904223U;
    byte = static_cast<std::uint8_t>(state >> 24);
  }
  return bytes;
}

// Each kernel this CPU has, forced in turn (R14: the AVX2 one must never be
// the only one tested), then the automatic choice again.
class MappedMemoryCopyTest
    : public ::testing::TestWithParam<MappedMemoryCopyKernel> {
 protected:
  void SetUp() override {
    if (!ForceMappedMemoryCopyKernelForTesting(GetParam())) {
      GTEST_SKIP() << "this CPU has no "
                   << MappedMemoryCopyKernelName(GetParam());
    }
    ASSERT_EQ(ActiveMappedMemoryCopyKernel(), GetParam());
  }
  void TearDown() override { ResetMappedMemoryCopyKernelForTesting(); }
};

TEST_P(MappedMemoryCopyTest, MatchesMemcpyForEveryAlignmentAndLength) {
  const std::vector<std::uint8_t> source = Pattern(70000, 1);
  std::vector<std::uint8_t> got(70100);
  std::vector<std::uint8_t> want(70100);
  for (std::size_t source_offset = 0; source_offset < 64; ++source_offset) {
    for (std::size_t destination_offset : {0U, 3U, 32U}) {
      for (std::size_t length : {0U, 1U, 15U, 31U, 63U, 64U, 65U, 4095U, 4096U,
                                 4097U, 65535U}) {
        std::fill(got.begin(), got.end(), 0x5A);
        std::fill(want.begin(), want.end(), 0x5A);
        CopyFromMappedMemory(got.data() + destination_offset,
                             source.data() + source_offset, length);
        std::memcpy(want.data() + destination_offset,
                    source.data() + source_offset, length);
        ASSERT_EQ(got, want) << "source offset " << source_offset
                             << ", destination offset " << destination_offset
                             << ", length " << length;
      }
    }
  }
}

TEST_P(MappedMemoryCopyTest, PacksStridedRows) {
  struct Case {
    std::size_t row_bytes;
    std::size_t rows;
    std::size_t stride;
  };
  // Dense spans stream whole (stride up to twice the row), sparse ones row
  // by row; rows of 8 and 16 bytes are single ETC2 block rows.
  const Case cases[] = {
      {8, 1, 64},      {8, 7, 8},       {8, 9, 16},      {16, 33, 24},
      {64, 64, 96},    {128, 40, 4096}, {200, 31, 384},  {4096, 5, 4160},
      {4097, 3, 8000}, {256, 300, 512}, {256, 300, 520}, {1000, 17, 1001},
  };
  for (const Case& test : cases) {
    for (std::size_t source_offset : {0U, 1U, 8U, 31U, 33U}) {
      const std::size_t span = (test.rows - 1) * test.stride + test.row_bytes;
      const std::vector<std::uint8_t> source =
          Pattern(source_offset + span + 64, static_cast<std::uint32_t>(span));
      std::vector<std::uint8_t> got(test.rows * test.row_bytes + 32, 0x5A);
      std::vector<std::uint8_t> want(got.size(), 0x5A);
      CopyStridedFromMappedMemory(got.data() + 16,
                                  source.data() + source_offset,
                                  test.row_bytes, test.rows, test.stride);
      for (std::size_t row = 0; row < test.rows; ++row) {
        std::memcpy(want.data() + 16 + row * test.row_bytes,
                    source.data() + source_offset + row * test.stride,
                    test.row_bytes);
      }
      ASSERT_EQ(got, want) << "rows of " << test.row_bytes << " x "
                           << test.rows << " every " << test.stride
                           << ", source offset " << source_offset;
    }
  }
  // Nothing to copy touches nothing.
  std::uint8_t untouched = 0x5A;
  CopyStridedFromMappedMemory(&untouched, nullptr, 0, 5, 16);
  CopyStridedFromMappedMemory(&untouched, nullptr, 16, 0, 16);
  EXPECT_EQ(untouched, 0x5A);
}

INSTANTIATE_TEST_SUITE_P(
    Kernels, MappedMemoryCopyTest, ::testing::ValuesIn(kKernels),
    [](const ::testing::TestParamInfo<MappedMemoryCopyKernel>& info) {
      switch (info.param) {
        case MappedMemoryCopyKernel::kMemcpy:
          return std::string("Memcpy");
        case MappedMemoryCopyKernel::kSse41:
          return std::string("Sse41");
        case MappedMemoryCopyKernel::kAvx2:
          return std::string("Avx2");
      }
      return std::string("Unknown");
    });

TEST(MappedMemoryCopyKernelTest, PicksTheBestKernelThisCpuHas) {
  ResetMappedMemoryCopyKernelForTesting();
  const MappedMemoryCopyKernel active = ActiveMappedMemoryCopyKernel();
  EXPECT_TRUE(MappedMemoryCopyKernelSupported(active));
  EXPECT_TRUE(MappedMemoryCopyKernelSupported(MappedMemoryCopyKernel::kMemcpy));
  if (MappedMemoryCopyKernelSupported(MappedMemoryCopyKernel::kAvx2)) {
    EXPECT_EQ(active, MappedMemoryCopyKernel::kAvx2);
  } else if (MappedMemoryCopyKernelSupported(MappedMemoryCopyKernel::kSse41)) {
    EXPECT_EQ(active, MappedMemoryCopyKernel::kSse41);
  } else {
    EXPECT_EQ(active, MappedMemoryCopyKernel::kMemcpy);
  }
  EXPECT_STREQ(MappedMemoryCopyKernelName(MappedMemoryCopyKernel::kAvx2),
               "avx2");
  EXPECT_STREQ(MappedMemoryCopyKernelName(MappedMemoryCopyKernel::kSse41),
               "sse4.1");
  EXPECT_STREQ(MappedMemoryCopyKernelName(MappedMemoryCopyKernel::kMemcpy),
               "memcpy");
}

}  // namespace
}  // namespace mocktail::graphics
