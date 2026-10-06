#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "mocktail/graphics/texture_override.h"
#include "support/resample_reference.h"

namespace mocktail::graphics {
namespace {

std::vector<std::uint8_t> RandomRgba(std::mt19937_64* rng, std::uint32_t width,
                                     std::uint32_t height) {
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height *
                                   4);
  for (std::size_t index = 0; index < pixels.size(); index += 8) {
    const std::uint64_t value = (*rng)();
    for (std::size_t byte = 0; byte < 8 && index + byte < pixels.size();
         ++byte) {
      pixels[index + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
    }
  }
  return pixels;
}

// Runs both implementations into buffers that start out different, so a
// texel one of them skips cannot compare equal by accident.
void ExpectMatchesReference(const std::vector<std::uint8_t>& source,
                            std::uint32_t source_width,
                            std::uint32_t source_height, std::uint32_t width,
                            std::uint32_t height) {
  const std::size_t size = static_cast<std::size_t>(width) * height * 4;
  std::vector<std::uint8_t> actual(size, 0x11);
  std::vector<std::uint8_t> expected(size, 0xEE);
  ResampleRgba(source.data(), source_width, source_height, width, height,
               actual.data());
  reference::ResampleRgba(source.data(), source_width, source_height, width,
                          height, expected.data());
  ASSERT_EQ(actual, expected) << source_width << "x" << source_height << " -> "
                              << width << "x" << height;
}

// Integer up- and downscales (x1..x8), arbitrary ratios and 1-texel edges,
// 20000 cases up to 280x280, as the small-texture upscale and override PNGs
// use them.
TEST(ResampleRgbaTest, MatchesReferenceOnRandomCases) {
  std::mt19937_64 rng(0x5EED0101);
  for (int index = 0; index < 20000; ++index) {
    std::uint32_t source_width = 1 + static_cast<std::uint32_t>(rng() % 70);
    std::uint32_t source_height = 1 + static_cast<std::uint32_t>(rng() % 70);
    std::uint32_t width = 1 + static_cast<std::uint32_t>(rng() % 280);
    std::uint32_t height = 1 + static_cast<std::uint32_t>(rng() % 280);
    const std::uint32_t factor = 1 + static_cast<std::uint32_t>(rng() % 8);
    switch (index % 5) {
      case 0:  // integer upscale, the small-texture path
        source_width = 1 + static_cast<std::uint32_t>(rng() % (280 / factor));
        source_height = 1 + static_cast<std::uint32_t>(rng() % (280 / factor));
        width = source_width * factor;
        height = source_height * factor;
        break;
      case 1:  // integer downscale
        width = std::max(1U, source_width / factor);
        height = std::max(1U, source_height / factor);
        break;
      case 2:  // a 1-texel row or column on either side
        if (rng() % 2 != 0) {
          source_width = 1;
          height = 1;
        } else {
          source_height = 1;
          width = 1;
        }
        break;
      default:  // non-integer ratios both ways
        break;
    }
    ExpectMatchesReference(RandomRgba(&rng, source_width, source_height),
                           source_width, source_height, width, height);
  }
}

TEST(ResampleRgbaTest, UpscalesByRepeatingTexels) {
  const std::vector<std::uint8_t> source = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<std::uint8_t> out(4 * 2 * 4);
  ResampleRgba(source.data(), 2, 1, 4, 2, out.data());
  const std::vector<std::uint8_t> expected = {1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7,
                                              8, 5, 6, 7, 8, 1, 2, 3, 4, 1, 2,
                                              3, 4, 5, 6, 7, 8, 5, 6, 7, 8};
  EXPECT_EQ(out, expected);
}

TEST(ResampleRgbaTest, DownscalesWithARoundedBoxAverage) {
  // (10 + 11 + 20 + 22) / 4 = 15.75 -> 16 per channel with offsets.
  const std::vector<std::uint8_t> source = {10, 0, 255, 1, 11, 0, 255, 2,
                                            20, 0, 254, 3, 22, 1, 254, 4};
  std::vector<std::uint8_t> out(4);
  ResampleRgba(source.data(), 2, 2, 1, 1, out.data());
  EXPECT_EQ(out, (std::vector<std::uint8_t>{16, 0, 255, 3}));
}

// A 4096x4096 box sums to 4096^2 * 255 + rounding, just under 2^32; any
// larger PNG (4096x8192) would overflow a 32-bit accumulator.
TEST(ResampleRgbaTest, LargeBoxesDoNotOverflow) {
  for (const std::uint32_t source_height : {4096U, 8192U}) {
    const std::vector<std::uint8_t> source(
        static_cast<std::size_t>(4096) * source_height * 4, 255);
    std::vector<std::uint8_t> out(4, 0);
    ResampleRgba(source.data(), 4096, source_height, 1, 1, out.data());
    EXPECT_EQ(out, (std::vector<std::uint8_t>{255, 255, 255, 255}))
        << "4096x" << source_height;
    ExpectMatchesReference(source, 4096, source_height, 1, 1);
  }
}

TEST(ResampleRgbaTest, IgnoresEmptyAndNullArguments) {
  std::vector<std::uint8_t> source(16, 7);
  std::vector<std::uint8_t> out(16, 0x42);
  ResampleRgba(nullptr, 2, 2, 2, 2, out.data());
  ResampleRgba(source.data(), 0, 2, 2, 2, out.data());
  ResampleRgba(source.data(), 2, 0, 2, 2, out.data());
  ResampleRgba(source.data(), 2, 2, 0, 2, out.data());
  ResampleRgba(source.data(), 2, 2, 2, 0, out.data());
  ResampleRgba(source.data(), 2, 2, 2, 2, nullptr);
  EXPECT_EQ(out, std::vector<std::uint8_t>(16, 0x42));
}

// The RgbaImage overload resamples the whole image and refuses one whose
// pixels are shorter than width * height * 4.
TEST(ResampleRgbaTest, ImageOverloadChecksPixelCount) {
  std::mt19937_64 rng(0x5EED0102);
  RgbaImage image;
  image.width = 13;
  image.height = 7;
  image.pixels = RandomRgba(&rng, 13, 7);
  std::vector<std::uint8_t> actual(52 * 28 * 4, 0x11);
  std::vector<std::uint8_t> expected(52 * 28 * 4, 0xEE);
  ResampleRgba(image, 52, 28, actual.data());
  reference::ResampleRgba(image.pixels.data(), 13, 7, 52, 28, expected.data());
  EXPECT_EQ(actual, expected);

  image.pixels.pop_back();
  std::vector<std::uint8_t> untouched(52 * 28 * 4, 0x11);
  ResampleRgba(image, 52, 28, untouched.data());
  EXPECT_EQ(untouched, std::vector<std::uint8_t>(52 * 28 * 4, 0x11));
}

}  // namespace
}  // namespace mocktail::graphics
