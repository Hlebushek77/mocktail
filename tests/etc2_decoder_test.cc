#include "mocktail/graphics/etc2_decoder.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "support/etc2_reference_decoder.h"

// The golden blocks below were derived by hand from the Khronos Data Format
// specification (ETC2 individual, differential, T, H and planar modes,
// punch-through alpha, EAC 8-bit alpha and 11-bit R/RG) and checked against
// the plus.1 decoder; the randomised tests compare the decoder with the frozen
// copy in support/etc2_reference_decoder.cc.

namespace mocktail::graphics {
namespace {

constexpr EtcFormat kAllFormats[] = {
    EtcFormat::kEtc2Rgb8,      EtcFormat::kEtc2Rgb8A1,   EtcFormat::kEtc2Rgba8,
    EtcFormat::kEacR11,        EtcFormat::kEacR11Signed, EtcFormat::kEacRg11,
    EtcFormat::kEacRg11Signed,
};

using Rgba = std::array<int, 4>;
using Block8 = std::array<std::uint8_t, 8>;

// Colour index bits for a 4x4 block where every texel in row y uses pixel
// index y: index (msb << 1) | lsb, texel number x * 4 + y.
constexpr std::uint8_t kRowIndexMsb = 0xCC;
constexpr std::uint8_t kRowIndexLsb = 0xAA;

std::vector<std::uint8_t> DecodeBlock(EtcFormat format,
                                      const std::vector<std::uint8_t>& block) {
  std::vector<std::uint8_t> out(16 * EtcDecodedTexelBytes(format), 0x5A);
  EXPECT_TRUE(DecodeEtcImage(format, block.data(), block.size(), 4, 4,
                             out.data(), out.size()));
  return out;
}

Rgba TexelAt(const std::vector<std::uint8_t>& rgba, int x, int y) {
  const std::size_t offset = static_cast<std::size_t>(y * 4 + x) * 4;
  return {rgba[offset], rgba[offset + 1], rgba[offset + 2], rgba[offset + 3]};
}

std::uint16_t R16At(const std::vector<std::uint8_t>& texels, int x, int y,
                    int channel, int channels) {
  const std::size_t offset =
      (static_cast<std::size_t>(y * 4 + x) * channels + channel) * 2;
  return static_cast<std::uint16_t>(texels[offset] | (texels[offset + 1] << 8));
}

// Every row y of the block is `rows[y]`, for both halves of the block when
// `right` is empty.
void ExpectRows(const std::vector<std::uint8_t>& rgba,
                const std::array<Rgba, 4>& left,
                const std::array<Rgba, 4>& right) {
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      SCOPED_TRACE("texel " + std::to_string(x) + "," + std::to_string(y));
      EXPECT_EQ(TexelAt(rgba, x, y), x < 2 ? left[y] : right[y]);
    }
  }
}

std::vector<std::uint8_t> ColourBlock(std::uint8_t b0, std::uint8_t b1,
                                      std::uint8_t b2, std::uint8_t b3) {
  return {b0,           b1,           b2,           b3,
          kRowIndexMsb, kRowIndexMsb, kRowIndexLsb, kRowIndexLsb};
}

// An EAC block whose texel number i = x * 4 + y uses modifier index i % 8.
Block8 EacBlock(std::uint8_t base, std::uint8_t multiplier_and_table) {
  std::uint64_t bits = 0;
  for (int texel = 0; texel < 16; ++texel) {
    bits |= static_cast<std::uint64_t>(texel % 8) << (45 - texel * 3);
  }
  Block8 block{base, multiplier_and_table};
  for (int byte = 0; byte < 6; ++byte) {
    block[2 + byte] = static_cast<std::uint8_t>(bits >> (40 - byte * 8));
  }
  return block;
}

int EacIndexAt(int x, int y) { return (x * 4 + y) % 8; }

std::vector<std::uint8_t> Concat(const Block8& a, const Block8& b) {
  std::vector<std::uint8_t> out(a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  return out;
}

std::vector<std::uint8_t> Concat(const Block8& a,
                                 const std::vector<std::uint8_t>& b) {
  std::vector<std::uint8_t> out(a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  return out;
}

// Individual mode, flip 0: base colours 4-bit expanded, tables 0 and 7,
// every texel index 0 (+a); red and blue clamp.
TEST(Etc2DecoderTest, IndividualModeSplitsColumns) {
  const std::vector<std::uint8_t> block = {0xF0, 0x80, 0x0F, 0x1C,
                                           0x00, 0x00, 0x00, 0x00};
  const auto rgba = DecodeBlock(EtcFormat::kEtc2Rgb8, block);
  const Rgba left = {255, 138, 2, 255};
  const Rgba right = {47, 47, 255, 255};
  ExpectRows(rgba, {left, left, left, left}, {right, right, right, right});
}

// Individual mode, flip 1: the halves are the top and bottom rows, and the
// pixel index walks +a, +b, -a, -b down the block (column-major texels).
TEST(Etc2DecoderTest, IndividualModeFlippedUsesEveryIndex) {
  const auto rgba =
      DecodeBlock(EtcFormat::kEtc2Rgb8, ColourBlock(0x5A, 0x3C, 0x96, 0x55));
  const std::array<Rgba, 4> rows = {
      Rgba{94, 60, 162, 255}, Rgba{114, 80, 182, 255}, Rgba{146, 180, 78, 255},
      Rgba{90, 124, 22, 255}};
  ExpectRows(rgba, rows, rows);
}

// Differential mode: 5-bit base plus a signed 3-bit delta, tables 1 and 6.
TEST(Etc2DecoderTest, DifferentialMode) {
  const auto rgba =
      DecodeBlock(EtcFormat::kEtc2Rgb8, ColourBlock(0xA3, 0x54, 0xFF, 0x3A));
  ExpectRows(rgba,
             {Rgba{170, 87, 255, 255}, Rgba{182, 99, 255, 255},
              Rgba{160, 77, 250, 255}, Rgba{148, 65, 238, 255}},
             {Rgba{222, 82, 255, 255}, Rgba{255, 155, 255, 255},
              Rgba{156, 16, 214, 255}, Rgba{83, 0, 141, 255}});
}

// T mode (red overflows): paint colours c1, c2 + d, c2, c2 - d, d = 32.
TEST(Etc2DecoderTest, TMode) {
  const auto rgba =
      DecodeBlock(EtcFormat::kEtc2Rgb8, ColourBlock(0xF3, 0x4E, 0x7C, 0x3B));
  const std::array<Rgba, 4> rows = {
      Rgba{187, 68, 238, 255}, Rgba{151, 236, 83, 255}, Rgba{119, 204, 51, 255},
      Rgba{87, 172, 19, 255}};
  ExpectRows(rgba, rows, rows);
}

// H mode (green overflows): c1 +- d, c2 +- d; the distance LSB comes from
// c1 >= c2.
TEST(Etc2DecoderTest, HMode) {
  const auto rgba =
      DecodeBlock(EtcFormat::kEtc2Rgb8, ColourBlock(0x52, 0x15, 0x96, 0x4E));
  const std::array<Rgba, 4> rows = {
      Rgba{202, 117, 83, 255}, Rgba{138, 53, 19, 255}, Rgba{66, 236, 185, 255},
      Rgba{2, 172, 121, 255}};
  ExpectRows(rgba, rows, rows);
}

// Planar mode (blue overflows). The block holds RO=32 GO=32 BO=22, RH=63
// GH=0 BH=63, RV=0 GV=127 BV=32 (6/7/6 bits); the expectation applies the
// specification's formula to those values.
const std::vector<std::uint8_t> kPlanarBlock = {0x40, 0x40, 0xF3, 0x7F,
                                                0x01, 0xF8, 0x1F, 0xE0};

int Expand6(int value) { return (value << 2) | (value >> 4); }
int Expand7(int value) { return (value << 1) | (value >> 6); }

Rgba PlanarTexel(int x, int y) {
  const int o[3] = {Expand6(32), Expand7(32), Expand6(22)};
  const int h[3] = {Expand6(63), Expand7(0), Expand6(63)};
  const int v[3] = {Expand6(0), Expand7(127), Expand6(32)};
  Rgba out{0, 0, 0, 255};
  for (int c = 0; c < 3; ++c) {
    const int value =
        (x * (h[c] - o[c]) + y * (v[c] - o[c]) + 4 * o[c] + 2) >> 2;
    out[c] = value < 0 ? 0 : value > 255 ? 255 : value;
  }
  return out;
}

TEST(Etc2DecoderTest, PlanarMode) {
  const auto rgba = DecodeBlock(EtcFormat::kEtc2Rgb8, kPlanarBlock);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      EXPECT_EQ(TexelAt(rgba, x, y), PlanarTexel(x, y)) << x << "," << y;
    }
  }
  // A few corners spelled out.
  EXPECT_EQ(TexelAt(rgba, 0, 0), (Rgba{130, 64, 89, 255}));
  EXPECT_EQ(TexelAt(rgba, 3, 0), (Rgba{224, 16, 214, 255}));
  EXPECT_EQ(TexelAt(rgba, 0, 3), (Rgba{33, 207, 120, 255}));
  EXPECT_EQ(TexelAt(rgba, 3, 3), (Rgba{126, 159, 244, 255}));
}

// Punch-through alpha with the opaque bit clear: index 2 is transparent
// black and index 0 takes no modifier.
TEST(Etc2DecoderTest, PunchThroughTransparentDifferential) {
  const auto rgba =
      DecodeBlock(EtcFormat::kEtc2Rgb8A1, ColourBlock(0xA3, 0x54, 0xFF, 0x38));
  ExpectRows(rgba,
             {Rgba{165, 82, 255, 255}, Rgba{182, 99, 255, 255},
              Rgba{0, 0, 0, 0}, Rgba{148, 65, 238, 255}},
             {Rgba{189, 49, 247, 255}, Rgba{255, 155, 255, 255},
              Rgba{0, 0, 0, 0}, Rgba{83, 0, 141, 255}});
}

// The opaque bit set decodes like ETC2 RGB8 with alpha 255.
TEST(Etc2DecoderTest, PunchThroughOpaqueMatchesRgb) {
  const auto block = ColourBlock(0xA3, 0x54, 0xFF, 0x3A);
  EXPECT_EQ(DecodeBlock(EtcFormat::kEtc2Rgb8A1, block),
            DecodeBlock(EtcFormat::kEtc2Rgb8, block));
}

// Punch-through has no individual mode: these bits decode as differential.
TEST(Etc2DecoderTest, PunchThroughReadsIndividualBitsAsDifferential) {
  const auto rgba =
      DecodeBlock(EtcFormat::kEtc2Rgb8A1, ColourBlock(0x5A, 0x3C, 0x96, 0x55));
  const std::array<Rgba, 4> rows = {Rgba{90, 57, 148, 255},
                                    Rgba{119, 86, 177, 255}, Rgba{0, 0, 0, 0},
                                    Rgba{27, 0, 52, 255}};
  ExpectRows(rgba, rows, rows);
}

TEST(Etc2DecoderTest, PunchThroughTransparentTMode) {
  const auto rgba =
      DecodeBlock(EtcFormat::kEtc2Rgb8A1, ColourBlock(0xF3, 0x4E, 0x7C, 0x39));
  const std::array<Rgba, 4> rows = {Rgba{187, 68, 238, 255},
                                    Rgba{151, 236, 83, 255}, Rgba{0, 0, 0, 0},
                                    Rgba{87, 172, 19, 255}};
  ExpectRows(rgba, rows, rows);
}

// Planar blocks stay opaque even with the opaque bit clear.
TEST(Etc2DecoderTest, PunchThroughPlanarIsOpaque) {
  std::vector<std::uint8_t> block = kPlanarBlock;
  block[3] = 0x7D;
  const auto rgba = DecodeBlock(EtcFormat::kEtc2Rgb8A1, block);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      EXPECT_EQ(TexelAt(rgba, x, y), PlanarTexel(x, y)) << x << "," << y;
    }
  }
}

// R11 values per modifier index 0..7 for the blocks below.
constexpr std::array<std::uint16_t, 8> kR11Base100Multiplier0 = {
    0x646C, 0x644C, 0x642C, 0x634C, 0x648C, 0x64AC, 0x64CC, 0x65AC};
constexpr std::array<std::uint16_t, 8> kR11Base255Multiplier15 = {
    0xD29A, 0xA594, 0x788F, 0x1E83, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
constexpr std::array<std::uint16_t, 8> kR11Base0Multiplier15 = {
    0x0000, 0x0000, 0x0000, 0x0000, 0x1E83, 0x4B89, 0x788F, 0xD29A};
constexpr std::array<std::uint16_t, 8> kR11SignedBaseMinus128Multiplier0 = {
    0x80C1, 0x80A1, 0x8081, 0x8001, 0x80E1, 0x8101, 0x8121, 0x8201};
constexpr std::array<std::uint16_t, 8> kR11SignedBaseMinus128Multiplier1 = {
    0x8001, 0x8001, 0x8001, 0x8001, 0x82E1, 0x85E2, 0x88E3, 0x8EE4};
constexpr std::array<std::uint16_t, 8> kR11SignedBase127Multiplier15 = {
    0x5214, 0x2509, 0xF7FE, 0x9DE8, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF};

void ExpectR11(EtcFormat format, const Block8& block,
               const std::array<std::uint16_t, 8>& expected) {
  const auto texels = DecodeBlock(
      format, std::vector<std::uint8_t>(block.begin(), block.end()));
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      EXPECT_EQ(R16At(texels, x, y, 0, 1), expected[EacIndexAt(x, y)])
          << x << "," << y;
    }
  }
}

// Multiplier 0 adds the modifier unscaled (base * 8 + 4 + modifier).
TEST(Etc2DecoderTest, EacR11MultiplierZero) {
  ExpectR11(EtcFormat::kEacR11, EacBlock(100, 0x0D), kR11Base100Multiplier0);
}

TEST(Etc2DecoderTest, EacR11ClampsAtBothEnds) {
  ExpectR11(EtcFormat::kEacR11, EacBlock(255, 0xF0), kR11Base255Multiplier15);
  ExpectR11(EtcFormat::kEacR11, EacBlock(0, 0xF0), kR11Base0Multiplier15);
}

// A signed base of -128 is read as -127, and values clamp to -1023.
TEST(Etc2DecoderTest, EacR11SignedBaseMinus128) {
  ExpectR11(EtcFormat::kEacR11Signed, EacBlock(0x80, 0x0D),
            kR11SignedBaseMinus128Multiplier0);
  ExpectR11(EtcFormat::kEacR11Signed, EacBlock(0x80, 0x10),
            kR11SignedBaseMinus128Multiplier1);
}

TEST(Etc2DecoderTest, EacR11SignedClampsTo1023) {
  ExpectR11(EtcFormat::kEacR11Signed, EacBlock(0x7F, 0xF0),
            kR11SignedBase127Multiplier15);
}

// The 11-bit to 16-bit expansion: unsigned v -> v << 5 | v >> 6; signed
// magnitude m -> m << 5 | m >> 5 with the sign reapplied.
TEST(Etc2DecoderTest, R16BitPatterns) {
  const auto r16 = [](EtcFormat format, const Block8& block, int x, int y) {
    return R16At(DecodeBlock(format, std::vector<std::uint8_t>(block.begin(),
                                                               block.end())),
                 x, y, 0, 1);
  };
  // Table 13 index 7 adds 9, index 4 adds 0; table 6 index 0 adds -4.
  EXPECT_EQ(r16(EtcFormat::kEacR11, EacBlock(255, 0x0D), 1, 3), 0xFFFF);
  EXPECT_EQ(r16(EtcFormat::kEacR11, EacBlock(255, 0x0D), 1, 0), 0xFF9F);
  EXPECT_EQ(r16(EtcFormat::kEacR11, EacBlock(128, 0x06), 0, 0), 0x8010);
  EXPECT_EQ(r16(EtcFormat::kEacR11, EacBlock(127, 0x0D), 1, 0), 0x7F8F);
  EXPECT_EQ(r16(EtcFormat::kEacR11, EacBlock(0, 0xF0), 0, 0), 0x0000);
  EXPECT_EQ(r16(EtcFormat::kEacR11Signed, EacBlock(0, 0x0D), 1, 0), 0x0000);
  EXPECT_EQ(r16(EtcFormat::kEacR11Signed, EacBlock(0x7F, 0x0D), 1, 3),
            0x7FFF);  // 1016 + 9 clamps to 1023
  EXPECT_EQ(r16(EtcFormat::kEacR11Signed, EacBlock(0x80, 0x0D), 0, 3),
            0x8001);  // -1016 - 10 clamps to -1023
}

TEST(Etc2DecoderTest, EacRg11DecodesRedThenGreen) {
  const auto texels = DecodeBlock(
      EtcFormat::kEacRg11, Concat(EacBlock(100, 0x0D), EacBlock(255, 0xF0)));
  const auto signed_texels =
      DecodeBlock(EtcFormat::kEacRg11Signed,
                  Concat(EacBlock(0x80, 0x0D), EacBlock(0x7F, 0xF0)));
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 4; ++x) {
      const int index = EacIndexAt(x, y);
      EXPECT_EQ(R16At(texels, x, y, 0, 2), kR11Base100Multiplier0[index]);
      EXPECT_EQ(R16At(texels, x, y, 1, 2), kR11Base255Multiplier15[index]);
      EXPECT_EQ(R16At(signed_texels, x, y, 0, 2),
                kR11SignedBaseMinus128Multiplier0[index]);
      EXPECT_EQ(R16At(signed_texels, x, y, 1, 2),
                kR11SignedBase127Multiplier15[index]);
    }
  }
}

// ETC2 RGBA8: an 8-bit EAC alpha block followed by an opaque ETC2 colour
// block; alpha = clamp(base + modifier * multiplier).
TEST(Etc2DecoderTest, Rgba8AlphaAndColour) {
  struct Case {
    Block8 alpha;
    std::vector<std::uint8_t> colour;
    std::array<int, 8> alpha_by_index;
  };
  const Case cases[] = {
      {EacBlock(128, 0x2E),
       ColourBlock(0x5A, 0x3C, 0x96, 0x55),
       {120, 116, 112, 110, 134, 138, 142, 144}},
      {EacBlock(250, 0xF0),
       ColourBlock(0x52, 0x15, 0x96, 0x4E),
       {205, 160, 115, 25, 255, 255, 255, 255}},
      {EacBlock(77, 0x07), kPlanarBlock, {77, 77, 77, 77, 77, 77, 77, 77}},
  };
  for (const Case& test : cases) {
    const auto rgba =
        DecodeBlock(EtcFormat::kEtc2Rgba8, Concat(test.alpha, test.colour));
    const auto colour = DecodeBlock(EtcFormat::kEtc2Rgb8, test.colour);
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        Rgba expected = TexelAt(colour, x, y);
        expected[3] = test.alpha_by_index[EacIndexAt(x, y)];
        EXPECT_EQ(TexelAt(rgba, x, y), expected) << x << "," << y;
      }
    }
  }
}

// Partial blocks at the right and bottom edges write only the texels inside
// the image, from the block's top-left corner.
TEST(Etc2DecoderTest, ClipsEdgeBlocks) {
  const auto block = ColourBlock(0xA3, 0x54, 0xFF, 0x3A);
  const auto whole = DecodeBlock(EtcFormat::kEtc2Rgb8, block);
  std::vector<std::uint8_t> out(3 * 2 * 4, 0x5A);
  ASSERT_TRUE(DecodeEtcImage(EtcFormat::kEtc2Rgb8, block.data(), block.size(),
                             3, 2, out.data(), out.size()));
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 3; ++x) {
      const std::size_t offset = static_cast<std::size_t>(y * 3 + x) * 4;
      EXPECT_EQ((Rgba{out[offset], out[offset + 1], out[offset + 2],
                      out[offset + 3]}),
                TexelAt(whole, x, y));
    }
  }
}

TEST(Etc2DecoderTest, RejectsShortBuffersAndEmptyImages) {
  std::vector<std::uint8_t> source(4 * 8);
  std::vector<std::uint8_t> out(8 * 8 * 4);
  EXPECT_TRUE(DecodeEtcImage(EtcFormat::kEtc2Rgb8, source.data(), 32, 8, 8,
                             out.data(), out.size()));
  EXPECT_FALSE(DecodeEtcImage(EtcFormat::kEtc2Rgb8, source.data(), 31, 8, 8,
                              out.data(), out.size()));
  EXPECT_FALSE(DecodeEtcImage(EtcFormat::kEtc2Rgb8, source.data(), 32, 8, 8,
                              out.data(), out.size() - 1));
  EXPECT_FALSE(DecodeEtcImage(EtcFormat::kEtc2Rgb8, source.data(), 32, 0, 8,
                              out.data(), out.size()));
  EXPECT_FALSE(DecodeEtcImage(EtcFormat::kEtc2Rgb8, nullptr, 32, 8, 8,
                              out.data(), out.size()));
  EXPECT_FALSE(DecodeEtcImageBlockRows(EtcFormat::kEtc2Rgb8, source.data(), 32,
                                       8, 8, 1, 2, out.data(), out.size()));
  EXPECT_EQ(EtcBlockBytes(EtcFormat::kEtc2Rgb8), 8U);
  EXPECT_EQ(EtcBlockBytes(EtcFormat::kEtc2Rgb8A1), 8U);
  EXPECT_EQ(EtcBlockBytes(EtcFormat::kEtc2Rgba8), 16U);
  EXPECT_EQ(EtcBlockBytes(EtcFormat::kEacR11), 8U);
  EXPECT_EQ(EtcBlockBytes(EtcFormat::kEacR11Signed), 8U);
  EXPECT_EQ(EtcBlockBytes(EtcFormat::kEacRg11), 16U);
  EXPECT_EQ(EtcBlockBytes(EtcFormat::kEacRg11Signed), 16U);
  EXPECT_EQ(EtcDecodedTexelBytes(EtcFormat::kEtc2Rgb8), 4U);
  EXPECT_EQ(EtcDecodedTexelBytes(EtcFormat::kEtc2Rgba8), 4U);
  EXPECT_EQ(EtcDecodedTexelBytes(EtcFormat::kEacR11), 2U);
  EXPECT_EQ(EtcDecodedTexelBytes(EtcFormat::kEacR11Signed), 2U);
  EXPECT_EQ(EtcDecodedTexelBytes(EtcFormat::kEacRg11Signed), 4U);
}

std::vector<std::uint8_t> RandomBytes(std::mt19937_64* rng, std::size_t size) {
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t index = 0; index < size; index += 8) {
    const std::uint64_t value = (*rng)();
    for (std::size_t byte = 0; byte < 8 && index + byte < size; ++byte) {
      bytes[index + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
    }
  }
  return bytes;
}

std::size_t CompressedBytes(EtcFormat format, std::uint32_t width,
                            std::uint32_t height) {
  return static_cast<std::size_t>((width + 3) / 4) * ((height + 3) / 4) *
         EtcBlockBytes(format);
}

std::vector<std::uint8_t> ReferenceDecode(EtcFormat format,
                                          const std::vector<std::uint8_t>& src,
                                          std::uint32_t width,
                                          std::uint32_t height) {
  std::vector<std::uint8_t> out(
      static_cast<std::size_t>(width) * height * EtcDecodedTexelBytes(format),
      0xA5);
  EXPECT_TRUE(reference::DecodeEtcImage(format, src.data(), src.size(), width,
                                        height, out.data(), out.size()));
  return out;
}

// Random blocks hit every mode (about half are individual, and the overflow
// modes and every EAC table appear in a few thousand blocks).
TEST(Etc2DecoderTest, MatchesReferenceOnRandomImages) {
  std::mt19937_64 rng(0x5EED0001);
  int images = 0;
  for (EtcFormat format : kAllFormats) {
    for (int iteration = 0; iteration < 300; ++iteration) {
      const std::uint32_t width = 1 + static_cast<std::uint32_t>(rng() % 70);
      const std::uint32_t height = 1 + static_cast<std::uint32_t>(rng() % 70);
      const auto source =
          RandomBytes(&rng, CompressedBytes(format, width, height));
      std::vector<std::uint8_t> out(static_cast<std::size_t>(width) * height *
                                        EtcDecodedTexelBytes(format),
                                    0x3C);
      ASSERT_TRUE(DecodeEtcImage(format, source.data(), source.size(), width,
                                 height, out.data(), out.size()));
      ASSERT_EQ(out, ReferenceDecode(format, source, width, height))
          << "format " << static_cast<int>(format) << " " << width << "x"
          << height;
      ++images;
    }
  }
  EXPECT_EQ(images, 2100);
}

// Decoding in block-row bands gives the same texels as one pass.
TEST(Etc2DecoderTest, BandsMatchReference) {
  std::mt19937_64 rng(0x5EED0002);
  for (EtcFormat format : kAllFormats) {
    for (int iteration = 0; iteration < 40; ++iteration) {
      const std::uint32_t width = 1 + static_cast<std::uint32_t>(rng() % 70);
      const std::uint32_t height = 1 + static_cast<std::uint32_t>(rng() % 70);
      const auto source =
          RandomBytes(&rng, CompressedBytes(format, width, height));
      std::vector<std::uint8_t> out(static_cast<std::size_t>(width) * height *
                                        EtcDecodedTexelBytes(format),
                                    0x3C);
      const std::uint32_t block_rows = (height + 3) / 4;
      std::uint32_t row = 0;
      while (row < block_rows) {
        const std::uint32_t count = std::min<std::uint32_t>(
            block_rows - row, 1 + static_cast<std::uint32_t>(rng() % 5));
        ASSERT_TRUE(DecodeEtcImageBlockRows(format, source.data(),
                                            source.size(), width, height, row,
                                            count, out.data(), out.size()));
        row += count;
      }
      ASSERT_EQ(out, ReferenceDecode(format, source, width, height));
    }
  }
}

// DecodeEtcJobs splits jobs into bands and, above 65536 blocks, runs them on
// worker threads; the output must not depend on either.
TEST(Etc2DecoderTest, JobsMatchReferenceSerialAndThreaded) {
  std::mt19937_64 rng(0x5EED0003);
  struct Image {
    EtcFormat format;
    std::uint32_t width;
    std::uint32_t height;
    std::vector<std::uint8_t> source;
    std::vector<std::uint8_t> out;
  };
  std::vector<Image> images;
  for (EtcFormat format : kAllFormats) {
    for (int iteration = 0; iteration < 6; ++iteration) {
      const std::uint32_t width = 1 + static_cast<std::uint32_t>(rng() % 300);
      const std::uint32_t height = 1 + static_cast<std::uint32_t>(rng() % 90);
      images.push_back(
          {format,
           width,
           height,
           RandomBytes(&rng, CompressedBytes(format, width, height)),
           {}});
    }
  }
  // 1024x1040 RGB8 alone is 66560 blocks: the threaded path with banding.
  images.push_back(
      {EtcFormat::kEtc2Rgb8,
       1024,
       1040,
       RandomBytes(&rng, CompressedBytes(EtcFormat::kEtc2Rgb8, 1024, 1040)),
       {}});
  for (const unsigned workers : {1U, 4U}) {
    std::vector<EtcDecodeJob> jobs;
    for (Image& image : images) {
      image.out.assign(static_cast<std::size_t>(image.width) * image.height *
                           EtcDecodedTexelBytes(image.format),
                       0x3C);
      EtcDecodeJob job;
      job.format = image.format;
      job.source = image.source.data();
      job.source_bytes = image.source.size();
      job.width = image.width;
      job.height = image.height;
      job.destination = image.out.data();
      job.destination_bytes = image.out.size();
      jobs.push_back(job);
    }
    // A broken job is reported and does not stop the others.
    std::vector<std::uint8_t> small(4);
    EtcDecodeJob broken;
    broken.source = small.data();
    broken.source_bytes = small.size();
    broken.width = 8;
    broken.height = 8;
    broken.destination = small.data();
    broken.destination_bytes = small.size();
    broken.ok = true;
    jobs.insert(jobs.begin() + 3, broken);
    DecodeEtcJobs(jobs.data(), jobs.size(), workers);
    EXPECT_FALSE(jobs[3].ok);
    for (std::size_t index = 0; index < images.size(); ++index) {
      const EtcDecodeJob& job = jobs[index < 3 ? index : index + 1];
      EXPECT_TRUE(job.ok);
      ASSERT_EQ(images[index].out,
                ReferenceDecode(images[index].format, images[index].source,
                                images[index].width, images[index].height))
          << "image " << index << " workers " << workers;
    }
  }
}

}  // namespace
}  // namespace mocktail::graphics
