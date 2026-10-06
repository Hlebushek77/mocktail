#include "mocktail/graphics/bc_encoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "support/bc_reference.h"
#include "support/bc_reference_transcode.h"
#include "support/etc2_reference_decoder.h"

// The CPU BC encoder of ETC2->BC mode (DESIGN C4.4) against the reference
// the GPU route's shader was verified with (support/bc_reference.h, a
// frozen copy of plus2-design/harness/bc_ref.h): every block must give the
// same bytes, so an image is identical whichever route served it. Plus the
// properties BC decoders rely on, checked on the encoded bytes directly.

namespace mocktail::graphics {
namespace {

constexpr EtcFormat kFormats[] = {
    EtcFormat::kEtc2Rgb8,      EtcFormat::kEtc2Rgb8A1,   EtcFormat::kEtc2Rgba8,
    EtcFormat::kEacR11,        EtcFormat::kEacR11Signed, EtcFormat::kEacRg11,
    EtcFormat::kEacRg11Signed,
};

const char* Name(EtcFormat format) {
  static const char* const kNames[] = {
      "RGB8->BC1",  "RGB8A1->BC1a", "RGBA8->BC3", "R11->BC4",
      "R11s->BC4s", "RG11->BC5",    "RG11s->BC5s"};
  return kNames[static_cast<int>(format)];
}

bool IsEac(EtcFormat format) {
  return format != EtcFormat::kEtc2Rgb8 && format != EtcFormat::kEtc2Rgb8A1 &&
         format != EtcFormat::kEtc2Rgba8;
}

// Random blocks as bc_test parity makes them: random bytes, with the
// multiplier of every 13th 8-byte EAC half cleared, so flat EAC blocks
// occur. Random ETC2 colour blocks cover every mode (individual,
// differential, T, H, planar, punch-through).
std::vector<std::uint8_t> RandomEtcBlocks(EtcFormat format, std::size_t count,
                                          std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::uint8_t> blocks(count * EtcBlockBytes(format));
  for (std::uint8_t& byte : blocks) {
    byte = static_cast<std::uint8_t>(rng());
  }
  if (IsEac(format) || format == EtcFormat::kEtc2Rgba8) {
    for (std::size_t i = 0; i < blocks.size(); i += 8 * 13) {
      blocks[i + 1] &= 0x0F;
    }
  }
  return blocks;
}

void StoreLe(const std::uint32_t* words, std::size_t count, std::uint8_t* out) {
  for (std::size_t i = 0; i < count * 4; ++i) {
    out[i] = static_cast<std::uint8_t>(words[i / 4] >> (8 * (i % 4)));
  }
}

std::uint64_t Fnv1a(std::uint64_t hash, const std::uint8_t* data,
                    std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    hash = (hash ^ data[i]) * 0x100000001b3ULL;
  }
  return hash;
}

std::string Hex(const std::uint8_t* data, std::size_t size) {
  std::string out;
  char byte[4];
  for (std::size_t i = 0; i < size; ++i) {
    std::snprintf(byte, sizeof(byte), "%02x", data[i]);
    out += byte;
  }
  return out;
}

constexpr std::size_t kParityBlocks = 16384;
constexpr std::uint64_t kParitySeed = 0xBC0E7C2ULL;

// FNV-1a over the reference's output for every parity block, refine off
// then on, formats in enum order: computed from the original
// plus2-design/harness/bc_ref.h, so this file's copy cannot drift either.
constexpr std::uint64_t kParityGolden = 0x860440c1828669a9ULL;

TEST(BcEncoderTest, MatchesTheReferenceOnRandomEtcBlocks) {
  std::uint64_t reference_hash = 0xcbf29ce484222325ULL;
  for (const bool refine : {false, true}) {
    for (const EtcFormat format : kFormats) {
      const std::size_t block_bytes = EtcBlockBytes(format);
      const std::vector<std::uint8_t> blocks = RandomEtcBlocks(
          format, kParityBlocks, kParitySeed + static_cast<int>(format));
      std::vector<std::uint8_t> expected(blocks.size());
      std::vector<std::uint8_t> one_by_one(blocks.size());
      std::vector<std::uint8_t> batched(blocks.size());
      for (std::size_t block = 0; block < kParityBlocks; ++block) {
        // bc_test's CpuTranscodeBlock: the plus.1 decoder, then the
        // reference encoder.
        reference::BcBlockFromEtc(format, blocks.data() + block * block_bytes,
                                  refine, expected.data() + block * block_bytes);
        TranscodeEtcBlockToBc(format, blocks.data() + block * block_bytes,
                              refine, one_by_one.data() + block * block_bytes);
      }
      ASSERT_TRUE(TranscodeEtcToBc(format, blocks.data(), blocks.size(),
                                   batched.data(), batched.size(), refine));
      reference_hash = Fnv1a(reference_hash, expected.data(), expected.size());
      int reported = 0;
      std::size_t mismatches = 0;
      for (std::size_t block = 0; block < kParityBlocks; ++block) {
        const std::size_t offset = block * block_bytes;
        const bool same =
            std::memcmp(expected.data() + offset, one_by_one.data() + offset,
                        block_bytes) == 0 &&
            std::memcmp(expected.data() + offset, batched.data() + offset,
                        block_bytes) == 0;
        if (!same) {
          ++mismatches;
          if (reported++ < 3) {
            ADD_FAILURE() << Name(format) << " refine=" << refine << " block "
                          << block << ": etc "
                          << Hex(blocks.data() + offset, block_bytes)
                          << " reference "
                          << Hex(expected.data() + offset, block_bytes)
                          << " encoder "
                          << Hex(one_by_one.data() + offset, block_bytes)
                          << " batched "
                          << Hex(batched.data() + offset, block_bytes);
          }
        }
      }
      EXPECT_EQ(mismatches, 0U) << Name(format) << " refine=" << refine;
    }
  }
  char golden[32];
  std::snprintf(golden, sizeof(golden), "0x%016llxULL",
                static_cast<unsigned long long>(reference_hash));
  EXPECT_EQ(reference_hash, kParityGolden)
      << "the reference copy no longer gives the output of the original "
         "bc_ref.h; this run gives "
      << golden;
}

// Random RGBA blocks with every colour distinct, and with a few colours
// only (what textures mostly hold), straight into the block encoders.
std::vector<std::uint8_t> RandomTexelBlock(std::mt19937_64* rng, int colours) {
  std::vector<std::uint8_t> rgba(64);
  std::uint8_t palette[4][4];
  for (auto& colour : palette) {
    for (std::uint8_t& channel : colour) {
      channel = static_cast<std::uint8_t>((*rng)());
    }
  }
  for (int i = 0; i < 16; ++i) {
    for (int c = 0; c < 4; ++c) {
      rgba[i * 4 + c] = colours == 0 ? static_cast<std::uint8_t>((*rng)())
                                     : palette[(*rng)() % colours][c];
    }
  }
  return rgba;
}

TEST(BcEncoderTest, MatchesTheReferenceOnRandomTexels) {
  std::mt19937_64 rng(0x7E3E15ULL);
  for (int trial = 0; trial < 20000; ++trial) {
    const int colours = trial % 5;  // 0: every texel random
    const std::vector<std::uint8_t> rgba = RandomTexelBlock(&rng, colours);
    for (const bool refine : {false, true}) {
      for (const bool punch : {false, true}) {
        std::uint32_t words[2];
        bcref::EncodeBc1(rgba.data(), punch, false, refine, words);
        std::uint8_t expected[8];
        StoreLe(words, 2, expected);
        std::uint8_t actual[8];
        EncodeBc1Block(rgba.data(), punch, refine, actual);
        ASSERT_EQ(Hex(actual, 8), Hex(expected, 8))
            << "BC1 trial " << trial << " punch=" << punch
            << " refine=" << refine << " texels " << Hex(rgba.data(), 64);
      }
      std::uint32_t words[4];
      int alpha[16];
      for (int i = 0; i < 16; ++i) {
        alpha[i] = rgba[i * 4 + 3];
      }
      bcref::EncodeBc4(alpha, false, false, words);
      bcref::EncodeBc1(rgba.data(), false, true, refine, words + 2);
      std::uint8_t expected[16];
      StoreLe(words, 4, expected);
      std::uint8_t actual[16];
      EncodeBc3Block(rgba.data(), refine, actual);
      ASSERT_EQ(Hex(actual, 16), Hex(expected, 16))
          << "BC3 trial " << trial << " refine=" << refine;
    }
  }
  // BC4 and BC5 on R16 values: random, a narrow range, and the extremes.
  for (int trial = 0; trial < 20000; ++trial) {
    std::uint16_t values[32];
    const int kind = trial % 3;
    const auto base = static_cast<std::uint16_t>(rng());
    for (std::uint16_t& value : values) {
      value = kind == 0 ? static_cast<std::uint16_t>(rng())
              : kind == 1
                  ? static_cast<std::uint16_t>(base + rng() % 700)
                  : static_cast<std::uint16_t>((rng() & 1) != 0
                                                   ? 0x8000 + rng() % 3
                                                   : 0x7FFF - rng() % 3);
    }
    for (const bool snorm : {false, true}) {
      int channel_values[2][16];
      for (int i = 0; i < 16; ++i) {
        for (int c = 0; c < 2; ++c) {
          channel_values[c][i] =
              snorm ? static_cast<std::int16_t>(values[i * 2 + c])
                    : values[i * 2 + c];
        }
      }
      std::uint32_t words[4];
      bcref::EncodeBc4(channel_values[0], true, snorm, words);
      bcref::EncodeBc4(channel_values[1], true, snorm, words + 2);
      std::uint8_t expected[16];
      StoreLe(words, 4, expected);
      std::uint8_t bc5[16];
      EncodeBc5Block(values, snorm, bc5);
      ASSERT_EQ(Hex(bc5, 16), Hex(expected, 16))
          << "BC5 trial " << trial << " snorm=" << snorm;
      std::uint16_t red[16];
      for (int i = 0; i < 16; ++i) {
        red[i] = values[i * 2];
      }
      std::uint8_t bc4[8];
      EncodeBc4Block(red, snorm, bc4);
      ASSERT_EQ(Hex(bc4, 8), Hex(expected, 8))
          << "BC4 trial " << trial << " snorm=" << snorm;
    }
  }
}

// ---- what BC decoders read from the bytes --------------------------------

std::uint32_t Endpoint(const std::uint8_t* block, int which) {
  return static_cast<std::uint32_t>(block[which * 2] |
                                    (block[which * 2 + 1] << 8));
}

int Bc1Index(const std::uint8_t* block, int texel) {
  return (block[4 + texel / 4] >> (2 * (texel % 4))) & 3;
}

// A BC4 block's 3-bit index of `texel`, read from its 48-bit field.
int Bc4Index(const std::uint8_t* block, int texel) {
  std::uint64_t bits = 0;
  for (int i = 0; i < 6; ++i) {
    bits |= static_cast<std::uint64_t>(block[2 + i]) << (8 * i);
  }
  return static_cast<int>((bits >> (3 * texel)) & 7);
}

// BC4 as D3D10 decodes it, in [0, 1] (unorm) or [-1, 1] (snorm).
double Bc4Value(const std::uint8_t* block, bool snorm, int texel) {
  double e0 = 0;
  double e1 = 0;
  bool eight = false;
  if (snorm) {
    const int s0 =
        std::max(-127, static_cast<int>(static_cast<std::int8_t>(block[0])));
    const int s1 =
        std::max(-127, static_cast<int>(static_cast<std::int8_t>(block[1])));
    e0 = s0 / 127.0;
    e1 = s1 / 127.0;
    eight =
        static_cast<std::int8_t>(block[0]) > static_cast<std::int8_t>(block[1]);
  } else {
    e0 = block[0] / 255.0;
    e1 = block[1] / 255.0;
    eight = block[0] > block[1];
  }
  const int index = Bc4Index(block, texel);
  if (index == 0) return e0;
  if (index == 1) return e1;
  if (eight) return ((8 - index) * e0 + (index - 1) * e1) / 7.0;
  if (index == 6) return snorm ? -1.0 : 0.0;
  if (index == 7) return 1.0;
  return ((6 - index) * e0 + (index - 1) * e1) / 5.0;
}

TEST(BcEncoderTest, OpaqueBc1BlocksDecodeInFourColourMode) {
  const std::vector<std::uint8_t> blocks =
      RandomEtcBlocks(EtcFormat::kEtc2Rgb8, 4096, 11);
  for (const bool refine : {false, true}) {
    for (std::size_t block = 0; block < 4096; ++block) {
      std::uint8_t bc[8];
      TranscodeEtcBlockToBc(EtcFormat::kEtc2Rgb8, blocks.data() + block * 8,
                            refine, bc);
      const std::uint32_t c0 = Endpoint(bc, 0);
      const std::uint32_t c1 = Endpoint(bc, 1);
      if (c0 == c1) {
        // One colour: 3-colour mode, which must then never pick black.
        for (int texel = 0; texel < 16; ++texel) {
          EXPECT_NE(Bc1Index(bc, texel), 3) << "block " << block;
        }
      } else {
        EXPECT_GT(c0, c1) << "block " << block;
      }
    }
  }
}

TEST(BcEncoderTest, PunchThroughUsesIndexThreeExactlyOnTransparentTexels) {
  const std::vector<std::uint8_t> blocks =
      RandomEtcBlocks(EtcFormat::kEtc2Rgb8A1, 8192, 12);
  int partly_transparent = 0;
  for (std::size_t block = 0; block < 8192; ++block) {
    std::uint8_t decoded[64];
    reference::DecodeEtcImage(EtcFormat::kEtc2Rgb8A1, blocks.data() + block * 8,
                              8, 4, 4, decoded, sizeof(decoded));
    std::uint8_t bc[8];
    TranscodeEtcBlockToBc(EtcFormat::kEtc2Rgb8A1, blocks.data() + block * 8,
                          true, bc);
    int transparent = 0;
    for (int texel = 0; texel < 16; ++texel) {
      transparent += decoded[texel * 4 + 3] < 128 ? 1 : 0;
    }
    if (transparent == 0) {
      // Opaque: 4-colour mode, where index 3 is a colour, or one colour.
      bool black = false;
      for (int texel = 0; texel < 16; ++texel) {
        black = black || Bc1Index(bc, texel) == 3;
      }
      EXPECT_TRUE(Endpoint(bc, 0) > Endpoint(bc, 1) || !black)
          << "block " << block;
      continue;
    }
    // 3-colour mode, index 3 (transparent black) on exactly those texels.
    EXPECT_LE(Endpoint(bc, 0), Endpoint(bc, 1)) << "block " << block;
    for (int texel = 0; texel < 16; ++texel) {
      EXPECT_EQ(Bc1Index(bc, texel) == 3, decoded[texel * 4 + 3] < 128)
          << "block " << block << " texel " << texel;
    }
    partly_transparent += transparent < 16 ? 1 : 0;
  }
  EXPECT_GT(partly_transparent, 100);

  // The threshold is alpha 128, as for an override's alpha (CR-24).
  std::uint8_t rgba[64];
  for (int texel = 0; texel < 16; ++texel) {
    rgba[texel * 4] = 200;
    rgba[texel * 4 + 1] = static_cast<std::uint8_t>(texel * 16);
    rgba[texel * 4 + 2] = 40;
    rgba[texel * 4 + 3] = texel % 2 == 0 ? 127 : 128;
  }
  std::uint8_t bc[8];
  EncodeBc1Block(rgba, true, true, bc);
  for (int texel = 0; texel < 16; ++texel) {
    EXPECT_EQ(Bc1Index(bc, texel) == 3, texel % 2 == 0) << texel;
  }
  // Without punch-through alpha is ignored.
  EncodeBc1Block(rgba, false, true, bc);
  EXPECT_GT(Endpoint(bc, 0), Endpoint(bc, 1));
}

TEST(BcEncoderTest, FullyTransparentAndFlatBlocks) {
  std::uint8_t rgba[64] = {};
  std::uint8_t bc[16];
  EncodeBc1Block(rgba, true, true, bc);
  EXPECT_EQ(Hex(bc, 8), "00000000ffffffff");

  // A flat colour that 565 holds exactly comes back exactly, in every
  // encoder; BC3 keeps a flat alpha exactly too.
  for (int texel = 0; texel < 16; ++texel) {
    rgba[texel * 4] = 255;
    rgba[texel * 4 + 1] = 130;  // 6-bit 32 expands to 130
    rgba[texel * 4 + 2] = 0;
    rgba[texel * 4 + 3] = 77;
  }
  for (const bool refine : {false, true}) {
    EncodeBc1Block(rgba, false, refine, bc);
    std::uint32_t words[2] = {
        static_cast<std::uint32_t>(bc[0] | bc[1] << 8 | bc[2] << 16) |
            static_cast<std::uint32_t>(bc[3]) << 24,
        static_cast<std::uint32_t>(bc[4] | bc[5] << 8 | bc[6] << 16) |
            static_cast<std::uint32_t>(bc[7]) << 24};
    std::uint8_t decoded[64];
    bcref::DecodeBc1(words, false, decoded);
    for (int texel = 0; texel < 16; ++texel) {
      EXPECT_EQ(decoded[texel * 4], 255);
      EXPECT_EQ(decoded[texel * 4 + 1], 130);
      EXPECT_EQ(decoded[texel * 4 + 2], 0);
    }
    EncodeBc3Block(rgba, refine, bc);
    EXPECT_EQ(bc[0], 77);
    EXPECT_EQ(bc[1], 77);
    EXPECT_EQ(Hex(bc + 2, 6), "000000000000");
  }
}

TEST(BcEncoderTest, Bc4PacksEndpointsAndIndicesAcrossTheWordBoundary) {
  // A ramp over the full range puts every one of the 8 palette entries in
  // the block, texel 10 (bits 30..32) and 11 included.
  std::uint16_t values[16];
  for (int texel = 0; texel < 16; ++texel) {
    values[texel] = static_cast<std::uint16_t>(texel * 65535 / 15);
  }
  std::uint8_t bc[8];
  EncodeBc4Block(values, false, bc);
  EXPECT_EQ(bc[0], 255);  // e0 > e1: 8-value mode
  EXPECT_EQ(bc[1], 0);
  for (int texel = 0; texel < 16; ++texel) {
    const double expected = values[texel] / 65535.0;
    EXPECT_NEAR(Bc4Value(bc, false, texel), expected, 1.0 / 14 + 1e-9)
        << "texel " << texel;
  }
  EXPECT_EQ(Bc4Index(bc, 0), 1);   // e1 (0)
  EXPECT_EQ(Bc4Index(bc, 15), 0);  // e0 (65535)
  // Texel 10 = 43690 = 2/3 of the range: palette position round(14/3) = 5,
  // index 3; texel 11 = 48059: position round(5.13) = 5, index 3.
  EXPECT_EQ(Bc4Index(bc, 10), 3);
  EXPECT_EQ(Bc4Index(bc, 11), 3);
  // Every value within half a palette step of what it decodes to.
  std::mt19937_64 rng(0xBC4);
  for (int trial = 0; trial < 2000; ++trial) {
    for (std::uint16_t& value : values) {
      value = static_cast<std::uint16_t>(rng());
    }
    EncodeBc4Block(values, false, bc);
    const double step =
        std::abs(static_cast<double>(bc[0]) - bc[1]) / 255.0 / 7.0;
    for (int texel = 0; texel < 16; ++texel) {
      EXPECT_NEAR(Bc4Value(bc, false, texel), values[texel] / 65535.0,
                  step / 2 + 1.0 / 255)
          << "trial " << trial << " texel " << texel;
    }
  }
}

TEST(BcEncoderTest, Bc4SnormMapsTheFullRange) {
  std::uint16_t values[16];
  const auto flat = [&](std::uint16_t value) {
    std::fill(std::begin(values), std::end(values), value);
    std::uint8_t bc[8];
    EncodeBc4Block(values, true, bc);
    return std::vector<std::uint8_t>(bc, bc + 8);
  };
  // -32768 reads as -32767, both -1.0: endpoint -127 (0x81).
  EXPECT_EQ(flat(0x8000)[0], 0x81);
  EXPECT_EQ(flat(0x8001)[0], 0x81);
  EXPECT_EQ(flat(0x7FFF)[0], 0x7F);  // +1.0: 127
  EXPECT_EQ(flat(0x0000)[0], 0x00);  // 0.0
  // A ramp from -1 to 1 decodes within half a step of every value.
  for (int texel = 0; texel < 16; ++texel) {
    values[texel] = static_cast<std::uint16_t>(
        static_cast<std::int16_t>(-32767 + texel * 65534 / 15));
  }
  std::uint8_t bc[8];
  EncodeBc4Block(values, true, bc);
  EXPECT_EQ(bc[0], 0x7F);
  EXPECT_EQ(bc[1], 0x81);
  for (int texel = 0; texel < 16; ++texel) {
    const double expected = static_cast<std::int16_t>(values[texel]) / 32767.0;
    EXPECT_NEAR(Bc4Value(bc, true, texel), expected, 2.0 / 14 + 1e-9)
        << "texel " << texel;
  }
}

double Bc1Psnr(const std::vector<std::uint8_t>& image, std::uint32_t width,
               std::uint32_t height, bool refine) {
  std::vector<std::uint8_t> bc(
      BcEncodedBytes(EtcFormat::kEtc2Rgb8, width, height, 1));
  EXPECT_TRUE(TranscodeDecodedToBc(EtcFormat::kEtc2Rgb8, image.data(), width,
                                   height, 1, bc.data(), bc.size(), refine));
  double squared = 0;
  const std::uint32_t blocks_wide = (width + 3) / 4;
  for (std::uint32_t by = 0; by < (height + 3) / 4; ++by) {
    for (std::uint32_t bx = 0; bx < blocks_wide; ++bx) {
      const std::uint8_t* block = bc.data() + (by * blocks_wide + bx) * 8;
      const std::uint32_t words[2] = {
          static_cast<std::uint32_t>(block[0] | block[1] << 8 |
                                     block[2] << 16) |
              static_cast<std::uint32_t>(block[3]) << 24,
          static_cast<std::uint32_t>(block[4] | block[5] << 8 |
                                     block[6] << 16) |
              static_cast<std::uint32_t>(block[7]) << 24};
      std::uint8_t decoded[64];
      bcref::DecodeBc1(words, false, decoded);
      for (int texel = 0; texel < 16; ++texel) {
        const std::uint32_t x = bx * 4 + texel % 4;
        const std::uint32_t y = by * 4 + texel / 4;
        for (int c = 0; c < 3; ++c) {
          const double d = static_cast<double>(decoded[texel * 4 + c]) -
                           image[(y * width + x) * 4 + c];
          squared += d * d;
        }
      }
    }
  }
  const double mse = squared / (3.0 * width * height);
  return mse == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

TEST(BcEncoderTest, RefinedBc1KeepsSmoothGradients) {
  constexpr std::uint32_t kSize = 128;
  const auto gradient = [](auto texel) {
    std::vector<std::uint8_t> image(kSize * kSize * 4);
    for (std::uint32_t y = 0; y < kSize; ++y) {
      for (std::uint32_t x = 0; x < kSize; ++x) {
        std::uint8_t* out = image.data() + (y * kSize + x) * 4;
        texel(x, y, out);
        out[3] = 255;
      }
    }
    return image;
  };
  const std::vector<std::vector<std::uint8_t>> images = {
      gradient([](std::uint32_t x, std::uint32_t, std::uint8_t* out) {
        out[0] = out[1] = out[2] = static_cast<std::uint8_t>(x * 2);
      }),
      gradient([](std::uint32_t x, std::uint32_t y, std::uint8_t* out) {
        out[0] = static_cast<std::uint8_t>(x * 2);
        out[1] = static_cast<std::uint8_t>(y * 2);
        out[2] = static_cast<std::uint8_t>(255 - x - y);
      }),
      gradient([](std::uint32_t x, std::uint32_t y, std::uint8_t* out) {
        out[0] = static_cast<std::uint8_t>(64 + (x + y) / 2);
        out[1] = static_cast<std::uint8_t>(128 + y / 2);
        out[2] = static_cast<std::uint8_t>(32 + x);
      }),
  };
  for (std::size_t index = 0; index < images.size(); ++index) {
    const double refined = Bc1Psnr(images[index], kSize, kSize, true);
    const double plain = Bc1Psnr(images[index], kSize, kSize, false);
    EXPECT_GE(refined, 40.0) << "gradient " << index;
    EXPECT_GE(refined, plain) << "gradient " << index;
  }
}

TEST(BcEncoderTest, TranscodeDecodedEncodesWholeBlocksAndRepeatsEdges) {
  std::mt19937_64 rng(0xED6E);
  for (const EtcFormat format : kFormats) {
    const std::size_t texel_bytes = EtcDecodedTexelBytes(format);
    for (const auto& [width, height] :
         {std::pair<std::uint32_t, std::uint32_t>{16, 8},
          {13, 6},
          {1, 1},
          {2, 7}}) {
      constexpr std::uint32_t kLayers = 2;
      const std::uint32_t blocks_wide = (width + 3) / 4;
      const std::uint32_t blocks_high = (height + 3) / 4;
      const std::size_t blocks = blocks_wide * blocks_high * kLayers;
      const std::vector<std::uint8_t> etc =
          RandomEtcBlocks(format, blocks, rng());
      const std::size_t layer_etc = etc.size() / kLayers;
      // Whole blocks decoded give exactly the GPU route's bytes.
      std::vector<std::uint8_t> padded(blocks * 16 * texel_bytes);
      const std::size_t padded_layer = padded.size() / kLayers;
      for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        ASSERT_TRUE(DecodeEtcImage(format, etc.data() + layer * layer_etc,
                                   layer_etc, blocks_wide * 4, blocks_high * 4,
                                   padded.data() + layer * padded_layer,
                                   padded_layer));
      }
      std::vector<std::uint8_t> expected(etc.size());
      ASSERT_TRUE(TranscodeEtcToBc(format, etc.data(), etc.size(),
                                   expected.data(), expected.size(), true));
      std::vector<std::uint8_t> actual(etc.size(), 0xEE);
      ASSERT_TRUE(TranscodeDecodedToBc(format, padded.data(), blocks_wide * 4,
                                       blocks_high * 4, kLayers, actual.data(),
                                       actual.size(), true));
      EXPECT_EQ(actual, expected)
          << Name(format) << " " << width << "x" << height;
      // The region itself: texels past the edge repeat the last column and
      // row, which equals encoding a copy padded that way.
      std::vector<std::uint8_t> region(static_cast<std::size_t>(width) *
                                       height * texel_bytes * kLayers);
      std::vector<std::uint8_t> repeated(padded.size());
      for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t y = 0; y < blocks_high * 4; ++y) {
          for (std::uint32_t x = 0; x < blocks_wide * 4; ++x) {
            const std::uint8_t* from =
                padded.data() + layer * padded_layer +
                (std::min(y, height - 1) * blocks_wide * 4 +
                 std::min(x, width - 1)) *
                    texel_bytes;
            std::memcpy(repeated.data() + layer * padded_layer +
                            (y * blocks_wide * 4 + x) * texel_bytes,
                        from, texel_bytes);
            if (x < width && y < height) {
              std::memcpy(region.data() +
                              ((layer * height + y) * width + x) * texel_bytes,
                          from, texel_bytes);
            }
          }
        }
      }
      ASSERT_TRUE(TranscodeDecodedToBc(
          format, repeated.data(), blocks_wide * 4, blocks_high * 4, kLayers,
          expected.data(), expected.size(), false));
      ASSERT_TRUE(TranscodeDecodedToBc(format, region.data(), width, height,
                                       kLayers, actual.data(), actual.size(),
                                       false));
      EXPECT_EQ(actual, expected)
          << Name(format) << " " << width << "x" << height;
    }
  }
}

TEST(BcEncoderTest, RejectsPartialBlocksAndShortOutput) {
  std::vector<std::uint8_t> etc = RandomEtcBlocks(EtcFormat::kEtc2Rgba8, 4, 3);
  std::vector<std::uint8_t> out(etc.size(), 0xEE);
  EXPECT_FALSE(TranscodeEtcToBc(EtcFormat::kEtc2Rgba8, etc.data(), 24,
                                out.data(), out.size(), true));
  EXPECT_FALSE(TranscodeEtcToBc(EtcFormat::kEtc2Rgba8, etc.data(), etc.size(),
                                out.data(), etc.size() - 1, true));
  EXPECT_TRUE(
      TranscodeEtcToBc(EtcFormat::kEtc2Rgba8, etc.data(), 0, nullptr, 0, true));
  EXPECT_EQ(out, std::vector<std::uint8_t>(etc.size(), 0xEE));

  std::vector<std::uint8_t> rgba(5 * 5 * 4, 100);
  EXPECT_EQ(BcEncodedBytes(EtcFormat::kEtc2Rgb8, 5, 5, 1), 32U);
  EXPECT_EQ(BcEncodedBytes(EtcFormat::kEtc2Rgba8, 5, 5, 3), 192U);
  EXPECT_EQ(BcEncodedBytes(EtcFormat::kEacR11, 1, 1, 1), 8U);
  EXPECT_FALSE(TranscodeDecodedToBc(EtcFormat::kEtc2Rgb8, rgba.data(), 5, 5, 1,
                                    out.data(), 31, true));
  EXPECT_FALSE(TranscodeDecodedToBc(EtcFormat::kEtc2Rgb8, rgba.data(), 0, 5, 1,
                                    out.data(), out.size(), true));
  EXPECT_EQ(out, std::vector<std::uint8_t>(etc.size(), 0xEE));
}

}  // namespace
}  // namespace mocktail::graphics
