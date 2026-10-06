#include "support/bc_reference_transcode.h"

#include <algorithm>
#include <cstring>

#include "support/bc_reference.h"
#include "support/etc2_reference_decoder.h"

namespace mocktail::graphics::reference {
namespace {

void StoreLe(const std::uint32_t* words, std::size_t count, std::uint8_t* out) {
  for (std::size_t i = 0; i < count * 4; ++i) {
    out[i] = static_cast<std::uint8_t>(words[i / 4] >> (8 * (i % 4)));
  }
}

}  // namespace

void BcBlockFromTexels(graphics::EtcFormat format, const std::uint8_t* texels,
                       bool refine, std::uint8_t* out) {
  std::uint32_t words[4] = {};
  switch (format) {
    case graphics::EtcFormat::kEtc2Rgb8:
    case graphics::EtcFormat::kEtc2Rgb8A1:
      bcref::EncodeBc1(texels, format == graphics::EtcFormat::kEtc2Rgb8A1,
                       false, refine, words);
      StoreLe(words, 2, out);
      return;
    case graphics::EtcFormat::kEtc2Rgba8: {
      int alpha[16];
      for (int i = 0; i < 16; ++i) {
        alpha[i] = texels[i * 4 + 3];
      }
      bcref::EncodeBc4(alpha, false, false, words);
      bcref::EncodeBc1(texels, false, true, refine, words + 2);
      StoreLe(words, 4, out);
      return;
    }
    default:
      break;
  }
  const bool is_signed = format == graphics::EtcFormat::kEacR11Signed ||
                         format == graphics::EtcFormat::kEacRg11Signed;
  const int channels = static_cast<int>(reference::EtcBlockBytes(format) / 8);
  for (int channel = 0; channel < channels; ++channel) {
    int values[16];
    for (int i = 0; i < 16; ++i) {
      const std::uint8_t* texel = texels + (i * channels + channel) * 2;
      const auto raw = static_cast<std::uint16_t>(texel[0] | (texel[1] << 8));
      values[i] = is_signed ? static_cast<std::int16_t>(raw) : raw;
    }
    bcref::EncodeBc4(values, true, is_signed, words + channel * 2);
  }
  StoreLe(words, static_cast<std::size_t>(channels) * 2, out);
}

void BcBlockFromEtc(graphics::EtcFormat format, const std::uint8_t* block,
                    bool refine, std::uint8_t* out) {
  std::uint8_t decoded[64];
  DecodeEtcImage(ToReference(format), block, reference::EtcBlockBytes(format),
                 4, 4, decoded, sizeof(decoded));
  BcBlockFromTexels(format, decoded, refine, out);
}

std::vector<std::uint8_t> BcFromEtc(graphics::EtcFormat format,
                                    const std::uint8_t* blocks,
                                    std::size_t bytes, bool refine) {
  const std::size_t block_bytes = reference::EtcBlockBytes(format);
  std::vector<std::uint8_t> out(bytes);
  for (std::size_t offset = 0; offset + block_bytes <= bytes;
       offset += block_bytes) {
    BcBlockFromEtc(format, blocks + offset, refine, out.data() + offset);
  }
  return out;
}

std::vector<std::uint8_t> BcFromTexels(graphics::EtcFormat format,
                                       const std::uint8_t* texels,
                                       std::uint32_t width,
                                       std::uint32_t height,
                                       std::uint32_t layers, bool refine) {
  const std::size_t texel_bytes = reference::EtcDecodedTexelBytes(format);
  const std::size_t block_bytes = reference::EtcBlockBytes(format);
  const std::uint32_t blocks_wide = (width + 3) / 4;
  const std::uint32_t blocks_high = (height + 3) / 4;
  std::vector<std::uint8_t> out(static_cast<std::size_t>(blocks_wide) *
                                blocks_high * layers * block_bytes);
  std::uint8_t* next = out.data();
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    const std::uint8_t* image =
        texels + static_cast<std::size_t>(layer) * width * height * texel_bytes;
    for (std::uint32_t by = 0; by < blocks_high; ++by) {
      for (std::uint32_t bx = 0; bx < blocks_wide; ++bx) {
        std::uint8_t block[64];
        for (std::uint32_t texel = 0; texel < 16; ++texel) {
          const std::uint32_t x = std::min(width - 1, bx * 4 + texel % 4);
          const std::uint32_t y = std::min(height - 1, by * 4 + texel / 4);
          std::memcpy(block + texel * texel_bytes,
                      image + (static_cast<std::size_t>(y) * width + x) *
                                  texel_bytes,
                      texel_bytes);
        }
        BcBlockFromTexels(format, block, refine, next);
        next += block_bytes;
      }
    }
  }
  return out;
}

}  // namespace mocktail::graphics::reference
