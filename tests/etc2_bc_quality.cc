// What ETC2->BC mode costs in quality (DESIGN WP3 acceptance, CR-12): PSNR
// of the BC images against the ETC2/EAC-decoded texels that the default
// mode shows. A manual tool, not a ctest:
//
//   etc2_bc_quality <corpus-dir> [--blocks <trace-dir>] [--max-png N]
//
// <corpus-dir> is searched for ETC1 .ktx files (Roblox's APK ships 26 under
// assets/android/textures) and .png files, which are re-encoded to ETC2
// RGB8 and RGBA8 and to EAC R11 and RG11 by the simple test encoders below
// (individual and differential ETC1 modes only, so this corpus is smoother
// than a production encoder's output). --blocks reads a
// MOCKTAIL_ETC2_TRACE_BLOCKS directory (etc2_blocks.bin and .csv): Roblox's
// own compressed uploads. The BC side is bc_encoder, which gives the GPU
// route's bytes, so these numbers hold for both routes. The decoders are
// the reference ones of tests/support/bc_reference.h (D3D10 rules).
//
// Exit status 1 when a CR-12 floor is missed (refined encoder, APK
// corpus): RGB8->BC1 PNG mean >= 45 dB and p10 >= 32 dB, RGBA8->BC3 alpha
// p10 >= 37 dB, R11->BC4 and RG11->BC5 p10 >= 41 dB, and the real-content
// floor, the APK .ktx at BC1 mean >= 35.0 dB.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "mocktail/graphics/bc_encoder.h"
#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/texture_override.h"
#include "support/bc_reference.h"

namespace {

using mocktail::graphics::DecodeEtcImage;
using mocktail::graphics::EtcBlockBytes;
using mocktail::graphics::EtcDecodedTexelBytes;
using mocktail::graphics::EtcFormat;

const char* const kNames[7] = {"RGB8->BC1",  "RGB8A1->BC1a", "RGBA8->BC3",
                               "R11->BC4",   "R11s->BC4s",   "RG11->BC5",
                               "RG11s->BC5s"};

// One image of bw x bh tightly packed ETC2/EAC blocks, and its BC form.
struct Image {
  EtcFormat format = EtcFormat::kEtc2Rgb8;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> blocks;
  std::vector<std::uint8_t> bc;
};

std::uint32_t LoadWord(const std::uint8_t* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         static_cast<std::uint32_t>(bytes[1]) << 8 |
         static_cast<std::uint32_t>(bytes[2]) << 16 |
         static_cast<std::uint32_t>(bytes[3]) << 24;
}

void Encode(std::vector<Image>* images, bool refine) {
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> threads;
  const unsigned count = std::max(1U, std::thread::hardware_concurrency() / 2);
  for (unsigned thread = 0; thread < count; ++thread) {
    threads.emplace_back([&] {
      for (std::size_t index; (index = next.fetch_add(1)) < images->size();) {
        Image& image = (*images)[index];
        image.bc.resize(image.blocks.size());
        mocktail::graphics::TranscodeEtcToBc(
            image.format, image.blocks.data(), image.blocks.size(),
            image.bc.data(), image.bc.size(), refine);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
}

double Psnr(double mse, double peak) {
  return mse <= 0 ? 99.0 : 10.0 * std::log10(peak * peak / mse);
}

struct Stats {
  std::vector<double> psnr;
  double Mean() const {
    double sum = 0;
    for (const double value : psnr) {
      sum += value;
    }
    return psnr.empty() ? 0 : sum / static_cast<double>(psnr.size());
  }
  double P10() const {
    std::vector<double> sorted = psnr;
    std::sort(sorted.begin(), sorted.end());
    return sorted.empty() ? 0 : sorted[sorted.size() / 10];
  }
  void Print(const std::string& name) const {
    if (psnr.empty()) {
      return;
    }
    std::vector<double> sorted = psnr;
    std::sort(sorted.begin(), sorted.end());
    std::printf(
        "  %-34s n=%4zu mean %.2f dB  median %.2f  p10 %.2f  min %.2f\n",
        name.c_str(), sorted.size(), Mean(), sorted[sorted.size() / 2], P10(),
        sorted.front());
  }
};

// Colour (and the alpha of RGBA8) against the ETC decode, over the image's
// own texels (block padding excluded).
void CompareColour(const Image& image, Stats* rgb, Stats* alpha) {
  const std::uint32_t w = image.width;
  const std::uint32_t h = image.height;
  std::vector<std::uint8_t> reference(static_cast<std::size_t>(w) * h * 4);
  DecodeEtcImage(image.format, image.blocks.data(), image.blocks.size(), w, h,
                 reference.data(), reference.size());
  const bool bc3 = image.format == EtcFormat::kEtc2Rgba8;
  const std::size_t block_bytes = EtcBlockBytes(image.format);
  const std::uint32_t blocks_wide = (w + 3) / 4;
  double colour_error = 0;
  double alpha_error = 0;
  for (std::uint32_t by = 0; by < (h + 3) / 4; ++by) {
    for (std::uint32_t bx = 0; bx < blocks_wide; ++bx) {
      const std::uint8_t* block =
          image.bc.data() + (by * blocks_wide + bx) * block_bytes;
      const std::uint8_t* colour = bc3 ? block + 8 : block;
      const std::uint32_t words[2] = {LoadWord(colour), LoadWord(colour + 4)};
      std::uint8_t texels[64];
      bcref::DecodeBc1(words, bc3, texels);
      int alpha_values[16] = {};
      if (bc3) {
        const std::uint32_t alpha_words[2] = {LoadWord(block),
                                              LoadWord(block + 4)};
        bcref::DecodeBc4Fixed(alpha_words, false, alpha_values);
      }
      for (int i = 0; i < 16; ++i) {
        const std::uint32_t x = bx * 4 + (i & 3);
        const std::uint32_t y = by * 4 + (i >> 2);
        if (x >= w || y >= h) {
          continue;
        }
        const std::uint8_t* expected =
            reference.data() + (static_cast<std::size_t>(y) * w + x) * 4;
        for (int c = 0; c < 3; ++c) {
          const double d = static_cast<double>(texels[i * 4 + c]) - expected[c];
          colour_error += d * d;
        }
        const double d =
            bc3 ? alpha_values[i] / 256.0 - expected[3]
                : static_cast<double>(texels[i * 4 + 3]) - expected[3];
        alpha_error += d * d;
      }
    }
  }
  const double texels = static_cast<double>(w) * h;
  rgb->psnr.push_back(Psnr(colour_error / (3 * texels), 255.0));
  if (alpha != nullptr) {
    alpha->psnr.push_back(Psnr(alpha_error / texels, 255.0));
  }
}

// EAC against the R16 decode, both in [0, 1] (unorm) or [-1, 1] (snorm).
void CompareEac(const Image& image, Stats* stats) {
  const std::uint32_t w = image.width;
  const std::uint32_t h = image.height;
  const bool is_signed = image.format == EtcFormat::kEacR11Signed ||
                         image.format == EtcFormat::kEacRg11Signed;
  const int channels = EtcBlockBytes(image.format) / 8;
  std::vector<std::uint8_t> reference(static_cast<std::size_t>(w) * h *
                                      channels * 2);
  DecodeEtcImage(image.format, image.blocks.data(), image.blocks.size(), w, h,
                 reference.data(), reference.size());
  const std::uint32_t blocks_wide = (w + 3) / 4;
  double error = 0;
  for (std::uint32_t by = 0; by < (h + 3) / 4; ++by) {
    for (std::uint32_t bx = 0; bx < blocks_wide; ++bx) {
      for (int channel = 0; channel < channels; ++channel) {
        const std::uint8_t* block =
            image.bc.data() +
            ((by * blocks_wide + bx) * channels + channel) * 8;
        const std::uint32_t words[2] = {LoadWord(block), LoadWord(block + 4)};
        int values[16];
        bcref::DecodeBc4Fixed(words, is_signed, values);
        for (int i = 0; i < 16; ++i) {
          const std::uint32_t x = bx * 4 + (i & 3);
          const std::uint32_t y = by * 4 + (i >> 2);
          if (x >= w || y >= h) {
            continue;
          }
          const std::uint8_t* texel =
              reference.data() +
              ((static_cast<std::size_t>(y) * w + x) * channels + channel) * 2;
          const int raw = texel[0] | (texel[1] << 8);
          const double expected =
              is_signed
                  ? std::max(-1.0, static_cast<std::int16_t>(raw) / 32767.0)
                  : raw / 65535.0;
          const double actual = is_signed ? values[i] / 256.0 / 127.0 - 1.0
                                          : values[i] / 256.0 / 255.0;
          error += (expected - actual) * (expected - actual);
        }
      }
    }
  }
  const double samples = static_cast<double>(w) * h * channels;
  stats->psnr.push_back(Psnr(error / samples, is_signed ? 2.0 : 1.0));
}

// ---- test-data ETC encoders (from plus2-design/harness/bc_test.cc) ------

const int kModifiers[8][4] = {{2, 8, -2, -8},       {5, 17, -5, -17},
                              {9, 29, -9, -29},     {13, 42, -13, -42},
                              {18, 60, -18, -60},   {24, 80, -24, -80},
                              {33, 106, -33, -106}, {47, 183, -47, -183}};
const int kEacModifiers[16][8] = {
    {-3, -6, -9, -15, 2, 5, 8, 14}, {-3, -7, -10, -13, 2, 6, 9, 12},
    {-2, -5, -8, -13, 1, 4, 7, 12}, {-2, -4, -6, -13, 1, 3, 5, 12},
    {-3, -6, -8, -12, 2, 5, 7, 11}, {-3, -7, -9, -11, 2, 6, 8, 10},
    {-4, -7, -8, -11, 3, 6, 7, 10}, {-3, -5, -8, -11, 2, 4, 7, 10},
    {-2, -6, -8, -10, 1, 5, 7, 9},  {-2, -5, -8, -10, 1, 4, 7, 9},
    {-2, -4, -8, -10, 1, 3, 7, 9},  {-2, -5, -7, -10, 1, 4, 6, 9},
    {-3, -4, -7, -10, 2, 3, 6, 9},  {-1, -2, -3, -10, 0, 1, 2, 9},
    {-4, -6, -8, -9, 3, 5, 7, 8},   {-3, -5, -7, -9, 2, 4, 6, 8}};

// An ETC1 block (individual or differential mode) for 16 row-major texels.
void EncodeEtc1(const std::uint8_t* t, std::uint8_t* out) {
  long best_error = -1;
  for (int flip = 0; flip < 2; ++flip) {
    int sum[2][3] = {};
    for (int y = 0; y < 4; ++y) {
      for (int x = 0; x < 4; ++x) {
        const int half = flip ? (y >= 2) : (x >= 2);
        for (int c = 0; c < 3; ++c) {
          sum[half][c] += t[(y * 4 + x) * 4 + c];
        }
      }
    }
    for (int diff = 0; diff < 2; ++diff) {
      int q[2][3];
      int base[2][3];
      bool ok = true;
      for (int half = 0; half < 2; ++half) {
        for (int c = 0; c < 3; ++c) {
          const int average = (sum[half][c] + 4) / 8;
          if (diff) {
            q[half][c] = (average * 31 + 127) / 255;
            base[half][c] = (q[half][c] << 3) | (q[half][c] >> 2);
          } else {
            q[half][c] = (average * 15 + 127) / 255;
            base[half][c] = (q[half][c] << 4) | q[half][c];
          }
        }
      }
      if (diff) {
        for (int c = 0; c < 3; ++c) {
          const int d = q[1][c] - q[0][c];
          ok = ok && d >= -4 && d <= 3;
        }
      }
      if (!ok) {
        continue;
      }
      long error = 0;
      int table[2] = {0, 0};
      std::uint32_t msb = 0;
      std::uint32_t lsb = 0;
      for (int half = 0; half < 2; ++half) {
        long best_table = -1;
        std::uint32_t table_msb = 0;
        std::uint32_t table_lsb = 0;
        for (int tb = 0; tb < 8; ++tb) {
          long e = 0;
          std::uint32_t m = 0;
          std::uint32_t l = 0;
          for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
              if ((flip ? (y >= 2) : (x >= 2)) != half) {
                continue;
              }
              int best_index = 0;
              long best_e = -1;
              for (int i = 0; i < 4; ++i) {
                long ee = 0;
                for (int c = 0; c < 3; ++c) {
                  const int v =
                      std::clamp(base[half][c] + kModifiers[tb][i], 0, 255) -
                      t[(y * 4 + x) * 4 + c];
                  ee += v * v;
                }
                if (best_e < 0 || ee < best_e) {
                  best_e = ee;
                  best_index = i;
                }
              }
              e += best_e;
              const int texel = x * 4 + y;
              m |= static_cast<std::uint32_t>(best_index >> 1) << texel;
              l |= static_cast<std::uint32_t>(best_index & 1) << texel;
            }
          }
          if (best_table < 0 || e < best_table) {
            best_table = e;
            table[half] = tb;
            table_msb = m;
            table_lsb = l;
          }
        }
        error += best_table;
        msb |= table_msb;
        lsb |= table_lsb;
      }
      if (best_error < 0 || error < best_error) {
        best_error = error;
        for (int c = 0; c < 3; ++c) {
          out[c] = diff ? static_cast<std::uint8_t>((q[0][c] << 3) |
                                                    ((q[1][c] - q[0][c]) & 7))
                        : static_cast<std::uint8_t>((q[0][c] << 4) | q[1][c]);
        }
        out[3] = static_cast<std::uint8_t>((table[0] << 5) | (table[1] << 2) |
                                           (diff << 1) | flip);
        out[4] = static_cast<std::uint8_t>(msb >> 8);
        out[5] = static_cast<std::uint8_t>(msb & 0xFF);
        out[6] = static_cast<std::uint8_t>(lsb >> 8);
        out[7] = static_cast<std::uint8_t>(lsb & 0xFF);
      }
    }
  }
}

// An EAC block for 16 row-major values: 8-bit alpha, or with `r11` 11-bit.
void EncodeEac(const int* v, bool r11, std::uint8_t* out) {
  long best = -1;
  int low = v[0];
  int high = v[0];
  for (int i = 1; i < 16; ++i) {
    low = std::min(low, v[i]);
    high = std::max(high, v[i]);
  }
  for (int tb = 0; tb < 16; ++tb) {
    const int table_span = kEacModifiers[tb][7] - kEacModifiers[tb][3];
    const int range = r11 ? (high - low + 7) / 8 : (high - low);
    const int guess = std::clamp((range + table_span / 2) / table_span, 1, 15);
    for (int multiplier = std::max(1, guess - 1);
         multiplier <= std::min(15, guess + 1); ++multiplier) {
      const int scale = r11 ? multiplier * 8 : multiplier;
      const int center = (low + high) / 2;
      for (int db = -1; db <= 1; ++db) {
        int b = r11 ? (center - 4) / 8 + db : center + db;
        b = std::clamp(b, 0, 255);
        long e = 0;
        std::uint64_t bits = 0;
        for (int y = 0; y < 4; ++y) {
          for (int x = 0; x < 4; ++x) {
            const int target = v[y * 4 + x];
            long best_e = -1;
            int best_index = 0;
            for (int i = 0; i < 8; ++i) {
              const int value =
                  r11 ? std::clamp(b * 8 + 4 + kEacModifiers[tb][i] * scale, 0,
                                   2047)
                      : std::clamp(b + kEacModifiers[tb][i] * multiplier, 0,
                                   255);
              const long d = value - target;
              if (best_e < 0 || d * d < best_e) {
                best_e = d * d;
                best_index = i;
              }
            }
            e += best_e;
            const int texel = x * 4 + y;
            bits |= static_cast<std::uint64_t>(best_index) << (45 - texel * 3);
          }
        }
        if (best < 0 || e < best) {
          best = e;
          out[0] = static_cast<std::uint8_t>(b);
          out[1] = static_cast<std::uint8_t>((multiplier << 4) | tb);
          for (int i = 0; i < 6; ++i) {
            out[2 + i] =
                static_cast<std::uint8_t>((bits >> (40 - 8 * i)) & 0xFF);
          }
        }
      }
    }
  }
}

struct Rgba {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::vector<std::uint8_t> pixels;
};

Image EtcFromRgba(const Rgba& rgba, EtcFormat format) {
  Image image;
  image.format = format;
  image.width = rgba.width;
  image.height = rgba.height;
  const std::uint32_t blocks_wide = (rgba.width + 3) / 4;
  const std::uint32_t blocks_high = (rgba.height + 3) / 4;
  const std::size_t block_bytes = EtcBlockBytes(format);
  image.blocks.resize(static_cast<std::size_t>(blocks_wide) * blocks_high *
                      block_bytes);
  for (std::uint32_t by = 0; by < blocks_high; ++by) {
    for (std::uint32_t bx = 0; bx < blocks_wide; ++bx) {
      std::uint8_t t[64];
      for (int i = 0; i < 16; ++i) {
        const std::uint32_t x = std::min(rgba.width - 1, bx * 4 + (i & 3));
        const std::uint32_t y = std::min(rgba.height - 1, by * 4 + (i >> 2));
        std::memcpy(t + i * 4,
                    rgba.pixels.data() +
                        (static_cast<std::size_t>(y) * rgba.width + x) * 4,
                    4);
      }
      std::uint8_t* out =
          image.blocks.data() + (by * blocks_wide + bx) * block_bytes;
      if (format == EtcFormat::kEtc2Rgb8) {
        EncodeEtc1(t, out);
      } else if (format == EtcFormat::kEtc2Rgba8) {
        int alpha[16];
        for (int i = 0; i < 16; ++i) {
          alpha[i] = t[i * 4 + 3];
        }
        EncodeEac(alpha, false, out);
        EncodeEtc1(t, out + 8);
      } else {  // R11 from luminance, RG11 from R and G
        const int channels = format == EtcFormat::kEacRg11 ? 2 : 1;
        for (int channel = 0; channel < channels; ++channel) {
          int values[16];
          for (int i = 0; i < 16; ++i) {
            const int value = channels == 1
                                  ? (t[i * 4] * 77 + t[i * 4 + 1] * 150 +
                                     t[i * 4 + 2] * 29) >>
                                        8
                                  : t[i * 4 + channel];
            values[i] = value * 2047 / 255;
          }
          EncodeEac(values, true, out + channel * 8);
        }
      }
    }
  }
  return image;
}

// The level-0 blocks of an ETC1 .ktx (glInternalFormat ETC1_RGB8_OES).
bool ReadKtxEtc1(const std::string& path, Image* image) {
  std::ifstream in(path, std::ios::binary);
  const std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
  if (data.size() < 68) {
    return false;
  }
  std::uint32_t header[13];
  std::memcpy(header, data.data() + 12, sizeof(header));
  const std::size_t offset = 64 + static_cast<std::size_t>(header[12]);
  if (offset + 4 > data.size()) {
    return false;
  }
  std::uint32_t size = 0;
  std::memcpy(&size, data.data() + offset, 4);
  if (offset + 4 + size > data.size()) {
    return false;
  }
  image->format = EtcFormat::kEtc2Rgb8;
  image->width = header[6];
  image->height = header[7];
  image->blocks.assign(
      data.begin() + static_cast<std::ptrdiff_t>(offset + 4),
      data.begin() + static_cast<std::ptrdiff_t>(offset + 4 + size));
  return image->width != 0 && image->height != 0 &&
         image->blocks.size() ==
             static_cast<std::size_t>((image->width + 3) / 4) *
                 ((image->height + 3) / 4) * 8;
}

// The uploads of a MOCKTAIL_ETC2_TRACE_BLOCKS directory, by format: each
// layer of each upload as one image.
bool ReadTraceBlocks(const std::string& directory, std::vector<Image>* images) {
  std::ifstream index(directory + "/etc2_blocks.csv");
  std::ifstream blob(directory + "/etc2_blocks.bin", std::ios::binary);
  if (!index || !blob) {
    return false;
  }
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(blob)),
                                        std::istreambuf_iterator<char>());
  std::string line;
  std::getline(index, line);  // header
  while (std::getline(index, line)) {
    // offset,bytes,vkformat,w,h,mip,layers,hash64
    unsigned long long offset = 0;
    unsigned long long size = 0;
    int vk_format = 0;
    unsigned width = 0;
    unsigned height = 0;
    unsigned mip = 0;
    unsigned layers = 0;
    if (std::sscanf(line.c_str(), "%llu,%llu,%d,%u,%u,%u,%u", &offset, &size,
                    &vk_format, &width, &height, &mip, &layers) != 7 ||
        layers == 0 || offset + size > bytes.size()) {
      continue;
    }
    // VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK (147) .. EAC_R11G11_SNORM (156).
    static const EtcFormat kByVkFormat[10] = {
        EtcFormat::kEtc2Rgb8,   EtcFormat::kEtc2Rgb8,
        EtcFormat::kEtc2Rgb8A1, EtcFormat::kEtc2Rgb8A1,
        EtcFormat::kEtc2Rgba8,  EtcFormat::kEtc2Rgba8,
        EtcFormat::kEacR11,     EtcFormat::kEacR11Signed,
        EtcFormat::kEacRg11,    EtcFormat::kEacRg11Signed};
    if (vk_format < 147 || vk_format > 156) {
      continue;
    }
    const EtcFormat format = kByVkFormat[vk_format - 147];
    const std::size_t layer_bytes = static_cast<std::size_t>(size) / layers;
    if (layer_bytes != static_cast<std::size_t>((width + 3) / 4) *
                           ((height + 3) / 4) * EtcBlockBytes(format)) {
      continue;
    }
    for (unsigned layer = 0; layer < layers; ++layer) {
      Image image;
      image.format = format;
      image.width = width;
      image.height = height;
      const auto first = bytes.begin() + static_cast<std::ptrdiff_t>(
                                             offset + layer * layer_bytes);
      image.blocks.assign(first,
                          first + static_cast<std::ptrdiff_t>(layer_bytes));
      images->push_back(std::move(image));
    }
  }
  return true;
}

struct Floor {
  const char* what;
  double value;
  double minimum;
};

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string corpus;
  std::string trace;
  std::size_t max_png = 1000;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--blocks" && index + 1 < argc) {
      trace = argv[++index];
    } else if (argument == "--max-png" && index + 1 < argc) {
      max_png = std::strtoul(argv[++index], nullptr, 10);
    } else if (corpus.empty()) {
      corpus = argument;
    } else {
      std::fprintf(stderr, "unexpected argument: %s\n", argument.c_str());
      return 2;
    }
  }
  if (corpus.empty()) {
    std::fprintf(stderr,
                 "usage: %s <corpus-dir> [--blocks <trace-dir>] "
                 "[--max-png N]\n",
                 argv[0]);
    return 2;
  }
  std::vector<std::string> ktx;
  std::vector<std::string> png;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(corpus)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string path = entry.path().string();
    const std::string extension = entry.path().extension().string();
    if (extension == ".ktx") {
      ktx.push_back(path);
    } else if (extension == ".png") {
      png.push_back(path);
    }
  }
  std::sort(ktx.begin(), ktx.end());
  std::sort(png.begin(), png.end());

  std::vector<Image> apk;
  for (const std::string& path : ktx) {
    Image image;
    if (ReadKtxEtc1(path, &image)) {
      apk.push_back(std::move(image));
    }
  }
  // PNGs: the top-left 256x256 of each one at least 16x16, re-encoded.
  std::vector<Rgba> pngs;
  for (const std::string& path : png) {
    if (pngs.size() >= max_png) {
      break;
    }
    mocktail::graphics::RgbaImage decoded;
    if (!mocktail::graphics::ReadPngRgba(path, &decoded) ||
        decoded.width < 16 || decoded.height < 16) {
      continue;
    }
    Rgba crop;
    crop.width = std::min<std::uint32_t>(decoded.width, 256);
    crop.height = std::min<std::uint32_t>(decoded.height, 256);
    crop.pixels.resize(static_cast<std::size_t>(crop.width) * crop.height * 4);
    for (std::uint32_t y = 0; y < crop.height; ++y) {
      std::memcpy(
          crop.pixels.data() + static_cast<std::size_t>(y) * crop.width * 4,
          decoded.pixels.data() +
              static_cast<std::size_t>(y) * decoded.width * 4,
          static_cast<std::size_t>(crop.width) * 4);
    }
    pngs.push_back(std::move(crop));
  }
  const EtcFormat kPngFormats[] = {EtcFormat::kEtc2Rgb8, EtcFormat::kEtc2Rgba8,
                                   EtcFormat::kEacR11, EtcFormat::kEacRg11};
  std::vector<std::vector<Image>> reencoded;
  for (const EtcFormat format : kPngFormats) {
    std::vector<Image> images(pngs.size());
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> threads;
    const unsigned count =
        std::max(1U, std::thread::hardware_concurrency() / 2);
    for (unsigned thread = 0; thread < count; ++thread) {
      threads.emplace_back([&] {
        for (std::size_t index; (index = next.fetch_add(1)) < pngs.size();) {
          images[index] = EtcFromRgba(pngs[index], format);
        }
      });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
    reencoded.push_back(std::move(images));
  }
  std::vector<Image> traced;
  if (!trace.empty() && !ReadTraceBlocks(trace, &traced)) {
    std::fprintf(stderr, "cannot read %s/etc2_blocks.{bin,csv}\n",
                 trace.c_str());
    return 2;
  }
  std::printf("corpus %s: %zu ETC1 .ktx, %zu PNG; traced uploads: %zu\n",
              corpus.c_str(), apk.size(), pngs.size(), traced.size());

  std::vector<Floor> floors;
  for (const bool refine : {false, true}) {
    std::printf("refine=%d\n", refine ? 1 : 0);
    Encode(&apk, refine);
    Stats apk_stats;
    for (const Image& image : apk) {
      CompareColour(image, &apk_stats, nullptr);
    }
    apk_stats.Print("APK ETC1 .ktx (RGB8->BC1)");
    if (refine) {
      floors.push_back({"APK .ktx BC1 mean", apk_stats.Mean(), 35.0});
    }
    for (std::size_t index = 0; index < reencoded.size(); ++index) {
      std::vector<Image>& images = reencoded[index];
      const EtcFormat format = kPngFormats[index];
      Encode(&images, refine);
      Stats rgb;
      Stats alpha;
      Stats eac;
      for (const Image& image : images) {
        if (format == EtcFormat::kEtc2Rgb8 || format == EtcFormat::kEtc2Rgba8) {
          CompareColour(image, &rgb,
                        format == EtcFormat::kEtc2Rgba8 ? &alpha : nullptr);
        } else {
          CompareEac(image, &eac);
        }
      }
      const std::string name =
          std::string("APK PNG -> ") + kNames[static_cast<int>(format)];
      rgb.Print(name + " rgb");
      alpha.Print(name + " alpha");
      eac.Print(name);
      if (!refine) {
        continue;
      }
      if (format == EtcFormat::kEtc2Rgb8) {
        floors.push_back({"PNG BC1 mean", rgb.Mean(), 45.0});
        floors.push_back({"PNG BC1 p10", rgb.P10(), 32.0});
      } else if (format == EtcFormat::kEtc2Rgba8) {
        floors.push_back({"PNG BC3 alpha p10", alpha.P10(), 37.0});
      } else {
        floors.push_back(
            {format == EtcFormat::kEacR11 ? "PNG BC4 p10" : "PNG BC5 p10",
             eac.P10(), 41.0});
      }
    }
    if (!traced.empty()) {
      Encode(&traced, refine);
      Stats colour[7];
      Stats alpha;
      for (const Image& image : traced) {
        const int format = static_cast<int>(image.format);
        if (format <= 2) {
          CompareColour(
              image, &colour[format],
              image.format == EtcFormat::kEtc2Rgba8 ? &alpha : nullptr);
        } else {
          CompareEac(image, &colour[format]);
        }
      }
      for (int format = 0; format < 7; ++format) {
        colour[format].Print(std::string("traced ") + kNames[format]);
      }
      alpha.Print("traced RGBA8->BC3 alpha");
    }
  }
  bool ok = true;
  std::printf("CR-12 floors (refine=1):\n");
  for (const Floor& floor : floors) {
    const bool met = floor.value >= floor.minimum;
    ok = ok && met;
    std::printf("  %-20s %6.2f dB >= %5.1f  %s\n", floor.what, floor.value,
                floor.minimum, met ? "ok" : "MISSED");
  }
  return ok ? 0 : 1;
}
