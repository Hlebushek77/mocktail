#include "mocktail/graphics/bc_encoder.h"

#include <algorithm>
#include <cstring>
#include <utility>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

// The arithmetic is plus2-design/harness/bc_ref.h's (the GPU's
// bc_common.glsl is its GLSL port): integers only, and divisions only see
// non-negative operands, so every compiler and every GPU get the same bits.
// Three places compute the same integers another way, for speed (about 2x
// over the reference on one core): the palette search and the refinement
// sums use SSE2 where the compiler targets it (always on x86-64), the
// refinement derives its second weight's sums from the first's, and BC4
// divides by a reciprocal that is exact over its whole input range.
// bc_encoder_test checks this file against a frozen copy of the reference
// over random blocks of every format; the scalar fallbacks pass it too
// (built with -U__SSE2__).
namespace mocktail::graphics {
namespace {

// round(v * 31 / 255) and round(v * 63 / 255), and the expansion D3D10
// decoders apply to the stored 5- and 6-bit values.
int Quant5(int v) { return (v * 31 + 127) / 255; }
int Quant6(int v) { return (v * 63 + 127) / 255; }
int Unq5(int q) { return (q << 3) | (q >> 2); }
int Unq6(int q) { return (q << 2) | (q >> 4); }

struct Rgb {
  int r = 0;
  int g = 0;
  int b = 0;
};

std::uint32_t Pack565(const Rgb& c) {
  return (static_cast<std::uint32_t>(Quant5(c.r)) << 11) |
         (static_cast<std::uint32_t>(Quant6(c.g)) << 5) |
         static_cast<std::uint32_t>(Quant5(c.b));
}

Rgb Unpack565(std::uint32_t c) {
  return {Unq5(static_cast<int>(c >> 11) & 31),
          Unq6(static_cast<int>(c >> 5) & 63), Unq5(static_cast<int>(c) & 31)};
}

// The palette as D3D10 and later hardware decode it: thirds, or a half
// and black in 3-colour mode, rounded.
void Bc1Palette(std::uint32_t c0, std::uint32_t c1, bool four, Rgb p[4]) {
  p[0] = Unpack565(c0);
  p[1] = Unpack565(c1);
  if (four) {
    p[2] = {(2 * p[0].r + p[1].r + 1) / 3, (2 * p[0].g + p[1].g + 1) / 3,
            (2 * p[0].b + p[1].b + 1) / 3};
    p[3] = {(p[0].r + 2 * p[1].r + 1) / 3, (p[0].g + 2 * p[1].g + 1) / 3,
            (p[0].b + 2 * p[1].b + 1) / 3};
  } else {
    p[2] = {(p[0].r + p[1].r + 1) / 2, (p[0].g + p[1].g + 1) / 2,
            (p[0].b + p[1].b + 1) / 2};
    p[3] = {0, 0, 0};
  }
}

// One block's colour, a plane per channel, and its transparent texels
// (-1, else 0).
struct ColourBlock {
  alignas(16) std::int16_t r[16];
  alignas(16) std::int16_t g[16];
  alignas(16) std::int16_t b[16];
  alignas(16) std::int32_t transparent[16];
};

// Indices (2 bits per texel, texel 0 lowest; and one per texel in `index`)
// for endpoints c0, c1 and their squared error; transparent texels take
// index 3 and add nothing. Ties keep the lower index.
int Bc1Choose(std::uint32_t c0, std::uint32_t c1, bool force_four,
              const ColourBlock& block, std::uint32_t* indices, int index[16]) {
  const bool four = force_four || c0 > c1;
  Rgb p[4];
  Bc1Palette(c0, c1, four, p);
  // In 3-colour mode entry 3 (black) never wins: its distance gets a bias
  // larger than any real one (3 * 255^2).
  const int last_bias = four ? 0 : 0x40000000;
  int error = 0;
#if defined(__SSE2__)
  // Four texels per register. _mm_madd_epi16 on (dr, dg) pairs gives
  // dr^2 + dg^2 exactly (differences fit in 16 bits, sums in 32), and on
  // (db, 0) pairs db^2.
  const __m128i zero = _mm_setzero_si128();
  const __m128i r[2] = {
      _mm_load_si128(reinterpret_cast<const __m128i*>(block.r)),
      _mm_load_si128(reinterpret_cast<const __m128i*>(block.r + 8))};
  const __m128i g[2] = {
      _mm_load_si128(reinterpret_cast<const __m128i*>(block.g)),
      _mm_load_si128(reinterpret_cast<const __m128i*>(block.g + 8))};
  const __m128i b[2] = {
      _mm_load_si128(reinterpret_cast<const __m128i*>(block.b)),
      _mm_load_si128(reinterpret_cast<const __m128i*>(block.b + 8))};
  __m128i nearest[4];
  __m128i best[4];
  for (int k = 0; k < 4; ++k) {
    const __m128i pr = _mm_set1_epi16(static_cast<std::int16_t>(p[k].r));
    const __m128i pg = _mm_set1_epi16(static_cast<std::int16_t>(p[k].g));
    const __m128i pb = _mm_set1_epi16(static_cast<std::int16_t>(p[k].b));
    const __m128i bias = _mm_set1_epi32(k == 3 ? last_bias : 0);
    const __m128i entry = _mm_set1_epi32(k);
    for (int half = 0; half < 2; ++half) {
      const __m128i dr = _mm_sub_epi16(r[half], pr);
      const __m128i dg = _mm_sub_epi16(g[half], pg);
      const __m128i db = _mm_sub_epi16(b[half], pb);
      const __m128i rg[2] = {_mm_unpacklo_epi16(dr, dg),
                             _mm_unpackhi_epi16(dr, dg)};
      const __m128i bz[2] = {_mm_unpacklo_epi16(db, zero),
                             _mm_unpackhi_epi16(db, zero)};
      for (int quarter = 0; quarter < 2; ++quarter) {
        const int j = half * 2 + quarter;
        const __m128i distance = _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(rg[quarter], rg[quarter]),
                          _mm_madd_epi16(bz[quarter], bz[quarter])),
            bias);
        if (k == 0) {
          nearest[j] = distance;
          best[j] = zero;
          continue;
        }
        const __m128i closer = _mm_cmpgt_epi32(nearest[j], distance);
        nearest[j] = _mm_or_si128(_mm_and_si128(closer, distance),
                                  _mm_andnot_si128(closer, nearest[j]));
        best[j] = _mm_or_si128(_mm_and_si128(closer, entry),
                               _mm_andnot_si128(closer, best[j]));
      }
    }
  }
  const __m128i three = _mm_set1_epi32(3);
  __m128i sum = zero;
  for (int j = 0; j < 4; ++j) {
    const __m128i clear = _mm_load_si128(
        reinterpret_cast<const __m128i*>(block.transparent + j * 4));
    sum = _mm_add_epi32(sum, _mm_andnot_si128(clear, nearest[j]));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(index + j * 4),
                     _mm_or_si128(best[j], _mm_and_si128(clear, three)));
  }
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
  sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
  error = _mm_cvtsi128_si32(sum);
#else
  for (int i = 0; i < 16; ++i) {
    int distance[4];
    for (int k = 0; k < 4; ++k) {
      const int dr = block.r[i] - p[k].r;
      const int dg = block.g[i] - p[k].g;
      const int db = block.b[i] - p[k].b;
      distance[k] = dr * dr + dg * dg + db * db;
    }
    distance[3] += last_bias;
    int best = 0;
    int nearest = distance[0];
    for (int k = 1; k < 4; ++k) {
      best = distance[k] < nearest ? k : best;
      nearest = distance[k] < nearest ? distance[k] : nearest;
    }
    index[i] = block.transparent[i] != 0 ? 3 : best;
    error += block.transparent[i] != 0 ? 0 : nearest;
  }
#endif
  std::uint32_t bits = 0;
  for (int i = 0; i < 16; ++i) {
    bits |= static_cast<std::uint32_t>(index[i]) << (2 * i);
  }
  *indices = bits;
  return error;
}

// Moves both ends of one axis 1/16 of the range towards the middle.
void Inset(int* lo, int* hi) {
  const int d = *hi > *lo ? *hi - *lo : *lo - *hi;
  const int s = d >> 4;
  if (*hi > *lo) {
    *hi -= s;
    *lo += s;
  } else {
    *hi += s;
    *lo -= s;
  }
}

// round(n / det) for det > 0, rounding halves away from zero.
int DivideRounded(int n, int det) {
  return n >= 0 ? (n + det / 2) / det : -((-n + det / 2) / det);
}

// `rgba`: 16 row-major RGBA8 texels. `punch`: alpha < 128 is transparent
// (3-colour mode). `force_four`: BC3's colour block, always decoded in
// 4-colour mode.
void EncodeBc1Words(const std::uint8_t* rgba, bool punch, bool force_four,
                    bool refine, std::uint32_t out[2]) {
  // Branch-free over the 16 texels: transparent texels count as neither
  // minimum nor maximum and add nothing to the covariance.
  ColourBlock block;
  int transparent_count = 0;
  int low_r = 255, low_g = 255, low_b = 255;
  int high_r = 0, high_g = 0, high_b = 0;
  int sum_r = 0, sum_g = 0, sum_b = 0;
  for (int i = 0; i < 16; ++i) {
    const int r = rgba[i * 4];
    const int g = rgba[i * 4 + 1];
    const int b = rgba[i * 4 + 2];
    const bool transparent = punch && rgba[i * 4 + 3] < 128;
    block.r[i] = static_cast<std::int16_t>(r);
    block.g[i] = static_cast<std::int16_t>(g);
    block.b[i] = static_cast<std::int16_t>(b);
    block.transparent[i] = transparent ? -1 : 0;
    transparent_count += transparent ? 1 : 0;
    low_r = std::min(low_r, transparent ? 255 : r);
    low_g = std::min(low_g, transparent ? 255 : g);
    low_b = std::min(low_b, transparent ? 255 : b);
    high_r = std::max(high_r, transparent ? 0 : r);
    high_g = std::max(high_g, transparent ? 0 : g);
    high_b = std::max(high_b, transparent ? 0 : b);
    sum_r += r;
    sum_g += g;
    sum_b += b;
  }
  if (transparent_count == 16) {
    // Fully transparent: c0 = c1 = 0 (3-colour mode), every index 3.
    out[0] = 0;
    out[1] = 0xFFFFFFFFU;
    return;
  }
  const bool any_transparent = transparent_count != 0;
  Rgb low{low_r, low_g, low_b};
  Rgb high{high_r, high_g, high_b};
  // The bounding-box diagonal that follows the colour covariance.
  const Rgb center{(low.r + high.r) >> 1, (low.g + high.g) >> 1,
                   (low.b + high.b) >> 1};
  int cov_rg = 0;
  int cov_bg = 0;
  for (int i = 0; i < 16; ++i) {
    const int r = block.r[i] - center.r;
    const int g = block.transparent[i] != 0 ? 0 : block.g[i] - center.g;
    const int b = block.b[i] - center.b;
    cov_rg += r * g;
    cov_bg += b * g;
  }
  if (cov_rg < 0) {
    std::swap(low.r, high.r);
  }
  if (cov_bg < 0) {
    std::swap(low.b, high.b);
  }
  Inset(&low.r, &high.r);
  Inset(&low.g, &high.g);
  Inset(&low.b, &high.b);
  std::uint32_t c0 = Pack565(high);
  std::uint32_t c1 = Pack565(low);
  // 3-colour mode needs c0 <= c1, 4-colour mode c0 > c1 (BC3 ignores it).
  const bool three = any_transparent && !force_four;
  if (three ? c0 > c1 : c0 < c1) {
    std::swap(c0, c1);
  }
  std::uint32_t indices = 0;
  int index[16];
  int error = Bc1Choose(c0, c1, force_four, block, &indices, index);
  if (refine && !three) {
    // Least squares on the 4-colour weights (3,0,2,1)/3 and (0,3,1,2)/3:
    // index 0 weighs c0 with a = 3, 1 with 0, 2 with 2 and 3 with 1, and c1
    // with b = 3 - a. With no transparent texel here, the b sums follow
    // from the a sums: bb = 16 * 9 - 6 * sum(a) + sum(a^2),
    // ab = 3 * sum(a) - sum(a^2), bx = 3 * sum(x) - ax.
    int sum_a = 0;
    int aa = 0;
    int ax[3] = {0, 0, 0};
#if defined(__SSE2__)
    // The weights as 16-bit lanes, a = 4 - k except 3 for k = 0 and 0 for
    // k = 1; every sum is _mm_madd_epi16 against a plane.
    const __m128i one = _mm_set1_epi16(1);
    __m128i a[2];
    for (int half = 0; half < 2; ++half) {
      const __m128i k = _mm_packs_epi32(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(index + half * 8)),
          _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(index + half * 8 + 4)));
      const __m128i first = _mm_cmpeq_epi16(k, _mm_setzero_si128());
      const __m128i second = _mm_cmpeq_epi16(k, one);
      a[half] = _mm_andnot_si128(
          second, _mm_sub_epi16(_mm_sub_epi16(_mm_set1_epi16(4), k),
                                _mm_and_si128(first, one)));
    }
    const auto dot = [&a](__m128i low, __m128i high) {
      __m128i sum =
          _mm_add_epi32(_mm_madd_epi16(a[0], low), _mm_madd_epi16(a[1], high));
      sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
      sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
      return _mm_cvtsi128_si32(sum);
    };
    const auto plane = [](const std::int16_t* values, int half) {
      return _mm_load_si128(
          reinterpret_cast<const __m128i*>(values + half * 8));
    };
    sum_a = dot(one, one);
    aa = dot(a[0], a[1]);
    ax[0] = dot(plane(block.r, 0), plane(block.r, 1));
    ax[1] = dot(plane(block.g, 0), plane(block.g, 1));
    ax[2] = dot(plane(block.b, 0), plane(block.b, 1));
#else
    for (int i = 0; i < 16; ++i) {
      const int k = index[i];
      const int a = k == 0 ? 3 : k == 1 ? 0 : 4 - k;
      sum_a += a;
      aa += a * a;
      ax[0] += a * block.r[i];
      ax[1] += a * block.g[i];
      ax[2] += a * block.b[i];
    }
#endif
    const int ab = 3 * sum_a - aa;
    const int bb = 144 - 6 * sum_a + aa;
    const int bx[3] = {3 * sum_r - ax[0], 3 * sum_g - ax[1], 3 * sum_b - ax[2]};
    const int det = aa * bb - ab * ab;  // 0 when every texel has one weight
    if (det > 0) {
      int e0[3];
      int e1[3];
      for (int c = 0; c < 3; ++c) {
        e0[c] = std::clamp(DivideRounded(3 * (bb * ax[c] - ab * bx[c]), det), 0,
                           255);
        e1[c] = std::clamp(DivideRounded(3 * (aa * bx[c] - ab * ax[c]), det), 0,
                           255);
      }
      std::uint32_t r0 = Pack565({e0[0], e0[1], e0[2]});
      std::uint32_t r1 = Pack565({e1[0], e1[1], e1[2]});
      if (r0 < r1) {
        std::swap(r0, r1);
      }
      std::uint32_t refined = 0;
      const int refined_error =
          Bc1Choose(r0, r1, force_four, block, &refined, index);
      if (refined_error < error) {
        c0 = r0;
        c1 = r1;
        indices = refined;
      }
    }
  }
  out[0] = c0 | (c1 << 16);
  out[1] = indices;
}

// BC4 on 16 values: 8-bit (`is16` false; BC3 alpha), R16_UNORM, or
// R16_SNORM patterns sign-extended (`signed_values`). Endpoints are chosen
// in an 8.8 fixed-point domain u in [0, 255] (unorm) or [0, 254] (snorm).
void EncodeBc4Words(const int values[16], bool is16, bool signed_values,
                    std::uint32_t out[2]) {
  int v[16];
  for (int i = 0; i < 16; ++i) {
    if (!is16) {
      v[i] = values[i] * 256;
    } else if (!signed_values) {
      v[i] = values[i] * 256 / 257;  // v * 255 / 65535 in 8.8
    } else {
      const int s = std::max(values[i], -32767);
      v[i] = (s + 32767) * 254 * 128 / 32767;  // (s + 1) * 127 in 8.8
    }
  }
  int low = v[0];
  int high = v[0];
  for (int i = 1; i < 16; ++i) {
    low = std::min(low, v[i]);
    high = std::max(high, v[i]);
  }
  const int top = signed_values ? 254 : 255;
  int e0 = std::min(top, (high + 128) >> 8);
  int e1 = std::min(top, (low + 128) >> 8);
  std::uint64_t indices = 0;  // the 48-bit index field, texel 0 lowest
  if (e0 != e1) {
    // 8-value mode: e0 > e1. Equal endpoints leave every index 0 (= e0).
    if (e0 < e1) {
      std::swap(e0, e1);
    }
    const int base = e1 * 256;
    const int span = (e0 - e1) * 256;
    // round(d * 7 / span) = (d * 14 + span) / (2 * span) for every texel,
    // as a multiplication by a reciprocal that is exact here: the dividend
    // stays below 2^20 and the divisor is at least 512, so the error of
    // m = 2^40 / divisor + 1, under 2^-20, never reaches the next integer
    // (a quotient's fraction is at most 1 - 1 / divisor).
    const std::uint64_t reciprocal =
        (std::uint64_t{1} << 40) / static_cast<std::uint64_t>(2 * span) + 1;
    std::uint32_t index[16];
    for (int i = 0; i < 16; ++i) {
      const int d = std::clamp(v[i] - base, 0, span);
      const auto position = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(d * 14 + span) * reciprocal) >> 40);
      // Position 7 is e0 (index 0), 0 is e1 (index 1), 6..1 are indices 2..7.
      index[i] = position == 7 ? 0U : position == 0 ? 1U : 8U - position;
    }
    for (int i = 0; i < 16; ++i) {
      indices |= static_cast<std::uint64_t>(index[i]) << (3 * i);
    }
  }
  const auto endpoint = [signed_values](int e) {
    return signed_values ? static_cast<std::uint32_t>((e - 127) & 0xFF)
                         : static_cast<std::uint32_t>(e);
  };
  out[0] = endpoint(e0) | (endpoint(e1) << 8) |
           (static_cast<std::uint32_t>(indices) << 16);
  out[1] = static_cast<std::uint32_t>(indices >> 16);
}

void StoreWords(const std::uint32_t* words, std::size_t count,
                std::uint8_t* out) {
  for (std::size_t i = 0; i < count; ++i) {
    out[i * 4] = static_cast<std::uint8_t>(words[i]);
    out[i * 4 + 1] = static_cast<std::uint8_t>(words[i] >> 8);
    out[i * 4 + 2] = static_cast<std::uint8_t>(words[i] >> 16);
    out[i * 4 + 3] = static_cast<std::uint8_t>(words[i] >> 24);
  }
}

int Signed16(std::uint16_t value) {
  return static_cast<int>(value ^ 0x8000U) - 0x8000;
}

bool IsSigned(EtcFormat format) {
  return format == EtcFormat::kEacR11Signed ||
         format == EtcFormat::kEacRg11Signed;
}

int EacChannels(EtcFormat format) {
  return format == EtcFormat::kEacRg11 || format == EtcFormat::kEacRg11Signed
             ? 2
             : 1;
}

// Encodes one block from its 4x4 decoded texels: row y starts at
// texels + y * row_stride, laid out as DecodeEtcImage writes `format`.
void EncodeDecodedBlock(EtcFormat format, const std::uint8_t* texels,
                        std::size_t row_stride, bool refine,
                        std::uint8_t* out) {
  switch (format) {
    case EtcFormat::kEtc2Rgb8:
    case EtcFormat::kEtc2Rgb8A1:
    case EtcFormat::kEtc2Rgba8: {
      std::uint8_t rgba[64];
      for (int y = 0; y < 4; ++y) {
        std::memcpy(rgba + y * 16, texels + y * row_stride, 16);
      }
      if (format == EtcFormat::kEtc2Rgba8) {
        EncodeBc3Block(rgba, refine, out);
      } else {
        EncodeBc1Block(rgba, format == EtcFormat::kEtc2Rgb8A1, refine, out);
      }
      return;
    }
    case EtcFormat::kEacR11:
    case EtcFormat::kEacR11Signed:
    case EtcFormat::kEacRg11:
    case EtcFormat::kEacRg11Signed: {
      const int channels = EacChannels(format);
      const bool snorm = IsSigned(format);
      for (int channel = 0; channel < channels; ++channel) {
        int values[16];
        for (int i = 0; i < 16; ++i) {
          const std::uint8_t* texel =
              texels + (i >> 2) * row_stride +
              static_cast<std::size_t>(((i & 3) * channels + channel) * 2);
          const auto raw =
              static_cast<std::uint16_t>(texel[0] | (texel[1] << 8));
          values[i] = snorm ? Signed16(raw) : raw;
        }
        std::uint32_t words[2];
        EncodeBc4Words(values, true, snorm, words);
        StoreWords(words, 2, out + channel * 8);
      }
      return;
    }
  }
}

// Blocks decoded per DecodeEtcImage call: a 4-texel-high strip of this
// many blocks stays within 4 KiB of RGBA8 (L1), and the call overhead is
// shared by all of them.
constexpr std::size_t kStripBlocks = 64;

}  // namespace

void EncodeBc1Block(const std::uint8_t rgba[64], bool punchthrough, bool refine,
                    std::uint8_t out[8]) {
  std::uint32_t words[2];
  EncodeBc1Words(rgba, punchthrough, false, refine, words);
  StoreWords(words, 2, out);
}

void EncodeBc3Block(const std::uint8_t rgba[64], bool refine,
                    std::uint8_t out[16]) {
  int alpha[16];
  for (int i = 0; i < 16; ++i) {
    alpha[i] = rgba[i * 4 + 3];
  }
  std::uint32_t words[4];
  EncodeBc4Words(alpha, false, false, words);
  EncodeBc1Words(rgba, false, true, refine, words + 2);
  StoreWords(words, 4, out);
}

void EncodeBc4Block(const std::uint16_t values[16], bool snorm,
                    std::uint8_t out[8]) {
  int converted[16];
  for (int i = 0; i < 16; ++i) {
    converted[i] = snorm ? Signed16(values[i]) : values[i];
  }
  std::uint32_t words[2];
  EncodeBc4Words(converted, true, snorm, words);
  StoreWords(words, 2, out);
}

void EncodeBc5Block(const std::uint16_t values[32], bool snorm,
                    std::uint8_t out[16]) {
  for (int channel = 0; channel < 2; ++channel) {
    std::uint16_t plane[16];
    for (int i = 0; i < 16; ++i) {
      plane[i] = values[i * 2 + channel];
    }
    EncodeBc4Block(plane, snorm, out + channel * 8);
  }
}

void TranscodeEtcBlockToBc(EtcFormat format, const std::uint8_t* etc_block,
                           bool refine, std::uint8_t* bc_block) {
  std::uint8_t texels[64];
  DecodeEtcImage(format, etc_block, EtcBlockBytes(format), 4, 4, texels,
                 sizeof(texels));
  EncodeDecodedBlock(format, texels, 4 * EtcDecodedTexelBytes(format), refine,
                     bc_block);
}

bool TranscodeEtcToBc(EtcFormat format, const std::uint8_t* source,
                      std::size_t source_bytes, std::uint8_t* out,
                      std::size_t out_bytes, bool refine) {
  const std::size_t block_bytes = EtcBlockBytes(format);
  if (source_bytes % block_bytes != 0 || out_bytes < source_bytes ||
      (source_bytes != 0 && (source == nullptr || out == nullptr))) {
    return false;
  }
  const std::size_t texel_bytes = EtcDecodedTexelBytes(format);
  std::uint8_t strip[kStripBlocks * 16 * 4];
  const std::size_t blocks = source_bytes / block_bytes;
  for (std::size_t first = 0; first < blocks; first += kStripBlocks) {
    const std::size_t count = std::min(kStripBlocks, blocks - first);
    // The blocks as one image of count x 1 blocks.
    const std::size_t row_stride = count * 4 * texel_bytes;
    DecodeEtcImage(format, source + first * block_bytes, count * block_bytes,
                   static_cast<std::uint32_t>(count * 4), 4, strip,
                   row_stride * 4);
    for (std::size_t block = 0; block < count; ++block) {
      EncodeDecodedBlock(format, strip + block * 4 * texel_bytes, row_stride,
                         refine, out + (first + block) * block_bytes);
    }
  }
  return true;
}

std::size_t BcEncodedBytes(EtcFormat format, std::uint32_t width,
                           std::uint32_t height, std::uint32_t layers) {
  return ((static_cast<std::size_t>(width) + 3) / 4) *
         ((static_cast<std::size_t>(height) + 3) / 4) * layers *
         EtcBlockBytes(format);
}

bool TranscodeDecodedToBc(EtcFormat format, const std::uint8_t* decoded,
                          std::uint32_t width, std::uint32_t height,
                          std::uint32_t layers, std::uint8_t* out,
                          std::size_t out_bytes, bool refine) {
  if (width == 0 || height == 0 || layers == 0 || decoded == nullptr ||
      out == nullptr ||
      out_bytes < BcEncodedBytes(format, width, height, layers)) {
    return false;
  }
  const std::size_t texel_bytes = EtcDecodedTexelBytes(format);
  const std::size_t row_bytes = static_cast<std::size_t>(width) * texel_bytes;
  const std::size_t layer_bytes = row_bytes * height;
  const std::size_t block_bytes = EtcBlockBytes(format);
  const std::uint32_t blocks_wide = (width + 3) / 4;
  const std::uint32_t blocks_high = (height + 3) / 4;
  std::uint8_t texels[64];
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    const std::uint8_t* image = decoded + layer * layer_bytes;
    for (std::uint32_t by = 0; by < blocks_high; ++by) {
      for (std::uint32_t bx = 0; bx < blocks_wide; ++bx) {
        for (std::uint32_t y = 0; y < 4; ++y) {
          const std::uint32_t source_y = std::min(height - 1, by * 4 + y);
          for (std::uint32_t x = 0; x < 4; ++x) {
            const std::uint32_t source_x = std::min(width - 1, bx * 4 + x);
            std::memcpy(texels + (y * 4 + x) * texel_bytes,
                        image + source_y * row_bytes + source_x * texel_bytes,
                        texel_bytes);
          }
        }
        EncodeDecodedBlock(format, texels, 4 * texel_bytes, refine, out);
        out += block_bytes;
      }
    }
  }
  return true;
}

}  // namespace mocktail::graphics
