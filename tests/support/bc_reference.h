// A frozen copy of plus2-design/harness/bc_ref.h (2026-10-06), the BC
// encoder the GPU route's etc2_bc.comp was verified against bit for bit
// (bc_test parity, 7 formats, refine on and off, RTX 3060 / 615.71.09).
// src/graphics/bc_encoder.cc must give the same bytes for every block;
// bc_encoder_test checks it. Do not change this file to make a test pass.
// (Only these comment lines and the diagnostic pragmas around the copy
// were added; the copy itself keeps the original's one-line `if`s.)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
// Integer-only BC1/BC3/BC4/BC5 block encoders: the CPU reference (and CPU
// fallback) for the GPU ETC2 -> BC transcoder. Every operation is integer and
// divisions only see non-negative operands, so the GLSL port is bit-exact.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace bcref {

inline int Quant5(int v) { return (v * 31 + 127) / 255; }  // round(v*31/255)
inline int Quant6(int v) { return (v * 63 + 127) / 255; }
inline int Unq5(int q) { return (q << 3) | (q >> 2); }
inline int Unq6(int q) { return (q << 2) | (q >> 4); }

struct Rgb { int r, g, b; };

inline std::uint32_t Pack565(const Rgb& c) {
  return (static_cast<std::uint32_t>(Quant5(c.r)) << 11) |
         (static_cast<std::uint32_t>(Quant6(c.g)) << 5) |
         static_cast<std::uint32_t>(Quant5(c.b));
}
inline Rgb Unpack565(std::uint32_t c) {
  return {Unq5(static_cast<int>(c >> 11) & 31), Unq6(static_cast<int>(c >> 5) & 63),
          Unq5(static_cast<int>(c) & 31)};
}

// Palette as decoded by D3D10+ hardware (rounded thirds / halves).
inline void Bc1Palette(std::uint32_t c0, std::uint32_t c1, bool four, Rgb p[4]) {
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

inline int Dist(const Rgb& a, const Rgb& b) {
  const int dr = a.r - b.r, dg = a.g - b.g, db = a.b - b.b;
  return dr * dr + dg * dg + db * db;
}

// texels: 16 RGBA8, row-major (y * 4 + x). mode: 0 = opaque (BC1 RGB / BC3
// colour, alpha ignored), 1 = punchthrough (alpha 0 -> index 3, 3-colour).
// force_four: BC3 colour blocks are always decoded in 4-colour mode.
// refine: one least-squares endpoint refinement pass.
inline void EncodeBc1(const std::uint8_t* texels, bool punch, bool force_four, bool refine,
                      std::uint32_t out[2]) {
  bool transparent[16];
  bool any_transparent = false;
  int count = 0;
  Rgb mn{255, 255, 255}, mx{0, 0, 0};
  for (int i = 0; i < 16; ++i) {
    transparent[i] = punch && texels[i * 4 + 3] < 128;
    any_transparent |= transparent[i];
    if (transparent[i]) continue;
    ++count;
    const Rgb c{texels[i * 4], texels[i * 4 + 1], texels[i * 4 + 2]};
    mn = {std::min(mn.r, c.r), std::min(mn.g, c.g), std::min(mn.b, c.b)};
    mx = {std::max(mx.r, c.r), std::max(mx.g, c.g), std::max(mx.b, c.b)};
  }
  if (count == 0) {  // fully transparent: 3-colour mode, all index 3
    out[0] = 0u;            // c0 = c1 = 0 -> c0 <= c1
    out[1] = 0xFFFFFFFFu;
    return;
  }
  // Pick the bounding-box diagonal that follows the colour covariance.
  const Rgb center{(mn.r + mx.r) >> 1, (mn.g + mx.g) >> 1, (mn.b + mx.b) >> 1};
  int cov_rg = 0, cov_bg = 0;
  for (int i = 0; i < 16; ++i) {
    if (transparent[i]) continue;
    const int r = texels[i * 4] - center.r, g = texels[i * 4 + 1] - center.g,
              b = texels[i * 4 + 2] - center.b;
    cov_rg += r * g;
    cov_bg += b * g;
  }
  if (cov_rg < 0) std::swap(mn.r, mx.r);
  if (cov_bg < 0) std::swap(mn.b, mx.b);
  // Inset by 1/16 of the range towards the middle.
  auto inset = [](int& lo, int& hi) {
    const int d = hi > lo ? hi - lo : lo - hi;
    const int s = d >> 4;
    if (hi > lo) { hi -= s; lo += s; } else { hi += s; lo -= s; }
  };
  inset(mn.r, mx.r);
  inset(mn.g, mx.g);
  inset(mn.b, mx.b);
  std::uint32_t c0 = Pack565(mx), c1 = Pack565(mn);
  const bool three = any_transparent && !force_four;
  auto order = [&](std::uint32_t& a, std::uint32_t& b) {
    if (three ? a > b : a < b) std::swap(a, b);
  };
  auto choose = [&](std::uint32_t a, std::uint32_t b, std::uint32_t* indices) {
    const bool four = force_four || a > b;
    Rgb p[4];
    Bc1Palette(a, b, four, p);
    const int usable = four ? 4 : 3;
    std::uint32_t bits = 0;
    int error = 0;
    for (int i = 0; i < 16; ++i) {
      int best = 0;
      if (transparent[i]) {
        best = 3;
      } else {
        const Rgb c{texels[i * 4], texels[i * 4 + 1], texels[i * 4 + 2]};
        int best_d = Dist(c, p[0]);
        for (int k = 1; k < usable; ++k) {
          const int d = Dist(c, p[k]);
          if (d < best_d) { best_d = d; best = k; }
        }
        error += best_d;
      }
      bits |= static_cast<std::uint32_t>(best) << (2 * i);
    }
    *indices = bits;
    return error;
  };
  order(c0, c1);
  std::uint32_t indices = 0;
  int error = choose(c0, c1, &indices);
  if (refine && !three) {
    // Least squares on the 4-colour weights (3,0,2,1)/3 and (0,3,1,2)/3.
    static const int kW0[4] = {3, 0, 2, 1};
    int aa = 0, ab = 0, bb = 0;
    int ax[3] = {0, 0, 0}, bx[3] = {0, 0, 0};
    for (int i = 0; i < 16; ++i) {
      const int k = static_cast<int>((indices >> (2 * i)) & 3);
      const int a = kW0[k], b = 3 - a;
      aa += a * a; ab += a * b; bb += b * b;
      for (int c = 0; c < 3; ++c) {
        ax[c] += a * texels[i * 4 + c];
        bx[c] += b * texels[i * 4 + c];
      }
    }
    const int det = aa * bb - ab * ab;  // >= 0; 0 when one index only
    if (det > 0) {
      // e0 = 3 * (bb*ax - ab*bx) / det, e1 = 3 * (aa*bx - ab*ax) / det.
      int e0[3], e1[3];
      for (int c = 0; c < 3; ++c) {
        const int n0 = 3 * (bb * ax[c] - ab * bx[c]);
        const int n1 = 3 * (aa * bx[c] - ab * ax[c]);
        auto div_round = [det](int n) {  // round(n / det), det > 0
          return n >= 0 ? (n + det / 2) / det : -((-n + det / 2) / det);
        };
        e0[c] = std::clamp(div_round(n0), 0, 255);
        e1[c] = std::clamp(div_round(n1), 0, 255);
      }
      std::uint32_t r0 = Pack565({e0[0], e0[1], e0[2]});
      std::uint32_t r1 = Pack565({e1[0], e1[1], e1[2]});
      order(r0, r1);
      std::uint32_t refined = 0;
      const int refined_error = choose(r0, r1, &refined);
      if (refined_error < error) {
        c0 = r0; c1 = r1; indices = refined; error = refined_error;
      }
    }
  }
  out[0] = c0 | (c1 << 16);
  out[1] = indices;
}

// BC4 on 16 values in [0, range] (range 255 for 8-bit alpha, 65535 for
// R16 sources). Endpoints are stored as 8-bit; `signed_values` maps
// R16_SNORM patterns (-32767..32767) onto BC4_SNORM endpoints (-127..127).
inline void EncodeBc4(const int* values, bool is16, bool signed_values, std::uint32_t out[2]) {
  // Work in an unsigned 8-bit endpoint domain u in [0, 254] (snorm) or
  // [0, 255] (unorm), with values scaled by 255 / range for comparison.
  int v[16];
  for (int i = 0; i < 16; ++i) {
    if (!is16) {
      v[i] = values[i] * 256;  // 8.8 fixed point
    } else if (!signed_values) {
      v[i] = values[i] * 256 / 257;  // 8.8 fixed point of v*255/65535
    } else {
      const int s = std::max(values[i], -32767);
      v[i] = (s + 32767) * 254 * 128 / 32767;  // 8.8 of (s+1)*127 -> [0, 254]
    }
  }
  int mn = v[0], mx = v[0];
  for (int i = 1; i < 16; ++i) { mn = std::min(mn, v[i]); mx = std::max(mx, v[i]); }
  const int top = signed_values ? 254 : 255;
  int e0 = std::min(top, (mx + 128) >> 8);  // endpoints in the u domain
  int e1 = std::min(top, (mn + 128) >> 8);
  std::uint32_t indices_lo = 0, indices_hi = 0;
  auto set_index = [&](int i, std::uint32_t idx) {
    const int bit = 3 * i;
    if (bit < 32) {
      indices_lo |= idx << bit;
      if (bit > 29) indices_hi |= idx >> (32 - bit);
    } else {
      indices_hi |= idx << (bit - 32);
    }
  };
  if (e0 == e1) {
    // All texels index 0 (= e0).
  } else {
    if (e0 < e1) std::swap(e0, e1);
    const int lo = e1 * 256, span = (e0 - e1) * 256;
    for (int i = 0; i < 16; ++i) {
      const int d = std::clamp(v[i] - lo, 0, span);
      const int pos = (d * 14 + span) / (2 * span);  // round(d * 7 / span)
      const std::uint32_t idx = pos == 7 ? 0u : pos == 0 ? 1u : static_cast<std::uint32_t>(8 - pos);
      set_index(i, idx);
    }
  }
  const auto byte = [&](int e) {
    return signed_values ? static_cast<std::uint32_t>((e - 127) & 0xFF) : static_cast<std::uint32_t>(e);
  };
  out[0] = byte(e0) | (byte(e1) << 8) | (indices_lo << 16);
  out[1] = (indices_lo >> 16) | (indices_hi << 16);
}

// ---- Reference decoders (for PSNR only; D3D10 rules) ----
inline void DecodeBc1(const std::uint32_t in[2], bool force_four, std::uint8_t* rgba /*16*4, row-major*/) {
  const std::uint32_t c0 = in[0] & 0xFFFF, c1 = in[0] >> 16;
  const bool four = force_four || c0 > c1;
  Rgb p[4];
  Bc1Palette(c0, c1, four, p);
  for (int i = 0; i < 16; ++i) {
    const int k = (in[1] >> (2 * i)) & 3;
    rgba[i * 4] = p[k].r; rgba[i * 4 + 1] = p[k].g; rgba[i * 4 + 2] = p[k].b;
    rgba[i * 4 + 3] = (!four && k == 3) ? 0 : 255;
  }
}

// Returns values in the 8.8 fixed domain of EncodeBc4 (u domain * 256).
inline void DecodeBc4Fixed(const std::uint32_t in[2], bool signed_values, int* out16) {
  int e0 = in[0] & 0xFF, e1 = (in[0] >> 8) & 0xFF;
  if (signed_values) {
    e0 = static_cast<std::int8_t>(e0) + 127; e1 = static_cast<std::int8_t>(e1) + 127;
    if (e0 < 0) e0 = 0; if (e1 < 0) e1 = 0;  // -128 == -127
  }
  std::uint64_t bits = (static_cast<std::uint64_t>(in[1]) << 16) | (in[0] >> 16);
  int pal[8];
  pal[0] = e0 * 256; pal[1] = e1 * 256;
  const bool eight = signed_values ? (static_cast<std::int8_t>(in[0] & 0xFF) > static_cast<std::int8_t>((in[0] >> 8) & 0xFF)) : e0 > e1;
  if (eight) {
    for (int k = 1; k <= 6; ++k) pal[k + 1] = ((7 - k) * e0 * 256 + k * e1 * 256) / 7;
  } else {
    for (int k = 1; k <= 4; ++k) pal[k + 1] = ((5 - k) * e0 * 256 + k * e1 * 256) / 5;
    pal[6] = 0; pal[7] = (signed_values ? 254 : 255) * 256;
  }
  for (int i = 0; i < 16; ++i) out16[i] = pal[(bits >> (3 * i)) & 7];
}

}  // namespace bcref

#pragma GCC diagnostic pop
