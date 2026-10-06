// GLSL port of harness/bc_ref.h (integer-only BC1/BC4 encoders).
#ifndef BC_COMMON_GLSL
#define BC_COMMON_GLSL

int Quant5(int v) { return (v * 31 + 127) / 255; }
int Quant6(int v) { return (v * 63 + 127) / 255; }
int Unq5(int q) { return (q << 3) | (q >> 2); }
int Unq6(int q) { return (q << 2) | (q >> 4); }

uint Pack565(ivec3 c) {
  return (uint(Quant5(c.r)) << 11) | (uint(Quant6(c.g)) << 5) | uint(Quant5(c.b));
}
ivec3 Unpack565(uint c) {
  return ivec3(Unq5(int(c >> 11) & 31), Unq6(int(c >> 5) & 63), Unq5(int(c) & 31));
}

void Bc1Palette(uint c0, uint c1, bool four, out ivec3 p[4]) {
  p[0] = Unpack565(c0);
  p[1] = Unpack565(c1);
  if (four) {
    p[2] = (2 * p[0] + p[1] + 1) / 3;
    p[3] = (p[0] + 2 * p[1] + 1) / 3;
  } else {
    p[2] = (p[0] + p[1] + 1) / 2;
    p[3] = ivec3(0);
  }
}

int Dist(ivec3 a, ivec3 b) {
  const ivec3 d = a - b;
  return d.r * d.r + d.g * d.g + d.b * d.b;
}

ivec3 TexelRgb(uint t) { return ivec3(int(t & 0xFFu), int((t >> 8) & 0xFFu), int((t >> 16) & 0xFFu)); }

// Returns the error; writes 2-bit indices (row-major texels) to `indices`.
int Bc1Choose(uint a, uint b, bool force_four, const uint tex[16], const bool transparent[16],
              out uint indices) {
  const bool four = force_four || a > b;
  ivec3 p[4];
  Bc1Palette(a, b, four, p);
  const int usable = four ? 4 : 3;
  uint bits = 0u;
  int error = 0;
  for (int i = 0; i < 16; ++i) {
    int best = 0;
    if (transparent[i]) {
      best = 3;
    } else {
      const ivec3 c = TexelRgb(tex[i]);
      int best_d = Dist(c, p[0]);
      for (int k = 1; k < usable; ++k) {
        const int d = Dist(c, p[k]);
        if (d < best_d) {
          best_d = d;
          best = k;
        }
      }
      error += best_d;
    }
    bits |= uint(best) << (2 * i);
  }
  indices = bits;
  return error;
}

void InsetAxis(inout int lo, inout int hi) {
  const int d = hi > lo ? hi - lo : lo - hi;
  const int s = d >> 4;
  if (hi > lo) {
    hi -= s;
    lo += s;
  } else {
    hi += s;
    lo -= s;
  }
}

int DivRound(int n, int det) {
  return n >= 0 ? (n + det / 2) / det : -((-n + det / 2) / det);
}

// tex: 16 RGBA8 texels, row-major.
uvec2 EncodeBc1(const uint tex[16], bool punch, bool force_four, bool refine) {
  bool transparent[16];
  bool any_transparent = false;
  int count = 0;
  ivec3 mn = ivec3(255);
  ivec3 mx = ivec3(0);
  for (int i = 0; i < 16; ++i) {
    transparent[i] = punch && (tex[i] >> 24) < 128u;
    any_transparent = any_transparent || transparent[i];
    if (transparent[i]) continue;
    ++count;
    const ivec3 c = TexelRgb(tex[i]);
    mn = min(mn, c);
    mx = max(mx, c);
  }
  if (count == 0) {
    return uvec2(0u, 0xFFFFFFFFu);
  }
  const ivec3 center = (mn + mx) >> 1;
  int cov_rg = 0;
  int cov_bg = 0;
  for (int i = 0; i < 16; ++i) {
    if (transparent[i]) continue;
    const ivec3 c = TexelRgb(tex[i]) - center;
    cov_rg += c.r * c.g;
    cov_bg += c.b * c.g;
  }
  if (cov_rg < 0) {
    const int t = mn.r; mn.r = mx.r; mx.r = t;
  }
  if (cov_bg < 0) {
    const int t = mn.b; mn.b = mx.b; mx.b = t;
  }
  InsetAxis(mn.r, mx.r);
  InsetAxis(mn.g, mx.g);
  InsetAxis(mn.b, mx.b);
  uint c0 = Pack565(mx);
  uint c1 = Pack565(mn);
  const bool three = any_transparent && !force_four;
  if (three ? c0 > c1 : c0 < c1) {
    const uint t = c0; c0 = c1; c1 = t;
  }
  uint indices;
  int error = Bc1Choose(c0, c1, force_four, tex, transparent, indices);
  if (refine && !three) {
    const int kW0[4] = int[4](3, 0, 2, 1);
    int aa = 0, ab = 0, bb = 0;
    ivec3 ax = ivec3(0), bx = ivec3(0);
    for (int i = 0; i < 16; ++i) {
      const int k = int((indices >> (2 * i)) & 3u);
      const int a = kW0[k];
      const int b = 3 - a;
      aa += a * a;
      ab += a * b;
      bb += b * b;
      const ivec3 c = TexelRgb(tex[i]);
      ax += a * c;
      bx += b * c;
    }
    const int det = aa * bb - ab * ab;
    if (det > 0) {
      ivec3 e0, e1;
      for (int c = 0; c < 3; ++c) {
        e0[c] = clamp(DivRound(3 * (bb * ax[c] - ab * bx[c]), det), 0, 255);
        e1[c] = clamp(DivRound(3 * (aa * bx[c] - ab * ax[c]), det), 0, 255);
      }
      uint r0 = Pack565(e0);
      uint r1 = Pack565(e1);
      if (r0 < r1) {
        const uint t = r0; r0 = r1; r1 = t;
      }
      uint refined;
      const int refined_error = Bc1Choose(r0, r1, force_four, tex, transparent, refined);
      if (refined_error < error) {
        c0 = r0;
        c1 = r1;
        indices = refined;
        error = refined_error;
      }
    }
  }
  return uvec2(c0 | (c1 << 16), indices);
}

// values: 16 inputs; is16 = R16 patterns, signed_values = R16_SNORM patterns
// already sign-extended to int.
uvec2 EncodeBc4(const int values[16], bool is16, bool signed_values) {
  int v[16];
  for (int i = 0; i < 16; ++i) {
    if (!is16) {
      v[i] = values[i] * 256;
    } else if (!signed_values) {
      v[i] = values[i] * 256 / 257;
    } else {
      const int s = max(values[i], -32767);
      v[i] = (s + 32767) * 254 * 128 / 32767;
    }
  }
  int mn = v[0];
  int mx = v[0];
  for (int i = 1; i < 16; ++i) {
    mn = min(mn, v[i]);
    mx = max(mx, v[i]);
  }
  const int top = signed_values ? 254 : 255;
  int e0 = min(top, (mx + 128) >> 8);
  int e1 = min(top, (mn + 128) >> 8);
  uint lo32 = 0u;
  uint hi32 = 0u;
  if (e0 != e1) {
    if (e0 < e1) {
      const int t = e0; e0 = e1; e1 = t;
    }
    const int lo = e1 * 256;
    const int span = (e0 - e1) * 256;
    for (int i = 0; i < 16; ++i) {
      const int d = clamp(v[i] - lo, 0, span);
      const int pos = (d * 14 + span) / (2 * span);
      const uint idx = pos == 7 ? 0u : (pos == 0 ? 1u : uint(8 - pos));
      const int bit = 3 * i;
      if (bit < 32) {
        lo32 |= idx << bit;
        if (bit > 29) hi32 |= idx >> (32 - bit);
      } else {
        hi32 |= idx << (bit - 32);
      }
    }
  }
  const uint b0 = signed_values ? uint((e0 - 127) & 0xFF) : uint(e0);
  const uint b1 = signed_values ? uint((e1 - 127) & 0xFF) : uint(e1);
  return uvec2(b0 | (b1 << 8) | (lo32 << 16), (lo32 >> 16) | (hi32 << 16));
}

#endif  // BC_COMMON_GLSL
