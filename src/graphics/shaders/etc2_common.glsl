// ETC2/EAC block decoding, a bit-exact port of src/graphics/etc2_decoder.cc.
// Core GLSL 4.50 only: 32-bit integers, no int64/int16/8-bit storage, so it
// runs on any VkDevice Roblox creates without enabling extra features.

#ifndef ETC2_COMMON_GLSL
#define ETC2_COMMON_GLSL

// EtcFormat order from include/mocktail/graphics/etc2_decoder.h.
#define FMT_ETC2_RGB8 0u
#define FMT_ETC2_RGB8A1 1u
#define FMT_ETC2_RGBA8 2u
#define FMT_EAC_R11 3u
#define FMT_EAC_R11_SIGNED 4u
#define FMT_EAC_RG11 5u
#define FMT_EAC_RG11_SIGNED 6u

const int kDistance[8] = int[8](3, 6, 11, 16, 23, 32, 41, 64);
// Indexed by table * 4 + ((msb << 1) | lsb).
const int kEtc1Modifiers[32] = int[32](
    2, 8, -2, -8, 5, 17, -5, -17, 9, 29, -9, -29, 13, 42, -13, -42,
    18, 60, -18, -60, 24, 80, -24, -80, 33, 106, -33, -106, 47, 183, -47, -183);
const int kEacModifiers[128] = int[128](
    -3, -6, -9, -15, 2, 5, 8, 14,   -3, -7, -10, -13, 2, 6, 9, 12,
    -2, -5, -8, -13, 1, 4, 7, 12,   -2, -4, -6, -13, 1, 3, 5, 12,
    -3, -6, -8, -12, 2, 5, 7, 11,   -3, -7, -9, -11, 2, 6, 8, 10,
    -4, -7, -8, -11, 3, 6, 7, 10,   -3, -5, -8, -11, 2, 4, 7, 10,
    -2, -6, -8, -10, 1, 5, 7, 9,    -2, -5, -8, -10, 1, 4, 7, 9,
    -2, -4, -8, -10, 1, 3, 7, 9,    -2, -5, -7, -10, 1, 4, 6, 9,
    -3, -4, -7, -10, 2, 3, 6, 9,    -1, -2, -3, -10, 0, 1, 2, 9,
    -4, -6, -8, -9, 3, 5, 7, 8,     -3, -5, -7, -9, 2, 4, 6, 8);
const int kDelta[8] = int[8](0, 1, 2, 3, -4, -3, -2, -1);

int Expand4(int v) { return (v << 4) | v; }
int Expand5(int v) { return (v << 3) | (v >> 2); }
int Expand6(int v) { return (v << 2) | (v >> 4); }
int Expand7(int v) { return (v << 1) | (v >> 6); }

uint PackRgba(int r, int g, int b, int a) {
  return uint(clamp(r, 0, 255)) | (uint(clamp(g, 0, 255)) << 8) |
         (uint(clamp(b, 0, 255)) << 16) | (uint(a) << 24);
}

// Byte i (0..7) of the 64-bit block stored little-endian in (w0, w1).
int BlockByte(uint w0, uint w1, int i) {
  return int(((i < 4 ? w0 : w1) >> (8 * (i & 3))) & 0xFFu);
}

// 16 RGBA8 texels, column-major (x * 4 + y) like the CPU decoder.
void DecodeRgbBlock(uint w0, uint w1, bool punchthrough, out uint px[16]) {
  int b[8];
  for (int i = 0; i < 8; ++i) {
    b[i] = BlockByte(w0, w1, i);
  }
  const bool diff_or_opaque = (b[3] & 2) != 0;
  const bool opaque = !punchthrough || diff_or_opaque;
  const int msb = (b[4] << 8) | b[5];
  const int lsb = (b[6] << 8) | b[7];
  const int r_sum = (b[0] >> 3) + kDelta[b[0] & 7];
  const int g_sum = (b[1] >> 3) + kDelta[b[1] & 7];
  const int b_sum = (b[2] >> 3) + kDelta[b[2] & 7];
  const bool individual = !punchthrough && !diff_or_opaque;

  if (!individual && (r_sum < 0 || r_sum > 31 || g_sum < 0 || g_sum > 31)) {
    ivec3 paint[4];
    if (r_sum < 0 || r_sum > 31) {  // T mode
      const ivec3 c1 = ivec3(Expand4((((b[0] >> 3) & 3) << 2) | (b[0] & 3)),
                             Expand4(b[1] >> 4), Expand4(b[1] & 15));
      const ivec3 c2 = ivec3(Expand4(b[2] >> 4), Expand4(b[2] & 15),
                             Expand4(b[3] >> 4));
      const int d = kDistance[(((b[3] >> 2) & 3) << 1) | (b[3] & 1)];
      paint[0] = c1;
      paint[1] = c2 + d;
      paint[2] = c2;
      paint[3] = c2 - d;
    } else {  // H mode
      const ivec3 c1 = ivec3(
          Expand4((b[0] >> 3) & 15),
          Expand4(((b[0] & 7) << 1) | ((b[1] >> 4) & 1)),
          Expand4((b[1] & 8) | ((b[1] & 3) << 1) | (b[2] >> 7)));
      const ivec3 c2 = ivec3(Expand4((b[2] >> 3) & 15),
                             Expand4(((b[2] & 7) << 1) | (b[3] >> 7)),
                             Expand4((b[3] >> 3) & 15));
      const int v1 = (c1.r << 16) | (c1.g << 8) | c1.b;
      const int v2 = (c2.r << 16) | (c2.g << 8) | c2.b;
      const int d =
          kDistance[(b[3] & 4) | ((b[3] & 1) << 1) | (v1 >= v2 ? 1 : 0)];
      paint[0] = c1 + d;
      paint[1] = c1 - d;
      paint[2] = c2 + d;
      paint[3] = c2 - d;
    }
    for (int t = 0; t < 16; ++t) {
      const int index = (((msb >> t) & 1) << 1) | ((lsb >> t) & 1);
      px[t] = (!opaque && index == 2)
                  ? 0u
                  : PackRgba(paint[index].r, paint[index].g, paint[index].b, 255);
    }
    return;
  }

  if (!individual && (b_sum < 0 || b_sum > 31)) {  // planar
    const int ro = Expand6((b[0] >> 1) & 63);
    const int go = Expand7(((b[0] & 1) << 6) | ((b[1] >> 1) & 63));
    const int bo = Expand6(((b[1] & 1) << 5) | (b[2] & 0x18) |
                           ((b[2] & 3) << 1) | (b[3] >> 7));
    const int rh = Expand6(((b[3] & 0x7c) >> 1) | (b[3] & 1));
    const int gh = Expand7(b[4] >> 1);
    const int bh = Expand6(((b[4] & 1) << 5) | (b[5] >> 3));
    const int rv = Expand6(((b[5] & 7) << 3) | (b[6] >> 5));
    const int gv = Expand7(((b[6] & 31) << 2) | (b[7] >> 6));
    const int bv = Expand6(b[7] & 63);
    for (int x = 0; x < 4; ++x) {
      for (int y = 0; y < 4; ++y) {
        px[x * 4 + y] =
            PackRgba((x * (rh - ro) + y * (rv - ro) + 4 * ro + 2) >> 2,
                     (x * (gh - go) + y * (gv - go) + 4 * go + 2) >> 2,
                     (x * (bh - bo) + y * (bv - bo) + 4 * bo + 2) >> 2, 255);
      }
    }
    return;
  }

  ivec3 base0;
  ivec3 base1;
  for (int c = 0; c < 3; ++c) {
    if (individual) {
      base0[c] = Expand4(b[c] >> 4);
      base1[c] = Expand4(b[c] & 15);
    } else {
      base0[c] = Expand5(b[c] >> 3);
      base1[c] = Expand5((b[c] >> 3) + kDelta[b[c] & 7]);
    }
  }
  const int table0 = b[3] >> 5;
  const int table1 = (b[3] >> 2) & 7;
  const bool flipped = (b[3] & 1) != 0;
  for (int x = 0; x < 4; ++x) {
    for (int y = 0; y < 4; ++y) {
      const int t = x * 4 + y;
      const int index = (((msb >> t) & 1) << 1) | ((lsb >> t) & 1);
      if (!opaque && index == 2) {
        px[t] = 0u;
        continue;
      }
      const bool second = flipped ? (y >= 2) : (x >= 2);
      const ivec3 base = second ? base1 : base0;
      const int modifier =
          (!opaque && index == 0)
              ? 0
              : kEtc1Modifiers[(second ? table1 : table0) * 4 + index];
      px[t] = PackRgba(base.r + modifier, base.g + modifier, base.b + modifier,
                       255);
    }
  }
}

// 3-bit index of texel t from the 48 index bits in bytes 2..7.
int EacIndex(uint w0, uint w1, int t) {
  const uint hi16 = (uint(BlockByte(w0, w1, 2)) << 8) | uint(BlockByte(w0, w1, 3));
  const uint lo32 = (uint(BlockByte(w0, w1, 4)) << 24) |
                    (uint(BlockByte(w0, w1, 5)) << 16) |
                    (uint(BlockByte(w0, w1, 6)) << 8) | uint(BlockByte(w0, w1, 7));
  const int s = 45 - 3 * t;
  uint v;
  if (s >= 32) {
    v = hi16 >> uint(s - 32);
  } else if (s == 0) {
    v = lo32;
  } else {
    v = (lo32 >> uint(s)) | (hi16 << uint(32 - s));
  }
  return int(v & 7u);
}

int EacModifier(uint w0, uint w1, int t) {
  return kEacModifiers[(BlockByte(w0, w1, 1) & 15) * 8 + EacIndex(w0, w1, t)];
}

// Returns the 16-bit R16_UNORM / R16_SNORM bit pattern in the low half.
uint DecodeR11(uint w0, uint w1, int t, bool is_signed) {
  const int b0 = BlockByte(w0, w1, 0);
  const int b1 = BlockByte(w0, w1, 1);
  const int multiplier = b1 >> 4;
  const int modifier = EacModifier(w0, w1, t);
  const int scaled = multiplier != 0 ? modifier * multiplier * 8 : modifier;
  if (!is_signed) {
    const int value = clamp(b0 * 8 + 4 + scaled, 0, 2047);
    return uint((value << 5) | (value >> 6));
  }
  const int base = max(-127, (b0 & 0x80) != 0 ? b0 - 256 : b0);
  const int value = clamp(base * 8 + scaled, -1023, 1023);
  const int magnitude = abs(value);
  const int expanded = (magnitude << 5) | (magnitude >> 5);
  return uint(value < 0 ? -expanded : expanded) & 0xFFFFu;
}

uint EtcBlockWords(uint format) {
  return (format == FMT_ETC2_RGBA8 || format == FMT_EAC_RG11 ||
          format == FMT_EAC_RG11_SIGNED)
             ? 4u
             : 2u;
}

#endif  // ETC2_COMMON_GLSL
