#ifndef MOCKTAIL_GRAPHICS_BC_ENCODER_H_
#define MOCKTAIL_GRAPHICS_BC_ENCODER_H_

#include <cstddef>
#include <cstdint>

#include "mocktail/graphics/etc2_decoder.h"

// ETC2 -> BC conversion on the CPU (ETC2->BC mode, MOCKTAIL_ETC2_TO_BC=1):
// integer-only BC1, BC3, BC4 and BC5 block encoders. Every result is
// bit-identical to the GPU route's etc2_bc.comp; both were checked against
// the same reference (plus2-design/harness/bc_ref.h, kept as
// tests/support/bc_reference.h), so an image looks the same whichever route
// served each of its uploads. Blocks are written in the byte order Vulkan's
// BCn formats store them in (little-endian 32-bit words).
namespace mocktail::graphics {

// ETC2->BC mode refines every BC1 colour block (EncodeBc1Block's `refine`),
// on the CPU route and in the GPU route's etc2_bc pipeline alike, so a BC
// image's bytes do not depend on which route served an upload. (CR-13
// leaves it to the user whether the CPU route may skip it: a quarter of the
// transcode time for RGB8->BC1, a fifth for RGBA8->BC3, at the price of
// route-dependent bytes.)
inline constexpr bool kEtc2ToBcRefine = true;

// `rgba`: 16 RGBA8 texels of one block, row-major (y * 4 + x). With
// `punchthrough` (ETC2 RGB8A1 -> BC1_RGBA) a texel with alpha below 128
// becomes transparent black: the block is then in 3-colour mode with index
// 3 on exactly those texels. Without it alpha is ignored. `refine` runs one
// least-squares pass over the endpoints and keeps it when it lowers the
// error.
void EncodeBc1Block(const std::uint8_t rgba[64], bool punchthrough, bool refine,
                    std::uint8_t out[8]);

// BC3: alpha as a BC4 block of the 8-bit alpha values, then the colour as a
// BC1 block, which BC3 always decodes in 4-colour mode.
void EncodeBc3Block(const std::uint8_t rgba[64], bool refine,
                    std::uint8_t out[16]);

// `values`: 16 texels as the EAC decoder writes them, R16_UNORM, or with
// `snorm` R16_SNORM bit patterns (-32768 reads as -32767).
void EncodeBc4Block(const std::uint16_t values[16], bool snorm,
                    std::uint8_t out[8]);

// `values`: 16 RG16 texels, R and G interleaved; BC5 is a BC4 block per
// channel.
void EncodeBc5Block(const std::uint16_t values[32], bool snorm,
                    std::uint8_t out[16]);

// One ETC2/EAC block to the BC block ETC2->BC mode stores for it, exactly as
// the GPU route does it: RGB8 -> BC1, RGB8A1 -> BC1 with punch-through,
// RGBA8 -> BC3, R11 -> BC4, RG11 -> BC5. Both blocks have EtcBlockBytes()
// bytes, so offsets and strides of an upload do not change.
void TranscodeEtcBlockToBc(EtcFormat format, const std::uint8_t* etc_block,
                           bool refine, std::uint8_t* bc_block);

// Every block of `source` (tightly packed ETC2/EAC blocks) to `out`. False,
// with nothing written, when `source_bytes` is not a whole number of blocks
// or `out_bytes` is smaller than `source_bytes`.
bool TranscodeEtcToBc(EtcFormat format, const std::uint8_t* source,
                      std::size_t source_bytes, std::uint8_t* out,
                      std::size_t out_bytes, bool refine);

// The bytes of tightly packed BC blocks for `layers` images of
// width x height texels (whole blocks, as for the ETC2/EAC data).
std::size_t BcEncodedBytes(EtcFormat format, std::uint32_t width,
                           std::uint32_t height, std::uint32_t layers);

// Decoded texels to BC blocks: `decoded` holds `layers` tightly packed
// images of width x height texels in the host format of the default mode
// (RGBA8 for the colour formats, R16 or RG16 for EAC), and `out` receives
// BcEncodedBytes() of tightly packed blocks. Texels past the right and
// bottom edge repeat the last column and row. This serves texture
// overrides, whose texels never were ETC2 blocks; for Roblox's own data
// TranscodeEtcToBc encodes every texel of the edge blocks as the GPU route
// does (decoding whole blocks here gives the same bytes). False, with
// nothing written, for a zero extent or a short `out_bytes`.
bool TranscodeDecodedToBc(EtcFormat format, const std::uint8_t* decoded,
                          std::uint32_t width, std::uint32_t height,
                          std::uint32_t layers, std::uint8_t* out,
                          std::size_t out_bytes, bool refine);

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_BC_ENCODER_H_
