#ifndef MOCKTAIL_TESTS_SUPPORT_BC_REFERENCE_TRANSCODE_H_
#define MOCKTAIL_TESTS_SUPPORT_BC_REFERENCE_TRANSCODE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mocktail/graphics/etc2_decoder.h"

// ETC2->BC mode as the references do it, which the GPU route's etc2_bc
// shader was verified against (bc_test's CpuTranscodeBlock): the plus.1
// decoder (support/etc2_reference_decoder.h), then the frozen encoder
// (support/bc_reference.h, a copy of plus2-design/harness/bc_ref.h). Blocks
// are stored as Vulkan's BCn formats store them (little-endian words).
namespace mocktail::graphics::reference {

// One BC block from 16 texels in the layout the decoder writes for
// `format` (row-major RGBA8, R16 or RG16): BC1 for RGB8, BC1 with
// punch-through for RGB8A1 (alpha below 128 is transparent), BC3 for
// RGBA8, BC4 for R11 and BC5 for RG11, signed for the SNORM formats.
void BcBlockFromTexels(graphics::EtcFormat format, const std::uint8_t* texels,
                       bool refine, std::uint8_t* out);

// The BC block ETC2->BC mode stores for one ETC2/EAC block: the whole 4x4
// block decoded, padding texels of edge blocks included, then encoded.
void BcBlockFromEtc(graphics::EtcFormat format, const std::uint8_t* block,
                    bool refine, std::uint8_t* out);

// Tightly packed ETC2/EAC blocks to as many bytes of BC blocks.
std::vector<std::uint8_t> BcFromEtc(graphics::EtcFormat format,
                                    const std::uint8_t* blocks,
                                    std::size_t bytes, bool refine);

// `layers` tightly packed images of width x height decoded texels to
// tightly packed BC blocks, texels past the right and bottom edge repeating
// the last column and row (how an override's texels are encoded).
std::vector<std::uint8_t> BcFromTexels(graphics::EtcFormat format,
                                       const std::uint8_t* texels,
                                       std::uint32_t width,
                                       std::uint32_t height,
                                       std::uint32_t layers, bool refine);

}  // namespace mocktail::graphics::reference

#endif  // MOCKTAIL_TESTS_SUPPORT_BC_REFERENCE_TRANSCODE_H_
