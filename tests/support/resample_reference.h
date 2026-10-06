#ifndef MOCKTAIL_TESTS_SUPPORT_RESAMPLE_REFERENCE_H_
#define MOCKTAIL_TESTS_SUPPORT_RESAMPLE_REFERENCE_H_

#include <cstdint>

// ResampleRgba as plus.1 shipped it: the box filter that upscales small
// textures and fits override PNGs. Faster versions must match it exactly.
namespace mocktail::graphics::reference {

void ResampleRgba(const std::uint8_t* source, std::uint32_t source_width,
                  std::uint32_t source_height, std::uint32_t width,
                  std::uint32_t height, std::uint8_t* destination);

}  // namespace mocktail::graphics::reference

#endif  // MOCKTAIL_TESTS_SUPPORT_RESAMPLE_REFERENCE_H_
