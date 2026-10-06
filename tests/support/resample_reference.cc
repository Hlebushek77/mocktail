// A frozen copy of ResampleRgba from src/graphics/texture_override.cc at
// 2249b6c: the body is verbatim, only the namespace differs.
#include "support/resample_reference.h"

#include <cstddef>

namespace mocktail::graphics::reference {

void ResampleRgba(const std::uint8_t* source, std::uint32_t source_width,
                  std::uint32_t source_height, std::uint32_t width,
                  std::uint32_t height, std::uint8_t* destination) {
  if (source == nullptr || destination == nullptr || width == 0 ||
      height == 0 || source_width == 0 || source_height == 0) {
    return;
  }
  for (std::uint32_t y = 0; y < height; ++y) {
    const std::uint64_t y0 = static_cast<std::uint64_t>(y) * source_height / height;
    std::uint64_t y1 =
        (static_cast<std::uint64_t>(y) + 1) * source_height / height;
    if (y1 <= y0) {
      y1 = y0 + 1;
    }
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint64_t x0 = static_cast<std::uint64_t>(x) * source_width / width;
      std::uint64_t x1 =
          (static_cast<std::uint64_t>(x) + 1) * source_width / width;
      if (x1 <= x0) {
        x1 = x0 + 1;
      }
      std::uint64_t sum[4] = {0, 0, 0, 0};
      for (std::uint64_t sy = y0; sy < y1; ++sy) {
        for (std::uint64_t sx = x0; sx < x1; ++sx) {
          const std::uint8_t* pixel = source + (sy * source_width + sx) * 4;
          for (int channel = 0; channel < 4; ++channel) {
            sum[channel] += pixel[channel];
          }
        }
      }
      const std::uint64_t count = (y1 - y0) * (x1 - x0);
      std::uint8_t* out = destination + (static_cast<std::size_t>(y) * width + x) * 4;
      for (int channel = 0; channel < 4; ++channel) {
        out[channel] =
            static_cast<std::uint8_t>((sum[channel] + count / 2) / count);
      }
    }
  }
}

}  // namespace mocktail::graphics::reference
