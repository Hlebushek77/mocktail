#ifndef MOCKTAIL_TESTS_SUPPORT_ETC2_REFERENCE_DECODER_H_
#define MOCKTAIL_TESTS_SUPPORT_ETC2_REFERENCE_DECODER_H_

#include <cstddef>
#include <cstdint>

#include "mocktail/graphics/etc2_decoder.h"

// The ETC2/EAC decoder as plus.1 shipped it, kept as the reference that
// faster decoders (the CPU fast path, the GPU shader) must match bit for bit.
namespace mocktail::graphics::reference {

// Its own copy of the format list, so calls inside the frozen decoder never
// reach the production functions through argument-dependent lookup.
enum class EtcFormat {
  kEtc2Rgb8,
  kEtc2Rgb8A1,
  kEtc2Rgba8,
  kEacR11,
  kEacR11Signed,
  kEacRg11,
  kEacRg11Signed,
};
static_assert(static_cast<int>(EtcFormat::kEacRg11Signed) ==
              static_cast<int>(graphics::EtcFormat::kEacRg11Signed));

std::size_t EtcBlockBytes(EtcFormat format);

std::size_t EtcDecodedTexelBytes(EtcFormat format);

bool DecodeEtcImage(EtcFormat format, const std::uint8_t* source,
                    std::size_t source_bytes, std::uint32_t width,
                    std::uint32_t height, std::uint8_t* destination,
                    std::size_t destination_bytes);

bool DecodeEtcImageBlockRows(EtcFormat format, const std::uint8_t* source,
                             std::size_t source_bytes, std::uint32_t width,
                             std::uint32_t height,
                             std::uint32_t first_block_row,
                             std::uint32_t block_row_count,
                             std::uint8_t* destination,
                             std::size_t destination_bytes);

inline EtcFormat ToReference(graphics::EtcFormat format) {
  return static_cast<EtcFormat>(format);
}

inline std::size_t EtcBlockBytes(graphics::EtcFormat format) {
  return EtcBlockBytes(ToReference(format));
}

inline std::size_t EtcDecodedTexelBytes(graphics::EtcFormat format) {
  return EtcDecodedTexelBytes(ToReference(format));
}

inline bool DecodeEtcImage(graphics::EtcFormat format,
                           const std::uint8_t* source, std::size_t source_bytes,
                           std::uint32_t width, std::uint32_t height,
                           std::uint8_t* destination,
                           std::size_t destination_bytes) {
  return DecodeEtcImage(ToReference(format), source, source_bytes, width,
                        height, destination, destination_bytes);
}

}  // namespace mocktail::graphics::reference

#endif  // MOCKTAIL_TESTS_SUPPORT_ETC2_REFERENCE_DECODER_H_
