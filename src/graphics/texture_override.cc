#include "mocktail/graphics/texture_override.h"

#include <png.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace mocktail::graphics {

namespace {

// One finished output row into the destination (staging, in the emulation).
// Streaming stores skip reading each line in before overwriting it, which
// halves the memory traffic of an enlargement; they need a 16-byte aligned
// destination and length, which upscaled rows of whole 4-texel groups have.
// The caller fences once after its last row.
void StoreRow(std::uint8_t* destination, const std::uint8_t* row,
              std::size_t bytes) {
#if defined(__SSE2__)
  if ((reinterpret_cast<std::uintptr_t>(destination) & 15) == 0 &&
      (bytes & 15) == 0) {
    auto* out = reinterpret_cast<__m128i*>(destination);
    const auto* in = reinterpret_cast<const __m128i*>(row);
    for (std::size_t index = 0; index < bytes / 16; ++index) {
      _mm_stream_si128(out + index, _mm_loadu_si128(in + index));
    }
    return;
  }
#endif
  std::memcpy(destination, row, bytes);
}

void FenceRows() {
#if defined(__SSE2__)
  _mm_sfence();
#endif
}

}  // namespace

std::uint64_t HashBytes(const std::uint8_t* data, std::size_t size) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (std::size_t index = 0; index < size; ++index) {
    hash ^= data[index];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

std::string HashName(std::uint64_t hash) {
  char text[17];
  std::snprintf(text, sizeof(text), "%016llx",
                static_cast<unsigned long long>(hash));
  return text;
}

bool ReadPngRgba(const std::string& path, RgbaImage* image) {
  if (image == nullptr) {
    return false;
  }
  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&png, path.c_str()) == 0) {
    return false;
  }
  png.format = PNG_FORMAT_RGBA;
  RgbaImage result;
  result.width = png.width;
  result.height = png.height;
  result.pixels.resize(PNG_IMAGE_SIZE(png));
  if (png_image_finish_read(&png, nullptr, result.pixels.data(), 0,
                            nullptr) == 0) {
    png_image_free(&png);
    return false;
  }
  *image = std::move(result);
  return true;
}

bool WritePngRgba(const std::string& path, const RgbaImage& image) {
  if (image.width == 0 || image.height == 0 ||
      image.pixels.size() !=
          static_cast<std::size_t>(image.width) * image.height * 4) {
    return false;
  }
  png_image png{};
  png.version = PNG_IMAGE_VERSION;
  png.width = image.width;
  png.height = image.height;
  png.format = PNG_FORMAT_RGBA;
  const int written = png_image_write_to_file(&png, path.c_str(), 0,
                                              image.pixels.data(), 0, nullptr);
  png_image_free(&png);
  return written != 0;
}

void ResampleRgba(const RgbaImage& source, std::uint32_t width,
                  std::uint32_t height, std::uint8_t* destination) {
  if (source.pixels.size() <
      static_cast<std::size_t>(source.width) * source.height * 4) {
    return;
  }
  ResampleRgba(source.pixels.data(), source.width, source.height, width,
               height, destination);
}

void ResampleRgba(const std::uint8_t* source, std::uint32_t source_width,
                  std::uint32_t source_height, std::uint32_t width,
                  std::uint32_t height, std::uint8_t* destination) {
  ResampleRgba(source, source_width, source_height,
               static_cast<std::size_t>(source_width) * 4, width, height,
               destination);
}

// Each output texel is the rounded mean of the source box
// [x0, x1) x [y0, y1) with x0 = x * source_width / width and x1 the next
// column's x0 (at least x0 + 1), likewise for rows. The spans are computed
// once per call and each output row is built in a cached row buffer, then
// written out whole (StoreRow), which suits staging memory of either kind:
// write-combined, or cached but never read back by the CPU. Rows whose
// source span repeats (every row of an enlargement) are copied from the
// previous one, and an enlargement reads single texels instead of summing
// boxes. The result is the same, bit for bit, as summing every box.
void ResampleRgba(const std::uint8_t* source, std::uint32_t source_width,
                  std::uint32_t source_height, std::size_t source_row_bytes,
                  std::uint32_t width, std::uint32_t height,
                  std::uint8_t* destination) {
  if (source == nullptr || destination == nullptr || width == 0 ||
      height == 0 || source_width == 0 || source_height == 0) {
    return;
  }
  thread_local std::vector<std::uint32_t> columns;  // x0, x1 per column
  thread_local std::vector<std::uint8_t> row;
  columns.resize(static_cast<std::size_t>(width) * 2);
  const std::size_t row_bytes = static_cast<std::size_t>(width) * 4;
  row.resize(row_bytes);
  for (std::uint32_t x = 0; x < width; ++x) {
    const std::uint64_t x0 =
        static_cast<std::uint64_t>(x) * source_width / width;
    std::uint64_t x1 =
        (static_cast<std::uint64_t>(x) + 1) * source_width / width;
    if (x1 <= x0) {
      x1 = x0 + 1;
    }
    columns[x * 2] = static_cast<std::uint32_t>(x0);
    columns[x * 2 + 1] = static_cast<std::uint32_t>(x1);
  }
  // Every column spans one source texel.
  const bool single_columns = width >= source_width;
  std::uint64_t previous_y0 = UINT64_MAX;
  std::uint64_t previous_y1 = UINT64_MAX;
  for (std::uint32_t y = 0; y < height; ++y) {
    const std::uint64_t y0 =
        static_cast<std::uint64_t>(y) * source_height / height;
    std::uint64_t y1 =
        (static_cast<std::uint64_t>(y) + 1) * source_height / height;
    if (y1 <= y0) {
      y1 = y0 + 1;
    }
    std::uint8_t* out = destination + static_cast<std::size_t>(y) * row_bytes;
    if (y0 != previous_y0 || y1 != previous_y1) {
      if (single_columns && y1 - y0 == 1) {
        const std::uint8_t* line = source + y0 * source_row_bytes;
        for (std::uint32_t x = 0; x < width; ++x) {
          std::memcpy(row.data() + static_cast<std::size_t>(x) * 4,
                      line + static_cast<std::size_t>(columns[x * 2]) * 4, 4);
        }
      } else {
        for (std::uint32_t x = 0; x < width; ++x) {
          const std::uint32_t x0 = columns[x * 2];
          const std::uint32_t x1 = columns[x * 2 + 1];
          // 64-bit sums: a large override PNG boxed down to a few texels
          // can pass 2^32 (DESIGN CR-31).
          std::uint64_t sum[4] = {0, 0, 0, 0};
          for (std::uint64_t sy = y0; sy < y1; ++sy) {
            const std::uint8_t* pixel = source + sy * source_row_bytes +
                                        static_cast<std::size_t>(x0) * 4;
            for (std::uint32_t sx = x0; sx < x1; ++sx, pixel += 4) {
              sum[0] += pixel[0];
              sum[1] += pixel[1];
              sum[2] += pixel[2];
              sum[3] += pixel[3];
            }
          }
          const std::uint64_t count = (y1 - y0) * (x1 - x0);
          std::uint8_t* texel = row.data() + static_cast<std::size_t>(x) * 4;
          for (int channel = 0; channel < 4; ++channel) {
            texel[channel] =
                static_cast<std::uint8_t>((sum[channel] + count / 2) / count);
          }
        }
      }
      previous_y0 = y0;
      previous_y1 = y1;
    }
    StoreRow(out, row.data(), row_bytes);
  }
  FenceRows();
}

namespace {

std::string EnvironmentString(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? value : "";
}

}  // namespace

TextureOverrides::TextureOverrides(std::string override_dir,
                                   std::string dump_dir)
    : override_dir_(std::move(override_dir)), dump_dir_(std::move(dump_dir)) {}

TextureOverrides TextureOverrides::FromEnvironment() {
  return TextureOverrides(EnvironmentString("MOCKTAIL_TEXTURE_OVERRIDE_DIR"),
                          EnvironmentString("MOCKTAIL_TEXTURE_DUMP_DIR"));
}

std::shared_ptr<const RgbaImage> TextureOverrides::Lookup(std::uint64_t hash) {
  if (override_dir_.empty()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const auto cached = cache_.find(hash);
  if (cached != cache_.end()) {
    return cached->second;
  }
  std::shared_ptr<const RgbaImage> loaded;
  RgbaImage image;
  const std::string path = override_dir_ + "/" + HashName(hash) + ".png";
  if (ReadPngRgba(path, &image) && image.width > 0 && image.height > 0) {
    loaded = std::make_shared<const RgbaImage>(std::move(image));
    std::fprintf(stderr, "  [vulkan] texture override loaded: %s (%ux%u)\n",
                 path.c_str(), loaded->width, loaded->height);
  }
  cache_[hash] = loaded;
  return loaded;
}

void TextureOverrides::Dump(std::uint64_t hash, std::uint32_t width,
                            std::uint32_t height, const std::uint8_t* rgba) {
  if (dump_dir_.empty() || rgba == nullptr || width == 0 || height == 0) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dumped_.insert(hash).second) {
      return;
    }
  }
  std::error_code error;
  std::filesystem::create_directories(dump_dir_, error);
  RgbaImage image;
  image.width = width;
  image.height = height;
  image.pixels.assign(rgba, rgba + static_cast<std::size_t>(width) * height * 4);
  const std::string path = dump_dir_ + "/" + HashName(hash) + "_" +
                           std::to_string(width) + "x" +
                           std::to_string(height) + ".png";
  if (!WritePngRgba(path, image)) {
    std::fprintf(stderr, "  [vulkan] texture dump failed: %s\n", path.c_str());
  }
}

}  // namespace mocktail::graphics
