#ifndef MOCKTAIL_GRAPHICS_VULKAN_ETC2_CAPS_H_
#define MOCKTAIL_GRAPHICS_VULKAN_ETC2_CAPS_H_

#include <cstdlib>
#include <cstring>

// Which ETC2/EAC capabilities an emulated device reports. Kept free of
// Vulkan calls so the selection is unit-testable; the adapter's feature and
// format-property hooks feed it what they know about the physical device.
namespace mocktail::graphics {

// How the format-property hooks answer for ETC2/EAC formats,
// MOCKTAIL_ETC2_FORMAT_PROPS. A switch for the plus.2 DXT probe only.
enum class Etc2FormatPropsMode {
  // The formats follow textureCompressionETC2: with MOCKTAIL_ADVERTISE_ETC2=0
  // they report what the bare device reports, no features and
  // VK_ERROR_FORMAT_NOT_SUPPORTED.
  kConsistent,
  // MOCKTAIL_ETC2_FORMAT_PROPS=legacy, plus.1: an emulated device reports the
  // emulated features even while the feature bit is hidden.
  kLegacy,
};

// Only "legacy" selects kLegacy; unset or any other value is kConsistent.
inline Etc2FormatPropsMode ParseEtc2FormatPropsMode(const char* value) {
  return value != nullptr && std::strcmp(value, "legacy") == 0
             ? Etc2FormatPropsMode::kLegacy
             : Etc2FormatPropsMode::kConsistent;
}

inline Etc2FormatPropsMode Etc2FormatPropsModeFromEnvironment() {
  static const Etc2FormatPropsMode mode =
      ParseEtc2FormatPropsMode(std::getenv("MOCKTAIL_ETC2_FORMAT_PROPS"));
  return mode;
}

// vkGetPhysicalDeviceFeatures(2) reports textureCompressionETC2 for an
// emulated device unless MOCKTAIL_ADVERTISE_ETC2=0 hides it. A device that is
// not emulated reports what its driver reports.
constexpr bool Etc2FeatureAdvertised(bool emulated_device,
                                     bool support_advertised) {
  return emulated_device && support_advertised;
}

// Whether vkGetPhysicalDeviceFormatProperties(2) and
// vkGetPhysicalDeviceImageFormatProperties(2) answer for an ETC2/EAC format
// with the emulated features and limits (from the host format). When false
// the query goes to the driver with the ETC2/EAC format unchanged.
constexpr bool Etc2FormatsAdvertised(bool emulated_device,
                                     bool support_advertised,
                                     Etc2FormatPropsMode mode) {
  return emulated_device &&
         (support_advertised || mode == Etc2FormatPropsMode::kLegacy);
}

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_VULKAN_ETC2_CAPS_H_
