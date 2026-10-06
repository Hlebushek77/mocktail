#ifndef MOCKTAIL_GRAPHICS_VULKAN_ETC2_CAPS_H_
#define MOCKTAIL_GRAPHICS_VULKAN_ETC2_CAPS_H_

// Which ETC2/EAC capabilities an emulated device reports. Kept free of
// Vulkan calls so the selection is unit-testable; the adapter's feature and
// format-property hooks feed it what they know about the physical device.
namespace mocktail::graphics {

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
// the query goes to the driver with the ETC2/EAC format unchanged. The
// formats follow textureCompressionETC2, so with MOCKTAIL_ADVERTISE_ETC2=0
// they report what the bare device reports: no features and
// VK_ERROR_FORMAT_NOT_SUPPORTED.
constexpr bool Etc2FormatsAdvertised(bool emulated_device,
                                     bool support_advertised) {
  return Etc2FeatureAdvertised(emulated_device, support_advertised);
}

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_VULKAN_ETC2_CAPS_H_
