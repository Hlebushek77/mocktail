#ifndef MOCKTAIL_GRAPHICS_VULKAN_ETC2_GPU_H_
#define MOCKTAIL_GRAPHICS_VULKAN_ETC2_GPU_H_

namespace mocktail::graphics {

// The GPU decode runtime of one emulated device (DESIGN C3): pipelines,
// output arenas and per-queue submit slots. The emulation owns one per
// device once the GPU route exists; until then no device has one and every
// upload takes the CPU route.
class Etc2GpuDevice {
 public:
  // Never calls Vulkan and never joins a thread: when the process exits
  // without vkDestroyDevice, the device may already be gone.
  virtual ~Etc2GpuDevice() = default;
  // vkDestroyDevice: joins the pipeline build, waits for Mocktail's own
  // fences (1 s at most) and destroys every Vulkan object it made.
  virtual void Destroy() = 0;
};

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_VULKAN_ETC2_GPU_H_
