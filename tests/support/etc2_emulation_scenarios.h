#ifndef MOCKTAIL_TESTS_SUPPORT_ETC2_EMULATION_SCENARIOS_H_
#define MOCKTAIL_TESTS_SUPPORT_ETC2_EMULATION_SCENARIOS_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/texture_override.h"
#include "support/fake_vulkan_device.h"

// Roblox-shaped upload scenarios for the ETC2/EAC emulation, a reference
// model of what each should leave in the host images, and the golden hashes
// frozen from the plus.1 code. None of this depends on the emulation's API:
// support/etc2_emulation_harness.cc drives the emulation through a scenario,
// and the same scenario runs against the fake device and a real GPU.
namespace mocktail::graphics::testing {

// What every subresource of a fresh host image holds, on both backends.
inline constexpr std::uint8_t kImageFill = 0xCD;

struct ScenarioImage {
  VkFormat format = VK_FORMAT_UNDEFINED;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t mip_levels = 1;
  std::uint32_t array_layers = 1;
};

// Roblox's upload buffer: a TRANSFER_SRC buffer in host-visible memory,
// bound at `bind_offset` and either kept mapped from `map_offset` (as Roblox
// does) or left unmapped, so the emulation maps it for the submit.
struct ScenarioSource {
  std::uint64_t seed = 0;
  VkDeviceSize size = 0;
  bool mapped = true;
  VkDeviceSize bind_offset = 0;
  VkDeviceSize map_offset = 0;
};

struct ScenarioCommandBuffer {
  bool secondary = false;
};

// Which hook the batches go through: vkQueueSubmit (VkSubmitInfo) or
// vkQueueSubmit2 (VkSubmitInfo2). Before plus.2 the vkQueueSubmit2 hook
// decoded each command buffer on its own; both now decode a batch at once.
enum class SubmitShape { kQueueSubmit, kQueueSubmit2 };

struct ScenarioStep {
  enum class Kind {
    kCopyBufferToImage,
    kCopyImage,
    kExecuteCommands,
    kSubmit,
    kReset,          // vkResetCommandBuffer, after the emulation releases it
    kRewriteSource,  // Roblox rewrites its upload buffer
    kWriteOverride,  // an override PNG for the compressed bytes of `regions[0]`
    kCheckpoint,     // read every image back and compare
  };
  Kind kind = Kind::kCheckpoint;
  std::uint32_t command_buffer = 0;
  std::uint32_t image = 0;   // destination; the source image for kCopyImage
  std::uint32_t image2 = 0;  // kCopyImage destination
  std::uint32_t source = 0;
  bool copy2 = false;  // vkCmdCopyBufferToImage2 / vkCmdCopyImage2
  std::vector<VkBufferImageCopy> regions;
  std::vector<VkImageCopy> image_regions;
  std::vector<std::uint32_t> command_buffers;  // secondaries, or to reset
  std::vector<std::vector<std::uint32_t>> batches;
  SubmitShape shape = SubmitShape::kQueueSubmit;
  std::uint64_t seed = 0;
  std::uint32_t override_width = 0;
  std::uint32_t override_height = 0;
};

struct Scenario {
  std::string name;
  std::string description;
  std::uint32_t upscale = 1;  // MOCKTAIL_SMALL_TEXTURE_UPSCALE
  bool emulated = true;       // the device lacks ETC2
  bool overrides = false;     // texture override and dump directories set
  std::uint32_t queue_family = 0;
  bool needs_copy_commands2 = false;
  // Why a real GPU cannot run it; empty when it can.
  std::string fake_only;
  // Fake-device errors this scenario expects today, as substrings.
  std::vector<std::string> allowed_fake_errors;
  std::vector<ScenarioImage> images;
  std::vector<ScenarioSource> sources;
  std::vector<ScenarioCommandBuffer> command_buffers;
  std::vector<ScenarioStep> steps;
};

const std::vector<Scenario>& Etc2EmulationScenarios();
const Scenario* FindEtc2EmulationScenario(const std::string& name);

std::vector<std::uint8_t> SeededBytes(std::uint64_t seed, std::size_t size);

// The image the emulation creates on the host for `image` (format, extent
// after the small-texture upscale).
struct HostImageShape {
  VkFormat format = VK_FORMAT_UNDEFINED;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t mip_levels = 1;
  std::uint32_t array_layers = 1;
};
HostImageShape HostShapeFor(const ScenarioImage& image, std::uint32_t upscale,
                            bool emulated);

// FNV-1a 64, folded over every checkpoint's images. Goldens are these.
class GoldenHasher {
 public:
  void Add(const void* data, std::size_t size);
  void AddU32(std::uint32_t value);
  std::uint64_t value() const { return hash_; }

 private:
  std::uint64_t hash_ = 0xcbf29ce484222325ULL;
};

// The frozen hash for a scenario, from vulkan_etc2_emulation_golden.inc.
bool FindEtc2EmulationGolden(const std::string& name, std::uint64_t* hash);

// What the plus.1 emulation does, written from its rules rather than its
// code: ETC2/EAC copies decode with the reference decoder at submit time and
// land in the host region; x4-upscaled colour images resample with the
// reference box filter (mip 0 sets the base image that later full mips are
// resampled from); override PNGs replace mip-0 colour uploads by hash; image
// copies touching an upscaled image become blits. Images are indexed in
// scenario order, command buffers likewise.
class Etc2EmulationModel {
 public:
  struct Transfer {
    bool blit = false;
    bool executed = true;
    std::uint32_t source = 0;
    std::uint32_t destination = 0;
    std::vector<VkImageCopy> copies;
    std::vector<VkImageBlit> blits;
    VkFilter filter = VK_FILTER_NEAREST;
  };
  struct Dump {
    std::uint64_t hash = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> rgba;
  };

  explicit Etc2EmulationModel(const Scenario& scenario);
  ~Etc2EmulationModel();

  const HostImageShape& host_shape(std::uint32_t image) const;
  // Returns false when the emulation drops the whole command.
  bool CopyBufferToImage(std::uint32_t command_buffer, std::uint32_t image,
                         std::uint32_t source,
                         const std::vector<VkBufferImageCopy>& regions);
  void CopyImage(std::uint32_t command_buffer, std::uint32_t source,
                 std::uint32_t destination,
                 const std::vector<VkImageCopy>& regions);
  void ExecuteCommands(std::uint32_t primary,
                       const std::vector<std::uint32_t>& secondaries);
  void Reset(std::uint32_t command_buffer);
  void AddOverride(std::uint64_t hash, std::shared_ptr<const RgbaImage> image);
  void Submit(const std::vector<std::vector<std::uint32_t>>& batches,
              const std::vector<std::vector<std::uint8_t>>& sources);
  const std::vector<std::uint8_t>& Subresource(std::uint32_t image,
                                               std::uint32_t mip,
                                               std::uint32_t layer) const;
  const std::vector<Dump>& dumps() const;
  const std::vector<Transfer>& transfers() const;

  // The compressed bytes of one region, rows packed, as the emulation hashes
  // them for override lookups.
  static std::vector<std::uint8_t> TightCompressed(
      VkFormat format, const VkBufferImageCopy& region,
      const std::vector<std::uint8_t>& source);
  static std::uint64_t OverrideHash(const std::vector<std::uint8_t>& bytes);

  struct State;

 private:
  std::unique_ptr<State> state_;
};

}  // namespace mocktail::graphics::testing

#endif  // MOCKTAIL_TESTS_SUPPORT_ETC2_EMULATION_SCENARIOS_H_
