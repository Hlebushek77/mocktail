#include "support/etc2_emulation_harness.h"

#include <stdlib.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>

#include "mocktail/graphics/texture_override.h"

namespace mocktail::graphics::testing {
namespace {

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Sets or clears one variable for the life of the object. The texture
// overrides read MOCKTAIL_TEXTURE_OVERRIDE_DIR and MOCKTAIL_TEXTURE_DUMP_DIR
// themselves when the emulation is constructed; Etc2EmulationOptions does
// not carry them.
class ScopedEnvironment {
 public:
  ScopedEnvironment(const char* name, const std::optional<std::string>& value)
      : name_(name) {
    if (const char* old = std::getenv(name); old != nullptr) {
      old_ = old;
    }
    if (value.has_value()) {
      setenv(name, value->c_str(), 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnvironment() {
    if (old_.has_value()) {
      setenv(name_, old_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }
  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

 private:
  const char* name_;
  std::optional<std::string> old_;
};

std::string MakeScratchDirectory(const std::string& parent) {
  std::string base = parent;
  if (base.empty()) {
    const char* tmpdir = std::getenv("TMPDIR");
    base = tmpdir != nullptr && tmpdir[0] != '\0' ? tmpdir : "/tmp";
  }
  std::string pattern = base + "/etc2-scenario-XXXXXX";
  std::vector<char> buffer(pattern.begin(), pattern.end());
  buffer.push_back('\0');
  if (mkdtemp(buffer.data()) == nullptr) {
    return {};
  }
  return buffer.data();
}

bool IsEtc2(VkFormat format) {
  Etc2EmulatedFormat emulated;
  return LookupEmulatedEtc2Format(format, &emulated);
}

std::string ImageName(const Scenario& scenario, std::uint32_t index) {
  const ScenarioImage& image = scenario.images[index];
  return "image " + std::to_string(index) + " (format " +
         std::to_string(static_cast<int>(image.format)) + " " +
         std::to_string(image.width) + "x" + std::to_string(image.height) + ")";
}

bool SameTransfer(const FakeVulkanDevice::ImageTransfer& actual,
                  const Etc2EmulationModel::Transfer& expected,
                  const std::vector<VkImage>& images) {
  if (actual.blit != expected.blit || actual.executed != expected.executed ||
      actual.source != images[expected.source] ||
      actual.destination != images[expected.destination] ||
      actual.copies.size() != expected.copies.size() ||
      actual.blits.size() != expected.blits.size() ||
      (expected.blit && actual.filter != expected.filter)) {
    return false;
  }
  return (expected.copies.empty() ||
          std::memcmp(actual.copies.data(), expected.copies.data(),
                      expected.copies.size() * sizeof(VkImageCopy)) == 0) &&
         (expected.blits.empty() ||
          std::memcmp(actual.blits.data(), expected.blits.data(),
                      expected.blits.size() * sizeof(VkImageBlit)) == 0);
}

}  // namespace

std::string GoldenLine(const std::string& name, std::uint64_t hash) {
  char text[160];
  std::snprintf(text, sizeof(text), "{\"%s\", 0x%016llxULL},", name.c_str(),
                static_cast<unsigned long long>(hash));
  return text;
}

// ---- the fake backend -------------------------------------------------------

FakeEmulationBackend::FakeEmulationBackend(FakeVulkanDevice* fake)
    : fake_(fake) {}

FakeEmulationBackend::~FakeEmulationBackend() = default;

VkDevice FakeEmulationBackend::device() const { return fake_->device(); }

VkPhysicalDevice FakeEmulationBackend::physical_device() const {
  return fake_->physical_device();
}

VkPhysicalDeviceProperties FakeEmulationBackend::properties() const {
  return fake_->config().properties;
}

VkPhysicalDeviceMemoryProperties FakeEmulationBackend::memory_properties()
    const {
  return fake_->config().memory;
}

std::vector<VkQueueFamilyProperties> FakeEmulationBackend::queue_families()
    const {
  return fake_->config().queue_families;
}

PFN_vkGetDeviceProcAddr FakeEmulationBackend::get_device_proc_addr() const {
  return FakeVulkanDevice::GetDeviceProcAddr();
}

std::string FakeEmulationBackend::Unsupported(const Scenario& scenario) const {
  if (scenario.queue_family >= fake_->config().queue_families.size()) {
    return "the fake has no queue family " +
           std::to_string(scenario.queue_family);
  }
  if (fake_->config().image_fill != kImageFill) {
    return "the fake must fill images with kImageFill";
  }
  return {};
}

VkQueue FakeEmulationBackend::Queue(std::uint32_t family) {
  return fake_->GetQueue(family, 0);
}

bool FakeEmulationBackend::CreateSource(VkDeviceSize size,
                                        VkDeviceSize bind_offset, Source* out) {
  if (fake_->CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          &out->buffer) != VK_SUCCESS) {
    return false;
  }
  VkMemoryRequirements requirements{};
  fake_->GetBufferMemoryRequirements(out->buffer, &requirements);
  // Roblox's upload memory: the first HOST_VISIBLE|HOST_COHERENT type.
  const VkPhysicalDeviceMemoryProperties& memory = fake_->config().memory;
  constexpr VkMemoryPropertyFlags kWanted =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  std::uint32_t type = UINT32_MAX;
  for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
    if ((requirements.memoryTypeBits & (1U << index)) != 0 &&
        (memory.memoryTypes[index].propertyFlags & kWanted) == kWanted) {
      type = index;
      break;
    }
  }
  out->bind_offset = AlignUp(bind_offset, requirements.alignment);
  out->memory_type = type;
  out->memory_size = out->bind_offset + requirements.size;
  return type != UINT32_MAX &&
         fake_->AllocateMemory(out->memory_size, type, &out->memory) ==
             VK_SUCCESS;
}

bool FakeEmulationBackend::WriteUnmapped(
    const Source& source, const std::vector<std::uint8_t>& bytes) {
  std::uint8_t* memory = fake_->MemoryBytes(source.memory);
  if (memory == nullptr || fake_->IsMapped(source.memory)) {
    return false;
  }
  std::memcpy(memory + source.bind_offset, bytes.data(), bytes.size());
  return true;
}

bool FakeEmulationBackend::PrepareImage(VkImage image,
                                        const HostImageShape& shape) {
  // The fake fills new images itself; check it made what the model expects.
  VkImageCreateInfo info{};
  return fake_->GetImageInfo(image, &info);
}

void FakeEmulationBackend::ReleaseImage(VkImage) {}

VkCommandPool FakeEmulationBackend::CreateCommandPool(
    std::uint32_t queue_family, VkCommandPoolCreateFlags flags) {
  return fake_->CreateCommandPool(queue_family, flags);
}

bool FakeEmulationBackend::ResetCommandPool(VkCommandPool pool) {
  return fake_->ResetCommandPool(pool) == VK_SUCCESS;
}

void FakeEmulationBackend::DestroyCommandPool(VkCommandPool pool) {
  fake_->DestroyCommandPool(pool);
}

VkCommandBuffer FakeEmulationBackend::AllocateCommandBuffer(VkCommandPool pool,
                                                            bool secondary) {
  return fake_->AllocateCommandBuffer(
      pool, secondary ? VK_COMMAND_BUFFER_LEVEL_SECONDARY
                      : VK_COMMAND_BUFFER_LEVEL_PRIMARY);
}

void FakeEmulationBackend::FreeCommandBuffer(VkCommandPool,
                                             VkCommandBuffer command_buffer) {
  fake_->FreeCommandBuffer(command_buffer);
}

bool FakeEmulationBackend::Begin(VkCommandBuffer command_buffer, bool,
                                 VkCommandBufferUsageFlags flags) {
  return fake_->BeginCommandBuffer(command_buffer, flags) == VK_SUCCESS;
}

bool FakeEmulationBackend::End(VkCommandBuffer command_buffer) {
  return fake_->EndCommandBuffer(command_buffer) == VK_SUCCESS;
}

bool FakeEmulationBackend::Reset(VkCommandBuffer command_buffer) {
  return fake_->ResetCommandBuffer(command_buffer) == VK_SUCCESS;
}

void FakeEmulationBackend::BeforeTransfer(VkCommandBuffer) {}

VkResult FakeEmulationBackend::QueueSubmit(VkQueue queue, std::uint32_t count,
                                           const VkSubmitInfo* submits,
                                           VkFence fence) {
  return fake_->QueueSubmit(queue, count, submits, fence);
}

VkResult FakeEmulationBackend::QueueSubmit2(VkQueue queue, std::uint32_t count,
                                            const VkSubmitInfo2* submits,
                                            VkFence fence) {
  return fake_->QueueSubmit2(queue, count, submits, fence);
}

bool FakeEmulationBackend::WaitIdle(VkQueue) { return true; }

bool FakeEmulationBackend::ReadImage(
    VkImage image, const HostImageShape& shape,
    std::vector<std::vector<std::uint8_t>>* texels) {
  texels->clear();
  for (std::uint32_t mip = 0; mip < shape.mip_levels; ++mip) {
    for (std::uint32_t layer = 0; layer < shape.array_layers; ++layer) {
      const std::vector<std::uint8_t>* subresource =
          fake_->ImageSubresource(image, mip, layer);
      if (subresource == nullptr) {
        return false;
      }
      texels->push_back(*subresource);
    }
  }
  return true;
}

void FakeEmulationBackend::DestroyImageDirect(VkImage image) {
  fake_->DestroyImage(image);
}

void FakeEmulationBackend::DestroyBufferDirect(VkBuffer buffer) {
  fake_->DestroyBuffer(buffer);
}

void FakeEmulationBackend::UnmapMemoryDirect(VkDeviceMemory memory) {
  fake_->UnmapMemory(memory);
}

void FakeEmulationBackend::FreeMemoryDirect(VkDeviceMemory memory) {
  fake_->FreeMemory(memory);
}

// ---- the runner -------------------------------------------------------------

Etc2EmulationOptions CpuRouteOptions() {
  Etc2EmulationOptions options;
  options.gpu = false;
  return options;
}

ScenarioRun RunEtc2Scenario(const Scenario& scenario, EmulationBackend* backend,
                            const ScenarioOptions& options) {
  ScenarioRun run;
  run.skip_reason = backend->Unsupported(scenario);
  if (!run.skip_reason.empty()) {
    run.skipped = true;
    return run;
  }
  auto fail = [&run](std::string what) {
    run.failures.push_back(std::move(what));
  };

  std::string scratch;
  std::optional<std::string> override_directory;
  std::optional<std::string> dump_directory;
  if (scenario.overrides) {
    scratch = MakeScratchDirectory(options.scratch_directory);
    if (scratch.empty()) {
      fail("could not create a scratch directory");
      return run;
    }
    override_directory = scratch + "/overrides";
    dump_directory = scratch + "/dumps";
    std::error_code error;
    std::filesystem::create_directories(*override_directory, error);
  }
  const ScopedEnvironment override_environment("MOCKTAIL_TEXTURE_OVERRIDE_DIR",
                                               override_directory);
  const ScopedEnvironment dump_environment("MOCKTAIL_TEXTURE_DUMP_DIR",
                                           dump_directory);

  // ---- the adapter's vkCreateDevice and vkGetDeviceQueue hooks ----
  Etc2EmulationOptions emulation_options = options.emulation;
  emulation_options.upscale = scenario.upscale;
  VulkanEtc2Emulation emulation(emulation_options);
  const VkDevice device = backend->device();
  const float priority = 1.0F;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = scenario.queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  emulation.RegisterDevice(
      device, backend->physical_device(), scenario.emulated,
      backend->memory_properties(), backend->properties(),
      backend->queue_families(), device_info, backend->get_device_proc_addr(),
      backend->get_instance_proc_addr(), backend->instance());
  const VkQueue queue = backend->Queue(scenario.queue_family);
  if (queue == VK_NULL_HANDLE) {
    fail("no queue in family " + std::to_string(scenario.queue_family));
  }
  emulation.RegisterQueue(device, queue, scenario.queue_family, 0);
  Etc2EmulationModel model(scenario);

  std::vector<VkImage> images;
  for (std::uint32_t index = 0; index < scenario.images.size(); ++index) {
    const ScenarioImage& spec = scenario.images[index];
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = spec.format;
    info.extent = {spec.width, spec.height, 1};
    info.mipLevels = spec.mip_levels;
    info.arrayLayers = spec.array_layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    const VkResult result =
        emulation.CreateImage(device, &info, nullptr, &image);
    if (result != VK_SUCCESS || image == VK_NULL_HANDLE) {
      fail("CreateImage failed for " + ImageName(scenario, index) + ": " +
           std::to_string(result));
      break;
    }
    images.push_back(image);
    const HostImageShape& shape = model.host_shape(index);
    if (options.fake != nullptr) {
      VkImageCreateInfo host{};
      options.fake->GetImageInfo(image, &host);
      if (host.format != shape.format || host.extent.width != shape.width ||
          host.extent.height != shape.height ||
          host.mipLevels != shape.mip_levels ||
          host.arrayLayers != shape.array_layers) {
        fail(ImageName(scenario, index) + ": host image is format " +
             std::to_string(static_cast<int>(host.format)) + " " +
             std::to_string(host.extent.width) + "x" +
             std::to_string(host.extent.height) + ", the model expects " +
             std::to_string(static_cast<int>(shape.format)) + " " +
             std::to_string(shape.width) + "x" + std::to_string(shape.height));
      }
    }
    if (!backend->PrepareImage(image, shape)) {
      fail("could not prepare " + ImageName(scenario, index));
    }
  }

  std::vector<EmulationBackend::Source> sources(scenario.sources.size());
  std::vector<std::vector<std::uint8_t>> bytes(scenario.sources.size());
  std::vector<std::uint8_t*> mapped(scenario.sources.size(), nullptr);
  for (std::size_t index = 0;
       index < scenario.sources.size() && run.failures.empty(); ++index) {
    const ScenarioSource& spec = scenario.sources[index];
    bytes[index] =
        SeededBytes(spec.seed, std::max<VkDeviceSize>(spec.size, 16));
    EmulationBackend::Source& source = sources[index];
    if (!backend->CreateSource(bytes[index].size(), spec.bind_offset,
                               &source)) {
      fail("could not create source " + std::to_string(index));
      break;
    }
    // The adapter's vkAllocateMemory hook notes the memory type of every
    // allocation on an emulated device.
    if (scenario.emulated) {
      emulation.NoteAllocateMemory(device, source.memory, source.memory_type,
                                   source.memory_size);
    }
    // The adapter routes vkBindBufferMemory and vkMapMemory through the
    // emulation so it can find the bytes at submit time.
    if (emulation.BindBufferMemory(device, source.buffer, source.memory,
                                   source.bind_offset) != VK_SUCCESS) {
      fail("BindBufferMemory failed for source " + std::to_string(index));
      break;
    }
    if (spec.mapped) {
      const VkDeviceSize map_offset =
          std::min(spec.map_offset, source.bind_offset);
      void* data = nullptr;
      if (emulation.MapMemory(device, source.memory, map_offset, VK_WHOLE_SIZE,
                              0, &data) != VK_SUCCESS) {
        fail("MapMemory failed for source " + std::to_string(index));
        break;
      }
      mapped[index] =
          static_cast<std::uint8_t*>(data) + (source.bind_offset - map_offset);
      std::memcpy(mapped[index], bytes[index].data(), bytes[index].size());
    } else if (!backend->WriteUnmapped(source, bytes[index])) {
      fail("could not write source " + std::to_string(index));
      break;
    }
  }

  // ---- the adapter's vkCreateCommandPool hook ----
  constexpr VkCommandPoolCreateFlags kPoolFlags =
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  const VkCommandPool pool =
      backend->CreateCommandPool(scenario.queue_family, kPoolFlags);
  if (pool == VK_NULL_HANDLE) {
    fail("could not create a command pool");
  } else {
    emulation.RegisterCommandPool(device, pool, scenario.queue_family,
                                  kPoolFlags);
  }
  enum class State { kInitial, kRecording, kExecutable };
  std::vector<VkCommandBuffer> command_buffers;
  std::vector<State> states(scenario.command_buffers.size(), State::kInitial);
  for (const ScenarioCommandBuffer& spec : scenario.command_buffers) {
    command_buffers.push_back(
        pool != VK_NULL_HANDLE
            ? backend->AllocateCommandBuffer(pool, spec.secondary)
            : VK_NULL_HANDLE);
    if (command_buffers.back() == VK_NULL_HANDLE) {
      fail("could not allocate a command buffer");
    }
  }
  constexpr VkCommandBufferUsageFlags kBeginFlags = 0;
  const auto recording = [&](std::uint32_t index) {
    if (states[index] == State::kRecording) {
      return;
    }
    if (states[index] != State::kInitial) {
      model.Reset(index);  // vkBeginCommandBuffer resets implicitly
    }
    // The adapter's vkBeginCommandBuffer hook releases the old recording.
    emulation.BeginCommandBuffer(command_buffers[index], pool, kBeginFlags);
    ++run.calls.begins;
    if (!backend->Begin(command_buffers[index],
                        scenario.command_buffers[index].secondary,
                        kBeginFlags)) {
      fail("vkBeginCommandBuffer failed");
    }
    states[index] = State::kRecording;
  };
  const auto executable = [&](std::uint32_t index) {
    if (states[index] == State::kInitial) {
      recording(index);
    }
    if (states[index] == State::kRecording) {
      if (!backend->End(command_buffers[index])) {
        fail("vkEndCommandBuffer failed");
      }
      states[index] = State::kExecutable;
    }
  };

  GoldenHasher hasher;
  std::uint32_t checkpoints = 0;
  const auto checkpoint = [&]() {
    hasher.AddU32(checkpoints);
    for (std::uint32_t index = 0; index < images.size(); ++index) {
      const HostImageShape& shape = model.host_shape(index);
      hasher.AddU32(static_cast<std::uint32_t>(shape.format));
      hasher.AddU32(shape.width);
      hasher.AddU32(shape.height);
      hasher.AddU32(shape.mip_levels);
      hasher.AddU32(shape.array_layers);
      std::vector<std::vector<std::uint8_t>> texels;
      if (!backend->ReadImage(images[index], shape, &texels) ||
          texels.size() != shape.mip_levels * shape.array_layers) {
        fail("could not read back " + ImageName(scenario, index));
        continue;
      }
      for (std::uint32_t mip = 0; mip < shape.mip_levels; ++mip) {
        for (std::uint32_t layer = 0; layer < shape.array_layers; ++layer) {
          const std::vector<std::uint8_t>& actual =
              texels[mip * shape.array_layers + layer];
          const std::vector<std::uint8_t>& expected =
              model.Subresource(index, mip, layer);
          hasher.Add(actual.data(), actual.size());
          if (actual == expected) {
            continue;
          }
          std::size_t first = 0;
          std::size_t different = 0;
          for (std::size_t byte = 0;
               byte < std::min(actual.size(), expected.size()); ++byte) {
            if (actual[byte] != expected[byte]) {
              first = different++ == 0 ? byte : first;
            }
          }
          fail("checkpoint " + std::to_string(checkpoints) + ", " +
               ImageName(scenario, index) + " mip " + std::to_string(mip) +
               " layer " + std::to_string(layer) + ": " +
               std::to_string(different) + " of " +
               std::to_string(expected.size()) +
               " bytes differ from the model, the first at byte " +
               std::to_string(first) + " (" +
               std::to_string(first < actual.size() ? actual[first] : -1) +
               " instead of " +
               std::to_string(first < expected.size() ? expected[first] : -1) +
               ")");
        }
      }
    }
    ++checkpoints;
  };

  const VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
  for (const ScenarioStep& step : scenario.steps) {
    if (!run.failures.empty()) {
      break;
    }
    switch (step.kind) {
      case ScenarioStep::Kind::kCopyBufferToImage: {
        const std::uint32_t index = step.command_buffer;
        recording(index);
        backend->BeforeTransfer(command_buffers[index]);
        if (step.copy2) {
          std::vector<VkBufferImageCopy2> regions;
          for (const VkBufferImageCopy& region : step.regions) {
            VkBufferImageCopy2 copy{};
            copy.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2;
            copy.bufferOffset = region.bufferOffset;
            copy.bufferRowLength = region.bufferRowLength;
            copy.bufferImageHeight = region.bufferImageHeight;
            copy.imageSubresource = region.imageSubresource;
            copy.imageOffset = region.imageOffset;
            copy.imageExtent = region.imageExtent;
            regions.push_back(copy);
          }
          VkCopyBufferToImageInfo2 info{};
          info.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2;
          info.srcBuffer = sources[step.source].buffer;
          info.dstImage = images[step.image];
          info.dstImageLayout = layout;
          info.regionCount = static_cast<std::uint32_t>(regions.size());
          info.pRegions = regions.data();
          emulation.CmdCopyBufferToImage2(device, command_buffers[index],
                                          &info);
        } else {
          emulation.CmdCopyBufferToImage(
              device, command_buffers[index], sources[step.source].buffer,
              images[step.image], layout,
              static_cast<std::uint32_t>(step.regions.size()),
              step.regions.data());
        }
        if (IsEtc2(scenario.images[step.image].format)) {
          ++run.calls.etc2_copies;
          run.calls.etc2_regions += step.regions.size();
        }
        model.CopyBufferToImage(index, step.image, step.source, step.regions);
        break;
      }
      case ScenarioStep::Kind::kCopyImage: {
        const std::uint32_t index = step.command_buffer;
        recording(index);
        backend->BeforeTransfer(command_buffers[index]);
        if (step.copy2) {
          std::vector<VkImageCopy2> regions;
          for (const VkImageCopy& region : step.image_regions) {
            VkImageCopy2 copy{};
            copy.sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2;
            copy.srcSubresource = region.srcSubresource;
            copy.srcOffset = region.srcOffset;
            copy.dstSubresource = region.dstSubresource;
            copy.dstOffset = region.dstOffset;
            copy.extent = region.extent;
            regions.push_back(copy);
          }
          VkCopyImageInfo2 info{};
          info.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2;
          info.srcImage = images[step.image];
          info.srcImageLayout = layout;
          info.dstImage = images[step.image2];
          info.dstImageLayout = layout;
          info.regionCount = static_cast<std::uint32_t>(regions.size());
          info.pRegions = regions.data();
          emulation.CmdCopyImage2(device, command_buffers[index], &info);
        } else {
          emulation.CmdCopyImage(
              device, command_buffers[index], images[step.image], layout,
              images[step.image2], layout,
              static_cast<std::uint32_t>(step.image_regions.size()),
              step.image_regions.data());
        }
        if (IsEtc2(scenario.images[step.image].format) ||
            IsEtc2(scenario.images[step.image2].format)) {
          ++run.calls.etc2_image_copies;
        }
        model.CopyImage(index, step.image, step.image2, step.image_regions);
        break;
      }
      case ScenarioStep::Kind::kExecuteCommands: {
        std::vector<VkCommandBuffer> secondaries;
        for (std::uint32_t secondary : step.command_buffers) {
          executable(secondary);
          secondaries.push_back(command_buffers[secondary]);
        }
        recording(step.command_buffer);
        emulation.CmdExecuteCommands(
            device, command_buffers[step.command_buffer],
            static_cast<std::uint32_t>(secondaries.size()), secondaries.data());
        run.calls.secondaries_executed += secondaries.size();
        model.ExecuteCommands(step.command_buffer, step.command_buffers);
        break;
      }
      case ScenarioStep::Kind::kSubmit: {
        std::vector<std::vector<VkCommandBuffer>> batches;
        for (const auto& batch : step.batches) {
          std::vector<VkCommandBuffer> handles;
          for (std::uint32_t index : batch) {
            executable(index);
            handles.push_back(command_buffers[index]);
          }
          batches.push_back(std::move(handles));
        }
        // ---- the adapter's vkQueueSubmit and vkQueueSubmit2 hooks ----
        const auto count = static_cast<std::uint32_t>(batches.size());
        Etc2SubmitPlan plan;
        const auto pre_submit = [&]() {
          for (const Etc2PreSubmit& pre : plan.pre_submits()) {
            if (backend->QueueSubmit(queue, 1, &pre.info, pre.fence) !=
                VK_SUCCESS) {
              fail("a pre-submit of the plan failed");
            }
          }
        };
        VkResult result = VK_SUCCESS;
        std::vector<VkSubmitInfo> infos;
        std::vector<std::vector<VkCommandBufferSubmitInfo>> buffer_infos;
        std::vector<VkSubmitInfo2> infos2;
        if (step.shape == SubmitShape::kQueueSubmit) {
          for (const auto& batch : batches) {
            VkSubmitInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            info.commandBufferCount = static_cast<std::uint32_t>(batch.size());
            info.pCommandBuffers = batch.empty() ? nullptr : batch.data();
            infos.push_back(info);
          }
          emulation.PrepareSubmit(queue, count, infos.data(), VK_NULL_HANDLE,
                                  &plan);
          ++run.calls.submit_calls;
          run.calls.submit_batches += count;
          pre_submit();
          result = backend->QueueSubmit(queue, plan.count(count),
                                        plan.submits(infos.data()),
                                        plan.fence(VK_NULL_HANDLE));
        } else {
          for (const auto& batch : batches) {
            std::vector<VkCommandBufferSubmitInfo> buffers;
            for (VkCommandBuffer command_buffer : batch) {
              VkCommandBufferSubmitInfo buffer{};
              buffer.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
              buffer.commandBuffer = command_buffer;
              buffers.push_back(buffer);
            }
            buffer_infos.push_back(std::move(buffers));
          }
          for (const auto& buffers : buffer_infos) {
            VkSubmitInfo2 info{};
            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            info.commandBufferInfoCount =
                static_cast<std::uint32_t>(buffers.size());
            info.pCommandBufferInfos =
                buffers.empty() ? nullptr : buffers.data();
            infos2.push_back(info);
          }
          emulation.PrepareSubmit2(queue, count, infos2.data(), VK_NULL_HANDLE,
                                   &plan);
          ++run.calls.submit2_calls;
          run.calls.submit2_batches += count;
          pre_submit();
          result = backend->QueueSubmit2(queue, plan.count(count),
                                         plan.submits2(infos2.data()),
                                         plan.fence(VK_NULL_HANDLE));
        }
        emulation.FinishSubmit(&plan, result, [&](VkFence fence) {
          return backend->QueueSubmit(queue, 0, nullptr, fence);
        });
        if (result != VK_SUCCESS || !backend->WaitIdle(queue)) {
          fail("the submit failed: " + std::to_string(result));
        }
        model.Submit(step.batches, bytes);
        break;
      }
      case ScenarioStep::Kind::kReset:
        for (std::uint32_t index : step.command_buffers) {
          // The adapter's vkResetCommandBuffer hook.
          emulation.ReleaseCommandBuffer(command_buffers[index]);
          if (!backend->Reset(command_buffers[index])) {
            fail("vkResetCommandBuffer failed");
          }
          states[index] = State::kInitial;
          model.Reset(index);
        }
        break;
      case ScenarioStep::Kind::kRewriteSource: {
        std::vector<std::uint8_t>& data = bytes[step.source];
        data = SeededBytes(step.seed, data.size());
        if (mapped[step.source] != nullptr) {
          std::memcpy(mapped[step.source], data.data(), data.size());
        } else if (!backend->WriteUnmapped(sources[step.source], data)) {
          fail("could not rewrite source " + std::to_string(step.source));
        }
        break;
      }
      case ScenarioStep::Kind::kWriteOverride: {
        const std::uint64_t hash = Etc2EmulationModel::OverrideHash(
            Etc2EmulationModel::TightCompressed(
                scenario.images[step.image].format, step.regions[0],
                bytes[step.source]));
        auto png = std::make_shared<RgbaImage>();
        png->width = step.override_width;
        png->height = step.override_height;
        png->pixels = SeededBytes(
            step.seed, static_cast<std::size_t>(png->width) * png->height * 4);
        if (!override_directory.has_value() ||
            !WritePngRgba(*override_directory + "/" + HashName(hash) + ".png",
                          *png)) {
          fail("could not write an override PNG");
          break;
        }
        model.AddOverride(hash, png);
        break;
      }
      case ScenarioStep::Kind::kCheckpoint:
        checkpoint();
        break;
    }
  }
  if (run.failures.empty()) {
    checkpoint();
  }
  run.hash = hasher.value();

  if (dump_directory.has_value() && run.failures.empty()) {
    std::size_t files = 0;
    std::error_code error;
    for (const auto& entry :
         std::filesystem::directory_iterator(*dump_directory, error)) {
      files += entry.is_regular_file() ? 1 : 0;
    }
    if (files != model.dumps().size()) {
      fail(std::to_string(files) + " texture dumps, the model expects " +
           std::to_string(model.dumps().size()));
    }
    for (const Etc2EmulationModel::Dump& dump : model.dumps()) {
      const std::string path = *dump_directory + "/" + HashName(dump.hash) +
                               "_" + std::to_string(dump.width) + "x" +
                               std::to_string(dump.height) + ".png";
      RgbaImage image;
      if (!ReadPngRgba(path, &image) || image.pixels != dump.rgba) {
        fail("dump " + path + " is missing or differs");
      }
    }
  }

  if (options.fake != nullptr && run.failures.empty()) {
    const auto& actual = options.fake->image_transfers();
    const auto& expected = model.transfers();
    if (actual.size() != expected.size()) {
      fail(std::to_string(actual.size()) + " image copies or blits ran, the " +
           "model expects " + std::to_string(expected.size()));
    } else {
      for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!SameTransfer(actual[index], expected[index], images)) {
          fail("image copy or blit " + std::to_string(index) +
               " differs from the model (blit " +
               std::to_string(actual[index].blit) + ", filter " +
               std::to_string(actual[index].filter) + ")");
        }
      }
    }
  }

  if (options.after_steps) {
    options.after_steps(&emulation);
  }

  // ---- teardown, as the adapter's hooks see Roblox's calls ----
  const bool device_first =
      options.teardown == ScenarioTeardown::kDestroyDeviceFirst;
  if (device_first) {
    emulation.DestroyDevice(device);
  }
  // vkResetCommandPool and vkDestroyCommandPool release every command
  // buffer of the pool first (ReleaseEtc2PoolCommandBuffers).
  const auto release_pool = [&]() {
    for (VkCommandBuffer command_buffer : command_buffers) {
      if (command_buffer != VK_NULL_HANDLE) {
        emulation.ReleaseCommandBuffer(command_buffer);
      }
    }
  };
  if (options.teardown == ScenarioTeardown::kResetPool &&
      pool != VK_NULL_HANDLE) {
    release_pool();
    if (!backend->ResetCommandPool(pool)) {
      fail("vkResetCommandPool failed");
    }
  }
  if (options.teardown != ScenarioTeardown::kDestroyPool) {
    for (VkCommandBuffer& command_buffer : command_buffers) {
      if (command_buffer != VK_NULL_HANDLE) {
        // vkFreeCommandBuffers releases each one first.
        emulation.ReleaseCommandBuffer(command_buffer);
        backend->FreeCommandBuffer(pool, command_buffer);
        command_buffer = VK_NULL_HANDLE;
      }
    }
  }
  if (pool != VK_NULL_HANDLE) {
    release_pool();
    backend->DestroyCommandPool(pool);
    emulation.ForgetCommandPool(device, pool);
  }
  if (!device_first && options.after_command_buffers) {
    options.after_command_buffers();
  }
  const bool orderly = !device_first;
  for (VkImage image : images) {
    if (orderly) {
      emulation.DestroyImage(device, image, nullptr);
    } else {
      backend->DestroyImageDirect(image);
    }
    backend->ReleaseImage(image);
  }
  for (std::size_t index = 0; index < sources.size(); ++index) {
    const EmulationBackend::Source& source = sources[index];
    if (source.buffer == VK_NULL_HANDLE) {
      continue;
    }
    if (orderly) {
      if (mapped[index] != nullptr) {
        emulation.UnmapMemory(device, source.memory);
      }
      emulation.DestroyBuffer(device, source.buffer, nullptr);
      emulation.FreeMemory(device, source.memory, nullptr);
    } else {
      if (mapped[index] != nullptr) {
        backend->UnmapMemoryDirect(source.memory);
      }
      backend->DestroyBufferDirect(source.buffer);
      backend->FreeMemoryDirect(source.memory);
    }
  }
  if (!device_first) {
    emulation.DestroyDevice(device);
  }
  if (!scratch.empty()) {
    std::error_code error;
    std::filesystem::remove_all(scratch, error);
  }
  return run;
}

}  // namespace mocktail::graphics::testing
