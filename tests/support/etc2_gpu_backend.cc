#include "support/etc2_gpu_backend.h"

#include <cstring>

namespace mocktail::graphics::testing {
namespace {

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

std::uint32_t Mip(std::uint32_t base, std::uint32_t mip) {
  return std::max<std::uint32_t>(1, base >> mip);
}

VkDeviceSize SubresourceBytes(const HostImageShape& shape, std::uint32_t mip) {
  FormatBlock block;
  LookupFormatBlock(shape.format, &block);
  return static_cast<VkDeviceSize>((Mip(shape.width, mip) + block.width - 1) /
                                   block.width) *
         ((Mip(shape.height, mip) + block.height - 1) / block.height) *
         block.bytes;
}

void TransferBarrier(const HeadlessVulkanDevice& device,
                     VkCommandBuffer command_buffer,
                     VkPipelineStageFlags source_stage, VkAccessFlags source,
                     VkPipelineStageFlags destination_stage,
                     VkAccessFlags destination) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = source;
  barrier.dstAccessMask = destination;
  device.vk().vkCmdPipelineBarrier(command_buffer, source_stage,
                                   destination_stage, 0, 1, &barrier, 0,
                                   nullptr, 0, nullptr);
}

}  // namespace

Etc2GpuBackend::Etc2GpuBackend(HeadlessVulkanDevice* device) : device_(device) {
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = device_->queue_family();
  device_->vk().vkCreateCommandPool(device_->device(), &pool_info, nullptr,
                                    &pool_);
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  device_->vk().vkCreateFence(device_->device(), &fence_info, nullptr, &fence_);
}

Etc2GpuBackend::~Etc2GpuBackend() {
  const auto& vk = device_->vk();
  vk.vkDeviceWaitIdle(device_->device());
  for (const auto& [image, memory] : image_memory_) {
    vk.vkFreeMemory(device_->device(), memory, nullptr);
  }
  if (fence_ != VK_NULL_HANDLE) {
    vk.vkDestroyFence(device_->device(), fence_, nullptr);
  }
  if (pool_ != VK_NULL_HANDLE) {
    vk.vkDestroyCommandPool(device_->device(), pool_, nullptr);
  }
}

VkDevice Etc2GpuBackend::device() const { return device_->device(); }

VkPhysicalDevice Etc2GpuBackend::physical_device() const {
  return device_->physical_device();
}

VkPhysicalDeviceProperties Etc2GpuBackend::properties() const {
  return device_->properties();
}

VkPhysicalDeviceMemoryProperties Etc2GpuBackend::memory_properties() const {
  return device_->memory();
}

std::vector<VkQueueFamilyProperties> Etc2GpuBackend::queue_families() const {
  return device_->queue_families();
}

PFN_vkGetDeviceProcAddr Etc2GpuBackend::get_device_proc_addr() const {
  return device_->get_device_proc_addr();
}

PFN_vkGetInstanceProcAddr Etc2GpuBackend::get_instance_proc_addr() const {
  return device_->get_instance_proc_addr();
}

VkInstance Etc2GpuBackend::instance() const { return device_->instance(); }

std::string Etc2GpuBackend::Unsupported(const Scenario& scenario) const {
  if (!scenario.fake_only.empty()) {
    return "fake device only: " + scenario.fake_only;
  }
  if (pool_ == VK_NULL_HANDLE || fence_ == VK_NULL_HANDLE) {
    return "could not create a command pool and a fence";
  }
  if (scenario.queue_family != device_->queue_family()) {
    return "needs queue family " + std::to_string(scenario.queue_family) +
           ", the test device has a queue in family " +
           std::to_string(device_->queue_family());
  }
  if (scenario.needs_copy_commands2 && !device_->copy_commands2()) {
    return "the device has no vkCmdCopyBufferToImage2 / vkCmdCopyImage2";
  }
  bool blits = false;
  bool submit2 = false;
  for (const ScenarioStep& step : scenario.steps) {
    blits |= step.kind == ScenarioStep::Kind::kCopyImage;
    submit2 |= step.kind == ScenarioStep::Kind::kSubmit &&
               step.shape == SubmitShape::kQueueSubmit2;
  }
  if (submit2 && !device_->synchronization2()) {
    return "the device has no vkQueueSubmit2 (Vulkan 1.3 synchronization2)";
  }
  for (const ScenarioImage& image : scenario.images) {
    const HostImageShape shape =
        HostShapeFor(image, scenario.upscale, scenario.emulated);
    VkFormatFeatureFlags needed =
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if (blits && shape.width != image.width) {
      needed |= VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    }
    if (!device_->OptimalTilingSupports(shape.format, needed)) {
      return "host format " + std::to_string(static_cast<int>(shape.format)) +
             " lacks transfer or blit support";
    }
  }
  return {};
}

bool Etc2GpuBackend::CreateHostBuffer(VkDeviceSize size,
                                      VkBufferUsageFlags usage,
                                      HostBuffer* out) {
  const auto& vk = device_->vk();
  VkBufferCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.size = size;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vk.vkCreateBuffer(device_->device(), &info, nullptr, &out->buffer) !=
      VK_SUCCESS) {
    return false;
  }
  VkMemoryRequirements requirements{};
  vk.vkGetBufferMemoryRequirements(device_->device(), out->buffer,
                                   &requirements);
  const std::uint32_t type = device_->FindMemoryType(
      requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkMemoryAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex = type;
  void* data = nullptr;
  if (type == UINT32_MAX ||
      vk.vkAllocateMemory(device_->device(), &allocate, nullptr,
                          &out->memory) != VK_SUCCESS ||
      vk.vkBindBufferMemory(device_->device(), out->buffer, out->memory, 0) !=
          VK_SUCCESS ||
      vk.vkMapMemory(device_->device(), out->memory, 0, VK_WHOLE_SIZE, 0,
                     &data) != VK_SUCCESS) {
    DestroyHostBuffer(out);
    return false;
  }
  out->mapped = static_cast<std::uint8_t*>(data);
  return true;
}

void Etc2GpuBackend::DestroyHostBuffer(HostBuffer* buffer) {
  const auto& vk = device_->vk();
  if (buffer->buffer != VK_NULL_HANDLE) {
    vk.vkDestroyBuffer(device_->device(), buffer->buffer, nullptr);
  }
  if (buffer->memory != VK_NULL_HANDLE) {
    vk.vkFreeMemory(device_->device(), buffer->memory, nullptr);
  }
  *buffer = {};
}

template <typename Record>
bool Etc2GpuBackend::RunOnce(Record record) {
  const auto& vk = device_->vk();
  VkCommandBufferAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate.commandPool = pool_;
  allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  if (vk.vkAllocateCommandBuffers(device_->device(), &allocate,
                                  &command_buffer) != VK_SUCCESS) {
    return false;
  }
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer(command_buffer, &begin);
  record(command_buffer);
  vk.vkEndCommandBuffer(command_buffer);
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &command_buffer;
  bool done =
      vk.vkQueueSubmit(device_->queue(), 1, &submit, fence_) == VK_SUCCESS;
  if (done) {
    done = vk.vkWaitForFences(device_->device(), 1, &fence_, VK_TRUE,
                              10ULL * 1000000000ULL) == VK_SUCCESS;
    vk.vkResetFences(device_->device(), 1, &fence_);
  }
  vk.vkFreeCommandBuffers(device_->device(), pool_, 1, &command_buffer);
  return done;
}

VkQueue Etc2GpuBackend::Queue(std::uint32_t family) {
  return family == device_->queue_family() ? device_->queue() : VK_NULL_HANDLE;
}

bool Etc2GpuBackend::CreateSource(VkDeviceSize size, VkDeviceSize bind_offset,
                                  Source* out) {
  const auto& vk = device_->vk();
  VkBufferCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.size = size;
  info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vk.vkCreateBuffer(device_->device(), &info, nullptr, &out->buffer) !=
      VK_SUCCESS) {
    return false;
  }
  VkMemoryRequirements requirements{};
  vk.vkGetBufferMemoryRequirements(device_->device(), out->buffer,
                                   &requirements);
  // Roblox's upload memory: write-combined system memory (type 3 here).
  const std::uint32_t type =
      device_->FindMemoryType(requirements.memoryTypeBits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  out->bind_offset = AlignUp(bind_offset, requirements.alignment);
  out->memory_type = type;
  out->memory_size = out->bind_offset + requirements.size;
  VkMemoryAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate.allocationSize = out->memory_size;
  allocate.memoryTypeIndex = type;
  return type != UINT32_MAX &&
         vk.vkAllocateMemory(device_->device(), &allocate, nullptr,
                             &out->memory) == VK_SUCCESS;
}

bool Etc2GpuBackend::WriteUnmapped(const Source& source,
                                   const std::vector<std::uint8_t>& bytes) {
  const auto& vk = device_->vk();
  void* data = nullptr;
  if (vk.vkMapMemory(device_->device(), source.memory, source.bind_offset,
                     bytes.size(), 0, &data) != VK_SUCCESS) {
    return false;
  }
  std::memcpy(data, bytes.data(), bytes.size());
  vk.vkUnmapMemory(device_->device(), source.memory);
  return true;
}

bool Etc2GpuBackend::PrepareImage(VkImage image, const HostImageShape& shape) {
  const auto& vk = device_->vk();
  VkMemoryRequirements requirements{};
  vk.vkGetImageMemoryRequirements(device_->device(), image, &requirements);
  VkMemoryAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate.allocationSize = requirements.size;
  allocate.memoryTypeIndex = device_->FindMemoryType(
      requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (allocate.memoryTypeIndex == UINT32_MAX ||
      vk.vkAllocateMemory(device_->device(), &allocate, nullptr, &memory) !=
          VK_SUCCESS) {
    return false;
  }
  image_memory_[image] = memory;
  if (vk.vkBindImageMemory(device_->device(), image, memory, 0) != VK_SUCCESS) {
    return false;
  }
  VkDeviceSize largest = 0;
  for (std::uint32_t mip = 0; mip < shape.mip_levels; ++mip) {
    largest =
        std::max(largest, SubresourceBytes(shape, mip) * shape.array_layers);
  }
  HostBuffer fill;
  if (!CreateHostBuffer(largest, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &fill)) {
    return false;
  }
  std::memset(fill.mapped, kImageFill, largest);
  const bool done = RunOnce([&](VkCommandBuffer command_buffer) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, shape.mip_levels,
                                0, shape.array_layers};
    vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                            nullptr, 1, &barrier);
    for (std::uint32_t mip = 0; mip < shape.mip_levels; ++mip) {
      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0,
                                 shape.array_layers};
      region.imageExtent = {Mip(shape.width, mip), Mip(shape.height, mip), 1};
      vk.vkCmdCopyBufferToImage(command_buffer, fill.buffer, image,
                                VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }
    TransferBarrier(*device_, command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
  });
  DestroyHostBuffer(&fill);
  return done;
}

void Etc2GpuBackend::ReleaseImage(VkImage image) {
  const auto found = image_memory_.find(image);
  if (found != image_memory_.end()) {
    device_->vk().vkFreeMemory(device_->device(), found->second, nullptr);
    image_memory_.erase(found);
  }
}

VkCommandPool Etc2GpuBackend::CreateCommandPool(
    std::uint32_t queue_family, VkCommandPoolCreateFlags flags) {
  VkCommandPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.flags = flags;
  info.queueFamilyIndex = queue_family;
  VkCommandPool pool = VK_NULL_HANDLE;
  if (device_->vk().vkCreateCommandPool(device_->device(), &info, nullptr,
                                        &pool) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  return pool;
}

bool Etc2GpuBackend::ResetCommandPool(VkCommandPool pool) {
  return device_->vk().vkResetCommandPool(device_->device(), pool, 0) ==
         VK_SUCCESS;
}

void Etc2GpuBackend::DestroyCommandPool(VkCommandPool pool) {
  device_->vk().vkDestroyCommandPool(device_->device(), pool, nullptr);
}

VkCommandBuffer Etc2GpuBackend::AllocateCommandBuffer(VkCommandPool pool,
                                                      bool secondary) {
  VkCommandBufferAllocateInfo allocate{};
  allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate.commandPool = pool;
  allocate.level = secondary ? VK_COMMAND_BUFFER_LEVEL_SECONDARY
                             : VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate.commandBufferCount = 1;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  if (device_->vk().vkAllocateCommandBuffers(device_->device(), &allocate,
                                             &command_buffer) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  return command_buffer;
}

bool Etc2GpuBackend::Begin(VkCommandBuffer command_buffer, bool secondary,
                           VkCommandBufferUsageFlags flags) {
  VkCommandBufferInheritanceInfo inheritance{};
  inheritance.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = flags;
  begin.pInheritanceInfo = secondary ? &inheritance : nullptr;
  return device_->vk().vkBeginCommandBuffer(command_buffer, &begin) ==
         VK_SUCCESS;
}

bool Etc2GpuBackend::End(VkCommandBuffer command_buffer) {
  return device_->vk().vkEndCommandBuffer(command_buffer) == VK_SUCCESS;
}

bool Etc2GpuBackend::Reset(VkCommandBuffer command_buffer) {
  return device_->vk().vkResetCommandBuffer(command_buffer, 0) == VK_SUCCESS;
}

void Etc2GpuBackend::FreeCommandBuffer(VkCommandPool pool,
                                       VkCommandBuffer command_buffer) {
  device_->vk().vkFreeCommandBuffers(device_->device(), pool, 1,
                                     &command_buffer);
}

void Etc2GpuBackend::BeforeTransfer(VkCommandBuffer command_buffer) {
  TransferBarrier(*device_, command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}

VkResult Etc2GpuBackend::QueueSubmit(VkQueue queue, std::uint32_t count,
                                     const VkSubmitInfo* submits,
                                     VkFence fence) {
  return device_->vk().vkQueueSubmit(queue, count, submits, fence);
}

VkResult Etc2GpuBackend::QueueSubmit2(VkQueue queue, std::uint32_t count,
                                      const VkSubmitInfo2* submits,
                                      VkFence fence) {
  if (device_->vk().vkQueueSubmit2 == nullptr) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }
  return device_->vk().vkQueueSubmit2(queue, count, submits, fence);
}

bool Etc2GpuBackend::WaitIdle(VkQueue queue) {
  return device_->vk().vkQueueWaitIdle(queue) == VK_SUCCESS;
}

bool Etc2GpuBackend::ReadImage(VkImage image, const HostImageShape& shape,
                               std::vector<std::vector<std::uint8_t>>* texels) {
  std::vector<VkDeviceSize> offsets;
  VkDeviceSize total = 0;
  for (std::uint32_t mip = 0; mip < shape.mip_levels; ++mip) {
    for (std::uint32_t layer = 0; layer < shape.array_layers; ++layer) {
      offsets.push_back(total);
      total += AlignUp(SubresourceBytes(shape, mip), 16);
    }
  }
  HostBuffer readback;
  if (!CreateHostBuffer(total, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback)) {
    return false;
  }
  const auto& vk = device_->vk();
  const bool done = RunOnce([&](VkCommandBuffer command_buffer) {
    TransferBarrier(*device_, command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT);
    std::vector<VkBufferImageCopy> regions;
    for (std::uint32_t mip = 0; mip < shape.mip_levels; ++mip) {
      for (std::uint32_t layer = 0; layer < shape.array_layers; ++layer) {
        VkBufferImageCopy region{};
        region.bufferOffset = offsets[mip * shape.array_layers + layer];
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
        region.imageExtent = {Mip(shape.width, mip), Mip(shape.height, mip), 1};
        regions.push_back(region);
      }
    }
    vk.vkCmdCopyImageToBuffer(
        command_buffer, image, VK_IMAGE_LAYOUT_GENERAL, readback.buffer,
        static_cast<std::uint32_t>(regions.size()), regions.data());
    TransferBarrier(*device_, command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                    VK_ACCESS_HOST_READ_BIT);
  });
  if (done) {
    texels->clear();
    for (std::uint32_t mip = 0; mip < shape.mip_levels; ++mip) {
      for (std::uint32_t layer = 0; layer < shape.array_layers; ++layer) {
        const std::uint8_t* data =
            readback.mapped + offsets[mip * shape.array_layers + layer];
        texels->emplace_back(data, data + SubresourceBytes(shape, mip));
      }
    }
  }
  DestroyHostBuffer(&readback);
  return done;
}

void Etc2GpuBackend::DestroyImageDirect(VkImage image) {
  device_->vk().vkDestroyImage(device_->device(), image, nullptr);
}

void Etc2GpuBackend::DestroyBufferDirect(VkBuffer buffer) {
  device_->vk().vkDestroyBuffer(device_->device(), buffer, nullptr);
}

void Etc2GpuBackend::UnmapMemoryDirect(VkDeviceMemory memory) {
  device_->vk().vkUnmapMemory(device_->device(), memory);
}

void Etc2GpuBackend::FreeMemoryDirect(VkDeviceMemory memory) {
  device_->vk().vkFreeMemory(device_->device(), memory, nullptr);
}

}  // namespace mocktail::graphics::testing
