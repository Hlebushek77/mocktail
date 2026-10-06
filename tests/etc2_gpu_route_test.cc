#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "mocktail/graphics/vulkan_etc2_gpu.h"
#include "support/etc2_emulation_scenarios.h"
#include "support/etc2_gpu_backend.h"
#include "support/vulkan_test_device.h"

// The GPU route of the production emulation on a real device, in the
// submit shapes the scenario runner does not make (DESIGN §9.3, CR-1,
// CR-7, CR-32): several batches in one call with and without an app fence,
// vkQueueSubmit2, a semaphore-guarded source another queue writes, two
// queues submitting from two threads, a compute-only family, a transfer-only
// family, a forced rescue with no slot, a failed selftest, and vkDestroyDevice
// with work in flight. Every image is read back and compared with the CPU
// decoder. Label "gpu"; skipped without a device and while a live game runs.

namespace mocktail::graphics::testing {
namespace {

using Clock = std::chrono::steady_clock;

struct RigImage {
  VkImage image = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  EtcFormat etc = EtcFormat::kEtc2Rgb8;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t mips = 1;
  std::uint32_t layers = 1;
  HostImageShape shape;
};

struct RigSource {
  EmulationBackend::Source source;
  std::uint8_t* mapped = nullptr;
  std::vector<std::uint8_t> bytes;  // what the GPU should read
};

// One upload of a whole mip, and the bytes it reads.
struct RigUpload {
  const RigImage* image = nullptr;
  VkBufferImageCopy region{};
};

// Plays Roblox and the adapter around one VulkanEtc2Emulation on a real
// device: images, upload buffers and command pools go through the
// emulation as the adapter routes them, submits through PrepareSubmit, the
// plan's pre-submits, the host submit and FinishSubmit.
class Rig {
 public:
  Rig(HeadlessVulkanDevice* device, Etc2EmulationOptions options)
      : device_(device), backend_(device), emulation_(options) {
    std::vector<VkDeviceQueueCreateInfo> queues;
    const float priority = 0.0F;
    for (std::uint32_t family = 0; family < device->queue_families().size();
         ++family) {
      std::uint32_t count = 0;
      while (device->Queue(family, count) != VK_NULL_HANDLE) {
        ++count;
      }
      if (count != 0) {
        VkDeviceQueueCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        info.queueFamilyIndex = family;
        info.queueCount = count;
        info.pQueuePriorities = &priority;
        queues.push_back(info);
      }
    }
    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount =
        static_cast<std::uint32_t>(queues.size());
    create_info.pQueueCreateInfos = queues.data();
    emulation_.RegisterDevice(
        device->device(), device->physical_device(), true, device->memory(),
        device->properties(), device->queue_families(), create_info,
        device->get_device_proc_addr(), device->get_instance_proc_addr(),
        device->instance());
    for (const VkDeviceQueueCreateInfo& info : queues) {
      for (std::uint32_t index = 0; index < info.queueCount; ++index) {
        emulation_.RegisterQueue(device->device(),
                                 device->Queue(info.queueFamilyIndex, index),
                                 info.queueFamilyIndex, index);
      }
    }
  }

  ~Rig() { Teardown(); }

  VulkanEtc2Emulation& emulation() { return emulation_; }
  Etc2GpuBackend& backend() { return backend_; }
  const HeadlessVulkanDevice::Dispatch& vk() const { return device_->vk(); }
  Etc2GpuDevice* gpu() const { return Etc2GpuDevice::Find(device_->device()); }

  // The adapter's vkDestroyDevice hook, before Roblox freed the rest.
  void DestroyEmulationDevice() {
    emulation_.DestroyDevice(device_->device());
    device_destroyed_ = true;
  }

  // Releases and frees everything, then destroys the emulation's device.
  void Teardown() {
    if (torn_down_) {
      return;
    }
    torn_down_ = true;
    vk().vkDeviceWaitIdle(device_->device());
    for (const auto& [pool, buffers] : pools_) {
      for (VkCommandBuffer command_buffer : buffers) {
        emulation_.ReleaseCommandBuffer(command_buffer);
      }
      backend_.DestroyCommandPool(pool);
      emulation_.ForgetCommandPool(device_->device(), pool);
    }
    for (const std::unique_ptr<RigImage>& image : images_) {
      if (device_destroyed_) {
        backend_.DestroyImageDirect(image->image);
      } else {
        emulation_.DestroyImage(device_->device(), image->image, nullptr);
      }
      backend_.ReleaseImage(image->image);
    }
    for (const std::unique_ptr<RigSource>& source : sources_) {
      if (device_destroyed_) {
        backend_.UnmapMemoryDirect(source->source.memory);
        backend_.DestroyBufferDirect(source->source.buffer);
        backend_.FreeMemoryDirect(source->source.memory);
        continue;
      }
      emulation_.UnmapMemory(device_->device(), source->source.memory);
      emulation_.DestroyBuffer(device_->device(), source->source.buffer,
                               nullptr);
      emulation_.FreeMemory(device_->device(), source->source.memory, nullptr);
    }
    for (VkSemaphore semaphore : semaphores_) {
      vk().vkDestroySemaphore(device_->device(), semaphore, nullptr);
    }
    for (VkFence fence : fences_) {
      vk().vkDestroyFence(device_->device(), fence, nullptr);
    }
    emulation_.DestroyDevice(device_->device());
  }

  // Waits for the pipelines, then runs the selftest with empty submits on
  // `queue`. True once the GPU route is ready.
  bool WarmUp(VkQueue queue) {
    Etc2GpuDevice* runtime = gpu();
    if (runtime == nullptr) {
      return false;
    }
    // The first emulated image would start the build; no need to wait.
    runtime->StartBuild();
    if (!runtime->WaitForBuild(std::chrono::seconds(60))) {
      return false;
    }
    for (int round = 0;
         round < 3 && runtime->state() != Etc2GpuDevice::State::kReady &&
         runtime->state() != Etc2GpuDevice::State::kFailed;
         ++round) {
      Submit(queue, {}, VK_NULL_HANDLE);
      vk().vkQueueWaitIdle(queue);
    }
    return runtime->state() == Etc2GpuDevice::State::kReady;
  }

  RigImage* Image(VkFormat format, std::uint32_t width, std::uint32_t height,
                  std::uint32_t mips, std::uint32_t layers = 1,
                  std::vector<std::uint32_t> concurrent_families = {}) {
    auto image = std::make_unique<RigImage>();
    image->format = format;
    Etc2EmulatedFormat emulated;
    EXPECT_TRUE(LookupEmulatedEtc2Format(format, &emulated));
    image->etc = emulated.etc_format;
    image->width = width;
    image->height = height;
    image->mips = mips;
    image->layers = layers;
    ScenarioImage spec{format, width, height, mips, layers};
    image->shape =
        HostShapeFor(spec, emulation_.options().upscale, /*emulated=*/true);
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = mips;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.sharingMode = concurrent_families.empty() ? VK_SHARING_MODE_EXCLUSIVE
                                                   : VK_SHARING_MODE_CONCURRENT;
    info.queueFamilyIndexCount =
        static_cast<std::uint32_t>(concurrent_families.size());
    info.pQueueFamilyIndices = concurrent_families.data();
    EXPECT_EQ(emulation_.CreateImage(device_->device(), &info, nullptr,
                                     &image->image),
              VK_SUCCESS);
    EXPECT_TRUE(backend_.PrepareImage(image->image, image->shape));
    images_.push_back(std::move(image));
    return images_.back().get();
  }

  // Roblox's upload buffer: host-visible, bound and kept mapped through the
  // emulation, filled with `seed`'s bytes. A copy can write it too.
  RigSource* Source(VkDeviceSize size, std::uint64_t seed) {
    auto source = std::make_unique<RigSource>();
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    EXPECT_EQ(vk().vkCreateBuffer(device_->device(), &info, nullptr,
                                  &source->source.buffer),
              VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vk().vkGetBufferMemoryRequirements(device_->device(), source->source.buffer,
                                       &requirements);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = device_->FindMemoryType(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    EXPECT_EQ(vk().vkAllocateMemory(device_->device(), &allocate, nullptr,
                                    &source->source.memory),
              VK_SUCCESS);
    source->source.memory_type = allocate.memoryTypeIndex;
    source->source.memory_size = requirements.size;
    source->bytes.resize(size);
    emulation_.NoteAllocateMemory(device_->device(), source->source.memory,
                                  source->source.memory_type,
                                  source->source.memory_size);
    EXPECT_EQ(emulation_.BindBufferMemory(device_->device(),
                                          source->source.buffer,
                                          source->source.memory, 0),
              VK_SUCCESS);
    void* data = nullptr;
    EXPECT_EQ(emulation_.MapMemory(device_->device(), source->source.memory, 0,
                                   VK_WHOLE_SIZE, 0, &data),
              VK_SUCCESS);
    source->mapped = static_cast<std::uint8_t*>(data);
    Rewrite(source.get(), seed);
    sources_.push_back(std::move(source));
    return sources_.back().get();
  }

  void Rewrite(RigSource* source, std::uint64_t seed) {
    source->bytes = SeededBytes(seed, source->bytes.size());
    std::memcpy(source->mapped, source->bytes.data(), source->bytes.size());
  }

  VkCommandPool Pool(std::uint32_t family) {
    const VkCommandPool pool = backend_.CreateCommandPool(
        family, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    EXPECT_NE(pool, VK_NULL_HANDLE);
    emulation_.RegisterCommandPool(
        device_->device(), pool, family,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    pools_.push_back({pool, {}});
    return pool;
  }

  VkCommandBuffer Begin(VkCommandPool pool,
                        VkCommandBufferUsageFlags flags =
                            VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT) {
    const VkCommandBuffer command_buffer =
        backend_.AllocateCommandBuffer(pool, false);
    for (auto& [owner, buffers] : pools_) {
      if (owner == pool) {
        buffers.push_back(command_buffer);
      }
    }
    Restart(command_buffer, pool, flags);
    return command_buffer;
  }

  // vkBeginCommandBuffer on a command buffer Roblox used before.
  void Restart(VkCommandBuffer command_buffer, VkCommandPool pool,
               VkCommandBufferUsageFlags flags) {
    emulation_.BeginCommandBuffer(command_buffer, pool, flags);
    EXPECT_TRUE(backend_.Begin(command_buffer, false, flags));
  }

  void End(VkCommandBuffer command_buffer) {
    EXPECT_TRUE(backend_.End(command_buffer));
  }

  // Every mip of `image` from `source`, tightly packed from `offset` on.
  // Returns the uploads, with the offset after them in `*end`.
  std::vector<RigUpload> UploadChain(VkCommandBuffer command_buffer,
                                     const RigImage& image,
                                     const RigSource& source,
                                     VkDeviceSize offset,
                                     VkDeviceSize* end = nullptr) {
    std::vector<RigUpload> uploads;
    std::vector<VkBufferImageCopy> regions;
    for (std::uint32_t mip = 0; mip < image.mips; ++mip) {
      RigUpload upload;
      upload.image = &image;
      VkBufferImageCopy& region = upload.region;
      region.bufferOffset = offset;
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0,
                                 image.layers};
      region.imageExtent = {std::max(1u, image.width >> mip),
                            std::max(1u, image.height >> mip), 1};
      VkDeviceSize compressed = 0;
      VkDeviceSize decoded = 0;
      Etc2UploadByteCounts(image.etc, region.imageExtent, image.layers,
                           &compressed, &decoded);
      offset += compressed;
      regions.push_back(region);
      uploads.push_back(upload);
    }
    backend_.BeforeTransfer(command_buffer);
    emulation_.CmdCopyBufferToImage(
        device_->device(), command_buffer, source.source.buffer, image.image,
        VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(regions.size()),
        regions.data());
    if (end != nullptr) {
      *end = offset;
    }
    return uploads;
  }

  VkResult Submit(VkQueue queue,
                  const std::vector<std::vector<VkCommandBuffer>>& batches,
                  VkFence fence,
                  const std::vector<VkSubmitInfo>* overrides = nullptr) {
    std::vector<VkSubmitInfo> infos;
    if (overrides != nullptr) {
      infos = *overrides;
    } else {
      for (const std::vector<VkCommandBuffer>& batch : batches) {
        VkSubmitInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        info.commandBufferCount = static_cast<std::uint32_t>(batch.size());
        info.pCommandBuffers = batch.empty() ? nullptr : batch.data();
        infos.push_back(info);
      }
    }
    const auto count = static_cast<std::uint32_t>(infos.size());
    Etc2SubmitPlan plan;
    emulation_.PrepareSubmit(queue, count, infos.empty() ? nullptr : infos.data(),
                             fence, &plan);
    for (const Etc2PreSubmit& pre : plan.pre_submits()) {
      EXPECT_EQ(vk().vkQueueSubmit(queue, 1, &pre.info, pre.fence),
                VK_SUCCESS);
    }
    const VkResult result = vk().vkQueueSubmit(
        queue, plan.count(count),
        plan.submits(infos.empty() ? nullptr : infos.data()),
        plan.fence(fence));
    emulation_.FinishSubmit(&plan, result, [&](VkFence trailing) {
      return vk().vkQueueSubmit(queue, 0, nullptr, trailing);
    });
    return result;
  }

  VkResult Submit2(VkQueue queue, const std::vector<VkCommandBuffer>& batch,
                   VkFence fence) {
    std::vector<VkCommandBufferSubmitInfo> buffers;
    for (VkCommandBuffer command_buffer : batch) {
      VkCommandBufferSubmitInfo info{};
      info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
      info.commandBuffer = command_buffer;
      buffers.push_back(info);
    }
    VkSubmitInfo2 info{};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    info.commandBufferInfoCount = static_cast<std::uint32_t>(buffers.size());
    info.pCommandBufferInfos = buffers.data();
    Etc2SubmitPlan plan;
    emulation_.PrepareSubmit2(queue, 1, &info, fence, &plan);
    for (const Etc2PreSubmit& pre : plan.pre_submits()) {
      EXPECT_EQ(vk().vkQueueSubmit(queue, 1, &pre.info, pre.fence),
                VK_SUCCESS);
    }
    const VkResult result = vk().vkQueueSubmit2(
        queue, plan.count(1), plan.submits2(&info), plan.fence(fence));
    emulation_.FinishSubmit(&plan, result, [&](VkFence trailing) {
      return vk().vkQueueSubmit(queue, 0, nullptr, trailing);
    });
    return result;
  }

  VkFence Fence() {
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    EXPECT_EQ(vk().vkCreateFence(device_->device(), &info, nullptr, &fence),
              VK_SUCCESS);
    fences_.push_back(fence);
    return fence;
  }

  VkSemaphore Semaphore() {
    VkSemaphoreCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    EXPECT_EQ(vk().vkCreateSemaphore(device_->device(), &info, nullptr,
                                     &semaphore),
              VK_SUCCESS);
    semaphores_.push_back(semaphore);
    return semaphore;
  }

  // Every upload's subresource must hold what the CPU decoder makes of
  // `bytes` (the source as the GPU read it).
  void Check(const std::vector<RigUpload>& uploads,
             const std::vector<std::uint8_t>& bytes, const std::string& what) {
    std::vector<std::vector<std::uint8_t>> texels;
    const RigImage* read = nullptr;
    for (const RigUpload& upload : uploads) {
      const RigImage& image = *upload.image;
      if (read != &image) {
        ASSERT_TRUE(backend_.ReadImage(image.image, image.shape, &texels));
        read = &image;
      }
      const VkBufferImageCopy& region = upload.region;
      const std::uint32_t mip = region.imageSubresource.mipLevel;
      const std::uint32_t width = region.imageExtent.width;
      const std::uint32_t height = region.imageExtent.height;
      VkDeviceSize compressed = 0;
      VkDeviceSize decoded = 0;
      Etc2UploadByteCounts(image.etc, region.imageExtent, 1, &compressed,
                           &decoded);
      for (std::uint32_t layer = 0; layer < image.layers; ++layer) {
        std::vector<std::uint8_t> expected(decoded);
        ASSERT_TRUE(DecodeEtcImage(
            image.etc, bytes.data() + region.bufferOffset + layer * compressed,
            compressed, width, height, expected.data(), expected.size()));
        EXPECT_EQ(texels[mip * image.layers + layer], expected)
            << what << ": " << image.width << "x" << image.height << " format "
            << image.format << " mip " << mip << " layer " << layer;
      }
    }
  }

 private:
  HeadlessVulkanDevice* device_;
  Etc2GpuBackend backend_;
  VulkanEtc2Emulation emulation_;
  std::vector<std::unique_ptr<RigImage>> images_;
  std::vector<std::unique_ptr<RigSource>> sources_;
  std::vector<std::pair<VkCommandPool, std::vector<VkCommandBuffer>>> pools_;
  std::vector<VkSemaphore> semaphores_;
  std::vector<VkFence> fences_;
  bool torn_down_ = false;
  bool device_destroyed_ = false;
};

Etc2EmulationOptions GpuOptions() {
  Etc2EmulationOptions options;
  options.gpu = true;
  options.upscale = 1;
  return options;
}

class Etc2GpuRouteTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    HeadlessVulkanDevice::Options options;
    options.queue_count = 2;
    options.compute_only_family = true;
    options.transfer_only_family = true;
    device_ = HeadlessVulkanDevice::Create(&skip_reason_, options).release();
    if (device_ != nullptr) {
      std::printf("GPU: %s\n", device_->Describe().c_str());
    }
  }
  static void TearDownTestSuite() {
    delete device_;
    device_ = nullptr;
  }
  void SetUp() override {
    if (device_ == nullptr) {
      GTEST_SKIP() << skip_reason_;
    }
    device_->TakeValidationMessages();
  }
  void TearDown() override {
    Etc2GpuTestHooks& hooks = Etc2GpuTesting();
    hooks.fail_slot_acquire = false;
    hooks.fail_rescue_slot = false;
    hooks.corrupt_selftest = false;
    {
      std::lock_guard<std::mutex> lock(hooks.mutex);
      hooks.rescue_decode = nullptr;
    }
    if (device_ != nullptr) {
      for (const std::string& message : device_->TakeValidationMessages()) {
        ADD_FAILURE() << "validation: " << message;
      }
    }
  }

  VkQueue queue(std::uint32_t index = 0) const {
    return device_->Queue(device_->queue_family(), index);
  }

  static HeadlessVulkanDevice* device_;
  static std::string skip_reason_;
};

HeadlessVulkanDevice* Etc2GpuRouteTest::device_ = nullptr;
std::string Etc2GpuRouteTest::skip_reason_;

// [CR-1] One call, three batches, two of them with uploads: both get a pass
// and one slot fence covers the call, with and without Roblox's fence. Then
// 40 more calls in a row never wait for a slot: every slot's fence signals.
TEST_F(Etc2GpuRouteTest, OneSlotServesEveryBatchOfACall) {
  Rig rig(device_, GpuOptions());
  ASSERT_TRUE(rig.WarmUp(queue()));
  const VkCommandPool pool = rig.Pool(device_->queue_family());
  RigImage* first = rig.Image(VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, 37, 37, 6);
  RigImage* second = rig.Image(VK_FORMAT_EAC_R11G11_SNORM_BLOCK, 60, 33, 4, 2);
  RigSource* source = rig.Source(64 * 1024, 7);
  for (const bool app_fence : {false, true}) {
    SCOPED_TRACE(app_fence ? "with an app fence" : "without an app fence");
    rig.Rewrite(source, app_fence ? 8 : 9);
    const VkCommandBuffer a = rig.Begin(pool);
    VkDeviceSize end = 0;
    std::vector<RigUpload> uploads = rig.UploadChain(a, *first, *source, 0, &end);
    rig.End(a);
    const VkCommandBuffer idle = rig.Begin(pool);
    rig.End(idle);
    const VkCommandBuffer c = rig.Begin(pool);
    const std::vector<RigUpload> more =
        rig.UploadChain(c, *second, *source, end);
    uploads.insert(uploads.end(), more.begin(), more.end());
    rig.End(c);
    const VkFence fence = app_fence ? rig.Fence() : VK_NULL_HANDLE;
    const std::uint64_t passes = rig.gpu()->stats().passes;
    ASSERT_EQ(rig.Submit(queue(), {{a}, {idle}, {c}}, fence), VK_SUCCESS);
    if (fence != VK_NULL_HANDLE) {
      ASSERT_EQ(rig.vk().vkWaitForFences(device_->device(), 1, &fence, VK_TRUE,
                                         10ULL * 1000000000ULL),
                VK_SUCCESS);
    }
    rig.vk().vkQueueWaitIdle(queue());
    EXPECT_EQ(rig.gpu()->stats().passes - passes, 2U);
    rig.Check(uploads, source->bytes, "three batches");
  }
  const std::uint64_t slots = rig.gpu()->stats().slots_created;
  RigImage* small = rig.Image(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 16, 16, 1);
  for (int round = 0; round < 40; ++round) {
    const VkCommandBuffer command_buffer = rig.Begin(pool);
    rig.UploadChain(command_buffer, *small, *source, 0);
    rig.End(command_buffer);
    const VkFence fence = round % 2 == 0 ? VK_NULL_HANDLE : rig.Fence();
    ASSERT_EQ(rig.Submit(queue(), {{command_buffer}}, fence), VK_SUCCESS);
    rig.vk().vkQueueWaitIdle(queue());
  }
  const Etc2GpuDevice::Stats stats = rig.gpu()->stats();
  EXPECT_EQ(stats.slot_waits, 0U);
  EXPECT_LE(stats.slots_created, slots + 1) << "signalled slots are reused";
  EXPECT_EQ(stats.dropped_uploads, 0U);
}

// vkQueueSubmit2 with three command buffers in one batch: one pass in front.
TEST_F(Etc2GpuRouteTest, Submit2PutsOnePassInFrontOfTheBatch) {
  if (!device_->synchronization2()) {
    GTEST_SKIP() << "no vkQueueSubmit2 (Vulkan 1.3 synchronization2)";
  }
  Rig rig(device_, GpuOptions());
  ASSERT_TRUE(rig.WarmUp(queue()));
  const VkCommandPool pool = rig.Pool(device_->queue_family());
  RigSource* source = rig.Source(64 * 1024, 11);
  std::vector<VkCommandBuffer> batch;
  std::vector<RigUpload> uploads;
  VkDeviceSize offset = 0;
  const VkFormat formats[3] = {VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
                               VK_FORMAT_EAC_R11_SNORM_BLOCK,
                               VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK};
  for (const VkFormat format : formats) {
    RigImage* image = rig.Image(format, 45, 29, 3);
    const VkCommandBuffer command_buffer = rig.Begin(pool);
    const std::vector<RigUpload> chain =
        rig.UploadChain(command_buffer, *image, *source, offset, &offset);
    uploads.insert(uploads.end(), chain.begin(), chain.end());
    rig.End(command_buffer);
    batch.push_back(command_buffer);
  }
  const std::uint64_t passes = rig.gpu()->stats().passes;
  ASSERT_EQ(rig.Submit2(queue(), batch, VK_NULL_HANDLE), VK_SUCCESS);
  rig.vk().vkQueueWaitIdle(queue());
  EXPECT_EQ(rig.gpu()->stats().passes - passes, 1U);
  rig.Check(uploads, source->bytes, "vkQueueSubmit2");
}

// A source another queue writes and signals: the pass's copy-in runs after
// Roblox's semaphore wait (TRANSFER stage), so it reads the new bytes. (The
// CPU route reads the source at vkQueueSubmit and cannot.)
TEST_F(Etc2GpuRouteTest, ReadsASourceASemaphoreGuards) {
  if (queue(1) == VK_NULL_HANDLE) {
    GTEST_SKIP() << "the device has one queue in its graphics family";
  }
  Rig rig(device_, GpuOptions());
  ASSERT_TRUE(rig.WarmUp(queue()));
  const VkCommandPool pool = rig.Pool(device_->queue_family());
  RigImage* image = rig.Image(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 64, 64, 7);
  RigSource* source = rig.Source(32 * 1024, 21);   // old bytes
  RigSource* staging = rig.Source(32 * 1024, 22);  // what queue 1 copies in
  const VkCommandBuffer writer = rig.Begin(pool);
  const VkBufferCopy all{0, 0, 32 * 1024};
  rig.vk().vkCmdCopyBuffer(writer, staging->source.buffer,
                           source->source.buffer, 1, &all);
  rig.End(writer);
  const VkSemaphore written = rig.Semaphore();
  VkSubmitInfo signal{};
  signal.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  signal.commandBufferCount = 1;
  signal.pCommandBuffers = &writer;
  signal.signalSemaphoreCount = 1;
  signal.pSignalSemaphores = &written;
  const std::vector<VkSubmitInfo> signal_batch = {signal};
  const VkCommandBuffer reader = rig.Begin(pool);
  const std::vector<RigUpload> uploads =
      rig.UploadChain(reader, *image, *source, 0);
  rig.End(reader);
  const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkSubmitInfo wait{};
  wait.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  wait.waitSemaphoreCount = 1;
  wait.pWaitSemaphores = &written;
  wait.pWaitDstStageMask = &stage;
  wait.commandBufferCount = 1;
  wait.pCommandBuffers = &reader;
  const std::vector<VkSubmitInfo> wait_batch = {wait};
  ASSERT_EQ(rig.Submit(queue(1), {}, VK_NULL_HANDLE, &signal_batch),
            VK_SUCCESS);
  ASSERT_EQ(rig.Submit(queue(0), {}, VK_NULL_HANDLE, &wait_batch), VK_SUCCESS);
  rig.vk().vkDeviceWaitIdle(device_->device());
  rig.Check(uploads, staging->bytes, "semaphore-guarded source");
}

// Two queues of one family, each fed by its own thread: their passes share
// the family's output arena and must not disturb each other.
TEST_F(Etc2GpuRouteTest, TwoQueuesUploadConcurrently) {
  if (queue(1) == VK_NULL_HANDLE) {
    GTEST_SKIP() << "the device has one queue in its graphics family";
  }
  Rig rig(device_, GpuOptions());
  ASSERT_TRUE(rig.WarmUp(queue()));
  struct Lane {
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    RigSource* source = nullptr;
    std::vector<RigImage*> images;
    std::vector<std::vector<RigUpload>> uploads;
  };
  Lane lanes[2];
  for (std::uint32_t index = 0; index < 2; ++index) {
    Lane& lane = lanes[index];
    lane.queue = queue(index);
    lane.pool = rig.Pool(device_->queue_family());
    lane.source = rig.Source(256 * 1024, 31 + index);
    for (int image = 0; image < 12; ++image) {
      lane.images.push_back(
          rig.Image(image % 2 == 0 ? VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK
                                   : VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
                    96 + 8 * image, 64, 5));
    }
    lane.uploads.resize(lane.images.size());
  }
  // Command buffers are allocated up front: a pool is used by one thread.
  std::vector<VkCommandBuffer> buffers[2];
  for (std::uint32_t index = 0; index < 2; ++index) {
    for (std::size_t image = 0; image < lanes[index].images.size(); ++image) {
      buffers[index].push_back(rig.Begin(lanes[index].pool));
    }
  }
  const auto run = [&](std::uint32_t index) {
    Lane& lane = lanes[index];
    VkDeviceSize offset = 0;
    for (std::size_t image = 0; image < lane.images.size(); ++image) {
      const VkCommandBuffer command_buffer = buffers[index][image];
      lane.uploads[image] = rig.UploadChain(
          command_buffer, *lane.images[image], *lane.source, offset, &offset);
      rig.End(command_buffer);
      EXPECT_EQ(rig.Submit(lane.queue, {{command_buffer}}, VK_NULL_HANDLE),
                VK_SUCCESS);
    }
  };
  std::thread second(run, 1);
  run(0);
  second.join();
  rig.vk().vkDeviceWaitIdle(device_->device());
  for (const Lane& lane : lanes) {
    for (const std::vector<RigUpload>& uploads : lane.uploads) {
      rig.Check(uploads, lane.source->bytes, "two queues");
    }
  }
  EXPECT_EQ(rig.gpu()->stats().dropped_uploads, 0U);
}

// A compute-only family (family 2 on NVIDIA) gets its own arena and slots.
TEST_F(Etc2GpuRouteTest, DecodesOnAComputeOnlyFamily) {
  const std::uint32_t family = device_->compute_only_family();
  if (family == UINT32_MAX) {
    GTEST_SKIP() << "the device has no compute-only queue family";
  }
  Rig rig(device_, GpuOptions());
  ASSERT_TRUE(rig.WarmUp(device_->Queue(family)));
  const VkCommandPool pool = rig.Pool(family);
  // Read back on the graphics family: share the image between both.
  RigImage* image = rig.Image(VK_FORMAT_EAC_R11_UNORM_BLOCK, 130, 66, 4, 2,
                              {device_->queue_family(), family});
  RigSource* source = rig.Source(64 * 1024, 41);
  const VkCommandBuffer command_buffer = rig.Begin(pool);
  const std::vector<RigUpload> uploads =
      rig.UploadChain(command_buffer, *image, *source, 0);
  rig.End(command_buffer);
  const std::uint64_t passes = rig.gpu()->stats().passes;
  ASSERT_EQ(rig.Submit(device_->Queue(family), {{command_buffer}},
                       VK_NULL_HANDLE),
            VK_SUCCESS);
  rig.vk().vkDeviceWaitIdle(device_->device());
  EXPECT_EQ(rig.gpu()->stats().passes - passes, 1U);
  rig.Check(uploads, source->bytes, "compute-only family");
}

// A transfer-only family has no compute: the CPU route serves it.
TEST_F(Etc2GpuRouteTest, ATransferOnlyFamilyTakesTheCpuRoute) {
  const std::uint32_t family = device_->transfer_only_family();
  if (family == UINT32_MAX) {
    GTEST_SKIP() << "the device has no transfer-only queue family";
  }
  Rig rig(device_, GpuOptions());
  ASSERT_TRUE(rig.WarmUp(queue()));
  const VkCommandPool pool = rig.Pool(family);
  RigImage* image = rig.Image(VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, 64, 32, 3, 1,
                              {device_->queue_family(), family});
  RigSource* source = rig.Source(32 * 1024, 51);
  const VkCommandBuffer command_buffer = rig.Begin(pool);
  const std::vector<RigUpload> uploads =
      rig.UploadChain(command_buffer, *image, *source, 0);
  rig.End(command_buffer);
  const std::uint64_t passes = rig.gpu()->stats().passes;
  ASSERT_EQ(rig.Submit(device_->Queue(family), {{command_buffer}},
                       VK_NULL_HANDLE),
            VK_SUCCESS);
  rig.vk().vkDeviceWaitIdle(device_->device());
  EXPECT_EQ(rig.gpu()->stats().passes, passes);
  rig.Check(uploads, source->bytes, "transfer-only family");
}

// [CR-7] No slot comes free: the reserved rescue slot copies what the CPU
// decoded into host staging (the arena's, or a one-off buffer with
// MOCKTAIL_ETC2_FAST_PATH=0) into the output arena, and Roblox's rewritten
// copies read it after the pass's TRANSFER->TRANSFER barrier. Whole mip
// chains, and padded rows and layers from a source Roblox does not keep
// mapped. With the rescue slot taken as well, the uploads are dropped:
// counted, and the call still goes through.
TEST_F(Etc2GpuRouteTest, ARescueWithNoSlotCopiesCpuTexelsIntoTheArena) {
  for (const bool fast_path : {true, false}) {
    SCOPED_TRACE(fast_path ? "host staging arena" : "one-off staging");
    Etc2EmulationOptions options = GpuOptions();
    options.fast_path = fast_path;
    Rig rig(device_, options);
    ASSERT_TRUE(rig.WarmUp(queue()));
    const VkCommandPool pool = rig.Pool(device_->queue_family());
    RigImage* chain = rig.Image(VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, 50, 30, 4);
    RigImage* layered =
        rig.Image(VK_FORMAT_EAC_R11G11_SNORM_BLOCK, 27, 14, 1, 2);
    RigSource* source = rig.Source(32 * 1024, 61);
    Etc2GpuTesting().fail_slot_acquire = true;
    const VkCommandBuffer command_buffer = rig.Begin(pool);
    VkDeviceSize end = 0;
    std::vector<RigUpload> uploads =
        rig.UploadChain(command_buffer, *chain, *source, 0, &end);
    // Rows and layers apart in Roblox's buffer.
    VkBufferImageCopy padded{};
    padded.bufferOffset = end + 48;
    padded.bufferRowLength = 27 + 13;
    padded.bufferImageHeight = 14 + 6;
    padded.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2};
    padded.imageExtent = {27, 14, 1};
    rig.emulation().CmdCopyBufferToImage(
        device_->device(), command_buffer, source->source.buffer,
        layered->image, VK_IMAGE_LAYOUT_GENERAL, 1, &padded);
    rig.End(command_buffer);
    rig.emulation().UnmapMemory(device_->device(), source->source.memory);
    ASSERT_EQ(rig.Submit(queue(), {{command_buffer}}, VK_NULL_HANDLE),
              VK_SUCCESS);
    rig.vk().vkQueueWaitIdle(queue());
    Etc2GpuTesting().fail_slot_acquire = false;
    Etc2GpuDevice::Stats stats = rig.gpu()->stats();
    EXPECT_EQ(stats.rescues, 1U);
    EXPECT_EQ(stats.rescued_uploads, uploads.size() + 1);
    EXPECT_EQ(stats.dropped_uploads, 0U);
    rig.Check(uploads, source->bytes, "rescue");
    std::vector<std::vector<std::uint8_t>> texels;
    ASSERT_TRUE(rig.backend().ReadImage(layered->image, layered->shape,
                                        &texels));
    const VkDeviceSize block = EtcBlockBytes(layered->etc);
    const VkDeviceSize row_stride = ((padded.bufferRowLength + 3) / 4) * block;
    const VkDeviceSize layer_stride =
        row_stride * ((padded.bufferImageHeight + 3) / 4);
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
      std::vector<std::uint8_t> packed;
      for (std::uint32_t row = 0; row < (14 + 3) / 4; ++row) {
        const std::uint8_t* from = source->bytes.data() + padded.bufferOffset +
                                   layer * layer_stride + row * row_stride;
        packed.insert(packed.end(), from, from + ((27 + 3) / 4) * block);
      }
      std::vector<std::uint8_t> expected(27 * 14 *
                                         EtcDecodedTexelBytes(layered->etc));
      ASSERT_TRUE(DecodeEtcImage(layered->etc, packed.data(), packed.size(),
                                 27, 14, expected.data(), expected.size()));
      EXPECT_EQ(texels[layer], expected) << "padded rows, layer " << layer;
    }
    void* data = nullptr;  // mapped again, as the rig keeps it
    ASSERT_EQ(rig.emulation().MapMemory(device_->device(),
                                        source->source.memory, 0,
                                        VK_WHOLE_SIZE, 0, &data),
              VK_SUCCESS);

    // Neither a slot nor the rescue slot: dropped, counted, submitted.
    rig.emulation().ReleaseCommandBuffer(command_buffer);
    rig.Restart(command_buffer, pool,
                VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    rig.UploadChain(command_buffer, *chain, *source, 0);
    rig.End(command_buffer);
    Etc2GpuTesting().fail_slot_acquire = true;
    Etc2GpuTesting().fail_rescue_slot = true;
    ASSERT_EQ(rig.Submit(queue(), {{command_buffer}}, VK_NULL_HANDLE),
              VK_SUCCESS);
    rig.vk().vkQueueWaitIdle(queue());
    Etc2GpuTesting().fail_slot_acquire = false;
    Etc2GpuTesting().fail_rescue_slot = false;
    stats = rig.gpu()->stats();
    EXPECT_EQ(stats.rescues, 1U);
    EXPECT_EQ(stats.dropped_uploads, uploads.size());
    rig.Teardown();
  }
}

// A selftest mismatch leaves the device on the CPU route for good.
TEST_F(Etc2GpuRouteTest, ASelftestMismatchKeepsTheCpuRoute) {
  Etc2GpuTesting().corrupt_selftest = true;
  Rig rig(device_, GpuOptions());
  EXPECT_FALSE(rig.WarmUp(queue()));
  ASSERT_NE(rig.gpu(), nullptr);
  EXPECT_EQ(rig.gpu()->state(), Etc2GpuDevice::State::kFailed);
  EXPECT_NE(rig.gpu()->failure().find("selftest mismatch: ETC2_RGBA8"),
            std::string::npos)
      << rig.gpu()->failure();
  const VkCommandPool pool = rig.Pool(device_->queue_family());
  RigImage* image = rig.Image(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 40, 40, 3);
  RigSource* source = rig.Source(32 * 1024, 71);
  const VkCommandBuffer command_buffer = rig.Begin(pool);
  const std::vector<RigUpload> uploads =
      rig.UploadChain(command_buffer, *image, *source, 0);
  rig.End(command_buffer);
  ASSERT_EQ(rig.Submit(queue(), {{command_buffer}}, VK_NULL_HANDLE),
            VK_SUCCESS);
  rig.vk().vkQueueWaitIdle(queue());
  EXPECT_EQ(rig.gpu()->stats().passes, 0U);
  rig.Check(uploads, source->bytes, "CPU route after a failed selftest");
}

// vkDestroyDevice right after a submit: Destroy waits for the slot's fence
// (which covers Roblox's batch) before it frees the arena the copies read.
TEST_F(Etc2GpuRouteTest, DestroyDeviceWaitsForPassesInFlight) {
  Rig rig(device_, GpuOptions());
  ASSERT_TRUE(rig.WarmUp(queue()));
  const VkCommandPool pool = rig.Pool(device_->queue_family());
  RigImage* image = rig.Image(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 1024, 1024, 1);
  RigSource* source = rig.Source(512 * 1024, 81);
  const VkCommandBuffer command_buffer = rig.Begin(pool);
  rig.UploadChain(command_buffer, *image, *source, 0);
  rig.End(command_buffer);
  ASSERT_EQ(rig.Submit(queue(), {{command_buffer}}, VK_NULL_HANDLE),
            VK_SUCCESS);
  // No wait: the emulation's device goes while the pass may be running.
  rig.DestroyEmulationDevice();
  EXPECT_EQ(Etc2GpuDevice::Find(device_->device()), nullptr);
  rig.vk().vkQueueWaitIdle(queue());
}

}  // namespace
}  // namespace mocktail::graphics::testing
