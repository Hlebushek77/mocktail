#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "generated/etc2_shaders_spv.h"
#include "mocktail/graphics/bc_encoder.h"
#include "mocktail/graphics/etc2_decoder.h"
#include "mocktail/graphics/vulkan_etc2_gpu_plan.h"
#include "support/bc_reference.h"
#include "support/vulkan_test_device.h"

// The decode shader on a real GPU (DESIGN §9.3, ctest label "gpu"), ported
// from plus2-design/harness/decode_test.cc and e2e_test.inc, but with the
// production pieces: the committed SPIR-V (src/graphics/generated) and the
// production planner (BuildDecodePass) that lays out the copies, jobs and
// dispatches.
//  - ShaderParity: every format, random blocks of every mode, sizes 1-140,
//    row and image padding, 1-3 layers, odd source offsets: bit-exact
//    against the CPU decoder.
//  - BcParity: the etc2_bc shader (ETC2->BC mode) against the CPU BC
//    encoder: every format, 16384+ blocks each (random ones of every mode,
//    and blocks with flat, two-colour and transparent texels), refinement
//    on and off, through the production planner's BC jobs: bit-exact.
//  - BcHardwareDecodes: the blocks ETC2->BC mode writes, decoded by the GPU
//    itself (a NEAREST blit from the BC image to a float image), within one
//    8-bit step of the reference BC decoders (D3D10 rules).
//  - PrependE2E: the integration's batch shape, [decode pass, "Roblox"
//    command buffer with layout barriers and the rewritten padded-row copy]
//    submitted without a fence and followed by a fence-only submit, on the
//    graphics family and on a compute-only family; every image read back.
// Both skip without a device and while a live game runs (CR-25).

namespace mocktail::graphics::testing {
namespace {

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

const char* const kFormatNames[7] = {"ETC2_RGB8",     "ETC2_RGB8A1",
                                     "ETC2_RGBA8",    "EAC_R11",
                                     "EAC_R11_SNORM", "EAC_RG11",
                                     "EAC_RG11_SNORM"};

struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  std::uint8_t* mapped = nullptr;
  VkDeviceSize size = 0;
};

// Device objects for one test, all freed by the destructor.
class Gpu {
 public:
  explicit Gpu(HeadlessVulkanDevice* device) : device_(device) {}
  ~Gpu() {
    const auto& vk = device_->vk();
    const VkDevice device = device_->device();
    vk.vkDeviceWaitIdle(device);
    for (Buffer& buffer : buffers_) {
      vk.vkDestroyBuffer(device, buffer.buffer, nullptr);
      vk.vkFreeMemory(device, buffer.memory, nullptr);
    }
    for (const auto& [image, memory] : images_) {
      vk.vkDestroyImage(device, image, nullptr);
      vk.vkFreeMemory(device, memory, nullptr);
    }
    for (VkCommandPool pool : pools_) {
      vk.vkDestroyCommandPool(device, pool, nullptr);
    }
    for (VkFence fence : fences_) {
      vk.vkDestroyFence(device, fence, nullptr);
    }
    if (descriptors_ != VK_NULL_HANDLE) {
      vk.vkDestroyDescriptorPool(device, descriptors_, nullptr);
    }
    if (pipeline_ != VK_NULL_HANDLE) {
      vk.vkDestroyPipeline(device, pipeline_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
      vk.vkDestroyPipelineLayout(device, layout_, nullptr);
    }
    for (VkDescriptorSetLayout set_layout : {set0_, set1_}) {
      if (set_layout != VK_NULL_HANDLE) {
        vk.vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
      }
    }
  }

  const HeadlessVulkanDevice::Dispatch& vk() const { return device_->vk(); }
  VkDevice device() const { return device_->device(); }

  // `host`: host-visible coherent (cached when possible), mapped.
  Buffer CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
    Buffer out;
    out.size = size;
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    EXPECT_EQ(vk().vkCreateBuffer(device(), &info, nullptr, &out.buffer),
              VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vk().vkGetBufferMemoryRequirements(device(), out.buffer, &requirements);
    const std::uint32_t type =
        host ? device_->FindMemoryType(
                   requirements.memoryTypeBits,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
             : device_->FindMemoryType(requirements.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type;
    EXPECT_EQ(vk().vkAllocateMemory(device(), &allocate, nullptr, &out.memory),
              VK_SUCCESS);
    EXPECT_EQ(vk().vkBindBufferMemory(device(), out.buffer, out.memory, 0),
              VK_SUCCESS);
    if (host) {
      void* data = nullptr;
      EXPECT_EQ(vk().vkMapMemory(device(), out.memory, 0, VK_WHOLE_SIZE, 0,
                                 &data),
                VK_SUCCESS);
      out.mapped = static_cast<std::uint8_t*>(data);
    }
    buffers_.push_back(out);
    return out;
  }

  VkImage CreateImage(VkFormat format, std::uint32_t width,
                      std::uint32_t height, std::uint32_t mips,
                      std::uint32_t layers) {
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = mips;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage image = VK_NULL_HANDLE;
    EXPECT_EQ(CreateImageRaw(info, &image), VK_SUCCESS);
    return image;
  }

  // The decode pipeline, or with `bc` the etc2_bc one (same layout).
  bool BuildPipeline(bool bc = false) {
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = bc ? sizeof(kEtc2BcSpv) : sizeof(kEtc2DecodeSpv);
    module_info.pCode = bc ? kEtc2BcSpv : kEtc2DecodeSpv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (vk().vkCreateShaderModule(device(), &module_info, nullptr, &module) !=
        VK_SUCCESS) {
      return false;
    }
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (std::uint32_t index = 0; index < 2; ++index) {
      bindings[index].binding = index;
      bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[index].descriptorCount = 1;
      bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;
    vk().vkCreateDescriptorSetLayout(device(), &layout_info, nullptr, &set0_);
    layout_info.bindingCount = 1;
    vk().vkCreateDescriptorSetLayout(device(), &layout_info, nullptr, &set1_);
    const VkDescriptorSetLayout sets[2] = {set0_, set1_};
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pipeline_layout{};
    pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout.setLayoutCount = 2;
    pipeline_layout.pSetLayouts = sets;
    pipeline_layout.pushConstantRangeCount = 1;
    pipeline_layout.pPushConstantRanges = &push;
    vk().vkCreatePipelineLayout(device(), &pipeline_layout, nullptr, &layout_);
    VkComputePipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = module;
    pipeline.stage.pName = "main";
    pipeline.layout = layout_;
    const VkResult result = vk().vkCreateComputePipelines(
        device(), VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_);
    vk().vkDestroyShaderModule(device(), module, nullptr);
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = 64;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    vk().vkCreateDescriptorPool(device(), &pool, nullptr, &descriptors_);
    return result == VK_SUCCESS;
  }

  VkDescriptorSet AllocateSet(bool set1, std::vector<const Buffer*> buffers) {
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = descriptors_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = set1 ? &set1_ : &set0_;
    VkDescriptorSet set = VK_NULL_HANDLE;
    EXPECT_EQ(vk().vkAllocateDescriptorSets(device(), &allocate, &set),
              VK_SUCCESS);
    std::vector<VkDescriptorBufferInfo> infos;
    for (const Buffer* buffer : buffers) {
      infos.push_back({buffer->buffer, 0, VK_WHOLE_SIZE});
    }
    std::vector<VkWriteDescriptorSet> writes(infos.size());
    for (std::uint32_t index = 0; index < infos.size(); ++index) {
      writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[index].dstSet = set;
      writes[index].dstBinding = index;
      writes[index].descriptorCount = 1;
      writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[index].pBufferInfo = &infos[index];
    }
    vk().vkUpdateDescriptorSets(device(),
                                static_cast<std::uint32_t>(writes.size()),
                                writes.data(), 0, nullptr);
    return set;
  }

  VkCommandBuffer CommandBuffer(std::uint32_t family) {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = family;
    VkCommandPool pool = VK_NULL_HANDLE;
    EXPECT_EQ(vk().vkCreateCommandPool(device(), &pool_info, nullptr, &pool),
              VK_SUCCESS);
    pools_.push_back(pool);
    VkCommandBufferAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    EXPECT_EQ(
        vk().vkAllocateCommandBuffers(device(), &allocate, &command_buffer),
        VK_SUCCESS);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk().vkBeginCommandBuffer(command_buffer, &begin);
    return command_buffer;
  }

  VkFence Fence() {
    VkFenceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    EXPECT_EQ(vk().vkCreateFence(device(), &info, nullptr, &fence), VK_SUCCESS);
    fences_.push_back(fence);
    return fence;
  }

  void Barrier(VkCommandBuffer command_buffer, VkPipelineStageFlags source,
               VkAccessFlags source_access, VkPipelineStageFlags destination,
               VkAccessFlags destination_access) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = source_access;
    barrier.dstAccessMask = destination_access;
    vk().vkCmdPipelineBarrier(command_buffer, source, destination, 0, 1,
                              &barrier, 0, nullptr, 0, nullptr);
  }

  // The production pass: leading barrier, copy-in, decode, the barrier
  // that hands the output to transfers.
  void RecordPass(VkCommandBuffer command_buffer, const Etc2DecodePass& pass,
                  const Buffer& input, VkDescriptorSet set0,
                  VkDescriptorSet set1) {
    constexpr VkPipelineStageFlags kStages =
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    Barrier(command_buffer, kStages,
            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, kStages,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    for (const Etc2InputCopies& copies : pass.copies) {
      vk().vkCmdCopyBuffer(command_buffer, copies.source, input.buffer,
                           static_cast<std::uint32_t>(copies.regions.size()),
                           copies.regions.data());
    }
    Barrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT);
    vk().vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_);
    const VkDescriptorSet sets[2] = {set0, set1};
    vk().vkCmdBindDescriptorSets(command_buffer,
                                 VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 2,
                                 sets, 0, nullptr);
    for (const Etc2DecodeDispatch& dispatch : pass.dispatches) {
      const std::uint32_t push[2] = {dispatch.job_first, dispatch.job_count};
      vk().vkCmdPushConstants(command_buffer, layout_,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                              push);
      vk().vkCmdDispatch(command_buffer, dispatch.groups_x, dispatch.groups_y,
                         1);
    }
    Barrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT);
  }

 private:
  VkResult CreateImageRaw(const VkImageCreateInfo& info, VkImage* image) {
    auto create = reinterpret_cast<PFN_vkCreateImage>(
        vk().vkGetDeviceProcAddr(device(), "vkCreateImage"));
    VkResult result = create(device(), &info, nullptr, image);
    if (result != VK_SUCCESS) {
      return result;
    }
    VkMemoryRequirements requirements{};
    vk().vkGetImageMemoryRequirements(device(), *image, &requirements);
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = device_->FindMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    result = vk().vkAllocateMemory(device(), &allocate, nullptr, &memory);
    if (result == VK_SUCCESS) {
      result = vk().vkBindImageMemory(device(), *image, memory, 0);
    }
    images_.push_back({*image, memory});
    return result;
  }

  HeadlessVulkanDevice* device_;
  std::vector<Buffer> buffers_;
  std::vector<std::pair<VkImage, VkDeviceMemory>> images_;
  std::vector<VkCommandPool> pools_;
  std::vector<VkFence> fences_;
  VkDescriptorSetLayout set0_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout set1_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptors_ = VK_NULL_HANDLE;
};

// One upload as Roblox's buffer holds it.
struct Upload {
  EtcFormat format = EtcFormat::kEtc2Rgb8;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t layers = 1;
  std::uint32_t row_texels = 0;  // bufferRowLength
  std::uint32_t layer_rows = 0;  // bufferImageHeight
  Etc2GpuUpload gpu;
};

// Places the uploads in Roblox's buffer (block aligned, sometimes a few
// blocks apart) and in the output buffer (256-byte aligned), as
// PlanGpuRegions would. Returns the source and output sizes.
std::pair<VkDeviceSize, VkDeviceSize> Place(std::vector<Upload>* uploads,
                                            VkBuffer source, VkBuffer output,
                                            std::mt19937_64& random) {
  VkDeviceSize source_cursor = 0;
  VkDeviceSize output_cursor = 0;
  for (Upload& upload : *uploads) {
    const VkDeviceSize block = EtcBlockBytes(upload.format);
    Etc2GpuUpload& gpu = upload.gpu;
    gpu.source = source;
    gpu.format = upload.format;
    gpu.blocks_wide = (upload.width + 3) / 4;
    gpu.blocks_high = (upload.height + 3) / 4;
    gpu.layers = upload.layers;
    gpu.source_row_stride = ((upload.row_texels + 3) / 4) * block;
    gpu.source_layer_stride =
        gpu.source_row_stride * ((upload.layer_rows + 3) / 4);
    gpu.source_span = (gpu.layers - 1) * gpu.source_layer_stride +
                      (gpu.blocks_high - 1) * gpu.source_row_stride +
                      gpu.blocks_wide * block;
    gpu.compressed = static_cast<VkDeviceSize>(gpu.blocks_wide) *
                     gpu.blocks_high * gpu.layers * block;
    source_cursor = AlignUp(source_cursor, block) + block * (random() % 3);
    gpu.source_offset = source_cursor;
    source_cursor += gpu.source_span;
    gpu.output = output;
    gpu.output_offset = AlignUp(output_cursor, kEtc2GpuRegionAlignment);
    gpu.output_bytes = static_cast<VkDeviceSize>(gpu.blocks_wide) *
                       gpu.blocks_high * gpu.layers * 16 *
                       EtcDecodedTexelBytes(upload.format);
    output_cursor = gpu.output_offset + gpu.output_bytes;
  }
  return {AlignUp(source_cursor, 16), AlignUp(output_cursor, 16)};
}

// The CPU decoder's texels of one layer: rows packed, as the CPU route
// packs them, then DecodeEtcImage.
std::vector<std::uint8_t> CpuDecodeLayer(const Upload& upload,
                                         const std::uint8_t* source,
                                         std::uint32_t layer) {
  const Etc2GpuUpload& gpu = upload.gpu;
  const VkDeviceSize packed_row =
      static_cast<VkDeviceSize>(gpu.blocks_wide) * EtcBlockBytes(gpu.format);
  std::vector<std::uint8_t> packed(packed_row * gpu.blocks_high);
  for (std::uint32_t row = 0; row < gpu.blocks_high; ++row) {
    std::memcpy(packed.data() + row * packed_row,
                source + gpu.source_offset + layer * gpu.source_layer_stride +
                    row * gpu.source_row_stride,
                packed_row);
  }
  std::vector<std::uint8_t> out(static_cast<std::size_t>(upload.width) *
                                upload.height *
                                EtcDecodedTexelBytes(upload.format));
  EXPECT_TRUE(DecodeEtcImage(upload.format, packed.data(), packed.size(),
                             upload.width, upload.height, out.data(),
                             out.size()));
  return out;
}

// Which branch of the ETC2 colour decoder a block takes (individual,
// differential, T, H, planar), to show the random blocks cover them all.
int ColorMode(const std::uint8_t* block, bool punchthrough) {
  static constexpr int kDelta[8] = {0, 1, 2, 3, -4, -3, -2, -1};
  if (!punchthrough && (block[3] & 2) == 0) {
    return 0;
  }
  const int r = (block[0] >> 3) + kDelta[block[0] & 7];
  const int g = (block[1] >> 3) + kDelta[block[1] & 7];
  const int b = (block[2] >> 3) + kDelta[block[2] & 7];
  if (r < 0 || r > 31) {
    return 2;
  }
  if (g < 0 || g > 31) {
    return 3;
  }
  return (b < 0 || b > 31) ? 4 : 1;
}

class Etc2GpuShaderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    HeadlessVulkanDevice::Options options;
    options.compute_only_family = true;
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
    if (device_ != nullptr) {
      for (const std::string& message : device_->TakeValidationMessages()) {
        ADD_FAILURE() << "validation: " << message;
      }
    }
  }

  static HeadlessVulkanDevice* device_;
  static std::string skip_reason_;
};

HeadlessVulkanDevice* Etc2GpuShaderTest::device_ = nullptr;
std::string Etc2GpuShaderTest::skip_reason_;

TEST_F(Etc2GpuShaderTest, ShaderParity) {
  Gpu gpu(device_);
  ASSERT_TRUE(gpu.BuildPipeline());
  const auto& vk = device_->vk();
  std::mt19937_64 random(1);
  std::array<std::uint64_t, 5> modes{};
  std::uint64_t blocks = 0;
  for (int f = 0; f < 7; ++f) {
    SCOPED_TRACE(kFormatNames[f]);
    const auto format = static_cast<EtcFormat>(f);
    std::vector<Upload> uploads;
    const std::uint32_t fixed[][2] = {{1, 1},   {2, 2},     {3, 5},   {4, 4},
                                      {5, 5},   {7, 3},     {64, 64}, {60, 33},
                                      {37, 37}, {1024, 1},  {1, 513}, {256, 256},
                                      {130, 66}};
    for (const auto& size : fixed) {
      uploads.push_back({format, size[0], size[1], 1, size[0], size[1], {}});
    }
    for (int index = 0; index < 400; ++index) {
      Upload upload;
      upload.format = format;
      upload.width = 1 + random() % 140;
      upload.height = 1 + random() % 140;
      upload.layers = random() % 4 == 0 ? 1 + random() % 3 : 1;
      // Row padding, sometimes far apart (block rows copied one by one).
      upload.row_texels = upload.width + (random() % 2 != 0
                                              ? 4 * (random() % 4) + random() % 4
                                              : 0);
      if (random() % 8 == 0) {
        upload.row_texels += 512;
      }
      upload.layer_rows =
          upload.height + (random() % 3 == 0 ? 4 * (random() % 3) : 0);
      uploads.push_back(upload);
    }
    Buffer source;
    Buffer output;
    // Sizes first, buffers after: placement needs the handles, so place
    // twice (the second time with the real ones, same random draws).
    std::mt19937_64 placement = random;
    const auto sizes = Place(&uploads, VK_NULL_HANDLE, VK_NULL_HANDLE,
                             placement);
    source = gpu.CreateBuffer(sizes.first, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              true);
    output = gpu.CreateBuffer(sizes.second,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              false);
    Place(&uploads, source.buffer, output.buffer, random);
    for (VkDeviceSize byte = 0; byte < sizes.first; ++byte) {
      source.mapped[byte] = static_cast<std::uint8_t>(random());
    }
    const std::vector<std::uint8_t> bytes(source.mapped,
                                          source.mapped + sizes.first);
    std::vector<Etc2GpuUpload> gpu_uploads;
    for (const Upload& upload : uploads) {
      gpu_uploads.push_back(upload.gpu);
    }
    Etc2DecodePass pass;
    ASSERT_TRUE(BuildDecodePass(gpu_uploads, 0, 0, &pass));
    const Buffer input = gpu.CreateBuffer(
        std::max<VkDeviceSize>(pass.input_bytes, 16),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        false);
    const Buffer jobs = gpu.CreateBuffer(
        pass.jobs.size() * sizeof(Etc2DecodeJob),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
    std::memcpy(jobs.mapped, pass.jobs.data(),
                pass.jobs.size() * sizeof(Etc2DecodeJob));
    const Buffer readback = gpu.CreateBuffer(
        sizes.second, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    std::memset(readback.mapped, 0xCD, sizes.second);
    const VkDescriptorSet set0 = gpu.AllocateSet(false, {&input, &jobs});
    const VkDescriptorSet set1 = gpu.AllocateSet(true, {&output});
    const VkCommandBuffer command_buffer =
        gpu.CommandBuffer(device_->queue_family());
    gpu.RecordPass(command_buffer, pass, input, set0, set1);
    const VkBufferCopy all{0, 0, sizes.second};
    vk.vkCmdCopyBuffer(command_buffer, output.buffer, readback.buffer, 1, &all);
    gpu.Barrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                VK_ACCESS_HOST_READ_BIT);
    vk.vkEndCommandBuffer(command_buffer);
    const VkFence fence = gpu.Fence();
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer;
    ASSERT_EQ(vk.vkQueueSubmit(device_->queue(), 1, &submit, fence),
              VK_SUCCESS);
    ASSERT_EQ(vk.vkWaitForFences(device_->device(), 1, &fence, VK_TRUE,
                                 30ULL * 1000000000ULL),
              VK_SUCCESS);

    int failures = 0;
    const std::uint32_t texel = EtcDecodedTexelBytes(format);
    for (const Upload& upload : uploads) {
      const Etc2GpuUpload& g = upload.gpu;
      blocks += static_cast<std::uint64_t>(g.blocks_wide) * g.blocks_high *
                g.layers;
      for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
        const std::vector<std::uint8_t> cpu =
            CpuDecodeLayer(upload, bytes.data(), layer);
        const std::uint8_t* layer_out =
            readback.mapped + g.output_offset +
            static_cast<VkDeviceSize>(layer) * g.blocks_high * 4 *
                g.blocks_wide * 4 * texel;
        for (std::uint32_t y = 0; y < upload.height; ++y) {
          const std::uint8_t* row =
              layer_out + static_cast<VkDeviceSize>(y) * g.blocks_wide * 4 *
                              texel;
          if (std::memcmp(row, cpu.data() + static_cast<std::size_t>(y) *
                                                upload.width * texel,
                          static_cast<std::size_t>(upload.width) * texel) !=
              0) {
            if (failures++ < 5) {
              ADD_FAILURE() << upload.width << "x" << upload.height
                            << " layer " << layer << " row " << y
                            << " differs from the CPU decoder";
            }
            break;
          }
        }
      }
      if (f <= 2) {
        for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
          for (std::uint32_t by = 0; by < g.blocks_high; ++by) {
            for (std::uint32_t bx = 0; bx < g.blocks_wide; ++bx) {
              const std::uint8_t* block =
                  bytes.data() + g.source_offset +
                  layer * g.source_layer_stride + by * g.source_row_stride +
                  bx * EtcBlockBytes(format) + (f == 2 ? 8 : 0);
              ++modes[ColorMode(block, f == 1)];
            }
          }
        }
      }
    }
    EXPECT_EQ(failures, 0);
  }
  std::printf("blocks=%llu modes: individual=%llu differential=%llu T=%llu "
              "H=%llu planar=%llu\n",
              static_cast<unsigned long long>(blocks),
              static_cast<unsigned long long>(modes[0]),
              static_cast<unsigned long long>(modes[1]),
              static_cast<unsigned long long>(modes[2]),
              static_cast<unsigned long long>(modes[3]),
              static_cast<unsigned long long>(modes[4]));
  for (const std::uint64_t count : modes) {
    EXPECT_GT(count, 1000U) << "every colour mode is covered";
  }
}

// A block for BcParity: random bytes, or (one in two) random bytes with
// their texel indices changed so that the BC encoders' special cases come
// up: every texel on one modifier (flat EAC blocks, two-colour ETC2
// sub-blocks), an EAC multiplier of 0, or (punch-through) the index that
// makes a texel transparent on every texel.
void BcParityBlock(EtcFormat format, std::mt19937_64& random,
                   std::uint8_t* block) {
  const std::uint32_t bytes = EtcBlockBytes(format);
  for (std::uint32_t byte = 0; byte < bytes; ++byte) {
    block[byte] = static_cast<std::uint8_t>(random());
  }
  const std::uint64_t pick = random() % 8;
  const auto eac = [&](std::uint8_t* half) {
    if (pick == 4) {
      std::memset(half + 2, 0, 6);  // every texel on index 0
    } else if (pick == 5) {
      half[1] &= 0x0F;  // multiplier 0: one value
    }
  };
  switch (format) {
    case EtcFormat::kEtc2Rgb8:
    case EtcFormat::kEtc2Rgb8A1:
    case EtcFormat::kEtc2Rgba8: {
      std::uint8_t* colour = block + (format == EtcFormat::kEtc2Rgba8 ? 8 : 0);
      if (pick == 4) {
        std::memset(colour + 4, 0, 4);
      } else if (pick == 5 || pick == 6) {
        // msb 1, lsb 0 everywhere; with the opaque bit clear that is
        // transparent in punch-through blocks (all modes but planar).
        colour[4] = colour[5] = 0xFF;
        colour[6] = colour[7] = 0x00;
        if (pick == 6) {
          colour[3] &= static_cast<std::uint8_t>(~2U);
        }
      }
      if (format == EtcFormat::kEtc2Rgba8) {
        eac(block);
      }
      break;
    }
    case EtcFormat::kEacR11:
    case EtcFormat::kEacR11Signed:
      eac(block);
      break;
    case EtcFormat::kEacRg11:
    case EtcFormat::kEacRg11Signed:
      eac(block);
      eac(block + 8);
      break;
  }
}

// DESIGN §9.3 BcParity: GPU BC equals CPU BC, 16384 blocks or more per
// format, with the BC1 refinement on and off.
TEST_F(Etc2GpuShaderTest, BcParity) {
  Gpu gpu(device_);
  ASSERT_TRUE(gpu.BuildPipeline(/*bc=*/true));
  const auto& vk = device_->vk();
  std::mt19937_64 random(3);
  // How often the encoders' special cases came up (CPU side): BC1 blocks
  // whose endpoints are equal, fully transparent BC1 blocks, BC1 blocks in
  // 3-colour mode, and BC4 blocks with a single value.
  std::uint64_t flat_bc1 = 0;
  std::uint64_t transparent_bc1 = 0;
  std::uint64_t three_colour_bc1 = 0;
  std::uint64_t flat_bc4 = 0;
  std::uint64_t total_blocks = 0;
  for (const bool refine : {true, false}) {
    for (int f = 0; f < 7; ++f) {
      SCOPED_TRACE(std::string(kFormatNames[f]) +
                   (refine ? ", refined" : ", not refined"));
      const auto format = static_cast<EtcFormat>(f);
      const VkDeviceSize block_bytes = EtcBlockBytes(format);
      // Uploads of 1-140 texels, padded rows (some far apart, copied block
      // row by block row), 1-3 layers, until 16384 blocks.
      std::vector<Upload> uploads;
      std::uint64_t format_blocks = 0;
      const std::uint32_t fixed[][2] = {{1, 1}, {4, 4}, {7, 3}, {128, 128}};
      for (const auto& size : fixed) {
        uploads.push_back({format, size[0], size[1], 1, size[0], size[1], {}});
        format_blocks += ((size[0] + 3) / 4) * ((size[1] + 3) / 4);
      }
      while (format_blocks < 16384) {
        Upload upload;
        upload.format = format;
        upload.width = 1 + random() % 140;
        upload.height = 1 + random() % 140;
        upload.layers = random() % 4 == 0 ? 1 + random() % 3 : 1;
        upload.row_texels =
            upload.width +
            (random() % 2 != 0 ? 4 * (random() % 4) + random() % 4 : 0);
        if (random() % 8 == 0) {
          upload.row_texels += 512;
        }
        upload.layer_rows =
            upload.height + (random() % 3 == 0 ? 4 * (random() % 3) : 0);
        format_blocks += static_cast<std::uint64_t>((upload.width + 3) / 4) *
                         ((upload.height + 3) / 4) * upload.layers;
        uploads.push_back(upload);
      }
      // Sizes first, then the buffers, then the real placement (the same
      // draws); BC output is the blocks themselves.
      std::mt19937_64 placement = random;
      auto sizes = Place(&uploads, VK_NULL_HANDLE, VK_NULL_HANDLE, placement);
      const Buffer source =
          gpu.CreateBuffer(sizes.first, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
      Place(&uploads, source.buffer, VK_NULL_HANDLE, random);
      VkDeviceSize output_bytes = 0;
      for (Upload& upload : uploads) {
        Etc2GpuUpload& g = upload.gpu;
        g.bc = true;
        g.output_offset = AlignUp(output_bytes, kEtc2GpuRegionAlignment);
        g.output_bytes = g.compressed;
        output_bytes = g.output_offset + g.output_bytes;
      }
      const Buffer output = gpu.CreateBuffer(
          output_bytes,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          false);
      for (VkDeviceSize byte = 0; byte < sizes.first; ++byte) {
        source.mapped[byte] = static_cast<std::uint8_t>(random());
      }
      for (const Upload& upload : uploads) {
        const Etc2GpuUpload& g = upload.gpu;
        for (std::uint32_t layer = 0; layer < g.layers; ++layer) {
          for (std::uint32_t by = 0; by < g.blocks_high; ++by) {
            for (std::uint32_t bx = 0; bx < g.blocks_wide; ++bx) {
              BcParityBlock(format, random,
                            source.mapped + g.source_offset +
                                layer * g.source_layer_stride +
                                by * g.source_row_stride + bx * block_bytes);
            }
          }
        }
      }
      const std::vector<std::uint8_t> bytes(source.mapped,
                                            source.mapped + sizes.first);
      std::vector<Etc2GpuUpload> gpu_uploads;
      for (Upload& upload : uploads) {
        upload.gpu.output = output.buffer;
        gpu_uploads.push_back(upload.gpu);
      }
      Etc2DecodePass pass;
      ASSERT_TRUE(BuildDecodePass(gpu_uploads, 0, 0, &pass));
      ASSERT_EQ(pass.dispatches.size(), 1U);
      EXPECT_TRUE(pass.dispatches[0].bc);
      for (Etc2DecodeJob& job : pass.jobs) {
        // The production jobs refine (ETC2->BC mode's choice); the other
        // way round is the shader's flag cleared.
        EXPECT_EQ(job.flags, kEtc2ToBcRefine ? kEtc2BcJobRefine : 0U);
        job.flags = refine ? kEtc2BcJobRefine : 0;
      }
      const Buffer input = gpu.CreateBuffer(
          std::max<VkDeviceSize>(pass.input_bytes, 16),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          false);
      const Buffer jobs =
          gpu.CreateBuffer(pass.jobs.size() * sizeof(Etc2DecodeJob),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
      std::memcpy(jobs.mapped, pass.jobs.data(),
                  pass.jobs.size() * sizeof(Etc2DecodeJob));
      const Buffer readback = gpu.CreateBuffer(
          output_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
      std::memset(readback.mapped, 0xCD, output_bytes);
      const VkDescriptorSet set0 = gpu.AllocateSet(false, {&input, &jobs});
      const VkDescriptorSet set1 = gpu.AllocateSet(true, {&output});
      const VkCommandBuffer command_buffer =
          gpu.CommandBuffer(device_->queue_family());
      gpu.RecordPass(command_buffer, pass, input, set0, set1);
      const VkBufferCopy all{0, 0, output_bytes};
      vk.vkCmdCopyBuffer(command_buffer, output.buffer, readback.buffer, 1,
                         &all);
      gpu.Barrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                  VK_ACCESS_HOST_READ_BIT);
      vk.vkEndCommandBuffer(command_buffer);
      const VkFence fence = gpu.Fence();
      VkSubmitInfo submit{};
      submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command_buffer;
      ASSERT_EQ(vk.vkQueueSubmit(device_->queue(), 1, &submit, fence),
                VK_SUCCESS);
      ASSERT_EQ(vk.vkWaitForFences(device_->device(), 1, &fence, VK_TRUE,
                                   30ULL * 1000000000ULL),
                VK_SUCCESS);

      int failures = 0;
      for (const Upload& upload : uploads) {
        const Etc2GpuUpload& g = upload.gpu;
        std::uint64_t index = 0;
        for (std::uint32_t layer = 0; layer < g.layers; ++layer) {
          for (std::uint32_t by = 0; by < g.blocks_high; ++by) {
            for (std::uint32_t bx = 0; bx < g.blocks_wide; ++bx, ++index) {
              std::uint8_t cpu[16] = {};
              TranscodeEtcBlockToBc(format,
                                    bytes.data() + g.source_offset +
                                        layer * g.source_layer_stride +
                                        by * g.source_row_stride +
                                        bx * block_bytes,
                                    refine, cpu);
              const std::uint8_t* gpu_block =
                  readback.mapped + g.output_offset + index * block_bytes;
              if (std::memcmp(gpu_block, cpu, block_bytes) != 0 &&
                  failures++ < 5) {
                std::string hex;
                char digits[4];
                for (VkDeviceSize byte = 0; byte < block_bytes; ++byte) {
                  std::snprintf(digits, sizeof(digits), "%02x",
                                gpu_block[byte]);
                  hex += digits;
                  hex += byte + 1 == block_bytes ? " vs cpu " : "";
                }
                for (VkDeviceSize byte = 0; byte < block_bytes; ++byte) {
                  std::snprintf(digits, sizeof(digits), "%02x", cpu[byte]);
                  hex += digits;
                }
                ADD_FAILURE() << upload.width << "x" << upload.height
                              << " layer " << layer << " block (" << bx << ", "
                              << by << "): gpu " << hex;
              }
              // The colour (BC1) half and the BC4 halves.
              const std::uint8_t* bc1 =
                  format == EtcFormat::kEtc2Rgba8 ? cpu + 8 : cpu;
              if (f <= 2) {
                const std::uint16_t c0 =
                    static_cast<std::uint16_t>(bc1[0] | (bc1[1] << 8));
                const std::uint16_t c1 =
                    static_cast<std::uint16_t>(bc1[2] | (bc1[3] << 8));
                flat_bc1 += c0 == c1 ? 1 : 0;
                three_colour_bc1 += c0 < c1 ? 1 : 0;
                transparent_bc1 += c0 == 0 && c1 == 0 && bc1[4] == 0xFF &&
                                           bc1[5] == 0xFF && bc1[6] == 0xFF &&
                                           bc1[7] == 0xFF
                                       ? 1
                                       : 0;
              }
              if (f >= 2) {
                for (VkDeviceSize half = 0; half < (f >= 5 ? 2U : 1U); ++half) {
                  const std::uint8_t* bc4 = cpu + 8 * half;
                  flat_bc4 += bc4[0] == bc4[1] ? 1 : 0;
                }
              }
            }
          }
        }
        total_blocks += index;
      }
      EXPECT_EQ(failures, 0);
      EXPECT_GE(format_blocks, 16384U);
    }
  }
  std::printf(
      "blocks=%llu bc1: flat=%llu three-colour=%llu transparent=%llu "
      "bc4: flat=%llu\n",
      static_cast<unsigned long long>(total_blocks),
      static_cast<unsigned long long>(flat_bc1),
      static_cast<unsigned long long>(three_colour_bc1),
      static_cast<unsigned long long>(transparent_bc1),
      static_cast<unsigned long long>(flat_bc4));
  EXPECT_GT(flat_bc1, 100U);
  EXPECT_GT(three_colour_bc1, 100U);
  EXPECT_GT(transparent_bc1, 100U);
  EXPECT_GT(flat_bc4, 100U);
}

// DESIGN §9.3 (BcParity, optional part): the BC blocks the CPU encoder
// (and so the etc2_bc shader) writes mean on the GPU what the reference
// decoders say: punch-through texels transparent, BC3 colour in 4-colour
// mode, BC4/BC5 SNORM endpoints as mapped. The hardware decodes them in a
// NEAREST blit to a float image, which keeps its values unrounded. An
// endpoint texel and every alpha of BC1 must be within one 8-bit step (one
// 127th for SNORM) of the reference; an interpolated one may also be off by
// kSpanTolerance of its two endpoints' distance, because GPUs approximate
// the D3D10 thirds and sevenths (NVIDIA's BC1 green is up to 5 steps off on
// a 231-step span, its BC4 up to about 2). A wrong mode, byte order or
// SNORM mapping is far outside that. Formats the device cannot blit are
// skipped.
constexpr double kSpanTolerance = 0.04;

TEST_F(Etc2GpuShaderTest, BcHardwareDecodesTheBlocksAsTheReference) {
  Gpu gpu(device_);
  const auto& vk = device_->vk();
  constexpr std::uint32_t kSize = 64;  // 16 x 16 blocks
  constexpr std::uint32_t kBlocks = (kSize / 4) * (kSize / 4);
  struct Case {
    VkFormat bc;
    VkFormat decoded;
    std::uint32_t channels;
  };
  const Case cases[7] = {
      {VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_R32G32B32A32_SFLOAT, 4},
      {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_R32G32B32A32_SFLOAT, 4},
      {VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_R32G32B32A32_SFLOAT, 4},
      {VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_R32_SFLOAT, 1},
      {VK_FORMAT_BC4_SNORM_BLOCK, VK_FORMAT_R32_SFLOAT, 1},
      {VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_R32G32_SFLOAT, 2},
      {VK_FORMAT_BC5_SNORM_BLOCK, VK_FORMAT_R32G32_SFLOAT, 2},
  };
  std::mt19937_64 random(5);
  int checked = 0;
  double worst_interpolated = 0;  // (error - 1 step) / span
  for (int f = 0; f < 7; ++f) {
    SCOPED_TRACE(kFormatNames[f]);
    const auto format = static_cast<EtcFormat>(f);
    const Case& c = cases[f];
    if (!device_->OptimalTilingSupports(
            c.bc, VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT) ||
        !device_->OptimalTilingSupports(c.decoded,
                                        VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
      std::printf("%s: the device cannot blit it, skipped\n", kFormatNames[f]);
      continue;
    }
    const VkDeviceSize block_bytes = EtcBlockBytes(format);
    std::vector<std::uint8_t> etc(kBlocks * block_bytes);
    for (std::uint32_t block = 0; block < kBlocks; ++block) {
      BcParityBlock(format, random, etc.data() + block * block_bytes);
    }
    const Buffer upload =
        gpu.CreateBuffer(etc.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    ASSERT_TRUE(TranscodeEtcToBc(format, etc.data(), etc.size(), upload.mapped,
                                 etc.size(), kEtc2ToBcRefine));
    const std::vector<std::uint8_t> bc(upload.mapped,
                                       upload.mapped + etc.size());
    const VkDeviceSize texel_bytes = 4ULL * c.channels;
    const Buffer readback = gpu.CreateBuffer(
        kSize * kSize * texel_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    const VkImage source = gpu.CreateImage(c.bc, kSize, kSize, 1, 1);
    const VkImage decoded = gpu.CreateImage(c.decoded, kSize, kSize, 1, 1);
    const VkCommandBuffer command_buffer =
        gpu.CommandBuffer(device_->queue_family());
    const auto transition = [&](VkImage image, VkImageLayout from,
                                VkImageLayout to, VkAccessFlags source_access,
                                VkAccessFlags destination_access) {
      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.oldLayout = from;
      barrier.newLayout = to;
      barrier.srcAccessMask = source_access;
      barrier.dstAccessMask = destination_access;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                              nullptr, 1, &barrier);
    };
    transition(source, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT);
    transition(decoded, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {kSize, kSize, 1};
    vk.vkCmdCopyBufferToImage(command_buffer, upload.buffer, source,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    transition(source, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {static_cast<std::int32_t>(kSize),
                          static_cast<std::int32_t>(kSize), 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[1] = blit.srcOffsets[1];
    const auto blit_image = reinterpret_cast<PFN_vkCmdBlitImage>(
        vk.vkGetDeviceProcAddr(device_->device(), "vkCmdBlitImage"));
    ASSERT_NE(blit_image, nullptr);
    blit_image(command_buffer, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               decoded, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
               VK_FILTER_NEAREST);
    transition(decoded, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    vk.vkCmdCopyImageToBuffer(command_buffer, decoded,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              readback.buffer, 1, &region);
    gpu.Barrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                VK_ACCESS_HOST_READ_BIT);
    vk.vkEndCommandBuffer(command_buffer);
    const VkFence fence = gpu.Fence();
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer;
    ASSERT_EQ(vk.vkQueueSubmit(device_->queue(), 1, &submit, fence),
              VK_SUCCESS);
    ASSERT_EQ(vk.vkWaitForFences(device_->device(), 1, &fence, VK_TRUE,
                                 30ULL * 1000000000ULL),
              VK_SUCCESS);

    const bool snorm = format == EtcFormat::kEacR11Signed ||
                       format == EtcFormat::kEacRg11Signed;
    int failures = 0;
    for (std::uint32_t block = 0; block < kBlocks; ++block) {
      const std::uint8_t* bytes = bc.data() + block * block_bytes;
      std::uint32_t words[4] = {};
      std::memcpy(words, bytes, block_bytes);
      // Each channel's endpoint distance, and which texels sit on an
      // endpoint (BC1 indices 0 and 1, BC4 indices 0 and 1).
      double span[4] = {};
      bool on_endpoint[16][4] = {};
      const auto bc4_span = [&](const std::uint32_t* half, int channel) {
        int e0 = static_cast<int>(half[0] & 0xFF);
        int e1 = static_cast<int>((half[0] >> 8) & 0xFF);
        if (snorm) {
          e0 = std::max<int>(static_cast<std::int8_t>(e0), -127);
          e1 = std::max<int>(static_cast<std::int8_t>(e1), -127);
        }
        span[channel] = std::abs(e0 - e1);
        const std::uint64_t indices =
            (static_cast<std::uint64_t>(half[1]) << 16) | (half[0] >> 16);
        for (int texel = 0; texel < 16; ++texel) {
          on_endpoint[texel][channel] = ((indices >> (3 * texel)) & 7) < 2;
        }
      };
      if (f <= 2) {
        const std::uint32_t* colour = f == 2 ? words + 2 : words;
        const bcref::Rgb e0 = bcref::Unpack565(colour[0] & 0xFFFF);
        const bcref::Rgb e1 = bcref::Unpack565(colour[0] >> 16);
        span[0] = std::abs(e0.r - e1.r);
        span[1] = std::abs(e0.g - e1.g);
        span[2] = std::abs(e0.b - e1.b);
        for (int texel = 0; texel < 16; ++texel) {
          const bool endpoint = ((colour[1] >> (2 * texel)) & 3) < 2;
          on_endpoint[texel][0] = on_endpoint[texel][1] =
              on_endpoint[texel][2] = endpoint;
          on_endpoint[texel][3] = true;  // BC1 alpha is 0 or 1
        }
        if (f == 2) {
          bc4_span(words, 3);
        }
      } else {
        for (std::uint32_t channel = 0; channel < c.channels; ++channel) {
          bc4_span(words + 2 * channel, static_cast<int>(channel));
        }
      }
      // The reference, per texel (row-major) and channel, in 8-bit steps
      // (SNORM: steps of 1/127, from -127).
      double expected[16][4] = {};
      if (f <= 2) {
        std::uint8_t rgba[64];
        bcref::DecodeBc1(f == 2 ? words + 2 : words, f == 2, rgba);
        for (int texel = 0; texel < 16; ++texel) {
          for (int channel = 0; channel < 4; ++channel) {
            expected[texel][channel] = rgba[texel * 4 + channel];
          }
        }
        if (f == 0) {
          for (auto& texel : expected) {
            texel[3] = 255;  // BC1_RGB: alpha is 1
          }
        } else if (f == 2) {
          int alpha[16];
          bcref::DecodeBc4Fixed(words, false, alpha);
          for (int texel = 0; texel < 16; ++texel) {
            expected[texel][3] = alpha[texel] / 256.0;
          }
        }
      } else {
        for (std::uint32_t channel = 0; channel < c.channels; ++channel) {
          int values[16];
          bcref::DecodeBc4Fixed(words + 2 * channel, snorm, values);
          for (int texel = 0; texel < 16; ++texel) {
            expected[texel][channel] =
                values[texel] / 256.0 - (snorm ? 127.0 : 0.0);
          }
        }
      }
      const std::uint32_t bx = block % (kSize / 4);
      const std::uint32_t by = block / (kSize / 4);
      for (int texel = 0; texel < 16; ++texel) {
        const std::uint32_t x = 4 * bx + texel % 4;
        const std::uint32_t y = 4 * by + texel / 4;
        float hardware[4] = {};
        std::memcpy(hardware, readback.mapped + (y * kSize + x) * texel_bytes,
                    texel_bytes);
        for (std::uint32_t channel = 0; channel < c.channels; ++channel) {
          const double got = hardware[channel] * (snorm ? 127.0 : 255.0);
          const double want = expected[texel][channel];
          const double error = std::abs(got - want);
          ++checked;
          if (!on_endpoint[texel][channel] && span[channel] > 0) {
            worst_interpolated =
                std::max(worst_interpolated, (error - 1.0) / span[channel]);
          }
          const double allowed = 1.0 + (on_endpoint[texel][channel]
                                            ? 0.0
                                            : kSpanTolerance * span[channel]);
          if (error > allowed && failures++ < 5) {
            char words_hex[48];
            std::snprintf(words_hex, sizeof(words_hex), "%08x %08x %08x %08x",
                          words[0], words[1], words[2], words[3]);
            ADD_FAILURE() << "block " << block << " (" << words_hex
                          << ") texel " << texel << " channel " << channel
                          << ": the GPU decodes " << got << ", the reference "
                          << want;
          }
        }
      }
    }
    EXPECT_EQ(failures, 0);
  }
  std::printf(
      "%d channel values checked; interpolated ones off by at most "
      "1 step + %.1f %% of their endpoints' distance\n",
      checked, 100.0 * worst_interpolated);
}

VkFormat HostFormat(EtcFormat format) {
  switch (format) {
    case EtcFormat::kEtc2Rgb8:
    case EtcFormat::kEtc2Rgb8A1:
    case EtcFormat::kEtc2Rgba8:
      return VK_FORMAT_R8G8B8A8_SRGB;
    case EtcFormat::kEacR11:
      return VK_FORMAT_R16_UNORM;
    case EtcFormat::kEacR11Signed:
      return VK_FORMAT_R16_SNORM;
    case EtcFormat::kEacRg11:
      return VK_FORMAT_R16G16_UNORM;
    case EtcFormat::kEacRg11Signed:
      break;
  }
  return VK_FORMAT_R16G16_SNORM;
}

// [decode pass, "Roblox" command buffer] in one batch without a fence, then
// a fence-only submit: the integration's shape (DESIGN §3.4, §5.1, §5.4).
void PrependEndToEnd(HeadlessVulkanDevice* device, std::uint32_t family) {
  const VkQueue queue = device->Queue(family);
  ASSERT_NE(queue, VK_NULL_HANDLE);
  Gpu gpu(device);
  ASSERT_TRUE(gpu.BuildPipeline());
  const auto& vk = device->vk();
  std::mt19937_64 random(3);
  struct Image {
    EtcFormat format = EtcFormat::kEtc2Rgb8;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mips = 0;
    std::uint32_t layers = 0;
    VkImage image = VK_NULL_HANDLE;
    std::vector<std::size_t> uploads;  // one per mip
    std::vector<VkDeviceSize> readback;
  };
  std::vector<Image> images;
  const std::uint32_t dims[][4] = {{37, 37, 6, 1},     {60, 33, 6, 1},
                                   {130, 66, 3, 2},    {1024, 1024, 11, 1},
                                   {5, 3, 3, 1},       {256, 64, 9, 3}};
  for (int f = 0; f < 7; ++f) {
    for (const auto& d : dims) {
      Image image;
      image.format = static_cast<EtcFormat>(f);
      image.width = d[0];
      image.height = d[1];
      image.mips = d[2];
      image.layers = d[3];
      images.push_back(image);
    }
  }
  std::vector<Upload> uploads;
  for (Image& image : images) {
    for (std::uint32_t mip = 0; mip < image.mips; ++mip) {
      const std::uint32_t width = std::max(1u, image.width >> mip);
      const std::uint32_t height = std::max(1u, image.height >> mip);
      Upload upload{image.format, width, height, image.layers, width, height,
                    {}};
      if (random() % 2 != 0) {
        upload.row_texels += 4 * (1 + random() % 3);  // Roblox's row padding
      }
      image.uploads.push_back(uploads.size());
      uploads.push_back(upload);
    }
  }
  std::mt19937_64 placement = random;
  const auto sizes = Place(&uploads, VK_NULL_HANDLE, VK_NULL_HANDLE, placement);
  const Buffer source =
      gpu.CreateBuffer(sizes.first, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
  const Buffer output = gpu.CreateBuffer(
      sizes.second,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      false);
  Place(&uploads, source.buffer, output.buffer, random);
  for (VkDeviceSize byte = 0; byte < sizes.first; ++byte) {
    source.mapped[byte] = static_cast<std::uint8_t>(random());
  }
  const std::vector<std::uint8_t> bytes(source.mapped,
                                        source.mapped + sizes.first);
  std::vector<Etc2GpuUpload> gpu_uploads;
  for (const Upload& upload : uploads) {
    gpu_uploads.push_back(upload.gpu);
  }
  Etc2DecodePass pass;
  ASSERT_TRUE(BuildDecodePass(gpu_uploads, 0, 0, &pass));
  const Buffer input = gpu.CreateBuffer(
      pass.input_bytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      false);
  const Buffer jobs =
      gpu.CreateBuffer(pass.jobs.size() * sizeof(Etc2DecodeJob),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
  std::memcpy(jobs.mapped, pass.jobs.data(),
              pass.jobs.size() * sizeof(Etc2DecodeJob));
  VkDeviceSize readback_bytes = 0;
  for (Image& image : images) {
    image.image = gpu.CreateImage(HostFormat(image.format), image.width,
                                  image.height, image.mips, image.layers);
    for (std::uint32_t mip = 0; mip < image.mips; ++mip) {
      const Upload& upload = uploads[image.uploads[mip]];
      image.readback.push_back(readback_bytes);
      readback_bytes += AlignUp(static_cast<VkDeviceSize>(upload.width) *
                                    upload.height *
                                    EtcDecodedTexelBytes(upload.format) *
                                    upload.layers,
                                16);
    }
  }
  const Buffer readback =
      gpu.CreateBuffer(readback_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  std::memset(readback.mapped, 0xCD, readback_bytes);
  const VkDescriptorSet set0 = gpu.AllocateSet(false, {&input, &jobs});
  const VkDescriptorSet set1 = gpu.AllocateSet(true, {&output});

  const VkCommandBuffer pass_buffer = gpu.CommandBuffer(family);
  gpu.RecordPass(pass_buffer, pass, input, set0, set1);
  vk.vkEndCommandBuffer(pass_buffer);
  // "Roblox": knows nothing about the pass.
  const VkCommandBuffer roblox = gpu.CommandBuffer(family);
  const auto transition = [&](VkImage image, VkImageLayout from,
                              VkImageLayout to, VkAccessFlags source_access,
                              VkAccessFlags destination_access,
                              std::uint32_t mips, std::uint32_t layers) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcAccessMask = source_access;
    barrier.dstAccessMask = destination_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, layers};
    vk.vkCmdPipelineBarrier(roblox, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                            nullptr, 1, &barrier);
  };
  for (const Image& image : images) {
    transition(image.image, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
               VK_ACCESS_TRANSFER_WRITE_BIT, image.mips, image.layers);
    std::vector<VkBufferImageCopy> regions;
    for (std::uint32_t mip = 0; mip < image.mips; ++mip) {
      const Upload& upload = uploads[image.uploads[mip]];
      VkBufferImageCopy region{};
      region.bufferOffset = upload.gpu.output_offset;
      region.bufferRowLength = 4 * upload.gpu.blocks_wide;
      region.bufferImageHeight = 4 * upload.gpu.blocks_high;
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0,
                                 image.layers};
      region.imageExtent = {upload.width, upload.height, 1};
      regions.push_back(region);
    }
    vk.vkCmdCopyBufferToImage(roblox, output.buffer, image.image,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              static_cast<std::uint32_t>(regions.size()),
                              regions.data());
    transition(image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
               image.mips, image.layers);
    std::vector<VkBufferImageCopy> back;
    for (std::uint32_t mip = 0; mip < image.mips; ++mip) {
      const Upload& upload = uploads[image.uploads[mip]];
      VkBufferImageCopy region{};
      region.bufferOffset = image.readback[mip];
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0,
                                 image.layers};
      region.imageExtent = {upload.width, upload.height, 1};
      back.push_back(region);
    }
    vk.vkCmdCopyImageToBuffer(roblox, image.image,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              readback.buffer,
                              static_cast<std::uint32_t>(back.size()),
                              back.data());
  }
  gpu.Barrier(roblox, VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
              VK_ACCESS_HOST_READ_BIT);
  vk.vkEndCommandBuffer(roblox);
  const VkCommandBuffer batch[2] = {pass_buffer, roblox};
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 2;
  submit.pCommandBuffers = batch;
  const VkFence fence = gpu.Fence();
  ASSERT_EQ(vk.vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), VK_SUCCESS);
  ASSERT_EQ(vk.vkQueueSubmit(queue, 0, nullptr, fence), VK_SUCCESS)
      << "the fence-only submit";
  ASSERT_EQ(vk.vkWaitForFences(device->device(), 1, &fence, VK_TRUE,
                               30ULL * 1000000000ULL),
            VK_SUCCESS);
  int failures = 0;
  int checked = 0;
  for (const Image& image : images) {
    for (std::uint32_t mip = 0; mip < image.mips; ++mip) {
      const Upload& upload = uploads[image.uploads[mip]];
      const std::size_t layer_bytes = static_cast<std::size_t>(upload.width) *
                                      upload.height *
                                      EtcDecodedTexelBytes(upload.format);
      for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
        ++checked;
        const std::vector<std::uint8_t> cpu =
            CpuDecodeLayer(upload, bytes.data(), layer);
        if (std::memcmp(readback.mapped + image.readback[mip] +
                            layer * layer_bytes,
                        cpu.data(), layer_bytes) != 0 &&
            failures++ < 5) {
          ADD_FAILURE() << kFormatNames[static_cast<int>(image.format)] << " "
                        << image.width << "x" << image.height << " mip "
                        << mip << " layer " << layer
                        << " differs from the CPU decoder";
        }
      }
    }
  }
  EXPECT_EQ(failures, 0);
  std::printf("family %u: %zu images, %d mip-layers bit-exact through "
              "[decode pass, roblox] + fence-only submit\n",
              family, images.size(), checked - failures);
}

// The control for MOCKTAIL_TEST_VALIDATION=sync: a read-after-write with no
// barrier in between must reach the messenger, or a clean validation run
// would prove nothing.
TEST_F(Etc2GpuShaderTest, SyncValidationSeesAMissingBarrier) {
  if (!device_->sync_validation()) {
    GTEST_SKIP() << "MOCKTAIL_TEST_VALIDATION=sync is not set";
  }
  Gpu gpu(device_);
  const auto& vk = device_->vk();
  const Buffer a = gpu.CreateBuffer(256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    false);
  const Buffer b = gpu.CreateBuffer(256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    false);
  const VkCommandBuffer command_buffer =
      gpu.CommandBuffer(device_->queue_family());
  const VkBufferCopy region{0, 0, 256};
  vk.vkCmdCopyBuffer(command_buffer, a.buffer, b.buffer, 1, &region);
  vk.vkCmdCopyBuffer(command_buffer, b.buffer, a.buffer, 1, &region);
  vk.vkEndCommandBuffer(command_buffer);
  bool hazard = false;
  for (const std::string& message : device_->TakeValidationMessages()) {
    hazard = hazard || message.find("SYNC-HAZARD") != std::string::npos;
  }
  EXPECT_TRUE(hazard) << "synchronization validation reported nothing";
}

TEST_F(Etc2GpuShaderTest, PrependE2EOnTheGraphicsFamily) {
  PrependEndToEnd(device_, device_->queue_family());
}

TEST_F(Etc2GpuShaderTest, PrependE2EOnAComputeOnlyFamily) {
  if (device_->compute_only_family() == UINT32_MAX) {
    GTEST_SKIP() << "the device has no compute-only queue family";
  }
  PrependEndToEnd(device_, device_->compute_only_family());
}

}  // namespace
}  // namespace mocktail::graphics::testing
