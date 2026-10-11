#include "mocktail/graphics/vulkan_etc2_gpu_plan.h"

#include <algorithm>
#include <limits>

#include "mocktail/graphics/bc_encoder.h"

namespace mocktail::graphics {

namespace {

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

bool FitsWord(VkDeviceSize value) {
  return value <= std::numeric_limits<std::uint32_t>::max();
}

// A span more than twice the compressed bytes (Roblox's rows padded far
// apart, or a few rows of a big layer stride) is copied row by row.
bool CopiesRows(const Etc2GpuUpload& upload) {
  return upload.source_span > 2 * upload.compressed;
}

VkDeviceSize PackedRow(const Etc2GpuUpload& upload) {
  return static_cast<VkDeviceSize>(upload.blocks_wide) *
         EtcBlockBytes(upload.format);
}

VkDeviceSize InputBytes(const Etc2GpuUpload& upload) {
  return CopiesRows(upload) ? upload.compressed : upload.source_span;
}

template <typename Info>
const VkBaseInStructure* FindNext(const Info& info, VkStructureType type) {
  for (auto* next = static_cast<const VkBaseInStructure*>(info.pNext);
       next != nullptr; next = next->pNext) {
    if (next->sType == type) {
      return next;
    }
  }
  return nullptr;
}

}  // namespace

template <typename Region>
bool PlanGpuRegions(const Etc2ImageShape& image, VkBuffer source,
                    std::uint32_t count, const Region* regions, bool bc,
                    std::vector<Region>* rewritten,
                    std::vector<Etc2GpuUpload>* uploads, VkDeviceSize* total) {
  *total = 0;
  if (regions == nullptr || count == 0 || image.scale > 1) {
    return false;
  }
  const EtcFormat format = image.format.etc_format;
  const VkDeviceSize block_bytes = EtcBlockBytes(format);
  const VkDeviceSize texel_bytes = EtcDecodedTexelBytes(format);
  rewritten->reserve(rewritten->size() + count);
  uploads->reserve(uploads->size() + count);
  for (std::uint32_t index = 0; index < count; ++index) {
    const Region& region = regions[index];
    Etc2GpuUpload upload;
    if (!Etc2UploadByteCounts(format, region.imageExtent,
                              region.imageSubresource.layerCount,
                              &upload.compressed, &upload.decoded)) {
      return false;
    }
    const std::uint32_t mip = region.imageSubresource.mipLevel;
    const std::uint32_t mip_width =
        mip < 32 ? std::max<std::uint32_t>(1, image.width >> mip) : 1;
    const std::uint32_t mip_height =
        mip < 32 ? std::max<std::uint32_t>(1, image.height >> mip) : 1;
    if (region.imageOffset.x < 0 || region.imageOffset.y < 0 ||
        region.imageOffset.z != 0 ||
        static_cast<std::uint32_t>(region.imageOffset.x) >= mip_width ||
        static_cast<std::uint32_t>(region.imageOffset.y) >= mip_height) {
      return false;
    }
    const std::uint32_t width = region.imageExtent.width;
    const std::uint32_t height = region.imageExtent.height;
    const VkDeviceSize row_texels =
        region.bufferRowLength != 0 ? region.bufferRowLength : width;
    const VkDeviceSize layer_rows =
        region.bufferImageHeight != 0 ? region.bufferImageHeight : height;
    if (row_texels < width || layer_rows < height) {
      return false;
    }
    upload.source = source;
    upload.source_offset = region.bufferOffset;
    upload.format = format;
    upload.blocks_wide = (width + 3) / 4;
    upload.blocks_high = (height + 3) / 4;
    upload.layers = region.imageSubresource.layerCount;
    upload.source_row_stride = ((row_texels + 3) / 4) * block_bytes;
    upload.source_layer_stride =
        upload.source_row_stride * ((layer_rows + 3) / 4);
    upload.source_span =
        (upload.layers - 1) * upload.source_layer_stride +
        (static_cast<VkDeviceSize>(upload.blocks_high) - 1) *
            upload.source_row_stride +
        static_cast<VkDeviceSize>(upload.blocks_wide) * block_bytes;
    const VkDeviceSize blocks = static_cast<VkDeviceSize>(upload.blocks_wide) *
                                upload.blocks_high * upload.layers;
    upload.output_offset = AlignUp(*total, kEtc2GpuRegionAlignment);
    upload.output_bytes = bc ? blocks * block_bytes : blocks * 16 * texel_bytes;
    upload.mip_level = mip;
    upload.bc = bc;
    Region copy = region;
    copy.bufferOffset = upload.output_offset;
    // Whole blocks: 4 x blocks texels per row and rows per layer. In BC
    // mode that is the tight layout too.
    copy.bufferRowLength = 4 * upload.blocks_wide;
    copy.bufferImageHeight = 4 * upload.blocks_high;
    copy.imageExtent.width = std::min(
        width, mip_width - static_cast<std::uint32_t>(region.imageOffset.x));
    copy.imageExtent.height = std::min(
        height, mip_height - static_cast<std::uint32_t>(region.imageOffset.y));
    // BC mode writes the blocks themselves, as the CPU route counts them.
    upload.decoded = bc ? upload.compressed
                        : static_cast<VkDeviceSize>(copy.imageExtent.width) *
                              copy.imageExtent.height * texel_bytes *
                              upload.layers;
    rewritten->push_back(copy);
    uploads->push_back(upload);
    *total = upload.output_offset + upload.output_bytes;
  }
  return true;
}

template bool PlanGpuRegions<VkBufferImageCopy>(
    const Etc2ImageShape&, VkBuffer, std::uint32_t, const VkBufferImageCopy*,
    bool, std::vector<VkBufferImageCopy>*, std::vector<Etc2GpuUpload>*,
    VkDeviceSize*);
template bool PlanGpuRegions<VkBufferImageCopy2>(
    const Etc2ImageShape&, VkBuffer, std::uint32_t, const VkBufferImageCopy2*,
    bool, std::vector<VkBufferImageCopy2>*, std::vector<Etc2GpuUpload>*,
    VkDeviceSize*);

VkDeviceSize Etc2DecodeInputBytes(const std::vector<Etc2GpuUpload>& uploads) {
  VkDeviceSize bytes = 0;
  for (const Etc2GpuUpload& upload : uploads) {
    bytes = AlignUp(bytes, kEtc2GpuInputAlignment) + InputBytes(upload);
  }
  return AlignUp(bytes, kEtc2GpuInputAlignment);
}

std::uint32_t Etc2DecodeGroups(std::uint64_t blocks) {
  return static_cast<std::uint32_t>((blocks + kEtc2DecodeGroupSize - 1) /
                                    kEtc2DecodeGroupSize);
}

void Etc2SplitDispatch(std::uint32_t groups, std::uint32_t* x,
                       std::uint32_t* y) {
  *x = std::min(groups, kEtc2DispatchMaxGroups);
  *y = *x == 0 ? 0 : (groups + *x - 1) / *x;
}

bool BuildDecodePass(const std::vector<Etc2GpuUpload>& uploads,
                     VkDeviceSize input_base, std::uint32_t job_base,
                     Etc2DecodePass* pass) {
  *pass = Etc2DecodePass{};
  input_base = AlignUp(input_base, kEtc2GpuInputAlignment);
  // Jobs grouped by pipeline (decode, then BC) and output buffer, so each
  // group is one dispatch; the order inside a group is the recording order.
  std::vector<std::size_t> order(uploads.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t left, std::size_t right) {
                     if (uploads[left].bc != uploads[right].bc) {
                       return uploads[right].bc;
                     }
                     return uploads[left].output < uploads[right].output;
                   });
  // The input layout follows the recording order, so neighbouring source
  // spans can share one copy region.
  std::vector<VkDeviceSize> input_offsets(uploads.size());
  VkDeviceSize cursor = 0;
  for (std::size_t index = 0; index < uploads.size(); ++index) {
    const Etc2GpuUpload& upload = uploads[index];
    if (upload.layers == 0 || upload.blocks_wide == 0 ||
        upload.blocks_high == 0 || upload.output == VK_NULL_HANDLE) {
      return false;
    }
    cursor = AlignUp(cursor, kEtc2GpuInputAlignment);
    input_offsets[index] = input_base + cursor;
    const VkDeviceSize destination = input_offsets[index];
    Etc2InputCopies* copies = nullptr;
    for (Etc2InputCopies& candidate : pass->copies) {
      if (candidate.source == upload.source) {
        copies = &candidate;
        break;
      }
    }
    if (copies == nullptr) {
      pass->copies.push_back({upload.source, {}});
      copies = &pass->copies.back();
    }
    const auto add = [copies](VkDeviceSize source, VkDeviceSize target,
                              VkDeviceSize size) {
      if (!copies->regions.empty()) {
        VkBufferCopy& last = copies->regions.back();
        if (last.srcOffset + last.size == source &&
            last.dstOffset + last.size == target) {
          last.size += size;
          return;
        }
      }
      copies->regions.push_back({source, target, size});
    };
    if (CopiesRows(upload)) {
      const VkDeviceSize packed = PackedRow(upload);
      for (std::uint32_t layer = 0; layer < upload.layers; ++layer) {
        for (std::uint32_t row = 0; row < upload.blocks_high; ++row) {
          add(upload.source_offset + layer * upload.source_layer_stride +
                  row * upload.source_row_stride,
              destination +
                  (static_cast<VkDeviceSize>(layer) * upload.blocks_high +
                   row) *
                      packed,
              packed);
        }
      }
    } else {
      add(upload.source_offset, destination, upload.source_span);
    }
    cursor += InputBytes(upload);
  }
  pass->input_bytes = AlignUp(cursor, kEtc2GpuInputAlignment);

  Etc2DecodeDispatch* dispatch = nullptr;
  std::uint64_t groups = 0;
  const auto close = [&]() {
    if (dispatch != nullptr) {
      Etc2SplitDispatch(static_cast<std::uint32_t>(groups),
                        &dispatch->groups_x, &dispatch->groups_y);
    }
  };
  for (const std::size_t index : order) {
    const Etc2GpuUpload& upload = uploads[index];
    if (dispatch == nullptr || dispatch->output != upload.output ||
        dispatch->bc != upload.bc) {
      close();
      pass->dispatches.push_back({});
      dispatch = &pass->dispatches.back();
      dispatch->output = upload.output;
      dispatch->bc = upload.bc;
      dispatch->job_first =
          job_base + static_cast<std::uint32_t>(pass->jobs.size());
      groups = 0;
    }
    const VkDeviceSize texel_bytes = EtcDecodedTexelBytes(upload.format);
    const bool rows = CopiesRows(upload);
    const VkDeviceSize row_stride =
        rows ? PackedRow(upload) : upload.source_row_stride;
    const VkDeviceSize layer_stride =
        rows ? PackedRow(upload) * upload.blocks_high
             : upload.source_layer_stride;
    const std::uint64_t blocks = static_cast<std::uint64_t>(
                                     upload.blocks_wide) *
                                 upload.blocks_high * upload.layers;
    const VkDeviceSize dst_row =
        upload.bc ? static_cast<VkDeviceSize>(upload.blocks_wide) *
                        EtcBlockBytes(upload.format)
                  : static_cast<VkDeviceSize>(upload.blocks_wide) * 4 *
                        texel_bytes;
    const VkDeviceSize dst_layer =
        upload.bc ? dst_row * upload.blocks_high : dst_row * 4 * upload.blocks_high;
    if (input_offsets[index] % 4 != 0 || upload.output_offset % 4 != 0 ||
        row_stride % 4 != 0 || layer_stride % 4 != 0 ||
        !FitsWord(input_offsets[index] / 4) ||
        !FitsWord(upload.output_offset / 4) || !FitsWord(row_stride / 4) ||
        !FitsWord(layer_stride / 4) || !FitsWord(blocks) ||
        !FitsWord(dst_layer / 4) ||
        groups + Etc2DecodeGroups(blocks) >
            static_cast<std::uint64_t>(kEtc2DispatchMaxGroups) *
                kEtc2DispatchMaxGroups) {
      return false;
    }
    Etc2DecodeJob job;
    job.format = static_cast<std::uint32_t>(upload.format);
    job.src_word = static_cast<std::uint32_t>(input_offsets[index] / 4);
    job.src_row_words = static_cast<std::uint32_t>(row_stride / 4);
    job.src_layer_words = static_cast<std::uint32_t>(layer_stride / 4);
    job.blocks_wide = upload.blocks_wide;
    job.blocks_high = upload.blocks_high;
    job.block_count = static_cast<std::uint32_t>(blocks);
    job.dst_word = static_cast<std::uint32_t>(upload.output_offset / 4);
    job.dst_row_words = static_cast<std::uint32_t>(dst_row / 4);
    job.dst_layer_words = static_cast<std::uint32_t>(dst_layer / 4);
    job.group_begin = static_cast<std::uint32_t>(groups);
    job.flags = upload.bc && kEtc2ToBcRefine ? kEtc2BcJobRefine : 0;
    groups += Etc2DecodeGroups(blocks);
    pass->jobs.push_back(job);
    ++dispatch->job_count;
    pass->blocks += blocks;
    pass->compressed += upload.compressed;
    pass->decoded += upload.decoded;
  }
  close();
  return true;
}

bool Etc2BatchNeedsPreSubmit(const VkSubmitInfo& submit) {
  if (FindNext(submit, VK_STRUCTURE_TYPE_DEVICE_GROUP_SUBMIT_INFO) !=
      nullptr) {
    return true;
  }
  const auto* protected_info = reinterpret_cast<const VkProtectedSubmitInfo*>(
      FindNext(submit, VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO));
  return protected_info != nullptr &&
         protected_info->protectedSubmit == VK_TRUE;
}

bool Etc2BatchNeedsPreSubmit2(const VkSubmitInfo2& submit) {
  return (submit.flags & VK_SUBMIT_PROTECTED_BIT) != 0;
}

namespace {

void SetFences(VkFence app_fence, VkFence slot_fence, Etc2SubmitPlan* plan) {
  if (app_fence == VK_NULL_HANDLE) {
    plan->substitute_fence = slot_fence;
    plan->trailing_fence = VK_NULL_HANDLE;
  } else {
    plan->substitute_fence = VK_NULL_HANDLE;
    plan->trailing_fence = slot_fence;
  }
}

// A pre-submit of the pass alone; `slot` points into plan->command_buffers.
void AddPreSubmit(const VkCommandBuffer* slot, std::uint64_t uploads,
                  Etc2SubmitPlan* plan) {
  Etc2PreSubmit pre;
  pre.info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  pre.info.commandBufferCount = 1;
  pre.info.pCommandBuffers = slot;
  pre.uploads = uploads;
  plan->pre_submit_batches.push_back(pre);
}

}  // namespace

std::uint32_t RewriteSubmits(std::uint32_t count, const VkSubmitInfo* submits,
                             const std::vector<Etc2BatchDecode>& decodes,
                             VkFence app_fence, VkFence slot_fence,
                             Etc2SubmitPlan* plan) {
  // Size the command-buffer array exactly first: the batches point into it.
  std::size_t handles = 0;
  for (const Etc2BatchDecode& decode : decodes) {
    if (submits == nullptr || decode.batch >= count) {
      continue;
    }
    handles += 1 + (Etc2BatchNeedsPreSubmit(submits[decode.batch])
                        ? 0
                        : submits[decode.batch].commandBufferCount);
  }
  if (handles == 0) {
    return 0;
  }
  plan->command_buffers.clear();
  plan->command_buffers.reserve(handles);
  plan->submit_infos.assign(submits, submits + count);
  plan->submit_infos2.clear();
  std::uint32_t pre_submits = 0;
  bool prepended = false;
  for (const Etc2BatchDecode& decode : decodes) {
    if (decode.batch >= count) {
      continue;
    }
    const VkSubmitInfo& batch = submits[decode.batch];
    const std::size_t at = plan->command_buffers.size();
    plan->command_buffers.push_back(decode.command_buffer);
    if (Etc2BatchNeedsPreSubmit(batch)) {
      AddPreSubmit(&plan->command_buffers[at], decode.uploads, plan);
      ++pre_submits;
      continue;
    }
    for (std::uint32_t index = 0; index < batch.commandBufferCount; ++index) {
      plan->command_buffers.push_back(batch.pCommandBuffers[index]);
    }
    VkSubmitInfo& rewritten = plan->submit_infos[decode.batch];
    rewritten.commandBufferCount = batch.commandBufferCount + 1;
    rewritten.pCommandBuffers = &plan->command_buffers[at];
    prepended = true;
  }
  plan->modified = prepended;
  if (!prepended) {
    plan->submit_infos.clear();
  }
  SetFences(app_fence, slot_fence, plan);
  return pre_submits;
}

std::uint32_t RewriteSubmits2(std::uint32_t count,
                              const VkSubmitInfo2* submits,
                              const std::vector<Etc2BatchDecode>& decodes,
                              VkFence app_fence, VkFence slot_fence,
                              Etc2SubmitPlan* plan) {
  std::size_t handles = 0;
  std::size_t infos = 0;
  for (const Etc2BatchDecode& decode : decodes) {
    if (submits == nullptr || decode.batch >= count) {
      continue;
    }
    if (Etc2BatchNeedsPreSubmit2(submits[decode.batch])) {
      ++handles;
    } else {
      infos += 1 + submits[decode.batch].commandBufferInfoCount;
    }
  }
  if (handles == 0 && infos == 0) {
    return 0;
  }
  plan->command_buffers.clear();
  plan->command_buffers.reserve(handles);
  plan->command_buffer_infos.clear();
  plan->command_buffer_infos.reserve(infos);
  plan->submit_infos2.assign(submits, submits + count);
  plan->submit_infos.clear();
  std::uint32_t pre_submits = 0;
  bool prepended = false;
  for (const Etc2BatchDecode& decode : decodes) {
    if (decode.batch >= count) {
      continue;
    }
    const VkSubmitInfo2& batch = submits[decode.batch];
    if (Etc2BatchNeedsPreSubmit2(batch)) {
      plan->command_buffers.push_back(decode.command_buffer);
      AddPreSubmit(&plan->command_buffers.back(), decode.uploads, plan);
      ++pre_submits;
      continue;
    }
    const std::size_t at = plan->command_buffer_infos.size();
    VkCommandBufferSubmitInfo pass{};
    pass.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    pass.commandBuffer = decode.command_buffer;
    plan->command_buffer_infos.push_back(pass);
    for (std::uint32_t index = 0; index < batch.commandBufferInfoCount;
         ++index) {
      plan->command_buffer_infos.push_back(batch.pCommandBufferInfos[index]);
    }
    VkSubmitInfo2& rewritten = plan->submit_infos2[decode.batch];
    rewritten.commandBufferInfoCount = batch.commandBufferInfoCount + 1;
    rewritten.pCommandBufferInfos = &plan->command_buffer_infos[at];
    prepended = true;
  }
  plan->modified = prepended;
  if (!prepended) {
    plan->submit_infos2.clear();
  }
  SetFences(app_fence, slot_fence, plan);
  return pre_submits;
}

}  // namespace mocktail::graphics
