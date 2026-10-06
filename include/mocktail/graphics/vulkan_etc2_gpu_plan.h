#ifndef MOCKTAIL_GRAPHICS_VULKAN_ETC2_GPU_PLAN_H_
#define MOCKTAIL_GRAPHICS_VULKAN_ETC2_GPU_PLAN_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

#include "mocktail/graphics/vulkan_etc2_emulation.h"
#include "mocktail/graphics/vulkan_etc2_planning.h"

// Pure planning for the GPU route's decode passes (DESIGN C3): the jobs the
// decode shader reads, the copies that bring Roblox's compressed bytes into
// a buffer the shader can read, and the rewritten submit arrays that put a
// decode pass in front of Roblox's batches. No Vulkan calls, no state.
// PlanGpuRegions, the record-time half, is declared in
// vulkan_etc2_planning.h and defined with these.

namespace mocktail::graphics {

// etc2_decode.comp's and etc2_bc.comp's `Job` (std430, 64 bytes): one
// upload, all its layers.
struct Etc2DecodeJob {
  std::uint32_t format = 0;           // EtcFormat
  std::uint32_t src_word = 0;         // first block, in words of the input
  std::uint32_t src_row_words = 0;    // words per block row
  std::uint32_t src_layer_words = 0;  // words per layer
  std::uint32_t blocks_wide = 0;
  std::uint32_t blocks_high = 0;
  std::uint32_t block_count = 0;      // blocks_wide * blocks_high * layers
  std::uint32_t dst_word = 0;         // first output word in its buffer
  std::uint32_t dst_row_words = 0;    // words per texel row (4 x blocks),
                                      // or per block row for BC
  std::uint32_t dst_layer_words = 0;
  std::uint32_t group_begin = 0;      // first workgroup, per dispatch
  std::uint32_t flags = 0;            // etc2_bc.comp: kEtc2BcJobRefine
  std::uint32_t pad[4] = {};
};
static_assert(sizeof(Etc2DecodeJob) == 64, "the shader's Job is 64 bytes");

// etc2_bc.comp's job flag: refine BC1 endpoints (EncodeBc1Block's `refine`).
inline constexpr std::uint32_t kEtc2BcJobRefine = 1;

// The shader's local_size_x: one invocation per 4x4 block.
inline constexpr std::uint32_t kEtc2DecodeGroupSize = 64;
// Dispatches split X here, the portable minimum of
// maxComputeWorkGroupCount[0] and [1].
inline constexpr std::uint32_t kEtc2DispatchMaxGroups = 65535;
// Uploads whose output exceeds min(maxStorageBufferRange, this) take the
// CPU route.
inline constexpr VkDeviceSize kEtc2GpuMaxOutputBytes = 1ULL << 30;
// Where each upload's compressed bytes start in the input buffer.
inline constexpr VkDeviceSize kEtc2GpuInputAlignment = 16;

// vkCmdCopyBuffer regions from one of Roblox's buffers into the input.
struct Etc2InputCopies {
  VkBuffer source = VK_NULL_HANDLE;
  std::vector<VkBufferCopy> regions;
};

// One vkCmdDispatch over jobs [job_first, job_first + job_count) of the jobs
// buffer, all of which write into `output` (bound as set 1). `bc`: the jobs
// transcode to BC blocks (the etc2_bc pipeline) rather than decode.
struct Etc2DecodeDispatch {
  VkBuffer output = VK_NULL_HANDLE;
  bool bc = false;
  std::uint32_t job_first = 0;
  std::uint32_t job_count = 0;
  std::uint32_t groups_x = 0;
  std::uint32_t groups_y = 0;
};

// The decode pass of one batch.
struct Etc2DecodePass {
  std::vector<Etc2InputCopies> copies;  // one entry per source buffer
  std::vector<Etc2DecodeJob> jobs;      // the jobs buffer from job_base on
  std::vector<Etc2DecodeDispatch> dispatches;
  VkDeviceSize input_bytes = 0;  // used from input_base on
  std::uint64_t blocks = 0;
  VkDeviceSize compressed = 0;
  VkDeviceSize decoded = 0;
};

// The input bytes BuildDecodePass uses for `uploads`, so that a slot can
// size its input buffer for every batch of a call before recording any.
VkDeviceSize Etc2DecodeInputBytes(const std::vector<Etc2GpuUpload>& uploads);

// Plans the decode pass of one batch's GPU uploads, whose `output` and
// `output_offset` are final: copies of each source span into the input
// buffer from `input_base` (16-byte aligned) on, merged where both sides are
// contiguous, or block row by block row when the span is more than twice
// the compressed bytes; one job per upload; dispatches grouped by pipeline
// (decode first, then the BC uploads of ETC2->BC mode, whose jobs refine as
// the CPU route does) and output buffer, with group_begin counted per
// dispatch and X split at 65535. False when a value does not fit the
// shader's 32-bit words.
bool BuildDecodePass(const std::vector<Etc2GpuUpload>& uploads,
                     VkDeviceSize input_base, std::uint32_t job_base,
                     Etc2DecodePass* pass);

// The workgroups for `blocks` blocks, and the 2-D split of `groups`.
std::uint32_t Etc2DecodeGroups(std::uint64_t blocks);
void Etc2SplitDispatch(std::uint32_t groups, std::uint32_t* x,
                       std::uint32_t* y);

// A batch that gets a decode pass in front of it: its index in pSubmits and
// the pass's command buffer.
struct Etc2BatchDecode {
  std::uint32_t batch = 0;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
};

// True for a batch the pass cannot join: a protected batch (its command
// buffers must all be protected, VUID-VkSubmitInfo-pNext-04148) or a device
// group batch (VkDeviceGroupSubmitInfo counts the command buffers,
// VUID-VkDeviceGroupSubmitInfo-commandBufferCount-00083).
bool Etc2BatchNeedsPreSubmit(const VkSubmitInfo& submit);
bool Etc2BatchNeedsPreSubmit2(const VkSubmitInfo2& submit);

// Rewrites one submit call into `plan`. Each batch in `decodes` (ascending
// batch order) gets its pass's command buffer prepended (index 0, so a
// render-pass suspend/resume pair is never split), or, for a batch
// Etc2BatchNeedsPreSubmit accepts, a separate pre-submit of the pass alone
// (no waits, signals or fence) that the caller submits first. Every other
// batch, its waits, signals and pNext, stay as they were. Without an app
// fence the slot fence is substituted; with one, plan->trailing_fence is
// the slot fence, which FinishSubmit signals with a fence-only submit. Returns
// the number of pre-submits made.
std::uint32_t RewriteSubmits(std::uint32_t count, const VkSubmitInfo* submits,
                             const std::vector<Etc2BatchDecode>& decodes,
                             VkFence app_fence, VkFence slot_fence,
                             Etc2SubmitPlan* plan);
std::uint32_t RewriteSubmits2(std::uint32_t count,
                              const VkSubmitInfo2* submits,
                              const std::vector<Etc2BatchDecode>& decodes,
                              VkFence app_fence, VkFence slot_fence,
                              Etc2SubmitPlan* plan);

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_VULKAN_ETC2_GPU_PLAN_H_
