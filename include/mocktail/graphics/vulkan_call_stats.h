#ifndef MOCKTAIL_GRAPHICS_VULKAN_CALL_STATS_H_
#define MOCKTAIL_GRAPHICS_VULKAN_CALL_STATS_H_

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

// MOCKTAIL_VK_CALL_STATS=1: per-frame Vulkan call counts, for telling the
// render paths Roblox takes apart (the plus.2 DXT probe). Entry points the
// adapter does not hook are counted by thin wrappers that
// vkGetDeviceProcAddr and vkGetInstanceProcAddr hand out instead of the
// loader's pointers; entry points it hooks count inside the hook. Without the
// variable the proc-address hooks return exactly what they returned before.
namespace mocktail::graphics {

enum class VkCallCounter : std::uint8_t {
  // Recording, shown per frame.
  kQueueSubmit,         // hooked: vkQueueSubmit, vkQueueSubmit2(KHR)
  kDraw,                // vkCmdDraw
  kDrawIndexed,         // vkCmdDrawIndexed
  kDrawIndirect,        // vkCmdDraw(Indexed)Indirect(Count)
  kDispatch,            // vkCmdDispatch, Indirect, Base
  kBindPipeline,
  kBindDescriptorSets,
  kPushConstants,
  kBindVertexBuffers,
  kBindIndexBuffer,
  kBeginRenderPass,     // vkCmdBeginRenderPass(2), vkCmdBeginRendering
  kPipelineBarrier,     // vkCmdPipelineBarrier(2)
  kCopyBuffer,          // vkCmdCopyBuffer(2), with bytes
  kCopyBufferToImage,   // hooked: vkCmdCopyBufferToImage(2), with bytes
  kUpdateBuffer,        // vkCmdUpdateBuffer, with bytes
  kClear,               // vkCmdClearColorImage, DepthStencilImage, Attachments
  kUpdateDescriptorSets,  // vkUpdateDescriptorSets, ...WithTemplate
  // Rarer work, shown as window totals.
  kCopyImage,           // hooked: vkCmdCopyImage(2)
  kBlitImage,           // vkCmdBlitImage(2)
  kCopyImageToBuffer,   // vkCmdCopyImageToBuffer(2)
  kResolveImage,        // vkCmdResolveImage(2)
  kAllocateDescriptorSets,
  kResetDescriptorPool,
  kFlushMappedMemory,       // with bytes; VK_WHOLE_SIZE ranges count 0
  kInvalidateMappedMemory,  // with bytes; VK_WHOLE_SIZE ranges count 0
  kMapMemory,               // hooked: vkMapMemory(2)
  kAllocateMemory,          // hooked
  kFreeMemory,              // hooked
  kCreateBuffer,
  kDestroyBuffer,           // hooked
  kCreateImage,             // hooked, also counted per format class
  kDestroyImage,            // hooked
  kAliasView,               // hooked: a view whose format is not its image's
  kCreateGraphicsPipelines,
  kCreateComputePipelines,
  kCreateShaderModule,
  kCreateSampler,
  kCreateFramebuffer,
  kAllocateCommandBuffers,  // hooked
  kBeginCommandBuffer,      // hooked
  kGetFenceStatus,
  kCount,
};
inline constexpr std::size_t kVkCallCounterCount =
    static_cast<std::size_t>(VkCallCounter::kCount);

// Format classes of created images. BC7 is apart from BC4-BC6H because
// Roblox ships a BC7 encoder that may run when ETC2 is missing.
enum class VkImageFormatClass : std::uint8_t {
  kEtc2Eac,
  kBc1To3,
  kBc4To6,
  kBc7,
  kAstc,
  kRgba8,  // R8G8B8A8 and B8G8R8A8
  kOther,
  kCount,
};
inline constexpr std::size_t kVkImageFormatClassCount =
    static_cast<std::size_t>(VkImageFormatClass::kCount);

VkImageFormatClass ClassifyVkImageFormat(VkFormat format);

// Time spent inside calls that block the calling thread.
enum class VkBlockedCall : std::uint8_t {
  kWaitForFences,
  kWaitSemaphores,
  kAcquireNextImage,
  kQueuePresent,
  kQueueSubmit,
  kWaitIdle,  // vkQueueWaitIdle and vkDeviceWaitIdle
  kCount,
};
inline constexpr std::size_t kVkBlockedCallCount =
    static_cast<std::size_t>(VkBlockedCall::kCount);

// The wrapped entry points: X(name, counter, return type, (parameters),
// (arguments), extra statement run before the call). None of them is hooked
// by the adapter (vulkan_call_stats_test keeps the two lists apart); the
// hooked ones count with VkCallStatsCount* instead. KHR aliases have their own
// wrappers, bound to their own loader entry points.
#define MOCKTAIL_VK_CALL_STATS_WRAPPED(X)                                      \
  X(vkCmdDraw, kDraw, void,                                                    \
    (VkCommandBuffer cb, uint32_t vertex_count, uint32_t instance_count,       \
     uint32_t first_vertex, uint32_t first_instance),                          \
    (cb, vertex_count, instance_count, first_vertex, first_instance), )        \
  X(vkCmdDrawIndexed, kDrawIndexed, void,                                      \
    (VkCommandBuffer cb, uint32_t index_count, uint32_t instance_count,        \
     uint32_t first_index, int32_t vertex_offset, uint32_t first_instance),    \
    (cb, index_count, instance_count, first_index, vertex_offset,              \
     first_instance), )                                                        \
  X(vkCmdDrawIndirect, kDrawIndirect, void,                                    \
    (VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,                 \
     uint32_t draw_count, uint32_t stride),                                    \
    (cb, buffer, offset, draw_count, stride), )                                \
  X(vkCmdDrawIndexedIndirect, kDrawIndirect, void,                             \
    (VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,                 \
     uint32_t draw_count, uint32_t stride),                                    \
    (cb, buffer, offset, draw_count, stride), )                                \
  X(vkCmdDrawIndirectCount, kDrawIndirect, void,                               \
    (VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,                 \
     VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t max_draws,     \
     uint32_t stride),                                                         \
    (cb, buffer, offset, count_buffer, count_offset, max_draws, stride), )     \
  X(vkCmdDrawIndirectCountKHR, kDrawIndirect, void,                            \
    (VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,                 \
     VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t max_draws,     \
     uint32_t stride),                                                         \
    (cb, buffer, offset, count_buffer, count_offset, max_draws, stride), )     \
  X(vkCmdDrawIndexedIndirectCount, kDrawIndirect, void,                        \
    (VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,                 \
     VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t max_draws,     \
     uint32_t stride),                                                         \
    (cb, buffer, offset, count_buffer, count_offset, max_draws, stride), )     \
  X(vkCmdDrawIndexedIndirectCountKHR, kDrawIndirect, void,                     \
    (VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,                 \
     VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t max_draws,     \
     uint32_t stride),                                                         \
    (cb, buffer, offset, count_buffer, count_offset, max_draws, stride), )     \
  X(vkCmdDispatch, kDispatch, void,                                            \
    (VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z), (cb, x, y, z),   \
    NotifyVkCallStatsDispatch(cb);)                                            \
  X(vkCmdDispatchIndirect, kDispatch, void,                                    \
    (VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset),                \
    (cb, buffer, offset), NotifyVkCallStatsDispatch(cb);)                      \
  X(vkCmdDispatchBase, kDispatch, void,                                        \
    (VkCommandBuffer cb, uint32_t base_x, uint32_t base_y, uint32_t base_z,    \
     uint32_t x, uint32_t y, uint32_t z),                                      \
    (cb, base_x, base_y, base_z, x, y, z), NotifyVkCallStatsDispatch(cb);)     \
  X(vkCmdDispatchBaseKHR, kDispatch, void,                                     \
    (VkCommandBuffer cb, uint32_t base_x, uint32_t base_y, uint32_t base_z,    \
     uint32_t x, uint32_t y, uint32_t z),                                      \
    (cb, base_x, base_y, base_z, x, y, z), NotifyVkCallStatsDispatch(cb);)     \
  X(vkCmdBindPipeline, kBindPipeline, void,                                    \
    (VkCommandBuffer cb, VkPipelineBindPoint bind_point, VkPipeline pipeline), \
    (cb, bind_point, pipeline), )                                              \
  X(vkCmdBindDescriptorSets, kBindDescriptorSets, void,                        \
    (VkCommandBuffer cb, VkPipelineBindPoint bind_point,                       \
     VkPipelineLayout layout, uint32_t first_set, uint32_t set_count,          \
     const VkDescriptorSet* sets, uint32_t dynamic_offset_count,               \
     const uint32_t* dynamic_offsets),                                         \
    (cb, bind_point, layout, first_set, set_count, sets,                       \
     dynamic_offset_count, dynamic_offsets), )                                 \
  X(vkCmdPushConstants, kPushConstants, void,                                  \
    (VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages,   \
     uint32_t offset, uint32_t size, const void* values),                      \
    (cb, layout, stages, offset, size, values), )                              \
  X(vkCmdBindVertexBuffers, kBindVertexBuffers, void,                          \
    (VkCommandBuffer cb, uint32_t first_binding, uint32_t binding_count,       \
     const VkBuffer* buffers, const VkDeviceSize* offsets),                    \
    (cb, first_binding, binding_count, buffers, offsets), )                    \
  X(vkCmdBindIndexBuffer, kBindIndexBuffer, void,                              \
    (VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset,                 \
     VkIndexType index_type),                                                  \
    (cb, buffer, offset, index_type), )                                        \
  X(vkCmdBeginRenderPass, kBeginRenderPass, void,                              \
    (VkCommandBuffer cb, const VkRenderPassBeginInfo* begin,                   \
     VkSubpassContents contents),                                              \
    (cb, begin, contents), )                                                   \
  X(vkCmdBeginRenderPass2, kBeginRenderPass, void,                             \
    (VkCommandBuffer cb, const VkRenderPassBeginInfo* begin,                   \
     const VkSubpassBeginInfo* subpass),                                       \
    (cb, begin, subpass), )                                                    \
  X(vkCmdBeginRenderPass2KHR, kBeginRenderPass, void,                          \
    (VkCommandBuffer cb, const VkRenderPassBeginInfo* begin,                   \
     const VkSubpassBeginInfo* subpass),                                       \
    (cb, begin, subpass), )                                                    \
  X(vkCmdBeginRendering, kBeginRenderPass, void,                               \
    (VkCommandBuffer cb, const VkRenderingInfo* info), (cb, info), )           \
  X(vkCmdBeginRenderingKHR, kBeginRenderPass, void,                            \
    (VkCommandBuffer cb, const VkRenderingInfo* info), (cb, info), )           \
  X(vkCmdPipelineBarrier, kPipelineBarrier, void,                              \
    (VkCommandBuffer cb, VkPipelineStageFlags src_stages,                      \
     VkPipelineStageFlags dst_stages, VkDependencyFlags dependency_flags,      \
     uint32_t memory_count, const VkMemoryBarrier* memory,                     \
     uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,              \
     uint32_t image_count, const VkImageMemoryBarrier* images),                \
    (cb, src_stages, dst_stages, dependency_flags, memory_count, memory,       \
     buffer_count, buffers, image_count, images), )                            \
  X(vkCmdPipelineBarrier2, kPipelineBarrier, void,                             \
    (VkCommandBuffer cb, const VkDependencyInfo* info), (cb, info), )          \
  X(vkCmdPipelineBarrier2KHR, kPipelineBarrier, void,                          \
    (VkCommandBuffer cb, const VkDependencyInfo* info), (cb, info), )          \
  X(vkCmdCopyBuffer, kCopyBuffer, void,                                        \
    (VkCommandBuffer cb, VkBuffer source, VkBuffer destination,                \
     uint32_t region_count, const VkBufferCopy* regions),                      \
    (cb, source, destination, region_count, regions),                          \
    AddVkCallStatsBytes(VkCallCounter::kCopyBuffer,                            \
                        BufferCopyBytes(region_count, regions));)              \
  X(vkCmdCopyBuffer2, kCopyBuffer, void,                                       \
    (VkCommandBuffer cb, const VkCopyBufferInfo2* info), (cb, info),           \
    AddVkCallStatsBytes(VkCallCounter::kCopyBuffer, BufferCopyBytes(info));)   \
  X(vkCmdCopyBuffer2KHR, kCopyBuffer, void,                                    \
    (VkCommandBuffer cb, const VkCopyBufferInfo2* info), (cb, info),           \
    AddVkCallStatsBytes(VkCallCounter::kCopyBuffer, BufferCopyBytes(info));)   \
  X(vkCmdBlitImage, kBlitImage, void,                                          \
    (VkCommandBuffer cb, VkImage source, VkImageLayout source_layout,          \
     VkImage destination, VkImageLayout destination_layout,                    \
     uint32_t region_count, const VkImageBlit* regions, VkFilter filter),      \
    (cb, source, source_layout, destination, destination_layout,               \
     region_count, regions, filter), )                                         \
  X(vkCmdBlitImage2, kBlitImage, void,                                         \
    (VkCommandBuffer cb, const VkBlitImageInfo2* info), (cb, info), )          \
  X(vkCmdBlitImage2KHR, kBlitImage, void,                                      \
    (VkCommandBuffer cb, const VkBlitImageInfo2* info), (cb, info), )          \
  X(vkCmdCopyImageToBuffer, kCopyImageToBuffer, void,                          \
    (VkCommandBuffer cb, VkImage source, VkImageLayout source_layout,          \
     VkBuffer destination, uint32_t region_count,                              \
     const VkBufferImageCopy* regions),                                        \
    (cb, source, source_layout, destination, region_count, regions), )         \
  X(vkCmdCopyImageToBuffer2, kCopyImageToBuffer, void,                         \
    (VkCommandBuffer cb, const VkCopyImageToBufferInfo2* info), (cb, info), )  \
  X(vkCmdCopyImageToBuffer2KHR, kCopyImageToBuffer, void,                      \
    (VkCommandBuffer cb, const VkCopyImageToBufferInfo2* info), (cb, info), )  \
  X(vkCmdUpdateBuffer, kUpdateBuffer, void,                                    \
    (VkCommandBuffer cb, VkBuffer destination, VkDeviceSize offset,            \
     VkDeviceSize size, const void* data),                                     \
    (cb, destination, offset, size, data),                                     \
    AddVkCallStatsBytes(VkCallCounter::kUpdateBuffer, size);)                  \
  X(vkCmdClearColorImage, kClear, void,                                        \
    (VkCommandBuffer cb, VkImage image, VkImageLayout layout,                  \
     const VkClearColorValue* color, uint32_t range_count,                     \
     const VkImageSubresourceRange* ranges),                                   \
    (cb, image, layout, color, range_count, ranges), )                         \
  X(vkCmdClearDepthStencilImage, kClear, void,                                 \
    (VkCommandBuffer cb, VkImage image, VkImageLayout layout,                  \
     const VkClearDepthStencilValue* value, uint32_t range_count,              \
     const VkImageSubresourceRange* ranges),                                   \
    (cb, image, layout, value, range_count, ranges), )                         \
  X(vkCmdClearAttachments, kClear, void,                                       \
    (VkCommandBuffer cb, uint32_t attachment_count,                            \
     const VkClearAttachment* attachments, uint32_t rect_count,                \
     const VkClearRect* rects),                                                \
    (cb, attachment_count, attachments, rect_count, rects), )                  \
  X(vkCmdResolveImage, kResolveImage, void,                                    \
    (VkCommandBuffer cb, VkImage source, VkImageLayout source_layout,          \
     VkImage destination, VkImageLayout destination_layout,                    \
     uint32_t region_count, const VkImageResolve* regions),                    \
    (cb, source, source_layout, destination, destination_layout,               \
     region_count, regions), )                                                 \
  X(vkCmdResolveImage2, kResolveImage, void,                                   \
    (VkCommandBuffer cb, const VkResolveImageInfo2* info), (cb, info), )       \
  X(vkCmdResolveImage2KHR, kResolveImage, void,                                \
    (VkCommandBuffer cb, const VkResolveImageInfo2* info), (cb, info), )       \
  X(vkUpdateDescriptorSets, kUpdateDescriptorSets, void,                       \
    (VkDevice device, uint32_t write_count,                                    \
     const VkWriteDescriptorSet* writes, uint32_t copy_count,                  \
     const VkCopyDescriptorSet* copies),                                       \
    (device, write_count, writes, copy_count, copies), )                       \
  X(vkUpdateDescriptorSetWithTemplate, kUpdateDescriptorSets, void,            \
    (VkDevice device, VkDescriptorSet set,                                     \
     VkDescriptorUpdateTemplate update_template, const void* data),            \
    (device, set, update_template, data), )                                    \
  X(vkUpdateDescriptorSetWithTemplateKHR, kUpdateDescriptorSets, void,         \
    (VkDevice device, VkDescriptorSet set,                                     \
     VkDescriptorUpdateTemplate update_template, const void* data),            \
    (device, set, update_template, data), )                                    \
  X(vkAllocateDescriptorSets, kAllocateDescriptorSets, VkResult,               \
    (VkDevice device, const VkDescriptorSetAllocateInfo* info,                 \
     VkDescriptorSet* sets),                                                   \
    (device, info, sets), )                                                    \
  X(vkResetDescriptorPool, kResetDescriptorPool, VkResult,                     \
    (VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags flags), \
    (device, pool, flags), )                                                   \
  X(vkFlushMappedMemoryRanges, kFlushMappedMemory, VkResult,                   \
    (VkDevice device, uint32_t range_count, const VkMappedMemoryRange* ranges), \
    (device, range_count, ranges),                                             \
    AddVkCallStatsBytes(VkCallCounter::kFlushMappedMemory,                     \
                        MappedRangeBytes(range_count, ranges));)               \
  X(vkInvalidateMappedMemoryRanges, kInvalidateMappedMemory, VkResult,         \
    (VkDevice device, uint32_t range_count, const VkMappedMemoryRange* ranges), \
    (device, range_count, ranges),                                             \
    AddVkCallStatsBytes(VkCallCounter::kInvalidateMappedMemory,                \
                        MappedRangeBytes(range_count, ranges));)               \
  X(vkCreateBuffer, kCreateBuffer, VkResult,                                   \
    (VkDevice device, const VkBufferCreateInfo* info,                          \
     const VkAllocationCallbacks* allocator, VkBuffer* buffer),                \
    (device, info, allocator, buffer), )                                       \
  X(vkCreateGraphicsPipelines, kCreateGraphicsPipelines, VkResult,             \
    (VkDevice device, VkPipelineCache cache, uint32_t count,                   \
     const VkGraphicsPipelineCreateInfo* infos,                                \
     const VkAllocationCallbacks* allocator, VkPipeline* pipelines),           \
    (device, cache, count, infos, allocator, pipelines), )                     \
  X(vkCreateComputePipelines, kCreateComputePipelines, VkResult,               \
    (VkDevice device, VkPipelineCache cache, uint32_t count,                   \
     const VkComputePipelineCreateInfo* infos,                                 \
     const VkAllocationCallbacks* allocator, VkPipeline* pipelines),           \
    (device, cache, count, infos, allocator, pipelines), )                     \
  X(vkCreateShaderModule, kCreateShaderModule, VkResult,                       \
    (VkDevice device, const VkShaderModuleCreateInfo* info,                    \
     const VkAllocationCallbacks* allocator, VkShaderModule* module),          \
    (device, info, allocator, module), )                                       \
  X(vkCreateSampler, kCreateSampler, VkResult,                                 \
    (VkDevice device, const VkSamplerCreateInfo* info,                         \
     const VkAllocationCallbacks* allocator, VkSampler* sampler),              \
    (device, info, allocator, sampler), )                                      \
  X(vkCreateFramebuffer, kCreateFramebuffer, VkResult,                         \
    (VkDevice device, const VkFramebufferCreateInfo* info,                     \
     const VkAllocationCallbacks* allocator, VkFramebuffer* framebuffer),      \
    (device, info, allocator, framebuffer), )                                  \
  X(vkGetFenceStatus, kGetFenceStatus, VkResult,                               \
    (VkDevice device, VkFence fence), (device, fence), )

// MOCKTAIL_VK_CALL_STATS is set and not "0", read once.
bool VkCallStatsFlagEnabled(const char* value);
bool VkCallStatsEnabled();

// Whether `name` has a counting wrapper.
bool IsVkCallStatsProc(const char* name);

// What vkGetDeviceProcAddr or vkGetInstanceProcAddr return for `name`, an
// entry point the adapter does not hook. `host` is what they returned before
// call stats existed; `trampoline` is the loader's instance-level pointer for
// `name` (valid for every device, looked up only when IsVkCallStatsProc).
// Off, unknown, unsupported (host null) or without a trampoline, the answer is
// `host` itself; otherwise the wrapper, bound to `trampoline`. A wrapper stays
// bound to the first trampoline; a different one later gets `host`.
PFN_vkVoidFunction SelectVkCallStatsProc(bool enabled, const char* name,
                                         PFN_vkVoidFunction host,
                                         PFN_vkVoidFunction trampoline);

struct VkCallStatsProcInfo {
  const char* name;
  VkCallCounter counter;
  PFN_vkVoidFunction wrapper;
};
// The wrapper table, in X-macro order.
const VkCallStatsProcInfo* VkCallStatsProcs(std::size_t* count);

// Called by the dispatch wrappers, for the emulation's
// "uploads recorded after a dispatch" counter.
using VkCallStatsDispatchObserver = void (*)(VkCommandBuffer command_buffer);
void SetVkCallStatsDispatchObserver(VkCallStatsDispatchObserver observer);
void NotifyVkCallStatsDispatch(VkCommandBuffer command_buffer);

// Counting from the adapter's own hooks. All of these do nothing unless
// VkCallStatsEnabled().
void VkCallStatsCount(VkCallCounter counter, std::uint64_t calls = 1);
void AddVkCallStatsBytes(VkCallCounter counter, std::uint64_t bytes);
void VkCallStatsCountCreateImage(VkFormat format);
void AddVkCallStatsBlocked(VkBlockedCall call, std::uint64_t nanoseconds);

std::uint64_t BufferCopyBytes(std::uint32_t region_count,
                              const VkBufferCopy* regions);
std::uint64_t BufferCopyBytes(const VkCopyBufferInfo2* info);
std::uint64_t MappedRangeBytes(std::uint32_t range_count,
                               const VkMappedMemoryRange* ranges);

// One 2-second window of counts.
struct VkCallStatsWindow {
  std::uint64_t frames = 0;
  std::uint64_t span_ns = 0;
  std::array<std::uint64_t, kVkCallCounterCount> calls{};
  std::array<std::uint64_t, kVkCallCounterCount> bytes{};
  std::array<std::uint64_t, kVkImageFormatClassCount> created_images{};
  std::array<std::uint64_t, kVkBlockedCallCount> blocked_ns{};
};

// Moves every count accumulated so far into *window (frames and span_ns are
// left alone) and starts again from zero.
void TakeVkCallStatsCounts(VkCallStatsWindow* window);

// Counts one present at now_ns. The first present starts the window; once it
// is 2 s old, *window receives it (calls made before the first present
// included) and true is returned.
bool AdvanceVkCallStatsFrame(std::uint64_t now_ns, VkCallStatsWindow* window);

// The [perf] vk_calls line for a window, without leading spaces or newline.
std::string FormatVkCallStatsLine(const VkCallStatsWindow& window);

// vkQueuePresentKHR's hook: prints the line every 2 s when enabled.
void VkCallStatsOnPresent();

// Clears counts, the frame window, wrapper bindings, the observer and a
// forced state, which tests set to count without the variable.
void ResetVkCallStatsForTesting();
void ForceVkCallStatsEnabledForTesting(bool enabled);

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_VULKAN_CALL_STATS_H_
