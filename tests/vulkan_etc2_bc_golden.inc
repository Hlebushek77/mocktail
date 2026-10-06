// Golden hashes of the ETC2/EAC emulation scenarios in ETC2->BC mode
// (MOCKTAIL_ETC2_TO_BC=1) on the CPU route: FNV-1a 64 over every host image
// (BC format, extent, every subresource's blocks) at each checkpoint, as in
// vulkan_etc2_emulation_golden.inc.
//
// Frozen on the fake device (plus.2 WP3a) after every scenario matched the
// reference model: blocks from the plus.1 decoder and the frozen
// plus2-design/harness/bc_ref.h encoder, which the GPU route's etc2_bc
// shader matches bit for bit. The GPU route in BC mode must give the same
// hashes. Scenarios with bc_mode_fake_errors have no line.
{"FullMipChains", 0xf9bd56db8591a0a2ULL},
{"PartialUploadsWithOffsets", 0x0a58ce5cb7518533ULL},
{"RowStridesAndOffsets", 0x3609b1eb3f5616e5ULL},
{"ArrayLayers", 0x05e2a6bc74908680ULL},
{"UpscaledMipChains", 0xd4bbc49371537b9bULL},
{"UpscaledPartialUpdates", 0x0bfd52609ce7335aULL},
{"UpscaleSelection", 0x80cefde3241edf6aULL},
{"UpscaleFactorTwo", 0x06ea9e5c93e5f0f0ULL},
{"CopyBufferToImage2", 0x95822bc686026b54ULL},
{"SecondaryCommandBuffers", 0x4226dba56f157774ULL},
{"ResubmitAfterSourceChange", 0x5f06340425bcf2f9ULL},
{"MultiCommandBufferSubmits", 0x1612bdf62729b354ULL},
{"UnmappedSources", 0x7b8c5cf33d871932ULL},
{"RejectedRegions", 0x58014f3e42683a9aULL},
{"NotEmulatedDevice", 0x77dccff1e29868c9ULL},
{"TransferOnlyQueueFamily", 0xd4b675784d8ca400ULL},
{"TextureOverridesAndDumps", 0x8f984479de8fa8ddULL},
{"CopyImageMipMoves", 0xff8ead29fb5605baULL},
