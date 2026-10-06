// Golden hashes of the ETC2/EAC emulation scenarios in
// tests/support/etc2_emulation_scenarios.cc: FNV-1a 64 over every host image
// (format, extent, every subresource's texels) at each checkpoint.
//
// Frozen from the plus.1 upload path (plus2/main 2249b6c, before any plus.2
// change to it) on the fake device; the same scenarios on a real GPU must
// give the same hashes. Later packages must keep every line unchanged: a
// different hash means different texels in Roblox's textures.
{"FullMipChains", 0xb47e3552cd3d2fd7ULL},
{"PartialUploadsWithOffsets", 0xf5fa83366fb6e65cULL},
{"RowStridesAndOffsets", 0xad9a24f44dcdef9aULL},
{"ArrayLayers", 0x1a299fb5219581e6ULL},
{"UpscaledMipChains", 0x13a5936b16ceaad0ULL},
{"UpscaledPartialUpdates", 0xd44b348571bc94abULL},
{"UpscaleSelection", 0x1b284701f7f43388ULL},
{"UpscaleFactorTwo", 0xfdf3a9d446150254ULL},
{"CopyBufferToImage2", 0xa7842a14219cde66ULL},
{"SecondaryCommandBuffers", 0xc0789db7591872d0ULL},
{"ResubmitAfterSourceChange", 0x72b08400c2d3ce01ULL},
{"MultiCommandBufferSubmits", 0xc8f464a77b3cf0b9ULL},
{"UnmappedSources", 0xa6317cd26eb8d98bULL},
{"CopyImageUnscaled", 0x431ef95f1b62aff1ULL},
{"CopyImageScaled", 0x0fd667ead317a848ULL},
{"CopyImageScaledToUnscaled", 0xceac477873f5a455ULL},
{"RejectedRegions", 0x679a7fc6eb1e747eULL},
{"NotEmulatedDevice", 0x77dccff1e29868c9ULL},
{"TransferOnlyQueueFamily", 0x4995331e4d312511ULL},
{"TextureOverridesAndDumps", 0xd804d9c02baf52caULL},
