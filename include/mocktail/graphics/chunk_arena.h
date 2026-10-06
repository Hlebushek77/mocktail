#ifndef MOCKTAIL_GRAPHICS_CHUNK_ARENA_H_
#define MOCKTAIL_GRAPHICS_CHUNK_ARENA_H_

#include <vulkan/vulkan.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace mocktail::graphics {

// One Vulkan buffer and its memory. `mapped` stays valid for the chunk's
// whole life when the memory is host-visible, and is null otherwise.
struct ArenaChunkStorage {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  std::uint8_t* mapped = nullptr;
  VkDeviceSize size = 0;
  // Set by the backend when the chunk came from the fallback memory and so
  // does not count against Config::cap_bytes.
  bool over_cap = false;
};

// A range handed out by ChunkArena::Take. A command buffer's record keeps
// only this: `arena_id` finds the arena again (ChunkArena::ReleaseRef).
struct ArenaRef {
  std::uint32_t arena_id = 0;  // 0: no range
  std::uint32_t chunk = 0;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  VkDeviceSize size = 0;           // the requested bytes
  std::uint8_t* mapped = nullptr;  // chunk mapping + offset, or null
};

// Sub-allocates ranges of a few large buffers, so that recording an upload
// costs no Vulkan call once the chunks exist (creating a 64 MiB chunk costs
// about 17 ms). Each chunk is a bump allocator that starts over when its last
// range is released.
//
// Thread-safe. The backend runs without the arena's lock held, so Release
// from the render thread never waits behind a chunk being created or freed.
// The destructor never calls the backend: only DestroyAll gives chunks back,
// because at process exit the Vulkan device may already be gone.
class ChunkArena {
 public:
  struct Config {
    VkDeviceSize chunk_bytes = 64ULL << 20;
    VkDeviceSize keep_idle_bytes = 0;  // Trim keeps this much of idle chunks
    VkDeviceSize alignment = 256;      // offsets and sizes, a power of two
    VkDeviceSize cap_bytes = 0;        // 0: no cap
    // How long a chunk must have been idle before Trim frees it.
    std::chrono::steady_clock::duration idle_delay = std::chrono::seconds(2);
    // Arenas of one device can share the cap through one counter (the GPU
    // route's arenas, one per queue family). Null: the arena counts alone.
    std::atomic<VkDeviceSize>* shared_cap_used = nullptr;
  };
  struct Backend {
    // Creates a chunk of at least `size` bytes. `over_cap` asks for the
    // fallback memory; the backend may also pick it on its own and says so
    // in ArenaChunkStorage::over_cap.
    std::function<bool(VkDeviceSize size, bool over_cap, ArenaChunkStorage*)>
        create;
    std::function<void(const ArenaChunkStorage&)> destroy;
    // Optional: the owner's submit epoch, stored with a chunk when its last
    // range is released; Trim frees it only once that epoch completed.
    std::function<std::uint64_t()> epoch;
  };
  struct Stats {
    std::uint32_t chunks = 0;
    VkDeviceSize bytes = 0;
    VkDeviceSize peak_bytes = 0;
    VkDeviceSize over_cap_bytes = 0;
    std::uint64_t creates = 0;
  };

  ChunkArena(Config config, Backend backend);
  ~ChunkArena();
  ChunkArena(const ChunkArena&) = delete;
  ChunkArena& operator=(const ChunkArena&) = delete;

  std::uint32_t id() const { return id_; }

  // Bumps a range in the first chunk with room; when none has room, creates
  // a chunk of max(chunk_bytes, bytes), so a large request gets one of its
  // own. False when the backend cannot create one.
  bool Take(VkDeviceSize bytes, ArenaRef* out);
  // Drops one live range; a chunk whose last range goes is reused from its
  // start. False for a range this arena does not hold (already released, or
  // handed out before DestroyAll).
  bool Release(const ArenaRef& ref);
  // Frees chunks that have been idle for idle_delay, largest first, until the
  // idle chunks add up to keep_idle_bytes or less. A chunk released at epoch
  // E is freed only when completed_epoch > E.
  void Trim(std::chrono::steady_clock::time_point now,
            std::uint64_t completed_epoch = UINT64_MAX);
  // Frees every chunk. Ranges handed out before stay invalid; a creation in
  // flight frees its chunk itself.
  void DestroyAll();
  Stats stats() const;

  // Releases `ref` into the arena that handed it out. A missing arena (never
  // existed, or destroyed) is a no-op that returns false.
  static bool ReleaseRef(const ArenaRef& ref);

 private:
  enum class ChunkState { kFree, kCreating, kReady };
  struct Chunk {
    ChunkState state = ChunkState::kFree;
    ArenaChunkStorage storage;
    VkDeviceSize used = 0;
    std::uint32_t live = 0;
    std::chrono::steady_clock::time_point idle_since{};
    std::uint64_t release_epoch = 0;
  };

  bool Hand(std::uint32_t index, VkDeviceSize bytes, VkDeviceSize aligned,
            ArenaRef* out);
  std::atomic<VkDeviceSize>& CapUsed() {
    return config_.shared_cap_used != nullptr ? *config_.shared_cap_used
                                              : own_cap_used_;
  }
  void Forget(const ArenaChunkStorage& storage);

  const std::uint32_t id_;
  Config config_;
  Backend backend_;
  std::atomic<VkDeviceSize> own_cap_used_{0};
  mutable std::mutex mutex_;
  std::vector<Chunk> chunks_;
  std::uint64_t generation_ = 0;
  Stats stats_;
};

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_CHUNK_ARENA_H_
