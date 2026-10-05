#include "mocktail/graphics/chunk_arena.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace mocktail::graphics {

namespace {

// Never destroyed: the emulation's arenas live in a function-local static
// that is torn down at exit, possibly after these would be.
std::mutex& RegistryMutex() {
  static auto* mutex = new std::mutex;
  return *mutex;
}

std::unordered_map<std::uint32_t, ChunkArena*>& Registry() {
  static auto* registry = new std::unordered_map<std::uint32_t, ChunkArena*>;
  return *registry;
}

std::uint32_t NextArenaId() {
  static std::atomic<std::uint32_t> next{1};
  std::uint32_t id = next.fetch_add(1, std::memory_order_relaxed);
  if (id == 0) {
    id = next.fetch_add(1, std::memory_order_relaxed);
  }
  return id;
}

VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

ChunkArena::ChunkArena(Config config, Backend backend)
    : id_(NextArenaId()), config_(config), backend_(std::move(backend)) {
  if (config_.alignment == 0 ||
      (config_.alignment & (config_.alignment - 1)) != 0) {
    config_.alignment = 256;
  }
  std::lock_guard<std::mutex> lock(RegistryMutex());
  Registry()[id_] = this;
}

ChunkArena::~ChunkArena() {
  std::lock_guard<std::mutex> lock(RegistryMutex());
  Registry().erase(id_);
}

bool ChunkArena::Hand(std::uint32_t index, VkDeviceSize bytes,
                      VkDeviceSize aligned, ArenaRef* out) {
  Chunk& chunk = chunks_[index];
  out->arena_id = id_;
  out->chunk = index;
  out->buffer = chunk.storage.buffer;
  out->offset = chunk.used;
  out->size = bytes;
  out->mapped = chunk.storage.mapped != nullptr
                    ? chunk.storage.mapped + chunk.used
                    : nullptr;
  chunk.used += aligned;
  ++chunk.live;
  return true;
}

void ChunkArena::Forget(const ArenaChunkStorage& storage) {
  if (!storage.over_cap && config_.cap_bytes != 0) {
    CapUsed().fetch_sub(storage.size, std::memory_order_relaxed);
  }
}

bool ChunkArena::Take(VkDeviceSize bytes, ArenaRef* out) {
  if (out == nullptr || bytes == 0) {
    return false;
  }
  const VkDeviceSize aligned = AlignUp(bytes, config_.alignment);
  if (aligned < bytes) {
    return false;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  for (std::uint32_t index = 0; index < chunks_.size(); ++index) {
    const Chunk& chunk = chunks_[index];
    if (chunk.state == ChunkState::kReady &&
        chunk.storage.size - chunk.used >= aligned) {
      return Hand(index, bytes, aligned, out);
    }
  }
  if (!backend_.create) {
    return false;
  }
  const VkDeviceSize size =
      std::max(AlignUp(config_.chunk_bytes, config_.alignment), aligned);
  bool over_cap = false;
  if (config_.cap_bytes != 0) {
    // Reserve the cap now, so that concurrent creations see it.
    const VkDeviceSize before =
        CapUsed().fetch_add(size, std::memory_order_relaxed);
    if (before + size > config_.cap_bytes || before + size < before) {
      CapUsed().fetch_sub(size, std::memory_order_relaxed);
      over_cap = true;
    }
  }
  std::uint32_t slot = 0;
  while (slot < chunks_.size() && chunks_[slot].state != ChunkState::kFree) {
    ++slot;
  }
  if (slot == chunks_.size()) {
    chunks_.emplace_back();
  }
  chunks_[slot] = Chunk{};
  chunks_[slot].state = ChunkState::kCreating;
  const std::uint64_t generation = generation_;
  lock.unlock();

  ArenaChunkStorage storage;
  storage.size = size;
  storage.over_cap = over_cap;
  const bool made = backend_.create(size, over_cap, &storage);
  const bool created =
      made && storage.buffer != VK_NULL_HANDLE && storage.size >= aligned;
  if (made && !created && backend_.destroy) {
    backend_.destroy(storage);
  }
  if (config_.cap_bytes != 0) {
    // Settle the reservation: a chunk counts against the cap with its real
    // size exactly when it is not over the cap (see Forget).
    const VkDeviceSize reserved = over_cap ? 0 : size;
    const VkDeviceSize counted =
        created && !storage.over_cap ? storage.size : 0;
    if (counted > reserved) {
      CapUsed().fetch_add(counted - reserved, std::memory_order_relaxed);
    } else if (counted < reserved) {
      CapUsed().fetch_sub(reserved - counted, std::memory_order_relaxed);
    }
  }

  lock.lock();
  if (generation != generation_) {
    // DestroyAll ran meanwhile and took the placeholder with it.
    lock.unlock();
    if (created) {
      if (backend_.destroy) {
        backend_.destroy(storage);
      }
      Forget(storage);
    }
    return false;
  }
  Chunk& chunk = chunks_[slot];
  if (!created) {
    chunk.state = ChunkState::kFree;
    return false;
  }
  chunk.state = ChunkState::kReady;
  chunk.storage = storage;
  ++stats_.chunks;
  ++stats_.creates;
  stats_.bytes += storage.size;
  stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.bytes);
  if (storage.over_cap) {
    stats_.over_cap_bytes += storage.size;
  }
  return Hand(slot, bytes, aligned, out);
}

bool ChunkArena::Release(const ArenaRef& ref) {
  if (ref.arena_id != id_) {
    return false;
  }
  const std::uint64_t epoch = backend_.epoch ? backend_.epoch() : 0;
  std::lock_guard<std::mutex> lock(mutex_);
  if (ref.chunk >= chunks_.size()) {
    return false;
  }
  Chunk& chunk = chunks_[ref.chunk];
  if (chunk.state != ChunkState::kReady || chunk.live == 0 ||
      chunk.storage.buffer != ref.buffer) {
    return false;
  }
  if (--chunk.live == 0) {
    chunk.used = 0;
    chunk.idle_since = std::chrono::steady_clock::now();
    chunk.release_epoch = epoch;
  }
  return true;
}

void ChunkArena::Trim(std::chrono::steady_clock::time_point now,
                      std::uint64_t completed_epoch) {
  std::vector<ArenaChunkStorage> doomed;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VkDeviceSize idle_bytes = 0;
    for (const Chunk& chunk : chunks_) {
      if (chunk.state == ChunkState::kReady && chunk.live == 0) {
        idle_bytes += chunk.storage.size;
      }
    }
    while (idle_bytes > config_.keep_idle_bytes) {
      Chunk* largest = nullptr;
      for (Chunk& chunk : chunks_) {
        if (chunk.state == ChunkState::kReady && chunk.live == 0 &&
            now - chunk.idle_since >= config_.idle_delay &&
            chunk.release_epoch < completed_epoch &&
            (largest == nullptr ||
             chunk.storage.size > largest->storage.size)) {
          largest = &chunk;
        }
      }
      if (largest == nullptr) {
        break;
      }
      idle_bytes -= largest->storage.size;
      stats_.bytes -= largest->storage.size;
      if (largest->storage.over_cap) {
        stats_.over_cap_bytes -= largest->storage.size;
      }
      --stats_.chunks;
      doomed.push_back(largest->storage);
      *largest = Chunk{};
    }
  }
  for (const ArenaChunkStorage& storage : doomed) {
    if (backend_.destroy) {
      backend_.destroy(storage);
    }
    Forget(storage);
  }
}

void ChunkArena::DestroyAll() {
  std::vector<ArenaChunkStorage> doomed;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++generation_;
    for (const Chunk& chunk : chunks_) {
      if (chunk.state == ChunkState::kReady) {
        doomed.push_back(chunk.storage);
      }
    }
    chunks_.clear();
    stats_.chunks = 0;
    stats_.bytes = 0;
    stats_.over_cap_bytes = 0;
  }
  for (const ArenaChunkStorage& storage : doomed) {
    if (backend_.destroy) {
      backend_.destroy(storage);
    }
    Forget(storage);
  }
}

ChunkArena::Stats ChunkArena::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

bool ChunkArena::ReleaseRef(const ArenaRef& ref) {
  if (ref.arena_id == 0) {
    return false;
  }
  std::lock_guard<std::mutex> lock(RegistryMutex());
  const auto found = Registry().find(ref.arena_id);
  return found != Registry().end() && found->second->Release(ref);
}

}  // namespace mocktail::graphics
