#include "mocktail/graphics/chunk_arena.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace mocktail::graphics {
namespace {

using std::chrono::seconds;
using std::chrono::steady_clock;

// Chunks are plain heap memory; the buffer handle is a counter.
class FakeBackend {
 public:
  ChunkArena::Backend backend() {
    ChunkArena::Backend backend;
    backend.create = [this](VkDeviceSize size, bool over_cap,
                            ArenaChunkStorage* out) {
      std::unique_lock<std::mutex> lock(mutex);
      ++create_calls;
      requested_over_cap.push_back(over_cap);
      gate_cv.wait(lock, [this] { return !hold_creates; });
      if (fail_creates) {
        return false;
      }
      out->buffer = reinterpret_cast<VkBuffer>(++next_handle);
      out->memory = reinterpret_cast<VkDeviceMemory>(next_handle);
      out->size = size;
      out->mapped = new std::uint8_t[size];
      if (force_over_cap) {
        out->over_cap = true;
      }
      ++live_chunks;
      return true;
    };
    backend.destroy = [this](const ArenaChunkStorage& storage) {
      std::lock_guard<std::mutex> lock(mutex);
      ++destroy_calls;
      destroyed_sizes.push_back(storage.size);
      delete[] storage.mapped;
      --live_chunks;
    };
    backend.epoch = [this] { return epoch.load(); };
    return backend;
  }

  void Release() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      hold_creates = false;
    }
    gate_cv.notify_all();
  }

  std::mutex mutex;
  std::condition_variable gate_cv;
  bool hold_creates = false;
  bool fail_creates = false;
  bool force_over_cap = false;
  std::uintptr_t next_handle = 0x1000;
  int create_calls = 0;
  int destroy_calls = 0;
  int live_chunks = 0;
  std::vector<bool> requested_over_cap;
  std::vector<VkDeviceSize> destroyed_sizes;
  std::atomic<std::uint64_t> epoch{0};
};

ChunkArena::Config SmallConfig() {
  ChunkArena::Config config;
  config.chunk_bytes = 4096;
  config.alignment = 256;
  config.keep_idle_bytes = 0;
  return config;
}

TEST(ChunkArenaTest, AlignsEveryRangeAndBumpsInsideOneChunk) {
  FakeBackend fake;
  ChunkArena arena(SmallConfig(), fake.backend());
  ArenaRef a;
  ArenaRef b;
  ArenaRef c;
  ASSERT_TRUE(arena.Take(1, &a));
  ASSERT_TRUE(arena.Take(300, &b));
  ASSERT_TRUE(arena.Take(10, &c));
  EXPECT_EQ(a.offset, 0U);
  EXPECT_EQ(b.offset, 256U);
  EXPECT_EQ(c.offset, 768U);
  EXPECT_EQ(b.size, 300U);
  EXPECT_EQ(a.buffer, b.buffer);
  EXPECT_EQ(b.buffer, c.buffer);
  EXPECT_EQ(a.arena_id, arena.id());
  EXPECT_NE(arena.id(), 0U);
  EXPECT_EQ(b.mapped, a.mapped + 256);
  EXPECT_EQ(fake.create_calls, 1);
  EXPECT_EQ(arena.stats().chunks, 1U);
  EXPECT_EQ(arena.stats().bytes, 4096U);
  arena.DestroyAll();
}

TEST(ChunkArenaTest, ReusesAChunkOnlyAfterItsLastRangeIsReleased) {
  FakeBackend fake;
  ChunkArena arena(SmallConfig(), fake.backend());
  ArenaRef first;
  ArenaRef second;
  ASSERT_TRUE(arena.Take(2048, &first));
  ASSERT_TRUE(arena.Take(2048, &second));
  ArenaRef third;
  ASSERT_TRUE(arena.Take(256, &third));
  EXPECT_NE(third.buffer, first.buffer);
  EXPECT_EQ(fake.create_calls, 2);

  EXPECT_TRUE(arena.Release(first));
  ArenaRef fourth;
  ASSERT_TRUE(arena.Take(4000, &fourth));
  EXPECT_NE(fourth.buffer, first.buffer) << "chunk 0 still holds a range";
  EXPECT_EQ(fake.create_calls, 3);

  EXPECT_TRUE(arena.Release(second));
  EXPECT_FALSE(arena.Release(second)) << "a range is released once";
  ArenaRef fifth;
  ASSERT_TRUE(arena.Take(4096, &fifth));
  EXPECT_EQ(fifth.buffer, first.buffer);
  EXPECT_EQ(fifth.offset, 0U);
  EXPECT_EQ(fake.create_calls, 3);
  arena.DestroyAll();
}

TEST(ChunkArenaTest, GivesALargeRequestAChunkOfItsOwn) {
  FakeBackend fake;
  ChunkArena arena(SmallConfig(), fake.backend());
  ArenaRef small;
  ArenaRef large;
  ASSERT_TRUE(arena.Take(100, &small));
  ASSERT_TRUE(arena.Take(10000, &large));
  EXPECT_NE(small.buffer, large.buffer);
  EXPECT_EQ(large.offset, 0U);
  EXPECT_EQ(arena.stats().bytes, 4096U + 10240U);
  EXPECT_EQ(arena.stats().peak_bytes, 4096U + 10240U);
  arena.DestroyAll();
  EXPECT_EQ(fake.live_chunks, 0);
}

TEST(ChunkArenaTest, TakesChunksPastTheCapFromTheFallbackMemory) {
  FakeBackend fake;
  ChunkArena::Config config = SmallConfig();
  config.cap_bytes = 8192;
  ChunkArena arena(config, fake.backend());
  ArenaRef refs[3];
  for (ArenaRef& ref : refs) {
    ASSERT_TRUE(arena.Take(4096, &ref));
  }
  ASSERT_EQ(fake.requested_over_cap.size(), 3U);
  EXPECT_FALSE(fake.requested_over_cap[0]);
  EXPECT_FALSE(fake.requested_over_cap[1]);
  EXPECT_TRUE(fake.requested_over_cap[2]);
  EXPECT_EQ(arena.stats().over_cap_bytes, 4096U);

  // Freeing a chunk under the cap makes room for the next one.
  for (const ArenaRef& ref : refs) {
    arena.Release(ref);
  }
  arena.Trim(steady_clock::now() + seconds(3));
  EXPECT_EQ(arena.stats().chunks, 0U);
  ArenaRef again;
  ASSERT_TRUE(arena.Take(4096, &again));
  EXPECT_FALSE(fake.requested_over_cap.back());
  arena.DestroyAll();
}

TEST(ChunkArenaTest, ArenasOfOneDeviceShareTheCap) {
  FakeBackend fake;
  std::atomic<VkDeviceSize> used{0};
  ChunkArena::Config config = SmallConfig();
  config.cap_bytes = 8192;
  config.shared_cap_used = &used;
  ChunkArena family0(config, fake.backend());
  ChunkArena family2(config, fake.backend());
  ArenaRef a;
  ArenaRef b;
  ArenaRef c;
  ASSERT_TRUE(family0.Take(4096, &a));
  ASSERT_TRUE(family2.Take(4096, &b));
  ASSERT_TRUE(family2.Take(4096, &c));
  EXPECT_EQ(used.load(), 8192U);
  EXPECT_TRUE(fake.requested_over_cap.back());
  EXPECT_EQ(family2.stats().over_cap_bytes, 4096U);
  family0.DestroyAll();
  family2.DestroyAll();
  EXPECT_EQ(used.load(), 0U);
}

TEST(ChunkArenaTest, BackendMayMoveAChunkPastTheCapItself) {
  FakeBackend fake;
  fake.force_over_cap = true;  // say, too little VRAM left for Roblox
  ChunkArena::Config config = SmallConfig();
  config.cap_bytes = 1 << 20;
  ChunkArena arena(config, fake.backend());
  ArenaRef ref;
  ASSERT_TRUE(arena.Take(100, &ref));
  EXPECT_FALSE(fake.requested_over_cap[0]);
  EXPECT_EQ(arena.stats().over_cap_bytes, 4096U);
  arena.DestroyAll();
}

TEST(ChunkArenaTest, TrimsIdleChunksLargestFirstDownToTheKeepSize) {
  FakeBackend fake;
  ChunkArena::Config config = SmallConfig();
  config.keep_idle_bytes = 4096;
  ChunkArena arena(config, fake.backend());
  ArenaRef refs[3];
  ASSERT_TRUE(arena.Take(4096, &refs[0]));
  ASSERT_TRUE(arena.Take(4096, &refs[1]));
  ASSERT_TRUE(arena.Take(12288, &refs[2]));
  for (const ArenaRef& ref : refs) {
    EXPECT_TRUE(arena.Release(ref));
  }
  arena.Trim(steady_clock::now());
  EXPECT_EQ(fake.destroy_calls, 0) << "not idle for two seconds yet";

  arena.Trim(steady_clock::now() + seconds(3));
  ASSERT_EQ(fake.destroyed_sizes.size(), 2U);
  EXPECT_EQ(fake.destroyed_sizes[0], 12288U);
  EXPECT_EQ(fake.destroyed_sizes[1], 4096U);
  EXPECT_EQ(arena.stats().chunks, 1U);
  EXPECT_EQ(arena.stats().bytes, 4096U);
  EXPECT_EQ(arena.stats().peak_bytes, 20480U);
  arena.DestroyAll();
}

TEST(ChunkArenaTest, KeepsChunksWithLiveRanges) {
  FakeBackend fake;
  ChunkArena arena(SmallConfig(), fake.backend());
  ArenaRef held;
  ASSERT_TRUE(arena.Take(64, &held));
  arena.Trim(steady_clock::now() + seconds(60));
  EXPECT_EQ(fake.destroy_calls, 0);
  arena.Release(held);
  arena.Trim(steady_clock::now() + seconds(60));
  EXPECT_EQ(fake.destroy_calls, 1);
}

TEST(ChunkArenaTest, TrimWaitsUntilTheReleaseEpochCompleted) {
  FakeBackend fake;
  ChunkArena arena(SmallConfig(), fake.backend());
  ArenaRef ref;
  ASSERT_TRUE(arena.Take(64, &ref));
  fake.epoch = 5;
  arena.Release(ref);
  fake.epoch = 9;
  arena.Trim(steady_clock::now() + seconds(3), 5);
  EXPECT_EQ(fake.destroy_calls, 0) << "a submit of epoch 5 may still run";
  arena.Trim(steady_clock::now() + seconds(3), 6);
  EXPECT_EQ(fake.destroy_calls, 1);
}

TEST(ChunkArenaTest, DestroyAllFreesEverythingAndForgetsOldRanges) {
  FakeBackend fake;
  ChunkArena arena(SmallConfig(), fake.backend());
  ArenaRef a;
  ArenaRef b;
  ASSERT_TRUE(arena.Take(100, &a));
  ASSERT_TRUE(arena.Take(5000, &b));
  arena.DestroyAll();
  EXPECT_EQ(fake.destroy_calls, 2);
  EXPECT_EQ(fake.live_chunks, 0);
  EXPECT_FALSE(arena.Release(a));
  EXPECT_FALSE(ChunkArena::ReleaseRef(b));
  EXPECT_EQ(arena.stats().chunks, 0U);
  EXPECT_EQ(arena.stats().creates, 2U);

  ArenaRef fresh;
  ASSERT_TRUE(arena.Take(100, &fresh));
  EXPECT_EQ(fake.create_calls, 3);
  arena.DestroyAll();
}

TEST(ChunkArenaTest, ReleaseRefFindsTheArenaByItsId) {
  FakeBackend fake;
  ArenaRef ref;
  std::uint32_t id = 0;
  {
    ChunkArena arena(SmallConfig(), fake.backend());
    id = arena.id();
    ASSERT_TRUE(arena.Take(100, &ref));
    ArenaRef other;
    ASSERT_TRUE(arena.Take(100, &other));
    EXPECT_TRUE(ChunkArena::ReleaseRef(other));
    EXPECT_FALSE(ChunkArena::ReleaseRef(ArenaRef{}));
    ArenaRef foreign = other;
    foreign.arena_id = id + 1000;
    EXPECT_FALSE(ChunkArena::ReleaseRef(foreign));
    arena.DestroyAll();
  }
  EXPECT_EQ(ref.arena_id, id);
  EXPECT_FALSE(ChunkArena::ReleaseRef(ref)) << "the arena is gone";
}

TEST(ChunkArenaTest, DestructorNeverCallsTheBackend) {
  // At process exit the device may already be gone (DESIGN CR-4).
  FakeBackend fake;
  std::vector<std::uint8_t*> leaked;
  {
    ChunkArena arena(SmallConfig(), fake.backend());
    ArenaRef ref;
    ASSERT_TRUE(arena.Take(100, &ref));
    leaked.push_back(ref.mapped);
  }
  EXPECT_EQ(fake.destroy_calls, 0);
  for (std::uint8_t* chunk : leaked) {
    delete[] chunk;
  }
}

TEST(ChunkArenaTest, FailedCreationLeavesNothingBehind) {
  FakeBackend fake;
  fake.fail_creates = true;
  ChunkArena::Config config = SmallConfig();
  config.cap_bytes = 8192;
  std::atomic<VkDeviceSize> used{0};
  config.shared_cap_used = &used;
  ChunkArena arena(config, fake.backend());
  ArenaRef ref;
  EXPECT_FALSE(arena.Take(100, &ref));
  EXPECT_EQ(arena.stats().chunks, 0U);
  EXPECT_EQ(used.load(), 0U);
  fake.fail_creates = false;
  EXPECT_TRUE(arena.Take(100, &ref));
  EXPECT_EQ(used.load(), 4096U);
  arena.DestroyAll();
  EXPECT_EQ(used.load(), 0U);
}

TEST(ChunkArenaTest, ReleaseDoesNotWaitForAChunkBeingCreated) {
  FakeBackend fake;
  ChunkArena arena(SmallConfig(), fake.backend());
  ArenaRef held;
  ASSERT_TRUE(arena.Take(4096, &held));
  fake.hold_creates = true;
  std::atomic<bool> taken{false};
  std::thread creator([&] {
    ArenaRef ref;
    EXPECT_TRUE(arena.Take(100, &ref));
    taken = true;
  });
  while (true) {
    std::lock_guard<std::mutex> lock(fake.mutex);
    if (fake.create_calls == 2) {
      break;
    }
  }
  // The creator sits inside Backend::create; the arena must stay usable.
  EXPECT_TRUE(arena.Release(held));
  EXPECT_EQ(arena.stats().chunks, 1U);
  ArenaRef reuse;
  EXPECT_TRUE(arena.Take(64, &reuse));
  EXPECT_FALSE(taken.load());
  fake.Release();
  creator.join();
  EXPECT_TRUE(taken.load());
  EXPECT_EQ(arena.stats().chunks, 2U);
  arena.DestroyAll();
  EXPECT_EQ(fake.live_chunks, 0);
}

TEST(ChunkArenaTest, DestroyAllDuringACreationFreesTheNewChunk) {
  FakeBackend fake;
  ChunkArena arena(SmallConfig(), fake.backend());
  fake.hold_creates = true;
  std::thread creator([&] {
    ArenaRef ref;
    EXPECT_FALSE(arena.Take(100, &ref));
  });
  while (true) {
    std::lock_guard<std::mutex> lock(fake.mutex);
    if (fake.create_calls == 1) {
      break;
    }
  }
  arena.DestroyAll();
  fake.Release();
  creator.join();
  EXPECT_EQ(fake.live_chunks, 0);
  EXPECT_EQ(arena.stats().chunks, 0U);
}

TEST(ChunkArenaTest, ConcurrentRangesNeverOverlap) {
  FakeBackend fake;
  ChunkArena::Config config;
  config.chunk_bytes = 64 * 1024;
  config.keep_idle_bytes = 128 * 1024;
  config.idle_delay = std::chrono::steady_clock::duration::zero();
  ChunkArena arena(config, fake.backend());
  constexpr int kThreads = 8;
  constexpr int kRounds = 1500;
  std::atomic<int> errors{0};
  std::vector<std::thread> threads;
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      std::mt19937 random(static_cast<unsigned>(thread) * 7919U + 1U);
      std::vector<ArenaRef> held;
      for (int round = 0; round < kRounds; ++round) {
        const VkDeviceSize bytes = 1 + random() % 9000;
        ArenaRef ref;
        if (!arena.Take(bytes, &ref)) {
          ++errors;
          continue;
        }
        if (ref.offset % 256 != 0) {
          ++errors;
        }
        std::memset(ref.mapped, thread + 1, static_cast<std::size_t>(bytes));
        held.push_back(ref);
        if (held.size() > 6 || random() % 3 == 0) {
          const ArenaRef old = held.front();
          held.erase(held.begin());
          for (VkDeviceSize index = 0; index < old.size; ++index) {
            if (old.mapped[index] != thread + 1) {
              ++errors;
              break;
            }
          }
          if (!ChunkArena::ReleaseRef(old)) {
            ++errors;
          }
        }
        if (round % 97 == 0) {
          arena.Trim(steady_clock::now());
        }
      }
      for (const ArenaRef& ref : held) {
        arena.Release(ref);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(errors.load(), 0);
  arena.Trim(steady_clock::now());
  EXPECT_LE(arena.stats().bytes, 128U * 1024U + 9216U);
  arena.DestroyAll();
  EXPECT_EQ(fake.live_chunks, 0);
  EXPECT_EQ(fake.create_calls, fake.destroy_calls);
}

}  // namespace
}  // namespace mocktail::graphics
