#ifndef MOCKTAIL_GRAPHICS_WORKER_POOL_H_
#define MOCKTAIL_GRAPHICS_WORKER_POOL_H_

#include <sys/types.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

#include <pthread.h>

namespace mocktail::graphics {

// Persistent threads for the ETC2 CPU route's copy-in, decode and emit, so a
// submit wakes sleeping threads (about 45 us) instead of creating and joining
// seven per call (about 230 us). The threads run pure CPU work only: no
// Vulkan, bionic or Roblox calls.
//
// The shared pool is created with `new` on first use and never deleted: the
// adapter's state is a function-local static torn down at exit, which must
// never join or destroy a sleeping worker. Its threads are named
// "mt-etc2-N", start with every signal blocked (a signal Roblox's bionic
// handlers expect never lands on them) and have 256 KiB stacks.
//
// After fork() the child has none of the parent's threads, and the pool's
// locks may be held forever: Shared() and ParallelFor() compare the process
// id before they take any lock, and the child leaves the inherited pool
// untouched and starts a pool of its own. No pthread_atfork handler is
// installed.
class WorkerPool {
 public:
  // `workers` threads take part in a ParallelFor: the caller and
  // `workers - 1` pool threads.
  explicit WorkerPool(unsigned workers);
  ~WorkerPool();  // joins the threads (not for the shared pool: immortal)
  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  // The process's pool. `workers` (0: leave as is) applies from the next
  // ParallelFor on; in production it never changes after the first call.
  static WorkerPool& Shared(unsigned workers);
  // Joins the shared pool's threads, when this process has started any; the
  // pool starts them again at its next ParallelFor. For the last
  // vkDestroyDevice of an emulated device, outside every emulation lock.
  static void ShutdownShared();

  // Runs body(index) exactly once for every index in [0, count), on the
  // pool's threads and the calling thread, and returns once all ran. Runs
  // everything on the caller when count is 1, when another ParallelFor is
  // running (one at a time), in a forked child that inherited the pool, or
  // when no thread can be created. Returns how many threads took part (1
  // when inline).
  unsigned ParallelFor(std::size_t count,
                       const std::function<void(std::size_t)>& body);

  // Joins the threads; the next ParallelFor starts them again.
  void Shutdown();

  unsigned workers() const { return workers_.load(std::memory_order_relaxed); }
  // Pool threads running now, and created so far (tests).
  std::size_t threads() const;
  std::uint64_t thread_starts() const;

  // Test hooks: thread creation fails as if the system had none left, or
  // after `threads` more (UINT_MAX: no limit).
  static void FailThreadCreationForTesting(bool fail);
  static void LimitThreadCreationForTesting(unsigned threads);

 private:
  struct Job {
    std::size_t count = 0;
    std::atomic<std::size_t> next{0};
    const std::function<void(std::size_t)>* body = nullptr;
  };

  static void Drain(Job* job);
  static void* ThreadMain(void* argument);
  // Under run_mutex_.
  bool EnsureStarted();
  void StopThreads();
  void Loop(unsigned long long seen);

  std::atomic<unsigned> workers_;
  const pid_t pid_;
  std::mutex run_mutex_;  // one ParallelFor (or Shutdown) at a time
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable done_;
  std::vector<pthread_t> threads_;
  unsigned started_for_ = 0;  // workers_ when threads_ was started
  std::uint64_t thread_starts_ = 0;
  Job* job_ = nullptr;
  unsigned long long generation_ = 0;
  std::size_t pending_ = 0;
  bool stop_ = false;
};

}  // namespace mocktail::graphics

#endif  // MOCKTAIL_GRAPHICS_WORKER_POOL_H_
