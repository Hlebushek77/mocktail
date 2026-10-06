#include "mocktail/graphics/worker_pool.h"

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <new>

namespace mocktail::graphics {

namespace {

constexpr std::size_t kWorkerStackBytes = 256U * 1024U;

// Constant-initialised and trivially destructible: nothing at exit.
std::atomic<WorkerPool*> g_shared{nullptr};
// Test hook: how many pool threads may still be created (UINT_MAX: any).
std::atomic<unsigned> g_thread_creation_limit{UINT_MAX};

struct ThreadStart {
  WorkerPool* pool;
  unsigned index;
  unsigned long long generation;  // the last job posted before it started
};

}  // namespace

WorkerPool::WorkerPool(unsigned workers)
    : workers_(std::max(1U, workers)), pid_(getpid()) {}

WorkerPool::~WorkerPool() { Shutdown(); }

WorkerPool& WorkerPool::Shared(unsigned workers) {
  const pid_t pid = getpid();
  WorkerPool* pool = g_shared.load(std::memory_order_acquire);
  while (pool == nullptr || pool->pid_ != pid) {
    // A pool inherited across fork() stays as it is: its threads do not
    // exist here and its locks may never be released.
    auto* fresh = new WorkerPool(
        workers != 0 ? workers : pool != nullptr ? pool->workers() : 1);
    if (g_shared.compare_exchange_strong(pool, fresh,
                                         std::memory_order_acq_rel)) {
      pool = fresh;
      break;
    }
    delete fresh;  // never started
  }
  if (workers != 0) {
    pool->workers_.store(workers, std::memory_order_relaxed);
  }
  return *pool;
}

void WorkerPool::ShutdownShared() {
  WorkerPool* pool = g_shared.load(std::memory_order_acquire);
  if (pool != nullptr && pool->pid_ == getpid()) {
    pool->Shutdown();
  }
}

void WorkerPool::FailThreadCreationForTesting(bool fail) {
  LimitThreadCreationForTesting(fail ? 0U : UINT_MAX);
}

void WorkerPool::LimitThreadCreationForTesting(unsigned threads) {
  g_thread_creation_limit.store(threads, std::memory_order_relaxed);
}

void WorkerPool::Drain(Job* job) {
  for (std::size_t index =
           job->next.fetch_add(1, std::memory_order_relaxed);
       index < job->count;
       index = job->next.fetch_add(1, std::memory_order_relaxed)) {
    (*job->body)(index);
  }
}

unsigned WorkerPool::ParallelFor(
    std::size_t count, const std::function<void(std::size_t)>& body) {
  if (count == 0) {
    return 0;
  }
  const auto run_inline = [&]() {
    for (std::size_t index = 0; index < count; ++index) {
      body(index);
    }
    return 1U;
  };
  if (count == 1 || pid_ != getpid()) {
    return run_inline();
  }
  std::unique_lock<std::mutex> run(run_mutex_, std::try_to_lock);
  if (!run.owns_lock() || !EnsureStarted()) {
    return run_inline();
  }
  Job job;
  job.count = count;
  job.body = &body;
  std::size_t threads = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    job_ = &job;
    threads = threads_.size();
    pending_ = threads;
    ++generation_;
  }
  wake_.notify_all();
  Drain(&job);
  // Every thread checks in, so none touches `job` after this returns.
  std::unique_lock<std::mutex> lock(mutex_);
  done_.wait(lock, [this] { return pending_ == 0; });
  job_ = nullptr;
  return static_cast<unsigned>(threads + 1);
}

bool WorkerPool::EnsureStarted() {
  const unsigned wanted = workers_.load(std::memory_order_relaxed);
  if (!threads_.empty() && started_for_ != wanted) {
    StopThreads();  // MOCKTAIL_ETC2_WORKERS changed (tests only)
  }
  if (!threads_.empty()) {
    // Possibly fewer than wanted, when the system had no more threads to
    // give: those few keep serving rather than being restarted each call.
    return true;
  }
  if (wanted <= 1) {
    return false;
  }
  pthread_attr_t attributes;
  if (pthread_attr_init(&attributes) != 0) {
    return false;
  }
  pthread_attr_setstacksize(&attributes, kWorkerStackBytes);
  // The threads inherit this mask: every signal blocked.
  sigset_t all;
  sigset_t previous;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &previous);
  unsigned long long generation = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    generation = generation_;
  }
  for (unsigned index = 0; index + 1 < wanted; ++index) {
    unsigned limit = g_thread_creation_limit.load(std::memory_order_relaxed);
    if (limit == 0 ||
        (limit != UINT_MAX &&
         !g_thread_creation_limit.compare_exchange_strong(
             limit, limit - 1, std::memory_order_relaxed))) {
      break;
    }
    auto* start = new (std::nothrow) ThreadStart{this, index, generation};
    pthread_t thread;
    if (start == nullptr ||
        pthread_create(&thread, &attributes, ThreadMain, start) != 0) {
      delete start;
      break;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    threads_.push_back(thread);
    ++thread_starts_;
  }
  pthread_sigmask(SIG_SETMASK, &previous, nullptr);
  pthread_attr_destroy(&attributes);
  started_for_ = wanted;
  return !threads_.empty();
}

void* WorkerPool::ThreadMain(void* argument) {
  const ThreadStart start = *static_cast<ThreadStart*>(argument);
  delete static_cast<ThreadStart*>(argument);
  char name[16];
  std::snprintf(name, sizeof(name), "mt-etc2-%u", start.index + 1);
  pthread_setname_np(pthread_self(), name);
  start.pool->Loop(start.generation);
  return nullptr;
}

void WorkerPool::Loop(unsigned long long seen) {
  for (;;) {
    Job* job = nullptr;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) {
        return;
      }
      seen = generation_;
      job = job_;
    }
    if (job != nullptr) {
      Drain(job);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (--pending_ == 0) {
      done_.notify_one();
    }
  }
}

void WorkerPool::StopThreads() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  for (pthread_t thread : threads_) {  // only run_mutex_'s holder changes it
    pthread_join(thread, nullptr);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  threads_.clear();
  stop_ = false;
}

void WorkerPool::Shutdown() {
  if (pid_ != getpid()) {
    return;
  }
  std::lock_guard<std::mutex> run(run_mutex_);
  StopThreads();
}

std::uint64_t WorkerPool::thread_starts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return thread_starts_;
}

std::size_t WorkerPool::threads() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return threads_.size();
}

}  // namespace mocktail::graphics
