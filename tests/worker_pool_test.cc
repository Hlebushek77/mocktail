#include "mocktail/graphics/worker_pool.h"

#include <gtest/gtest.h>

#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace mocktail::graphics {
namespace {

// Every index of [0, count) ran exactly once.
void ExpectEachOnce(const std::vector<std::atomic<int>>& hits, int times = 1) {
  for (std::size_t index = 0; index < hits.size(); ++index) {
    ASSERT_EQ(hits[index].load(), times) << "index " << index;
  }
}

TEST(WorkerPoolTest, RunsEveryIndexExactlyOnce) {
  WorkerPool pool(8);
  for (std::size_t count : {0U, 1U, 2U, 7U, 8U, 9U, 1000U, 100000U}) {
    std::vector<std::atomic<int>> hits(count);
    const unsigned threads = pool.ParallelFor(
        count, [&](std::size_t index) { hits[index].fetch_add(1); });
    ExpectEachOnce(hits);
    EXPECT_EQ(threads, count == 0 ? 0U : count == 1 ? 1U : 8U) << count;
  }
  EXPECT_EQ(pool.threads(), 7U);
}

TEST(WorkerPoolTest, SpreadsWorkOverNamedThreadsWithSignalsBlocked) {
  WorkerPool pool(4);
  std::mutex mutex;
  std::set<std::string> names;
  std::set<std::thread::id> ids;
  bool unblocked_signal = false;
  std::atomic<int> arrived{0};
  pool.ParallelFor(4, [&](std::size_t) {
    // Hold every task until all four threads have one, so each runs one.
    arrived.fetch_add(1);
    while (arrived.load() < 4) {
      std::this_thread::yield();
    }
    char name[16] = "";
    pthread_getname_np(pthread_self(), name, sizeof(name));
    sigset_t mask;
    pthread_sigmask(SIG_SETMASK, nullptr, &mask);
    std::lock_guard<std::mutex> lock(mutex);
    ids.insert(std::this_thread::get_id());
    if (std::strncmp(name, "mt-etc2-", 8) == 0) {
      names.insert(name);
      unblocked_signal |= sigismember(&mask, SIGINT) != 1 ||
                          sigismember(&mask, SIGSEGV) != 1 ||
                          sigismember(&mask, SIGUSR1) != 1;
    }
  });
  EXPECT_EQ(ids.size(), 4U);
  EXPECT_EQ(names, (std::set<std::string>{"mt-etc2-1", "mt-etc2-2",
                                          "mt-etc2-3"}));
  EXPECT_FALSE(unblocked_signal) << "pool threads must block every signal";
  // The caller's own mask is untouched.
  sigset_t mask;
  pthread_sigmask(SIG_SETMASK, nullptr, &mask);
  EXPECT_NE(sigismember(&mask, SIGINT), 1);
}

TEST(WorkerPoolTest, TwoCallersAtOnceBothFinish) {
  WorkerPool pool(4);
  std::vector<std::atomic<int>> hits(200000);
  std::atomic<int> inline_runs{0};
  const auto run = [&]() {
    for (int round = 0; round < 20; ++round) {
      const unsigned threads = pool.ParallelFor(
          hits.size() / 20, [&, round](std::size_t index) {
            hits[round * (hits.size() / 20) + index].fetch_add(1);
          });
      if (threads == 1) {
        inline_runs.fetch_add(1);
      }
    }
  };
  std::thread other(run);
  run();
  other.join();
  ExpectEachOnce(hits, 2);
}

TEST(WorkerPoolTest, RunsInlineWhenNoThreadCanBeCreated) {
  WorkerPool::FailThreadCreationForTesting(true);
  WorkerPool pool(6);
  std::vector<std::atomic<int>> hits(1000);
  std::set<std::thread::id> ids;
  std::mutex mutex;
  EXPECT_EQ(pool.ParallelFor(hits.size(),
                             [&](std::size_t index) {
                               hits[index].fetch_add(1);
                               std::lock_guard<std::mutex> lock(mutex);
                               ids.insert(std::this_thread::get_id());
                             }),
            1U);
  WorkerPool::FailThreadCreationForTesting(false);
  ExpectEachOnce(hits);
  EXPECT_EQ(ids, std::set<std::thread::id>{std::this_thread::get_id()});
  EXPECT_EQ(pool.threads(), 0U);
  // A single worker never starts a thread either.
  WorkerPool single(1);
  EXPECT_EQ(single.ParallelFor(50, [](std::size_t) {}), 1U);
  EXPECT_EQ(single.threads(), 0U);
}

// When the system gives fewer threads than asked for, those few serve every
// call: the pool does not join and recreate them each time.
TEST(WorkerPoolTest, KeepsTheThreadsItCouldCreate) {
  WorkerPool::LimitThreadCreationForTesting(2);
  WorkerPool pool(6);
  std::vector<std::atomic<int>> hits(3000);
  for (int round = 0; round < 3; ++round) {
    EXPECT_EQ(pool.ParallelFor(hits.size(),
                               [&](std::size_t index) {
                                 hits[index].fetch_add(1);
                               }),
              3U)
        << "round " << round;
  }
  WorkerPool::LimitThreadCreationForTesting(UINT_MAX);
  ExpectEachOnce(hits, 3);
  EXPECT_EQ(pool.threads(), 2U);
  EXPECT_EQ(pool.thread_starts(), 2U) << "no restart per call";
}

TEST(WorkerPoolTest, ShutdownJoinsAndTheNextCallRestarts) {
  WorkerPool pool(3);
  std::vector<std::atomic<int>> hits(5000);
  pool.ParallelFor(hits.size(),
                   [&](std::size_t index) { hits[index].fetch_add(1); });
  EXPECT_EQ(pool.threads(), 2U);
  pool.Shutdown();
  EXPECT_EQ(pool.threads(), 0U);
  pool.Shutdown();  // twice is harmless
  EXPECT_EQ(pool.ParallelFor(hits.size(),
                             [&](std::size_t index) {
                               hits[index].fetch_add(1);
                             }),
            3U);
  EXPECT_EQ(pool.threads(), 2U);
  ExpectEachOnce(hits, 2);
}

TEST(WorkerPoolTest, SharedPoolFollowsTheRequestedWorkers) {
  WorkerPool& pool = WorkerPool::Shared(3);
  EXPECT_EQ(&WorkerPool::Shared(0), &pool) << "one pool per process";
  EXPECT_EQ(pool.workers(), 3U);
  EXPECT_EQ(pool.ParallelFor(64, [](std::size_t) {}), 3U);
  EXPECT_EQ(WorkerPool::Shared(5).ParallelFor(64, [](std::size_t) {}), 5U);
  EXPECT_EQ(pool.threads(), 4U);
  WorkerPool::ShutdownShared();
  EXPECT_EQ(pool.threads(), 0U);
  EXPECT_EQ(WorkerPool::Shared(0).ParallelFor(64, [](std::size_t) {}), 5U);
  WorkerPool::ShutdownShared();
}

// fork() while the shared pool has threads: the child must neither use the
// parent's pool (its threads do not exist there) nor take its locks.
TEST(WorkerPoolTest, ForkedChildStartsAPoolOfItsOwn) {
  WorkerPool& parent_pool = WorkerPool::Shared(4);
  ASSERT_EQ(parent_pool.ParallelFor(64, [](std::size_t) {}), 4U);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    int status = 0;
    WorkerPool& pool = WorkerPool::Shared(0);
    if (&pool == &parent_pool) {
      status |= 1;
    }
    std::vector<std::atomic<int>> hits(10000);
    if (pool.ParallelFor(hits.size(), [&](std::size_t index) {
          hits[index].fetch_add(1);
        }) != 4) {
      status |= 2;
    }
    for (const auto& hit : hits) {
      if (hit.load() != 1) {
        status |= 4;
        break;
      }
    }
    // The inherited pool runs inline and never waits for its threads.
    if (parent_pool.ParallelFor(100, [](std::size_t) {}) != 1) {
      status |= 8;
    }
    parent_pool.Shutdown();  // a no-op in the child
    WorkerPool::ShutdownShared();
    _exit(status);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "the child hung or crashed";
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "1: same pool, 2: no threads, 4: indices, 8: inherited pool waited";
  // The parent's pool still works.
  EXPECT_EQ(WorkerPool::Shared(0).ParallelFor(64, [](std::size_t) {}), 4U);
  WorkerPool::ShutdownShared();
}

}  // namespace
}  // namespace mocktail::graphics
