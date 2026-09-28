#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <latch>
#include <numeric>
#include <print>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

using namespace std;

constexpr size_t kRepeats = 9;
const size_t kThreadCount = max(2u, thread::hardware_concurrency());
using Timings = array<double, kRepeats>;

double averageTime(Timings timings) {
  ranges::sort(timings);
  return accumulate(timings.begin() + 2, timings.end() - 2, 0.0) /
         (kRepeats - 4);
}

double averageFastTime(Timings timings) {
  ranges::sort(timings);
  return (timings[1] + timings[2] + timings[3]) / 3;
}

double timeForGivenPossibleCacheLineSize(size_t possibleCacheLineSize) {
  struct alignas(4096) PairData {
    size_t values[1024]{};
  };
  vector<PairData> data((kThreadCount + 1) / 2);
  latch ready(kThreadCount), startSignal(1), finished(kThreadCount);
  vector<thread> threads;
  threads.reserve(kThreadCount);
  constexpr size_t kIterations = 1 << 25;

  for (size_t t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([&, t] {
      volatile size_t &value =
          data[t / 2].values[(t % 2) * possibleCacheLineSize / sizeof(size_t)];
      for (size_t i = 0; i < 4096; ++i)
        value = value + 1;
      value = 0;
      ready.count_down();
      startSignal.wait();
      for (size_t i = 0; i < kIterations; ++i)
        value = value + 1;
      finished.count_down();
    });
  }
  ready.wait();
  const auto start = chrono::steady_clock::now();
  startSignal.count_down();
  finished.wait();
  const auto end = chrono::steady_clock::now();
  for (auto &worker : threads)
    worker.join();
  for (size_t t = 0; t < kThreadCount; ++t) {
    assert(
        data[t / 2].values[(t % 2) * possibleCacheLineSize / sizeof(size_t)] ==
        kIterations);
  }
  return chrono::duration<double>(end - start).count();
}

size_t calculateLineSize(bool debug) {
  vector<size_t> offsets;
  for (size_t offset = sizeof(size_t); offset <= 1024; offset *= 2)
    offsets.push_back(offset);

  vector<Timings> timings(offsets.size());
  vector<size_t> order(offsets.size());
  iota(order.begin(), order.end(), 0);

  mt19937 rng(12345);
  for (size_t repeat = 0; repeat < kRepeats; ++repeat) {
    ranges::shuffle(order, rng);
    for (size_t i : order)
      timings[i][repeat] = timeForGivenPossibleCacheLineSize(offsets[i]);
  }
  vector<double> times;
  for (size_t i = 0; i < offsets.size(); ++i) {
    times.push_back(averageTime(timings[i]));
    if (debug)
      println("Offset = {} (in bytes), time = {}", offsets[i], times.back());
  }
  size_t lineSize = 0;
  double greatestDecrease = 0;
  for (size_t i = 1; i < offsets.size(); ++i) {
    const double decrease = times[i - 1] - times[i];
    if (decrease <= greatestDecrease || decrease <= times[i - 1] * 0.05)
      continue;
    if (i + 1 < times.size() && times[i - 1] - times[i + 1] < decrease / 2)
      continue;
    size_t fasterRepeats = 0;
    for (size_t r = 0; r < kRepeats; ++r)
      fasterRepeats += timings[i][r] < timings[i - 1][r];
    if (fasterRepeats >= 6) {
      greatestDecrease = decrease;
      lineSize = offsets[i];
    }
  }
  return lineSize;
}

double timeForPointerCycle(size_t *data, vector<size_t> &positions,
                           unsigned seed) {
  mt19937 rng(seed);
  ranges::shuffle(positions, rng);

  for (size_t i = 0; i < positions.size(); ++i) {
    data[positions[i]] = positions[(i + 1) % positions.size()];
  }

  const volatile size_t *links = data;
  size_t index = positions[0];
  const size_t warmup = max(size_t(8192), positions.size() * 32);
  for (size_t i = 0; i < warmup; ++i)
    index = links[index];
  constexpr size_t kIterations = 1 << 21;

  const auto start = chrono::steady_clock::now();
  atomic_signal_fence(memory_order_seq_cst);
  for (size_t i = 0; i < kIterations; ++i)
    index = links[index];
  atomic_signal_fence(memory_order_seq_cst);
  const auto end = chrono::steady_clock::now();

  assert(index == positions[(warmup + kIterations) % positions.size()]);
  return chrono::duration<double, nano>(end - start).count() / kIterations;
}

double timeForL1WorkingSet(size_t bytes, size_t lineSize, unsigned seed) {
  constexpr size_t kMaxBytes = 512 * 1024;
  alignas(4096) static size_t data[kMaxBytes / sizeof(size_t)];
  vector<size_t> positions(bytes / lineSize);
  for (size_t i = 0; i < positions.size(); ++i)
    positions[i] = i * lineSize / sizeof(size_t);
  return timeForPointerCycle(data, positions, seed);
}

size_t calculateL1CacheSize(size_t lineSize, bool debug) {
  constexpr size_t kStep = 8 * 1024, kMaxBytes = 512 * 1024;
  mt19937 rng(12345);
  size_t firstSize = 0, lastSize = 0;
  double previousTime = 0;
  for (size_t bytes = kStep; bytes <= kMaxBytes; bytes *= 2) {
    Timings timings;
    for (double &time : timings)
      time = timeForL1WorkingSet(bytes, lineSize, rng());
    const double currentTime = averageTime(timings);
    if (previousTime > 0 && currentTime > previousTime * 1.25) {
      firstSize = max(kStep, bytes / 4);
      lastSize = min(kMaxBytes, bytes * 2 + kStep);
      break;
    }
    previousTime = currentTime;
  }
  if (lastSize == 0)
    return 0;

  const size_t count = (lastSize - firstSize) / kStep + 1;
  vector<Timings> timings(count);
  vector<size_t> order(count);
  iota(order.begin(), order.end(), 0);
  for (size_t repeat = 0; repeat < kRepeats; ++repeat) {
    ranges::shuffle(order, rng);
    for (size_t i : order)
      timings[i][repeat] =
          timeForL1WorkingSet(firstSize + i * kStep, lineSize, rng());
  }
  vector<double> times;
  for (size_t i = 0; i < count; ++i) {
    times.push_back(averageTime(timings[i]));
    if (debug) {
      println("L1 data size = {} KiB, time = {} ns/access",
              (firstSize + i * kStep) / 1024, times.back());
    }
  }
  size_t cacheSize = 0;
  double greatestIncrease = 0;
  for (size_t i = 1; i < count; ++i) {
    const double increase = times[i] - times[i - 1];
    if (increase <= greatestIncrease || increase <= times[i - 1] * 0.05)
      continue;
    if (i + 1 < count && times[i + 1] - times[i - 1] < increase / 2)
      continue;
    size_t slowerRepeats = 0;
    for (size_t r = 0; r < kRepeats; ++r)
      slowerRepeats += timings[i][r] > timings[i - 1][r];
    if (slowerRepeats >= 6) {
      greatestIncrease = increase;
      cacheSize = firstSize + (i - 1) * kStep;
    }
  }
  return cacheSize;
}

double timeForL1Set(size_t cacheSize, size_t lines, size_t lineSize,
                    unsigned seed, bool control) {
  constexpr size_t kMaxCacheSize = 512 * 1024;
  constexpr size_t kMaxLines = 34;
  constexpr size_t kBlockBytes = 4096;
  alignas(kBlockBytes) static size_t
      data[kMaxLines * kMaxCacheSize / sizeof(size_t)];
  const size_t linesPerBlock = kBlockBytes / lineSize;
  size_t lineInBlock = seed % linesPerBlock;
  vector<size_t> positions(lines);
  for (size_t i = 0; i < lines; ++i) {
    positions[i] = (i * cacheSize + lineInBlock * lineSize) / sizeof(size_t);
    if (control)
      lineInBlock = (lineInBlock + 1) % linesPerBlock;
  }
  return timeForPointerCycle(data, positions, seed);
}

size_t calculateL1Associativity(size_t cacheSize, size_t lineSize, bool debug) {
  constexpr size_t kMaxLines = 34;
  array<Timings, kMaxLines + 1> sameSet{}, control{};
  vector<size_t> order(kMaxLines - 1);
  iota(order.begin(), order.end(), 2);
  mt19937 rng(12345);
  for (size_t repeat = 0; repeat < kRepeats; ++repeat) {
    ranges::shuffle(order, rng);
    for (size_t lines : order) {
      const unsigned seed = rng();
      if (repeat % 2 == 0) {
        sameSet[lines][repeat] =
            timeForL1Set(cacheSize, lines, lineSize, seed, false);
        control[lines][repeat] =
            timeForL1Set(cacheSize, lines, lineSize, seed, true);
      } else {
        control[lines][repeat] =
            timeForL1Set(cacheSize, lines, lineSize, seed, true);
        sameSet[lines][repeat] =
            timeForL1Set(cacheSize, lines, lineSize, seed, false);
      }
    }
  }
  array<double, kMaxLines + 1> extraTime{};
  for (size_t lines = 2; lines <= kMaxLines; ++lines) {
    const double sameSetTime = averageFastTime(sameSet[lines]);
    const double controlTime = averageFastTime(control[lines]);
    extraTime[lines] = sameSetTime - controlTime;
    if (debug)
      println("L1 set lines = {}, same set = {} ns, control = {} ns", lines,
              sameSetTime, controlTime);
  }
  size_t lastLines = kMaxLines;
  const double minimumExtraTime = averageFastTime(sameSet[2]) * 0.15;
  for (size_t lines = 4; lines <= kMaxLines; ++lines) {
    if (extraTime[lines - 2] > minimumExtraTime &&
        extraTime[lines - 1] > minimumExtraTime &&
        extraTime[lines] > minimumExtraTime) {
      lastLines = lines;
      break;
    }
  }
  size_t associativity = 0;
  double greatestIncrease = 0;
  for (size_t lines = 3; lines <= lastLines; ++lines) {
    const double increase = extraTime[lines] - extraTime[lines - 1];
    if (increase <= greatestIncrease ||
        increase <= abs(extraTime[lines - 1]) * 0.05)
      continue;
    if (lines < lastLines &&
        extraTime[lines + 1] - extraTime[lines - 1] < increase / 2)
      continue;
    size_t slowerRepeats = 0;
    for (size_t r = 0; r < kRepeats; ++r) {
      const auto current = sameSet[lines][r] - control[lines][r];
      const auto previous = sameSet[lines - 1][r] - control[lines - 1][r];
      slowerRepeats += current > previous;
    }
    if (slowerRepeats >= 6) {
      greatestIncrease = increase;
      associativity = lines - 1;
    }
  }
  return associativity;
}

int main(int argc, char *argv[]) {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  bool debug = false;
  for (int i = 1; i < argc; ++i) {
    if (string_view(argv[i]) == "--debug")
      debug = true;
  }
  if (debug)
    println("Worker threads = {}, repeats = {}", kThreadCount, kRepeats);

  const size_t lineSize = calculateLineSize(debug);
  if (lineSize != 0) {
    println("Cache line size = {} bytes", lineSize);
    const size_t l1Size = calculateL1CacheSize(lineSize, debug);
    if (l1Size != 0) {
      println("L1 data cache size = {} KiB", l1Size / 1024);
      const size_t associativity =
          calculateL1Associativity(l1Size, lineSize, debug);
      if (associativity != 0) {
        println("L1 data cache associativity = {}", associativity);
        return 0;
      } else {
        println("Can't find L1 data cache associativity");
      }
    } else {
      println("Can't find L1 data cache size");
    }
  } else {
    println("Can't find Cache line size");
  }
  return 1;
}
