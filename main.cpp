#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <print>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

auto timeForGivenPossibleCacheLineSize(const size_t possibleCacheLineSize) {
    alignas(4096) volatile size_t data[1024];

    static_assert(sizeof(size_t) == 8);
    assert(possibleCacheLineSize % 8 == 0);
    assert(possibleCacheLineSize / 8 < 1024);

    data[0] = 0;
    data[possibleCacheLineSize >> 3] = 0;

    volatile size_t &x = data[0];
    volatile size_t &y = data[possibleCacheLineSize >> 3];

    constexpr size_t kIterations = 1ull << 31;

    const auto start = std::chrono::steady_clock::now();
    auto cycleIncrement = [](volatile size_t &value) {
        for (size_t i = 0; i < kIterations; ++i) {
            value += 1;
        }
    };

    std::thread t1([&] {
        cycleIncrement(x);
    }), t2([&] {
        cycleIncrement(y);
    });

    t1.join();
    t2.join();

    const auto end = std::chrono::steady_clock::now();
    const auto diff = end - start;
    return std::chrono::duration<double>(diff).count();
}

size_t calculateLineSize(bool debug) {
    double previousTime = 0;
    for (size_t offset = 8; offset <= 1024; offset <<= 1) {
        const double currentTime = timeForGivenPossibleCacheLineSize(offset);
        if (debug) {
            std::println("Offset = {} (in bytes), time = {}", offset, currentTime);
        }
        if (previousTime > 0 && currentTime * 2 <= previousTime) {
            return offset;
        }
        previousTime = currentTime;
    }
    return 0;
}

double timeForL1WorkingSet(const size_t bytes, const size_t lineSize) {
    constexpr size_t kMaxBytes = 512 * 1024;
    constexpr size_t kIterations = 1 << 22;
    alignas(4096) static size_t data[kMaxBytes / sizeof(size_t)];

    const size_t stride = lineSize / sizeof(size_t);
    const size_t count = bytes / lineSize;
    std::vector<size_t> positions(count);
    std::iota(positions.begin(), positions.end(), 0);
    std::mt19937 rng(12345);
    std::ranges::shuffle(positions, rng);

    for (size_t i = 0; i < count; ++i) {
        data[positions[i] * stride] = positions[(i + 1) % count] * stride;
    }

    const volatile size_t *const links = data;
    size_t index = positions[0] * stride;
    for (size_t i = 0; i < count * 2; ++i) {
        index = links[index];
    }

    std::array<double, 3> timings{};
    for (double &timing: timings) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < kIterations; ++i) {
            index = links[index];
        }
        const auto end = std::chrono::steady_clock::now();
        timing = std::chrono::duration<double, std::nano>(end - start).count() / kIterations;
    }
    std::ranges::sort(timings);
    return timings[1];
}

size_t calculateL1CacheSize(size_t lineSize, bool debug) {
    constexpr size_t kStep = 8 * 1024;
    constexpr size_t kMaxBytes = 512 * 1024;
    double fastestTime = 0;
    size_t slowMeasurements = 0;

    for (size_t bytes = kStep; bytes <= kMaxBytes; bytes += kStep) {
        const double nanoseconds = timeForL1WorkingSet(bytes, lineSize);
        if (debug) {
            std::println("L1 data size = {} KiB, time = {} ns/access", bytes / 1024, nanoseconds);
        }

        if (fastestTime == 0 || nanoseconds < fastestTime) {
            fastestTime = nanoseconds;
            slowMeasurements = 0;
            continue;
        }
        if (nanoseconds >= fastestTime * 1.5) {
            if (++slowMeasurements == 2) {
                return bytes - 2 * kStep;
            }
        } else {
            slowMeasurements = 0;
        }
    }
    return 0;
}

double timeForL1Set(size_t cacheSize, size_t lineSize, size_t lines, bool sameSet) {
    constexpr size_t kMaxCacheSize = 512 * 1024;
    constexpr size_t kMaxLineSize = 1024;
    constexpr size_t kMaxLines = 34;
    constexpr size_t kIterations = 1 << 20;
    alignas(4096) static size_t data[kMaxLines * (kMaxCacheSize + kMaxLineSize) / sizeof(size_t)];

    const size_t stride = (cacheSize + (sameSet ? 0 : lineSize)) / sizeof(size_t);
    std::vector<size_t> positions(lines);
    std::iota(positions.begin(), positions.end(), 0);
    std::mt19937 rng(12345);
    std::ranges::shuffle(positions, rng);

    for (size_t i = 0; i < lines; ++i) {
        data[positions[i] * stride] = positions[(i + 1) % lines] * stride;
    }

    const volatile size_t *const links = data;
    size_t index = positions[0] * stride;
    for (size_t i = 0; i < lines * 64; ++i) {
        index = links[index];
    }

    std::array<double, 3> timings{};
    for (double &timing: timings) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < kIterations; ++i) {
            index = links[index];
        }
        const auto end = std::chrono::steady_clock::now();
        timing = std::chrono::duration<double, std::nano>(end - start).count() / kIterations;
    }
    std::ranges::sort(timings);
    return timings[1];
}

size_t calculateL1Associativity(size_t cacheSize, size_t lineSize, bool debug) {
    size_t slowMeasurements = 0;
    for (size_t lines = 2; lines <= 34; ++lines) {
        const auto sameSet = timeForL1Set(cacheSize, lineSize, lines, true);
        const auto control = timeForL1Set(cacheSize, lineSize, lines, false);
        if (debug) {
            std::println("L1 set lines = {}, same set = {} ns, control = {} ns", lines, sameSet, control);
        }

        if (sameSet >= control * 1.5) {
            if (++slowMeasurements == 2) {
                return lines - 2;
            }
        } else {
            slowMeasurements = 0;
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    bool debug = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--debug") {
            debug = true;
        }
    }

    const size_t lineSize = calculateLineSize(debug);
    if (lineSize != 0) {
        std::println("Cache line size = {} bytes", lineSize);
        const size_t l1Size = calculateL1CacheSize(lineSize, debug);
        if (l1Size != 0) {
            std::println("L1 data cache size = {} KiB", l1Size / 1024);
            const size_t associativity = calculateL1Associativity(l1Size, lineSize, debug);
            if (associativity != 0) {
                std::println("L1 data cache associativity = {}", associativity);
            } else {
                std::println("Can't find L1 data cache associativity");
            }
        } else {
            std::println("Can't find L1 data cache size");
        }
    } else {
        std::println("Can't find Cache line size");
    }
}
