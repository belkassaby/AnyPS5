#include "prx/libSceAgcDriver/Execution/include/HostThreads.hpp"
#include "prx/libSceAgcDriver/Graphics/include/CpuTimeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/ReplayProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawSpeculation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::array<const char*, static_cast<std::size_t>(DrawPipeline::DrainReason::Count)> DrainNames{"packet", "flush", "labels", "capture", "indirect", "submission"};

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

// Debug aid: APS5_PROFILE_COMMIT_KINDS=1 splits committer time by the commit's lambda type (one per
// enqueue site) in the 10 s report.
struct CommitKinds {
    std::mutex mutex;
    std::vector<std::tuple<const char*, std::uint64_t, std::uint64_t>> kinds;
    static bool Enabled() {
        static const bool enabled = std::getenv("APS5_PROFILE_COMMIT_KINDS") != nullptr;
        return enabled;
    }
    void Note(const char* name, std::uint64_t ns) {
        std::lock_guard lock(mutex);
        for (auto& [kind, count, total] : kinds) {
            if (kind != name) continue;
            ++count;
            total += ns;
            return;
        }
        kinds.emplace_back(name, 1, ns);
    }
    void Report() {
        std::lock_guard lock(mutex);
        std::sort(kinds.begin(), kinds.end(), [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
        for (const auto& [kind, count, total] : kinds) std::fprintf(stderr, "[commit-kinds] %8llu x %6.1f us = %7.1f ms  %.160s\n", static_cast<unsigned long long>(count), static_cast<double>(total) / 1e3 / static_cast<double>(count), static_cast<double>(total) / 1e6, kind);
        kinds.clear();
    }
};
CommitKinds commitKinds;

std::atomic<DrawPipeline::ForkRunner*> forkRunner{nullptr};

bool forkStatsOn() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile && Graphics::ForkStatsEnabled();
}

class ForkPool {
public:
    using Job = std::function<void(std::size_t, std::size_t)>;

    static ForkPool& Get() {
        static auto* pool = new ForkPool();
        return *pool;
    }

    void Run(std::size_t count, const Job& job) {
        if (count == 0) return;
        auto batch = std::make_shared<Batch>();
        batch->job = &job;
        batch->count = count;
        std::unique_lock lock(mutex);
        while (threads.size() < DrawPipeline::ForkWorkers()) threads.emplace_back([this, slot = threads.size() + 1] { work(slot); });
        current = batch;
        ++generation;
        wake.notify_all();
        done.wait(lock, [&] { return batch->finished == batch->count; });
        current.reset();
    }

private:
    struct Batch {
        const Job* job = nullptr;
        std::size_t count = 0;
        std::atomic<std::size_t> next{0};
        std::size_t finished = 0;
    };

    void work(std::size_t slot) {
        PinWorkerThread("fork build");
        std::uint64_t seen = 0;
        for (;;) {
            std::shared_ptr<Batch> batch;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return generation != seen; });
                seen = generation;
                batch = current;
            }
            if (batch == nullptr) continue;
            std::size_t processed = 0;
            for (auto index = batch->next.fetch_add(1, std::memory_order_relaxed); index < batch->count; index = batch->next.fetch_add(1, std::memory_order_relaxed)) {
                try {
                    (*batch->job)(index, slot);
                } catch (...) {
                }
                ++processed;
            }
            if (processed == 0) continue;
            std::lock_guard lock(mutex);
            batch->finished += processed;
            if (batch->finished == batch->count) done.notify_all();
        }
    }

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable done;
    std::shared_ptr<Batch> current;
    std::uint64_t generation = 0;
    std::vector<std::thread> threads;
};

}

void DrawPipeline::SetForkRunner(ForkRunner* runner) {
    forkRunner.store(runner, std::memory_order_release);
}

std::size_t DrawPipeline::ForkWorkers() {
    static const std::size_t workers = [] {
        const char* text = std::getenv("APS5_FORK_WORKERS");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 4ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(value, 1, 14));
    }();
    return workers;
}

std::size_t DrawPipeline::ForkRunLimit() {
    static const std::size_t limit = [] {
        const char* text = std::getenv("APS5_FORK_RUN");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 32ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(value, 2, 64));
    }();
    return limit;
}

void DrawPipeline::RunForkJobs(std::size_t count, const std::function<void(std::size_t index, std::size_t slot)>& job) {
    ForkPool::Get().Run(count, job);
}

DrawPipeline& DrawPipeline::Queue0() {
    static DrawPipeline* pipeline = new DrawPipeline();
    return *pipeline;
}

std::size_t DrawPipeline::Depth() {
    static const std::size_t depth = [] {
        const char* text = std::getenv("APS5_PIPELINED_DRAWS");
        if (std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr || std::getenv("APS5_DRAW_DRAIN") != nullptr || std::getenv("APS5_DRAIN_ALL") != nullptr || std::getenv("APS5_NO_WORDWISE_CAPTURE") != nullptr) return std::size_t{0};
        if (text == nullptr) return std::size_t{8};
        const auto value = std::strtoull(text, nullptr, 10);
        return value == 0 ? std::size_t{0} : value == 1 ? std::size_t{8} : static_cast<std::size_t>(std::min<unsigned long long>(value, 256));
    }();
    static const std::size_t forkDepth = [] {
        const char* text = std::getenv("APS5_FORK_DEPTH");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        return value == 0 ? std::size_t{0} : static_cast<std::size_t>(std::clamp<unsigned long long>(value, 8, 256));
    }();
    return depth != 0 && forkDepth != 0 && Graphics::ForkBuildEnabled() && Graphics::LiveFlag("fork_depth", true) ? std::max(depth, forkDepth) : depth;
}

bool& DrawPipeline::Active() {
    static thread_local bool active = false;
    return active;
}

namespace {

struct PipeEvent {
    std::int64_t ns;
    std::uint32_t thread;
    std::uint16_t kind;
    std::uint16_t argument;
};

struct PipeTraceState {
    std::atomic<std::int64_t> from{0};
    std::atomic<std::int64_t> until{0};
    std::atomic<std::int64_t> nextLiveCheck{0};
    bool live = false;
    std::vector<PipeEvent> events;
    std::atomic<std::size_t> next{0};
    std::atomic<bool> dumped{false};
    std::atomic<std::uint32_t> threads{0};
};

PipeTraceState* PipeTracing() {
    static PipeTraceState* state = [] () -> PipeTraceState* {
        const char* text = std::getenv("APS5_TRACE_PIPE");
        if (text == nullptr) return nullptr;
        auto* created = new PipeTraceState();
        created->events.resize(std::size_t{1} << 21u);
        const auto start = std::chrono::steady_clock::now().time_since_epoch().count();
        created->live = std::strcmp(text, "live") == 0;
        const auto from = created->live ? std::numeric_limits<std::int64_t>::max() - 4'000'000'000ll : start + static_cast<std::int64_t>(std::strtod(text, nullptr) * 1e9);
        created->from.store(from);
        created->until.store(from + 2'000'000'000ll);
        return created;
    }();
    return state;
}

void dumpPipeTrace(PipeTraceState& state) {
    if (state.dumped.exchange(true)) return;
    const auto count = std::min(state.next.load(), state.events.size());
    if (FILE* file = std::fopen("pipe-trace.txt", "w")) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto& event = state.events[i];
            std::fprintf(file, "%.3f %u %u %u%c", static_cast<double>(event.ns - state.from.load()) / 1e3, event.thread, event.kind, event.argument, 10);
        }
        std::fclose(file);
    }
    std::fprintf(stderr, "[pipe-trace] wrote %zu events to pipe-trace.txt%c", count, 10);
}

}

void DrawPipeline::PipeTrace(Trace kind, std::uint32_t argument) {
    auto* state = PipeTracing();
    if (state == nullptr || state->dumped.load(std::memory_order_relaxed)) return;
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    if (now < state->from.load(std::memory_order_relaxed)) {
        auto check = state->nextLiveCheck.load(std::memory_order_relaxed);
        if (!state->live || now < check || !state->nextLiveCheck.compare_exchange_strong(check, now + 250'000'000ll)) return;
        if (!Graphics::LiveFlag("trace", false)) return;
        state->until.store(now + 2'000'000'000ll);
        state->from.store(now);
        std::fprintf(stderr, "[pipe-trace] live capture started%c", 10);
    }
    if (now > state->until.load(std::memory_order_relaxed)) {
        dumpPipeTrace(*state);
        return;
    }
    static thread_local std::uint32_t thread = 0;
    if (thread == 0) thread = state->threads.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto index = state->next.fetch_add(1, std::memory_order_relaxed);
    if (index >= state->events.size()) {
        dumpPipeTrace(*state);
        return;
    }
    state->events[index] = {now, thread, static_cast<std::uint16_t>(kind), static_cast<std::uint16_t>(argument)};
}

std::atomic<std::uint8_t>& DrawPipeline::ProducerState() {
    static std::atomic<std::uint8_t> state{0};
    return state;
}

std::atomic<std::uint64_t>& DrawPipeline::EpochToken() {
    static std::atomic<std::uint64_t> token{1};
    return token;
}

void DrawPipeline::FollowEpoch(std::uint64_t token) {
    static thread_local std::uint64_t followed = 0;
    if (followed == token) return;
    followed = token;
    if (GuestMemory::SharedEpochMemo()) GuestMemory::FollowCollectEpoch(token);
    else GuestMemory::BumpCollectEpoch();
}

void DrawPipeline::rethrowFailure() {
    if (failure == nullptr) return;
    auto error = std::exchange(failure, nullptr);
    std::rethrow_exception(error);
}

bool DrawPipeline::SplitDraws() {
    static const bool enabled = std::getenv("APS5_SPLIT_DRAWS") != nullptr && Depth() != 0;
    return enabled;
}

std::size_t DrawPipeline::PrepWorkers() {
    static const std::size_t workers = [] {
        const char* text = std::getenv("APS5_SPLIT_DRAWS");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 1ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(value, 1, 16));
    }();
    return workers;
}

std::uint64_t& DrawPipeline::PrepSequence() {
    static thread_local std::uint64_t sequence = 0;
    return sequence;
}

bool& DrawPipeline::PrepEnqueued() {
    static thread_local bool enqueued = false;
    return enqueued;
}

void DrawPipeline::Prepare(Commit job, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes) {
    if (auto segment = ReplayProbe::Hold()) {
        job = [segment = std::move(segment), inner = std::move(job)] {
            const ReplayProbe::Scope probeScope(segment);
            inner();
        };
    }
    std::unique_lock lock(prepMutex);
    const auto room = Depth() * PrepWorkers();
    if (prepOutstanding.load(std::memory_order_relaxed) >= room) {
        const ProducerScope producer(Producer::PrepRoom);
        const auto start = std::chrono::steady_clock::now();
        prepIdle.wait(lock, [&] { return prepOutstanding.load(std::memory_order_relaxed) < room; });
        prepWaitNs += elapsedNs(start);
    }
    if (!writes.empty()) prepWrites.push_back({prepNextSequence, std::move(writes), labelAddress, std::move(labelBytes)});
    prepItems.emplace_back(prepNextSequence++, std::move(job));
    prepOutstanding.fetch_add(1, std::memory_order_release);
    while (prepThreads.size() < PrepWorkers()) prepThreads.emplace_back([this] { prepRun(); });
    prepWake.notify_one();
}

namespace {
// Per-thread waits inside the pipeline (ms), for the [slow-prep] lines.
thread_local double threadTurnWaitMs = 0;
thread_local double threadRoomWaitMs = 0;
thread_local double threadDrainWaitMs = 0;
}

void DrawPipeline::waitPrepTurn(std::uint64_t sequence) {
    const TraceScope trace(Trace::PrepTurn, Trace::PrepTurnEnd);
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock lock(prepMutex);
    prepTurnChanged.wait(lock, [&] { return prepTurn == sequence; });
    threadTurnWaitMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void DrawPipeline::finishPrepTurn(std::uint64_t sequence) {
    std::lock_guard lock(prepMutex);
    if (prepTurn == sequence) ++prepTurn;
    prepTurnChanged.notify_all();
}

bool DrawPipeline::PrepReorder() {
    static const bool enabled = std::getenv("APS5_PREP_REORDER") != nullptr;
    return Graphics::LiveFlag("prep_reorder", enabled);
}

void DrawPipeline::deliverPrepared(std::uint64_t sequence, std::optional<Item> item) {
    {
        std::lock_guard lock(prepMutex);
        if (prepTurn != sequence || prepFlushing) {
            prepParked.emplace(sequence, std::move(item));
            ++prepParkedCount;
            return;
        }
        prepFlushing = true;
    }
    for (;;) {
        if (item.has_value()) pushItem(std::move(*item));
        std::lock_guard lock(prepMutex);
        std::erase_if(prepWrites, [&](const PrepWrites& entry) { return entry.sequence == sequence; });
        prepOutstanding.fetch_sub(1, std::memory_order_release);
        prepIdle.notify_all();
        ++prepTurn;
        prepTurnChanged.notify_all();
        const auto next = prepParked.find(prepTurn);
        if (next == prepParked.end()) {
            prepFlushing = false;
            return;
        }
        sequence = next->first;
        item = std::move(next->second);
        prepParked.erase(next);
    }
}

// Live `room_spin` (env APS5_ROOM_SPIN=0 turns it off; HandoffSpin.hpp RoomSpinWindow): a producer
// finding the queue full used to sleep on `idle` at once, so with the committer the bottleneck every
// commit had to wake it: sampling the World Map committer put the condition variable broadcast
// (ReleaseSemaphore) and the wake's mutex hand-back (SetEvent) at about 6% of its time, roughly 2 us
// a commit. The producer now first spins outside the mutex on `outstanding` (items.size(), kept with
// it under the mutex) for one window, about five commits, and sleeps only if the queue is still
// full. It pushes only with the mutex held and room in the queue, as before.
bool DrawPipeline::spinForRoom(std::unique_lock<std::mutex>& lock) {
    static const bool enabledDefault = [] {
        const char* value = std::getenv("APS5_ROOM_SPIN");
        return value == nullptr || value[0] != '0';
    }();
    const auto window = RoomSpinWindow();
    if (window.count() <= 0 || !Graphics::LiveFlag("room_spin", enabledDefault)) return false;
    const auto depth = Depth();
    lock.unlock();
    SpinUntil(window, [&] { return outstanding.load(std::memory_order_acquire) < depth; });
    lock.lock();
    return items.size() < Depth();
}

void DrawPipeline::pushItem(Item item) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::vector<Commit> finished;
    std::unique_lock lock(mutex);
    finished.swap(retired);
    if (items.size() >= Depth()) {
        const auto start = std::chrono::steady_clock::now();
        if (!spinForRoom(lock)) {
            ++idleWaiters;
            idle.wait(lock, [&] { return items.size() < Depth(); });
            --idleWaiters;
        }
        fullWaitNs += elapsedNs(start);
        threadRoomWaitMs += static_cast<double>(elapsedNs(start)) / 1e6;
    }
    depthSum += items.size();
    ++enqueued;
    item.sequence = ++nextItemSequence;
    items.push_back(std::move(item));
    outstanding.fetch_add(1, std::memory_order_release);
    if (!thread.joinable()) thread = std::thread([this] { run(); });
    if (committerWaiting) wake.notify_one();
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
}

void DrawPipeline::prepRun() {
    AgcDriver::JoinGamesSchedulingClass();
    Active() = true;
    GuestMemory::SetCollectRole(2);
    for (;;) {
        std::pair<std::uint64_t, Commit> job;
        {
            std::unique_lock lock(prepMutex);
            prepWake.wait(lock, [&] { return !prepItems.empty(); });
            job = std::move(prepItems.front());
            prepItems.pop_front();
        }
        PrepSequence() = job.first;
        PrepEnqueued() = false;
        const auto start = std::chrono::steady_clock::now();
        bool failed = false;
        PipeTrace(Trace::Prep, static_cast<std::uint32_t>(job.first & 0xffffu));
        const auto syncsBefore = Graphics::ThreadHookSyncs();
        struct Waits {
            double turn, room, drain, lock;
        };
        const Waits waitsBefore{threadTurnWaitMs, threadRoomWaitMs, threadDrainWaitMs, GuestMemory::ThreadGpuLockWaitMs()};
        struct SlowPrep {
            std::chrono::steady_clock::time_point start;
            std::pair<double, std::uint64_t> syncsBefore;
            Waits waitsBefore;
            ~SlowPrep() {
                // [slow-prep] (APS5_PROFILE_DRAW): a preparation job over 5 ms, with the GPU-sync
                // waits it made, since one slow job stalls the in-order hand-off behind it.
                static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
                if (!profile) return;
                const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                if (ms < 5.0) return;
                // At most 25 lines per 10 s, so every phase of a run (boot, load, race) gets some.
                static std::atomic<std::int64_t> windowStart{0};
                static std::atomic<int> told{0};
                const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
                auto began = windowStart.load(std::memory_order_relaxed);
                if (now - began > 10'000'000'000ll && windowStart.compare_exchange_strong(began, now)) told.store(0, std::memory_order_relaxed);
                if (told.fetch_add(1, std::memory_order_relaxed) >= 25) return;
                const auto syncs = Graphics::ThreadHookSyncs();
                std::fprintf(stderr, "[slow-prep] %.1f ms (GPU-sync waits %.1f ms in %llu; waited for: turn %.1f ms, committer room %.1f ms, drains %.1f ms, GPU mutex %.1f ms)\n", ms, syncs.first - syncsBefore.first, static_cast<unsigned long long>(syncs.second - syncsBefore.second), threadTurnWaitMs - waitsBefore.turn, threadRoomWaitMs - waitsBefore.room, threadDrainWaitMs - waitsBefore.drain, GuestMemory::ThreadGpuLockWaitMs() - waitsBefore.lock);
            }
        } slowPrep{start, syncsBefore, waitsBefore};
        try {
            job.second();
        } catch (const std::exception& error) {
            failed = true;
            static std::atomic<std::uint64_t> reported{0};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 20) std::fprintf(stderr, "[draw] split draw preparation failed: %.200s\n", error.what());
        } catch (...) {
            failed = true;
        }
        if (!PrepEnqueued()) {
            if (!PrepReorder()) waitPrepTurn(job.first);
            PrepEnqueued() = true;
            deliverPrepared(job.first, std::nullopt);
        }
        PipeTrace(Trace::PrepEnd);
        PrepSequence() = 0;
        const auto spent = elapsedNs(start);
        std::lock_guard lock(prepMutex);
        ++prepJobs;
        prepNs += spent;
        if (failed) ++prepFailures;
    }
}

void DrawPipeline::EnqueueOrdered(Commit commit, std::vector<Range> writes, bool unknownWrites) {
    if (!SplitDraws() || OnPrepThread() || prepOutstanding.load(std::memory_order_acquire) == 0) {
        Enqueue(std::move(commit), std::move(writes), 0, {}, unknownWrites);
        return;
    }
    auto held = std::make_shared<std::pair<Commit, std::vector<Range>>>(std::move(commit), writes);
    Prepare([this, held, unknownWrites] { Enqueue(std::move(held->first), std::move(held->second), 0, {}, unknownWrites); }, std::move(writes));
}

void DrawPipeline::WaitEarlierPreparations() {
    if (OnPrepThread() && !PrepEnqueued()) waitPrepTurn(PrepSequence());
}

void DrawPipeline::drainPrep() {
    if (prepOutstanding.load(std::memory_order_acquire) == 0 || OnPrepThread()) return;
    std::unique_lock lock(prepMutex);
    prepIdle.wait(lock, [&] { return prepOutstanding.load(std::memory_order_relaxed) == 0; });
}

void DrawPipeline::Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, bool unknownWrites, std::shared_ptr<ForkJob> fork) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    // A job the queue thread hands over (a pipelined DMA store or copy, an ordered fill or label,
    // a pipelined dispatch) collects under the epoch of the packet that made it, as the queue
    // thread would have: it follows the token current here. Without it the committer would collect
    // under whatever token the last draw followed, an older ordering point. Draw commits follow
    // their own token inside (the later FollowEpoch wins); a newer token only asks for newer walks.
    if (!OnPrepThread()) {
        commit = [token = EpochToken().load(std::memory_order_acquire), inner = std::move(commit)] {
            FollowEpoch(token);
            inner();
        };
    }
    static const bool orderedHandoff = std::getenv("APS5_NO_ORDERED_HANDOFF") == nullptr;
    if (orderedHandoff && SplitDraws() && !OnPrepThread() && prepOutstanding.load(std::memory_order_acquire) != 0) {
        // Behind the outstanding preparations, in their order, instead of waiting for them.
        struct Held {
            Commit commit;
            std::vector<Range> writes;
            std::uint64_t labelAddress;
            std::vector<std::byte> labelBytes;
            bool unknownWrites;
            std::shared_ptr<ForkJob> fork;
        };
        auto held = std::make_shared<Held>(Held{std::move(commit), writes, labelAddress, labelBytes, unknownWrites, std::move(fork)});
        Prepare([this, held] { Enqueue(std::move(held->commit), std::move(held->writes), held->labelAddress, std::move(held->labelBytes), held->unknownWrites, std::move(held->fork)); }, std::move(writes), labelAddress, std::move(labelBytes));
        return;
    }
    // Finished commits are destroyed here, off the committer (their captures are a draw's whole
    // front end), after the mutex and the preparation turn are released (declared before both).
    const auto prepared = PrepSequence();
    if (prepared != 0 && !PrepEnqueued()) {
        rethrowFailure();
        if (!PrepReorder()) waitPrepTurn(prepared);
        PrepEnqueued() = true;
        deliverPrepared(prepared, Item{std::move(commit), std::move(writes), labelAddress, std::move(labelBytes), 0, unknownWrites, std::move(fork), ReplayProbe::Tag()});
        return;
    }
    if (prepared != 0) {
        std::unique_lock prepLock(prepMutex);
        prepTurnChanged.wait(prepLock, [&] { return prepTurn > prepared; });
    }
    std::vector<Commit> finished;
    drainPrep();
    const auto sequence = PrepSequence();
    if (sequence != 0 && !PrepEnqueued()) waitPrepTurn(sequence);
    struct TurnRelease {
        DrawPipeline& pipeline;
        std::uint64_t sequence;
        ~TurnRelease() {
            if (sequence == 0 || PrepEnqueued()) return;
            PrepEnqueued() = true;
            pipeline.finishPrepTurn(sequence);
        }
    } release{*this, sequence};
    std::unique_lock lock(mutex);
    rethrowFailure();
    finished.swap(retired);
    if (items.size() >= Depth()) {
        const ProducerScope producer(Producer::EnqueueRoom);
        const auto start = std::chrono::steady_clock::now();
        if (!spinForRoom(lock)) {
            ++idleWaiters;
            idle.wait(lock, [&] { return items.size() < Depth(); });
            --idleWaiters;
        }
        fullWaitNs += elapsedNs(start);
        rethrowFailure();
    }
    depthSum += items.size();
    ++enqueued;
    items.push_back({std::move(commit), std::move(writes), labelAddress, std::move(labelBytes), ++nextItemSequence, unknownWrites, std::move(fork), ReplayProbe::Tag()});
    outstanding.fetch_add(1, std::memory_order_release);
    if (!thread.joinable()) thread = std::thread([this] { run(); });
    if (committerWaiting) wake.notify_one();
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
}

// [parcommit] (APS5_PROFILE_DRAW, every 10 s, under `mutex`): why render passes ended (the
// recorder's counts: a pass that ends for anything but the guest's next draw is a pass break the
// next draw pays for as a new head) and, with live `parallel_commit`, the pass runs.
void DrawPipeline::reportPassRuns() {
    const auto ends = Graphics::Recorder::TakePassEndCounts();
    std::uint64_t total = 0;
    for (const auto count : ends.causes) total += count;
    std::string line = "[parcommit] (10 s) " + std::to_string(total) + " render pass ends by cause:";
    for (std::size_t index = 1; index < ends.causes.size(); ++index) {
        if (ends.causes[index] == 0) continue;
        line += std::string(" ") + Graphics::Recorder::PassEndCauseName(static_cast<Graphics::Recorder::PassEndCause>(index)) + " " + std::to_string(ends.causes[index]);
        if (static_cast<Graphics::Recorder::PassEndCause>(index) != Graphics::Recorder::PassEndCause::Submit) continue;
        std::string reasons;
        for (std::size_t reason = 0; reason < ends.submits.size(); ++reason) {
            if (ends.submits[reason] == 0) continue;
            reasons += std::string(reasons.empty() ? "" : " ") + Graphics::Recorder::SubmitReasonName(static_cast<Graphics::Recorder::SubmitReason>(reason)) + " " + std::to_string(ends.submits[reason]);
        }
        if (!reasons.empty()) line += " (" + reasons + ")";
    }
    line += ";";
    char item[512];
    std::snprintf(item, sizeof(item), " pass runs (parallel_commit %s): %llu runs of %.1f draws, %llu continued while others waited, %llu continuations (each a deferred starve check under APS5_STARVE_SUBMIT), %llu heads left outside a run to let waiters in; ended by:", Graphics::ParallelCommitEnabled() ? "on" : "off", static_cast<unsigned long long>(passRunStats.runs), passRunStats.runs != 0 ? static_cast<double>(passRunStats.draws) / static_cast<double>(passRunStats.runs) : 0.0, static_cast<unsigned long long>(passRunStats.contendedContinuations), static_cast<unsigned long long>(passRunStats.deferredStarveChecks), static_cast<unsigned long long>(passRunStats.cooldowns));
    line += item;
    for (std::size_t index = 0; index < passRunStats.ends.size(); ++index) {
        if (passRunStats.ends[index] == 0) continue;
        line += std::string(" ") + PassRunEndName(static_cast<PassRunEnd>(index)) + " " + std::to_string(passRunStats.ends[index]);
    }
    if (Graphics::VerifyParallelCommit()) {
        std::snprintf(item, sizeof(item), "; verified %llu runs: draws after the head continued the pass %llu, began one %llu", static_cast<unsigned long long>(verifiedRuns), static_cast<unsigned long long>(verifiedContinued), static_cast<unsigned long long>(verifiedBegun));
        line += item;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    passRunStats = {};
    verifiedRuns = verifiedContinued = verifiedBegun = 0;
}

void DrawPipeline::report(std::chrono::steady_clock::time_point now) {
    lastReport = now;
    std::string drainText;
    for (std::size_t i = 0; i < drains.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f ms)", DrainNames[i], static_cast<unsigned long long>(drains[i]), static_cast<double>(drainWaitNs[i]) / 1e6);
        drainText += item;
    }
    {
        char item[96];
        std::snprintf(item, sizeof(item), "; waits taken before the drain %llu (%.1f ms)", static_cast<unsigned long long>(earlyWaits), static_cast<double>(earlyWaitNs) / 1e6);
        drainText += item;
        earlyWaits = earlyWaitNs = 0;
    }
    drainText += "; packet drains by opcode:";
    for (int shown = 0; shown < 5; ++shown) {
        const auto top = std::max_element(drainOpcodes.begin(), drainOpcodes.begin() + 0x100);
        if (*top == 0) break;
        char item[32];
        std::snprintf(item, sizeof(item), " 0x%02x x%llu", static_cast<unsigned>(top - drainOpcodes.begin()), static_cast<unsigned long long>(*top));
        drainText += item;
        *top = 0;
    }
    drainOpcodes = {};
    const auto perCommit = commits == 0 ? 0.0 : static_cast<double>(commitNs) / 1e3 / static_cast<double>(commits);
    const auto perDraw = enqueued == 0 ? 0.0 : static_cast<double>(fullWaitNs) / 1e3 / static_cast<double>(enqueued);
    const auto queued = enqueued == 0 ? 0.0 : static_cast<double>(depthSum) / static_cast<double>(enqueued);
    std::fprintf(stderr, "[draw] committer idle by queue-thread state (10 s): working %.0f ms, memory wait %.0f ms, prep room %.0f ms, enqueue room %.0f ms, draining %.0f ms, polling for submissions %.0f ms\n", idleNs[0] / 1e6, idleNs[1] / 1e6, idleNs[2] / 1e6, idleNs[3] / 1e6, idleNs[4] / 1e6, idleNs[5] / 1e6);
    idleNs = {};
    std::fprintf(stderr, "[draw] pipelined (10 s, depth %zu): %llu commits at %.1f us each on the committer, %llu failed; %llu enqueued behind %.2f on average, waiting %.1f us per draw for room; drains:%s\n", Depth(), static_cast<unsigned long long>(commits), perCommit, static_cast<unsigned long long>(commitErrors), static_cast<unsigned long long>(enqueued), queued, perDraw, drainText.c_str());
    if (SplitDraws()) {
        std::lock_guard prepLock(prepMutex);
        std::fprintf(stderr, "[draw] split preparation (10 s): %llu draws at %.1f us each, %llu failed, %.1f ms waiting for room\n", static_cast<unsigned long long>(prepJobs), prepJobs == 0 ? 0.0 : static_cast<double>(prepNs) / 1e3 / static_cast<double>(prepJobs), static_cast<unsigned long long>(prepFailures), static_cast<double>(prepWaitNs) / 1e6);
        prepJobs = prepNs = prepWaitNs = prepFailures = 0;
    }
    if (CommitKinds::Enabled()) commitKinds.Report();
    reportPassRuns();
    commits = commitNs = commitErrors = enqueued = fullWaitNs = depthSum = 0;
    drains = {};
    drainWaitNs = {};
}

namespace {

PassRunLimits passRunLimits() {
    static const PassRunLimits limits = [] {
        PassRunLimits configured;
        if (const char* text = std::getenv("APS5_PASS_RUN_US")) configured.budget = std::chrono::microseconds(std::clamp<unsigned long long>(std::strtoull(text, nullptr, 10), 10, 20000));
        if (const char* text = std::getenv("APS5_PASS_RUN_DRAWS")) configured.maxDraws = static_cast<std::size_t>(std::clamp<unsigned long long>(std::strtoull(text, nullptr, 10), 2, 4096));
        return configured;
    }();
    return limits;
}

[[noreturn]] void passRunVerifyFailed(const char* what) {
    std::fprintf(stderr, "[parcommit] APS5_VERIFY_PARALLEL_COMMIT: %s\n", what);
    std::fflush(stderr);
    std::abort();
}

}

void DrawPipeline::run() {
    AgcDriver::JoinGamesSchedulingClass();
    GuestMemory::SetCollectRole(1);
    Graphics::Recorder::MarkCommitterThread();
    static const auto holdBudget = [] {
        const char* text = std::getenv("APS5_COMMIT_HOLD_US");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 300ull;
        return std::chrono::microseconds(std::clamp<unsigned long long>(value, 10, 20000));
    }();
    std::vector<ForkJob*> forkJobs;
    std::unique_lock<GuestMemory::GpuMutexType> hold(GuestMemory::GpuMutex(), std::defer_lock);
    auto heldSince = std::chrono::steady_clock::time_point{};
    // Live `parallel_commit` pass runs (PassRun.hpp): while a run is active the committer holds
    // `hold` across its commits whoever waits for the mutex, and defers its starve check to the
    // run's end. Only this thread touches the tracker; its statistics move to `passRunStats`.
    PassRunTracker passRun;
    passRun.SetLimits(passRunLimits());
    const auto runItem = [](const Item& item) { return PassRunItem{item.fork != nullptr, item.fork != nullptr ? item.fork->passHint : 0}; };
    const auto publishRunStatistics = [&] {
        const auto taken = passRun.TakeStatistics();
        std::lock_guard lock(mutex);
        passRunStats.runs += taken.runs;
        passRunStats.draws += taken.draws;
        passRunStats.contendedContinuations += taken.contendedContinuations;
        passRunStats.deferredStarveChecks += taken.deferredStarveChecks;
        passRunStats.cooldowns += taken.cooldowns;
        for (std::size_t index = 0; index < passRunStats.ends.size(); ++index) passRunStats.ends[index] += taken.ends[index];
    };
    for (;;) {
        if (hold.owns_lock() && !passRun.Active() && (outstanding.load(std::memory_order_acquire) == 0 || GuestMemory::GpuMutex().Contenders() != 0 || std::chrono::steady_clock::now() - heldSince > holdBudget || !Graphics::CommitHoldEnabled())) hold.unlock();
        if (!hold.owns_lock() && outstanding.load(std::memory_order_acquire) == 0) {
            committerIdle.store(true, std::memory_order_relaxed);
            SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
            committerIdle.store(false, std::memory_order_relaxed);
        }
        Commit* commit = nullptr;
        std::uint64_t sequence = 0;
        std::shared_ptr<ReplaySegment> segment;
        bool stream = false;
        forkJobs.clear();
        PassRunItem front;
        std::optional<PassRunItem> next;
        {
            std::unique_lock lock(mutex);
            committerWaiting = true;
            if (items.empty()) {
                const auto state = std::min<std::size_t>(ProducerState().load(std::memory_order_relaxed), idleNs.size() - 1);
                const auto idleStart = std::chrono::steady_clock::now();
                PipeTrace(Trace::CommitIdle);
                committerIdle.store(true, std::memory_order_relaxed);
                wake.wait(lock, [&] { return !items.empty(); });
                committerIdle.store(false, std::memory_order_relaxed);
                PipeTrace(Trace::CommitIdleEnd);
                idleNs[state] += elapsedNs(idleStart);
            }
            committerWaiting = false;
            commit = &items.front().commit;
            sequence = items.front().sequence;
            segment = items.front().segment;
            front = runItem(items.front());
            if (items.size() > 1) next = runItem(items[1]);
            gatherForkRun(forkJobs);
            stream = forkRunner.load(std::memory_order_acquire) != nullptr && items.front().fork != nullptr && items.front().fork->forFork && items.front().fork->eligible && Graphics::ForkStreamEnabled();
        }
        // A pass run starts at a draw that the queued draw behind it continues, and goes on while
        // AfterCommit lets it; the flag switched off ends it at once.
        const bool parallelCommit = Graphics::ParallelCommitEnabled();
        if (!parallelCommit && passRun.Active()) {
            passRun.Abandon(PassRunEnd::FlagOff);
            publishRunStatistics();
        }
        const bool inRun = parallelCommit && passRun.BeforeCommit(front, next, std::chrono::steady_clock::now());
        if (inRun && !hold.owns_lock()) {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
            hold.lock();
            heldSince = std::chrono::steady_clock::now();
        }
        const bool verify = inRun && Graphics::VerifyParallelCommit();
        const bool runHead = inRun && passRun.Committed() == 0;
        const auto othersBefore = verify ? Graphics::Recorder::PassEndsByOtherThreads() : 0;
        if (verify) {
            if (!GuestMemory::GpuMutex().HeldByThisThread()) passRunVerifyFailed("a pass run commits without holding the GPU mutex");
            static_cast<void>(Graphics::TakeDrawPassOutcome());
        }
        // After `count` commits: the run goes on only with the next queued draw of its pass while
        // the recorder's pass stays continuable (read under `hold`), within its budget.
        const auto afterCommits = [&](std::size_t count) {
            if (!passRun.Active()) return;
            if (verify) {
                if (!GuestMemory::GpuMutex().HeldByThisThread()) passRunVerifyFailed("a pass run lost the GPU mutex during a commit");
                if (Graphics::Recorder::PassEndsByOtherThreads() != othersBefore) passRunVerifyFailed("another thread ended the render pass inside a pass run");
                const auto outcome = Graphics::TakeDrawPassOutcome();
                std::lock_guard lock(mutex);
                if (count == 1 && !runHead && outcome != 0) ++(outcome == 1 ? verifiedContinued : verifiedBegun);
                if (runHead) ++verifiedRuns;
            }
            std::optional<PassRunItem> queued;
            {
                std::lock_guard lock(mutex);
                if (!items.empty()) queued = runItem(items.front());
            }
            auto* recorder = Graphics::Recorder::Active();
            const bool continuable = recorder != nullptr && recorder->PassContinuable();
            static_cast<void>(passRun.AfterCommit(count, queued, continuable, GuestMemory::GpuMutex().Contenders() != 0, std::chrono::steady_clock::now()));
            if (!passRun.Active()) publishRunStatistics();
        };
        // APS5_STARVE_SUBMIT=<commands> (experiment): between commits, the open batch goes to the GPU
        // once it holds that many commands and every submitted batch has completed.
        static const std::uint64_t starveSubmit = [] {
            const char* text = std::getenv("APS5_STARVE_SUBMIT");
            return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        }();
        const auto starveCheck = [&] {
            if (starveSubmit == 0) return;
            // Inside a pass run the submit waits for the run's end: it would end the render pass.
            if (passRun.Active()) return;
            if (auto* recorder = Graphics::Recorder::Active(); recorder != nullptr && Graphics::Recorder::RecordedWorkSinceSubmit() >= starveSubmit) {
                std::unique_lock<GuestMemory::GpuMutexType> submitLock(GuestMemory::GpuMutex(), std::defer_lock);
                if (!hold.owns_lock()) submitLock.lock();
                // Nothing left to commit: the committer goes idle next, so a young batch is not
                // held for its age (Recorder::StarveAge) with nothing behind it to submit it.
                if (recorder->SubmitIfStarved(starveSubmit, outstanding.load(std::memory_order_acquire) == 0)) ++starvedSubmits;
            }
        };
        if (stream) {
            if (const auto handled = runStream(); handled != 0) {
                // A stream window commits across passes (DrawForkRunner::Stream schedules ahead
                // within a pass but keeps committing behind it): it ends the run.
                if (handled > 1 && passRun.Active()) {
                    passRun.Abandon(PassRunEnd::Stream);
                    publishRunStatistics();
                }
                afterCommits(handled);
                // With starve_age a batch held for its age must not be left behind a forked run.
                if (Graphics::Recorder::StarveAge().count() != 0) starveCheck();
                continue;
            }
        } else if (forkJobs.size() >= 2) {
            if (const auto handled = runForked(forkJobs); handled != 0) {
                afterCommits(handled);
                if (Graphics::Recorder::StarveAge().count() != 0) starveCheck();
                continue;
            }
        }
        if (!hold.owns_lock() && Graphics::CommitHoldEnabled()) {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
            hold.lock();
            heldSince = std::chrono::steady_clock::now();
        }
        static_cast<void>(runCommit(*commit, sequence, std::move(segment)));
        afterCommits(1);
        starveCheck();
    }
}

bool DrawPipeline::runCommit(const Commit& commit, std::uint64_t sequence, std::shared_ptr<ReplaySegment> segment) {
    const auto start = std::chrono::steady_clock::now();
    // [slow-commit] (APS5_PROFILE_DRAW): a commit over 5 ms with what it waited for, at most 25
    // lines per 10 s; a slow commit stalls every preparation queued behind it.
    struct SlowCommit {
        std::chrono::steady_clock::time_point start;
        const Commit& commit;
        std::pair<double, std::uint64_t> syncs = Graphics::ThreadHookSyncs();
        double locked = GuestMemory::ThreadLockedGpuWaitMs();
        double mutex = GuestMemory::ThreadGpuLockWaitMs();
        ~SlowCommit() {
            static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
            if (!profile) return;
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (ms < 5.0) return;
            static std::atomic<std::int64_t> windowStart{0};
            static std::atomic<int> told{0};
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            auto began = windowStart.load(std::memory_order_relaxed);
            if (now - began > 10'000'000'000ll && windowStart.compare_exchange_strong(began, now)) told.store(0, std::memory_order_relaxed);
            if (told.fetch_add(1, std::memory_order_relaxed) >= 25) return;
            const auto after = Graphics::ThreadHookSyncs();
            const char* kind = commit.target_type().name();
            const auto length = std::strlen(kind);
            std::fprintf(stderr, "[slow-commit] %.1f ms (GPU-sync %.1f ms in %llu, locked GPU waits %.1f ms, GPU mutex %.1f ms) kind ...%s\n", ms, after.first - syncs.first, static_cast<unsigned long long>(after.second - syncs.second), GuestMemory::ThreadLockedGpuWaitMs() - locked, GuestMemory::ThreadGpuLockWaitMs() - mutex, length > 90 ? kind + length - 90 : kind);
        }
    } slowCommit{start, commit};
    std::exception_ptr fatal;
    bool failed = false;
    PipeTrace(Trace::Commit);
    CommittingSequence() = sequence;
    try {
        const ReplayProbe::Scope probeScope(segment);
        commit();
    } catch (const std::exception& error) {
        failed = true;
        static std::atomic<std::uint64_t> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 20) std::fprintf(stderr, "[draw] pipelined commit failed: %.200s\n", error.what());
    } catch (...) {
        fatal = std::current_exception();
    }
    CommittingSequence() = 0;
    PipeTrace(Trace::CommitEnd);
    const auto spent = elapsedNs(start);
    if (spent >= 1'000'000 && Graphics::CpuTimeline::Active()) Graphics::CpuTimeline::Event("commit", start, start + std::chrono::nanoseconds(spent), sequence, 0);
    if (CommitKinds::Enabled()) commitKinds.Note(commit.target_type().name(), spent);
    if (segment != nullptr) ReplayProbe::Committed(*segment, spent, failed || fatal != nullptr);
    std::lock_guard lock(mutex);
    commitNsTotal += spent;
    if (Graphics::DrawTokensEnabled()) {
        std::lock_guard appliedLock(appliedMutex);
        if (applied.size() != AppliedCapacity) applied.resize(AppliedCapacity);
        auto& slot = applied[static_cast<std::size_t>(sequence % AppliedCapacity)];
        slot.sequence = sequence;
        slot.unknown = items.front().unknown;
        slot.writes = std::move(items.front().writes);
        appliedFrontier.store(sequence, std::memory_order_release);
    }
    static const bool retireOffCommitter = std::getenv("APS5_NO_RETIRED_COMMITS") == nullptr;
    if (retireOffCommitter && retired.size() < 4096) retired.push_back(std::move(items.front().commit));
    items.pop_front();
    ++commits;
    commitNs += spent;
    if (failed) ++commitErrors;
    if (fatal != nullptr && failure == nullptr) failure = fatal;
    outstanding.fetch_sub(1, std::memory_order_release);
    if (idleWaiters != 0) idle.notify_all();
    return !failed && fatal == nullptr;
}

void DrawPipeline::gatherForkRun(std::vector<ForkJob*>& jobs) {
    const bool stats = forkStatsOn();
    const bool build = forkRunner.load(std::memory_order_acquire) != nullptr && Graphics::ForkBuildEnabled();
    if (!stats && !build) return;
    const auto& front = items.front();
    if (front.fork == nullptr || !front.fork->forFork || !front.fork->eligible) {
        lastHinted = false;
        return;
    }
    const auto hint = front.fork->passHint;
    const bool continuing = lastHinted && lastHint == hint;
    const auto limit = ForkRunLimit();
    std::size_t length = 0;
    for (const auto& item : items) {
        if (item.fork == nullptr || !item.fork->forFork || !item.fork->eligible || item.fork->passHint != hint) break;
        if (build && continuing && jobs.size() < limit) jobs.push_back(item.fork.get());
        ++length;
        if (!stats && (!continuing || jobs.size() >= limit)) break;
    }
    if (stats && !continuing) {
        std::size_t bucket = 0;
        while (bucket + 1 < runLengths.size() && length >= (std::size_t{2} << bucket)) ++bucket;
        ++runLengths[bucket];
        ++runStarts;
    }
    lastHinted = true;
    lastHint = hint;
}

std::size_t DrawPipeline::runForked(std::span<ForkJob* const> jobs) {
    auto* runner = forkRunner.load(std::memory_order_acquire);
    if (runner == nullptr) return 0;
    std::size_t committed = 0;
    std::size_t handled = 0;
    try {
        handled = runner->Run(jobs, [this, &committed](std::size_t, const Commit* body) {
            const Commit* own = nullptr;
            std::uint64_t sequence = 0;
            std::shared_ptr<ReplaySegment> segment;
            {
                std::lock_guard lock(mutex);
                own = &items.front().commit;
                sequence = items.front().sequence;
                segment = items.front().segment;
            }
            ++committed;
            return runCommit(body != nullptr ? *body : *own, sequence, std::move(segment));
        });
    } catch (...) {
        handled = committed;
    }
    std::lock_guard lock(mutex);
    ++forkRuns;
    forkRunItems += jobs.size();
    forkHandled += handled;
    return handled;
}

std::size_t DrawPipeline::runStream() {
    auto* runner = forkRunner.load(std::memory_order_acquire);
    if (runner == nullptr) return 0;
    std::size_t committed = 0;
    std::size_t handled = 0;
    try {
        handled = runner->Stream([this](std::size_t limit) {
            std::vector<std::shared_ptr<ForkJob>> jobs;
            std::lock_guard lock(mutex);
            for (const auto& item : items) {
                if (item.fork == nullptr || !item.fork->forFork || jobs.size() >= limit) break;
                jobs.push_back(item.fork);
            }
            return jobs;
        }, [this, &committed](const Commit* body) {
            const Commit* own = nullptr;
            std::uint64_t sequence = 0;
            std::shared_ptr<ReplaySegment> segment;
            {
                std::lock_guard lock(mutex);
                own = &items.front().commit;
                sequence = items.front().sequence;
                segment = items.front().segment;
            }
            ++committed;
            return runCommit(body != nullptr ? *body : *own, sequence, std::move(segment));
        });
    } catch (...) {
        handled = committed;
    }
    return handled;
}

std::string DrawPipeline::TakeForkRuns() {
    std::lock_guard lock(mutex);
    constexpr std::array<const char*, 7> names{"1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+"};
    std::string line = "visible runs " + std::to_string(runStarts) + " (";
    for (std::size_t index = 0; index < runLengths.size(); ++index) line += std::string(index == 0 ? "" : " ") + names[index] + ":" + std::to_string(runLengths[index]);
    line += "), forked runs " + std::to_string(forkRuns) + " of " + std::to_string(forkRunItems) + " draws, " + std::to_string(forkHandled) + " committed by the runner";
    runLengths = {};
    runStarts = forkRuns = forkRunItems = forkHandled = 0;
    return line;
}

void DrawPipeline::Drain(DrainReason reason, std::uint32_t opcode) {
    if (!Busy()) return;
    const ProducerScope producer(Producer::Draining);
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    const auto start = std::chrono::steady_clock::now();
    if (!OnPrepThread()) SpinUntil(HandoffSpinWindow() * 2, [&] { return !Busy(); });
    drainPrep();
    std::unique_lock lock(mutex);
    ++idleWaiters;
    idle.wait(lock, [&] { return items.empty(); });
    --idleWaiters;
    const auto index = static_cast<std::size_t>(reason);
    ++drains[index];
    ++drainOpcodes[std::min<std::uint32_t>(opcode, 0x100)];
    drainWaitNs[index] += elapsedNs(start);
    threadDrainWaitMs += static_cast<double>(elapsedNs(start)) / 1e6;
    rethrowFailure();
}

bool DrawPipeline::DrainUnless(DrainReason reason, std::uint32_t opcode, const std::function<bool()>& ready) {
    if (!Busy()) return false;
    const ProducerScope producer(Producer::Draining);
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    const auto start = std::chrono::steady_clock::now();
    constexpr auto poll = std::chrono::microseconds(100);
    bool early = false;
    for (;;) {
        // The predicate runs without the pipeline's locks (it reads labels and guest memory).
        if (ready()) {
            early = true;
            break;
        }
        if (prepOutstanding.load(std::memory_order_acquire) != 0) {
            std::unique_lock prepLock(prepMutex);
            prepIdle.wait_for(prepLock, poll, [&] { return prepOutstanding.load(std::memory_order_relaxed) == 0; });
            continue;
        }
        std::unique_lock lock(mutex);
        if (items.empty()) break;
        ++idleWaiters;
        idle.wait_for(lock, poll, [&] { return items.empty(); });
        --idleWaiters;
        if (items.empty()) break;
    }
    std::lock_guard lock(mutex);
    const auto index = static_cast<std::size_t>(reason);
    if (early) {
        ++earlyWaits;
        earlyWaitNs += elapsedNs(start);
    } else {
        ++drains[index];
        ++drainOpcodes[std::min<std::uint32_t>(opcode, 0x100)];
        drainWaitNs[index] += elapsedNs(start);
    }
    rethrowFailure();
    return early;
}

std::uint64_t DrawPipeline::CommitNsTotal() {
    std::lock_guard lock(mutex);
    return commitNsTotal;
}

std::uint64_t& DrawPipeline::CommittingSequence() {
    static thread_local std::uint64_t sequence = 0;
    return sequence;
}

DrawPipeline::Window DrawPipeline::CheckWindow(std::uint64_t fromExclusive, std::uint64_t toExclusive, std::span<const Range> reads) {
    if (toExclusive == 0 || toExclusive <= fromExclusive) return Window::Overflow;
    if (toExclusive - fromExclusive - 1 > AppliedCapacity) return Window::Overflow;
    std::lock_guard lock(appliedMutex);
    if (applied.size() != AppliedCapacity || appliedFrontier.load(std::memory_order_relaxed) + 1 < toExclusive) return Window::Overflow;
    // Live `window_early_out` (env APS5_WINDOW_EARLY_OUT=0 turns it off): see ScanWindow. Sampling
    // the World Map committer put this scan (every write of every slot against every read) at about
    // 2.5% of Graphics::Draw.
    static const bool earlyOutDefault = [] {
        const char* value = std::getenv("APS5_WINDOW_EARLY_OUT");
        return value == nullptr || value[0] != '0';
    }();
    const bool earlyOut = Graphics::LiveFlag("window_early_out", earlyOutDefault);
    return ScanWindow(fromExclusive, toExclusive, [&](std::uint64_t sequence) -> const Applied& { return applied[static_cast<std::size_t>(sequence % AppliedCapacity)]; }, reads, earlyOut);
}

std::optional<std::uint64_t> DrawPipeline::PendingLabel(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > 8) return std::nullopt;
    const auto end = address + bytes;
    if (prepOutstanding.load(std::memory_order_acquire) != 0 && !OnPrepThread()) {
        // The newest writer decides: jobs still held by the preparation come after every queued item.
        std::lock_guard prepLock(prepMutex);
        for (auto entry = prepWrites.rbegin(); entry != prepWrites.rend(); ++entry) {
            const bool overlaps = std::any_of(entry->writes.begin(), entry->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
            if (!overlaps) continue;
            if (entry->labelBytes.empty() || address < entry->labelAddress || end > entry->labelAddress + entry->labelBytes.size()) return std::nullopt;
            std::uint64_t value = 0;
            std::memcpy(&value, entry->labelBytes.data() + (address - entry->labelAddress), bytes);
            return value;
        }
    }
    std::lock_guard lock(mutex);
    for (auto item = items.rbegin(); item != items.rend(); ++item) {
        const bool overlaps = std::any_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) continue;
        if (item->labelBytes.empty() || address < item->labelAddress || end > item->labelAddress + item->labelBytes.size()) return std::nullopt;
        std::uint64_t value = 0;
        std::memcpy(&value, item->labelBytes.data() + (address - item->labelAddress), bytes);
        return value;
    }
    return std::nullopt;
}

std::pair<std::uint64_t, std::uint64_t> DrawPipeline::OverlapRange(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    std::lock_guard lock(mutex);
    for (const auto& item : items) {
        for (const auto& [begin, limit] : item.writes) {
            if (address < limit && begin < end) return {begin, limit};
        }
    }
    return {0, 0};
}

bool DrawPipeline::Overlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    if (prepOutstanding.load(std::memory_order_acquire) != 0) {
        std::lock_guard prepLock(prepMutex);
        const auto own = PrepSequence();
        for (const auto& entry : prepWrites) {
            if (own != 0 && entry.sequence >= own) continue;
            for (const auto& [begin, limit] : entry.writes) {
                if (address < limit && begin < end) return true;
            }
        }
    }
    std::lock_guard lock(mutex);
    for (const auto& item : items) {
        for (const auto& [begin, limit] : item.writes) {
            if (address < limit && begin < end) return true;
        }
    }
    return false;
}

}
