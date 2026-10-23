#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RECORDER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RECORDER_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <array>
#include <deque>
#include <mutex>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

class Buffer;

struct ForkRefusal {
    const char* site;
    bool backstop;
};

class ForkLog {
public:
    enum class Effect : std::uint8_t { Keep, Complete, Read, Reads, Snapshot, Touch, Erase, Sampled, Counter, Depth, Count };
    struct Entry {
        Effect kind;
        std::uint64_t address;
        std::uint64_t bytes;
    };
    explicit ForkLog(bool witness = false, std::size_t slot = 0) : witness(witness), slot(slot) {}
    void Defer(Effect kind, std::uint64_t address, std::uint64_t bytes, std::function<void()> effect);
    void Note(Effect kind, std::uint64_t address, std::uint64_t bytes);
    void Expect(std::function<bool()> check);
    bool Holds() const;
    void Replay();
    const std::vector<Entry>& Entries() const { return entries; }
    std::size_t Effects() const { return effects.size(); }
    const bool witness;
    const std::size_t slot;

private:
    std::vector<Entry> entries;
    std::vector<std::function<void()>> effects;
    std::vector<std::function<bool()>> checks;
};

class ForkScope {
public:
    explicit ForkScope(ForkLog& log);
    ~ForkScope();
    ForkScope(const ForkScope&) = delete;
    ForkScope& operator=(const ForkScope&) = delete;

private:
    ForkLog* previous;
};

ForkLog* ForkLogged();
ForkLog* ForkWitnessed();
class ForkQuiesceScope {
public:
    explicit ForkQuiesceScope(const std::function<void(const char*)>* hook);
    ~ForkQuiesceScope();
    ForkQuiesceScope(const ForkQuiesceScope&) = delete;
    ForkQuiesceScope& operator=(const ForkQuiesceScope&) = delete;

private:
    const std::function<void(const char*)>* previous;
};
bool ForkQuiescing();
void ForkQuiesce(const char* reason);
void SetForkPausePending(bool pending);
bool ForkPausePending();
void CountForkRefusal(const char* site, bool backstop = false);
[[noreturn]] void ForkRefuse(const char* site, bool backstop = false);
inline void ForkTrap(const char* site) {
    if (ForkLogged() != nullptr) ForkRefuse(site);
}
// A contended ForkLock outside a fork: spins for the live `lock_spin` window before blocking
// (Recorder.cpp; HandoffSpin.hpp LockSpinWindow).
void LockContended(std::mutex& mutex);
inline void ForkLock(std::mutex& mutex) {
    if (ForkLogged() == nullptr) {
        if (!mutex.try_lock()) LockContended(mutex);
        return;
    }
    for (std::uint32_t spin = 0; !mutex.try_lock(); ++spin) {
        if (ForkPausePending()) ForkRefuse("paused");
        if (spin > 64) std::this_thread::yield();
    }
}
class ForkGuard {
public:
    explicit ForkGuard(std::mutex& mutex) : mutex(mutex) { ForkLock(mutex); }
    ~ForkGuard() { mutex.unlock(); }
    ForkGuard(const ForkGuard&) = delete;
    ForkGuard& operator=(const ForkGuard&) = delete;

private:
    std::mutex& mutex;
};
struct ForkRefusalCounts {
    std::vector<std::pair<std::string, std::uint64_t>> sites;
    std::uint64_t total = 0;
    std::uint64_t backstops = 0;
};
ForkRefusalCounts TakeForkRefusals();

// Accumulates GPU work across guest commands so the CPU does not wait for each one. Dispatches and the
// copies that feed them record into one open batch; Submit sends it to the queue without waiting and
// Sync waits for every batch, then runs its completion actions (write-backs) in order. Objects handed
// to Keep live until the batch that recorded them completed. Guest memory ranges the recorded work
// will write are noted, so a CPU access to such a range (the GuestMemory flush hook) syncs first.
// Every call happens under GuestMemory::GpuMutex, except WaitSerial and the static lock-free readers
// below. The noted ranges are also published as an immutable snapshot (rebuilt under the mutex at
// every note and after a batch's completions ran) that the flush hook reads without the mutex, so
// accesses overlapping nothing never wait behind device work.
//
// Submissions are numbered (serials) and, when the device has timeline semaphores, every Submit
// signals a timeline semaphore with its serial: a thread can then wait for "everything submitted
// up to serial S" WITHOUT the mutex (WaitSerial), because the monotonic value cannot be reset or
// recycled the way the pooled batch fences are, and afterwards run the completions under the mutex
// (FinishUpTo). Without timeline semaphores callers use the locked Sync as before.
// The calling thread's GPU-sync waits in the flush hook so far (ms, count; profiling only).
std::pair<double, std::uint64_t> ThreadHookSyncs();

class Recorder {
public:
    // `timelineSemaphores`: the device enabled VK_KHR_timeline_semaphore (WaitSerial is usable).
    explicit Recorder(const Context& context, bool timelineSemaphores = false);
    ~Recorder();
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    // The open batch's command buffer, starting a batch when none is open. `coveredAccess`, when
    // given, receives the destination access mask of the last recorded command's trailing barrier
    // when that barrier had ALL_COMMANDS as its destination stage (MarkCovered), else 0: every
    // write recorded so far is then available and visible to those accesses of any later stage,
    // so a dispatch, fill or copy whose leading barrier asks for a subset of them may skip it
    // (the [barriers] line counts the merges). The call clears the mask (whatever the caller
    // records is assumed uncovered until it marks); a new batch starts uncovered.
    VkCommandBuffer Commands(VkAccessFlags* coveredAccess = nullptr);
    VkCommandBuffer OpenBatch();
    // After a trailing barrier with ALL_COMMANDS as its destination stage: `access` is its
    // destination access mask.
    void MarkCovered(VkAccessFlags access);
    void MarkShaderReadsCovered() { MarkCovered(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT); }
    // A recorded draw's render pass (Draw.cpp) is left open after the draw: the next draw of the
    // same attachments (`key`: the views and the extent) continues it when nothing was recorded in
    // between and the earlier draw allowed it (`continuable`: it wrote nothing but its
    // attachments, so no barrier is owed inside the pass), and anything else recorded first ends
    // it (vkCmdEndRenderPass, one trailing barrier for the pass, the end of the draw class range
    // `timing`): Commands(), RecordStore and Submit end it; CommandsInRenderPass hands a
    // continuing draw the command buffer without ending it.
    bool ContinuesRenderPass(std::uint64_t key) const;
    VkCommandBuffer CommandsInRenderPass();
    void LeaveRenderPassOpen(std::uint64_t key, std::uint32_t timing, bool continuable, std::function<void(VkCommandBuffer)> afterPass = {});
    // DCC "uncompressed" key stores (DccMetadata.cpp StoreUncompressedOnGpu): queued on the open
    // batch and recorded as one run (one barrier pair for every queued fill) at Submit, before a
    // label store (RecordStore), or before a command that writes or reads a queued range (the
    // fill must precede a later key writer or reader in the batch: FlushKeyStores by the callers
    // that know their ranges). `seed` (kept) supplies an
    // unaligned head or tail; [begin, end) is the guest range. Debug aid: APS5_DCC_KEYS_EACH=1
    // records every store at once, as before (QueueKeyStore then records and returns).
    void QueueKeyStore(VkBuffer buffer, VkDeviceSize first, VkDeviceSize last, std::shared_ptr<void> seed, std::uint64_t begin, std::uint64_t end);
    bool HasQueuedKeyStores() const { return open != nullptr && !open->keyStores.empty(); }
    bool QueuedKeyStoreOverlaps(std::uint64_t address, std::size_t bytes) const;
    // Whether `overlaps(begin, end)` holds for a queued store's range.
    bool AnyQueuedKeyStore(const std::function<bool(std::uint64_t, std::uint64_t)>& overlaps) const;
    void FlushKeyStores();
    void FlushKeyStoresOverlapping(std::uint64_t address, std::size_t bytes) { if (HasQueuedKeyStores() && QueuedKeyStoreOverlaps(address, bytes)) FlushKeyStores(); }
    bool Recording() const { return open != nullptr; }
    bool PassContinuable() const { return open != nullptr && open->renderPass.open && open->renderPass.continuable; }
    std::uint64_t WriteNotes() const { return writeNoteCount; }
    std::uint64_t PassGeneration() const { return passGeneration; }
    std::uint64_t FinishedBatches() const { return finishedBatches; }
    std::size_t QueuedKeyStoreCount() const { return open != nullptr ? open->keyStores.size() : 0; }
    std::size_t QueuedStoreCount() const { return open != nullptr ? open->run.queued.size() : 0; }
    bool Idle() const { return open == nullptr && inFlight.empty(); }
    // Whether recorded work still has completion actions (write-backs the CPU must see) to run.
    bool HasCompletions() const;
    void Keep(std::shared_ptr<void> object, std::size_t bytes = 0);
    static constexpr std::size_t KeptBytesBudget = std::size_t{512} << 20u;
    void BoundKeptBytes();
    std::size_t InFlightKeptBytes() const { return inFlightKeptBytes; }
    enum class SnapshotUse : std::uint8_t { Storage, Vertex, Index16, Index32 };
    static constexpr std::size_t DrawSnapshotBudget = std::size_t{256} << 20u;
    static constexpr std::size_t DrawSnapshotEntries = 1024;
    static constexpr std::size_t DrawInputBudget = std::size_t{1024} << 20u;
    static constexpr std::size_t DrawInputEntries = 16384;
    std::shared_ptr<Buffer> ReusableDrawSnapshot(std::uint64_t address, std::size_t bytes, SnapshotUse use = SnapshotUse::Storage, std::uint32_t* derived = nullptr, VkDeviceSize* offset = nullptr);
    void KeepDrawSnapshot(std::uint64_t address, std::size_t bytes, std::uint64_t generation, std::uint64_t registryGeneration, std::shared_ptr<Buffer> buffer, SnapshotUse use = SnapshotUse::Storage, std::uint32_t derived = 0, VkDeviceSize offset = 0);
    struct SnapshotSlice {
        std::shared_ptr<Buffer> buffer;
        VkDeviceSize offset = 0;
    };
    static bool SnapshotRingEnabled();
    SnapshotSlice AllocateDrawSnapshot(std::size_t bytes);
    void OnComplete(std::function<void()> action);
    void NotePendingWrite(std::uint64_t address, std::size_t bytes);
    void NotePendingFill(std::uint64_t address, std::size_t bytes, std::uint8_t value);
    // Notes several [begin, end) ranges and publishes the snapshot once (a dispatch writes many buffers).
    void NotePendingWrites(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
    bool PendingWriteOverlaps(std::uint64_t address, std::size_t bytes) const;
    bool PendingWriteOverlapsScreened(std::uint64_t address, std::size_t bytes) const;
    // Whether no unfinished batch writes the range any more: no open-batch overlap, and every
    // overlapping in-flight batch's fence has signaled (its in-place GPU stores are final in host
    // memory; its completion stores are the caller's business, see VulkanDevice::CopyBuffer).
    bool PendingWriteSettled(std::uint64_t address, std::size_t bytes) const;
    std::uint64_t LastWriteNote(std::uint64_t address, std::size_t bytes) const;
    std::uint64_t NewestWriteNote(std::uint64_t address, std::size_t bytes) const;
    struct PendingWrite {
        std::uint64_t begin;
        std::uint64_t end;
        std::uint64_t note;
        int value;
    };
    std::vector<PendingWrite> PendingWritesOver(std::uint64_t address, std::size_t bytes) const;
    // Batch read tracking: the guest ranges the recorded work reads IN PLACE through a host import
    // and the GPU has not executed yet (a V# element bound in place, a region the GPU copies out of
    // an import, an address-based build's leased heaps, indirect arguments, a GPU-direct storage
    // upload, the copy HLE's source). A CPU store into such a range before the batch ran would be
    // seen by the reader early (WAR); VulkanDevice::CopyBuffer refuses its CPU path on a hit. Noted
    // on the open batch, no snapshot: the only reader holds the mutex. `kind` names the reader for
    // the [recorder] hit counters. APS5_COPY_READ_TRACKING=0 notes nothing (ReadTracking() is then
    // false and the CPU copy falls back to Idle()).
    enum class ReadKind : std::uint8_t { DispatchElement = 0, GpuCopy, AddressBased, Indirect, StorageUpload, CopySource, DrawInput, Count };
    static bool ReadTracking();
    void NotePendingRead(std::uint64_t address, std::size_t bytes, ReadKind kind);
    void NotePendingReads(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, ReadKind kind);
    // As NotePendingReads for a large, shared set of sorted and disjoint ranges (an address space's
    // in-place reads): the batch keeps a reference and searches it instead of copying the ranges.
    void NoteSharedReads(std::shared_ptr<const std::vector<std::pair<std::uint64_t, std::uint64_t>>> sorted, ReadKind kind);
    // Whether the open batch (opened now if needed) already noted the read set `serial` (an
    // address space's leased regions, identical for every build sharing it); marks it noted. The
    // caller notes the set's ranges only on false. Always false with read tracking off.
    bool ReadSetNoted(std::uint64_t serial);
    // Whether an unexecuted batch (open, or in flight with its fence unsignaled) reads the range in
    // place. A hit in an in-flight batch whose fence signaled meanwhile is a miss (one status
    // query per hit). `ignoreSignaled` false: every in-flight batch counts, whatever its fence.
    bool PendingReadOverlaps(std::uint64_t address, std::size_t bytes, bool ignoreSignaled = true) const;
    // The newest batch whose noted read overlaps the range (for the [copyhle] trace and the verify
    // switch): its serial (the open batch reports the serial it will get), recording queue, reader
    // kind, and whether its fence has signaled. Nullopt: no overlap.
    struct PendingReadInfo {
        std::uint64_t serial;
        std::uint32_t queue;
        ReadKind kind;
        bool open;
        bool signaled;
    };
    std::optional<PendingReadInfo> DescribePendingRead(std::uint64_t address, std::size_t bytes) const;
    // Counters of the read tracking (cumulative, for the [recorder] line): ranges noted, overlap
    // queries, hits by reader kind, and hits ignored because the reader's fence had signaled.
    struct ReadStatistics {
        std::uint64_t noted, queries, staleIgnored;
        std::array<std::uint64_t, static_cast<std::size_t>(ReadKind::Count)> hits;
    };
    static ReadStatistics ReadCounts();
    // Whether the OPEN (unsubmitted) batch writes the range: a wait on such a range must submit it.
    bool OpenWriteOverlaps(std::uint64_t address, std::size_t bytes) const;
    // What SyncThrough(address, bytes) would wait for, for the [hooksync] attribution: the newest
    // batch whose noted range overlaps the access (the open batch counts first, with the serial it
    // will get), whether that batch's fence has already signaled (its GPU work is done, so the
    // access's bytes are already final; never for the open batch), that noted range, and how many
    // batches the wait finishes. Nullopt: no overlap.
    struct PendingWriteInfo {
        std::uint64_t serial;
        bool open;
        bool signaled;
        std::uint64_t rangeBegin;
        std::uint64_t rangeEnd;
        std::size_t batchesToFinish;
    };
    std::optional<PendingWriteInfo> DescribePendingWrite(std::uint64_t address, std::size_t bytes) const;
    // Why a batch went to the queue ([submits] line, cpu-timeline.txt). A caller announces the
    // reason for the Submit this thread makes next (AnnounceSubmit; the Submit consumes it); a
    // sync that submits without an announcement takes the reason of its CountSync source, and any
    // other Submit counts as Other with its call site kept for the [submits] line's site table.
    enum class SubmitReason : std::uint8_t {
        Other = 0, Starve, LabelDeadline, BatchCap, PacketFlush, Label, Eop, SubmissionEnd, Suspend, Flip, Present, Wait, WaitDeadline, Drain, Presync,
        SyncIdle, SyncPendingWrite, SyncRecordedStore, SyncAddressBased, SyncOther, KeptBytes, Immediate, Lease, Count
    };
    static constexpr std::size_t SubmitReasonCount = static_cast<std::size_t>(SubmitReason::Count);
    static const char* SubmitReasonName(SubmitReason reason);
    // `site`: the code address to name when the reason is Other (null: the Submit's caller).
    static void AnnounceSubmit(SubmitReason reason, const void* site = nullptr);
    // Submit reasons and starvation are tracked: APS5_PROFILE_DRAW, APS5_PROFILE_SUBMITS=1 or the
    // GPU timeline window. Otherwise Submit records the reason only (no status query).
    static bool SubmitProfiled();
    // Cumulative counts (all recorders): batches by reason, and of them those submitted while the
    // GPU had nothing queued (the newest earlier batch's fence had signaled: it went idle waiting
    // for this submit), plus the top unannounced call sites (module offsets) with their counts.
    struct SubmitStatistics {
        std::array<std::uint64_t, SubmitReasonCount> batches{};
        std::array<std::uint64_t, SubmitReasonCount> starved{};
        std::vector<std::pair<std::uint64_t, std::uint64_t>> otherSites;
    };
    static SubmitStatistics SubmitCounts();
    // Why the render pass a draw left open ended (the [parcommit] line, APS5_PROFILE_DRAW; see
    // parcommit-design.md): a submit (split by its SubmitReason), the next draw itself (other
    // attachments, a pass the previous draw left not continuable, a draw reading its own targets,
    // one that records outside a pass first, the queued stores a draw flushes, or work the draw's
    // build recorded: a target refresh, a depth copy, an upload), a label store run, or anything else recorded by the
    // draw committer (a dispatch, a store, a copy), by a queue worker or by another thread (the
    // presenter, a flush hook on a game thread). A pass that ends for anything but the guest's own
    // next draw is a pass break the committer pays for again at the next draw (a new head).
    enum class PassEndCause : std::uint8_t { Auto = 0, Submit, DrawOtherPass, DrawNotContinuable, DrawReadsTarget, DrawOutsidePass, DrawStoreFlush, DrawBuild, LabelStores, CommitterOther, QueueWorker, OtherThread, Count };
    static constexpr std::size_t PassEndCauseCount = static_cast<std::size_t>(PassEndCause::Count);
    static const char* PassEndCauseName(PassEndCause cause);
    struct PassEndCounts {
        std::array<std::uint64_t, PassEndCauseCount> causes{};
        std::array<std::uint64_t, SubmitReasonCount> submits{};
    };
    // Counts since the last call (all recorders).
    static PassEndCounts TakePassEndCounts();
    // Pass ends recorded by threads other than the draw committer, ever (monotonic): the
    // APS5_VERIFY_PARALLEL_COMMIT check that nothing interleaved inside a pass run.
    static std::uint64_t PassEndsByOtherThreads();
    // The draw committer's thread (DrawPipeline::run): its pass ends that no scope names count as
    // CommitterOther, everyone else's as QueueWorker (a tagged queue thread) or OtherThread.
    static void MarkCommitterThread();
    // Names the cause of the pass end the calling thread's next Commands call may make.
    class PassEndCauseScope {
    public:
        explicit PassEndCauseScope(PassEndCause cause);
        ~PassEndCauseScope();
        PassEndCauseScope(const PassEndCauseScope&) = delete;
        PassEndCauseScope& operator=(const PassEndCauseScope&) = delete;

    private:
        PassEndCause previous;
    };
    // Ends and submits the open batch without waiting.
    void Submit();
    // Submits the open batch when it holds at least `minWork` commands and the newest submitted
    // batch has completed, so the GPU does not idle while the committer records (APS5_STARVE_SUBMIT).
    // Returns whether it submitted. Requires the GPU mutex.
    // `ignoreAge`: the caller is about to go idle (nothing else queued), so StarveAge does not hold.
    bool SubmitIfStarved(std::uint64_t minWork, bool ignoreAge = false);
    // Submits the open batch and returns the serial of the newest submitted batch (0 when nothing was
    // ever submitted); WaitSerial(serial) then covers all recorded work.
    std::uint64_t SubmitAndEpoch();
    bool HasTimeline() const { return timeline != VK_NULL_HANDLE; }
    // Waits until every batch up to `serial` completed on the GPU. Called WITHOUT GuestMemory::GpuMutex:
    // it touches only the immutable timeline semaphore (the caller keeps the device alive). Requires
    // HasTimeline(); the debug switch APS5_NO_TIMELINE makes the device create the recorder without one.
    void WaitSerial(std::uint64_t serial);
    // Under the mutex: finishes (completions, release) the in-flight batches up to `serial`, from the
    // front only; tolerates batches another thread finished meanwhile. Later batches stay in flight.
    void FinishUpTo(std::uint64_t serial);
    // Submits and waits for every batch, running completions in order.
    void Sync();
    void CountSamples();
    bool RecordMeshArguments(VkCommandBuffer commands, VkDeviceAddress record, VkDeviceAddress arguments, std::span<const std::uint32_t, 7> rules);
    std::uint64_t SamplesTotal();
    bool DumpSamples(VkDeviceAddress target);
    void NoteSampledDraw();
    // Waits only for the batches up to the newest one that writes the range (submitting the open
    // batch when it is that one); later batches stay in flight. Fences of one queue signal in
    // submission order, so completions still run in order. Debug aid: APS5_NO_SYNC_THROUGH=1 syncs all.
    // `waitUnlocked` (the flush hook): when the caller's GpuMutex acquisition is the outermost one
    // on this thread and the device has a timeline, the GPU wait runs WITHOUT the mutex (released
    // and retaken here; the caller must hold nothing else that orders after it) and only the
    // completions run under it, so other queues do not queue behind a CPU read's wait. Inside a
    // batch's completion action no wait is made at all: every batch still in flight was recorded
    // after the completing one, and its store is that batch's in-order write (see Recorder.cpp).
    void SyncThrough(std::uint64_t address, std::size_t bytes, bool waitUnlocked = false);
    // Completes batches that already finished; returns whether nothing is in flight.
    bool Reap();
    // Live `fence_nodrain` (env APS5_FENCE_NO_DRAIN=1), with the recorder thread
    // (APS5_RECORDER_THREAD=1): a finished batch's fence reset and its timestamp read go straight
    // to the driver instead of first waiting for the recorder thread to make every call streamed
    // so far (DeferredVulkan's waiting calls), which held whoever reaped (under the GPU mutex)
    // behind the whole stream at every retired batch. Exact: taken only once the recorder thread
    // returned from the batch's vkQueueSubmit (its stream ticket, DeferredVulkanExecuted) and the
    // fence signaled, so every command of it ran and no streamed call still names the fence, the
    // command buffer or the batch's query pool. Counted for the [submits] line and the tests.
    static bool FenceNoDrain();
    static std::uint64_t UndrainedFenceResets();
    // Live `starve_age` (env APS5_STARVE_AGE_US=<us>, default 250 when only the live flag is set),
    // with APS5_STARVE_SUBMIT: SubmitIfStarved also waits until the open batch is that old (from
    // its first command) or holds 4 times the minimum work, so a GPU that runs a few draws faster
    // than the committer records them gets them in fewer, larger batches. Exact: only when
    // recorded work reaches the queue changes; labels, EOPs, waits, flips and syncs still submit
    // at once, and the committer passes ignoreAge before it goes idle.
    static std::chrono::microseconds StarveAge();
    // SubmitIfStarved checks StarveAge held back so far (cumulative; one batch can be held by
    // several checks).
    static std::uint64_t StarveAgeHolds();
    // When the open batch began (its first command), the steady-clock zero when none is open.
    std::chrono::steady_clock::time_point OpenedAt() const { return open != nullptr ? open->openedAt : std::chrono::steady_clock::time_point{}; }
    std::uint64_t Submissions() const { return submissions; }
    // Batches submitted and not yet finished (what a full Sync() waits for); under GuestMemory::GpuMutex.
    std::size_t InFlightBatches() const { return inFlight.size(); }
    // Reaps that retired at least one batch so far (the [recorder] line's 'with work' count).
    static std::uint64_t ReapsWithWork();

    // A store of `bytes` into a host import (a label, or a COPY_DATA/DMA_DATA/DUMP_CONST_RAM store
    // of up to 64 KiB; `address` is its guest address) recorded into the open batch: ordered
    // after everything recorded before it and visible to the host and to later work. The stores
    // of a batch form one run recorded as the batch's last commands at Submit: one barrier
    // (everything so far -> transfer and host: every result of the batch reaches host memory
    // before any label of it lands, the order the title's pollers rely on), the
    // vkCmdUpdateBuffers, one transfer -> host barrier. A later command of the batch that writes
    // over a queued store's bytes or reads them in place must find the store recorded (the
    // callers that know their ranges call FlushStoresOverlapping, or FlushStores when they do
    // not, before recording): the run is then recorded in place, with a trailing barrier toward
    // the later work, and the next store starts a new run. Within a run a store contiguous with
    // the last queued one joins its vkCmdUpdateBuffer, one inside it replaces its bytes (the
    // later store wins, as in program order), and one overlapping a store queued earlier gets a
    // transfer-to-transfer barrier first (WAW). Debug aids: APS5_LABEL_RUNS_INLINE=1 records the
    // stores at once and closes the run at the next command (Commands()) or at Submit, as
    // before; APS5_NO_LABEL_RUNS=1 gives every store barriers of its own.
    void RecordStore(VkBuffer buffer, VkDeviceSize offset, std::span<const std::byte> bytes, std::uint64_t address);
    bool HasQueuedStores() const { return open != nullptr && !open->run.queued.empty(); }
    bool QueuedStoreOverlaps(std::uint64_t address, std::size_t bytes) const;
    // Whether `overlaps(begin, end)` holds for a queued store's guest range.
    bool AnyQueuedStore(const std::function<bool(std::uint64_t, std::uint64_t)>& overlaps) const;
    void FlushStores();
    void FlushStoresOverlapping(std::uint64_t address, std::size_t bytes) { if (HasQueuedStores() && QueuedStoreOverlaps(address, bytes)) FlushStores(); }

    // Pending-label table. A GPU label (RELEASE_MEM/WRITE_DATA recorded as a store into the batch)
    // is noted per 4-byte dword with the value it will store and a record-order stamp (the driver's
    // event serial, taken when the label is recorded). A WAIT_REG_MEM whose submission was received
    // (stamped) BEFORE the label was recorded may take the value from the table instead of waiting
    // for the GPU: any CPU store the game ordered before that submit call precedes the label in
    // program order exactly as on hardware, and later recorded work follows the label in queue
    // order. Entries recorded before the submission (the previous frame's label at the same address)
    // are trusted only under the late rule below (APS5_LABEL_TRUST_LATE=0 never trusts them).
    // Entries leave the table with their batch (a memory bound only). The note also enters the
    // label's range as a pending write of the open batch (the flush hook syncs CPU reads), marked
    // as the label's own so it does not count as an overwrite of the entry.
    static constexpr std::size_t LabelTableBytes = 64;
    void NoteLabel(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue);
    // Late rule: an entry whose stamp is not newer than the wait's submission is still the value
    // memory will hold, unless a CPU store touched the dword since it was recorded (the table
    // cannot see CPU stores; the title recycles label slots by CPU writes) or a later recorded
    // non-label GPU write covers it. The table refuses a queued entry, one flagged `overwritten`
    // (noteWrite), and one whose record group has not closed (generation 0); the caller checks the
    // CPU side with GuestMemory::UnchangedSinceCollected against the hit's generation. Labels are
    // noted in groups (a worker's deferred labels recorded back to back, or one immediate label
    // or store): CloseLabelGroup, called by the recording thread after the group's last
    // MarkWritten, stamps the group's entries with the write tracker's generation at that point;
    // per-entry capture would fail, as the next label of the group stamps the shared 64 KiB block
    // newer. A generation of 0 leaves the entries unclosed (a failed record), as does a batch
    // already submitted at the close (its value may have landed before the generation was taken).
    static void CloseLabelGroup(std::uint64_t trackerGeneration);
    static bool LateTrust();
    struct LabelHit {
        std::uint64_t value;
        // The recording queue of the first dword, and its record stamp.
        std::uint32_t queue;
        std::uint64_t stamp;
        // Taken under the late rule: `generation` is the oldest group generation of the dwords.
        bool late;
        std::uint64_t generation;
    };
    // Why a lookup refused a late candidate (an entry with stamp <= afterStamp), for the trace.
    // BehindCompletion replaces the other reasons when the refused entry's value reaches memory
    // only through a completion action of its batch (AfterCompletions: the GPU has no view of the
    // range, or a write-back overwrote the GPU's store): a poller must reap that batch to converge.
    enum class LabelRefusal : std::uint8_t { None = 0, TrustOff, Queued, Overwritten, Unclosed, BehindCompletion };
    // Per calling thread (lookups run on the waiting worker): late candidates seen, in both modes,
    // and the refusals by reason ([packets] line deltas).
    struct LateStatistics {
        std::uint64_t candidates, queued, overwritten, unclosed;
    };
    static LateStatistics LateCounts();
    // The 4- or 8-byte value the table holds for `address` when every dword is present with a stamp
    // newer than `afterStamp`, or trustable under the late rule.
    std::optional<LabelHit> PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, LabelRefusal* refusal = nullptr) const;
    std::size_t PendingLabels() const;
    // Whether any tracked label dword (recorded, or queued by a worker and not recorded yet) lies
    // inside [address, address + bytes): a range query for large ranges (a fill of megabytes),
    // where a per-dword lookup would not do.
    bool PendingLabelIn(std::uint64_t address, std::size_t bytes) const;
    bool CompletionLabelIn(std::uint64_t address, std::size_t bytes) const;
    // PendingLabel of the active recorder WITHOUT GuestMemory::GpuMutex: the table has a small mutex
    // of its own (every mutation holds both), so a WAIT_REG_MEM consults it without queueing behind
    // device work. Nothing is done under the table mutex but the lookup (it never takes the GPU mutex).
    static std::optional<LabelHit> LookupLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, LabelRefusal* refusal = nullptr);
    // The value alone, for callers asking whether any label is pending in a dword (afterStamp 0).
    static std::optional<std::uint64_t> LookupLabelValue(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp);
    // The value a recorded label store (WRITE_DATA, RELEASE_MEM data) leaves in a dword, when nothing
    // recorded after it is known to write the dword: the entry is recorded (not queued), no later
    // noted GPU write overlaps it, its value does not wait for completion actions, and no wide
    // label covers it. Work recorded after the store reads this value, so a capture for such work
    // can take it instead of waiting for the GPU (live `label_known`).
    static bool CurrentLabelDword(std::uint64_t address, std::uint32_t& value);
    // The bytes a read ordered after all recorded work will see in [address, address +
    // out.size()) (a DMA_DATA source, driver live `dma_known`), without waiting for the GPU: each
    // dword the pending-write snapshot covers must hold a recorded label store's value
    // (CurrentLabelDword) and is taken from the table, the others are read from memory as they
    // are. Unknown at the first covered dword with no such value (a non-label writer, a label
    // behind a completion or written over, a wide label) or one of this thread's queued labels;
    // NotPending when nothing covers the range (a plain read waits for nothing either). The
    // caller checks the range is accessible and holds no unit-shadow or storage-image results.
    enum class KnownSource : std::uint8_t { Served, Unaligned, NotPending, Queued, Unknown };
    static KnownSource KnownSourceBytes(std::uint64_t address, std::span<std::byte> out);
    // APS5_TRACE_DCC_SYNC: the code address that noted a pending write (the caller of
    // NotePendingWrite/NotePendingFill/NotePendingWrites), from a ring of the last 8192 notes;
    // null when the note is older or was noted another way.
    static const void* NoteSite(std::uint64_t note);
    // The guest program whose recorded work notes writes next on this thread (a dispatch's compute
    // program, 1 for a draw; 0 clears), kept with the note sites for the same trace.
    static void SetNoteProgram(std::uint64_t program);
    // The write notes the calling thread's last recorded dispatch made: notes in (first, last]
    // (VulkanDevice::recordDispatch around MarkGpuWrites). A caller that knows what the dispatch
    // stored (the pattern-fill kernel's DCC keys) attributes the range to the dispatch only while
    // the newest note over it is one of these.
    static void SetDispatchNotes(std::uint64_t first, std::uint64_t last);
    static std::pair<std::uint64_t, std::uint64_t> DispatchNotes();
    static std::uint64_t NoteProgram(std::uint64_t note);
    static bool WideLabelIn(std::uint64_t address, std::size_t bytes);
    // A label a queue worker decoded but has not recorded yet (Driver.cpp DeferredLabels): it
    // enters the table with no batch, and only a lookup made on the noting thread (a WAIT_REG_MEM
    // of the same queue) takes its value, since that queue's later work follows the label in queue
    // order whenever the group is recorded; other queues and the CPU see it once it is recorded.
    // The range is also kept per thread, so a CPU access of it by the same worker (the flush hook)
    // has the group recorded first through the function SetQueuedLabelRecorder installed. Takes
    // the table mutex only. The queued entries live apart from the recorded ones: a recorded
    // label of the same dword (an older store still pending on the GPU) keeps serving the other
    // queues' waits and the driver's pending-label checks until the queued one is recorded.
    // Debug aid: APS5_NO_SEPARATE_QUEUED_LABELS=1 lets a queued entry replace the recorded one.
    static void NoteQueuedLabel(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue);
    // After the calling worker recorded (or dropped) its queued labels: its ranges are cleared and
    // the entries of its dwords still without a batch (stored by the CPU, or dropped) leave the table.
    static void ForgetQueuedLabels();
    // The function the flush hook calls, on the accessing thread (which may hold the GPU mutex),
    // when an access overlaps one of that thread's queued labels; it records them. Set once.
    static void SetQueuedLabelRecorder(void (*recorder)());
    // Whether the lock-free pending-write snapshot (open, in-flight and finishing batches) overlaps
    // the range: false means no recorded work writes it, so a wait on it has nothing to submit.
    static bool SnapshotWriteOverlaps(std::uint64_t address, std::size_t bytes);
    static bool QueuedLabelsOverlap(std::uint64_t address, std::size_t bytes);
    // The snapshot itself (the sorted, merged union of the pending ranges; null when none), for a
    // reader that tests many ranges against one loaded snapshot: one atomic shared_ptr load per
    // validation instead of one per run, and every test sees the same snapshot (design13 R1's p0).
    using WriteRanges = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
    static std::shared_ptr<const WriteRanges> PendingWriteSnapshot();
    static bool SnapshotOverlaps(const WriteRanges* snapshot, std::uint64_t address, std::size_t bytes);
    // A label behind pending completions (CPU write-backs the label must not precede): stored to
    // guest memory by a completion action appended to the open batch, or to the newest in-flight
    // batch when none is open (no empty submission), so it runs after every earlier write-back in
    // submission order. The range is noted as a pending write of that batch so CPU reads through the
    // flush hook still sync, and the label enters the table like a GPU one.
    // `storedOnGpu`: the same bytes were also recorded into the open batch (a host import), so the
    // completion re-stores them only when a CPU write-back noted by NoteWrittenBack overlapped the
    // range since; the plain store of a label the GPU has no view of always runs. An unconditional
    // completion store would land on memory the game may have reused by then (a stale label value
    // over a fresh command buffer). Debug aid: APS5_LABEL_STORE_ALWAYS=1 stores unconditionally.
    void AfterCompletions(std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue, bool storedOnGpu);
    bool AfterRecordedWork(std::function<void()> action);
    // A CPU store of GPU results into guest memory (GuestBufferMemory::WriteBack): recorded so a
    // completion label store can tell whether its bytes were overwritten. Under GuestMemory::
    // GpuMutex (every write-back runs inside finish() or a synchronous draw's wait): the first
    // write-back overlapping a completion label stored on the GPU is what makes that label's
    // completion store necessary, so it is counted into PendingCompletionLabels here rather than
    // at registration (APS5_COUNT_ALL_COMPLETION_LABELS=1 counts every one at registration, as
    // before). A write-back over any tracked label dword is counted on the [recorder] line (must
    // stay 0: it would mean a CPU store landed over a label recorded on the GPU).
    static void NoteWrittenBack(std::uint64_t address, std::size_t bytes);

    // Lock-free readers for the queue workers' packet loops and WAIT_REG_MEM polls (values of the
    // active recorder; a torn-down recorder never publishes):
    // when the open batch received its first label (nullopt: none pending), so the label's
    // submission can be bounded (APS5_LABEL_FLUSH_US) without taking the mutex to look;
    static std::optional<std::chrono::steady_clock::time_point> PendingLabelSince();
    // bumped by every note of a pending write into the open batch, so a poller learns that the
    // producer may have recorded the label it waits for;
    static std::uint64_t WriteGeneration();
    // bumped right AFTER every store of the pending-write snapshot (the generation above moves
    // before the store), so a reader that loads it before and after a set of in-place reads knows
    // the snapshot it consulted in between was the one in force for all of them;
    static std::uint64_t PublishGeneration();
    // labels waiting in completion actions (the CPU must reap their batch for them to land);
    static std::uint64_t PendingCompletionLabels();
    // write-back completions (OnComplete) of unfinished batches: a reaper retires them when every
    // worker sleeps, as for the labels above;
    static std::uint64_t PendingWriteBackCompletions();
    // dispatches and draws recorded since the last submit (the driver counts them: CountRecordedWork),
    // so a batch is submitted after a bounded amount of work (APS5_BATCH_CAP).
    static std::uint64_t RecordedWorkSinceSubmit();
    static void CountRecordedWork();
    // APS5_EOP_COMPLETION_THREAD=1: a submitted batch holding completion labels or EOP interrupts
    // is tracked, and a driver thread waits for it on the timeline WITHOUT the mutex
    // (WaitCompletionWork), then finishes the batches up to it under the mutex
    // (FinishCompletionWork), front to back like every other reap. APS5_PROFILE_DRAW tracks the
    // same batches (without the thread option the thread only stamps when the timeline reached
    // them) for the [eopcompletion] line. StopCompletionWork makes WaitCompletionWork return false.
    static bool CompletionThreadEnabled();
    static bool CompletionWorkTracked();
    static bool WaitCompletionWork(std::uint64_t& recorder, std::uint64_t& serial);
    static std::size_t FinishCompletionWork(std::uint64_t recorder, std::uint64_t serial);
    static void StopCompletionWork();
    // A queue worker's poll-loop reap attempt (try_lock on the mutex), for the [eopcompletion] line.
    static void CountCompletionReapTry(bool failed);

    // The recorder of the current device, for code that only has guest addresses (the flush hook)
    // and for helpers that must not recycle resources the recorded work still uses.
    static Recorder* Active();
    // Whether the calling thread runs inside a batch's completion action (finish()): a store made
    // there must not have a publish recorded for it (UnitShadow), see SyncThrough.
    static bool InCompletion();
    void Activate();
    // APS5_PROFILE_DRAW: syncs by source (0 device idle, 1 CPU access to pending writes, 2 CPU read
    // after a recorded store, 3 address-based dispatch, 4 other). Contract: the call announces the
    // source for the Sync/SyncThrough this thread makes NEXT, which consumes it (a sync without a
    // preceding CountSync counts as "other"); call it right before the sync and never without one,
    // or the announcement attributes an unrelated later wait. `site` names the call site for the
    // [recorder] "top sync sites" table (a return address, printed as a module offset like the
    // [guestmem] callers): a caller that syncs on behalf of others (VulkanDevice::WaitIdle) passes
    // its own return address, so the table names the driver site, not the wrapper; without it the
    // return address of the Sync/SyncThrough call itself is taken, so an unannounced sync still
    // names its real caller.
    static void CountSync(int source, const void* site = nullptr);
    // Names only the site for the sync or drain this thread makes next (its source stays as
    // announced, or as the drain counts it): for a wrapper (VulkanDevice::ReapRecorded) whose
    // FinishUpTo counts the drain itself, so a CountSync there would count it twice.
    static void AnnounceSyncSite(const void* site);
    // APS5_PROFILE_DRAW: the calling thread's fence and timeline waits so far, in milliseconds (0
    // when not profiling): a caller reads it around a span of its own work to learn how much of
    // that span waited for the GPU (a resource build's nested flush-hook waits, a draw's).
    static double ThreadWaitedMs();
    // The calling thread's flush-hook syncs (SyncThrough) that waited for a target batch whose
    // fence had not signaled when the wait began: a read whose bytes are compared around such a
    // sync observed the GPU's work; one that found the fence signaled already read the GPU's bytes
    // both times and observed nothing (Driver.cpp's dword evidence reads it before and after).
    static std::uint64_t ThreadHookWaits();
    // Cumulative counts for the [labels] line (APS5_PROFILE_DRAW): stores recorded (RecordStore),
    // the runs they formed (one barrier pair each), stores joined to a pending contiguous one,
    // stores that replaced pending bytes, WAW barriers inside runs, joins refused because the
    // joined store would overlap one the run already recorded, queued labels noted (of which
    // over a recorded entry of the same dword), lookups a queued label satisfied on its own
    // queue, flush-hook accesses that had queued labels recorded first, and hook accesses made
    // by a completion action whose queued-label record was skipped.
    struct StoreStatistics {
        std::uint64_t stores, runs, joined, replaced, wawBarriers, joinsRefused, queuedNoted, queuedOverRecorded, queuedHits, queuedHookRecords, queuedHookInCompletion;
        // DCC key stores queued, the runs that recorded them (at Submit, or before a writer of a
        // queued range), and the queued stores a duplicate range joined.
        std::uint64_t keyStores, keyStoreRuns, keyStoreRunsForWriter, keyStoresJoined;
        // Store runs recorded at Submit, and in place before a later writer or reader of a queued
        // store's bytes (FlushStores).
        std::uint64_t runsAtSubmit, runsForced;
    };
    static StoreStatistics StoreCounts();
    // APS5_PROFILE_GPU=1: GPU time of recorded work by key (a guest program address), from timestamp
    // queries around each timed range; the totals per key are reported every 10 s. Begin returns the
    // range index to pass to End, or NoTiming when timing is off or the batch's queries are used up.
    static constexpr std::uint32_t NoTiming = 0xffffffffu;
    static bool GpuTimingEnabled();
    static bool BatchStampsEnabled();
    std::uint32_t BeginGpuTiming(std::uint64_t key);
    // APS5_DEBUG_LABELS: a named debug label around one command (a GPU Trace attributes it alone).
    static void DebugLabelBegin(const Context& context, VkCommandBuffer commands, const char* name);
    static void DebugLabelEnd(const Context& context, VkCommandBuffer commands);
    // APS5_DEBUG_LABELS=2 (flat per-command labels).
    static bool FlatDebugLabels();
    // `bytes`: what the range moved (a fill's, a copy's), summed per key on the [gputime] line.
    void EndGpuTiming(std::uint32_t index, std::uint64_t bytes = 0);
    // The key of the whole-batch range (first to last command; reported as "batch" on the [gputime]
    // line, apart from the per-program totals).
    static constexpr std::uint64_t BatchTimingKey = 0x3;
    // What a timed range worked on, appended to its gpu-timeline.txt line (APS5_GPU_TIMELINE with
    // APS5_PROFILE_GPU; nothing is kept outside that window's configuration). A draw pass range
    // carries the targets of its first draw, its draw count and the guest program addresses of
    // the pixel and vertex shaders that drew most in it; a surface range (storage upload or
    // write-back, staging copies) the surface's guest address, size, formats and the path that
    // moved it. `index` is BeginGpuTiming's (NoTiming and label-only ranges are ignored).
    struct TimedDraw {
        std::uint64_t color = 0;
        std::uint32_t colorWidth = 0, colorHeight = 0, colorFormat = 0, colorTargets = 0;
        std::uint64_t depth = 0;
        std::uint32_t depthWidth = 0, depthHeight = 0, depthFormat = 0;
        std::uint64_t pixelProgram = 0, vertexProgram = 0;
    };
    void NoteTimedDraw(std::uint32_t index, const TimedDraw& draw);
    // The timing index of the open render pass (a continued draw lies in its range), or NoTiming.
    std::uint32_t RenderPassTiming() const;
    // `format`: the host VkFormat (0 when none), `guestFormat` the guest's; `kind` a string literal;
    // `pieces`: the ranges or regions the command moved (staging copies cover several buffers,
    // `address` is then the first one's and `bytes` their sum).
    void NoteTimedSurface(std::uint32_t index, std::uint64_t address, std::uint64_t bytes, std::uint32_t format, std::uint32_t guestFormat, std::uint32_t width, std::uint32_t height, const char* kind, std::uint32_t pieces = 1);
    static bool TimedIdentityWanted();
    // The driver's own command classes: their [gputime] ranges take the reserved keys 0x10 + class
    // (printed by name, with counts and bytes, per 10 s and per present) and their barriers are
    // counted on the [barriers] line (CountBarriers, printed every 10 s under APS5_PROFILE_DRAW
    // or APS5_PROFILE_GPU). A class range covers the command with its own barriers (a dispatch's
    // barriers are classes of their own, its program range stays keyed by the program), so the
    // union of every range of a batch and the batch span differ by what no class times ('untimed').
    enum class CommandClass : std::uint8_t { DispatchLeading = 0, DispatchTrailing, IndirectArguments, LabelRun, Fill, FillClear, Copy, StagingIn, StagingOut, Draw, StorageUpload, StorageWriteBack, DccClear, DccKeyStore, PresentBlit, ShadowPublish, TemplateDataRefresh, Count };
    static constexpr std::uint64_t ClassKey(CommandClass which) { return 0x10 + static_cast<std::uint64_t>(which); }
    std::uint32_t BeginGpuTiming(CommandClass which) { return BeginGpuTiming(ClassKey(which)); }
    static void CountBarriers(CommandClass which, std::uint32_t count = 1);
    // A leading barrier a command left out because the previous trailing barrier covered its
    // accesses (see Commands), on the [barriers] line as 'merged'. Debug aid: APS5_FULL_BARRIERS=1
    // (or APS5_NO_BARRIER_ELISION=1) records every leading barrier, as before.
    static bool MergeBarriers();
    static bool MergeDrawBarriers();
    // Dispatch runs (live `dispatch_runs`, env APS5_DISPATCH_RUNS): on the console, consecutive
    // dispatches of one queue overlap unless the title asks for a wait between them (a
    // CS_PARTIAL_FLUSH, an ACQUIRE_MEM, a release). The driver gives each dispatch a token that
    // stays the same while only register writes separate the dispatches of a queue
    // (SetDispatchRunToken, thread-local; 0 = none). A dispatch whose token matches the run left
    // open by the previous command skips its leading barrier, and every dispatch of a run leaves
    // its trailing barrier owed: any other command (or a dispatch outside the run, or Submit)
    // records it first.
    static bool DispatchRunsEnabled();
    // Live `dispatch_run_stores` (env APS5_DISPATCH_RUN_STORES), with dispatch runs: a WRITE_DATA
    // between the dispatches of a run does not end it (the console's ME writes it without waiting
    // for the dispatches before it). Its store is recorded in place inside the run (RecordStore),
    // with only a transfer-to-compute barrier, so the dispatches after it read it.
    static bool DispatchRunStoresEnabled();
    // Live `dispatch_run_data` (env APS5_DISPATCH_RUN_DATA), with dispatch runs: a dispatch whose
    // template data words differ continues the run with its words in snapshot-ring slices
    // (ShaderResources::DataBindings) instead of ending it to refresh the template's buffers.
    static bool DispatchRunDataEnabled();
    static void SetDispatchRunToken(std::uint64_t token);
    static std::uint64_t DispatchRunToken();
    bool ContinuesDispatchRun(std::uint64_t token) const;
    VkCommandBuffer CommandsInDispatchRun();
    void LeaveDispatchRunOpen(std::uint64_t token);
    static constexpr VkAccessFlags DrawBarrierAccess = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    static void CountMerged(CommandClass which);
    static std::uint64_t MergedBarriers(CommandClass which);
    // Hazard tracker, counting mode (APS5_BARRIER_VALIDATE=1): every command tells the tracker
    // what it reads and writes (guest ranges through host imports, images) with its stage before
    // it is recorded; the tracker simulates barriers emitted only on a hazard against the accesses
    // since its last simulated barrier (RAW, WAW, WAR, an image written; a command whose ranges are
    // unknown, an address-based build or a BDA draw, is a hazard by itself) and counts, per class,
    // the leading barriers a hazard-driven recorder would have emitted and skipped, on the
    // [barriers] line; the real barriers are recorded as before. APS5_TRACE_BARRIERS=<n> prints
    // the first n skipped decisions with their ranges. Costs nothing when off.
    struct Access {
        std::span<const std::pair<std::uint64_t, std::uint64_t>> reads;
        std::span<const std::pair<std::uint64_t, std::uint64_t>> writes;
        // (image, written)
        std::span<const std::pair<VkImage, bool>> images;
        VkPipelineStageFlags stages;
        bool conservative = false;
        bool ordered = false;
        bool keepBarriers = false;
        // Reads guest memory it cannot name (an address-based shader): conflicts with every earlier
        // write to memory and every later one, but not with images (live `lease_reads`).
        bool readsAnyMemory = false;
    };
    static bool BarrierValidate();
    static bool BarrierTracking();
    bool NoteAccess(CommandClass which, const Access& access);
    VkCommandBuffer Commands(CommandClass which, const Access& access, bool& elided, VkAccessFlags* coveredAccess = nullptr);
    bool SkipsBarriers(CommandClass which) const;
    static void CountSkipped(CommandClass which, std::uint32_t count = 1);
    // A range timed outside the recorder's batches (the presenter's blit, stamped into its own
    // pool and read after its fence): enters the class totals like a recorded one.
    static void AddGpuTiming(CommandClass which, double nanoseconds, std::uint64_t bytes);
    // Presentations, so the [gputime] class totals can be given per present.
    static void CountPresent();
    // Presentations so far (never reset): a frame clock for policies that count per frame, such as
    // StorageTexture's write-back hysteresis.
    static std::uint64_t Presents();
    // The GPU milliseconds per MiB of the storage-writeback class in the last [gputime] window
    // (APS5_PROFILE_GPU), 0 before one measured any.
    static double StorageWritebackMsPerMiB();

    // What the presenter learns about finished batches (Driver::Present's [present] line and the
    // frame record): finish() writes an entry per submitted batch into a ring of the last 128,
    // under GuestMemory::GpuMutex and the ring's own mutex; the readers below take only the ring
    // mutex, so the presenter reads them WITHOUT the GpuMutex. gpuStartNs/gpuEndNs are the
    // whole-batch stamps (0 unless BatchStampsEnabled); `reads` and `readGeneration` (the
    // write-watch generation collected over the noted in-place reads at submit) are filled with
    // APS5_FLIP_READ_CHECK=1 only.
    struct Completed {
        std::uint64_t serial = 0;
        double gpuStartNs = 0;
        double gpuEndNs = 0;
        // Why it was submitted, whether the GPU had nothing queued then (SubmitProfiled only), and
        // the GpuMutex tag of the thread that opened it.
        SubmitReason reason = SubmitReason::Other;
        bool starved = false;
        std::uint32_t queue = 0xffffffffu;
        std::chrono::steady_clock::time_point submittedAt{};
        std::chrono::steady_clock::time_point fenceSeenAt{};
        std::uint64_t readGeneration = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
    };
    // The entries with serial in (afterSerial, throughSerial], oldest first; `missing` counts the
    // serials of that range not in the ring (not finished yet, or overwritten by newer batches).
    // An entry is overwritten once the batch CompletedRingSize serials later finishes.
    static constexpr std::size_t CompletedRingSize = 512;
    std::vector<Completed> CompletedBatches(std::uint64_t afterSerial, std::uint64_t throughSerial, std::size_t& missing) const;
    // The newest submitted serial and its vkQueueSubmit time (ring mutex only).
    std::uint64_t NewestSubmitted(std::chrono::steady_clock::time_point* submittedAt = nullptr) const;
    // In-flight batches whose fence has not signaled (one status query each); under the mutex.
    std::size_t UnsignaledBatches() const;
    static bool FlipReadCheck();

private:
    struct Batch {
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        std::vector<std::shared_ptr<void>> kept;
        std::size_t keptBytes = 0;
        std::vector<std::function<void()>> completions;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
        std::vector<std::uint64_t> writeNotes;
        std::vector<std::int16_t> writeValues;
        // In-place reads (see NotePendingRead), dying with the batch: a finished batch's reads are done.
        struct Read {
            std::uint64_t begin;
            std::uint64_t end;
            ReadKind kind;
        };
        std::vector<Read> reads;
        std::vector<std::uint64_t> readSets;
        // Shared read sets (NoteSharedReads), sorted and disjoint; `bounds` spans the set and carries
        // its kind (what readOverlap returns for a hit inside it).
        struct SharedReads {
            std::shared_ptr<const std::vector<std::pair<std::uint64_t, std::uint64_t>>> ranges;
            Read bounds;
        };
        std::vector<SharedReads> sharedReads;
        // Submission number (1-based): identifies a batch after its allocation may have been reused.
        std::uint64_t serial = 0;
        bool submitted = false;
        // The GpuMutex queue tag of the thread that opened the batch, and when it was submitted
        // (the [syncwait] line reports what a wait's target and the batches ahead of it were).
        std::uint32_t queue = 0xffffffffu;
        std::chrono::steady_clock::time_point submittedAt{};
        // See SubmitReason; `starved`: the GPU had nothing queued when it was submitted.
        SubmitReason reason = SubmitReason::Other;
        bool starved = false;
        std::chrono::steady_clock::time_point openedAt{};
        // The recorder thread's stream position of its vkQueueSubmit (DeferredVulkanTicket; 0
        // without the thread).
        std::uint64_t submitTicket = 0;
        VkQueryPool queries = VK_NULL_HANDLE;
        VkQueryPool samples = VK_NULL_HANDLE;
        std::shared_ptr<void> samplePool;
        bool sampleActive = false;
        bool samplesDrawn = false;
        std::vector<std::uint64_t> timedKeys;
        std::vector<std::uint64_t> timedBytes;
        // Parallel to timedKeys while TimedIdentityWanted (NoteTimedDraw, NoteTimedSurface).
        struct TimedIdentity {
            TimedDraw targets;
            std::uint32_t draws = 0;
            // The four programs that drew most (address, draws) and the draws of all others.
            std::array<std::pair<std::uint64_t, std::uint32_t>, 4> pixel{}, vertex{};
            std::uint32_t otherPixel = 0, otherVertex = 0;
            std::uint64_t surface = 0, surfaceBytes = 0;
            std::uint32_t surfaceFormat = 0, surfaceGuestFormat = 0, surfaceWidth = 0, surfaceHeight = 0, surfacePieces = 0;
            const char* surfaceKind = nullptr;
        };
        std::vector<TimedIdentity> timedIdentity;
        // The whole-batch timed range (BatchTimingKey) and its stamps once read (see Completed).
        std::uint32_t batchTiming = NoTiming;
        double gpuStartNs = 0;
        double gpuEndNs = 0;
        std::uint64_t readGeneration = 0;
        // Dword addresses this batch noted in the label table (removed when it finishes).
        std::vector<std::uint64_t> labelDwords;
        std::vector<std::multimap<std::uint64_t, std::uint64_t>::iterator> wideLabels;
        // Labels stored by this batch's completion actions (see AfterCompletions); subtracted from
        // the lock-free pending count when the batch finishes, whether or not its completions ran.
        std::uint32_t completionLabelCount = 0;
        // Write-back completions (OnComplete), likewise.
        std::uint32_t writeBackCompletionCount = 0;
        // EOP interrupts appended by AfterRecordedWork.
        std::uint32_t recordedWorkActions = 0;
        // The ranges of the completion labels also stored on the GPU, and whether each entered the
        // pending count (NoteWrittenBack counts one the first time a write-back overlaps it).
        struct CompletionLabel {
            std::uint64_t begin;
            std::uint64_t end;
            bool counted;
        };
        std::vector<CompletionLabel> completionLabelRanges;
        // The store run in progress (RecordStore): whether its leading barrier was recorded and
        // the trailing one is still owed, the store not yet recorded (joined by contiguous ones),
        // and the [buffer, offset, end) of the stores already recorded (the WAW check).
        struct StoreRun {
            bool open = false;
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceSize offset = 0;
            std::vector<std::byte> bytes;
            std::vector<std::tuple<VkBuffer, VkDeviceSize, VkDeviceSize>> recorded;
            // The run's [gputime] range (leading barrier to trailing barrier) and its stored bytes.
            std::uint32_t timing = NoTiming;
            std::uint64_t storedBytes = 0;
            // The stores queued for the batch's run (per-batch runs; see RecordStore), in order.
            struct Queued {
                VkBuffer buffer;
                VkDeviceSize offset;
                std::vector<std::byte> bytes;
                std::uint64_t address;
            };
            std::vector<Queued> queued;
        } run;
        // See Commands(): the accesses the last recorded command's trailing barrier covers.
        VkAccessFlags coveredAccess = 0;
        // The token of the dispatch run the last command left open (its trailing barrier owed;
        // see DispatchRunsEnabled), 0 when none.
        std::uint64_t dispatchRun = 0;
        // The hazard tracker's accesses since its last simulated barrier (see NoteAccess).
        struct Tracker {
            struct Range {
                std::uint64_t begin;
                std::uint64_t end;
                VkPipelineStageFlags stages;
            };
            std::vector<Range> reads;
            std::vector<Range> writes;
            struct Image {
                VkImage image;
                bool written;
                VkPipelineStageFlags stages;
            };
            std::vector<Image> images;
            bool unknown = true;
            // An access since the barrier read memory it could not name (Access::readsAnyMemory).
            bool readsAnyMemory = false;
            bool pending = false;
            bool elided = false;
            std::uint32_t classes = 0;
        } tracker;
        std::uint32_t skipMask = 0;
        // The render pass a recorded draw left open (see ContinuesRenderPass).
        struct RenderPass {
            bool open = false;
            bool continuable = false;
            std::uint64_t key = 0;
            std::uint32_t timing = NoTiming;
            std::function<void(VkCommandBuffer)> afterPass;
        } renderPass;
        // A pass ended in this batch: Submit records the host-read barrier its draws left out.
        bool hostReadOwed = false;
        // Queued DCC key stores (see QueueKeyStore).
        struct KeyStore {
            VkBuffer buffer;
            VkDeviceSize first;
            VkDeviceSize last;
            std::shared_ptr<void> seed;
            std::uint64_t begin;
            std::uint64_t end;
        };
        std::vector<KeyStore> keyStores;
    };
    struct LabelEntry {
        std::uint32_t value;
        std::uint32_t queue;
        std::uint64_t stamp;
        // Null for a queued label (NoteQueuedLabel): not recorded yet, so no batch removes it.
        const Batch* batch;
        // Write-tracker generation at the close of the entry's record group (CloseLabelGroup);
        // 0 while the group is open. A recorded non-label GPU write over the dword noted after the
        // entry sets `overwritten`: memory will not end with `value`. Both serve the late rule only.
        std::uint64_t generation = 0;
        bool overwritten = false;
        // The value lands in memory only when the batch's completion actions run (LabelRefusal::BehindCompletion).
        bool behindCompletion = false;
    };
    void readGpuTiming(Batch& batch);
    void beginSamples(Batch& batch);
    void readSamples(Batch& batch);
    bool gpuSampleCounter();
    void endSamples(Batch& batch);
    void foldSamples(Batch& batch, VkDeviceAddress target);
    struct SampleSegment {
        std::shared_ptr<void> pool;
        VkQueryPool handle;
    };
    std::vector<SampleSegment> pendingSamples;
    bool countingSamples = false;
    int meshArgumentState = 0;
    VkPipelineLayout meshArgumentLayout = VK_NULL_HANDLE;
    VkPipeline meshArgumentPipeline = VK_NULL_HANDLE;
    int sampleCounterState = 0;
    std::unique_ptr<Buffer> sampleCounter;
    VkPipelineLayout sampleLayout = VK_NULL_HANDLE;
    VkPipeline samplePipeline = VK_NULL_HANDLE;
    std::shared_ptr<void> samplePools;
    // BeginGpuTiming on the open batch without Commands() (RecordStore times its own run, which
    // Commands() would close).
    std::uint32_t beginTiming(std::uint64_t key);
    // Vulkan entry points resolved once (constructor): the loader's vkGetDeviceProcAddr is a name
    // lookup per call, paid by every reap, store and submit otherwise. APS5_NO_PROC_TABLE=1 leaves
    // them null and resolves per call as before; vkWaitSemaphoresKHR stays a lazy lookup (timeline
    // semaphores are optional).
    PFN_vkGetFenceStatus getFenceStatus = nullptr;
    PFN_vkResetFences resetFences = nullptr;
    PFN_vkWaitForFences waitForFences = nullptr;
    PFN_vkCmdUpdateBuffer cmdUpdateBuffer = nullptr;
    PFN_vkBeginCommandBuffer beginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer endCommandBuffer = nullptr;
    PFN_vkQueueSubmit queueSubmit = nullptr;
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
    PFN_vkCmdBeginQuery cmdBeginQuery = nullptr;
    PFN_vkCmdEndQuery cmdEndQuery = nullptr;
    PFN_vkCmdResetQueryPool cmdResetQueryPool = nullptr;
    PFN_vkCmdCopyQueryPoolResults cmdCopyQueryPoolResults = nullptr;
    PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
    PFN_vkCmdPushConstants cmdPushConstants = nullptr;
    PFN_vkCmdDispatch cmdDispatch = nullptr;
    // This device's entry points past DeferredVulkan's waiting thunks (FenceNoDrain); null
    // without the recorder thread.
    PFN_vkResetFences resetFencesUndrained = nullptr;
    PFN_vkGetQueryPoolResults queryResultsUndrained = nullptr;
    template<typename TFunction>
    TFunction function(TFunction resolved, const char* name) const {
        return resolved != nullptr ? resolved : context.Function<TFunction>(name);
    }
    VkResult fenceStatus(VkFence fence) const { return function(getFenceStatus, "vkGetFenceStatus")(context.device, fence); }
    void recordBarrier(VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) const;
    // Opens a batch when none is open, without closing a store run (Commands() does both).
    void ensureOpen();
    void prepareCommands();
    void untrackedUse();
    void recordFullBarrier();
    // Records the run's pending store (vkCmdUpdateBuffer), if any.
    void flushPendingStore();
    // Records the pending store and the run's trailing barrier; the next store starts a new run.
    // Per-batch runs: records the queued stores with their barriers (`atSubmit`: the batch-end
    // form, toward the host only); returns whether the batch's host-read barrier was included.
    bool closeStoreRun(bool atSubmit = false);
    // Ends the render pass a draw left open (vkCmdEndRenderPass, the pass's trailing barrier).
    // `cause` Auto: the calling thread's PassEndCauseScope, else its role (MarkCommitterThread).
    void endOpenRenderPass(PassEndCause cause = PassEndCause::Auto, SubmitReason submitReason = SubmitReason::Other);
    // Records the trailing barrier an open dispatch run owes (see DispatchRunsEnabled).
    void closeDispatchRun();
    // Whether the last RecordStore went in place inside a dispatch run: its NoteLabel then does
    // not arm the prompt submit at the next packet (the run would end at every WRITE_DATA); the
    // store is still in the label table and under a pending-write note.
    bool lastStoreInRun = false;
    // Records the queued key stores as one run (`forWriter`: before a command writing, reading or
    // labelling over one, not at Submit).
    void recordKeyStores(bool forWriter);

    // The unlocked wait of SyncThrough(waitUnlocked): the target batch (submitting the open one when
    // it is the target), the timeline wait with the mutex released, then the completions up to it.
    // Returns false when the locked path must run instead (no timeline, nested acquisition, off).
    bool syncThroughUnlocked(std::uint64_t address, std::uint64_t end, int source, const void* site);
    // `source` is the CountSync source the wait is attributed to.
    void finish(std::unique_ptr<Batch> batch, bool wait, int source, bool releaseNow = false);
    void release(Batch& batch) noexcept;
    static bool overlaps(const Batch& batch, std::uint64_t address, std::uint64_t end);
    // The first noted read of `batch` overlapping [address, end), or null.
    static const Batch::Read* readOverlap(const Batch& batch, std::uint64_t address, std::uint64_t end);
    bool signaled(const Batch& batch) const;
    // Rebuilds the lock-free snapshot of pending writes from open, inFlight and finishing.
    void publishPendingWrites() const;
    // Appends one range to the open batch; returns whether the snapshot must be rebuilt for it.
    // `ownLabel`: the range is a label's own store (NoteLabel, AfterCompletions), which does not
    // overwrite the table entries it covers; any other range flags them (table mutex, briefly).
    bool noteWrite(std::uint64_t address, std::size_t bytes, bool ownLabel = false, int value = -1);
    // Appends one range to `batch` (open or in flight) and publishes the snapshot if needed.
    void noteWriteOn(Batch& batch, std::uint64_t address, std::size_t bytes, bool ownLabel = false);
    void noteLabelOn(Batch& batch, std::uint64_t address, std::span<const std::byte> bytes, std::uint64_t stamp, std::uint32_t queue, bool behindCompletion = false);
    // Flags the recorded entries of `batch` over [begin, end) as stored by a completion action.
    void markBehindCompletion(const Batch& batch, std::uint64_t begin, std::uint64_t end);
    // Enters a submitted batch with completion labels into the completion-work tracking.
    void trackCompletionWork(const Batch& batch);
    // Flags the recorded entries of the dwords overlapping [address, end) as overwritten.
    void markOverwritten(std::uint64_t address, std::uint64_t end);
    // Whether a write-back noted after `sequence` overlapped [begin, end); true when the ring no
    // longer reaches back to `sequence` (conservative: the store runs as before).
    bool writtenBackSince(std::uint64_t sequence, std::uint64_t begin, std::uint64_t end);
    std::mutex writtenBackMutex;
    std::deque<std::array<std::uint64_t, 3>> writtenBack;
    std::uint64_t writtenBackSequence = 0;
    // PendingLabel without the table mutex (the caller holds it, or the GPU mutex).
    std::optional<LabelHit> lookupLabel(std::uint64_t address, std::size_t bytes, std::uint64_t afterStamp, LabelRefusal* refusal) const;
    // Whether the in-flight batch with that serial has not signaled its fence (false when it is
    // not in flight any more): what makes a wait for it a real GPU wait (ThreadHookWaits).
    bool unsignaled(std::uint64_t serial) const;

    Context context;
    VkSemaphore timeline = VK_NULL_HANDLE;
    // Identity for a thread that released the mutex around a wait (syncThroughUnlocked): the
    // recorder may have been torn down meanwhile, so it is looked up by this id, not by pointer.
    std::uint64_t id = 0;
    // Every access holds the table mutex (Recorder.cpp): the recorded entries are mutated under
    // GuestMemory::GpuMutex as well, the queued ones (NoteQueuedLabel) under the table mutex alone.
    // Ordered, so a range (a noted write, PendingLabelIn) finds its dwords by lower_bound.
    std::map<std::uint64_t, LabelEntry> labels;
    // The queued entries (batch always null), by dword; a lookup on the noting queue's thread
    // prefers them (its newest store in program order), every other lookup sees `labels` alone.
    std::map<std::uint64_t, LabelEntry> queuedLabels;
    std::multimap<std::uint64_t, std::uint64_t> wideLabels;
    std::uint64_t wideLabelBytes = 0;
    bool wideLabelInLocked(std::uint64_t address, std::size_t bytes) const;
    // labels.size(), readable without the table mutex: a noted write skips the mutex while the
    // table is empty (most of the time between label groups).
    std::atomic<std::size_t> recordedLabels{0};
    std::unique_ptr<Batch> open;
    std::deque<std::unique_ptr<Batch>> inFlight;
    // Batches popped from inFlight whose completions have not run yet: their writes stay in the
    // snapshot (a rebuild from inside a completion must not drop them) until finish returns.
    std::vector<const Batch*> finishing;
    std::uint64_t submissions = 0;
    std::size_t inFlightKeptBytes = 0;
    std::uint64_t writeNoteCount = 0;
    // Command buffers and fences of completed batches, reused by later ones (hundreds of batches per
    // frame would otherwise allocate and free their objects each time).
    std::vector<std::pair<VkCommandBuffer, VkFence>> spare;
    std::vector<VkQueryPool> sparePools;
    mutable std::mutex completedMutex;
    std::array<Completed, CompletedRingSize> completed;
    std::uint64_t newestSubmitted = 0;
    std::chrono::steady_clock::time_point newestSubmittedAt{};
    using DrawSnapshotKey = std::tuple<std::uint64_t, SnapshotUse, std::size_t>;
    struct DrawSnapshot {
        std::uint64_t generation;
        std::uint64_t registryGeneration;
        std::list<DrawSnapshotKey>::iterator recent;
        std::shared_ptr<Buffer> buffer;
        std::uint32_t derived;
        VkDeviceSize offset;
    };
    struct DrawSnapshotPool {
        std::list<DrawSnapshotKey> recency;
        std::size_t bytes = 0;
    };
    std::map<DrawSnapshotKey, DrawSnapshot> drawSnapshots;
    std::mutex snapshotMutex;
    std::array<DrawSnapshotPool, 2> drawSnapshotPools;
    std::shared_ptr<Buffer> snapshotArena;
    std::size_t snapshotArenaCursor = 0;
    struct ForkArena {
        std::shared_ptr<Buffer> arena;
        std::size_t cursor = 0;
    };
    std::array<ForkArena, 16> forkArenas;
    std::uint64_t finishedBatches = 0;
    std::uint64_t passGeneration = 0;
    SnapshotSlice allocateSnapshot(std::shared_ptr<Buffer>& arena, std::size_t& cursor, std::size_t bytes);
    void eraseDrawSnapshot(std::map<DrawSnapshotKey, DrawSnapshot>::iterator entry);
};

}

#endif
