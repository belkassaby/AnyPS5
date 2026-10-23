#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUESTBUFFERMEMORY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUESTBUFFERMEMORY_HPP

#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "BdaAbi.hpp"
#include "BufferStoreCover.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

void ShutdownGuestBufferWorkers();

struct GuestMemorySnapshot {
    std::uint64_t address;
    std::span<const std::byte> bytes;
};

// A guest allocation imported with VK_EXT_external_memory_host, usable by the GPU in place.
struct HostImport {
    std::uint64_t base;
    std::uint64_t bytes;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceAddress address;
    void* alias = nullptr;
    std::weak_ptr<const GuestAllocations::Range> range {};
    // A span import (several adjacent registered ranges imported as one, for a region crossing
    // their boundary): the ranges it covers, all of which must stay registered unchanged.
    std::vector<std::weak_ptr<const GuestAllocations::Range>> parts {};
    // Identity for the life of this import (see HostImportSerial); 0 until first asked for.
    std::uint64_t serial = 0;
    bool unwatched = false;
    bool dmaBuf = false;
};

enum class ImportWatch : std::uint8_t { Watch, Unwatch };

struct ImportProbe {
    const char* failure = nullptr;
    VkResult result = VK_SUCCESS;
    std::uint32_t pages = 0;
    std::uint32_t writtenAtImport = 0;
    std::uint32_t writtenAfterSubmit = 0;
    std::uint32_t writtenByCpu = 0;
};

ImportProbe ProbeImportWriteProtection(const Context& context);
ImportProbe ProbeDmaBufImportWriteProtection(const Context& context);
ImportWatch PrepareImportWatch(const Context& context);
void SetImportWatch(const Context& context, ImportWatch watch);

// The host import of the registered allocation containing [address, address + bytes), made on demand
// (alignment and budget permitting), or null. Bytes at `address` are at `address - import->base` in
// the import's buffer.
const HostImport* HostImportFor(const Context& context, std::uint64_t address, std::size_t bytes);
// Whether an existing import covers [address, address + bytes), without reconciling the imports
// with the registry or making one (HostImportFor may take a registry lease): a hint for choices
// made outside the device lock (a sampled texture's path, a dispatch's pre-sync); the path taken
// re-checks with HostImportFor when it binds the import.
bool HostImportCovers(const Context& context, std::uint64_t address, std::size_t bytes);
// Whether a readable registered allocation contains [address, address + bytes) right now (one
// registry lease): a storage image whose memory was freed or re-registered has nothing to store to.
bool RegisteredReadableCovers(std::uint64_t address, std::size_t bytes);

// A persistent device copy of one registered range of the main guest image (which cannot be host
// imported); see GuestBufferMemory.cpp.
struct ImageMirror;
class Recorder;

// Identity of the import serving [address, address + bytes): a serial unique for the life of one
// import, 0 when no import serves the range. An equal serial later means the same VkBuffer still
// backs the range (a re-import gets a new serial), which is what a descriptor set written against it
// needs to stay valid (see the ShaderResources cache). With `reconcile`, imports whose registered
// range changed are dropped first, as an upload does; without it (right after an upload, before the
// work using the import is recorded) the imports are left as they are.
std::uint64_t HostImportSerial(const Context& context, std::uint64_t address, std::size_t bytes, bool reconcile);

// The import table's identity (the fast Revalidate): serials proved under one identity stand
// while the table still has it, i.e. the same device, the same epoch (bumped by every retire)
// and the registry generation it was reconciled with, which must still be the live one.
struct HostImportsProof {
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t epoch = 0;
    std::uint64_t refreshedGeneration = 0;
};
// Whether the table still matches `proof` (one lock); false for an empty or stale proof.
bool HostImportsUnchanged(const Context& context, const HostImportsProof& proof);
// The table's identity when it is reconciled with the live registry (one lock), else empty.
HostImportsProof HostImportsIdentity(const Context& context);

// Identity of the read-only image mirror serving [address, address + bytes) (the counterpart of
// HostImportSerial for main-image ranges, which no import serves): a serial unique for the life of
// one mirror with its top bit set, 0 when no read-only mirror whose registered range still exists
// covers the range. A read-only mirror's bytes and VkBuffer never change, so a descriptor set written
// against it stays valid while the serial is unchanged.
std::uint64_t ImageMirrorSerial(const Context& context, std::uint64_t address, std::size_t bytes);

// Deferred lease release (see GuestBufferMemory.cpp). An address-based build leases every readable
// registered allocation until its write-back, which runs when its batch completed; the guest's
// unmap/mprotect/free of a leased allocation waits for that through the registry's pin waiter, which
// submits the recorder and finishes its batches up to the newest one recorded with a lease. SyncLeaseWork
// says whether the build's work must instead be synced as soon as it is recorded
// (APS5_SYNC_LEASE_DISPATCH=1, the behaviour before deferral). The driver and the draw path count
// their choice with CountLeaseOutcome right after the lease-holding resources were kept by the open
// batch: `batchSerial` is that batch's serial (Recorder::Submissions() + 1 under the device lock), 0
// when synced, and the waiter targets the newest of them. Under APS5_PROFILE_DRAW CountLeaseOutcome
// prints the [address-sync] leases line every 10 s from LeaseCounters (cumulative).
bool SyncLeaseWork();
void CountLeaseOutcome(bool synced, std::uint64_t batchSerial);
struct LeaseStats {
    std::uint64_t deferred = 0;
    std::uint64_t synced = 0;
    // Pin-contention waits made by guest threads (the waiter calls), how many of them finished the
    // newest lease batch, how many had to drain the whole recorder instead (no lease batch noted),
    // how many only dropped the cached address space (no GPU wait, see AddressSpaceStats), and
    // their total time.
    std::uint64_t contentionWaits = 0;
    std::uint64_t contentionSyncs = 0;
    std::uint64_t contentionDrains = 0;
    std::uint64_t cacheDrops = 0;
    double contentionMs = 0;
};
LeaseStats LeaseCounters();

// The cached address space of address-based builds (see GuestBufferMemory.cpp): builds served by
// the cached space, rebuilds by reason, builds whose space could not be published (an import
// retired while it was built) and the pin waiter's drops (cumulative). `enabled` is false under
// APS5_NO_ADDRESS_SPACE_CACHE=1, when every build takes the per-build path.
struct AddressSpaceStats {
    bool enabled = true;
    std::uint64_t hits = 0;
    std::uint64_t rebuiltFirst = 0;
    std::uint64_t rebuiltGeneration = 0;
    std::uint64_t rebuiltEpoch = 0;
    std::uint64_t rebuiltDevice = 0;
    std::uint64_t rebuiltWaiterDrop = 0;
    std::uint64_t unpublished = 0;
    // Builds that left the space for the per-build path: a region partially overlapping one of
    // its regions, or an import retired under it before the upload.
    std::uint64_t dissolvedOverlap = 0;
    std::uint64_t dissolvedImports = 0;
    std::uint64_t waiterDrops = 0;
};
AddressSpaceStats AddressSpaceCounters();

struct MirrorStats {
    std::uint64_t heapMirrors = 0;
    std::uint64_t heapBytes = 0;
    std::uint64_t rebuilds = 0;
    std::uint64_t blocksCopied = 0;
    std::uint64_t heapRefills = 0;
};
MirrorStats MirrorCounters();
void ClearImageMirrors(VkDevice device);
void ClearHostImports(VkDevice device);

// The write-back elision by buffer usage (live storage_usage_elide, env APS5_STORAGE_USAGE_ELIDE=1).
// A compute dispatch's guest buffer region that the shader never reads (no load, no atomic, through
// any element over it) and that it is proved to overwrite in part (DescriptorBinding::bufferCovers,
// evaluated for the dispatch's grid) does not take the write-back of the pending storage-image units
// lying wholly inside the overwritten bytes: those results are dead once the dispatch ran, as on
// the hardware, where the dispatch's stores land after them. The rest of the region is flushed as
// before, and the skipped units are dropped right after the dispatch is recorded (DropElided), so no
// later write-back can land over its stores. Live storage_usage_elide_verify (env
// APS5_STORAGE_USAGE_ELIDE_VERIFY=1) keeps every write-back and checks the decision instead: the
// bytes the dispatch reads under the skipped units (none by construction), and, on host-imported
// regions, that the dispatch did overwrite them (a canary filled into them after the write-back,
// synchronously checked and repaired after the dispatch). APS5_PROFILE_DRAW prints the [storage]
// usage elide line every 10 s with the MiB not written back and the GPU time that saves.
bool StorageUsageElide();
bool StorageUsageVerify();
// Tests: forces the two modes (nothing: back to the live flags and the environment).
void SetStorageUsageModes(std::optional<bool> elide, std::optional<bool> verify);
// The elision's counters (APS5_PROFILE_DRAW: since the last [storage] usage elide line, which resets them).
struct StorageUsageCounts {
    std::uint64_t elidedRegions = 0;
    std::uint64_t droppedUnits = 0;
    std::uint64_t droppedBytes = 0;
    std::uint64_t verifyRegions = 0;
    std::uint64_t canaryWords = 0;
    std::uint64_t readBytes = 0;
    std::uint64_t refused = 0;
};
StorageUsageCounts CurrentStorageUsageCounts();

// The compute dispatch whose resources this thread prepares and records, for evaluating the
// shader's store covers (set by the driver around a direct dispatch with a known grid); null
// otherwise (draws, indirect dispatches, tests that set none).
const ShaderRecompiler::DispatchGrid* CurrentDispatchGrid();
class DispatchGridScope {
public:
    explicit DispatchGridScope(const ShaderRecompiler::DispatchGrid& grid);
    ~DispatchGridScope();
    DispatchGridScope(const DispatchGridScope&) = delete;
    DispatchGridScope& operator=(const DispatchGridScope&) = delete;

private:
    ShaderRecompiler::DispatchGrid grid;
    const ShaderRecompiler::DispatchGrid* previous;
};

// What a shader does with one guest buffer element (DescriptorBinding::bufferRead and
// bufferCovers), with the element's V# words for evaluating the covers.
struct BufferElementUsage {
    bool read = true;
    std::array<std::uint32_t, 4> words{};
    std::vector<ShaderRecompiler::BufferStoreCover> covers;
};

struct AddressCopy {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint64_t committed;
    const char* reason;
};
std::string AddressCopyOverflow(std::vector<AddressCopy> copies, std::uint64_t limit);

class GuestBufferMemory {
public:
    explicit GuestBufferMemory(const Context& context);
    ~GuestBufferMemory();
    void AcquireRegistered();
    // APS5_PROFILE_DRAW: the parts of one address-based build ([address] line every 10 s): the
    // registry lease, the imports pass, the mirror preparation and the mirror compare (blocks and
    // copied blocks) are timed by AcquireRegistered on the calling thread; the caller adds the
    // snapshot compares it made after it (ShaderResources::prepareAddressBindings).
    static void CountAddressBuild(double snapshotsUs);
    // A guest range bound through a descriptor. Both read live guest memory at upload and bind the
    // same way (a storage buffer); AddWritable also notes the range in Writes(), so it gets the
    // write-back's reference copy, a write-back, the recorder's pending-write note and the
    // direct-write marks. AddReadable is for an element the shader is proved never to store to
    // (DescriptorBinding::bufferWritten): the CPU never has to wait for it. `atomic` marks an
    // element the shader updates atomically (DescriptorBinding::bufferAtomic), staged in device
    // memory whatever its size (see AllowDeviceStaging).
    // `usage` (a guest buffer element's, see BufferElementUsage) lets the write-back elision skip
    // results the element's stores overwrite; without it the element counts as read.
    void AddWritable(std::uint64_t address, std::size_t bytes, bool atomic = false, const BufferElementUsage* usage = nullptr);
    void AddReadable(std::uint64_t address, std::size_t bytes);
    // Ranges the recorded work writes through other bindings (storage images it stores to): a
    // region overlapping one is never elided.
    void SetOtherWrites(std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges) { otherWrites = std::move(ranges); }
    // The flush of pending storage results over [begin, end), a region this upload binds in place
    // (a reused build's proof, before the work is recorded again): as UploadFinish's own flush,
    // with the write-back elision. Starts a use: elisions of an earlier use are forgotten.
    void FlushBoundRange(std::uint64_t begin, std::uint64_t end, const char* reason);
    // Forgets the elisions of an earlier use that never reached DropElided (UploadFinish and a
    // reused build's proof call it before their flushes).
    void BeginElisionUse();
    // Whether this use elided write-backs not yet dropped (DropElided).
    bool HasElided() const { return !elided.empty(); }
    // Drops the pending storage-image units the elided flushes of this use skipped, right after the
    // work was recorded (ShaderResources::MarkGpuWrites, under GuestMemory::GpuMutex): those of
    // regions bound in place, or (`copied`) of regions the GPU copies out of their import, once
    // their copy-backs are recorded too. In verify mode, checks the canaries (regions in place).
    // Nothing to do without elisions.
    void DropElided(bool copied);
    // Device-local staging of written and atomic elements inside host imports (see
    // GuestBufferMemory.cpp): allowed only for a build whose every use records its work and then
    // calls RecordCopyBacks (a dispatch), since a staged region's results reach guest memory by
    // that copy alone. Call before Upload.
    void AllowDeviceStaging() { stagingAllowed = true; }
    // Records the copy-in of every staged region anew for another use of this upload (a resource
    // cache hit, from ShaderResources::Revalidate, under GuestMemory::GpuMutex, once the imports
    // were confirmed unchanged): the previous use's copy-back left the shadow behind, and the next
    // RecordCopyBacks copies it back again. Nothing to do without staged regions.
    void RecordStagingCopies(Recorder& recorder);
    void AddSnapshot(const GuestMemorySnapshot& snapshot);
    // Upload is the two stages below back to back. UploadPrepare needs no device lock: it merges the
    // regions, binds the image mirrors and host imports that already serve them (an import pointer is
    // re-checked against the registry epoch later) and copies read-only regions. UploadFinish runs
    // under GuestMemory::GpuMutex: it reconciles and makes imports (retiring one hands its buffer to
    // the recorder), refreshes writable mirrors and flushes pending results (both may wait for
    // recorded work) and copies the regions a descriptor may write, so the window between that copy
    // and the recorder's pending-write note stays closed.
    void Upload(bool addressable);
    void UploadPrepare(bool addressable);
    void UploadFinish(bool addressable);
    VkDescriptorBufferInfo Descriptor(std::uint64_t address, std::size_t bytes, std::uint32_t& adjustment) const;
    static std::uint64_t ViewBytes(std::uint64_t bytes, std::uint32_t adjustment);
    std::vector<ShaderRecompiler::BdaAbi::Range> AddressRanges() const;
    // The BDA table of the cached address space when it serves this upload alone (an address-based
    // build with no region outside it): its ranges, immutable while the space lives, and the
    // space's serial, by which BdaResources reuses the table buffer built for it without a
    // compare. Nothing when a region of this build is not in the space (AddressRanges then merges).
    struct CachedTable {
        std::uint64_t serial;
        const std::vector<ShaderRecompiler::BdaAbi::Range>* ranges;
    };
    std::optional<CachedTable> CachedAddressTable() const;
    // The registered memory an address-based build maps, shared by consecutive builds (see
    // GuestBufferMemory.cpp): its lease and the sorted, immutable regions of the ranges served in
    // place by imports and mirrors. Opaque outside GuestBufferMemory.cpp.
    struct AddressSpace;
    void WriteBack();
    bool WritesOverlap(std::uint64_t address, std::size_t bytes) const;
    // The GPU buffer and byte offset holding [address, address + bytes) of a copied region (not a
    // host import or mirror) this upload's shader writes: in recorded order its contents are the
    // shader's results before any write-back reaches guest memory. Nothing otherwise.
    std::optional<std::pair<VkBuffer, VkDeviceSize>> CopiedSource(std::uint64_t address, std::size_t bytes) const;
    // Whether [address, address + bytes) lies in a region the shader reads and writes in place in a
    // host import: its writes land in guest memory in recorded order, with no write-back.
    bool InPlaceRange(std::uint64_t address, std::size_t bytes) const;
    // Guest ranges the shader may write through descriptors, as [begin, end).
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& Writes() const { return writes; }
    // Whether any written range lives in a copied buffer, so a write-back must run once the GPU is done.
    bool HasCopiedWrites() const;
    // Reports writes into host-imported memory (made by the GPU in place, or copied back into it by
    // RecordCopyBacks) to the write tracking now.
    void MarkDirectWrites() const;
    // Records, into the recorder's open batch, the copy of every written sub-range of a region the
    // GPU copied out of a host import (see Region::gpuCopy) back into the import: what the shader
    // wrote lands in guest memory by the GPU, ordered after the recorded work, so the region needs
    // no CPU write-back (HasCopiedWrites no longer counts it) and the caller notes the ranges as
    // pending writes and marks them as MarkDirectWrites does. Called right after the work using the
    // regions was recorded (ShaderResources::MarkGpuWrites), under GuestMemory::GpuMutex. A use that
    // never calls it (a synchronous draw) stores the staging bytes from the CPU in WriteBack.
    void RecordCopyBacks(Recorder& recorder);
    // Whether registered allocations are pinned until write-back (address-based shaders): by this
    // build's own lease, or by the cached address space it holds.
    bool HoldsLease() const { return !lease.empty() || space != nullptr; }
    // Every uploaded region as [begin, end) when all of them are served by host imports, in place or
    // through a device-local staging copy of the import (nothing was copied through the CPU, so the
    // upload can serve a later identical build), else nothing.
    std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> DirectRegions() const;
    // Stream templates (ShaderResources::StreamRing): an upload DirectRegions refuses only because
    // some regions were copied through the CPU can still serve a later identical draw when every
    // such region is small (at most `limit` bytes), read-only to the shader, live guest bytes (not a
    // snapshot) and not sparse: a hit then copies the current bytes of those ranges into a per-draw
    // ring slice instead of trusting the copy this upload made. `direct` gets the regions
    // DirectRegions would have listed (imports and read-only mirrors, proved by serial as before),
    // `copied` the CPU-copied ones (outside imports, or a non-staged misaligned copy inside one).
    // Nothing (and `why`, when given, the first refusal met) when a region is none of these: a
    // range the shader may write, one over the limit, a sparse heap, a writable or heap mirror (its
    // shadow is diffed at write-back), a snapshot, a staged region, or an address-based space;
    // nothing with `why` None when no region was copied at all (DirectRegions decides) or the upload
    // is not complete. Changes nothing (the [reuse] counters call it with the switch off too).
    enum class StreamRefusal : std::uint8_t { None, Written, Large, Sparse, WritableMirror, Snapshot, Staged, Space, Count };
    struct StreamRegions {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> direct;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> copied;
    };
    std::optional<StreamRegions> StreamableRegions(std::size_t limit, StreamRefusal* why) const;
    // Every region the recorded work reads in place through a host import, as [begin, end): the
    // regions bound in place (`direct`), an address-based build's leased heaps included. A region
    // the GPU copies out of an import (gpuCopy) notes its read itself when the copy is recorded.
    // For the recorder's read tracking (ShaderResources::MarkGpuWrites); nothing once committed.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> InPlaceReads() const;
    bool ReadsInPlace(std::uint64_t address, std::uint64_t bytes) const;
    bool InPlaceReadsOverlap(std::uint64_t address, std::uint64_t bytes) const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> TrackedReads() const;
    // The same split: the cached address space's serial (0 without one) and its in-place regions,
    // and the regions of this build outside it.
    std::uint64_t AddressSpaceSerial() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> AddressSpaceReads() const;
    // The same in-place regions, built once per space, sorted and disjoint (Recorder::NoteSharedReads);
    // null when the space's regions are not disjoint (AddressSpaceReads then).
    std::shared_ptr<const std::vector<std::pair<std::uint64_t, std::uint64_t>>> AddressSpaceReadSet() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> OwnInPlaceReads() const;
    bool SpaceOnly() const { return uploaded && !committed && space != nullptr && regions.empty() && lease.empty(); }
    enum class SpaceShare : std::uint8_t { Taken, Replaced, Imports };
    SpaceShare ShareSpace(std::uint64_t serial, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& sourceWrites);
    bool SpaceCovers(std::uint64_t address, std::size_t bytes) const;
    std::optional<VkDescriptorBufferInfo> SpaceDescriptor(std::uint64_t address, std::size_t bytes, std::uint32_t& adjustment) const;
    bool FinishSharedSpace();
    void TakeSpace(GuestBufferMemory& source);
    struct SpaceView {
        VkDescriptorBufferInfo info;
        std::uint32_t adjustment;
    };
    static std::shared_ptr<const AddressSpace> SpeculatedSpace(const Context& context, std::uint64_t serial, std::span<const std::pair<std::uint64_t, std::size_t>> ranges, std::vector<std::optional<SpaceView>>& views);
    static bool SpaceCovers(const AddressSpace& space, std::uint64_t address, std::size_t bytes);
    bool SharesSpace(const AddressSpace* candidate) const { return space != nullptr && space.get() == candidate; }

private:
    struct Region {
        std::uint64_t begin;
        std::uint64_t end;
        bool writable;
        std::vector<std::byte> snapshot;
        // Shared so a recorded GPU copy into it (see gpuCopy) can keep it in the batch itself, before
        // the caller keeps the whole resources: a throw between the two would otherwise return it to
        // the pool under an unsubmitted copy command.
        std::shared_ptr<Buffer> buffer;
        // Guest bytes as uploaded; write-back only stores bytes the GPU changed.
        std::vector<std::byte> uploaded {};
        // Registered allocation that is imported: its bytes are read from live guest memory, not a snapshot.
        bool hostBacked = false;
        // Set when the region is served by an imported allocation; nothing is copied or written back.
        const HostImport* direct = nullptr;
        // Set when the region covers uncommitted pages: only the `backed` parts (possibly none) are guest
        // memory; the rest reads as zeros and is never stored.
        bool sparse = false;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> backed {};
        // Set when the region is served by an image mirror: nothing is copied; writable mirrors are
        // written back by comparing with the mirror's shadow. Kept alive here for recorded work.
        std::shared_ptr<ImageMirror> mirror {};
        // The mirror is a descriptor sub-range one found by UploadPrepare (not a lease mirror), which
        // UploadFinish confirms still has its Range before the region binds it.
        bool subrangeMirror = false;
        // Set by UploadPrepare when UploadFinish still has device-lock work for the region: a copy a
        // descriptor writes, a writable mirror's refresh, an import to reconcile or make, or the
        // pending-results flush of an import it took.
        bool pending = false;
        // The region lies inside a host import but is misaligned for binding in place: `buffer` is
        // filled by a vkCmdCopyBuffer from the import recorded into the batch (no CPU read, so no
        // flush-hook wait), from `copySource` (the import's VkBuffer, alive until the batch
        // completed, with the guest address of its first byte). `copiedBack` once RecordCopyBacks
        // recorded the written sub-ranges' copies back into the import.
        bool gpuCopy = false;
        VkBuffer copySource = VK_NULL_HANDLE;
        std::uint64_t copySourceBase = 0;
        bool copiedBack = false;
        // An element the shader updates atomically lies inside (AddWritable's `atomic`).
        bool atomic = false;
        // The gpuCopy buffer is a device-local staging shadow (see stagingEligible): no host
        // mapping, so nothing is ever stored from it by the CPU, and the region is taken even when
        // the import could bind it in place.
        bool deviceLocal = false;
        // The device refused the shadow (memory or allocations exhausted): the region takes the
        // path it would take without staging, in both upload stages.
        bool unstaged = false;
        // Whether any part of the region may be read by the recorded work (a readable or snapshot
        // range, an element without usage or one the shader loads from); the write-only elements
        // over it with store covers (`elements`), each at its V#'s base address.
        bool read = true;
        struct CoveredElement {
            std::uint64_t base;
            std::array<std::uint32_t, 4> words;
            std::vector<ShaderRecompiler::BufferStoreCover> covers;
        };
        std::vector<CoveredElement> elements {};
    };
    // Flushes pending storage results over a region the recorded work binds in place or copies on
    // the GPU (`reason` the flush's), eliding the write-back of what the dispatch overwrites.
    void flushBoundRegion(const Region& region, const char* reason, bool addressable, bool copied);
    // The elision's decision for a region (`copied`: the GPU copies it out of its import and back),
    // or false when the whole region must be flushed.
    bool elideFlush(const Region& region, const char* reason, bool addressable, bool copied);
    struct ElidedRange {
        std::uint64_t begin;
        std::uint64_t end;
        // Verify mode: the import's bytes after the write-back, checked after the dispatch.
        std::vector<std::byte> stored;
        bool canary = false;
        // A verify range: its units were stored as without the elision, only checked; the canary
        // still to go in at the end of UploadFinish (armCanaries).
        bool verify = false;
        bool wantsCanary = false;
        // The import table's epoch when the region's import was confirmed (canaries only).
        std::uint64_t importsEpoch = 0;
        // The region is copied out of its import and back by the GPU (dropped after the copy-back).
        bool copied = false;
    };
    // Verify mode: fills the canaries of this use's verify ranges (end of UploadFinish).
    void armCanaries();
    // UploadFinish is running (its regions' imports are the ones the work binds).
    bool uploadFinishing = false;
    std::vector<ElidedRange> elided;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> otherWrites;

    // How [begin, end) lies against the space's base regions.
    enum class BaseOverlap { None, Inside, Partial };
    BaseOverlap baseOverlap(std::uint64_t begin, std::uint64_t end, const Region** owner) const;
    static BaseOverlap baseOverlapIn(const AddressSpace& space, std::uint64_t begin, std::uint64_t end, const Region** owner);
    static std::optional<VkDescriptorBufferInfo> spaceDescriptorIn(const Context& context, const AddressSpace& space, std::uint64_t address, std::size_t bytes, std::uint32_t& adjustment);
    // Copies the space's base regions into this build's own regions (today's per-build form) and
    // drops the space: for a region that partially overlaps a base region, and for a build whose
    // imports changed under the space (`resolve`: the direct regions are re-resolved by UploadFinish).
    void dissolveSpace(bool resolve);
    // The region containing `address`: this build's own regions (sorted after UploadPrepare) or
    // the space's base regions, or null.
    const Region* owner(std::uint64_t address) const;
    // Builds this build's regions from a lease as the per-build path always did: `base` gets the
    // ranges served in place (imports, mirrors), `copied` the ones copied per build.
    struct CopiedRange {
        std::uint64_t begin;
        std::uint64_t end;
        bool writable;
    };
    void addCopiedRange(const CopiedRange& range);
    // The BDA table entry of an uploaded region.
    static ShaderRecompiler::BdaAbi::Range addressRange(const Region& region);
    void validate(std::uint64_t address, std::size_t bytes) const;
    // The region of a descriptor-bound range (AddWritable/AddReadable), committed pages only. A
    // range inside a base region of the space adds nothing: the region serves it, as today's merge
    // of the two did.
    void addDescriptorRegion(std::uint64_t address, std::size_t bytes, bool atomic);
    // Gives a region a buffer of its own with its bytes (guest memory for host-backed and writable
    // ranges, plus the write-back's reference copy for a range a descriptor writes; else its snapshot).
    void copyRegion(Region& region, bool addressable);
    // Whether a region inside a host import that cannot be bound in place is copied by the GPU
    // instead of the CPU (see Region::gpuCopy): live guest bytes, not sparse, at most the size
    // APS5_GPU_COPY_MAX_KIB allows, and APS5_CPU_COPIES unset.
    bool gpuCopyEligible(const Region& region) const;
    // Whether a region inside a host import is staged in device memory instead of bound in place
    // (see GuestBufferMemory.cpp): staging allowed, not address-based, a written element inside,
    // and an atomic element or a size within the written-shadow window. Independent of the import,
    // so UploadPrepare and UploadFinish decide alike.
    bool stagingEligible(const Region& region, bool addressable) const;
    // Records the import-to-buffer copies of the given gpuCopy regions into the open batch, with
    // the barriers that order them after earlier recorded writes and before the shaders reading them.
    void recordGpuCopies(std::span<Region* const> copies, bool addressable);
    void takeHeapReferences();
    Context context;
    bool stagingAllowed = false;
    GuestAllocations::Lease lease;
    // The cached address space this build maps through (its lease pins the ranges); `regions` then
    // holds only the regions outside it (V#s, snapshots, ranges copied per build).
    std::shared_ptr<const AddressSpace> space;
    mutable std::uint64_t writeTableSerial = 0;
    mutable std::shared_ptr<const std::vector<ShaderRecompiler::BdaAbi::Range>> writeTableRanges;
    // Import registry epoch when `direct` pointers were taken at acquire time; they are reused while
    // no import was destroyed since.
    std::uint64_t importsEpoch = 0;
    std::vector<Region> regions;
    // Whether `regions` is in ascending address order (true right after AcquireRegistered, whose
    // regions follow the registry's order), so AddSnapshot can search instead of scanning.
    bool regionsSorted = false;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    std::vector<std::pair<std::uint64_t, std::vector<std::byte>>> heapReferences;
    // UploadPrepare ran (regions are frozen); `uploaded` once UploadFinish ran.
    bool prepared = false;
    bool uploaded = false;
    bool committed = false;
};

}

#endif
