#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "ThreadOwned.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawSpeculation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DebugControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libc/include/General.hpp"
#include "Optimization/include/Optimization/ShaderStageInputInfo.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <filesystem>
#include <string>
#include <unordered_map>
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuCheckpoints.hpp"

namespace AgcDriver::Graphics {
namespace {

bool overlap(std::uint64_t first, std::size_t firstSize, std::uint64_t second, std::size_t secondSize) {
    return first < second + secondSize && second < first + firstSize;
}

VkComponentSwizzle ComponentSwizzleFor(std::uint8_t dstSel) {
    switch (dstSel) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 4: return VK_COMPONENT_SWIZZLE_R;
        case 5: return VK_COMPONENT_SWIZZLE_G;
        case 6: return VK_COMPONENT_SWIZZLE_B;
        case 7: return VK_COMPONENT_SWIZZLE_A;
        default: throw std::runtime_error("AGC graphics: guest texture descriptor has an invalid destination channel selector " + std::to_string(dstSel));
    }
}

VkComponentMapping ViewComponents(const GuestTextureResource& resource) {
    if (IsRepackedTextureFormat(resource.format)) {
        // The guest R field is stored in the Vulkan B channel and guest B in Vulkan R.
        const auto channel = [](VkComponentSwizzle swizzle) { return swizzle == VK_COMPONENT_SWIZZLE_R ? VK_COMPONENT_SWIZZLE_B : swizzle == VK_COMPONENT_SWIZZLE_B ? VK_COMPONENT_SWIZZLE_R : swizzle; };
        return {channel(ComponentSwizzleFor(resource.dstSelX)), channel(ComponentSwizzleFor(resource.dstSelY)), channel(ComponentSwizzleFor(resource.dstSelZ)), channel(ComponentSwizzleFor(resource.dstSelW))};
    }
    if (IsConvertedTextureFormat(resource.format)) return {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    return {ComponentSwizzleFor(resource.dstSelX), ComponentSwizzleFor(resource.dstSelY), ComponentSwizzleFor(resource.dstSelZ), ComponentSwizzleFor(resource.dstSelW)};
}

// Sampled textures are reused across draws and dispatches while their guest bytes are unchanged; a
// byte copy of the guest surface validates each reuse, so CPU or GPU writes to it force a re-upload.
// The key is everything a lookup matches: the device, the eight descriptor words and the swizzle.
struct TextureKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    std::array<std::uint32_t, 4> components;
    bool depthCompare = false;
    bool operator==(const TextureKey&) const = default;
};

std::uint64_t hashWords(std::uint64_t hash, const std::uint32_t* words, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) hash = (hash ^ words[i]) * 1099511628211ull;
    return hash;
}

struct TextureKeyHash {
    std::size_t operator()(const TextureKey& key) const noexcept {
        auto hash = 14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device);
        hash = hashWords(hash, key.words.data(), key.words.size()) ^ static_cast<std::uint64_t>(key.depthCompare);
        return static_cast<std::size_t>(hashWords(hash, key.components.data(), key.components.size()));
    }
};

TextureKey MakeTextureKey(VkDevice device, std::span<const std::uint32_t> words, VkComponentMapping components, bool depthCompare = false) {
    TextureKey key{device, {}, {static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a)}};
    key.depthCompare = depthCompare;
    std::copy(words.begin(), words.end(), key.words.begin());
    return key;
}

// Leaves new elements uninitialized: a texture snapshot is overwritten whole by ReadTextureSurface,
// and zeroing it first (often megabytes per cache miss) cost the committer a sixth of its time.
template<typename T>
struct DefaultInitAllocator : std::allocator<T> {
    template<typename U>
    struct rebind {
        using other = DefaultInitAllocator<U>;
    };
    DefaultInitAllocator() = default;
    template<typename U>
    DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    template<typename U>
    void construct(U* pointer) noexcept(std::is_nothrow_default_constructible_v<U>) { ::new (static_cast<void*>(pointer)) U; }
    template<typename U, typename... Args>
    void construct(U* pointer, Args&&... args) { std::allocator_traits<std::allocator<T>>::construct(static_cast<std::allocator<T>&>(*this), pointer, std::forward<Args>(args)...); }
};
using SnapshotBytes = std::vector<std::byte, DefaultInitAllocator<std::byte>>;

struct CachedTexture {
    TextureKey key;
    std::uint64_t address;
    SnapshotBytes bytes;
    std::shared_ptr<Texture> texture;
    // A fast-cleared surface is cached as its clear texels; it stays valid while the keys are unchanged.
    DccKeys keys = DccKeys::Uncompressed;
    // Write generation the snapshot is known current at (see GuestMemory::CollectWrites).
    std::uint64_t generation = 0;
    // A texture viewing a storage image on the GPU has no snapshot; it stays valid while that image
    // is still its surface's cached image and nothing wrote the guest memory since it matched.
    std::shared_ptr<StorageTexture> source;
    std::uint64_t sourceVersion = 0;
    std::uint64_t accounted = 0;
};

// Entries in use order (front = most recent) with a hash index by key: a lookup is O(1) and the
// eviction takes the back. APS5_NO_TEXTURE_HASH=1 finds entries by scanning the list (the index is
// still kept), the linear lookup of before.
struct TextureCache {
    std::mutex mutex;
    std::list<CachedTexture> entries;
    std::unordered_map<TextureKey, std::list<CachedTexture>::iterator, TextureKeyHash> index;
    std::uint64_t bytes = 0;
};

TextureCache& Textures() {
    static TextureCache cache;
    return cache;
}

bool TextureHashEnabled() {
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_HASH") != nullptr;
    return !disabled;
}

std::list<CachedTexture>::iterator findTexture(TextureCache& cache, const TextureKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

void eraseTexture(TextureCache& cache, std::list<CachedTexture>::iterator it) {
    cache.bytes -= it->accounted;
    cache.index.erase(it->key);
    cache.entries.erase(it);
}

// Moves an entry to the front (most recently used).
void touchTexture(TextureCache& cache, std::list<CachedTexture>::iterator it) {
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
}

// APS5_PROFILE_DRAW: what the sampled-texture and storage-image lookups did, printed as [textures]
// every 10 s (cumulative). Storage-sourced sampled textures need no CPU read; snapshots do, and a
// pending read is a snapshot compare or read made while recorded work still wrote the surface (the
// flush hook may find that work already signaled, so this bounds the hook syncs from above; the
// [hooksync] line attributes the actual waits).
struct TextureCounters {
    std::atomic<std::uint64_t> fromStorage{0};
    std::atomic<std::uint64_t> snapshots{0};
    std::atomic<std::uint64_t> pendingReads{0};
    std::atomic<std::uint64_t> storageFallbacks{0};
    std::atomic<std::uint64_t> records{0};
    std::atomic<std::uint64_t> fastHits{0};
    std::atomic<std::uint64_t> fastMisses{0};
    std::atomic<std::uint64_t> storageHits{0};
    std::atomic<std::uint64_t> storageCreated{0};
    // Storage images a Revalidate refreshed directly instead of through the lookups (T1, see
    // ShaderResources::refreshOwnObjects).
    std::atomic<std::uint64_t> ownRefreshes{0};
    std::atomic<std::int64_t> lastReport{0};
};

TextureCounters& TextureCounts() {
    static TextureCounters counters;
    return counters;
}

void reportTextureCounters() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& counters = TextureCounts();
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto count = [](const std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
    AgcDriver::ProfilePrint_nid_no_patch("[textures] sampled created: %llu from storage images, %llu snapshots (%llu snapshot reads over recorded writes inside cachedTexture, %llu storage-path fallbacks); stage-A records %llu: %llu fast hits, %llu full lookups; storage images %llu hits, %llu created, %llu own-object refreshes\n", count(counters.fromStorage), count(counters.snapshots), count(counters.pendingReads), count(counters.storageFallbacks), count(counters.records), count(counters.fastHits), count(counters.fastMisses), count(counters.storageHits), count(counters.storageCreated), count(counters.ownRefreshes));
}

// What the sampled-texture lookups on this thread proved their returned objects current against,
// taken by object into the ShaderResources being built or revalidated (captureValidation), which
// then repeats the proof from write stamps on its next Revalidate. The lookups run inside one build
// on one thread; records nobody takes (a build that threw, a resident target looked up outside a
// build, a thread that never captures) are dropped by the next capture or by the bound below.
struct LookupRecord {
    const void* object;
    GuestTextureResource resource;
    std::uint64_t bytes;
    DccKeys keys;
    std::uint64_t generation;
    const StorageTexture* source;
    bool depth = false;
    std::array<std::uint32_t, 8> words{};
    std::size_t wordCount = 0;
    VkComponentMapping components{};
};

thread_local std::vector<LookupRecord>* lookupLogSlot = nullptr;

std::vector<LookupRecord>& lookupLog() {
    return ShaderRecompiler::ThreadOwned(lookupLogSlot);
}

void logLookup(const LookupRecord& record) {
    // The bound drops the oldest half, never everything: one build's lookups (a few dozen at most)
    // are the newest records, and a capture must find them even when the bound trips mid-build.
    constexpr std::size_t bound = 1024;
    auto& log = lookupLog();
    if (log.size() >= bound) log.erase(log.begin(), log.begin() + bound / 2);
    log.push_back(record);
}

std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip, std::uint64_t guestBytes = 0);

bool MetadataMoved(const StorageTexture& image, const GuestTextureResource& resource) {
    return resource.dccAddress != 0 && image.Descriptor().dccAddress != resource.dccAddress && !image.ServesKeysAt(resource.dccAddress);
}

// Whether a sampled texture over `resource` can be a view of the surface's cached storage image
// instead of a CPU snapshot (see cachedTexture): the format has a storage form and is not block
// compressed, and the surface lives in host-imported memory, where the image uploads and refreshes
// GPU-direct (a snapshot of such a surface reads and compares its bytes on the CPU, waiting for
// the recorded work that wrote them). APS5_NO_SAMPLED_FROM_STORAGE=1 keeps every sampled texture a
// snapshot as before.
bool SampledFromStorageEligible(const Context& context, std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes) {
    static const bool disabled = std::getenv("APS5_NO_SAMPLED_FROM_STORAGE") != nullptr;
    if (disabled || IsBlockCompressed(format) || !StorageFormatAvailable(context, format)) return false;
    return HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
}

bool SampledFromStorageEligible(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    return SampledFromStorageEligible(context, resource.format, resource.baseAddress, guestBytes);
}

// Whether a sampled texture over a fast-cleared, storage-eligible surface views the cleared storage
// image (see cachedTexture). APS5_NO_CLEARED_VIEW=1 keeps such surfaces snapshots, as before.
bool ClearedViewEnabled() {
    return DebugValue(DebugKey::NoClearedView) == 0;
}

// Surfaces whose storage image could not be made (no writable committed pages, say): remembered
// so the sampled lookup does not throw and fall back on every use. A failure is keyed by the surface
// (address and size: the heap reuses addresses) and holds only while the guest mappings are what
// they were when it failed (the allocation generation): pages committed later, or another surface
// at the address, get a fresh attempt.
struct StorageFailures {
    std::mutex mutex;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> generations;
};

StorageFailures& StorageFailed() {
    static StorageFailures failures;
    return failures;
}

// The cached storage image for a sampled descriptor, or null when it cannot be made (then the
// snapshot path serves the descriptor, as before). Every null return counts as a fallback.
std::shared_ptr<StorageTexture> sampledStorageSource(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    auto& counters = TextureCounts();
    auto& failures = StorageFailed();
    const auto surface = std::make_pair(resource.baseAddress, guestBytes);
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    {
        std::lock_guard lock(failures.mutex);
        if (const auto it = failures.generations.find(surface); it != failures.generations.end()) {
            if (it->second == generation) {
                counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }
            failures.generations.erase(it);
        }
    }
    try {
        return cachedStorageTexture(context, {}, resource, 0, guestBytes);
    } catch (const std::exception& error) {
        counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock(failures.mutex);
        if (failures.generations.size() < 4096 && failures.generations.emplace(surface, generation).second) std::fprintf(stderr, "[textures] sampled texture 0x%llx (%ux%u format %u) keeps the snapshot path: %s\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, error.what());
        return nullptr;
    }
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
std::shared_ptr<Texture> cachedTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components, std::uint64_t guestBytes = 0, bool depthCompare = false) {
    ForkTrap("texture lookup");
    const RefreshSiteScope site("sampled");
    CaptureTrace::Log("sampled-lookup address=%llx width=%u height=%u dcc=%llx", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, static_cast<unsigned long long>(resource.dccAddress));
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
    if (auto depth = DepthSurfaceTexture(context, words, resource, components)) {
        LookupRecord record{depth.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true, {}, std::min<std::size_t>(words.size(), 8), components};
        std::copy_n(words.begin(), record.wordCount, record.words.begin());
        logLookup(record);
        return depth;
    }
    const auto depthBitsWidth = words.size() >= 4 ? ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) : 0u;
    if (depthBitsWidth == 32u) {
        char text[160];
        std::snprintf(text, sizeof(text), "AGC graphics: 32-bit integer read of the depth-layout texture 0x%llx, which is no depth surface drawn with, is not implemented", static_cast<unsigned long long>(resource.baseAddress));
        throw std::runtime_error(text);
    }
    constexpr auto unorm16 = static_cast<std::uint32_t>(ShaderRecompiler::IrBufferFormat::Format16UNorm);
    if (depthBitsWidth == 16u && resource.format != unorm16) {
        auto normalized = resource;
        normalized.format = unorm16;
        return cachedTexture(context, words, normalized, components, guestBytes, depthCompare);
    }
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    const bool profile = LookupOutcomes::Profiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto scanKeys = [&] {
        const auto scanStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto keys = TextureClearKeys(resource, guestBytes);
        if (profile && resource.dccAddress != 0) LookupOutcomes::Add(LookupOutcomes::DccScan, scanStart);
        return keys;
    };
    if (guestBytes == 0) guestBytes = DescribeSurface(resource).guestBytes;
    auto& counters = TextureCounts();
    const auto address = resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(guestBytes);
    // A storage image whose results for this surface (or for a mip chain containing it) are still on
    // the GPU supplies the texture by a view of it; anything else needs those results in guest
    // memory first.
    auto source = StorageTexture::FindPending(address, guestBytes);
    // A repacked (format-43) texture is never a view of the storage image, whose words keep the
    // guest layout: it is a copy of it through the repack pass (Texture::RepackCopyTag), or, when the
    // image cannot serve one, the pending results are flushed and the surface uploaded.
    const bool repacked = IsRepackedTextureFormat(resource.format);
    if (source != nullptr && (depthCompare || (repacked ? !Texture::CanRepackCopyFrom(*source, resource) : !Texture::CanCopyFrom(*source, resource)) || MetadataMoved(*source, resource))) source.reset();
    std::optional<DccKeys> keys;
    bool clearThroughKeys = false;
    if (source != nullptr && resource.dccAddress != 0 && IsDccClear(source->FilledKeys())) {
        keys = scanKeys();
        if (*keys == source->FilledKeys()) {
            std::array<std::byte, 16> probe{};
            if (!FillDccClear(ResolveTextureFormat(resource.format), *keys, resource.dccAlphaOnMsb, probe)) {
                char text[256];
                std::snprintf(text, sizeof(text), "AGC graphics: sampled texture 0x%llx (%ux%u format %u, dcc 0x%llx) reads %s DCC keys filled over its pending image, a clear value the format has no encoding for", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
                throw std::runtime_error(text);
            }
            static const bool traceKeys = std::getenv("APS5_TRACE_DCC_KEYS") != nullptr;
            if (traceKeys) std::fprintf(stderr, "[dcc-keys] sampled 0x%llx through keys 0x%llx reads the %s fill, not the pending image\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
            source.reset();
            clearThroughKeys = true;
        }
    }

    // Otherwise a surface in host-imported memory is viewed through its cached storage image (made
    // here when there is none): its refresh after a CPU or GPU write is a GPU-direct detile from the
    // import, recorded behind the producer, so no bytes are read or compared on the CPU and nothing
    // waits for the producer. A fast-cleared surface (keys) is viewed only through an image whose own
    // descriptor carries the DCC address (below); it stays a snapshot otherwise, its texels not read.
    if (!depthCompare && !repacked && source == nullptr && !clearThroughKeys && SampledFromStorageEligible(context, resource, guestBytes)) {

        keys = scanKeys();
        if (*keys == DccKeys::Uncompressed) {
            source = sampledStorageSource(context, resource, guestBytes);
        } else if (ClearedViewEnabled() && StorageClearAvailable(context, resource.format, *keys)) {
            // A fast-cleared surface is viewed as well, through its image cleared on the GPU (the
            // lookup's Refresh, see StorageTexture::upload), when the image's own descriptor names
            // the same DCC metadata so the refresh sees the keys. A snapshot of the clear texels
            // (filled and uploaded on the CPU, then replaced by a view once results are pending: two
            // 4K uploads per clear at the movie stage) serves the surface otherwise, as before.
            auto candidate = sampledStorageSource(context, resource, guestBytes);
            if (candidate != nullptr && candidate->Descriptor().dccAddress == resource.dccAddress) source = std::move(candidate);
        }
        // The lookup's Refresh may have flushed another image's results over the memory (which
        // marks the keys uncompressed): a surface that stays a snapshot re-reads them behind that
        // flush, as it does behind its own. APS5_NO_KEYS_RESCAN=1 keeps the keys scanned above.
        static const bool rescan = std::getenv("APS5_NO_KEYS_RESCAN") == nullptr;
        if (rescan && source == nullptr) keys.reset();
    }
    // Pages of the surface written since they were last collected are stamped now, before any
    // UnchangedSince (here and on a cache hit) looks at them: the checks only compare stamped blocks,
    // so a write the game made since the last walk is invisible until a collect stamps its page. The
    // walk is memoized per packet, so a texture viewed from a storage image repeats it for free.
    auto generation = GuestMemory::CollectWrites(address, bytes);
    static const bool alwaysRefreshViews = std::getenv("APS5_ALWAYS_REFRESH_VIEWS") != nullptr;
    if (source != nullptr && (alwaysRefreshViews || !GuestMemory::UnchangedSince(address, bytes, source->Generation()))) {
        // The CPU wrote the memory while results were pending: Refresh merges them the usual way.
        source->Refresh();
    }
    // Results stored to guest memory just now are stamped newer than `generation`; a snapshot read
    // after them is current at the generation of a second (memoized) collect. The store marks the
    // surface's DCC keys uncompressed, so the keys are read after it.
    const auto flushStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (source == nullptr && !clearThroughKeys && StorageTexture::FlushPending(address, bytes, nullptr, "sampled texture")) {
        if (profile) LookupOutcomes::Add(LookupOutcomes::PendingFlush, flushStart);
        generation = GuestMemory::CollectWrites(address, bytes);
        keys.reset();
    }
    if (!keys.has_value()) keys = scanKeys();
    if (disabled) {
        if (source != nullptr && repacked) return std::make_shared<Texture>(context, source, resource, components, Texture::RepackCopyTag{});
        if (source != nullptr) return std::make_shared<Texture>(context, source, resource, components);
        std::vector<std::byte> snapshot(bytes);
        ReadTextureSurface(resource, *keys, snapshot);
        return std::make_shared<Texture>(context, *context.detiler, resource, components, snapshot, depthCompare);
    }
    auto& cache = Textures();
    const auto key = MakeTextureKey(context.device, words, components, depthCompare);
    std::lock_guard lock(cache.mutex);
    if (auto it = findTexture(cache, key); it != cache.entries.end()) {
        if (it->source != nullptr) {
            // The view follows the storage image, whatever the GPU wrote to it since; guest memory
            // written meanwhile is taken in by refreshing the image. A fast clear the image cannot
            // see (keys, and no pending results to prefer) ends the view: a snapshot holds the clear.
            if ((source == nullptr || source == it->source) && (source != nullptr || *keys == DccKeys::Uncompressed) && StorageImageServesKeys(*it->source, resource.dccAddress)) {
                static const bool alwaysRefresh = std::getenv("APS5_ALWAYS_REFRESH_VIEWS") != nullptr;
                if (alwaysRefresh || !GuestMemory::UnchangedSince(address, bytes, it->source->Generation())) it->source->Refresh();
                if (it->texture->RepackCopy()) it->texture->RefreshRepackCopy();
                static const bool verifyViews = std::getenv("APS5_VERIFY_TEXTURES") != nullptr;
                static std::atomic<std::uint64_t> viewHits{0};
                if (verifyViews && (viewHits.fetch_add(1, std::memory_order_relaxed) % 20000u) == 0u) std::fprintf(stderr, "[verify-tex] %llu storage-view hits (not byte-checked)\n", static_cast<unsigned long long>(viewHits.load(std::memory_order_relaxed)));
                it->keys = *keys;
                touchTexture(cache, it);
                logLookup({it->texture.get(), resource, guestBytes, *keys, 0, it->source.get()});
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(*keys != DccKeys::Uncompressed ? LookupOutcomes::SampledHitClearedView : LookupOutcomes::SampledHitView, start);
                return it->texture;
            }
        } else if (source == nullptr && it->bytes.size() == guestBytes && it->keys == *keys) {
            // Unwritten pages need no comparison; partially resident textures compare only committed
            // pages. The compare goes through the flush hook: it waits for recorded work over the
            // surface (counted, and named for the [hooksync] line).
            const auto equalsCommitted = [&] {
                if (Recorder::SnapshotWriteOverlaps(address, bytes)) counters.pendingReads.fetch_add(1, std::memory_order_relaxed);
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureCompare);
                return GuestMemory::EqualsCommitted(address, it->bytes);
            };
            static const bool verify = std::getenv("APS5_VERIFY_TEXTURES") != nullptr;
            bool stale = false;
            static std::atomic<std::uint64_t> verified{0};
            if (verify && *keys == DccKeys::Uncompressed && (verified.fetch_add(1, std::memory_order_relaxed) % 20000u) == 0u) std::fprintf(stderr, "[verify-tex] %llu snapshot hits checked\n", static_cast<unsigned long long>(verified.load(std::memory_order_relaxed)));
            if (verify && *keys == DccKeys::Uncompressed && GuestMemory::UnchangedSince(address, it->bytes.size(), it->generation) && !equalsCommitted()) {
                stale = true;
                static std::atomic<std::uint32_t> reported{0};
                if (reported.fetch_add(1, std::memory_order_relaxed) < 200u) {
                    std::fprintf(stderr, "[verify-tex] stale snapshot 0x%llx %ux%u format %u tile %d bytes 0x%zx: memory changed without a tracked write\n", static_cast<unsigned long long>(address), resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), it->bytes.size());
                }
            }
            if (!stale && (*keys != DccKeys::Uncompressed || GuestMemory::UnchangedSince(address, it->bytes.size(), it->generation) || equalsCommitted())) {
                it->generation = generation;
                touchTexture(cache, it);
                logLookup({it->texture.get(), resource, guestBytes, *keys, generation, nullptr});
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(LookupOutcomes::SampledHitSnapshot, start);
                return it->texture;
            }
        }
        eraseTexture(cache, it);
    }
    CachedTexture entry{key, address, SnapshotBytes(source != nullptr ? 0u : bytes), nullptr, *keys, generation};
    entry.accounted = guestBytes;
    if (source != nullptr) {
        entry.source = source;
        entry.sourceVersion = source->Version();
        entry.texture = repacked ? std::make_shared<Texture>(context, source, resource, components, Texture::RepackCopyTag{}) : std::make_shared<Texture>(context, source, resource, components);
        counters.fromStorage.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Snapshot before the upload so a write racing with it is caught by the next comparison.
        if (*keys == DccKeys::Uncompressed && Recorder::SnapshotWriteOverlaps(address, bytes)) counters.pendingReads.fetch_add(1, std::memory_order_relaxed);
        // A clear fill may leave bytes unwritten (an unknown layout, a partial element): zero first.
        if (*keys != DccKeys::Uncompressed) std::fill(entry.bytes.begin(), entry.bytes.end(), std::byte{0});
        ReadTextureSurface(resource, *keys, entry.bytes);
        static const bool traceTextures = std::getenv("APS5_TRACE_TEXTURES") != nullptr;
        if (traceTextures) {
            std::size_t nonzero = 0;
            for (std::size_t i = 0; i < entry.bytes.size(); i += 64) nonzero += entry.bytes[i] != std::byte{0};
            std::fprintf(stderr, "[texture] 0x%llx %ux%u format %u tile %d: %zu of %zu sampled bytes nonzero\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), nonzero, entry.bytes.size() / 64);
        }
        entry.texture = std::make_shared<Texture>(context, *context.detiler, resource, components, entry.bytes, depthCompare);
        counters.snapshots.fetch_add(1, std::memory_order_relaxed);
    }
    constexpr std::uint64_t budget = 2048ull << 20u;
    while (!cache.entries.empty() && cache.bytes + entry.accounted > budget) eraseTexture(cache, std::prev(cache.entries.end()));
    cache.bytes += entry.accounted;
    auto texture = entry.texture;
    logLookup({texture.get(), resource, guestBytes, *keys, generation, source.get()});
    cache.entries.push_front(std::move(entry));
    cache.index[key] = cache.entries.begin();
    reportTextureCounters();
    if (profile) LookupOutcomes::Add(source != nullptr ? LookupOutcomes::SampledMadeView : LookupOutcomes::SampledMadeSnapshot, start);
    return texture;
}

// Storage images stay on the GPU between dispatches: while guest memory still holds what an image was
// last uploaded from or written back as, its next use skips the upload and detile.
struct StorageKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    bool operator==(const StorageKey&) const = default;
};

struct StorageKeyHash {
    std::size_t operator()(const StorageKey& key) const noexcept {
        return static_cast<std::size_t>(hashWords(14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device), key.words.data(), key.words.size()));
    }
};

struct CachedStorageTexture {
    StorageKey key;
    std::uint32_t mip;
    std::shared_ptr<StorageTexture> texture;
};

// As TextureCache: use order with a hash index by key, plus one by image for StorageImageCached.
struct StorageTextureCache {
    std::mutex mutex;
    std::list<CachedStorageTexture> entries;
    std::unordered_map<StorageKey, std::list<CachedStorageTexture>::iterator, StorageKeyHash> index;
    std::unordered_map<const StorageTexture*, std::list<CachedStorageTexture>::iterator> byImage;
    std::unordered_multimap<std::uint64_t, std::pair<StorageKey, std::uint64_t>> arrays;
    std::uint64_t bytes = 0;
};

StorageTextureCache& StorageTextures() {
    static StorageTextureCache cache;
    return cache;
}

std::list<CachedStorageTexture>::iterator findStorage(StorageTextureCache& cache, const StorageKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

std::list<CachedStorageTexture>::iterator findStorageByImage(StorageTextureCache& cache, VkDevice device, const StorageTexture* image) {
    if (TextureHashEnabled()) {
        const auto found = cache.byImage.find(image);
        return found == cache.byImage.end() || found->second->key.device != device ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key.device == device && it->texture.get() == image) return it;
    }
    return cache.entries.end();
}

// Moves whenever a storage image leaves the cache: a proof made while it was unchanged knows every
// image it found cached still is.
std::atomic<std::uint64_t> storageEvictions{0};

bool FastProofEnabled() {
    static const bool enabled = std::getenv("APS5_FAST_PROOF") != nullptr;
    return LiveFlag("fast_proof", enabled);
}

bool VerifyFastProof() {
    static const bool enabled = std::getenv("APS5_VERIFY_FAST_PROOF") != nullptr;
    return enabled;
}

struct FastProofCounts {
    std::atomic<std::uint64_t> calls{0}, accepted{0}, noTokens{0}, pending{0}, evicted{0}, generation{0}, stamps{0};
    std::atomic<std::int64_t> reported{0};
};

FastProofCounts& FastProofStats() {
    static FastProofCounts counts;
    return counts;
}

// [fast-proof] every 10 s under APS5_PROFILE_DRAW: speculated proofs taken without their checks,
// and why the others ran them.
void ReportFastProof() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& stats = FastProofStats();
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = stats.reported.load(std::memory_order_relaxed);
    if (now - last <= 10'000'000'000ll || !stats.reported.compare_exchange_strong(last, now)) return;
    std::fprintf(stderr, "[fast-proof] (10 s) %llu speculated proofs: %llu accepted without checks; checked because: no tokens %llu, stamps moved %llu, pending registry moved %llu, storage evicted %llu, image generation moved %llu\n", static_cast<unsigned long long>(stats.calls.exchange(0)), static_cast<unsigned long long>(stats.accepted.exchange(0)), static_cast<unsigned long long>(stats.noTokens.exchange(0)), static_cast<unsigned long long>(stats.stamps.exchange(0)), static_cast<unsigned long long>(stats.pending.exchange(0)), static_cast<unsigned long long>(stats.evicted.exchange(0)), static_cast<unsigned long long>(stats.generation.exchange(0)));
}

// Evicts an entry: its pending results go to guest memory first (the image may die with the entry).
void evictStorage(StorageTextureCache& cache, std::list<CachedStorageTexture>::iterator it) {
    storageEvictions.fetch_add(1, std::memory_order_acq_rel);
    it->texture->SetCached(false);
    it->texture->Flush();
    cache.bytes -= it->texture->GuestBytes();
    cache.index.erase(it->key);
    cache.byImage.erase(it->texture.get());
    const auto [first, last] = cache.arrays.equal_range(it->texture->Descriptor().baseAddress);
    for (auto array = first; array != last; ++array) {
        if (array->second.first != it->key) continue;
        cache.arrays.erase(array);
        break;
    }
    cache.entries.erase(it);
}

// Storage images are shared by every descriptor of one surface (address, extent, layers, format, tile
// mode): the image holds the whole mip chain, and render targets in the same memory attach to it.
std::array<std::uint32_t, 8> SurfaceKey(const Context& context, const GuestTextureResource& resource) {
    const auto dimension = resource.dimension == TextureDimension::kCube && DebugValue(DebugKey::SplitCubeStorage) == 0 ? TextureDimension::k2DArray : resource.dimension;
    // Guest formats that store in the same Vulkan format share the image (views carry the difference).
    return {static_cast<std::uint32_t>(resource.baseAddress), static_cast<std::uint32_t>(resource.baseAddress >> 32u), resource.width, resource.height, (resource.depthOrLastArray << 16u) | (resource.mipCount & 0xffffu), (static_cast<std::uint32_t>(resource.tileMode) << 12u) | (static_cast<std::uint32_t>(dimension) << 20u), resource.baseArray, static_cast<std::uint32_t>(StorageFormatForGuest(context, resource.format))};
}

struct ExtendedSurfaces {
    std::mutex mutex;
    std::map<std::array<std::uint32_t, 8>, std::uint32_t> levels;
    std::atomic<bool> any{false};
};

ExtendedSurfaces& ExtendedChains() {
    static ExtendedSurfaces surfaces;
    return surfaces;
}

std::uint32_t AllocatedLevels(const GuestTextureResource& resource) {
    return resource.allocatedMipCount != 0 ? resource.allocatedMipCount : resource.mipCount;
}

GuestTextureResource StorageSurface(const Context& context, const GuestTextureResource& viewed) {
    auto& surfaces = ExtendedChains();
    const bool extended = viewed.mipCount > AllocatedLevels(viewed);
    if (!extended && !surfaces.any.load(std::memory_order_acquire)) return viewed;
    auto surface = viewed;
    surface.mipCount = AllocatedLevels(viewed);
    const auto identity = SurfaceKey(context, surface);
    std::lock_guard lock(surfaces.mutex);
    if (extended) {
        auto& levels = surfaces.levels[identity];
        levels = std::max(levels, viewed.mipCount);
        surfaces.any.store(true, std::memory_order_release);
        surface.mipCount = levels;
    } else if (const auto it = surfaces.levels.find(identity); it != surfaces.levels.end()) {
        surface.mipCount = it->second;
    } else {
        surface.mipCount = viewed.mipCount;
    }
    return surface;
}

struct NullTextures {
    std::mutex mutex;
    std::map<std::tuple<VkDevice, int, bool>, std::shared_ptr<Texture>> textures;
};
NullTextures& NullTextureCache() {
    static NullTextures cache;
    return cache;
}

std::shared_ptr<Texture> nullTexture(const Context& context, ShaderRecompiler::DescriptorImageShape shape, bool depthCompare) {
    constexpr VkComponentMapping mapping{VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO};
    auto& cache = NullTextureCache();
    std::lock_guard lock(cache.mutex);
    auto& texture = cache.textures[{context.device, static_cast<int>(shape), depthCompare}];
    if (texture == nullptr) {
        GuestTextureResource resource{};
        resource.width = 1;
        resource.height = 1;
        resource.mipCount = 1;
        resource.tileMode = TextureTileMode::kLinear;
        resource.format = depthCompare ? 22u : 56u;
        switch (shape) {
            case ShaderRecompiler::DescriptorImageShape::Image1D: resource.dimension = TextureDimension::k1D; break;
            case ShaderRecompiler::DescriptorImageShape::Image2D: resource.dimension = TextureDimension::k2D; break;
            case ShaderRecompiler::DescriptorImageShape::Image2DArray: resource.dimension = TextureDimension::k2DArray; break;
            case ShaderRecompiler::DescriptorImageShape::ImageCube: resource.dimension = TextureDimension::kCube; resource.depthOrLastArray = 5; break;
            case ShaderRecompiler::DescriptorImageShape::Image3D: resource.dimension = TextureDimension::k3D; break;
        }
        resource.dstSelX = 4;
        resource.dstSelY = 5;
        resource.dstSelZ = 6;
        resource.dstSelW = 7;
        const std::vector<std::byte> zeros(static_cast<std::size_t>(DescribeSurface(resource).guestBytes));
        texture = std::make_shared<Texture>(context, *context.detiler, resource, mapping, zeros, depthCompare);
        texture->MarkNull();
    }
    return texture;
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
std::shared_ptr<StorageTexture> lookupStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& viewed, std::uint32_t mip, std::uint64_t guestBytes);

std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& viewed, std::uint32_t mip, std::uint64_t guestBytes) {
    const RefreshSiteScope site(std::string_view(g_refreshSite) == "other" ? "storage" : g_refreshSite);
    Require(!DepthStencilPlaneAt(viewed.baseAddress), "storage access to a depth surface's stencil plane is not implemented");
    auto texture = lookupStorageTexture(context, words, viewed, mip, guestBytes);
    if (DepthSurfaceAt(viewed.baseAddress)) SeedStorageFromDepth(context, texture);
    return texture;
}

std::shared_ptr<StorageTexture> lookupStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& viewed, std::uint32_t mip, std::uint64_t guestBytes) {
    ForkTrap("storage lookup");
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    if (disabled) return std::make_shared<StorageTexture>(context, *context.detiler, viewed, mip);
    static_cast<void>(words);
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
    const bool profile = LookupOutcomes::Profiled();
    auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& counters = TextureCounts();
    auto resource = StorageSurface(context, viewed);
    if (resource.dimension == TextureDimension::kCube && DebugValue(DebugKey::SplitCubeStorage) == 0) resource.dimension = TextureDimension::k2DArray;
    static const bool layerViews = std::getenv("APS5_NO_SHARED_STORAGE_LAYERS") == nullptr;
    if (layerViews && (resource.dimension == TextureDimension::k2DArray || resource.dimension == TextureDimension::k1DArray)) resource.baseArray = 0;
    const StorageKey key{context.device, SurfaceKey(context, resource)};
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    if (resource.mipCount > AllocatedLevels(resource)) {
        auto allocated = resource;
        allocated.mipCount = AllocatedLevels(resource);
        if (const auto found = findStorage(cache, {context.device, SurfaceKey(context, allocated)}); found != cache.entries.end()) evictStorage(cache, found);
    }
    // Debug aid: APS5_TRACE_STORAGE_AT=<hex address> prints the first 200 storage lookups at that base.
    static const std::uint64_t traceAt = [] { const char* text = std::getenv("APS5_TRACE_STORAGE_AT"); return text != nullptr ? std::strtoull(text, nullptr, 16) : 0ull; }();
    if (traceAt != 0 && resource.baseAddress == traceAt) {
        static std::atomic<int> traced{0};
        if (traced.fetch_add(1, std::memory_order_relaxed) < 200) std::fprintf(stderr, "[storage-at] lookup %ux%u layers %u mips %u dim %u fmt %u site %s arrays %zu entry %s\n", resource.width, resource.height, resource.depthOrLastArray + 1u, resource.mipCount, static_cast<unsigned>(resource.dimension), resource.format, g_refreshSite, static_cast<std::size_t>(cache.arrays.count(resource.baseAddress)), findStorage(cache, key) != cache.entries.end() ? "cached" : "new");
    }
    // Debug aid: APS5_TRACE_LAYER_VIEWS names why a 2D lookup at an array image's base does not take
    // the array's layer (once per address and reason).
    static const bool traceLayers = std::getenv("APS5_TRACE_LAYER_VIEWS") != nullptr;
    const auto traceLayer = [&](const char* why) {
        if (!traceLayers) return;
        static std::mutex traceMutex;
        static std::set<std::pair<std::uint64_t, std::string>> seen;
        std::lock_guard traceLock(traceMutex);
        if (seen.size() < 256 && seen.emplace(resource.baseAddress, why).second) std::fprintf(stderr, "[layer-views] 0x%llx %ux%u mips %u dim %u site %s dcc 0x%llx: %s\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.mipCount, static_cast<unsigned>(resource.dimension), g_refreshSite, static_cast<unsigned long long>(resource.dccAddress), why);
    };
    if (traceLayers && resource.dimension == TextureDimension::k2D && cache.arrays.find(resource.baseAddress) == cache.arrays.end()) {
        for (const auto& entry : cache.entries) {
            const auto& other = entry.texture->Descriptor();
            if (entry.key.device == context.device && other.baseAddress == resource.baseAddress && entry.texture.get() != nullptr && (other.dimension == TextureDimension::k2DArray || other.dimension == TextureDimension::kCube)) {
                traceLayer(other.dccAddress != 0 ? "an array image at the base is unregistered (it has DCC)" : "an array image at the base is unregistered (geometry)");
                break;
            }
        }
    }
    if (layerViews && resource.dimension == TextureDimension::k2D && !(std::string_view(g_refreshSite) == "storage" || (std::string_view(g_refreshSite) == "render-target" && resource.mipCount > 1)) && cache.arrays.contains(resource.baseAddress)) traceLayer("the lookup site takes no layer view");
    if (layerViews && resource.dimension == TextureDimension::k2D && (std::string_view(g_refreshSite) == "storage" || (std::string_view(g_refreshSite) == "render-target" && resource.mipCount > 1))) {
        // Arrays of several shapes take turns at one base (cube maps of each size reuse one heap
        // block): the one whose layer is this surface serves it.
        const auto layerBytes = guestBytes != 0 ? guestBytes : DescribeSurface(resource).guestBytes;
        const auto format = static_cast<std::uint32_t>(StorageFormatForGuest(context, resource.format));
        const auto [first, last] = cache.arrays.equal_range(resource.baseAddress);
        for (auto array = first; array != last; ++array) {
            if (array->second.first.device != context.device) continue;
            const auto found = findStorage(cache, array->second.first);
            if (found == cache.entries.end()) {
                traceLayer("the registered array image is gone");
                continue;
            }
            const auto& surface = found->texture->Descriptor();
            if (traceLayers) {
                char why[160];
                const char* reason = surface.width != resource.width || surface.height != resource.height ? "extent differs" : surface.mipCount != resource.mipCount ? "mip count differs" : surface.tileMode != resource.tileMode ? "tile mode differs" : array->second.first.words[7] != format ? "storage format differs" : MetadataMoved(*found->texture, resource) ? "metadata moved" : array->second.second != layerBytes ? "layer bytes differ" : "matched";
                std::snprintf(why, sizeof(why), "array %ux%u mips %u layer 0x%llx (lookup layer 0x%llx): %s", surface.width, surface.height, surface.mipCount, static_cast<unsigned long long>(array->second.second), static_cast<unsigned long long>(layerBytes), reason);
                traceLayer(why);
            }
            if (surface.width == resource.width && surface.height == resource.height && surface.mipCount == resource.mipCount && surface.tileMode == resource.tileMode && array->second.first.words[7] == format && !MetadataMoved(*found->texture, resource) && array->second.second == layerBytes) {
                found->texture->Refresh();
                cache.entries.splice(cache.entries.begin(), cache.entries, found);
                counters.storageHits.fetch_add(1, std::memory_order_relaxed);
                if (profile) LookupOutcomes::Add(LookupOutcomes::StorageHit, start);
                return found->texture;
            }
        }
        if (first != last) {
            char why[96];
            std::snprintf(why, sizeof(why), "%ux%u mips %u fell through to its own 2D image", resource.width, resource.height, resource.mipCount);
            traceLayer(why);
        }
    }
    if (layerViews && resource.dimension == TextureDimension::k2DArray && std::string_view(g_refreshSite) == "render-target") {
        // A color target that is one layer of an array surface (APS5_RT_SLICE_LAYERS: cube faces)
        // takes the array image registered at its base with the same geometry: an attachment view
        // ignores the sampled swizzle that gives the two lookups different keys.
        const auto layerBytes = DescribeSurface(resource).layerBytes;
        const auto format = static_cast<std::uint32_t>(StorageFormatForGuest(context, resource.format));
        const auto [first, last] = cache.arrays.equal_range(resource.baseAddress);
        for (auto array = first; array != last; ++array) {
            if (array->second.first.device != context.device || array->second.second != layerBytes || array->second.first.words[7] != format) continue;
            const auto found = findStorage(cache, array->second.first);
            if (found == cache.entries.end()) continue;
            const auto& surface = found->texture->Descriptor();
            if (surface.width != resource.width || surface.height != resource.height || surface.mipCount != resource.mipCount || surface.tileMode != resource.tileMode || surface.depthOrLastArray != resource.depthOrLastArray || MetadataMoved(*found->texture, resource)) continue;
            found->texture->Refresh();
            cache.entries.splice(cache.entries.begin(), cache.entries, found);
            counters.storageHits.fetch_add(1, std::memory_order_relaxed);
            if (profile) LookupOutcomes::Add(LookupOutcomes::StorageHit, start);
            return found->texture;
        }
    }
    auto it = findStorage(cache, key);
    if (it != cache.entries.end() && MetadataMoved(*it->texture, resource)) {
        evictStorage(cache, it);
        it = cache.entries.end();
    }
    if (it != cache.entries.end()) {
        it->texture->Refresh();
        cache.entries.splice(cache.entries.begin(), cache.entries, it);
        counters.storageHits.fetch_add(1, std::memory_order_relaxed);
        if (profile) LookupOutcomes::Add(LookupOutcomes::StorageHit, start);
        return it->texture;
    }
    // Results other images hold over this memory reach it before the new image reads it: a hit's
    // Refresh flushes them, the constructor's upload does not, and a GPU-direct upload reads the
    // import buffer without the flush hook. The flush is recorded ahead of the upload in the batch.
    if (guestBytes == 0) guestBytes = DescribeSurface(resource).guestBytes;
    if (StorageTexture::FlushPending(resource.baseAddress, static_cast<std::size_t>(guestBytes), nullptr, "storage image creation", PublishScope::None) && profile) start = LookupOutcomes::Add(LookupOutcomes::PendingFlush, start);
    CachedStorageTexture entry{key, mip, std::make_shared<StorageTexture>(context, *context.detiler, resource, mip)};
    // The constructor's upload may have recorded into the open batch (a GPU clear, a direct
    // detile) before the image could keep itself (no weak_from_this yet): the batch keeps it here,
    // so an eviction or a failed view before it ran cannot destroy a referenced image.
    // APS5_NO_KEEP_NEW_STORAGE=1 leaves the image to its cache entry alone, as before.
    static const bool keepNew = std::getenv("APS5_NO_KEEP_NEW_STORAGE") == nullptr;
    if (auto* recorder = Recorder::Active(); keepNew && recorder != nullptr && GuestMemory::GpuMutex().HeldByThisThread() && recorder->Recording()) recorder->Keep(entry.texture);
    constexpr std::uint64_t budget = 2048ull << 20u;
    while (!cache.entries.empty() && cache.bytes + entry.texture->GuestBytes() > budget) evictStorage(cache, std::prev(cache.entries.end()));
    cache.bytes += entry.texture->GuestBytes();
    auto texture = entry.texture;
    cache.entries.push_front(std::move(entry));
    cache.index[key] = cache.entries.begin();
    cache.byImage[texture.get()] = cache.entries.begin();
    if (layerViews && resource.dimension == TextureDimension::k2DArray && resource.dccAddress != 0) traceLayer("an array image here has DCC and is not registered");
    if (layerViews && resource.dimension == TextureDimension::k2DArray && resource.dccAddress == 0) {
        const auto geometry = DescribeSurface(resource);
        if (traceLayers) {
            static std::mutex registerMutex;
            static std::set<std::array<std::uint32_t, 8>> registered;
            std::lock_guard registerLock(registerMutex);
            if (registered.size() < 256 && registered.insert(key.words).second) std::fprintf(stderr, "[layer-views] array 0x%llx %ux%u mips %u: %u layers of 0x%llx bytes, 0x%llx in all%s: %s\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.mipCount, geometry.layers, static_cast<unsigned long long>(geometry.layerBytes), static_cast<unsigned long long>(geometry.guestBytes), geometry.thick ? " (thick)" : "", !geometry.thick && geometry.layers >= 2 && geometry.layerBytes * geometry.layers == geometry.guestBytes ? "registered" : "refused");
        }
        if (!geometry.thick && geometry.layers >= 2 && geometry.layerBytes * geometry.layers == geometry.guestBytes) {
            const auto [first, last] = cache.arrays.equal_range(resource.baseAddress);
            if (std::none_of(first, last, [&](const auto& array) { return array.second.first == key; })) cache.arrays.emplace(resource.baseAddress, std::make_pair(key, geometry.layerBytes));
        }
    }
    texture->SetCached(true);
    counters.storageCreated.fetch_add(1, std::memory_order_relaxed);
    if (profile) LookupOutcomes::Add(LookupOutcomes::StorageMade, start);
    return texture;
}

}

void FlushCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot flush textures without a Vulkan device");
    GuestMemory::AssertGpuLockHeld("FlushCachedTextures");
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    for (const auto& entry : cache.entries) {
        if (entry.key.device == device) entry.texture->Flush();
    }
}

std::size_t DumpCachedStorageImages(VkDevice device, const std::string& directory, std::span<const std::uint64_t> addresses, std::optional<std::uint32_t> minimumWidth) {
    ForkTrap("storage dump");
    std::unique_lock gpu(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) gpu.lock();
    const std::uint64_t minimum = minimumWidth ? *minimumWidth : addresses.empty() ? DebugValue(DebugKey::DumpMinWidth) : 0u;
    std::vector<std::shared_ptr<StorageTexture>> images;
    {
        auto& cache = StorageTextures();
        std::lock_guard lock(cache.mutex);
        for (const auto& entry : cache.entries) {
            if ((device == VK_NULL_HANDLE || entry.key.device == device) && (addresses.empty() || std::ranges::find(addresses, entry.texture->Descriptor().baseAddress) != addresses.end()) && entry.texture->Descriptor().width >= minimum) images.push_back(entry.texture);
        }
    }
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    std::size_t saved = 0;
    for (const auto& image : images) {
        const auto& descriptor = image->Descriptor();
        char name[128];
        std::snprintf(name, sizeof(name), "/storage_%llx_%ux%u_t%u_f%u.raw", static_cast<unsigned long long>(descriptor.baseAddress), descriptor.width, descriptor.height, static_cast<unsigned>(descriptor.tileMode), static_cast<unsigned>(image->StorageFormat()));
        try {
            image->DumpBaseLevel(directory + name);
            ++saved;
        } catch (const std::exception& failure) {
            std::fprintf(stderr, "[dump-images] %s: %s\n", name, failure.what());
        }
    }
    std::fprintf(stderr, "[dump-images] saved %zu of %zu storage images to %s\n", saved, images.size(), directory.c_str());
    if (!addresses.empty() || minimumWidth) return saved;
    static const std::uint64_t snapshotAddress = [] { const char* text = std::getenv("APS5_DUMP_TEXTURE"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();
    auto* listing = std::fopen((directory + "/sampled.txt").c_str(), "w");
    auto& sampled = Textures();
    std::lock_guard lock(sampled.mutex);
    std::size_t snapshots = 0;
    for (const auto& entry : sampled.entries) {
        if (device != VK_NULL_HANDLE && entry.key.device != device) continue;
        const auto& words = entry.key.words;
        const auto format = (words[1] >> 20u) & 0x1ffu;
        const auto width = ((words[1] >> 30u) | ((words[2] & 0xfffu) << 2u)) + 1u;
        const auto height = ((words[2] >> 14u) & 0x3fffu) + 1u;
        const auto nonzero = static_cast<std::size_t>(std::count_if(entry.bytes.begin(), entry.bytes.end(), [](std::byte value) { return value != std::byte{0}; }));
        if (listing != nullptr) {
            std::fprintf(listing, "%llx f%u %ux%u sw%u type%u bytes %zu nonzero %zu null %d storage %d keys %d gen %llu words", static_cast<unsigned long long>(entry.address), format, width, height, (words[3] >> 20u) & 31u, words[3] >> 28u, entry.bytes.size(), nonzero, entry.texture && entry.texture->Null() ? 1 : 0, entry.source != nullptr ? 1 : 0, static_cast<int>(entry.keys), static_cast<unsigned long long>(entry.generation));
            for (const auto word : words) std::fprintf(listing, " %08x", word);
            std::fprintf(listing, "\n");
        }
        if (snapshotAddress != 0 && entry.address == snapshotAddress && !entry.bytes.empty()) {
            char name[96];
            std::snprintf(name, sizeof(name), "/snapshot_%llx_f%u_%ux%u_%zu.raw", static_cast<unsigned long long>(entry.address), format, width, height, snapshots++);
            if (auto* file = std::fopen((directory + name).c_str(), "wb")) {
                std::fwrite(entry.bytes.data(), 1, entry.bytes.size(), file);
                std::fclose(file);
            }
        }
    }
    if (listing != nullptr) std::fclose(listing);
    std::fprintf(stderr, "[dump-images] listed %zu sampled textures, %zu snapshots\n", sampled.entries.size(), snapshots);
    return saved;
}

void ClearCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot clear textures without a Vulkan device");
    auto& sampled = Textures();
    {
        std::lock_guard lock(sampled.mutex);
        for (auto it = sampled.entries.begin(); it != sampled.entries.end();) {
            if (it->key.device == device) eraseTexture(sampled, it++);
            else ++it;
        }
    }
    {
        auto& zeros = NullTextureCache();
        std::lock_guard lock(zeros.mutex);
        for (auto it = zeros.textures.begin(); it != zeros.textures.end();) {
            if (std::get<0>(it->first) == device) it = zeros.textures.erase(it);
            else ++it;
        }
    }
    auto& storage = StorageTextures();
    std::lock_guard lock(storage.mutex);
    for (auto it = storage.entries.begin(); it != storage.entries.end();) {
        if (it->key.device != device) {
            ++it;
            continue;
        }
        storageEvictions.fetch_add(1, std::memory_order_acq_rel);
        it->texture->SetCached(false);
        storage.bytes -= it->texture->GuestBytes();
        storage.index.erase(it->key);
        storage.byImage.erase(it->texture.get());
        it = storage.entries.erase(it);
    }
}

bool StorageImageCached(const Context& context, const StorageTexture* image) {
    auto& cache = StorageTextures();
    const ForkGuard lock(cache.mutex);
    const auto it = findStorageByImage(cache, context.device, image);
    if (it == cache.entries.end()) return false;
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
    return true;
}

namespace {

// StorageImageCached for several images under one acquisition of the cache mutex.
bool StorageImagesCached(const Context& context, std::span<const StorageTexture* const> images) {
    if (images.empty()) return true;
    auto& cache = StorageTextures();
    const ForkGuard lock(cache.mutex);
    for (const auto* image : images) {
        const auto it = findStorageByImage(cache, context.device, image);
        if (it == cache.entries.end()) return false;
        cache.entries.splice(cache.entries.begin(), cache.entries, it);
    }
    return true;
}

bool StorageImagesPeek(const Context& context, std::span<const StorageTexture* const> images) {
    if (images.empty()) return true;
    auto& cache = StorageTextures();
    const ForkGuard lock(cache.mutex);
    return std::all_of(images.begin(), images.end(), [&](const StorageTexture* image) { return findStorageByImage(cache, context.device, image) != cache.entries.end(); });
}

}

std::shared_ptr<StorageTexture> CachedStorageSurface(const Context& context, const GuestTextureResource& resource) {
    return cachedStorageTexture(context, {}, resource, 0);
}

bool StorageImageServesKeys(const StorageTexture& image, std::uint64_t dccAddress) {
    GuestTextureResource resource{};
    resource.dccAddress = dccAddress;
    return !MetadataMoved(image, resource);
}

namespace {

const char* roleName(ShaderRecompiler::DescriptorRole role) {
    switch (role) {
        case ShaderRecompiler::DescriptorRole::GuestBuffers: return "GuestBuffers";
        case ShaderRecompiler::DescriptorRole::GuestImages: return "GuestImages";
        case ShaderRecompiler::DescriptorRole::GuestSamplers: return "GuestSamplers";
        case ShaderRecompiler::DescriptorRole::Gds: return "Gds";
        case ShaderRecompiler::DescriptorRole::BdaPagetable: return "BdaPagetable";
        case ShaderRecompiler::DescriptorRole::FaultBuffer: return "FaultBuffer";
        case ShaderRecompiler::DescriptorRole::FlattenedSrt: return "FlattenedSrt";
        case ShaderRecompiler::DescriptorRole::ShaderData: return "ShaderData";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor role");
}

const char* kindName(ShaderRecompiler::DescriptorKind kind) {
    switch (kind) {
        case ShaderRecompiler::DescriptorKind::UniformBuffer: return "UniformBuffer";
        case ShaderRecompiler::DescriptorKind::StorageBuffer: return "StorageBuffer";
        case ShaderRecompiler::DescriptorKind::UniformTexelBuffer: return "UniformTexelBuffer";
        case ShaderRecompiler::DescriptorKind::StorageTexelBuffer: return "StorageTexelBuffer";
        case ShaderRecompiler::DescriptorKind::SampledImage: return "SampledImage";
        case ShaderRecompiler::DescriptorKind::StorageImage: return "StorageImage";
        case ShaderRecompiler::DescriptorKind::Sampler: return "Sampler";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor kind");
}

// Consecutive identical storage descriptors address successive mips of one texture (dynamic-mip
// storage writes), so the second and later ones share the first one's image lookup.
// APS5_NO_STORAGE_DEDUPE=1 looks each one up (and refreshes the image) separately as before.
bool StorageDedupeEnabled() {
    static const bool disabled = std::getenv("APS5_NO_STORAGE_DEDUPE") != nullptr;
    return !disabled;
}

// Whether storage element `element` of `binding` repeats the previous element's eight words.
bool SameAsPreviousStorageElement(const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t element) {
    if (element == 0) return false;
    const auto words = binding.guestDescriptor.begin() + static_cast<std::size_t>(element) * 8u;
    return std::equal(words, words + 8, words - 8);
}

}

ShaderResources::ShaderResources(const Context& context, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes) : ShaderResources(context, std::array<CompiledShader, 2>{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, {ShaderRecompiler::ShaderStage::Fragment, &fragment, static_cast<std::uint32_t>(vertex.pushConstants.size())}}}, target, indexAddress, indexBytes) {}

ShaderResources::ShaderResources(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes, std::span<const GuestMemorySnapshot> snapshots) : context(context), guestMemory(context) {
    ForkTrap("resource build");
    drawBuild = true;
    prepareAddressBindings(shaders, snapshots);
    build(shaders, &target, indexAddress, indexBytes);
}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots) : ShaderResources(context, compute, snapshots, false) {}

namespace {

// Whether a compute stage maps registered guest memory through BDA tables (see prepareAddressBindings).
bool UsesAddressTables(const CompiledShader& compute) {
    Require(compute.program != nullptr, "missing compiled shader");
    return std::any_of(compute.program->bindings.begin(), compute.program->bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; });
}

}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots, bool deferred) : context(context), guestMemory(context), deferredCompute(compute), deferredSnapshots(snapshots) {
    Require(compute.stage == ShaderRecompiler::ShaderStage::Compute, "compute resources require a compute shader");
    // Every use of a compute build is a recorded dispatch that calls MarkGpuWrites, which staged
    // buffers need (a synchronous draw's use would not).
    guestMemory.AllowDeviceStaging();
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    if (!deferred) {
        prepareAddressBindings(shaders, snapshots);
        build(shaders, nullptr, 0, 0);
        forgetDeferredInputs();
        return;
    }
    // Address-based shaders pin and mirror registered memory in prepareAddressBindings (reconciling
    // imports retires buffers to the recorder, refreshing mirrors waits for recorded work): that is
    // device-lock work, so their whole build waits for Complete(). Without tables the call only
    // checks the bindings.
    lockedBuild = UsesAddressTables(compute);
    if (lockedBuild) return;
    unlockedPrepare = true;
    prepareAddressBindings(shaders, snapshots);
    buildPrepare(shaders, nullptr, 0, 0);
}

void ShaderResources::Complete() {
    Require(!completed, "shader resources were already completed");
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    if (lockedBuild) {
        prepareAddressBindings(shaders, deferredSnapshots);
        buildPrepare(shaders, nullptr, 0, 0);
    }
    buildComplete();
    forgetDeferredInputs();
}

void ShaderResources::forgetDeferredInputs() {
    // The compiled shader and the captured regions belong to the dispatch, which returns while this
    // object lives on in the resource cache and the recorder: nothing may reach for them after the
    // build, so they are dropped rather than left dangling.
    deferredCompute = {};
    deferredSnapshots = {};
}

void ShaderResources::build(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes) {
    buildPrepare(shaders, target, indexAddress, indexBytes);
    buildComplete();
}

namespace {

// APS5_PROFILE_DRAW: the [resources] phase totals. Each thread accumulates its builds' phases in
// arrays of its own and merges them into the shared totals every 1000 of its builds (and when it
// ends), so no build takes the shared mutex per phase; the totals lag by up to 999 builds per
// worker.
constexpr std::size_t BuildPhaseCount = static_cast<std::size_t>(ShaderResources::BuildPhase::Count);
constexpr std::array<const char*, BuildPhaseCount> BuildPhaseNames{"bindings", "precollect", "guest memory upload", "descriptors", "stage A", "images", "bda", "stage B"};

struct BuildProfile {
    std::mutex mutex;
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
};

BuildProfile& Builds() {
    static BuildProfile profile;
    return profile;
}

bool BuildProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

struct ThreadBuildProfile {
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
    ~ThreadBuildProfile() { merge(); }
    void merge() {
        auto& profile = Builds();
        std::lock_guard lock(profile.mutex);
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) profile.ms[i] += ms[i];
        profile.builds += builds;
        ms = {};
        if (builds == 0) return;
        builds = 0;
        std::string report;
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) report += " " + std::string(BuildPhaseNames[i]) + "=" + std::to_string(static_cast<long long>(profile.ms[i])) + "ms";
        AgcDriver::ProfilePrint_nid_no_patch("[resources] %llu builds, phase totals:%s\n", static_cast<unsigned long long>(profile.builds), report.c_str());
    }
};

ThreadBuildProfile& ThreadBuilds() {
    thread_local ThreadBuildProfile* profileSlot = nullptr;
    auto& profile = ShaderRecompiler::ThreadOwned(profileSlot);
    return profile;
}

void addBuildPhase(ShaderResources::BuildPhase which, double ms) {
    ThreadBuilds().ms[static_cast<std::size_t>(which)] += ms;
}

// APS5_PROFILE_DRAW: guest buffer elements bound by descriptor (addGuestBuffer) and how many the
// recompiler proved read-only, plus the pending-write notes their uses skipped (MarkGpuWrites),
// printed as [buffers] every 10 s from MarkGpuWrites. Cumulative.
struct BufferWriteCounters {
    std::atomic<std::uint64_t> elements{0};
    std::atomic<std::uint64_t> readOnly{0};
    std::atomic<std::uint64_t> notesSkipped{0};
    std::atomic<std::int64_t> lastReport{0};
};

BufferWriteCounters& BufferWrites() {
    static BufferWriteCounters counters;
    return counters;
}

}

double ShaderResources::phase(BuildPhase which) {
    if (!BuildProfiled()) return 0.0;
    const auto now = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration<double, std::milli>(now - phaseStart).count();
    addBuildPhase(which, ms);
    if (ms > 50) AgcDriver::ProfilePrint_nid_no_patch("[resources] %s took %.0f ms (%zu textures, %zu storage images, %zu buffers, bda %d)\n", BuildPhaseNames[static_cast<std::size_t>(which)], ms, textures.size(), storageTextures.size(), allocations.size(), usesBda ? 1 : 0);
    phaseStart = now;
    return ms;
}

namespace {

void ValidateDescriptorLayout(const Context& context, std::span<const VkDescriptorSetLayoutBinding> bindings) {
    std::map<VkDescriptorType, std::uint64_t> heapTotals;
    std::map<std::pair<VkShaderStageFlags, VkDescriptorType>, std::uint64_t> stageTotals;
    std::map<VkShaderStageFlags, std::uint64_t> stageResources;
    for (const auto& layout : bindings) {
        heapTotals[layout.descriptorType] += layout.descriptorCount;
        for (std::uint32_t bit = 0u; bit < 32u; ++bit) {
            const auto stage = VkShaderStageFlags{1u} << bit;
            if ((layout.stageFlags & stage) == 0u) continue;
            const auto count = stageTotals[{stage, layout.descriptorType}] += layout.descriptorCount;
            const auto all = stageResources[stage] += layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ? 0u : layout.descriptorCount;
            Require(all <= context.limits.maxPerStageResources, "typed heaps exceed per-stage resource capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) Require(count <= context.limits.maxPerStageDescriptorSampledImages, "typed heaps exceed per-stage sampled image capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) Require(count <= context.limits.maxPerStageDescriptorStorageImages, "typed heaps exceed per-stage storage image capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) Require(count <= context.limits.maxPerStageDescriptorSamplers, "typed heaps exceed per-stage sampler capacity");
        }
    }
    Require(heapTotals[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] <= context.limits.maxDescriptorSetSampledImages && heapTotals[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] <= context.limits.maxDescriptorSetStorageImages && heapTotals[VK_DESCRIPTOR_TYPE_SAMPLER] <= context.limits.maxDescriptorSetSamplers, "typed heaps exceed descriptor set capacity");
}

void ValidateRuntimeResources(const CompiledShader& shader, std::span<const std::uint32_t> words) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    const auto& descriptors = program.bindings;
    constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    std::array<const ShaderRecompiler::DescriptorBinding*, bindingCount> heaps{};
    if (program.runtimeImageCount != 0u) {
        for (const auto& binding : descriptors) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
            auto& heap = heaps[binding.binding % bindingCount];
            Require(heap == nullptr, "duplicate runtime image binding");
            heap = &binding;
        }
    }
    Require(words.size() == program.shaderDataDwords, "invalid compact shader data size");
    constexpr auto metadataWords = sizeof(ShaderRecompiler::RuntimeAbi::ResourceMetadata) / sizeof(std::uint32_t);
    Require(program.imageMetadataDword <= words.size() && program.runtimeImageCount <= (words.size() - program.imageMetadataDword) / metadataWords, "runtime image metadata exceeds shader data");
    for (const auto index : program.runtimeImageResources) {
        Require(index < program.runtimeImageCount, "runtime image resource exceeds its compact layout");
        const auto offset = program.imageMetadataDword + index * metadataWords;
        const auto kind = words[offset];
        const auto first = words[offset + 1u];
        const auto count = words[offset + 2u];
        Require(count != 0u, "runtime image metadata has no descriptor elements");
        Require(kind < heaps.size() && heaps[kind] != nullptr, "runtime metadata references an unbound image heap");
        Require(first < heaps[kind]->count && count <= heaps[kind]->count - first, "runtime metadata exceeds its bound heap");
    }
}

}

void ShaderResources::buildPrepare(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes) {
    storesThroughAddresses = std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program == nullptr || shader.program->bdaWrites; });
    PerformanceTimer frameTiming("Resources.Prepare");
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    if (BuildProfiled()) {
        auto& profile = ThreadBuilds();
        if (++profile.builds % 1000 == 0) profile.merge();
    }
    try {
        Require(!shaders.empty() && context.limits.maxBoundDescriptorSets >= 1, "shader descriptor set exceeds device limits");
        if (shaders.size() == 1u) refreshResourceKey = ContentKey(shaders.front(), false, true);
        std::set<std::uint32_t> occupied;
        for (const auto& shader : shaders) {
            Require(shader.program != nullptr, "missing compiled shader");
            ShaderRecompiler::RuntimeAbi::RequireVersion(shader.program->runtimeAbiVersion);
            const VkShaderStageFlags flags = VulkanStage(shader.stage);
            std::uint64_t stageStorageBuffers = 0;
            std::uint64_t stageResources = 0;
            std::vector<std::size_t> offsetsInData;
            std::int64_t shaderData = -1;
            const auto firstSampler = samplers.size();
            const auto firstDeferred = deferredImages.size();
            for (const auto& binding : shader.program->bindings) {
                Require(binding.descriptorSet == 0, "unexpected descriptor set: every shader resource must use descriptor set zero");
                Require(occupied.insert(binding.binding).second, "duplicate shader binding");
                const bool addressRole = binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable || binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer;
                const bool bufferRole = addressRole || binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers || binding.role == ShaderRecompiler::DescriptorRole::ShaderData || binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt || binding.role == ShaderRecompiler::DescriptorRole::Gds;
                const bool imageRole = binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers;
                if (imageRole) {
                    if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage || binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) stageResources += binding.count;
                    Require(stageResources <= context.limits.maxPerStageResources, "shader descriptors exceed per-stage limits");
                    addImageBinding(binding, flags);
                    continue;
                }
                if (!bufferRole) Require(false, std::string("unsupported descriptor role ") + roleName(binding.role));
                if (binding.kind != ShaderRecompiler::DescriptorKind::StorageBuffer) Require(false, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role " + roleName(binding.role) + ": only StorageBuffer is supported");
                Require(!binding.readOnly, "read-only descriptors are unsupported because the recompiler emits no NonWritable decoration");
                Require(binding.count != 0, "empty descriptor binding");
                stageStorageBuffers += binding.count;
                stageResources += binding.count;
                storageBuffers += binding.count;
                Require(stageStorageBuffers <= context.limits.maxPerStageDescriptorStorageBuffers && stageResources <= context.limits.maxPerStageResources, "shader descriptors exceed per-stage limits");
                Binding item{{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding.count, flags, nullptr}, {}};
                if (binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
                    Require(binding.guestDescriptor.size() == static_cast<std::uint64_t>(binding.count) * 4, "guest buffer descriptor must contain four DWORDs per array element");
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        // An element the recompiler did not classify (a producer without the
                        // vector) counts as written, like imageWritten below.
                        const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                        const bool atomic = element < binding.bufferAtomic.size() && binding.bufferAtomic[element];
                        // Likewise an element without the vectors counts as read, with no covers.
                        const bool read = element >= binding.bufferRead.size() || binding.bufferRead[element];
                        const auto* covers = element < binding.bufferCovers.size() ? &binding.bufferCovers[element] : nullptr;
                        const auto index = addGuestBuffer(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 4, 4), target, indexAddress, indexBytes, written, atomic, read, covers);
                        if (const auto traceWrites = DebugValue(DebugKey::TraceBufferWrites); traceWrites != 0 && written && allocations[index].address <= traceWrites && traceWrites < allocations[index].address + allocations[index].size) {
                            static std::atomic<int> traced{0};
                            if (traced.fetch_add(1) < 400) std::fprintf(stderr, "[trace-buffer-write] 0x%llx+0x%zx binding %u element %u by shader 0x%llx stage %d (packet 0x%x)\n", static_cast<unsigned long long>(allocations[index].address), allocations[index].size, binding.binding, element, static_cast<unsigned long long>(GpuCheckpoints::VariantAddress(shader.program->variantId)), static_cast<int>(shader.stage), GuestMemory::CurrentPacket().opcode);
                        }
                        const auto& push = shader.program->pushConstants;
                        if (!push.empty()) {
                            const auto position = shader.program->memoryOffsetDword * 4u + element;
                            Require(position < push.size(), "guest buffer offset lies outside the shader's push constants");
                            allocations[index].pushByte = static_cast<std::int32_t>(shader.pushConstantOffset + position);
                        } else {
                            allocations[index].dataByte = shader.program->memoryOffsetDword * 4u + element;
                            offsetsInData.push_back(index);
                        }
                        item.allocations.push_back(index);
                    }
                } else if (addressRole) {
                    item.allocations.push_back(allocations.size());
                    allocations.push_back({0, 0, false, nullptr, binding.role});
                } else if (binding.role == ShaderRecompiler::DescriptorRole::Gds) {
                    Require(binding.count == 1 && binding.guestDescriptor.empty(), "invalid GDS descriptor contract");
                    const auto gds = Pm4::GdsAddress();
                    guestMemory.AddWritable(gds, Pm4::GdsBytes, true);
                    item.allocations.push_back(allocations.size());
                    allocations.push_back({gds, Pm4::GdsBytes, true, nullptr, ShaderRecompiler::DescriptorRole::ShaderData, true});
                } else {
                    Require(binding.count == 1, "shader data and flattened SRT descriptors must not be arrays");
                    Require(!binding.guestDescriptor.empty(), "empty shader data descriptor");
                    if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) {
                        ValidateRuntimeResources(shader, binding.guestDescriptor);
                    }
                    item.allocations.push_back(addDataBuffer(binding.guestDescriptor));
                    if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) shaderData = static_cast<std::int64_t>(item.allocations.back());
                }
                bindings.push_back(std::move(item));
            }
            for (const auto index : offsetsInData) allocations[index].dataAllocation = shaderData;
            for (auto index = firstDeferred; index < deferredImages.size(); ++index) {
                deferredImages[index].firstSampler = firstSampler;
                deferredImages[index].samplerCount = samplers.size() - firstSampler;
            }
        }
        Require(storageBuffers <= context.limits.maxDescriptorSetStorageBuffers, "pipeline descriptors exceed device limits");
        timing.bindingsMs = phase(BuildPhase::Bindings);
        // For every build, locked ones included: their stage B then takes the fast path too, and the
        // collects cost the same wherever they run.
        if (precollectImages()) phase(BuildPhase::Precollect);
        guestMemory.UploadPrepare(usesBda);
        timing.uploadMs = phase(BuildPhase::Upload);
        std::vector<VkDescriptorSetLayoutBinding> description;
        for (const auto& binding : bindings) {
            description.push_back(binding.layout);
            layoutKey.insert(layoutKey.end(), {binding.layout.binding, static_cast<std::uint32_t>(binding.layout.descriptorType), binding.layout.descriptorCount, binding.layout.stageFlags});
        }
        static const bool noLayoutCache = std::getenv("APS5_NO_LAYOUT_CACHE") != nullptr;
        static const bool noPoolCache = std::getenv("APS5_NO_POOL_CACHE") != nullptr;
        if (context.descriptorCache != nullptr && !noLayoutCache) {
            _layout = context.descriptorCache->Layout(layoutKey, description);
        } else {
            ValidateDescriptorLayout(context, description);
            VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            info.bindingCount = static_cast<std::uint32_t>(description.size());
            info.pBindings = description.data();
            Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &_layout), "vkCreateDescriptorSetLayout");
            ownsLayout = true;
        }
        if (!bindings.empty()) {
            // The set is sized from the plan: every image element becomes exactly one descriptor
            // when stage B looks it up.
            std::vector<VkDescriptorPoolSize> sizes;
            if (storageBuffers != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<std::uint32_t>(storageBuffers)});
            if (plannedSampledImages != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, plannedSampledImages});
            if (plannedStorageImages != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, plannedStorageImages});
            if (!samplers.empty()) sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER, static_cast<std::uint32_t>(samplers.size())});
            if (context.descriptorCache != nullptr && !noPoolCache) {
                const auto allocated = context.descriptorCache->Allocate(_layout, sizes);
                _set = allocated.set;
                cachePool = allocated.pool;
            }
            if (_set == VK_NULL_HANDLE) {
                VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
                poolInfo.maxSets = 1;
                poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
                poolInfo.pPoolSizes = sizes.data();
                Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
                VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                allocation.descriptorPool = pool;
                allocation.descriptorSetCount = 1;
                allocation.pSetLayouts = &_layout;
                Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &_set), "vkAllocateDescriptorSets");
            }
        }
        timing.descriptorsMs = phase(BuildPhase::Descriptors);
        if (BuildProfiled()) {
            timing.prepareMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageA, timing.prepareMs);
        }
    } catch (...) {
        release();
        throw;
    }
}

void ShaderResources::buildComplete() {
    PerformanceTimer frameTiming("Resources.Complete");
    Require(!completed, "cannot update a published descriptor heap");
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    try {
        // The lookups run in plan order: Revalidate walks the bindings the same way, and consecutive
        // storage elements of one mip chain share the previous element's image.
        for (const auto& deferred : deferredImages) resolveImageBinding(*deferred.binding, bindings[deferred.index], std::span<const std::shared_ptr<Sampler>>(samplers).subspan(deferred.firstSampler, deferred.samplerCount));
        deferredImages.clear();
        // The records served their purpose: each holds the cache entry's objects as of stage A,
        // which would otherwise keep a replaced texture or an evicted storage image (and its device
        // memory) alive, outside the caches' budgets, for as long as this object is cached.
        imageRecords.clear();
        imageRecords.shrink_to_fit();
        nextImageRecord = 0;
        Require(textures.size() == plannedSampledImages && storageTextures.size() == plannedStorageImages, "image lookups disagree with the descriptor plan");
        timing.bindingsMs += phase(BuildPhase::Images);
        guestMemory.SetOtherWrites(writtenImageRanges());
        guestMemory.UploadFinish(usesBda);
        timing.uploadMs += phase(BuildPhase::Upload);
        if (usesBda) bda = std::make_unique<BdaResources>(context, guestMemory);
        else if (usesFaultBuffer) bda = std::make_unique<BdaResources>(context);
        if (bda) {
            bda->SetMayWrite(storesThroughAddresses);
            bda->SetCompute(!drawBuild);
        }
        phase(BuildPhase::Bda);
        if (_set != VK_NULL_HANDLE) {
            // One update call for the whole set: the info arrays are sized up front so every write's
            // pointer into them stays valid until the call.
            std::size_t bufferCount = 0;
            std::size_t imageCount = 0;
            for (const auto& binding : bindings) {
                bufferCount += binding.allocations.size();
                if (binding.layout.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) imageCount += binding.layout.descriptorCount;
            }
            std::vector<VkDescriptorBufferInfo> buffers;
            std::vector<VkDescriptorImageInfo> images;
            buffers.reserve(bufferCount);
            images.reserve(imageCount);
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(bindings.size());
            for (const auto& binding : bindings) {
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = _set;
                write.dstBinding = binding.layout.binding;
                write.descriptorCount = binding.layout.descriptorCount;
                write.descriptorType = binding.layout.descriptorType;
                switch (binding.layout.descriptorType) {
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                        write.pBufferInfo = buffers.data() + buffers.size();
                        for (const auto index : binding.allocations) buffers.push_back(descriptor(allocations[index]));
                        break;
                    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) {
                            if (index != std::numeric_limits<std::size_t>::max()) {
                                if (addressTextureSlots.size() < textures.size()) addressTextureSlots.resize(textures.size(), std::numeric_limits<std::size_t>::max());
                                addressTextureSlots[index] = images.size();
                            }
                            if (index == std::numeric_limits<std::size_t>::max()) images.push_back({VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL});
                            else images.push_back({VK_NULL_HANDLE, textureBaseLevel[index] ? textures[index]->BaseLevelView() : textureFirstLayer[index] ? textures[index]->FirstLayerView() : textures[index]->View(), textures[index]->Layout()});
                        }
                        Require(binding.imageAllocations.size() == binding.layout.descriptorCount, "descriptor allocations disagree with compact binding");
                        break;
                    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) {
                            if (index == std::numeric_limits<std::size_t>::max()) images.push_back({VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL});
                            else images.push_back({VK_NULL_HANDLE, storageAtomic64[index] ? storageTextures[index]->Atomic64View(storageMips[index], storageFirstLayer[index], storageLayers[index]) : storageAtomic[index] ? storageTextures[index]->AtomicView(storageMips[index], storageFirstLayer[index], storageLayers[index]) : storageTextures[index]->StorageView(storageMips[index], storageFirstLayer[index], storageLayers[index]), VK_IMAGE_LAYOUT_GENERAL});
                        }
                        Require(binding.imageAllocations.size() == binding.layout.descriptorCount, "descriptor allocations disagree with compact binding");
                        break;
                    case VK_DESCRIPTOR_TYPE_SAMPLER:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) images.push_back({samplers[index]->Handle(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
                        Require(binding.imageAllocations.size() == binding.layout.descriptorCount, "descriptor allocations disagree with compact binding");
                        break;
                    default: throw std::runtime_error("AGC graphics: ShaderResources encountered an unknown descriptor type while writing the descriptor set");
                }
                writes.push_back(write);
            }
            context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            if (!usesBda && DrawSetWrites()) {
                setBuffers = buffers;
                setImages = images;
                setWrites.reserve(writes.size());
                for (const auto& write : writes) {
                    const bool image = write.pImageInfo != nullptr;
                    auto kept = write;
                    kept.dstSet = VK_NULL_HANDLE;
                    kept.pBufferInfo = nullptr;
                    kept.pImageInfo = nullptr;
                    setWrites.push_back({kept, image ? static_cast<std::size_t>(write.pImageInfo - images.data()) : static_cast<std::size_t>(write.pBufferInfo - buffers.data()), image});
                }
            }
            if (drawBuild && usesBda && AddressTemplates()) {
                addressBuffers = buffers;
                addressImages = images;
                addressInfos.assign(allocations.size(), std::numeric_limits<std::size_t>::max());
                std::size_t position = 0;
                for (const auto& binding : bindings) {
                    if (binding.layout.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) continue;
                    for (const auto index : binding.allocations) {
                        addressInfos[index] = position;
                        if (allocations[index].role == ShaderRecompiler::DescriptorRole::FaultBuffer) addressFaults.push_back(position);
                        ++position;
                    }
                }
                addressWrites.reserve(writes.size());
                for (const auto& write : writes) {
                    const bool image = write.pImageInfo != nullptr;
                    auto kept = write;
                    kept.dstSet = VK_NULL_HANDLE;
                    kept.pBufferInfo = nullptr;
                    kept.pImageInfo = nullptr;
                    addressWrites.push_back({kept, image ? static_cast<std::size_t>(write.pImageInfo - images.data()) : static_cast<std::size_t>(write.pBufferInfo - buffers.data()), image});
                }
            }
        }
        for (const auto& allocation : allocations) {
            if (allocation.adjustment == 0) continue;
            if (allocation.pushByte >= 0) {
                pushPatches.emplace_back(static_cast<std::uint32_t>(allocation.pushByte), allocation.adjustment);
                continue;
            }
            Require(allocation.dataAllocation >= 0, "a guest buffer off the storage buffer offset alignment in a shader without shader data is not implemented");
            auto& data = allocations[static_cast<std::size_t>(allocation.dataAllocation)];
            Require(data.buffer != nullptr && allocation.dataByte < data.size, "guest buffer offset lies outside the shader's data buffer");
            data.buffer->Bytes()[allocation.dataByte] = static_cast<std::byte>(allocation.adjustment);
            dataPatches.push_back({static_cast<std::size_t>(allocation.dataAllocation), allocation.dataByte, allocation.adjustment});
        }
        timing.descriptorsMs += phase(BuildPhase::Descriptors);
        noteReusable();
        completed = true;
        if (BuildProfiled()) {
            timing.completeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageB, timing.completeMs);
            reportDescriptorCaches();
        }
    } catch (...) {
        release();
        throw;
    }
}

// Whether this build can serve later dispatches of the same content (see ContentKey): nothing to do
// once the GPU completed (no BDA, no copied written buffers, no lease) and every guest buffer bound
// in place through a host import (or staged in device memory from one, see
// GuestBufferMemory::DirectRegions) whose identity is recorded for Revalidate.
namespace {

// APS5_NO_TEMPLATE_DATA_REFRESH=1: compute keys keep the ShaderData/FlattenedSrt words and no
// template refreshes its data buffers (see ShaderResources::ContentKey).
bool TemplateDataRefresh() {
    static const bool enabled = std::getenv("APS5_NO_TEMPLATE_DATA_REFRESH") == nullptr;
    return enabled;
}

bool DataRole(ShaderRecompiler::DescriptorRole role) {
    return role == ShaderRecompiler::DescriptorRole::ShaderData || role == ShaderRecompiler::DescriptorRole::FlattenedSrt;
}

// The largest data buffer vkCmdUpdateBuffer refreshes; a template with a bigger one is not reused.
constexpr std::size_t MaxRefreshBytes = 65536;

// FNV-1a over one data buffer's words, its count first (DataWordsHash).
constexpr std::uint64_t FnvOffset = 14695981039346656037ull;
constexpr std::uint64_t FnvPrime = 1099511628211ull;
void mixDataWords(std::uint64_t& hash, std::span<const std::uint32_t> words) {
    hash = (hash ^ static_cast<std::uint64_t>(words.size())) * FnvPrime;
    for (const auto word : words) hash = (hash ^ word) * FnvPrime;
}

// ResourceCache::Find and Touch calls (the [rescache] line).
std::atomic<std::uint64_t> resourceCacheFinds{0};
std::atomic<std::uint64_t> resourceCacheTouches{0};

// APS5_PROFILE_DRAW: noteReusable's verdicts on draw builds and the stream hits' copies
// (ShaderResources::ReuseCounters). Relaxed atomics: one add per build or streamed range.
constexpr std::size_t StreamRefusalCount = static_cast<std::size_t>(GuestBufferMemory::StreamRefusal::Count);
struct ReuseCounts {
    std::atomic<std::uint64_t> builds{0};
    std::atomic<std::uint64_t> inPlace{0};
    std::atomic<std::uint64_t> streamMade{0};
    std::atomic<std::uint64_t> streamableOff{0};
    std::atomic<std::uint64_t> completion{0};
    std::atomic<std::uint64_t> perUseTexture{0};
    std::atomic<std::uint64_t> largeData{0};
    std::array<std::atomic<std::uint64_t>, StreamRefusalCount> copyRefused{};
    std::atomic<std::uint64_t> straddled{0};
    std::atomic<std::uint64_t> noSerial{0};
    std::atomic<std::uint64_t> copiedRanges{0};
    std::atomic<std::uint64_t> copiedBytes{0};
    std::atomic<std::uint64_t> streamedRanges{0};
    std::atomic<std::uint64_t> streamedBytes{0};
    std::atomic<std::uint64_t> verified{0};
};

ReuseCounts& Reuses() {
    static ReuseCounts counts;
    return counts;
}

void countCopied(std::span<const std::pair<std::uint64_t, std::uint64_t>> copied) {
    std::uint64_t bytes = 0;
    for (const auto& [begin, end] : copied) bytes += end - begin;
    Reuses().copiedRanges.fetch_add(copied.size(), std::memory_order_relaxed);
    Reuses().copiedBytes.fetch_add(bytes, std::memory_order_relaxed);
}

}

bool ShaderResources::StreamRing() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_STREAM_RING");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return LiveFlag("stream_ring", enabled);
}

std::size_t ShaderResources::StreamLimit() {
    // The snapshot ring hands out slices of at most 256 KiB (Recorder::AllocateDrawSnapshot); the
    // default keeps a hit's copies to the small constant ranges the design targets.
    static const std::size_t limit = [] {
        const char* value = std::getenv("APS5_STREAM_RING_MAX_KIB");
        const long kib = value != nullptr ? std::strtol(value, nullptr, 10) : 16;
        return static_cast<std::size_t>(std::clamp<long>(kib, 1, 256)) * 1024u;
    }();
    return limit;
}

ShaderResources::ReuseStats ShaderResources::ReuseCounters() {
    auto& counts = Reuses();
    const auto load = [](const std::atomic<std::uint64_t>& value) { return value.load(std::memory_order_relaxed); };
    ReuseStats stats;
    stats.builds = load(counts.builds);
    stats.inPlace = load(counts.inPlace);
    stats.streamMade = load(counts.streamMade);
    stats.streamableOff = load(counts.streamableOff);
    stats.completion = load(counts.completion);
    stats.perUseTexture = load(counts.perUseTexture);
    stats.largeData = load(counts.largeData);
    for (std::size_t index = 0; index < StreamRefusalCount; ++index) stats.copyRefused[index] = load(counts.copyRefused[index]);
    stats.straddled = load(counts.straddled);
    stats.noSerial = load(counts.noSerial);
    stats.copiedRanges = load(counts.copiedRanges);
    stats.copiedBytes = load(counts.copiedBytes);
    stats.streamedRanges = load(counts.streamedRanges);
    stats.streamedBytes = load(counts.streamedBytes);
    stats.verified = load(counts.verified);
    return stats;
}

void ShaderResources::clearStreamed() {
    if (!streamed) return;
    streamed = false;
    for (auto& allocation : allocations) allocation.streamed = false;
}

bool ShaderResources::tagStreamed(std::span<const std::pair<std::uint64_t, std::uint64_t>> copied) {
    // Every guest buffer lies inside the one merged region its descriptor range made
    // (addDescriptorRegion widens it to DWORDs; UploadPrepare merges overlaps), so one that only
    // overlaps a copied region cannot happen; it is refused rather than half rebound.
    bool any = false;
    for (auto& allocation : allocations) {
        if (!allocation.guest || allocation.size == 0) continue;
        const auto begin = allocation.address;
        const auto end = allocation.address + allocation.size;
        const auto inside = std::find_if(copied.begin(), copied.end(), [&](const auto& range) { return begin >= range.first && end <= range.second; });
        if (inside == copied.end()) {
            if (std::any_of(copied.begin(), copied.end(), [&](const auto& range) { return begin < range.second && range.first < end; })) {
                clearStreamed();
                return false;
            }
            continue;
        }
        // StreamableRegions refused every region a written element lies in (WritesOverlap).
        if (allocation.written) {
            clearStreamed();
            return false;
        }
        allocation.streamed = true;
        streamed = true;
        any = true;
    }
    if (!any) clearStreamed();
    return any;
}

void ShaderResources::noteReusable() {
    // Taken whether or not the object turns out reusable: the lookups' records are this build's.
    captureValidation();
    reusable = false;
    addressReusable = false;
    directRegions.clear();
    clearStreamed();
    // The [reuse] verdict of a draw build (dispatches are not counted).
    const bool counted = drawBuild && BuildProfiled();
    auto& counts = Reuses();
    const auto count = [&](std::atomic<std::uint64_t>& verdict) {
        if (!counted) return;
        counts.builds.fetch_add(1, std::memory_order_relaxed);
        verdict.fetch_add(1, std::memory_order_relaxed);
    };
    if (NeedsCompletion() || HoldsLease()) {
        count(counts.completion);
        noteAddressReusable();
        return;
    }
    if (std::any_of(textures.begin(), textures.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->RefreshedPerUse(); })) {
        count(counts.perUseTexture);
        return;
    }
    if (TemplateDataRefresh() && std::any_of(allocations.begin(), allocations.end(), [](const Allocation& allocation) { return allocation.buffer != nullptr && !allocation.guest && allocation.size > MaxRefreshBytes; })) {
        count(counts.largeData);
        return;
    }
    auto regions = guestMemory.DirectRegions();
    std::optional<GuestBufferMemory::StreamRegions> stream;
    if (!regions.has_value()) {
        // A region was copied through the CPU. A draw build may still become a stream template
        // (see StreamRing); with the switch off the same predicate runs for the [reuse] count only,
        // so a baseline run measures the share of builds a stream template would serve.
        if (!drawBuild) return;
        const bool enabled = StreamRing();
        if (!enabled && !counted) return;
        auto why = GuestBufferMemory::StreamRefusal::None;
        stream = guestMemory.StreamableRegions(StreamLimit(), &why);
        if (!stream.has_value()) {
            count(counts.copyRefused[static_cast<std::size_t>(why)]);
            return;
        }
        if (!enabled) {
            count(counts.streamableOff);
            if (counted) countCopied(stream->copied);
            return;
        }
        if (!tagStreamed(stream->copied)) {
            count(counts.straddled);
            return;
        }
        regions = std::move(stream->direct);
    }
    for (const auto& [begin, end] : *regions) {
        // No reconcile here: the upload just took these imports, and the set is about to be recorded
        // against them.
        auto serial = HostImportSerial(context, begin, static_cast<std::size_t>(end - begin), false);
        // Read-only image mirrors (exe ranges) are as stable as imports; their serials have the top
        // bit set, so the two spaces never collide.
        if (serial == 0) serial = ImageMirrorSerial(context, begin, static_cast<std::size_t>(end - begin));
        if (serial == 0) {
            // Not reusable after all: no stream tag may outlive the refusal.
            clearStreamed();
            count(counts.noSerial);
            return;
        }
        directRegions.push_back({begin, end, serial});
    }
    reusable = true;
    count(streamed ? counts.streamMade : counts.inPlace);
    if (counted && streamed) countCopied(stream->copied);
}

namespace {

struct AddressTemplateCounts {
    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> detached{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShaderResources::AddressMiss::Count)> misses{};
    std::atomic<std::uint64_t> movedBuffers{0};
    std::atomic<std::uint64_t> timedHits{0};
    std::array<std::atomic<std::uint64_t>, 5> nanoseconds{};
};

AddressTemplateCounts& AddressCounts() {
    static AddressTemplateCounts counts;
    return counts;
}

}

bool ShaderResources::DrawSetWrites() {
    static const bool requested = std::getenv("APS5_DRAW_SET_WRITES") != nullptr;
    return LiveFlag("draw_set_writes", requested);
}

bool ShaderResources::AddressTemplates() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_ADDRESS_TEMPLATES");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

ShaderResources::AddressTemplateStats ShaderResources::AddressTemplateCounters() {
    auto& counts = AddressCounts();
    AddressTemplateStats stats;
    stats.hits = counts.hits.load(std::memory_order_relaxed);
    stats.detached = counts.detached.load(std::memory_order_relaxed);
    for (std::size_t index = 0; index < stats.misses.size(); ++index) stats.misses[index] = counts.misses[index].load(std::memory_order_relaxed);
    stats.movedBuffers = counts.movedBuffers.load(std::memory_order_relaxed);
    stats.timedHits = counts.timedHits.load(std::memory_order_relaxed);
    const auto us = [&](std::size_t phase) { return static_cast<double>(counts.nanoseconds[phase].load(std::memory_order_relaxed)) / 1000.0; };
    stats.movedUs = us(0);
    stats.spaceUs = us(1);
    stats.proofUs = us(2);
    stats.finishUs = us(3);
    stats.setUs = us(4);
    return stats;
}

void ShaderResources::noteAddressReusable() {
    const bool eligible = drawBuild && usesBda && bda != nullptr && !addressWrites.empty() && AddressTemplates() && guestMemory.SpaceOnly() && !guestMemory.HasCopiedWrites()
        && std::none_of(textures.begin(), textures.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->RefreshedPerUse(); })
        && !(TemplateDataRefresh() && std::any_of(allocations.begin(), allocations.end(), [](const Allocation& allocation) { return allocation.buffer != nullptr && !allocation.guest && allocation.size > MaxRefreshBytes; }));
    addressSpaceSerial = eligible ? guestMemory.AddressSpaceSerial() : 0;
    addressReusable = addressSpaceSerial != 0;
    if (addressReusable) return;
    addressWrites = {};
    addressBuffers = {};
    addressImages = {};
    addressFaults = {};
    addressInfos = {};
}

ShaderResources::ShaderResources(AddressUseTag, const std::shared_ptr<ShaderResources>& source) : context(source->context), guestMemory(source->context) {
    layoutKey = source->layoutKey;
    usesBda = source->usesBda;
    usesFaultBuffer = source->usesFaultBuffer;
    _layout = source->_layout;
    readOnlyBuffers = source->readOnlyBuffers;
    textures = source->textures;
    storageTextures = source->storageTextures;
    storageWritten = source->storageWritten;
    pushPatches = source->pushPatches;
    describedRanges = source->describedRanges;
    textureFirstLayer = source->textureFirstLayer;
    textureBaseLevel = source->textureBaseLevel;
    addressSource = source;
    completed = true;
}

bool ShaderResources::writeAddressSet(std::span<const MovedBuffer> moved, std::span<const std::optional<GuestBufferMemory::SpaceView>> planned, std::size_t* movedGuestOut) {
    const auto& source = *addressSource;
    Require(bda != nullptr && !source.addressWrites.empty(), "an address template use has no descriptor plan");
    auto buffers = source.addressBuffers;
    std::size_t movedGuest = 0;
    for (const auto& entry : moved) {
        if (!entry.words.empty()) continue;
        Require(entry.allocation < source.addressInfos.size() && entry.allocation < source.allocations.size() && source.addressInfos[entry.allocation] < buffers.size(), "a moved guest buffer of an address template has no descriptor");
        const auto& item = source.allocations[entry.allocation];
        std::uint32_t adjustment = 0;
        std::optional<VkDescriptorBufferInfo> info;
        if (planned.empty()) info = guestMemory.SpaceDescriptor(entry.address, entry.size, adjustment);
        else if (movedGuest < planned.size() && planned[movedGuest].has_value()) {
            info = planned[movedGuest]->info;
            adjustment = planned[movedGuest]->adjustment;
        }
        if (!info.has_value()) return false;
        if (adjustment != item.adjustment) {
            if (item.pushByte < 0) return false;
            const auto position = static_cast<std::uint32_t>(item.pushByte);
            std::erase_if(pushPatches, [position](const auto& patch) { return patch.first == position; });
            if (adjustment != 0) pushPatches.emplace_back(position, adjustment);
        }
        buffers[source.addressInfos[entry.allocation]] = *info;
        ++movedGuest;
    }
    std::vector<VkDescriptorPoolSize> sizes;
    for (std::size_t at = 0; at + 3 < layoutKey.size(); at += 4) {
        const auto type = static_cast<VkDescriptorType>(layoutKey[at + 1]);
        const auto found = std::find_if(sizes.begin(), sizes.end(), [type](const VkDescriptorPoolSize& size) { return size.type == type; });
        if (found != sizes.end()) found->descriptorCount += layoutKey[at + 2];
        else sizes.push_back({type, layoutKey[at + 2]});
    }
    static const bool noPoolCache = std::getenv("APS5_NO_POOL_CACHE") != nullptr;
    if (context.descriptorCache != nullptr && !noPoolCache) {
        const auto allocated = context.descriptorCache->AllocateRewritten(_layout, sizes);
        _set = allocated.set;
        cachePool = allocated.pool;
        recycledSet = true;
    } else {
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        poolInfo.pPoolSizes = sizes.data();
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &_layout;
        Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &_set), "vkAllocateDescriptorSets");
    }
    Require(_set != VK_NULL_HANDLE, "address template descriptor allocation failed");
    const auto fault = bda->Fault();
    for (const auto position : source.addressFaults) buffers.at(position) = fault;
    addressData.reserve(moved.size());
    for (const auto& entry : moved) {
        if (entry.words.empty()) continue;
        Require(entry.allocation < source.addressInfos.size() && source.addressInfos[entry.allocation] < buffers.size() && entry.words.size() * sizeof(std::uint32_t) == entry.size, "a moved data buffer of an address template has no descriptor");
        auto buffer = std::make_shared<Buffer>(context, entry.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto* bytes = buffer->Bytes().data();
        std::memcpy(bytes, entry.words.data(), entry.size);
        for (const auto& patch : source.dataPatches) {
            if (patch.allocation == entry.allocation && patch.byte < entry.size) bytes[patch.byte] = static_cast<std::byte>(patch.adjustment);
        }
        buffers[source.addressInfos[entry.allocation]] = {buffer->Handle(), 0, entry.size};
        addressData.push_back(std::move(buffer));
    }
    auto images = source.addressImages;
    for (const auto& [slot, info] : movedImageViews) {
        if (slot >= images.size()) return false;
        images[slot] = info;
    }
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(source.addressWrites.size());
    for (const auto& kept : source.addressWrites) {
        auto write = kept.write;
        write.dstSet = _set;
        if (kept.image) write.pImageInfo = images.data() + kept.first;
        else write.pBufferInfo = buffers.data() + kept.first;
        writes.push_back(write);
    }
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (movedGuestOut != nullptr) *movedGuestOut = movedGuest;
    else if (movedGuest != 0) AddressCounts().movedBuffers.fetch_add(movedGuest, std::memory_order_relaxed);
    return true;
}

namespace {

// Whether an address speculation records its moved images on the prep thread (live `spec_images`,
// env APS5_SPEC_IMAGE_RECORDS), so the committer's rebind is the fastTexture predicate.
bool SpeculatedImageRecords() {
    static const bool enabled = std::getenv("APS5_SPEC_IMAGE_RECORDS") != nullptr;
    return LiveFlag("spec_images", enabled);
}

// Rebinds served by the record (0) and records the predicate refused (1).
std::array<std::atomic<std::uint64_t>, 2>& SpeculatedImageCounts() {
    static std::array<std::atomic<std::uint64_t>, 2> counts{};
    return counts;
}

}

bool ShaderResources::rebindSpeculatedImages(std::span<const SpeculatedImage> images) {
    // APS5_PROFILE_LOOKUP: [rebind-split] the texture lookups apart from the descriptor writes.
    static const bool split = std::getenv("APS5_PROFILE_LOOKUP") != nullptr;
    static std::atomic<std::uint64_t> rebinds{0}, rebound{0}, lookupNs{0}, writeNs{0};
    static std::atomic<std::int64_t> reported{0};
    const auto started = split ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto looked = started;
    struct Report {
        bool enabled;
        const std::chrono::steady_clock::time_point& started;
        const std::chrono::steady_clock::time_point& looked;
        std::size_t images;
        ~Report() {
            if (!enabled) return;
            const auto now = std::chrono::steady_clock::now();
            lookupNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(looked - started).count()), std::memory_order_relaxed);
            writeNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - looked).count()), std::memory_order_relaxed);
            rebound.fetch_add(images, std::memory_order_relaxed);
            const auto count = rebinds.fetch_add(1, std::memory_order_relaxed) + 1;
            const auto stamp = now.time_since_epoch().count();
            auto last = reported.load(std::memory_order_relaxed);
            if (stamp - last <= 10'000'000'000ll || !reported.compare_exchange_strong(last, stamp)) return;
            rebinds.store(0);
            std::fprintf(stderr, "[rebind-split] %llu rebinds, %.2f images each (records: %llu fast, %llu refused), us per rebind: texture lookups %.2f descriptor writes %.2f\n", static_cast<unsigned long long>(count), static_cast<double>(rebound.exchange(0)) / static_cast<double>(count), static_cast<unsigned long long>(SpeculatedImageCounts()[0].exchange(0)), static_cast<unsigned long long>(SpeculatedImageCounts()[1].exchange(0)), static_cast<double>(lookupNs.exchange(0)) / 1000.0 / static_cast<double>(count), static_cast<double>(writeNs.exchange(0)) / 1000.0 / static_cast<double>(count));
        }
    } report{split, started, looked, images.size()};
    const auto& source = *addressSource;
    std::vector<VkDescriptorImageInfo> infos;
    infos.reserve(images.size());
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(images.size());
    for (const auto& image : images) {
        if (image.at >= textures.size() || image.at >= source.textureDepthCompare.size() || image.at >= source.textureUnnormalized.size()) return false;
        const auto words = std::span<const std::uint32_t>(image.words);
        const auto resource = image.recorded ? image.resource : DecodeTextureResource(words);
        std::shared_ptr<Texture> texture;
        if (image.recorded) {
            ImageRecord record;
            record.sampled = true;
            record.decoded = true;
            record.words = image.words;
            record.resource = image.resource;
            record.guestBytes = image.guestBytes;
            record.components = image.components;
            record.keys = image.keys;
            record.generation = image.generation;
            record.texture = image.texture;
            record.source = image.source;
            record.entryKeys = image.entryKeys;
            record.entryGeneration = image.entryGeneration;
            texture = fastTexture(record);
            SpeculatedImageCounts()[texture != nullptr ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
        }
        if (texture == nullptr) {
            try {
                texture = cachedTexture(source.context, words, resource, ViewComponents(resource), DescribeSurface(resource).guestBytes, source.textureDepthCompare[image.at]);
            } catch (const std::exception&) {
                return false;
            }
        }
        if (texture == nullptr) return false;
        bool baseLevel = false;
        if (source.textureUnnormalized[image.at]) {
            const auto range = texture->SampledViewRange(image.firstLayer);
            if (range.layers != 1u || EffectiveMinLod(resource) != 0.0f || (range.type != VK_IMAGE_VIEW_TYPE_1D && range.type != VK_IMAGE_VIEW_TYPE_2D)) return false;
            baseLevel = range.levels > 1u;
        }
        const auto kept = std::find_if(source.addressWrites.begin(), source.addressWrites.end(), [&](const AddressWrite& entry) { return entry.image && image.slot >= entry.first && image.slot < entry.first + entry.write.descriptorCount; });
        if (kept == source.addressWrites.end()) return false;
        const VkDescriptorImageInfo info{VK_NULL_HANDLE, baseLevel ? texture->BaseLevelView() : image.firstLayer ? texture->FirstLayerView() : texture->View(), texture->Layout()};
        movedImageViews.push_back({image.slot, info});
        textures[image.at] = std::move(texture);
        infos.push_back(info);
        auto write = kept->write;
        write.dstSet = _set;
        write.dstArrayElement += static_cast<std::uint32_t>(image.slot - kept->first);
        write.descriptorCount = 1;
        writes.push_back(write);
    }
    if (split) looked = std::chrono::steady_clock::now();
    for (std::size_t k = 0; k < writes.size(); ++k) writes[k].pImageInfo = &infos[k];
    if (!writes.empty()) context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    AddressCounts().movedBuffers.fetch_add(images.size(), std::memory_order_relaxed);
    return true;
}

std::shared_ptr<ShaderResources> ShaderResources::AddressUse(const std::shared_ptr<ShaderResources>& source, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, AddressMiss& miss) {
    ForkTrap("address template");
    Require(source != nullptr && source->addressReusable, "an address template use needs an address template");
    auto& counts = AddressCounts();
    const auto refuse = [&](AddressMiss reason) {
        miss = reason;
        counts.misses[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
        return std::shared_ptr<ShaderResources>{};
    };
    miss = AddressMiss::None;
    static const bool windowed = std::getenv("APS5_PROFILE_LOOKUP") != nullptr;
    const bool profile = BuildProfiled() || windowed;
    auto lap = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::array<std::uint64_t, 5> spent{};
    const auto mark = [&](std::size_t phase) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        spent[phase] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - lap).count());
        lap = now;
    };
    auto moved = source->movedBuffers(shaders, Recorder::Active(), false);
    if (!moved.has_value()) return refuse(AddressMiss::Moved);
    std::shared_ptr<ShaderResources> use(new ShaderResources(AddressUseTag{}, source));
    mark(0);
    switch (use->guestMemory.ShareSpace(source->addressSpaceSerial, source->guestMemory.Writes())) {
        case GuestBufferMemory::SpaceShare::Taken: break;
        case GuestBufferMemory::SpaceShare::Replaced: return refuse(AddressMiss::Replaced);
        case GuestBufferMemory::SpaceShare::Imports: return refuse(AddressMiss::Imports);
    }
    for (const auto& snapshot : snapshots) {
        if (!use->guestMemory.SpaceCovers(snapshot.address, snapshot.bytes.size())) return refuse(AddressMiss::Outside);
    }
    mark(1);
    if (!source->Revalidate(shaders, nullptr, MovableImages())) return refuse(AddressMiss::Proof);
    mark(2);
    if (!use->guestMemory.FinishSharedSpace()) return refuse(AddressMiss::Imports);
    if (MovableImages()) {
        std::size_t index = 0;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) return refuse(AddressMiss::Moved);
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.kind != ShaderRecompiler::DescriptorKind::SampledImage || binding.count == 0) continue;
                if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 8u || !binding.imageShape.has_value()) return refuse(AddressMiss::Moved);
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                    if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                    if (index >= source->textureWords.size()) return refuse(AddressMiss::Moved);
                    const auto at = index++;
                    if (std::equal(source->textureWords[at].begin(), source->textureWords[at].end(), words.begin())) continue;
                    static std::atomic<int> refusals{0};
                    const auto note = [&](const char* why) { if (refusals.fetch_add(1, std::memory_order_relaxed) < 40) std::fprintf(stderr, "[rescache] moved image refused: %s (texture %zu of %zu, slots %zu)\n", why, at, source->textureWords.size(), source->addressTextureSlots.size()); };
                    if (at >= source->addressTextureSlots.size() || source->addressTextureSlots[at] == std::numeric_limits<std::size_t>::max()) { note("no slot"); return refuse(AddressMiss::Moved); }
                    const auto resource = DecodeTextureResource(words);
                    const bool firstLayer = *binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
                    if (firstLayer != source->textureFirstLayer[at] || (!firstLayer && !MatchesGuestDimension(*binding.imageShape, resource.dimension))) return refuse(AddressMiss::Moved);
                    std::shared_ptr<Texture> texture;
                    try {
                        texture = cachedTexture(source->context, words, resource, ViewComponents(resource), DescribeSurface(resource).guestBytes, source->textureDepthCompare[at]);
                    } catch (const std::exception&) {
                        return refuse(AddressMiss::Moved);
                    }
                    if (texture == nullptr) return refuse(AddressMiss::Moved);
                    bool baseLevel = false;
                    if (source->textureUnnormalized[at]) {
                        const auto range = texture->SampledViewRange(firstLayer);
                        if (range.layers != 1u || EffectiveMinLod(resource) != 0.0f || (range.type != VK_IMAGE_VIEW_TYPE_1D && range.type != VK_IMAGE_VIEW_TYPE_2D)) {
                            note("unnormalized view");
                            return refuse(AddressMiss::Moved);
                        }
                        baseLevel = range.levels > 1u;
                    }
                    use->movedImageViews.push_back({source->addressTextureSlots[at], {VK_NULL_HANDLE, baseLevel ? texture->BaseLevelView() : firstLayer ? texture->FirstLayerView() : texture->View(), texture->Layout()}});
                    use->textures[at] = std::move(texture);
                }
            }
        }
        if (index != source->textureWords.size()) return refuse(AddressMiss::Moved);
        if (!use->movedImageViews.empty()) counts.movedBuffers.fetch_add(use->movedImageViews.size(), std::memory_order_relaxed);
    }
    mark(3);
    use->bda = source->bda->ShareTable(source->context);
    use->bda->SetMayWrite(std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program == nullptr || shader.program->bdaWrites; }));
    if (!use->writeAddressSet(*moved)) return refuse(AddressMiss::MovedOutside);
    mark(4);
    counts.hits.fetch_add(1, std::memory_order_relaxed);
    if (profile) {
        counts.timedHits.fetch_add(1, std::memory_order_relaxed);
        for (std::size_t phase = 0; phase < spent.size(); ++phase) counts.nanoseconds[phase].fetch_add(spent[phase], std::memory_order_relaxed);
    }
    if (windowed) {
        static std::array<std::atomic<std::uint64_t>, 5> window{};
        static std::atomic<std::uint64_t> windowHits{0};
        static std::atomic<std::int64_t> windowReport{0};
        for (std::size_t phase = 0; phase < spent.size(); ++phase) window[phase].fetch_add(spent[phase], std::memory_order_relaxed);
        const auto hits = windowHits.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = windowReport.load(std::memory_order_relaxed);
        if (now - last > 10'000'000'000ll && windowReport.compare_exchange_strong(last, now)) {
            const auto us = [&](std::size_t phase) { return static_cast<double>(window[phase].exchange(0)) / 1000.0 / static_cast<double>(hits); };
            windowHits.store(0);
            std::fprintf(stderr, "[use-split] %llu address uses, us per use: moved check %.1f space %.1f proof %.1f finish %.1f descriptor set %.1f%c", static_cast<unsigned long long>(hits), us(0), us(1), us(2), us(3), us(4), 10);
        }
    }
    return use;
}

std::shared_ptr<ShaderResources> ShaderResources::DetachAddressUse(const std::shared_ptr<ShaderResources>& source) {
    Require(source != nullptr && source->addressReusable && source->guestMemory.SpaceOnly(), "only an address template holding its own use's address space detaches it");
    std::shared_ptr<ShaderResources> use(new ShaderResources(AddressUseTag{}, source));
    use->bda = source->bda->ShareTable(source->context);
    Require(use->writeAddressSet({}), "an address template could not describe its own use");
    use->guestMemory.TakeSpace(source->guestMemory);
    AddressCounts().detached.fetch_add(1, std::memory_order_relaxed);
    return use;
}

// APS5_PROFILE_DRAW: the per-device descriptor caches' counters, every 10 s.
void ShaderResources::reportDescriptorCaches() const {
    static std::mutex reportMutex;
    static auto lastReport = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(reportMutex);
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    const auto descriptors = context.descriptorCache != nullptr ? context.descriptorCache->Counters() : DescriptorCache::Stats{};
    const auto samplerHits = context.samplerCache != nullptr ? context.samplerCache->Hits() : 0;
    const auto samplerMisses = context.samplerCache != nullptr ? context.samplerCache->Misses() : 0;
    AgcDriver::ProfilePrint_nid_no_patch("[descriptors] layouts %llu hits / %llu created, sets %llu from %llu pools and %llu dedicated pools, samplers %llu hits / %llu created\n", static_cast<unsigned long long>(descriptors.layoutHits), static_cast<unsigned long long>(descriptors.layoutMisses), static_cast<unsigned long long>(descriptors.sets), static_cast<unsigned long long>(descriptors.pools), static_cast<unsigned long long>(descriptors.dedicatedPools), static_cast<unsigned long long>(samplerHits), static_cast<unsigned long long>(samplerMisses));
}

template<typename TVisitor>
void VisitContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers, const TVisitor& visit, bool movableImages = false) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    visit(dataWords ? 1u : 0u);
    visit(static_cast<std::uint32_t>(shader.stage));
    visit(static_cast<std::uint32_t>(program.PipelineVariantId()));
    visit(static_cast<std::uint32_t>(program.PipelineVariantId() >> 32u));
    visit(static_cast<std::uint32_t>(program.bindings.size()));
    const auto packBits = [&](const std::vector<bool>& bits) {
        visit(static_cast<std::uint32_t>(bits.size()));
        std::uint32_t word = 0;
        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (bits[i]) word |= 1u << (i % 32u);
            if (i % 32u == 31u || i + 1 == bits.size()) {
                visit(word);
                word = 0;
            }
        }
    };
    for (const auto& binding : program.bindings) {
        for (const auto word : {static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.descriptorSet, binding.binding, binding.count, binding.readOnly ? 1u : 0u, binding.imageShape.has_value() ? static_cast<std::uint32_t>(*binding.imageShape) + 1u : 0u, static_cast<std::uint32_t>(binding.guestDescriptor.size())}) visit(word);
        if (movableBuffers && binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers && binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 4u) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                const bool empty = words[2] == 0 || (words[0] == 0 && (words[1] & 0xffffu) == 0);
                if (words[2] == 0 && (words[0] != 0 || (words[1] & 0xffffu) != 0)) { for (const auto word : {1u, words[1] & 0xffff0000u, 0u, words[3]}) visit(word); }
                else if (written || empty) { for (std::uint32_t index = 0; index < 4u; ++index) visit(words[index]); }
                else { for (const auto word : {0u, words[1] & 0xffff0000u, 0u, words[3]}) visit(word); }
            }
        } else if (movableImages && binding.role == ShaderRecompiler::DescriptorRole::GuestImages && binding.kind == ShaderRecompiler::DescriptorKind::SampledImage && binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 8u) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 8u;
                if (words[0] == 0u && (words[1] & 0xffu) == 0u) { for (std::uint32_t index = 0; index < 8u; ++index) visit(words[index]); }
                else { for (const auto word : {1u, words[1] & ~0xffu, 0u, words[3], words[4], words[5], words[6], words[7]}) visit(word); }
            }
        } else if ((dataWords && !movableBuffers) || !DataRole(binding.role)) {
            for (const auto word : binding.guestDescriptor) visit(word);
        }
        packBits(binding.imageWritten);
        packBits(binding.samplerDepthCompare);
        packBits(binding.imageDepthCompare);
        packBits(binding.imageAtomic);
        packBits(binding.samplerUnnormalized);
        packBits(binding.imageUnnormalized);
        // Read-only elements are bound without a write set: an object built for one written set
        // must not serve a build with another (the variant implies it, this makes it explicit).
        packBits(binding.bufferWritten);
        visit(static_cast<std::uint32_t>(binding.imageSamplers.size()));
        for (const auto sampler : binding.imageSamplers) {
            visit(static_cast<std::uint32_t>(sampler));
            visit(static_cast<std::uint32_t>(sampler >> 32u));
        }
    }
}

bool ShaderResources::MovableImages() {
    static const bool enabled = std::getenv("APS5_MOVABLE_IMAGES") != nullptr;
    return enabled;
}

bool ShaderResources::SameImageWords(std::span<const CompiledShader> shaders) const {
    std::size_t index = 0;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) return false;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.kind != ShaderRecompiler::DescriptorKind::SampledImage || binding.count == 0) continue;
            const auto elementWords = binding.guestDescriptor.size() / binding.count;
            if (elementWords != 8) return false;
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 8u;
                if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                if (index >= textureWords.size() || !std::equal(textureWords[index].begin(), textureWords[index].end(), words)) return false;
                ++index;
            }
        }
    }
    return index == textureWords.size();
}

std::vector<std::uint32_t> ShaderResources::ContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers, bool movableImages) {
    Require(shader.program != nullptr, "missing compiled shader");
    std::vector<std::uint32_t> key;
    key.reserve(8u + shader.program->bindings.size() * 12u);
    VisitContentKey(shader, dataWords, movableBuffers, [&](std::uint32_t word) { key.push_back(word); }, movableImages);
    return key;
}

bool MatchesContentKey(const CompiledShader& shader, std::span<const std::uint32_t> key) {
    std::size_t cursor = 0;
    bool equal = true;
    VisitContentKey(shader, false, true, [&](std::uint32_t word) {
        equal &= cursor < key.size() && key[cursor] == word;
        ++cursor;
    });
    return equal && cursor == key.size();
}

namespace {

// APS5_PROFILE_DRAW: Revalidate outcomes and time, printed as [rescache] every 10 s next to the
// device's hit/miss line: failed = the object could not be reused (whichever image path it took);
// of the reused ones, fast = every image proved current from stamps, full = the lookups were repeated.
// Why the fast path left an object to the full walk (the "fast-fail by reason" counts).
using FastFail = ShaderResources::FastFail;
constexpr std::array<const char*, static_cast<std::size_t>(FastFail::Count)> FastFailNames{"no record", "collect", "pending image", "evicted image", "memory changed", "keys", "cleared view", "storage keys"};
using OwnRefreshFallback = ShaderResources::OwnRefreshFallback;
constexpr std::array<const char*, static_cast<std::size_t>(OwnRefreshFallback::Count)> OwnRefreshFallbackNames{"disabled", "snapshot texture", "cleared view", "foreign view", "surface key", "not imported", "uncached", "re-run failed"};

struct RevalidateProfile {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> nanoseconds{0};
    std::atomic<std::uint64_t> fast{0};
    std::atomic<std::uint64_t> full{0};
    std::atomic<std::uint64_t> failed{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FastFail::Count)> fastFails{};
    // Epoch gate: registry scans skipped because the pending serial was unchanged, and import
    // serial loops skipped because the table's identity was unchanged.
    std::atomic<std::uint64_t> serialSkips{0};
    std::atomic<std::uint64_t> importSkips{0};
    // Elements with DCC keys the fast path accepted through the images' key proofs.
    std::atomic<std::uint64_t> keysProven{0};
    std::atomic<std::uint64_t> storageKeysProven{0};
    // The 'pending image' fast-fails by what overlapped: a surface with an own object (the query
    // excepts it, so the overlapping image is foreign to the surface) or a snapshot texture (no own
    // object); and the fails whose overlapping surfaces all have an own object (the T1 trim of
    // design_cpu_final applies: flush the foreign image, refresh the own object, no full walk).
    std::atomic<std::uint64_t> pendingForeign{0};
    std::atomic<std::uint64_t> pendingSnapshot{0};
    std::atomic<std::uint64_t> pendingT1Eligible{0};
    // The full walks by the reason of the fast proof that preceded them (fastFails also counts the
    // Pending failures the own-object refresh resolved), the Pending failures it resolved (with the
    // objects it refreshed: storage images, view sources) and those it left to the walk by reason;
    // view surfaces a foreign image overlapped that the proof accepted because FindPending names
    // their own source; surfaces a foreign image was still pending over after their own object's
    // refresh (its alias, a straddling image) that the re-run accepted; proofs the verify switch
    // checked against the full walk.
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FastFail::Count)> fullByReason{};
    std::atomic<std::uint64_t> ownRefreshed{0};
    std::atomic<std::uint64_t> ownStorageRefreshes{0};
    std::atomic<std::uint64_t> ownViewRefreshes{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(OwnRefreshFallback::Count)> ownFallbacks{};
    std::atomic<std::uint64_t> ownSourceViews{0};
    std::atomic<std::uint64_t> refreshedOverlaps{0};
    std::atomic<std::uint64_t> proofsVerified{0};
    std::atomic<std::int64_t> lastReport{0};
};

RevalidateProfile& Revalidations() {
    static RevalidateProfile profile;
    return profile;
}

void countFastFail(FastFail reason) {
    if (BuildProfiled()) Revalidations().fastFails[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countPendingFail(std::span<const StorageTexture::PendingQuery> pending) {
    if (!BuildProfiled()) return;
    auto& profile = Revalidations();
    bool eligible = true;
    for (const auto& query : pending) {
        if (!query.overlaps) continue;
        if (query.except != nullptr) {
            profile.pendingForeign.fetch_add(1, std::memory_order_relaxed);
        } else {
            profile.pendingSnapshot.fetch_add(1, std::memory_order_relaxed);
            eligible = false;
        }
    }
    if (eligible) profile.pendingT1Eligible.fetch_add(1, std::memory_order_relaxed);
}

void countKeysProven(bool storage) {
    if (BuildProfiled()) (storage ? Revalidations().storageKeysProven : Revalidations().keysProven).fetch_add(1, std::memory_order_relaxed);
}

void countFullWalk(FastFail reason) {
    if (BuildProfiled() && reason != FastFail::Count) Revalidations().fullByReason[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countOwnRefreshFallback(OwnRefreshFallback reason) {
    if (BuildProfiled()) Revalidations().ownFallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countOwnRefresh(bool storage) {
    TextureCounts().ownRefreshes.fetch_add(1, std::memory_order_relaxed);
    if (BuildProfiled()) (storage ? Revalidations().ownStorageRefreshes : Revalidations().ownViewRefreshes).fetch_add(1, std::memory_order_relaxed);
}

// APS5_NO_OWN_IMAGE_REFRESH=1: a Pending failure of the fast proof takes the full walk, as before
// the own-object refresh (T1).
bool OwnImageRefreshEnabled() {
    static const bool disabled = std::getenv("APS5_NO_OWN_IMAGE_REFRESH") != nullptr;
    return !disabled;
}

// APS5_VERIFY_PROOFS=1: after every own-object refresh the full walk runs beside the proof and the
// process aborts when it would return another object or upload again (a different decision).
bool VerifyProofs() {
    static const bool enabled = std::getenv("APS5_VERIFY_PROOFS") != nullptr;
    return enabled;
}

// Whether a sampled surface's keys are the very keys of the image the view follows (the same
// metadata, extent, format and alpha placement: the same scan), so the image's proof serves it.
bool SameKeySurface(const StorageTexture* source, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    if (source == nullptr) return false;
    const auto& own = source->Descriptor();
    return own.dccAddress == resource.dccAddress && source->GuestBytes() == guestBytes && own.format == resource.format && own.dccAlphaOnMsb == resource.dccAlphaOnMsb;
}


std::atomic<std::uint64_t> lookupMemoHits{0};
std::atomic<std::uint64_t> lookupMemoMisses{0};

void countLookupMemo(bool hit) {
    (hit ? lookupMemoHits : lookupMemoMisses).fetch_add(1, std::memory_order_relaxed);
}

void countRevalidate(bool fast, bool ok, std::chrono::steady_clock::time_point start) {
    auto& profile = Revalidations();
    const auto now = std::chrono::steady_clock::now();
    profile.calls.fetch_add(1, std::memory_order_relaxed);
    profile.nanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count()), std::memory_order_relaxed);
    (!ok ? profile.failed : fast ? profile.fast : profile.full).fetch_add(1, std::memory_order_relaxed);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto last = profile.lastReport.load();
    if (nowMs - last < 10000 || !profile.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto byReason = [](const auto& counts, const auto& names) {
        std::string reasons;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto count = counts[i].load(std::memory_order_relaxed);
            if (count == 0) continue;
            reasons += " " + std::string(names[i]) + " " + std::to_string(count);
        }
        return reasons;
    };
    const auto reasons = byReason(profile.fastFails, FastFailNames);
    const auto fullReasons = byReason(profile.fullByReason, FastFailNames);
    const auto fallbacks = byReason(profile.ownFallbacks, OwnRefreshFallbackNames);
    AgcDriver::ProfilePrint_nid_no_patch( "[rescache] revalidate %llu calls %.1f ms: fast %llu, full %llu, failed %llu; fast-fail by reason:%s; full walks: T1 refreshed %llu (%llu storage images, %llu view sources, %llu views served by their own pending source, %llu overlaps left by the refresh accepted), full by reason:%s, T1 fallback by reason:%s, proofs verified %llu; fast-fail pending image: own 0 (the query excepts the own object), foreign %llu, snapshot %llu, T1 eligible %llu; keys proven: %llu sampled, %llu storage; epoch gate: %llu registry scans skipped, %llu import loops skipped; lookup memo (APS5_LOOKUP_MEMO, cumulative): %llu hits, %llu misses\n", static_cast<unsigned long long>(profile.calls.load()), profile.nanoseconds.load() / 1e6, static_cast<unsigned long long>(profile.fast.load()), static_cast<unsigned long long>(profile.full.load()), static_cast<unsigned long long>(profile.failed.load()), reasons.c_str(), static_cast<unsigned long long>(profile.ownRefreshed.load()), static_cast<unsigned long long>(profile.ownStorageRefreshes.load()), static_cast<unsigned long long>(profile.ownViewRefreshes.load()), static_cast<unsigned long long>(profile.ownSourceViews.load()), static_cast<unsigned long long>(profile.refreshedOverlaps.load()), fullReasons.c_str(), fallbacks.c_str(), static_cast<unsigned long long>(profile.proofsVerified.load()), static_cast<unsigned long long>(profile.pendingForeign.load()), static_cast<unsigned long long>(profile.pendingSnapshot.load()), static_cast<unsigned long long>(profile.pendingT1Eligible.load()), static_cast<unsigned long long>(profile.keysProven.load()), static_cast<unsigned long long>(profile.storageKeysProven.load()), static_cast<unsigned long long>(profile.serialSkips.load()), static_cast<unsigned long long>(profile.importSkips.load()), static_cast<unsigned long long>(lookupMemoHits.load()), static_cast<unsigned long long>(lookupMemoMisses.load()));
}

// APS5_NO_EPOCH_REVALIDATE=1: the per-element registry, cache and stamp checks and the per-region
// flush and import lookups of every Revalidate, as before the epoch gate.
bool EpochRevalidate() {
    static const bool enabled = std::getenv("APS5_NO_EPOCH_REVALIDATE") == nullptr;
    return enabled;
}

bool LookupMemoEnabled() {
    // Live `lookup_memo` (env APS5_LOOKUP_MEMO the default): a memo is kept only while it is on,
    // and lookupMemoHolds checks its epoch and serials, so a toggle takes effect at the next build.
    static const bool requested = std::getenv("APS5_LOOKUP_MEMO") != nullptr;
    return EpochRevalidate() && LiveFlag("lookup_memo", requested);
}

}

// Turns the lookups' records into this object's per-texture validation records (see
// ValidatedSurface); the last record of an object is the freshest.
void ShaderResources::captureValidation() {
    const auto record = [](const void* object) -> const LookupRecord* {
        const auto& log = lookupLog();
        for (auto it = log.rbegin(); it != log.rend(); ++it) {
            if (it->object == object) return &*it;
        }
        return nullptr;
    };
    ++proofVersion;
    validatedTextures.assign(textures.size(), {});
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (const auto* found = record(textures[i].get())) validatedTextures[i] = {found->resource, found->bytes, found->keys, found->generation, 0, found->source, true, found->depth, found->words, found->wordCount, found->components};
    }
    lookupLog().clear();
}

// Proves every texture and storage image still current without repeating its lookup: exactly the
// "unchanged" branches of cachedTexture and StorageTexture::Refresh, evaluated from write stamps,
// the pending-results registry and the DCC keys. Per element: collect first (stamps come only from
// collects, and the generation a record moves to must predate the checks, so a write landing between
// them is stamped newer: the rule of Driver's dispatch cache), then no other image may have results
// pending over the memory (a lookup would flush them into it), a storage image must still be the
// cache's image of its surface, the memory must be unchanged since the object's content matched it,
// and the DCC keys must be what the content was made under whenever the surface has any (a fast
// clear touches only the keys). Anything else, including a failed collect (memory outside the arena,
// uncommitted pages), is left to the full walk.
// The epoch gate: while the pending registry's serial is what this object's last proof saw
// (pendingSerialSeen), the registry is what it was then (every change bumps the serial under the
// registry mutex), so the pending-image checks are skipped; otherwise one scan answers them for
// every surface at once. The stamp checks go through one tracker lock, the cache checks through
// the images' cached flags and one touch of the cache.
struct FastSplit {
    std::atomic<std::uint64_t> calls{0}, collectNs{0}, scanNs{0}, unchangedNs{0}, cachedNs{0}, totalNs{0}, textures{0};
    std::atomic<std::int64_t> reported{0};
};

FastSplit& FastSplitStats() {
    static FastSplit stats;
    return stats;
}

struct FastTimer {
    bool on;
    std::chrono::steady_clock::time_point at;
    explicit FastTimer(bool on) : on(on), at(on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    std::uint64_t ns() const { return on ? static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - at).count()) : 0; }
};

bool ShaderResources::fastRevalidate(std::uint64_t serialBefore, std::span<const PendingOverlap> refreshed, FastFail& reason, std::vector<PendingOverlap>& overlapping, bool& accepted) {
    static const bool fastSplit = std::getenv("APS5_PROFILE_LOOKUP") != nullptr;
    auto& fastStats = FastSplitStats();
    const FastTimer fastTotal(fastSplit);
    struct FastReport {
        bool on;
        const FastTimer& total;
        FastSplit& stats;
        std::size_t textures;
        ~FastReport() {
            if (!on) return;
            stats.totalNs.fetch_add(total.ns(), std::memory_order_relaxed);
            stats.textures.fetch_add(textures, std::memory_order_relaxed);
            const auto count = stats.calls.fetch_add(1, std::memory_order_relaxed) + 1;
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            auto last = stats.reported.load(std::memory_order_relaxed);
            if (now - last > 10'000'000'000ll && stats.reported.compare_exchange_strong(last, now)) {
                const auto us = [&](std::atomic<std::uint64_t>& value) { return static_cast<double>(value.exchange(0)) / 1000.0 / static_cast<double>(count); };
                const auto textureCount = stats.textures.exchange(0);
                stats.calls.store(0);
                std::fprintf(stderr, "[fast-split] %llu fast proofs (%.1f textures each), us per proof: total %.1f collect %.1f scan %.1f unchanged %.1f cached %.1f%c", static_cast<unsigned long long>(count), static_cast<double>(textureCount) / static_cast<double>(count), us(stats.totalNs), us(stats.collectNs), us(stats.scanNs), us(stats.unchangedNs), us(stats.cachedNs), 10);
            }
        }
    } fastReport{fastSplit, fastTotal, fastStats, textures.size() + storageTextures.size()};
    reason = FastFail::Count;
    overlapping.clear();
    accepted = false;
    if (!EpochRevalidate()) return fastRevalidateEach();
    const auto fail = [&reason](FastFail why) {
        reason = why;
        countFastFail(why);
        return false;
    };
    static std::array<std::atomic<std::uint64_t>, 4> noRecordKinds{};
    static std::atomic<std::int64_t> noRecordReport{0};
    const auto noteNoRecord = [&](std::size_t kind) {
        noRecordKinds[kind].fetch_add(1, std::memory_order_relaxed);
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = noRecordReport.load(std::memory_order_relaxed);
        if (now - last > 10'000'000'000ll && noRecordReport.compare_exchange_strong(last, now)) std::fprintf(stderr, "[rescache] no record (10 s): record count differs %llu, texture without a record %llu (of them refreshed per use %llu), storage slot empty %llu\n", static_cast<unsigned long long>(noRecordKinds[0].exchange(0)), static_cast<unsigned long long>(noRecordKinds[1].exchange(0)), static_cast<unsigned long long>(noRecordKinds[3].exchange(0)), static_cast<unsigned long long>(noRecordKinds[2].exchange(0)));
    };
    if (validatedTextures.size() != textures.size()) {
        noteNoRecord(0);
        return fail(FastFail::NoRecord);
    }
    bool depthSeen = false;
    const bool unchanged = pendingSerialSeen != 0 && pendingSerialSeen == serialBefore;
    const bool keyProofs = KeyFastPath();
    thread_local std::vector<GuestMemory::UnchangedQuery>* queriesSlot = nullptr;
    auto& queries = ShaderRecompiler::ThreadOwned(queriesSlot);
    thread_local std::vector<StorageTexture::PendingQuery>* pendingSlot = nullptr;
    auto& pending = ShaderRecompiler::ThreadOwned(pendingSlot);
    // The element each pending query stands for, in query order.
    thread_local std::vector<PendingOverlap>* ownersSlot = nullptr;
    auto& owners = ShaderRecompiler::ThreadOwned(ownersSlot);
    thread_local std::vector<const StorageTexture*>* imagesSlot = nullptr;
    auto& images = ShaderRecompiler::ThreadOwned(imagesSlot);
    thread_local std::vector<DccKeys>* scannedKeysSlot = nullptr;
    auto& scannedKeys = ShaderRecompiler::ThreadOwned(scannedKeysSlot);
    queries.clear();
    pending.clear();
    owners.clear();
    images.clear();
    scannedKeys.assign(textures.size(), DccKeys::Uncompressed);
    const bool memo = LookupMemoEnabled();
    lookupMemo.epoch = 0;
    lookupMemo.stamps.clear();
    lookupMemo.keysKept = keyProofs;
    const auto noteKeys = [&](const GuestTextureResource& resource, std::uint64_t surfaceBytes, const DccKeyProof& proof) {
        if (!memo || resource.dccAddress == 0) return;
        const auto keyBytes = static_cast<std::size_t>(surfaceBytes / 256);
        if (keyBytes == 0) return;
        if (proof.generation == 0) lookupMemo.keysKept = false;
        lookupMemo.stamps.push_back({resource.dccAddress, keyBytes, proof.generation});
    };
    const auto query = [&](std::uint64_t begin, std::uint64_t end, const StorageTexture* except, const StorageTexture* identity, PendingOverlap owner) {
        pending.push_back({begin, end, except, identity, false});
        owners.push_back(owner);
    };
    bool skippedAny = false;
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (fastSkip != nullptr && i < fastSkip->size() && (*fastSkip)[i]) {
            skippedAny = true;
            continue;
        }
        auto& surface = validatedTextures[i];
        if (!surface.valid) {
            noteNoRecord(1);
            if (textures[i] != nullptr && textures[i]->RefreshedPerUse()) noteNoRecord(3);
            return fail(FastFail::NoRecord);
        }
        if (surface.depth) {
            ForkTrap("depth texture proof");
            static const bool depthFastPath = std::getenv("APS5_NO_DEPTH_FAST_PROOF") == nullptr;
            if (!depthFastPath || DepthSurfaceTexture(context, std::span<const std::uint32_t>(surface.words.data(), surface.wordCount), surface.resource, surface.components) != textures[i]) return fail(FastFail::Changed);
            depthSeen = true;
            continue;
        }
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        {
            const FastTimer timer(fastSplit);
            surface.collected = GuestMemory::CollectWrites(address, bytes);
            if (fastSplit) fastStats.collectNs.fetch_add(timer.ns(), std::memory_order_relaxed);
        }
        if (surface.collected == 0) return fail(FastFail::Collect);
        const auto* source = surface.source;
        if (source != nullptr && DepthSurfaceAt(address)) return fail(FastFail::Changed);
        if (!keyProofs) {
            // Without proofs the keys are compared with the record's (a cleared view's identity
            // below is the record's too): the scan repeats on every call, as before.
            if (surface.resource.dccAddress != 0 && TextureClearKeys(surface.resource, surface.bytes) != surface.keys) return fail(FastFail::Keys);
            scannedKeys[i] = surface.keys;
            if (!unchanged) query(address, address + bytes, source, source != nullptr && surface.keys != DccKeys::Uncompressed ? source : nullptr, {i, false, source != nullptr && surface.keys == DccKeys::Uncompressed});
        } else if (source != nullptr) {
            // The view follows its image, so it stays valid while the image would Refresh as
            // unchanged: the image's own keys must be what its content was uploaded under (its
            // proof answers without a scan while they are unstamped), and its memory unchanged
            // since its generation (the query below). The surface's own keys come from the same
            // proof when they are the image's, else from the texture's.
            const auto& own = source->Descriptor();
            const auto sourceKeys = own.dccAddress != 0 ? ProvedClearKeys(own, source->GuestBytes(), source->KeyProof()) : DccKeys::Uncompressed;
            noteKeys(own, source->GuestBytes(), source->KeyProof());
            if (sourceKeys != source->UploadedKeys()) return fail(FastFail::Keys);
            const bool sameKeys = SameKeySurface(source, surface.resource, surface.bytes);
            const auto keys = surface.resource.dccAddress == 0 ? DccKeys::Uncompressed : sameKeys ? sourceKeys : ProvedClearKeys(surface.resource, surface.bytes, textures[i]->KeyProof());
            if (!sameKeys) noteKeys(surface.resource, surface.bytes, textures[i]->KeyProof());
            if (surface.resource.dccAddress != 0 && surface.resource.dccAddress != own.dccAddress && (keys != DccKeys::Uncompressed || !StorageImageServesKeys(*source, surface.resource.dccAddress))) return fail(FastFail::Keys);
            scannedKeys[i] = keys;
            // A view of a fast-cleared surface stays one only while its image still has results
            // pending over it (cachedTexture's hit rule), unless the image's own descriptor names
            // the surface's metadata (the lookup views the cleared image then). The identity is
            // asked whatever the registry's serial did: the keys may have moved since the last
            // proof while the registry did not.
            if (keys != DccKeys::Uncompressed && !(own.dccAddress == surface.resource.dccAddress && ClearedViewEnabled() && StorageClearAvailable(context, surface.resource.format, keys))) query(address, address + bytes, source, source, {i, false, false});
            else if (!unchanged) query(address, address + bytes, source, nullptr, {i, false, keys == DccKeys::Uncompressed});
            if (surface.resource.dccAddress != 0 || own.dccAddress != 0) countKeysProven(false);
        } else {
            // A snapshot holds its clear texels or the guest bytes: the keys must be what it was made under.
            const auto keys = surface.resource.dccAddress != 0 ? ProvedClearKeys(surface.resource, surface.bytes, textures[i]->KeyProof()) : DccKeys::Uncompressed;
            noteKeys(surface.resource, surface.bytes, textures[i]->KeyProof());
            if (keys != surface.keys) return fail(FastFail::Keys);
            scannedKeys[i] = keys;
            if (!unchanged) query(address, address + bytes, nullptr, nullptr, {i, false, false});
            if (surface.resource.dccAddress != 0) countKeysProven(false);
        }
        if (source != nullptr) {
            if (!source->Cached()) return fail(FastFail::Evicted);
            queries.push_back({address, bytes, source->Generation()});
            images.push_back(source);
        } else {
            queries.push_back({address, bytes, surface.generation});
        }
    }
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        if (storageTextures[i] != nullptr && (i >= storageKeys.size() || !StorageImageServesKeys(*storageTextures[i], storageKeys[i]))) return fail(FastFail::StorageKeys);
        if (i != 0 && storageTextures[i] == storageTextures[i - 1]) continue;
        const auto* image = storageTextures[i].get();
        if (image == nullptr) {
            noteNoRecord(2);
            return fail(FastFail::NoRecord);
        }
        const auto& own = image->Descriptor();
        const auto address = own.baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        {
            const FastTimer timer(fastSplit);
            const auto collected = GuestMemory::CollectWrites(address, bytes);
            if (fastSplit) fastStats.collectNs.fetch_add(timer.ns(), std::memory_order_relaxed);
            if (collected == 0) return fail(FastFail::Collect);
        }
        if (own.dccAddress != 0) {
            // Refresh's unchanged branch (its key compare against the keys the content was
            // uploaded under, through the image's proof; the memory query below), minus its
            // byte-compare fallback; the image's generation is left where it is.
            if (!keyProofs || ProvedClearKeys(own, bytes, image->KeyProof()) != image->UploadedKeys()) return fail(FastFail::StorageKeys);
            noteKeys(own, bytes, image->KeyProof());
            countKeysProven(true);
        }
        if (!unchanged) query(address, address + bytes, image, nullptr, {i, true, false});
        if (!image->Cached()) return fail(FastFail::Evicted);
        queries.push_back({address, bytes, image->Generation()});
        images.push_back(image);
    }
    if (!pending.empty()) {
        const FastTimer scanTimer(fastSplit);
        const bool scanned = StorageTexture::ScanPending(pending);
        if (fastSplit) fastStats.scanNs.fetch_add(scanTimer.ns(), std::memory_order_relaxed);
        if (!scanned) return fail(FastFail::ClearedView);
        // A foreign image over a view under uncompressed keys is no failure while FindPending
        // names the view's own source: the lookup views the source (cachedTexture's hit rule)
        // and touches nothing else. A surface whose own object T1 refreshed in this Revalidate
        // (`refreshed`) is no failure either: a storage element's lookup is that Refresh and
        // returns the image whatever stays registered over it; a view's lookup hits on the source
        // when FindPending names it, and views the cached image of its key (the source, proved by
        // refreshOwnObjects) when FindPending names nothing. Every other overlap is the full
        // walk's, or T1's (Revalidate).
        const auto refreshedOwner = [&](const PendingOverlap& owner) {
            return std::find_if(refreshed.begin(), refreshed.end(), [&](const PendingOverlap& entry) { return entry.element == owner.element && entry.storage == owner.storage; });
        };
        for (std::size_t k = 0; k < pending.size(); ++k) {
            if (!pending[k].overlaps) continue;
            if (OwnImageRefreshEnabled() && owners[k].viewUncompressed && pending[k].found == pending[k].except) {
                if (BuildProfiled()) Revalidations().ownSourceViews.fetch_add(1, std::memory_order_relaxed);
                accepted = true;
                continue;
            }
            if (const auto entry = refreshedOwner(owners[k]); entry != refreshed.end() && (entry->storage || pending[k].found == pending[k].except || (pending[k].found == nullptr && entry->sourceEligible))) {
                if (BuildProfiled()) Revalidations().refreshedOverlaps.fetch_add(1, std::memory_order_relaxed);
                accepted = true;
                continue;
            }
            overlapping.push_back(owners[k]);
        }
        if (!overlapping.empty()) {
            countPendingFail(pending);
            return fail(FastFail::Pending);
        }
    } else if (unchanged && BuildProfiled()) {
        Revalidations().serialSkips.fetch_add(1, std::memory_order_relaxed);
    }
    {
        const FastTimer timer(fastSplit);
        const bool same = GuestMemory::UnchangedSinceAll(queries);
        if (fastSplit) fastStats.unchangedNs.fetch_add(timer.ns(), std::memory_order_relaxed);
        if (!same) return fail(FastFail::Changed);
    }
    {
        const FastTimer timer(fastSplit);
        const bool cached = StorageImagesCached(context, images);
        if (fastSplit) fastStats.cachedNs.fetch_add(timer.ns(), std::memory_order_relaxed);
        if (!cached) return fail(FastFail::Evicted);
    }
    for (const auto* image : images) image->NoteProved();
    for (std::size_t i = 0; i < validatedTextures.size(); ++i) {
        if (fastSkip != nullptr && i < fastSkip->size() && (*fastSkip)[i]) continue;
        auto& surface = validatedTextures[i];
        if (surface.source == nullptr) surface.generation = surface.collected;
        surface.keys = scannedKeys[i];
    }
    if (memo && !depthSeen && !skippedAny && lookupMemo.keysKept && !accepted && refreshed.empty() && GuestMemory::ThreadCollectEpoch() != 0) {
        for (const auto& entry : queries) lookupMemo.stamps.push_back({entry.address, entry.bytes, entry.generation});
        for (std::size_t i = 0, k = 0; i < validatedTextures.size(); ++i, ++k) {
            const auto& surface = validatedTextures[i];
            if (surface.source == nullptr) lookupMemo.stamps[lookupMemo.stamps.size() - queries.size() + k].generation = surface.generation;
        }
        lookupMemo.images = images;
        lookupMemo.epoch = GuestMemory::ThreadCollectEpoch();
        lookupMemo.unwatched = GuestMemory::UnwatchSerial();
        lookupMemo.pendingSerial = serialBefore;
        lookupMemo.registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    }
    return true;
}

LookupMemoCounts LookupMemoCounters() {
    return {lookupMemoHits.load(std::memory_order_relaxed), lookupMemoMisses.load(std::memory_order_relaxed)};
}

bool ShaderResources::lookupMemoHolds(std::uint64_t serialBefore) {
    if (!LookupMemoEnabled() || lookupMemo.epoch == 0) return false;
    const auto epoch = GuestMemory::ThreadCollectEpoch();
    if (epoch == 0 || epoch != lookupMemo.epoch || GuestMemory::UnwatchSerial() != lookupMemo.unwatched || serialBefore != lookupMemo.pendingSerial || GuestAllocations::GuestAllocationsGeneration_nid_postfix() != lookupMemo.registryGeneration) return false;
    if (validatedTextures.size() != textures.size()) return false;
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        if (storageTextures[i] == nullptr || i >= storageKeys.size() || !StorageImageServesKeys(*storageTextures[i], storageKeys[i])) return false;
    }
    thread_local std::vector<GuestMemory::UnchangedQuery>* stampsSlot = nullptr;
    auto& stamps = ShaderRecompiler::ThreadOwned(stampsSlot);
    stamps.clear();
    for (const auto& entry : lookupMemo.stamps) stamps.push_back({entry.address, entry.bytes, entry.generation});
    if (!GuestMemory::UnchangedSinceAll(stamps)) return false;
    for (const auto* image : lookupMemo.images) {
        if (!image->Cached()) return false;
    }
    if (!StorageImagesCached(context, lookupMemo.images)) return false;
    for (const auto* image : lookupMemo.images) image->NoteProved();
    return true;
}

bool ShaderResources::fastRevalidateEach() {
    if (validatedTextures.size() != textures.size()) return false;
    for (std::size_t i = 0; i < textures.size(); ++i) {
        auto& surface = validatedTextures[i];
        if (!surface.valid) return false;
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        surface.collected = GuestMemory::CollectWrites(address, bytes);
        if (surface.collected == 0) return false;
        if (PendingStorageOverlaps(address, bytes, surface.source)) return false;
        if (surface.source != nullptr && surface.resource.dccAddress != 0 && surface.resource.dccAddress != surface.source->Descriptor().dccAddress) return false;
        if (surface.source != nullptr) {
            // The view follows the image: it needs the image current with guest memory, as the
            // lookup's Refresh would make it.
            if (!StorageImageCached(context, surface.source) || !GuestMemory::UnchangedSince(address, bytes, surface.source->Generation())) return false;
        } else if (!GuestMemory::UnchangedSince(address, bytes, surface.generation)) {
            return false;
        }
        if (surface.resource.dccAddress != 0 && TextureClearKeys(surface.resource, surface.bytes) != surface.keys) return false;
        // A view made under fast-clear keys stays one only while its image still has results pending
        // over the surface (the lookup prefers them to the clear); once they are flushed the clear
        // the image cannot see wins and the lookup makes a snapshot (cachedTexture's hit rule).
        if (surface.source != nullptr && surface.keys != DccKeys::Uncompressed && StorageTexture::FindPending(address, surface.bytes).get() != surface.source) return false;
    }
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        if (storageTextures[i] != nullptr && (i >= storageKeys.size() || (storageKeys[i] != 0 && storageKeys[i] != storageTextures[i]->Descriptor().dccAddress))) return false;
        // Consecutive elements of one mip chain share the image (see addStorageImageBinding).
        if (i != 0 && storageTextures[i] == storageTextures[i - 1]) continue;
        const auto* image = storageTextures[i].get();
        if (image == nullptr) return false;
        // StorageTexture::Refresh checks the image's own surface, not the element's descriptor (one
        // image serves every descriptor of its memory, with or without a DCC address), and compares
        // its keys with the keys it was uploaded under, which only the image knows (they move to
        // Uncompressed on every write-back; a fast clear after it puts the key bytes back to what a
        // record saw while the image holds last frame's results), so a surface with keys is left to
        // the full walk. Without a DCC address both sides of that compare are Uncompressed whatever
        // happened, and the memory checks below are Refresh's.
        const auto& own = image->Descriptor();
        if (own.dccAddress != 0) return false;
        const auto address = own.baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (GuestMemory::CollectWrites(address, bytes) == 0) return false;
        if (PendingStorageOverlaps(address, bytes, image)) return false;
        if (!StorageImageCached(context, image)) return false;
        if (!GuestMemory::UnchangedSince(address, bytes, image->Generation())) return false;
    }
    // Every check passed: snapshot textures are current at the collects issued before them. Storage
    // images keep their own generation: nothing was stamped above it, so the blocks a later write
    // stamps are the same set either way.
    for (auto& surface : validatedTextures) {
        if (surface.source == nullptr) surface.generation = surface.collected;
    }
    return true;
}

ShaderResources::OwnRefreshFallback ShaderResources::refreshOwnObjects(std::span<const CompiledShader> shaders, std::span<PendingOverlap> overlapping) {
    ForkTrap("own object refresh");
    const auto listed = [&](std::size_t element, bool storage) {
        return std::find_if(overlapping.begin(), overlapping.end(), [&](const PendingOverlap& overlap) { return overlap.element == element && overlap.storage == storage; });
    };
    // A sampled view's lookup (cachedTexture): the pending image FindPending serves the surface
    // by, if it fits the view, else the cached storage image of the surface's key, refreshed; the
    // view stays the lookup's answer only when that image is its own source. `sourceEligible` is
    // recorded whichever branch answers: a later refresh of this call may flush the source, and
    // the re-run then judges the surface by the other branch.
    const auto refreshView = [&](std::size_t i, PendingOverlap& overlap) {
        const auto& surface = validatedTextures[i];
        const auto& source = textures[i]->SharedStorageSource();
        if (source == nullptr || source.get() != surface.source) return OwnRefreshFallback::Snapshot;
        if (!source->Cached()) return OwnRefreshFallback::Uncached;
        if (!StorageImageServesKeys(*source, surface.resource.dccAddress)) return OwnRefreshFallback::Keys;
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        const bool imported = SampledFromStorageEligible(context, surface.resource, surface.bytes);
        overlap.sourceEligible = imported && SurfaceKey(context, source->Descriptor()) == SurfaceKey(context, surface.resource);
        auto found = StorageTexture::FindPending(address, surface.bytes);
        if (found != nullptr && !Texture::CanCopyFrom(*found, surface.resource)) found.reset();
        if (found != nullptr) {
            if (found != source) return OwnRefreshFallback::ForeignView;
            if (!GuestMemory::UnchangedSince(address, bytes, source->Generation())) {
                countOwnRefresh(false);
                source->Refresh();
            }
            return OwnRefreshFallback::Count;
        }
        if (!imported) return OwnRefreshFallback::NotImported;
        if (!overlap.sourceEligible) return OwnRefreshFallback::SurfaceKey;
        countOwnRefresh(false);
        // The cache's image of the key is this source while it is cached (an entry is replaced
        // only after its image left the cache), so the lookup would refresh and return it.
        source->Refresh();
        return OwnRefreshFallback::Count;
    };
    // A storage element's lookup (cachedStorageTexture): the cached image of its surface key,
    // refreshed; the key is the element's words', which mapped to this image at the build.
    const auto refreshStorage = [&](std::size_t i) {
        auto& image = storageTextures[i];
        if (image == nullptr || !image->Cached()) return OwnRefreshFallback::Uncached;
        if (i >= storageKeys.size() || !StorageImageServesKeys(*image, storageKeys[i])) return OwnRefreshFallback::Keys;
        countOwnRefresh(true);
        image->Refresh();
        return OwnRefreshFallback::Count;
    };
    std::size_t textureIndex = 0;
    std::size_t storageIndex = 0;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) return OwnRefreshFallback::Uncached;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
            if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                    if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                    if (textureIndex >= textures.size()) return OwnRefreshFallback::Uncached;
                    const auto index = textureIndex++;
                    const auto overlap = listed(index, false);
                    if (overlap == overlapping.end()) continue;
                    if (const auto fallback = refreshView(index, *overlap); fallback != OwnRefreshFallback::Count) return fallback;
                }
            } else if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                    if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                    if (storageIndex >= storageTextures.size()) return OwnRefreshFallback::Uncached;
                    const auto index = storageIndex++;
                    if (listed(index, true) == overlapping.end()) continue;
                    if (const auto fallback = refreshStorage(index); fallback != OwnRefreshFallback::Count) return fallback;
                }
            }
        }
    }
    return OwnRefreshFallback::Count;
}

struct ProofSplit {
    std::atomic<std::uint64_t> calls{0}, fastNs{0}, fullNs{0}, importNs{0}, stagingNs{0}, fulls{0};
    std::atomic<std::int64_t> reported{0};
};

ProofSplit& ProofSplitStats() {
    static ProofSplit stats;
    return stats;
}

struct ProofLap {
    bool on = false;
    std::chrono::steady_clock::time_point at{};
    void lap(std::atomic<std::uint64_t>& into) {
        if (!on) return;
        const auto now = std::chrono::steady_clock::now();
        into.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - at).count()), std::memory_order_relaxed);
        at = now;
    }
    ~ProofLap() {
        if (!on) return;
        auto& stats = ProofSplitStats();
        const auto count = stats.calls.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = stats.reported.load(std::memory_order_relaxed);
        if (now - last > 10'000'000'000ll && stats.reported.compare_exchange_strong(last, now)) {
            const auto us = [&](std::atomic<std::uint64_t>& value) { return static_cast<double>(value.exchange(0)) / 1000.0 / static_cast<double>(count); };
            const auto fulls = stats.fulls.exchange(0);
            stats.calls.store(0);
            std::fprintf(stderr, "[proof-split] %llu proofs (%llu full walks), us per proof: fast %.1f full %.1f imports %.1f staging %.1f\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(fulls), us(stats.fastNs), us(stats.fullNs), us(stats.importNs), us(stats.stagingNs));
        }
    }
};

bool ShaderResources::Revalidate(std::span<const CompiledShader> shaders, ProofReport* report, bool movedImagesRebound) {
    static const bool splitProfile = std::getenv("APS5_PROFILE_LOOKUP") != nullptr;
    ProofLap proofLap{splitProfile, splitProfile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
    if (report != nullptr) *report = {ProofPath::Full, ProofFailure::Other};
    if (!(reusable || addressReusable) || shaders.empty()) return false;
    const ForkGuard proofLock(proofMutex);
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    // APS5_NO_FAST_REVALIDATE=1 always repeats the lookups.
    static const bool noFast = std::getenv("APS5_NO_FAST_REVALIDATE") != nullptr;
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto finish = [&](bool fast, bool ok, ProofFailure failure = ProofFailure::Other) {
        if (profile) countRevalidate(fast, ok, start);
        if (report != nullptr) report->failure = ok ? ProofFailure::None : failure;
        return ok;
    };
    // Loaded before any check (the memo rule of fastRevalidate): kept as the memo only when the
    // registry did not move through the whole proof, flushes of this call included.
    const auto serialBefore = StorageTexture::PendingSerial();
    // (1) Textures and storage images, from stamps when every element allows it (fastRevalidate),
    // else by the same lookups as the build (which refresh or replace them as guest memory changed):
    // they must hand back the very objects the set's views belong to. The build appended them stage
    // by stage, binding by binding, so the walk repeats that order.
    const auto fullWalk = [&] {
        lookupLog().clear();
        std::size_t textureIndex = 0;
        std::size_t storageIndex = 0;
        bool skipped = false;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) return false;
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
                if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
                    const auto elementWords = binding.count != 0 ? binding.guestDescriptor.size() / binding.count : 0;
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
                        if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                        if (movedImagesRebound && elementWords == 8 && textureIndex < textureWords.size() && textureIndex < textures.size() && !std::equal(textureWords[textureIndex].begin(), textureWords[textureIndex].end(), words.begin())) {
                            skipped = true;
                            ++textureIndex;
                            continue;
                        }
                        const auto resource = DecodeTextureResource(words);
                        const VkComponentMapping components = ViewComponents(resource);
                        if (textureIndex >= textures.size() || cachedTexture(context, words, resource, components, 0, !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element)) != textures[textureIndex]) return false;
                        ++textureIndex;
                    }
                } else if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                        if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                        if (storageIndex >= storageTextures.size()) return false;
                        std::shared_ptr<StorageTexture> expected;
                        if (SameAsPreviousStorageElement(binding, element) && StorageDedupeEnabled()) expected = storageTextures[storageIndex - 1];
                        else expected = cachedStorageTexture(context, words, DecodeTextureResource(words), storageMips[storageIndex]);
                        if (expected != storageTextures[storageIndex]) return false;
                        ++storageIndex;
                    }
                }
            }
        }
        if (textureIndex != textures.size() || storageIndex != storageTextures.size()) return false;
        // The walk proved every object current again: the records move to what it proved, or one
        // spurious stamp (a label sharing a 64 KiB block with a surface's edge) would keep this
        // object on the full walk for good.
        if (!skipped) captureValidation();
        return true;
    };
    thread_local std::vector<bool>* movedSlot = nullptr;
    auto& moved = ShaderRecompiler::ThreadOwned(movedSlot);
    moved.assign(textures.size(), false);
    bool anyMoved = false;
    if (movedImagesRebound) {
        std::size_t index = 0;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) break;
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.kind != ShaderRecompiler::DescriptorKind::SampledImage || binding.count == 0) continue;
                const auto elementWords = binding.guestDescriptor.size() / binding.count;
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * elementWords;
                    if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                    if (elementWords == 8 && index < textureWords.size() && index < moved.size() && !std::equal(textureWords[index].begin(), textureWords[index].end(), words)) moved[index] = anyMoved = true;
                    ++index;
                }
            }
        }
    }
    struct SkipScope {
        ShaderResources& owner;
        ~SkipScope() { owner.fastSkip = nullptr; }
    } skipScope{*this};
    fastSkip = anyMoved ? &moved : nullptr;
    thread_local std::vector<PendingOverlap>* overlappingSlot = nullptr;
    auto& overlapping = ShaderRecompiler::ThreadOwned(overlappingSlot);
    thread_local std::vector<PendingOverlap>* refreshedSlot = nullptr;
    auto& refreshed = ShaderRecompiler::ThreadOwned(refreshedSlot);
    refreshed.clear();
    FastFail reason = FastFail::Count;
    bool accepted = false;
    const bool memoHit = !noFast && lookupMemoHolds(serialBefore);
    if (memoHit) countLookupMemo(true);
    else if (LookupMemoEnabled()) countLookupMemo(false);
    if (memoHit && VerifyProofs()) {
        thread_local std::vector<PendingOverlap>* checkSlot = nullptr;
        auto& check = ShaderRecompiler::ThreadOwned(checkSlot);
        FastFail checkReason = FastFail::Count;
        bool checkAccepted = false;
        if (!fastRevalidate(serialBefore, refreshed, checkReason, check, checkAccepted)) {
            std::fprintf(stderr, "[rescache] APS5_VERIFY_PROOFS: the lookup memo (APS5_LOOKUP_MEMO) proved a template its fast proof refuses (reason %d)\n", static_cast<int>(checkReason));
            std::fflush(stderr);
            std::abort();
        }
        if (BuildProfiled()) Revalidations().proofsVerified.fetch_add(1, std::memory_order_relaxed);
    }
    bool fast = memoHit || (!noFast && fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted));
    if (fast) GuestMemory::NoteProbeEvent(memoHit ? "memo lookup" : "memo fast revalidate");
    // T1 (design_cpu_final M3, rule RT1): a Pending failure whose overlapping images are foreign
    // to the surfaces is resolved by the own objects' refresh (what the walk's lookups would do to
    // them) and the fast proof run again, which is then authoritative (it accepts what stays
    // pending over the refreshed surfaces, see fastRevalidate); a fallback, or a second failure,
    // takes the full walk as before.
    bool ownRefreshed = false;
    if (!fast && !overlapping.empty()) {
        auto fallback = OwnRefreshFallback::Count;
        if (!OwnImageRefreshEnabled()) fallback = OwnRefreshFallback::Disabled;
        else {
            refreshed = overlapping;
            fallback = refreshOwnObjects(shaders, refreshed);
        }
        if (fallback == OwnRefreshFallback::Count) {
            fast = fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted);
            if (fast) ownRefreshed = true;
            else fallback = OwnRefreshFallback::Rerun;
        }
        if (fallback != OwnRefreshFallback::Count) countOwnRefreshFallback(fallback);
    }
    proofLap.lap(ProofSplitStats().fastNs);
    if (report != nullptr) report->path = !fast ? ProofPath::Full : ownRefreshed ? ProofPath::OwnRefreshed : ProofPath::Fast;
    if (!fast) {
        if (splitProfile) ProofSplitStats().fulls.fetch_add(1, std::memory_order_relaxed);
        countFullWalk(reason);
        if (!fullWalk()) {
            const auto failure = [&] {
                switch (reason) {
                    case FastFail::Pending: return ProofFailure::Pending;
                    case FastFail::Evicted: return ProofFailure::Evicted;
                    case FastFail::Changed: return ProofFailure::Changed;
                    case FastFail::Keys: case FastFail::ClearedView: case FastFail::StorageKeys: return ProofFailure::Keys;
                    default: return ProofFailure::Other;
                }
            }();
            return finish(false, false, failure);
        }
    } else if (ownRefreshed || accepted) {
        if (profile && ownRefreshed) Revalidations().ownRefreshed.fetch_add(1, std::memory_order_relaxed);
        if (VerifyProofs()) {
            // The walk after the refresh, or beside a proof that accepted a foreign overlap, must
            // find every object in place and current: another object, or an upload (a moved
            // content version), is a decision the proof got wrong.
            thread_local std::vector<std::pair<const StorageTexture*, std::uint64_t>>* versionsSlot = nullptr;
            auto& versions = ShaderRecompiler::ThreadOwned(versionsSlot);
            versions.clear();
            for (const auto& surface : validatedTextures) {
                if (surface.source != nullptr) versions.emplace_back(surface.source, surface.source->Version());
            }
            for (const auto& image : storageTextures) versions.emplace_back(image.get(), image->Version());
            const bool same = fullWalk();
            const auto moved = std::find_if(versions.begin(), versions.end(), [](const auto& entry) { return entry.first->Version() != entry.second; });
            if (!same || moved != versions.end()) {
                std::fprintf(stderr, "[rescache] APS5_VERIFY_PROOFS: the T1 proof (%s) disagrees with the full walk (%s)\n", ownRefreshed ? "own-object refresh" : "accepted overlap", !same ? "another object" : "an upload");
                std::fflush(stderr);
                std::abort();
            }
            if (profile) Revalidations().proofsVerified.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const auto failure = proofTail(serialBefore, &proofLap);
    return finish(fast, failure == ProofFailure::None, failure);
}

ShaderResources::ProofFailure ShaderResources::proofTail(std::uint64_t serialBefore, ProofLap* lap) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    // (2) The imported buffers the set reads in place: results of storage images pending in them go
    // to guest memory first (as an upload does), then each import must still be the one the set was
    // written against. This comes last because the lookups and flushes above can reconcile imports
    // themselves; nothing else touches them between here and the dispatch being recorded. Under the
    // epoch gate the flush runs only for regions one registry scan finds pending images over (none
    // while the serial is the memo's), and the serial loop only while the import table's identity
    // moved since the last proof (a retire bumps its epoch, a registry change its generation).
    if (lap != nullptr) lap->lap(ProofSplitStats().fullNs);
    // A new use of the object: elisions of an earlier use that never reached MarkGpuWrites go.
    guestMemory.BeginElisionUse();
    guestMemory.SetOtherWrites(writtenImageRanges());
    const auto serialLoop = [&] {
        for (const auto& region : directRegions) {
            auto serial = HostImportSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin), true);
            if (serial == 0) serial = ImageMirrorSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin));
            if (serial != region.serial) return false;
        }
        return true;
    };
    if (EpochRevalidate()) {
        if (!directRegions.empty() && !(pendingSerialSeen != 0 && pendingSerialSeen == StorageTexture::PendingSerial())) {
            thread_local std::vector<StorageTexture::PendingQuery>* regionsSlot = nullptr;
            auto& regions = ShaderRecompiler::ThreadOwned(regionsSlot);
            regions.clear();
            for (const auto& region : directRegions) regions.push_back({region.begin, region.end, nullptr, nullptr, false});
            StorageTexture::ScanPending(regions);
            // A region over a unit shadow's fresh results (a retile bumped the serial) is
            // published by the flush, as one over a pending image is stored.
            for (const auto& region : regions) {
                if (region.overlaps || AnyShadowedOverlaps(region.begin, static_cast<std::size_t>(region.end - region.begin))) guestMemory.FlushBoundRange(region.begin, region.end, "imported buffer region");
            }
        }
        if (HostImportsUnchanged(context, importsProof)) {
            if (profile) Revalidations().importSkips.fetch_add(1, std::memory_order_relaxed);
        } else {
            if (!serialLoop()) return ProofFailure::Imports;
            importsProof = HostImportsIdentity(context);
        }
    } else {
        for (const auto& region : directRegions) guestMemory.FlushBoundRange(region.begin, region.end, "imported buffer region");
        if (!serialLoop()) return ProofFailure::Imports;
    }
    // (3) Buffers staged in device memory (GuestBufferMemory::AllowDeviceStaging) are copied in
    // from their imports anew for this use, after the flushes above and before the work is
    // recorded; a failure to record leaves the object unusable for this dispatch, not the batch.
    if (lap != nullptr) lap->lap(ProofSplitStats().importNs);
    if (auto* recorder = Recorder::Active(); recorder != nullptr) {
        try {
            guestMemory.RecordStagingCopies(*recorder);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[resources] staging copies of a reused build failed: %s\n", error.what());
            return ProofFailure::Other;
        }
    }
    if (lap != nullptr) lap->lap(ProofSplitStats().stagingNs);
    // Units the write-back elision left pending over the regions (dropped only once the dispatch is
    // recorded) must not let the memos skip the next proof's flush, should this use never get there.
    const bool elided = guestMemory.HasElided();
    pendingSerialSeen = EpochRevalidate() && StorageTexture::PendingSerial() == serialBefore && !elided ? serialBefore : 0;
    if (StorageTexture::PendingSerial() != serialBefore || elided) lookupMemo.epoch = 0;
    return ProofFailure::None;
}

namespace {

// The prep threads' pending refusals by verdict ([Clear]: proofs that accepted own-source overlaps).
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShaderResources::PendingVerdict::Count)>& PendingVerdictCounts() {
    static std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShaderResources::PendingVerdict::Count)> counts{};
    return counts;
}

}

namespace {

// See TakeSpeculationRefusalDetail. `identity`: ScanPending found another image than the one
// speculated at a surface's base; otherwise the first overlap that is not an own-source view.
thread_local SpeculationRefusalDetail pendingRefusalDetail;

void notePendingRefusal(std::span<const StorageTexture::PendingQuery> scan, std::uint8_t kind, bool identity) {
    SpeculationRefusalDetail detail;
    detail.kind = kind;
    const StorageTexture::PendingQuery* chosen = nullptr;
    for (const auto& query : scan) {
        const bool hit = identity ? (query.pending != nullptr && query.found != query.pending) : query.overlaps;
        if (!hit) continue;
        if (chosen == nullptr) chosen = &query;
        if (identity || !(query.found != nullptr && query.found == query.except)) {
            chosen = &query;
            break;
        }
    }
    if (chosen != nullptr) {
        detail.surface = chosen->begin;
        detail.surfaceBytes = chosen->end - chosen->begin;
        if (const auto* blocker = identity ? chosen->found : chosen->blocker; blocker != nullptr) {
            const auto& d = blocker->Descriptor();
            detail.blocker = d.baseAddress;
            detail.blockerBytes = blocker->GuestBytes();
            detail.blockerFormat = d.format;
            detail.blockerWidth = d.width;
            detail.blockerHeight = d.height;
        }
    }
    pendingRefusalDetail = detail;
}

}

SpeculationRefusalDetail TakeSpeculationRefusalDetail() {
    return std::exchange(pendingRefusalDetail, SpeculationRefusalDetail{});
}

ShaderResources::PendingVerdict ShaderResources::SpeculatedPendingVerdict(std::span<const ProofSpeculation::Surface> surfaces, std::span<const StorageTexture::PendingQuery> scan, bool acceptOwnSource) {
    bool own = false;
    for (std::size_t k = 0; k < scan.size(); ++k) {
        if (!scan[k].overlaps) continue;
        const auto* surface = k < surfaces.size() ? &surfaces[k] : nullptr;
        // fastRevalidate's own-source rule: a sampled view (keys uncompressed: a speculated
        // surface has no DCC metadata) over which FindPending names the view's own image.
        if (surface != nullptr && !surface->storage && surface->image != nullptr && scan[k].except == surface->image && scan[k].found == surface->image) {
            own = true;
            continue;
        }
        if (surface == nullptr || surface->image == nullptr) return PendingVerdict::Snapshot;
        return surface->storage ? PendingVerdict::Storage : PendingVerdict::ForeignView;
    }
    return own && !acceptOwnSource ? PendingVerdict::OwnSource : PendingVerdict::Clear;
}

bool ShaderResources::SpeculateOwnPending() {
    static const bool requested = std::getenv("APS5_SPEC_OWN_PENDING") != nullptr;
    return OwnImageRefreshEnabled() && LiveFlag("spec_own_pending", requested);
}

std::array<std::uint64_t, static_cast<std::size_t>(ShaderResources::PendingVerdict::Count)> ShaderResources::TakePendingVerdicts() {
    std::array<std::uint64_t, static_cast<std::size_t>(PendingVerdict::Count)> taken{};
    auto& counts = PendingVerdictCounts();
    for (std::size_t i = 0; i < taken.size(); ++i) taken[i] = counts[i].exchange(0, std::memory_order_relaxed);
    return taken;
}

ShaderResources::SpeculationMiss ShaderResources::SpeculateProof(ProofSpeculation& speculation) const {
    using Miss = SpeculationMiss;
    static const bool noFast = std::getenv("APS5_NO_FAST_REVALIDATE") != nullptr;
    if (noFast || !EpochRevalidate() || !(reusable || addressReusable)) return Miss::Record;
    speculation.surfaces.clear();
    speculation.pending.clear();
    speculation.depths.clear();
    speculation.scanned = false;
    speculation.ownPending = SpeculateOwnPending();
    speculation.ownAccepted = false;
    speculation.tokens = false;
    speculation.evictions = storageEvictions.load(std::memory_order_acquire);
    speculation.stamps = GuestMemory::StampSerial();
    const bool tokens = DrawTokensEnabled();
    if (tokens) {
        speculation.collectEpoch = GuestMemory::ThreadCollectEpoch();
        speculation.unwatchSerial = GuestMemory::UnwatchSerial();
        speculation.allocations = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    }
    std::uint64_t seen = 0;
    {
        const ForkGuard lock(proofMutex);
        if (validatedTextures.size() != textures.size()) return Miss::Record;
        speculation.version = proofVersion;
        seen = pendingSerialSeen;
        for (std::size_t i = 0; i < validatedTextures.size(); ++i) {
            if (i < speculation.skip.size() && speculation.skip[i]) continue;
            const auto& surface = validatedTextures[i];
            if (!surface.valid) return Miss::Record;
            if (surface.depth) {
                static const bool depthDefault = std::getenv("APS5_SPEC_DEPTH") != nullptr;
                if (ForkLogged() == nullptr && !LiveFlag("spec_depth", depthDefault)) return Miss::Depth;
                speculation.depths.push_back(i);
                continue;
            }
            if (surface.resource.dccAddress != 0 || surface.keys != DccKeys::Uncompressed) return Miss::Keys;
            speculation.surfaces.push_back({surface.resource.baseAddress, static_cast<std::size_t>(surface.bytes), surface.source, surface.generation, i, false});
        }
    }
    speculation.pendingSerial = StorageTexture::PendingSerial();
    thread_local std::vector<GuestMemory::UnchangedQuery>* queriesSlot = nullptr;
    auto& queries = ShaderRecompiler::ThreadOwned(queriesSlot);
    thread_local std::vector<const StorageTexture*>* imagesSlot = nullptr;
    auto& images = ShaderRecompiler::ThreadOwned(imagesSlot);
    queries.clear();
    images.clear();
    const bool fast = FastProofEnabled();
    for (auto& surface : speculation.surfaces) {
        if (surface.image != nullptr && surface.image->Descriptor().dccAddress != 0) return Miss::Keys;
        if (GuestMemory::CollectWrites(surface.address, surface.bytes) == 0) return Miss::Collect;
        if (surface.image != nullptr) {
            if (DepthSurfaceAt(surface.address)) return Miss::Depth;
            if (!surface.image->Cached()) return Miss::Evicted;
            images.push_back(surface.image);
            // The committer's check of the image (memory unchanged since its generation), made here
            // too for the fast accept, which then needs only the generation to stay put.
            if (fast) {
                surface.generation = surface.image->Generation();
                queries.push_back({surface.address, surface.bytes, surface.generation});
            }
        } else {
            queries.push_back({surface.address, surface.bytes, surface.generation});
        }
        speculation.pending.push_back({surface.address, surface.address + surface.bytes, surface.image, nullptr, false});
    }
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        const auto* image = storageTextures[i].get();
        if (image == nullptr) return Miss::Record;
        if (i >= storageKeys.size() || storageKeys[i] != 0) return Miss::Keys;
        if (i != 0 && storageTextures[i] == storageTextures[i - 1]) continue;
        const auto& own = image->Descriptor();
        if (own.dccAddress != 0) return Miss::Keys;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (GuestMemory::CollectWrites(own.baseAddress, bytes) == 0) return Miss::Collect;
        if (!image->Cached()) return Miss::Evicted;
        const auto imageGeneration = fast ? image->Generation() : 0;
        if (fast) queries.push_back({own.baseAddress, bytes, imageGeneration});
        speculation.surfaces.push_back({own.baseAddress, bytes, image, imageGeneration, i, true});
        speculation.pending.push_back({own.baseAddress, own.baseAddress + bytes, image, nullptr, false});
        images.push_back(image);
    }
    const bool unchanged = seen != 0 && seen == speculation.pendingSerial;
    if (!unchanged && !speculation.pending.empty()) {
        thread_local std::vector<StorageTexture::PendingQuery>* scanSlot = nullptr;
        auto& scan = ShaderRecompiler::ThreadOwned(scanSlot);
        scan.assign(speculation.pending.begin(), speculation.pending.end());
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        if (!StorageTexture::ScanPending(scan)) {
            if (profile) notePendingRefusal(scan, SpeculationRefusalDetail::Identity, true);
            return Miss::Pending;
        }
        const auto verdict = SpeculatedPendingVerdict(speculation.surfaces, scan, speculation.ownPending);
        if (verdict != PendingVerdict::Clear) {
            PendingVerdictCounts()[static_cast<std::size_t>(verdict)].fetch_add(1, std::memory_order_relaxed);
            if (profile) notePendingRefusal(scan, static_cast<std::uint8_t>(static_cast<std::uint8_t>(verdict) + 1u), false);
            return Miss::Pending;
        }
        if (std::any_of(scan.begin(), scan.end(), [](const StorageTexture::PendingQuery& query) { return query.overlaps; })) {
            speculation.ownAccepted = true;
            PendingVerdictCounts()[static_cast<std::size_t>(PendingVerdict::Clear)].fetch_add(1, std::memory_order_relaxed);
        }
        speculation.scanned = true;
    }
    if (!queries.empty() && !GuestMemory::UnchangedSinceAll(queries)) return Miss::Changed;
    if (!StorageImagesPeek(context, images)) return Miss::Evicted;
    speculation.tokens = tokens && speculation.collectEpoch != 0 && GuestMemory::ThreadCollectEpoch() == speculation.collectEpoch && GuestMemory::UnwatchSerial() == speculation.unwatchSerial && GuestAllocations::GuestAllocationsGeneration_nid_postfix() == speculation.allocations;
    return Miss::None;
}

ShaderResources::SpeculationMiss ShaderResources::ProveSpeculated(const ProofSpeculation& speculation) {
    using Miss = SpeculationMiss;
    if (!(reusable || addressReusable)) return Miss::Record;
    std::lock_guard lock(proofMutex);
    if (proofVersion != speculation.version || validatedTextures.size() != textures.size()) return Miss::Version;
    // An own-source overlap accepted on prep stands only while the switch stays on.
    if (speculation.ownAccepted && !SpeculateOwnPending()) return Miss::Pending;
    if (!speculation.depths.empty()) {
        static const bool depthFastPath = std::getenv("APS5_NO_DEPTH_FAST_PROOF") == nullptr;
        if (!depthFastPath) return Miss::Depth;
        for (const auto index : speculation.depths) {
            if (index >= validatedTextures.size() || index >= textures.size()) return Miss::Depth;
            const auto& surface = validatedTextures[index];
            if (!surface.depth || DepthSurfaceTexture(context, std::span<const std::uint32_t>(surface.words.data(), surface.wordCount), surface.resource, surface.components) != textures[index]) return Miss::Depth;
        }
    }
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto serialBefore = StorageTexture::PendingSerial();
    thread_local std::vector<GuestMemory::UnchangedQuery>* queriesSlot = nullptr;
    auto& queries = ShaderRecompiler::ThreadOwned(queriesSlot);
    thread_local std::vector<const StorageTexture*>* imagesSlot = nullptr;
    auto& images = ShaderRecompiler::ThreadOwned(imagesSlot);
    thread_local std::vector<std::uint64_t>* collectedSlot = nullptr;
    auto& collected = ShaderRecompiler::ThreadOwned(collectedSlot);
    const auto checks = [&](bool memoized) {
        queries.clear();
        images.clear();
        collected.assign(speculation.surfaces.size(), 0);
        const auto current = memoized ? GuestMemory::CollectedGeneration() : 0;
        for (std::size_t k = 0; k < speculation.surfaces.size(); ++k) {
            const auto& surface = speculation.surfaces[k];
            collected[k] = memoized ? current : GuestMemory::CollectWrites(surface.address, surface.bytes);
            if (collected[k] == 0) return Miss::Collect;
            if (surface.image == nullptr) {
                queries.push_back({surface.address, surface.bytes, surface.generation});
                continue;
            }
            if (!surface.storage) {
                if (DepthSurfaceAt(surface.address)) return Miss::Depth;
                if (surface.image->UploadedKeys() != DccKeys::Uncompressed) return Miss::Keys;
            }
            if (!surface.image->Cached()) return Miss::Evicted;
            queries.push_back({surface.address, surface.bytes, surface.image->Generation()});
            images.push_back(surface.image);
        }
        const bool unchanged = pendingSerialSeen != 0 && pendingSerialSeen == serialBefore;
        if (!unchanged && !speculation.pending.empty() && !(speculation.scanned && speculation.pendingSerial == serialBefore)) {
            thread_local std::vector<StorageTexture::PendingQuery>* scanSlot = nullptr;
            auto& scan = ShaderRecompiler::ThreadOwned(scanSlot);
            scan.assign(speculation.pending.begin(), speculation.pending.end());
            // The prep thread's rule while the switch stays on (off: any overlap refuses, as before).
            if (!StorageTexture::ScanPending(scan) || SpeculatedPendingVerdict(speculation.surfaces, scan, speculation.ownPending && SpeculateOwnPending()) != PendingVerdict::Clear) return Miss::Pending;
        }
        if (!GuestMemory::UnchangedSinceAll(queries)) return Miss::Changed;
        return Miss::None;
    };
    auto fallback = TokenFallback::Count;
    if (DrawTokensEnabled()) {
        fallback = !speculation.tokens ? TokenFallback::NoTokens
            : GuestMemory::ThreadCollectEpoch() != speculation.collectEpoch ? TokenFallback::Epoch
            : GuestMemory::UnwatchSerial() != speculation.unwatchSerial ? TokenFallback::Unwatch
            : GuestAllocations::GuestAllocationsGeneration_nid_postfix() != speculation.allocations ? TokenFallback::Allocations
            : speculation.window;
        CountTokenOutcome(TokenFamily::Proof, fallback);
    }
    const bool memoized = fallback == TokenFallback::None;
    if (memoized && VerifyDrawTokens()) {
        if (const auto full = checks(false); full != Miss::None && checks(true) == Miss::None) {
            std::fprintf(stderr, "[tokens] APS5_VERIFY_DRAW_TOKENS: a token-accepted proof is refused by the full check (miss %zu)\n", static_cast<std::size_t>(full));
            std::fflush(stderr);
            std::abort();
        }
    }
    // The fast accept (live `fast_proof`, env APS5_FAST_PROOF): with the token window clear, the
    // collect epoch, unwatch and allocation serials what the prep thread proved under, the pending
    // registry unchanged since (its serial) and no storage image evicted since, every check below
    // would repeat the prep thread's answer, so they are skipped. APS5_VERIFY_FAST_PROOF=1 runs
    // them anyway and aborts on a disagreement.
    auto& fastStats = FastProofStats();
    const bool evictionsSame = speculation.evictions == storageEvictions.load(std::memory_order_acquire);
    const bool generationsSame = std::all_of(speculation.surfaces.begin(), speculation.surfaces.end(), [](const ProofSpeculation::Surface& surface) { return surface.image == nullptr || (surface.generation != 0 && surface.image->Generation() == surface.generation); });
    const bool stampsSame = speculation.stamps == GuestMemory::StampSerial();
    const bool fastAccept = memoized && FastProofEnabled() && speculation.pendingSerial == serialBefore && evictionsSame && generationsSame && stampsSame;
    if (FastProofEnabled()) {
        fastStats.calls.fetch_add(1, std::memory_order_relaxed);
        if (!memoized) fastStats.noTokens.fetch_add(1, std::memory_order_relaxed);
        else if (!stampsSame) fastStats.stamps.fetch_add(1, std::memory_order_relaxed);
        else if (speculation.pendingSerial != serialBefore) fastStats.pending.fetch_add(1, std::memory_order_relaxed);
        else if (!evictionsSame) fastStats.evicted.fetch_add(1, std::memory_order_relaxed);
        else if (!fastAccept) fastStats.generation.fetch_add(1, std::memory_order_relaxed);
        else fastStats.accepted.fetch_add(1, std::memory_order_relaxed);
        ReportFastProof();
    }
    if (fastAccept && !VerifyFastProof()) {
        images.clear();
        collected.assign(speculation.surfaces.size(), GuestMemory::CollectedGeneration());
        for (const auto& surface : speculation.surfaces) {
            if (surface.image != nullptr) images.push_back(surface.image);
        }
    } else {
        const auto miss = checks(memoized);
        const bool cached = miss == Miss::None && StorageImagesCached(context, images);
        if (fastAccept && (miss != Miss::None || !cached)) {
            std::fprintf(stderr, "[fast-proof] APS5_VERIFY_FAST_PROOF: a fast-accepted proof fails the full checks (miss %zu, cached %d)\n", static_cast<std::size_t>(miss), cached ? 1 : 0);
            std::fflush(stderr);
            std::abort();
        }
        if (miss != Miss::None) return miss;
        if (!cached) return Miss::Evicted;
    }
    for (const auto* image : images) image->NoteProved();
    for (std::size_t k = 0; k < speculation.surfaces.size(); ++k) {
        const auto& surface = speculation.surfaces[k];
        if (surface.storage || surface.image != nullptr || surface.element >= validatedTextures.size()) continue;
        auto& record = validatedTextures[surface.element];
        record.generation = std::max(record.generation, collected[k]);
    }
    if (LookupMemoEnabled()) lookupMemo.epoch = 0;
    const auto failure = proofTail(serialBefore, nullptr);
    if (profile) countRevalidate(true, failure == ProofFailure::None, start);
    return failure == ProofFailure::None ? Miss::None : Miss::Tail;
}

namespace {

// Whether a DCC key store or label store queued in the open batch, a label this thread deferred but
// has not recorded yet (the flush hook would record it before a hooked read), or a recorded label
// still pending in the batch overlaps a streamed element's range (movedBuffers, MovedStillFree):
// the stream copy is a raw memcpy, so it must refuse every case the hooked read would act on.
bool StreamStoreQueued(const Recorder& recorder, std::uint64_t begin, std::size_t bytes) {
    return (recorder.HasQueuedKeyStores() && recorder.QueuedKeyStoreOverlaps(begin, bytes)) || (recorder.HasQueuedStores() && recorder.QueuedStoreOverlaps(begin, bytes)) || Recorder::QueuedLabelsOverlap(begin, bytes) || recorder.PendingLabelIn(begin, bytes);
}

}

namespace {

std::atomic<std::uint64_t> movedPublishChecks{0};

// The ranges SpeculateMovedFree lists for `moved` (its entries without words).
std::size_t movedRangeCount(std::span<const ShaderResources::MovedBuffer> moved) {
    return static_cast<std::size_t>(std::count_if(moved.begin(), moved.end(), [](const ShaderResources::MovedBuffer& entry) { return entry.words.empty(); }));
}

}

bool ShaderResources::MovedPublishCheck() {
    static const bool requested = std::getenv("APS5_MOVED_PUBLISH_CHECK") != nullptr;
    return LiveFlag("moved_publish_check", requested);
}

std::uint64_t ShaderResources::TakeMovedPublishChecks() {
    return movedPublishChecks.exchange(0, std::memory_order_relaxed);
}

bool ShaderResources::MovedStillFree(std::span<const MovedBuffer> moved, Recorder* recorder, const MovedProof* proof) const {
    // Streamed elements refuse queued stores at commit whatever the token proof says (movedBuffers
    // ran on the prep thread without a recorder).
    if (streamed && recorder != nullptr) {
        for (const auto& entry : moved) {
            if (!entry.words.empty() || entry.allocation >= allocations.size() || !allocations[entry.allocation].streamed) continue;
            const auto& item = allocations[entry.allocation];
            if (StreamStoreQueued(*recorder, entry.address - item.adjustment, static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment)))) return false;
        }
    }
    const bool tokens = DrawTokensEnabled();
    const auto full = [&] {
        for (const auto& entry : moved) {
            if (!entry.words.empty()) continue;
            if (entry.allocation >= allocations.size()) return false;
            const auto& item = allocations[entry.allocation];
            const auto begin = entry.address - item.adjustment;
            const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment));
            if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return false;
            if (guestMemory.WritesOverlap(begin, bytes) || (recorder != nullptr && (tokens ? recorder->PendingWriteOverlapsScreened(begin, bytes) : recorder->PendingWriteOverlaps(begin, bytes))) || PendingStorageOverlaps(begin, bytes, nullptr) || AnyShadowedOverlaps(begin, bytes)) return false;
        }
        return true;
    };
    if (!tokens || proof == nullptr) return full();
    const auto fallback = !proof->tokens ? TokenFallback::NoTokens
        : Recorder::PublishGeneration() != proof->publish ? TokenFallback::Publish
        : StorageTexture::PendingSerial() != proof->pendingSerial ? TokenFallback::Pending
        : GuestAllocations::GuestAllocationsGeneration_nid_postfix() != proof->allocations || GuestMemory::ForgetSerial() != proof->forget ? TokenFallback::Allocations
        : proof->window;
    CountTokenOutcome(TokenFamily::Moved, fallback);
    // Only the pending-write snapshot moved since the prep thread's proof (live
    // `moved_publish_check`): every other input of the token (the pending registry serial, the
    // allocation and forget serials, the window) still holds, and the publish generation guards
    // only the recorder's pending writes. So the full check's other tests repeat the prep thread's
    // answers, and the pending writes of the ranges are tested again now, as full() tests them.
    // The fallback names the first input that moved, so the others are compared here.
    const bool publishOnly = fallback == TokenFallback::Publish && StorageTexture::PendingSerial() == proof->pendingSerial && GuestAllocations::GuestAllocationsGeneration_nid_postfix() == proof->allocations && GuestMemory::ForgetSerial() == proof->forget && proof->window == TokenFallback::None;
    if (publishOnly && MovedPublishCheck() && proof->ranges.size() == movedRangeCount(moved)) {
        bool free = true;
        if (recorder != nullptr) {
            for (const auto& [begin, end] : proof->ranges) {
                if (recorder->PendingWriteOverlapsScreened(begin, static_cast<std::size_t>(end - begin))) {
                    free = false;
                    break;
                }
            }
        }
        movedPublishChecks.fetch_add(1, std::memory_order_relaxed);
        if (VerifyDrawTokens() && free != full()) {
            std::fprintf(stderr, "[tokens] APS5_VERIFY_DRAW_TOKENS: the publish-only moved check disagrees with the full check\n");
            std::fflush(stderr);
            std::abort();
        }
        return free;
    }
    if (fallback != TokenFallback::None) return full();
    if (VerifyDrawTokens() && !full()) {
        std::fprintf(stderr, "[tokens] APS5_VERIFY_DRAW_TOKENS: token-accepted moved buffers are refused by the full check\n");
        std::fflush(stderr);
        std::abort();
    }
    return true;
}

void ShaderResources::SpeculateMovedFree(std::span<const MovedBuffer> moved, MovedProof& proof) const {
    proof = {};
    if (!DrawTokensEnabled()) return;
    proof.publish = Recorder::PublishGeneration();
    proof.pendingSerial = StorageTexture::PendingSerial();
    proof.allocations = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    proof.forget = GuestMemory::ForgetSerial();
    if ((proof.forget & 1u) != 0) return;
    const auto snapshot = Recorder::PendingWriteSnapshot();
    for (const auto& entry : moved) {
        if (!entry.words.empty()) continue;
        if (entry.allocation >= allocations.size()) return;
        const auto& item = allocations[entry.allocation];
        const auto begin = entry.address - item.adjustment;
        const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment));
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return;
        if (guestMemory.WritesOverlap(begin, bytes) || Recorder::SnapshotOverlaps(snapshot.get(), begin, bytes) || PendingStorageOverlaps(begin, bytes, nullptr) || AnyShadowedOverlaps(begin, bytes)) return;
        proof.ranges.emplace_back(begin, begin + bytes);
    }
    proof.tokens = Recorder::PublishGeneration() == proof.publish && StorageTexture::PendingSerial() == proof.pendingSerial && GuestAllocations::GuestAllocationsGeneration_nid_postfix() == proof.allocations && GuestMemory::ForgetSerial() == proof.forget;
}

bool ShaderResources::SpeculateTemplate(std::span<const CompiledShader> shaders, DrawSpeculation& speculation) const {
    using Miss = SpeculationMiss;
    if (!SameImageWords(shaders)) {
        speculation.miss = Miss::ImageWords;
        return false;
    }
    if (const auto miss = SpeculateProof(speculation.proof); miss != Miss::None) {
        speculation.miss = miss;
        return false;
    }
    auto moved = _set == VK_NULL_HANDLE || usesBda ? std::optional<std::vector<MovedBuffer>>(std::vector<MovedBuffer>{}) : movedBuffers(shaders, nullptr, false);
    if (!moved.has_value()) {
        speculation.miss = Miss::Moved;
        return false;
    }
    speculation.moved = std::move(*moved);
    SpeculateMovedFree(speculation.moved, speculation.movedProof);
    speculation.kind = DrawSpeculation::Kind::Template;
    return true;
}

bool ShaderResources::SpeculateAddressUse(const std::shared_ptr<ShaderResources>& source, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, DrawSpeculation& speculation) {
    using Miss = SpeculationMiss;
    const auto refuse = [&](Miss miss) {
        speculation.miss = miss;
        speculation.use.reset();
        return false;
    };
    if (source == nullptr || !source->addressReusable || source->bda == nullptr) return refuse(Miss::NoEntry);
    static const bool movedDefault = std::getenv("APS5_NO_SPEC_MOVED") == nullptr;
    if (!MovableImages() || !LiveFlag("spec_moved", movedDefault)) {
        if (!source->SameImageWords(shaders)) return refuse(Miss::ImageWords);
    } else {
        speculation.proof.skip.assign(source->textures.size(), false);
        speculation.movedImages.clear();
        std::size_t index = 0;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) return refuse(Miss::Moved);
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.kind != ShaderRecompiler::DescriptorKind::SampledImage || binding.count == 0) continue;
                if (binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 8u || !binding.imageShape.has_value()) return refuse(Miss::Moved);
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                    if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                    if (index >= source->textureWords.size()) return refuse(Miss::ImageWords);
                    const auto at = index++;
                    if (std::equal(source->textureWords[at].begin(), source->textureWords[at].end(), words.begin())) continue;
                    if (at >= source->addressTextureSlots.size() || source->addressTextureSlots[at] == std::numeric_limits<std::size_t>::max() || at >= source->textureFirstLayer.size()) return refuse(Miss::ImageWords);
                    const auto resource = DecodeTextureResource(words);
                    const bool firstLayer = *binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
                    if (firstLayer != source->textureFirstLayer[at] || (!firstLayer && !MatchesGuestDimension(*binding.imageShape, resource.dimension))) return refuse(Miss::ImageWords);
                    if (at < speculation.proof.skip.size()) speculation.proof.skip[at] = true;
                    SpeculatedImage image{at, source->addressTextureSlots[at], firstLayer, {}};
                    std::copy(words.begin(), words.end(), image.words.begin());
                    // The element's stage-A record (precollectImages): collect, keys, then the entry.
                    if (SpeculatedImageRecords() && at < source->textureDepthCompare.size() && !source->textureDepthCompare[at]) {
                        image.resource = resource;
                        image.guestBytes = DescribeSurface(resource).guestBytes;
                        image.generation = GuestMemory::CollectWrites(resource.baseAddress, static_cast<std::size_t>(image.guestBytes));
                        image.components = ViewComponents(resource);
                        image.keys = TextureClearKeys(resource, image.guestBytes);
                        auto& cache = Textures();
                        std::lock_guard lock(cache.mutex);
                        if (const auto it = findTexture(cache, MakeTextureKey(source->context.device, words, image.components)); it != cache.entries.end()) {
                            image.texture = it->texture;
                            image.source = it->source;
                            image.entryKeys = it->keys;
                            image.entryGeneration = it->generation;
                            image.recorded = true;
                        }
                    }
                    speculation.movedImages.push_back(std::move(image));
                }
            }
        }
        if (index != source->textureWords.size()) return refuse(Miss::ImageWords);
    }
    auto moved = source->movedBuffers(shaders, nullptr, false);
    if (!moved.has_value()) return refuse(Miss::Moved);
    std::vector<std::pair<std::uint64_t, std::size_t>> ranges;
    for (const auto& entry : *moved) {
        if (entry.words.empty()) ranges.emplace_back(entry.address, entry.size);
    }
    std::vector<std::optional<GuestBufferMemory::SpaceView>> views;
    const auto space = GuestBufferMemory::SpeculatedSpace(source->context, source->addressSpaceSerial, ranges, views);
    if (space == nullptr) return refuse(Miss::Space);
    for (const auto& snapshot : snapshots) {
        if (!GuestBufferMemory::SpaceCovers(*space, snapshot.address, snapshot.bytes.size())) return refuse(Miss::Outside);
    }
    if (const auto miss = source->SpeculateProof(speculation.proof); miss != Miss::None) return refuse(miss);
    std::shared_ptr<ShaderResources> use(new ShaderResources(AddressUseTag{}, source));
    use->bda = source->bda->ShareTable(source->context);
    use->bda->SetMayWrite(std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program == nullptr || shader.program->bdaWrites; }));
    if (!use->writeAddressSet(*moved, views, &speculation.movedGuest)) return refuse(Miss::Set);
    speculation.moved = std::move(*moved);
    source->SpeculateMovedFree(speculation.moved, speculation.movedProof);
    speculation.space = space;
    speculation.use = std::move(use);
    speculation.kind = DrawSpeculation::Kind::Address;
    return true;
}

std::shared_ptr<ShaderResources> ShaderResources::UseAddressSpeculation(const std::shared_ptr<ShaderResources>& source, DrawSpeculation& speculation) {
    using Miss = SpeculationMiss;
    auto use = std::move(speculation.use);
    const auto refuse = [&](Miss miss) {
        speculation.miss = miss;
        // Kept for the committer to release with the batch (defer_drops); otherwise, and for a
        // use holding a lease, it goes here as before.
        if (use != nullptr && DeferDropsEnabled() && !use->HoldsLease()) speculation.refusedUse = std::move(use);
        return std::shared_ptr<ShaderResources>{};
    };
    // APS5_PROFILE_LOOKUP: [spec-split] times the committer's steps of an accepted address speculation.
    static const bool split = std::getenv("APS5_PROFILE_LOOKUP") != nullptr;
    struct Split {
        std::array<std::atomic<std::uint64_t>, 6> ns{};
        std::atomic<std::uint64_t> uses{0};
        std::atomic<std::int64_t> reported{0};
    };
    static Split splitStats;
    auto lapAt = split ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto lap = [&](std::size_t step) {
        if (!split) return;
        const auto now = std::chrono::steady_clock::now();
        splitStats.ns[step].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - lapAt).count()), std::memory_order_relaxed);
        lapAt = now;
    };
    if (use == nullptr || source == nullptr || !source->addressReusable || use->addressSource != source) return refuse(Miss::Identity);
    if (!source->MovedStillFree(speculation.moved, Recorder::Active(), &speculation.movedProof)) return refuse(Miss::MovedRecheck);
    lap(0);
    const auto space = speculation.space.lock();
    if (space == nullptr) return refuse(Miss::Space);
    if (use->guestMemory.ShareSpace(source->addressSpaceSerial, source->guestMemory.Writes()) != GuestBufferMemory::SpaceShare::Taken) return refuse(Miss::Shared);
    lap(1);
    if (!use->guestMemory.SharesSpace(space.get())) return refuse(Miss::Space);
    lap(2);
    if (const auto miss = source->ProveSpeculated(speculation.proof); miss != Miss::None) return refuse(miss);
    lap(3);
    if (!use->guestMemory.FinishSharedSpace()) return refuse(Miss::Finish);
    lap(4);
    if (!speculation.movedImages.empty() && !use->rebindSpeculatedImages(speculation.movedImages)) return refuse(Miss::MovedImage);
    lap(5);
    if (split) {
        const auto uses = splitStats.uses.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = splitStats.reported.load(std::memory_order_relaxed);
        if (now - last > 10'000'000'000ll && splitStats.reported.compare_exchange_strong(last, now)) {
            splitStats.uses.store(0);
            const auto us = [&](std::size_t step) { return static_cast<double>(splitStats.ns[step].exchange(0)) / 1000.0 / static_cast<double>(uses); };
            std::fprintf(stderr, "[spec-split] %llu accepted address speculations, us per use: moved recheck %.2f space share %.2f shares space %.2f proof %.2f finish %.2f images %.2f\n", static_cast<unsigned long long>(uses), us(0), us(1), us(2), us(3), us(4), us(5));
        }
    }
    auto& counts = AddressCounts();
    counts.hits.fetch_add(1, std::memory_order_relaxed);
    if (speculation.movedGuest != 0) counts.movedBuffers.fetch_add(speculation.movedGuest, std::memory_order_relaxed);
    return use;
}

namespace {

// APS5_PROFILE_DRAW: what makes content keys miss. On a miss the key is compared with the last key
// seen for the same shader variant(s) and the first differing word is charged to its binding's role
// (and to its index within the V# or T# element), printed as [rescache] every 10 s: it says whether
// V# bases of ring allocations, SRT words or image descriptors churn.
struct ChurnCounts {
    std::array<std::uint64_t, 8> roles{};
    std::array<std::uint64_t, 4> bufferWords{};
    std::array<std::uint64_t, 8> imageWords{};
    std::uint64_t firstSeen = 0;
    std::uint64_t sameKey = 0;
    std::uint64_t layout = 0;
    std::uint64_t drawWords = 0;
    std::array<std::uint64_t, 3> bufferKinds{};
    std::array<std::uint64_t, 8> flagGroups{};
    std::array<std::uint64_t, 8> headerWords{};
};

// Dispatch keys and draw keys are counted apart: what churns a draw's key decides the draw steps.
struct ChurnProfile {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, ResourceCache::Key> lastByVariant;
    ChurnCounts dispatch;
    ChurnCounts draws;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

ChurnProfile& Churn() {
    static ChurnProfile profile;
    return profile;
}

// Charges word `diff` of `key` to the binding of the ContentKey that starts at `at` (the layout of
// ShaderResources::ContentKey), moving `at` past that key; false when `diff` lies beyond it.
bool chargeShaderKey(ChurnCounts& profile, const ResourceCache::Key& key, std::size_t& at, std::size_t diff, bool movable = false) {
    if (at + 5 > key.size() || diff < at + 5) {
        ++profile.layout;
        return true;
    }
    const bool dataWords = key[at] != 0;
    const auto bindings = key[at + 4];
    at += 5;
    for (std::uint32_t binding = 0; binding < bindings; ++binding) {
        if (at + 8 > key.size() || diff < at + 8) {
            if (at + 8 <= key.size()) ++profile.headerWords[diff - at];
            ++profile.layout;
            return true;
        }
        const auto kind = key[at];
        const auto role = key[at + 1];
        const auto descriptorWords = (dataWords && !movable) || !DataRole(static_cast<ShaderRecompiler::DescriptorRole>(role)) ? key[at + 7] : 0u;
        at += 8;
        if (diff < at + descriptorWords) {
            if (role < profile.roles.size()) ++profile.roles[role];
            const auto index = diff - at;
            if (role == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorRole::GuestBuffers)) {
                ++profile.bufferWords[index % 4];
                const auto element = index / 4;
                const auto* words = key.data() + at + element * 4;
                std::size_t flagsAt = at + descriptorWords;
                for (int flags = 0; flags < 6 && flagsAt < key.size(); ++flags) flagsAt += 1 + (static_cast<std::size_t>(key[flagsAt]) + 31) / 32;
                const bool flagged = flagsAt < key.size() && (element >= key[flagsAt] || (flagsAt + 1 + element / 32 < key.size() && ((key[flagsAt + 1 + element / 32] >> (element % 32)) & 1u) != 0));
                const bool empty = words[2] == 0 || (words[0] == 0 && (words[1] & 0xffffu) == 0);
                ++profile.bufferKinds[flagged ? 0 : empty ? 1 : 2];
            } else if (role == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorRole::GuestImages) && (kind == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorKind::SampledImage) || kind == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorKind::StorageImage))) ++profile.imageWords[index % 8];
            return true;
        }
        at += descriptorWords;
        for (int flags = 0; flags < 8; ++flags) {
            if (at >= key.size()) {
                ++profile.layout;
                return true;
            }
            const auto words = flags == 7 ? 1 + static_cast<std::size_t>(key[at]) : 1 + (static_cast<std::size_t>(key[at]) + 31) / 32;
            if (diff < at + words) {
                ++profile.flagGroups[flags];
                ++profile.layout;
                return true;
            }
            at += words;
        }
    }
    return false;
}

}

std::size_t ResourceCache::HashKey(const Key& key) noexcept {
    constexpr std::uint64_t Prime = 0x9e3779b97f4a7c15ull;
    std::uint64_t lanes[4] = {0x243f6a8885a308d3ull, 0x13198a2e03707344ull, 0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull};
    const auto* words = key.data();
    const auto count = key.size();
    std::size_t at = 0;
    for (; at + 8u <= count; at += 8u) {
        for (std::size_t lane = 0; lane < 4u; ++lane) {
            const auto value = static_cast<std::uint64_t>(words[at + lane * 2u]) | static_cast<std::uint64_t>(words[at + lane * 2u + 1u]) << 32u;
            lanes[lane] = (lanes[lane] ^ value) * Prime;
            lanes[lane] ^= lanes[lane] >> 29u;
        }
    }
    std::uint64_t hash = lanes[0] ^ (lanes[1] << 17u | lanes[1] >> 47u) ^ (lanes[2] << 31u | lanes[2] >> 33u) ^ (lanes[3] << 47u | lanes[3] >> 17u) ^ count;
    for (; at < count; ++at) hash = (hash ^ words[at]) * Prime;
    hash ^= hash >> 33u;
    hash *= 0xff51afd7ed558ccdull;
    hash ^= hash >> 33u;
    hash *= 0xc4ceb9fe1a85ec53ull;
    hash ^= hash >> 33u;
    return static_cast<std::size_t>(hash);
}

std::shared_ptr<ShaderResources> ResourceCache::Find(const Key& key) {
    return Find(key, HashKey(key));
}

std::shared_ptr<ShaderResources> ResourceCache::Find(const Key& key, std::size_t hash) {
    resourceCacheFinds.fetch_add(1, std::memory_order_relaxed);
    const ForkGuard lock(mutex);
    const auto found = index.find(HashedKey{&key, hash});
    if (found == index.end()) {
        noteMiss(key);
        return nullptr;
    }
    entries.splice(entries.begin(), entries, found->second);
    return found->second->second;
}

std::shared_ptr<ShaderResources> ResourceCache::Peek(const Key& key) const {
    return Peek(key, HashKey(key));
}

std::shared_ptr<ShaderResources> ResourceCache::Peek(const Key& key, std::size_t hash) const {
    const ForkGuard lock(mutex);
    const auto found = index.find(HashedKey{&key, hash});
    return found != index.end() ? found->second->second : nullptr;
}

std::shared_ptr<ShaderResources> ResourceCache::Peek(const Key& key, std::size_t hash, Handle& handle) const {
    handle = {};
    const ForkGuard lock(mutex);
    const auto found = index.find(HashedKey{&key, hash});
    if (found == index.end()) return nullptr;
    handle.cache = this;
    handle.entry = found->second;
    handle.stamp = found->second->stamp;
    return found->second->second;
}

bool ResourceCache::TouchStamped(const Handle& handle, const ShaderResources& object) {
    if (handle.cache != this || handle.stamp == 0) return false;
    const ForkGuard lock(mutex);
    // The stamp is set and cleared under this mutex for this cache's entries: still the handle's,
    // the entry has not left the cache (an erase or an eviction clears it, an insert of the same
    // key makes a new entry with a new stamp), so the position is valid and holds `object`.
    if (object.cacheStamp.load(std::memory_order_relaxed) != handle.stamp) return false;
    resourceCacheFinds.fetch_add(1, std::memory_order_relaxed);
    entries.splice(entries.begin(), entries, handle.entry);
    return true;
}

std::uint64_t ResourceCache::stampEntry(ShaderResources& object) {
    static std::atomic<std::uint64_t> next{0};
    const auto stamp = next.fetch_add(1, std::memory_order_relaxed) + 1;
    object.cacheStamp.store(stamp, std::memory_order_relaxed);
    return stamp;
}

void ResourceCache::unstampEntry(const Entry& entry) {
    if (entry.second == nullptr) return;
    auto expected = entry.stamp;
    entry.second->cacheStamp.compare_exchange_strong(expected, 0, std::memory_order_relaxed);
}

void ResourceCache::noteMiss(const Key& key) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile || key.size() < 5) return;
    // The variant(s) the key belongs to: a compute key starts with the data-words flag, the stage
    // and its variant, a draw key with a marker, the device and its stages' keys (Draw.cpp
    // DrawResourceKey).
    const bool draw = key[0] == 0xffffffffu;
    std::uint64_t variants = 0;
    if (draw) {
        std::size_t at = 5;
        for (std::uint32_t stage = 0; stage < key[3] && at + 4 < key.size(); ++stage) {
            variants = variants * 1000003ull ^ (key[at + 3] | (static_cast<std::uint64_t>(key[at + 4]) << 32u));
            at += 1 + key[at];
        }
    } else {
        variants = key[2] | (static_cast<std::uint64_t>(key[3]) << 32u);
    }
    auto& churn = Churn();
    std::lock_guard lock(churn.mutex);
    auto& counts = draw ? churn.draws : churn.dispatch;
    const auto found = churn.lastByVariant.find(variants);
    if (found == churn.lastByVariant.end()) {
        ++counts.firstSeen;
        churn.lastByVariant.emplace(variants, key);
    } else {
        const auto& previous = found->second;
        const auto common = std::min(key.size(), previous.size());
        std::size_t diff = 0;
        while (diff < common && key[diff] == previous[diff]) ++diff;
        if (diff == common && key.size() == previous.size()) {
            ++counts.sameKey;
        } else if (draw) {
            std::size_t at = 5;
            bool charged = false;
            for (std::uint32_t stage = 0; stage < key[3] && at < key.size() && !charged; ++stage) {
                const auto partWords = static_cast<std::size_t>(key[at]);
                const auto partBegin = at + 1;
                const auto partEnd = std::min(key.size(), partBegin + partWords);
                if (diff < partEnd) {
                    const ResourceCache::Key part(key.begin() + static_cast<std::ptrdiff_t>(partBegin), key.begin() + static_cast<std::ptrdiff_t>(partEnd));
                    std::size_t partAt = 0;
                    if (!chargeShaderKey(counts, part, partAt, diff - partBegin, true)) ++counts.layout;
                    charged = true;
                    static std::atomic<int> imageSamples{0};
                    if (diff + 8 <= key.size() && diff + 8 <= previous.size() && imageSamples.fetch_add(1, std::memory_order_relaxed) < 60) std::fprintf(stderr, "[rescache] churn word %zu: old %08x %08x %08x %08x %08x %08x %08x %08x new %08x %08x %08x %08x %08x %08x %08x %08x\n", diff, previous[diff], previous[diff + 1], previous[diff + 2], previous[diff + 3], previous[diff + 4], previous[diff + 5], previous[diff + 6], previous[diff + 7], key[diff], key[diff + 1], key[diff + 2], key[diff + 3], key[diff + 4], key[diff + 5], key[diff + 6], key[diff + 7]);
                }
                at = partEnd;
            }
            if (!charged) ++counts.drawWords;
        } else {
            std::size_t at = 0;
            if (!chargeShaderKey(counts, key, at, diff)) ++counts.layout;
        }
        found->second = key;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - churn.lastReport < std::chrono::seconds(10)) return;
    churn.lastReport = now;
    const auto report = [](const char* what, const ChurnCounts& churn) {
        std::string line = std::string("[rescache] miss churn (") + what + "), first differing word by role:";
        char item[256];
        for (std::size_t role = 0; role < churn.roles.size(); ++role) {
            if (churn.roles[role] == 0) continue;
            std::snprintf(item, sizeof(item), " %s %llu", roleName(static_cast<ShaderRecompiler::DescriptorRole>(role)), static_cast<unsigned long long>(churn.roles[role]));
            line += item;
            if (role == static_cast<std::size_t>(ShaderRecompiler::DescriptorRole::GuestBuffers)) {
                std::snprintf(item, sizeof(item), " (V# word %llu/%llu/%llu/%llu)", static_cast<unsigned long long>(churn.bufferWords[0]), static_cast<unsigned long long>(churn.bufferWords[1]), static_cast<unsigned long long>(churn.bufferWords[2]), static_cast<unsigned long long>(churn.bufferWords[3]));
                line += item;
                std::snprintf(item, sizeof(item), " (written %llu, empty %llu, read-only %llu)", static_cast<unsigned long long>(churn.bufferKinds[0]), static_cast<unsigned long long>(churn.bufferKinds[1]), static_cast<unsigned long long>(churn.bufferKinds[2]));
                line += item;
            } else if (role == static_cast<std::size_t>(ShaderRecompiler::DescriptorRole::GuestImages)) {
                std::snprintf(item, sizeof(item), " (T# word %llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu)", static_cast<unsigned long long>(churn.imageWords[0]), static_cast<unsigned long long>(churn.imageWords[1]), static_cast<unsigned long long>(churn.imageWords[2]), static_cast<unsigned long long>(churn.imageWords[3]), static_cast<unsigned long long>(churn.imageWords[4]), static_cast<unsigned long long>(churn.imageWords[5]), static_cast<unsigned long long>(churn.imageWords[6]), static_cast<unsigned long long>(churn.imageWords[7]));
                line += item;
            }
        }
        std::snprintf(item, sizeof(item), "; header words %llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu, flag groups %llu/%llu/%llu/%llu/%llu/%llu/%llu, samplers %llu", static_cast<unsigned long long>(churn.headerWords[0]), static_cast<unsigned long long>(churn.headerWords[1]), static_cast<unsigned long long>(churn.headerWords[2]), static_cast<unsigned long long>(churn.headerWords[3]), static_cast<unsigned long long>(churn.headerWords[4]), static_cast<unsigned long long>(churn.headerWords[5]), static_cast<unsigned long long>(churn.headerWords[6]), static_cast<unsigned long long>(churn.headerWords[7]), static_cast<unsigned long long>(churn.flagGroups[0]), static_cast<unsigned long long>(churn.flagGroups[1]), static_cast<unsigned long long>(churn.flagGroups[2]), static_cast<unsigned long long>(churn.flagGroups[3]), static_cast<unsigned long long>(churn.flagGroups[4]), static_cast<unsigned long long>(churn.flagGroups[5]), static_cast<unsigned long long>(churn.flagGroups[6]), static_cast<unsigned long long>(churn.flagGroups[7]));
        line += item;
        std::snprintf(item, sizeof(item), "; layout %llu, draw target/index %llu, same key %llu (not inserted or evicted), first seen %llu", static_cast<unsigned long long>(churn.layout), static_cast<unsigned long long>(churn.drawWords), static_cast<unsigned long long>(churn.sameKey), static_cast<unsigned long long>(churn.firstSeen));
        line += item;
        AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    };
    report("dispatch", churn.dispatch);
    report("draws", churn.draws);
}

void ResourceCache::Insert(const Key& key, std::shared_ptr<ShaderResources> resources, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    std::lock_guard lock(mutex);
    if (const auto found = index.find(key); found != index.end()) {
        unstampEntry(*found->second);
        if (evicted != nullptr) evicted->push_back(std::move(found->second->second));
        entries.erase(found->second);
        index.erase(found);
    }
    const auto stamp = resources != nullptr ? stampEntry(*resources) : 0;
    entries.emplace_front(key, std::move(resources), stamp);
    index.emplace(key, entries.begin());
    // Entries pin their textures and storage images past the texture caches' budgets, so the bound
    // stays modest: APS5_RESOURCE_CACHE_ENTRIES (default 1024) covers several frames of distinct
    // dispatch and draw content.
    static const std::size_t capacity = [] {
        const char* value = std::getenv("APS5_RESOURCE_CACHE_ENTRIES");
        const auto parsed = value ? std::strtoull(value, nullptr, 10) : 1024ull;
        return static_cast<std::size_t>(parsed != 0 ? parsed : 1024ull);
    }();
    while (entries.size() > capacity) {
        unstampEntry(entries.back());
        if (evicted != nullptr) evicted->push_back(std::move(entries.back().second));
        index.erase(entries.back().first);
        entries.pop_back();
    }
}

void ResourceCache::Remove(const Key& key, const ShaderResources* object, std::shared_ptr<ShaderResources>* removed) {
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    if (found == index.end() || (object != nullptr && found->second->second.get() != object)) return;
    unstampEntry(*found->second);
    if (removed != nullptr) *removed = std::move(found->second->second);
    entries.erase(found->second);
    index.erase(found);
}

bool ResourceCache::Touch(const Key& key) {
    resourceCacheTouches.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    if (found == index.end()) return false;
    entries.splice(entries.begin(), entries, found->second);
    return true;
}

std::uint64_t ResourceCache::Finds() {
    return resourceCacheFinds.load(std::memory_order_relaxed);
}

std::uint64_t ResourceCache::Touches() {
    return resourceCacheTouches.load(std::memory_order_relaxed);
}

ResourceCache::~ResourceCache() {
    for (const auto& entry : entries) unstampEntry(entry);
}

void ResourceCache::Clear() {
    std::lock_guard lock(mutex);
    for (const auto& entry : entries) unstampEntry(entry);
    index.clear();
    entries.clear();
}

std::size_t ResourceCache::Size() const {
    std::lock_guard lock(mutex);
    return entries.size();
}

void ResourceCache::erase(const Key& key) {
    const auto found = index.find(key);
    if (found == index.end()) return;
    unstampEntry(*found->second);
    entries.erase(found->second);
    index.erase(found);
}

ResourceCache& SharedResourceCache() {
    static auto* cache = new ResourceCache();
    return *cache;
}

DescriptorCache::DescriptorCache(const Context& context) : context(context), destroyLayout(context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")), destroyPool(context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")), freeSets(context.Function<PFN_vkFreeDescriptorSets>("vkFreeDescriptorSets")) {}

DescriptorCache::~DescriptorCache() {
    for (const auto pool : pools) destroyPool(context.device, pool, nullptr);
    for (const auto pool : dedicated) destroyPool(context.device, pool, nullptr);
    for (const auto& [key, layout] : layouts) destroyLayout(context.device, layout, nullptr);
}

VkDescriptorSetLayout DescriptorCache::Layout(std::span<const std::uint32_t> key, std::span<const VkDescriptorSetLayoutBinding> bindings) {
    std::lock_guard lock(mutex);
    std::vector<std::uint32_t> keyCopy(key.begin(), key.end());
    if (const auto found = layouts.find(keyCopy); found != layouts.end()) {
        ++stats.layoutHits;
        return found->second;
    }
    ValidateDescriptorLayout(context, bindings);
    ++stats.layoutMisses;
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &layout), "vkCreateDescriptorSetLayout");
    layouts.emplace(std::move(keyCopy), layout);
    return layout;
}

namespace {
std::uint32_t SetDescriptorLimit(const VkPhysicalDeviceLimits& limits, VkDescriptorType type) {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return limits.maxDescriptorSetStorageBuffers;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return limits.maxDescriptorSetSampledImages;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return limits.maxDescriptorSetStorageImages;
        case VK_DESCRIPTOR_TYPE_SAMPLER: return limits.maxDescriptorSetSamplers;
        default: throw std::runtime_error("AGC graphics: descriptor set uses an unsupported descriptor type " + std::to_string(type));
    }
}

// What one chain pool holds; a set needing more of any type gets a dedicated pool.
constexpr std::uint32_t ChainPoolSets = 1024;
constexpr std::array<VkDescriptorPoolSize, 4> ChainPoolSizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 512}}};
constexpr std::array<VkDescriptorPoolSize, 4> ChainPoolCreateSizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 + 2}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 512}}};
}

DescriptorCache::SetAllocation DescriptorCache::Allocate(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes) {
    bool fits = true;
    for (const auto& size : sizes) {
        Require(size.descriptorCount <= SetDescriptorLimit(context.limits, size.type), "descriptor set exceeds the device's per-set descriptor limit");
        const auto capacity = std::find_if(ChainPoolSizes.begin(), ChainPoolSizes.end(), [&](const auto& item) { return item.type == size.type; });
        if (capacity == ChainPoolSizes.end() || size.descriptorCount > capacity->descriptorCount) fits = false;
    }
    std::lock_guard lock(mutex);
    const auto allocate = context.Resolved(&DeviceFunctions::allocateDescriptorSets, "vkAllocateDescriptorSets");
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &layout;
    if (!fits) {
        dedicated.reserve(dedicated.size() + 1);
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        poolInfo.pPoolSizes = sizes.data();
        VkDescriptorPool pool = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
        allocation.descriptorPool = pool;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const auto result = allocate(context.device, &allocation, &set);
        if (result != VK_SUCCESS) {
            destroyPool(context.device, pool, nullptr);
            Check(result, "vkAllocateDescriptorSets");
        }
        dedicated.push_back(pool);
        ++stats.dedicatedPools;
        ++stats.sets;
        return {set, pool};
    }
    // Newest pool first: it has the most room; a full or fragmented pool is left for its sets to
    // drain and tried again later.
    for (auto it = pools.rbegin(); it != pools.rend(); ++it) {
        allocation.descriptorPool = *it;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const auto result = allocate(context.device, &allocation, &set);
        if (result == VK_SUCCESS) {
            ++stats.sets;
            return {set, *it};
        }
        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL) Check(result, "vkAllocateDescriptorSets");
    }
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = ChainPoolSets;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(ChainPoolCreateSizes.size());
    poolInfo.pPoolSizes = ChainPoolCreateSizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
    pools.push_back(pool);
    ++stats.pools;
    allocation.descriptorPool = pool;
    VkDescriptorSet set = VK_NULL_HANDLE;
    Check(allocate(context.device, &allocation, &set), "vkAllocateDescriptorSets");
    ++stats.sets;
    return {set, pool};
}

DescriptorCache::SetAllocation DescriptorCache::AllocateRewritten(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes) {
    {
        std::lock_guard lock(mutex);
        if (const auto found = recycled.find(layout); found != recycled.end() && !found->second.empty()) {
            const auto allocation = found->second.back();
            found->second.pop_back();
            return allocation;
        }
    }
    return Allocate(layout, sizes);
}

void DescriptorCache::Recycle(VkDescriptorSetLayout layout, const SetAllocation& allocation) noexcept {
    if (allocation.set == VK_NULL_HANDLE || allocation.pool == VK_NULL_HANDLE) return;
    std::lock_guard lock(mutex);
    try {
        auto& list = recycled[layout];
        if (list.size() < 4096) {
            list.push_back(allocation);
            return;
        }
    } catch (...) {
    }
    static_cast<void>(freeSets(context.device, allocation.pool, 1, &allocation.set));
}

void DescriptorCache::Free(const SetAllocation& allocation) noexcept {
    if (allocation.set == VK_NULL_HANDLE || allocation.pool == VK_NULL_HANDLE) return;
    std::lock_guard lock(mutex);
    if (const auto found = std::find(dedicated.begin(), dedicated.end(), allocation.pool); found != dedicated.end()) {
        dedicated.erase(found);
        destroyPool(context.device, allocation.pool, nullptr);
        return;
    }
    static_cast<void>(freeSets(context.device, allocation.pool, 1, &allocation.set));
}

DescriptorCache::Stats DescriptorCache::Counters() const {
    std::lock_guard lock(mutex);
    return stats;
}

std::size_t ShaderResources::addGuestBuffer(std::span<const std::uint32_t> words, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes, bool written, bool atomic, bool read, const std::vector<ShaderRecompiler::BufferStoreCover>* covers) {
    Require(words.size() == 4, "buffer descriptor must contain four DWORDs");
    Require((words[1] & 0x40000000u) == 0, "buffer descriptor has reserved bits set");
    const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
    Require(descriptor.Type() == 0u, "buffer descriptor uses an unsupported type");
    const auto address = descriptor.Base48();
    const auto byteSize = descriptor.GetSize();
    if (byteSize == 0 || address == 0) {
        allocations.push_back({0, EmptyBufferBytes, false, nullptr, ShaderRecompiler::DescriptorRole::GuestBuffers, false});
        return allocations.size() - 1;
    }
    Require(byteSize <= context.limits.maxStorageBufferRange, "shader buffer exceeds descriptor range limit");
    Require(byteSize <= std::numeric_limits<std::size_t>::max(), "shader buffer size exceeds host address space");
    const auto size = static_cast<std::size_t>(byteSize);
    Require(target == nullptr || !overlap(address, size, target->address, target->bytes), "shader buffer aliases the render target");
    // APS5_ALL_BUFFERS_WRITTEN=1: every element is noted as written, as before bufferWritten existed.
    static const bool allWritten = std::getenv("APS5_ALL_BUFFERS_WRITTEN") != nullptr;
    Require(!written || !overlap(address, size, indexAddress, indexBytes), "writable shader buffer aliases the index buffer");
    written = written || allWritten;
    if (written) {
        // What the shader does with the element, for the write-back elision (StorageUsageElide):
        // only a written element that is never read and has store covers can matter.
        BufferElementUsage usage;
        usage.read = read || atomic;
        if (!usage.read && covers != nullptr && !covers->empty()) {
            std::copy(words.begin(), words.end(), usage.words.begin());
            usage.covers = *covers;
        }
        guestMemory.AddWritable(address, size, atomic, &usage);
    } else {
        guestMemory.AddReadable(address, size);
        ++readOnlyBuffers;
    }
    if (BuildProfiled()) {
        auto& counters = BufferWrites();
        counters.elements.fetch_add(1, std::memory_order_relaxed);
        if (!written) counters.readOnly.fetch_add(1, std::memory_order_relaxed);
    }
    allocations.push_back({address, size, true, nullptr, ShaderRecompiler::DescriptorRole::ShaderData, written});
    return allocations.size() - 1;
}

std::string ShaderResources::Describe() const {
    const auto sample = [](std::uint64_t address, std::uint64_t bytes) {
        // Heaps bound whole can include uncommitted pages; those ranges are not sampled.
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes))) return -1.0;
        std::size_t nonzero = 0;
        std::size_t samples = 0;
        const auto* data = reinterpret_cast<const std::uint32_t*>(address);
        const auto words = bytes / 4;
        const auto step = std::max<std::uint64_t>(1, words / 4096);
        for (std::uint64_t i = 0; i < words; i += step, ++samples) nonzero += data[i] != 0;
        return samples == 0 ? 0.0 : static_cast<double>(nonzero) / samples;
    };
    std::string text;
    char line[160];
    for (const auto& range : describedRanges) {
        std::snprintf(line, sizeof(line), " %s 0x%llx+0x%llx(%ux%u f%u t%d) nz=%.2f", range.kind, static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(range.bytes), range.width, range.height, range.format, range.tileMode, sample(range.address, range.bytes));
        text += line;
        if (range.dccAddress != 0) {
            const auto keys = ReadDccKeys(range.dccAddress, range.bytes);
            std::snprintf(line, sizeof(line), " dcc=%s@0x%llx", DccKeysName(keys), static_cast<unsigned long long>(range.dccAddress));
            text += line;
            if (keys == DccKeys::Mixed) {
                // Where the keys change: first key, how many leading keys match it, and the next key.
                const auto* bytes = reinterpret_cast<const std::uint8_t*>(range.dccAddress);
                const auto count = static_cast<std::size_t>(range.bytes / 256u);
                std::size_t run = 1;
                while (run < count && bytes[run] == bytes[0]) ++run;
                std::snprintf(line, sizeof(line), "(%02x x%zu then %02x of %zu)", bytes[0], run, run < count ? bytes[run] : 0u, count);
                text += line;
            }
        }
    }
    // APS5_TRACE_DISPATCH_IO=2 also lists the words of small buffers (constants).
    static const bool words = [] { const char* value = std::getenv("APS5_TRACE_DISPATCH_IO"); return value != nullptr && value[0] == '2'; }();
    const auto appendWords = [&](const std::uint32_t* data, std::size_t bytes) {
        if (!words || bytes > 0x200) return;
        text += " [";
        for (std::size_t i = 0; i < bytes / 4; ++i) {
            char word[12];
            std::snprintf(word, sizeof(word), "%s%08x", i == 0 ? "" : " ", data[i]);
            text += word;
        }
        text += "]";
    };
    for (const auto& allocation : allocations) {
        if (allocation.guest) {
            std::snprintf(line, sizeof(line), " buffer%s 0x%llx+0x%zx nz=%.2f", allocation.written ? "" : "(ro)", static_cast<unsigned long long>(allocation.address), allocation.size, sample(allocation.address, allocation.size));
            text += line;
            if (GuestMemory::Accessible(reinterpret_cast<const void*>(allocation.address), allocation.size)) appendWords(reinterpret_cast<const std::uint32_t*>(allocation.address), allocation.size);
        } else if (allocation.buffer) {
            const auto bytes = allocation.buffer->Bytes();
            std::snprintf(line, sizeof(line), " data+0x%zx nz=%.2f", allocation.size, sample(reinterpret_cast<std::uint64_t>(bytes.data()), allocation.size));
            text += line;
            appendWords(reinterpret_cast<const std::uint32_t*>(bytes.data()), allocation.size);
        }
    }
    return text;
}

std::size_t ShaderResources::addDataBuffer(std::span<const std::uint32_t> words) {
    const auto size = words.size() * sizeof(std::uint32_t);
    Require(size <= context.limits.maxStorageBufferRange, "shader data buffer exceeds descriptor range limit");
    const bool refreshable = TemplateDataRefresh() && size <= MaxRefreshBytes;
    auto buffer = MakeShaderDataBuffer(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | (refreshable ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0u));
    std::memcpy(buffer->Bytes().data(), words.data(), size);
    Allocation allocation{0, size, false, std::move(buffer)};
    if (refreshable) allocation.dataWords.assign(words.begin(), words.end());
    allocations.push_back(std::move(allocation));
    mixDataWords(dataWordsHash, allocations.back().dataWords);
    return allocations.size() - 1;
}

void ShaderResources::rehashDataWords() {
    dataWordsHash = FnvOffset;
    // Data buffers are the non-guest allocations with a buffer (an address-role allocation has
    // none), appended in binding order by buildPrepare: the order DataWordsHash(shader) hashes.
    for (const auto& allocation : allocations) {
        if (!allocation.guest && allocation.buffer != nullptr) mixDataWords(dataWordsHash, allocation.dataWords);
    }
}

std::uint64_t ShaderResources::DataWordsHash(const CompiledShader& shader) {
    Require(shader.program != nullptr, "missing compiled shader");
    std::uint64_t hash = FnvOffset;
    for (const auto& binding : shader.program->bindings) {
        if (DataRole(binding.role)) mixDataWords(hash, binding.guestDescriptor);
    }
    return hash;
}

bool ShaderResources::DataWordsDiffer(const CompiledShader& shader) const {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    if (program.bindings.size() != bindings.size()) return true;
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        const auto& binding = program.bindings[index];
        if (!DataRole(binding.role)) continue;
        if (bindings[index].allocations.size() != 1) return true;
        const auto& allocation = allocations[bindings[index].allocations.front()];
        if (allocation.dataWords.size() != binding.guestDescriptor.size() || !std::equal(allocation.dataWords.begin(), allocation.dataWords.end(), binding.guestDescriptor.begin())) return true;
    }
    return false;
}

bool ShaderResources::RefreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder) {
    Require(shader.program != nullptr, "missing compiled shader");
    Require(!refreshResourceKey.empty() && MatchesContentKey(shader, refreshResourceKey), "runtime data refresh cannot replace bound resources");
    return refreshData(commands, shader, recorder);
}

bool ShaderResources::refreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    Require(program.bindings.size() == bindings.size(), "template bindings disagree with the shader");
    bool recorded = false;
    // One [gputime] range of class TemplateDataRefresh around the updates, begun at the first one
    // (a refresh that finds every word equal records nothing and times nothing).
    auto timing = Recorder::NoTiming;
    std::uint64_t refreshedBytes = 0;
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        const auto& binding = program.bindings[index];
        if (!DataRole(binding.role)) continue;
        Require(bindings[index].allocations.size() == 1, "data binding without its buffer");
        auto& allocation = allocations[bindings[index].allocations.front()];
        const auto size = binding.guestDescriptor.size() * sizeof(std::uint32_t);
        Require(allocation.buffer != nullptr && !allocation.guest && allocation.size == size && size <= MaxRefreshBytes, "template data buffer cannot take the dispatch's words");
        if (allocation.dataWords.size() == binding.guestDescriptor.size() && std::equal(allocation.dataWords.begin(), allocation.dataWords.end(), binding.guestDescriptor.begin())) continue;
        if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) ValidateRuntimeResources(shader, binding.guestDescriptor);
        if (recorder != nullptr && timing == Recorder::NoTiming) timing = recorder->BeginGpuTiming(Recorder::CommandClass::TemplateDataRefresh);
        writeDataWords(commands, bindings[index].allocations.front(), binding.guestDescriptor);
        allocation.dataWords.assign(binding.guestDescriptor.begin(), binding.guestDescriptor.end());
        refreshedBytes += size;
        recorded = true;
    }
    if (timing != Recorder::NoTiming) recorder->EndGpuTiming(timing, refreshedBytes);
    if (recorded) rehashDataWords();
    return recorded;
}

void ShaderResources::writeDataWords(VkCommandBuffer commands, std::size_t allocation, std::span<const std::uint32_t> words) const {
    const auto& buffer = *allocations[allocation].buffer;
    const auto size = words.size() * sizeof(std::uint32_t);
    if (std::none_of(dataPatches.begin(), dataPatches.end(), [&](const DataPatch& patch) { return patch.allocation == allocation; })) {
        context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, buffer.Handle(), 0, size, words.data());
        return;
    }
    std::vector<std::uint32_t> patched(words.begin(), words.end());
    auto* bytes = reinterpret_cast<std::byte*>(patched.data());
    for (const auto& patch : dataPatches) {
        if (patch.allocation == allocation && patch.byte < size) bytes[patch.byte] = static_cast<std::byte>(patch.adjustment);
    }
    context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, buffer.Handle(), 0, size, patched.data());
}

void ShaderResources::PrecollectSurfaces() const {
    for (const auto& range : describedRanges) GuestMemory::CollectWrites(range.address, static_cast<std::size_t>(range.bytes));
}

void ShaderResources::addImageBinding(const ShaderRecompiler::DescriptorBinding& binding, VkShaderStageFlags flags) {
    Require(binding.count != 0, "empty descriptor binding");
    const auto heap = static_cast<ShaderRecompiler::RuntimeAbi::Binding>(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count));
    const auto capacity = binding.count;
    Require(binding.count <= ShaderRecompiler::RuntimeAbi::HeapCapacity(heap), "typed descriptor heap capacity exceeded");
    const auto localBinding = static_cast<std::uint32_t>(heap);
    const bool sampledHeap = localBinding >= ShaderRecompiler::RuntimeAbi::FirstImageBinding && localBinding < ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
    const bool storageHeap = localBinding >= ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding && localBinding < static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers);
    Require((binding.kind == ShaderRecompiler::DescriptorKind::SampledImage && sampledHeap) || (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage && storageHeap) || (binding.kind == ShaderRecompiler::DescriptorKind::Sampler && heap == ShaderRecompiler::RuntimeAbi::Binding::Samplers), "resource class disagrees with typed descriptor heap");
    Require(binding.kind == ShaderRecompiler::DescriptorKind::Sampler || context.nullDescriptors, "typed image heaps require enabled nullDescriptor");
    if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
        // Storage images are guest textures the shader writes (looked up in stage B).
        Require(binding.role == ShaderRecompiler::DescriptorRole::GuestImages, "storage image binding has a non-image role");
        Require(binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 8u, "guest storage image descriptors must contain 8 dwords each");
        Require(context.detiler != nullptr, "device texture detiler is unavailable");
        Require(capacity <= context.limits.maxPerStageDescriptorStorageImages, "shader storage-image descriptors exceed per-stage limits");
        bindings.push_back({{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, capacity, flags, nullptr}, {}, {}});
        plannedStorageImages += binding.count;
        deferredImages.push_back({&binding, bindings.size() - 1});
        return;
    }
    const bool sampledImage = binding.kind == ShaderRecompiler::DescriptorKind::SampledImage;
    const bool samplerKind = binding.kind == ShaderRecompiler::DescriptorKind::Sampler;
    if (!(sampledImage || samplerKind)) Require(false, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role " + roleName(binding.role));
    Require((sampledImage && binding.role == ShaderRecompiler::DescriptorRole::GuestImages) || (samplerKind && binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers), "guest image descriptor role disagrees with its kind");
    Require(binding.guestDescriptor.size() % binding.count == 0, "guest image descriptor size is not a multiple of the binding count");
    const auto elementWords = binding.guestDescriptor.size() / binding.count;

    Binding item{{binding.binding, sampledImage ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER, capacity, flags, nullptr}, {}, {}};

    if (sampledImage) {
        Require(elementWords == 8, "guest texture descriptor must contain 8 dwords");
        Require(binding.imageShape.has_value(), "guest image binding is missing an image shape");
        Require(context.detiler != nullptr, "device texture detiler is unavailable");
        Require(context.textureCache != nullptr, "device texture cache is unavailable");
        Require(capacity <= context.limits.maxPerStageDescriptorSampledImages, "shader sampled-image descriptors exceed per-stage limits");
        plannedSampledImages += binding.count;
        Require(plannedSampledImages <= context.limits.maxDescriptorSetSampledImages, "pipeline sampled-image descriptors exceed device limits");
    } else {
        Require(elementWords == 4, "guest sampler descriptor must contain 4 dwords");
        Require(capacity <= context.limits.maxPerStageDescriptorSamplers, "shader sampler descriptors exceed per-stage limits");
        Require(binding.samplerDepthCompare.size() == binding.count, "guest sampler binding is missing depth comparison metadata");
        Require(binding.samplerUnnormalized.empty() || binding.samplerUnnormalized.size() == binding.count, "guest sampler binding has unnormalized coordinate metadata of another size");
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            const bool compareEnable = binding.samplerDepthCompare.at(element);
            const bool unnormalized = element < binding.samplerUnnormalized.size() && binding.samplerUnnormalized[element];
            static const bool noSamplerCache = std::getenv("APS5_NO_SAMPLER_CACHE") != nullptr;
            if (context.samplerCache != nullptr && !noSamplerCache) {
                samplers.push_back(context.samplerCache->Get(context, words, compareEnable, unnormalized));
            } else {
                auto resource = DecodeSamplerResource(words, unnormalized);
                resource.compareEnable = compareEnable;
                samplers.push_back(std::make_shared<Sampler>(context, resource));
            }
            item.imageAllocations.push_back(samplers.size() - 1);
        }
        Require(samplers.size() <= context.limits.maxDescriptorSetSamplers, "pipeline sampler descriptors exceed device limits");
    }

    bindings.push_back(std::move(item));
    if (sampledImage) deferredImages.push_back({&binding, bindings.size() - 1});
}

namespace {

// Calls `visit(binding, element, words)` for every sampled and storage image element of a compiled
// shader's bindings, in plan order (the order addImageBinding and resolveImageBinding walk).
template <typename Visit>
void forEachImageElement(const ShaderRecompiler::RecompileResult& program, Visit&& visit) {
    for (const auto& binding : program.bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.count == 0) continue;
        if (binding.kind != ShaderRecompiler::DescriptorKind::SampledImage && binding.kind != ShaderRecompiler::DescriptorKind::StorageImage) continue;
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) visit(binding, element, std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords));
    }
}

}

bool ShaderResources::precollectImages() {
    // The image lookups of stage B (cachedTexture, StorageTexture::Refresh) each start with a
    // GuestMemory::CollectWrites of their surface, the GetWriteWatch walk that is most of the
    // lookups' time under the lock. The walk is memoized per thread epoch, and the worker thread
    // passes no ordering point between the stages (see GuestMemory::BumpCollectEpoch), so walking
    // every surface here makes stage B's collects memo hits: the walk leaves the lock, the lookups'
    // checks stay where they were. A memo miss (the ring overflowed, an uncommitted page) just walks
    // under the lock as before; a descriptor stage B rejects is left to it. APS5_NO_PRECOLLECT=1
    // disables the pass.
    // The pass also leaves one ImageRecord per element: the decode and the surface description are
    // made once for the build, and a sampled element's cache entry is taken here (a hash lookup
    // under the cache's own mutex) so that stage B, under the device lock, runs only the
    // fastRevalidate predicate on it (see fastTexture). The collect comes before the entry is read,
    // as the predicate's rule demands (the generation the entry moves to must predate the checks).
    static const bool disabled = std::getenv("APS5_NO_PRECOLLECT") != nullptr;
    static const bool noRecords = std::getenv("APS5_NO_STAGE_A_IMAGES") != nullptr;
    if (disabled || deferredImages.empty()) return false;
    imageRecords.clear();
    nextImageRecord = 0;
    auto& counters = TextureCounts();
    for (const auto& deferred : deferredImages) {
        const auto& binding = *deferred.binding;
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            ImageRecord record;
            record.sampled = binding.kind == ShaderRecompiler::DescriptorKind::SampledImage;
            if (words[0] != 0u || (words[1] & 0xffu) != 0u) {
                record.resource = DecodeTextureResource(words);
                record.guestBytes = DescribeSurface(record.resource).guestBytes;
                record.generation = GuestMemory::CollectWrites(record.resource.baseAddress, static_cast<std::size_t>(record.guestBytes));
                record.decoded = true;
                if (record.sampled && !noRecords && words.size() == 8 && (binding.imageDepthCompare.empty() || !binding.imageDepthCompare.at(element))) {
                    std::copy(words.begin(), words.end(), record.words.begin());
                    record.components = ViewComponents(record.resource);
                    record.keys = TextureClearKeys(record.resource, record.guestBytes);
                    auto& cache = Textures();
                    std::lock_guard lock(cache.mutex);
                    if (const auto it = findTexture(cache, MakeTextureKey(context.device, words, record.components)); it != cache.entries.end()) {
                        record.texture = it->texture;
                        record.source = it->source;
                        record.entryKeys = it->keys;
                        record.entryGeneration = it->generation;
                        counters.records.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            imageRecords.push_back(std::move(record));
        }
    }
    return true;
}

std::shared_ptr<Texture> ShaderResources::fastTexture(const ImageRecord& record) {
    const RefreshSiteScope site("sampled-fast");
    if (record.texture == nullptr || record.texture->RefreshedPerUse()) return nullptr;
    struct Outcome {
        bool profile;
        std::chrono::steady_clock::time_point start;
        bool hit = false;
        ~Outcome() {
            if (profile) LookupOutcomes::Add(hit ? LookupOutcomes::SampledFast : LookupOutcomes::SampledFastMiss, start);
        }
    } outcome{LookupOutcomes::Profiled(), LookupOutcomes::Profiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
    // Exactly the "unchanged" branches of cachedTexture, from write stamps, the pending-results
    // registry and the DCC keys, the way fastRevalidate proves a built object current: the collect
    // first (a memo hit: stage A walked the range), then no other image may have results pending
    // over the memory (a lookup would flush them into it, or view them instead), a storage-sourced
    // view needs its image to be the cache's image of the surface and current with guest memory,
    // a snapshot needs the memory unchanged since its content matched, and the keys must be what
    // the content was made under whenever the surface has any (re-read under the lock: a flush
    // between the stages marks them uncompressed).
    const auto address = record.resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(record.guestBytes);
    if (record.source != nullptr && DepthSurfaceAt(address)) return nullptr;
    if (GuestMemory::CollectWrites(address, bytes) == 0) return nullptr;
    if (PendingStorageOverlaps(address, bytes, record.source.get())) return nullptr;
    auto keys = record.keys;
    if (record.resource.dccAddress != 0) {
        const auto scanStart = outcome.profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // Through the proof of the image whose keys they are (the view's source when the surface
        // is the image's, else the texture's): a scan only when the key range was stamped.
        keys = ProvedClearKeys(record.resource, record.guestBytes, SameKeySurface(record.source.get(), record.resource, record.guestBytes) ? record.source->KeyProof() : record.texture->KeyProof());
        if (outcome.profile) LookupOutcomes::Add(LookupOutcomes::DccScan, scanStart);
    }
    // The keys the entry was made under (a view of a pending render target was made under its clear
    // keys and stays valid while they are unchanged); a change is the full lookup's to judge.
    if (keys != record.entryKeys) return nullptr;
    if (record.source != nullptr) {
        if (!StorageImageCached(context, record.source.get()) || !GuestMemory::UnchangedSince(address, bytes, record.source->Generation())) return nullptr;
        if (record.resource.dccAddress != 0 && IsDccClear(record.source->FilledKeys())) return nullptr;
        // Under fast-clear keys the view holds only while its image's results are still pending over
        // the surface; flushed, the clear the image cannot see makes the lookup take a snapshot.
        if (keys != DccKeys::Uncompressed && StorageTexture::FindPending(address, record.guestBytes) != record.source) return nullptr;
    } else if (keys == DccKeys::Uncompressed && !GuestMemory::UnchangedSince(address, bytes, record.entryGeneration)) {
        return nullptr;
    }
    // The entry must still be the cache's, holding this object (an evicted object is not reused: the
    // next lookup would make another), and it takes the stage-A generation like a hit would.
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    const auto it = findTexture(cache, MakeTextureKey(context.device, record.words, record.components));
    if (it == cache.entries.end() || it->texture != record.texture) return nullptr;
    if (it->source == nullptr) it->generation = record.generation;
    touchTexture(cache, it);
    logLookup({record.texture.get(), record.resource, record.guestBytes, keys, record.source != nullptr ? 0 : record.generation, record.source.get()});
    outcome.hit = true;
    return record.texture;
}

void ShaderResources::resolveImageBinding(const ShaderRecompiler::DescriptorBinding& binding, Binding& item, std::span<const std::shared_ptr<Sampler>> shaderSamplers) {
    auto& counters = TextureCounts();
    // The element's stage-A record, when the pass ran (records follow the plan order exactly).
    const auto nextRecord = [&]() -> const ImageRecord* { return nextImageRecord < imageRecords.size() ? &imageRecords[nextImageRecord++] : nullptr; };
    if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
        Require(binding.imageSamplers.size() == binding.count, "guest sampled image binding is missing its image-sampler pairs");
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            const auto* record = nextRecord();
            if (words[0] == 0u && (words[1] & 0xffu) == 0u) {
                item.imageAllocations.push_back(std::numeric_limits<std::size_t>::max());
                --plannedSampledImages;
                continue;
            }
            const auto resource = record != nullptr && record->decoded ? record->resource : DecodeTextureResource(words);
            const bool firstLayer = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
            if (!firstLayer && !MatchesGuestDimension(*binding.imageShape, resource.dimension)) throw std::runtime_error("AGC graphics: guest texture dimension disagrees with the shader's declared image shape (shape " + std::to_string(static_cast<int>(*binding.imageShape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
            const VkComponentMapping components = ViewComponents(resource);
            const auto guestBytes = record != nullptr && record->decoded ? record->guestBytes : DescribeSurface(resource).guestBytes;
            std::shared_ptr<Texture> texture;
            if (record != nullptr && record->texture != nullptr) {
                texture = fastTexture(*record);
                (texture != nullptr ? counters.fastHits : counters.fastMisses).fetch_add(1, std::memory_order_relaxed);
            }
            const bool fast = texture != nullptr;
            if (texture == nullptr) texture = cachedTexture(context, words, resource, components, guestBytes, !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element));
            const auto traceSampled = DebugValue(DebugKey::TraceSampled);
            static std::atomic<int> tracedSampled{0};
            if (traceSampled != 0 && resource.baseAddress == traceSampled && tracedSampled.fetch_add(1) < 400) {
                std::fprintf(stderr, "[trace-sampled] 0x%llx fast %d view %d null %d keys %s pending %d format %u dcc 0x%llx texture %p\n", static_cast<unsigned long long>(resource.baseAddress), fast ? 1 : 0, texture->ViewsStorageImage() ? 1 : 0, texture->Null() ? 1 : 0, DccKeysName(TextureClearKeys(resource, guestBytes)), StorageTexture::FindPending(resource.baseAddress, guestBytes) != nullptr ? 1 : 0, resource.format, static_cast<unsigned long long>(resource.dccAddress), static_cast<const void*>(texture.get()));
            }
            bool baseLevel = false;
            if (element < binding.imageUnnormalized.size() && binding.imageUnnormalized[element]) {
                const auto range = texture->SampledViewRange(firstLayer);
                const bool supported = range.layers == 1u && EffectiveMinLod(resource) == 0.0f;
                if (!supported || (range.type != VK_IMAGE_VIEW_TYPE_1D && range.type != VK_IMAGE_VIEW_TYPE_2D)) throw std::runtime_error("AGC graphics: guest texture sampled with unnormalized coordinates is not a single-layer 1D or 2D view without a minimum LOD clamp, which is not implemented (base level " + std::to_string(resource.baseLevel) + ", levels " + std::to_string(range.levels) + ", layers " + std::to_string(range.layers) + ", view type " + std::to_string(static_cast<int>(range.type)) + ")");
                baseLevel = range.levels > 1u;
            }
            RequireFilterMinmax(context, texture->ViewFormat(), binding.imageSamplers[element], shaderSamplers);
            textures.push_back(std::move(texture));
            textureFirstLayer.push_back(firstLayer);
            textureBaseLevel.push_back(baseLevel);
            textureWords.emplace_back();
            std::copy_n(words.begin(), 8, textureWords.back().begin());
            textureDepthCompare.push_back(!binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element));
            textureUnnormalized.push_back(element < binding.imageUnnormalized.size() && binding.imageUnnormalized[element]);
            describedRanges.push_back({"texture", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress});
            item.imageAllocations.push_back(textures.size() - 1);
        }
        // The line is due even when every element took the fast path (cachedTexture reports too).
        reportTextureCounters();
        return;
    }
    // Consecutive identical storage descriptors address successive mips of one texture (dynamic-mip
    // storage writes).
    std::uint32_t mipOffset = 0;
    for (std::uint32_t element = 0; element < binding.count; ++element) {
        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
        const bool sameAsPrevious = SameAsPreviousStorageElement(binding, element);
        if (sameAsPrevious) ++mipOffset;
        else mipOffset = 0;
        const auto* record = nextRecord();
        if (words[0] == 0u && (words[1] & 0xffu) == 0u) {
            item.imageAllocations.push_back(std::numeric_limits<std::size_t>::max());
            --plannedStorageImages;
            continue;
        }
        const auto resource = record != nullptr && record->decoded ? record->resource : DecodeTextureResource(words);
        const bool firstLayer = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
        if (binding.imageShape.has_value() && !firstLayer && !MatchesGuestDimension(*binding.imageShape, resource.dimension)) throw std::runtime_error("AGC graphics: guest storage texture dimension disagrees with the shader's declared image shape (shape " + std::to_string(static_cast<int>(*binding.imageShape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
        const auto mip = std::min(resource.baseLevel + mipOffset, resource.mipCount - 1u);
        Require(resource.minLod <= mip * 256u, "guest storage texture descriptor clamps its minimum LOD above the level it addresses, which is not implemented");
        const auto guestBytes = record != nullptr && record->decoded ? record->guestBytes : DescribeSurface(resource).guestBytes;
        // The same surface as the previous element: its image was just looked up and refreshed.
        if (sameAsPrevious && StorageDedupeEnabled()) storageTextures.push_back(storageTextures.back());
        else storageTextures.push_back(cachedStorageTexture(context, words, resource, mip, guestBytes));
        if (const auto traceStorage = DebugValue(DebugKey::TraceStorage); traceStorage != 0 && resource.baseAddress == traceStorage) {
            const auto packet = GuestMemory::CurrentPacket();
            std::fprintf(stderr, "[trace-storage] 0x%llx binding %u element %u written %d atomic %d %ux%u mip %u format %u (packet 0x%x queue 0x%x)\n", static_cast<unsigned long long>(resource.baseAddress), binding.binding, element, element >= binding.imageWritten.size() || binding.imageWritten[element] ? 1 : 0, element < binding.imageAtomic.size() && binding.imageAtomic[element] ? 1 : 0, resource.width, resource.height, mip, resource.format, packet.opcode, packet.queue);
        }
        storageMips.push_back(mip);
        storageKeys.push_back(resource.dccAddress);
        storageFirstLayer.push_back(firstLayer || (resource.dimension == TextureDimension::k2D && storageTextures.back()->Descriptor().dimension == TextureDimension::k2DArray));
        storageLayers.push_back(resource.baseArray - std::min(resource.baseArray, storageTextures.back()->Descriptor().baseArray));
        // Images the shader only reads have nothing to store back.
        storageWritten.push_back(element >= binding.imageWritten.size() || binding.imageWritten[element]);
        storageAtomic.push_back(element < binding.imageAtomic.size() && binding.imageAtomic[element]);
        storageAtomic64.push_back(element < binding.imageAtomic64.size() && binding.imageAtomic64[element]);
        describedRanges.push_back({"storage", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress});
        item.imageAllocations.push_back(storageTextures.size() - 1);
    }
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ShaderResources::PresyncSurfaces() const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> surfaces;
    // A surface's lookup reads guest memory on the CPU unless it is served GPU-direct from a host
    // import: a storage image of imported memory uploads and refreshes from the import, and a
    // sampled texture of one views that image (SampledFromStorageEligible), so neither waits for
    // the producer.
    const auto consider = [&](std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes, bool sampled) {
        const bool gpuDirect = sampled ? SampledFromStorageEligible(context, format, address, guestBytes) : HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
        if (!gpuDirect) surfaces.emplace_back(address, guestBytes);
    };
    if (completed) {
        // A cached object (the resource cache): Revalidate repeats the lookups of the objects the
        // build made, which never change afterwards (a lookup returning another object fails the
        // revalidation instead), so the surfaces come from the build's own record of them. A
        // sampled texture viewing a storage image refreshes like that image; a snapshot texture
        // over an eligible surface is replaced by a view without reading. Storage images are
        // checked on their own surface (a mip chain can exceed the element's descriptor). Read
        // without the device lock: these members are fixed once the build completed.
        std::size_t textureIndex = 0;
        for (const auto& range : describedRanges) {
            if (std::strcmp(range.kind, "texture") == 0 && textureIndex < textures.size()) {
                const auto& texture = textures[textureIndex++];
                if (range.bytes == 0) continue;
                if (texture->ViewsStorageImage()) consider(range.format, range.address, range.bytes, false);
                else consider(range.format, range.address, range.bytes, true);
            }
        }
        for (std::size_t index = 0; index < storageTextures.size(); ++index) {
            if (index != 0 && storageTextures[index] == storageTextures[index - 1]) continue;
            const auto& image = *storageTextures[index];
            consider(image.Descriptor().format, image.Descriptor().baseAddress, image.GuestBytes(), false);
        }
        return surfaces;
    }
    if (!imageRecords.empty()) {
        for (const auto& record : imageRecords) {
            if (record.decoded) consider(record.resource.format, record.resource.baseAddress, record.guestBytes, record.sampled);
        }
        return surfaces;
    }
    // Stage A has not run (an address-based build, or the pass is off): decoded from the shader.
    if (deferredCompute.program == nullptr) return surfaces;
    forEachImageElement(*deferredCompute.program, [&](const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t, std::span<const std::uint32_t> words) {
        if (words[0] == 0u && (words[1] & 0xffu) == 0u) return;
        const auto resource = DecodeTextureResource(words);
        consider(resource.format, resource.baseAddress, DescribeSurface(resource).guestBytes, binding.kind == ShaderRecompiler::DescriptorKind::SampledImage);
    });
    return surfaces;
}

ShaderResources::~ShaderResources() {
    release();
}

void ShaderResources::release() noexcept {
    // A set from the cache's pool chain goes back to it; a dedicated pool dies with its set.
    if (cachePool && context.descriptorCache != nullptr) {
        if (recycledSet) context.descriptorCache->Recycle(_layout, {_set, cachePool});
        else context.descriptorCache->Free({_set, cachePool});
    }
    if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    if (_layout && ownsLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, _layout, nullptr);
    cachePool = VK_NULL_HANDLE;
    pool = VK_NULL_HANDLE;
    _set = VK_NULL_HANDLE;
    _layout = VK_NULL_HANDLE;
}

VkDescriptorSetLayout ShaderResources::Layout() const {
    return _layout;
}

ShaderResources::DrawBindings::~DrawBindings() {
    if (cache == nullptr || allocation.set == VK_NULL_HANDLE) return;
    if (layout != VK_NULL_HANDLE) cache->Recycle(layout, allocation);
    else cache->Free(allocation);
}

std::optional<std::vector<ShaderResources::MovedBuffer>> ShaderResources::MovedReadOnlyBuffers(std::span<const CompiledShader> shaders, Recorder& recorder) const {
    if (_set == VK_NULL_HANDLE || usesBda) return std::vector<MovedBuffer>{};
    return movedBuffers(shaders, &recorder, false);
}

std::optional<std::vector<ShaderResources::MovedBuffer>> ShaderResources::movedBuffers(std::span<const CompiledShader> shaders, Recorder* recorder, bool dataOnly) const {
    std::vector<MovedBuffer> moved;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) return std::nullopt;
        for (const auto& binding : shader.program->bindings) {
            if (DataRole(binding.role)) {
                const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
                if (kept == bindings.end() || kept->allocations.size() != 1) return std::nullopt;
                const auto index = kept->allocations.front();
                const auto& item = allocations[index];
                if (item.guest || item.buffer == nullptr || item.size != binding.guestDescriptor.size() * sizeof(std::uint32_t)) return std::nullopt;
                const bool patched = std::any_of(dataPatches.begin(), dataPatches.end(), [&](const DataPatch& patch) { return patch.allocation == index; });
                const bool same = !item.dataWords.empty() ? item.dataWords == binding.guestDescriptor : !patched && std::memcmp(item.buffer->Bytes().data(), binding.guestDescriptor.data(), item.size) == 0;
                if (same) continue;
                if (item.dataWords.empty() && patched) return std::nullopt;
                if (dataOnly && binding.role == ShaderRecompiler::DescriptorRole::ShaderData) ValidateRuntimeResources(shader, binding.guestDescriptor);
                moved.push_back({index, 0, item.size, binding.guestDescriptor});
                continue;
            }
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
            const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
            if (kept == bindings.end() || kept->allocations.size() != binding.count || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) return std::nullopt;
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
                const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
                const auto address = descriptor.Base48();
                const auto size = descriptor.GetSize();
                const auto index = kept->allocations[element];
                const auto& item = allocations[index];
                const bool empty = size == 0 || address == 0;
                if (!item.guest) {
                    if (empty) continue;
                    return std::nullopt;
                }
                // A streamed element (stream template, see StreamRing) stays in the moved list at its
                // own address: the template's copy of it is stale by design, so the hit copies the
                // current bytes through the same refusals as any moved read-only buffer. A dataOnly
                // use (which rebinds nothing) refuses it below.
                if (item.address == address && item.size == size && !item.streamed) continue;
                if (dataOnly) return std::nullopt;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                if (written || empty || item.written || size > context.limits.maxStorageBufferRange) return std::nullopt;
                const auto begin = address - item.adjustment;
                const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
                // Bounds a hit's per-draw copy when a later V# under the same key is large: the draw
                // builds as before instead.
                if (item.streamed && bytes > StreamLimit()) return std::nullopt;
                if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return std::nullopt;
                if (guestMemory.WritesOverlap(begin, bytes) || (recorder != nullptr && recorder->PendingWriteOverlaps(begin, bytes)) || PendingStorageOverlaps(begin, bytes, nullptr) || AnyShadowedOverlaps(begin, bytes)) return std::nullopt;
                // The build a stream hit replaces may copy the range on the GPU (a misaligned region
                // inside an import), after the queued DCC key and label stores over it land
                // (recordGpuCopies); the hit's CPU copy cannot wait for those, so it refuses them.
                if (item.streamed && recorder != nullptr && StreamStoreQueued(*recorder, begin, bytes)) return std::nullopt;
                moved.push_back({index, address, static_cast<std::size_t>(size)});
            }
        }
    }
    return moved;
}

constexpr std::size_t FreshCopyLimit = 4096;

namespace {

// Debug aid: APS5_VERIFY_STREAM_RING=1 compares every stream hit's copy with a read of the same
// range through the flush hook (what the build the hit replaces would have read) and aborts on a
// difference: a case the moved checks let through that the hook would have acted on.
bool VerifyStreamRing() {
    static const bool enabled = std::getenv("APS5_VERIFY_STREAM_RING") != nullptr;
    return enabled;
}

void VerifyStreamedCopy(std::uint64_t begin, std::span<const std::byte> copy) {
    std::vector<std::byte> hooked(copy.size());
    GuestMemory::Read(begin, hooked);
    if (std::memcmp(hooked.data(), copy.data(), copy.size()) != 0) {
        std::fprintf(stderr, "[reuse] APS5_VERIFY_STREAM_RING: a stream hit's copy of 0x%llx+0x%zx differs from the hooked read\n", static_cast<unsigned long long>(begin), copy.size());
        std::fflush(stderr);
        std::abort();
    }
    Reuses().verified.fetch_add(1, std::memory_order_relaxed);
}

}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::PrepareDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved) const {
    if (_set == VK_NULL_HANDLE || usesBda) return {};
    auto result = std::make_shared<DrawBindings>();
    thread_local std::vector<std::size_t>* selectedSlot = nullptr;
    auto& selected = ShaderRecompiler::ThreadOwned(selectedSlot);
    selected.clear();
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        const auto override = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return entry.allocation == index; });
        if (override != moved.end() && !override->words.empty()) {
            Recorder::SnapshotSlice slice{};
            static const bool movedRing = std::getenv("APS5_MOVED_RING") == nullptr || std::strcmp(std::getenv("APS5_MOVED_RING"), "0") != 0;
            if (movedRing && Recorder::SnapshotRingEnabled()) slice = recorder.AllocateDrawSnapshot(override->size);
            auto buffer = slice.buffer != nullptr ? std::move(slice.buffer) : std::make_shared<Buffer>(context, override->size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            const VkDeviceSize offset = buffer != nullptr && slice.offset != 0 ? slice.offset : 0;
            auto* bytes = buffer->Bytes().data() + offset;
            std::memcpy(bytes, override->words.data(), override->size);
            for (const auto& patch : dataPatches) {
                if (patch.allocation == index && patch.byte < override->size) bytes[patch.byte] = static_cast<std::byte>(patch.adjustment);
            }
            selected.push_back(index);
            result->snapshots.push_back({0, std::move(buffer), offset, override->size});
            continue;
        }
        std::uint64_t address = item.address;
        std::size_t size = item.size;
        if (override != moved.end()) {
            address = override->address;
            size = override->size;
        } else {
            if (!item.guest || item.written || guestMemory.WritesOverlap(item.address, item.size)) continue;
            const bool direct = guestMemory.ReadsInPlace(item.address, item.size);
            if (!direct || recorder.PendingWriteOverlaps(item.address, item.size)) continue;
        }
        const auto begin = address - item.adjustment;
        const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
        if (item.streamed && override != moved.end()) {
            // A stream hit: always a fresh copy, with no write tracking or snapshot reuse (the
            // range is usually outside the watched arena, where CollectWrites has no generation to
            // offer), taken here under the commit's device lock in stream order, where the build it
            // replaces copied the same range. movedBuffers refused every case the flush hook would
            // have acted on; the slice lives until the batch completed (recorder.Keep below).
            VkDeviceSize offset = 0;
            std::shared_ptr<Buffer> buffer;
            if (Recorder::SnapshotRingEnabled()) {
                auto slice = recorder.AllocateDrawSnapshot(bytes);
                buffer = std::move(slice.buffer);
                offset = slice.offset;
            }
            if (buffer == nullptr) {
                buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                offset = 0;
            }
            const auto copy = buffer->Bytes().subspan(static_cast<std::size_t>(offset), bytes);
            std::memcpy(copy.data(), reinterpret_cast<const void*>(begin), bytes);
            if (VerifyStreamRing()) VerifyStreamedCopy(begin, copy);
            if (BuildProfiled()) {
                Reuses().streamedRanges.fetch_add(1, std::memory_order_relaxed);
                Reuses().streamedBytes.fetch_add(bytes, std::memory_order_relaxed);
            }
            selected.push_back(index);
            result->snapshots.push_back({begin, std::move(buffer), offset, bytes});
            CaptureTrace::Log("draw-snapshot batch=%llu address=%llx bytes=%zu", static_cast<unsigned long long>(recorder.Submissions() + 1), static_cast<unsigned long long>(begin), bytes);
            continue;
        }
        const auto registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        const auto generation = GuestMemory::CollectWrites(begin, bytes);
        VkDeviceSize offset = 0;
        auto buffer = recorder.ReusableDrawSnapshot(begin, bytes, Recorder::SnapshotUse::Storage, nullptr, &offset);
        if (buffer == nullptr) {
            if (Recorder::SnapshotRingEnabled()) {
                auto slice = recorder.AllocateDrawSnapshot(bytes);
                buffer = std::move(slice.buffer);
                offset = slice.offset;
            }
            if (buffer == nullptr) {
                buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                offset = 0;
            }
            std::memcpy(buffer->Bytes().data() + offset, reinterpret_cast<const void*>(begin), bytes);
            recorder.KeepDrawSnapshot(begin, bytes, generation, registryGeneration, buffer, Recorder::SnapshotUse::Storage, 0, offset);
        }
        selected.push_back(index);
        result->snapshots.push_back({begin, std::move(buffer), offset, bytes});
        CaptureTrace::Log("draw-snapshot batch=%llu address=%llx bytes=%zu", static_cast<unsigned long long>(recorder.Submissions() + 1), static_cast<unsigned long long>(begin), bytes);
    }
    if (selected.empty()) return {};
    Require(context.descriptorCache != nullptr, "draw snapshots require a descriptor cache");
    thread_local std::vector<VkDescriptorPoolSize>* sizesSlot = nullptr;
    auto& sizes = ShaderRecompiler::ThreadOwned(sizesSlot);
    sizes.clear();
    for (const auto& binding : bindings) {
        const auto type = binding.layout.descriptorType;
        const auto found = std::find_if(sizes.begin(), sizes.end(), [type](const VkDescriptorPoolSize& size) { return size.type == type; });
        if (found != sizes.end()) found->descriptorCount += binding.layout.descriptorCount;
        else sizes.push_back({type, binding.layout.descriptorCount});
    }
    // The set written whole from buildComplete's writes (live `draw_set_writes`): the snapshot
    // elements' infos replace the template's, every other descriptor is the one `_set` holds (it is
    // never rewritten after the build), so the set equals the copy-then-patch result below.
    if (!setWrites.empty() && DrawSetWrites()) {
        result->cache = context.descriptorCache;
        result->allocation = result->cache->AllocateRewritten(_layout, sizes);
        result->layout = _layout;
        Require(result->allocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
        thread_local std::vector<VkDescriptorBufferInfo>* setBuffersSlot = nullptr;
        auto& buffers = ShaderRecompiler::ThreadOwned(setBuffersSlot);
        buffers.assign(setBuffers.begin(), setBuffers.end());
        thread_local std::vector<VkWriteDescriptorSet>* setWritesSlot = nullptr;
        auto& writes = ShaderRecompiler::ThreadOwned(setWritesSlot);
        writes.clear();
        std::size_t binding = 0;
        for (const auto& kept : setWrites) {
            auto write = kept.write;
            write.dstSet = result->allocation.set;
            if (kept.image) {
                write.pImageInfo = setImages.data() + kept.first;
            } else {
                write.pBufferInfo = buffers.data() + kept.first;
                // buildComplete wrote one write per binding, in order, its buffer infos in the
                // binding's allocation order.
                const auto& allocated = bindings[binding].allocations;
                for (std::size_t element = 0; element < allocated.size(); ++element) {
                    const auto found = std::find(selected.begin(), selected.end(), allocated[element]);
                    if (found == selected.end()) continue;
                    const auto& snapshot = result->snapshots[static_cast<std::size_t>(found - selected.begin())];
                    buffers[kept.first + element] = {snapshot.buffer->Handle(), snapshot.offset, snapshot.bytes};
                }
            }
            writes.push_back(write);
            ++binding;
        }
        context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        recorder.Keep(result);
        return result;
    }
    result->cache = context.descriptorCache;
    result->allocation = result->cache->AllocateRewritten(_layout, sizes);
    result->layout = _layout;
    Require(result->allocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    thread_local std::vector<VkCopyDescriptorSet>* copiesSlot = nullptr;
    auto& copies = ShaderRecompiler::ThreadOwned(copiesSlot);
    copies.clear();
    for (const auto& binding : bindings) {
        VkCopyDescriptorSet copy{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
        copy.srcSet = _set;
        copy.srcBinding = binding.layout.binding;
        copy.dstSet = result->allocation.set;
        copy.dstBinding = binding.layout.binding;
        copy.descriptorCount = binding.layout.descriptorCount;
        copies.push_back(copy);
    }
    const auto update = context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets");
    update(context.device, 0, nullptr, static_cast<std::uint32_t>(copies.size()), copies.data());
    thread_local std::vector<VkDescriptorBufferInfo>* infosSlot = nullptr;
    auto& infos = ShaderRecompiler::ThreadOwned(infosSlot);
    infos.clear();
    infos.reserve(selected.size());
    for (const auto& snapshot : result->snapshots) infos.push_back({snapshot.buffer->Handle(), snapshot.offset, snapshot.bytes});
    thread_local std::vector<VkWriteDescriptorSet>* writesSlot = nullptr;
    auto& writes = ShaderRecompiler::ThreadOwned(writesSlot);
    writes.clear();
    for (const auto& binding : bindings) {
        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
            const auto found = std::find(selected.begin(), selected.end(), binding.allocations[element]);
            if (found == selected.end()) continue;
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = result->allocation.set;
            write.dstBinding = binding.layout.binding;
            write.dstArrayElement = static_cast<std::uint32_t>(element);
            write.descriptorCount = 1;
            write.descriptorType = binding.layout.descriptorType;
            write.pBufferInfo = &infos[static_cast<std::size_t>(found - selected.begin())];
            writes.push_back(write);
        }
    }
    update(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    recorder.Keep(result);
    return result;
}

void ShaderResources::Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const {
    if (_set == VK_NULL_HANDLE) return;
    context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, bindPoint, layout, 0, 1, &_set, 0, nullptr);
}

void ShaderResources::BindSet(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout, VkDescriptorSet set) const {
    context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, bindPoint, layout, 0, 1, &set, 0, nullptr);
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::DataBindings(Recorder& recorder, const CompiledShader& shader, bool keyChecked, bool& usable) const {
    usable = false;
    if (_set == VK_NULL_HANDLE || shader.program == nullptr || context.descriptorCache == nullptr || !Recorder::SnapshotRingEnabled()) return {};
    const auto& program = *shader.program;
    if (program.bindings.size() != bindings.size()) return {};
    if (!keyChecked && (refreshResourceKey.empty() || !MatchesContentKey(shader, refreshResourceKey))) return {};
    auto result = std::make_shared<DrawBindings>();
    thread_local std::vector<std::size_t>* selectedSlot = nullptr;
    auto& selected = ShaderRecompiler::ThreadOwned(selectedSlot);
    selected.clear();
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        const auto& binding = program.bindings[index];
        if (!DataRole(binding.role)) continue;
        if (bindings[index].allocations.size() != 1) return {};
        const auto allocationIndex = bindings[index].allocations.front();
        const auto& allocation = allocations[allocationIndex];
        const auto size = binding.guestDescriptor.size() * sizeof(std::uint32_t);
        if (allocation.buffer == nullptr || allocation.guest || allocation.size != size || size > MaxRefreshBytes) return {};
        if (allocation.dataWords.size() == binding.guestDescriptor.size() && std::equal(allocation.dataWords.begin(), allocation.dataWords.end(), binding.guestDescriptor.begin())) continue;
        if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) ValidateRuntimeResources(shader, binding.guestDescriptor);
        auto slice = recorder.AllocateDrawSnapshot(size);
        if (slice.buffer == nullptr) return {};
        auto* bytes = slice.buffer->Bytes().data() + slice.offset;
        std::memcpy(bytes, binding.guestDescriptor.data(), size);
        for (const auto& patch : dataPatches) {
            if (patch.allocation == allocationIndex && patch.byte < size) bytes[patch.byte] = static_cast<std::byte>(patch.adjustment);
        }
        selected.push_back(allocationIndex);
        result->snapshots.push_back({0, std::move(slice.buffer), slice.offset, size});
    }
    usable = true;
    if (selected.empty()) return {};
    thread_local std::vector<VkDescriptorPoolSize>* sizesSlot = nullptr;
    auto& sizes = ShaderRecompiler::ThreadOwned(sizesSlot);
    sizes.clear();
    for (const auto& binding : bindings) {
        const auto type = binding.layout.descriptorType;
        const auto found = std::find_if(sizes.begin(), sizes.end(), [type](const VkDescriptorPoolSize& size) { return size.type == type; });
        if (found != sizes.end()) found->descriptorCount += binding.layout.descriptorCount;
        else sizes.push_back({type, binding.layout.descriptorCount});
    }
    result->cache = context.descriptorCache;
    result->allocation = result->cache->AllocateRewritten(_layout, sizes);
    result->layout = _layout;
    if (result->allocation.set == VK_NULL_HANDLE) {
        usable = false;
        return {};
    }
    thread_local std::vector<VkCopyDescriptorSet>* copiesSlot = nullptr;
    auto& copies = ShaderRecompiler::ThreadOwned(copiesSlot);
    copies.clear();
    for (const auto& binding : bindings) {
        VkCopyDescriptorSet copy{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
        copy.srcSet = _set;
        copy.srcBinding = binding.layout.binding;
        copy.dstSet = result->allocation.set;
        copy.dstBinding = binding.layout.binding;
        copy.descriptorCount = binding.layout.descriptorCount;
        copies.push_back(copy);
    }
    const auto update = context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets");
    update(context.device, 0, nullptr, static_cast<std::uint32_t>(copies.size()), copies.data());
    thread_local std::vector<VkDescriptorBufferInfo>* infosSlot = nullptr;
    auto& infos = ShaderRecompiler::ThreadOwned(infosSlot);
    infos.clear();
    infos.reserve(selected.size());
    for (const auto& snapshot : result->snapshots) infos.push_back({snapshot.buffer->Handle(), snapshot.offset, snapshot.bytes});
    thread_local std::vector<VkWriteDescriptorSet>* writesSlot = nullptr;
    auto& writes = ShaderRecompiler::ThreadOwned(writesSlot);
    writes.clear();
    for (const auto& binding : bindings) {
        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
            const auto found = std::find(selected.begin(), selected.end(), binding.allocations[element]);
            if (found == selected.end()) continue;
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = result->allocation.set;
            write.dstBinding = binding.layout.binding;
            write.dstArrayElement = static_cast<std::uint32_t>(element);
            write.descriptorCount = 1;
            write.descriptorType = binding.layout.descriptorType;
            write.pBufferInfo = &infos[static_cast<std::size_t>(found - selected.begin())];
            writes.push_back(write);
        }
    }
    update(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    recorder.Keep(result);
    return result;
}

namespace {
// Debug aid: APS5_GPU_NO_WRITEBACK=1 keeps GPU results out of guest memory.
bool SkipWriteBack() {
    static const bool skip = std::getenv("APS5_GPU_NO_WRITEBACK") != nullptr;
    return skip;
}
}

void ShaderResources::WriteBack() {
    WriteBackBuffers();
    if (SkipWriteBack()) return;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (storageWritten[index]) storageTextures[index]->MarkDirty();
    }
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ShaderResources::writtenImageRanges() const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (std::size_t index = 0; index < storageTextures.size() && index < storageWritten.size(); ++index) {
        const auto& image = storageTextures[index];
        if (image == nullptr || !storageWritten[index]) continue;
        ranges.emplace_back(image->Descriptor().baseAddress, image->Descriptor().baseAddress + image->GuestBytes());
    }
    return ranges;
}

void ShaderResources::MarkGpuWrites(Recorder& recorder) {
    // First: the pending units the write-back elision skipped under regions bound in place are
    // dropped now that the dispatch overwriting them is recorded (those of staged regions follow
    // their copy-backs, below).
    guestMemory.DropElided(false);
    // The ranges this use reads in place through their host imports (read-only and written elements
    // alike, and an address-based build's whole leased heaps), before the writes: a CPU store into
    // one of them (the copy HLE) must not land before the recorded work read it.
    static const bool dedupeSpaceReads = std::getenv("APS5_NO_READ_SET_DEDUPE") == nullptr;
    const auto readKind = guestMemory.HoldsLease() ? Recorder::ReadKind::AddressBased : Recorder::ReadKind::DispatchElement;
    if (const auto spaceSerial = dedupeSpaceReads ? guestMemory.AddressSpaceSerial() : 0; spaceSerial != 0) {
        // APS5_NO_SHARED_READS=1 copies the space's ranges into each batch, as before.
        static const bool sharedReads = std::getenv("APS5_NO_SHARED_READS") == nullptr;
        if (!recorder.ReadSetNoted(spaceSerial)) {
            if (auto set = sharedReads ? guestMemory.AddressSpaceReadSet() : nullptr) recorder.NoteSharedReads(std::move(set), readKind);
            else recorder.NotePendingReads(guestMemory.AddressSpaceReads(), readKind);
        }
        recorder.NotePendingReads(guestMemory.OwnInPlaceReads(), readKind);
    } else {
        recorder.NotePendingReads(guestMemory.InPlaceReads(), readKind);
    }
    if (SkipWriteBack()) {
        guestMemory.DropElided(true);
        return;
    }
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (storageWritten[index]) storageTextures[index]->MarkDirty();
    }
    // Written sub-ranges of buffers the GPU copied out of a host import go back into it by the GPU,
    // recorded here after the work: those regions then need no CPU write-back (HasCopiedWrites),
    // and the note and mark below cover them like direct writes.
    guestMemory.RecordCopyBacks(recorder);
    // The skipped units under staged regions, now that their copy-backs are recorded (a throw before
    // here leaves them pending over import bytes that never changed). No storage image this use
    // writes overlaps any skipped unit (GuestBufferMemory::SetOtherWrites).
    guestMemory.DropElided(true);
    // Only the written elements' ranges (AddWritable): a read-only element is neither noted here
    // nor marked as a direct write, so CPU reads of its memory never wait for this work.
    recorder.NotePendingWrites(guestMemory.Writes());
    guestMemory.MarkDirectWrites();
    if (!BuildProfiled()) return;
    auto& counters = BufferWrites();
    // Recorded uses only (dispatches and recorded draws): a synchronous draw writes back in
    // WriteBack() and publishes no pending writes, so it has no notes to skip.
    counters.notesSkipped.fetch_add(readOnlyBuffers, std::memory_order_relaxed);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto elements = counters.elements.load();
    const auto readOnly = counters.readOnly.load();
    AgcDriver::ProfilePrint_nid_no_patch("[buffers] descriptor elements bound: %llu total, %llu written, %llu read-only; %llu pending-write notes skipped\n", static_cast<unsigned long long>(elements), static_cast<unsigned long long>(elements - readOnly), static_cast<unsigned long long>(readOnly), static_cast<unsigned long long>(counters.notesSkipped.load()));
}

void ShaderResources::WriteBackBuffers() {
    if (bda) bda->CheckFault();
    if (SkipWriteBack()) return;
    guestMemory.WriteBack();
}

bool ShaderResources::WritesMemory() const {
    return HoldsLease() || NeedsCompletion() || !guestMemory.Writes().empty() || std::any_of(storageWritten.begin(), storageWritten.end(), [](bool written) { return written; });
}

bool ShaderResources::ReadsOverlap(std::uint64_t address, std::size_t bytes) const {
    return guestMemory.InPlaceReadsOverlap(address, bytes);
}

std::vector<std::pair<VkImage, bool>> ShaderResources::StorageImages() const {
    std::vector<std::pair<VkImage, bool>> images;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (storageTextures[index] != nullptr) images.emplace_back(storageTextures[index]->Image(), storageWritten[index]);
    }
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->StorageSource() != nullptr) images.emplace_back(texture->StorageSource()->Image(), false);
    }
    return images;
}

bool ShaderResources::ReadsImage(const StorageTexture* image) const {
    if (image == nullptr) return false;
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->StorageSource() == image) return true;
    }
    return std::any_of(storageTextures.begin(), storageTextures.end(), [&](const auto& storage) { return storage.get() == image; });
}

}
