#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/AddressLedger.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuCheckpoints.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PendingUnits.hpp"
#include "prx/libSceAgcDriver/Graphics/shaders/Float11Repack_spv.h"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/SlimMutex.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
// For the declaration of PendingStorageOverlaps, defined below.
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawSpeculation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <list>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>
#include <atomic>
#include <mutex>
#include <map>
#include <limits>
#include <exception>
#include <optional>
#include <unordered_map>
#include <utility>

extern "C" __declspec(dllimport) unsigned short __stdcall RtlCaptureStackBackTrace(unsigned long framesToSkip, unsigned long framesToCapture, void** backTrace, unsigned long* backTraceHash);

// The [alias-stack] trace prints this function's address so its return addresses can be placed in
// the module (its symbol address minus this one is the load offset).
extern "C" void AliasStackAnchor() {}

namespace AgcDriver::Graphics {

namespace {
// APS5_TRACE_TEXTURES=1: a stdout line per texture and mip created (each flushes; at a few hundred
// textures per second in a race it cost the committer tens of ms per frame when unconditional).
bool TraceTextures() {
    static const bool trace = std::getenv("APS5_TRACE_TEXTURES") != nullptr;
    return LiveFlag("trace_textures", trace);
}
}


namespace {

// APS5_PROFILE_DRAW: accumulate texture setup phases and report every 200 textures.
struct TextureProfile {
    double allocate = 0, read = 0, gpu = 0, view = 0;
    std::uint64_t count = 0;
    std::uint64_t fromStorage = 0;
    double storageCreate = 0, storageWriteBack = 0, storageAlloc = 0, storageHostCopy = 0, storageGpu = 0, storageStore = 0;
    std::uint64_t storageCount = 0;
    std::uint64_t storageBytes = 0;
    std::uint64_t storageReused = 0;
    std::uint64_t storageDirectUploads = 0;
    std::uint64_t storageDirectWriteBacks = 0;
    // Sampled-texture uploads recorded into the open batch instead of waited for.
    std::uint64_t recordedUploads = 0;
    // GPU clears of fast-cleared storage images recorded into the open batch, and waited for in a
    // batch of their own (APS5_NO_RECORDED_CLEAR=1, or no recorder).
    std::uint64_t storageRecordedClears = 0;
    std::uint64_t storageWaitedClears = 0;
};

TextureProfile& Profile() {
    static TextureProfile profile;
    return profile;
}

// APS5_PROFILE_DRAW, every 10 s on the [storage] line: storage image bytes uploaded by path
// (recorded DCC clear, direct detile from the import, CPU) and written back by the reason of the
// FlushPending that forced the write-back (the other callers name themselves: "refresh", "cache
// eviction", "explicit"). `flushReason` is set by the caller around its writeBack calls.
thread_local const char* flushReason = nullptr;
// Likewise what made an upload necessary: "first" (a new image), "keys" (the DCC keys changed),
// "cpu" (a CPU store stamped a changed unit), "flushed" (another image's results were stored over
// the surface by this refresh), "store" (a driver store: a fill, a copy, a label, an earlier
// write-back), "untracked" (no write stamps for the surface: every refresh re-uploads), "alias"
// (units taken from an alias image on the device); set by the constructor and Refresh around
// upload() and borrowUnits().
thread_local const char* uploadReason = "first";
// The size of the range the FlushPending storing an image asked for (the [wb-cause] line: an access
// much smaller than the stored surface marks whole-surface stores); 0 outside FlushPending.
thread_local std::size_t flushRequestBytes = 0;
// Block-unit traffic, per 10 s like the rest of the [storage] line: partial uploads and write-backs
// (count and bytes), write-backs that stored nothing (every selected block CPU-written), partial
// selections widened to every pending unit (the hysteresis in writeBack), units dropped as dead
// (DiscardPendingInside) and units an alias's own results superseded (markLayersPending); relaxed,
// reported only.
std::atomic<std::uint64_t> partialUploads{0}, partialUploadBytes{0}, partialWriteBacks{0}, partialWriteBackBytes{0}, emptyWriteBacks{0}, coalescedWriteBacks{0}, unitsDropped{0}, unitsSuperseded{0}, fusedWriteBacks{0};
// FlushPending listings skipped because no pending unit of the image lay inside the range, images
// whose selected pending units were dropped because their memory is no longer registered (see
// writeBackLayers), and pending units a DCC clear -> uncompressed key flip kept as the texels
// (Refresh).
std::atomic<std::uint64_t> pretestSkipped{0}, unregisteredDropped{0}, keyFlipKept{0}, aliasDiscards{0};
// Images stored by a FlushPending after a hook skip of theirs (AccessKeptByCpu), of which by the
// hook for the read site of the last skip; images evictStale dropped (the [hooksync] and [storage] lines).
std::atomic<std::uint64_t> flushedAfterSkip{0}, flushedAfterSkipSameSite{0}, staleEvicted{0};

// Live `lazy_writeback` (default on; APS5_LAZY_WRITEBACK=0 makes the default off): results stay on
// the GPU for consumers that do not need guest bytes. A thick (3D) surface created while it is on
// tracks 64 KiB block units like a thin one, so an access to part of it stores the units it
// touches instead of the whole volume (the flag off stores every pending unit at once, as the
// whole-surface tracking did); and a refresh takes the pending units of an image at the same base
// with the same width and layout but another height (the 1920x2160 pool over two 1920x1080 images)
// on the device, its edge units (bytes one of the two surfaces holds no element of) stored first.
// APS5_VERIFY_LAZY_WRITEBACK=1 checks each thick partial store's windows against the whole-surface
// retile and each such borrow against the alias's own retile (a GPU wait each: slow, debug only).
std::atomic<int> lazyOverride{-1}, lazyVerifyOverride{-1};
std::atomic<std::uint64_t> lazyThickStores{0}, lazyThickStoredBytes{0}, lazyThickKeptBytes{0}, lazyRowBorrows{0}, lazyRowBorrowBytes{0}, lazyEdgeUnits{0}, lazyVerifyChecks{0}, lazyVerifyMismatches{0};

bool LazyWriteBack() {
    if (const auto forced = lazyOverride.load(std::memory_order_relaxed); forced >= 0) return forced != 0;
    static const bool fallback = [] {
        const char* text = std::getenv("APS5_LAZY_WRITEBACK");
        return text == nullptr || std::strcmp(text, "0") != 0;
    }();
    return LiveFlag("lazy_writeback", fallback);
}

bool VerifyLazyWriteBack() {
    if (const auto forced = lazyVerifyOverride.load(std::memory_order_relaxed); forced >= 0) return forced != 0;
    static const bool requested = [] {
        const char* text = std::getenv("APS5_VERIFY_LAZY_WRITEBACK");
        return text != nullptr && std::strcmp(text, "0") != 0;
    }();
    return requested;
}

// The tile block a write-back or upload window moves whole: {block bytes, width, height} in
// elements (a thick block also spans blockDepth slices, every one of which a window of its slab
// carries).
std::array<std::uint32_t, 3> WindowBlockLayout(const GuestTextureResource& descriptor, const SurfaceGeometry& geometry) {
    const auto elementBytes = BytesPerElement(descriptor.format);
    if (!geometry.thick) return ThinBlockLayout(descriptor.tileMode, elementBytes);
    const auto extent = ThickBlockExtent(descriptor.tileMode, elementBytes);
    return {descriptor.tileMode == TextureTileMode::kStandard4KB ? 4096u : 65536u, extent[0], extent[1]};
}

// APS5_PROFILE_DRAW: what each write-back stored and why, per surface (10 s, the [wb-cause] line),
// and the timeline kind of its range carrying the cause ("writeback-windows/storage-refresh").
struct WriteBackCauses {
    std::mutex mutex;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> totals;
};

WriteBackCauses& Causes() {
    static WriteBackCauses causes;
    return causes;
}

const char* CauseKind(const char* kind) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile || flushReason == nullptr) return kind;
    static std::mutex mutex;
    static std::map<std::pair<const char*, const char*>, std::string> names;
    std::lock_guard lock(mutex);
    auto& name = names[{kind, flushReason}];
    if (name.empty()) {
        name = std::string(kind) + "/" + flushReason;
        std::replace(name.begin(), name.end(), ' ', '-');
    }
    return name.c_str();
}

void reportLazyWriteBack() {
    static std::uint64_t seen[8] = {};
    const std::uint64_t now[8] = {lazyThickStores.load(std::memory_order_relaxed), lazyThickStoredBytes.load(std::memory_order_relaxed), lazyThickKeptBytes.load(std::memory_order_relaxed), lazyRowBorrows.load(std::memory_order_relaxed), lazyRowBorrowBytes.load(std::memory_order_relaxed), lazyEdgeUnits.load(std::memory_order_relaxed), lazyVerifyChecks.load(std::memory_order_relaxed), lazyVerifyMismatches.load(std::memory_order_relaxed)};
    std::uint64_t delta[8];
    for (std::size_t i = 0; i < 8; ++i) {
        delta[i] = now[i] - seen[i];
        seen[i] = now[i];
    }
    AgcDriver::ProfilePrint_nid_no_patch("[lazy-wb] (10 s, lazy_writeback %d) thick partial stores %llu: %.1f MiB stored, %.1f MiB left pending on the GPU; row-alias borrows %llu: %.1f MiB copied on the device, %llu edge units stored first; verify %llu checks, %llu mismatches\n", LazyWriteBack() ? 1 : 0, static_cast<unsigned long long>(delta[0]), delta[1] / 1048576.0, delta[2] / 1048576.0, static_cast<unsigned long long>(delta[3]), delta[4] / 1048576.0, static_cast<unsigned long long>(delta[5]), static_cast<unsigned long long>(delta[6]), static_cast<unsigned long long>(delta[7]));
    auto& causes = Causes();
    std::vector<std::pair<std::uint64_t, std::string>> heavy;
    {
        std::lock_guard lock(causes.mutex);
        for (const auto& [name, totals] : causes.totals) heavy.emplace_back(totals.second, std::to_string(totals.first) + "x " + name);
        causes.totals.clear();
    }
    if (heavy.empty()) return;
    std::sort(heavy.rbegin(), heavy.rend());
    std::string line = "[wb-cause] (10 s) write-backs by surface and cause:";
    char text[32];
    for (std::size_t i = 0; i < std::min<std::size_t>(heavy.size(), 16); ++i) {
        std::snprintf(text, sizeof(text), " | %.1f MiB ", heavy[i].first / 1048576.0);
        line += text + heavy[i].second;
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
}

struct StorageTraffic {
    std::mutex mutex;
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> writeBacks;
    std::array<std::pair<std::uint64_t, std::uint64_t>, 4> uploads{};
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> uploadReasons;
    std::uint64_t directWriteBacks = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

StorageTraffic& Traffic() {
    static StorageTraffic traffic;
    return traffic;
}

void reportStorageTraffic(StorageTraffic& traffic) {
    const auto now = std::chrono::steady_clock::now();
    if (now - traffic.lastReport < std::chrono::seconds(10)) return;
    traffic.lastReport = now;
    static constexpr const char* uploadNames[4] = {"clear", "direct", "cpu", "alias"};
    std::string line;
    char text[96];
    for (std::size_t i = 0; i < 4; ++i) {
        std::snprintf(text, sizeof(text), " %s %llu/%.1f", uploadNames[i], static_cast<unsigned long long>(traffic.uploads[i].first), traffic.uploads[i].second / 1048576.0);
        line += text;
        traffic.uploads[i] = {};
    }
    line += "; by reason:";
    for (const auto& [reason, totals] : traffic.uploadReasons) {
        std::snprintf(text, sizeof(text), " %llu/%.1f", static_cast<unsigned long long>(totals.first), totals.second / 1048576.0);
        line += " " + reason + text;
    }
    line += "; write-backs by reason (count/MiB):";
    for (const auto& [reason, totals] : traffic.writeBacks) {
        std::snprintf(text, sizeof(text), " %llu/%.1f", static_cast<unsigned long long>(totals.first), totals.second / 1048576.0);
        line += " " + reason + text;
    }
    const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    const auto partialUploadCount = take(partialUploads), partialWriteBackCount = take(partialWriteBacks);
    const auto uploadMiB = take(partialUploadBytes) / 1048576.0, writeBackMiB = take(partialWriteBackBytes) / 1048576.0;
    AgcDriver::ProfilePrint_nid_no_patch( "[storage] uploads by path (count/MiB, 10 s):%s; %llu write-backs GPU-direct (%llu fused), %llu stored nothing; block units: partial uploads %llu/%.1f, partial write-backs %llu/%.1f, %llu widened to every pending unit, units dropped %llu, superseded %llu, dropped unregistered %llu, kept over a key flip %llu, render-target alias uploads discarded %llu%s, pretest skipped %llu, evicted stale %llu\n", line.c_str(), static_cast<unsigned long long>(traffic.directWriteBacks), take(fusedWriteBacks), take(emptyWriteBacks), partialUploadCount, uploadMiB, partialWriteBackCount, writeBackMiB, take(coalescedWriteBacks), take(unitsDropped), take(unitsSuperseded), take(unregisteredDropped), take(keyFlipKept), take(aliasDiscards), ShadowReport().c_str(), take(pretestSkipped), take(staleEvicted));
    traffic.writeBacks.clear();
    traffic.uploadReasons.clear();
    traffic.directWriteBacks = 0;
    reportLazyWriteBack();
}

void countStorageUpload(std::size_t path, std::uint64_t bytes) {
    if (!LookupOutcomes::Profiled()) return;
    auto& traffic = Traffic();
    std::lock_guard lock(traffic.mutex);
    ++traffic.uploads[path].first;
    traffic.uploads[path].second += bytes;
    auto& reason = traffic.uploadReasons[uploadReason != nullptr ? uploadReason : "other"];
    ++reason.first;
    reason.second += bytes;
    reportStorageTraffic(traffic);
}

void countStorageWriteBack(std::uint64_t bytes, bool direct) {
    if (!LookupOutcomes::Profiled()) return;
    auto& traffic = Traffic();
    std::lock_guard lock(traffic.mutex);
    auto& totals = traffic.writeBacks[flushReason != nullptr ? flushReason : "other"];
    ++totals.first;
    totals.second += bytes;
    if (direct) ++traffic.directWriteBacks;
    reportStorageTraffic(traffic);
}

// Every storage image alive, for the fill HLE's cover check (StorageTexture::ClassifyFill): the
// storage cache indexes surfaces by key, not by address range. Its mutex is a leaf.
struct LiveImages {
    std::mutex mutex;
    std::vector<StorageTexture*> textures;
};

LiveImages& Live() {
    static LiveImages live;
    return live;
}

// Per-layer validity of array surfaces (see StorageTexture::layerBegin): a fill of one layer clears
// it on the image and only that layer's guest bytes go stale, so the other layers' fills, uploads
// and write-backs stay separate. APS5_NO_LAYER_CLEAR=1 tracks every surface as one layer, and a
// one-layer FillClear then demands the rest of the surface unchanged, as before.
bool LayerTrackingEnabled() {
    static const bool disabled = std::getenv("APS5_NO_LAYER_CLEAR") != nullptr;
    return !disabled;
}

// APS5_NO_BLOCK_TRACKING=1 tracks array layers (or the whole surface) instead of 64 KiB blocks.
bool BlockTrackingEnabled() {
    static const bool disabled = std::getenv("APS5_NO_BLOCK_TRACKING") != nullptr;
    return !disabled;
}

struct PhaseTimer {
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    double lap() {
        const auto now = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration<double, std::milli>(now - last).count();
        last = now;
        return ms;
    }
};


VkImageType ImageTypeFor(TextureDimension dimension) {
    return dimension == TextureDimension::k1D || dimension == TextureDimension::k1DArray ? VK_IMAGE_TYPE_1D : dimension == TextureDimension::k3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
}

VkImageViewType ViewTypeFor(TextureDimension dimension, [[maybe_unused]] std::uint32_t viewLayerCount) {
    switch (dimension) {
        case TextureDimension::k1D: return VK_IMAGE_VIEW_TYPE_1D;
        case TextureDimension::k2D: return VK_IMAGE_VIEW_TYPE_2D;
        case TextureDimension::k2DArray: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        // Shaders address cube maps as 2D arrays of faces.
        case TextureDimension::kCube: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        case TextureDimension::k3D: return VK_IMAGE_VIEW_TYPE_3D;
        case TextureDimension::k1DArray: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    }
    throw std::runtime_error("AGC graphics: Texture encountered an unknown guest texture dimension");
}

}

namespace {

void ChainMinLod(const Context& context, const GuestTextureResource& descriptor, VkImageViewCreateInfo& viewInfo, VkImageViewMinLodCreateInfoEXT& minLod) {
    const auto clamp = EffectiveMinLod(descriptor);
    if (clamp == 0.0f) return;
    Require(context.imageViewMinLod, "guest texture descriptor uses a minimum LOD clamp, which needs VK_EXT_image_view_min_lod");
    minLod.minLod = clamp;
    minLod.pNext = viewInfo.pNext;
    viewInfo.pNext = &minLod;
}

}


namespace ImagePool {
namespace {

struct Pooled {
    Context context;
    std::uint64_t key;
    VkImage image;
    VkDeviceMemory memory;
    VkDeviceSize bytes;
};

struct State {
    std::mutex mutex;
    // Most recently given first; evicted from the back past the byte cap.
    std::list<Pooled> entries;
    std::unordered_multimap<std::uint64_t, std::list<Pooled>::iterator> byKey;
    VkDeviceSize bytes = 0;
    std::uint64_t takes = 0, misses = 0, gives = 0, evictions = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

State& Pool() {
    static auto* state = new State;
    return *state;
}

bool Enabled() {
    static const bool enabled = std::getenv("APS5_NO_IMAGE_POOL") == nullptr;
    return enabled;
}

VkDeviceSize CapBytes() {
    static const VkDeviceSize cap = [] {
        const char* text = std::getenv("APS5_IMAGE_POOL_MIB");
        return static_cast<VkDeviceSize>(text != nullptr ? std::strtoull(text, nullptr, 10) : 1024ull) << 20u;
    }();
    return cap;
}

void Destroy(const Pooled& entry) {
    entry.context.Function<PFN_vkDestroyImage>("vkDestroyImage")(entry.context.device, entry.image, nullptr);
    entry.context.Function<PFN_vkFreeMemory>("vkFreeMemory")(entry.context.device, entry.memory, nullptr);
}

void Report(State& state) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile || std::chrono::steady_clock::now() - state.lastReport < std::chrono::seconds(10)) return;
    state.lastReport = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[image-pool] (10 s) %llu taken, %llu missed, %llu given, %llu evicted; %zu held (%.1f MiB)\n", static_cast<unsigned long long>(state.takes), static_cast<unsigned long long>(state.misses), static_cast<unsigned long long>(state.gives), static_cast<unsigned long long>(state.evictions), state.entries.size(), state.bytes / 1048576.0);
    state.takes = state.misses = state.gives = state.evictions = 0;
}

}

std::uint64_t Key(VkDevice device, const VkImageCreateInfo& info) {
    if (!Enabled() || info.pNext != nullptr) return 0;
    std::uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 0x100000001b3ull;
    };
    mix(reinterpret_cast<std::uintptr_t>(device));
    mix(info.flags);
    mix(info.imageType);
    mix(info.format);
    mix(info.extent.width);
    mix(info.extent.height);
    mix(info.extent.depth);
    mix(info.mipLevels);
    mix(info.arrayLayers);
    mix(info.samples);
    mix(info.tiling);
    mix(info.usage);
    mix(info.sharingMode);
    return hash == 0 ? 1 : hash;
}

bool Take(VkDevice device, std::uint64_t key, VkImage& image, VkDeviceMemory& memory, VkDeviceSize& bytes) {
    if (key == 0) return false;
    auto& state = Pool();
    std::lock_guard lock(state.mutex);
    Report(state);
    const auto found = state.byKey.find(key);
    if (found == state.byKey.end() || found->second->context.device != device) {
        ++state.misses;
        return false;
    }
    const auto entry = found->second;
    image = entry->image;
    memory = entry->memory;
    bytes = entry->bytes;
    state.bytes -= entry->bytes;
    state.byKey.erase(found);
    state.entries.erase(entry);
    ++state.takes;
    return true;
}

bool Give(const Context& context, std::uint64_t key, VkImage image, VkDeviceMemory memory, VkDeviceSize bytes) {
    if (key == 0 || !Enabled() || bytes > CapBytes() / 4) return false;
    auto& state = Pool();
    std::vector<Pooled> evicted;
    {
        std::lock_guard lock(state.mutex);
        state.entries.push_front({context, key, image, memory, bytes});
        state.byKey.emplace(key, state.entries.begin());
        state.bytes += bytes;
        ++state.gives;
        while (state.bytes > CapBytes() && !state.entries.empty()) {
            auto last = std::prev(state.entries.end());
            const auto range = state.byKey.equal_range(last->key);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second == last) {
                    state.byKey.erase(it);
                    break;
                }
            }
            state.bytes -= last->bytes;
            evicted.push_back(*last);
            state.entries.erase(last);
            ++state.evictions;
        }
    }
    for (const auto& entry : evicted) Destroy(entry);
    return true;
}

}

Texture::Texture(const Context& context, TextureDetiler& detiler, const GuestTextureResource& descriptor, VkComponentMapping components, std::span<const std::byte> snapshot, bool depthCompare) : context(context) {

    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    PhaseTimer timer;
    try {
        const auto colorFormat = SampledTextureFormat(context, descriptor.format);
        Require(!depthCompare || colorFormat == VK_FORMAT_R32_SFLOAT || colorFormat == VK_FORMAT_R16_UNORM, "comparison sampling requires an R32 float or R16 unorm depth texture");
        Require(!depthCompare || descriptor.dimension != TextureDimension::k3D, "comparison sampling does not support 3D depth textures");
        const auto vkFormat = depthCompare ? (colorFormat == VK_FORMAT_R32_SFLOAT ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_D16_UNORM) : colorFormat;
        const VkImageAspectFlags aspect = depthCompare ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        if (IsBlockCompressed(descriptor.format)) {
            Require(context.textureCompressionBC, "device does not support BC compressed textures");
        }

        const auto geometry = DescribeSurface(descriptor);
        const auto& mips = geometry.mips;
        const auto arrayLayers = geometry.layers;
        const auto elementBytes = BytesPerElement(descriptor.format);
        APS5_LOG_OUT_IF(TraceTextures(), "Texture address=0x%llx %ux%u mips=%u layers=%u dim=%d tile=%d format=%u vk=%d element=%u", static_cast<unsigned long long>(descriptor.baseAddress), descriptor.width, descriptor.height, descriptor.mipCount, arrayLayers,
                     static_cast<int>(descriptor.dimension), static_cast<int>(descriptor.tileMode), descriptor.format, static_cast<int>(vkFormat), elementBytes);
        for (const auto& mip : mips) APS5_LOG_OUT_IF(TraceTextures(), "  mip %ux%u tiled=0x%llx+0x%llx linear=0x%llx+0x%llx blocksPerRow=%u pitch=%u tail=%d", mip.width, mip.height, static_cast<unsigned long long>(mip.tiledOffset), static_cast<unsigned long long>(mip.tiledSize), static_cast<unsigned long long>(mip.linearOffset), static_cast<unsigned long long>(mip.linearSize), mip.blocksPerRow, mip.pitchBytes, mip.tail ? 1 : 0);

        const auto guestBytes = geometry.guestBytes;
        Require(snapshot.size() == guestBytes, "texture snapshot size mismatch");

        const auto sliceLinearBytes = geometry.sliceLinearBytes;
        Require(arrayLayers == 0 || sliceLinearBytes <= UINT64_MAX / arrayLayers, "detiled texture buffer size overflows");
        const auto linearBytes = sliceLinearBytes * arrayLayers;

        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.flags = descriptor.dimension == TextureDimension::kCube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
        imageInfo.imageType = ImageTypeFor(descriptor.dimension);
        imageInfo.format = vkFormat;
        imageInfo.extent = {descriptor.width, descriptor.height, geometry.imageDepth};
        imageInfo.mipLevels = descriptor.mipCount;
        imageInfo.arrayLayers = geometry.imageLayers;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        const auto poolKey = ImagePool::Key(context.device, imageInfo);
        VkDeviceMemory pooledMemory = VK_NULL_HANDLE;
        VkDeviceSize pooledBytes = 0;
        if (ImagePool::Take(context.device, poolKey, image, pooledMemory, pooledBytes)) {
            owned = std::make_shared<OwnedImage>(context, image, pooledMemory);
            allocationBytes = pooledBytes;
        } else {
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &imageInfo, nullptr, &image), "vkCreateImage");
            if (AddressLedger::Enabled()) {
                char name[112];
                std::snprintf(name, sizeof(name), "texture 0x%llx %ux%ux%u layers %u mips %u fmt %d", static_cast<unsigned long long>(descriptor.baseAddress), imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth, imageInfo.arrayLayers, imageInfo.mipLevels, static_cast<int>(imageInfo.format));
                AddressLedger::NameObject(reinterpret_cast<std::uint64_t>(image), name);
            }
            owned = std::make_shared<OwnedImage>(context, image, VK_NULL_HANDLE);

            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocationBytes = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &owned->memory), "vkAllocateMemory texture");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, owned->memory, 0), "vkBindImageMemory");
        }
        owned->poolKey = poolKey;
        owned->poolBytes = allocationBytes;

        {
            // Debug aid: APS5_DUMP_TEXTURE=<hex addresses, comma separated> saves the detiled first mip
            // of those sampled textures on their first 8 uploads as texture_<address>_<n>.raw (u32
            // width, height, VkFormat, then tightly packed rows).
            static const std::string dumpList = [] { const char* text = std::getenv("APS5_DUMP_TEXTURE"); return text ? std::string(text) : std::string(); }();
            bool dumpWanted = false;
            if (!dumpList.empty()) {
                char address[32];
                std::snprintf(address, sizeof(address), "%llx", static_cast<unsigned long long>(descriptor.baseAddress));
                char extent[32];
                std::snprintf(extent, sizeof(extent), "%ux%u", descriptor.width, descriptor.height);
                // Entries may also be extents ("3840x2160"), since heap addresses change between runs.
                dumpWanted = dumpList.find(address) != std::string::npos || dumpList.find(extent) != std::string::npos;
            }
            auto staging = std::make_shared<Buffer>(context, static_cast<std::size_t>(guestBytes), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            std::memcpy(staging->Bytes().data(), snapshot.data(), snapshot.size());
            auto tiled = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(guestBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            auto linear = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(linearBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            if (profile) Profile().allocate += timer.lap();
            if (profile) Profile().read += timer.lap();

            detiler.BeginBatch();
            // The upload is recorded into the open batch, like a storage image's: the work that samples
            // the texture is recorded after it, and a batch of its own (SubmitAndWait) was a full GPU
            // drain under the device lock for every new texture. The buffers and the image live with
            // the batch (the cache may drop the texture before it completes). The dump aid needs the
            // linear bytes on the CPU, so it keeps the waiting batch, as does a build without a
            // recorder (tests). APS5_SYNC_TEXTURE_UPLOAD=1 restores the waiting batch for every upload.
            static const bool syncUploads = std::getenv("APS5_SYNC_TEXTURE_UPLOAD") != nullptr;
            auto* recorder = dumpWanted || syncUploads ? nullptr : Recorder::Active();
            std::unique_ptr<CommandBatch> batch;
            VkCommandBuffer commands = VK_NULL_HANDLE;
            if (recorder != nullptr) {
                commands = recorder->Commands();
                recorder->Keep(staging, static_cast<std::size_t>(guestBytes));
                recorder->Keep(tiled, static_cast<std::size_t>(guestBytes));
                recorder->Keep(linear, static_cast<std::size_t>(linearBytes));
                recorder->Keep(owned);
            } else {
                batch = std::make_unique<CommandBatch>(context);
                commands = batch->Handle();
            }

            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            CopyBuffer(context, commands, staging->Handle(), 0, tiled->Handle(), 0, guestBytes);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
                const auto guestLayerOffset = geometry.GuestLayerOffset(layer);
                const auto linearLayerOffset = geometry.LinearLayerOffset(layer);
                for (std::uint32_t level = 0; level < mips.size(); ++level) {
                    if (!geometry.HasLayer(level, layer)) continue;
                    const auto& mip = mips[level];
                    detiler.Dispatch(commands, descriptor.tileMode, elementBytes, tiled->Handle(), guestLayerOffset + mip.tiledOffset, linear->Handle(), linearLayerOffset + mip.linearOffset, mip, false, layer, geometry.thick);
                }
            }
            if (!depthCompare && IsRepackedTextureFormat(descriptor.format)) {
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
                    for (std::uint32_t level = 0; level < mips.size(); ++level) {
                        if (!geometry.HasLayer(level, layer)) continue;
                        const auto offset = geometry.LinearLayerOffset(layer) + mips[level].linearOffset;
                        detiler.DispatchRepack(commands, linear->Handle(), offset, linear->Handle(), offset, mips[level].linearSize);
                    }
                }
            }

            VkBufferMemoryBarrier linearReadBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            linearReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            linearReadBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            linearReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            linearReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            linearReadBarrier.buffer = linear->Handle();
            linearReadBarrier.offset = 0;
            linearReadBarrier.size = VK_WHOLE_SIZE;

            VkImageMemoryBarrier toTransferDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toTransferDst.srcAccessMask = 0;
            toTransferDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toTransferDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toTransferDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toTransferDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransferDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransferDst.image = image;
            toTransferDst.subresourceRange = {aspect, 0, descriptor.mipCount, 0, geometry.imageLayers};
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &linearReadBarrier, 1, &toTransferDst);

            std::vector<VkBufferImageCopy> regions;
            regions.reserve(static_cast<std::size_t>(arrayLayers) * mips.size());
            for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
                const auto linearLayerOffset = static_cast<std::uint64_t>(layer) * sliceLinearBytes;
                for (std::uint32_t level = 0; level < descriptor.mipCount; ++level) {
                    if (!geometry.HasLayer(level, layer)) continue;
                    const auto& mip = mips[level];
                    VkBufferImageCopy region{};
                    region.bufferOffset = linearLayerOffset + mip.linearOffset;
                    region.bufferRowLength = mip.pitchBytes / BytesPerElement(descriptor.format) * BlockWidth(descriptor.format);
                    region.bufferImageHeight = 0;
                    region.imageSubresource = {aspect, level, geometry.CopyLayer(layer), 1};
                    region.imageOffset = {0, 0, geometry.CopyDepth(layer)};
                    region.imageExtent = {std::max(descriptor.width >> level, 1u), std::max(descriptor.height >> level, 1u), 1u};
                    regions.push_back(region);
                }
            }
            context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, linear->Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()), regions.data());
            std::unique_ptr<Buffer> dump;
            if (dumpWanted) {
                dump = std::make_unique<Buffer>(context, static_cast<std::size_t>(mips[0].linearSize), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                CopyBuffer(context, commands, linear->Handle(), mips[0].linearOffset, dump->Handle(), 0, mips[0].linearSize);
                RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            }

            VkImageMemoryBarrier toShaderRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toShaderRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toShaderRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            toShaderRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toShaderRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toShaderRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toShaderRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toShaderRead.image = image;
            toShaderRead.subresourceRange = toTransferDst.subresourceRange;
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toShaderRead);

            if (batch) batch->SubmitAndWait();
            else ++Profile().recordedUploads;
            if (profile) Profile().gpu += timer.lap();
            if (dump) {
                static std::mutex dumpMutex;
                static std::map<std::uint64_t, int> dumped;
                std::lock_guard lock(dumpMutex);
                auto& count = dumped[descriptor.baseAddress];
                if (count < 8) {
                    char name[64];
                    std::snprintf(name, sizeof(name), "texture_%llx_%d.raw", static_cast<unsigned long long>(descriptor.baseAddress), count++);
                    if (std::FILE* file = std::fopen(name, "wb")) {
                        const std::uint32_t header[3] = {mips[0].pitchBytes / static_cast<std::uint32_t>(elementBytes), mips[0].height, static_cast<std::uint32_t>(vkFormat)};
                        std::fwrite(header, sizeof(header), 1, file);
                        std::fwrite(dump->Bytes().data(), 1, dump->Bytes().size(), file);
                        std::fclose(file);
                        std::fprintf(stderr, "[texture] dumped %s (%ux%u VkFormat %d, tile %d)\n", name, header[0], header[1], static_cast<int>(vkFormat), static_cast<int>(descriptor.tileMode));
                    }
                }
            }
        }

        const auto viewLevelCount = std::min(descriptor.lastLevel, descriptor.mipCount - 1u) - descriptor.baseLevel + 1u;
        const auto viewLayerCount = geometry.imageLayers - descriptor.baseArray;
        if (descriptor.dimension == TextureDimension::kCube) {
            Require(viewLayerCount % 6u == 0, "guest cube texture view does not contain a multiple of 6 array slices");
        }

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = ViewTypeFor(descriptor.dimension, viewLayerCount);
        viewInfo.format = vkFormat;
        viewFormat = vkFormat;
        viewInfo.components = depthCompare ? VkComponentMapping{} : components;
        viewInfo.subresourceRange = {aspect, descriptor.baseLevel, viewLevelCount, descriptor.baseArray, viewLayerCount};
        VkImageViewMinLodCreateInfoEXT minLod{VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT};
        ChainMinLod(context, descriptor, viewInfo, minLod);

        Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView");
        viewRange = {viewInfo.viewType, viewInfo.subresourceRange.levelCount, viewInfo.subresourceRange.layerCount};
        createFirstLayerView(descriptor, viewInfo);
        prepareBaseLevelView(descriptor, viewInfo);
        if (profile) {
            auto& totals = Profile();
            totals.view += timer.lap();
            if (++totals.count % 200 == 0) std::fprintf(stderr, "[texture] %llu textures (%llu copied from storage images, %llu uploads recorded): allocate+image %.0f ms, guest read %.0f ms, detile+copy %.0f ms, view+buffer release %.0f ms\n", static_cast<unsigned long long>(totals.count), static_cast<unsigned long long>(totals.fromStorage), static_cast<unsigned long long>(totals.recordedUploads), totals.allocate, totals.read, totals.gpu, totals.view);
        }
    } catch (...) {
        release();
        throw;
    }
}

bool Texture::CanCopyFrom(const StorageTexture& source, const GuestTextureResource& descriptor) {
    const auto& from = source.Descriptor();
    if (IsBlockCompressed(descriptor.format) || IsBlockCompressed(from.format)) return false;
    // Same memory, same layout, same texel size: the GPU copy reinterprets the texels exactly as a
    // guest read through the sampled descriptor would.
    return descriptor.baseAddress == from.baseAddress && descriptor.width == from.width && descriptor.height == from.height && (descriptor.dimension == from.dimension || (source.CubeCompatible() && (descriptor.dimension == TextureDimension::kCube || descriptor.dimension == TextureDimension::k2DArray) && (from.dimension == TextureDimension::kCube || from.dimension == TextureDimension::k2DArray))) && descriptor.tileMode == from.tileMode && descriptor.mipCount <= from.mipCount && descriptor.depthOrLastArray == from.depthOrLastArray && BytesPerElement(descriptor.format) == BytesPerElement(from.format) && BlockWidth(descriptor.format) == BlockWidth(from.format);
}

Texture::Texture(const Context& context, const std::shared_ptr<StorageTexture>& source, const GuestTextureResource& descriptor, VkComponentMapping components) : context(context), storageSource(source) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    PhaseTimer timer;
    try {
        Require(source != nullptr && CanCopyFrom(*source, descriptor), "storage image does not match the sampled texture");
        const auto vkFormat = SampledTextureFormat(context, descriptor.format);
        const auto geometry = DescribeSurface(descriptor);
        APS5_LOG_OUT_IF(TraceTextures(), "Texture address=0x%llx %ux%u mips=%u viewed from storage image (vk=%d)", static_cast<unsigned long long>(descriptor.baseAddress), descriptor.width, descriptor.height, descriptor.mipCount, static_cast<int>(vkFormat));
        // Storage images stay in the general layout; the view samples them there.
        layout = VK_IMAGE_LAYOUT_GENERAL;
        const auto viewLevelCount = std::min(descriptor.lastLevel, descriptor.mipCount - 1u) - descriptor.baseLevel + 1u;
        const auto viewLayerCount = std::min(geometry.imageLayers, source->ImageLayers()) - descriptor.baseArray;
        if (descriptor.dimension == TextureDimension::kCube) {
            Require(viewLayerCount % 6u == 0, "guest cube texture view does not contain a multiple of 6 array slices");
        }
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = source->Image();
        viewInfo.viewType = ViewTypeFor(descriptor.dimension, viewLayerCount);
        viewInfo.format = vkFormat;
        viewFormat = vkFormat;
        viewInfo.components = components;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, descriptor.baseLevel, viewLevelCount, descriptor.baseArray, viewLayerCount};
        VkImageViewMinLodCreateInfoEXT minLod{VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT};
        ChainMinLod(context, descriptor, viewInfo, minLod);
        Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView storage view");
        viewRange = {viewInfo.viewType, viewInfo.subresourceRange.levelCount, viewInfo.subresourceRange.layerCount};
        createFirstLayerView(descriptor, viewInfo);
        prepareBaseLevelView(descriptor, viewInfo);
        if (profile) {
            auto& totals = Profile();
            totals.view += timer.lap();
            ++totals.fromStorage;
            if (++totals.count % 200 == 0) std::fprintf(stderr, "[texture] %llu textures (%llu viewed from storage images): allocate+image %.0f ms, guest read %.0f ms, detile+copy %.0f ms, view+buffer release %.0f ms\n", static_cast<unsigned long long>(totals.count), static_cast<unsigned long long>(totals.fromStorage), totals.allocate, totals.read, totals.gpu, totals.view);
        }
    } catch (...) {
        release();
        throw;
    }
}

bool Texture::CanRepackCopyFrom(const StorageTexture& source, const GuestTextureResource& descriptor) {
    const auto& from = source.Descriptor();
    if (!IsRepackedTextureFormat(descriptor.format) || descriptor.dimension == TextureDimension::k3D || source.StorageFormat() != VK_FORMAT_R32_UINT) return false;
    if (descriptor.baseAddress != from.baseAddress || descriptor.width != from.width || descriptor.height != from.height || descriptor.tileMode != from.tileMode || descriptor.mipCount > from.mipCount || descriptor.depthOrLastArray != from.depthOrLastArray || BytesPerElement(from.format) != 4u) return false;
    if (descriptor.dimension != from.dimension && !(source.CubeCompatible() && (descriptor.dimension == TextureDimension::kCube || descriptor.dimension == TextureDimension::k2DArray) && (from.dimension == TextureDimension::kCube || from.dimension == TextureDimension::k2DArray))) return false;
    const auto geometry = DescribeSurface(descriptor);
    return geometry.imageDepth == 1u && source.ImageDepth() == 1u && source.ImageLayers() >= geometry.imageLayers;
}

Texture::Texture(const Context& context, const std::shared_ptr<StorageTexture>& source, const GuestTextureResource& descriptor, VkComponentMapping components, RepackCopyTag) : context(context), repackSource(source) {
    try {
        Require(source != nullptr && CanRepackCopyFrom(*source, descriptor), "storage image cannot serve a repacked copy of the sampled texture");
        Require(context.detiler != nullptr, "a repacked texture copy needs the texture detiler");
        // Every use goes through the lookup (no memoized or fast reuse), which re-copies the image
        // once the GPU wrote it: guest memory does not change when it does.
        refreshedPerUse = true;
        const auto geometry = DescribeSurface(descriptor);
        copyWidth = descriptor.width;
        copyHeight = descriptor.height;
        copyLevels = descriptor.mipCount;
        copyLayers = geometry.imageLayers;
        APS5_LOG_OUT_IF(TraceTextures(), "Texture address=0x%llx %ux%u mips=%u layers=%u repacked copy of storage image", static_cast<unsigned long long>(descriptor.baseAddress), descriptor.width, descriptor.height, descriptor.mipCount, copyLayers);

        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.flags = descriptor.dimension == TextureDimension::kCube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
        imageInfo.imageType = ImageTypeFor(descriptor.dimension);
        imageInfo.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        imageInfo.extent = {descriptor.width, descriptor.height, 1u};
        imageInfo.mipLevels = copyLevels;
        imageInfo.arrayLayers = copyLayers;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &imageInfo, nullptr, &image), "vkCreateImage repacked copy");
        owned = std::make_shared<OwnedImage>(context, image, VK_NULL_HANDLE);
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocationBytes = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &owned->memory), "vkAllocateMemory repacked copy");
        Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, owned->memory, 0), "vkBindImageMemory repacked copy");

        const auto viewLevelCount = std::min(descriptor.lastLevel, descriptor.mipCount - 1u) - descriptor.baseLevel + 1u;
        const auto viewLayerCount = copyLayers - descriptor.baseArray;
        if (descriptor.dimension == TextureDimension::kCube) {
            Require(viewLayerCount % 6u == 0, "guest cube texture view does not contain a multiple of 6 array slices");
        }
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = ViewTypeFor(descriptor.dimension, viewLayerCount);
        viewInfo.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        viewFormat = viewInfo.format;
        viewInfo.components = components;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, descriptor.baseLevel, viewLevelCount, descriptor.baseArray, viewLayerCount};
        VkImageViewMinLodCreateInfoEXT minLod{VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT};
        ChainMinLod(context, descriptor, viewInfo, minLod);
        Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView repacked copy");
        viewRange = {viewInfo.viewType, viewInfo.subresourceRange.levelCount, viewInfo.subresourceRange.layerCount};
        createFirstLayerView(descriptor, viewInfo);
        prepareBaseLevelView(descriptor, viewInfo);
        RefreshRepackCopy();
    } catch (...) {
        release();
        throw;
    }
}

void Texture::RefreshRepackCopy() {
    if (repackSource == nullptr || (repackedOnce && repackedVersion == repackSource->Version())) return;
    static std::atomic<std::uint64_t> copies{0};
    static const bool traceCopies = std::getenv("APS5_TRACE_REPACK_COPIES") != nullptr;
    const auto copied = copies.fetch_add(1, std::memory_order_relaxed) + 1u;
    if (traceCopies && copied % 500u == 0u) std::fprintf(stderr, "[repack-copy] %llu format-43 copies of storage images\n", static_cast<unsigned long long>(copied));
    // The copy goes into the open batch like a texture upload, ahead of the work that samples it.
    auto* recorder = Recorder::Active();
    std::unique_ptr<CommandBatch> batch;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    std::uint64_t linearBytes = 0;
    for (std::uint32_t level = 0; level < copyLevels; ++level) linearBytes += static_cast<std::uint64_t>(std::max(copyWidth >> level, 1u)) * std::max(copyHeight >> level, 1u) * 4u * copyLayers;
    auto linear = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(linearBytes), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (recorder != nullptr) {
        commands = recorder->Commands();
        recorder->Keep(linear, static_cast<std::size_t>(linearBytes));
        recorder->Keep(owned);
    } else {
        batch = std::make_unique<CommandBatch>(context);
        commands = batch->Handle();
    }
    std::vector<VkBufferImageCopy> regions;
    std::uint64_t offset = 0;
    for (std::uint32_t level = 0; level < copyLevels; ++level) {
        VkBufferImageCopy region{};
        region.bufferOffset = offset;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, copyLayers};
        region.imageExtent = {std::max(copyWidth >> level, 1u), std::max(copyHeight >> level, 1u), 1u};
        regions.push_back(region);
        offset += static_cast<std::uint64_t>(region.imageExtent.width) * region.imageExtent.height * 4u * copyLayers;
    }
    const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    // The storage image stays in the general layout: wait for whatever wrote it.
    VkImageMemoryBarrier produced{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    produced.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    produced.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    produced.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    produced.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    produced.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    produced.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    produced.image = repackSource->Image();
    produced.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, copyLevels, 0, copyLayers};
    barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &produced);
    context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, repackSource->Image(), VK_IMAGE_LAYOUT_GENERAL, linear->Handle(), static_cast<std::uint32_t>(regions.size()), regions.data());
    // Later writes of the storage image wait for the copy's reads.
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    for (const auto& region : regions) {
        const auto bytes = static_cast<std::uint64_t>(region.imageExtent.width) * region.imageExtent.height * 4u * copyLayers;
        context.detiler->DispatchRepack(commands, linear->Handle(), region.bufferOffset, linear->Handle(), region.bufferOffset, bytes);
    }
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkImageMemoryBarrier toTransferDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toTransferDst.srcAccessMask = repackedOnce ? VK_ACCESS_SHADER_READ_BIT : 0u;
    toTransferDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransferDst.oldLayout = repackedOnce ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    toTransferDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDst.image = image;
    toTransferDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, copyLevels, 0, copyLayers};
    barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toTransferDst);
    context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, linear->Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()), regions.data());
    VkImageMemoryBarrier toShaderRead = toTransferDst;
    toShaderRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toShaderRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toShaderRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toShaderRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toShaderRead);
    if (batch) batch->SubmitAndWait();
    repackedOnce = true;
    repackedVersion = repackSource->Version();
}

Texture::Texture(const Context& context, VkImage depthImage, VkFormat depthFormat, VkImageAspectFlags aspect, VkComponentMapping components, VkImageViewType viewType) : context(context) {
    layout = VK_IMAGE_LAYOUT_GENERAL;
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = depthImage;
    viewInfo.viewType = viewType;
    viewInfo.format = depthFormat;
    viewFormat = depthFormat;
    viewInfo.components = components;
    viewInfo.subresourceRange = {aspect, 0, 1, 0, viewType == VK_IMAGE_VIEW_TYPE_2D ? 1u : VK_REMAINING_ARRAY_LAYERS};
    Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView depth plane");
    viewRange = {viewInfo.viewType, viewInfo.subresourceRange.levelCount, viewInfo.subresourceRange.layerCount};
}

Texture::~Texture() {
    release();
}

void Texture::createFirstLayerView(const GuestTextureResource& descriptor, VkImageViewCreateInfo viewInfo) {
    if (descriptor.dimension != TextureDimension::k2DArray) return;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.subresourceRange.layerCount = 1;
    Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &firstLayerView), "vkCreateImageView first layer");
    firstLayerRange = {viewInfo.viewType, viewInfo.subresourceRange.levelCount, viewInfo.subresourceRange.layerCount};
}

void Texture::prepareBaseLevelView(const GuestTextureResource& descriptor, VkImageViewCreateInfo viewInfo) {
    if (viewInfo.subresourceRange.levelCount == 1u) return;
    if (descriptor.dimension != TextureDimension::k1D && descriptor.dimension != TextureDimension::k2D && descriptor.dimension != TextureDimension::k2DArray) return;
    viewInfo.pNext = nullptr;
    viewInfo.viewType = descriptor.dimension == TextureDimension::k1D ? VK_IMAGE_VIEW_TYPE_1D : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    baseLevelInfo = viewInfo;
}

VkImageView Texture::BaseLevelView() const {
    Require(baseLevelInfo.has_value(), "a base-level view needs a mipmapped 1D, 2D or 2D array texture");
    std::lock_guard lock(baseLevelMutex);
    if (!baseLevelView) Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &*baseLevelInfo, nullptr, &baseLevelView), "vkCreateImageView base level");
    return baseLevelView;
}

void Texture::release() noexcept {
    upload.reset();
    if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
    view = VK_NULL_HANDLE;
    if (firstLayerView) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, firstLayerView, nullptr);
    firstLayerView = VK_NULL_HANDLE;
    if (baseLevelView) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, baseLevelView, nullptr);
    baseLevelView = VK_NULL_HANDLE;
    // The image and its memory go with the last holder: this texture, or the batch still uploading it.
    owned.reset();
    image = VK_NULL_HANDLE;
}

VkImageView Texture::View() const {
    return view;
}


namespace {

VkBufferMemoryBarrier WholeBufferBarrier(VkBuffer buffer, VkAccessFlags from, VkAccessFlags to) {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = from;
    barrier.dstAccessMask = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
    return barrier;
}

// Storage images cannot use sRGB formats; the shader works on the raw encoded values either way.
// The answer is a property of the physical device, so it is computed once per format (every storage
// image lookup asks: a live vkGetPhysicalDeviceFormatProperties call each time was measurable) and
// remembered as VK_FORMAT_UNDEFINED when the format has no storage form.
VkFormat StorageFormatOrUndefined(const Context& context, VkFormat format) {
    struct Table {
        std::mutex mutex;
        std::unordered_map<std::uint64_t, VkFormat> formats;
    };
    static Table table;
    // The key names the physical device: the headless and the windowed device share one GPU, but a
    // second GPU would answer differently.
    const auto key = (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(context.physical)) << 20u) ^ static_cast<std::uint64_t>(format);
    {
        std::lock_guard lock(table.mutex);
        if (const auto found = table.formats.find(key); found != table.formats.end()) return found->second;
    }
    const auto supports = [&](VkFormat candidate) {
        VkFormatProperties properties{};
        context.formatProperties(context.physical, candidate, &properties);
        return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
    };
    VkFormat storage = VK_FORMAT_UNDEFINED;
    if (supports(format)) {
        storage = format;
    } else {
        VkFormat linear = VK_FORMAT_UNDEFINED;
        switch (format) {
            case VK_FORMAT_R8G8B8A8_SRGB: linear = VK_FORMAT_R8G8B8A8_UNORM; break;
            case VK_FORMAT_B8G8R8A8_SRGB: linear = VK_FORMAT_B8G8R8A8_UNORM; break;
            case VK_FORMAT_A8B8G8R8_SRGB_PACK32: linear = VK_FORMAT_A8B8G8R8_UNORM_PACK32; break;
            case VK_FORMAT_R8_SRGB: linear = VK_FORMAT_R8_UNORM; break;
            case VK_FORMAT_R8G8_SRGB: linear = VK_FORMAT_R8G8_UNORM; break;
            default: break;
        }
        if (linear != VK_FORMAT_UNDEFINED && supports(linear)) storage = linear;
    }
    std::lock_guard lock(table.mutex);
    table.formats.emplace(key, storage);
    return storage;
}

VkFormat UintFormatOfSint(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8G8_SINT: return VK_FORMAT_R8G8_UINT;
        case VK_FORMAT_R8G8B8A8_SINT: return VK_FORMAT_R8G8B8A8_UINT;
        case VK_FORMAT_R16_SINT: return VK_FORMAT_R16_UINT;
        case VK_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_UINT;
        case VK_FORMAT_R16G16B16A16_SINT: return VK_FORMAT_R16G16B16A16_UINT;
        case VK_FORMAT_R32_SINT: return VK_FORMAT_R32_UINT;
        case VK_FORMAT_R32G32_SINT: return VK_FORMAT_R32G32_UINT;
        case VK_FORMAT_R32G32B32A32_SINT: return VK_FORMAT_R32G32B32A32_UINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

VkFormat StorageFormatFor(const Context& context, VkFormat format) {
    const auto storage = StorageFormatOrUndefined(context, format);
    if (storage == VK_FORMAT_UNDEFINED) Require(false, "guest storage texture format " + std::to_string(format) + " cannot be used as a storage image");
    return storage;
}

}

bool StorageFormatAvailable(const Context& context, std::uint32_t guestFormat) {
    try {
        return StorageFormatOrUndefined(context, ResolveTextureFormat(guestFormat)) != VK_FORMAT_UNDEFINED;
    } catch (const std::exception&) {
        // An unknown guest format: the sampled texture path reports it when it gets there.
        return false;
    }
}

// Integer formats take integer clear values; the DCC clear codes are only mapped for the others.
bool IntegerFormat(VkFormat format) {
    switch (format) {
        case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
        case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_B8G8R8A8_SINT:
        case VK_FORMAT_A8B8G8R8_UINT_PACK32: case VK_FORMAT_A8B8G8R8_SINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT:
        case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT:
        case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R64_UINT: case VK_FORMAT_R64_SINT:
            return true;
        default: return false;
    }
}

// The clear value a DCC clear code stands for, for non-integer formats.
bool ClearColorFor(VkFormat format, DccKeys keys, VkClearColorValue& clear) {
    if (IntegerFormat(format)) return false;
    switch (keys) {
        case DccKeys::Clear0000: clear.float32[0] = clear.float32[1] = clear.float32[2] = clear.float32[3] = 0.0f; return true;
        case DccKeys::Clear0001: clear.float32[0] = clear.float32[1] = clear.float32[2] = 0.0f; clear.float32[3] = 1.0f; return true;
        case DccKeys::Clear1110: clear.float32[0] = clear.float32[1] = clear.float32[2] = 1.0f; clear.float32[3] = 0.0f; return true;
        case DccKeys::Clear1111: clear.float32[0] = clear.float32[1] = clear.float32[2] = clear.float32[3] = 1.0f; return true;
        default: return false;
    }
}

VkFormat StorageFormatForGuest(const Context& context, std::uint32_t guestFormat) {
    return StorageFormatFor(context, ResolveTextureFormat(guestFormat));
}

bool StorageClearAvailable(const Context& context, std::uint32_t guestFormat, DccKeys keys) {
    VkClearColorValue clear{};
    return IsDccClear(keys) && StorageFormatAvailable(context, guestFormat) && ClearColorFor(StorageFormatForGuest(context, guestFormat), keys, clear);
}

VkFormat AttachmentProxyFormat(const Context& context, VkFormat format) {
    if (format != VK_FORMAT_R8_SRGB) return VK_FORMAT_UNDEFINED;
    struct Table {
        std::mutex mutex;
        std::unordered_map<VkPhysicalDevice, VkFormat> formats;
    };
    static Table table;
    {
        std::lock_guard lock(table.mutex);
        if (const auto found = table.formats.find(context.physical); found != table.formats.end()) return found->second;
    }
    const char* forced = std::getenv("APS5_SRGB_ATTACHMENT_PROXY");
    VkFormatProperties properties{};
    context.formatProperties(context.physical, format, &properties);
    constexpr VkFormatFeatureFlags attachment = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
    const bool proxied = (forced != nullptr && std::strcmp(forced, "1") == 0) || (properties.optimalTilingFeatures & attachment) != attachment;
    const auto proxy = proxied ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_UNDEFINED;
    std::lock_guard lock(table.mutex);
    table.formats.emplace(context.physical, proxy);
    return proxy;
}

StorageTexture::StorageTexture(const Context& context, TextureDetiler& detiler, const GuestTextureResource& descriptor, std::uint32_t mipLevel) : context(context), detiler(detiler), descriptor(descriptor) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    PhaseTimer timer;
    try {
        Require(!IsBlockCompressed(descriptor.format), "block-compressed textures cannot be storage images");
        Require(mipLevel < descriptor.mipCount, "storage texture mip level is outside the texture");
        const auto vkFormat = StorageFormatFor(context, ResolveTextureFormat(descriptor.format));
        storageFormat = vkFormat;
        APS5_LOG_OUT_IF(TraceTextures(), "StorageTexture address=0x%llx %ux%u mips=%u mip=%u layers=%u base=%u dim=%d tile=%d format=%u vk=%d", static_cast<unsigned long long>(descriptor.baseAddress), descriptor.width, descriptor.height, descriptor.mipCount, mipLevel,
                     descriptor.depthOrLastArray, descriptor.baseArray, static_cast<int>(descriptor.dimension), static_cast<int>(descriptor.tileMode), descriptor.format, static_cast<int>(vkFormat));
        geometry = DescribeSurface(descriptor);
        mips = geometry.mips;
        arrayLayers = geometry.layers;
        const auto elementBytes = BytesPerElement(descriptor.format);
        guestBytes = geometry.guestBytes;
        // Storage images in heaps the guest commits on demand only read and store committed pages.
        Require(!GuestMemory::CommittedRanges(descriptor.baseAddress, static_cast<std::size_t>(guestBytes), true).empty(), "storage texture has no committed guest pages");
        // Write stamps are per 64 KiB block, so only 64 KiB-aligned layer slices can be told apart.
        const bool layered = geometry.layers >= 2 && !geometry.thick && geometry.imageDepth == 1 && geometry.layers == geometry.imageLayers && geometry.layerBytes * geometry.layers == guestBytes && geometry.layerBytes % 65536 == 0 && descriptor.baseAddress % 65536 == 0;
        // Tile blocks divide the stamp blocks, so at an aligned base every unit is whole tile blocks
        // of its (layer, mip) slices (sliceWindows).
        // A thick surface of one mip (no tail block) takes block units under `lazy_writeback`: its
        // windows are whole tile blocks of a slab's rows, every depth slice of the slab one window.
        const bool thickBlocks = geometry.thick && descriptor.mipCount == 1 && !mips.empty() && !mips[0].tail && LazyWriteBack();
        blockUnits = LayerTrackingEnabled() && BlockTrackingEnabled() && (!geometry.thick || thickBlocks) && descriptor.tileMode != TextureTileMode::kLinear && descriptor.baseAddress % 65536 == 0;
        // APS5_THICK_UNITS=1 (experiment): a thick (3D) surface tracks its 64 KiB blocks as units
        // too, without the windowed block paths: a write-back retiles the whole image on the GPU but
        // stores only the requested units into guest memory, the others staying pending (uploads
        // stay whole, after every pending unit is stored).
        static const bool thickUnits = std::getenv("APS5_THICK_UNITS") != nullptr;
        const bool unitTracked = blockUnits || (thickUnits && LayerTrackingEnabled() && geometry.thick && descriptor.tileMode != TextureTileMode::kLinear && descriptor.baseAddress % 65536 == 0);
        trackedLayers = unitTracked ? static_cast<std::uint32_t>((guestBytes + 65535) / 65536) : layered && LayerTrackingEnabled() ? geometry.layers : 1u;
        trackedLayerBytes = unitTracked ? 65536 : guestBytes / trackedLayers;
        layerGeneration.assign(trackedLayers, 0);
        layerPending.assign(trackedLayers, false);
        sliceLinearBytes = geometry.sliceLinearBytes;
        const auto linearBytes = sliceLinearBytes * arrayLayers;

        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        // Sampled views of other same-size formats (sRGB, reinterpretations) read the image directly.
        cubeCompatible = descriptor.dimension == TextureDimension::kCube || (descriptor.dimension == TextureDimension::k2DArray && descriptor.width == descriptor.height && geometry.imageLayers % 6u == 0u);
        imageInfo.flags = (cubeCompatible ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u) | (descriptor.dimension == TextureDimension::k3D ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0u) | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        imageInfo.imageType = ImageTypeFor(descriptor.dimension);
        imageInfo.format = vkFormat;
        imageInfo.extent = {descriptor.width, descriptor.height, geometry.imageDepth};
        imageInfo.mipLevels = descriptor.mipCount;
        imageInfo.arrayLayers = geometry.imageLayers;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        {
            VkFormatProperties properties{};
            context.formatProperties(context.physical, vkFormat, &properties);
            attachable = (descriptor.dimension == TextureDimension::k2D || descriptor.dimension == TextureDimension::k2DArray || descriptor.dimension == TextureDimension::k3D) && (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0;
            if (attachable) imageInfo.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        }
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &imageInfo, nullptr, &image), "vkCreateImage storage");
        if (AddressLedger::Enabled()) {
            char name[112];
            std::snprintf(name, sizeof(name), "storage 0x%llx %ux%ux%u layers %u mips %u fmt %d", static_cast<unsigned long long>(descriptor.baseAddress), imageInfo.extent.width, imageInfo.extent.height, imageInfo.extent.depth, imageInfo.arrayLayers, imageInfo.mipLevels, static_cast<int>(imageInfo.format));
            AddressLedger::NameObject(reinterpret_cast<std::uint64_t>(image), name);
        }
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory storage texture");
        Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory storage");
        uploadReason = "first";
        upload();
        defaultMip = mipLevel;
        view = createView(mipLevel, false, storageFormat);
        {
            auto& live = Live();
            std::lock_guard lock(live.mutex);
            live.textures.push_back(this);
        }
        if (profile) {
            auto& totals = Profile();
            totals.storageCreate += timer.lap();
            totals.storageBytes += guestBytes;
            if (++totals.storageCount % 50 == 0) std::fprintf(stderr, "[texture] %llu storage images (%.0f MiB): create %.0f ms, write-back %.0f ms (alloc %.0f, host copy %.0f, gpu %.0f, store %.0f), %llu reused, %llu direct uploads, %llu direct write-backs, clears %llu recorded / %llu waited\n", static_cast<unsigned long long>(totals.storageCount), totals.storageBytes / 1048576.0, totals.storageCreate, totals.storageWriteBack, totals.storageAlloc, totals.storageHostCopy, totals.storageGpu, totals.storageStore, static_cast<unsigned long long>(totals.storageReused), static_cast<unsigned long long>(totals.storageDirectUploads), static_cast<unsigned long long>(totals.storageDirectWriteBacks), static_cast<unsigned long long>(totals.storageRecordedClears), static_cast<unsigned long long>(totals.storageWaitedClears));
        }
    } catch (...) {
        release();
        throw;
    }
}

namespace {

class PendingList {
public:
    using iterator = std::vector<StorageTexture*>::iterator;
    iterator begin() { return textures.begin(); }
    iterator end() { return textures.end(); }
    void push_back(StorageTexture* texture) {
        textures.push_back(texture);
        ++version;
    }
    iterator erase(iterator it) {
        ++version;
        return textures.erase(it);
    }
    void remove(const StorageTexture* texture) {
        if (std::erase(textures, texture) != 0) ++version;
    }
    bool MayOverlap(std::uint64_t address, std::size_t bytes) {
        if (textures.empty() || bytes == 0) return false;
        if (indexed != version) rebuild();
        const auto end = bytes > std::numeric_limits<std::uint64_t>::max() - address ? std::numeric_limits<std::uint64_t>::max() : address + bytes;
        const auto it = std::upper_bound(ranges.begin(), ranges.end(), address, [](std::uint64_t value, const std::pair<std::uint64_t, std::uint64_t>& range) { return value < range.second; });
        return it != ranges.end() && it->first < end;
    }

private:
    void rebuild() {
        ranges.clear();
        for (const auto* texture : textures) ranges.emplace_back(texture->Descriptor().baseAddress, texture->Descriptor().baseAddress + texture->GuestBytes());
        std::sort(ranges.begin(), ranges.end());
        std::size_t merged = 0;
        for (const auto& range : ranges) {
            if (merged != 0 && range.first <= ranges[merged - 1].second) ranges[merged - 1].second = std::max(ranges[merged - 1].second, range.second);
            else ranges[merged++] = range;
        }
        ranges.resize(merged);
        indexed = version;
    }
    std::vector<StorageTexture*> textures;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::uint64_t version = 0;
    std::uint64_t indexed = ~std::uint64_t{0};
};

// Storage images whose results have not reached guest memory yet.
struct PendingWrites {
    SlimMutex mutex;
    PendingList textures;
    // Images taken out of `textures` by a FlushPending still storing them (see adjacentPendingUnchanged).
    std::vector<StorageTexture*> flushing;
};

PendingWrites& Pending() {
    static PendingWrites pending;
    return pending;
}

// See StorageTexture::PendingSerial: bumped after every mutation of the registry above.
std::atomic<std::uint64_t> pendingSerial{0};

// The texel layout a clear value is built from (FillClearColor): channels in memory order with
// their bit offset and width inside the element and the clear component they feed (R, G, B, A).
enum class ClearKind { Unorm, Snorm, Uint, Sint, Float, UFloat };

struct ClearChannel {
    std::uint32_t offset;
    std::uint32_t bits;
    std::uint32_t component;
};

struct ClearLayout {
    ClearKind kind;
    std::uint32_t channels;
    ClearChannel channel[4];
};

bool ClearLayoutFor(VkFormat format, ClearLayout& layout) {
    const auto uniform = [&](ClearKind kind, std::uint32_t count, std::uint32_t width, bool reversed = false) {
        layout.kind = kind;
        layout.channels = count;
        for (std::uint32_t i = 0; i < count; ++i) layout.channel[i] = {i * width, width, reversed && i < 3 ? 2u - i : i};
    };
    switch (format) {
        case VK_FORMAT_R8_UNORM: uniform(ClearKind::Unorm, 1, 8); return true;
        case VK_FORMAT_R8_SNORM: uniform(ClearKind::Snorm, 1, 8); return true;
        case VK_FORMAT_R8_UINT: uniform(ClearKind::Uint, 1, 8); return true;
        case VK_FORMAT_R8_SINT: uniform(ClearKind::Sint, 1, 8); return true;
        case VK_FORMAT_R8G8_UNORM: uniform(ClearKind::Unorm, 2, 8); return true;
        case VK_FORMAT_R8G8_SNORM: uniform(ClearKind::Snorm, 2, 8); return true;
        case VK_FORMAT_R8G8_UINT: uniform(ClearKind::Uint, 2, 8); return true;
        case VK_FORMAT_R8G8_SINT: uniform(ClearKind::Sint, 2, 8); return true;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_A8B8G8R8_UNORM_PACK32: uniform(ClearKind::Unorm, 4, 8); return true;
        case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_A8B8G8R8_SNORM_PACK32: uniform(ClearKind::Snorm, 4, 8); return true;
        case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_A8B8G8R8_UINT_PACK32: uniform(ClearKind::Uint, 4, 8); return true;
        case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_A8B8G8R8_SINT_PACK32: uniform(ClearKind::Sint, 4, 8); return true;
        case VK_FORMAT_B8G8R8A8_UNORM: uniform(ClearKind::Unorm, 4, 8, true); return true;
        case VK_FORMAT_B8G8R8A8_SNORM: uniform(ClearKind::Snorm, 4, 8, true); return true;
        case VK_FORMAT_B8G8R8A8_UINT: uniform(ClearKind::Uint, 4, 8, true); return true;
        case VK_FORMAT_B8G8R8A8_SINT: uniform(ClearKind::Sint, 4, 8, true); return true;
        case VK_FORMAT_R16_UNORM: uniform(ClearKind::Unorm, 1, 16); return true;
        case VK_FORMAT_R16_SNORM: uniform(ClearKind::Snorm, 1, 16); return true;
        case VK_FORMAT_R16_UINT: uniform(ClearKind::Uint, 1, 16); return true;
        case VK_FORMAT_R16_SINT: uniform(ClearKind::Sint, 1, 16); return true;
        case VK_FORMAT_R16_SFLOAT: uniform(ClearKind::Float, 1, 16); return true;
        case VK_FORMAT_R16G16_UNORM: uniform(ClearKind::Unorm, 2, 16); return true;
        case VK_FORMAT_R16G16_SNORM: uniform(ClearKind::Snorm, 2, 16); return true;
        case VK_FORMAT_R16G16_UINT: uniform(ClearKind::Uint, 2, 16); return true;
        case VK_FORMAT_R16G16_SINT: uniform(ClearKind::Sint, 2, 16); return true;
        case VK_FORMAT_R16G16_SFLOAT: uniform(ClearKind::Float, 2, 16); return true;
        case VK_FORMAT_R16G16B16A16_UNORM: uniform(ClearKind::Unorm, 4, 16); return true;
        case VK_FORMAT_R16G16B16A16_SNORM: uniform(ClearKind::Snorm, 4, 16); return true;
        case VK_FORMAT_R16G16B16A16_UINT: uniform(ClearKind::Uint, 4, 16); return true;
        case VK_FORMAT_R16G16B16A16_SINT: uniform(ClearKind::Sint, 4, 16); return true;
        case VK_FORMAT_R16G16B16A16_SFLOAT: uniform(ClearKind::Float, 4, 16); return true;
        case VK_FORMAT_R32_UINT: uniform(ClearKind::Uint, 1, 32); return true;
        case VK_FORMAT_R32_SINT: uniform(ClearKind::Sint, 1, 32); return true;
        case VK_FORMAT_R32_SFLOAT: uniform(ClearKind::Float, 1, 32); return true;
        case VK_FORMAT_R32G32_UINT: uniform(ClearKind::Uint, 2, 32); return true;
        case VK_FORMAT_R32G32_SINT: uniform(ClearKind::Sint, 2, 32); return true;
        case VK_FORMAT_R32G32_SFLOAT: uniform(ClearKind::Float, 2, 32); return true;
        case VK_FORMAT_R32G32B32A32_UINT: uniform(ClearKind::Uint, 4, 32); return true;
        case VK_FORMAT_R32G32B32A32_SINT: uniform(ClearKind::Sint, 4, 32); return true;
        case VK_FORMAT_R32G32B32A32_SFLOAT: uniform(ClearKind::Float, 4, 32); return true;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: layout = {ClearKind::Unorm, 4, {{0, 10, 0}, {10, 10, 1}, {20, 10, 2}, {30, 2, 3}}}; return true;
        case VK_FORMAT_A2B10G10R10_UINT_PACK32: layout = {ClearKind::Uint, 4, {{0, 10, 0}, {10, 10, 1}, {20, 10, 2}, {30, 2, 3}}}; return true;
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32: layout = {ClearKind::Unorm, 4, {{0, 10, 2}, {10, 10, 1}, {20, 10, 0}, {30, 2, 3}}}; return true;
        case VK_FORMAT_A2R10G10B10_UINT_PACK32: layout = {ClearKind::Uint, 4, {{0, 10, 2}, {10, 10, 1}, {20, 10, 0}, {30, 2, 3}}}; return true;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: layout = {ClearKind::UFloat, 3, {{0, 11, 0}, {11, 11, 1}, {22, 10, 2}}}; return true;
        default: return false;
    }
}

// A float of `bits` (16: IEEE half; 11 and 10: the unsigned 5-bit-exponent floats of
// B10G11R11) as a float32, or nullopt for a NaN, a denormal or (unsigned) an infinity.
std::optional<float> SmallFloat(std::uint32_t code, std::uint32_t bits) {
    const std::uint32_t mantissaBits = bits == 16 ? 10u : bits - 5u;
    const bool negative = bits == 16 && (code >> 15u) != 0;
    const std::uint32_t exponent = (code >> mantissaBits) & 0x1fu;
    const std::uint32_t mantissa = code & ((1u << mantissaBits) - 1u);
    if (exponent == 31u) {
        if (mantissa != 0 || bits != 16) return std::nullopt;
        return negative ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
    }
    if (exponent == 0) {
        if (mantissa != 0) return std::nullopt;
        return negative ? -0.0f : 0.0f;
    }
    const float value = std::ldexp(1.0f + static_cast<float>(mantissa) / static_cast<float>(1u << mantissaBits), static_cast<int>(exponent) - 15);
    return negative ? -value : value;
}

// One texel of `format` (`elementBytes` wide) taken from a 16-byte fill pattern as the clear value
// that stores exactly those bits, when the pattern is that texel repeated: integer formats take the
// codes as they are, normalized ones the code over the channel's maximum (the conversion back rounds
// to the same code), float ones the decoded value. Codes a clear might store differently are
// refused: NaNs and denormals (an implementation may canonicalize or flush them) and the most
// negative SNORM code (stored as its neighbour, the same -1.0).
bool FillClearColor(VkFormat format, std::uint32_t elementBytes, std::span<const std::uint32_t, 4> pattern, VkClearColorValue& clear) {
    ClearLayout layout{};
    if (elementBytes == 0 || elementBytes > 16 || 16 % elementBytes != 0 || !ClearLayoutFor(format, layout)) return false;
    std::array<std::byte, 16> bytes{};
    std::memcpy(bytes.data(), pattern.data(), bytes.size());
    for (std::size_t at = elementBytes; at < bytes.size(); at += elementBytes) {
        if (std::memcmp(bytes.data(), bytes.data() + at, elementBytes) != 0) return false;
    }
    for (std::uint32_t i = 0; i < layout.channels; ++i) {
        const auto& channel = layout.channel[i];
        const auto word = pattern[channel.offset / 32u];
        const auto code = channel.bits == 32 ? word : (word >> (channel.offset % 32u)) & ((1u << channel.bits) - 1u);
        const auto signedCode = static_cast<std::int32_t>(code << (32u - channel.bits)) >> (32u - channel.bits);
        switch (layout.kind) {
            case ClearKind::Unorm:
                clear.float32[channel.component] = static_cast<float>(code) / static_cast<float>((1u << channel.bits) - 1u);
                break;
            case ClearKind::Snorm: {
                const auto maximum = static_cast<std::int32_t>((1u << (channel.bits - 1u)) - 1u);
                if (signedCode < -maximum) return false;
                clear.float32[channel.component] = static_cast<float>(signedCode) / static_cast<float>(maximum);
                break;
            }
            case ClearKind::Uint:
                clear.uint32[channel.component] = code;
                break;
            case ClearKind::Sint:
                clear.int32[channel.component] = signedCode;
                break;
            case ClearKind::Float:
                if (channel.bits == 32) {
                    const auto exponent = (code >> 23u) & 0xffu;
                    const auto mantissa = code & 0x7fffffu;
                    if ((exponent == 0xffu && mantissa != 0) || (exponent == 0 && mantissa != 0)) return false;
                    clear.float32[channel.component] = std::bit_cast<float>(code);
                } else {
                    const auto value = SmallFloat(code, channel.bits);
                    if (!value) return false;
                    clear.float32[channel.component] = *value;
                }
                break;
            case ClearKind::UFloat: {
                const auto value = SmallFloat(code, channel.bits);
                if (!value) return false;
                clear.float32[channel.component] = *value;
                break;
            }
        }
    }
    return true;
}

// The image being validated by Refresh: its own pending results are what the next dispatch wants,
// so the comparison with guest memory must not flush them.
thread_local const StorageTexture* refreshing = nullptr;
// Layer aliases Refresh takes on the device (pendingLayerAliases): their pending results stay
// pending through its flush, like the alias passed as `except`.
thread_local const std::vector<const StorageTexture*>* exemptLayerAliases = nullptr;

bool LayerAliasExempted(const StorageTexture* texture) {
    return exemptLayerAliases != nullptr && std::find(exemptLayerAliases->begin(), exemptLayerAliases->end(), texture) != exemptLayerAliases->end();
}

void FlushHook(std::uint64_t address, std::size_t bytes) {
    StorageTexture::FlushPending(address, bytes);
}

// Write stamps are per 64 KiB block (GuestMemory::UnchangedSince), so a write-back's own MarkWritten
// stamps the blocks its surface shares with an adjacent surface (video planes packed back to back:
// the luma plane's last block is the chroma plane's first). A pending image of that adjacent surface
// would take the stamp for a CPU write since its generation and, at its own write-back, keep the
// guest bytes of the shared block in place of its results (a stale band at the plane's start, zero
// chroma on a fresh buffer). So a write-back first finds the adjacent pending images unchanged since
// their generation and, once its stamps are made, advances them past the stamps: nothing of theirs
// changed. APS5_NO_ADJACENT_GENERATION=1 leaves them at their generation, as before.
bool AdjacentGenerationEnabled() {
    static const bool disabled = std::getenv("APS5_NO_ADJACENT_GENERATION") != nullptr;
    return !disabled;
}

}

VkImageView StorageTexture::createView(std::uint32_t mip, bool firstLayer, VkFormat format, std::uint32_t layer) const {
    Require(mip < descriptor.mipCount, "storage texture mip level is outside the texture");
    Require(!firstLayer || descriptor.dimension == TextureDimension::k2DArray, "a first-layer storage view needs a 2D array surface");
    const auto baseLayer = descriptor.baseArray + layer;
    Require(baseLayer < geometry.imageLayers, "storage view base layer is outside the texture");
    const auto viewLayerCount = firstLayer ? 1u : geometry.imageLayers - baseLayer;
    VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    usage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.pNext = format == storageFormat ? nullptr : &usage;
    viewInfo.image = image;
    // Storage views address one mip; cube faces are written as array layers.
    viewInfo.viewType = firstLayer ? VK_IMAGE_VIEW_TYPE_2D : descriptor.dimension == TextureDimension::k1D ? VK_IMAGE_VIEW_TYPE_1D : descriptor.dimension == TextureDimension::k2D ? VK_IMAGE_VIEW_TYPE_2D : descriptor.dimension == TextureDimension::k3D ? VK_IMAGE_VIEW_TYPE_3D : descriptor.dimension == TextureDimension::k1DArray ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = format;
    viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1u, baseLayer, viewLayerCount};
    VkImageView created = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &created), "vkCreateImageView storage");
    return created;
}

VkImageView StorageTexture::AttachmentView(VkFormat format, std::uint32_t mip, std::uint32_t depthSlice, std::uint32_t layers) {
    Require(attachable, "storage image cannot be a color attachment");
    Require(mip < descriptor.mipCount, "attachment mip exceeds the storage image");
    const bool volume = descriptor.dimension == TextureDimension::k3D;
    // A 3D image's slices exist at mip 0 only here; an array layer has every mip.
    Require((depthSlice == 0 && layers == 1) || (layers != 0 && depthSlice + layers - 1u <= descriptor.depthOrLastArray && ((volume && mip == 0) || descriptor.dimension == TextureDimension::k2DArray)), "attachment slice is outside the storage image");
    const auto found = attachmentViews.find({format, mip, depthSlice, layers});
    if (found != attachmentViews.end()) return found->second;
    ForkTrap("attachment view");
    VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    usage.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.pNext = &usage;
    viewInfo.image = image;
    viewInfo.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1u, volume ? depthSlice : descriptor.baseArray + depthSlice, layers};
    {
        // The view must lie inside the image (a layer or mip past it makes the device write outside
        // the image's memory). Logged, so a mismatch between the descriptor and the image is visible.
        const auto& range = viewInfo.subresourceRange;
        const auto imageLayers = std::max<std::uint32_t>(ImageLayers(), 1u);
        if (!volume && range.baseArrayLayer + range.layerCount > imageLayers) {
            static std::atomic<int> told{0};
            if (told.fetch_add(1) < 16) std::fprintf(stderr, "[attachment-view] 0x%llx: layers %u..%u (baseArray %u + slice %u, %u layers) exceed the image's %u layers; mip %u/%u\n", static_cast<unsigned long long>(descriptor.baseAddress), range.baseArrayLayer, range.baseArrayLayer + range.layerCount - 1u, descriptor.baseArray, depthSlice, layers, imageLayers, mip, descriptor.mipCount);
        }
    }
    VkImageView created = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &created), "vkCreateImageView attachment");
    attachmentViews.emplace(std::tuple{format, mip, depthSlice, layers}, created);
    return created;
}

namespace {
struct Float11Repack;
const Float11Repack& float11Repack(const Context& context);
}

VkImageView StorageTexture::AttachmentProxyView(bool float11, std::uint32_t mip, std::uint32_t layer, std::uint32_t layers) {
    if (proxyImage != VK_NULL_HANDLE) {
        Require(proxyFloat11 == float11, "the storage image's attachment proxy is of another kind");
        if (float11) return proxyFloat11View(mip, layer, layers);
        return proxyView;
    }
    ForkTrap("attachment proxy view");
    if (float11) {
        Require(storageFormat == VK_FORMAT_R32_UINT && descriptor.dimension != TextureDimension::k3D, "a COLOR_11_11_10 blend proxy is only modeled for a 2D, array or cube R32 surface");
        float11Repack(context);
        try {
            VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
            imageInfo.extent = {descriptor.width, descriptor.height, 1};
            imageInfo.mipLevels = descriptor.mipCount;
            imageInfo.arrayLayers = geometry.imageLayers;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &imageInfo, nullptr, &proxyImage), "vkCreateImage float11 proxy");
            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, proxyImage, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &proxyMemory), "vkAllocateMemory float11 proxy");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, proxyImage, proxyMemory, 0), "vkBindImageMemory float11 proxy");
            proxyFloat11 = true;
        } catch (...) {
            if (proxyImage) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, proxyImage, nullptr);
            if (proxyMemory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, proxyMemory, nullptr);
            proxyImage = VK_NULL_HANDLE;
            proxyMemory = VK_NULL_HANDLE;
            proxyView = VK_NULL_HANDLE;
            throw;
        }
        return proxyFloat11View(mip, layer, layers);
    }
    Require(storageFormat == VK_FORMAT_R8_UNORM && descriptor.dimension == TextureDimension::k2D && descriptor.mipCount == 1 && geometry.imageLayers == 1, "an RGBA8_SRGB attachment proxy is only modeled for a single-mip 2D R8 surface");
    try {
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.extent = {descriptor.width, descriptor.height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &imageInfo, nullptr, &proxyImage), "vkCreateImage attachment proxy");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, proxyImage, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &proxyMemory), "vkAllocateMemory attachment proxy");
        Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, proxyImage, proxyMemory, 0), "vkBindImageMemory attachment proxy");
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = proxyImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &proxyView), "vkCreateImageView attachment proxy");
    } catch (...) {
        if (proxyImage) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, proxyImage, nullptr);
        if (proxyMemory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, proxyMemory, nullptr);
        proxyImage = VK_NULL_HANDLE;
        proxyMemory = VK_NULL_HANDLE;
        proxyView = VK_NULL_HANDLE;
        throw;
    }
    return proxyView;
}

namespace {

void blitWhole(const Context& context, VkCommandBuffer commands, VkImage source, VkImage destination, std::uint32_t width, std::uint32_t height) {
    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[1] = {static_cast<std::int32_t>(width), static_cast<std::int32_t>(height), 1};
    region.dstSubresource = region.srcSubresource;
    region.dstOffsets[1] = region.srcOffsets[1];
    context.Function<PFN_vkCmdBlitImage>("vkCmdBlitImage")(commands, source, VK_IMAGE_LAYOUT_GENERAL, destination, VK_IMAGE_LAYOUT_GENERAL, 1, &region, VK_FILTER_NEAREST);
}

VkImageMemoryBarrier proxyBarrier(VkImage image, VkAccessFlags source, VkAccessFlags destination, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = source;
    barrier.dstAccessMask = destination;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return barrier;
}

// The Float11Repack compute pipeline of a device (push descriptors: source and destination R32_UINT
// storage views), built on first use.
struct Float11Repack {
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

const Float11Repack& float11Repack(const Context& context) {
    static std::mutex mutex;
    static std::unordered_map<VkDevice, Float11Repack> pipelines;
    std::lock_guard lock(mutex);
    auto& entry = pipelines[context.device];
    if (entry.pipeline != VK_NULL_HANDLE) return entry;
    Require(context.pushDescriptors, "a COLOR_11_11_10 blend proxy needs VK_KHR_push_descriptor");
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i) bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    setInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    setInfo.pBindings = bindings.data();
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &setInfo, nullptr, &entry.setLayout), "vkCreateDescriptorSetLayout float11 repack");
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &entry.setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &push;
    Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &entry.layout), "vkCreatePipelineLayout float11 repack");
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = sizeof(FLOAT11_REPACK_SPV);
    moduleInfo.pCode = FLOAT11_REPACK_SPV;
    VkShaderModule module = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule float11 repack");
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = entry.layout;
    const auto created = context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, context.pipelineCache, 1, &pipelineInfo, nullptr, &entry.pipeline);
    context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
    Check(created, "vkCreateComputePipelines float11 repack");
    return entry;
}

// Repacks every texel of `source` into `destination` (both R32_UINT storage views in GENERAL layout).
void recordFloat11Repack(const Context& context, VkCommandBuffer commands, VkImageView source, VkImageView destination, std::uint32_t width, std::uint32_t height, bool toNative) {
    const auto& repack = float11Repack(context);
    std::array<VkDescriptorImageInfo, 2> images{{{VK_NULL_HANDLE, source, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, destination, VK_IMAGE_LAYOUT_GENERAL}}};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (std::uint32_t i = 0; i < writes.size(); ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &images[i];
    }
    const std::array<std::uint32_t, 3> push{width, height, toNative ? 1u : 0u};
    context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, repack.pipeline);
    context.Function<PFN_vkCmdPushDescriptorSetKHR>("vkCmdPushDescriptorSetKHR")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, repack.layout, 0, static_cast<std::uint32_t>(writes.size()), writes.data());
    context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, repack.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push.data());
    context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, (width + 7u) / 8u, (height + 7u) / 8u, 1);
}

}

// A view of `layers` layers (2D, or 2D array for several) of one level of `target` in `format`,
// restricted to `usage`.
VkImageView StorageTexture::proxySubresourceView(VkImage target, VkFormat format, VkImageUsageFlags usage, std::uint32_t mip, std::uint32_t layer, std::uint32_t layers) const {
    VkImageViewUsageCreateInfo usageInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, nullptr, usage};
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.pNext = &usageInfo;
    viewInfo.image = target;
    viewInfo.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, layer, layers};
    VkImageView created = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &created), "vkCreateImageView float11 proxy subresource");
    return created;
}

// The float11 proxy's attachment view of `layers` slices from `slice` of level `mip` (image layers
// from the descriptor's base array, as AttachmentView), with the R32 storage views of each of those
// layers in the proxy and in the image made alongside.
VkImageView StorageTexture::proxyFloat11View(std::uint32_t mip, std::uint32_t slice, std::uint32_t layers) {
    const auto first = descriptor.baseArray + slice;
    Require(mip < descriptor.mipCount && layers != 0 && first + layers <= geometry.imageLayers, "a COLOR_11_11_10 blend proxy subresource lies outside the surface");
    const auto key = std::tuple{mip, first, layers};
    if (const auto found = proxyAttachmentViews.find(key); found != proxyAttachmentViews.end()) return found->second;
    for (auto layer = first; layer < first + layers; ++layer) {
        const auto single = mip | (layer << 16u);
        if (proxyStorageViews.contains(single)) continue;
        proxyStorageViews.emplace(single, proxySubresourceView(proxyImage, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT, mip, layer));
        proxyImageViews.emplace(single, proxySubresourceView(image, VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT, mip, layer));
    }
    return proxyAttachmentViews.emplace(key, proxySubresourceView(proxyImage, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, mip, first, layers)).first->second;
}

// Repacks the draw's layers of level `mip` between the image (packed) and the proxy (B10G11R11), with
// the proxy's layout moving between GENERAL and the attachment layout around it.
void StorageTexture::recordProxyRepack(VkCommandBuffer commands, VkImageLayout attachmentLayout, std::uint32_t mip, std::uint32_t slice, std::uint32_t layers, bool toNative) const {
    const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
    const auto first = descriptor.baseArray + slice;
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, first, layers};
    const std::uint32_t width = std::max(1u, descriptor.width >> mip);
    const std::uint32_t height = std::max(1u, descriptor.height >> mip);
    if (toNative) {
        const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        auto discard = proxyBarrier(proxyImage, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        discard.subresourceRange = range;
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 1, &discard);
    } else {
        auto detach = proxyBarrier(proxyImage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, attachmentLayout, VK_IMAGE_LAYOUT_GENERAL);
        detach.subresourceRange = range;
        const VkMemoryBarrier reuse{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT};
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &reuse, 0, nullptr, 1, &detach);
    }
    for (auto layer = first; layer < first + layers; ++layer) {
        const auto single = mip | (layer << 16u);
        const auto proxyView = proxyStorageViews.find(single);
        const auto imageView = proxyImageViews.find(single);
        Require(proxyView != proxyStorageViews.end() && imageView != proxyImageViews.end(), "the COLOR_11_11_10 blend proxy has no view of that subresource");
        if (toNative) recordFloat11Repack(context, commands, imageView->second, proxyView->second, width, height, true);
        else recordFloat11Repack(context, commands, proxyView->second, imageView->second, width, height, false);
    }
    if (toNative) {
        auto attach = proxyBarrier(proxyImage, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, attachmentLayout);
        attach.subresourceRange = range;
        barrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &attach);
    } else {
        const VkMemoryBarrier stored{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};
        barrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &stored, 0, nullptr, 0, nullptr);
    }
}

void StorageTexture::RecordAttachmentProxyLoad(VkCommandBuffer commands, VkImageLayout attachmentLayout, std::uint32_t mip, std::uint32_t layer, std::uint32_t layers) const {
    Require(proxyImage != VK_NULL_HANDLE, "the storage image has no attachment proxy");
    if (proxyFloat11) {
        recordProxyRepack(commands, attachmentLayout, mip, layer, layers, true);
        return;
    }
    const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
    const VkMemoryBarrier written{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    const auto discard = proxyBarrier(proxyImage, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &written, 0, nullptr, 1, &discard);
    blitWhole(context, commands, image, proxyImage, descriptor.width, descriptor.height);
    const auto attach = proxyBarrier(proxyImage, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, attachmentLayout);
    barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &attach);
}

void StorageTexture::RecordAttachmentProxyStore(VkCommandBuffer commands, VkImageLayout attachmentLayout, std::uint32_t mip, std::uint32_t layer, std::uint32_t layers) const {
    Require(proxyImage != VK_NULL_HANDLE, "the storage image has no attachment proxy");
    if (proxyFloat11) {
        recordProxyRepack(commands, attachmentLayout, mip, layer, layers, false);
        return;
    }
    const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
    const auto detach = proxyBarrier(proxyImage, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, attachmentLayout, VK_IMAGE_LAYOUT_GENERAL);
    barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &detach);
    blitWhole(context, commands, proxyImage, image, descriptor.width, descriptor.height);
    const VkMemoryBarrier stored{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};
    barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &stored, 0, nullptr, 0, nullptr);
}

VkImageView StorageTexture::View(std::uint32_t mip, std::uint32_t layer) {
    if (mip == defaultMip && layer == 0) return view;
    const auto key = mip | (layer << 16u);
    const auto found = extraViews.find(key);
    if (found != extraViews.end()) return found->second;
    const auto created = createView(mip, false, storageFormat, layer);
    extraViews.emplace(key, created);
    return created;
}

VkImageView StorageTexture::FirstLayerView(std::uint32_t mip, std::uint32_t layer) {
    const auto key = mip | (layer << 16u);
    const auto found = firstLayerViews.find(key);
    if (found != firstLayerViews.end()) return found->second;
    const auto created = createView(mip, true, storageFormat, layer);
    firstLayerViews.emplace(key, created);
    return created;
}

VkImageView StorageTexture::StorageView(std::uint32_t mip, bool firstLayer, std::uint32_t layer) {
    const auto format = UintFormatOfSint(storageFormat);
    if (format == VK_FORMAT_UNDEFINED) return firstLayer ? FirstLayerView(mip, layer) : View(mip, layer);
    if (format == VK_FORMAT_R32_UINT) return AtomicView(mip, firstLayer, layer);
    const auto key = std::pair{mip | (layer << 16u), firstLayer};
    const auto found = uintViews.find(key);
    if (found != uintViews.end()) return found->second;
    Require(StorageFormatOrUndefined(context, format) == format, "storage image of format " + std::to_string(storageFormat) + " has no storage view of its UINT format " + std::to_string(format));
    const auto created = createView(mip, firstLayer, format, layer);
    uintViews.emplace(key, created);
    return created;
}

VkFormat StorageTexture::rawFormat() const {
    if (descriptor.tileMode == TextureTileMode::kLinear || BlockWidth(descriptor.format) != 1 || geometry.imageDepth != 1) return VK_FORMAT_UNDEFINED;
    if (descriptor.dimension != TextureDimension::k2D && descriptor.dimension != TextureDimension::k2DArray && descriptor.dimension != TextureDimension::kCube) return VK_FORMAT_UNDEFINED;
    VkFormat format = VK_FORMAT_UNDEFINED;
    switch (BytesPerElement(descriptor.format)) {
        case 1: format = VK_FORMAT_R8_UINT; break;
        case 2: format = VK_FORMAT_R16_UINT; break;
        case 4: format = VK_FORMAT_R32_UINT; break;
        case 8: format = VK_FORMAT_R32G32_UINT; break;
        case 16: format = VK_FORMAT_R32G32B32A32_UINT; break;
        default: return VK_FORMAT_UNDEFINED;
    }
    return StorageFormatOrUndefined(context, format) == format ? format : VK_FORMAT_UNDEFINED;
}

VkImageView StorageTexture::rawView(std::uint32_t mip) {
    if (const auto found = rawViews.find(mip); found != rawViews.end()) return found->second;
    const auto format = rawFormat();
    Require(format != VK_FORMAT_UNDEFINED && mip < descriptor.mipCount, "storage image has no raw view");
    VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    usage.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.pNext = &usage;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = format;
    viewInfo.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1u, 0u, geometry.imageLayers};
    VkImageView created = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &created), "vkCreateImageView raw");
    rawViews.emplace(mip, created);
    return created;
}

VkImageView StorageTexture::AtomicView(std::uint32_t mip, bool firstLayer, std::uint32_t layer) {
    if (storageFormat == VK_FORMAT_R32_UINT) return firstLayer ? FirstLayerView(mip, layer) : View(mip, layer);
    Require(storageFormat == VK_FORMAT_R32_SINT || storageFormat == VK_FORMAT_R32_SFLOAT, "storage image atomics need a surface of one 32-bit component");
    const auto key = std::pair{mip | (layer << 16u), firstLayer};
    const auto found = atomicViews.find(key);
    if (found != atomicViews.end()) return found->second;
    const auto created = createView(mip, firstLayer, VK_FORMAT_R32_UINT, layer);
    atomicViews.emplace(key, created);
    return created;
}

VkImageView StorageTexture::Atomic64View(std::uint32_t mip, bool firstLayer, std::uint32_t layer) {
    if (storageFormat == VK_FORMAT_R64_UINT) return firstLayer ? FirstLayerView(mip, layer) : View(mip, layer);
    Require(storageFormat == VK_FORMAT_R32G32_UINT || storageFormat == VK_FORMAT_R32G32_SINT || storageFormat == VK_FORMAT_R32G32_SFLOAT, "64-bit storage image atomics need a surface of two 32-bit components");
    const auto key = std::pair{mip | (layer << 16u), firstLayer};
    const auto found = atomicViews.find(key);
    if (found != atomicViews.end()) return found->second;
    const auto created = createView(mip, firstLayer, VK_FORMAT_R64_UINT, layer);
    atomicViews.emplace(key, created);
    return created;
}

const char* LookupOutcomes::Name(Kind kind) {
    static constexpr const char* names[Count] = {"sampled fast hit", "sampled fast miss", "sampled hit view", "sampled hit cleared view", "sampled hit snapshot", "sampled made view", "sampled made snapshot", "storage hit", "storage made", "refresh unchanged", "refresh compared", "upload direct", "upload cpu", "upload clear", "dcc scan", "pending flush"};
    return kind < Count ? names[kind] : "?";
}

bool LookupOutcomes::Profiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

std::chrono::steady_clock::time_point LookupOutcomes::Add(Kind kind, std::chrono::steady_clock::time_point start) {
    const auto now = std::chrono::steady_clock::now();
    auto& outcomes = ThreadLookupOutcomes();
    ++outcomes.counts[kind];
    outcomes.ms[kind] += std::chrono::duration<double, std::milli>(now - start).count();
    return now;
}

LookupOutcomes& ThreadLookupOutcomes() {
    thread_local LookupOutcomes outcomes;
    return outcomes;
}

namespace {

bool targetProofMemo() {
    static const bool enabled = [] { const char* text = std::getenv("APS5_TARGET_PROOF_MEMO"); return text == nullptr || std::strcmp(text, "0") != 0; }();
    return enabled;
}

std::atomic<std::uint64_t> targetProofHits{0};
std::atomic<std::uint64_t> targetProofMisses{0};
std::atomic<std::uint64_t> crossEpochProofs{0};

}

std::uint64_t TakeCrossEpochTargetProofs() {
    return crossEpochProofs.exchange(0, std::memory_order_relaxed);
}

std::pair<std::uint64_t, std::uint64_t> StorageTexture::TargetProofCounts() {
    return {targetProofHits.load(std::memory_order_relaxed), targetProofMisses.load(std::memory_order_relaxed)};
}

bool StorageTexture::refreshProved() const {
    if (!targetProofMemo() || refreshProof.epoch == 0) return false;
    const auto epoch = GuestMemory::ThreadCollectEpoch();
    if (epoch == 0 || GuestMemory::UnwatchSerial() != refreshProof.unwatched || GuestAllocations::GuestAllocationsGeneration_nid_postfix() != refreshProof.registryGeneration) return false;
    if (epoch != refreshProof.epoch) {
        if (!TargetTokensEnabled() || GuestMemory::CollectWrites(descriptor.baseAddress, static_cast<std::size_t>(guestBytes)) == 0) return false;
        if (descriptor.dccAddress != 0 && guestBytes / 256 != 0 && GuestMemory::CollectWrites(descriptor.dccAddress, static_cast<std::size_t>(guestBytes / 256)) == 0) return false;
        if (GuestMemory::UnwatchSerial() != refreshProof.unwatched) return false;
    }
    if (PendingSerial() != refreshProof.pendingSerial) {
        static const bool overlapCheck = std::getenv("APS5_NO_TARGET_PROOF_OVERLAP") == nullptr;
        if (!overlapCheck) return false;
        for (const auto& image : overlappingPending(descriptor.baseAddress, static_cast<std::size_t>(guestBytes))) {
            if (image.get() != this) return false;
        }
        if (descriptor.dccAddress != 0 && guestBytes / 256 != 0) {
            for (const auto& image : overlappingPending(descriptor.dccAddress, static_cast<std::size_t>(guestBytes / 256))) {
                if (image.get() != this) return false;
            }
        }
    }
    if (!GuestMemory::UnchangedSince(descriptor.baseAddress, static_cast<std::size_t>(guestBytes), refreshProof.generation)) return false;
    const auto keyBytes = static_cast<std::size_t>(guestBytes / 256);
    return descriptor.dccAddress == 0 || keyBytes == 0 || GuestMemory::UnchangedSince(descriptor.dccAddress, keyBytes, refreshProof.generation);
}

bool StorageTexture::Refresh() {
    ForkTrap("storage refresh");
    ForkQuiesce("storage refresh");
    const bool profile = LookupOutcomes::Profiled();
    auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (targetProofMemo()) {
        const auto provedEpoch = refreshProof.epoch;
        if (refreshProved()) {
            if (const auto epoch = GuestMemory::ThreadCollectEpoch(); epoch != provedEpoch) {
                refreshProof.epoch = epoch;
                crossEpochProofs.fetch_add(1, std::memory_order_relaxed);
            }
            if (provedEpoch != refreshProof.epoch && VerifyDrawTokens()) {
                bool current = ProvedKeys() == uploadedKeys;
                if (current && blockUnits) current = GuestMemory::UnchangedSince(descriptor.baseAddress, static_cast<std::size_t>(guestBytes), generation);
                for (std::uint32_t layer = 0; current && !blockUnits && layer < trackedLayers; ++layer) current = GuestMemory::UnchangedSince(layerBegin(layer), static_cast<std::size_t>(layerBytes(layer)), layerGeneration[layer]);
                if (!current) {
                    std::fprintf(stderr, "[tokens] APS5_VERIFY_DRAW_TOKENS: a cross-epoch target proof of 0x%llx is refused by the refresh check\n", static_cast<unsigned long long>(descriptor.baseAddress));
                    std::fflush(stderr);
                    std::abort();
                }
            }
            NoteProved();
            GuestMemory::NoteProbeEvent("memo target proof");
            ++Profile().storageReused;
            const auto hits = targetProofHits.fetch_add(1, std::memory_order_relaxed) + 1;
            if (profile && hits % 100000 == 0) std::fprintf(stderr, "[target-proof] %llu refreshes answered by the proof, %llu not\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(targetProofMisses.load(std::memory_order_relaxed)));
            if (profile) LookupOutcomes::Add(LookupOutcomes::RefreshUnchanged, start);
            return true;
        }
        targetProofMisses.fetch_add(1, std::memory_order_relaxed);
        refreshProof = {};
    }
    struct Exempt {
        const StorageTexture* previous;
        ~Exempt() { refreshing = previous; }
    } exempt{refreshing};
    // This image's own pending results stay on the GPU, where the next dispatch wants them (the
    // flush below and the compare through the hook skip it).
    refreshing = this;
    NoteProved();
    // Results of other images pending in this memory must reach it first, except an alias's: its
    // units are taken on the device below (borrowUnits), so it stays pending. The keys read for
    // that decision are read again after the flush, which may store keys itself. The exemption
    // holds only when every pending unit of the alias is borrowable (tracked, unstamped since its
    // generation, no results of this image's own there) or dead to its own write-back (tracked and
    // stamped); an untracked unit is stored whole by that write-back, and a unit pending in both
    // images takes the old order (store, then this image re-uploads it).
    auto alias = pendingAlias();
    if (alias != nullptr && !(ProvedClearKeys(descriptor, guestBytes, keyProof) == DccKeys::Uncompressed && HostImportFor(context, descriptor.baseAddress, static_cast<std::size_t>(guestBytes)) != nullptr)) alias = nullptr;
    // A row alias (same base, width and layout, another height; `lazy_writeback`): the units the
    // two share are taken on the device only where every byte holds an element of both surfaces.
    // An edge unit (the last block row of the shorter one, whose padding rows are the taller one's
    // texels, or a unit the shorter surface ends inside) has its alias results stored first: this
    // image then reads it from memory, and no borrow of it can later supersede results of the
    // alias this image does not hold.
    std::vector<bool> aliasEdge;
    const bool rowAlias = alias != nullptr && (alias->guestBytes != guestBytes || alias->descriptor.height != descriptor.height);
    if (rowAlias) {
        aliasEdge = rowAliasEdges(*alias);
        std::vector<bool> stored(trackedLayers, false);
        bool any = false;
        for (std::uint32_t unit = 0; unit < std::min(trackedLayers, alias->trackedLayers); ++unit) {
            if (!aliasEdge[unit] || !alias->layerPending[unit]) continue;
            stored[unit] = true;
            any = true;
            lazyEdgeUnits.fetch_add(1, std::memory_order_relaxed);
        }
        if (any) {
            for (const auto& [begin, end] : unitRuns(stored)) static_cast<void>(FlushPending(descriptor.baseAddress + begin, static_cast<std::size_t>(end - begin), nullptr, "alias edge", PublishScope::None));
        }
    }
    // Units of this image the alias has (the shorter of the two surfaces bounds them).
    const auto aliasUnits = alias != nullptr ? std::min(trackedLayers, alias->trackedLayers) : 0u;
    if (alias != nullptr) {
        std::vector<std::uint8_t> aliasStamped(trackedLayers);
        if (!GuestMemory::ChangedBlocks(descriptor.baseAddress, static_cast<std::size_t>(std::min(guestBytes, alias->guestBytes)), alias->layerGeneration, aliasStamped)) alias = nullptr;
        for (std::uint32_t unit = 0; alias != nullptr && unit < aliasUnits; ++unit) {
            if (!alias->layerPending[unit]) continue;
            if ((rowAlias && aliasEdge[unit]) || alias->layerGeneration[unit] == 0 || aliasStamped[unit] == GuestMemory::BlockMaybeWritten || (aliasStamped[unit] == GuestMemory::BlockUnchanged && layerPending[unit])) {
                alias = nullptr;
                break;
            }
        }
    }
    // Another pending image over a unit the alias holds pending results in: the flush below stores
    // it and stamps the unit, which the alias (exempt) would then take for a CPU write and drop at
    // its own write-back though its results may be the newer. The order is unknown: both are stored.
    if (alias != nullptr) {
        for (const auto& other : overlappingPending(descriptor.baseAddress, static_cast<std::size_t>(guestBytes))) {
            if (other == alias) continue;
            for (std::uint32_t unit = 0; alias != nullptr && unit < aliasUnits; ++unit) {
                if (alias->layerPending[unit] && other->overlaps(layerBegin(unit), static_cast<std::size_t>(layerBytes(unit))) && other->pendingUnitInside(layerBegin(unit), static_cast<std::size_t>(layerBytes(unit)))) alias = nullptr;
            }
            if (alias == nullptr) break;
        }
    }
    // Layer aliases (a cube map and the 2D images of its faces) under the same rule, each over the
    // units it shares with this image. Two of them pending in one unit would need their order:
    // then every one is stored as before.
    std::vector<LayerAlias> layerAliases;
    if (alias == nullptr) {
        layerAliases = pendingLayerAliases();
        if (!layerAliases.empty() && HostImportFor(context, descriptor.baseAddress, static_cast<std::size_t>(guestBytes)) == nullptr) layerAliases.clear();
        std::erase_if(layerAliases, [&](const LayerAlias& link) { return !layerAliasExempt(link); });
        std::vector<bool> claimed(trackedLayers, false);
        bool collide = false;
        for (const auto& link : layerAliases) {
            for (auto unit = link.first; unit < link.end && !collide; ++unit) {
                if (!link.source->layerPending[static_cast<std::size_t>(unit + link.offset)]) continue;
                collide = claimed[unit];
                claimed[unit] = true;
            }
        }
        if (collide) layerAliases.clear();
    }
    std::vector<const StorageTexture*> exemptLinks;
    for (const auto& link : layerAliases) exemptLinks.push_back(link.source.get());
    struct ExemptLinks {
        const std::vector<const StorageTexture*>* previous;
        ~ExemptLinks() { exemptLayerAliases = previous; }
    } exemptLinksScope{exemptLayerAliases};
    if (!exemptLinks.empty()) exemptLayerAliases = &exemptLinks;
    // No publish: the upload below reads the stored units from the unit shadow itself.
    const bool flushed = FlushPending(descriptor.baseAddress, static_cast<std::size_t>(guestBytes), alias.get(), "storage refresh", PublishScope::None);
    exemptLayerAliases = exemptLinksScope.previous;
    if (profile && flushed) start = LookupOutcomes::Add(LookupOutcomes::PendingFlush, start);
    // `original` holds the guest bytes the image was last uploaded from or written back as; while the
    // guest memory and the DCC keys still match, the image content is current. Pages nobody wrote
    // since `generation` need no comparison.
    const auto current = GuestMemory::CollectWrites(descriptor.baseAddress, static_cast<std::size_t>(guestBytes));
    // Per tracked layer: changed since its generation (a keys change makes every layer stale).
    std::vector<bool> changed(trackedLayers, false);
    // Per 64 KiB block of the surface (blockGenerations): stamped since its layer's generation, and
    // by a CPU store; scanned once when anything changed.
    std::vector<std::uint64_t> generations;
    std::vector<std::uint8_t> stampedBlocks, cpuBlocks;
    // Units taken from the alias's image instead of guest memory.
    std::vector<bool> borrowed;
    bool anyBorrowed = false;
    // APS5_VERIFY_LAZY_WRITEBACK: the alias whose borrowed units the end of the refresh checks.
    std::shared_ptr<StorageTexture> verifyAlias;
    // Per layer alias: the units taken from it (empty: none).
    std::vector<std::vector<bool>> layerBorrowed;
    bool unchanged = true;
    bool stamped = true;
    bool tracked = true;
    bool keysChanged = false;
    bool cpuWrote = false;
    bool direct = false;
    DccKeys keys = uploadedKeys;
    {
        const auto equalsOriginal = [&] {
            // Named for the [hooksync] attribution: the compare goes through the flush hook.
            const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureCompare);
            return GuestMemory::EqualsCommitted(descriptor.baseAddress, original);
        };
        const auto scanBlocks = [&] {
            blockGenerations(generations);
            stampedBlocks.assign(generations.size(), 0);
            cpuBlocks.assign(generations.size(), 0);
            tracked = GuestMemory::ChangedBlocks(descriptor.baseAddress, static_cast<std::size_t>(guestBytes), generations, stampedBlocks, cpuBlocks);
        };
        const auto keysStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        keys = ProvedKeys();
        if (profile && descriptor.dccAddress != 0) LookupOutcomes::Add(LookupOutcomes::DccScan, keysStart);
        if (keys != uploadedKeys) {
            changed.assign(trackedLayers, true);
            unchanged = false;
            keysChanged = true;
            // Which units the CPU stamped still decides what the drop rule below may drop.
            if (blockUnits) scanBlocks();
        } else if (blockUnits) {
            if (!GuestMemory::UnchangedSince(descriptor.baseAddress, static_cast<std::size_t>(guestBytes), generation)) {
                // One stamp scan for every unit (each is one stamp block); the unchanged units move
                // to the current generation, so the whole-surface check answers next time.
                scanBlocks();
                for (std::uint32_t unit = 0; unit < trackedLayers; ++unit) {
                    if (stampedBlocks[unit] == 0) {
                        layerGeneration[unit] = current;
                        continue;
                    }
                    changed[unit] = true;
                    unchanged = false;
                }
                refreshGeneration();
            }
        } else {
            for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
                if (GuestMemory::UnchangedSince(layerBegin(layer), static_cast<std::size_t>(layerBytes(layer)), layerGeneration[layer])) continue;
                changed[layer] = true;
                unchanged = false;
            }
            if (!unchanged) scanBlocks();
        }
        // "cpu" names a CPU store in both models; a block without a generation (untracked) says
        // nothing about who wrote it.
        for (std::size_t k = 0; k < cpuBlocks.size(); ++k) {
            if (tracked && cpuBlocks[k] != 0 && generations[k] != 0) cpuWrote = true;
        }
        // Unchanged bytes rescue an upload only when nothing else moved: with the keys changed
        // the same bytes read differently, and a changed unit holding pending results must take
        // the drop rule below (the image holds the results, `original` what the CPU wrote back).
        bool pendingChanged = false;
        for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
            if (changed[layer] && layerPending[layer]) pendingChanged = true;
        }
        if (!unchanged && !keysChanged && !pendingChanged && originalValid && equalsOriginal()) {
            unchanged = true;
            stamped = false;
        }
        // Only the direct path uploads the selected layers alone (see upload); the others replace
        // the whole image, so every pending layer's results are stored first.
        direct = keys == DccKeys::Uncompressed && HostImportFor(context, descriptor.baseAddress, static_cast<std::size_t>(guestBytes)) != nullptr && (blockUnits || trackedLayers == arrayLayers || trackedLayers == 1);
        if (alias != nullptr && !direct) {
            // The keys moved under the flush (a key store it made): nothing can be borrowed, so the
            // exempt alias's results reach memory now and this image re-reads those units.
            std::vector<bool> held(trackedLayers, false);
            for (std::uint32_t unit = 0; unit < aliasUnits; ++unit) held[unit] = alias->layerPending[unit];
            static_cast<void>(FlushPending(descriptor.baseAddress, static_cast<std::size_t>(std::min(guestBytes, alias->guestBytes)), nullptr, "storage refresh", PublishScope::None));
            for (std::uint32_t unit = 0; unit < trackedLayers; ++unit) {
                if (!held[unit]) continue;
                changed[unit] = true;
                unchanged = false;
            }
        }
        if (alias != nullptr && direct) {
            // The alias's pending units that nothing stamped since its generation (a CPU store or a
            // driver store there wins over its results, as at its own write-back) are taken from
            // its image: every one of them, since its write-back would have stamped them all. Units
            // already holding them at the same alias version stay as they are.
            const bool sameBorrow = borrowedFrom.lock() == alias && borrowedVersion == alias->version && borrowedUnits.size() == trackedLayers;
            std::vector<std::uint8_t> aliasStamped(trackedLayers);
            const bool aliasTracked = GuestMemory::ChangedBlocks(descriptor.baseAddress, static_cast<std::size_t>(std::min(guestBytes, alias->guestBytes)), alias->layerGeneration, aliasStamped);
            borrowed.assign(trackedLayers, false);
            for (std::uint32_t unit = 0; unit < aliasUnits; ++unit) {
                if (!aliasTracked || !alias->layerPending[unit] || aliasStamped[unit] != 0 || alias->layerGeneration[unit] == 0 || layerPending[unit]) continue;
                if (rowAlias && aliasEdge[unit]) continue;
                if (sameBorrow && borrowedUnits[unit]) continue;
                borrowed[unit] = true;
                anyBorrowed = true;
                changed[unit] = false;
                unchanged = false;
            }
            if (!sameBorrow) {
                borrowedFrom.reset();
                borrowedUnits.clear();
            }
        }
        // The layer aliases' units by the same rule, each through its own unit offset.
        layerBorrowed.assign(layerAliases.size(), {});
        for (std::size_t index = 0; direct && index < layerAliases.size(); ++index) {
            const auto& link = layerAliases[index];
            const auto& source = *link.source;
            const auto count = link.end - link.first;
            std::vector<std::uint64_t> sourceGenerations(count);
            for (std::uint32_t k = 0; k < count; ++k) sourceGenerations[k] = source.layerGeneration[static_cast<std::size_t>(link.first + k + link.offset)];
            std::vector<std::uint8_t> sourceStamped(count);
            const auto begin = static_cast<std::uint64_t>(link.first) * trackedLayerBytes;
            const auto end = std::min<std::uint64_t>(guestBytes, static_cast<std::uint64_t>(link.end) * trackedLayerBytes);
            if (!GuestMemory::ChangedBlocks(descriptor.baseAddress + begin, static_cast<std::size_t>(end - begin), sourceGenerations, sourceStamped)) continue;
            const LayerBorrow* held = nullptr;
            for (const auto& record : layerBorrows) {
                if (record.from.lock() == link.source && record.offset == link.offset && record.version == source.version && record.units.size() == trackedLayers) held = &record;
            }
            auto& units = layerBorrowed[index];
            for (std::uint32_t k = 0; k < count; ++k) {
                const auto unit = link.first + k;
                const auto sourceUnit = static_cast<std::size_t>(unit + link.offset);
                if (!source.layerPending[sourceUnit] || sourceStamped[k] != 0 || source.layerGeneration[sourceUnit] == 0 || layerPending[unit]) continue;
                if (held != nullptr && held->units[unit]) continue;
                if (units.empty()) units.assign(trackedLayers, false);
                units[unit] = true;
                changed[unit] = false;
                unchanged = false;
            }
        }
    }
    if (unchanged) {
        ++Profile().storageReused;
        layerGeneration.assign(trackedLayers, current);
        refreshGeneration();
        if (targetProofMemo() && current != 0 && (descriptor.dccAddress == 0 || keyProof.generation != 0)) {
            refreshProof = {GuestMemory::ThreadCollectEpoch(), GuestMemory::UnwatchSerial(), PendingSerial(), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), current};
        }
        if (profile) LookupOutcomes::Add(stamped ? LookupOutcomes::RefreshUnchanged : LookupOutcomes::RefreshCompared, start);
        return true;
    }
    CaptureTrace::Log("refresh image=%llx bytes=%llu generation=%llu dirty=%d keysChanged=%d cpuWrote=%d", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), static_cast<unsigned long long>(generation), dirty, keysChanged, cpuWrote);
    // Debug aid: APS5_TRACE_UPLOAD names why a storage image is uploaded again.
    static const bool traceUpload = std::getenv("APS5_TRACE_UPLOAD") != nullptr;
    if (traceUpload) {
        const auto keys = TextureClearKeys(descriptor, guestBytes);
        std::fprintf(stderr, "[upload] 0x%llx+0x%llx %ux%u mips %u: %s (keys %s -> %s, originalValid %d, dirty %d)\n", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), descriptor.width, descriptor.height, descriptor.mipCount, keys != uploadedKeys ? "DCC keys changed" : "guest memory changed", DccKeysName(uploadedKeys), DccKeysName(keys), originalValid ? 1 : 0, dirty ? 1 : 0);
    }
    // Changed layers with results pending: the CPU wrote this memory while GPU results were
    // pending, so as with an immediate write-back followed by the CPU write, its blocks win and the
    // results land everywhere else in those layers; the unchanged pending layers keep their
    // results on the GPU when the direct path leaves them alone, and are stored otherwise. The
    // image stays registered throughout (the store clears what it stored).
    std::vector<bool> stored(trackedLayers, false);
    bool pendingResults = false;
    bool anyStored = false;
    bool dropped = false;
    // A unit is one 64 KiB block, which the CPU's bytes win whole once a store stamped it:
    // nothing of its results is stored, and the unit re-uploads from memory below. A keys change
    // alone (the title marking the surface uncompressed behind a pass) leaves unstamped units
    // with their results, which are stored first, as the layer model does; new keys that are a
    // clear code make every result dead on hardware too. Without a generation (no tracking) a
    // unit's stamps say nothing, so it is stored.
    const auto droppable = [&](std::uint32_t unit) {
        if (keysChanged && IsDccClear(keys)) return true;
        if (!tracked || layerGeneration[unit] == 0 || unit >= stampedBlocks.size() || stampedBlocks[unit] != GuestMemory::BlockWritten) return false;
        const auto begin = layerBegin(unit);
        const auto bytes = layerBytes(unit);
        const bool edge = begin % 65536 != 0 || bytes != 65536;
        return !edge || GuestMemory::StoredOver(begin, static_cast<std::size_t>(bytes), layerGeneration[unit]);
    };
    // A clear code -> uncompressed flip on an image with results pending: an unstamped pending
    // unit's results ARE the uncompressed texels (nothing wrote its memory since), so it keeps
    // them instead of being dropped and re-read from stale guest bytes; a stamped one takes the
    // drop rule. Debug aid: APS5_KEYFLIP_DROP=1 drops as before.
    static const bool keyFlipDrop = std::getenv("APS5_KEYFLIP_DROP") != nullptr;
    static const bool traceKeys = std::getenv("APS5_TRACE_DCC_KEYS") != nullptr;
    const bool keyFlip = keysChanged && blockUnits && !keyFlipDrop && IsDccClear(uploadedKeys) && keys == DccKeys::Uncompressed && stampedBlocks.size() >= trackedLayers;
    if (traceKeys && keysChanged) {
        std::size_t pendingUnits = 0, stampedUnits = 0;
        for (std::uint32_t unit = 0; unit < trackedLayers; ++unit) {
            if (layerPending[unit]) ++pendingUnits;
            if (unit < cpuBlocks.size() && cpuBlocks[unit] != 0 && generations[unit] != 0) ++stampedUnits;
        }
        std::fprintf(stderr, "[dcc-keys] refresh 0x%llx+0x%llx keys %s -> %s (dirty %d, %zu pending units, %zu cpu-stamped units%s) site %s tdraw %llu\n",static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), DccKeysName(uploadedKeys), DccKeysName(keys), dirty ? 1 : 0, pendingUnits, stampedUnits, keyFlip ? ", flip keeps unstamped results" : "", g_refreshSite, static_cast<unsigned long long>(g_traceTargetDraws.load(std::memory_order_relaxed)));
    }
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (!layerPending[layer]) continue;
        if (keyFlip && layerGeneration[layer] != 0 && stampedBlocks[layer] != GuestMemory::BlockWritten) {
            changed[layer] = false;
            keyFlipKept.fetch_add(1, std::memory_order_relaxed);
            if (direct) continue;
            stored[layer] = true;
            anyStored = true;
            continue;
        }
        if (changed[layer]) {
            pendingResults = true;
            if (blockUnits && droppable(layer)) {
                CaptureTrace::Log("drop-unit image=%llx unit=%u generation=%llu stamped=%d keys=%d", static_cast<unsigned long long>(descriptor.baseAddress), layer, static_cast<unsigned long long>(layerGeneration[layer]), layer < stampedBlocks.size() && stampedBlocks[layer] != 0, static_cast<int>(keys));
                layerPending[layer] = false;
                dropped = true;
                continue;
            }
        } else if (direct) {
            continue;
        }
        stored[layer] = true;
        anyStored = true;
    }
    if (anyStored && !keysChanged && descriptor.dccAddress != 0 && IsDccClear(uploadedKeys)) stored.assign(trackedLayers, true);
    if (pendingResults) {
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1) < 8) std::fprintf(stderr, "[gpu] storage image 0x%llx: guest memory changed while GPU results were pending; keeping the CPU's blocks\n", static_cast<unsigned long long>(descriptor.baseAddress));
    }
    if (anyStored) {
        const auto previous = std::exchange(flushReason, "refresh");
        writeBackLayers(stored);
        flushReason = previous;
    } else if (dropped) {
        reconcilePending();
    }
    if (anyBorrowed) {
        GuestMemory::NoteProbeEvent("alias borrow");
        uploadReason = "alias";
        const auto bytes = borrowUnits(*alias, borrowed);
        if (rowAlias) {
            lazyRowBorrows.fetch_add(1, std::memory_order_relaxed);
            lazyRowBorrowBytes.fetch_add(bytes, std::memory_order_relaxed);
        }
        // Checked once the refresh is done (the check waits for the GPU, which runs completions).
        if (VerifyLazyWriteBack()) verifyAlias = alias;
        for (std::uint32_t unit = 0; unit < trackedLayers; ++unit) {
            if (borrowed[unit]) layerGeneration[unit] = current;
        }
        refreshGeneration();
    }
    for (std::size_t index = 0; index < layerBorrowed.size(); ++index) {
        const auto& units = layerBorrowed[index];
        if (units.empty()) continue;
        GuestMemory::NoteProbeEvent("layer alias borrow");
        uploadReason = "layer alias";
        borrowLayerUnits(*layerAliases[index].source, layerAliases[index].offset, units);
        for (std::uint32_t unit = 0; unit < trackedLayers; ++unit) {
            if (units[unit]) layerGeneration[unit] = current;
        }
        refreshGeneration();
    }
    // Live `rt_alias_discard` (APS5_RT_ALIAS_DISCARD=1), experimental: a render target whose memory
    // changed only because another image's results were just flushed into it (a transient pool
    // handing the same block to targets of another shape) keeps its own content instead of
    // uploading the other image's bytes, which the title renders over. Not exact: a title reading
    // or blending over reinterpreted memory would see the target's old texels.
    static const bool aliasDiscardDefault = std::getenv("APS5_RT_ALIAS_DISCARD") != nullptr;
    // Live `volume_alias_discard` (APS5_VOLUME_ALIAS_DISCARD=1), experimental and inexact the same
    // way: a thick (3D) storage image whose memory changed only through another image's flush (a
    // transient heap block shared with cube maps) keeps its content instead of uploading it.
    static const bool volumeDiscardDefault = std::getenv("APS5_VOLUME_ALIAS_DISCARD") != nullptr;
    // Live `storage_alias_discard` (APS5_STORAGE_ALIAS_DISCARD=1), experimental and inexact the same
    // way, for the thin (2D) storage images of the same transient blocks (PPSA01317 hands one block
    // to 1920x1080, 1920x2160, 666x524 and 512x256 targets of different formats every frame).
    static const bool storageDiscardDefault = std::getenv("APS5_STORAGE_ALIAS_DISCARD") != nullptr;
    const bool storageSite = std::string_view(g_refreshSite) == "storage" || std::string_view(g_refreshSite) == "other";
    const bool aliasSite = std::string_view(g_refreshSite) == "render-target" ? LiveFlag("rt_alias_discard", aliasDiscardDefault) : storageSite && (geometry.thick ? LiveFlag("volume_alias_discard", volumeDiscardDefault) : LiveFlag("storage_alias_discard", storageDiscardDefault));
    const bool aliasDiscard = flushed && !keysChanged && !cpuWrote && generation != 0 && keys == DccKeys::Uncompressed && aliasSite;
    static const bool traceVolume = std::getenv("APS5_TRACE_VOLUME_DISCARD") != nullptr;
    if (traceVolume && geometry.thick) {
        static std::atomic<int> traced{0};
        if (traced.fetch_add(1) < 60) std::fprintf(stderr, "[volume-discard] 0x%llx flushed %d keysChanged %d cpuWrote %d generation %llu keys %d site %s discard %d\n", static_cast<unsigned long long>(descriptor.baseAddress), flushed ? 1 : 0, keysChanged ? 1 : 0, cpuWrote ? 1 : 0, static_cast<unsigned long long>(generation), static_cast<int>(keys), g_refreshSite, aliasDiscard ? 1 : 0);
    }
    if (aliasDiscard && std::any_of(changed.begin(), changed.end(), [](bool selected) { return selected; })) {
        aliasDiscards.fetch_add(1, std::memory_order_relaxed);
        layerGeneration.assign(trackedLayers, current);
        refreshGeneration();
        originalValid = false;
        uploadedKeys = keys;
    } else if (std::any_of(changed.begin(), changed.end(), [](bool selected) { return selected; })) {
        uploadReason = keysChanged ? "keys" : cpuWrote ? "cpu" : generation == 0 ? "untracked" : flushed ? "flushed" : "store";
        upload(&changed);
    } else {
        uploadedKeys = keys;
    }
    if (verifyAlias != nullptr) verifyBorrow(*verifyAlias, borrowed);
    return false;
}

DccKeys StorageTexture::ProvedKeys() const {
    std::optional<DccKeys> keys;
    if (descriptor.dccAddress != 0 && uploadedKeys == DccKeys::Uncompressed) keys = WaitForKeyWriters(descriptor, guestBytes);
    if (keys.has_value()) keyProof = {};
    else keys = ProvedClearKeys(descriptor, guestBytes, keyProof);
    if (IsDccClear(filledKeys) && *keys != filledKeys && !anyLayerPending()) filledKeys = DccKeys::Uncompressed;
    return *keys;
}

bool StorageTexture::ServesKeysAt(std::uint64_t dccAddress) const {
    const bool locked = GuestMemory::GpuMutex().HeldByThisThread();
    const auto readFollowed = [&] {
        ForkTrap("image keys");
        DccKeyProof unlocked;
        return ProvedClearKeys(descriptor, guestBytes, locked ? keyProof : unlocked);
    };
    const auto readNamed = [&] {
        ForkTrap("image keys");
        auto named = descriptor;
        named.dccAddress = dccAddress;
        DccKeyProof unlocked;
        if (!locked) return ProvedClearKeys(named, guestBytes, unlocked);
        auto slot = std::find_if(foreignKeyProofs.begin(), foreignKeyProofs.end(), [&](const ForeignKeyProof& entry) { return entry.dccAddress == dccAddress; });
        if (slot == foreignKeyProofs.end()) {
            slot = foreignKeyProofs.begin() + nextForeignKeyProof++ % foreignKeyProofs.size();
            *slot = ForeignKeyProof{dccAddress, {}};
        }
        return ProvedClearKeys(named, guestBytes, slot->proof);
    };
    return KeysServeSurface(descriptor.dccAddress, uploadedKeys, filledKeys, dccAddress, readFollowed, readNamed);
}

namespace {

// Debug aid (APS5_TRACE_DCC_KEYS=1): who stores uncompressed keys over a surface (recommendation
// 2's tracer): the write-back path, the surface and its key range.
void traceKeyStore(const char* path, const GuestTextureResource& descriptor, std::uint64_t guestBytes) {
    static const bool trace = std::getenv("APS5_TRACE_DCC_KEYS") != nullptr;
    if (!trace || descriptor.dccAddress == 0) return;
    const auto packet = GuestMemory::CurrentPacket();
    std::fprintf(stderr, "[dcc-keys] uncompressed store by %s (%s) for surface 0x%llx+0x%llx: keys 0x%llx+0x%llx (packet 0x%x queue 0x%x) site %s tdraw %llu\n", path, flushReason != nullptr ? flushReason : "?", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), static_cast<unsigned long long>(descriptor.dccAddress), static_cast<unsigned long long>(DccKeyCount(descriptor, guestBytes)), packet.opcode, packet.queue, g_refreshSite, static_cast<unsigned long long>(g_traceTargetDraws.load(std::memory_order_relaxed)));
}

}

void StorageTexture::upload(const std::vector<bool>* layers) {
    GuestMemory::NoteProbeEvent("storage upload");
    CaptureTrace::Log("upload image=%llx bytes=%llu generation=%llu reason=%s partial=%d", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), static_cast<unsigned long long>(generation), uploadReason, layers != nullptr);
    const bool profile = LookupOutcomes::Profiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto elementBytes = BytesPerElement(descriptor.format);
    const auto linearBytes = sliceLinearBytes * arrayLayers;
    original.resize(static_cast<std::size_t>(guestBytes));
    // A layer selection applies to the direct path alone (the others upload everything); it names
    // array layers, which the tracked layers are when there are several.
    if (layers != nullptr && ((!blockUnits && trackedLayers != arrayLayers) || std::all_of(layers->begin(), layers->end(), [](bool selected) { return selected; }))) layers = nullptr;
    const auto now = GuestMemory::CollectWrites(descriptor.baseAddress, static_cast<std::size_t>(guestBytes));
    const auto stampLayers = [&](bool selectedOnly) {
        for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
            if (!selectedOnly || layers == nullptr || (*layers)[layer]) layerGeneration[layer] = now;
        }
        refreshGeneration();
    };
    uploadedKeys = ProvedClearKeys(descriptor, guestBytes, keyProof);
    filledKeys = DccKeys::Uncompressed;
    VkClearColorValue clearValue{};
    if (IsDccClear(uploadedKeys) && ClearColorFor(storageFormat, uploadedKeys, clearValue)) {
        // A fast-cleared surface is cleared on the GPU; its texel memory is neither read nor filled.
        // The clear is recorded into the open batch like a direct upload (the image kept by it): a
        // batch of its own submitted the recorder's work first and waited for all of it, 25-40 ms
        // under the GPU mutex at the movie stage. APS5_NO_RECORDED_CLEAR=1 waits as before.
        originalValid = false;
        forgetBorrowed(0, trackedLayers);
        stampLayers(false);
        static const bool recordClear = std::getenv("APS5_NO_RECORDED_CLEAR") == nullptr;
        auto* recorder = recordClear ? Recorder::Active() : nullptr;
        std::unique_ptr<CommandBatch> batch;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        auto timing = Recorder::NoTiming;
        bool elided = false;
        if (recorder != nullptr) {
            const std::pair<VkImage, bool> cleared{image, true};
            commands = Recorder::BarrierTracking() ? recorder->Commands(Recorder::CommandClass::DccClear, Recorder::Access{{}, {}, std::span(&cleared, 1), VK_PIPELINE_STAGE_TRANSFER_BIT}, elided) : recorder->Commands();
            timing = recorder->BeginGpuTiming(Recorder::CommandClass::DccClear);
            if (auto self = weak_from_this().lock()) recorder->Keep(std::move(self));
            ++Profile().storageRecordedClears;
            Recorder::CountBarriers(Recorder::CommandClass::DccClear, elided ? 1 : 2);
            if (elided) Recorder::CountSkipped(Recorder::CommandClass::DccClear);
        } else {
            batch = std::make_unique<CommandBatch>(context);
            commands = batch->Handle();
            ++Profile().storageWaitedClears;
        }
        VkImageMemoryBarrier toClear{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toClear.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        toClear.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toClear.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toClear.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toClear.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toClear.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toClear.image = image;
        toClear.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
        if (elided) {
            toClear.srcAccessMask = 0;
            toClear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toClear);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &toClear.subresourceRange);
        } else {
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toClear);
            context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &toClear.subresourceRange);
            VkImageMemoryBarrier toGeneral = toClear;
            toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        }
        if (batch) batch->SubmitAndWait();
        else recorder->EndGpuTiming(timing, guestBytes);
        countStorageUpload(0, guestBytes);
        ++version;
        if (profile) LookupOutcomes::Add(LookupOutcomes::UploadClear, start);
        return;
    }
    if (const auto* import = uploadedKeys == DccKeys::Uncompressed ? HostImportFor(context, descriptor.baseAddress, static_cast<std::size_t>(guestBytes)) : nullptr) {
        // The surface lives in host-imported memory: the detiler reads it in place, no guest bytes
        // are copied, and write tracking alone validates the image (a change re-runs this).
        originalValid = false;
        stampLayers(true);
        // A whole-surface upload of a block-unit image goes through the windows too when a unit
        // shadow holds part of the surface (the detile then reads the slabs; a new image has no
        // layout yet); a partial one always does.
        if (blockUnits && (layers != nullptr || AnyShadowedOverlaps(descriptor.baseAddress, static_cast<std::size_t>(guestBytes)))) {
            std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
            if (layers != nullptr) {
                for (std::uint32_t unit = 0; unit < trackedLayers; ++unit) {
                    if ((*layers)[unit]) forgetBorrowed(unit, 1);
                }
                runs = unitRuns(*layers);
            } else {
                forgetBorrowed(0, trackedLayers);
                runs.emplace_back(0, guestBytes);
            }
            const auto uploadedBytes = uploadWindows(*import, runs, layers == nullptr && version == 0);
            countStorageUpload(1, uploadedBytes);
            if (layers != nullptr) {
                partialUploads.fetch_add(1, std::memory_order_relaxed);
                partialUploadBytes.fetch_add(uploadedBytes, std::memory_order_relaxed);
            }
            ++Profile().storageDirectUploads;
            ++version;
            if (profile) LookupOutcomes::Add(LookupOutcomes::UploadDirect, start);
            return;
        }
        // Units another image's results shadow reach the import before the detile reads it in place.
        PublishShadow(descriptor.baseAddress, static_cast<std::size_t>(guestBytes), PublishScope::Whole, PublishReason::Upload);
        forgetBorrowed(0, trackedLayers);
        std::uint64_t uploadedBytes = layers == nullptr ? guestBytes : 0;
        if (layers != nullptr) {
            for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
                if ((*layers)[layer]) uploadedBytes += trackedLayerBytes;
            }
        }
        auto linear = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(linearBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        detiler.BeginBatch();
        auto* recorder = Recorder::Active();
        std::unique_ptr<CommandBatch> batch;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        auto timing = Recorder::NoTiming;
        bool elided = false;
        if (recorder != nullptr) {
            // A queued label store in the surface lands before the detile reads it.
            recorder->FlushStoresOverlapping(descriptor.baseAddress, static_cast<std::size_t>(guestBytes));
            const std::pair<std::uint64_t, std::uint64_t> read{descriptor.baseAddress, descriptor.baseAddress + guestBytes};
            const std::pair<VkImage, bool> written{image, true};
            commands = Recorder::BarrierTracking() ? recorder->Commands(Recorder::CommandClass::StorageUpload, Recorder::Access{std::span(&read, 1), {}, std::span(&written, 1), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT}, elided) : recorder->Commands();
            timing = recorder->BeginGpuTiming(Recorder::CommandClass::StorageUpload);
            recorder->NoteTimedSurface(timing, descriptor.baseAddress, guestBytes, static_cast<std::uint32_t>(storageFormat), descriptor.format, descriptor.width, descriptor.height, "upload", 1);
            if (GpuCheckpoints::Active().load(std::memory_order_relaxed)) {
                char text[160];
                std::snprintf(text, sizeof(text), "detile upload %ux%u fmt%u tile%d mips %u layers %u, import 0x%llx+0x%llx", descriptor.width, descriptor.height, descriptor.format, static_cast<int>(descriptor.tileMode), descriptor.mipCount, arrayLayers, static_cast<unsigned long long>(import->base), static_cast<unsigned long long>(import->bytes));
                GpuCheckpoints::Set(context, commands, "storage-upload detile (surface, bytes, linear)", descriptor.baseAddress, guestBytes, linearBytes, text);
            }
            recorder->Keep(linear, linear->Size());
            // The image itself must outlive the recorded copy: the cache may evict it right after.
            if (auto self = weak_from_this().lock()) recorder->Keep(std::move(self));
            // The detile reads the tiled bytes from the import when the batch runs.
            recorder->NotePendingRead(descriptor.baseAddress, static_cast<std::size_t>(guestBytes), Recorder::ReadKind::StorageUpload);
            Recorder::CountBarriers(Recorder::CommandClass::StorageUpload, elided ? 1 : 3);
            if (elided) Recorder::CountSkipped(Recorder::CommandClass::StorageUpload, 2);
        } else {
            batch = std::make_unique<CommandBatch>(context);
            commands = batch->Handle();
        }
        const auto importOffset = descriptor.baseAddress - import->base;
        if (!elided) RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
            if (layers != nullptr && !(*layers)[layer]) continue;
            for (std::uint32_t level = 0; level < mips.size(); ++level) {
                if (!geometry.HasLayer(level, layer)) continue;
                const auto& mip = mips[level];
                detiler.Dispatch(commands, descriptor.tileMode, elementBytes, import->buffer, importOffset + geometry.GuestLayerOffset(layer) + mip.tiledOffset, linear->Handle(), geometry.LinearLayerOffset(layer) + mip.linearOffset, mip, false, layer, geometry.thick);
            }
        }
        const auto linearRead = WholeBufferBarrier(linear->Handle(), VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.srcAccessMask = layers != nullptr ? VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0u;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        // Some layers keep their content: the others are replaced, not the whole image.
        toTransfer.oldLayout = layers != nullptr ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = image;
        toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
        const auto regions = CopyRegions(layers);
        if (elided) {
            toTransfer.srcAccessMask = 0;
            toTransfer.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &linearRead, layers != nullptr ? 0u : 1u, &toTransfer);
            context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, linear->Handle(), image, VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(regions.size()), regions.data());
        } else {
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &linearRead, 1, &toTransfer);
            context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, linear->Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()), regions.data());
            VkImageMemoryBarrier toGeneral = toTransfer;
            toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        }
        if (batch) batch->SubmitAndWait();
        else recorder->EndGpuTiming(timing, uploadedBytes);
        countStorageUpload(1, uploadedBytes);
        ++Profile().storageDirectUploads;
        ++version;
        if (profile) LookupOutcomes::Add(LookupOutcomes::UploadDirect, start);
        return;
    }
    originalValid = true;
    forgetBorrowed(0, trackedLayers);
    stampLayers(false);
    GuestMemory::ReadCommitted(descriptor.baseAddress, original);
    {
            Buffer staging(context, original.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            if (uploadedKeys == DccKeys::Uncompressed) std::memcpy(staging.Bytes().data(), original.data(), original.size());
            else ReadTextureSurface(descriptor, uploadedKeys, staging.Bytes().first(original.size()));
            DeviceBuffer tiled(context, original.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            DeviceBuffer linear(context, static_cast<std::size_t>(linearBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            detiler.BeginBatch();
            CommandBatch batch(context);
            const auto commands = batch.Handle();
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            CopyBuffer(context, commands, staging.Handle(), 0, tiled.Handle(), 0, original.size());
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
                for (std::uint32_t level = 0; level < mips.size(); ++level) {
                    if (!geometry.HasLayer(level, layer)) continue;
                    const auto& mip = mips[level];
                    detiler.Dispatch(commands, descriptor.tileMode, elementBytes, tiled.Handle(), geometry.GuestLayerOffset(layer) + mip.tiledOffset, linear.Handle(), geometry.LinearLayerOffset(layer) + mip.linearOffset, mip, false, layer, geometry.thick);
                }
            }
            const auto linearRead = WholeBufferBarrier(linear.Handle(), VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toTransfer.image = image;
            toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &linearRead, 1, &toTransfer);
            const auto regions = CopyRegions();
            context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, linear.Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()), regions.data());
            VkImageMemoryBarrier toGeneral = toTransfer;
            toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
            APS5_LOG_CHARS_OUT("StorageTexture upload submit");
            batch.SubmitAndWait();
            APS5_LOG_CHARS_OUT("StorageTexture upload done");
    }
    countStorageUpload(2, guestBytes);
    ++version;
    if (profile) LookupOutcomes::Add(LookupOutcomes::UploadCpu, start);
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> StorageTexture::unitRuns(const std::vector<bool>& units) const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    for (std::uint32_t unit = 0; unit < trackedLayers; ++unit) {
        if (!units[unit]) continue;
        const auto begin = static_cast<std::uint64_t>(unit) * trackedLayerBytes;
        const auto end = begin + layerBytes(unit);
        if (!runs.empty() && runs.back().second == begin) runs.back().second = end;
        else runs.emplace_back(begin, end);
    }
    return runs;
}

std::vector<StorageTexture::SliceWindow> StorageTexture::sliceWindows(std::span<const std::pair<std::uint64_t, std::uint64_t>> runs) const {
    std::vector<SliceWindow> windows;
    const auto elementBytes = BytesPerElement(descriptor.format);
    const auto block = WindowBlockLayout(descriptor, geometry);
    for (const auto& [runBegin, runEnd] : runs) {
        for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
            for (std::uint32_t level = 0; level < descriptor.mipCount; ++level) {
                if (!geometry.HasLayer(level, layer)) continue;
                const auto& mip = mips[level];
                const auto sliceBegin = geometry.GuestLayerOffset(layer) + mip.tiledOffset;
                const auto sliceEnd = sliceBegin + mip.tiledSize;
                if (runEnd <= sliceBegin || sliceEnd <= runBegin) continue;
                SliceWindow window{layer, level, 0, 0, {}, 0, {}};
                const auto pitch = static_cast<std::uint32_t>(mip.pitchBytes / elementBytes * BlockWidth(descriptor.format));
                if (mip.tail) {
                    // The tail block holds every tail level: each is moved whole.
                    window.tiledBegin = sliceBegin;
                    window.tiledEnd = sliceEnd;
                    window.window = {0, static_cast<std::uint32_t>(mip.tiledSize), 0, 0, mip.linearSize};
                    window.linearBytes = mip.linearSize;
                    VkBufferImageCopy region{};
                    region.bufferRowLength = pitch;
                    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, geometry.CopyLayer(layer), 1};
                    region.imageOffset = {0, 0, geometry.CopyDepth(layer)};
                    region.imageExtent = {mip.width, mip.height, 1u};
                    window.regions.push_back(region);
                    windows.push_back(std::move(window));
                    continue;
                }
                const auto low = std::max(runBegin, sliceBegin) - sliceBegin;
                const auto high = std::min(runEnd, sliceEnd) - sliceBegin;
                Require(low % block[0] == 0 && high % block[0] == 0, "storage image window is not tile-block aligned");
                const auto firstBlock = low / block[0];
                const auto lastBlock = (high - 1) / block[0];
                const auto firstRow = static_cast<std::uint32_t>(firstBlock / mip.blocksPerRow);
                const auto lastRow = static_cast<std::uint32_t>(lastBlock / mip.blocksPerRow);
                const auto rowBegin = firstRow * block[2];
                const auto rowEnd = std::min(mip.height, (lastRow + 1u) * block[2]);
                window.tiledBegin = sliceBegin + low;
                window.tiledEnd = sliceBegin + high;
                window.linearBytes = static_cast<std::uint64_t>(rowEnd - rowBegin) * mip.pitchBytes;
                window.window = {static_cast<std::uint32_t>(low), static_cast<std::uint32_t>(high), static_cast<std::uint32_t>(low), static_cast<std::uint32_t>(rowBegin) * mip.pitchBytes, window.linearBytes};
                // The dispatched rectangle: the block rows, and inside one block row the block span.
                window.window.rowBegin = rowBegin;
                window.window.rowEnd = rowEnd;
                if (firstRow == lastRow) {
                    window.window.columnBegin = static_cast<std::uint32_t>(firstBlock - static_cast<std::uint64_t>(firstRow) * mip.blocksPerRow) * block[1];
                    window.window.columnEnd = std::min(mip.width, static_cast<std::uint32_t>(lastBlock - static_cast<std::uint64_t>(firstRow) * mip.blocksPerRow + 1u) * block[1]);
                }
                // One region per block row; whole rows merge (a run's middle rows are whole).
                for (auto row = firstRow; row <= lastRow; ++row) {
                    const auto rowFirst = std::max<std::uint64_t>(firstBlock, static_cast<std::uint64_t>(row) * mip.blocksPerRow);
                    const auto rowLast = std::min<std::uint64_t>(lastBlock, static_cast<std::uint64_t>(row + 1u) * mip.blocksPerRow - 1u);
                    const auto x0 = static_cast<std::uint32_t>(rowFirst - static_cast<std::uint64_t>(row) * mip.blocksPerRow) * block[1];
                    const auto x1 = std::min(mip.width, static_cast<std::uint32_t>(rowLast - static_cast<std::uint64_t>(row) * mip.blocksPerRow + 1u) * block[1]);
                    const auto y0 = row * block[2];
                    const auto y1 = std::min(mip.height, (row + 1u) * block[2]);
                    if (x0 >= x1 || y0 >= y1) continue;
                    if (!window.regions.empty()) {
                        auto& previous = window.regions.back();
                        const bool wholeRows = previous.imageOffset.x == 0 && previous.imageExtent.width == mip.width && x0 == 0 && x1 == mip.width;
                        if (wholeRows && static_cast<std::uint32_t>(previous.imageOffset.y) + previous.imageExtent.height == y0) {
                            previous.imageExtent.height += y1 - y0;
                            continue;
                        }
                    }
                    VkBufferImageCopy region{};
                    region.bufferOffset = static_cast<std::uint64_t>(y0 - rowBegin) * mip.pitchBytes + static_cast<std::uint64_t>(x0) * elementBytes;
                    region.bufferRowLength = pitch;
                    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, geometry.CopyLayer(layer), 1};
                    region.imageOffset = {static_cast<std::int32_t>(x0), static_cast<std::int32_t>(y0), geometry.CopyDepth(layer)};
                    region.imageExtent = {x1 - x0, y1 - y0, 1u};
                    window.regions.push_back(region);
                }
                windows.push_back(std::move(window));
            }
        }
    }
    return windows;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> StorageTexture::uncoveredBytes(std::span<const std::pair<std::uint64_t, std::uint64_t>> runs) const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> covered;
    if (!geometry.thick) {
        const auto elementBytes = BytesPerElement(descriptor.format);
        for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
            for (std::uint32_t level = 0; level < descriptor.mipCount; ++level) {
                if (!geometry.HasLayer(level, layer)) continue;
                const auto sliceBegin = geometry.GuestLayerOffset(layer) + mips[level].tiledOffset;
                for (const auto& [begin, end] : CoveredMipBytes(descriptor.tileMode, elementBytes, mips[level])) covered.emplace_back(sliceBegin + begin, sliceBegin + end);
            }
        }
        std::sort(covered.begin(), covered.end());
    } else if (blockUnits && geometry.layerBytes != 0) {
        // A slab's whole tile blocks, when every depth slice of the slab exists at that level: each
        // block of a covered row holds one element at every byte (the thin rule per slab).
        const auto block = WindowBlockLayout(descriptor, geometry);
        const auto slabs = static_cast<std::uint32_t>(guestBytes / geometry.layerBytes);
        for (std::uint32_t level = 0; level < descriptor.mipCount; ++level) {
            const auto& mip = mips[level];
            if (mip.tail) continue;
            const auto depth = std::max(geometry.imageDepth >> level, 1u);
            const auto wholeColumns = std::min(mip.width / block[1], mip.blocksPerRow);
            const auto wholeRows = mip.height / block[2];
            for (std::uint32_t slab = 0; slab < slabs && (slab + 1u) * geometry.blockDepth <= depth; ++slab) {
                const auto sliceBegin = static_cast<std::uint64_t>(slab) * geometry.layerBytes + mip.tiledOffset;
                for (std::uint32_t row = 0; row < wholeRows && wholeColumns != 0; ++row) {
                    const auto first = static_cast<std::uint64_t>(row) * mip.blocksPerRow;
                    covered.emplace_back(sliceBegin + first * block[0], sliceBegin + (first + wholeColumns) * block[0]);
                }
            }
        }
        std::sort(covered.begin(), covered.end());
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted(runs.begin(), runs.end());
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> uncovered;
    const auto add = [&](std::uint64_t begin, std::uint64_t end) {
        if (begin >= end) return;
        if (!uncovered.empty() && uncovered.back().second >= begin) uncovered.back().second = std::max(uncovered.back().second, end);
        else uncovered.emplace_back(begin, end);
    };
    std::size_t next = 0;
    for (const auto& [runBegin, runEnd] : sorted) {
        while (next < covered.size() && covered[next].second <= runBegin) ++next;
        auto at = runBegin;
        for (auto i = next; i < covered.size() && covered[i].first < runEnd; ++i) {
            add(at, std::min(covered[i].first, runEnd));
            at = std::max(at, covered[i].second);
        }
        add(at, runEnd);
    }
    return uncovered;
}

StorageTexture::PaddingSeeds StorageTexture::paddingSeeds(const HostImport& import, std::vector<CopiedBytes> copied) const {
    PaddingSeeds seeds;
    std::sort(copied.begin(), copied.end(), [](const CopiedBytes& a, const CopiedBytes& b) { return a.begin < b.begin; });
    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    for (const auto& range : copied) runs.emplace_back(range.begin, range.end);
    const auto uncovered = uncoveredBytes(runs);
    if (uncovered.empty()) return seeds;
    std::vector<CopiedBytes> pieces;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pieceRuns;
    std::size_t next = 0;
    for (const auto& [begin, end] : uncovered) {
        while (next < copied.size() && copied[next].end <= begin) ++next;
        for (auto i = next; i < copied.size() && copied[i].begin < end; ++i) {
            const auto from = std::max(begin, copied[i].begin);
            const auto to = std::min(end, copied[i].end);
            if (from >= to) continue;
            pieces.push_back({from, to, copied[i].scratch + (from - copied[i].begin)});
            pieceRuns.emplace_back(from, to);
        }
    }
    if (pieces.empty()) return seeds;
    const auto sources = ShadowSources(context, import, descriptor.baseAddress, pieceRuns, {}, false);
    std::size_t first = 0;
    for (const auto& source : sources) {
        while (first < pieces.size() && pieces[first].end <= source.begin) ++first;
        for (auto i = first; i < pieces.size() && pieces[i].begin < source.end; ++i) {
            const auto from = std::max(source.begin, pieces[i].begin);
            const auto to = std::min(source.end, pieces[i].end);
            if (from >= to) continue;
            auto target = std::find_if(seeds.copies.begin(), seeds.copies.end(), [&](const auto& entry) { return entry.first == source.buffer; });
            if (target == seeds.copies.end()) target = seeds.copies.insert(seeds.copies.end(), {source.buffer, {}});
            target->second.push_back({source.offset + (from - source.begin), pieces[i].scratch + (from - pieces[i].begin), to - from});
            if (!source.shadow) {
                const auto guestBegin = descriptor.baseAddress + from;
                if (!seeds.importReads.empty() && seeds.importReads.back().second == guestBegin) seeds.importReads.back().second = descriptor.baseAddress + to;
                else seeds.importReads.emplace_back(guestBegin, descriptor.baseAddress + to);
            } else {
                seeds.shadowReads.emplace_back(descriptor.baseAddress + from, descriptor.baseAddress + to);
            }
        }
        if (source.shadow && std::find(seeds.slabs.begin(), seeds.slabs.end(), source.slab) == seeds.slabs.end()) seeds.slabs.push_back(source.slab);
    }
    return seeds;
}

std::uint64_t StorageTexture::uploadWindows(const HostImport& import, std::span<const std::pair<std::uint64_t, std::uint64_t>> runs, bool discard) {
    const auto elementBytes = BytesPerElement(descriptor.format);
    // Each run's pieces by source (a unit shadow's slab while fresh, else the import), the tail
    // blocks moved whole by one window each; the windows are built per piece.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> tailBlocks;
    for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
        for (std::uint32_t level = 0; level < descriptor.mipCount; ++level) {
            if (!geometry.HasLayer(level, layer)) continue;
            const auto& mip = mips[level];
            if (!mip.tail) continue;
            const auto sliceBegin = geometry.GuestLayerOffset(layer) + mip.tiledOffset;
            if (tailBlocks.empty() || tailBlocks.back().first != sliceBegin) tailBlocks.emplace_back(sliceBegin, sliceBegin + mip.tiledSize);
        }
    }
    const auto sources = ShadowSources(context, import, descriptor.baseAddress, runs, tailBlocks);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pieces;
    bool allShadow = true;
    for (const auto& source : sources) {
        pieces.emplace_back(source.begin, source.end);
        if (!source.shadow) allShadow = false;
    }
    if (allShadow && !sources.empty() && (std::strcmp(uploadReason, "flushed") == 0 || std::strcmp(uploadReason, "store") == 0)) uploadReason = "shadow";
    auto windows = sliceWindows(pieces);
    Require(!windows.empty(), "storage image upload selects no unit");
    const auto sourceOf = [&](const SliceWindow& window) -> const ShadowRun& {
        for (const auto& source : sources) {
            if (source.begin <= window.tiledBegin && window.tiledBegin < source.end) return source;
        }
        throw std::runtime_error("AGC graphics: storage image window has no source");
    };
    // One linear buffer for every window, each at a 256-byte position (the detiler's descriptor
    // offsets are aligned by the dispatch).
    std::vector<std::uint64_t> positions;
    std::uint64_t linearTotal = 0, uploadedBytes = 0;
    for (const auto& window : windows) {
        positions.push_back(linearTotal);
        linearTotal += (window.linearBytes + 255) & ~std::uint64_t{255};
    }
    for (const auto& [begin, end] : runs) uploadedBytes += end - begin;
    auto linear = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(linearTotal), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    detiler.BeginBatch();
    auto* recorder = Recorder::Active();
    std::unique_ptr<CommandBatch> batch;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    auto timing = Recorder::NoTiming;
    bool elided = false;
    if (recorder != nullptr) {
        // The import pieces are read in place when the batch runs; the slab pieces are the
        // device's own.
        std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
        for (const auto& source : sources) {
            if (!source.shadow) reads.emplace_back(descriptor.baseAddress + source.begin, descriptor.baseAddress + source.end);
        }
        // A queued label store in a run lands before the detile reads it.
        for (const auto& [begin, end] : reads) recorder->FlushStoresOverlapping(begin, static_cast<std::size_t>(end - begin));
        std::vector<std::pair<std::uint64_t, std::uint64_t>> tracked;
        for (const auto& source : sources) tracked.emplace_back(descriptor.baseAddress + source.begin, descriptor.baseAddress + source.end);
        const std::pair<VkImage, bool> written{image, true};
        commands = Recorder::BarrierTracking() ? recorder->Commands(Recorder::CommandClass::StorageUpload, Recorder::Access{tracked, {}, std::span(&written, 1), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT}, elided) : recorder->Commands();
        timing = recorder->BeginGpuTiming(Recorder::CommandClass::StorageUpload);
        recorder->NoteTimedSurface(timing, descriptor.baseAddress, guestBytes, static_cast<std::uint32_t>(storageFormat), descriptor.format, descriptor.width, descriptor.height, "upload-windows", static_cast<std::uint32_t>(runs.size()));
        if (GpuCheckpoints::Active().load(std::memory_order_relaxed)) {
            char text[160];
            std::snprintf(text, sizeof(text), "window upload %ux%u fmt%u tile%d mips %u, %zu runs %zu sources", descriptor.width, descriptor.height, descriptor.format, static_cast<int>(descriptor.tileMode), descriptor.mipCount, runs.size(), sources.size());
            GpuCheckpoints::Set(context, commands, "storage-upload windows (surface, bytes, linear)", descriptor.baseAddress, guestBytes, linear->Size(), text);
        }
        recorder->Keep(linear, linear->Size());
        for (const auto& source : sources) {
            if (source.slab != nullptr) recorder->Keep(source.slab);
        }
        if (auto self = weak_from_this().lock()) recorder->Keep(std::move(self));
        if (!reads.empty()) recorder->NotePendingReads(reads, Recorder::ReadKind::StorageUpload);
        Recorder::CountBarriers(Recorder::CommandClass::StorageUpload, elided ? 1 : 3);
        if (elided) Recorder::CountSkipped(Recorder::CommandClass::StorageUpload, 2);
    } else {
        batch = std::make_unique<CommandBatch>(context);
        commands = batch->Handle();
    }
    const auto importOffset = descriptor.baseAddress - import.base;
    if (!elided) RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    std::vector<VkBufferImageCopy> regions;
    for (std::size_t i = 0; i < windows.size(); ++i) {
        const auto& window = windows[i];
        const auto& mip = mips[window.level];
        const auto& source = sourceOf(window);
        auto detile = window.window;
        if (source.shadow) {
            // The slab holds the piece from the window's first tiled byte: the window's tiled base
            // is its range begin (sliceWindows builds it so), read from the piece's slab offset.
            detiler.Dispatch(commands, descriptor.tileMode, elementBytes, source.buffer, source.offset + (window.tiledBegin - source.begin), linear->Handle(), positions[i], mip, false, window.layer, geometry.thick, detile);
        } else {
            // The import holds the whole mip: the window's tiled base is the mip's.
            detile.tiledBase = 0;
            detiler.Dispatch(commands, descriptor.tileMode, elementBytes, import.buffer, importOffset + geometry.GuestLayerOffset(window.layer) + mip.tiledOffset, linear->Handle(), positions[i], mip, false, window.layer, geometry.thick, detile);
        }
        for (auto region : window.regions) {
            region.bufferOffset += positions[i];
            regions.push_back(region);
        }
    }
    const auto linearRead = WholeBufferBarrier(linear->Handle(), VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toTransfer.srcAccessMask = discard ? 0u : VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    // The other blocks keep their content (a new image, every block of which is written, has none).
    toTransfer.oldLayout = discard ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = image;
    toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
    if (elided) {
        toTransfer.srcAccessMask = 0;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &linearRead, discard ? 1u : 0u, &toTransfer);
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, linear->Handle(), image, VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(regions.size()), regions.data());
    } else {
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &linearRead, 1, &toTransfer);
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, linear->Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()), regions.data());
        VkImageMemoryBarrier toGeneral = toTransfer;
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
    }
    if (batch) batch->SubmitAndWait();
    else recorder->EndGpuTiming(timing, uploadedBytes);
    return uploadedBytes;
}

std::uint64_t StorageTexture::writeBackWindows(const HostImport& import, std::span<const std::pair<std::uint64_t, std::uint64_t>> keep, std::uint64_t firstStored, std::uint64_t lastStored, std::vector<ShadowedRange>& shadowed, std::vector<std::pair<std::uint64_t, std::uint64_t>>& imported) {
    const auto elementBytes = BytesPerElement(descriptor.format);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
    std::uint64_t storedBytes = 0;
    for (const auto& [from, to] : keep) {
        runs.emplace_back(from - descriptor.baseAddress, to - descriptor.baseAddress);
        storedBytes += to - from;
    }
    if (runs.empty()) return 0;
    auto windows = sliceWindows(runs);
    // The linear rows and the tiled bytes of every window, each at a 256-byte position; the tail
    // levels of a layer retile into one scratch region (their block is shared).
    std::vector<std::uint64_t> linearPositions, scratchPositions;
    std::uint64_t linearTotal = 0, scratchTotal = 0;
    for (std::size_t i = 0; i < windows.size(); ++i) {
        const auto& window = windows[i];
        linearPositions.push_back(linearTotal);
        linearTotal += (window.linearBytes + 255) & ~std::uint64_t{255};
        std::size_t shared = 0;
        while (shared < i && (windows[shared].tiledBegin != window.tiledBegin || windows[shared].tiledEnd != window.tiledEnd)) ++shared;
        if (shared < i) {
            scratchPositions.push_back(scratchPositions[shared]);
            continue;
        }
        scratchPositions.push_back(scratchTotal);
        scratchTotal += (window.tiledEnd - window.tiledBegin + 255) & ~std::uint64_t{255};
    }
    // The kept ranges inside each window, from the window's scratch (a shared tail region once)
    // to their destination: the import's unit shadow where a slab takes the piece (split at slab
    // boundaries; units a piece covers partly and does not already hold fresh are seeded from the
    // import first), else the import as before. Decided before anything is recorded: making a
    // slab may evict another with a publish of its own (a chosen slab stays pinned by its
    // destination until the pieces are marked, so no later choice evicts it).
    struct SlabPieces {
        std::shared_ptr<ShadowSlab> slab;
        std::shared_ptr<ShadowSlabPin> pin;
        std::vector<VkBufferCopy> copies;
    };
    std::vector<SlabPieces> slabPieces;
    std::vector<VkBufferCopy> importCopies;
    std::vector<ShadowedRange> seeds;
    // The same pieces for the fused write-back: window, guest range, destination buffer and offset.
    struct DirectPiece {
        std::size_t window;
        std::uint64_t begin;
        std::uint64_t end;
        VkBuffer buffer;
        std::uint64_t offset;
    };
    std::vector<DirectPiece> directPieces;
    std::size_t currentWindow = 0;
    const auto addPiece = [&](std::uint64_t scratchOffset, std::uint64_t guestBegin, std::uint64_t guestEnd) {
        while (guestBegin < guestEnd) {
            const auto pieceEnd = UnitShadowEnabled() ? std::min(guestEnd, SlabBoundary(import, guestBegin)) : guestEnd;
            const auto destination = ShadowDestinationFor(context, import, guestBegin, pieceEnd);
            if (destination.has_value()) {
                if (slabPieces.empty() || slabPieces.back().slab != destination->slab) slabPieces.push_back({destination->slab, destination->pin, {}});
                slabPieces.back().copies.push_back({scratchOffset, destination->offset, pieceEnd - guestBegin});
                directPieces.push_back({currentWindow, guestBegin, pieceEnd, destination->slab->buffer, destination->offset});
                for (const auto& [seedBegin, seedEnd] : destination->seedUnits) {
                    if (std::none_of(seeds.begin(), seeds.end(), [&](const ShadowedRange& seed) { return seed.begin == seedBegin; })) seeds.push_back({seedBegin, seedEnd, destination->slab});
                }
                if (!shadowed.empty() && shadowed.back().end == guestBegin && shadowed.back().slab == destination->slab) shadowed.back().end = pieceEnd;
                else shadowed.push_back({guestBegin, pieceEnd, destination->slab});
            } else {
                importCopies.push_back({scratchOffset, guestBegin - import.base, pieceEnd - guestBegin});
                directPieces.push_back({currentWindow, guestBegin, pieceEnd, import.buffer, guestBegin - import.base});
                if (!imported.empty() && imported.back().second == guestBegin) imported.back().second = pieceEnd;
                else imported.emplace_back(guestBegin, pieceEnd);
            }
            scratchOffset += pieceEnd - guestBegin;
            guestBegin = pieceEnd;
        }
    };
    for (std::size_t i = 0; i < windows.size(); ++i) {
        const auto& window = windows[i];
        // The tail levels follow each other, sharing the previous one's region.
        if (i != 0 && scratchPositions[i] == scratchPositions[i - 1]) continue;
        currentWindow = i;
        for (const auto& [from, to] : runs) {
            const auto begin = std::max(from, window.tiledBegin);
            const auto end = std::min(to, window.tiledEnd);
            if (begin >= end) continue;
            addPiece(scratchPositions[i] + (begin - window.tiledBegin), descriptor.baseAddress + begin, descriptor.baseAddress + end);
        }
    }
    std::sort(seeds.begin(), seeds.end(), [](const ShadowedRange& a, const ShadowedRange& b) { return a.begin < b.begin; });
    std::vector<CopiedBytes> copied;
    for (std::size_t i = 0; i < windows.size(); ++i) {
        const auto& window = windows[i];
        if (i != 0 && scratchPositions[i] == scratchPositions[i - 1]) continue;
        for (const auto& [from, to] : runs) {
            const auto begin = std::max(from, window.tiledBegin);
            const auto end = std::min(to, window.tiledEnd);
            if (begin < end) copied.push_back({begin, end, scratchPositions[i] + (begin - window.tiledBegin)});
        }
    }
    const auto padding = paddingSeeds(import, std::move(copied));
    // A queued label or key store inside a seeded unit lands before the seed copies the import.
    auto flushBegin = firstStored;
    auto flushEnd = lastStored;
    for (const auto& seed : seeds) {
        flushBegin = std::min(flushBegin, seed.begin);
        flushEnd = std::max(flushEnd, seed.end);
    }
    // APS5_FUSED_WRITEBACK=1: one dispatch per piece retiles the image's texels straight into the
    // piece's destination (no linear copy, no scratch, no copies out): two barriers instead of four
    // or five. Not with padding bytes to carry (they go through the scratch).
    static const bool fusedWriteBack = std::getenv("APS5_FUSED_WRITEBACK") != nullptr;
    const bool fused = fusedWriteBack && context.pushDescriptors && padding.copies.empty() && rawFormat() != VK_FORMAT_UNDEFINED;
    std::shared_ptr<DeviceBuffer> linear, tiledScratch;
    if (!fused) {
        linear = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(linearTotal), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        tiledScratch = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(scratchTotal), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }
    detiler.BeginBatch();
    auto* recorder = Recorder::Active();
    std::unique_ptr<CommandBatch> batch;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkAccessFlags covered = 0;
    auto timing = Recorder::NoTiming;
    bool elided = false;
    if (recorder != nullptr) {
        recorder->FlushKeyStoresOverlapping(flushBegin, static_cast<std::size_t>(flushEnd - flushBegin));
        recorder->FlushStoresOverlapping(flushBegin, static_cast<std::size_t>(flushEnd - flushBegin));
        if (Recorder::BarrierTracking()) {
            const std::pair<VkImage, bool> read{image, false};
            std::vector<std::pair<std::uint64_t, std::uint64_t>> seedReads;
            for (const auto& seed : seeds) seedReads.emplace_back(seed.begin, seed.end);
            seedReads.insert(seedReads.end(), padding.importReads.begin(), padding.importReads.end());
            seedReads.insert(seedReads.end(), padding.shadowReads.begin(), padding.shadowReads.end());
            std::vector<std::pair<std::uint64_t, std::uint64_t>> written(imported.begin(), imported.end());
            for (const auto& seed : seeds) written.emplace_back(seed.begin, seed.end);
            for (const auto& range : shadowed) written.emplace_back(range.begin, range.end);
            commands = recorder->Commands(Recorder::CommandClass::StorageWriteBack, Recorder::Access{seedReads, written, std::span(&read, 1), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT}, elided, &covered);
        } else {
            commands = recorder->Commands(&covered);
        }
        timing = recorder->BeginGpuTiming(Recorder::CommandClass::StorageWriteBack);
        recorder->NoteTimedSurface(timing, descriptor.baseAddress, guestBytes, static_cast<std::uint32_t>(storageFormat), descriptor.format, descriptor.width, descriptor.height, CauseKind("writeback-windows"), static_cast<std::uint32_t>(keep.size()));
        if (linear) recorder->Keep(linear, linear->Size());
        if (tiledScratch) recorder->Keep(tiledScratch, tiledScratch->Size());
        for (const auto& pieces : slabPieces) recorder->Keep(pieces.slab);
        for (const auto& slab : padding.slabs) recorder->Keep(slab);
        if (auto self = weak_from_this().lock()) recorder->Keep(std::move(self));
        Recorder::CountBarriers(Recorder::CommandClass::StorageWriteBack, fused ? 2 : elided ? 2 : 4);
        if (elided) Recorder::CountSkipped(Recorder::CommandClass::StorageWriteBack, 2);
    } else {
        batch = std::make_unique<CommandBatch>(context);
        commands = batch->Handle();
    }
    // Transfer writes are in the destination mask too: a following publish or detile of the slabs
    // merges its leading barrier.
    constexpr VkAccessFlags storedAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    if (fused) {
        fusedWriteBacks.fetch_add(1, std::memory_order_relaxed);
        // The image's writers (its attachment and shader writes) and every earlier writer or
        // reader of the destinations come first; the image stays in the general layout.
        if (!elided) RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        if (!seeds.empty()) {
            for (const auto& [begin, end, slab] : seeds) {
                const VkBufferCopy seed{begin - import.base, SlabOffset(import, *slab, begin), end - begin};
                context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, import.buffer, slab->buffer, 1, &seed);
                NoteShadowSeed(begin, end);
            }
            // The seeded bytes precede the retile that overwrites part of them.
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            Recorder::CountBarriers(Recorder::CommandClass::StorageWriteBack);
        }
        for (const auto& piece : directPieces) {
            const auto& first = windows[piece.window];
            for (std::size_t j = piece.window; j < windows.size(); ++j) {
                // The tail levels of one layer share the piece's block: each writes its texels.
                const auto& window = windows[j];
                if (j != piece.window && (window.tiledBegin != first.tiledBegin || window.tiledEnd != first.tiledEnd)) break;
                const auto sliceBegin = descriptor.baseAddress + window.tiledBegin - window.window.rangeBegin;
                auto range = window.window;
                range.rangeBegin = static_cast<std::uint32_t>(piece.begin - sliceBegin);
                range.rangeEnd = static_cast<std::uint32_t>(piece.end - sliceBegin);
                range.tiledBase = range.rangeBegin;
                detiler.DispatchImage(commands, descriptor.tileMode, elementBytes, rawView(window.level), piece.buffer, piece.offset, mips[window.level], true, geometry.CopyLayer(window.layer), range);
            }
        }
        const VkMemoryBarrier stored{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, storedAccess};
        if (!elided) context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &stored, 0, nullptr, 0, nullptr);
    } else {
    VkImageMemoryBarrier toSource{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toSource.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toSource.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSource.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSource.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSource.image = image;
    toSource.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
    if (!elided) context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toSource);
    if (!seeds.empty() || !padding.copies.empty()) {
        // The seeded units' import bytes (every earlier writer of them, host stores included)
        // precede the seed copies; the previous command's trailing barrier may have covered that.
        constexpr VkAccessFlags transferAccess = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        if (elided) {
            Recorder::CountSkipped(Recorder::CommandClass::StorageWriteBack);
        } else if (recorder != nullptr && (covered & transferAccess) == transferAccess && Recorder::MergeBarriers()) {
            Recorder::CountMerged(Recorder::CommandClass::StorageWriteBack);
        } else {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, transferAccess);
            Recorder::CountBarriers(Recorder::CommandClass::StorageWriteBack);
        }
        for (const auto& [begin, end, slab] : seeds) {
            // The piece's own slab (a seeded unit is one the piece covers partly).
            const VkBufferCopy seed{begin - import.base, SlabOffset(import, *slab, begin), end - begin};
            context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, import.buffer, slab->buffer, 1, &seed);
            NoteShadowSeed(begin, end);
        }
        // The padding's current bytes, under the retile that follows (the import-ready barrier
        // orders the scratch writes).
        for (const auto& [source, copies] : padding.copies) context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, source, tiledScratch->Handle(), static_cast<std::uint32_t>(copies.size()), copies.data());
    }
    std::vector<VkBufferImageCopy> regions;
    for (std::size_t i = 0; i < windows.size(); ++i) {
        for (auto region : windows[i].regions) {
            region.bufferOffset += linearPositions[i];
            regions.push_back(region);
        }
    }
    context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image, elided ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, linear->Handle(), static_cast<std::uint32_t>(regions.size()), regions.data());
    const auto linearRead = WholeBufferBarrier(linear->Handle(), VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    const VkMemoryBarrier importReady{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, elided ? VkAccessFlags{VK_ACCESS_TRANSFER_WRITE_BIT} : VkAccessFlags{VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT}, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT};
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, elided ? VkPipelineStageFlags{VK_PIPELINE_STAGE_TRANSFER_BIT} : VkPipelineStageFlags{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &importReady, 1, &linearRead, 0, nullptr);
    for (std::size_t i = 0; i < windows.size(); ++i) {
        const auto& window = windows[i];
        detiler.Dispatch(commands, descriptor.tileMode, elementBytes, linear->Handle(), linearPositions[i], tiledScratch->Handle(), scratchPositions[i], mips[window.level], true, window.layer, geometry.thick, window.window);
    }
    {
        // The retiled scratch (and a seed's slab bytes, which the scratch copies overwrite in
        // part) precede the copies into the slabs and the import.
        const VkMemoryBarrier scratchDone{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &scratchDone, 0, nullptr, 0, nullptr);
        const auto copyBuffer = context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer");
        for (const auto& pieces : slabPieces) copyBuffer(commands, tiledScratch->Handle(), pieces.slab->buffer, static_cast<std::uint32_t>(pieces.copies.size()), pieces.copies.data());
        if (!importCopies.empty()) copyBuffer(commands, tiledScratch->Handle(), import.buffer, static_cast<std::uint32_t>(importCopies.size()), importCopies.data());
    }
    VkImageMemoryBarrier backToGeneral = toSource;
    backToGeneral.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    backToGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    backToGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    backToGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    const VkMemoryBarrier stored{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, storedAccess};
    if (!elided) context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &stored, 0, nullptr, 1, &backToGeneral);
    }
    if (batch) {
        batch->SubmitAndWait();
    } else {
        recorder->EndGpuTiming(timing, storedBytes);
        if (!elided) recorder->MarkCovered(storedAccess);
        // Only the pieces written into the import are stores a CPU reader must wait for; the
        // shadowed ones reach it through a publish, which notes its own.
        if (!imported.empty()) recorder->NotePendingWrites(imported);
    }
    partialWriteBacks.fetch_add(1, std::memory_order_relaxed);
    partialWriteBackBytes.fetch_add(storedBytes, std::memory_order_relaxed);
    return storedBytes;
}

std::vector<VkBufferImageCopy> StorageTexture::CopyRegions(const std::vector<bool>* layers) const {
    std::vector<VkBufferImageCopy> regions;
    for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
        if (layers != nullptr && !(*layers)[layer]) continue;
        for (std::uint32_t level = 0; level < descriptor.mipCount; ++level) {
            if (!geometry.HasLayer(level, layer)) continue;
            const auto& mip = mips[level];
            VkBufferImageCopy region{};
            region.bufferOffset = layer * sliceLinearBytes + mip.linearOffset;
            region.bufferRowLength = mip.pitchBytes / BytesPerElement(descriptor.format) * BlockWidth(descriptor.format);
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, geometry.CopyLayer(layer), 1};
            region.imageOffset = {0, 0, geometry.CopyDepth(layer)};
            region.imageExtent = {std::max(descriptor.width >> level, 1u), std::max(descriptor.height >> level, 1u), 1u};
            regions.push_back(region);
        }
    }
    return regions;
}

bool StorageTexture::overlaps(std::uint64_t address, std::size_t bytes) const {
    return address < descriptor.baseAddress + guestBytes && descriptor.baseAddress < address + bytes;
}

bool StorageTexture::DirtyQuiet() const {
    static const bool eager = std::getenv("APS5_EAGER_WRITEBACK") != nullptr || std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    if (eager || (descriptor.dccAddress != 0 && IsDccClear(uploadedKeys) && !IsDccClear(filledKeys)) || !borrowedFrom.expired() || !layerBorrows.empty()) return false;
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    if (!dirty || lent || layerPending.size() < trackedLayers) return false;
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (!layerPending[layer]) return false;
    }
    return true;
}

void StorageTexture::MarkDirty() {
    ForkTrap("storage dirty");
    if (ForkQuiescing() && !DirtyQuiet()) ForkQuiesce("storage dirty");
    if (descriptor.dccAddress != 0 && IsDccClear(uploadedKeys) && !IsDccClear(filledKeys)) {
        traceKeyStore("first write", descriptor, guestBytes);
        MarkDccUncompressed(context, descriptor.dccAddress, guestBytes, DccKeyCount(descriptor, guestBytes));
        uploadedKeys = DccKeys::Uncompressed;
    }
    markLayersPending(0, trackedLayers);
}

bool StorageTexture::anyLayerPending() const {
    return std::any_of(layerPending.begin(), layerPending.end(), [](bool pending) { return pending; });
}

void StorageTexture::refreshGeneration() {
    generation = *std::min_element(layerGeneration.begin(), layerGeneration.end());
}

void StorageTexture::markLayersPending(std::uint32_t first, std::uint32_t count) {
    CaptureTrace::Log("image-write image=%llx first=%u count=%u generation=%llu", static_cast<unsigned long long>(descriptor.baseAddress), first, count, static_cast<unsigned long long>(generation));
    static const bool eager = std::getenv("APS5_EAGER_WRITEBACK") != nullptr || std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    // Live `pending_fill` (env APS5_PENDING_FILL=0 turns it off; PendingUnits.hpp): whole words of
    // the bit vector at once. One bit at a time was about 2.7% of Graphics::Draw on the World Map,
    // where every draw marks its whole target pending.
    static const bool fillDefault = [] {
        const char* value = std::getenv("APS5_PENDING_FILL");
        return value == nullptr || value[0] != '0';
    }();
    MarkUnitsPending(layerPending, first, count, LiveFlag("pending_fill", fillDefault));
    // Results of this image now cover the alias's results borrowed into these units: this image
    // stores them, the alias no longer has to (its results, unchanged since the borrow, are in
    // this image's content). An alias written since keeps its own newer results pending.
    if (auto source = borrowedFrom.lock(); source != nullptr && borrowedUnits.size() == trackedLayers) {
        bool superseded = false;
        for (std::uint32_t unit = first; unit < first + count; ++unit) {
            if (!borrowedUnits[unit]) continue;
            borrowedUnits[unit] = false;
            if (source->version != borrowedVersion || !source->layerPending[unit]) continue;
            source->layerPending[unit] = false;
            unitsSuperseded.fetch_add(1, std::memory_order_relaxed);
            superseded = true;
        }
        if (superseded) source->reconcilePending();
    }
    // Likewise for the units taken from layer aliases, each at its own unit offset.
    for (auto& record : layerBorrows) {
        auto source = record.from.lock();
        bool superseded = false;
        for (std::uint32_t unit = first; unit < first + count && unit < record.units.size(); ++unit) {
            if (!record.units[unit]) continue;
            record.units[unit] = false;
            const auto sourceUnit = static_cast<std::int64_t>(unit) + record.offset;
            if (source == nullptr || source->version != record.version || sourceUnit < 0 || sourceUnit >= static_cast<std::int64_t>(source->trackedLayers) || !source->layerPending[static_cast<std::size_t>(sourceUnit)]) continue;
            source->layerPending[static_cast<std::size_t>(sourceUnit)] = false;
            unitsSuperseded.fetch_add(1, std::memory_order_relaxed);
            superseded = true;
        }
        if (superseded) source->reconcilePending();
    }
    std::erase_if(layerBorrows, [](const LayerBorrow& record) { return std::none_of(record.units.begin(), record.units.end(), [](bool held) { return held; }); });
    if (eager) {
        WriteBack();
        return;
    }
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    // The active recorder installs a hook that also waits for recorded work; without one (tests),
    // pending storage results alone are flushed.
    if (Recorder::Active() == nullptr) {
        static const bool hooked = [] {
            GuestMemory::SetFlushHook(&FlushHook);
            return true;
        }();
        static_cast<void>(hooked);
    }
    ++version;
    const bool wasLent = std::exchange(lent, false);
    if (dirty) {
        if (wasLent) BumpPendingSerial();
        return;
    }
    dirty = true;
    pending.textures.push_back(this);
    BumpPendingSerial();
}

void StorageTexture::reconcilePending() {
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    const bool any = anyLayerPending();
    if (any == dirty) return;
    dirty = any;
    if (any) pending.textures.push_back(this);
    else pending.textures.remove(this);
    BumpPendingSerial();
}

void StorageTexture::DumpBaseLevel(const std::string& path) {
    GuestMemory::AssertGpuLockHeld("StorageTexture::DumpBaseLevel");
    Require(!mips.empty(), "storage image dump of an empty image");
    if (auto* recorder = Recorder::Active()) recorder->Sync();
    Buffer host(context, static_cast<std::size_t>(sliceLinearBytes * arrayLayers), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    CommandBatch batch(context);
    const auto commands = batch.Handle();
    VkImageMemoryBarrier toSource{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toSource.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toSource.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSource.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSource.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSource.image = image;
    toSource.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toSource);
    const auto regions = CopyRegions();
    context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, host.Handle(), static_cast<std::uint32_t>(regions.size()), regions.data());
    VkImageMemoryBarrier back = toSource;
    back.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    back.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    back.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    back.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &back);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    batch.SubmitAndWait();
    const auto& mip = mips[0];
    for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
        const auto offset = static_cast<std::size_t>(sliceLinearBytes * layer + mip.linearOffset);
        const auto bytes = host.Bytes().subspan(offset, static_cast<std::size_t>(mip.linearSize));
        auto name = path;
        if (layer != 0) {
            char suffix[16];
            std::snprintf(suffix, sizeof(suffix), "_L%02u.raw", layer);
            name = path.substr(0, path.size() - 4) + suffix;
        }
        std::FILE* file = std::fopen(name.c_str(), "wb");
        Require(file != nullptr, "cannot open the storage image dump file");
        const std::uint32_t header[5] = {0x31474d49u, mip.width, mip.height, mip.pitchBytes, static_cast<std::uint32_t>(storageFormat)};
        std::fwrite(header, sizeof(header), 1, file);
        std::fwrite(bytes.data(), 1, bytes.size(), file);
        std::fclose(file);
    }
}

void StorageTexture::Flush() {
    ForkTrap("storage flush");
    ForkQuiesce("storage flush");
    {
        auto& live = Live();
        std::lock_guard lock(live.mutex);
        released = true;
    }
    {
        auto& pending = Pending();
        std::lock_guard lock(pending.mutex);
        if (!dirty) return;
        dirty = false;
        pending.textures.remove(this);
        BumpPendingSerial();
    }
    std::lock_guard gpu(GuestMemory::GpuMutex());
    const auto previous = std::exchange(flushReason, "cache eviction");
    writeBack(descriptor.baseAddress, static_cast<std::size_t>(guestBytes));
    flushReason = previous;
}

// Whether a pending unit of the image lies inside the range (writeBack's own selection): an image
// listed by FlushPending without one stored nothing and only cost the GPU mutex.
bool StorageTexture::pendingUnitInside(std::uint64_t address, std::size_t bytes) const {
    static const bool pretest = std::getenv("APS5_NO_FLUSH_PRETEST") == nullptr;
    if (!pretest) return true;
    const bool whole = descriptor.dccAddress != 0 && uploadedKeys != DccKeys::Uncompressed;
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (!layerPending[layer]) continue;
        if (whole) return true;
        const auto begin = layerBegin(layer);
        if (address < begin + layerBytes(layer) && begin < address + bytes) return true;
    }
    return false;
}

bool StorageTexture::FlushPending(std::uint64_t address, std::size_t bytes, const StorageTexture* except, const char* reason, PublishScope scope, bool* published) {
    if (published != nullptr) *published = false;
    if (ForkLogged() != nullptr) {
        {
            auto& pending = Pending();
            std::lock_guard lock(pending.mutex);
            const auto* exempt = refreshing;
            for (auto it = pending.textures.MayOverlap(address, bytes) ? pending.textures.begin() : pending.textures.end(); it != pending.textures.end(); ++it) {
                const auto* texture = *it;
                if (texture != except && texture != exempt && !LayerAliasExempted(texture) && texture->overlaps(address, bytes) && texture->pendingUnitInside(address, bytes)) ForkRefuse("storage results flush");
            }
        }
        if (scope != PublishScope::None && AnyShadowedOverlaps(address, bytes)) ForkRefuse("shadow publish");
        return false;
    }
    if (ForkQuiescing()) {
        bool mayFlush = false;
        {
            auto& pending = Pending();
            std::lock_guard lock(pending.mutex);
            mayFlush = pending.textures.MayOverlap(address, bytes);
        }
        if (mayFlush || (scope != PublishScope::None && AnyShadowedOverlaps(address, bytes))) ForkQuiesce("storage results flush");
    }
    // The images stay alive across the scan: their last owner may be a batch's kept list, which
    // another thread releases outside the GPU mutex once the batch completed.
    std::vector<std::shared_ptr<StorageTexture>> flush;
    {
        auto& pending = Pending();
        std::lock_guard lock(pending.mutex);
        const auto* exempt = refreshing;
        for (auto it = pending.textures.MayOverlap(address, bytes) ? pending.textures.begin() : pending.textures.end(); it != pending.textures.end();) {
            auto* texture = *it;
            if (texture != except && texture != exempt && !LayerAliasExempted(texture) && texture->overlaps(address, bytes)) {
                if (!texture->pendingUnitInside(address, bytes)) {
                    pretestSkipped.fetch_add(1, std::memory_order_relaxed);
                    ++it;
                    continue;
                }
                texture->dirty = false;
                if (const auto skips = texture->hookSkips.exchange(0, std::memory_order_relaxed); skips != 0 && texture->skippedResultsInside(address, bytes)) {
                    flushedAfterSkip.fetch_add(1, std::memory_order_relaxed);
                    if (PublishReasonFor(reason) == PublishReason::Hook && texture->hookSkipSite.load(std::memory_order_relaxed) == static_cast<std::uint8_t>(GuestMemory::CurrentReadSite())) flushedAfterSkipSameSite.fetch_add(1, std::memory_order_relaxed);
                }
                if (auto alive = texture->weak_from_this().lock()) flush.push_back(std::move(alive));
                it = pending.textures.erase(it);
            } else {
                ++it;
            }
        }
        // Still pending for the adjacency rule of writeBack until each one is stored.
        for (const auto& texture : flush) pending.flushing.push_back(texture.get());
        if (!flush.empty()) BumpPendingSerial();
    }
    if (flush.empty()) {
        // Units shadowed over the range (another image's results retiled into the import's unit
        // shadow, or an earlier flush's) still reach the import as the scope asks.
        const bool units = PublishShadowsOnly(address, bytes, scope, reason);
        if (published != nullptr) *published = units;
        return false;
    }
    GuestMemory::NoteProbeEvent("storage results flush");
    struct Unregister {
        const std::vector<std::shared_ptr<StorageTexture>>& flush;
        ~Unregister() {
            auto& pending = Pending();
            std::lock_guard lock(pending.mutex);
            for (const auto& texture : flush) std::erase(pending.flushing, texture.get());
            BumpPendingSerial();
        }
    } unregister{flush};
    // Debug aid: APS5_TRACE_FLUSH names what forces pending results to guest memory.
    static const bool trace = std::getenv("APS5_TRACE_FLUSH") != nullptr;
    if (trace) {
        // The flushed image and, for a refresh, the image refreshing (the one that forced it), each
        // with format, tiling, dimension and size, so profiling/cprof/diag-summary.py can aggregate pairs.
        const auto describe = [](const StorageTexture* texture) {
            if (texture == nullptr) return std::string("-");
            const auto& d = texture->descriptor;
            char text[160];
            std::snprintf(text, sizeof(text), "0x%llx+0x%llx fmt%u tile%u dim%u %ux%ux%u", static_cast<unsigned long long>(d.baseAddress), static_cast<unsigned long long>(texture->guestBytes), d.format, static_cast<unsigned>(d.tileMode), static_cast<unsigned>(d.dimension), d.width, d.height, d.depthOrLastArray);
            return std::string(text);
        };
        const auto packet = GuestMemory::CurrentPacket();
        const auto by = describe(refreshing);
        for (const auto& texture : flush) std::fprintf(stderr, "[flush] image 0x%llx+0x%llx for %s 0x%llx+0x%zx (packet 0x%x) flushed {%s} by {%s} site %s\n", static_cast<unsigned long long>(texture->descriptor.baseAddress), static_cast<unsigned long long>(texture->guestBytes), reason, static_cast<unsigned long long>(address), bytes, packet.opcode, describe(texture.get()).c_str(), by.c_str(), g_refreshSite);
    }
    static const bool traceAliases = std::getenv("APS5_TRACE_ALIAS_FLUSH") != nullptr;
    if (traceAliases) {
        static std::mutex aliasMutex;
        static std::map<std::string, std::uint64_t> aliasCounts;
        static std::map<std::string, std::uint64_t> aliasBytes;
        static auto aliasReport = std::chrono::steady_clock::now();
        const auto describe = [](const StorageTexture* texture) {
            if (texture == nullptr) return std::string("cpu");
            const auto& d = texture->descriptor;
            char text[192];
            std::snprintf(text, sizeof(text), "@%llx fmt%u tile%u dim%u %ux%ux%u mips%u/%u base%u arr%u sel%u%u%u%u", static_cast<unsigned long long>(d.baseAddress), d.format, static_cast<unsigned>(d.tileMode), static_cast<unsigned>(d.dimension), d.width, d.height, d.depthOrLastArray, d.baseLevel, d.mipCount, d.lastLevel, d.baseArray, d.dstSelX, d.dstSelY, d.dstSelZ, d.dstSelW);
            return std::string(text);
        };
        std::lock_guard aliasLock(aliasMutex);
        // A 2D image refreshing over a pending array image at its base: the first few say who
        // refreshes it (return addresses, with an anchor to place them in the module).
        if (refreshing != nullptr && refreshing->descriptor.dimension == TextureDimension::k2D && std::string_view(reason) == "storage refresh") {
            static int stacks = 0;
            for (const auto& texture : flush) {
                if (stacks >= 6 || texture->descriptor.baseAddress != refreshing->descriptor.baseAddress || texture->descriptor.dimension == TextureDimension::k2D) continue;
                ++stacks;
                void* frames[24] = {};
                const auto count = RtlCaptureStackBackTrace(0, 24, frames, nullptr);
                char anchor[64];
                std::snprintf(anchor, sizeof(anchor), "[alias-stack] anchor %llx:", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(&AliasStackAnchor)));
                std::string text = anchor;
                for (unsigned i = 0; i < count; ++i) {
                    char item[24];
                    std::snprintf(item, sizeof(item), " %llx", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(frames[i])));
                    text += item;
                }
                std::fprintf(stderr, "%s\n", text.c_str());
                break;
            }
        }
        for (const auto& texture : flush) {
            char key[96];
            std::snprintf(key, sizeof(key), "%s@%s 0x%llx+0x%llx ", reason, g_refreshSite, static_cast<unsigned long long>(texture->descriptor.baseAddress), static_cast<unsigned long long>(texture->guestBytes));
            const auto name = std::string(key) + "{" + describe(texture.get()) + "} -> {" + describe(refreshing) + "}";
            ++aliasCounts[name];
            // Bytes the store will select (writeBack's per-unit overlap, before any widening).
            std::uint64_t selected = 0;
            for (std::uint32_t layer = 0; layer < texture->trackedLayers; ++layer) {
                if (texture->layerPending[layer] && address < texture->layerBegin(layer) + texture->layerBytes(layer) && texture->layerBegin(layer) < address + bytes) selected += texture->layerBytes(layer);
            }
            aliasBytes[name] += selected;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - aliasReport >= std::chrono::seconds(10)) {
            aliasReport = now;
            std::vector<std::pair<std::uint64_t, std::string>> sorted;
            for (const auto& [name, count] : aliasCounts) sorted.emplace_back(count, name);
            std::sort(sorted.rbegin(), sorted.rend());
            std::string text = "[alias-flush] top (10 s):";
            for (std::size_t i = 0; i < std::min<std::size_t>(sorted.size(), 12); ++i) text += " | " + std::to_string(sorted[i].first) + "x " + sorted[i].second;
            std::fprintf(stderr, "%s\n", text.c_str());
            std::vector<std::pair<std::uint64_t, std::string>> heavy;
            for (const auto& [name, total] : aliasBytes) heavy.emplace_back(total, name);
            std::sort(heavy.rbegin(), heavy.rend());
            text = "[alias-flush] bytes (10 s):";
            for (std::size_t i = 0; i < std::min<std::size_t>(heavy.size(), 16); ++i) text += " | " + std::to_string(heavy[i].first >> 20) + " MiB " + std::to_string(aliasCounts[heavy[i].second]) + "x " + heavy[i].second;
            std::fprintf(stderr, "%s\n", text.c_str());
            aliasBytes.clear();
            aliasCounts.clear();
        }
    }
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::exception_ptr failure;
    const auto previousReason = std::exchange(flushReason, reason);
    const auto previousBytes = std::exchange(flushRequestBytes, bytes);
    for (const auto& texture : flush) {
        try {
            // Only the pending layers the access overlaps are stored; the others stay pending (the
            // store re-registers the image for them).
            texture->writeBack(address, bytes);
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    }
    flushReason = previousReason;
    flushRequestBytes = previousBytes;
    if (failure) std::rethrow_exception(failure);
    // The stores above went into the unit shadows where slabs took them: the consumer reads the
    // import, so they are published under the same hold, after every store.
    if (scope != PublishScope::None) {
        const auto units = PublishShadow(address, bytes, scope, PublishReasonFor(reason));
        if (published != nullptr) *published = units != 0;
    }
    return true;
}

bool StorageTexture::PublishShadowsOnly(std::uint64_t address, std::size_t bytes, PublishScope scope, const char* reason) {
    if (ForkLogged() != nullptr && scope != PublishScope::None && AnyShadowedOverlaps(address, bytes)) ForkRefuse("shadow publish");
    if (scope == PublishScope::None || !AnyShadowedOverlaps(address, bytes)) return false;
    std::lock_guard gpu(GuestMemory::GpuMutex());
    return PublishShadow(address, bytes, scope, PublishReasonFor(reason)) != 0;
}

void StorageTexture::FlushAllPending(const char* reason) {
    static_cast<void>(FlushPending(0, std::numeric_limits<std::size_t>::max(), nullptr, reason));
}

std::size_t StorageTexture::StoreSmallPending() {
    std::vector<std::shared_ptr<StorageTexture>> candidates;
    {
        auto& pending = Pending();
        std::lock_guard lock(pending.mutex);
        for (auto* texture : pending.textures) {
            if (texture->guestBytes > SmallStoreBytes) continue;
            if (auto alive = texture->weak_from_this().lock()) candidates.push_back(std::move(alive));
        }
    }
    std::size_t stored = 0;
    for (const auto& texture : candidates) {
        const auto address = texture->descriptor.baseAddress;
        const auto bytes = static_cast<std::size_t>(texture->guestBytes);
        // Elsewhere the store goes through the CPU, which would wait for the frame's work here.
        if (!HostImportCovers(texture->context, address, bytes)) continue;
        if (FlushPending(address, bytes, nullptr, "flip")) ++stored;
    }
    return stored;
}

std::shared_ptr<StorageTexture> StorageTexture::FindPending(std::uint64_t address, std::uint64_t bytes) {
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    for (auto* texture : pending.textures) {
        // Containment, not equality: a descriptor of a chain's first mips (its own guestBytes are
        // shorter) is served by the chain's image; CanCopyFrom then checks the geometry.
        if (texture->descriptor.baseAddress == address && texture->guestBytes >= bytes) return texture->weak_from_this().lock();
    }
    return nullptr;
}

bool PendingStorageOverlaps(std::uint64_t address, std::size_t bytes, const StorageTexture* except) {
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    if (!pending.textures.MayOverlap(address, bytes)) return false;
    for (const auto* texture : pending.textures) {
        // A free function (declared in ShaderResources.hpp): the overlap is computed from the public
        // surface description rather than the private helper.
        const auto begin = texture->Descriptor().baseAddress;
        if (texture != except && address < begin + texture->GuestBytes() && begin < address + bytes) return true;
    }
    return false;
}

bool StorageTexture::AnyPendingOverlaps(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (ranges.empty()) return false;
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    const auto overlapsAny = [&](const StorageTexture* texture) {
        for (const auto& [begin, end] : ranges) {
            if (end > begin && texture->overlaps(begin, static_cast<std::size_t>(end - begin))) return true;
        }
        return false;
    };
    for (const auto* texture : pending.textures) {
        if (overlapsAny(texture)) return true;
    }
    for (const auto* texture : pending.flushing) {
        if (overlapsAny(texture)) return true;
    }
    return false;
}

std::uint64_t StorageTexture::PendingSerial() {
    return pendingSerial.load(std::memory_order_acquire);
}

void StorageTexture::NoteProved() const {
    provedPresent.store(Recorder::Presents(), std::memory_order_relaxed);
}

std::vector<std::shared_ptr<StorageTexture>> StorageTexture::overlappingPending(std::uint64_t address, std::size_t bytes) {
    // FlushPending's listing; the images stay alive across the caller's scan as there.
    std::vector<std::shared_ptr<StorageTexture>> overlapping;
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    if (!pending.textures.MayOverlap(address, bytes)) return overlapping;
    const auto* exempt = refreshing;
    for (auto* texture : pending.textures) {
        if (texture == exempt || !texture->overlaps(address, bytes) || !texture->pendingUnitInside(address, bytes)) continue;
        if (auto alive = texture->weak_from_this().lock()) overlapping.push_back(std::move(alive));
    }
    return overlapping;
}

bool StorageTexture::blocksKept(std::span<const std::shared_ptr<StorageTexture>> images, std::uint64_t address, std::size_t bytes) {
    // The stamps are read outside the registry mutex (the tracker lock is never taken under it);
    // per pending layer, as the keep decision of writeBackLayers reads them.
    constexpr std::uint64_t block = 65536;
    const auto accessEnd = address + bytes;
    for (const auto& texture : images) {
        for (std::uint32_t layer = 0; layer < texture->trackedLayers; ++layer) {
            if (!texture->layerPending[layer]) continue;
            const auto begin = std::max(address, texture->layerBegin(layer));
            const auto end = std::min(accessEnd, texture->layerBegin(layer) + texture->layerBytes(layer));
            if (end <= begin) continue;
            if (texture->layerGeneration[layer] == 0) return false;
            for (auto at = begin & ~(block - 1); at < end; at += block) {
                const auto from = std::max(at, begin);
                const auto to = std::min(at + block, end);
                if (!GuestMemory::WrittenSince(from, static_cast<std::size_t>(to - from), texture->layerGeneration[layer])) return false;
            }
        }
    }
    return true;
}

void StorageTexture::ClassifyAccess(std::uint64_t address, std::size_t bytes, AccessClassification& out) {
    out = {};
    const auto overlapping = overlappingPending(address, bytes);
    if (overlapping.empty()) return;
    out.images = overlapping.size();
    const auto present = Recorder::Presents();
    for (const auto& texture : overlapping) {
        if (texture->provedPresent.load(std::memory_order_relaxed) + DeadImagePresents <= present) ++out.dead;
        else ++out.live;
    }
    if (bytes > ClassifyLimit) return;
    out.checked = true;
    GuestMemory::CollectWritesUncached(address, bytes);
    out.allCpuWritten = blocksKept(overlapping, address, bytes);
}

bool StorageTexture::AccessKeptByCpu(std::uint64_t address, std::size_t bytes, std::size_t* images, std::size_t* evicted) {
    if (evicted != nullptr) *evicted = 0;
    const auto overlapping = overlappingPending(address, bytes);
    if (images != nullptr) *images = overlapping.size();
    if (overlapping.empty() || bytes > ClassifyLimit) return false;
    // The keep decision of a store sees every CPU write to the range up to now (an uncached
    // collect, as writeBackLayers makes it: a memo hit could leave the store's own faults for the
    // next walk); so must this one.
    GuestMemory::CollectWritesUncached(address, bytes);
    // A neighbour's write-back stamps the 64 KiB block its surface shares with one of these images
    // and then advances that image's layer generation past the stamp (advanceAdjacent), both under
    // the GpuMutex; read between the two, the stamp would pass for a CPU write and the access
    // would read the image's stale bytes in the block. Under the mutex the layer state is what a
    // store there would see, and the stamps only grow.
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
    std::lock_guard gpu(GuestMemory::GpuMutex());
    if (!blocksKept(overlapping, address, bytes)) return false;
    const auto site = static_cast<std::uint8_t>(GuestMemory::CurrentReadSite());
    std::size_t dropped = 0;
    for (const auto& texture : overlapping) {
        if (texture->evictStale(address, bytes, dropped)) continue;
        texture->hookSkips.fetch_add(1, std::memory_order_relaxed);
        texture->hookSkipSite.store(site, std::memory_order_relaxed);
        for (std::uint32_t layer = 0; layer < texture->trackedLayers; ++layer) {
            if (!texture->layerPending[layer] || address >= texture->layerBegin(layer) + texture->layerBytes(layer) || texture->layerBegin(layer) >= address + bytes) continue;
            texture->hookSkipLayer = layer;
            texture->hookSkipVersion = texture->version;
            texture->hookSkipGeneration = texture->layerGeneration[layer];
            break;
        }
    }
    if (evicted != nullptr) *evicted = dropped;
    return true;
}

bool StorageTexture::evictStale(std::uint64_t address, std::size_t bytes, std::size_t& dropped) {
    static const bool staleEvict = std::getenv("APS5_STALE_EVICT") != nullptr;
    constexpr std::uint64_t block = 65536;
    // writeBack's selection for the access: the pending layers it overlaps, every one under a
    // clear code (see there), or every one of a dead image (H3).
    const bool whole = descriptor.dccAddress != 0 && uploadedKeys != DccKeys::Uncompressed;
    const bool all = whole || (staleEvict && provedPresent.load(std::memory_order_relaxed) + DeadImagePresents <= Recorder::Presents());
    const auto accessEnd = address + bytes;
    const auto selected = [&](std::uint32_t layer) {
        return layerPending[layer] && (all || (address < layerBegin(layer) + layerBytes(layer) && layerBegin(layer) < accessEnd));
    };
    std::uint64_t first = descriptor.baseAddress + guestBytes;
    std::uint64_t last = descriptor.baseAddress;
    bool beyond = false;
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (!selected(layer)) continue;
        if (layerGeneration[layer] == 0) return false;
        const auto begin = layerBegin(layer);
        const auto end = begin + layerBytes(layer);
        first = std::min(first, begin);
        last = std::max(last, end);
        if (begin < (address & ~(block - 1)) || end > ((accessEnd + block - 1) & ~(block - 1))) beyond = true;
    }
    if (last <= first) return false;
    if (beyond) {
        // Blocks outside the access's own were not collected by this call: the walk of the layers'
        // span (a write-back's own) runs once per present per image.
        const auto present = Recorder::Presents();
        if (staleCheckPresent == present) return false;
        staleCheckPresent = present;
        GuestMemory::CollectWritesUncached(first, static_cast<std::size_t>(last - first));
    }
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (!selected(layer)) continue;
        const auto begin = layerBegin(layer);
        const auto end = begin + layerBytes(layer);
        for (auto at = begin & ~(block - 1); at < end; at += block) {
            const auto from = std::max(at, begin);
            if (!GuestMemory::WrittenSince(from, static_cast<std::size_t>(std::min(at + block, end) - from), layerGeneration[layer])) return false;
        }
    }
    // As the empty write-back leaves the layers: no results pending, the generation unchanged (the
    // stamps make their next Refresh re-upload), `original` no longer vouching for the bytes.
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (selected(layer)) layerPending[layer] = false;
    }
    originalValid = false;
    if (!anyLayerPending()) {
        ++dropped;
        staleEvicted.fetch_add(1, std::memory_order_relaxed);
    }
    reconcilePending();
    return true;
}

bool StorageTexture::skippedResultsInside(std::uint64_t address, std::size_t bytes) const {
    const auto layer = hookSkipLayer;
    if (layer >= trackedLayers || !layerPending[layer] || version != hookSkipVersion || layerGeneration[layer] != hookSkipGeneration) return false;
    const bool whole = descriptor.dccAddress != 0 && uploadedKeys != DccKeys::Uncompressed;
    return whole || (address < layerBegin(layer) + layerBytes(layer) && layerBegin(layer) < address + bytes);
}

StorageTexture::HookSkipCounts StorageTexture::TakeHookSkipCounts() {
    return {flushedAfterSkip.exchange(0, std::memory_order_relaxed), flushedAfterSkipSameSite.exchange(0, std::memory_order_relaxed)};
}

void StorageTexture::BumpPendingSerial() {
    pendingSerial.fetch_add(1, std::memory_order_release);
}

bool StorageTexture::ScanPending(std::span<PendingQuery> queries) {
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    bool identities = true;
    for (auto& query : queries) {
        query.overlaps = false;
        query.found = nullptr;
        query.blocker = nullptr;
        if (query.end <= query.begin) continue;
        const auto bytes = static_cast<std::size_t>(query.end - query.begin);
        if (pending.textures.MayOverlap(query.begin, bytes)) {
            for (const auto* texture : pending.textures) {
                if (query.found == nullptr && texture->descriptor.baseAddress == query.begin && texture->guestBytes >= bytes) query.found = texture;
                if (texture != query.except && texture->overlaps(query.begin, bytes)) {
                    query.overlaps = true;
                    if (query.blocker == nullptr) query.blocker = texture;
                }
            }
        }
        for (const auto* texture : pending.flushing) {
            if (texture != query.except && texture->overlaps(query.begin, bytes)) {
                query.overlaps = true;
                if (query.blocker == nullptr) query.blocker = texture;
            }
        }
        if (query.pending != nullptr && query.found != query.pending) identities = false;
    }
    return identities;
}

std::size_t StorageTexture::DiscardPendingInside(std::uint64_t address, std::size_t bytes) {
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    std::size_t discarded = 0;
    const auto end = address + bytes;
    const auto* exempt = refreshing;
    for (auto it = pending.textures.begin(); it != pending.textures.end();) {
        auto* texture = *it;
        const auto begin = texture->descriptor.baseAddress;
        if (texture != exempt && begin >= address && begin + texture->guestBytes <= end) {
            // The content no longer matches guest memory anywhere: the next use re-uploads once the
            // fill stamped the range, never from a matching `original` (the fill may not have
            // touched the bytes a CPU-path upload copied).
            texture->dirty = false;
            texture->layerPending.assign(texture->trackedLayers, false);
            texture->originalValid = false;
            it = pending.textures.erase(it);
            ++discarded;
        } else if (texture != exempt && texture->blockUnits && texture->overlaps(address, bytes)) {
            // Its units wholly inside the range are dead too (re-uploaded from the range's bytes at
            // the next use, never from a matching `original`: the dropped units hold results the
            // bytes never received); the others stay pending.
            bool dropped = false;
            for (std::uint32_t unit = 0; unit < texture->trackedLayers; ++unit) {
                const auto unitBegin = texture->layerBegin(unit);
                if (!texture->layerPending[unit] || unitBegin < address || unitBegin + texture->layerBytes(unit) > end) continue;
                texture->layerPending[unit] = false;
                unitsDropped.fetch_add(1, std::memory_order_relaxed);
                dropped = true;
            }
            if (dropped) {
                texture->originalValid = false;
                ++discarded;
            }
            if (dropped && !texture->anyLayerPending()) {
                texture->dirty = false;
                it = pending.textures.erase(it);
                continue;
            }
            ++it;
        } else {
            ++it;
        }
    }
    if (discarded != 0) BumpPendingSerial();
    return discarded;
}

namespace {

// Whether [begin, end) lies wholly inside one of the sorted, disjoint `ranges`.
bool insideRanges(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, std::uint64_t begin, std::uint64_t end) {
    const auto found = std::upper_bound(ranges.begin(), ranges.end(), begin, [](std::uint64_t value, const auto& range) { return value < range.first; });
    if (found == ranges.begin()) return false;
    const auto& range = *std::prev(found);
    return range.first <= begin && end <= range.second;
}

bool overlapsRanges(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, std::uint64_t begin, std::uint64_t end) {
    return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.first < end && begin < range.second; });
}

}

bool StorageTexture::CoveredUnits(std::uint64_t address, std::size_t bytes, std::span<const std::pair<std::uint64_t, std::uint64_t>> covered, std::vector<std::pair<std::uint64_t, std::uint64_t>>& units) {
    units.clear();
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    // An image another thread is storing right now (listed by its FlushPending, not yet stored)
    // would land its results after the dispatch: nothing is elided over it.
    for (const auto* texture : pending.flushing) {
        if (overlapsRanges(covered, texture->descriptor.baseAddress, texture->descriptor.baseAddress + texture->guestBytes)) return false;
    }
    if (!pending.textures.MayOverlap(address, bytes)) return true;
    const auto* exempt = refreshing;
    // The pending units over the region: the ones wholly inside the covered bytes (candidates),
    // and every other one (another image's unit sticking out of the region or only partly covered).
    std::vector<std::pair<std::uint64_t, std::uint64_t>> candidates;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> others;
    for (const auto* texture : pending.textures) {
        if (!texture->overlaps(address, bytes)) continue;
        bool anyCovered = false;
        for (std::uint32_t unit = 0; unit < texture->trackedLayers && !anyCovered; ++unit) {
            if (texture->layerPending[unit] && overlapsRanges(covered, texture->layerBegin(unit), texture->layerBegin(unit) + texture->layerBytes(unit))) anyCovered = true;
        }
        // Results shared with an alias, an image under validation, or one whose write-back stores
        // its DCC keys with every unit keep the plain flush.
        const bool whole = texture->descriptor.dccAddress != 0 && texture->uploadedKeys != DccKeys::Uncompressed;
        if (anyCovered && (texture == exempt || LayerAliasExempted(texture) || texture->lent || whole || !texture->borrowedFrom.expired() || !texture->layerBorrows.empty() || texture->layerPending.size() < texture->trackedLayers)) return false;
        for (std::uint32_t unit = 0; unit < texture->trackedLayers && unit < texture->layerPending.size(); ++unit) {
            if (!texture->layerPending[unit]) continue;
            const auto begin = texture->layerBegin(unit);
            const auto end = begin + texture->layerBytes(unit);
            if (end <= address || address + bytes <= begin) continue;
            if (anyCovered && address <= begin && end <= address + bytes && insideRanges(covered, begin, end)) candidates.emplace_back(begin, end);
            else others.emplace_back(begin, end);
        }
    }
    // A candidate overlapped by any other pending unit is stored instead: the flush of the gaps
    // then stores that other unit too, which no drop would ever reach.
    for (const auto& range : candidates) {
        if (!overlapsRanges(others, range.first, range.second)) units.push_back(range);
    }
    std::sort(units.begin(), units.end());
    std::vector<std::pair<std::uint64_t, std::uint64_t>> merged;
    for (const auto& range : units) {
        if (!merged.empty() && range.first <= merged.back().second) merged.back().second = std::max(merged.back().second, range.second);
        else merged.push_back(range);
    }
    units = std::move(merged);
    return true;
}

std::pair<std::size_t, std::uint64_t> StorageTexture::DropCoveredUnits(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    std::pair<std::size_t, std::uint64_t> dropped{0, 0};
    if (ranges.empty()) return dropped;
    auto& pending = Pending();
    std::lock_guard lock(pending.mutex);
    const auto first = ranges.front().first;
    const auto span = static_cast<std::size_t>(ranges.back().second - first);
    // Clears the texture's pending units wholly inside the ranges; whether any was.
    const auto drop = [&](StorageTexture* texture) {
        if (!texture->overlaps(first, span) || texture->layerPending.size() < texture->trackedLayers) return false;
        bool any = false;
        for (std::uint32_t unit = 0; unit < texture->trackedLayers; ++unit) {
            const auto begin = texture->layerBegin(unit);
            const auto end = begin + texture->layerBytes(unit);
            if (!texture->layerPending[unit] || !insideRanges(ranges, begin, end)) continue;
            texture->layerPending[unit] = false;
            ++dropped.first;
            dropped.second += end - begin;
            unitsDropped.fetch_add(1, std::memory_order_relaxed);
            any = true;
        }
        // The dropped units hold results guest memory never received: the next use takes them from
        // memory (the dispatch's bytes), never from a matching `original`.
        if (any) texture->originalValid = false;
        return any;
    };
    // An image another thread's FlushPending took off the list since the elision decided (it waits
    // for the GPU mutex this thread holds, then stores the units still pending): its covered units
    // are cleared too, so that store, recorded after the dispatch, leaves the dispatch's bytes.
    bool anyDropped = false;
    for (auto* texture : pending.flushing) anyDropped = drop(texture) || anyDropped;
    if (pending.textures.MayOverlap(first, span)) {
        for (auto it = pending.textures.begin(); it != pending.textures.end();) {
            auto* texture = *it;
            const bool any = drop(texture);
            anyDropped = anyDropped || any;
            if (any && !texture->anyLayerPending()) {
                texture->dirty = false;
                it = pending.textures.erase(it);
                continue;
            }
            ++it;
        }
    }
    if (anyDropped) BumpPendingSerial();
    return dropped;
}

StorageTexture::FillCoverage StorageTexture::ClassifyFill(std::uint64_t address, std::size_t bytes) {
    FillCoverage coverage;
    auto& live = Live();
    std::lock_guard lock(live.mutex);
    std::vector<StorageTexture*> overlapping;
    for (auto* texture : live.textures) {
        if (!texture->released && texture->guestBytes != 0 && texture->overlaps(address, bytes)) overlapping.push_back(texture);
    }
    const auto end = address + bytes;
    StorageTexture* covered = nullptr;
    // The newest of several exact matches (the registry is in construction order): the same
    // surface under two storage formats keeps two images, and the older one may be bound by
    // nothing any more, so clearing it would only revive it for a write-back.
    for (auto* texture : overlapping) {
        if (texture->descriptor.baseAddress == address && texture->guestBytes == bytes) {
            covered = texture;
            coverage.cover = FillCover::Exact;
            coverage.layer = WholeImage;
        }
    }
    // APS5_NO_LAYER_REFUSAL_MEMO=1: match a layer of an image whose surface check failed at its
    // current generation anyway (the whole-surface collect that check costs then runs per fill).
    static const bool layerMemo = std::getenv("APS5_NO_LAYER_REFUSAL_MEMO") == nullptr;
    for (auto* texture : overlapping) {
        if (covered != nullptr) break;
        // One array layer of a thin surface: its guest bytes are one contiguous slice.
        const auto& geometry = texture->geometry;
        const auto base = texture->descriptor.baseAddress;
        if (geometry.layers < 2 || geometry.thick || geometry.imageDepth != 1 || geometry.layers != geometry.imageLayers || geometry.layerBytes != bytes || geometry.layerBytes * geometry.layers != texture->guestBytes || address < base) continue;
        if (layerMemo && texture->layerRefusedGeneration == texture->generation) continue;
        const auto offset = address - base;
        if (offset % bytes != 0 || offset / bytes >= geometry.layers) continue;
        covered = texture;
        coverage.cover = FillCover::Layer;
        coverage.layer = static_cast<std::uint32_t>(offset / bytes);
    }
    if (covered != nullptr) {
        coverage.image = covered->weak_from_this().lock();
        if (coverage.image == nullptr) {
            coverage.cover = FillCover::None;
        } else {
            for (const auto* texture : overlapping) {
                if (texture == covered) continue;
                ++coverage.others;
                if (texture->descriptor.baseAddress >= address && texture->descriptor.baseAddress + texture->guestBytes <= end) ++coverage.inside;
            }
        }
    } else if (overlapping.empty()) {
        constexpr std::uint64_t keyBytes = 256;
        for (const auto* texture : live.textures) {
            if (!texture->released && texture->descriptor.dccAddress == address && texture->guestBytes / keyBytes != 0 && bytes >= texture->guestBytes / keyBytes) coverage.cover = FillCover::Keys;
        }
    } else if (overlapping.size() > 1) {
        coverage.cover = FillCover::Several;
    } else {
        auto* single = overlapping.front();
        const auto begin = single->descriptor.baseAddress;
        const auto stop = begin + single->guestBytes;
        if (address >= begin && end <= stop) coverage.cover = FillCover::Inside;
        else if (address <= begin && end >= stop) coverage.cover = FillCover::Around;
        else coverage.cover = FillCover::Straddle;
    }
    // Debug aid: APS5_TRACE_FILL_COVER=1 prints the first fills that meet images without covering
    // one of them (or a layer of one), with the images' surfaces, so a wider conversion can be
    // designed from them.
    static const bool trace = std::getenv("APS5_TRACE_FILL_COVER") != nullptr;
    static int traced = 0;
    const auto cover = coverage.cover;
    if (trace && (cover == FillCover::Inside || cover == FillCover::Around || cover == FillCover::Straddle || cover == FillCover::Several) && traced < 96) {
        ++traced;
        std::fprintf(stderr, "[fill-cover] fill 0x%llx+0x%zx meets %zu image(s):", static_cast<unsigned long long>(address), bytes, overlapping.size());
        for (const auto* texture : overlapping) {
            const auto& d = texture->descriptor;
            std::fprintf(stderr, " [0x%llx+0x%llx %ux%u mips %u layers %u dim %d tile %d format %u dcc 0x%llx%s]", static_cast<unsigned long long>(d.baseAddress), static_cast<unsigned long long>(texture->guestBytes), d.width, d.height, d.mipCount, texture->geometry.imageLayers, static_cast<int>(d.dimension), static_cast<int>(d.tileMode), d.format, static_cast<unsigned long long>(d.dccAddress), texture->dirty ? " dirty" : "");
        }
        std::fprintf(stderr, "\n");
    }
    return coverage;
}

std::size_t StorageTexture::NoteKeysFill(std::uint64_t address, std::size_t bytes, std::uint8_t key) {
    static const bool refillDisabled = std::getenv("APS5_NO_KEYS_REFILL") != nullptr;
    DccKeys keys = DccKeys::Mixed;
    switch (key) {
        case 0x00: keys = DccKeys::Clear0000; break;
        case 0x40: keys = DccKeys::Clear0001; break;
        case 0x80: keys = DccKeys::Clear1110; break;
        case 0xc0: keys = DccKeys::Clear1111; break;
        case 0x20: keys = DccKeys::ClearRegister; break;
        case 0xff: keys = DccKeys::Uncompressed; break;
        default: return 0;
    }
    constexpr std::uint64_t keyBytes = 256;
    static const bool traceKeys = std::getenv("APS5_TRACE_DCC_KEYS") != nullptr;
    auto& live = Live();
    std::lock_guard lock(live.mutex);
    std::size_t covered = 0;
    for (auto* texture : live.textures) {
        if (texture->released || texture->descriptor.dccAddress != address || texture->guestBytes / keyBytes == 0 || bytes < texture->guestBytes / keyBytes) continue;
        texture->filledKeys = keys;
        ++covered;
        if (traceKeys && IsDccClear(keys)) std::fprintf(stderr, "[dcc-keys] %s key fill over 0x%llx+0x%llx (uploaded %s, dirty %d)\n", DccKeysName(keys), static_cast<unsigned long long>(texture->descriptor.baseAddress), static_cast<unsigned long long>(texture->guestBytes), DccKeysName(texture->uploadedKeys), texture->dirty ? 1 : 0);
        if (refillDisabled || !IsDccClear(keys) || texture->uploadedKeys != keys) continue;
        texture->uploadedKeys = DccKeys::Uncompressed;
    }
    return covered;
}

std::size_t StorageTexture::ClearByKeysFill(std::uint64_t address, std::size_t bytes, std::uint8_t key) {
    static const bool enabled = [] { const char* text = std::getenv("APS5_KEYS_FILL_CLEAR"); return text != nullptr && std::strcmp(text, "1") == 0; }();
    if (!enabled) return 0;
    DccKeys keys = DccKeys::Mixed;
    switch (key) {
        case 0x00: keys = DccKeys::Clear0000; break;
        case 0x40: keys = DccKeys::Clear0001; break;
        case 0x80: keys = DccKeys::Clear1110; break;
        case 0xc0: keys = DccKeys::Clear1111; break;
        case 0x20: keys = DccKeys::ClearRegister; break;
        default: return 0;
    }
    constexpr std::uint64_t keyBytes = 256;
    std::vector<std::shared_ptr<StorageTexture>> covered;
    {
        auto& live = Live();
        std::lock_guard lock(live.mutex);
        for (auto* texture : live.textures) {
            if (texture->released || texture->descriptor.dccAddress != address || texture->guestBytes / keyBytes == 0 || bytes < texture->guestBytes / keyBytes) continue;
            if (auto shared = texture->weak_from_this().lock()) covered.push_back(std::move(shared));
        }
    }
    std::size_t cleared = 0;
    for (const auto& texture : covered) {
        if (texture->clearByKeysFill(keys, key)) ++cleared;
    }
    return cleared;
}

bool StorageTexture::clearByKeysFill(DccKeys keys, std::uint8_t key) {
    static const bool traceKeys = std::getenv("APS5_TRACE_DCC_KEYS") != nullptr;
    VkClearColorValue clearValue{};
    auto* recorder = Recorder::Active();
    const char* refusal = !ClearColorFor(storageFormat, keys, clearValue) ? (keys == DccKeys::ClearRegister ? "clear-register code" : "no clear value in the storage format") : recorder == nullptr ? "no recorder" : nullptr;
    if (refusal != nullptr) {
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true)) std::fprintf(stderr, "[dcc-keys] key fill 0x%02x (%s) over surface 0x%llx+0x%llx (guest format %u, vk format %d, keys 0x%llx) not applied at once: %s\n", key, DccKeysName(keys), static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), descriptor.format, static_cast<int>(storageFormat), static_cast<unsigned long long>(descriptor.dccAddress), refusal);
        return false;
    }
    std::size_t droppedUnits = 0;
    const bool wasDirty = dirty;
    {
        auto& pending = Pending();
        std::lock_guard lock(pending.mutex);
        droppedUnits = static_cast<std::size_t>(std::count(layerPending.begin(), layerPending.end(), true));
        layerPending.assign(trackedLayers, false);
        if (dirty) {
            dirty = false;
            pending.textures.remove(this);
            BumpPendingSerial();
        }
    }
    const std::pair<VkImage, bool> clearedImage{image, true};
    bool elided = false;
    const auto commands = Recorder::BarrierTracking() ? recorder->Commands(Recorder::CommandClass::DccClear, Recorder::Access{{}, {}, std::span(&clearedImage, 1), VK_PIPELINE_STAGE_TRANSFER_BIT}, elided) : recorder->Commands();
    const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::DccClear);
    if (auto self = weak_from_this().lock()) recorder->Keep(std::move(self));
    Recorder::CountBarriers(Recorder::CommandClass::DccClear, elided ? 1 : 2);
    if (elided) Recorder::CountSkipped(Recorder::CommandClass::DccClear);
    VkImageMemoryBarrier toClear{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toClear.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toClear.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toClear.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toClear.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toClear.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toClear.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toClear.image = image;
    toClear.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
    if (elided) {
        toClear.srcAccessMask = 0;
        toClear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toClear);
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &toClear.subresourceRange);
    } else {
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toClear);
        context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &toClear.subresourceRange);
        VkImageMemoryBarrier toGeneral = toClear;
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
    }
    recorder->EndGpuTiming(timing, guestBytes);
    originalValid = false;
    uploadedKeys = keys;
    filledKeys = DccKeys::Uncompressed;
    keyProof = {};
    forgetBorrowed(0, trackedLayers);
    layerGeneration.assign(trackedLayers, GuestMemory::CollectWrites(descriptor.baseAddress, static_cast<std::size_t>(guestBytes)));
    refreshGeneration();
    ++version;
    if (traceKeys) std::fprintf(stderr, "[dcc-keys] key fill %s clears 0x%llx+0x%llx at once (keys 0x%llx, %zu pending units dropped, dirty %d)\n", DccKeysName(keys), static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), static_cast<unsigned long long>(descriptor.dccAddress), droppedUnits, wasDirty ? 1 : 0);
    return true;
}

bool StorageTexture::FillClear(std::span<const std::uint32_t, 4> pattern, std::uint32_t layer, const char*& refusal) {
    ForkTrap("storage clear");
    // Debug aid (APS5_TRACE_FILL_COVER=1): the first layer covers refused, with the surface, the
    // pattern and the generation the refusal was memoized at (ClassifyFill's layerRefusedGeneration).
    static const bool trace = std::getenv("APS5_TRACE_FILL_COVER") != nullptr;
    struct TraceLayerRefusal {
        const StorageTexture& texture;
        std::span<const std::uint32_t, 4> pattern;
        std::uint32_t layer;
        const char*& refusal;
        bool cleared = false;
        ~TraceLayerRefusal() {
            static std::atomic<int> traced{0};
            if (cleared || !trace || layer == WholeImage || refusal == nullptr || traced.fetch_add(1) >= 64) return;
            const auto& d = texture.descriptor;
            std::fprintf(stderr, "[fill-cover] layer %u of 0x%llx+0x%llx (%ux%u mips %u layers %u, %llu bytes per layer, format %u tile %d dcc 0x%llx%s) refused: %s; pattern %08x %08x %08x %08x; generation %llu\n", layer, static_cast<unsigned long long>(d.baseAddress), static_cast<unsigned long long>(texture.guestBytes), d.width, d.height, d.mipCount, texture.geometry.imageLayers, static_cast<unsigned long long>(texture.geometry.layerBytes), d.format, static_cast<int>(d.tileMode), static_cast<unsigned long long>(d.dccAddress), texture.dirty ? ", dirty" : "", refusal, pattern[0], pattern[1], pattern[2], pattern[3], static_cast<unsigned long long>(texture.generation));
        }
    } traceRefusal{*this, pattern, layer, refusal};
    VkClearColorValue clearValue{};
    if (!FillClearColor(storageFormat, BytesPerElement(descriptor.format), pattern, clearValue)) {
        refusal = "pattern";
        return false;
    }
    if (layer != WholeImage && (layer >= geometry.imageLayers || geometry.thick || geometry.imageDepth != 1)) {
        refusal = "layer";
        return false;
    }
    if (descriptor.dccAddress != 0) {
        // Under a clear code reads see that value whatever the texels hold, and keys a recorded
        // kernel still writes are not in the bytes yet: only keys that read as uncompressed now let
        // the image stand for the texels.
        if (Recorder::SnapshotWriteOverlaps(descriptor.dccAddress, static_cast<std::size_t>(guestBytes / 256))) {
            refusal = "keys pending";
            return false;
        }
        if (TextureClearKeys(descriptor, guestBytes) != DccKeys::Uncompressed) {
            refusal = "keys";
            return false;
        }
    }
    if (HostImportFor(context, descriptor.baseAddress, static_cast<std::size_t>(guestBytes)) == nullptr) {
        refusal = "not imported";
        return false;
    }
    auto* recorder = Recorder::Active();
    if (recorder == nullptr) {
        refusal = "no recorder";
        return false;
    }
    const auto begin = layer == WholeImage ? descriptor.baseAddress : descriptor.baseAddress + geometry.GuestLayerOffset(layer);
    const auto bytes = static_cast<std::size_t>(layer == WholeImage ? guestBytes : geometry.layerBytes);
    // A tracked layer answers for itself (its own generation and pending flag): the rest of the
    // surface may have changed under the image, which its own layers' next use uploads.
    if (layer != WholeImage && trackedLayers == 1) {
        // The rest of the surface keeps the image's content, which must still be what guest memory
        // holds: the one generation the write-back keeps CPU-written blocks after moves to now. A
        // surface that changed under the image (another image's results stored into its memory, a
        // fill of another layer stored as bytes) is not refreshed here: the images so filled at the
        // movie stage are stale ones of an earlier use of the memory, and reviving one every frame
        // (a 60 MiB write-back and re-upload inside the fill) cost what the conversion gained.
        // Uncached, as writeBack's keep decision walks: a memo hit of this submission's epoch
        // would leave pages the game dirtied since that walk unconsumed, and the check below
        // would pass over them.
        GuestMemory::CollectWritesUncached(descriptor.baseAddress, static_cast<std::size_t>(guestBytes));
        const auto before = static_cast<std::size_t>(begin - descriptor.baseAddress);
        const auto after = static_cast<std::size_t>(descriptor.baseAddress + guestBytes - (begin + bytes));
        if ((before != 0 && !GuestMemory::UnchangedSince(descriptor.baseAddress, before, generation)) || (after != 0 && !GuestMemory::UnchangedSince(begin + bytes, after, generation))) {
            refusal = "surface changed";
            layerRefusedGeneration = generation;
            return false;
        }
    }
    if (layer != WholeImage && blockUnits && ((begin - descriptor.baseAddress) % trackedLayerBytes != 0 || bytes % trackedLayerBytes != 0)) {
        refusal = "layer alignment";
        return false;
    }
    // Other images over the range hold content the fill supersedes: stamped below, they re-upload
    // at their next use (behind this image's write-back), and never from a matching `original`,
    // since the fill leaves the bytes a CPU-path upload copied as they are.
    bool aliasedKeys = false;
    {
        auto& live = Live();
        std::lock_guard lock(live.mutex);
        for (auto* texture : live.textures) {
            if (texture == this || texture->released || !texture->overlaps(begin, bytes)) continue;
            texture->originalValid = false;
            if (texture->descriptor.dccAddress != 0) aliasedKeys = true;
        }
    }
    // Debug aid (APS5_TRACE_FILL_COVER=1): the first clears of a surface with DCC keys (its own or
    // another image's over the same memory), with the pattern: a title that fills the keys to a
    // clear code right after makes the clear moot for that image (its next lookup re-uploads
    // under the code) unless the pattern is that code's.
    static std::atomic<int> traced{0};
    if (trace && (descriptor.dccAddress != 0 || aliasedKeys) && traced.fetch_add(1) < 16) std::fprintf(stderr, "[fill-cover] clear of surface 0x%llx+0x%llx (format %u, keys 0x%llx%s): pattern %08x %08x %08x %08x\n", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), descriptor.format, static_cast<unsigned long long>(descriptor.dccAddress), aliasedKeys ? ", another image over the memory has keys" : "", pattern[0], pattern[1], pattern[2], pattern[3]);
    // A neighbour's results shadowed in the range's edge units reach the import before the stamp
    // below makes those units stale (nothing for a 64 KiB-multiple surface).
    PublishShadow(begin, bytes, PublishScope::PartialUnits, PublishReason::FillClear);
    GuestMemory::MarkWritten(begin, bytes);
    const std::pair<VkImage, bool> cleared{image, true};
    bool elided = false;
    const auto commands = Recorder::BarrierTracking() ? recorder->Commands(Recorder::CommandClass::FillClear, Recorder::Access{{}, {}, std::span(&cleared, 1), VK_PIPELINE_STAGE_TRANSFER_BIT, false, false, true}, elided) : recorder->Commands();
    const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::FillClear);
    Recorder::CountBarriers(Recorder::CommandClass::FillClear, 2);
    if (auto self = weak_from_this().lock()) recorder->Keep(std::move(self));
    VkImageMemoryBarrier toClear{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toClear.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toClear.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    // A whole image is discarded into the clear; one layer keeps the others' content.
    toClear.oldLayout = layer == WholeImage ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
    toClear.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toClear.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toClear.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toClear.image = image;
    toClear.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, layer == WholeImage ? 0u : layer, layer == WholeImage ? geometry.imageLayers : 1u};
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toClear);
    context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearValue, 1, &toClear.subresourceRange);
    VkImageMemoryBarrier toGeneral = toClear;
    toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
    recorder->EndGpuTiming(timing, bytes);
    traceRefusal.cleared = true;
    // The image holds the fill and guest memory what it held: the texels reach it at the
    // write-back, which keeps the 64 KiB blocks the CPU writes from now on (a CPU write before the
    // fill is overwritten by it in program order; the stamps above are below this generation), and
    // the keys were read uncompressed. The collect is uncached (see writeBack's keep decision): a
    // memo hit of this submission's epoch would not consume pages the game dirtied since that
    // walk, the write-back's own walk would stamp them above this generation and keep their
    // blocks, and the fill's pattern would be lost in them.
    originalValid = false;
    uploadedKeys = DccKeys::Uncompressed;
    if (layer != WholeImage && trackedLayers > 1) {
        // The cleared layer's units alone: their stamps above lie below this generation, the other
        // units keep theirs (their pages are consumed by their own collects).
        const auto first = blockUnits ? static_cast<std::uint32_t>((begin - descriptor.baseAddress) / trackedLayerBytes) : layer;
        const auto count = blockUnits ? static_cast<std::uint32_t>(bytes / trackedLayerBytes) : 1u;
        const auto now = GuestMemory::CollectWritesUncached(begin, bytes);
        for (auto unit = first; unit < first + count; ++unit) layerGeneration[unit] = now;
        refreshGeneration();
        forgetBorrowed(first, count);
        markLayersPending(first, count);
        return true;
    }
    layerGeneration.assign(trackedLayers, GuestMemory::CollectWritesUncached(descriptor.baseAddress, static_cast<std::size_t>(guestBytes)));
    refreshGeneration();
    forgetBorrowed(0, trackedLayers);
    MarkDirty();
    return true;
}

void StorageTexture::WriteBack() {
    GuestMemory::NoteProbeEvent("write-back");
    const auto previous = std::exchange(flushReason, "explicit");
    writeBack(descriptor.baseAddress, static_cast<std::size_t>(guestBytes));
    flushReason = previous;
}

std::shared_ptr<StorageTexture> StorageTexture::FindLive(std::uint64_t address, std::uint64_t bytes) {
    auto& live = Live();
    std::lock_guard lock(live.mutex);
    for (auto it = live.textures.rbegin(); it != live.textures.rend(); ++it) {
        auto* texture = *it;
        if (texture->released || !texture->Cached() || texture->descriptor.baseAddress != address || texture->guestBytes != bytes) continue;
        return texture->weak_from_this().lock();
    }
    return nullptr;
}

bool StorageTexture::SameSurfaceShape(const StorageTexture& other) const {
    const auto& mine = descriptor;
    const auto& theirs = other.descriptor;
    return guestBytes == other.guestBytes && mine.width == theirs.width && mine.height == theirs.height && mine.depthOrLastArray == theirs.depthOrLastArray && mine.baseArray == theirs.baseArray && mine.mipCount == theirs.mipCount && mine.tileMode == theirs.tileMode && mine.dimension == theirs.dimension && mine.format == theirs.format && storageFormat == other.storageFormat && geometry.imageLayers == other.geometry.imageLayers && geometry.imageDepth == other.geometry.imageDepth;
}

bool StorageTexture::CopyFrom(StorageTexture& source, const char*& refusal) {
    if (&source == this || !SameSurfaceShape(source)) {
        refusal = "shape";
        return false;
    }
    if (descriptor.dccAddress != 0 || source.descriptor.dccAddress != 0) {
        refusal = "keys";
        return false;
    }
    if (released || source.released || !Cached() || !source.Cached()) {
        refusal = "released";
        return false;
    }
    if (HostImportFor(context, descriptor.baseAddress, static_cast<std::size_t>(guestBytes)) == nullptr) {
        refusal = "not imported";
        return false;
    }
    auto* recorder = Recorder::Active();
    if (recorder == nullptr) {
        refusal = "no recorder";
        return false;
    }
    // The source image must hold the surface: a stale one uploads first (and other images'
    // results pending over its memory land first, as the transfer's flush would have them).
    source.Refresh();
    // Results pending over the destination range from before the copy are dead inside it and land
    // first around it; other images over the range hold content the copy supersedes (they
    // re-upload at their next use, behind this image's write-back).
    static_cast<void>(DiscardPendingInside(descriptor.baseAddress, static_cast<std::size_t>(guestBytes)));
    FlushPending(descriptor.baseAddress, static_cast<std::size_t>(guestBytes), this, "buffer copy destination", PublishScope::PartialUnits);
    {
        auto& live = Live();
        std::lock_guard lock(live.mutex);
        for (auto* texture : live.textures) {
            if (texture != this && !texture->released && texture->overlaps(descriptor.baseAddress, static_cast<std::size_t>(guestBytes))) texture->originalValid = false;
        }
    }
    GuestMemory::MarkWritten(descriptor.baseAddress, static_cast<std::size_t>(guestBytes));
    const std::array<std::pair<VkImage, bool>, 2> copiedImages{{{source.image, false}, {image, true}}};
    bool elided = false;
    const auto commands = Recorder::BarrierTracking() ? recorder->Commands(Recorder::CommandClass::Copy, Recorder::Access{{}, {}, copiedImages, VK_PIPELINE_STAGE_TRANSFER_BIT, false, false, true}, elided) : recorder->Commands();
    const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
    Recorder::CountBarriers(Recorder::CommandClass::Copy, 2);
    if (auto self = weak_from_this().lock()) recorder->Keep(std::move(self));
    if (auto other = source.weak_from_this().lock()) recorder->Keep(std::move(other));
    VkImageMemoryBarrier barriers[2]{};
    for (auto& barrier : barriers) {
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
    }
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].image = source.image;
    barriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].image = image;
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
    std::vector<VkImageCopy> regions;
    for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
        for (std::uint32_t level = 0; level < descriptor.mipCount; ++level) {
            if (!geometry.HasLayer(level, layer)) continue;
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, geometry.CopyLayer(layer), 1};
            region.srcOffset = {0, 0, geometry.CopyDepth(layer)};
            region.dstSubresource = region.srcSubresource;
            region.dstOffset = region.srcOffset;
            region.extent = {std::max(descriptor.width >> level, 1u), std::max(descriptor.height >> level, 1u), 1u};
            regions.push_back(region);
        }
    }
    context.Function<PFN_vkCmdCopyImage>("vkCmdCopyImage")(commands, source.image, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(regions.size()), regions.data());
    VkImageMemoryBarrier copied = barriers[1];
    copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &copied);
    recorder->EndGpuTiming(timing, guestBytes);
    // As after a clear: the image holds the surface, guest memory follows at the write-back (which
    // keeps the 64 KiB blocks the CPU writes from now on; the collect is uncached for the reason
    // given in FillClear).
    originalValid = false;
    uploadedKeys = DccKeys::Uncompressed;
    layerGeneration.assign(trackedLayers, GuestMemory::CollectWritesUncached(descriptor.baseAddress, static_cast<std::size_t>(guestBytes)));
    refreshGeneration();
    forgetBorrowed(0, trackedLayers);
    MarkDirty();
    return true;
}

// See AdjacentGenerationEnabled above.
std::vector<StorageTexture::Adjacent> StorageTexture::adjacentPendingUnchanged(std::uint64_t firstBlock, std::uint64_t lastBlock) const {
    std::vector<Adjacent> adjacent;
    if (!AdjacentGenerationEnabled() || guestBytes == 0) return adjacent;
    constexpr std::uint64_t block = 65536;
    {
        auto& pending = Pending();
        std::lock_guard lock(pending.mutex);
        const auto consider = [&](StorageTexture* texture) {
            if (texture == this || texture->guestBytes == 0 || texture->overlaps(descriptor.baseAddress, static_cast<std::size_t>(guestBytes))) return;
            const auto begin = texture->descriptor.baseAddress;
            const auto neighbourFirst = begin / block;
            const auto neighbourLast = (begin + texture->guestBytes - 1) / block;
            if (neighbourFirst > lastBlock || neighbourLast < firstBlock) return;
            // Surfaces that do not overlap share at most one block: the neighbour's tracked layer
            // holding it is what the stamps touch.
            const auto shared = std::max(neighbourFirst, firstBlock);
            const auto layer = static_cast<std::uint32_t>((std::max(shared * block, begin) - begin) / texture->trackedLayerBytes);
            if (auto alive = texture->weak_from_this().lock()) adjacent.push_back({std::move(alive), layer, texture->layerGeneration[layer]});
        };
        for (auto* texture : pending.textures) consider(texture);
        for (auto* texture : pending.flushing) consider(texture);
    }
    // Only a layer whose memory has not changed since its generation can be advanced: the collect
    // stamps the CPU's writes so far, and a later one lands at a newer generation either way.
    std::erase_if(adjacent, [](const Adjacent& entry) {
        const auto& texture = entry.texture;
        const auto begin = texture->layerBegin(entry.layer);
        const auto bytes = static_cast<std::size_t>(texture->layerBytes(entry.layer));
        GuestMemory::CollectWritesUncached(begin, bytes);
        return !GuestMemory::UnchangedSince(begin, bytes, entry.generation);
    });
    return adjacent;
}

void StorageTexture::advanceAdjacent(const std::vector<Adjacent>& adjacent, std::uint64_t now, std::uint64_t firstBlock, std::uint64_t lastBlock) {
    static const bool trace = std::getenv("APS5_TRACE_FLUSH") != nullptr;
    constexpr std::uint64_t block = 65536;
    for (const auto& [texture, layer, seen] : adjacent) {
        // A layer stored or refreshed meanwhile set its own generation (an old one on purpose when
        // it kept blocks for the CPU): only the value seen at the check is advanced.
        if (texture->layerGeneration[layer] != seen || now <= seen) continue;
        // A CPU write to the neighbour landing since the check may have been stamped by another
        // thread's walk of its range at a value in (seen, now]; the advance would hide it. Its blocks
        // outside this write-back's span carry no stamp of ours, so they are re-checked against the
        // seen generation (a stamp scan, no walk). A write inside the shared block itself in that
        // window is indistinguishable from this write-back's stamp, as it is for the block rule.
        const auto begin = texture->layerBegin(layer);
        const auto end = begin + texture->layerBytes(layer);
        const auto beforeEnd = std::min(end, firstBlock * block);
        const auto afterBegin = std::max(begin, (lastBlock + 1) * block);
        if (begin < beforeEnd && !GuestMemory::UnchangedSince(begin, static_cast<std::size_t>(beforeEnd - begin), seen)) continue;
        if (afterBegin < end && !GuestMemory::UnchangedSince(afterBegin, static_cast<std::size_t>(end - afterBegin), seen)) continue;
        if (trace) std::fprintf(stderr, "[flush] adjacent pending image 0x%llx+0x%llx layer %u advanced past a write-back's stamps (generation %llu -> %llu)\n", static_cast<unsigned long long>(texture->descriptor.baseAddress), static_cast<unsigned long long>(texture->guestBytes), layer, static_cast<unsigned long long>(seen), static_cast<unsigned long long>(now));
        texture->layerGeneration[layer] = now;
        texture->refreshGeneration();
    }
}

void StorageTexture::blockGenerations(std::vector<std::uint64_t>& generations) const {
    constexpr std::uint64_t block = 65536;
    const auto spanBegin = descriptor.baseAddress & ~(block - 1);
    generations.assign(static_cast<std::size_t>((descriptor.baseAddress + guestBytes - spanBegin + block - 1) / block), 0);
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        const auto end = layerBegin(layer) + layerBytes(layer);
        for (auto at = layerBegin(layer) & ~(block - 1); at < end; at += block) generations[static_cast<std::size_t>((at - spanBegin) / block)] = layerGeneration[layer];
    }
}

std::vector<bool> StorageTexture::rowAliasEdges(const StorageTexture& alias) const {
    std::vector<bool> edge(trackedLayers, false);
    const auto shared = std::min(guestBytes, alias.guestBytes);
    const std::pair<std::uint64_t, std::uint64_t> span{0, shared};
    // Padding of both (the right edge's partial blocks at one width) is no surface's element: the
    // copy leaves it and both write-backs keep the guest's bytes there. Only bytes that are an
    // element of exactly one of the two make a unit an edge.
    const auto mine = uncoveredBytes(std::span(&span, 1));
    const auto theirs = alias.uncoveredBytes(std::span(&span, 1));
    const auto clipped = [](const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, std::uint64_t begin, std::uint64_t end) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> inside;
        for (const auto& [from, to] : ranges) {
            if (to <= begin || from >= end) continue;
            inside.emplace_back(std::max(from, begin), std::min(to, end));
        }
        return inside;
    };
    for (std::uint32_t unit = 0; unit < trackedLayers && static_cast<std::uint64_t>(unit) * trackedLayerBytes < shared; ++unit) {
        const auto begin = static_cast<std::uint64_t>(unit) * trackedLayerBytes;
        const auto end = std::min(begin + trackedLayerBytes, shared);
        // The unit the shared span ends inside holds bytes of one of the surfaces only.
        if (end - begin != trackedLayerBytes || clipped(mine, begin, end) != clipped(theirs, begin, end)) edge[unit] = true;
    }
    return edge;
}

namespace {

// APS5_VERIFY_LAZY_WRITEBACK=1 checks the first 16 cases of each kind and every 16th after (each
// waits for the GPU); the test override checks every one.
bool verifySample(std::atomic<std::uint64_t>& calls) {
    const auto call = calls.fetch_add(1, std::memory_order_relaxed);
    return lazyVerifyOverride.load(std::memory_order_relaxed) > 0 || call < 16 || call % 16 == 0;
}

std::atomic<std::uint64_t> verifyBorrowCalls{0}, verifyThickCalls{0};

void compareRetiled(const char* what, const GuestTextureResource& descriptor, std::span<const std::byte> mine, std::span<const std::byte> reference, std::span<const std::pair<std::uint64_t, std::uint64_t>> runs) {
    lazyVerifyChecks.fetch_add(1, std::memory_order_relaxed);
    for (const auto& [begin, end] : runs) {
        if (end > mine.size() || end > reference.size()) continue;
        if (std::memcmp(mine.data() + begin, reference.data() + begin, static_cast<std::size_t>(end - begin)) == 0) continue;
        auto at = begin;
        while (at < end && mine[static_cast<std::size_t>(at)] == reference[static_cast<std::size_t>(at)]) ++at;
        lazyVerifyMismatches.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1, std::memory_order_relaxed) < 32) {
            std::fprintf(stderr, "[lazy-wb] APS5_VERIFY_LAZY_WRITEBACK: %s of 0x%llx %ux%u format %u tile %d differs from the eager bytes at +0x%llx (run +0x%llx..+0x%llx)\n", what, static_cast<unsigned long long>(descriptor.baseAddress), descriptor.width, descriptor.height, descriptor.format, static_cast<int>(descriptor.tileMode), static_cast<unsigned long long>(at), static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end));
            std::fflush(stderr);
        }
        return;
    }
}

}

std::vector<std::byte> StorageTexture::retiledForVerify(std::span<const std::pair<std::uint64_t, std::uint64_t>> runs, bool windowed) {
    // Every byte no retile writes keeps the canary, in both the windowed and the whole result.
    constexpr std::uint32_t canary = 0xa5c3e1f7u;
    const auto elementBytes = BytesPerElement(descriptor.format);
    const auto tiledBytes = (guestBytes + 3) & ~std::uint64_t{3};
    DeviceBuffer tiled(context, static_cast<std::size_t>(tiledBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer host(context, static_cast<std::size_t>(tiledBytes), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::vector<SliceWindow> windows;
    std::vector<std::uint64_t> positions;
    std::uint64_t linearTotal = sliceLinearBytes * arrayLayers;
    if (windowed) {
        windows = sliceWindows(runs);
        linearTotal = 0;
        for (const auto& window : windows) {
            positions.push_back(linearTotal);
            linearTotal += (window.linearBytes + 255) & ~std::uint64_t{255};
        }
    }
    DeviceBuffer linear(context, static_cast<std::size_t>(std::max<std::uint64_t>(linearTotal, 256)), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    detiler.BeginBatch();
    CommandBatch batch(context);
    const auto commands = batch.Handle();
    context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer")(commands, tiled.Handle(), 0, VK_WHOLE_SIZE, canary);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    std::vector<VkBufferImageCopy> regions;
    if (windowed) {
        for (std::size_t i = 0; i < windows.size(); ++i) {
            for (auto region : windows[i].regions) {
                region.bufferOffset += positions[i];
                regions.push_back(region);
            }
        }
    } else {
        regions = CopyRegions();
    }
    if (!regions.empty()) context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image, VK_IMAGE_LAYOUT_GENERAL, linear.Handle(), static_cast<std::uint32_t>(regions.size()), regions.data());
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    if (windowed) {
        for (std::size_t i = 0; i < windows.size(); ++i) {
            const auto& window = windows[i];
            const auto& mip = mips[window.level];
            // The tiled buffer holds the whole surface: the window's tiled bytes at their own place.
            auto range = window.window;
            range.tiledBase = 0;
            detiler.Dispatch(commands, descriptor.tileMode, elementBytes, linear.Handle(), positions[i], tiled.Handle(), geometry.GuestLayerOffset(window.layer) + mip.tiledOffset, mip, true, window.layer, geometry.thick, range);
        }
    } else {
        for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
            for (std::uint32_t level = 0; level < mips.size(); ++level) {
                if (!geometry.HasLayer(level, layer)) continue;
                const auto& mip = mips[level];
                detiler.Dispatch(commands, descriptor.tileMode, elementBytes, linear.Handle(), geometry.LinearLayerOffset(layer) + mip.linearOffset, tiled.Handle(), geometry.GuestLayerOffset(layer) + mip.tiledOffset, mip, true, layer, geometry.thick);
            }
        }
    }
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    CopyBuffer(context, commands, tiled.Handle(), 0, host.Handle(), 0, static_cast<std::size_t>(tiledBytes));
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    batch.SubmitAndWait();
    const auto bytes = host.Bytes();
    return std::vector<std::byte>(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(guestBytes));
}

void StorageTexture::verifyBorrow(StorageTexture& source, const std::vector<bool>& units) {
    if (!verifySample(verifyBorrowCalls)) return;
    // Outside the refresh's exemption: a completion the wait runs flushes this image like any other.
    struct Exempt {
        const StorageTexture* previous;
        ~Exempt() { refreshing = previous; }
    } exempt{std::exchange(refreshing, nullptr)};
    if (auto* recorder = Recorder::Active()) recorder->Sync();
    // The borrowed units as this image's retile stores them, against the alias's: what its eager
    // write-back would have put in memory for this image to read.
    const auto runs = unitRuns(units);
    const auto mine = retiledForVerify(runs, true);
    const auto theirs = source.retiledForVerify(runs, true);
    compareRetiled("an alias borrow", descriptor, mine, theirs, runs);
}

void StorageTexture::verifyThickWindows(std::span<const std::pair<std::uint64_t, std::uint64_t>> runs) {
    if (!verifySample(verifyThickCalls)) return;
    if (auto* recorder = Recorder::Active()) recorder->Sync();
    const auto windowed = retiledForVerify(runs, true);
    const auto whole = retiledForVerify(runs, false);
    compareRetiled("a thick partial write-back", descriptor, windowed, whole, runs);
}

std::shared_ptr<StorageTexture> StorageTexture::pendingAlias() const {
    static const bool disabled = std::getenv("APS5_NO_ALIAS_BORROW") != nullptr;
    if (disabled || !blockUnits) return nullptr;
    static const bool subrectDefault = std::getenv("APS5_SUBRECT_BORROW") != nullptr;
    // Row aliases both ways under `lazy_writeback` (Refresh stores their edge units first), the
    // shorter refreshing over the taller alone under `subrect_borrow` as before.
    const bool lazy = LazyWriteBack();
    const bool subrect = (lazy || LiveFlag("subrect_borrow", subrectDefault)) && descriptor.mipCount == 1 && geometry.imageLayers == 1 && geometry.imageDepth == 1;
    std::vector<std::shared_ptr<StorageTexture>> candidates;
    {
        auto& pending = Pending();
        std::lock_guard lock(pending.mutex);
        for (auto* texture : pending.textures) {
            if (texture == this || texture->descriptor.baseAddress != descriptor.baseAddress || !texture->blockUnits) continue;
            if (texture->guestBytes != guestBytes && !(subrect && (texture->guestBytes > guestBytes || lazy))) continue;
            if (auto alive = texture->weak_from_this().lock()) candidates.push_back(std::move(alive));
        }
    }
    for (const auto& candidate : candidates) {
        const auto& mine = descriptor;
        const auto& theirs = candidate->descriptor;
        // The same texels under another format of the same size: the device copy moves them bit
        // for bit, as the guest bytes would.
        const bool sameShape = mine.width == theirs.width && mine.height == theirs.height && mine.depthOrLastArray == theirs.depthOrLastArray && mine.baseArray == theirs.baseArray && mine.mipCount == theirs.mipCount && mine.tileMode == theirs.tileMode && mine.dimension == theirs.dimension && BytesPerElement(mine.format) == BytesPerElement(theirs.format) && geometry.imageLayers == candidate->geometry.imageLayers && geometry.imageDepth == candidate->geometry.imageDepth && trackedLayers == candidate->trackedLayers;
        const bool topRows = subrect && mine.width == theirs.width && mine.height <= theirs.height && mine.depthOrLastArray == theirs.depthOrLastArray && mine.baseArray == theirs.baseArray && theirs.mipCount == 1 && mine.tileMode == theirs.tileMode && mine.dimension == theirs.dimension && BytesPerElement(mine.format) == BytesPerElement(theirs.format) && candidate->geometry.imageLayers == 1 && candidate->geometry.imageDepth == 1 && trackedLayers <= candidate->trackedLayers;
        // The taller one refreshing over the shorter: its rows below the alias's are its own.
        const bool tallRows = subrect && lazy && mine.width == theirs.width && mine.height > theirs.height && mine.depthOrLastArray == theirs.depthOrLastArray && mine.baseArray == theirs.baseArray && theirs.mipCount == 1 && mine.tileMode == theirs.tileMode && mine.dimension == theirs.dimension && BytesPerElement(mine.format) == BytesPerElement(theirs.format) && candidate->geometry.imageLayers == 1 && candidate->geometry.imageDepth == 1 && candidate->trackedLayers <= trackedLayers;
        if ((sameShape || topRows || tallRows) && candidate->Cached() && candidate->uploadedKeys == DccKeys::Uncompressed) return candidate;
    }
    return nullptr;
}

void StorageTexture::forgetBorrowed(std::uint32_t first, std::uint32_t count) {
    for (auto& record : layerBorrows) {
        for (auto unit = first; unit < first + count && unit < record.units.size(); ++unit) record.units[unit] = false;
    }
    std::erase_if(layerBorrows, [](const LayerBorrow& record) { return std::none_of(record.units.begin(), record.units.end(), [](bool held) { return held; }); });
    if (borrowedUnits.empty()) return;
    for (auto unit = first; unit < first + count && unit < borrowedUnits.size(); ++unit) borrowedUnits[unit] = false;
    if (std::none_of(borrowedUnits.begin(), borrowedUnits.end(), [](bool held) { return held; })) {
        borrowedUnits.clear();
        borrowedFrom.reset();
    }
}

std::uint64_t StorageTexture::borrowUnits(StorageTexture& source, const std::vector<bool>& units) {
    const auto runs = unitRuns(units);
    const auto windows = sliceWindows(runs);
    Require(!windows.empty(), "storage image borrows no unit");
    std::vector<VkImageCopy> regions;
    for (const auto& window : windows) {
        for (const auto& region : window.regions) regions.push_back({region.imageSubresource, region.imageOffset, region.imageSubresource, region.imageOffset, region.imageExtent});
    }
    std::uint64_t bytes = 0;
    for (const auto& [begin, end] : runs) bytes += end - begin;
    recordBorrowCopy(source, regions, bytes);
    if (borrowedFrom.lock() != source.weak_from_this().lock() || borrowedUnits.size() != trackedLayers) {
        borrowedUnits.assign(trackedLayers, false);
        borrowedFrom = source.weak_from_this();
    }
    borrowedVersion = source.version;
    source.lent = true;
    for (std::uint32_t unit = 0; unit < trackedLayers; ++unit) {
        if (units[unit]) borrowedUnits[unit] = true;
    }
    ++version;
    return bytes;
}

void StorageTexture::recordBorrowCopy(StorageTexture& source, const std::vector<VkImageCopy>& regions, std::uint64_t bytes) {
    auto* recorder = Recorder::Active();
    std::unique_ptr<CommandBatch> batch;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    auto timing = Recorder::NoTiming;
    bool elided = false;
    if (recorder != nullptr) {
        const std::array<std::pair<VkImage, bool>, 2> images{{{source.image, false}, {image, true}}};
        commands = Recorder::BarrierTracking() ? recorder->Commands(Recorder::CommandClass::StorageUpload, Recorder::Access{{}, {}, images, VK_PIPELINE_STAGE_TRANSFER_BIT}, elided) : recorder->Commands();
        timing = recorder->BeginGpuTiming(Recorder::CommandClass::StorageUpload);
        recorder->NoteTimedSurface(timing, descriptor.baseAddress, guestBytes, static_cast<std::uint32_t>(storageFormat), descriptor.format, descriptor.width, descriptor.height, "borrow-copy", static_cast<std::uint32_t>(regions.size()));
        if (auto self = weak_from_this().lock()) recorder->Keep(std::move(self));
        if (auto other = source.weak_from_this().lock()) recorder->Keep(std::move(other));
        if (elided) Recorder::CountSkipped(Recorder::CommandClass::StorageUpload, 2);
        else Recorder::CountBarriers(Recorder::CommandClass::StorageUpload, 2);
    } else {
        batch = std::make_unique<CommandBatch>(context);
        commands = batch->Handle();
    }
    // APS5_GPU_CHECKPOINTS: the copy names both surfaces, so a fault inside it names the alias.
    {
        char text[160];
        std::snprintf(text, sizeof(text), "borrow copy %ux%u fmt%u <- 0x%llx %ux%u fmt%u, %zu regions", descriptor.width, descriptor.height, descriptor.format, static_cast<unsigned long long>(source.descriptor.baseAddress), source.descriptor.width, source.descriptor.height, source.descriptor.format, regions.size());
        GpuCheckpoints::Set(context, commands, "storage-upload borrow (dst, bytes, regions)", descriptor.baseAddress, bytes, regions.size(), text);
    }
    // The regions come from this image's unit windows; a source of another shape over the same
    // memory (a shorter or narrower target the transient pool handed out before) can end inside
    // them. A copy region outside either image is undefined behaviour on the device: on NVIDIA it
    // faults the GPU (VK_ERROR_DEVICE_LOST, PPSA01317's World Map with a progressed save). Each
    // region is clipped to what both images hold; one with nothing left, or with a mip or layer
    // one image lacks, is left out. The rows past the source's end keep this image's content, as
    // the source holds none for them. APS5_TRACE_BORROW_CLIP=1 logs the clipped regions.
    std::vector<VkImageCopy> fitting;
    fitting.reserve(regions.size());
    {
        struct Room {
            bool exists;
            std::uint32_t x, y, z;
        };
        const auto room = [](const StorageTexture& surface, const VkImageSubresourceLayers& layers, const VkOffset3D& offset) {
            if (layers.mipLevel >= surface.descriptor.mipCount || layers.baseArrayLayer + layers.layerCount > surface.geometry.imageLayers || offset.x < 0 || offset.y < 0 || offset.z < 0) return Room{false, 0, 0, 0};
            const auto width = std::max(surface.descriptor.width >> layers.mipLevel, 1u);
            const auto height = std::max(surface.descriptor.height >> layers.mipLevel, 1u);
            const auto depth = std::max(surface.geometry.imageDepth >> layers.mipLevel, 1u);
            const auto left = [](std::uint32_t size, std::int32_t at) { return static_cast<std::uint32_t>(at) < size ? size - static_cast<std::uint32_t>(at) : 0u; };
            return Room{true, left(width, offset.x), left(height, offset.y), left(depth, offset.z)};
        };
        // Block-compressed formats copy whole 4x4 blocks except at an image's edge.
        const bool compressed = storageFormat >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && storageFormat <= VK_FORMAT_BC7_SRGB_BLOCK;
        static const bool trace = std::getenv("APS5_TRACE_BORROW_CLIP") != nullptr;
        static std::atomic<int> traced{0};
        static std::atomic<std::uint64_t> clipped{0}, dropped{0};
        for (const auto& region : regions) {
            const auto from = room(source, region.srcSubresource, region.srcOffset);
            const auto to = room(*this, region.dstSubresource, region.dstOffset);
            VkImageCopy kept = region;
            kept.extent.width = std::min({region.extent.width, from.x, to.x});
            kept.extent.height = std::min({region.extent.height, from.y, to.y});
            kept.extent.depth = std::min({region.extent.depth, from.z, to.z});
            const bool changed = kept.extent.width != region.extent.width || kept.extent.height != region.extent.height || kept.extent.depth != region.extent.depth;
            const auto endsMidBlock = [&](std::uint32_t extent, std::uint32_t destinationRoom) { return extent % 4u != 0 && extent != destinationRoom; };
            const bool keep = from.exists && to.exists && kept.extent.width != 0 && kept.extent.height != 0 && kept.extent.depth != 0 && !(compressed && changed && (endsMidBlock(kept.extent.width, to.x) || endsMidBlock(kept.extent.height, to.y)));
            if (keep) fitting.push_back(kept);
            if (!changed && keep) continue;
            (keep ? clipped : dropped).fetch_add(1, std::memory_order_relaxed);
            if (trace && traced.fetch_add(1, std::memory_order_relaxed) < 40) {
                std::fprintf(stderr, "[borrow-copy] %s: dst 0x%llx %ux%u fmt%u mips %u layers %u depth %u <- src 0x%llx %ux%u fmt%u mips %u layers %u depth %u; region src mip %u layer %u+%u off %d,%d,%d extent %ux%ux%u -> %ux%ux%u\n", keep ? "clipped" : "dropped", static_cast<unsigned long long>(descriptor.baseAddress), descriptor.width, descriptor.height, descriptor.format, descriptor.mipCount, geometry.imageLayers, geometry.imageDepth, static_cast<unsigned long long>(source.descriptor.baseAddress), source.descriptor.width, source.descriptor.height, source.descriptor.format, source.descriptor.mipCount, source.geometry.imageLayers, source.geometry.imageDepth, region.srcSubresource.mipLevel, region.srcSubresource.baseArrayLayer, region.srcSubresource.layerCount, region.srcOffset.x, region.srcOffset.y, region.srcOffset.z, region.extent.width, region.extent.height, region.extent.depth, kept.extent.width, kept.extent.height, kept.extent.depth);
                std::fflush(stderr);
            }
        }
        // One line every 10 s while borrows clip, so a run shows the fix at work without tracing.
        static std::atomic<std::int64_t> reported{0};
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = reported.load(std::memory_order_relaxed);
        if ((clipped.load(std::memory_order_relaxed) | dropped.load(std::memory_order_relaxed)) != 0 && now - last > 10'000'000'000ll && reported.compare_exchange_strong(last, now)) {
            std::fprintf(stderr, "[borrow-copy] (10 s) regions clipped to both images %llu, dropped %llu\n", static_cast<unsigned long long>(clipped.exchange(0)), static_cast<unsigned long long>(dropped.exchange(0)));
        }
    }
    VkImageMemoryBarrier barriers[2]{};
    for (auto& barrier : barriers) {
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
    }
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, source.descriptor.mipCount, 0, source.geometry.imageLayers};
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].image = source.image;
    barriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].image = image;
    if (!elided) context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
    if (!fitting.empty()) context.Function<PFN_vkCmdCopyImage>("vkCmdCopyImage")(commands, source.image, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(fitting.size()), fitting.data());
    VkImageMemoryBarrier copied = barriers[1];
    copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    if (!elided) context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &copied);
    if (batch) batch->SubmitAndWait();
    else recorder->EndGpuTiming(timing, bytes);
    countStorageUpload(3, bytes);
}

std::vector<StorageTexture::LayerAlias> StorageTexture::pendingLayerAliases() const {
    static const bool disabled = std::getenv("APS5_NO_LAYER_BORROW") != nullptr;
    std::vector<LayerAlias> links;
    if (disabled || !blockUnits || descriptor.dccAddress != 0 || geometry.thick || geometry.imageDepth != 1 || uploadedKeys != DccKeys::Uncompressed || trackedLayerBytes != 65536) return links;
    std::vector<std::shared_ptr<StorageTexture>> candidates;
    {
        auto& pending = Pending();
        std::lock_guard lock(pending.mutex);
        if (!pending.textures.MayOverlap(descriptor.baseAddress, static_cast<std::size_t>(guestBytes))) return links;
        for (auto* texture : pending.textures) {
            if (texture == this || !texture->blockUnits || !texture->overlaps(descriptor.baseAddress, static_cast<std::size_t>(guestBytes))) continue;
            if (auto alive = texture->weak_from_this().lock()) candidates.push_back(std::move(alive));
        }
    }
    const auto sameMips = [](const std::vector<TileMipLayout>& a, const std::vector<TileMipLayout>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t level = 0; level < a.size(); ++level) {
            if (a[level].width != b[level].width || a[level].height != b[level].height || a[level].tiledOffset != b[level].tiledOffset || a[level].tiledSize != b[level].tiledSize || a[level].pitchBytes != b[level].pitchBytes || a[level].tail != b[level].tail || a[level].tailX != b[level].tailX || a[level].tailY != b[level].tailY || a[level].blocksPerRow != b[level].blocksPerRow) return false;
        }
        return true;
    };
    for (auto& candidate : candidates) {
        const auto& theirs = candidate->descriptor;
        // The same per-layer layout: the same texels of each (layer, mip) slice at the same offset
        // inside a layer, and layers of the same size, a whole number of them apart.
        if (theirs.width != descriptor.width || theirs.height != descriptor.height || theirs.mipCount != descriptor.mipCount || theirs.tileMode != descriptor.tileMode || BytesPerElement(theirs.format) != BytesPerElement(descriptor.format) || theirs.dccAddress != 0) continue;
        if (candidate->geometry.thick || candidate->geometry.imageDepth != 1 || candidate->trackedLayerBytes != 65536 || candidate->storageFormat != storageFormat) continue;
        if (!candidate->Cached() || candidate->uploadedKeys != DccKeys::Uncompressed) continue;
        const auto layerBytes = geometry.layerBytes;
        if (layerBytes == 0 || candidate->geometry.layerBytes != layerBytes || !sameMips(mips, candidate->mips)) continue;
        if (geometry.layerBytes * geometry.layers != guestBytes || candidate->geometry.layerBytes * candidate->geometry.layers != candidate->guestBytes) continue;
        const auto delta = static_cast<std::int64_t>(descriptor.baseAddress) - static_cast<std::int64_t>(theirs.baseAddress);
        if (delta % static_cast<std::int64_t>(layerBytes) != 0 || delta % 65536 != 0) continue;
        // A same-base alias of the same shape is pendingAlias's.
        if (delta == 0 && geometry.layers == candidate->geometry.layers) continue;
        const auto begin = std::max(descriptor.baseAddress, theirs.baseAddress);
        const auto end = std::min(descriptor.baseAddress + guestBytes, theirs.baseAddress + candidate->guestBytes);
        if (begin >= end) continue;
        LayerAlias link{candidate, delta / 65536, static_cast<std::uint32_t>((begin - descriptor.baseAddress) / 65536), static_cast<std::uint32_t>((end - descriptor.baseAddress + 65535) / 65536)};
        bool aligned = static_cast<std::int64_t>(link.first) + link.offset >= 0 && static_cast<std::int64_t>(link.end) + link.offset <= static_cast<std::int64_t>(candidate->trackedLayers) && link.end <= trackedLayers;
        for (auto unit = link.first; aligned && unit < link.end; ++unit) aligned = this->layerBytes(unit) == candidate->layerBytes(static_cast<std::uint32_t>(unit + link.offset));
        if (!aligned) continue;
        std::vector<bool> every(trackedLayers, false);
        for (auto unit = link.first; unit < link.end; ++unit) every[unit] = true;
        if (!layerCopies(*candidate, link.offset, every, nullptr)) continue;
        links.push_back(std::move(link));
    }
    return links;
}

bool StorageTexture::layerAliasExempt(const LayerAlias& link) const {
    const auto& source = *link.source;
    const auto count = link.end - link.first;
    std::vector<std::uint64_t> sourceGenerations(count);
    for (std::uint32_t k = 0; k < count; ++k) sourceGenerations[k] = source.layerGeneration[static_cast<std::size_t>(link.first + k + link.offset)];
    std::vector<std::uint8_t> sourceStamped(count);
    const auto begin = static_cast<std::uint64_t>(link.first) * trackedLayerBytes;
    const auto end = std::min<std::uint64_t>(guestBytes, static_cast<std::uint64_t>(link.end) * trackedLayerBytes);
    if (!GuestMemory::ChangedBlocks(descriptor.baseAddress + begin, static_cast<std::size_t>(end - begin), sourceGenerations, sourceStamped)) return false;
    bool any = false;
    for (std::uint32_t k = 0; k < count; ++k) {
        const auto unit = link.first + k;
        const auto sourceUnit = static_cast<std::size_t>(unit + link.offset);
        if (!source.layerPending[sourceUnit]) continue;
        if (source.layerGeneration[sourceUnit] == 0 || sourceStamped[k] == GuestMemory::BlockMaybeWritten || (sourceStamped[k] == GuestMemory::BlockUnchanged && layerPending[unit])) return false;
        any = true;
    }
    return any;
}

bool StorageTexture::layerCopies(const StorageTexture& source, std::int64_t offset, const std::vector<bool>& units, std::vector<VkImageCopy>* regions) const {
    const auto runs = unitRuns(units);
    const auto delta = static_cast<std::uint64_t>(offset * static_cast<std::int64_t>(trackedLayerBytes));
    std::vector<std::pair<std::uint64_t, std::uint64_t>> sourceRuns;
    for (const auto& [begin, end] : runs) sourceRuns.emplace_back(begin + delta, end + delta);
    const auto mine = sliceWindows(runs);
    const auto theirs = source.sliceWindows(sourceRuns);
    if (mine.empty() || mine.size() != theirs.size()) return false;
    for (std::size_t w = 0; w < mine.size(); ++w) {
        const auto& a = mine[w];
        const auto& b = theirs[w];
        if (a.level != b.level || a.tiledBegin + delta != b.tiledBegin || a.tiledEnd + delta != b.tiledEnd || a.regions.size() != b.regions.size()) return false;
        for (std::size_t r = 0; r < a.regions.size(); ++r) {
            const auto& x = a.regions[r];
            const auto& y = b.regions[r];
            if (x.imageSubresource.mipLevel != y.imageSubresource.mipLevel || x.imageOffset.x != y.imageOffset.x || x.imageOffset.y != y.imageOffset.y || x.imageOffset.z != y.imageOffset.z || x.imageExtent.width != y.imageExtent.width || x.imageExtent.height != y.imageExtent.height || x.imageExtent.depth != y.imageExtent.depth) return false;
            if (regions != nullptr) regions->push_back({y.imageSubresource, y.imageOffset, x.imageSubresource, x.imageOffset, x.imageExtent});
        }
    }
    return true;
}

void StorageTexture::borrowLayerUnits(StorageTexture& source, std::int64_t offset, const std::vector<bool>& units) {
    std::vector<VkImageCopy> regions;
    Require(layerCopies(source, offset, units, &regions) && !regions.empty(), "storage image layer borrow does not pair with its source");
    std::uint64_t bytes = 0;
    for (const auto& [begin, end] : unitRuns(units)) bytes += end - begin;
    recordBorrowCopy(source, regions, bytes);
    const auto from = source.weak_from_this().lock();
    auto record = std::find_if(layerBorrows.begin(), layerBorrows.end(), [&](const LayerBorrow& entry) { return entry.from.lock() == from && entry.offset == offset; });
    if (record == layerBorrows.end()) record = layerBorrows.insert(layerBorrows.end(), LayerBorrow{from, offset, 0, {}});
    if (record->units.size() != trackedLayers) record->units.assign(trackedLayers, false);
    record->version = source.version;
    source.lent = true;
    for (std::uint32_t unit = 0; unit < trackedLayers; ++unit) {
        if (units[unit]) record->units[unit] = true;
    }
    ++version;
}

void StorageTexture::writeBack(std::uint64_t address, std::size_t bytes) {
    // Block units are stored per access; an image whose units keep being asked for in pieces (a
    // consumer touching it through many small ranges, each piece re-arming the pending memos of
    // the next dispatch) stores every pending unit once more than `pieces` partial stores fall
    // within `window` presents, and keeps doing so for another window. Off by default: the title's
    // aliased images are stored in pieces by design (every piece is another image's hand-over) and
    // widening them stored whole images again (movie stage: 39 vs 35 ms of GPU per present, 188 vs
    // 214 presents per 10 s); APS5_BLOCK_WRITEBACK_WIDEN=1 enables the widening, and
    // APS5_BLOCK_WRITEBACK_EACH=1 stores the touched units only, every time.
    static const bool each = std::getenv("APS5_BLOCK_WRITEBACK_EACH") != nullptr || std::getenv("APS5_BLOCK_WRITEBACK_WIDEN") == nullptr;
    static const std::uint32_t pieces = [] { const char* text = std::getenv("APS5_BLOCK_WRITEBACK_PIECES"); return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 10)) : 4u; }();
    static const std::uint64_t window = [] { const char* text = std::getenv("APS5_BLOCK_WRITEBACK_FRAMES"); return text != nullptr ? std::strtoull(text, nullptr, 10) : 8ull; }();
    std::vector<bool> layers(trackedLayers, false);
    std::size_t selected = 0, pendingCount = 0;
    // Under a clear code the texels outside the stored units would read as the clear while the
    // keys, marked uncompressed for the whole surface, say texels: such an image stores whole.
    const bool whole = descriptor.dccAddress != 0 && uploadedKeys != DccKeys::Uncompressed;
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (!layerPending[layer]) continue;
        ++pendingCount;
        layers[layer] = whole || (address < layerBegin(layer) + layerBytes(layer) && layerBegin(layer) < address + bytes);
        if (layers[layer]) ++selected;
    }
    if (selected == 0) {
        reconcilePending();
        return;
    }
    // A thick surface's units under `lazy_writeback` off: every pending one at once, as the whole-
    // surface tracking stored it; on, the units the access touches (the others stay on the GPU).
    if (geometry.thick && blockUnits && selected < pendingCount) {
        if (!LazyWriteBack()) {
            layers = layerPending;
        } else {
            std::uint64_t storedBytes = 0, keptBytes = 0;
            for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
                if (!layerPending[layer]) continue;
                (layers[layer] ? storedBytes : keptBytes) += layerBytes(layer);
            }
            lazyThickStores.fetch_add(1, std::memory_order_relaxed);
            lazyThickStoredBytes.fetch_add(storedBytes, std::memory_order_relaxed);
            lazyThickKeptBytes.fetch_add(keptBytes, std::memory_order_relaxed);
        }
    }
    if (blockUnits && !each && selected < pendingCount) {
        const auto present = Recorder::Presents();
        if (present - partialWindowStart >= window) {
            partialWindowStart = present;
            partialStores = 0;
        }
        if (present < storeWholeUntil || ++partialStores > pieces) {
            storeWholeUntil = present + window;
            layers = layerPending;
            coalescedWriteBacks.fetch_add(1, std::memory_order_relaxed);
        }
    }
    writeBackLayers(layers);
}

void StorageTexture::writeBackLayers(const std::vector<bool>& layers) {
    GuestMemory::NoteProbeEvent("storage write-back");
    CaptureTrace::Log("writeback image=%llx generation=%llu reason=%s units=%zu", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(generation), flushReason, static_cast<std::size_t>(std::count(layers.begin(), layers.end(), true)));
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    PhaseTimer timer;
    struct Account {
        bool enabled;
        PhaseTimer& timer;
        ~Account() { if (enabled) Profile().storageWriteBack += timer.lap(); }
    } account{profile, timer};
    // Whatever the outcome, the registration follows the pending flags (a throw leaves the
    // layers pending).
    struct Reconcile {
        StorageTexture& texture;
        ~Reconcile() { texture.reconcilePending(); }
    } reconcile{*this};
    // APS5_VERIFY_LAZY_WRITEBACK: a thick block-unit store's windows against the whole-surface
    // retile, before the keep decision below reads the stamps (the check's GPU wait runs
    // completions, whose stores that decision must see).
    if (geometry.thick && blockUnits && VerifyLazyWriteBack() && HostImportFor(context, descriptor.baseAddress, static_cast<std::size_t>(guestBytes)) != nullptr) verifyThickWindows(unitRuns(layers));
    const auto elementBytes = BytesPerElement(descriptor.format);
    // 64 KiB blocks the CPU wrote since the layer was last in sync keep the CPU's bytes: the game
    // may have reused the memory for something else entirely (see layerGeneration). A generation of
    // zero means no tracking, and the layer is stored whole.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> keep;
    std::vector<bool> skippedLayer(trackedLayers, false);
    bool skippedAny = false;
    std::size_t skipped = 0;
    std::uint64_t firstStored = descriptor.baseAddress + guestBytes;
    std::uint64_t lastStored = descriptor.baseAddress;
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (!layers[layer]) continue;
        firstStored = std::min(firstStored, layerBegin(layer));
        lastStored = std::max(lastStored, layerBegin(layer) + layerBytes(layer));
    }
    if (lastStored <= firstStored) return;
    constexpr std::uint64_t block = 65536;
    const auto spanBegin = firstStored & ~(block - 1);
    const auto spanBlocks = static_cast<std::size_t>((lastStored - spanBegin + block - 1) / block);
    // The keep-blocks decision must see every CPU write to the stored span up to now, so it
    // bypasses the per-packet collect memo (a Refresh collect earlier in the same packet would
    // satisfy it); the unselected layers keep their own generations and are collected by their
    // own refreshes. One tracker lock then reads the span's stamps against the layers' generations.
    GuestMemory::CollectWritesUncached(firstStored, static_cast<std::size_t>(lastStored - firstStored));
    std::vector<std::uint64_t> generations(spanBlocks, 0);
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (!layers[layer]) continue;
        const auto end = layerBegin(layer) + layerBytes(layer);
        for (auto at = layerBegin(layer) & ~(block - 1); at < end; at += block) generations[static_cast<std::size_t>((at - spanBegin) / block)] = layerGeneration[layer];
    }
    std::vector<std::uint8_t> changedBlocks(spanBlocks);
    const bool tracked = GuestMemory::ChangedBlocks(firstStored, static_cast<std::size_t>(lastStored - firstStored), generations, changedBlocks);
    if (!tracked) {
        // Debug count: results stored whole over memory outside the write watch, where CPU writes
        // made while they were pending are not detected and get overwritten.
        static std::atomic<int> untrackedStores{0};
        if (untrackedStores.fetch_add(1, std::memory_order_relaxed) < 40) std::fprintf(stderr, "[untracked-store] image 0x%llx+0x%llx %ux%u format %u stores 0x%llx..0x%llx whole (outside the write watch)\n", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), descriptor.width, descriptor.height, descriptor.format, static_cast<unsigned long long>(firstStored), static_cast<unsigned long long>(lastStored));
    }
    for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
        if (!layers[layer]) continue;
        const auto begin = layerBegin(layer);
        const auto end = begin + layerBytes(layer);
        if (!tracked || layerGeneration[layer] == 0) {
            keep.emplace_back(begin, end);
            continue;
        }
        for (auto at = begin & ~(block - 1); at < end; at += block) {
            const auto from = std::max(at, begin);
            const auto to = std::min(at + block, end);
            if (from >= to) continue;
            const bool edge = from != at || to != at + block;
            if (changedBlocks[static_cast<std::size_t>((at - spanBegin) / block)] == GuestMemory::BlockWritten && (!edge || GuestMemory::StoredOver(from, static_cast<std::size_t>(to - from), layerGeneration[layer]))) {
                skippedAny = true;
                skippedLayer[layer] = true;
                ++skipped;
                continue;
            }
            if (!keep.empty() && keep.back().second == from) keep.back().second = to;
            else keep.emplace_back(from, to);
        }
    }
    // Debug aid: APS5_TRACE_FLUSH also names the blocks a write-back leaves to the CPU.
    static const bool traceKept = std::getenv("APS5_TRACE_FLUSH") != nullptr;
    if (traceKept && skippedAny) {
        const auto first = keep.empty() ? descriptor.baseAddress + guestBytes : keep.front().first;
        std::fprintf(stderr, "[flush] image 0x%llx+0x%llx keeps %zu CPU-written 64 KiB blocks (first stored byte at +0x%llx, %zu ranges stored, generation %llu)\n", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), skipped, static_cast<unsigned long long>(first - descriptor.baseAddress), keep.size(), static_cast<unsigned long long>(generation));
    }
    // Taken before this write-back stamps anything (see advanceAdjacent), with the 64 KiB block span
    // its stamps cover.
    const auto firstBlock = firstStored / block;
    const auto lastBlock = (lastStored - 1) / block;
    const auto adjacent = adjacentPendingUnchanged(firstBlock, lastBlock);
    // After the store: the stored layers are in sync at a fresh generation unless blocks were kept
    // for the CPU (then the layer is stale there, keeps its old generation and re-uploads at its
    // next use); their results are no longer pending either way. The collect covers the stored
    // span, which holds every stored layer's pages. Pieces retiled into the import's unit shadow
    // (`shadowed`, with its import) are fresh at the generation taken after this write-back's
    // own stamps, so every later import writer stamps newer.
    std::vector<ShadowedRange> shadowed;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> imported;
    const HostImport* shadowImport = nullptr;
    const auto settle = [&](bool memoizedCollect) {
        for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
            if (layers[layer]) layerPending[layer] = false;
        }
        const bool allKept = std::none_of(skippedLayer.begin(), skippedLayer.end(), [](bool kept) { return kept; });
        if (allKept || !adjacent.empty()) {
            const auto now = memoizedCollect ? GuestMemory::CollectWrites(firstStored, static_cast<std::size_t>(lastStored - firstStored)) : GuestMemory::CollectWritesUncached(firstStored, static_cast<std::size_t>(lastStored - firstStored));
            for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
                if (layers[layer] && !skippedLayer[layer]) layerGeneration[layer] = now;
            }
            refreshGeneration();
            advanceAdjacent(adjacent, now, firstBlock, lastBlock);
            if (!shadowed.empty()) MarkShadowed(*shadowImport, shadowed, now);
        } else if (!shadowed.empty()) {
            MarkShadowed(*shadowImport, shadowed, GuestMemory::TrackerGeneration());
        }
    };
    if (keep.empty()) {
        // Every selected block was written by the CPU since the layer's generation: nothing to
        // store, the keys stay as they are, and the image is stale there (re-uploaded at its next
        // use, never from a matching `original`).
        emptyWriteBacks.fetch_add(1, std::memory_order_relaxed);
        originalValid = false;
        settle(true);
        return;
    }
    if (profile) {
        // The [wb-cause] attribution: the surface, why it is stored, and for another image's
        // refresh which image refreshes over it.
        std::uint64_t storedBytes = 0;
        for (const auto& [from, to] : keep) storedBytes += to - from;
        char by[112] = "";
        if (refreshing != nullptr && refreshing != this) {
            const auto& other = refreshing->descriptor;
            std::snprintf(by, sizeof(by), " for 0x%llx %ux%u fmt%u tile%d dim%d", static_cast<unsigned long long>(other.baseAddress), other.width, other.height, other.format, static_cast<int>(other.tileMode), static_cast<int>(other.dimension));
        }
        // The access the store answers, to a power of two (its own refresh or an eviction: none).
        char access[32] = "";
        if (flushRequestBytes != 0 && flushRequestBytes < guestBytes) std::snprintf(access, sizeof(access), " (access <=%llu KiB)", static_cast<unsigned long long>(std::bit_ceil(static_cast<std::uint64_t>(flushRequestBytes)) / 1024));
        char key[288];
        std::snprintf(key, sizeof(key), "0x%llx+0x%llx %ux%ux%u fmt%u tile%d dim%d%s <- %s%s%s", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), descriptor.width, descriptor.height, descriptor.depthOrLastArray + 1u, descriptor.format, static_cast<int>(descriptor.tileMode), static_cast<int>(descriptor.dimension), blockUnits ? "" : " whole", flushReason != nullptr ? flushReason : "other", access, by);
        auto& causes = Causes();
        std::lock_guard lock(causes.mutex);
        auto& totals = causes.totals[key];
        ++totals.first;
        totals.second += storedBytes;
    }
    if (const auto* import = HostImportFor(context, descriptor.baseAddress, static_cast<std::size_t>(guestBytes))) {
        if (blockUnits) {
            // The kept blocks alone pass through the retiler (windows of their slices).
            shadowImport = import;
            const auto storedBytes = writeBackWindows(*import, keep, firstStored, lastStored, shadowed, imported);
            countStorageWriteBack(storedBytes, true);
            if (profile) Profile().storageGpu += timer.lap();
            originalValid = false;
            traceKeyStore("block write-back", descriptor, guestBytes);
            if (!IsDccClear(filledKeys)) MarkDccUncompressed(context, descriptor.dccAddress, guestBytes, DccKeyCount(descriptor, guestBytes));
            uploadedKeys = DccKeys::Uncompressed;
            for (const auto& [from, to] : keep) GuestMemory::MarkWritten(from, static_cast<std::size_t>(to - from));
            settle(true);
            ++Profile().storageDirectWriteBacks;
            return;
        }
        // The whole-layer copies below overwrite the interior units of the span; the edge units
        // keep a neighbour's shadowed results in the rest of their bytes, which reach the import
        // first (this write-back's stamps then make those units stale).
        PublishShadow(firstStored, static_cast<std::size_t>(lastStored - firstStored), PublishScope::PartialUnits, PublishReason::Upload);
        // The surface lives in host-imported memory: the retiler writes into device scratch and the
        // untouched blocks are copied into the imported bytes in place, recorded behind the work that
        // produced the image; nothing crosses to the CPU.
        auto linear = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(sliceLinearBytes * arrayLayers), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        auto tiledScratch = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(guestBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        std::vector<CopiedBytes> copied;
        for (const auto& [from, to] : keep) copied.push_back({from - descriptor.baseAddress, to - descriptor.baseAddress, from - descriptor.baseAddress});
        const auto padding = paddingSeeds(*import, std::move(copied));
        detiler.BeginBatch();
        auto* recorder = Recorder::Active();
        std::unique_ptr<CommandBatch> batch;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        auto timing = Recorder::NoTiming;
        std::uint64_t storedBytes = 0;
        for (const auto& [from, to] : keep) storedBytes += to - from;
        bool elided = false;
        if (recorder != nullptr) {
            recorder->FlushKeyStoresOverlapping(firstStored, static_cast<std::size_t>(lastStored - firstStored));
            recorder->FlushStoresOverlapping(firstStored, static_cast<std::size_t>(lastStored - firstStored));
            if (Recorder::BarrierTracking()) {
                const std::pair<VkImage, bool> read{image, false};
                std::vector<std::pair<std::uint64_t, std::uint64_t>> paddingReads(padding.importReads.begin(), padding.importReads.end());
                paddingReads.insert(paddingReads.end(), padding.shadowReads.begin(), padding.shadowReads.end());
                commands = recorder->Commands(Recorder::CommandClass::StorageWriteBack, Recorder::Access{paddingReads, keep, std::span(&read, 1), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT}, elided);
            } else {
                commands = recorder->Commands();
            }
            timing = recorder->BeginGpuTiming(Recorder::CommandClass::StorageWriteBack);
            recorder->NoteTimedSurface(timing, descriptor.baseAddress, guestBytes, static_cast<std::uint32_t>(storageFormat), descriptor.format, descriptor.width, descriptor.height, CauseKind("writeback-layers"), static_cast<std::uint32_t>(keep.size()));
            recorder->Keep(linear, linear->Size());
            recorder->Keep(tiledScratch, tiledScratch->Size());
            for (const auto& slab : padding.slabs) recorder->Keep(slab);
            // The image itself must outlive the recorded retile: the cache may evict it right after.
            if (auto self = weak_from_this().lock()) recorder->Keep(std::move(self));
            Recorder::CountBarriers(Recorder::CommandClass::StorageWriteBack, elided ? 2 : padding.copies.empty() ? 4 : 5);
            if (elided) Recorder::CountSkipped(Recorder::CommandClass::StorageWriteBack, padding.copies.empty() ? 2 : 3);
        } else {
            batch = std::make_unique<CommandBatch>(context);
            commands = batch->Handle();
        }
        const auto importOffset = descriptor.baseAddress - import->base;
        // The tracked layers are the array layers when there are several; one tracked layer means
        // the whole surface.
        const auto* storedLayers = trackedLayers == arrayLayers && trackedLayers > 1 ? &layers : nullptr;
        VkImageMemoryBarrier toSource{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toSource.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toSource.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSource.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSource.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toSource.image = image;
        toSource.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
        if (!elided) context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toSource);
        if (!padding.copies.empty()) {
            // The current bytes of the copied ranges no element holds go under the retile (every
            // earlier writer of the import, host stores included, first; the import-ready barrier
            // below orders the scratch writes).
            if (!elided) RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            for (const auto& [source, copies] : padding.copies) context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, source, tiledScratch->Handle(), static_cast<std::uint32_t>(copies.size()), copies.data());
        }
        const auto regions = CopyRegions(storedLayers);
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image, elided ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, linear->Handle(), static_cast<std::uint32_t>(regions.size()), regions.data());
        const auto linearRead = WholeBufferBarrier(linear->Handle(), VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        const VkMemoryBarrier importReady{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, elided ? VkAccessFlags{VK_ACCESS_TRANSFER_WRITE_BIT} : VkAccessFlags{VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT}, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, elided ? VkPipelineStageFlags{VK_PIPELINE_STAGE_TRANSFER_BIT} : VkPipelineStageFlags{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT}, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &importReady, 1, &linearRead, 0, nullptr);
        for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
            if (storedLayers != nullptr && !(*storedLayers)[layer]) continue;
            for (std::uint32_t level = 0; level < mips.size(); ++level) {
                if (!geometry.HasLayer(level, layer)) continue;
                const auto& mip = mips[level];
                detiler.Dispatch(commands, descriptor.tileMode, elementBytes, linear->Handle(), geometry.LinearLayerOffset(layer) + mip.linearOffset, tiledScratch->Handle(), geometry.GuestLayerOffset(layer) + mip.tiledOffset, mip, true, layer, geometry.thick);
            }
        }
        {
            const auto scratchDone = WholeBufferBarrier(tiledScratch->Handle(), VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &scratchDone, 0, nullptr);
            std::vector<VkBufferCopy> copies;
            for (const auto& [from, to] : keep) copies.push_back({from - descriptor.baseAddress, importOffset + (from - descriptor.baseAddress), to - from});
            if (!copies.empty()) context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, tiledScratch->Handle(), import->buffer, static_cast<std::uint32_t>(copies.size()), copies.data());
        }
        VkImageMemoryBarrier backToGeneral = toSource;
        backToGeneral.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        backToGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        backToGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        backToGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        const VkMemoryBarrier stored{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT};
        if (!elided) context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &stored, 0, nullptr, 1, &backToGeneral);
        if (batch) {
            batch->SubmitAndWait();
        } else {
            recorder->EndGpuTiming(timing, storedBytes);
            if (!elided) recorder->MarkShaderReadsCovered();
            recorder->NotePendingWrite(firstStored, static_cast<std::size_t>(lastStored - firstStored));
        }
        countStorageWriteBack(storedBytes, true);
        if (profile) Profile().storageGpu += timer.lap();
        // The guest bytes now differ from `original`; other caches of the range see the write.
        originalValid = false;
        // The keys go the same way as the texels: a fill recorded behind the retile when the
        // metadata is host-imported (no CPU wait for the title's key-writing kernels), else a CPU
        // store (APS5_CPU_DCC_KEYS=1 keeps the CPU store; see DccMetadata.hpp).
        traceKeyStore("layer write-back", descriptor, guestBytes);
        if (!IsDccClear(filledKeys)) MarkDccUncompressed(context, descriptor.dccAddress, guestBytes, DccKeyCount(descriptor, guestBytes));
        uploadedKeys = DccKeys::Uncompressed;
        for (const auto& [from, to] : keep) GuestMemory::MarkWritten(from, static_cast<std::size_t>(to - from));
        // The walk covers this surface's pages only: an adjacent image's later CPU write is stamped
        // newer than this value when its own range is collected. No CPU wrote the pages here
        // (MarkWritten made the only stamps), so the memoized collect is exact.
        settle(true);
        ++Profile().storageDirectWriteBacks;
        return;
    }
    // A surface whose memory is no longer registered (freed, or re-registered under another
    // allocation: no readable registered range contains it) has nothing to receive its results;
    // the round trip through the CPU below would store stale texels over whatever the memory
    // holds now. Its selected units are dropped instead. APS5_NO_UNREGISTERED_DROP=1 stores.
    static const bool unregisteredDrop = std::getenv("APS5_NO_UNREGISTERED_DROP") == nullptr;
    if (unregisteredDrop && !RegisteredReadableCovers(descriptor.baseAddress, static_cast<std::size_t>(guestBytes))) {
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1) < 4) std::fprintf(stderr, "[gpu] storage image 0x%llx+0x%llx: memory no longer registered; %zu pending ranges dropped\n", static_cast<unsigned long long>(descriptor.baseAddress), static_cast<unsigned long long>(guestBytes), keep.size());
        unregisteredDropped.fetch_add(1, std::memory_order_relaxed);
        originalValid = false;
        for (std::uint32_t layer = 0; layer < trackedLayers; ++layer) {
            if (layers[layer]) layerPending[layer] = false;
        }
        return;
    }
    // The store below compares against the guest bytes, read now when `original` does not hold
    // them (a GPU clear, a dropped or kept unit). Reading them says nothing about the image: a
    // stale unit (dropped, or kept for the CPU) still differs from them, so `original` vouches for
    // the content again only where it did before, or once every unit is stored.
    const bool wasValid = originalValid;
    if (!wasValid) GuestMemory::ReadCommitted(descriptor.baseAddress, original);
    DeviceBuffer linear(context, static_cast<std::size_t>(sliceLinearBytes * arrayLayers), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    DeviceBuffer tiled(context, original.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    Buffer host(context, original.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (profile) Profile().storageAlloc += timer.lap();
    // Start from the uploaded bytes so padding and untouched texels keep their guest values.
    std::memcpy(host.Bytes().data(), original.data(), original.size());
    if (profile) Profile().storageHostCopy += timer.lap();
    detiler.BeginBatch();
    CommandBatch batch(context);
    const auto commands = batch.Handle();
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    CopyBuffer(context, commands, host.Handle(), 0, tiled.Handle(), 0, original.size());
    VkImageMemoryBarrier toSource{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toSource.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toSource.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toSource.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toSource.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSource.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSource.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSource.image = image;
    toSource.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, descriptor.mipCount, 0, geometry.imageLayers};
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toSource);
    const auto regions = CopyRegions();
    context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, linear.Handle(), static_cast<std::uint32_t>(regions.size()), regions.data());
    const VkBufferMemoryBarrier toShader[] = {WholeBufferBarrier(linear.Handle(), VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT), WholeBufferBarrier(tiled.Handle(), VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)};
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 2, toShader, 0, nullptr);
    for (std::uint32_t layer = 0; layer < arrayLayers; ++layer) {
        for (std::uint32_t level = 0; level < mips.size(); ++level) {
            if (!geometry.HasLayer(level, layer)) continue;
            const auto& mip = mips[level];
            detiler.Dispatch(commands, descriptor.tileMode, elementBytes, linear.Handle(), geometry.LinearLayerOffset(layer) + mip.linearOffset, tiled.Handle(), geometry.GuestLayerOffset(layer) + mip.tiledOffset, mip, true, layer, geometry.thick);
        }
    }
    // Debug aid: APS5_DUMP_STORAGE=<hex address> saves that storage image's first mip after each of
    // its first 8 write-backs as storage_<address>_<n>.raw (u32 width, height, VkFormat, then rows).
    static const std::uint64_t dumpAddress = [] { const char* text = std::getenv("APS5_DUMP_STORAGE"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();
    static int dumps = 0;
    std::unique_ptr<Buffer> dump;
    if (dumpAddress != 0 && descriptor.baseAddress == dumpAddress && dumps < 8) {
        dump = std::make_unique<Buffer>(context, static_cast<std::size_t>(mips[0].linearSize), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        CopyBuffer(context, commands, linear.Handle(), mips[0].linearOffset, dump->Handle(), 0, mips[0].linearSize);
    }
    const auto toCopy = WholeBufferBarrier(tiled.Handle(), VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkImageMemoryBarrier backToGeneral = toSource;
    backToGeneral.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    backToGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    backToGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    backToGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &toCopy, 1, &backToGeneral);
    CopyBuffer(context, commands, tiled.Handle(), 0, host.Handle(), 0, original.size());
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    APS5_LOG_CHARS_OUT("StorageTexture writeback submit");
    batch.SubmitAndWait();
    APS5_LOG_CHARS_OUT("StorageTexture writeback done");
    countStorageWriteBack(guestBytes, false);
    if (profile) Profile().storageGpu += timer.lap();
    if (dump) {
        char name[64];
        std::snprintf(name, sizeof(name), "storage_%llx_%d.raw", static_cast<unsigned long long>(descriptor.baseAddress), dumps++);
        if (std::FILE* file = std::fopen(name, "wb")) {
            const std::uint32_t header[3] = {mips[0].pitchBytes / static_cast<std::uint32_t>(elementBytes), mips[0].height, static_cast<std::uint32_t>(storageFormat)};
            std::fwrite(header, sizeof(header), 1, file);
            std::fwrite(dump->Bytes().data(), 1, dump->Bytes().size(), file);
            std::fclose(file);
        }
    }
    // Store only the kept ranges' blocks that changed, so concurrent CPU writes to untouched texels
    // survive; `original` follows for the stored layers (the others' guest bytes are unchanged).
    const auto current = host.Bytes();
    for (const auto& [from, to] : keep) {
        const auto offset = static_cast<std::size_t>(from - descriptor.baseAddress);
        const auto length = static_cast<std::size_t>(to - from);
        GuestMemory::WriteChangedCommitted(from, current.subspan(offset, length), std::span<const std::byte>(original).subspan(offset, length));
        std::memcpy(original.data() + offset, current.data() + offset, length);
    }
    // The texels now hold the whole image, so later reads must see them rather than a fast clear
    // (the keys may be host-imported although the texels were not: then a recorded fill, else a CPU store).
    traceKeyStore("cpu write-back", descriptor, guestBytes);
    if (!IsDccClear(filledKeys)) MarkDccUncompressed(context, descriptor.dccAddress, guestBytes, DccKeyCount(descriptor, guestBytes));
    uploadedKeys = DccKeys::Uncompressed;
    // The store above is the only write to these pages, so `original` is current at a fresh
    // generation, unless blocks were kept for the CPU: then the image is stale there.
    originalValid = !skippedAny && (wasValid || std::all_of(layers.begin(), layers.end(), [](bool selected) { return selected; }));
    // As in the GPU-direct path, but the store's memcpy dirtied this surface's pages in the write
    // watch: a memoized collect would leave them for the next walk to stamp newer than this value
    // (the shared edge block included, undoing the advance), so the walk is made here and
    // consumes them.
    settle(false);
    if (profile) Profile().storageStore += timer.lap();
}

StorageTexture::~StorageTexture() {
    {
        auto& live = Live();
        std::lock_guard lock(live.mutex);
        std::erase(live.textures, this);
    }
    // Cache eviction flushes first; anything still pending here is being torn down with the device.
    {
        auto& pending = Pending();
        std::lock_guard lock(pending.mutex);
        if (dirty) {
            dirty = false;
            pending.textures.remove(this);
            BumpPendingSerial();
            static std::atomic<int> reports{0};
            if (reports.fetch_add(1) < 4) std::fprintf(stderr, "[gpu] storage image 0x%llx destroyed with GPU results pending\n", static_cast<unsigned long long>(descriptor.baseAddress));
        }
    }
    release();
}

void StorageTexture::release() noexcept {
    for (const auto& [mip, extra] : extraViews) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, extra, nullptr);
    extraViews.clear();
    for (const auto& [mip, extra] : firstLayerViews) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, extra, nullptr);
    firstLayerViews.clear();
    for (const auto& [key, atomic] : atomicViews) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, atomic, nullptr);
    atomicViews.clear();
    for (const auto& [key, uint] : uintViews) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, uint, nullptr);
    uintViews.clear();
    for (const auto& [mip, raw] : rawViews) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, raw, nullptr);
    rawViews.clear();
    for (const auto& [format, attachment] : attachmentViews) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, attachment, nullptr);
    attachmentViews.clear();
    if (proxyView) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, proxyView, nullptr);
    if (proxyStorageView) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, proxyStorageView, nullptr);
    for (const auto& [key, created] : proxyAttachmentViews) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, created, nullptr);
    proxyAttachmentViews.clear();
    for (auto* views : {&proxyStorageViews, &proxyImageViews}) {
        for (const auto& [key, created] : *views) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, created, nullptr);
        views->clear();
    }
    if (proxyImage) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, proxyImage, nullptr);
    if (proxyMemory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, proxyMemory, nullptr);
    if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
    if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
    if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
}

std::uint64_t StorageTexture::GuestBytes() const {
    return guestBytes;
}

VkImageView StorageTexture::View() const {
    return view;
}

LazyWriteBackCounts CurrentLazyWriteBackCounts() {
    LazyWriteBackCounts counts;
    counts.thickStores = lazyThickStores.load(std::memory_order_relaxed);
    counts.thickStoredBytes = lazyThickStoredBytes.load(std::memory_order_relaxed);
    counts.thickKeptBytes = lazyThickKeptBytes.load(std::memory_order_relaxed);
    counts.rowBorrows = lazyRowBorrows.load(std::memory_order_relaxed);
    counts.rowBorrowBytes = lazyRowBorrowBytes.load(std::memory_order_relaxed);
    counts.edgeUnits = lazyEdgeUnits.load(std::memory_order_relaxed);
    counts.verifyChecks = lazyVerifyChecks.load(std::memory_order_relaxed);
    counts.verifyMismatches = lazyVerifyMismatches.load(std::memory_order_relaxed);
    return counts;
}

void SetLazyWriteBackModes(std::optional<bool> lazy, std::optional<bool> verify) {
    lazyOverride.store(lazy.has_value() ? (*lazy ? 1 : 0) : -1, std::memory_order_relaxed);
    lazyVerifyOverride.store(verify.has_value() ? (*verify ? 1 : 0) : -1, std::memory_order_relaxed);
}

}
