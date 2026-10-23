#include "prx/libSceAgcDriver/Graphics/include/PipelineJournal.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "ShaderCacheDirectory.hpp"
#include "ShaderCacheVersion.hpp"
#include "ShaderDiskCache.hpp"
#include "RuntimeAbi.hpp"
#include "prx/libkernel/AppMetadata/include/AppMetadata.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <pthread.h>
#include <set>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace AgcDriver::Graphics {

namespace {

// Per thread, by the word buffer's address. An entry's pin keeps its buffer alive, so a live
// buffer at the entry's address with the entry's size is the very buffer it hashed. The modules
// are the recompiler's cached variants and specialized modules (results share their buffers),
// which outlive the entries anyway; past the bound the thread forgets them all.
struct ModuleHashEntry {
    ShaderRecompiler::SharedSpirv pin;
    std::size_t size = 0;
    SpirvContentHash hash;
};

using ModuleHashes = std::unordered_map<const std::uint32_t*, ModuleHashEntry>;

ModuleHashes& threadModuleHashes() {
    thread_local ModuleHashes known;
    return known;
}

}

SpirvContentHash ModuleContentHash(const ShaderRecompiler::SharedSpirv& spirv) {
    const auto& words = spirv.Words();
    if (words.empty()) return {};
    constexpr std::size_t bound = 16384;
    auto& known = threadModuleHashes();
    if (const auto found = known.find(words.data()); found != known.end() && found->second.size == words.size()) return found->second.hash;
    const auto hash = HashSpirvWordsTwice(words, 0x5eed5a1d00000001ull, 0x5eed5a1d00000002ull);
    if (known.size() >= bound) known.clear();
    known.insert_or_assign(words.data(), ModuleHashEntry{spirv, words.size(), hash});
    return hash;
}

PipelineIdentity StageIdentity(const ShaderRecompiler::RecompileResult& result) {
    const auto hash = ModuleContentHash(result.spirv);
    if (hash == SpirvContentHash{}) return {};
    PipelineIdentity identity{hash.low, hash.high};
    const auto mix = [&](std::uint64_t value) {
        identity.low = (identity.low ^ value) * 0x9e3779b97f4a7c15ull;
        identity.low ^= identity.low >> 29u;
        identity.high = (identity.high ^ (value + 0x7f4a7c159e3779b9ull)) * 0xc2b2ae3d27d4eb4full;
        identity.high ^= identity.high >> 31u;
    };
    mix(result.specialization.size());
    for (const auto& constant : result.specialization) mix(constant.id | (static_cast<std::uint64_t>(constant.value) << 32u));
    // The high half is the workgroup memory slice upstream keys; this tree has none, so it is zero.
    mix(result.hostSubgroupSize);
    mix(result.bindings.size());
    for (const auto& binding : result.bindings) mix(binding.descriptorSet | (static_cast<std::uint64_t>(binding.binding) << 16u) | (static_cast<std::uint64_t>(binding.kind) << 40u) | (static_cast<std::uint64_t>(binding.count) << 48u));
    if (identity.Empty()) identity.low = 1;
    return identity;
}

namespace {

static_assert(sizeof(VkPipelineColorBlendAttachmentState) == 32, "VkPipelineColorBlendAttachmentState changed: update the recipe codec");
static_assert(sizeof(VkStencilOpState) == 28, "VkStencilOpState changed: update the recipe codec");
static_assert(sizeof(VkVertexInputBindingDescription) == 12, "VkVertexInputBindingDescription changed: update the recipe codec");
static_assert(sizeof(VkVertexInputAttributeDescription) == 16, "VkVertexInputAttributeDescription changed: update the recipe codec");
static_assert(sizeof(ShaderRecompiler::MeshConfiguration) == 36, "MeshConfiguration changed: update the recipe codec");
static_assert(sizeof(ShaderRecompiler::TessellationConfiguration) == 20, "TessellationConfiguration changed: update the recipe codec");
static_assert(sizeof(ShaderRecompiler::PipelineSpecializationConstant) == 8, "PipelineSpecializationConstant changed: update the module codec");
#if defined(_WIN32) && defined(__GLIBCXX__) && defined(__x86_64__)
static_assert(sizeof(State) == 592, "State changed: update CanonicalPipelineState and the recipe codec");
static_assert(sizeof(ShaderStages) == 80, "ShaderStages changed: update the recipe codec");
#endif

constexpr std::uint32_t FileMagic = 0x4c4a5041u;
// Bumped when the recipe or module codec, CanonicalPipelineState or the store key change in a way
// the layout guards above do not catch (a field renumbered or reinterpreted at the same size). The
// header also names the recompiler's source version, so a journal of another recompiler starts over.
constexpr std::uint32_t FileFormat = 0x10001u;
constexpr std::uint32_t ModuleRecord = 1u;
constexpr std::uint32_t GraphicsRecord = 2u;
constexpr std::uint32_t ComputeRecord = 3u;
constexpr std::uint64_t RecordAlignment = 8u;

struct FileHeader {
    std::uint32_t magic;
    std::uint32_t format;
    std::uint32_t runtimeAbi;
    std::uint32_t reserved;
    std::uint64_t sourceVersion;
    std::uint64_t device;
};
static_assert(sizeof(FileHeader) == 32 && std::is_trivially_copyable_v<FileHeader>);

struct RecordHeader {
    std::uint32_t kind;
    std::uint32_t reserved;
    std::uint64_t payloadBytes;
    std::uint64_t payloadHash;
};
static_assert(sizeof(RecordHeader) == 24 && std::is_trivially_copyable_v<RecordHeader>);

class Writer {
public:
    explicit Writer(std::vector<std::byte>& out) : out(out) {}

    template<typename TValue>
    void Raw(const TValue& value) requires std::is_trivially_copyable_v<TValue> {
        const auto offset = out.size();
        out.resize(offset + sizeof(value));
        std::memcpy(out.data() + offset, &value, sizeof(value));
    }

    void Bool(bool value) { Raw<std::uint8_t>(value ? 1u : 0u); }
    void U32(std::uint32_t value) { Raw(value); }
    void U64(std::uint64_t value) { Raw(value); }

    template<typename TValue>
    void Raws(std::span<const TValue> values) requires std::is_trivially_copyable_v<TValue> {
        U64(values.size());
        if (values.empty()) return;
        const auto offset = out.size();
        out.resize(offset + values.size_bytes());
        std::memcpy(out.data() + offset, values.data(), values.size_bytes());
    }

private:
    std::vector<std::byte>& out;
};

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes(bytes) {}

    [[nodiscard]] bool Ok() const { return ok; }
    [[nodiscard]] bool Done() const { return ok && position == bytes.size(); }
    void Fail() { ok = false; }

    template<typename TValue>
    [[nodiscard]] TValue Raw() requires std::is_trivially_copyable_v<TValue> {
        TValue value{};
        if (!ok || bytes.size() - position < sizeof(value)) {
            ok = false;
            return value;
        }
        std::memcpy(&value, bytes.data() + position, sizeof(value));
        position += sizeof(value);
        return value;
    }

    [[nodiscard]] bool Bool() {
        const auto value = Raw<std::uint8_t>();
        if (value > 1u) ok = false;
        return ok && value == 1u;
    }

    [[nodiscard]] std::uint32_t U32() { return Raw<std::uint32_t>(); }
    [[nodiscard]] std::uint64_t U64() { return Raw<std::uint64_t>(); }

    template<typename TValue>
    void Raws(std::vector<TValue>& values, std::size_t limit) requires std::is_trivially_copyable_v<TValue> {
        const auto count = U64();
        if (!ok || count > limit || count > (bytes.size() - position) / sizeof(TValue)) {
            ok = false;
            values.clear();
            return;
        }
        values.resize(static_cast<std::size_t>(count));
        if (!values.empty()) std::memcpy(values.data(), bytes.data() + position, values.size() * sizeof(TValue));
        position += values.size() * sizeof(TValue);
    }

private:
    std::span<const std::byte> bytes;
    std::size_t position = 0;
    bool ok = true;
};

void encodeMesh(Writer& writer, const ShaderRecompiler::MeshConfiguration& mesh) {
    for (const auto value : {mesh.inputPrimitive, mesh.primitivesPerGroup, mesh.verticesPerGroup, mesh.maxVertices, mesh.maxPrimitives, mesh.threadsPerGroup, mesh.ldsSizeDwords, mesh.provokingVertex, mesh.esgsItemSize}) writer.U32(value);
}

void decodeMesh(Reader& reader, ShaderRecompiler::MeshConfiguration& mesh) {
    for (auto* value : {&mesh.inputPrimitive, &mesh.primitivesPerGroup, &mesh.verticesPerGroup, &mesh.maxVertices, &mesh.maxPrimitives, &mesh.threadsPerGroup, &mesh.ldsSizeDwords, &mesh.provokingVertex, &mesh.esgsItemSize}) *value = reader.U32();
}

void encodeTessellation(Writer& writer, const ShaderRecompiler::TessellationConfiguration& tessellation) {
    for (const auto value : {tessellation.inputControlPoints, tessellation.outputControlPoints, tessellation.domain, tessellation.partitioning, tessellation.outputTopology}) writer.U32(value);
}

void decodeTessellation(Reader& reader, ShaderRecompiler::TessellationConfiguration& tessellation) {
    for (auto* value : {&tessellation.inputControlPoints, &tessellation.outputControlPoints, &tessellation.domain, &tessellation.partitioning, &tessellation.outputTopology}) *value = reader.U32();
}

// Exactly the fields CanonicalPipelineState keeps.
void encodeState(Writer& writer, const State& state) {
    writer.U32(static_cast<std::uint32_t>(state.stages.path));
    writer.U32(state.stages.vertexWaveSize);
    writer.U32(state.stages.fragmentWaveSize);
    writer.Bool(state.stages.mesh.has_value());
    if (state.stages.mesh) encodeMesh(writer, *state.stages.mesh);
    writer.Bool(state.stages.tessellation.has_value());
    if (state.stages.tessellation) encodeTessellation(writer, *state.stages.tessellation);
    writer.Bool(state.depth.has_value());
    if (state.depth) writer.U32(static_cast<std::uint32_t>(state.depth->format));
    writer.Bool(state.depthTest);
    writer.Bool(state.depthWrite);
    writer.U32(static_cast<std::uint32_t>(state.depthCompare));
    writer.Bool(state.depthBoundsTest);
    writer.Bool(state.depthBias);
    writer.Raw(state.depthBiasClamp);
    writer.Bool(state.stencilTest);
    writer.Raw(state.stencilFront);
    writer.Raw(state.stencilBack);
    writer.U64(state.colors.size());
    for (const auto& color : state.colors) {
        writer.U32(static_cast<std::uint32_t>(color.format));
        writer.U32(color.exportIndex);
    }
    writer.Raws(std::span<const VkPipelineColorBlendAttachmentState>(state.blends));
    for (const auto value : state.blendConstants) writer.Raw(value);
    writer.Bool(state.hasColorTarget);
    writer.Bool(state.rectList);
    writer.U32(static_cast<std::uint32_t>(state.topology));
    writer.Bool(state.primitiveRestart);
    writer.Bool(state.negativeOneToOne);
    writer.Bool(state.depthClamp);
    writer.U32(static_cast<std::uint32_t>(state.conservativeRasterization));
    writer.U32(state.cullMode);
    writer.U32(static_cast<std::uint32_t>(state.frontFace));
    writer.U32(state.reversedBlend);
}

void decodeState(Reader& reader, State& state) {
    state = State{};
    state.stages.path = static_cast<ShaderPath>(reader.U32());
    state.stages.vertexWaveSize = reader.U32();
    state.stages.fragmentWaveSize = reader.U32();
    if (reader.Bool()) decodeMesh(reader, state.stages.mesh.emplace());
    if (reader.Bool()) decodeTessellation(reader, state.stages.tessellation.emplace());
    if (reader.Bool()) {
        DepthTarget depth{};
        depth.format = static_cast<VkFormat>(reader.U32());
        state.depth = depth;
    }
    state.depthTest = reader.Bool();
    state.depthWrite = reader.Bool();
    state.depthCompare = static_cast<VkCompareOp>(reader.U32());
    state.depthBoundsTest = reader.Bool();
    state.depthBias = reader.Bool();
    state.depthBiasClamp = reader.Raw<float>();
    state.stencilTest = reader.Bool();
    state.stencilFront = reader.Raw<VkStencilOpState>();
    state.stencilBack = reader.Raw<VkStencilOpState>();
    const auto colors = reader.U64();
    if (!reader.Ok() || colors > 8u) {
        reader.Fail();
        state = State{};
        return;
    }
    for (std::uint64_t index = 0; index < colors; ++index) {
        ColorTarget color{};
        color.format = static_cast<VkFormat>(reader.U32());
        color.exportIndex = reader.U32();
        state.colors.push_back(color);
    }
    reader.Raws(state.blends, 8u);
    for (auto& value : state.blendConstants) value = reader.Raw<float>();
    state.hasColorTarget = reader.Bool();
    state.rectList = reader.Bool();
    state.topology = static_cast<VkPrimitiveTopology>(reader.U32());
    state.primitiveRestart = reader.Bool();
    state.negativeOneToOne = reader.Bool();
    state.depthClamp = reader.Bool();
    state.conservativeRasterization = static_cast<VkConservativeRasterizationModeEXT>(reader.U32());
    state.cullMode = reader.U32();
    state.frontFace = static_cast<VkFrontFace>(reader.U32());
    state.reversedBlend = reader.U32();
}

void encodeRecipe(Writer& writer, const GraphicsPipelineRecipe& recipe) {
    writer.U32(static_cast<std::uint32_t>(recipe.attachmentLayout));
    writer.U64(recipe.stages.size());
    for (const auto& stage : recipe.stages) {
        writer.U32(static_cast<std::uint32_t>(stage.stage));
        writer.U64(stage.identity.low);
        writer.U64(stage.identity.high);
        writer.U32(stage.pushConstantOffset);
        writer.Bool(stage.generated);
    }
    writer.Raws(std::span<const VkVertexInputBindingDescription>(recipe.vertexInput.bindings));
    writer.Raws(std::span<const VkVertexInputAttributeDescription>(recipe.vertexInput.attributes));
    writer.Raws(std::span<const std::uint32_t>(recipe.layoutKey));
    encodeState(writer, recipe.state);
}

bool decodeRecipe(Reader& reader, GraphicsPipelineRecipe& recipe) {
    recipe = GraphicsPipelineRecipe{};
    recipe.attachmentLayout = static_cast<VkImageLayout>(reader.U32());
    const auto stages = reader.U64();
    if (!reader.Ok() || stages > 8u) return false;
    for (std::uint64_t index = 0; index < stages; ++index) {
        GraphicsPipelineRecipe::Stage stage{};
        stage.stage = static_cast<ShaderRecompiler::ShaderStage>(reader.U32());
        stage.identity.low = reader.U64();
        stage.identity.high = reader.U64();
        stage.pushConstantOffset = reader.U32();
        stage.generated = reader.Bool();
        recipe.stages.push_back(stage);
    }
    reader.Raws(recipe.vertexInput.bindings, 64u);
    reader.Raws(recipe.vertexInput.attributes, 64u);
    reader.Raws(recipe.layoutKey, 4u * 4096u);
    decodeState(reader, recipe.state);
    return reader.Done() && recipe.layoutKey.size() % 4u == 0;
}

std::uint64_t padded(std::uint64_t bytes) {
    return (bytes + RecordAlignment - 1u) / RecordAlignment * RecordAlignment;
}

bool profiling() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

thread_local bool replayingHere = false;

// A malformed value is reported once and the default kept: these are read on draw and dispatch
// paths, which must not fail for a typo in a tuning switch.
std::uint64_t configured(const char* name, std::uint64_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') return fallback;
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        std::fprintf(stderr, "[pipeline-journal] %s=%s is not a decimal integer; using %llu\n", name, text, static_cast<unsigned long long>(fallback));
        return fallback;
    }
    return value;
}

// The journal file's name: the title's ID (several titles commonly share one install's shader
// cache directory, and each replays only its own pipelines), or a plain name without one.
std::string journalName() {
    std::string id;
    try {
        const auto title = GetAppTitleId_nid_postfix();
        for (const char c : title.value) {
            if (c == '\0') break;
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) id += c;
        }
    } catch (...) {
        id.clear();
    }
    return id.empty() ? std::string("pipelines") : "pipelines-" + id;
}

// What the journal stores of a stage result, copied on the recording thread (the words are shared,
// not copied) and encoded on the writer thread.
ShaderRecompiler::RecompileResult trimmedModule(const ShaderRecompiler::RecompileResult& result) {
    ShaderRecompiler::RecompileResult trimmed;
    trimmed.spirv = result.spirv;
    trimmed.runtimeAbiVersion = result.runtimeAbiVersion;
    trimmed.hostSubgroupSize = result.hostSubgroupSize;
    trimmed.specialization = result.specialization;
    trimmed.pushConstants = result.pushConstants;
    trimmed.bindings.reserve(result.bindings.size());
    for (const auto& binding : result.bindings) {
        ShaderRecompiler::DescriptorBinding shape{};
        shape.kind = binding.kind;
        shape.role = binding.role;
        shape.descriptorSet = binding.descriptorSet;
        shape.binding = binding.binding;
        shape.count = binding.count;
        trimmed.bindings.push_back(std::move(shape));
    }
    return trimmed;
}

// Held while the journal is read, cut or appended: another process sharing the directory (a
// second run of the same install) takes the same lock file, so its appends and this run's never
// interleave and neither cuts a record the other is writing. Taken before the journal's mutex,
// never while holding it.
class FileLock {
public:
    explicit FileLock(const std::filesystem::path& path) {
#ifdef _WIN32
        handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) return;
        OVERLAPPED overlapped{};
        if (!LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &overlapped)) {
            CloseHandle(handle);
            handle = INVALID_HANDLE_VALUE;
        }
#else
        descriptor = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (descriptor >= 0 && ::flock(descriptor, LOCK_EX) != 0) {
            ::close(descriptor);
            descriptor = -1;
        }
#endif
    }

    ~FileLock() {
#ifdef _WIN32
        if (handle == INVALID_HANDLE_VALUE) return;
        OVERLAPPED overlapped{};
        UnlockFileEx(handle, 0, 1, 0, &overlapped);
        CloseHandle(handle);
#else
        if (descriptor < 0) return;
        ::flock(descriptor, LOCK_UN);
        ::close(descriptor);
#endif
    }

    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

private:
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int descriptor = -1;
#endif
};

// The end of the last whole record in [from, size) of the file: records another process appended
// after this one loaded the file are kept, the torn tail of a record a crashed run left is cut.
std::uint64_t validTail(const std::filesystem::path& path, std::uint64_t from, std::uint64_t size);

struct LoadedModule {
    ShaderRecompiler::ShaderStage stage;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
};

struct LoadedEntry {
    std::uint32_t kind;
    GraphicsPipelineRecipe graphics;
    PipelineIdentity compute;
    bool pushConstants = false;
    std::vector<std::uint32_t> computeLayoutKey;
};

struct LoadedJournal {
    std::map<PipelineIdentity, LoadedModule> modules;
    std::vector<LoadedEntry> entries;
    std::set<PipelineIdentity> indexModules;
    std::set<std::uint64_t> indexRecords;
    std::uint64_t validBytes = 0;
    std::uint64_t bytes = 0;
    std::uint64_t dropped = 0;
    bool headerValid = false;
};

std::uint64_t sourceVersion() {
    return ShaderRecompiler::ShaderDiskCache::BaseSourceVersion();
}

class Journal {
public:
    static Journal& Instance() {
        static auto* journal = new Journal();
        return *journal;
    }

    [[nodiscard]] bool Enabled() const { return !path.empty() && !mismatched.load(std::memory_order_relaxed); }
    [[nodiscard]] const std::filesystem::path& Path() const { return path; }

    // The journal belongs to one device per process: the first device to replay binds its
    // fingerprint, the file header names it and a file written by another device starts over. A
    // later device with another fingerprint (the window's device is the headless device's GPU, so a
    // run does not see one) turns the journal off for the rest of the run.
    bool Bind(std::uint64_t fingerprint) {
        std::lock_guard lock(mutex);
        if (!bound.load(std::memory_order_relaxed)) {
            device.store(fingerprint, std::memory_order_relaxed);
            bound.store(true, std::memory_order_release);
            return true;
        }
        if (device.load(std::memory_order_relaxed) == fingerprint) return true;
        if (!mismatched.exchange(true)) std::fprintf(stderr, "[pipeline-journal] %s belongs to another device; the journal is off for this run\n", path.string().c_str());
        return false;
    }

    // Reads the whole file, fills the dedupe index, and remembers how many bytes parsed so a torn
    // tail is cut before the next append.
    LoadedJournal Load() {
        LoadedJournal loaded;
        if (!Enabled() || !bound.load(std::memory_order_acquire)) return loaded;
        const auto started = std::chrono::steady_clock::now();
        std::vector<std::byte> file;
        bool read = false;
        {
            FileLock fileLock(lockPath);
            read = ShaderRecompiler::ReadWholeFile(path, file);
        }
        if (read) parse(file, loaded, true);
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        {
            std::lock_guard lock(mutex);
            std::error_code error;
            if (!indexed && (read || !std::filesystem::exists(path, error))) {
                indexed = true;
                validBytes = loaded.validBytes;
                headerValid = loaded.headerValid;
                writtenModules = loaded.indexModules;
                writtenRecords = loaded.indexRecords;
            }
            counters.loadedRecipes.store(loaded.entries.size(), std::memory_order_relaxed);
            counters.loadedModules.store(loaded.modules.size(), std::memory_order_relaxed);
            counters.loadedBytes.store(loaded.bytes, std::memory_order_relaxed);
            counters.droppedRecords.store(loaded.dropped, std::memory_order_relaxed);
            loadMs.store(ms, std::memory_order_relaxed);
        }
        if (loaded.bytes != 0 || loaded.dropped != 0) std::fprintf(stderr, "[pipeline-journal] loaded %zu pipelines and %zu modules (%.1f MiB) from %s in %.0f ms%s\n", loaded.entries.size(), loaded.modules.size(), static_cast<double>(loaded.bytes) / (1024.0 * 1024.0), path.string().c_str(), ms, !loaded.headerValid ? "; written by another build or device, starting over" : loaded.dropped != 0 ? "; the file's tail was unreadable and is dropped" : "");
        return loaded;
    }

    void RecordGraphics(const GraphicsPipelineRecipe& recipe, std::span<const CompiledShader> shaders) {
        if (!Enabled() || !bound.load(std::memory_order_acquire)) return;
        Pending pending;
        pending.kind = GraphicsRecord;
        for (std::size_t index = 0; index < shaders.size() && index < recipe.stages.size(); ++index) {
            Require(shaders[index].program != nullptr, "a recorded pipeline stage has no program");
            pending.modules.push_back({recipe.stages[index].identity, recipe.stages[index].stage, std::make_shared<const ShaderRecompiler::RecompileResult>(trimmedModule(*shaders[index].program))});
        }
        Writer writer(pending.payload);
        encodeRecipe(writer, recipe);
        enqueue(std::move(pending));
    }

    void RecordCompute(const ShaderRecompiler::RecompileResult& shader, bool pushConstants, std::span<const std::uint32_t> layoutKey) {
        if (!Enabled() || !bound.load(std::memory_order_acquire)) return;
        const auto identity = StageIdentity(shader);
        if (identity.Empty()) return;
        Pending pending;
        pending.kind = ComputeRecord;
        pending.modules.push_back({identity, ShaderRecompiler::ShaderStage::Compute, std::make_shared<const ShaderRecompiler::RecompileResult>(trimmedModule(shader))});
        Writer writer(pending.payload);
        writer.U64(identity.low);
        writer.U64(identity.high);
        writer.Bool(pushConstants);
        writer.Raws(layoutKey);
        enqueue(std::move(pending));
    }

    void Flush() {
        std::unique_lock lock(mutex);
        idle.wait(lock, [&] { return queue.empty() && !writing; });
    }

    PipelineJournalCounters Totals() const {
        PipelineJournalCounters totals;
        totals.loadedRecipes = counters.loadedRecipes.load(std::memory_order_relaxed);
        totals.loadedModules = counters.loadedModules.load(std::memory_order_relaxed);
        totals.loadedBytes = counters.loadedBytes.load(std::memory_order_relaxed);
        totals.droppedRecords = counters.droppedRecords.load(std::memory_order_relaxed);
        totals.loadMs = loadMs.load(std::memory_order_relaxed);
        totals.recordedGraphics = counters.recordedGraphics.load(std::memory_order_relaxed);
        totals.recordedCompute = counters.recordedCompute.load(std::memory_order_relaxed);
        totals.recordedModules = counters.recordedModules.load(std::memory_order_relaxed);
        totals.writtenBytes = counters.writtenBytes.load(std::memory_order_relaxed);
        totals.writeFailures = counters.writeFailures.load(std::memory_order_relaxed);
        return totals;
    }

    struct Counters {
        std::atomic<std::uint64_t> loadedRecipes{0}, loadedModules{0}, loadedBytes{0}, droppedRecords{0};
        std::atomic<std::uint64_t> recordedGraphics{0}, recordedCompute{0}, recordedModules{0}, writtenBytes{0}, writeFailures{0};
    };

private:
    struct PendingModule {
        PipelineIdentity identity;
        ShaderRecompiler::ShaderStage stage;
        std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
    };

    struct Pending {
        std::uint32_t kind = 0;
        std::vector<PendingModule> modules;
        std::vector<std::byte> payload;
    };

    // The journal lives next to the driver's pipeline cache file in the shader cache directory, not
    // in the shader cache's per-source-version entry directory: that directory is the recompiler's
    // (whose functions the driver may not call without a recompiler source change, which would
    // start the shader cache over). The header names the source version instead.
    Journal() {
        const char* off = std::getenv("APS5_NO_PIPELINE_JOURNAL");
        if (off != nullptr && *off != '\0' && std::strcmp(off, "0") != 0) return;
        const auto directory = ShaderRecompiler::ShaderCacheDirectory();
        if (directory.empty()) return;
        const auto name = journalName();
        path = directory / (name + ".journal");
        lockPath = directory / (name + ".journal.lock");
        maxBytes = configured("APS5_PIPELINE_JOURNAL_MAX_MIB", 256u) << 20u;
        std::thread([this] { run(); }).detach();
    }

    // Parses the file: with `decode` every record is decoded, otherwise only the index (the module
    // identities and record hashes) is built. Stops at the first record that does not parse; the
    // bytes before it stay valid.
    void parse(const std::vector<std::byte>& file, LoadedJournal& loaded, bool decode) {
        std::uint64_t valid = 0;
        bool header = false;
        std::uint64_t dropped = 0;
        const auto bytes = std::span<const std::byte>(file);
        if (bytes.size() >= sizeof(FileHeader)) {
            FileHeader fileHeader;
            std::memcpy(&fileHeader, bytes.data(), sizeof(fileHeader));
            header = fileHeader.magic == FileMagic && fileHeader.format == FileFormat && fileHeader.runtimeAbi == ShaderRecompiler::RuntimeAbi::Version && fileHeader.sourceVersion == sourceVersion() && fileHeader.device == device.load(std::memory_order_relaxed);
        }
        if (header) {
            valid = padded(sizeof(FileHeader));
            std::uint64_t cursor = valid;
            while (cursor + sizeof(RecordHeader) <= bytes.size()) {
                RecordHeader record;
                std::memcpy(&record, bytes.data() + cursor, sizeof(record));
                const auto payloadOffset = cursor + sizeof(RecordHeader);
                if (record.payloadBytes > bytes.size() - payloadOffset) break;
                const auto end = padded(payloadOffset + record.payloadBytes);
                if (end > bytes.size()) break;
                const auto payload = bytes.subspan(static_cast<std::size_t>(payloadOffset), static_cast<std::size_t>(record.payloadBytes));
                if (ShaderRecompiler::HashBytes(payload) != record.payloadHash) break;
                if (record.kind == ModuleRecord) {
                    if (payload.size() < 20u) break;
                    PipelineIdentity identity;
                    std::memcpy(&identity.low, payload.data(), sizeof(identity.low));
                    std::memcpy(&identity.high, payload.data() + 8, sizeof(identity.high));
                    std::uint32_t stage = 0;
                    std::memcpy(&stage, payload.data() + 16, sizeof(stage));
                    bool decoded = !decode;
                    if (decode && !loaded.indexModules.contains(identity)) {
                        auto result = std::make_shared<ShaderRecompiler::RecompileResult>();
                        decoded = DecodePipelineModule(payload.subspan(20), *result) && StageIdentity(*result) == identity;
                        if (decoded) loaded.modules.emplace(identity, LoadedModule{static_cast<ShaderRecompiler::ShaderStage>(stage), std::move(result)});
                        else ++dropped;
                    }
                    if (decoded) loaded.indexModules.insert(identity);
                } else if (record.kind == GraphicsRecord || record.kind == ComputeRecord) {
                    if (decode && !loaded.indexRecords.contains(record.payloadHash)) {
                        LoadedEntry entry;
                        entry.kind = record.kind;
                        bool decoded = false;
                        Reader reader(payload);
                        if (record.kind == GraphicsRecord) {
                            decoded = decodeRecipe(reader, entry.graphics);
                        } else {
                            entry.compute.low = reader.U64();
                            entry.compute.high = reader.U64();
                            entry.pushConstants = reader.Bool();
                            reader.Raws(entry.computeLayoutKey, 4u * 4096u);
                            decoded = reader.Done() && entry.computeLayoutKey.size() % 4u == 0;
                        }
                        if (decoded) loaded.entries.push_back(std::move(entry));
                        else ++dropped;
                    }
                    loaded.indexRecords.insert(record.payloadHash);
                } else {
                    break;
                }
                valid = end;
                cursor = end;
            }
        }
        if (valid < bytes.size()) ++dropped;
        loaded.validBytes = valid;
        loaded.bytes = bytes.size();
        loaded.dropped = dropped;
        loaded.headerValid = header;
    }

    // The index the writer dedupes against, replaced from the file: what it holds and where it
    // ends. Under the file lock. False when the file exists but could not be read, so nothing is
    // written until it can be (the next record tries again) and the file is never started over for
    // a read failure.
    bool indexLocked() {
        std::vector<std::byte> file;
        LoadedJournal index;
        std::error_code error;
        const bool exists = std::filesystem::exists(path, error);
        if (exists && !ShaderRecompiler::ReadWholeFile(path, file)) return false;
        if (exists) parse(file, index, false);
        std::lock_guard lock(mutex);
        indexed = true;
        validBytes = index.validBytes;
        headerValid = index.headerValid;
        writtenModules = std::move(index.indexModules);
        writtenRecords = std::move(index.indexRecords);
        return true;
    }

    bool ensureIndexed() {
        {
            std::lock_guard lock(mutex);
            if (indexed) return true;
        }
        FileLock fileLock(lockPath);
        {
            std::lock_guard lock(mutex);
            if (indexed) return true;
        }
        return indexLocked();
    }

    void enqueue(Pending pending) {
        {
            std::lock_guard lock(mutex);
            queue.push_back(std::move(pending));
        }
        wake.notify_one();
    }

    void run() {
        for (;;) {
            Pending pending;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return !queue.empty(); });
                pending = std::move(queue.front());
                queue.pop_front();
                writing = true;
            }
            try {
                write(pending);
            } catch (const std::exception& error) {
                counters.writeFailures.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<std::uint64_t> reported{0};
                if (reported.fetch_add(1) < 4) std::fprintf(stderr, "[pipeline-journal] cannot write %s: %s\n", path.string().c_str(), error.what());
            }
            pending = Pending{};
            {
                std::lock_guard lock(mutex);
                writing = false;
                if (queue.empty()) idle.notify_all();
            }
        }
    }

    // One record per module the file lacks, then the pipeline record; a record whose payload the
    // file already holds is skipped. The file is opened for appending per batch, after the torn
    // tail of a crashed run (bytes past the last parsed record) was cut.
    void write(const Pending& pending) {
        if (!ensureIndexed()) throw std::runtime_error("the journal could not be read");
        for (unsigned attempt = 0; attempt < 2; ++attempt) {
            if (append(pending)) return;
        }
        throw std::runtime_error("the journal keeps changing under this run");
    }

    // One attempt: false when the file was cut or replaced by someone else since it was indexed
    // (it is indexed again, and the batch is rebuilt against that index).
    bool append(const Pending& pending) {
        const auto recordHash = ShaderRecompiler::HashBytes(pending.payload);
        std::vector<std::byte> batch;
        std::vector<PipelineIdentity> newModules;
        {
            std::lock_guard lock(mutex);
            if (writtenRecords.contains(recordHash)) return true;
            for (const auto& module : pending.modules) {
                if (!writtenModules.contains(module.identity) && std::ranges::find(newModules, module.identity) == newModules.end()) newModules.push_back(module.identity);
            }
        }
        const auto appendRecord = [&](std::uint32_t kind, std::span<const std::byte> payload) {
            const RecordHeader header{kind, 0u, payload.size(), ShaderRecompiler::HashBytes(payload)};
            Writer writer(batch);
            writer.Raw(header);
            batch.insert(batch.end(), payload.begin(), payload.end());
            batch.resize(static_cast<std::size_t>(padded(batch.size())));
        };
        for (const auto& identity : newModules) {
            const auto module = std::ranges::find_if(pending.modules, [&](const PendingModule& candidate) { return candidate.identity == identity; });
            std::vector<std::byte> payload;
            Writer writer(payload);
            writer.U64(identity.low);
            writer.U64(identity.high);
            writer.U32(static_cast<std::uint32_t>(module->stage));
            EncodePipelineModule(*module->result, payload);
            appendRecord(ModuleRecord, payload);
        }
        appendRecord(pending.kind, pending.payload);
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        FileLock fileLock(lockPath);
        bool fresh = false;
        {
            std::lock_guard lock(mutex);
            fresh = !headerValid;
            if (!fresh) {
                const auto size = std::filesystem::file_size(path, error);
                if (error) fresh = true;
                else if (size > validBytes) {
                    const auto tail = validTail(path, validBytes, size);
                    if (tail < size) std::filesystem::resize_file(path, tail, error);
                    if (error) fresh = true;
                    else validBytes = tail;
                }
            }
            if (fresh && (!writtenModules.empty() || !writtenRecords.empty())) {
                // The file this run indexed is gone (deleted or unreadable): a fresh file holds none
                // of the modules the batch left out as written, so the batch is built again.
                writtenModules.clear();
                writtenRecords.clear();
                headerValid = false;
                return false;
            }
            if (!fresh && validBytes + batch.size() > maxBytes) {
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true)) std::fprintf(stderr, "[pipeline-journal] %s reached APS5_PIPELINE_JOURNAL_MAX_MIB; later pipelines are not recorded\n", path.string().c_str());
                counters.writeFailures.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }
        if (!fresh) {
            const auto size = std::filesystem::file_size(path, error);
            if (!error && size < validBytes) {
                if (!indexLocked()) throw std::runtime_error("the journal could not be read");
                return false;
            }
        }
        std::vector<std::byte> prefix;
        if (fresh) {
            const FileHeader header{FileMagic, FileFormat, ShaderRecompiler::RuntimeAbi::Version, 0u, sourceVersion(), device.load(std::memory_order_relaxed)};
            Writer writer(prefix);
            writer.Raw(header);
            prefix.resize(static_cast<std::size_t>(padded(prefix.size())));
        }
        std::ofstream file(path, fresh ? std::ios::binary | std::ios::trunc : std::ios::binary | std::ios::app);
        if (!file) throw std::runtime_error("cannot open the journal for writing");
        if (!prefix.empty()) file.write(reinterpret_cast<const char*>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
        file.write(reinterpret_cast<const char*>(batch.data()), static_cast<std::streamsize>(batch.size()));
        file.flush();
        if (file.fail()) throw std::runtime_error("the journal write failed");
        std::lock_guard lock(mutex);
        headerValid = true;
        validBytes = (fresh ? padded(sizeof(FileHeader)) : validBytes) + batch.size();
        for (const auto& identity : newModules) writtenModules.insert(identity);
        writtenRecords.insert(recordHash);
        counters.recordedModules.fetch_add(newModules.size(), std::memory_order_relaxed);
        if (pending.kind == GraphicsRecord) counters.recordedGraphics.fetch_add(1, std::memory_order_relaxed);
        else counters.recordedCompute.fetch_add(1, std::memory_order_relaxed);
        counters.writtenBytes.fetch_add(prefix.size() + batch.size(), std::memory_order_relaxed);
        return true;
    }

    std::filesystem::path path;
    std::filesystem::path lockPath;
    std::uint64_t maxBytes = 0;
    std::atomic<std::uint64_t> device{0};
    std::atomic<bool> bound{false};
    std::atomic<bool> mismatched{false};
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<Pending> queue;
    bool writing = false;
    bool indexed = false;
    bool headerValid = false;
    std::uint64_t validBytes = 0;
    std::set<PipelineIdentity> writtenModules;
    std::set<std::uint64_t> writtenRecords;
    Counters counters;
    std::atomic<double> loadMs{0};
};

std::uint64_t validTail(const std::filesystem::path& path, std::uint64_t from, std::uint64_t size) {
    if (size <= from) return from;
    std::ifstream file(path, std::ios::binary);
    if (!file) return from;
    file.seekg(static_cast<std::streamoff>(from));
    std::vector<std::byte> bytes(static_cast<std::size_t>(size - from));
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (static_cast<std::uint64_t>(file.gcount()) != bytes.size()) return from;
    const auto span = std::span<const std::byte>(bytes);
    std::uint64_t cursor = 0;
    std::uint64_t valid = 0;
    while (cursor + sizeof(RecordHeader) <= span.size()) {
        RecordHeader record;
        std::memcpy(&record, span.data() + cursor, sizeof(record));
        const auto payloadOffset = cursor + sizeof(RecordHeader);
        if (record.kind == 0 || record.kind > ComputeRecord || record.payloadBytes > span.size() - payloadOffset) break;
        const auto end = padded(payloadOffset + record.payloadBytes);
        if (end > span.size()) break;
        if (ShaderRecompiler::HashBytes(span.subspan(static_cast<std::size_t>(payloadOffset), static_cast<std::size_t>(record.payloadBytes))) != record.payloadHash) break;
        valid = end;
        cursor = end;
    }
    return from + valid;
}

struct ReplayCounters {
    std::atomic<std::uint64_t> graphics{0}, compute{0}, present{0}, skipped{0}, rejected{0}, failed{0};
    std::atomic<std::uint32_t> threads{0};
    std::atomic<double> ms{0};
    std::atomic<std::uint64_t> pauses{0}, deferred{0}, overlapped{0}, gameWaits{0};
    std::atomic<double> pausedMs{0}, gameWaitMs{0}, buildMs{0}, longestBuildMs{0};
};

ReplayCounters& replayCounters() {
    static ReplayCounters counters;
    return counters;
}

void addMs(std::atomic<double>& total, double ms) {
    auto value = total.load(std::memory_order_relaxed);
    while (!total.compare_exchange_weak(value, value + ms, std::memory_order_relaxed)) {
    }
}

void maxMs(std::atomic<double>& longest, double ms) {
    auto value = longest.load(std::memory_order_relaxed);
    while (value < ms && !longest.compare_exchange_weak(value, ms, std::memory_order_relaxed)) {
    }
}

std::int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A replay worker's handle to itself, for the title to raise its priority during a compile.
#ifdef _WIN32
using ThreadHandle = HANDLE;
#else
using ThreadHandle = void*;
#endif

// The priority a replay compile runs at, and the normal priority of everything else a replay
// thread does (the title's own threads run at normal priority; a starved low-priority thread
// holding a lock or a driver resource the title needs would stall it).
void setThreadPriority(ThreadHandle thread, bool raised) {
#ifdef _WIN32
    if (thread != nullptr) SetThreadPriority(thread, raised ? THREAD_PRIORITY_NORMAL : THREAD_PRIORITY_LOWEST);
#else
    static_cast<void>(thread);
    static_cast<void>(raised);
#endif
}

// The game-first gate (GamePipelineBuild). `pending` counts the title builds and waits held now,
// `lastGameNs` is when one last started or ended: a worker starts a build only once none is held
// and none was for the quiet window, so a burst of title builds (a scene load) is not interleaved
// with replay builds between its members. `workers` publishes every live worker's handle for the
// whole of its life; a title build raises them all (lock-free, so it may run under the store's or
// the compute map's mutex), and a worker lowers itself only inside a compile (ReplayCompileScope),
// re-checking `pending` after lowering. The handles are never closed (the pooled threads never
// end). The mutex and condition only wake paused workers on cancel; title threads never take them.
struct ReplayGate {
    static constexpr std::int64_t QuietNs = 250'000'000;
    static constexpr std::size_t Slots = 16;
    std::atomic<std::uint32_t> pending{0};
    std::atomic<std::int64_t> lastGameNs{0};
    std::atomic<std::uint32_t> lowered{0};
    std::array<std::atomic<ThreadHandle>, Slots> workers{};
    std::mutex mutex;
    std::condition_variable wake;

    [[nodiscard]] bool quiet() const {
        return pending.load() == 0 && nowNs() - lastGameNs.load() >= QuietNs;
    }

    // A published worker slot, or Slots when all are taken (that worker is then never lowered).
    std::size_t publish(ThreadHandle thread) {
        if (thread == ThreadHandle{}) return Slots;
        for (std::size_t slot = 0; slot < Slots; ++slot) {
            ThreadHandle empty{};
            if (workers[slot].compare_exchange_strong(empty, thread)) return slot;
        }
        return Slots;
    }

    void raiseAll() {
        for (auto& worker : workers) setThreadPriority(worker.load(), true);
    }
};

ReplayGate& replayGate() {
    static auto* gate = new ReplayGate();
    return *gate;
}

// This replay thread's published handle (null when it has none, or on other threads).
thread_local ThreadHandle replayThreadHandle{};

// The replay's threads: started on demand, never ending (like the driver's pipeline compile pool),
// with 64 MiB stacks like the driver's other pipeline compile threads (a driver's pipeline compile
// can run deep). Each publishes its handle to the gate once, for the title to raise it during a
// compile, and runs at normal priority outside ReplayCompileScope.
class ReplayWorkers {
public:
    static ReplayWorkers& Instance() {
        static auto* workers = new ReplayWorkers();
        return *workers;
    }

    // Queues `count` runs of `task`, starting threads so at least `count` exist; returns how
    // many runs were queued (fewer when threads could not be started).
    unsigned Submit(unsigned count, const std::function<void()>& task) {
        std::lock_guard lock(mutex);
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setstacksize(&attributes, std::size_t{64} << 20u);
        while (threads < count) {
            pthread_t thread;
            if (pthread_create(&thread, &attributes, &ReplayWorkers::Main, this) != 0) break;
            pthread_detach(thread);
            ++threads;
        }
        pthread_attr_destroy(&attributes);
        const auto queued = std::min(count, threads);
        for (unsigned index = 0; index < queued; ++index) tasks.push_back(task);
        ready.notify_all();
        return queued;
    }

private:
    static void* Main(void* argument) {
        auto& pool = *static_cast<ReplayWorkers*>(argument);
        ThreadHandle self{};
#ifdef _WIN32
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self, THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE, 0)) self = nullptr;
#endif
        // Published for the thread's whole (endless) life; only a published thread lowers itself
        // to compile.
        if (replayGate().publish(self) != ReplayGate::Slots) replayThreadHandle = self;
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(pool.mutex);
                pool.ready.wait(lock, [&] { return !pool.tasks.empty(); });
                task = std::move(pool.tasks.front());
                pool.tasks.pop_front();
            }
            task();
        }
        return nullptr;
    }

    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> tasks;
    unsigned threads = 0;
};

// Builds the journal's pipelines for one device on a few pooled threads: the first thread
// loads the file, every thread then takes the next entry once the title is quiet (ReplayGate). A
// pipeline the store already holds, or one the title or another thread is building, counts as
// present; an entry whose modules are missing, or whose layout key does not describe a layout, is
// rejected; an entry deferred because a title build started is tried again once the title is quiet.
class Replay : public PipelineJournalReplay {
public:
    // The context keeps the device's buffer pool: a store entry is alive for a draw only while its
    // pool is the draw's (the pool tells device instances apart), so the entries built here must
    // name it. The pool is released with the replay, before the device's teardown resets it.
    Replay(Context context, PipelineJournalHooks hooks, unsigned threads) : context(std::move(context)), hooks(std::move(hooks)), active(threads) {
        started = std::chrono::steady_clock::now();
        replayCounters().threads.store(threads, std::memory_order_relaxed);
        const auto submitted = ReplayWorkers::Instance().Submit(threads, [this] { run(); });
        // Workers that could not be started never run: the replay ends once the started ones did
        // (at once when none started).
        for (auto missing = submitted; missing < threads; ++missing) leave();
    }

    ~Replay() override {
        Cancel();
        Wait();
    }

    void Cancel() override {
        cancelled.store(true, std::memory_order_relaxed);
        auto& gate = replayGate();
        {
            std::lock_guard lock(gate.mutex);
        }
        gate.wake.notify_all();
    }

    // The workers are pooled threads that outlive the replay (a thread's exit runs its
    // thread_local destructors, which faulted on replay threads in tests): waiting is for every
    // worker to have left run().
    void Wait() override {
        std::unique_lock lock(doneMutex);
        doneCondition.wait(lock, [&] { return active.load(std::memory_order_acquire) == 0; });
    }

    [[nodiscard]] bool Done() const override { return finished.load(std::memory_order_acquire); }

private:
    // One worker left run() (or never started): the last one finishes the replay. The count
    // drops and the waiters are woken under the mutex, so a waiter that sees zero and destroys
    // the replay does so only after this thread is done with it.
    void leave() {
        std::lock_guard lock(doneMutex);
        if (active.load(std::memory_order_acquire) == 1) finish();
        active.fetch_sub(1, std::memory_order_acq_rel);
        doneCondition.notify_all();
    }

    // Blocks until the title holds no build and has been quiet for the gate's window; false once
    // the replay is cancelled. A wait that had to pause is counted (and its length).
    bool waitForTurn() {
        auto& gate = replayGate();
        if (cancelled.load(std::memory_order_relaxed)) return false;
        if (gate.quiet()) return true;
        const auto pausedAt = std::chrono::steady_clock::now();
        replayCounters().pauses.fetch_add(1, std::memory_order_relaxed);
        std::unique_lock lock(gate.mutex);
        while (!cancelled.load(std::memory_order_relaxed) && !gate.quiet()) gate.wake.wait_for(lock, std::chrono::milliseconds(25));
        lock.unlock();
        addMs(replayCounters().pausedMs, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pausedAt).count());
        return !cancelled.load(std::memory_order_relaxed);
    }

    void run() {
        std::call_once(loadOnce, [&] {
            try {
                auto journal = Journal::Instance().Load();
                const auto limit = configured("APS5_PIPELINE_JOURNAL_ENTRIES", std::max<std::uint64_t>(1u, PipelineStoreTotals().bound));
                // The newest records: what the latest runs reached (a record is written once, at
                // the run that first built its pipeline, so older ones may belong to scenes or
                // builds a run no longer reaches).
                if (journal.entries.size() > limit) journal.entries.erase(journal.entries.begin(), journal.entries.end() - static_cast<std::ptrdiff_t>(limit));
                loaded = std::move(journal);
                if (!loaded.entries.empty()) std::fprintf(stderr, "[pipeline-journal] replaying %zu pipelines on %u low-priority threads, paused while the title builds pipelines\n", loaded.entries.size(), replayCounters().threads.load(std::memory_order_relaxed));
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[pipeline-journal] cannot load %s: %s\n", Journal::Instance().Path().string().c_str(), error.what());
            }
        });
        for (;;) {
            if (!waitForTurn()) break;
            const auto index = cursor.fetch_add(1, std::memory_order_relaxed);
            if (index >= loaded.entries.size()) break;
            for (;;) {
                PipelineReplayOutcome outcome = PipelineReplayOutcome::Rejected;
                try {
                    outcome = replay(loaded.entries[index]);
                } catch (const std::exception& error) {
                    replayCounters().failed.fetch_add(1, std::memory_order_relaxed);
                    static std::atomic<std::uint64_t> reported{0};
                    if (reported.fetch_add(1) < 8) std::fprintf(stderr, "[pipeline-journal] replay failed: %s\n", error.what());
                }
                if (outcome != PipelineReplayOutcome::Deferred) break;
                replayCounters().deferred.fetch_add(1, std::memory_order_relaxed);
                if (!waitForTurn()) break;
            }
            reportProgress();
        }
        // The module hashes this thread pinned (the journal's modules, freed with the replay)
        // are released with it rather than kept by the pooled thread.
        ModuleHashes().swap(threadModuleHashes());
        leave();
    }

    // APS5_PROFILE_DRAW: the replay's progress every 10 s while it runs.
    void reportProgress() {
        if (!profiling()) return;
        const auto now = nowNs();
        auto last = lastReportNs.load(std::memory_order_relaxed);
        if (now - last < 10'000'000'000 || !lastReportNs.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
        ReportPipelineJournal();
    }

    void finish() {
        loaded = LoadedJournal{};
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        replayCounters().ms.store(ms, std::memory_order_relaxed);
        finished.store(true, std::memory_order_release);
        const auto totals = PipelineJournalTotals();
        if (totals.loadedRecipes != 0) std::fprintf(stderr, "[pipeline-journal] replay %s after %.1f s: %llu graphics and %llu compute built (%.1f s building, longest %.0f ms), %llu present, %llu rejected, %llu failed; paused %llu times (%.1f s) for the title's builds, %llu title waits on a replayed build (%.0f ms)\n",
                                                     cancelled.load(std::memory_order_relaxed) ? "cancelled" : "done", ms / 1000.0, static_cast<unsigned long long>(totals.replayedGraphics), static_cast<unsigned long long>(totals.replayedCompute), totals.replayBuildMs / 1000.0, totals.replayLongestBuildMs, static_cast<unsigned long long>(totals.replayPresent + totals.replaySkipped), static_cast<unsigned long long>(totals.replayRejected), static_cast<unsigned long long>(totals.replayFailed),
                                                     static_cast<unsigned long long>(totals.replayPauses), totals.replayPausedMs / 1000.0, static_cast<unsigned long long>(totals.gameWaitsOnReplay), totals.gameWaitOnReplayMs);
        ReportPipelineJournal();
    }

    // One entry. The thread compiles at low priority (ReplayCompileScope, around the driver call)
    // and does everything else, the locks it shares with the title included, at normal priority.
    PipelineReplayOutcome replay(const LoadedEntry& entry) {
        struct Replaying {
            Replaying() { replayingHere = true; }
            ~Replaying() { replayingHere = false; }
        } replaying;
        auto& counters = replayCounters();
        // The time of one entry's build (outcome Built), for the counters.
        struct Timed {
            std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now();
            void Done() const {
                const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - at).count();
                addMs(replayCounters().buildMs, ms);
                maxMs(replayCounters().longestBuildMs, ms);
            }
        };
        const auto count = [&](PipelineReplayOutcome outcome) {
            switch (outcome) {
                case PipelineReplayOutcome::Built: break;
                case PipelineReplayOutcome::Present: counters.present.fetch_add(1, std::memory_order_relaxed); break;
                case PipelineReplayOutcome::Rejected: counters.rejected.fetch_add(1, std::memory_order_relaxed); break;
                case PipelineReplayOutcome::Deferred: break;
            }
            return outcome;
        };
        if (entry.kind == ComputeRecord) {
            const auto module = loaded.modules.find(entry.compute);
            if (module == loaded.modules.end() || module->second.stage != ShaderRecompiler::ShaderStage::Compute || !hooks.compute) return count(PipelineReplayOutcome::Rejected);
            const Timed timed;
            const auto outcome = hooks.compute(*module->second.result, entry.pushConstants, entry.computeLayoutKey);
            if (outcome == PipelineReplayOutcome::Built) {
                counters.compute.fetch_add(1, std::memory_order_relaxed);
                timed.Done();
            }
            return count(outcome);
        }
        std::vector<CompiledShader> shaders;
        for (const auto& stage : entry.graphics.stages) {
            const auto module = loaded.modules.find(stage.identity);
            if (module == loaded.modules.end() || module->second.stage != stage.stage) return count(PipelineReplayOutcome::Rejected);
            shaders.push_back({stage.stage, module->second.result.get(), stage.pushConstantOffset});
        }
        std::vector<VkDescriptorSetLayoutBinding> description;
        if (shaders.empty() || context.descriptorCache == nullptr || !DescribeLayoutKey(entry.graphics.layoutKey, description)) return count(PipelineReplayOutcome::Rejected);
        const auto store = PipelineStoreTotals();
        if (store.cached >= store.bound) {
            counters.skipped.fetch_add(1, std::memory_order_relaxed);
            return PipelineReplayOutcome::Present;
        }
        const auto setLayout = context.descriptorCache->Layout(entry.graphics.layoutKey, description);
        const Timed timed;
        const auto outcome = ReplayCachedPipeline(context, entry.graphics.state, entry.graphics.vertexInput, entry.graphics.layoutKey, setLayout, shaders, entry.graphics.attachmentLayout);
        if (outcome == PipelineReplayOutcome::Built) {
            counters.graphics.fetch_add(1, std::memory_order_relaxed);
            timed.Done();
        }
        return count(outcome);
    }

    Context context;
    PipelineJournalHooks hooks;
    std::mutex doneMutex;
    std::condition_variable doneCondition;
    std::once_flag loadOnce;
    LoadedJournal loaded;
    std::atomic<std::size_t> cursor{0};
    std::atomic<unsigned> active{0};
    std::atomic<bool> finished{false};
    std::atomic<bool> cancelled{false};
    std::atomic<std::int64_t> lastReportNs{0};
    std::chrono::steady_clock::time_point started;
};

}

// Exactly what the store key reads (pipelineKeyInto), in the form it reads it, so configurations
// with one key give one recipe: the depth block only with a depth target, the depth bias clamp as
// whether it is used, the vertex wave size only for a mesh stage, reversed blending as whether it
// is on. Every field the Pipeline constructor reads is among them (the others only in checks that
// these forms decide alike).
State CanonicalPipelineState(const State& state) {
    State canonical{};
    canonical.stages.mesh = state.stages.mesh;
    if (state.stages.mesh) canonical.stages.vertexWaveSize = state.stages.vertexWaveSize;
    canonical.stages.tessellation = state.stages.tessellation;
    if (state.depth) {
        DepthTarget depth{};
        depth.format = state.depth->format;
        canonical.depth = depth;
        canonical.depthTest = state.depthTest;
        canonical.depthWrite = state.depthWrite;
        canonical.depthCompare = state.depthCompare;
        canonical.depthBoundsTest = state.depthBoundsTest;
        canonical.depthBias = state.depthBias;
        canonical.depthBiasClamp = state.depthBias && state.depthBiasClamp != 0.0f ? 1.0f : 0.0f;
        canonical.stencilTest = state.stencilTest;
        canonical.stencilFront = state.stencilFront;
        canonical.stencilBack = state.stencilBack;
    }
    for (const auto& color : state.colors) {
        ColorTarget canonicalColor{};
        canonicalColor.format = color.format;
        canonicalColor.exportIndex = color.exportIndex;
        canonical.colors.push_back(canonicalColor);
    }
    canonical.blends = state.blends;
    canonical.blendConstants = state.blendConstants;
    canonical.hasColorTarget = state.hasColorTarget;
    canonical.rectList = state.rectList;
    canonical.topology = state.topology;
    canonical.primitiveRestart = state.primitiveRestart;
    canonical.negativeOneToOne = state.negativeOneToOne;
    canonical.depthClamp = state.depthClamp;
    canonical.conservativeRasterization = state.conservativeRasterization;
    canonical.cullMode = state.cullMode;
    canonical.frontFace = state.frontFace;
    canonical.reversedBlend = state.reversedBlend != 0 ? 1u : 0u;
    return canonical;
}

bool GeneratedPipelineStage(const State& state, ShaderRecompiler::ShaderStage stage) {
    using Stage = ShaderRecompiler::ShaderStage;
    return state.rectList && (stage == Stage::TessellationControl || stage == Stage::TessellationEvaluation);
}

GraphicsPipelineRecipe MakeGraphicsPipelineRecipe(const State& state, const VertexInputLayout& vertexInput, std::span<const std::uint32_t> layoutKey, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) {
    GraphicsPipelineRecipe recipe;
    recipe.attachmentLayout = attachmentLayout;
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        recipe.stages.push_back({shader.stage, StageIdentity(*shader.program), shader.pushConstantOffset, GeneratedPipelineStage(state, shader.stage)});
    }
    recipe.vertexInput = vertexInput;
    recipe.layoutKey.assign(layoutKey.begin(), layoutKey.end());
    recipe.state = CanonicalPipelineState(state);
    return recipe;
}

std::vector<std::byte> EncodeGraphicsPipelineRecipe(const GraphicsPipelineRecipe& recipe) {
    std::vector<std::byte> bytes;
    Writer writer(bytes);
    encodeRecipe(writer, recipe);
    return bytes;
}

bool DecodeGraphicsPipelineRecipe(std::span<const std::byte> bytes, GraphicsPipelineRecipe& recipe) {
    Reader reader(bytes);
    return decodeRecipe(reader, recipe);
}

void EncodePipelineModule(const ShaderRecompiler::RecompileResult& result, std::vector<std::byte>& out) {
    Writer writer(out);
    writer.U32(result.runtimeAbiVersion);
    writer.U32(result.hostSubgroupSize);
    writer.Raws(std::span<const std::uint32_t>(result.spirv.Words()));
    writer.Raws(std::span<const ShaderRecompiler::PipelineSpecializationConstant>(result.specialization));
    writer.Raws(std::span<const std::byte>(result.pushConstants));
    writer.U64(result.bindings.size());
    for (const auto& binding : result.bindings) {
        writer.U32(static_cast<std::uint32_t>(binding.kind));
        writer.U32(static_cast<std::uint32_t>(binding.role));
        writer.U32(binding.descriptorSet);
        writer.U32(binding.binding);
        writer.U32(binding.count);
    }
}

bool DecodePipelineModule(std::span<const std::byte> bytes, ShaderRecompiler::RecompileResult& result) {
    result = ShaderRecompiler::RecompileResult{};
    Reader reader(bytes);
    result.runtimeAbiVersion = reader.U32();
    result.hostSubgroupSize = reader.U32();
    std::vector<std::uint32_t> words;
    reader.Raws(words, std::size_t{1} << 26u);
    reader.Raws(result.specialization, 4096u);
    reader.Raws(result.pushConstants, PipelinePushConstantBytes);
    const auto bindings = reader.U64();
    if (!reader.Ok() || bindings > 4096u) return false;
    for (std::uint64_t index = 0; index < bindings; ++index) {
        ShaderRecompiler::DescriptorBinding binding{};
        binding.kind = static_cast<ShaderRecompiler::DescriptorKind>(reader.U32());
        binding.role = static_cast<ShaderRecompiler::DescriptorRole>(reader.U32());
        binding.descriptorSet = reader.U32();
        binding.binding = reader.U32();
        binding.count = reader.U32();
        result.bindings.push_back(std::move(binding));
    }
    if (!reader.Done() || words.size() < 5 || result.runtimeAbiVersion != ShaderRecompiler::RuntimeAbi::Version) return false;
    result.spirv = std::move(words);
    result.variantId = 0;
    result.specializationId = 0;
    return true;
}

bool DescribeLayoutKey(std::span<const std::uint32_t> key, std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    bindings.clear();
    if (key.size() % 4u != 0) return false;
    for (std::size_t at = 0; at < key.size(); at += 4) bindings.push_back({key[at], static_cast<VkDescriptorType>(key[at + 1]), key[at + 2], key[at + 3], nullptr});
    return true;
}

std::uint64_t PipelineSwitch(const char* name, std::uint64_t fallback) {
    return configured(name, fallback);
}

bool PipelineJournalEnabled() {
    return Journal::Instance().Enabled();
}

bool PipelineJournalReplaying() {
    return replayingHere;
}

GamePipelineBuild::GamePipelineBuild() {
    if (replayingHere) return;
    auto& gate = replayGate();
    counted = true;
    // Sequentially consistent with ReplayCompileScope: either the worker sees this build pending
    // after lowering itself, or this build sees the worker published and raises it.
    gate.pending.fetch_add(1);
    gate.lastGameNs.store(nowNs());
    if (gate.lowered.load() != 0) replayCounters().overlapped.fetch_add(1, std::memory_order_relaxed);
    gate.raiseAll();
}

GamePipelineBuild::~GamePipelineBuild() {
    if (!counted) return;
    auto& gate = replayGate();
    gate.lastGameNs.store(nowNs());
    // A paused worker polls (it waits for the quiet window after the last build anyway), so the
    // end of a build needs no wake-up.
    gate.pending.fetch_sub(1);
}

ReplayCompileScope::ReplayCompileScope() {
    const auto self = replayThreadHandle;
    if (!replayingHere || self == ThreadHandle{}) return;
    auto& gate = replayGate();
    active = true;
    gate.lowered.fetch_add(1);
    if (gate.pending.load() != 0) return;
    setThreadPriority(self, false);
    // A title build that started before the lowering raised a thread still at normal priority:
    // raise again (one that starts after this check raises it itself).
    if (gate.pending.load() != 0) setThreadPriority(self, true);
}

ReplayCompileScope::~ReplayCompileScope() {
    if (!active) return;
    setThreadPriority(replayThreadHandle, true);
    replayGate().lowered.fetch_sub(1);
}

bool GamePipelineBuildPending() {
    return replayGate().pending.load() != 0;
}

void CountGameWaitOnReplay(double milliseconds) {
    replayCounters().gameWaits.fetch_add(1, std::memory_order_relaxed);
    addMs(replayCounters().gameWaitMs, milliseconds);
}

std::filesystem::path PipelineJournalPath() {
    return Journal::Instance().Path();
}

PipelineJournalCounters PipelineJournalTotals() {
    auto totals = Journal::Instance().Totals();
    const auto& replay = replayCounters();
    totals.replayedGraphics = replay.graphics.load(std::memory_order_relaxed);
    totals.replayedCompute = replay.compute.load(std::memory_order_relaxed);
    totals.replayPresent = replay.present.load(std::memory_order_relaxed);
    totals.replaySkipped = replay.skipped.load(std::memory_order_relaxed);
    totals.replayRejected = replay.rejected.load(std::memory_order_relaxed);
    totals.replayFailed = replay.failed.load(std::memory_order_relaxed);
    totals.replayMs = replay.ms.load(std::memory_order_relaxed);
    totals.replayThreads = replay.threads.load(std::memory_order_relaxed);
    totals.replayPauses = replay.pauses.load(std::memory_order_relaxed);
    totals.replayPausedMs = replay.pausedMs.load(std::memory_order_relaxed);
    totals.replayDeferred = replay.deferred.load(std::memory_order_relaxed);
    totals.gameBuildsOverlapped = replay.overlapped.load(std::memory_order_relaxed);
    totals.gameWaitsOnReplay = replay.gameWaits.load(std::memory_order_relaxed);
    totals.gameWaitOnReplayMs = replay.gameWaitMs.load(std::memory_order_relaxed);
    totals.replayBuildMs = replay.buildMs.load(std::memory_order_relaxed);
    totals.replayLongestBuildMs = replay.longestBuildMs.load(std::memory_order_relaxed);
    return totals;
}

void RecordGraphicsPipeline(const GraphicsPipelineRecipe& recipe, std::span<const CompiledShader> shaders) {
    Journal::Instance().RecordGraphics(recipe, shaders);
}

void RecordComputePipeline(const ShaderRecompiler::RecompileResult& shader, bool pushConstants, std::span<const std::uint32_t> layoutKey) {
    Journal::Instance().RecordCompute(shader, pushConstants, layoutKey);
}

void FlushPipelineJournal() {
    if (Journal::Instance().Enabled()) Journal::Instance().Flush();
}

void ReportPipelineJournal() {
    if (!profiling() || !Journal::Instance().Enabled()) return;
    const auto totals = PipelineJournalTotals();
    AgcDriver::ProfilePrint_nid_no_patch("[pipeline-journal] loaded %llu pipelines, %llu modules (%.1f MiB, %.0f ms); replay on %u threads: %llu graphics and %llu compute built, %llu present, %llu skipped (store full), %llu rejected, %llu failed in %.0f ms; recorded this run: %llu graphics, %llu compute, %llu modules (%.1f MiB), %llu write failures\n",
                                         static_cast<unsigned long long>(totals.loadedRecipes), static_cast<unsigned long long>(totals.loadedModules), static_cast<double>(totals.loadedBytes) / (1024.0 * 1024.0), totals.loadMs, totals.replayThreads,
                                         static_cast<unsigned long long>(totals.replayedGraphics), static_cast<unsigned long long>(totals.replayedCompute), static_cast<unsigned long long>(totals.replayPresent), static_cast<unsigned long long>(totals.replaySkipped), static_cast<unsigned long long>(totals.replayRejected), static_cast<unsigned long long>(totals.replayFailed), totals.replayMs,
                                         static_cast<unsigned long long>(totals.recordedGraphics), static_cast<unsigned long long>(totals.recordedCompute), static_cast<unsigned long long>(totals.recordedModules), static_cast<double>(totals.writtenBytes) / (1024.0 * 1024.0), static_cast<unsigned long long>(totals.writeFailures));
    AgcDriver::ProfilePrint_nid_no_patch("[pipeline-journal] game first: replay paused %llu times (%.0f ms), %llu entries deferred; %llu title builds started during a replay compile (raised it), %llu title waits on a replayed build (%.1f ms); replay building %.0f ms (longest %.1f ms)\n",
                                         static_cast<unsigned long long>(totals.replayPauses), totals.replayPausedMs, static_cast<unsigned long long>(totals.replayDeferred), static_cast<unsigned long long>(totals.gameBuildsOverlapped), static_cast<unsigned long long>(totals.gameWaitsOnReplay), totals.gameWaitOnReplayMs, totals.replayBuildMs, totals.replayLongestBuildMs);
}

std::uint64_t PipelineJournalDeviceFingerprint(const VkPhysicalDeviceProperties& properties) {
    std::array<std::uint32_t, 7> words{properties.vendorID, properties.deviceID, properties.driverVersion, 0u, 0u, 0u, 0u};
    static_assert(VK_UUID_SIZE == 16);
    std::memcpy(words.data() + 3, properties.pipelineCacheUUID, VK_UUID_SIZE);
    return HashSpirvWords(words, 0x6a6f75726e616c01ull);
}

std::shared_ptr<PipelineJournalReplay> StartPipelineJournalReplay(const Context& context, std::uint64_t deviceFingerprint, PipelineJournalHooks hooks) {
    if (!Journal::Instance().Enabled() || !Journal::Instance().Bind(deviceFingerprint)) return nullptr;
    // Few threads: the replay is background work that yields to the title (ReplayGate), not a race
    // to finish before the title's first use.
    // APS5_PRECOMPILE=1 holds the title until the replay is done, so it takes most of the cores.
    const char* precompile = std::getenv("APS5_PRECOMPILE");
    const auto cores = std::max(1u, std::thread::hardware_concurrency());
    const auto fallback = precompile != nullptr && std::strcmp(precompile, "0") != 0 ? std::clamp(cores - 2u, 2u, 32u) : 2u;
    const auto threads = static_cast<unsigned>(configured("APS5_PIPELINE_JOURNAL_THREADS", fallback));
    if (threads == 0) return nullptr;
    return std::make_shared<Replay>(context, std::move(hooks), threads);
}

}
