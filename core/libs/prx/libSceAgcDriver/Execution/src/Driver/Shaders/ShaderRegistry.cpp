#include <string>
#include <tuple>
#include <map>
#include <atomic>
#include <windows.h>
#include <cstdio>
#include <mutex>
#include <set>
#include "prx/libc/include/general/LogMacros.hpp"
#include "ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/UserDataCalls.hpp"
#include "ControlFlow/Structurizer.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Capture/FrameCapture.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "CacheKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparation.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparationScope.hpp"
#include "CompiledVariant.hpp"
#include "ShaderWarmList.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <optional>
#include <pthread.h>
#include <thread>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"

namespace AgcDriver::DriverDetail {

static std::uint64_t CaptureValidationCode(const ShaderSnapshot& snapshot) {
    static const char* directory = std::getenv("APS5_CAPTURE_VALIDATION");
    if (directory == nullptr) return 0;
    std::uint64_t hash = 14695981039346656037ull;
    const auto bytes = std::as_bytes(std::span(snapshot.code));
    for (const auto byte : bytes) hash = (hash ^ std::to_integer<unsigned char>(byte)) * 1099511628211ull;
    static std::mutex mutex;
    static std::set<std::uint64_t> captured;
    static std::size_t written = 0;
    std::lock_guard lock(mutex);
    if (captured.contains(hash)) return hash;
    if (captured.size() >= 4096 || bytes.size() > 64 * 1024 * 1024 - written) return hash;
    char name[40];
    std::snprintf(name, sizeof(name), "validation-%016llx.bin", static_cast<unsigned long long>(hash));
    std::ofstream file(std::filesystem::path(directory) / name, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.close();
    if (file) {
        captured.insert(hash);
        written += bytes.size();
    }
    return hash;
}

void ReuseShaderControlFlowValidation(ShaderSnapshot& snapshot, const ShaderSnapshot& previous) {
    if (snapshot.code != previous.code) return;
    snapshot.controlFlowValidation = previous.controlFlowValidation;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static std::atomic<std::uint64_t> replacements{0};
    if (profile && (snapshot.headerAddress != previous.headerAddress || snapshot.type != previous.type || snapshot.header != previous.header)) {
        const auto count = ++replacements;
        if (count <= 8 || count % 100 == 0)
            ProfilePrint_nid_no_patch("[control-flow] reused code=0x%llx old_header=0x%llx new_header=0x%llx replacements=%llu\n", static_cast<unsigned long long>(snapshot.codeAddress), static_cast<unsigned long long>(previous.headerAddress), static_cast<unsigned long long>(snapshot.headerAddress), static_cast<unsigned long long>(count));
    }
}

void ValidateShaderControlFlow(const ShaderSnapshot& snapshot, std::size_t codeOffset) {
    require(codeOffset < snapshot.code.size(), "control-flow entry point is outside shader snapshot");
    auto& validation = *snapshot.controlFlowValidation;
    std::lock_guard lock(validation.mutex);
    if (const auto found = validation.entries.find(codeOffset); found != validation.entries.end()) {
        if (found->second) std::rethrow_exception(found->second);
        return;
    }
    const auto codeHash = CaptureValidationCode(snapshot);
    const auto started = std::chrono::steady_clock::now();
    std::exception_ptr failure;
    try {
        const auto decoded = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(std::span(snapshot.code).subspan(codeOffset));
        if (std::ranges::none_of(decoded.instructions, [](const auto& instruction) { return instruction.op == ShaderRecompiler::RdnaOpcode::SSwappcB64; })) {
            auto graph = ShaderRecompiler::GraphBuilder{}.Build(decoded);
            ShaderRecompiler::Structurizer{}.Structurize(graph);
        }
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception&) {
        failure = std::current_exception();
    }
    validation.entries.emplace(codeOffset, failure);
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static std::atomic<std::uint64_t> validations{0};
    if (profile) {
        const auto count = ++validations;
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        if (ms >= 10.0 || count <= 8 || count % 100 == 0)
            ProfilePrint_nid_no_patch("[control-flow] validation=%llu code=0x%llx header=0x%llx offset=%zu words=%zu ms=%.3f failed=%d type=%u hash=%016llx\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(snapshot.codeAddress), static_cast<unsigned long long>(snapshot.headerAddress), codeOffset, snapshot.code.size(), ms, failure != nullptr, snapshot.type, static_cast<unsigned long long>(codeHash));
    }
    if (failure) std::rethrow_exception(failure);
}

std::shared_ptr<const ShaderSnapshot> ReadRawComputeShader(std::uint64_t address) {
    GuestMemory::CheckRange(reinterpret_cast<const void*>(address), sizeof(std::uint32_t), 256);
    static std::mutex cacheMutex;
    static std::list<std::shared_ptr<const ShaderSnapshot>> cache;
    static std::size_t cacheBytes = 0;
    std::shared_ptr<const ShaderSnapshot> cached;
    {
        std::lock_guard lock(cacheMutex);
        const auto found = std::find_if(cache.begin(), cache.end(), [address](const auto& entry) { return entry->codeAddress == address; });
        if (found != cache.end()) cached = *found;
    }
    if (cached) {
        const auto code = std::as_bytes(std::span(cached->code));
        GuestMemory::FlushGpuWrites(address, code.size());
        if (GuestMemory::CompareMapped(address, code) == GuestMemory::Compare::Equal) {
            std::lock_guard lock(cacheMutex);
            const auto found = std::find(cache.begin(), cache.end(), cached);
            if (found != cache.end()) cache.splice(cache.begin(), cache, found);
            return cached;
        }
    }
    constexpr std::size_t limit = 1024 * 1024;
    const auto ranges = GuestMemory::CommittedRanges(address, limit);
    std::uint64_t end = address;
    for (const auto& range : ranges) {
        if (range.first != end) break;
        end = range.second;
    }
    const auto available = static_cast<std::size_t>(end - address) / sizeof(std::uint32_t);
    ShaderSnapshot snapshot{address, 0, 0, {}, {}};
    while (snapshot.code.size() < available) {
        const auto previous = snapshot.code.size();
        snapshot.code.resize(std::min(available, std::max<std::size_t>(64, previous * 2)));
        GuestMemory::Read(address + previous * sizeof(std::uint32_t),
            std::as_writable_bytes(std::span(snapshot.code).subspan(previous)), alignof(std::uint32_t));
        try {
            const auto decoded = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(snapshot.code);
            const auto& last = decoded.instructions.back();
            snapshot.code.resize(last.programCounter / sizeof(std::uint32_t) + last.wordCount);
            auto result = std::make_shared<const ShaderSnapshot>(std::move(snapshot));
            std::lock_guard lock(cacheMutex);
            const auto found = std::find_if(cache.begin(), cache.end(), [address](const auto& entry) { return entry->codeAddress == address; });
            if (found != cache.end()) {
                if ((*found)->code == result->code) {
                    result = *found;
                    cache.splice(cache.begin(), cache, found);
                    return result;
                }
                cacheBytes -= (*found)->code.size() * sizeof(std::uint32_t);
                cache.erase(found);
            }
            const auto bytes = result->code.size() * sizeof(std::uint32_t);
            while (!cache.empty() && (cache.size() >= 64 || cacheBytes + bytes > 8 * 1024 * 1024)) {
                cacheBytes -= cache.back()->code.size() * sizeof(std::uint32_t);
                cache.pop_back();
            }
            cache.push_front(result);
            cacheBytes += bytes;
            return result;
        } catch (const std::out_of_range&) {
            if (snapshot.code.size() == available) break;
        }
    }
    throw std::runtime_error("AGC driver: raw compute program has no reachable end within mapped code or the size limit");
}


struct ShaderPreparationTransaction::State {
    struct Change {
        std::shared_ptr<PreparedShaders> destination;
        PreparedShaderState prepared;
        bool edited = false;
        std::size_t baseEntries = 0;
        std::size_t baseAbis = 0;
    };
    static std::mutex writers;
    static thread_local State* active;
    std::unique_lock<std::mutex> lock{writers};
    std::map<PreparedShaders*, Change> changes;
    bool failed = false;
};

std::mutex ShaderPreparationTransaction::State::writers;
thread_local ShaderPreparationTransaction::State* ShaderPreparationTransaction::State::active = nullptr;

ShaderPreparationTransaction::ShaderPreparationTransaction() {
    if (State::active == nullptr) {
        state = std::make_unique<State>();
        State::active = state.get();
    }
    root = State::active;
}

ShaderPreparationTransaction::~ShaderPreparationTransaction() {
    if (!committed) root->failed = true;
    if (state != nullptr) State::active = nullptr;
}

PreparedShaderState& ShaderPreparationTransaction::Edit(const ShaderSnapshot& snapshot) {
    require(!committed, "shader preparation transaction is already committed");
    auto& changes = root->changes;
    const auto found = changes.find(snapshot.prepared.get());
    if (found != changes.end()) {
        found->second.edited = true;
        return found->second.prepared;
    }
    std::lock_guard lock(snapshot.prepared->mutex);
    return changes.emplace(snapshot.prepared.get(), State::Change{snapshot.prepared, *snapshot.prepared, true, snapshot.prepared->entries.size(), snapshot.prepared->registeredAbis.size()}).first->second.prepared;
}

const PreparedShaderState& ShaderPreparationTransaction::Read(const ShaderSnapshot& snapshot) const {
    require(!committed, "shader preparation transaction is already committed");
    auto& changes = root->changes;
    const auto found = changes.find(snapshot.prepared.get());
    if (found != changes.end()) return found->second.prepared;
    std::lock_guard lock(snapshot.prepared->mutex);
    return changes.emplace(snapshot.prepared.get(), State::Change{snapshot.prepared, *snapshot.prepared, false, snapshot.prepared->entries.size(), snapshot.prepared->registeredAbis.size()}).first->second.prepared;
}

void ShaderPreparationTransaction::Commit() {
    require(!committed, "shader preparation transaction was committed twice");
    if (root->failed) {
        committed = true;
        return;
    }
    if (state != nullptr) {
        std::vector<std::unique_lock<std::mutex>> locks;
        locks.reserve(state->changes.size());
        for (const auto& [key, change] : state->changes) locks.emplace_back(change.destination->mutex);
        for (auto& [key, change] : state->changes) {
            if (!change.edited) continue;
            auto& destination = *change.destination;
            for (std::size_t index = std::min(change.baseEntries, destination.entries.size()); index < destination.entries.size(); ++index) {
                const auto& added = destination.entries[index];
                const auto duplicate = std::ranges::any_of(change.prepared.entries, [&](const auto& existing) { return existing.codeOffset == added.codeOffset && existing.handle == added.handle; });
                if (!duplicate) change.prepared.entries.push_back(added);
            }
            for (std::size_t index = std::min(change.baseAbis, destination.registeredAbis.size()); index < destination.registeredAbis.size(); ++index) {
                if (std::ranges::find(change.prepared.registeredAbis, destination.registeredAbis[index]) == change.prepared.registeredAbis.end()) change.prepared.registeredAbis.push_back(destination.registeredAbis[index]);
            }
            destination.entries.swap(change.prepared.entries);
            destination.registeredAbis.swap(change.prepared.registeredAbis);
            destination.graphicsAbis.swap(change.prepared.graphicsAbis);
            destination.rectangles.swap(change.prepared.rectangles);
            destination.rectangleProgress.swap(change.prepared.rectangleProgress);
            destination.fragments.swap(change.prepared.fragments);
            std::swap(destination.rectangleRequested, change.prepared.rectangleRequested);
        }
    }
    committed = true;
}

void PublishRegisteredShader(std::shared_ptr<ShaderRegistry>& registry, const std::shared_ptr<const ShaderSnapshot>& snapshot) {
    ShaderPreparationTransaction transaction;
    if (registry != nullptr) {
        const auto found = registry->find(snapshot->codeAddress);
        if (found != registry->end()) {
            const auto& current = *found->second;
            if (current.headerAddress == snapshot->headerAddress && current.type == snapshot->type && current.code == snapshot->code && current.header == snapshot->header) {
                transaction.Commit();
                return;
            }
        }
    }
    auto next = registry != nullptr ? std::make_shared<ShaderRegistry>(*registry) : std::make_shared<ShaderRegistry>();
    next->insert_or_assign(snapshot->codeAddress, snapshot);
    registry = std::move(next);
    transaction.Commit();
}

std::shared_ptr<const ShaderRecompiler::SourceHandle> PrepareAtFirstUse(const ShaderRecompiler::RecompileRequest& request) {
    static std::mutex failuresMutex;
    static std::map<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t>, std::string> failures;
    const auto key = std::make_tuple(request.shader.codeAddress, static_cast<std::uint32_t>(request.shader.stage), request.layout.pushConstantOffsetBytes, request.layout.pushConstantSizeBytes);
    {
        std::lock_guard lock(failuresMutex);
        if (const auto found = failures.find(key); found != failures.end()) throw std::runtime_error(found->second);
    }
    try {
        return ShaderRecompiler::PrepareShader(request);
    } catch (const std::exception& error) {
        std::lock_guard lock(failuresMutex);
        failures.emplace(key, error.what());
        throw;
    }
}

std::shared_ptr<const ShaderRecompiler::SourceHandle> SourceHandleFor(const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request) {
    struct PreparedKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, PreparedKeyStorage>();
    ShaderRecompiler::BuildPreparedShaderKey(request, key);
    std::lock_guard lock(snapshot.prepared->mutex);
    for (const auto& entry : snapshot.prepared->entries) {
        if (entry.codeOffset == codeOffset && ShaderRecompiler::MatchesPreparedShader(request, *entry.handle, key)) return entry.handle;
    }
    if (snapshot.header.empty() || !request.shader.capturedCalls.empty()) {
        if (snapshot.header.empty()) {
            if (snapshot.type != 0 || request.shader.stage != ShaderRecompiler::ShaderStage::Compute) throw std::runtime_error("AGC driver: unregistered program is not a compute shader");
            APS5_LOG_ERR("Compute shader 0x%llx was not registered; preparing its artifact at dispatch", static_cast<unsigned long long>(snapshot.codeAddress));
        }
        auto handle = ShaderRecompiler::PrepareShader(request);
        snapshot.prepared->entries.push_back({codeOffset, handle});
        return handle;
    }
    {
        static std::atomic<std::uint32_t> lateReports{0};
        if (lateReports.fetch_add(1) < 40) std::fprintf(stderr, "[prepare] shader 0x%llx stage %u prepared at first use for push offset %u\n", static_cast<unsigned long long>(request.shader.codeAddress), static_cast<unsigned>(request.shader.stage), request.layout.pushConstantOffsetBytes);
        auto handle = PrepareAtFirstUse(request);
        snapshot.prepared->entries.push_back({codeOffset, handle});
        return handle;
    }
    std::string layouts;
    for (const auto& entry : snapshot.prepared->entries) {
        if (entry.codeOffset != codeOffset || entry.handle == nullptr || entry.handle->artifact == nullptr) continue;
        const auto& layout = entry.handle->artifact->layout;
        layouts += " [" + std::to_string(layout.pushConstantOffsetBytes) + "," + std::to_string(layout.pushConstantSizeBytes) + "]";
    }
    throw std::runtime_error("AGC driver: prepared shader artifact is missing for the requested static ABI: address=" + std::to_string(request.shader.codeAddress) + " stage=" + std::to_string(static_cast<std::uint32_t>(request.shader.stage)) + " wave=" + std::to_string(request.context.waveSize) + " pushOffset=" + std::to_string(request.layout.pushConstantOffsetBytes) + " pushCapacity=" + std::to_string(request.layout.pushConstantSizeBytes) + " preparedLayouts=" + layouts);
}

ShaderRecompiler::RectListShaders PreparedRectangle(const ShaderSnapshot& snapshot, std::uint64_t vertexId, std::uint64_t fragmentId) {
    std::lock_guard lock(snapshot.prepared->mutex);
    for (const auto& entry : snapshot.prepared->rectangles) {
        if (entry.vertexId == vertexId && entry.fragmentId == fragmentId) {
            require(entry.shaders != nullptr, "prepared rectangle artifact is missing");
            return *entry.shaders;
        }
    }
    throw std::runtime_error("AGC driver: prepared rectangle artifacts are missing");
}

ShaderRecompiler::PreparedShaderInvocation InvocationFor(const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, ShaderMemory* memory) {
    require(codeOffset < snapshot.code.size(), "prepared shader code offset is outside the snapshot");
    const auto code = std::span(snapshot.code).subspan(codeOffset);
    require(request.shader.code.data() == code.data() && request.shader.code.size() == code.size(), "prepared invocation does not refer to registered code");
    auto invocationRequest = request;
    ShaderRecompiler::CapturedCallProgram captured;
    struct PreparedKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, PreparedKeyStorage>();
    std::lock_guard lock(snapshot.prepared->mutex);
    const bool captureCalls = std::ranges::none_of(snapshot.prepared->entries, [&](const auto& entry) { return entry.codeOffset == codeOffset; }) || std::ranges::any_of(snapshot.prepared->entries, [&](const auto& entry) {
        return entry.codeOffset == codeOffset && !ShaderRecompiler::GetPreparedCalls(*entry.handle).empty();
    });
    if (memory != nullptr && captureCalls && request.shader.stage == ShaderRecompiler::ShaderStage::Compute) {
        captured = memory->ResolveCalls(request);
        if (!captured.code.empty()) {
            invocationRequest.shader.code = captured.code;
            invocationRequest.shader.capturedCalls = captured.calls;
        }
    }
    ShaderRecompiler::BuildPreparedShaderKey(invocationRequest, key);
    for (const auto& entry : snapshot.prepared->entries) {
        if (entry.codeOffset != codeOffset) continue;
        if (auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(invocationRequest, entry.handle, key)) return std::move(*invocation);
    }
    if (snapshot.header.empty() || !captured.code.empty()) {
        if (snapshot.header.empty() && (snapshot.type != 0 || request.shader.stage != ShaderRecompiler::ShaderStage::Compute)) throw std::runtime_error("AGC driver: unregistered program is not a compute shader");
        auto handle = ShaderRecompiler::PrepareShader(invocationRequest);
        auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(invocationRequest, handle, key);
        if (!invocation.has_value()) throw std::runtime_error("AGC driver: raw compute artifact does not match its invocation");
        snapshot.prepared->entries.push_back({codeOffset, std::move(handle)});
        return std::move(*invocation);
    }
    {
        static std::atomic<std::uint32_t> lateReports{0};
        if (lateReports.fetch_add(1) < 40) std::fprintf(stderr, "[prepare] shader 0x%llx stage %u prepared at first use for push offset %u\n", static_cast<unsigned long long>(request.shader.codeAddress), static_cast<unsigned>(request.shader.stage), request.layout.pushConstantOffsetBytes);
        auto handle = PrepareAtFirstUse(request);
        invocationRequest = request;
        invocationRequest.shader.code = ShaderRecompiler::GetPreparedCode(*handle);
        auto invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(invocationRequest, handle, key);
        if (invocation.has_value()) {
            snapshot.prepared->entries.push_back({codeOffset, std::move(handle)});
            return std::move(*invocation);
        }
    }
    std::string layouts;
    for (const auto& entry : snapshot.prepared->entries) {
        if (entry.codeOffset != codeOffset || entry.handle == nullptr || entry.handle->artifact == nullptr) continue;
        const auto& layout = entry.handle->artifact->layout;
        layouts += " [" + std::to_string(layout.pushConstantOffsetBytes) + "," + std::to_string(layout.pushConstantSizeBytes) + "]";
    }
    throw std::runtime_error("AGC driver: prepared shader artifact is missing for the requested static ABI: address=" + std::to_string(request.shader.codeAddress) + " stage=" + std::to_string(static_cast<std::uint32_t>(request.shader.stage)) + " wave=" + std::to_string(request.context.waveSize) + " pushOffset=" + std::to_string(request.layout.pushConstantOffsetBytes) + " pushCapacity=" + std::to_string(request.layout.pushConstantSizeBytes) + " preparedLayouts=" + layouts);
}
namespace {

template<typename TValue>
std::vector<TValue> ReadHeaderArray(const ShaderSnapshot& snapshot, const TValue* pointer, std::size_t count) {
    if (count == 0) return {};
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (address < snapshot.headerAddress || address - snapshot.headerAddress > snapshot.header.size()) throw std::runtime_error("AGC driver: shader metadata is outside the registered header");
    const auto offset = static_cast<std::size_t>(address - snapshot.headerAddress);
    if (count > (snapshot.header.size() - offset) / sizeof(TValue)) throw std::runtime_error("AGC driver: truncated shader metadata");
    std::vector<TValue> result(count);
    std::memcpy(result.data(), snapshot.header.data() + offset, count * sizeof(TValue));
    return result;
}

Shader ReadHeader(const ShaderSnapshot& snapshot) {
    Shader header;
    std::memcpy(&header, snapshot.header.data(), sizeof(header));
    return header;
}

RegisteredShaderState DecodeRegisteredState(const ShaderSnapshot& snapshot) {
    const auto header = ReadHeader(snapshot);
    RegisteredShaderState state{{}, InitialContextRegisters(), {{0x24a, 0}, {0x24b, 0}}};
    for (const auto reg : ReadHeaderArray(snapshot, header.sh_registers, header.num_sh_registers)) {
        state.shader.insert_or_assign(reg.offset, reg.value);
    }
    for (const auto reg : ReadHeaderArray(snapshot, header.cx_registers, header.num_cx_registers)) {
        state.context.insert_or_assign(reg.offset, reg.value);
    }
    if (header.specials != nullptr) {
        const auto special = ReadHeaderArray(snapshot, header.specials, 1).front();
        state.context[special.vgt_shader_stages_en.offset] = special.vgt_shader_stages_en.value;
        state.context[special.vgt_gs_out_prim_type.offset] = special.vgt_gs_out_prim_type.value;
        state.userConfig[special.ge_cntl.offset] = special.ge_cntl.value;
        state.userConfig[special.ge_user_vgpr_en.offset] = special.ge_user_vgpr_en.value;
    }
    return state;
}

std::uint32_t RegisterValue(const Registers& registers, std::uint32_t offset) {
    const auto found = registers.find(offset);
    if (found == registers.end()) throw std::runtime_error("AGC driver: missing static shader ABI register " + std::to_string(offset));
    return found->second;
}

void BuildRegisteredAbiKey(const QueueState& state, const VulkanDevice& device, std::vector<std::uint64_t>& key) {
    key.clear();
    key.reserve(7u + state.shader.size() + state.context.size() + state.userConfig.size());
    key.insert(key.end(), {device.Serial(), ShaderRecompiler::DebugProbeActive(), ShaderRecompiler::RayTracingStrict(), ShaderRecompiler::RayTracingMiss()});
    for (const auto* registers : {&state.shader, &state.context, &state.userConfig}) {
        key.push_back(registers->size());
        for (const auto& [offset, value] : *registers) key.push_back((static_cast<std::uint64_t>(offset) << 32u) | value);
    }
}

// Registration outcomes for the console line: prepared, deferred (nothing to prepare at
// registration: a type no draw or dispatch binds, or a call target only the dispatch's user data
// names; prepared at first use if ever used) and failed.
enum class CompileOutcome { Compiled, Deferred, Failed };

void ReportCompileProgress(CompileOutcome outcome) {
    static std::atomic<std::uint32_t> compiled{0};
    static std::atomic<std::uint32_t> deferred{0};
    static std::atomic<std::uint32_t> failures{0};
    static std::atomic<std::uint64_t> lastMs{0};
    (outcome == CompileOutcome::Compiled ? compiled : outcome == CompileOutcome::Deferred ? deferred : failures).fetch_add(1, std::memory_order_relaxed);
    const auto nowMs = static_cast<std::uint64_t>(GetTickCount64());
    auto last = lastMs.load(std::memory_order_relaxed);
    if (nowMs - last < 250 || !lastMs.compare_exchange_strong(last, nowMs)) return;
    char text[128];
    const int length = std::snprintf(text, sizeof(text), "Compiling shaders: %u compiled, %u deferred to first use, %u failed", compiled.load(), deferred.load(), failures.load());
    SetConsoleTitleA(text);
    static const HANDLE console = CreateFileA("CONOUT$", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (console == INVALID_HANDLE_VALUE || length <= 0) return;
    DWORD written = 0;
    WriteFile(console, "\r", 1, &written, nullptr);
    WriteFile(console, text, static_cast<DWORD>(length), &written, nullptr);
}

// A registration-time preparation that cannot run yet rather than one that failed: a call target
// computed from data the dispatch supplies (resolved when the dispatch links its callees).
bool DeferredAtRegistration(const char* what) {
    return std::strstr(what, "is not statically resolvable") != nullptr;
}

std::shared_ptr<const ShaderRecompiler::SourceHandle> TryPrepareShader(const ShaderRecompiler::RecompileRequest& request) {
    try {
        const auto started = std::chrono::steady_clock::now();
        auto handle = ShaderRecompiler::PrepareShader(request);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (seconds > 0.2) std::fprintf(stderr, "[prepare] shader 0x%llx stage %u (%zu words) took %.2f s on thread %lu\n", static_cast<unsigned long long>(request.shader.codeAddress), static_cast<unsigned>(request.shader.stage), request.shader.code.size(), seconds, GetCurrentThreadId());
        ReportCompileProgress(CompileOutcome::Compiled);
        return handle;
    } catch (const std::exception& error) {
        ReportCompileProgress(DeferredAtRegistration(error.what()) ? CompileOutcome::Deferred : CompileOutcome::Failed);
        static std::mutex reportMutex;
        static std::set<std::uint64_t> reported;
        bool first = false;
        {
            std::lock_guard lock(reportMutex);
            first = reported.insert(request.shader.codeAddress).second;
        }
        if (first) {
            char line[320];
            const int length = std::snprintf(line, sizeof(line), "[prepare] shader 0x%llx left unprepared: %.200s\n", static_cast<unsigned long long>(request.shader.codeAddress), error.what());
            if (length > 0) std::fwrite(line, 1, std::min<std::size_t>(static_cast<std::size_t>(length), sizeof(line) - 1), stderr);
            // APS5_DUMP_SHADER_FAILURES also saves registration-time failures (shader_<address>.req)
            // for agc_shader_replay; deferred call targets are not failures.
            static const bool dumpFailures = std::getenv("APS5_DUMP_SHADER_FAILURES") != nullptr;
            if (dumpFailures && !DeferredAtRegistration(error.what())) {
                try {
                    const auto text = ShaderRecompiler::RequestSerializer{}.Serialize(request);
                    char name[64];
                    std::snprintf(name, sizeof(name), "shader_%llx.req", static_cast<unsigned long long>(request.shader.codeAddress));
                    if (std::FILE* file = std::fopen(name, "wb")) {
                        std::fwrite(text.data(), 1, text.size(), file);
                        std::fclose(file);
                    }
                } catch (const std::exception& serializeError) {
                    std::fprintf(stderr, "[prepare] could not serialize request for 0x%llx: %s\n", static_cast<unsigned long long>(request.shader.codeAddress), serializeError.what());
                }
            }
        }
        return nullptr;
    }
}

std::vector<PreparedShaders::Entry> PrepareRegisteredImpl(const ShaderSnapshot& snapshot, const VulkanDevice& device, const QueueState& state, bool registration) {
    using Stage = ShaderRecompiler::ShaderStage;
    const auto header = ReadHeader(snapshot);
    std::uint32_t programRegister;
    std::uint32_t resourceRegister;
    std::uint32_t firstUser = 0;
    Stage stage;
    switch (snapshot.type) {
    case 0: stage = Stage::Compute; programRegister = 0x20c; resourceRegister = 0x213; break;
    case 1: stage = Stage::Fragment; programRegister = 0x008; resourceRegister = 0x00b; break;
    case 2: stage = Stage::Vertex; programRegister = 0x0c8; resourceRegister = 0x08b; firstUser = 8; break;
    case 4: stage = Stage::Mesh; programRegister = 0x0c8; resourceRegister = 0x08b; break;
    // Local programs are prepared at their first draw, for its tessellation configuration (their
    // LDS layout is the hull's input).
    case 5: return {};
    case 6: stage = Stage::Mesh; programRegister = 0x088; resourceRegister = 0x08b; break;
    case 7: stage = Stage::TessellationControl; programRegister = 0x108; resourceRegister = 0x10b; break;
    default: throw std::runtime_error("AGC driver: unsupported registered shader type");
    }
    const auto high = RegisterValue(state.shader, programRegister + 1);
    if ((high & ~0xffu) != 0) throw std::runtime_error("AGC driver: invalid registered program address");
    const auto address = (static_cast<std::uint64_t>(RegisterValue(state.shader, programRegister)) << 8u) | (static_cast<std::uint64_t>(high) << 40u);
    if (address < snapshot.codeAddress || address - snapshot.codeAddress >= snapshot.code.size() * 4u) throw std::runtime_error("AGC driver: registered entry point is outside shader code");
    const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / 4u);
    const auto code = std::span(snapshot.code).subspan(codeOffset);
    ValidateShaderControlFlow(snapshot, codeOffset);
    std::optional<ShaderRecompiler::ShaderComputeStageInfo> compute;
    std::optional<ShaderRecompiler::ShaderPixelStageInfo> pixel;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex;
    std::optional<ShaderRecompiler::GraphicsCompileContext> graphics;
    std::uint32_t wave;
    const auto resources = RegisterValue(state.shader, resourceRegister);
    auto userCount = (resources >> 1u) & 0x1fu;
    if (stage != Stage::Compute) userCount |= ((resources >> 27u) & 1u) << 5u;
    if (userCount > 32) throw std::runtime_error("AGC driver: static user SGPR count exceeds the register bank");
    if (stage == Stage::Compute) {
        compute = Graphics::DecodeComputeStageInfo(state.shader, snapshot.header);
        const auto special = ReadHeaderArray(snapshot, header.specials, 1).front();
        wave = (special.dispatch_modifier & 0x8000u) != 0 ? 32u : 64u;
    } else if (stage == Stage::Fragment) {
        const auto count = RegisterValue(state.context, 0x1b6) & 0x3fu;
        if (registration && count != 0) return {};
        pixel = Graphics::DecodePixelStageInfo(state.context, {});
        wave = pixel->wave32 ? 32u : 64u;
    } else {
        const auto routing = RegisterValue(state.context, 0x2d5);
        wave = (routing & 0x00400000u) != 0 ? 32u : 64u;
        if ((routing & 0x20u) != 0 || stage == Stage::Mesh || (routing & 4u) != 0 || ((routing >> 3u) & 3u) == 1u) {
            if (registration) return {};
            auto stageState = state;
            stageState.context[0x1b6] = 0;
            const auto stages = Graphics::DecodeShaderStages(stageState);
            graphics = ShaderRecompiler::GraphicsCompileContext{0, {}, stages.mesh, stages.tessellation, {}};
            if (stages.mesh) { stage = Stage::Mesh; firstUser = 0; }
            if (stages.tessellation && snapshot.type == 2) stage = Stage::TessellationEvaluation;
        }
        if (firstUser == 0) userCount += 8;
    }
    std::vector<std::uint32_t> userData(userCount);
    if (stage != Stage::Compute && stage != Stage::Fragment && snapshot.type != 6) vertex = Graphics::DecodeVertexStageInfo(snapshot.header, snapshot.headerAddress, userData, nullptr, true);
    if (stage == Stage::Compute) {
        const auto decoded = ShaderRecompiler::RdnaInstructionDecoder{}.Decode(code);
        const ShaderRecompiler::SwappcInfo swappc{false, firstUser, userCount};
        if (!ShaderRecompiler::AnalyzeUserDataCalls(decoded, swappc).empty()) return {};
    }
    const std::array<ShaderRecompiler::MemoryRegion, 2> memory{{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}, {snapshot.headerAddress, snapshot.header}}};
    ShaderRecompiler::RecompileRequest request{{stage, address, code, snapshot.headerAddress, snapshot.header}, {wave, firstUser, userData, compute, pixel, vertex, memory}, stage == Stage::Compute ? device.ComputeTarget(wave) : device.Target(), {0, 0, 0, 128}, graphics};
    if (graphics && graphics->mesh) request.layout.pushConstantSizeBytes = ShaderRecompiler::MeshDrawPushOffsetBytes;
    std::vector<PreparedShaders::Entry> entries;
    const auto append = [&] {
        PerformanceTimer timing("Shader.PrepareArtifact");
        if (auto handle = TryPrepareShader(request)) entries.push_back({codeOffset, std::move(handle)});
    };
    append();
    if (compute) {
        request.context.compute->partialThreads = {1, 1, 1};
        append();
    } else if (pixel) {
        request.layout.pushConstantSizeBytes = 0;
        append();
    }
    return entries;
}

std::vector<PreparedShaders::Entry> PrepareRegistered(const ShaderSnapshot& snapshot, const VulkanDevice& device, const QueueState& state, bool registration) {
    // A type outside the stage table (e.g. 8) has no program registers a draw or dispatch binds:
    // nothing to prepare at registration (it still throws if ever prepared for use).
    if (registration && (snapshot.type == 3 || snapshot.type > 7)) {
        ReportCompileProgress(CompileOutcome::Deferred);
        static std::mutex unsupportedMutex;
        static std::set<std::uint32_t> told;
        std::lock_guard lock(unsupportedMutex);
        if (told.insert(snapshot.type).second) std::fprintf(stderr, "[prepare] shaders of registered type %u are not prepared at registration (no stage uses them)\n", snapshot.type);
        return {};
    }
    // Geometry (4, 6), tessellation local (5) and hull (7) programs only run in a mesh or
    // tessellation pipeline, so they are prepared with the draw's stage state at first use; their
    // headers need not carry the program address registers (e.g. the high word) the draw sets.
    if (registration && snapshot.type >= 4 && snapshot.type <= 7) {
        ReportCompileProgress(CompileOutcome::Deferred);
        return {};
    }
    // Registration prepares these at every boot: the shader warm list neither records them nor
    // counts them as first uses (a static ABI resolution, registration == false, is a first use).
    std::optional<ShaderRecompiler::WarmList::RegistrationScope> registrationScope;
    if (registration) registrationScope.emplace();
    try {
        return PrepareRegisteredImpl(snapshot, device, state, registration);
    } catch (const std::exception& error) {
        ReportCompileProgress(DeferredAtRegistration(error.what()) ? CompileOutcome::Deferred : CompileOutcome::Failed);
        static std::mutex reportMutex;
        static std::set<std::uint64_t> reported;
        bool first = false;
        {
            std::lock_guard lock(reportMutex);
            first = reported.insert(snapshot.codeAddress).second;
        }
        if (first) {
            char line[384];
            const int length = std::snprintf(line, sizeof(line), "[prepare] shader 0x%llx left unprepared: %.200s [type=%u words=%zu hash=%016llx]\n", static_cast<unsigned long long>(snapshot.codeAddress), error.what(), snapshot.type, snapshot.code.size(), static_cast<unsigned long long>(CaptureValidationCode(snapshot)));
            if (length > 0) std::fwrite(line, 1, std::min<std::size_t>(static_cast<std::size_t>(length), sizeof(line) - 1), stderr);
        }
        return {};
    }
}

}

namespace {

class RegistrationPreparePool {
public:
    static RegistrationPreparePool& Instance() {
        static auto* pool = new RegistrationPreparePool();
        return *pool;
    }

    void Submit(std::function<void()> task) {
        outstanding.fetch_add(1, std::memory_order_relaxed);
        busyAt.store(NowMs(), std::memory_order_relaxed);
        {
            std::lock_guard lock(mutex);
            tasks.push_back(std::move(task));
        }
        ready.notify_one();
    }

    // Whether no preparation is queued or running and none was for `quietMs` (the shader warm
    // list's replay yields to registration).
    bool IdleFor(std::int64_t quietMs) const {
        return outstanding.load(std::memory_order_relaxed) == 0 && NowMs() - busyAt.load(std::memory_order_relaxed) >= quietMs;
    }

private:
    static std::int64_t NowMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    RegistrationPreparePool() {
        const auto count = std::max(2u, std::thread::hardware_concurrency() / 2u);
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setstacksize(&attributes, 64u << 20u);
        for (unsigned index = 0; index < count; ++index) {
            pthread_t thread;
            if (pthread_create(&thread, &attributes, &RegistrationPreparePool::Main, this) == 0) pthread_detach(thread);
        }
        pthread_attr_destroy(&attributes);
    }

    static void* Main(void* argument) {
        auto& pool = *static_cast<RegistrationPreparePool*>(argument);
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(pool.mutex);
                pool.ready.wait(lock, [&] { return !pool.tasks.empty(); });
                task = std::move(pool.tasks.front());
                pool.tasks.pop_front();
            }
            task();
            pool.busyAt.store(NowMs(), std::memory_order_relaxed);
            pool.outstanding.fetch_sub(1, std::memory_order_relaxed);
        }
        return nullptr;
    }

    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> tasks;
    std::atomic<std::int64_t> outstanding{0};
    // Starts busy: the first registration follows the pool's creation.
    std::atomic<std::int64_t> busyAt{NowMs()};
};

bool AsyncRegistrationPrepare() {
    static const bool sync = std::getenv("APS5_SYNC_SHADER_PREPARE") != nullptr;
    return !sync;
}

// Starts the shader warm list (ShaderWarmList.hpp) at the first registration: what earlier sessions
// prepared at first use (deferred programs with their draw state, captured-call dispatches, static
// ABI resolutions, specialized modules) is replayed on a few below-normal threads whenever the
// registration pool has been idle for a second. APS5_SHADER_WARM=0 turns recording and replay off;
// APS5_SHADER_WARM_THREADS=<n> sets the replay threads (default: a sixth of the cores, 1 to 4).
void StartShaderWarmList(const VulkanDevice& device) {
    static std::once_flag once;
    std::call_once(once, [&] {
        const char* enabled = std::getenv("APS5_SHADER_WARM");
        if (enabled != nullptr && std::strcmp(enabled, "0") == 0) {
            std::fprintf(stderr, "[shader-warm] off (APS5_SHADER_WARM=0)\n");
            return;
        }
        ShaderRecompiler::WarmList::ReplayOptions options;
        const char* threads = std::getenv("APS5_SHADER_WARM_THREADS");
        const auto requested = threads != nullptr ? static_cast<unsigned>(std::strtoul(threads, nullptr, 10)) : 0u;
        // APS5_PRECOMPILE=1: the title waits at its first submission until the replay is done
        // (Driver::Submit), so the replay takes most of the cores at normal priority and never yields.
        const char* precompileValue = std::getenv("APS5_PRECOMPILE");
        const bool precompile = precompileValue != nullptr && std::strcmp(precompileValue, "0") != 0;
        const auto cores = std::max(1u, std::thread::hardware_concurrency());
        options.threads = requested != 0 ? std::min(requested, 32u) : precompile ? std::clamp(cores - 2u, 2u, 32u) : std::clamp(cores / 6u, 1u, 4u);
        if (precompile) {
            options.lowPriority = false;
            options.waitTurn = [] { return true; };
        } else {
            options.waitTurn = [] {
                while (!RegistrationPreparePool::Instance().IdleFor(1000)) std::this_thread::sleep_for(std::chrono::milliseconds(200));
                return true;
            };
        }
        try {
            options.target = device.Target();
        } catch (const std::exception&) {
            options.target.reset();
        }
        ShaderRecompiler::WarmList::Start(std::move(options));
    });
}

void PrepareRegisteredInBackground(std::shared_ptr<const ShaderSnapshot> snapshot, std::shared_ptr<VulkanDevice> device, QueueState registered) {
    RegistrationPreparePool::Instance().Submit([snapshot = std::move(snapshot), device = std::move(device), registered = std::move(registered)] {
        auto entries = PrepareRegistered(*snapshot, *device, registered, true);
        std::vector<std::uint64_t> key;
        if ((snapshot->type == 0 || snapshot->type == 1) && !entries.empty()) BuildRegisteredAbiKey(registered, *device, key);
        std::lock_guard lock(snapshot->prepared->mutex);
        for (auto& entry : entries) {
            const auto duplicate = std::ranges::any_of(snapshot->prepared->entries, [&](const auto& existing) { return existing.codeOffset == entry.codeOffset && existing.handle->artifact == entry.handle->artifact; });
            if (!duplicate) snapshot->prepared->entries.push_back(std::move(entry));
        }
        if (!key.empty() && std::ranges::find(snapshot->prepared->registeredAbis, key) == snapshot->prepared->registeredAbis.end()) snapshot->prepared->registeredAbis.push_back(std::move(key));
    });
}

}

std::vector<PreparedGraphicsStage> PrepareGraphicsStages(const DrawDecode& decoded, const ShaderRecompiler::SpirvTarget& target) {
    ShaderPreparationTransaction transaction;
    require(decoded.programs.size() == decoded.roles.size(), "graphics ABI program roles are incomplete");
    for (const auto& program : decoded.programs) require(program.snapshot != nullptr, "graphics ABI has no registered shader snapshot");
    std::vector<ShaderRecompiler::ProgramRole> expected;
    if (decoded.state.stages.path == Graphics::ShaderPath::Tessellation) expected = {ShaderRecompiler::ProgramRole::Local, ShaderRecompiler::ProgramRole::Hull, ShaderRecompiler::ProgramRole::Domain};
    else {
        expected.push_back(ShaderRecompiler::ProgramRole::Main);
        if (decoded.state.stages.path == Graphics::ShaderPath::Geometry && !decoded.programs.empty() && decoded.programs.front().snapshot->type == 4) expected.push_back(ShaderRecompiler::ProgramRole::GeometryBack);
    }
    if (!decoded.roles.empty() && decoded.roles.back() == ShaderRecompiler::ProgramRole::Fragment) expected.push_back(ShaderRecompiler::ProgramRole::Fragment);
    require(expected == decoded.roles, "graphics ABI linked programs do not match the stage routing");
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    for (std::size_t index = 0; index < decoded.programs.size(); ++index) {
        const auto& program = decoded.programs[index];
        require(program.snapshot != nullptr && program.codeOffset < program.snapshot->code.size(), "graphics ABI has an invalid registered entry point");
        const auto code = std::span(program.snapshot->code).subspan(program.codeOffset);
        require(program.binary.code.data() == code.data() && program.binary.code.size() == code.size() && program.binary.codeAddress == program.snapshot->codeAddress + program.codeOffset * sizeof(std::uint32_t), "graphics ABI entry point differs from its snapshot");
        linked.push_back({decoded.roles[index], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
        memory.insert(memory.end(), program.memory.begin(), program.memory.end());
    }
    std::vector<PreparedGraphicsStage> prepared;
    std::uint32_t pushOffset = 0;
    const auto capacity = decoded.state.stages.mesh ? ShaderRecompiler::MeshDrawPushOffsetBytes : ShaderRecompiler::LayerLimitPushBytes;
    for (std::size_t index = 0; index < decoded.programs.size(); ++index) {
        if (decoded.roles[index] == ShaderRecompiler::ProgramRole::GeometryBack) continue;
        const auto& program = decoded.programs[index];
        const bool fragment = program.binary.stage == ShaderRecompiler::ShaderStage::Fragment;
        const auto wave = fragment ? decoded.state.stages.fragmentWaveSize : decoded.state.stages.vertexWaveSize;
        std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex;
        if (!fragment) vertex = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, nullptr, true);
        ShaderRecompiler::RecompileRequest request{program.binary, {wave, program.firstUserSgpr, program.userData, {}, fragment ? std::optional(decoded.pixel) : std::nullopt, vertex, memory}, target, {0, 0, pushOffset, capacity - pushOffset}, ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, decoded.state.stages.mesh, decoded.state.stages.tessellation, {}}};
        std::vector<std::uint64_t> key;
        ShaderRecompiler::BuildPreparedShaderKey(request, key);
        std::shared_ptr<const ShaderRecompiler::SourceHandle> handle;
        for (const auto& entry : transaction.Read(*program.snapshot).entries) {
            if (entry.codeOffset == program.codeOffset && ShaderRecompiler::MatchesPreparedShader(request, *entry.handle, key)) {
                handle = entry.handle;
                break;
            }
        }
        if (handle == nullptr) {
            PerformanceTimer timing("Shader.PrepareArtifact");
            handle = TryPrepareShader(request);
            if (handle == nullptr) return {};
        }
        const auto bytes = handle->artifact->bindings.pushConstantSizeBytes;
        require(bytes <= capacity - pushOffset, "prepared graphics stages exceed the push constant block");
        prepared.push_back({program.snapshot, {program.codeOffset, handle}});
        pushOffset += bytes;
    }
    transaction.Commit();
    return prepared;
}

namespace {

class AbiTimer {
public:
    explicit AbiTimer(int kind) : kind(kind), started(std::chrono::steady_clock::now()) {}
    ~AbiTimer() {
        static std::mutex mutex;
        static std::array<std::uint64_t, 2> calls{};
        static std::array<double, 2> seconds{};
        static std::array<double, 2> longest{};
        static auto window = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<double>(now - started).count();
        std::lock_guard lock(mutex);
        ++calls[kind];
        seconds[kind] += elapsed;
        longest[kind] = std::max(longest[kind], elapsed);
        if (now - window < std::chrono::seconds(1)) return;
        window = now;
        char line[256];
        const int length = std::snprintf(line, sizeof(line), "[abi] resolve shader %llu calls %.2f s (max %.3f), graphics stages %llu calls %.2f s (max %.3f)\n", static_cast<unsigned long long>(calls[0]), seconds[0], longest[0], static_cast<unsigned long long>(calls[1]), seconds[1], longest[1]);
        if (length > 0) std::fwrite(line, 1, std::min<std::size_t>(static_cast<std::size_t>(length), sizeof(line) - 1), stderr);
        calls = {};
        seconds = {};
        longest = {};
    }

private:
    int kind;
    std::chrono::steady_clock::time_point started;
};

}

void Driver::ResolveGraphicsStagesAbi(std::span<const Shader* const> stages, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    AbiTimer abiTimer(1);
    PerformanceContext timingContext(FrameTiming::Preparation());
    PerformanceTimer timing("Shader.ResolveGraphicsStagesAbi");
    ShaderPreparationTransaction transaction;
    CheckFailure();
    require(!stages.empty(), "graphics ABI has no shader headers");
    QueueState state{};
    std::shared_ptr<const ShaderRegistry> registry;
    std::shared_ptr<const ShaderSnapshot> owner;
    {
        std::lock_guard lock(mutex);
        registry = shaders;
        for (const auto* shader : stages) {
            GuestMemory::CheckRange(shader, sizeof(Shader), 4);
            const auto address = reinterpret_cast<std::uintptr_t>(const_cast<const void*>(shader->code));
            require(registry != nullptr && registry->contains(address), "graphics ABI refers to an unregistered shader");
            if (owner == nullptr) owner = registry->at(address);
            const auto& snapshot = *registry->at(address);
            static_cast<void>(snapshot.headerAddress == reinterpret_cast<std::uintptr_t>(shader));
            require(snapshot.registeredState != nullptr, "registered shader state is missing");
            const auto& registered = *snapshot.registeredState;
            for (const auto& [offset, value] : registered.shader) state.shader.insert_or_assign(offset, value);
            for (const auto& [offset, value] : registered.context) state.context.insert_or_assign(offset, value);
            for (const auto& [offset, value] : registered.userConfig) state.userConfig.insert_or_assign(offset, value);
        }
    }
    for (const auto reg : context) state.context.insert_or_assign(reg.offset, reg.value);
    for (const auto reg : primitive) state.userConfig.insert_or_assign(reg.offset, reg.value);
    const auto localDevice = device.Load();
    require(localDevice != nullptr, "graphics ABI device is missing");
    struct GraphicsAbiKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, GraphicsAbiKeyStorage>();
    BuildRegisteredAbiKey(state, *localDevice, key);
    const auto primitiveType = state.userConfig.find(0x242u);
    for (const auto& abi : transaction.Read(*owner).graphicsAbis) {
        if (abi.key != key || abi.registry.lock() != registry) continue;
        for (const auto& stage : abi.stages) {
            const auto snapshot = stage.lock();
            require(snapshot != nullptr, "prepared graphics ABI lost a registered stage");
            if (snapshot->type != 1) ResolvePreparedGraphics(*snapshot, {}, primitiveType == state.userConfig.end() ? 0u : primitiveType->second, localDevice->Target());
        }
        transaction.Commit();
        return;
    }
    DrawDecode decoded{};
    decoded.state.stages = Graphics::DecodeShaderStages(state);
    DecodeGraphicsPrograms(decoded, state, *registry, true, false);
    auto prepared = PrepareGraphicsStages(decoded, localDevice->Target());
    for (auto& stage : prepared) {
        const auto& entries = transaction.Read(*stage.snapshot).entries;
        const auto duplicate = std::ranges::any_of(entries, [&](const auto& existing) { return existing.codeOffset == stage.entry.codeOffset && existing.handle->artifact == stage.entry.handle->artifact; });
        if (!duplicate) transaction.Edit(*stage.snapshot).entries.push_back(std::move(stage.entry));
    }
    for (const auto& stage : prepared) {
        if (stage.snapshot->type == 1) continue;
        ResolvePreparedGraphics(*stage.snapshot, {}, primitiveType == state.userConfig.end() ? 0u : primitiveType->second, localDevice->Target());
    }
    PreparedShaderState::GraphicsAbi abi{key, registry, {}};
    for (const auto& stage : prepared) abi.stages.push_back(stage.snapshot);
    auto& graphicsAbis = transaction.Edit(*owner).graphicsAbis;
    std::erase_if(graphicsAbis, [](const auto& entry) { return entry.registry.expired(); });
    graphicsAbis.push_back(std::move(abi));
    transaction.Commit();
}

void Driver::ResolveShaderAbi(const Shader* shader, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    AbiTimer abiTimer(0);
    PerformanceContext timingContext(FrameTiming::Preparation());
    PerformanceTimer timing("Shader.ResolveAbi");
    ShaderPreparationTransaction transaction;
    CheckFailure();
    GuestMemory::CheckRange(shader, sizeof(Shader), 4);
    if (shader->type != 0 && shader->type != 1) {
        const std::array<const Shader*, 1> stages{shader};
        ResolveGraphicsStagesAbi(stages, context, primitive);
        transaction.Commit();
        return;
    }
    std::shared_ptr<const ShaderSnapshot> snapshot;
    {
        std::lock_guard lock(mutex);
        const auto address = reinterpret_cast<std::uintptr_t>(const_cast<const void*>(shader->code));
        require(shaders != nullptr && shaders->contains(address), "static ABI refers to an unregistered shader");
        snapshot = shaders->at(address);
        static_cast<void>(snapshot->headerAddress == reinterpret_cast<std::uintptr_t>(shader));
    }
    require(snapshot->registeredState != nullptr, "registered shader state is missing");
    QueueState state{};
    state.shader = snapshot->registeredState->shader;
    state.context = snapshot->registeredState->context;
    state.userConfig = snapshot->registeredState->userConfig;
    for (const auto reg : context) state.context[reg.offset] = reg.value;
    for (const auto reg : primitive) state.userConfig[reg.offset] = reg.value;
    const auto localDevice = device.Load();
    require(localDevice != nullptr, "shader registration device is missing");
    struct RegisteredAbiKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uint64_t>, RegisteredAbiKeyStorage>();
    BuildRegisteredAbiKey(state, *localDevice, key);
    const auto& registeredAbis = transaction.Read(*snapshot).registeredAbis;
    if (std::ranges::find(registeredAbis, key) != registeredAbis.end()) {
        transaction.Commit();
        return;
    }
    if (AsyncRegistrationPrepare()) {
        {
            std::lock_guard lock(snapshot->prepared->mutex);
            if (std::ranges::find(snapshot->prepared->registeredAbis, key) != snapshot->prepared->registeredAbis.end() || std::ranges::find(snapshot->prepared->pendingAbis, key) != snapshot->prepared->pendingAbis.end()) {
                transaction.Commit();
                return;
            }
            snapshot->prepared->pendingAbis.push_back(key);
        }
        RegistrationPreparePool::Instance().Submit([snapshot, device = localDevice, state, abi = key] {
            auto entries = PrepareRegistered(*snapshot, *device, state, false);
            std::lock_guard lock(snapshot->prepared->mutex);
            for (auto& entry : entries) {
                const auto duplicate = std::ranges::any_of(snapshot->prepared->entries, [&](const auto& existing) { return existing.handle->artifact == entry.handle->artifact; });
                if (!duplicate) snapshot->prepared->entries.push_back(std::move(entry));
            }
            if (std::ranges::find(snapshot->prepared->registeredAbis, abi) == snapshot->prepared->registeredAbis.end()) snapshot->prepared->registeredAbis.push_back(abi);
            std::erase(snapshot->prepared->pendingAbis, abi);
        });
        transaction.Commit();
        return;
    }
    auto entries = PrepareRegistered(*snapshot, *localDevice, state, false);
    auto& prepared = transaction.Edit(*snapshot);
    for (auto& entry : entries) {
        const auto duplicate = std::ranges::any_of(prepared.entries, [&](const auto& existing) { return existing.handle->artifact == entry.handle->artifact; });
        if (!duplicate) prepared.entries.push_back(std::move(entry));
    }
    prepared.registeredAbis.push_back(key);
    transaction.Commit();
}

alignas(256) static const std::uint32_t NullPixelCode[64] = {0xbf810000u};
static const Shader NullPixelShader = [] {
    Shader shader{};
    shader.file_header = 0x34333231u;
    shader.version = 0x18u;
    shader.code = NullPixelCode;
    shader.header_size = sizeof(Shader);
    shader.shader_size = sizeof(NullPixelCode);
    shader.type = 1;
    return shader;
}();

std::uint64_t NullPixelProgramAddress() {
    return reinterpret_cast<std::uintptr_t>(NullPixelCode);
}

// The registered state of the driver's null pixel program at the snapshot's address (a replayed
// capture restores the capturing process's null program next to its own).
RegisteredShaderState NullPixelRegisteredState(const ShaderSnapshot& snapshot) {
    auto state = DecodeRegisteredState(snapshot);
    state.shader.emplace(0x008u, static_cast<std::uint32_t>(snapshot.codeAddress >> 8u));
    state.shader.emplace(0x009u, static_cast<std::uint32_t>(snapshot.codeAddress >> 40u));
    state.shader.emplace(0x00bu, 0u);
    state.context.emplace(0x1b3u, 0x2u);
    state.context.emplace(0x1b4u, 0x2u);
    return state;
}

std::uint64_t SnapshotCodeHash(const ShaderSnapshot&, std::size_t) {
    return 0;
}

void Driver::ResolveGraphicsAbi(const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType) {
    PerformanceContext timingContext(FrameTiming::Preparation());
    PerformanceTimer timing("Shader.ResolveGraphicsAbi");
    CheckFailure();
    if (primitiveType != 0 && primitiveType != 7 && primitiveType != 17) return;
    ShaderPreparationTransaction transaction;
    require(vertex != nullptr && pixel != nullptr, "rectangle ABI requires vertex and fragment shaders");
    std::shared_ptr<const ShaderSnapshot> front;
    std::shared_ptr<const ShaderSnapshot> fragment;
    {
        std::lock_guard lock(mutex);
        GuestMemory::CheckRange(vertex, sizeof(Shader), 4);
        GuestMemory::CheckRange(pixel, sizeof(Shader), 4);
        const auto lookup = [&](const Shader* shader) {
            const auto address = reinterpret_cast<std::uintptr_t>(const_cast<const void*>(shader->code));
            require(shaders != nullptr && shaders->contains(address), "rectangle ABI refers to an unregistered shader");
            const auto snapshot = shaders->at(address);
            static_cast<void>(snapshot->headerAddress == reinterpret_cast<std::uintptr_t>(shader));
            return snapshot;
        };
        front = lookup(vertex);
        fragment = lookup(pixel);
    }
    require(front != fragment, "rectangle stages refer to the same shader");
    const auto localDevice = device.Load();
    require(localDevice != nullptr, "shader registration device is missing");
    ResolvePreparedGraphics(*front, fragment, primitiveType, localDevice->Target());
    transaction.Commit();
}

void ResolvePreparedGraphics(const ShaderSnapshot& front, const std::shared_ptr<const ShaderSnapshot>& fragment, std::uint32_t primitiveType, const ShaderRecompiler::SpirvTarget& target) {
    ShaderPreparationTransaction transaction;
    require(fragment.get() != &front, "rectangle stages refer to the same shader");
    std::vector<std::shared_ptr<const ShaderSnapshot>> fragments;
    {
        const auto& current = transaction.Read(front);
        const bool newFragment = fragment && !std::ranges::any_of(current.fragments, [&](const auto& entry) { return entry.lock() == fragment; });
        const bool newRequest = !current.rectangleRequested && (primitiveType == 7 || primitiveType == 17);
        const bool expiredFragment = std::ranges::any_of(current.fragments, [](const auto& entry) { return entry.expired(); });
        if (newFragment || newRequest || expiredFragment) {
            auto& prepared = transaction.Edit(front);
            std::erase_if(prepared.fragments, [](const auto& entry) { return entry.expired(); });
            std::erase_if(prepared.rectangleProgress, [](const auto& entry) { return entry.fragment.expired(); });
            if (newFragment) prepared.fragments.push_back(fragment);
            prepared.rectangleRequested |= newRequest;
        }
        const auto& prepared = transaction.Read(front);
        if (!prepared.rectangleRequested) {
            transaction.Commit();
            return;
        }
        for (const auto& entry : prepared.fragments) {
            if (auto snapshot = entry.lock()) fragments.push_back(std::move(snapshot));
        }
    }
    for (const auto& pixel : fragments) {
        const auto& current = transaction.Read(front);
        const auto& pixelPrepared = transaction.Read(*pixel);
        require(!current.entries.empty() && !pixelPrepared.entries.empty(), "rectangle ABI has missing stage artifacts");
        const auto progress = std::ranges::find_if(current.rectangleProgress, [&](const auto& entry) { return entry.fragment.lock() == pixel; });
        const auto vertexCount = progress == current.rectangleProgress.end() ? 0u : progress->vertexCount;
        const auto fragmentCount = progress == current.rectangleProgress.end() ? 0u : progress->fragmentCount;
        if (vertexCount == current.entries.size() && fragmentCount == pixelPrepared.entries.size()) continue;
        auto& prepared = transaction.Edit(front);
        std::vector<PreparedShaders::Rectangle> rectangles;
        for (std::size_t vertexIndex = 0; vertexIndex < prepared.entries.size(); ++vertexIndex) {
            const auto& vertexEntry = prepared.entries[vertexIndex];
            for (std::size_t fragmentIndex = vertexIndex < vertexCount ? fragmentCount : 0u; fragmentIndex < pixelPrepared.entries.size(); ++fragmentIndex) {
                const auto& fragmentEntry = pixelPrepared.entries[fragmentIndex];
                const auto& vertexArtifact = ShaderRecompiler::GetPreparedArtifact(*vertexEntry.handle);
                const auto& fragmentArtifact = ShaderRecompiler::GetPreparedArtifact(*fragmentEntry.handle);
                const auto exists = [&](const auto& entry) { return entry.vertexId == vertexArtifact.variantId && entry.fragmentId == fragmentArtifact.variantId; };
                if (std::ranges::any_of(prepared.rectangles, exists) || std::ranges::any_of(rectangles, exists)) continue;
                ShaderRecompiler::RecompileResult vertexResult;
                ShaderRecompiler::RecompileResult fragmentResult;
                static_cast<ShaderRecompiler::CompiledShaderArtifact&>(vertexResult) = vertexArtifact;
                static_cast<ShaderRecompiler::CompiledShaderArtifact&>(fragmentResult) = fragmentArtifact;
                rectangles.push_back({vertexArtifact.variantId, fragmentArtifact.variantId, std::make_shared<const ShaderRecompiler::RectListShaders>(ShaderRecompiler::BuildRectListShaders(vertexResult, fragmentResult, target))});
            }
        }
        prepared.rectangles.insert(prepared.rectangles.end(), std::make_move_iterator(rectangles.begin()), std::make_move_iterator(rectangles.end()));
        std::erase_if(prepared.rectangleProgress, [&](const auto& entry) { return entry.fragment.expired() || entry.fragment.lock() == pixel; });
        prepared.rectangleProgress.push_back({pixel, prepared.entries.size(), pixelPrepared.entries.size()});
    }
    transaction.Commit();
}

void Driver::RegisterShader(const Shader* shader) {
    PerformanceContext timingContext(FrameTiming::Preparation());
    PerformanceTimer timing("Shader.Register");
    ShaderPreparationTransaction transaction;
    CheckFailure();
    const auto entered = FrameCapture::Enter(*this);
    GuestMemory::CheckRange(shader, sizeof(Shader), 1);
    Shader fields;
    std::memcpy(&fields, static_cast<const void*>(shader), sizeof(Shader));
    require(fields.file_header == 0x34333231u && fields.version == 0x18u, "invalid shader header");
    require(fields.header_size >= sizeof(Shader), "shader header is smaller than its fixed fields");
    require(fields.shader_size != 0 && (fields.shader_size & 3u) == 0, "invalid shader size");
    GuestMemory::CheckRange(shader, fields.header_size, 1);
    const auto* code = const_cast<const void*>(fields.code);
    GuestMemory::CheckRange(code, fields.shader_size, 256);
    ShaderSnapshot snapshot{reinterpret_cast<std::uintptr_t>(code), reinterpret_cast<std::uintptr_t>(shader), fields.type, {}, {}};
    snapshot.code.resize(fields.shader_size / sizeof(std::uint32_t));
    std::memcpy(snapshot.code.data(), code, fields.shader_size);
    snapshot.header.resize(fields.header_size);
    std::memcpy(snapshot.header.data(), static_cast<const void*>(shader), fields.header_size);
    static const char* traceRegs = std::getenv("APS5_TRACE_SHADER_REGS");
    if (traceRegs != nullptr && (std::string(traceRegs) == "all" || std::strtoull(traceRegs, nullptr, 16) == snapshot.codeAddress)) {
        const auto print = [](const ShaderRegister* registers, std::uint32_t count) {
            for (std::uint32_t i = 0; i < count && registers != nullptr; ++i) {
                ShaderRegister value;
                std::memcpy(&value, static_cast<const void*>(registers + i), sizeof(value));
                std::fprintf(stderr, " %x=%08x", value.offset, value.value);
            }
        };
        std::fprintf(stderr, "[shader] 0x%llx type %u cx", static_cast<unsigned long long>(snapshot.codeAddress), fields.type);
        print(fields.cx_registers, fields.num_cx_registers);
        std::fprintf(stderr, " sh");
        print(fields.sh_registers, fields.num_sh_registers);
        std::fprintf(stderr, "\n");
    }
    if (entered.owns_lock()) FrameCapture::RecordShader(snapshot.codeAddress, snapshot.headerAddress, snapshot.type, snapshot.code, snapshot.header);
    registerSnapshot(std::move(snapshot), transaction, true);
}

void Driver::RestoreShader(std::uint64_t codeAddress, std::uint64_t headerAddress, std::uint8_t type, std::vector<std::uint32_t> code, std::vector<std::byte> header, bool prepare) {
    ShaderPreparationTransaction transaction;
    CheckFailure();
    ShaderSnapshot snapshot{codeAddress, headerAddress, type, std::move(code), std::move(header)};
    registerSnapshot(std::move(snapshot), transaction, prepare);
}

void Driver::RestoreRegistry(std::vector<ShaderSnapshot> snapshots, bool prepare) {
    ShaderPreparationTransaction transaction;
    CheckFailure();
    std::shared_ptr<VulkanDevice> localDevice = device.Load();
    if (localDevice == nullptr) {
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    std::shared_ptr<ShaderRegistry> next;
    {
        std::lock_guard lock(mutex);
        next = shaders != nullptr ? std::make_shared<ShaderRegistry>(*shaders) : std::make_shared<ShaderRegistry>();
    }
    for (auto& snapshot : snapshots) {
        try {
            const bool nullPixel = snapshot.type == NullPixelShader.type && snapshot.code.size() == std::size(NullPixelCode) && std::equal(snapshot.code.begin(), snapshot.code.end(), std::begin(NullPixelCode));
            snapshot.registeredState = std::make_shared<const RegisteredShaderState>(nullPixel ? NullPixelRegisteredState(snapshot) : DecodeRegisteredState(snapshot));
            if (prepare) {
                QueueState registered{};
                registered.shader = snapshot.registeredState->shader;
                registered.context = snapshot.registeredState->context;
                registered.userConfig = snapshot.registeredState->userConfig;
                snapshot.prepared->entries = PrepareRegistered(snapshot, *localDevice, registered, true);
                if ((snapshot.type == 0 || snapshot.type == 1) && !snapshot.prepared->entries.empty()) {
                    std::vector<std::uint64_t> key;
                    BuildRegisteredAbiKey(registered, *localDevice, key);
                    snapshot.prepared->registeredAbis.push_back(std::move(key));
                }
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[replay] shader 0x%llx was not restored: %s\n", static_cast<unsigned long long>(snapshot.codeAddress), error.what());
            continue;
        }
        const auto address = snapshot.codeAddress;
        next->insert_or_assign(address, std::make_shared<const ShaderSnapshot>(std::move(snapshot)));
    }
    std::lock_guard lock(mutex);
    rethrowFailure();
    shaders = std::move(next);
    publishNullPixelShader(*localDevice);
    transaction.Commit();
}

void Driver::registerSnapshot(ShaderSnapshot snapshot, ShaderPreparationTransaction& transaction, bool prepare) {
    {
        std::lock_guard lock(mutex);
        rethrowFailure();
        if (shaders != nullptr) {
            const auto found = shaders->find(snapshot.codeAddress);
            if (found != shaders->end()) {
                const auto& current = *found->second;
                ReuseShaderControlFlowValidation(snapshot, current);
                if (current.headerAddress == snapshot.headerAddress && current.type == snapshot.type && current.code == snapshot.code && current.header == snapshot.header) {
                    transaction.Commit();
                    return;
                }
            }
        }
    }

    std::shared_ptr<VulkanDevice> localDevice = device.Load();
    if (localDevice == nullptr) {
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    snapshot.registeredState = std::make_shared<const RegisteredShaderState>(DecodeRegisteredState(snapshot));
    StartShaderWarmList(*localDevice);
    if (prepare && AsyncRegistrationPrepare()) {
        QueueState registered{};
        registered.shader = snapshot.registeredState->shader;
        registered.context = snapshot.registeredState->context;
        registered.userConfig = snapshot.registeredState->userConfig;
        auto published = std::make_shared<const ShaderSnapshot>(std::move(snapshot));
        {
            std::lock_guard lock(mutex);
            rethrowFailure();
            PublishRegisteredShader(shaders, published);
            publishNullPixelShader(*localDevice);
        }
        PrepareRegisteredInBackground(std::move(published), localDevice, std::move(registered));
        transaction.Commit();
        return;
    }
    if (prepare) {
        QueueState registered{};
        registered.shader = snapshot.registeredState->shader;
        registered.context = snapshot.registeredState->context;
        registered.userConfig = snapshot.registeredState->userConfig;
        snapshot.prepared->entries = PrepareRegistered(snapshot, *localDevice, registered, true);
        if ((snapshot.type == 0 || snapshot.type == 1) && !snapshot.prepared->entries.empty()) {
            std::vector<std::uint64_t> key;
            BuildRegisteredAbiKey(registered, *localDevice, key);
            snapshot.prepared->registeredAbis.push_back(std::move(key));
        }
    }
    std::lock_guard lock(mutex);
    rethrowFailure();
    PublishRegisteredShader(shaders, std::make_shared<const ShaderSnapshot>(std::move(snapshot)));
    publishNullPixelShader(*localDevice);
    transaction.Commit();
}

void Driver::publishNullPixelShader(const VulkanDevice& localDevice) {
    if (shaders->find(NullPixelProgramAddress()) == shaders->end()) {
        ShaderSnapshot null{NullPixelProgramAddress(), reinterpret_cast<std::uintptr_t>(&NullPixelShader), NullPixelShader.type, {}, {}};
        null.code.assign(std::begin(NullPixelCode), std::end(NullPixelCode));
        null.header.resize(sizeof(Shader));
        std::memcpy(null.header.data(), &NullPixelShader, sizeof(Shader));
        null.registeredState = std::make_shared<const RegisteredShaderState>(NullPixelRegisteredState(null));
        QueueState nullState{};
        nullState.shader = null.registeredState->shader;
        nullState.context = null.registeredState->context;
        nullState.userConfig = null.registeredState->userConfig;
        null.prepared->entries = PrepareRegistered(null, localDevice, nullState, true);
        PublishRegisteredShader(shaders, std::make_shared<const ShaderSnapshot>(std::move(null)));
    }
}

}

extern "C" AgcDriver::DriverDetail::ShaderPreparationTransaction* AgcDriverBeginShaderPreparation_nid_postfix() {
    return new AgcDriver::DriverDetail::ShaderPreparationTransaction();
}

extern "C" void AgcDriverCommitShaderPreparation_nid_postfix(AgcDriver::DriverDetail::ShaderPreparationTransaction* transaction) {
    transaction->Commit();
}

extern "C" void AgcDriverEndShaderPreparation_nid_postfix(AgcDriver::DriverDetail::ShaderPreparationTransaction* transaction) noexcept {
    delete transaction;
}
