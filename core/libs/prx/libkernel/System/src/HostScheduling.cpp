#ifdef _WIN32
// CPU sets and power throttling are Windows 10 APIs (the library targets 0x0602 otherwise).
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#endif

#include "prx/libkernel/System/include/HostScheduling.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>


namespace HostScheduling {

#ifdef _WIN32
namespace {

bool Disabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "0") != 0;
}

// The CPU set ids of the most capable efficiency class (P-cores on a hybrid CPU), and every id.
struct CpuSets {
    std::vector<ULONG> performance;
    std::vector<ULONG> all;
};

CpuSets ReadCpuSets() {
    CpuSets sets;
    using GetInfo = BOOL(WINAPI*)(PSYSTEM_CPU_SET_INFORMATION, ULONG, PULONG, HANDLE, ULONG);
    const auto get = reinterpret_cast<GetInfo>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetSystemCpuSetInformation")));
    if (get == nullptr) return sets;
    ULONG bytes = 0;
    get(nullptr, 0, &bytes, GetCurrentProcess(), 0);
    if (bytes == 0) return sets;
    std::vector<unsigned char> buffer(bytes);
    if (!get(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), bytes, &bytes, GetCurrentProcess(), 0)) return sets;
    BYTE best = 0;
    for (ULONG offset = 0; offset < bytes;) {
        const auto* entry = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (entry->Type == CpuSetInformation) best = std::max(best, entry->CpuSet.EfficiencyClass);
        offset += entry->Size;
    }
    for (ULONG offset = 0; offset < bytes;) {
        const auto* entry = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (entry->Type == CpuSetInformation) {
            sets.all.push_back(entry->CpuSet.Id);
            if (entry->CpuSet.EfficiencyClass == best) sets.performance.push_back(entry->CpuSet.Id);
        }
        offset += entry->Size;
    }
    return sets;
}

const CpuSets& Sets() {
    static const CpuSets sets = ReadCpuSets();
    return sets;
}

void Configure() {
    // 1 ms timer resolution: guest sleeps and timed waits resolve at it (Windows otherwise keeps
    // 15.6 ms for a process it considers in the background).
    if (!Disabled("APS5_NO_TIMER_RESOLUTION")) {
        if (const auto winmm = LoadLibraryW(L"winmm.dll")) {
            using BeginPeriod = UINT(WINAPI*)(UINT);
            if (const auto begin = reinterpret_cast<BeginPeriod>(reinterpret_cast<void*>(GetProcAddress(winmm, "timeBeginPeriod")))) begin(1);
        }
    }
    // No power throttling (EcoQoS): Windows may run a process it deems inefficient on efficiency
    // cores at low clocks and coarsen its timers.
    if (!Disabled("APS5_KEEP_POWER_THROTTLING")) {
        PROCESS_POWER_THROTTLING_STATE state{};
        state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED | 0x4u;  // | IGNORE_TIMER_RESOLUTION
        state.StateMask = 0;
        using SetInfo = BOOL(WINAPI*)(HANDLE, PROCESS_INFORMATION_CLASS, LPVOID, DWORD);
        if (const auto set = reinterpret_cast<SetInfo>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetProcessInformation")))) set(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof(state));
    }
    // On a hybrid CPU every thread prefers the performance cores by default (a soft preference:
    // Windows still uses the other cores when these are all busy). PPSA01317's threads hand work
    // to each other thousands of times a frame; one landing on an efficiency core stalls the
    // chain. Background pools opt back into every core (AllowEveryCore). APS5_NO_PCORE_PREFERENCE=1
    // leaves the default.
    const auto& sets = Sets();
    if (!Disabled("APS5_NO_PCORE_PREFERENCE") && !sets.performance.empty() && sets.performance.size() < sets.all.size()) {
        using SetDefault = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);
        if (const auto set = reinterpret_cast<SetDefault>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetProcessDefaultCpuSets")))) {
            if (set(GetCurrentProcess(), sets.performance.data(), static_cast<ULONG>(sets.performance.size()))) {
                std::fprintf(stderr, "[scheduling] threads prefer the %zu performance-core CPU sets of %zu\n", sets.performance.size(), sets.all.size());
            }
        }
    }
}

}  // namespace

void ConfigureProcessOnce() {
    static std::once_flag once;
    std::call_once(once, Configure);
}

void AllowEveryCore() {
    const auto& sets = Sets();
    if (sets.all.empty()) return;
    using SetThreadSets = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);
    if (const auto set = reinterpret_cast<SetThreadSets>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadSelectedCpuSets")))) set(GetCurrentThread(), sets.all.data(), static_cast<ULONG>(sets.all.size()));
}
#else
void ConfigureProcessOnce() {}
void AllowEveryCore() {}
#endif

}  // namespace HostScheduling
