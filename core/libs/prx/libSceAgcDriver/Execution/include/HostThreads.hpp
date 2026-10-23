#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTTHREADS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTTHREADS_HPP

#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

namespace AgcDriver {

// The driver's latency-critical threads (queue workers, draw committer and preparation workers,
// recorder) join the Multimedia Class Scheduler's "Games" task: a raised, steady scheduling
// priority, as a game's render thread gets. APS5_NO_MMCSS=1 leaves them at normal priority.
inline void JoinGamesSchedulingClass() {
#ifdef _WIN32
    static const bool disabled = [] {
        const char* value = std::getenv("APS5_NO_MMCSS");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    if (disabled) return;
    static const auto join = [] {
        using Join = HANDLE(WINAPI*)(LPCWSTR, LPDWORD);
        const auto avrt = LoadLibraryW(L"avrt.dll");
        return avrt != nullptr ? reinterpret_cast<Join>(reinterpret_cast<void*>(GetProcAddress(avrt, "AvSetMmThreadCharacteristicsW"))) : nullptr;
    }();
    if (join == nullptr) return;
    DWORD task = 0;
    join(L"Games", &task);
#endif
}

}  // namespace AgcDriver

#endif
