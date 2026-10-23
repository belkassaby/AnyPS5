#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <system_error>

namespace GuestArena {

// Suspends, for its lifetime, the other threads of this process that are executing guest code
// (app.exe or a relinked *.guest.prx). Converting resident guest memory in place (cut, share, evict)
// releases and recommits pages that guest threads may be using: one touching them meanwhile takes an
// access fault, and Windows dispatches it on that thread's stack, writing its exception frame over the
// 128 bytes below rsp that SysV guest code keeps locals in (PPSA01317's BVH jobs then read a clobbered
// pointer). A thread running guest code holds no emulator lock, so suspending it cannot deadlock the
// conversion; threads inside the emulator or the OS are resumed at once. No heap is used.
// APS5_NO_GUEST_QUIESCE=1 turns it off.
class GuestQuiesce {
public:
    GuestQuiesce() {
        static const bool enabled = std::getenv("APS5_NO_GUEST_QUIESCE") == nullptr;
        if (!enabled) return;
        const auto t0 = std::chrono::steady_clock::now();
        collectRanges();
        const auto t1 = std::chrono::steady_clock::now();
        // Threads blocked in a kernel wait are not running guest code (their user-mode rip is in
        // ntdll), so only running and ready threads are suspended and inspected: suspending and
        // reading the context of every thread (~200 in PPSA01317) took 130-300 ms per conversion,
        // which stretched each streamed map of the licence track to that. A waiting thread that wakes
        // during the conversion is the remaining (narrow) window. APS5_FULL_GUEST_QUIESCE=1 inspects
        // every thread as before.
        static const bool full = std::getenv("APS5_FULL_GUEST_QUIESCE") != nullptr;
        const bool stated = !full && collectStates();
        const auto t2 = std::chrono::steady_clock::now();
        if (!stated) collectSnapshot();
        const auto t3 = std::chrono::steady_clock::now();
        // Every candidate is suspended first and inspected after, so the threads' suspensions
        // take effect concurrently (a thread inside a long kernel call delays its own).
        HANDLE pending[MaxThreads];
        std::size_t suspended = 0;
        for (std::size_t i = 0; i < candidateCount; ++i) {
            const HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, candidates[i]);
            if (thread == nullptr) continue;
            if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
                CloseHandle(thread);
                continue;
            }
            pending[suspended++] = thread;
        }
        for (std::size_t i = 0; i < suspended; ++i) {
            // GetThreadContext also waits until the suspension has taken effect.
            context.ContextFlags = CONTEXT_CONTROL;
            if (GetThreadContext(pending[i], &context) && inGuest(context.Rip)) {
                held[count++] = pending[i];
                continue;
            }
            ResumeThread(pending[i]);
            CloseHandle(pending[i]);
        }
        const auto t4 = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        if (ms(t0, t4) > 20.0) std::fprintf(stderr, "[quiesce] %.1f ms: modules %.1f, thread states %.1f, snapshot %.1f, suspend %.1f (%zu inspected, %zu held, %zu waiting skipped)\n", ms(t0, t4), ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), suspended, count, waitingCount);
        heldTotal().fetch_add(count, std::memory_order_relaxed);
        quiesces().fetch_add(1, std::memory_order_relaxed);
    }
    GuestQuiesce(const GuestQuiesce&) = delete;
    GuestQuiesce& operator=(const GuestQuiesce&) = delete;
    ~GuestQuiesce() {
        for (std::size_t i = 0; i < count; ++i) {
            ResumeThread(held[i]);
            CloseHandle(held[i]);
        }
    }
    static std::atomic<std::uint64_t>& quiesces() {
        static std::atomic<std::uint64_t> value{0};
        return value;
    }
    static std::atomic<std::uint64_t>& heldTotal() {
        static std::atomic<std::uint64_t> value{0};
        return value;
    }

private:
    static constexpr std::size_t MaxThreads = 1024;
    static constexpr std::size_t MaxRanges = 64;
    struct Range {
        std::uintptr_t begin;
        std::uintptr_t end;
    };

    // The ids of this process's other threads not in a kernel wait (ThreadState 5), from one
    // NtQuerySystemInformation(SystemProcessInformation) call; false when it is unavailable.
    bool collectStates() {
        using Query = LONG(WINAPI*)(ULONG, void*, ULONG, ULONG*);
        static const auto query = reinterpret_cast<Query>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation")));
        if (query == nullptr) return false;
        static std::vector<unsigned char> buffer(1u << 20);
        ULONG needed = 0;
        LONG status = 0;
        for (int attempt = 0; attempt < 4; ++attempt) {
            status = query(5, buffer.data(), static_cast<ULONG>(buffer.size()), &needed);
            if (status != static_cast<LONG>(0xC0000004)) break;  // STATUS_INFO_LENGTH_MISMATCH
            buffer.resize(std::max<std::size_t>(buffer.size() * 2, needed + (64u << 10)));
        }
        if (status < 0) return false;
        // x64 layouts: SYSTEM_PROCESS_INFORMATION is 0x100 bytes (NumberOfThreads at 0x04,
        // UniqueProcessId at 0x50), followed by its SYSTEM_THREAD_INFORMATION entries of 0x50 bytes
        // (ClientId.UniqueThread at 0x30, ThreadState at 0x44).
        const auto process = static_cast<std::uintptr_t>(GetCurrentProcessId());
        for (std::size_t offset = 0;;) {
            const auto* entry = buffer.data() + offset;
            std::uintptr_t id = 0;
            std::memcpy(&id, entry + 0x50, sizeof(id));
            if (id == process) {
                ULONG threads = 0;
                std::memcpy(&threads, entry + 0x04, sizeof(threads));
                const auto self = GetCurrentThreadId();
                for (ULONG i = 0; i < threads; ++i) {
                    const auto* thread = entry + 0x100 + i * 0x50;
                    std::uintptr_t tid = 0;
                    ULONG state = 0;
                    std::memcpy(&tid, thread + 0x30, sizeof(tid));
                    std::memcpy(&state, thread + 0x44, sizeof(state));
                    if (static_cast<DWORD>(tid) == self) continue;
                    if (state == 5) ++waitingCount;
                    else if (candidateCount < MaxThreads) candidates[candidateCount++] = static_cast<DWORD>(tid);
                }
                return true;
            }
            ULONG next = 0;
            std::memcpy(&next, entry, sizeof(next));
            if (next == 0) return false;
            offset += next;
        }
    }

    // Every other thread of this process (APS5_FULL_GUEST_QUIESCE, or no thread states).
    void collectSnapshot() {
        const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snapshot == INVALID_HANDLE_VALUE) return;
        const auto process = GetCurrentProcessId();
        const auto self = GetCurrentThreadId();
        THREADENTRY32 entry{};
        entry.dwSize = sizeof(entry);
        for (BOOL more = Thread32First(snapshot, &entry); more && candidateCount < MaxThreads; more = Thread32Next(snapshot, &entry)) {
            if (entry.th32OwnerProcessID == process && entry.th32ThreadID != self) candidates[candidateCount++] = entry.th32ThreadID;
        }
        CloseHandle(snapshot);
    }

    // The guest module ranges, enumerated again at most every 5 s (guest modules load at boot; the
    // enumeration cost ~5 ms per quiesce). Quiesces run under the mappings mutex, one at a time.
    void collectRanges() {
        static Range cached[MaxRanges];
        static std::size_t cachedCount = 0;
        static std::chrono::steady_clock::time_point taken{};
        const auto now = std::chrono::steady_clock::now();
        if (cachedCount == 0 || now - taken > std::chrono::seconds(5)) {
            rangeCount = 0;
            enumerateRanges();
            std::copy(ranges, ranges + rangeCount, cached);
            cachedCount = rangeCount;
            taken = now;
            return;
        }
        std::copy(cached, cached + cachedCount, ranges);
        rangeCount = cachedCount;
    }

    void enumerateRanges() {
        HMODULE modules[1024];
        DWORD needed = 0;
        if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed)) return;
        const auto total = std::min<std::size_t>(needed / sizeof(HMODULE), std::size(modules));
        const HMODULE executable = GetModuleHandleW(nullptr);
        for (std::size_t i = 0; i < total && rangeCount < MaxRanges; ++i) {
            char name[MAX_PATH] = {};
            if (modules[i] != executable && (K32GetModuleBaseNameA(GetCurrentProcess(), modules[i], name, sizeof(name)) == 0 || std::strstr(name, ".guest.") == nullptr)) continue;
            MODULEINFO info{};
            if (!K32GetModuleInformation(GetCurrentProcess(), modules[i], &info, sizeof(info))) continue;
            const auto begin = reinterpret_cast<std::uintptr_t>(info.lpBaseOfDll);
            ranges[rangeCount++] = {begin, begin + info.SizeOfImage};
        }
    }

    bool inGuest(std::uint64_t rip) const {
        for (std::size_t i = 0; i < rangeCount; ++i) {
            if (rip >= ranges[i].begin && rip < ranges[i].end) return true;
        }
        return false;
    }

    alignas(16) CONTEXT context{};
    HANDLE held[MaxThreads];
    std::size_t count = 0;
    Range ranges[MaxRanges];
    std::size_t rangeCount = 0;
    DWORD candidates[MaxThreads];
    std::size_t candidateCount = 0;
    std::size_t waitingCount = 0;
};

class WindowsMappings {
public:
    static WindowsMappings& Get() {
        static WindowsMappings mappings;
        return mappings;
    }

    void* Reserve(void* address, std::size_t bytes) {
        return allocate(GetCurrentProcess(), address, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    }

    std::vector<std::pair<std::uintptr_t, std::size_t>> Commit(void* address, std::size_t bytes, DWORD protection, std::size_t granule, bool watched) {
        std::vector<std::pair<std::uintptr_t, std::size_t>> created;
        std::lock_guard lock(mutex);
        const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
        for (auto cursor = reinterpret_cast<std::uintptr_t>(address); cursor < end;) {
            const auto memory = query(cursor);
            const auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
            if (memory.State == MEM_RESERVE) {
                const auto limit = std::min(end, cursor + granule);
                auto placeholderEnd = stop;
                while (placeholderEnd < limit) {
                    const auto next = query(placeholderEnd);
                    if (next.State != MEM_RESERVE) break;
                    placeholderEnd = reinterpret_cast<std::uintptr_t>(next.BaseAddress) + next.RegionSize;
                }
                const auto size = std::min(limit, placeholderEnd) - cursor;
                reset(cursor, size);
                const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watched ? MEM_WRITE_WATCH : 0);
                if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(cursor), size, flags, protection, nullptr, 0)) fail("replace guest placeholder with private memory");
                note(1, cursor, cursor + size);
                if (!created.empty() && created.back().first + created.back().second == cursor) created.back().second += size;
                else created.emplace_back(cursor, size);
                cursor += size;
            } else {
                if (memory.State != MEM_COMMIT) throw std::runtime_error("guest memory is not committed");
                const auto mapped = views.find(cursor & ~(pageBytes - 1));
                if (mapped != views.end()) {
                    mapped->second.protection = protection;
                    mapped->second.armed = false;
                    invalidate(*mapped->second.page);
                }
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(cursor), stop - cursor, protection, &previous)) fail("protect guest memory");
                cursor = stop;
            }
        }
        return created;
    }

    void Reset(void* address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        ConversionScope conversion(*this);
        reset(reinterpret_cast<std::uintptr_t>(address), bytes);
    }

    void Map(void* address, std::size_t bytes, HANDLE section, std::uint64_t offset, DWORD protection, bool watched) {
        std::lock_guard lock(mutex);
        auto cursor = reinterpret_cast<std::uintptr_t>(address);
        // A map repeating the current one (every page already shows the same physical page with the
        // same protection) changes nothing: it returns before the reset, which unmaps and remaps
        // every 16 KiB view (PPSA01317 remaps a 200 MiB render target over and over during the
        // licence briefing, and each view's unmap took ~30 ms once the GPU imported it: minutes
        // per map). APS5_NO_SAME_MAP_SKIP=1 always resets.
        static const bool skipSame = std::getenv("APS5_NO_SAME_MAP_SKIP") == nullptr;
        ++quiesceStats.maps;
        const auto mapStart = std::chrono::steady_clock::now();
        struct MapTimed {
            WindowsMappings& owner;
            std::chrono::steady_clock::time_point start;
            ~MapTimed() {
                const auto now = std::chrono::steady_clock::now();
                auto& stats = owner.quiesceStats;
                stats.mapMs += std::chrono::duration<double, std::milli>(now - start).count();
                if (now - stats.since >= std::chrono::seconds(10)) {
                    std::fprintf(stderr, "[mappings] (10 s) %llu maps taking %.0f ms, %llu page shares taking %.0f ms, %llu guest quiesces taking %.0f ms (max %.1f ms)\n", static_cast<unsigned long long>(stats.maps), stats.mapMs, static_cast<unsigned long long>(stats.shares), stats.shareMs, static_cast<unsigned long long>(stats.count), stats.ms, stats.maxMs);
                    stats = {};
                    stats.since = now;
                }
            }
        } mapTimed{*this, mapStart};
        if (skipSame && sameMapping(cursor, bytes, section, offset, protection)) {
            watchResident = watched;
            note(10, cursor, cursor + bytes);
            return;
        }
        ConversionScope conversion(*this);
        const auto resetStart = std::chrono::steady_clock::now();
        const auto quiesceBefore = quiesceMs;
        reset(cursor, bytes);
        const auto resetMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - resetStart).count();
        if (resetMs > 100.0) std::fprintf(stderr, "[mappings] slow map reset: 0x%llx+0x%zx took %.0f ms (%.0f ms suspending guest threads)\n", static_cast<unsigned long long>(cursor), bytes, resetMs, quiesceMs - quiesceBefore);
        watchResident = watched;
        HANDLE duplicate = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), section, GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) fail("keep shared guest section");
        const auto owned = std::make_shared<Section>(duplicate);
        const auto slotOf = [&](std::size_t done) -> Physical& { return physical[std::make_pair(reinterpret_cast<std::uintptr_t>(section), offset + done)]; };
        const auto pageOf = [](Physical& slot) {
            auto shared = slot.page.lock();
            if (!shared) {
                shared = std::make_shared<SharedPage>();
                slot.page = shared;
            }
            return shared;
        };
        for (std::size_t done = 0; done < bytes;) {
            const auto base = cursor + done;
            if (pageOf(slotOf(done))->aliases.empty()) {
                auto runBytes = pageBytes;
                while (done + runBytes < bytes && (base + runBytes) % chunkBytes != 0) {
                    const auto next = slotOf(done + runBytes).page.lock();
                    if (next && !next->aliases.empty()) break;
                    runBytes += pageBytes;
                }
                split(base, runBytes);
                commitResident(base, runBytes);
                note(2, base, base + runBytes);
                residentRuns.emplace(base, Run{base + runBytes, {}, static_cast<std::uint32_t>(runBytes >> 12)});
                Window window;
                for (std::size_t at = 0; at < runBytes; at += pageBytes) {
                    auto& slot = slotOf(done + at);
                    const auto shared = pageOf(slot);
                    if (slot.stored) {
                        if (window.view == nullptr) window = open(section, offset + done, runBytes);
                        commitSection(window.bytes + at);
                        std::memcpy(reinterpret_cast<void*>(base + at), window.bytes + at, pageBytes);
                    }
                    shared->aliases.push_back(base + at);
                    views.emplace(base + at, View{shared, protection, 0, false, owned, offset + done + at, 0, true, &slot});
                    invalidate(*shared);
                }
                close(window);
                DWORD previous;
                if (protection != PAGE_READWRITE && !VirtualProtect(reinterpret_cast<void*>(base), runBytes, protection, &previous)) fail("protect resident guest pages");
                done += runBytes;
                continue;
            }
            auto& slot = slotOf(done);
            const auto shared = pageOf(slot);
            const auto aliases = shared->aliases;
            for (const auto alias : aliases) {
                if (views.at(alias).resident) share(alias);
            }
            split(base, pageBytes);
            mapShared(base, section, offset + done, protection);
            note(3, base, base + pageBytes);
            slot.stored = true;
            shared->aliases.push_back(base);
            views.emplace(base, View{shared, protection, 0, false, owned, offset + done, 0, false, &slot});
            invalidate(*shared);
            done += pageBytes;
        }
    }

    void SetProtection(std::uintptr_t address, std::size_t bytes, DWORD protection) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address); it != views.end() && it->first < address + bytes; ++it) {
            it->second.protection = protection;
            it->second.armed = false;
            invalidate(*it->second.page);
        }
    }

    void Pin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& page = *it->second.page;
            ++page.pins;
            for (const auto alias : page.aliases) {
                auto& view = views.at(alias);
                if (!view.armed) continue;
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(alias), pageBytes, view.protection, &previous)) fail("pin shared guest page writable");
                view.armed = false;
            }
            invalidate(page);
        }
    }

    void Unpin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& page = *it->second.page;
            if (page.pins != 0) --page.pins;
            invalidate(page);
        }
    }

    bool HandleWrite(std::uintptr_t address) {
        std::lock_guard lock(mutex);
        const auto base = address & ~(pageBytes - 1);
        const auto found = views.find(base);
        if (found == views.end() || !writable(found->second.protection)) return false;
        auto& view = found->second;
        if (view.resident) {
            if (view.armed) {
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(base), pageBytes, view.protection, &previous)) fail("resume resident guest write");
                view.armed = false;
                view.cleanStreak = 0;
                view.armAfter = 255;
                forgetClean(base, base + pageBytes);
                return true;
            }
            const auto memory = query(address);
            return memory.State == MEM_COMMIT && writable(memory.Protect & 0xffu);
        }
        invalidate(*view.page);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(base), pageBytes, view.protection, &previous)) fail("resume shared memory write");
        view.armed = false;
        return true;
    }

    bool HandleRead(std::uintptr_t address) {
        std::lock_guard lock(mutex);
        if (!views.contains(address & ~(pageBytes - 1))) return false;
        const auto memory = query(address);
        return memory.State == MEM_COMMIT && (memory.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
    }

    bool BeginHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto first = views.lower_bound(address & ~(pageBytes - 1));
        const auto end = address + bytes;
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            if (!writable(it->second.protection)) return false;
        }
        for (auto cursor = address; cursor < end;) {
            if (views.find(cursor & ~(pageBytes - 1)) != views.end()) {
                cursor = (cursor & ~(pageBytes - 1)) + pageBytes;
                continue;
            }
            MEMORY_BASIC_INFORMATION memory{};
            if (VirtualQuery(reinterpret_cast<void*>(cursor), &memory, sizeof(memory)) != sizeof(memory)) return false;
            auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
            if (const auto view = views.lower_bound(cursor); view != views.end()) stop = std::min(stop, view->first);
            if (memory.Type == MEM_PRIVATE && (memory.State != MEM_COMMIT || (memory.Protect & PAGE_GUARD) != 0 || !writable(memory.Protect & 0xffu))) return false;
            cursor = stop;
        }
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            auto& view = it->second;
            ++view.hostWrites;
            invalidate(*view.page);
            if (!view.armed) continue;
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(it->first), pageBytes, view.protection, &previous)) fail("open shared memory to a host write");
            view.armed = false;
        }
        return true;
    }

    void EndHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            --it->second.hostWrites;
            invalidate(*it->second.page);
        }
    }

    void* MapAlias(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        ConversionScope conversion(*this);
        const auto refuse = [&](const char* reason) {
            char text[192];
            std::snprintf(text, sizeof(text), "read-write alias of shared guest memory 0x%llx+0x%llx: %s", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), reason);
            return std::runtime_error(text);
        };
        if (address % pageBytes != 0 || bytes % pageBytes != 0 || bytes == 0) throw refuse("the range is not made of whole shared pages");
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            const auto found = views.find(address + done);
            if (found != views.end() && found->second.resident) share(address + done);
        }
        note(9, address, address + bytes);
        auto view = views.find(address);
        if (view == views.end()) throw refuse("the range does not start at a shared view");
        const auto section = view->second.section;
        const auto offset = view->second.offset;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        for (std::size_t done = 0; done < bytes; done += pageBytes, ++view) {
            if (view == views.end() || view->first != address + done || view->second.offset != offset + done) throw refuse("the range is not one contiguous run of views of a section");
            if (view->second.section != section && !sameSection(view->second.section->handle, section->handle)) throw refuse("the range spans several sections");
        }
        const auto lead = offset % system.dwAllocationGranularity;
        void* alias = map(section->handle, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (alias == nullptr) {
            char text[160];
            std::snprintf(text, sizeof(text), "MapViewOfFile3 of a read-write alias of shared guest memory 0x%llx+0x%llx", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), text);
        }
        return static_cast<char*>(alias) + lead;
    }

    void UnmapAlias(void* alias) {
        if (alias == nullptr) return;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        const auto base = reinterpret_cast<std::uintptr_t>(alias) & ~(static_cast<std::uintptr_t>(system.dwAllocationGranularity) - 1);
        if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(base), 0)) fail("unmap shared guest alias");
    }

    bool Protection(std::uintptr_t address, std::uint32_t* protection) {
        std::lock_guard lock(mutex);
        const auto found = views.find(address & ~(pageBytes - 1));
        if (found == views.end()) return false;
        *protection = found->second.protection;
        return true;
    }

    // Local diagnostic: how a guest page is backed, and the recent backing conversions near it.
    void Describe(std::uintptr_t address, char* out, std::size_t size) {
        int used = 0;
        const auto put = [&](const char* format, auto... args) {
            if (used >= static_cast<int>(size)) return;
            const int written = std::snprintf(out + used, size - used, format, args...);
            if (written > 0) used += written;
        };
        out[0] = 0;
        std::unique_lock lock(mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            put("(mappings busy) ");
        } else {
            const auto base = address & ~(pageBytes - 1);
            const auto found = views.find(base);
            if (found == views.end()) {
                put("no view; ");
            } else {
                const auto& view = found->second;
                put("%s view armed %d protect 0x%lx aliases %zu pins %u host writes %u stored %d", view.resident ? "resident" : "shared", view.armed ? 1 : 0, view.protection, view.page->aliases.size(), view.page->pins, view.hostWrites, view.slot != nullptr && view.slot->stored ? 1 : 0);
                if (view.resident) {
                    auto run = residentRuns.upper_bound(address);
                    if (run != residentRuns.begin()) {
                        --run;
                        put(" run 0x%llx-0x%llx", static_cast<unsigned long long>(run->first), static_cast<unsigned long long>(run->second.end));
                    }
                }
                put("; ");
            }
        }
        static const char* const names[] = {"?", "commit", "map-resident", "map-shared", "cut", "share", "evict", "unmap-view", "release", "alias", "same-map"};
        const auto now = GetTickCount64();
        const auto total = conversionCount.load(std::memory_order_relaxed);
        int shown = 0;
        for (std::uint64_t back = 0; back < total && back < ConversionLog && shown < 10; ++back) {
            const auto& entry = conversions[(total - 1 - back) % ConversionLog];
            if (entry.end + 0x200000 <= address || entry.start >= address + 0x200000) continue;
            put("[%s 0x%llx-0x%llx %llu ms ago thread %lu] ", names[entry.kind < 11 ? entry.kind : 0], static_cast<unsigned long long>(entry.start), static_cast<unsigned long long>(entry.end), static_cast<unsigned long long>(now - entry.when), entry.thread);
            ++shown;
        }
        put("(%llu conversions logged)", static_cast<unsigned long long>(total));
    }

    // [collect-libc] (APS5_PROFILE_COLLECT): where Collect's time goes, printed every 10 s.
    struct CollectProfile {
        std::atomic<std::uint64_t> unlockedRetries{0};
        std::atomic<std::uint64_t> calls{0}, totalTicks{0}, lockWaits{0}, lockWaitTicks{0}, cleanSkips{0}, residentRuns{0}, residentFresh{0}, viewPages{0}, viewArms{0}, privateRegions{0}, queryTicks{0}, watchCalls{0}, watchTicks{0}, watchPages{0};
        // GetWriteWatch call durations: <1, <4, <16, <64, <256, >=256 us (calls, ticks).
        std::array<std::atomic<std::uint64_t>, 6> watchHistogram{};
        std::array<std::atomic<std::uint64_t>, 6> watchHistogramTicks{};
        std::atomic<std::int64_t> reported{0};
    };
    static std::atomic<bool (*)()>& unlockedWatchQuery() {
        static std::atomic<bool (*)()> query{nullptr};
        return query;
    }
    static bool unlockedWatch() {
        static const bool enabled = std::getenv("APS5_COLLECT_UNLOCKED_WATCH") != nullptr;
        if (const auto query = unlockedWatchQuery().load(std::memory_order_acquire)) return query();
        return enabled;
    }
    static bool collectProfiled() {
        static const bool enabled = std::getenv("APS5_PROFILE_COLLECT") != nullptr;
        return enabled;
    }
    static CollectProfile& collectProfile() {
        static CollectProfile profile;
        return profile;
    }
    static std::int64_t profileNow() {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        return now.QuadPart;
    }
    static double profileMs(std::uint64_t ticks) {
        static const double scale = [] {
            LARGE_INTEGER frequency;
            QueryPerformanceFrequency(&frequency);
            return 1e3 / static_cast<double>(frequency.QuadPart);
        }();
        return static_cast<double>(ticks) * scale;
    }
    static void reportCollectProfile() {
        auto& profile = collectProfile();
        const auto now = profileNow();
        auto last = profile.reported.load(std::memory_order_relaxed);
        if (last == 0) {
            profile.reported.compare_exchange_strong(last, now);
            return;
        }
        if (profileMs(static_cast<std::uint64_t>(now - last)) < 10e3 || !profile.reported.compare_exchange_strong(last, now)) return;
        const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
        const auto ms = [](std::atomic<std::uint64_t>& value) { return profileMs(value.exchange(0, std::memory_order_relaxed)); };
        const auto calls = take(profile.calls);
        const auto total = ms(profile.totalTicks);
        const auto waits = take(profile.lockWaits);
        const auto waited = ms(profile.lockWaitTicks);
        const auto clean = take(profile.cleanSkips);
        const auto resident = take(profile.residentRuns);
        const auto fresh = take(profile.residentFresh);
        const auto viewPages = take(profile.viewPages);
        const auto viewArms = take(profile.viewArms);
        const auto regions = take(profile.privateRegions);
        const auto queried = ms(profile.queryTicks);
        const auto watchCalls = take(profile.watchCalls);
        const auto watched = ms(profile.watchTicks);
        const auto watchPages = take(profile.watchPages);
        const auto retries = take(profile.unlockedRetries);
        char histogram[256];
        int used = 0;
        static const char* const names[] = {"<1us", "<4us", "<16us", "<64us", "<256us", ">=256us"};
        for (std::size_t i = 0; i < profile.watchHistogram.size() && used < static_cast<int>(sizeof(histogram)) - 40; ++i) {
            used += std::snprintf(histogram + used, sizeof(histogram) - used, " %s %llu/%.0fms", names[i], take(profile.watchHistogram[i]), ms(profile.watchHistogramTicks[i]));
        }
        std::fprintf(stderr, "[collect-libc] (10 s) %llu calls %.1f ms (lock waits %llu, %.1f ms); clean skips %llu; resident runs %llu (%llu fresh); view pages %llu (%llu armed aliases); private regions %llu (VirtualQuery %.1f ms); GetWriteWatch %llu calls %.1f ms over %llu MiB (%llu unlocked calls redone locked); durations%s\n", calls, total, waits, waited, clean, resident, fresh, viewPages, viewArms, regions, queried, watchCalls, watched, watchPages * 4096ull >> 20, retries, histogram);
    }

    bool Collect(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
        if (!collectProfiled()) return collect(address, bytes, pages, count, clear);
        const auto start = profileNow();
        const bool result = collect(address, bytes, pages, count, clear);
        auto& profile = collectProfile();
        profile.calls.fetch_add(1, std::memory_order_relaxed);
        profile.totalTicks.fetch_add(static_cast<std::uint64_t>(profileNow() - start), std::memory_order_relaxed);
        reportCollectProfile();
        return result;
    }

    bool collect(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
        const bool profiled = collectProfiled();
        std::unique_lock lock(mutex, std::defer_lock);
        if (!profiled) {
            lock.lock();
        } else if (!lock.try_lock()) {
            const auto waitStart = profileNow();
            lock.lock();
            collectProfile().lockWaits.fetch_add(1, std::memory_order_relaxed);
            collectProfile().lockWaitTicks.fetch_add(static_cast<std::uint64_t>(profileNow() - waitStart), std::memory_order_relaxed);
        }
        const auto watch = [&](DWORD flags, std::uintptr_t at, std::size_t length, void** out, ULONG_PTR* found, DWORD* granularity) {
            if (!profiled) return GetWriteWatch(flags, reinterpret_cast<void*>(at), length, out, found, granularity);
            const auto watchStart = profileNow();
            const auto result = GetWriteWatch(flags, reinterpret_cast<void*>(at), length, out, found, granularity);
            auto& profile = collectProfile();
            profile.watchCalls.fetch_add(1, std::memory_order_relaxed);
            profile.watchPages.fetch_add(length / 4096, std::memory_order_relaxed);
            const auto ticks = static_cast<std::uint64_t>(profileNow() - watchStart);
            profile.watchTicks.fetch_add(ticks, std::memory_order_relaxed);
            const auto us = profileMs(ticks) * 1000.0;
            const std::size_t bucket = us < 1 ? 0 : us < 4 ? 1 : us < 16 ? 2 : us < 64 ? 3 : us < 256 ? 4 : 5;
            profile.watchHistogram[bucket].fetch_add(1, std::memory_order_relaxed);
            profile.watchHistogramTicks[bucket].fetch_add(ticks, std::memory_order_relaxed);
            return result;
        };
        const auto capacity = *count;
        *count = 0;
        const auto end = address + bytes;
        bool retryLocked = false;
        for (auto cursor = address; cursor < end;) {
            const auto nextClean = cleanRanges.upper_bound(cursor);
            if (nextClean != cleanRanges.begin()) {
                const auto clean = std::prev(nextClean);
                if (cursor < clean->second) {
                    if (profiled) collectProfile().cleanSkips.fetch_add(1, std::memory_order_relaxed);
                    cursor = std::min(end, clean->second);
                    continue;
                }
            }
            const auto base = cursor & ~(pageBytes - 1);
            const auto found = views.find(base);
            if (found != views.end() && found->second.resident) {
                const auto run = std::prev(residentRuns.upper_bound(cursor));
                auto& state = run->second;
                const auto stop = nextClean != cleanRanges.end() ? std::min({end, state.end, nextClean->first}) : std::min(end, state.end);
                const auto firstBit = (cursor - run->first) >> 12;
                const auto lastBit = (stop - 1 - run->first) >> 12;
                bool fresh = false;
                for (auto bit = firstBit; state.fresh != 0 && bit <= lastBit && !fresh; ++bit) fresh = (state.collected[bit >> 6] >> (bit & 63) & 1) == 0;
                if (profiled) {
                    collectProfile().residentRuns.fetch_add(1, std::memory_order_relaxed);
                    if (fresh) collectProfile().residentFresh.fetch_add(1, std::memory_order_relaxed);
                }
                if (fresh) {
                    if (lastBit - firstBit + 1 > capacity - *count) {
                        for (auto at = cursor; *count < capacity; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                        return true;
                    }
                    for (auto at = cursor; at < stop; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                    if (clear) {
                        if (watchResident && ResetWriteWatch(reinterpret_cast<void*>(cursor), stop - cursor) != 0) fail("reset resident guest writes");
                        for (auto bit = firstBit; bit <= lastBit; ++bit) {
                            auto& word = state.collected[bit >> 6];
                            if ((word >> (bit & 63) & 1) != 0) continue;
                            word |= 1ull << (bit & 63);
                            --state.fresh;
                        }
                    }
                } else {
                    // At most the segment's pages: GetWriteWatch's cost grows with the output count
                    // it is given (a 65,536-entry buffer costs ~7 us per call against ~0.3 us for a
                    // run's 48 pages), and it cannot report more pages than the range holds.
                    ULONG_PTR available = std::min<ULONG_PTR>(capacity - *count, (((stop + 4095) & ~std::uintptr_t{4095}) - (cursor & ~std::uintptr_t{4095})) / 4096);
                    if (available == 0) return true;
                    DWORD granularity = 0;
                    const auto before = *count;
                    const DWORD flags = clear ? WRITE_WATCH_FLAG_RESET : 0;
                    if (unlockedWatch() && watchResident && !armResident() && !retryLocked) {
                        // APS5_COLLECT_UNLOCKED_WATCH=1: the kernel call over a settled run needs
                        // nothing of this object but the addresses, so the mutex is released for it
                        // and collects of other threads (and the mappers) run meanwhile. A run
                        // converted meanwhile (cut, shared, evicted, unmapped) makes the call fail,
                        // and the segment is resolved again under the mutex; a call that succeeded
                        // on the old allocation reported its writes. The next segment is looked up
                        // afresh after relocking.
                        lock.unlock();
                        const auto result = watch(flags, cursor, stop - cursor, pages + *count, &available, &granularity);
                        lock.lock();
                        if (result != 0) {
                            retryLocked = true;
                            if (profiled) collectProfile().unlockedRetries.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                    } else if (watch(flags, cursor, stop - cursor, pages + *count, &available, &granularity) != 0) {
                        fail("collect resident guest writes");
                    }
                    retryLocked = false;
                    *count += available;
                    if (*count == capacity) return true;
                    if (clear && watchResident) armClean(cursor, stop, pages + before, *count - before);
                }
                cursor = stop;
            } else if (found != views.end()) {
                auto& view = found->second;
                const auto stop = std::min(end, base + pageBytes);
                if (profiled) collectProfile().viewPages.fetch_add(1, std::memory_order_relaxed);
                if (view.protection == PAGE_NOACCESS) return false;
                // APS5_NO_ARM_SHARED=1: shared pages are never write-protected and every collect
                // reports them written, as for a pinned page. A tracking fault is dispatched on the
                // faulting guest thread's stack, and Windows writes its exception frame over the
                // 128 bytes below rsp that SysV guest code keeps locals in (the red zone): PPSA01317's
                // BVH jobs then read a clobbered pointer (app.exe+0x169a7b8) after a fault in the
                // same frame.
                static const bool noArmShared = std::getenv("APS5_NO_ARM_SHARED") != nullptr;
                // APS5_HASH_SHARED=1: shared pages are not write-protected either (no tracking
                // faults, so no red-zone clobbering), but a collect reports a page written only when
                // a host write moved its generation or its content hash differs from the one taken
                // when it was last consumed, so unchanged pages are not uploaded again.
                static const bool hashShared = std::getenv("APS5_HASH_SHARED") != nullptr;
                if (hashShared && view.page->pins == 0) {
                    const auto hash = HashPage(base);
                    if (!view.hashed || hash != view.contentHash || view.seen != view.page->generation) {
                        const auto needed = (stop - cursor + 4095) / 4096;
                        if (needed > capacity - *count) {
                            for (auto at = cursor; *count < capacity; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                            return true;
                        }
                        for (auto at = cursor; at < stop; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                    }
                    if (clear) {
                        view.contentHash = hash;
                        view.hashed = true;
                        view.seen = view.page->generation;
                    }
                    cursor = stop;
                    continue;
                }
                const bool pinned = view.page->pins != 0 || noArmShared;
                if (pinned || view.seen != view.page->generation) {
                    const auto needed = (stop - cursor + 4095) / 4096;
                    if (needed > capacity - *count) {
                        for (auto at = cursor; *count < capacity; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                        return true;
                    }
                    for (auto at = cursor; at < stop; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                }
                if (clear && !pinned) {
                    for (const auto alias : view.page->aliases) {
                        auto& other = views.at(alias);
                        if (!writable(other.protection) || other.armed || other.hostWrites != 0) continue;
                        DWORD previous;
                        const DWORD protection = other.protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY;
                        if (!VirtualProtect(reinterpret_cast<void*>(alias), pageBytes, protection, &previous)) fail("arm shared memory write tracking");
                        other.armed = true;
                        if (profiled) collectProfile().viewArms.fetch_add(1, std::memory_order_relaxed);
                    }
                    view.seen = view.page->generation;
                    rememberClean(base, base + pageBytes);
                }
                cursor = stop;
            } else {
                const auto queryStart = profiled ? profileNow() : 0;
                const auto memory = query(cursor);
                if (profiled) {
                    collectProfile().privateRegions.fetch_add(1, std::memory_order_relaxed);
                    collectProfile().queryTicks.fetch_add(static_cast<std::uint64_t>(profileNow() - queryStart), std::memory_order_relaxed);
                }
                if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE) return false;
                const auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                ULONG_PTR available = std::min<ULONG_PTR>(capacity - *count, (((stop + 4095) & ~std::uintptr_t{4095}) - (cursor & ~std::uintptr_t{4095})) / 4096);
                if (available == 0) return true;
                DWORD granularity = 0;
                if (watch(clear ? WRITE_WATCH_FLAG_RESET : 0, cursor, stop - cursor, pages + *count, &available, &granularity) != 0) fail("collect private guest writes");
                *count += available;
                if (*count == capacity) return true;
                cursor = stop;
            }
        }
        return true;
    }

private:
    struct Conversion {
        std::uintptr_t start;
        std::uintptr_t end;
        std::uint64_t when;
        unsigned long thread;
        std::uint32_t kind;
    };
    static constexpr std::size_t ConversionLog = 1024;
    std::array<Conversion, ConversionLog> conversions{};
    std::atomic<std::uint64_t> conversionCount{0};
    void note(std::uint32_t kind, std::uintptr_t start, std::uintptr_t end) {
        const auto index = conversionCount.load(std::memory_order_relaxed);
        conversions[index % ConversionLog] = Conversion{start, end, GetTickCount64(), GetCurrentThreadId(), kind};
        conversionCount.store(index + 1, std::memory_order_relaxed);
    }
    static constexpr std::size_t pageBytes = 0x4000;
    static constexpr std::size_t chunkBytes = 0x200000;
    struct SharedPage {
        std::uint64_t generation = 1;
        std::vector<std::uintptr_t> aliases;
        std::uint32_t pins = 0;
    };
    struct Section {
        HANDLE handle;
        explicit Section(HANDLE handle) : handle(handle) {}
        Section(const Section&) = delete;
        Section& operator=(const Section&) = delete;
        ~Section() { CloseHandle(handle); }
    };
    struct Physical {
        std::weak_ptr<SharedPage> page;
        bool stored = false;
    };
    struct View {
        std::shared_ptr<SharedPage> page;
        DWORD protection;
        std::uint64_t seen;
        bool armed;
        std::shared_ptr<Section> section;
        std::uint64_t offset;
        std::uint32_t hostWrites;
        bool resident;
        Physical* slot;
        std::uint8_t cleanStreak = 0;
        std::uint8_t armAfter = InitialArmAfter();
        // APS5_HASH_SHARED: the page's content when it was last consumed (hashed = taken).
        std::uint64_t contentHash = 0;
        bool hashed = false;
    };
    // A 64-bit hash of one guest page, four independent lanes so it runs near memory speed.
    static std::uint64_t HashPage(std::uintptr_t base) {
        const auto* words = reinterpret_cast<const std::uint64_t*>(base);
        std::uint64_t lanes[4] = {0x9e3779b97f4a7c15ull, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0x27d4eb2f165667c5ull};
        for (std::size_t i = 0; i < pageBytes / sizeof(std::uint64_t); i += 4) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
                lanes[lane] = (lanes[lane] ^ words[i + lane]) * 0x100000001b3ull;
                lanes[lane] ^= lanes[lane] >> 29;
            }
        }
        return lanes[0] ^ (lanes[1] * 3u) ^ (lanes[2] * 5u) ^ (lanes[3] * 7u);
    }
    static std::uint8_t InitialArmAfter() {
        static const std::uint8_t value = [] {
            const char* text = std::getenv("APS5_ARM_AFTER");
            const auto parsed = text != nullptr ? std::strtoul(text, nullptr, 10) : 64ul;
            return static_cast<std::uint8_t>(std::clamp<unsigned long>(parsed, 1ul, 255ul));
        }();
        return value;
    }
    struct Run {
        std::uintptr_t end;
        std::array<std::uint64_t, 8> collected;
        std::uint32_t fresh;
    };
    struct Window {
        void* view = nullptr;
        unsigned char* bytes = nullptr;
    };

    Window open(HANDLE section, std::uint64_t offset, std::size_t bytes) {
        const auto lead = offset % granularity;
        Window window;
        window.view = map(section, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (window.view == nullptr) fail("open a window on direct memory");
        window.bytes = static_cast<unsigned char*>(window.view) + lead;
        return window;
    }

    void close(Window& window) {
        if (window.view != nullptr && !unmap(GetCurrentProcess(), window.view, 0)) fail("close a window on direct memory");
        window = {};
    }

    static void commitSection(void* page) {
        if (!VirtualAlloc(page, pageBytes, MEM_COMMIT, PAGE_READWRITE)) fail("commit a direct memory page");
    }

    void commitResident(std::uintptr_t base, std::size_t bytes) {
        const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watchResident ? MEM_WRITE_WATCH : 0);
        if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(base), bytes, flags, PAGE_READWRITE, nullptr, 0)) fail("replace guest placeholder with resident direct memory");
    }

    void mapShared(std::uintptr_t base, HANDLE section, std::uint64_t offset, DWORD protection) {
        void* page = reinterpret_cast<void*>(base);
        if (!map(section, GetCurrentProcess(), page, offset, pageBytes, MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0)) fail("map shared guest page");
        if (!VirtualAlloc(page, pageBytes, MEM_COMMIT, PAGE_EXECUTE_READWRITE)) fail("commit shared guest page");
        DWORD previous;
        if (!VirtualProtect(page, pageBytes, protection, &previous)) fail("protect shared guest page");
    }

    static bool blank(std::uintptr_t page) {
        const auto* words = reinterpret_cast<const std::uint64_t*>(page);
        for (std::size_t i = 0; i < pageBytes / sizeof(std::uint64_t); ++i) {
            if (words[i] != 0) return false;
        }
        return true;
    }


    // Open around an entry point that may convert resident memory; the guest threads are suspended
    // (GuestQuiesce) by the first conversion inside it and resumed when it closes.
    struct ConversionScope {
        explicit ConversionScope(WindowsMappings& owner) : owner(owner), outer(owner.conversionScope) { owner.conversionScope = this; }
        ConversionScope(const ConversionScope&) = delete;
        ConversionScope& operator=(const ConversionScope&) = delete;
        ~ConversionScope() {
            owner.conversionScope = outer;
            delete quiesce;
        }
        WindowsMappings& owner;
        ConversionScope* outer;
        GuestQuiesce* quiesce = nullptr;
    };
    ConversionScope* conversionScope = nullptr;
    // Time spent suspending guest threads (GuestQuiesce), for the slow-map report.
    double quiesceMs = 0.0;
    struct QuiesceStats {
        std::uint64_t maps = 0;
        std::uint64_t shares = 0;
        double shareMs = 0.0;
        double mapMs = 0.0;
        std::uint64_t count = 0;
        double ms = 0.0;
        double maxMs = 0.0;
        std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    } quiesceStats;
    void quiesceForConversion() {
        // The outermost open scope holds the suspension (the GuestQuiesce is new'd before any
        // thread is suspended, so the heap is free then).
        auto* scope = conversionScope;
        while (scope != nullptr && scope->outer != nullptr) scope = scope->outer;
        if (scope != nullptr && scope->quiesce == nullptr) {
            const auto start = std::chrono::steady_clock::now();
            scope->quiesce = new GuestQuiesce;
            const auto now = std::chrono::steady_clock::now();
            const auto ms = std::chrono::duration<double, std::milli>(now - start).count();
            quiesceMs += ms;
            // Every 10 s: how many conversions suspended the guest, and what that cost.
            ++quiesceStats.count;
            quiesceStats.ms += ms;
            quiesceStats.maxMs = std::max(quiesceStats.maxMs, ms);
        }
    }
    void cut(std::uintptr_t at) {
        auto it = residentRuns.upper_bound(at);
        if (it == residentRuns.begin()) return;
        --it;
        const auto start = it->first;
        const auto stop = it->second.end;
        if (at <= start || at >= stop) return;
        quiesceForConversion();
        note(4, start, stop);
        disarm(start, stop);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READONLY, &previous)) fail("hold resident guest pages for a split");
        const std::vector<unsigned char> bytes(reinterpret_cast<const unsigned char*>(start), reinterpret_cast<const unsigned char*>(stop));
        if (!VirtualFree(reinterpret_cast<void*>(start), stop - start, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release resident guest pages");
        if (!VirtualFree(reinterpret_cast<void*>(start), at - start, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split resident guest placeholder");
        commitResident(start, at - start);
        commitResident(at, stop - at);
        std::memcpy(reinterpret_cast<void*>(start), bytes.data(), bytes.size());
        for (auto page = start; page < stop; page += pageBytes) {
            const auto protection = views.at(page).protection;
            if (protection != PAGE_READWRITE && !VirtualProtect(reinterpret_cast<void*>(page), pageBytes, protection, &previous)) fail("protect resident guest pages");
        }
        residentRuns.erase(it);
        residentRuns.emplace(start, Run{at, {}, static_cast<std::uint32_t>((at - start) >> 12)});
        residentRuns.emplace(at, Run{stop, {}, static_cast<std::uint32_t>((stop - at) >> 12)});
    }

    void share(std::uintptr_t base) {
        const auto shareStart = std::chrono::steady_clock::now();
        struct Timed {
            QuiesceStats& stats;
            std::chrono::steady_clock::time_point start;
            ~Timed() {
                ++stats.shares;
                stats.shareMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            }
        } timed{quiesceStats, shareStart};
        quiesceForConversion();
        cut(base);
        cut(base + pageBytes);
        note(5, base, base + pageBytes);
        auto& view = views.at(base);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(base), pageBytes, PAGE_READONLY, &previous)) fail("hold a resident guest page for sharing");
        auto window = open(view.section->handle, view.offset, pageBytes);
        commitSection(window.bytes);
        std::memcpy(window.bytes, reinterpret_cast<const void*>(base), pageBytes);
        close(window);
        view.slot->stored = true;
        if (!VirtualFree(reinterpret_cast<void*>(base), pageBytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release a resident guest page");
        residentRuns.erase(base);
        mapShared(base, view.section->handle, view.offset, view.protection);
        view.resident = false;
        view.seen = 0;
        view.armed = false;
        invalidate(*view.page);
    }

    void evict(std::map<std::uintptr_t, Run>::iterator run, bool live = true) {
        const auto start = run->first;
        const auto stop = run->second.end;
        // A run the guest is unmapping (reset) is not used by guest code meanwhile: no quiesce.
        if (live) quiesceForConversion();
        note(6, start, stop);
        disarm(start, stop);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READWRITE, &previous)) fail("open resident guest pages");
        const auto section = views.at(start).section;
        auto window = open(section->handle, views.at(start).offset, stop - start);
        for (auto page = start; page < stop; page += pageBytes) {
            const auto found = views.find(page);
            auto& view = found->second;
            if (view.slot->stored || !blank(page)) {
                commitSection(window.bytes + (page - start));
                std::memcpy(window.bytes + (page - start), reinterpret_cast<const void*>(page), pageBytes);
                view.slot->stored = true;
            }
            std::erase(view.page->aliases, page);
            views.erase(found);
        }
        close(window);
        residentRuns.erase(run);
    }
    static bool armResident() {
        static const bool enabled = std::getenv("APS5_NO_ARM_RESIDENT") == nullptr;
        return enabled;
    }

    void disarm(std::uintptr_t start, std::uintptr_t stop) {
        for (auto it = views.lower_bound(start); it != views.end() && it->first < stop; ++it) {
            if (!it->second.armed || !it->second.resident) continue;
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(it->first), pageBytes, it->second.protection, &previous)) fail("disarm resident guest page");
            it->second.armed = false;
            it->second.cleanStreak = 0;
        }
        forgetClean(start, stop);
    }

    void armClean(std::uintptr_t begin, std::uintptr_t end, void* const* written, std::size_t count) {
        if (!armResident()) return;
        std::vector<std::uintptr_t> armed;
        for (auto base = (begin + pageBytes - 1) & ~(pageBytes - 1); base + pageBytes <= end; base += pageBytes) {
            const auto found = views.find(base);
            if (found == views.end() || !found->second.resident) continue;
            auto& view = found->second;
            const bool dirty = std::any_of(written, written + count, [&](void* page) { const auto at = reinterpret_cast<std::uintptr_t>(page); return at >= base && at < base + pageBytes; });
            if (dirty) {
                view.cleanStreak = 0;
                continue;
            }
            if (view.armed || view.cleanStreak < 255) ++view.cleanStreak;
            static const bool armPinned = std::getenv("APS5_ARM_PINNED") != nullptr;
            if (view.armed || view.cleanStreak < view.armAfter || !writable(view.protection) || view.hostWrites != 0 || (view.page->pins != 0 && !armPinned)) continue;
            DWORD previous;
            const DWORD protection = view.protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY;
            if (!VirtualProtect(reinterpret_cast<void*>(base), pageBytes, protection, &previous)) fail("arm resident guest write tracking");
            view.armed = true;
            armed.push_back(base);
        }
        for (const auto base : armed) {
            std::array<void*, pageBytes / 4096> raced{};
            ULONG_PTR found = raced.size();
            DWORD granularity = 0;
            if (GetWriteWatch(0, reinterpret_cast<void*>(base), pageBytes, raced.data(), &found, &granularity) != 0) fail("recheck armed resident guest writes");
            auto& view = views.at(base);
            if (found != 0) {
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(base), pageBytes, view.protection, &previous)) fail("disarm raced resident guest page");
                view.armed = false;
                view.cleanStreak = 0;
                continue;
            }
            rememberClean(base, base + pageBytes);
        }
    }

    void forgetClean(std::uintptr_t start, std::uintptr_t end) {
        auto it = cleanRanges.lower_bound(start);
        if (it != cleanRanges.begin() && std::prev(it)->second > start) --it;
        while (it != cleanRanges.end() && it->first < end) {
            const auto first = it->first;
            const auto last = it->second;
            it = cleanRanges.erase(it);
            if (first < start) cleanRanges.emplace(first, start);
            if (last > end) it = cleanRanges.emplace(end, last).first;
        }
    }

    void rememberClean(std::uintptr_t start, std::uintptr_t end) {
        auto it = cleanRanges.lower_bound(start);
        if (it != cleanRanges.begin() && std::prev(it)->second >= start) --it;
        while (it != cleanRanges.end() && it->first <= end) {
            start = std::min(start, it->first);
            end = std::max(end, it->second);
            it = cleanRanges.erase(it);
        }
        cleanRanges.emplace(start, end);
    }

    void invalidate(SharedPage& page) {
        ++page.generation;
        for (const auto alias : page.aliases) forgetClean(alias, alias + pageBytes);
    }

    bool sameSection(HANDLE first, HANDLE second) const {
        return compare != nullptr && compare(first, second);
    }

    static bool writable(DWORD protection) {
        return protection == PAGE_READWRITE || protection == PAGE_EXECUTE_READWRITE;
    }
    using AllocateFunction = PVOID (WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using MapFunction = PVOID (WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using UnmapFunction = BOOL (WINAPI*)(HANDLE, PVOID, ULONG);
    using CompareFunction = BOOL (WINAPI*)(HANDLE, HANDLE);

    WindowsMappings() {
        const auto module = GetModuleHandleW(L"KernelBase.dll");
        if (!module) fail("load Windows memory API");
        allocate = reinterpret_cast<AllocateFunction>(GetProcAddress(module, "VirtualAlloc2"));
        map = reinterpret_cast<MapFunction>(GetProcAddress(module, "MapViewOfFile3"));
        unmap = reinterpret_cast<UnmapFunction>(GetProcAddress(module, "UnmapViewOfFile2"));
        if (!allocate || !map || !unmap) throw std::runtime_error("Windows placeholder memory APIs are required");
        compare = reinterpret_cast<CompareFunction>(GetProcAddress(module, "CompareObjectHandles"));
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        granularity = system.dwAllocationGranularity;
    }

    [[noreturn]] static void fail(const char* operation) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
    }

    static MEMORY_BASIC_INFORMATION query(std::uintptr_t address) {
        MEMORY_BASIC_INFORMATION memory{};
        if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) != sizeof(memory)) fail("query guest memory");
        return memory;
    }

    static void split(std::uintptr_t address, std::size_t bytes) {
        auto memory = query(address);
        memory = query(reinterpret_cast<std::uintptr_t>(memory.AllocationBase));
        if (memory.State != MEM_RESERVE) throw std::runtime_error("guest mapping requires a placeholder");
        const auto base = reinterpret_cast<std::uintptr_t>(memory.BaseAddress);
        if (address != base) {
            if (!VirtualFree(reinterpret_cast<void*>(base), address - base, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder prefix");
            memory = query(address);
        }
        if (memory.RegionSize < bytes) throw std::runtime_error("guest placeholder is too small");
        if (memory.RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder suffix");
    }

    bool sameMapping(std::uintptr_t address, std::size_t bytes, HANDLE section, std::uint64_t offset, DWORD protection) const {
        if (bytes == 0 || address % pageBytes != 0 || bytes % pageBytes != 0) return false;
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            const auto view = views.find(address + done);
            if (view == views.end() || view->second.protection != protection) return false;
            const auto slot = physical.find(std::make_pair(reinterpret_cast<std::uintptr_t>(section), offset + done));
            if (slot == physical.end() || view->second.slot != &slot->second) return false;
        }
        return true;
    }

    void reset(std::uintptr_t address, std::size_t bytes) {
        const auto end = address + bytes;
        forgetClean(address, end);
        cut(address);
        cut(end);
        for (auto cursor = address; cursor < end;) {
            const auto memory = query(cursor);
            if (memory.State == MEM_RESERVE) {
                cursor = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                continue;
            }
            if (reinterpret_cast<std::uintptr_t>(memory.AllocationBase) != cursor) throw std::runtime_error("cannot release part of a host allocation");
            auto allocationEnd = cursor;
            do {
                const auto part = query(allocationEnd);
                if (part.AllocationBase != memory.AllocationBase) break;
                allocationEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            } while (allocationEnd < end);
            if (allocationEnd > end || query(allocationEnd).AllocationBase == memory.AllocationBase) throw std::runtime_error("guest release truncates a host allocation");
            note(memory.Type == MEM_MAPPED ? 7 : 8, cursor, allocationEnd);
            if (memory.Type == MEM_MAPPED) {
                if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(cursor), MEM_PRESERVE_PLACEHOLDER)) fail("unmap shared guest page");
                const auto found = views.find(cursor);
                if (found != views.end()) {
                    std::erase(found->second.page->aliases, cursor);
                    views.erase(found);
                }
            } else if (memory.Type == MEM_PRIVATE) {
                if (const auto run = residentRuns.find(cursor); run != residentRuns.end()) evict(run, false);
                if (!VirtualFree(reinterpret_cast<void*>(cursor), allocationEnd - cursor, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release private guest memory");
            } else {
                throw std::runtime_error("unsupported guest mapping type");
            }
            cursor = allocationEnd;
        }
        const auto last = query(reinterpret_cast<std::uintptr_t>(query(end - 1).AllocationBase));
        const auto lastBase = reinterpret_cast<std::uintptr_t>(last.BaseAddress);
        if (lastBase + last.RegionSize > end) split(lastBase, end - lastBase);
        const auto first = query(address);
        split(address, std::min(bytes, reinterpret_cast<std::uintptr_t>(first.BaseAddress) + first.RegionSize - address));
        if (query(address).RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) fail("coalesce guest placeholders");
    }

    std::map<std::uintptr_t, std::uintptr_t> cleanRanges;
    std::map<std::uintptr_t, View> views;
    std::map<std::pair<std::uintptr_t, std::uint64_t>, Physical> physical;
    std::map<std::uintptr_t, Run> residentRuns;
    std::size_t granularity = 0x10000;
    bool watchResident = false;
    std::mutex mutex;
    AllocateFunction allocate = nullptr;
    MapFunction map = nullptr;
    UnmapFunction unmap = nullptr;
    CompareFunction compare = nullptr;
};

}
#endif

#endif
