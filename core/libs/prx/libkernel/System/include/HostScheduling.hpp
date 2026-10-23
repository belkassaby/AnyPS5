#ifndef CORE_LIBS_PRX_LIBKERNEL_SYSTEM_INCLUDE_HOSTSCHEDULING_HPP
#define CORE_LIBS_PRX_LIBKERNEL_SYSTEM_INCLUDE_HOSTSCHEDULING_HPP

// How the host schedules the emulated title's threads (Windows): 1 ms timer resolution, no power
// throttling, and on a hybrid CPU a default preference for the performance cores.
namespace HostScheduling {

// Applied once, at the first guest thread start (not under the loader lock).
void ConfigureProcessOnce();
// The calling thread may run on every core again (background pools: shader and pipeline replay).
void AllowEveryCore();

}  // namespace HostScheduling

#endif
