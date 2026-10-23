# PPSA01317 bleeding-edge branch: how to run it

This branch is an experimental performance branch for PPSA01317 and is not meant for upstream. Many of its changes are opt-in, so a plain launch is slower than our measurements. Set the variables below to match the setup that reached about 19 FPS on the World Map with a warm shader cache.

## Environment variables

| Variable | Value | What it does |
|---|---|---|
| `APS5_NET_OFFLINE` | `1` | Makes network calls fail at once. Without it the title waits minutes at boot for server responses. |
| `APS5_NO_PIPELINE_JOURNAL` | `1` | Turns off the pipeline journal replay at boot (it costs FPS while it runs). |
| `APS5_OPERATION_SUBMODE` | `1` | Makes the title use its native 1920x1080 mode instead of 4K (World Map about 19 -> 21.5 FPS on our machine). Leave it unset for 4K. |
| `APS5_LIVE_FLAGS` | path to `live-flags.txt` | Live switches, re-read while the game runs. Copy `docs/ppsa01317/live-flags.txt` next to the game and point this at it. |
| `APS5_ADDRESS_TEMPLATES` | `1` | |
| `APS5_RESOURCE_CACHE_ENTRIES` | `8192` | |
| `APS5_SPLIT_DRAWS` | `6` | |
| `APS5_SPLIT_DRAW_FRONT` | `1` | |
| `APS5_MOVABLE_IMAGES` | `1` | |
| `APS5_SHARED_EPOCH_MEMO` | `1` | |
| `APS5_PIPELINED_DISPATCHES` | `1` | |
| `APS5_PRESENT_MODE` | `mailbox` | |
| `APS5_FLIP_EARLY` | `1` | |
| `APS5_BDA_HINT_FAST` | `1` | |
| `APS5_BDA_INLINE_HIT` | `1` | |
| `APS5_FAST_HEAP` | `1` | |
| `APS5_NO_ARM_RESIDENT` | `1` | |
| `APS5_HOST_IMPORT_MIB` | `12288` | Host-import budget in MiB. Lower it on machines with less than 32 GB of RAM. |
| `APS5_RECORDER_THREAD` | `1` | |
| `APS5_STARVE_SUBMIT` | `4` | |
| `APS5_VOLUME_ALIAS_DISCARD` | `1` | |
| `APS5_RT_SLICE_LAYERS` | `1` | |
| `APS5_MOUSE_SPACE` | `1920x1080` | Only needed for our automation scripts. |
| `SDL_AUDIODRIVER` | `dummy` | Mutes audio (used for measurements). Leave it unset to hear the game. |

The unlabelled rows are performance switches from earlier experiments. Each is described where it is read in the code.

## PowerShell launch example

```powershell
$env:APS5_NET_OFFLINE = '1'
$env:APS5_NO_PIPELINE_JOURNAL = '1'
$env:APS5_OPERATION_SUBMODE = '1'
$env:APS5_LIVE_FLAGS = "$PWD\live-flags.txt"
$env:APS5_ADDRESS_TEMPLATES = '1'
$env:APS5_RESOURCE_CACHE_ENTRIES = '8192'
$env:APS5_SPLIT_DRAWS = '6'
$env:APS5_SPLIT_DRAW_FRONT = '1'
$env:APS5_MOVABLE_IMAGES = '1'
$env:APS5_SHARED_EPOCH_MEMO = '1'
$env:APS5_PIPELINED_DISPATCHES = '1'
$env:APS5_PRESENT_MODE = 'mailbox'
$env:APS5_FLIP_EARLY = '1'
$env:APS5_BDA_HINT_FAST = '1'
$env:APS5_BDA_INLINE_HIT = '1'
$env:APS5_FAST_HEAP = '1'
$env:APS5_NO_ARM_RESIDENT = '1'
$env:APS5_HOST_IMPORT_MIB = '12288'
$env:APS5_RECORDER_THREAD = '1'
$env:APS5_STARVE_SUBMIT = '4'
$env:APS5_VOLUME_ALIAS_DISCARD = '1'
$env:APS5_RT_SLICE_LAYERS = '1'
& .\app.exe 1> run-out.log 2> run-err.log
```

The window title shows the current FPS.

## Shader precompilation and graphics settings

- `APS5_PRECOMPILE=1` holds the title at its first GPU submission until the shader warm list and the pipeline journal have been replayed on all cores (normal priority, no yielding). Every shader and pipeline an earlier session met is then built before the first frame instead of stalling a draw. On an i9-13900HX with an RTX 4090-class GPU that took 71 s for 13,229 variants, 21,445 modules and 2,242 pipelines. `[precompile]` lines in `run-err.log` report the progress. `APS5_PRECOMPILE_MAX_S` bounds the wait (1800 s by default).
- `APS5_PRESENT_FILTER=nearest` scales the frame to the window without filtering.
- `APS5_TEXTURE_CACHE_MIB` sets the sampled-texture cache budget (2048 by default). On licence B-1, 2 GiB evicted most textures before their next use, so they were re-read and re-uploaded every frame. 6144 cut the snapshot reads by about 6x and raised the FPS from 1.24 to 1.55 (RTX 4090 Laptop, 16 GB VRAM, 64 GB RAM). `[textures] new sampled textures by cause` reports the misses.
- `APS5_NO_ARM_SHARED=1 APS5_HASH_SHARED=1`: shared guest pages are never write-protected, so no tracking fault clobbers a guest thread's red zone (the `Job#` crash at `app.exe+0x169a7b8`, seen in three of five boots without it). A collect reports a page written only when its content hash changed, so unchanged pages are not re-uploaded. On the licence B-1 briefing, `NO_ARM_SHARED` alone ran at 1.24 FPS and both together at 1.69, with no crash.
- `gt7-settings.ps1` is a small dialog with presets for resolution (`APS5_OPERATION_SUBMODE`), presentation mode, 120 Hz output, precompilation, geometry in VRAM, the image pool, the host-import budget and the working set. It writes `graphics-settings.json` next to the launcher, which applies it as environment defaults. DLSS is listed but disabled, because it needs the title's motion vectors, which the port does not extract yet.

## Notes

- The first boot compiles shaders and is slow. FPS settles after a couple of minutes on the World Map, and later boots start warm from the saved pipeline cache.
- Newer switches (lazy write-back, native pattern fill, and the committer cuts `room_spin`, `fast_heap_prefetch`, `lock_spin`, `pending_fill`, `index_scan_memo`, `window_early_out`) are on by default. Setting one to `0` in `live-flags.txt` turns it off while the game runs.
- The bs53 committer cuts are on by default too: `unchanged_lockfree`, `collect_local_copy`, `spec_entry_stamp`, `stream_batch`, `stream_inline` and `import_memo`. Each has an env switch (`APS5_UNCHANGED_LOCKFREE`, `APS5_COLLECT_LOCAL_COPY`, `APS5_SPEC_ENTRY_STAMP`, `APS5_STREAM_BATCH`, `APS5_STREAM_INLINE`, `APS5_IMPORT_MEMO`) whose value `0` turns it off from the start. The live flags take effect while the game runs.
- `APS5_OPERATION_SUBMODE=1` makes `sceKernelGetOperationMode` report submode 1, which makes PPSA01317 render at its native 1920x1080 (no binary patch needed). The default stays 0.
- Results were measured on an NVIDIA GPU at 4K. Other GPUs are untested.

AI-assisted: yes

## Known issues

- **Random crashes, usually within the first few minutes:** the title sometimes crashes inside its own memory allocator. The likely cause is the GPU copy-back of staged buffer ranges, which copies a whole range back and can undo CPU writes made while the GPU work was running. A fix that copies back only the bytes the GPU changed exists, but it currently costs about half the frame rate on the World Map, so it isn't on this branch yet. If the game crashes, relaunch it.
