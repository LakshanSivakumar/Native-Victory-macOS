---
name: debug-graphics
description: Diagnose rendering problems on the native Metal host (black, green or flickering frames, boxed text, missing geometry, unknown formats) with the VICTORY_* trace and dump switches. Use when the image is wrong or a draw seems missing.
---

# Debug rendering on the native host

Always start from evidence: dump what was drawn, then form one hypothesis. Keep dumps outside the repo, and never commit or upload them; they contain game content.

## Switches (all off by default)

| Variable | What it does |
|---|---|
| `VICTORY_GUESS_GRAPHICS=1` | Builds Metal pipelines for draws (needed for anything to draw) |
| `VICTORY_PRESENT_PLANE=<n>` | Shows only plane n, counted from 0, disabling native two-plane compositing. Out-of-range indices fall back to the largest plane. |
| `VICTORY_PRESENT_TOP=0` | Swaps the native composite order. Default plane 1 (UI) over plane 0 (scene) is a guess about console order based on the measured run. Both planes scale linearly; output alpha is 1 even when scene alpha is 0. |
| `VICTORY_PRESENT_PREMULTIPLIED=1` | Uses `bottom.rgb * (1 - top.a) + top.rgb`; default straight alpha uses `bottom.rgb * (1 - top.a) + top.rgb * top.a`. Ignored for a single-plane present. |
| `VICTORY_TRACE_TARGETS=1` | Logs per present which textures were drawn to, cleared or lost draws (`dropped`), which was presented, the work dropped since the last present by reason (`reasons=`), and each pipeline whose draws, dispatches or indirect executions were dropped (`draws dropped (no host pipeline, or an indirect execution)`, with its shaders and targets, or for a dispatch its group counts or `indirect`: 160x90 groups is a 1280x720 pass of 8x8 threads). Needed for the dumps below. |
| `VICTORY_DUMP_TARGETS=<dir>,<seconds>[,<every>]` | Writes the 8-bit colour targets as PPM files from that second (repeating every N s): `t<n>_<second>_<address>.ppm`. The presented image is the swapchain address. HDR and 10-bit targets (DXGI 2, 10, 24, 26) are tone-mapped into `..._f<format>.ppm`, with their maximum, mean and lit fraction logged (`target dumped`), so a black pass shows in the log without opening an image. |
| `VICTORY_TRACE_PASSES=<dir>,<skip>,<presents>` | After `<skip>` world presents, every draw, dispatch, clear and copy of the next `<presents>` frames in queue order (`PASS`: shaders, targets, views read and written, `ran` or why dropped), commands the submission could not prepare (`PASSFAIL`), uploads over GPU-written textures (`PASSUPLOAD`), and at the last present each named texture's statistics (`PASSTEX`) with a `.ppm` in `<dir>`. Walk backwards from the swapchain to the first pass that writes nothing. Details in docs/native-host.md. |
| `VICTORY_TRACE_CONTENT=<dir>,<seconds>,<format>` | Traces only draws into targets of one format, and dumps A8/R8 textures as grey images |
| `VICTORY_TRACE_FLOW=<seconds>[,<frames>]` with `VICTORY_TRACE_FLOW_DRAWS=1` | "flow: draw" lines for every draw into HDR targets in the captured frames (shaders, call, targets, views, pipeline state), then statistics of up to 400 distinct textures they named, including placed keys resolved through aliases in the placed format: which pass leaves the HDR target dark |
| `VICTORY_TRACE_FLOW_SNAP=<hash>,...` with `VICTORY_TRACE_FLOW` | GPU copies after captured indirect draws 1, 30, 60, 90 and 120 (`VICTORY_TRACE_FLOW_SNAP_AT=<n>,...` selects up to 16 draw numbers). Compute hashes copy up to four distinct table UAV textures before and after the first captured dispatch per hash, direct or indirect (the first record, with GPU count gating preserved), including placed views and thread-count dispatches: level 0 / slice 0, at most `1 << 21` texels, no depth or block formats. Copies split compute batches into ordered blit passes. "flow: target as the GPU had it" reports before/after statistics, HDR maximum and grid; dropped/refused or zero-sized direct dispatches log "flow: snapshot dispatch was not encoded". Compute dispatches do not count toward draw snapshots |
| `VICTORY_TRACE_FLOW_TEXTURES=<hash>,...` with `VICTORY_TRACE_FLOW` | Adds the named compute shaders' textures to "flow: late draw texture", including `sN.M placed tex 0x<key> at 0x<address> ...` and UAV views, sharing the 400-texture cap with draws. Aliases use the placed format and the view's level-0 extent |
| `VICTORY_DEBUG_FILL_INPUT=<hash>:<root slot>.<entry>:<byte>` | Fills a compute shader's input texture (the table entry the flow trace prints as `s<slot>.<entry>`) with a byte before each of its dispatches: a stand-in for a pass the host does not run, to test what is downstream of it |
| `VICTORY_DEBUG_DEPTH_ALWAYS=<hash>[+w],...` | Pipelines with a stage of one of these hashes draw with the depth test ALWAYS (`+w`: and write depth): tells a depth test that fails from a pixel shader that writes nothing. With the flow trace, "flow: late depth texture against its twin" then shows where those draws rasterised and the depth they had (14x8 grid, values), and "flow: target as the GPU had it" has a grid of the non-zero texels |
| `VICTORY_TRACE_FLOW_DUMP=<hash>,...` | "flow: buffer words": the constants of the compute shaders named, and of the first HDR draws of the vertex or pixel shaders named |
| `VICTORY_TRACE_RECORD_WRITERS=1` | Logs per present how many indirect mesh records were 0, which compute programs (ran or dropped) hold the record buffers, what they read and who writes that (are culling passes dropped?) |
| `VICTORY_TRACE_PIPES=1` | Logs each guessed pipeline's depth, stencil and blend state |
| `VICTORY_DUMP_DESC_DIR=<dir>` | Writes each pipeline's register image (device slot 0x1c8) to a file |
| `VICTORY_DEBUG_PS_RED=1` | Every translated pixel shader writes red (does geometry land at all?). Like the other `VICTORY_DEBUG_*` translation switches, it is part of the shader cache key, so it works on a warm cache |
| `VICTORY_DEBUG_PS=id` | Every translated pixel shader writes a colour from a hash of its code (which shader drew this pixel?). Read a UNORM target; details in docs/native-host.md |
| `VICTORY_DEBUG_VS_TRI=1` | Every vertex shader draws a full-target triangle (is the pixel shader right?) |
| `VICTORY_DEBUG_NO_DEPTH=1` | Disables the depth test |

With two native planes, `frame targets present=` and `VICTORY_TRACE_PRESENT=<every>` identify
the bottom picture; `overlay=` and `overlay_alpha=` name the top texture and alpha convention.
Target dumps remain separate textures, not the composited window. The title's measured plane 0
is a 1280x720 scene with alpha near zero, and plane 1 is 3840x2160 transparent UI. A HUD over
black when showing plane 1 alone is expected; use the default composite to see the scene too.

Convert a dump to look at it: `sips -s format png -Z 1280 t1_50_<addr>.ppm --out t1_50.png`.

## Lessons already learned (check these first)

- **Solid green:** video planes uploaded before the game wrote them. Textures written after their first upload are re-checked at each submission (`native/textures.rs`).
- **Glyphs drawn as boxes, opaque quads:** the pipeline had no blending. Blend, cull, depth, stencil and write mask come from the register image (`native/state.rs`, `Graphics::guess_from`).
- **Black content:** the guessed pipeline lacked the host root layout, or descriptor records the title copies weren't re-mirrored (`NativeHost::remirror_tables`).
- **Black present:** static samplers missing from the root layout.
- **Unknown format warning:** the title's format numbers are plain DXGI numbers; map them in `native/descriptors.rs` with a size test.

More detail, with addresses: `docs/native-host.md`.
