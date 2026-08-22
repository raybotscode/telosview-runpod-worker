# TelosView Architecture Plan — Using What We Have

## What Already Works (PROVEN)
- **Client-side splat.js** runs perfectly in-browser (tested on mobile Adreno 8xx)
- Full pipeline: video → sharp frames → SfM → 3DGS training → .ply export → viewer
- Tested with ~30s video, completed end-to-end, 33k splats, 20.4 dB PSNR
- Source: ~/splat-test (fork of arrival-space/splat.js)
- UI: ~/splat-test/telosview/index.html (client-side version)
- Repo: github.com/raybotscode/splat-test

## What We Have for Infrastructure
- **Mini PC**: Ryzen 3 4300U, 16GB RAM, wired Ethernet, no GPU
- **BOSGAME M5** (arriving soon): Ryzen AI Max 395, 128GB unified RAM, Radeon 8060S GPU
- Both run Windows + WSL

## What Failed
- Server-side processing via Playwright: headless Chromium can't decode video (no codecs)
- Running the full pipeline in headless Chrome is fragile and overengineered

## The Real Question
How do we build TelosView so that:
1. Users get a simple upload → result experience
2. Processing happens on OUR hardware (not the user's device)
3. We use the existing splat.js pipeline that already works
4. It runs on the mini PC now, scales to M5 later

## Constraints
- splat.js is browser-native (WebGPU, ES modules). It MUST run in a browser context.
- The mini PC has no GPU — WebGPU will use SwiftShader (software) which is slow but functional
- The M5 has a real GPU — WebGPU will use hardware acceleration
- We need video → frames extraction. The browser does this natively via extractSharpFrames()
- Server-side video decoding in headless Chrome fails due to missing codecs

## Possible Approaches

### A: Keep it client-side (simplest)
- The user's browser does everything (already works)
- Mini PC just serves static files
- Pro: zero server cost, already proven
- Con: user waits, quality depends on their device, can't close tab

### B: Server-side with real Chrome (not headless)
- Run Chrome with a virtual display (Xvfb) on the server
- Chrome has full codec support when running with a display
- Upload video → server saves it → Chrome opens the worker page → feeds video → processes
- Pro: uses existing splat.js code, Chrome has codecs
- Con: needs Xvfb setup, still uses SwiftShader on mini PC

### C: Hybrid — server extracts frames, browser trains
- Server uses ffmpeg to extract frames from uploaded video (no browser needed)
- Frames sent to a browser context for SfM + training
- Could use a real Chrome instance (not headless) for the GPU work
- Pro: ffmpeg handles any video format, separates concerns
- Con: more moving parts

### D: Pre-process on server, client does GPU work
- Upload video → server extracts frames with ffmpeg → stores them
- Client loads pre-extracted frames → runs SfM + training in their browser
- Pro: fast upload (smaller than video), client still does GPU work
- Con: still depends on client GPU

### E: Full server pipeline with real Chrome + virtual display
- Mini PC/M5 runs Chrome in a virtual display (Xvfb)
- Express API accepts video upload
- ffmpeg extracts frames (server-side, fast, reliable)
- Chrome (in virtual display) loads a worker page
- Frames are injected into the worker page
- splat.js runs SfM + training using the machine's GPU (SwiftShader on mini PC, real GPU on M5)
- PLY exported and stored for download
- Pro: full server-side, user just uploads and waits
- Con: more complex setup, but uses what we have

## My Recommendation
**Option E is the right architecture** for a real product:
1. ffmpeg for video → frames (reliable, fast, any format)
2. Real Chrome (not headless) for splat.js processing (has codecs + WebGPU)
3. Express API for upload/status/download
4. Works on mini PC (slow via SwiftShader), flies on M5 (real GPU)

The key insight: use ffmpeg for what it's good at (video decoding) and Chrome for what it's good at (WebGPU rendering). Don't try to make one tool do everything.

## Task
Plan the implementation for Option E. Consider:
- How to run Chrome with a virtual display on WSL/Linux
- How to inject extracted frames into a Chrome worker page
- How to track processing progress
- How to handle errors and timeouts
- File structure using the existing ~/splat-test codebase
- Migration path from mini PC to M5
