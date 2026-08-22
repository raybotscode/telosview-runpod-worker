# TelosView Server-Side Processing — Implementation Plan

## Problem
The server-side processing fails because Playwright's headless Chromium can't decode video files (no codecs installed). The client-side version works perfectly because the user's browser has hardware video decoders.

## Solution
Split the pipeline:
1. **Server (Node.js + ffmpeg)**: Video → extracted JPEG frames (fast, reliable, any format)
2. **Browser worker (Playwright + WebGPU)**: JPEG frames → SfM → 3DGS training → PLY export

This way the browser worker only receives images (JPEG), never video. No video codecs needed in headless Chromium.

## Architecture

```
User uploads video
    ↓
Express API saves video to disk
    ↓
ffmpeg extracts frames (server-side, fast, reliable)
    ↓
Frames saved to disk as JPEG sequence
    ↓
Playwright launches Chromium with WebGPU
    ↓
Worker page loads, fetches frames from server
    ↓
splat.js runs: decodeFrames → solve → seed → train → export PLY
    ↓
PLY saved to disk, user gets download link
```

## What to Build

### 1. Frame extraction endpoint (server-side ffmpeg)
Add to `~/splat-test/server/src/index.js`:
- After video upload, call ffmpeg to extract frames
- Extract at ~2fps for 30s video = ~60 frames (good for SfM)
- Save to `~/splat-test/server/frames/<job-id>/` as `frame_00001.jpg`, etc.
- Use `child_process.execFile` to call ffmpeg
- Progress: report frame count as they're extracted

ffmpeg command:
```bash
ffmpeg -i input.mp4 -vf "fps=2" -q:v 2 frames/%05d.jpg
```

### 2. Worker page changes (`~/splat-test/server/worker.html`)
Instead of receiving a video file, the worker should:
- Fetch the list of frames from the server: `GET /api/projects/:id/frames`
- Load each frame as a File/Blob
- Pass them to `session.load()` (which expects File[] or Blob[])
- Skip `extractSharpFrames` entirely (frames already extracted)

### 3. Worker HTML import fix
The worker.html currently imports from `/src/index.js`. This works because the Express server serves `/src` statically from the repo root. Keep this.

### 4. Playwright integration
- Launch Chromium with WebGPU flags (already in the code)
- Navigate to `http://localhost:PORT/worker.html`
- Inject the job ID so the worker knows which frames to fetch
- Poll `window.__worker` for progress
- Extract PLY base64 when done

### 5. Static file serving
The server already serves:
- `/src` → `~/splat-test/src/` (splat.js library)
- `/telosview` → `~/splat-test/telosview/` (UI)
- `/worker.html` → `~/splat-test/server/worker.html`

Add:
- `/frames` → `~/splat-test/server/frames/` (extracted frames)

## Key Files to Modify

1. **`~/splat-test/server/src/index.js`** — Add ffmpeg frame extraction after upload, add frames endpoint, fix Playwright integration
2. **`~/splat-test/server/worker.html`** — Change from video input to frame fetching, skip extractSharpFrames

## Constraints
- Mini PC: Ryzen 3 4300U, 16GB RAM, no GPU → WebGPU uses SwiftShader (slow but works)
- ffmpeg must be installed: `sudo apt install ffmpeg` or it may already be present
- splat.js is ES modules served from `/src/index.js`
- The worker page runs in Playwright's Chromium context with WebGPU enabled
- Processing will be slow on mini PC (SwiftShader) but will be fast on M5 (real GPU)

## Testing
1. Start server: `cd ~/splat-test && node server/src/index.js`
2. Open `http://localhost:3457/telosview/server.html`
3. Create project, upload a short video (10-30s)
4. Watch progress: frames extracted → SfM → training → PLY export
5. Download the .ply file

## Notes
- The client-side version (`~/splat-test/telosview/index.html`) should still work independently
- The server-side version is at `~/splat-test/telosview/server.html`
- Both use the same splat.js library from `/src/`
- Don't break the client-side version while building the server-side one
