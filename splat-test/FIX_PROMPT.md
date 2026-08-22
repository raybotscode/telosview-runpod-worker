# FIX_PROMPT.md — TelosView Server Bug Fixes

## Bug 1: Jobs lost on server restart
Jobs are stored in an in-memory `Map`. When the server restarts (which happens often during development), all jobs are lost. The worker page then calls `/api/projects/:id/frames` and gets 404 because the job doesn't exist in memory anymore.

**Fix:** On startup, scan the `jobs/` directory and load all existing `.json` files into the in-memory Map. This way jobs survive restarts.

In `~/splat-test/server/src/index.js`, after the `const jobs = new Map();` line, add startup loading:

```js
// Load persisted jobs on startup
try {
  const files = await readdir(JOBS_DIR);
  for (const f of files) {
    if (!f.endsWith('.json')) continue;
    const data = JSON.parse(await readFile(join(JOBS_DIR, f), 'utf-8'));
    jobs.set(data.id, data);
  }
  console.log(`Loaded ${jobs.size} persisted jobs`);
} catch {}
```

## Bug 2: Worker timeout too short
The `page.waitForFunction` timeout is 30 seconds. On SwiftShader (software WebGPU), training takes much longer — 5-15 minutes. Even on a real GPU, SfM + training takes 2-5 minutes.

**Fix:** Change the timeout from 30 minutes (already set correctly) but also check if there's a 30-second timeout somewhere else. The main timeout is `30 * 60 * 1000` which is 30 minutes — that should be fine. But look for any other `waitForFunction` calls with shorter timeouts.

Actually, looking at the error: `page.waitForFunction: Timeout 30000ms exceeded` — that's 30 seconds, not 30 minutes. There must be a second waitForFunction call with a 30s default timeout. The `waitForFunction` for `window.__worker?.status === 'ready'` has `{ timeout: 10000 }` which is fine. But the one waiting for completion might be using the Playwright default (30s) instead of the explicit timeout.

**Fix:** Make sure ALL `waitForFunction` calls have explicit, generous timeouts. The completion wait should be at least 10 minutes:

```js
await page.waitForFunction(
  () => window.__worker?.status === 'done' || window.__worker?.status === 'error',
  { timeout: 10 * 60 * 1000 }  // 10 minutes
);
```

## Bug 3: Frame extraction progress not reported to client
When the user uploads a video, the progress jumps from "queued" to "extracting" but the client doesn't see intermediate ffmpeg progress updates because the SSE polling interval is too slow.

**Fix:** This is minor — just make sure the `extractFrames` function persists progress updates to the job JSON, and the client polls frequently enough.

## Files to modify
- `~/splat-test/server/src/index.js` — fix job persistence on startup, fix timeout values

## Testing
After fixes:
1. Start server: `cd ~/splat-test && node server/src/index.js`
2. Upload a short video via the UI
3. Verify: frames extracted (check `server/frames/<job-id>/` has JPEGs)
4. Verify: worker page loads and starts processing
5. Verify: progress updates show in the UI
6. Verify: processing completes (may take 5-10 min on SwiftShader)
