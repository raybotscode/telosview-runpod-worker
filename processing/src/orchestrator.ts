import { chromium, type Browser, type BrowserContext, type Page } from 'playwright';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import {
  launchPod,
  terminatePod,
  waitForReady,
  waitForHttpReady,
  getPodEndpoint,
} from './runpod.js';
import type {
  ProcessingJob,
  ProcessingResult,
  WorkerState,
  ProcessingProgressCallback,
  RunPodWorkerState,
  RunPodUploadResponse,
} from './types.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WORKER_HTML_PATH = path.resolve(__dirname, '..', 'worker.html');
const PLY_OUTPUT_DIR = path.resolve(__dirname, '..', 'output');

fs.mkdirSync(PLY_OUTPUT_DIR, { recursive: true });

let browser: Browser | null = null;

/** Check if RunPod mode should be used */
function useRunPod(): boolean {
  return !!process.env.RUNPOD_API_KEY;
}

/**
 * Launch or reuse a Chromium browser with WebGPU/Vulkan support.
 * On a real GPU server, remove '--use-vulkan=swiftshader'.
 */
export async function getBrowser(): Promise<Browser> {
  if (browser && browser.isConnected()) return browser;

  console.log('[orchestrator] Launching Chromium with WebGPU support...');
  const isGpuServer = process.env.GPU_SERVER === 'true';

  const args = [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--enable-webgpu-developer-features',
    '--disable-gpu-sandbox',
    '--no-sandbox',
    '--disable-dev-shm-usage',
  ];

  if (!isGpuServer) {
    args.push('--use-vulkan=swiftshader');
  }

  browser = await chromium.launch({ headless: true, args });
  browser.on('disconnected', () => {
    browser = null;
  });
  console.log('[orchestrator] Browser launched');
  return browser;
}

/**
 * Process a project: launch Playwright, navigate to worker, run splat.js pipeline.
 */
export async function processProject(
  job: ProcessingJob,
  onProgress?: ProcessingProgressCallback
): Promise<ProcessingResult> {
  if (useRunPod()) {
    return processProjectRunPod(job, onProgress);
  }
  return processProjectLocal(job, onProgress);
}

/** Local Playwright processing (existing flow) */
async function processProjectLocal(
  job: ProcessingJob,
  onProgress?: ProcessingProgressCallback
): Promise<ProcessingResult> {
  const startTime = Date.now();
  let context: BrowserContext | undefined;
  let progressInterval: ReturnType<typeof setInterval> | undefined;

  try {
    onProgress?.({
      projectId: job.projectId,
      status: 'processing',
      stage: 'initializing',
      progress: 0,
      message: 'Launching browser...',
      metrics: null,
    });

    const b = await getBrowser();
    context = await b.newContext();
    const page = await context.newPage();

    const workerUrl = `${job.apiBaseUrl}/worker.html`;
    console.log(`[orchestrator] Navigating to ${workerUrl}`);
    await page.goto(workerUrl, { waitUntil: 'domcontentloaded' });

    console.log('[orchestrator] Waiting for worker ready...');
    await page.waitForFunction(
      () => (window as any).__worker?.status === 'ready',
      null,
      { timeout: 2 * 60 * 1000 }
    );
    console.log('[orchestrator] Worker ready');

    onProgress?.({
      projectId: job.projectId,
      status: 'processing',
      stage: 'loading',
      progress: 5,
      message: 'Worker ready, starting processing...',
      metrics: null,
    });

    await page.evaluate((projectId: string) => {
      (window as any).__process(projectId);
    }, job.projectId);

    progressInterval = setInterval(async () => {
      try {
        const state: WorkerState = await page.evaluate(() => (window as any).__worker);
        if (state) {
          onProgress?.({
            projectId: job.projectId,
            status: state.status === 'done' ? 'complete' : 'processing',
            stage: state.stage,
            progress: Math.round(state.progress),
            message: state.message,
            metrics: state.metrics,
          });
        }
      } catch {}
    }, 2000);

    console.log('[orchestrator] Waiting for processing to complete...');
    await page.waitForFunction(
      () => {
        const w = (window as any).__worker;
        return w?.status === 'done' || w?.status === 'error';
      },
      null,
      { timeout: 30 * 60 * 1000 }
    );

    clearInterval(progressInterval);
    progressInterval = undefined;

    const finalState: WorkerState = await page.evaluate(() => (window as any).__worker);

    if (finalState.status === 'error') {
      return {
        success: false,
        error: finalState.error || 'Unknown processing error',
        durationMs: Date.now() - startTime,
      };
    }

    if (!finalState.plyBase64) {
      return {
        success: false,
        error: 'Processing completed but no PLY data was produced',
        durationMs: Date.now() - startTime,
      };
    }

    const plyPath = path.join(PLY_OUTPUT_DIR, `${job.projectId}.ply`);
    const plyBuf = Buffer.from(finalState.plyBase64, 'base64');
    fs.writeFileSync(plyPath, plyBuf);

    console.log(`[orchestrator] PLY saved: ${plyPath} (${(plyBuf.length / 1024 / 1024).toFixed(1)} MB)`);

    onProgress?.({
      projectId: job.projectId,
      status: 'complete',
      stage: 'done',
      progress: 100,
      message: `Complete! PLY: ${(plyBuf.length / 1024 / 1024).toFixed(1)} MB`,
      metrics: finalState.metrics,
    });

    return {
      success: true,
      plyPath,
      plySize: plyBuf.length,
      metrics: finalState.metrics,
      durationMs: Date.now() - startTime,
      backend: 'local' as const,
    };
  } catch (err: any) {
    console.error(`[orchestrator] Processing failed:`, err.message);
    return {
      success: false,
      error: err.message,
      durationMs: Date.now() - startTime,
    };
  } finally {
    if (progressInterval) clearInterval(progressInterval);
    await context?.close().catch(() => {});
  }
}

/**
 * RunPod GPU processing — launches a remote pod with real GPU.
 * Uses a pre-built Docker image with the worker server baked in.
 */
async function processProjectRunPod(
  job: ProcessingJob,
  onProgress?: ProcessingProgressCallback
): Promise<ProcessingResult> {
  const startTime = Date.now();
  let podId: string | null = null;

  try {
    onProgress?.({
      projectId: job.projectId,
      status: 'processing',
      stage: 'launching',
      progress: 0,
      message: 'Launching RunPod GPU pod...',
      metrics: null,
    });

    // 1. Launch GPU pod with pre-built worker image
    podId = await launchPod({
      name: `telosview-${job.projectId}`,
      ports: ['8080/http'],
    });
    console.log(`[orchestrator] Launched pod ${podId}`);

    // 2. Wait for pod to reach RUNNING status
    onProgress?.({
      projectId: job.projectId,
      status: 'processing',
      stage: 'launching',
      progress: 2,
      message: `Pod ${podId} launching, waiting for GPU...`,
      metrics: null,
    });

    const ready = await waitForReady(podId, 10 * 60 * 1000);
    if (!ready) throw new Error('Pod did not become ready in time');

    // 3. Wait for HTTP service (worker starts automatically from Docker image)
    onProgress?.({
      projectId: job.projectId,
      status: 'processing',
      stage: 'loading',
      progress: 4,
      message: 'Waiting for worker HTTP service...',
      metrics: null,
    });

    const endpoint = await waitForHttpReady(podId, 8080, 5 * 60 * 1000);
    console.log(`[orchestrator] Pod endpoint: ${endpoint}`);

    // 4. Upload frames
    onProgress?.({
      projectId: job.projectId,
      status: 'processing',
      stage: 'uploading',
      progress: 7,
      message: 'Uploading frames to GPU pod...',
      metrics: null,
    });

    await uploadFramesToPod(endpoint, job.projectId, job.framesPath);

    // 5. Start processing
    onProgress?.({
      projectId: job.projectId,
      status: 'processing',
      stage: 'processing',
      progress: 8,
      message: 'Starting Gaussian splatting on GPU...',
      metrics: null,
    });

    const processRes = await fetch(`${endpoint}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        projectId: job.projectId,
        maxIters: job.maxIters || 20000,
      }),
    });
    if (!processRes.ok) {
      const body = await processRes.text();
      throw new Error(`Failed to start processing: ${processRes.status} ${body}`);
    }

    // 6. Stream progress via SSE
    const result = await streamRunPodProgress(
      endpoint, job.projectId, onProgress
    );

    if (!result.success) {
      return {
        success: false,
        error: result.error,
        durationMs: Date.now() - startTime,
        backend: 'runpod',
        podId,
      };
    }

    // 7. Download PLY
    onProgress?.({
      projectId: job.projectId,
      status: 'processing',
      stage: 'downloading',
      progress: 98,
      message: 'Downloading PLY from GPU pod...',
      metrics: null,
    });

    const plyPath = await downloadPlyFromPod(endpoint, job.projectId);

    onProgress?.({
      projectId: job.projectId,
      status: 'complete',
      stage: 'done',
      progress: 100,
      message: `Complete! PLY: ${(result.plySize / 1024 / 1024).toFixed(1)} MB`,
      metrics: result.metrics,
    });

    return {
      success: true,
      plyPath,
      plySize: result.plySize,
      metrics: result.metrics,
      durationMs: Date.now() - startTime,
      backend: 'runpod',
      podId,
    };
  } catch (err: any) {
    console.error(`[orchestrator] RunPod processing failed:`, err.message);
    return {
      success: false,
      error: err.message,
      durationMs: Date.now() - startTime,
      backend: 'runpod',
      podId: podId || undefined,
    };
  } finally {
    if (podId) {
      console.log(`[orchestrator] Terminating pod ${podId}...`);
      await terminatePod(podId).catch(err =>
        console.error(`[orchestrator] Failed to terminate pod ${podId}:`, err.message)
      );
    }
  }
}


/** Upload all JPEG frames from a local directory to the RunPod worker */
async function uploadFramesToPod(
  endpoint: string,
  projectId: string,
  framesPath: string
): Promise<void> {
  const files = fs.readdirSync(framesPath)
    .filter((n: string) => /^frame_\d{5}\.jpg$/i.test(n))
    .sort();

  if (files.length === 0) throw new Error('No frame files found');

  const BATCH = 20;
  for (let i = 0; i < files.length; i += BATCH) {
    const batch = files.slice(i, i + BATCH);
    const form = new FormData();
    for (const name of batch) {
      const buf = fs.readFileSync(path.join(framesPath, name));
      form.append('frames', new Blob([buf], { type: 'image/jpeg' }), name);
    }
    const res = await fetch(`${endpoint}/upload-frames/${projectId}`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Frame upload failed: ${res.status} ${body}`);
    }
    const data: RunPodUploadResponse = await res.json();
    console.log(`[orchestrator] Uploaded batch ${i}-${i + batch.length}: ${data.frameCount} total`);
  }
}

/** Stream progress from the RunPod worker's SSE endpoint */
async function streamRunPodProgress(
  endpoint: string,
  projectId: string,
  onProgress?: ProcessingProgressCallback
): Promise<{ success: boolean; error?: string; plySize: number; metrics: any }> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('RunPod processing timed out (30 min)'));
    }, 30 * 60 * 1000);

    const sseUrl = `${endpoint}/process/${projectId}/progress`;
    console.log(`[orchestrator] Connecting to SSE: ${sseUrl}`);

    fetch(sseUrl).then(res => {
      if (!res.ok) {
        clearTimeout(timeout);
        reject(new Error(`SSE connection failed: ${res.status}`));
        return;
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      function processChunk(): Promise<any> {
        return reader.read().then(({ done, value }) => {
          if (done) {
            clearTimeout(timeout);
            resolve({ success: false, error: 'SSE stream ended unexpectedly', plySize: 0, metrics: null });
            return;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
              const state: RunPodWorkerState = JSON.parse(line.slice(6));

              onProgress?.({
                projectId,
                status: state.status === 'done' ? 'complete' : 'processing',
                stage: state.stage,
                progress: Math.round(state.progress),
                message: state.message,
                metrics: state.metrics,
              });

              if (state.status === 'done') {
                clearTimeout(timeout);
                resolve({
                  success: true,
                  plySize: 0,
                  metrics: state.metrics,
                });
                reader.cancel();
                return;
              }
              if (state.status === 'error') {
                clearTimeout(timeout);
                resolve({
                  success: false,
                  error: state.error || 'Processing error on GPU pod',
                  plySize: 0,
                  metrics: null,
                });
                reader.cancel();
                return;
              }
            } catch { /* ignore parse errors */ }
          }

          return processChunk();
        });
      }

      return processChunk();
    }).catch(err => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

/** Download the PLY result file from the RunPod worker */
async function downloadPlyFromPod(
  endpoint: string,
  projectId: string
): Promise<string> {
  const res = await fetch(`${endpoint}/process/${projectId}/result`);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`PLY download failed: ${res.status} ${body}`);
  }

  const arrayBuf = await res.arrayBuffer();
  const plyPath = path.join(PLY_OUTPUT_DIR, `${projectId}.ply`);
  fs.writeFileSync(plyPath, Buffer.from(arrayBuf));

  console.log(`[orchestrator] PLY downloaded: ${plyPath} (${(arrayBuf.byteLength / 1024 / 1024).toFixed(1)} MB)`);
  return plyPath;
}

/**
 * Gracefully shut down the browser.
 */
export async function shutdownBrowser(): Promise<void> {
  if (browser) {
    await browser.close().catch(() => {});
    browser = null;
  }
}
