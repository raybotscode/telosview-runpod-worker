/**
 * RunPod API client for programmatic GPU pod management.
 * Uses runpodctl CLI (which uses REST v2 internally).
 * REST v1 is deprecated and broken — CLI is the reliable path.
 */

import { execSync } from 'child_process';

const RUNPODCTL = process.env.RUNPODCTL_PATH || `${process.env.HOME}/runpodctl`;

function runPodctl(args: string): string {
  const cmd = `${RUNPODCTL} ${args}`;
  console.log(`[runpod] ${cmd}`);
  try {
    return execSync(cmd, {
      encoding: 'utf-8',
      stdio: 'pipe',
      timeout: 30000,
      env: { ...process.env },
    });
  } catch (err: any) {
    throw new Error(`runpodctl failed: ${err.stderr || err.message}`);
  }
}

// ── Types ──

export interface PodOptions {
  gpuTypeId?: string;
  gpuCount?: number;
  containerDiskInGb?: number;
  imageName?: string;
  cloudType?: 'SECURE' | 'COMMUNITY';
  name?: string;
  ports?: string[];
  env?: Record<string, string>;
  dockerArgs?: string;
}

export interface PodInfo {
  id: string;
  desiredStatus: string;
  runtimeStatus?: string;
  machine?: {
    gpuDisplayName?: string;
    location?: string;
  };
  ports?: string[];
  costPerHr?: number;
  uptimeSeconds?: number;
}

// ── API Functions ──

/**
 * Launch a new GPU pod via runpodctl CLI.
 *
 * Uses a fallback chain: RunPod's catalog-level "stock" is not a guarantee that
 * a specific machine has free capacity ("This machine does not have the
 * resources to deploy your pod"), so if the primary GPU fails to provision we
 * try the next candidate. All candidates are NVIDIA + Vulkan-capable — native
 * Dawn talks to the GPU over Vulkan (not CUDA), so any of them runs splat.js.
 * Ordered by stock reliability + price.
 */
const GPU_FALLBACKS = [
  'NVIDIA GeForce RTX 4090',   // 24GB, High stock, well-tested
  'NVIDIA A40',                 // 48GB, High stock, cheaper ($0.44/hr)
  'NVIDIA RTX PRO 4500 Blackwell', // 32GB, High stock
  'NVIDIA GeForce RTX 3090',   // 24GB, former default
  'NVIDIA L4',                 // 24GB
];

export async function launchPod(options: PodOptions = {}): Promise<string> {
  const {
    gpuTypeId,
    gpuCount = 1,
    containerDiskInGb = 50,
    imageName = process.env.RUNPOD_DOCKER_IMAGE || 'raybotsemail/telosview-worker:latest',
    cloudType = 'SECURE',
    name = 'telosview-processor',
    ports = ['8080/http', '22/tcp'],
    env,
    dockerArgs,
  } = options;

  // Candidate list: explicit option/env override first, then the fallback chain.
  const candidates: string[] = [];
  if (gpuTypeId) candidates.push(gpuTypeId);
  else if (process.env.RUNPOD_GPU_TYPE) candidates.push(process.env.RUNPOD_GPU_TYPE);
  for (const g of GPU_FALLBACKS) {
    if (!candidates.includes(g)) candidates.push(g);
  }

  let lastError = '';
  for (const gpu of candidates) {
    try {
      return await tryCreatePod({
        gpu, gpuCount, containerDiskInGb, imageName, cloudType, name, ports, env, dockerArgs,
      });
    } catch (err: any) {
      lastError = err.message;
      console.log(`[runpod] GPU "${gpu}" unavailable, trying next: ${err.message}`);
    }
  }
  throw new Error(`Pod launch failed after trying ${candidates.length} GPU types. Last error: ${lastError}`);
}

async function tryCreatePod(opts: {
  gpu: string; gpuCount: number; containerDiskInGb: number; imageName: string;
  cloudType: string; name: string; ports: string[]; env?: Record<string, string>; dockerArgs?: string;
}): Promise<string> {
  const { gpu, gpuCount, containerDiskInGb, imageName, cloudType, name, ports, env, dockerArgs } = opts;

  let args = `pod create`;
  args += ` --name "${name}"`;
  args += ` --image "${imageName}"`;
  args += ` --gpu-id "${gpu}"`;
  args += ` --gpu-count ${gpuCount}`;
  args += ` --container-disk-in-gb ${containerDiskInGb}`;
  args += ` --ports "${ports.join(',')}"`;
  args += ` -o json`;

  if (dockerArgs) {
    args += ` --docker-args '${dockerArgs}'`;
  }

  if (env && Object.keys(env).length > 0) {
    args += ` --env '${JSON.stringify(env)}'`;
  }

  const output = runPodctl(args);
  let pod: any;
  try {
    pod = JSON.parse(output);
  } catch {
    throw new Error(`runpodctl pod create returned non-JSON: ${output}`);
  }
  if (!pod || !pod.id) {
    const msg = pod?.error || output;
    throw new Error(`Pod launch failed (GPU "${gpu}" may be unavailable): ${msg}`);
  }
  console.log(`[runpod] Launched pod ${pod.id} on "${gpu}" (status: ${pod.desiredStatus})`);
  return pod.id;
}

/**
 * Terminate a running pod.
 */
export async function terminatePod(podId: string): Promise<void> {
  runPodctl(`pod delete ${podId}`);
  console.log(`[runpod] Terminated pod ${podId}`);
}

/**
 * Get pod status via CLI.
 */
export async function getPodStatus(podId: string): Promise<PodInfo> {
  const output = runPodctl(`pod get ${podId} -o json`);
  return JSON.parse(output);
}

/**
 * Wait for a pod to reach running status with container ready.
 */
export async function waitForReady(
  podId: string,
  timeoutMs: number = 10 * 60 * 1000,
  pollIntervalMs: number = 10_000
): Promise<PodInfo> {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const pod = await getPodStatus(podId);
    console.log(`[runpod] Pod ${podId} runtimeStatus: ${pod.runtimeStatus || 'unknown'}`);

    if (pod.runtimeStatus === 'running') {
      console.log(`[runpod] Pod ${podId} ready — container running`);
      return pod;
    }

    if (pod.runtimeStatus === 'exited' || pod.runtimeStatus === 'dead') {
      throw new Error(`Pod ${podId} entered terminal state: ${pod.runtimeStatus}`);
    }

    await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
  }

  throw new Error(`Pod ${podId} not ready after ${timeoutMs / 1000}s`);
}

/**
 * Get the public HTTP endpoint URL for a pod's exposed port.
 * RunPod proxies exposed ports through: https://<podId>-<port>.proxy.runpod.net
 */
export function getPodEndpoint(podId: string, port: number = 8080): string {
  return `https://${podId}-${port}.proxy.runpod.net`;
}

/**
 * Wait for the pod's HTTP service to be reachable.
 * Polls the /health endpoint until it returns JSON (not RunPod's HTML waiting page).
 */
export async function waitForHttpReady(
  podId: string,
  port: number = 8080,
  timeoutMs: number = 8 * 60 * 1000,
  pollIntervalMs: number = 5_000
): Promise<string> {
  const endpoint = getPodEndpoint(podId, port);
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${endpoint}/health`, { signal: AbortSignal.timeout(10_000) });
      if (res.ok) {
        const contentType = res.headers.get('content-type') || '';
        // RunPod proxy returns HTML "waiting" page when service isn't ready
        if (contentType.includes('application/json')) {
          console.log(`[runpod] HTTP service ready at ${endpoint}`);
          return endpoint;
        }
      }
    } catch {
      // Service not ready yet
    }
    await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
  }

  throw new Error(`HTTP service at ${endpoint} did not become ready within ${timeoutMs / 1000}s`);
}

/**
 * List all pods for the account.
 */
export async function listPods(): Promise<PodInfo[]> {
  const output = runPodctl(`pod list -o json`);
  return JSON.parse(output);
}
