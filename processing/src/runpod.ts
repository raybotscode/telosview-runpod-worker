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
  registryAuthId?: string;
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
 */
export async function launchPod(options: PodOptions = {}): Promise<string> {
  const {
    gpuTypeId = process.env.RUNPOD_GPU_TYPE || 'NVIDIA GeForce RTX 3090',
    gpuCount = 1,
    containerDiskInGb = 50,
    imageName = process.env.RUNPOD_DOCKER_IMAGE || 'ghcr.io/raybotscode/telosview-runpod-worker-worker:latest',
    cloudType = 'SECURE',
    name = 'telosview-processor',
    ports = ['8080/http', '22/tcp'],
    registryAuthId = process.env.RUNPOD_REGISTRY_AUTH_ID || 'cmt4tl46r005k8he9y66fo6jp',
    env,
    dockerArgs,
  } = options;

  let args = `pod create`;
  args += ` --name "${name}"`;
  args += ` --image "${imageName}"`;
  args += ` --gpu-id "${gpuTypeId}"`;
  args += ` --gpu-count ${gpuCount}`;
  args += ` --container-disk-in-gb ${containerDiskInGb}`;
  args += ` --cloud-type ${cloudType}`;
  args += ` --ports "${ports.join(',')}"`;
  args += ` --ssh`;
  args += ` -o json`;

  if (registryAuthId) {
    args += ` --registry-auth-id ${registryAuthId}`;
  }

  if (dockerArgs) {
    args += ` --docker-args '${dockerArgs}'`;
  }

  if (env && Object.keys(env).length > 0) {
    args += ` --env '${JSON.stringify(env)}'`;
  }

  const output = runPodctl(args);
  const pod = JSON.parse(output);
  console.log(`[runpod] Launched pod ${pod.id} (status: ${pod.desiredStatus})`);
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
