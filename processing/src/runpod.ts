/**
 * RunPod API client for programmatic GPU pod management.
 * Uses RunPod's REST API v1 for pod lifecycle management.
 * Uses base image + startup script (GHCR pull not reliable on RunPod).
 */

const RUNPOD_REST_BASE = 'https://rest.runpod.io/v1';

function getApiKey(): string {
  const key = process.env.RUNPOD_API_KEY;
  if (!key) throw new Error('RUNPOD_API_KEY environment variable is required');
  return key;
}

async function restFetch<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${RUNPOD_REST_BASE}${path}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${getApiKey()}`,
      ...options.headers,
    },
  });

  const data = await response.json();

  if (!response.ok) {
    const detail = typeof data === 'object' ? JSON.stringify(data) : String(data);
    throw new Error(`RunPod API error: ${response.status} ${response.statusText} — ${detail}`);
  }

  return data as T;
}

// ── Types ──

export interface PodOptions {
  gpuTypeIds?: string[];
  gpuCount?: number;
  containerDiskInGb?: number;
  volumeInGb?: number;
  imageName?: string;
  cloudType?: 'SECURE' | 'COMMUNITY';
  name?: string;
  ports?: string[];
  env?: Array<{ key: string; value: string }>;
  dockerStartCmd?: string[];
}

export interface PodInfo {
  id: string;
  desiredStatus: string;
  publicIp?: string;
  machineId?: string;
  runtime: {
    uptimeInSeconds: number;
    gpus: { id: string; gpuUtilPercent: number; memoryUtilPercent: number }[];
    ports: { ip: string; isIpPublic: boolean; privatePort: number; publicPort: number; type: string }[];
  } | null;
  machine: {
    podHostId: string;
  } | null;
}

// ── API Functions ──

/**
 * Launch a new GPU pod on demand via REST API.
 */
export async function launchPod(options: PodOptions = {}): Promise<string> {
  const {
    gpuTypeIds = [process.env.RUNPOD_GPU_TYPE || 'NVIDIA GeForce RTX 3090'],
    gpuCount = 1,
    containerDiskInGb = 50,
    volumeInGb = 0,
    imageName = process.env.RUNPOD_DOCKER_IMAGE || 'runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204',
    cloudType = 'COMMUNITY',
    name = 'telosview-processor',
    ports = ['8080/http'],
    env,
    dockerStartCmd,
  } = options;

  const body: Record<string, any> = {
    cloudType,
    gpuCount,
    gpuTypeIds,
    containerDiskInGb,
    volumeInGb,
    imageName,
    name,
    ports,
  };

  if (env && env.length > 0) {
    body.env = Object.fromEntries(env.map(e => [e.key, e.value]));
  }

  if (dockerStartCmd) {
    body.dockerStartCmd = dockerStartCmd;
  }

  const pod = await restFetch<{ id: string; desiredStatus: string }>('/pods', {
    method: 'POST',
    body: JSON.stringify(body),
  });

  console.log(`[runpod] Launched pod ${pod.id} (status: ${pod.desiredStatus})`);
  return pod.id;
}

/**
 * Terminate a running pod.
 */
export async function terminatePod(podId: string): Promise<void> {
  await restFetch(`/pods/${podId}`, { method: 'DELETE' });
  console.log(`[runpod] Terminated pod ${podId}`);
}

/** Get pod status via REST API. */
export async function getPodStatus(podId: string): Promise<PodInfo> {
  return restFetch<PodInfo>(`/pods/${podId}`);
}

/**
 * Wait for a pod to reach RUNNING status.
 */
export async function waitForReady(
  podId: string,
  timeoutMs: number = 10 * 60 * 1000,
  pollIntervalMs: number = 10_000
): Promise<PodInfo> {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const pod = await getPodStatus(podId);
    console.log(`[runpod] Pod ${podId} status: ${pod.desiredStatus} runtime: ${pod.runtime ? 'yes' : 'no'} ip: ${pod.publicIp || 'none'}`);

    if (pod.desiredStatus === 'EXITED' || pod.desiredStatus === 'DEAD') {
      throw new Error(`Pod ${podId} entered terminal state: ${pod.desiredStatus}`);
    }

    // Pod is ready when runtime is non-null (container started, ports available)
    if (pod.runtime) {
      console.log(`[runpod] Pod ${podId} ready — runtime active`);
      return pod;
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
  return restFetch<PodInfo[]>('/pods');
}
