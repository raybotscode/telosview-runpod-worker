/**
 * RunPod API client for programmatic GPU pod management.
 * Uses RunPod's GraphQL API to launch, monitor, and terminate GPU pods.
 */

const RUNPOD_API_BASE = 'https://api.runpod.io/graphql';

function getApiKey(): string {
  const key = process.env.RUNPOD_API_KEY;
  if (!key) throw new Error('RUNPOD_API_KEY environment variable is required');
  return key;
}

async function graphql<T = any>(query: string, variables?: Record<string, any>): Promise<T> {
  const response = await fetch(RUNPOD_API_BASE, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${getApiKey()}`,
    },
    body: JSON.stringify({ query, variables }),
  });

  if (!response.ok) {
    throw new Error(`RunPod API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  if (data.errors?.length) {
    throw new Error(`RunPod GraphQL error: ${data.errors.map((e: any) => e.message).join(', ')}`);
  }
  return data.data;
}

// ── Types ──

export interface PodOptions {
  gpuTypeId?: string;
  gpuCount?: number;
  containerDiskInGb?: number;
  volumeInGb?: number;
  imageName?: string;
  cloudType?: 'SECURE' | 'ALL';
  name?: string;
  ports?: string;       // e.g. "8080/http"
  env?: Record<string, string>;
  dockerArgs?: string;  // Startup command passed to the container
}

export interface PodInfo {
  id: string;
  desiredStatus: string;
  runtime: {
    uptimeInSeconds: number;
    gpus: { id: string; gpuUtilPercent: number; memoryUtilPercent: number }[];
    ports: { ip: string; isIpPublic: boolean; privatePort: number; publicPort: number; type: string }[];
  } | null;
  machine: {
    podHostId: string;
  } | null;
}

export interface PodLaunchResult {
  id: string;
  desiredStatus: string;
  runtime: {
    uptimeInSeconds: number;
  } | null;
}

// ── API Functions ──

/**
 * Launch a new GPU pod on demand.
 */
export async function launchPod(options: PodOptions = {}): Promise<string> {
  const {
    gpuTypeId = process.env.RUNPOD_GPU_TYPE || 'NVIDIA GeForce RTX 3090',
    gpuCount = 1,
    containerDiskInGb = 50,
    volumeInGb = 0,
    imageName = process.env.RUNPOD_DOCKER_IMAGE || 'runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204',
    cloudType = 'SECURE',
    name = 'telosview-processor',
    ports = '22/tcp,8080/http',
    env = {},
    dockerArgs,
  } = options;

  const mutation = `
    mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        desiredStatus
        runtime { uptimeInSeconds }
      }
    }
  `;

  const input: Record<string, any> = {
    cloudType,
    gpuCount,
    gpuTypeId,
    containerDiskInGb,
    volumeInGb,
    imageName,
    name,
    ports,
    env: Object.entries(env).map(([key, value]) => ({ key, value })),
  };

  if (dockerArgs) {
    input.dockerArgs = dockerArgs;
  }

  const data = await graphql<{ podFindAndDeployOnDemand: PodLaunchResult }>(mutation, { input });

  const pod = data.podFindAndDeployOnDemand;
  console.log(`[runpod] Launched pod ${pod.id} (status: ${pod.desiredStatus})`);
  return pod.id;
}

/**
 * Terminate a running pod.
 */
export async function terminatePod(podId: string): Promise<void> {
  const mutation = `
    mutation podTerminate($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
  `;
  await graphql(mutation, { input: { podId } });
  console.log(`[runpod] Terminated pod ${podId}`);
}

/**
 * Get the current status of a pod.
 */
export async function getPodStatus(podId: string): Promise<PodInfo> {
  const query = `
    query pod($input: PodFilter!) {
      pod(input: $input) {
        id
        desiredStatus
        runtime {
          uptimeInSeconds
          gpus { id gpuUtilPercent memoryUtilPercent }
          ports { ip isIpPublic privatePort publicPort type }
        }
        machine { podHostId }
      }
    }
  `;

  const data = await graphql<{ pod: PodInfo }>(query, { input: { podId } });
  return data.pod;
}

/**
 * Wait for a pod to reach RUNNING status.
 */
export async function waitForReady(
  podId: string,
  timeoutMs: number = 10 * 60 * 1000,
  pollIntervalMs: number = 10_000
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    const pod = await getPodStatus(podId);
    console.log(`[runpod] Pod ${podId} status: ${pod.desiredStatus}`);

    if (pod.desiredStatus === 'RUNNING' && pod.runtime) {
      console.log(`[runpod] Pod ${podId} is ready (uptime: ${pod.runtime.uptimeInSeconds}s)`);
      return true;
    }

    if (pod.desiredStatus === 'EXITED' || pod.desiredStatus === 'DEAD') {
      throw new Error(`Pod ${podId} entered terminal state: ${pod.desiredStatus}`);
    }

    await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
  }

  return false;
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
 * Polls the /health endpoint until it responds or times out.
 */
export async function waitForHttpReady(
  podId: string,
  port: number = 8080,
  timeoutMs: number = 5 * 60 * 1000,
  pollIntervalMs: number = 5_000
): Promise<string> {
  const endpoint = getPodEndpoint(podId, port);
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${endpoint}/health`, { signal: AbortSignal.timeout(10_000) });
      if (res.ok) {
        console.log(`[runpod] HTTP service ready at ${endpoint}`);
        return endpoint;
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
  const query = `
    query {
      myself {
        pods {
          id
          desiredStatus
          runtime {
            uptimeInSeconds
            gpus { id gpuUtilPercent memoryUtilPercent }
            ports { ip isIpPublic privatePort publicPort type }
          }
          machine { podHostId }
        }
      }
    }
  `;

  const data = await graphql<{ myself: { pods: PodInfo[] } }>(query);
  return data.myself.pods;
}
