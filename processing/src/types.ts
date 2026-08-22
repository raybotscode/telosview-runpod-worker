export interface ExtractionResult {
  frameCount: number;
  duration: number;
  fps: number;
  outputDir: string;
}

export interface ExtractionProgress {
  frameCount: number;
  percent: number;
  message: string;
}

export type ProgressCallback = (progress: ExtractionProgress) => void;

// ── Processing Types ──

export interface ProcessingJob {
  projectId: string;
  framesPath: string;
  apiBaseUrl: string;
  /** Override max training iterations (default: 20000) */
  maxIters?: number;
}

export interface ProcessingResult {
  success: boolean;
  plyPath?: string;
  plySize?: number;
  metrics?: SplatMetrics | null;
  error?: string;
  durationMs: number;
  /** Where processing ran: 'local' or 'runpod' */
  backend?: 'local' | 'runpod';
  /** RunPod pod ID (if applicable) */
  podId?: string;
}

export interface SplatMetrics {
  iter: number;
  splats: number;
  itersPerSec: number;
  psnrTrain: string | null;
}

export interface WorkerState {
  status: 'ready' | 'processing' | 'done' | 'error';
  progress: number;
  stage: string;
  message: string;
  metrics: SplatMetrics | null;
  plyBase64: string | null;
  error: string | null;
  log: string[];
}

export interface ProcessingProgress {
  projectId: string;
  status: string;
  stage: string;
  progress: number;
  message: string;
  metrics: SplatMetrics | null;
}

export type ProcessingProgressCallback = (progress: ProcessingProgress) => void;

// ── RunPod Worker Types ──

/** State received from the RunPod worker's SSE stream */
export interface RunPodWorkerState {
  status: string;
  stage: string;
  progress: number;
  message: string;
  metrics: SplatMetrics | null;
  error: string | null;
}

/** Response from POST /upload-frames on the RunPod worker */
export interface RunPodUploadResponse {
  projectId: string;
  frameCount: number;
  message: string;
}

/** Response from POST /process on the RunPod worker */
export interface RunPodProcessResponse {
  projectId: string;
  status: string;
  message: string;
}
