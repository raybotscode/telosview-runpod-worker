export interface Project {
  id: string;
  name: string;
  description: string | null;
  status: ProjectStatus;
  video_path: string | null;
  frame_count: number;
  splat_url: string | null;
  scene_analysis: string | null;
  error: string | null;
  user_id: string | null;
  created_at: string;
  updated_at: string;
}

export type ProjectStatus =
  | 'created'
  | 'uploading'
  | 'extracting'
  | 'extracted'
  | 'processing'
  | 'complete'
  | 'error';

export interface CreateProjectBody {
  name: string;
  description?: string;
}

export interface ProgressEvent {
  type?: 'extraction' | 'extraction_complete' | 'progress' | 'complete' | 'error';
  projectId: string;
  status: ProjectStatus;
  frameCount: number;
  totalFrames: number;
  percent: number;
  message: string;
  stage?: string;
  metrics?: {
    iter: number;
    splats: number;
    itersPerSec: number;
    psnrTrain: string | null;
  } | null;
}
