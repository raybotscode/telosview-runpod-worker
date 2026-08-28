export type ProjectStatus = 'created' | 'extracting' | 'extracted' | 'processing' | 'preview' | 'complete' | 'error';

export interface Hotspot {
  id: string;
  position: { x: number; y: number; z: number };
  label: string;
  description?: string;
  color?: string;
  icon?: string;
  panelTitle?: string;
  panelContent?: string;  // markdown or HTML
  images?: string[];
  videoEmbed?: string;  // YouTube/Vimeo URL
  linkedScene?: string;
}

export interface TourStep {
  id: string;
  label: string;
  hotspotId?: string;
  camera: {
    position: { x: number; y: number; z: number };
    target: { x: number; y: number; z: number };
  };
  duration?: number;  // seconds
  autoAdvance?: boolean;
}

export interface Tour {
  id: string;
  name: string;
  steps: TourStep[];
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  status: ProjectStatus;
  video_path?: string;
  frames_dir?: string;
  splat_url?: string;
  splat_count?: number;
  thumbnail_url?: string;
  created_at: string;
  updated_at: string;
  error?: string;
  hotspots?: Hotspot[];
  tours?: Tour[];
}

export interface ProcessingProgress {
  stage: string;
  iteration: number;
  total_iterations: number;
  splats: number;
  iter_per_sec: number;
  psnr: number;
  percent: number;
}

export interface ExtractionProgress {
  frame: number;
  total_frames: number;
  percent: number;
}
