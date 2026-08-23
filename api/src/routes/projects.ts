import { Router } from 'express';
import type { Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import {
  createProject,
  getProject,
  getAllProjects,
  updateProjectStatus,
  updateProjectVideoPath,
  updateProjectFrameCount,
  updateProjectSplatUrl,
  getProjectSceneAnalysis,
  updateProjectSceneAnalysis,
  deleteProject
} from '../db.js';
import { extractFrames } from '../../../processing/src/extract-frames.js';
import { processProject } from '../../../processing/src/orchestrator.js';
import { analyzeScene, answerQuestion } from '../../../processing/src/scene-analyzer.js';
import type { CreateProjectBody, ProgressEvent } from '../types.js';
import { requireAuth } from '../middleware/auth.js';
import type { AuthRequest } from '../middleware/auth.js';

const router = Router();

// Multer setup for video uploads
const uploadsDir = path.resolve(import.meta.dirname, '..', '..', 'uploads');
fs.mkdirSync(uploadsDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (req, _file, cb) => {
    const ext = '.mp4'; // Normalize to mp4
    cb(null, `${req.params.id}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 500 * 1024 * 1024 }, // 500MB
  fileFilter: (_req, file, cb) => {
    const allowed = ['.mp4', '.mov', '.webm', '.avi', '.mkv'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowed.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported video format: ${ext}`));
    }
  }
});

// SSE clients per project
const sseClients = new Map<string, Set<Response>>();

function broadcastProgress(projectId: string, event: ProgressEvent) {
  const clients = sseClients.get(projectId);
  if (clients) {
    const data = JSON.stringify(event);
    for (const client of clients) {
      client.write(`data: ${data}\n\n`);
    }
  }
}

// POST /api/projects — create new project
router.post('/', requireAuth, (req: AuthRequest, res: Response) => {
  const { name, description } = req.body as CreateProjectBody;
  if (!name) {
    res.status(400).json({ error: 'Name is required' });
    return;
  }
  const id = uuidv4();
  const project = createProject(id, name, description, req.userId);
  res.status(201).json(project);
});

// GET /api/projects — list all projects
router.get('/', requireAuth, (req: AuthRequest, res: Response) => {
  const projects = getAllProjects(req.userId);
  res.json(projects);
});

// GET /api/projects/:id — get project details
router.get('/:id', requireAuth, (req: AuthRequest, res: Response) => {
  const project = getProject(req.params.id);
  if (!project) {
    res.status(404).json({ error: 'Project not found' });
    return;
  }
  if (project.user_id && project.user_id !== req.userId) {
    res.status(403).json({ error: 'Access denied' });
    return;
  }
  res.json(project);
});

// GET /api/projects/:id/frames — list extracted frames for a project
router.get('/:id/frames', (req: Request, res: Response) => {
  const projectId = req.params.id;
  const project = getProject(projectId);
  if (!project) {
    res.status(404).json({ error: 'Project not found' });
    return;
  }

  const framesDir = path.resolve(
    import.meta.dirname, '..', '..', '..', 'processing', 'frames', projectId
  );

  if (!fs.existsSync(framesDir)) {
    res.status(409).json({ error: 'Frames not extracted yet' });
    return;
  }

  try {
    const names = fs.readdirSync(framesDir)
      .filter(name => /^frame_\d{5}\.jpg$/i.test(name))
      .sort();

    res.json({
      count: names.length,
      frames: names.map(name => ({
        name,
        url: `/frames/${encodeURIComponent(projectId)}/${encodeURIComponent(name)}`,
      })),
    });
  } catch {
    res.status(409).json({ error: 'Frames not available' });
  }
});

// GET /api/projects/:id/status — SSE endpoint for real-time progress
router.get('/:id/status', (req: Request, res: Response) => {
  const projectId = req.params.id;
  const project = getProject(projectId);
  if (!project) {
    res.status(404).json({ error: 'Project not found' });
    return;
  }

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  // Send current status immediately
  const initialEvent: ProgressEvent = {
    type: project.status === 'complete' ? 'complete' : project.status === 'error' ? 'error' : 'progress',
    projectId,
    status: project.status,
    frameCount: project.frame_count,
    totalFrames: project.frame_count,
    percent: project.status === 'extracted' || project.status === 'complete' ? 100 : 0,
    message: `Status: ${project.status}`
  };
  res.write(`data: ${JSON.stringify(initialEvent)}\n\n`);

  // Register SSE client
  if (!sseClients.has(projectId)) {
    sseClients.set(projectId, new Set());
  }
  sseClients.get(projectId)!.add(res);

  // Cleanup on disconnect
  req.on('close', () => {
    sseClients.get(projectId)?.delete(res);
    if (sseClients.get(projectId)?.size === 0) {
      sseClients.delete(projectId);
    }
  });
});

// POST /api/projects/:id/upload — upload video and extract frames
router.post('/:id/upload', (req: Request, res: Response) => {
  const projectId = req.params.id;
  const project = getProject(projectId);
  if (!project) {
    res.status(404).json({ error: 'Project not found' });
    return;
  }

  upload.single('video')(req, res, async (err) => {
    if (err) {
      updateProjectStatus(projectId, 'error', err.message);
      res.status(400).json({ error: err.message });
      return;
    }
    if (!req.file) {
      res.status(400).json({ error: 'No video file provided' });
      return;
    }

    const videoPath = req.file.path;
    updateProjectVideoPath(projectId, videoPath);

    // Respond with the FULL updated project row. Clients type this as
    // Project; returning a { message } stub left them rendering
    // "Created Invalid Date" (created_at missing until the next refetch).
    res.json(getProject(projectId));

    try {
      updateProjectStatus(projectId, 'extracting');
      broadcastProgress(projectId, {
        type: 'extraction',
        projectId,
        status: 'extracting',
        frameCount: 0,
        totalFrames: 0,
        percent: 0,
        message: 'Starting frame extraction...'
      });

      const result = await extractFrames(videoPath, projectId, (progress) => {
        broadcastProgress(projectId, {
          type: 'extraction',
          projectId,
          status: 'extracting',
          frameCount: progress.frameCount,
          totalFrames: 0,
          percent: progress.percent,
          message: progress.message
        });
      });

      updateProjectFrameCount(projectId, result.frameCount);
      updateProjectStatus(projectId, 'extracted');

      broadcastProgress(projectId, {
        type: 'extraction_complete',
        projectId,
        status: 'extracted',
        frameCount: result.frameCount,
        totalFrames: result.frameCount,
        percent: 100,
        message: `Extraction complete: ${result.frameCount} frames from ${result.duration.toFixed(1)}s video`
      });
    } catch (error: any) {
      updateProjectStatus(projectId, 'error', error.message);
      broadcastProgress(projectId, {
        projectId,
        status: 'error',
        frameCount: 0,
        totalFrames: 0,
        percent: 0,
        message: `Error: ${error.message}`
      });
    }
  });
});

// POST /api/projects/:id/process — trigger splat processing
router.post('/:id/process', (req: Request, res: Response) => {
  const projectId = req.params.id;
  const project = getProject(projectId);
  if (!project) {
    res.status(404).json({ error: 'Project not found' });
    return;
  }

  if (project.status !== 'extracted') {
    res.status(409).json({
      error: `Project must be in 'extracted' state to process (current: ${project.status})`
    });
    return;
  }

  const framesDir = path.resolve(
    import.meta.dirname, '..', '..', '..', 'processing', 'frames', projectId
  );

  if (!fs.existsSync(framesDir)) {
    res.status(409).json({ error: 'No extracted frames found' });
    return;
  }

  // Update status and respond immediately
  updateProjectStatus(projectId, 'processing');
  res.json({ message: 'Processing started', projectId });

  // Run processing asynchronously
  const apiBaseUrl = `http://localhost:${process.env.PORT || 3457}`;

  processProject(
    {
      projectId,
      framesPath: framesDir,
      apiBaseUrl,
    },
    (progress) => {
      // Broadcast progress via SSE
      broadcastProgress(projectId, {
        type: 'progress',
        projectId,
        status: progress.status as any,
        frameCount: project.frame_count,
        totalFrames: project.frame_count,
        percent: progress.progress,
        message: progress.message,
        stage: progress.stage,
        metrics: progress.metrics,
      });
    }
  ).then((result) => {
    console.log(`[process] Project ${projectId} finished via ${result.backend || 'local'} in ${(result.durationMs / 1000).toFixed(1)}s`);
    if (result.success && result.plyPath) {
      // For now, store the local path as the splat URL
      // In production, this would be an R2 URL after upload
      updateProjectSplatUrl(projectId, `/api/projects/${projectId}/model.ply`);
      broadcastProgress(projectId, {
        type: 'complete',
        projectId,
        status: 'complete',
        frameCount: project.frame_count,
        totalFrames: project.frame_count,
        percent: 100,
        message: `Processing complete! PLY: ${((result.plySize || 0) / 1024 / 1024).toFixed(1)} MB`,
        metrics: result.metrics,
      });
    } else {
      updateProjectStatus(projectId, 'error', result.error);
      broadcastProgress(projectId, {
        type: 'error',
        projectId,
        status: 'error',
        frameCount: 0,
        totalFrames: 0,
        percent: 0,
        message: `Processing failed: ${result.error}`,
      });
    }
  }).catch((err) => {
    console.error(`[process] Unexpected error for ${projectId}:`, err);
    updateProjectStatus(projectId, 'error', err.message);
  });
});

// GET /api/projects/:id/model.ply — download PLY file
// The path MUST end in `.ply` (or `.splat`/`.spz`): Spark's SplatMesh infers the
// file format from the URL extension and throws "Unable to determine file type"
// on a bare `/ply` path. Kept a legacy `/:id/ply` alias for older records.
const downloadPly = (req: Request, res: Response) => {
  const projectId = req.params.id;
  const plyPath = path.resolve(
    import.meta.dirname, '..', '..', '..', 'processing', 'output', `${projectId}.ply`
  );

  if (!fs.existsSync(plyPath)) {
    res.status(404).json({ error: 'PLY file not found' });
    return;
  }

  res.download(plyPath, `${projectId}.ply`);
};
router.get('/:id/model.ply', downloadPly);
router.get('/:id/ply', downloadPly);

// POST /api/projects/:id/analyze — trigger AI scene analysis
router.post('/:id/analyze', async (req: Request, res: Response) => {
  const projectId = req.params.id;
  const project = getProject(projectId);
  if (!project) {
    res.status(404).json({ error: 'Project not found' });
    return;
  }

  // Check if already analyzed (cache)
  const existing = getProjectSceneAnalysis(projectId);
  if (existing) {
    try {
      res.json({ cached: true, analysis: JSON.parse(existing) });
      return;
    } catch {
      // Corrupted cache, re-analyze
    }
  }

  const framesDir = path.resolve(
    import.meta.dirname, '..', '..', '..', 'processing', 'frames', projectId
  );

  if (!fs.existsSync(framesDir)) {
    res.status(409).json({ error: 'No extracted frames found. Upload and extract a video first.' });
    return;
  }

  try {
    const analysis = await analyzeScene(framesDir);
    updateProjectSceneAnalysis(projectId, JSON.stringify(analysis));
    res.json({ cached: false, analysis });
  } catch (err: any) {
    console.error(`[analyze] Failed for ${projectId}:`, err.message);
    res.status(502).json({ error: `AI analysis failed: ${err.message}` });
  }
});

// POST /api/projects/:id/ask — ask a question about the scene
router.post('/:id/ask', async (req: Request, res: Response) => {
  const projectId = req.params.id;
  const { question } = req.body as { question?: string };

  if (!question || typeof question !== 'string' || question.trim().length === 0) {
    res.status(400).json({ error: 'Question is required' });
    return;
  }

  const project = getProject(projectId);
  if (!project) {
    res.status(404).json({ error: 'Project not found' });
    return;
  }

  const framesDir = path.resolve(
    import.meta.dirname, '..', '..', '..', 'processing', 'frames', projectId
  );

  if (!fs.existsSync(framesDir)) {
    res.status(409).json({ error: 'No extracted frames found' });
    return;
  }

  // Use cached scene analysis as context if available
  const sceneContext = getProjectSceneAnalysis(projectId) || 'No prior analysis available.';

  try {
    const result = await answerQuestion(question.trim(), sceneContext, framesDir);
    res.json(result);
  } catch (err: any) {
    console.error(`[ask] Failed for ${projectId}:`, err.message);
    res.status(502).json({ error: `AI query failed: ${err.message}` });
  }
});

// DELETE /api/projects/:id — delete project
router.delete('/:id', requireAuth, (req: AuthRequest, res: Response) => {
  const projectId = req.params.id;
  const project = getProject(projectId);
  if (!project) {
    res.status(404).json({ error: 'Project not found' });
    return;
  }
  if (project.user_id && project.user_id !== req.userId) {
    res.status(403).json({ error: 'Access denied' });
    return;
  }

  // Clean up files
  if (project.video_path && fs.existsSync(project.video_path)) {
    fs.unlinkSync(project.video_path);
  }
  const framesDir = path.resolve(import.meta.dirname, '..', '..', '..', 'processing', 'frames', projectId);
  if (fs.existsSync(framesDir)) {
    fs.rmSync(framesDir, { recursive: true, force: true });
  }
  const plyPath = path.resolve(import.meta.dirname, '..', '..', '..', 'processing', 'output', `${projectId}.ply`);
  if (fs.existsSync(plyPath)) {
    fs.unlinkSync(plyPath);
  }

  deleteProject(projectId);
  res.json({ message: 'Project deleted' });
});

export default router;
