import { execSync, spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import type { ExtractionResult, ProgressCallback } from './types.js';

const FRAMES_DIR = path.resolve(import.meta.dirname, '..', 'frames');

function getVideoDuration(videoPath: string): number {
  const result = execSync(
    `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${videoPath}"`,
    { encoding: 'utf-8' }
  );
  return parseFloat(result.trim());
}

function calculateExtractionRate(duration: number): { fps: string; estimatedFrames: number } {
  if (duration < 10) {
    // Extract every frame, cap at 300
    const fps = '30'; // Assume max 30fps source
    return { fps, estimatedFrames: Math.min(Math.ceil(duration * 30), 300) };
  } else if (duration <= 60) {
    // Every 3rd frame
    return { fps: '10', estimatedFrames: Math.ceil(duration * 10) };
  } else if (duration <= 300) {
    // Every 5th frame (1-5 min)
    return { fps: '6', estimatedFrames: Math.ceil(duration * 6) };
  } else {
    // Every 10th frame (>5 min), cap at 1000
    const frames = Math.min(Math.ceil(duration * 3), 1000);
    return { fps: '3', estimatedFrames: frames };
  }
}

export async function extractFrames(
  videoPath: string,
  projectId: string,
  onProgress?: ProgressCallback
): Promise<ExtractionResult> {
  const outputDir = path.join(FRAMES_DIR, projectId);
  fs.mkdirSync(outputDir, { recursive: true });

  const duration = getVideoDuration(videoPath);
  const { fps, estimatedFrames } = calculateExtractionRate(duration);

  onProgress?.({
    frameCount: 0,
    percent: 0,
    message: `Starting extraction: ${duration.toFixed(1)}s video, ~${estimatedFrames} frames at ${fps}fps`
  });

  return new Promise((resolve, reject) => {
    const args = [
      '-i', videoPath,
      '-vf', `fps=${fps},scale='min(1920,iw)':-2`,
      '-q:v', '2',
      path.join(outputDir, 'frame_%05d.jpg')
    ];

    const ffmpeg = spawn('ffmpeg', args);
    let stderr = '';

    ffmpeg.stderr.on('data', (data: Buffer) => {
      stderr += data.toString();
      // Parse frame count from ffmpeg output
      const frameMatch = stderr.match(/frame=\s*(\d+)/);
      if (frameMatch) {
        const currentFrame = parseInt(frameMatch[1]);
        const percent = Math.min(Math.round((currentFrame / estimatedFrames) * 100), 99);
        onProgress?.({
          frameCount: currentFrame,
          percent,
          message: `Extracted ${currentFrame} frames (${percent}%)`
        });
      }
    });

    ffmpeg.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`ffmpeg exited with code ${code}: ${stderr}`));
        return;
      }

      // Count actual extracted frames
      const frames = fs.readdirSync(outputDir).filter(f => f.endsWith('.jpg'));
      const frameCount = frames.length;

      onProgress?.({
        frameCount,
        percent: 100,
        message: `Extraction complete: ${frameCount} frames`
      });

      resolve({
        frameCount,
        duration,
        fps: parseFloat(fps),
        outputDir
      });
    });

    ffmpeg.on('error', (err) => {
      reject(new Error(`Failed to start ffmpeg: ${err.message}`));
    });
  });
}
