import { processProject } from './processing/src/orchestrator.js';
import path from 'path';

async function main() {
  const projectId = '626b2824-1987-4643-8a67-54cd81ae139e';
  const framesPath = path.resolve('processing/frames', projectId);

  console.log('Testing RunPod processing...');
  console.log('Project:', projectId);
  console.log('Frames:', framesPath);
  console.log('RUNPOD_API_KEY:', process.env.RUNPOD_API_KEY ? 'set (' + process.env.RUNPOD_API_KEY.slice(0,8) + '...)' : 'NOT SET');

  try {
    const result = await processProject(
      { projectId, framesPath, apiBaseUrl: 'http://localhost:3457' },
      (progress) => {
        console.log(`[${progress.stage}] ${progress.progress}% - ${progress.message}`);
      }
    );
    console.log('Result:', JSON.stringify(result, null, 2));
  } catch (err) {
    console.error('FAILED:', err);
  }
}

main();
