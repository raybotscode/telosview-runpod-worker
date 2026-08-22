// quality.js — in-browser quality scenarios, driven by tests/quality/run.mjs.
//
// Deliberately consumes ONLY the public library surface (src/index.js) — the
// gates double as an integration test of the packaged API. Results are POSTed
// to /scratch/quality_<run>.json where the node runner asserts thresholds.
//
//   ?scene=synthetic-solve   SfM accuracy on the bundled synthetic set
//   ?scene=synthetic-train   short training run, PSNR floors
//   ?scene=truck-ate         truck-42 SfM, poses posted for ATE vs COLMAP GT
//   ?scene=camping-ate       camping-113 SfM, poses posted for ATE vs GT

import { createSession } from '../../src/index.js';

const q = new URLSearchParams(location.search);
const scene = q.get('scene') || 'synthetic-solve';
const runId = q.get('run') || 'local';
const out = document.getElementById('out');
const lines = [];
const say = (m) => { lines.push(m); out.textContent = lines.slice(-40).join('\n'); };

async function fetchSet(dir, pattern, start, end, stride = 1, max = 1000) {
  const files = [];
  for (let i = start; i <= end && files.length < max; i += stride) {
    const name = pattern.replace(/\{i:(\d+)\}/, (_, w) => String(i).padStart(+w, '0'));
    const resp = await fetch(`../../data/${dir}/${name}`);
    if (!resp.ok) throw new Error(`dataset missing: data/${dir}/${name}`);
    files.push({ source: await resp.blob(), name });
  }
  return files;
}

async function post(result) {
  result.scene = scene;
  result.gpu = !!navigator.gpu;
  await fetch(`/scratch/quality_${runId}.json`, { method: 'POST', body: JSON.stringify(result) });
  say('\nresult posted: ' + JSON.stringify(result, null, 1));
}

const SETS = {
  synthetic: { dir: 'synthetic', pattern: 'synthetic_{i:2}.png', start: 0, end: 11 },
  truck: { dir: 'truck', pattern: '{i:6}.jpg', start: 1, end: 42 },
  camping: { dir: 'camping', pattern: 'frame_{i:5}.jpg', start: 1, end: 113 },
};

async function solveSet(setName, session) {
  const s = SETS[setName];
  say(`fetching ${setName} ...`);
  const files = await fetchSet(s.dir, s.pattern, s.start, s.end);
  await session.load(files);
  say(`solving ${files.length} frames ...`);
  const t0 = performance.now();
  const recon = await session.solve();
  const solveSec = (performance.now() - t0) / 1000;
  say(`solved: ${recon.cams.length}/${files.length} cams, rms ${recon.rmsBA && recon.rmsBA.toFixed(2)}px, ${solveSec.toFixed(0)}s`);
  return { recon, solveSec, total: files.length };
}

try {
  const session = createSession({ maxIters: 3000, evalHoldEvery: 1500 });
  session.on('log', say);

  if (scene === 'synthetic-solve' || scene === 'synthetic-train') {
    const { recon, solveSec, total } = await solveSet('synthetic', session);
    const gt = await (await fetch('../../data/synthetic/cameras.json')).json();
    const fGT = gt.cameras[0].fx;
    const result = {
      cams: recon.cams.length, total,
      fEst: recon.cams[0].f, fGT, fErr: Math.abs(recon.cams[0].f - fGT) / fGT,
      rmsBA: recon.rmsBA, solveSec,
    };
    if (scene === 'synthetic-train') {
      await session.seed();
      say('training 3000 iterations ...');
      const t1 = performance.now();
      await new Promise((res) => { session.on('event', (e) => { if (e.kind === 'train-complete') res(); }); session.start(); });
      result.trainSec = (performance.now() - t1) / 1000;
      const m = await session.metrics();
      result.iter = m.iter;
      result.splats = m.splats;
      result.psnrHold = m.psnrHold;
      const hist = session.lossHistory;
      result.psnrTrain = hist.length ? hist[hist.length - 1][1] : null;
      // interactive view render at a LARGER-than-training canvas (the black-
      // render class of bug: view buffers sized only for the training res)
      const cv = document.createElement('canvas');
      cv.width = 1200; cv.height = 800;
      session.view.attach(cv);
      const meta = session.trainer.camMeta[0];
      const s = Math.min(1200 / meta.w, 800 / meta.h);
      session.view.setCamera({
        R: meta.R, t: meta.t, f: meta.f * s,
        cx: 600, cy: 400, w: 1200, h: 800,
      });
      session.view.renderNow();
      const c2 = document.createElement('canvas');
      c2.width = 64; c2.height = 64;
      const x2 = c2.getContext('2d');
      x2.drawImage(cv, 0, 0, 64, 64);
      const px = x2.getImageData(0, 0, 64, 64).data;
      let sum = 0;
      for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
      result.viewPixelSum = sum;
    }
    await post(result);
  } else if (scene === 'truck-ate' || scene === 'camping-ate') {
    const setName = scene.split('-')[0];
    const { recon, solveSec, total } = await solveSet(setName, session);
    await post({
      cams: recon.cams.length, total, rmsBA: recon.rmsBA, solveSec,
      poses: recon.cams.map((c) => ({
        name: session.frames[c.imgIdx].name, R: Array.from(c.R), t: Array.from(c.t), f: c.f,
      })),
    });
  } else {
    throw new Error('unknown scene: ' + scene);
  }
  say('DONE');
} catch (e) {
  say('ERROR: ' + (e.stack || e.message));
  await post({ error: String(e.message || e) });
}
