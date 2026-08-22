// tests/quality/run.mjs — library-level quality gates in a real browser.
//
//   node tests/quality/run.mjs [scene ...] [--headed]
//
// Scenes (default: synthetic-solve synthetic-train, plus truck-ate and
// camping-ate when their datasets are present):
//   synthetic-solve   12/12 cams, focal within 2% of GT, BA rms floor
//   synthetic-train   3k-iteration training run, PSNR floors (needs WebGPU)
//   truck-ate         SfM camera path vs COLMAP GT (ATE % of path)
//   camping-ate       SfM camera path vs server-COLMAP GT
//
// Spawns its own static server (port 8763) and one Chrome per scene
// (--headless=new with WebGPU; falls back to a visible window when headless
// WebGPU is unavailable). Thresholds live in thresholds.json next to this
// file — every number is a measured baseline minus an explicit margin.

import { spawn, spawnSync } from 'child_process';
import { existsSync, readFileSync, writeFileSync, rmSync, mkdtempSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { tmpdir } from 'os';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
const PORT = 8763;
const thresholds = JSON.parse(readFileSync(join(here, 'thresholds.json'), 'utf8'));

const argv = process.argv.slice(2);
const headed = argv.includes('--headed');
let scenes = argv.filter((a) => !a.startsWith('--'));
if (!scenes.length) {
  scenes = ['synthetic-solve', 'synthetic-train'];
  if (existsSync(join(root, 'data/truck/000001.jpg'))) scenes.push('truck-ate');
  if (existsSync(join(root, 'data/camping/frame_00001.jpg'))) scenes.push('camping-ate');
}

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const cands = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    (process.env.LOCALAPPDATA || '') + '/Google/Chrome/Application/chrome.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ];
  for (const c of cands) if (c && existsSync(c)) return c;
  throw new Error('Chrome not found — set the CHROME env var');
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function runScene(chrome, scene, { headless }) {
  const runId = `${scene}-${process.pid}-${Math.floor(Math.random() * 1e6)}`;
  const resultFile = join(root, 'scratch', `quality_${runId}.json`);
  const profile = mkdtempSync(join(tmpdir(), 'splatjs-quality-'));
  const url = `http://localhost:${PORT}/tests/quality/?scene=${scene}&run=${runId}`;
  const args = [
    ...(headless ? ['--headless=new'] : ['--window-position=40,40', '--window-size=900,700']),
    '--enable-unsafe-webgpu', '--use-angle=d3d11',
    '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    `--user-data-dir=${profile}`,
    url,
  ];
  const proc = spawn(chrome, args, { stdio: 'ignore' });
  const timeoutMin = thresholds[scene].timeoutMin || 6;
  const deadline = Date.now() + timeoutMin * 60_000;
  let result = null;
  while (Date.now() < deadline) {
    await sleep(1500);
    if (existsSync(resultFile)) {
      try { result = JSON.parse(readFileSync(resultFile, 'utf8')); break; } catch {}
    }
    if (proc.exitCode !== null) break; // chrome died
  }
  proc.kill();
  try { rmSync(profile, { recursive: true, force: true }); } catch {}
  if (!result) throw new Error(`no result within ${timeoutMin}min (chrome exit ${proc.exitCode})`);
  return result;
}

function ateFromPoses(scene, result) {
  // reuse the standalone comparer: write poses, parse its ATE line
  const gt = scene === 'truck-ate'
    ? join(root, 'data/downloads/extracted/tandt/truck/sparse/0/images.bin')
    : join(root, 'data/camping/gt_images.txt');
  if (!existsSync(gt)) return { skip: `GT not present (${gt})` };
  const posesFile = join(root, 'scratch', `quality_poses_${scene}.json`);
  writeFileSync(posesFile, JSON.stringify({ cams: result.poses }));
  const r = spawnSync(process.execPath, [join(root, 'tests/compare_colmap.mjs'), gt, posesFile], { encoding: 'utf8' });
  const m = (r.stdout || '').match(/ATE rms [\d.]+\s+median [\d.]+\s+max [\d.]+\s+\(([\d.]+)% of path\)/);
  if (!m) throw new Error('ATE parse failed:\n' + r.stdout + r.stderr);
  return { atePct: parseFloat(m[1]) };
}

function check(name, value, limit, cmp = '<=') {
  const ok = cmp === '<=' ? value <= limit : value >= limit;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name} = ${typeof value === 'number' ? +value.toFixed(4) : value} (${cmp} ${limit})`);
  return ok;
}

// ---- main ----
const chrome = findChrome();
const server = spawn(process.execPath, [join(root, 'serve.mjs'), String(PORT)], { stdio: 'ignore' });
await sleep(700);

let failed = 0;
try {
  for (const scene of scenes) {
    const th = thresholds[scene];
    if (!th) { console.log(`${scene}: no thresholds defined — skipping`); continue; }
    console.log(`\n=== ${scene} ===`);
    let result;
    try {
      result = await runScene(chrome, scene, { headless: !headed });
      if (result.error) throw new Error(result.error);
      // training needs real WebGPU; retry headed if the headless run had none
      if (scene.includes('train') && !result.gpu) throw new Error('no WebGPU headless');
    } catch (e) {
      if (!headed) {
        console.log(`  headless failed (${e.message}) — retrying with a visible window`);
        result = await runScene(chrome, scene, { headless: false });
        if (result.error) throw new Error(result.error);
      } else {
        throw e;
      }
    }
    let ok = true;
    if (th.cams) ok = check('registered cams', result.cams, th.cams, '>=') && ok;
    if (th.fErrMax != null) ok = check('focal rel. error', result.fErr, th.fErrMax) && ok;
    if (th.rmsMax != null) ok = check('BA rms px', result.rmsBA, th.rmsMax) && ok;
    if (th.psnrTrainMin != null) ok = check('train PSNR dB', result.psnrTrain, th.psnrTrainMin, '>=') && ok;
    if (th.psnrHoldMin != null) ok = check('holdout PSNR dB', result.psnrHold, th.psnrHoldMin, '>=') && ok;
    if (result.viewPixelSum != null) ok = check('view render pixel sum', result.viewPixelSum, 1000, '>=') && ok;
    if (th.atePctMax != null) {
      const a = ateFromPoses(scene, result);
      if (a.skip) console.log(`  skip ATE: ${a.skip}`);
      else ok = check('ATE % of path', a.atePct, th.atePctMax) && ok;
    }
    if (!ok) failed++;
  }
} finally {
  server.kill();
}
console.log(failed ? `\n${failed} scene(s) FAILED` : '\nall quality gates passed');
process.exit(failed ? 1 : 0);
