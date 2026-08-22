// io/frames.js — decode & downscale input photographs into pipeline Frames.
//
// A Frame carries two resolutions of the same photograph:
//   feature scale — grayscale for SfM (corner/SIFT detection)
//   training scale — RGB float for the 3DGS photometric loss
//
// No DOM requirement: uses OffscreenCanvas when available (workers, headless)
// and falls back to document canvases in a page.

/**
 * @typedef {object} Frame
 * @property {string} name          source file name
 * @property {number} fw            feature-scale width
 * @property {number} fh            feature-scale height
 * @property {Float32Array} gray    feature-scale grayscale, fw*fh, 0..1
 * @property {number} tw            training-scale width
 * @property {number} th            training-scale height
 * @property {Float32Array} rgb     training-scale RGB, tw*th*3, 0..1
 * @property {number} sharpness    Laplacian variance (motion-blur indicator)
 * @property {HTMLCanvasElement|OffscreenCanvas} thumb  small preview
 * @property {(x:number, y:number) => number[]} sampleColor  training-res RGB at
 *   feature-scale pixel coords
 */

/**
 * @typedef {object} FrameOptions
 * @property {number} [featMaxDim=960]   feature-scale max dimension. 960 (was
 *   640): SfM feature localization is the pose-precision ceiling — measured on
 *   camping vs the server-COLMAP reference: 0.44% -> 0.27% ATE, tail halved.
 * @property {number} [trainMaxDim]      training-scale max dimension override.
 *   Default: native resolution up to 1600, shrunk only if the whole set would
 *   blow the GPU target budget (see adaptiveTrainCap).
 * @property {number} [targetBudgetBytes=700e6]  GPU budget for the training
 *   target buffer (all images, RGB float32).
 * @property {(msg: string) => void} [log]
 */

export const FEAT_MAX_DIM = 960;
export const TRAIN_MAX_DIM = 1600; // hard ceiling; actual res = native, memory permitting
const TARGET_BUDGET_BYTES = 700e6;

/** Train at the PROVIDED resolution up to trainMaxDim, shrunk only if the
 *  whole image set would blow the GPU target budget. Call once per dataset
 *  (needs the image count and one representative size); pass the result to
 *  processSource. An explicit opts.trainMaxDim wins unconditionally. */
export function adaptiveTrainCap(nImages, w, h, opts = {}) {
  const nat = Math.max(w, h);
  const scale = opts.trainScale || 1;
  // trainMaxDim CAPS the input scale (never upscales); trainScale then
  // multiplies the working buffer on top (may upscale — the photos are only
  // re-gridded; supervising above input res suppresses super-resolution
  // ringing at the cost of native-res PSNR and speed)
  if (opts.trainMaxDim) return Math.round(Math.min(opts.trainMaxDim, nat) * scale);
  if (scale !== 1) return Math.round(nat * scale);
  const native = Math.max(w, h);
  const full = Math.min(native, TRAIN_MAX_DIM);
  const [fw, fh] = fitDims(w, h, full);
  const bytes = nImages * fw * fh * 12; // 3 channels x float32
  const budget = opts.targetBudgetBytes || TARGET_BUDGET_BYTES;
  if (bytes <= budget) return full;
  const s = Math.sqrt(budget / bytes);
  return Math.max(320, Math.floor(full * s));
}

function fitDims(w, h, maxDim, allowUp = false) {
  const s0 = maxDim / Math.max(w, h);
  const s = allowUp ? s0 : Math.min(1, s0);
  return [Math.max(2, Math.round(w * s)), Math.max(2, Math.round(h * s))];
}

function mkCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  return cv;
}

/** High-quality downscale: iterative halving until within 2x of the target,
 *  then a final filtered draw. A single drawImage at large factors decimates
 *  (effectively point-samples), leaving aliasing in the training targets and
 *  the SfM grayscale. */
function drawScaled(src, srcW, srcH, w, h) {
  let cur = src, cw = srcW, ch = srcH;
  while (cw >= 2 * w && ch >= 2 * h) {
    const nw = Math.max(w, Math.round(cw / 2));
    const nh = Math.max(h, Math.round(ch / 2));
    const cv = mkCanvas(nw, nh);
    const cctx = cv.getContext('2d');
    cctx.imageSmoothingEnabled = true;
    cctx.imageSmoothingQuality = 'high';
    cctx.drawImage(cur, 0, 0, cw, ch, 0, 0, nw, nh);
    cur = cv; cw = nw; ch = nh;
  }
  const cv = mkCanvas(w, h);
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(cur, 0, 0, cw, ch, 0, 0, w, h);
  return ctx;
}

/** Single-step downscale (default bilinear). Deliberately used for the SfM
 *  grayscale: the residual high-frequency detail (incl. mild aliasing) is
 *  discriminative texture for corner detection — the smooth variant registers
 *  measurably fewer cameras. */
function drawScaledFast(src, w, h) {
  const cv = mkCanvas(w, h);
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(src, 0, 0, w, h);
  return ctx;
}

/** Process one drawable source (ImageBitmap or canvas) into a Frame.
 *  trainCap: per-dataset training resolution from adaptiveTrainCap(). */
export function processSource(src, srcW, srcH, name, trainCap, opts = {}) {
  const featCap = opts.featMaxDim || FEAT_MAX_DIM;
  const [fw, fh] = fitDims(srcW, srcH, featCap);
  // only an explicit trainScale may push the working buffer past native
  const [tw, th] = fitDims(srcW, srcH, trainCap || TRAIN_MAX_DIM, !!opts.trainScale);

  const fctx = drawScaledFast(src, fw, fh);
  const fdata = fctx.getImageData(0, 0, fw, fh).data;
  const gray = new Float32Array(fw * fh);
  for (let i = 0; i < fw * fh; i++) {
    gray[i] = (0.299 * fdata[i * 4] + 0.587 * fdata[i * 4 + 1] + 0.114 * fdata[i * 4 + 2]) / 255;
  }

  // sharpness = Laplacian variance (motion-blur indicator; used to exclude
  // the blurriest frames from the training loss)
  let lapSum = 0, lapSq = 0;
  const nl = (fw - 2) * (fh - 2);
  for (let y = 1; y < fh - 1; y++) {
    for (let x = 1; x < fw - 1; x++) {
      const i = y * fw + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - fw] - gray[i + fw];
      lapSum += lap; lapSq += lap * lap;
    }
  }
  const lapMean = lapSum / nl;
  const sharpness = lapSq / nl - lapMean * lapMean;

  const tctx = drawScaled(src, srcW, srcH, tw, th);
  const tdata = tctx.getImageData(0, 0, tw, th).data;
  const rgb = new Float32Array(tw * th * 3);
  for (let i = 0; i < tw * th; i++) {
    rgb[i * 3] = tdata[i * 4] / 255;
    rgb[i * 3 + 1] = tdata[i * 4 + 1] / 255;
    rgb[i * 3 + 2] = tdata[i * 4 + 2] / 255;
  }

  // small thumbnail for UIs
  const [thw, thh] = fitDims(srcW, srcH, 96);
  const thumbCtx = drawScaled(src, srcW, srcH, thw, thh);

  return {
    name, fw, fh, gray, tw, th, rgb, sharpness,
    thumb: thumbCtx.canvas,
    /** sample training-res RGB at feature-scale pixel coords */
    sampleColor(x, y) {
      const sx = Math.min(tw - 1, Math.max(0, Math.round(x * tw / fw)));
      const sy = Math.min(th - 1, Math.max(0, Math.round(y * th / fh)));
      const i = (sy * tw + sx) * 3;
      return [rgb[i], rgb[i + 1], rgb[i + 2]];
    },
  };
}

/** Decode image Files/Blobs (or {source, name} pairs of anything
 *  createImageBitmap accepts) into Frames.
 *  @param {Array<File|Blob|{source:*, name:string}>} files
 *  @param {FrameOptions} [opts]
 *  @returns {Promise<Frame[]>} */
export async function decodeFrames(files, opts = {}) {
  const log = opts.log || (() => {});
  const out = [];
  let trainCap = 0;
  for (const file of files) {
    const source = file.source || file;
    const name = file.name || 'frame';
    try {
      const bmp = await createImageBitmap(source);
      if (!trainCap) {
        trainCap = adaptiveTrainCap(files.length, bmp.width, bmp.height, opts);
        log(`training resolution: ${trainCap}px max dim (${files.length} images)`);
      }
      out.push(processSource(bmp, bmp.width, bmp.height, name, trainCap, opts));
      bmp.close();
    } catch (e) {
      log(`skipped ${name}: ${e.message}`);
    }
  }
  return out;
}

/** @deprecated use decodeFrames(files, { log }) */
export async function loadImageFiles(files, log) {
  return decodeFrames(files, { log });
}
