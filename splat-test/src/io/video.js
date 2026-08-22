// io/video.js — turn a video file into training photographs, in the browser.
//
// Users film places; they rarely have photo sets. This mirrors the sharp-frames
// policy of the arrival.space server pipeline (sample ~10/s, score sharpness,
// pick the best N with a minimum temporal buffer between picks — motion-blurred
// frames lose to their sharp neighbours), then re-captures the winners at full
// resolution as JPEG blobs the rest of the pipeline treats like photo files.
//
// Two passes over a muted <video>:
//   scan     playback at up to 3x with requestVideoFrameCallback, scoring a
//            downscaled grayscale Laplacian variance per ~0.1s of video time
//            (seek-stepping fallback when rVFC is unavailable)
//   capture  seek to the selected timestamps in order (fast, forward-only
//            decode) and encode full-resolution JPEGs
//
// No dependencies, no DOM attachment; works wherever <video> can decode the
// file (H.264/HEVC .mp4/.mov on Safari and Chrome, plus webm).

/**
 * @typedef {object} VideoExtractOptions
 * @property {number} [samplesPerSec=10]   sharpness sampling rate (video time)
 * @property {number} [targetFrames]       frames to keep; default scales with
 *   duration: clamp(round(4/s), 24, 140)
 * @property {number} [minBufferSec]       minimum spacing between kept frames;
 *   default 2 samples (0.2s). The server pipeline uses 3, but it also keeps
 *   300 frames — our smaller sets need the density (measured: video-camping
 *   at 0.3s gaps behaves like the stride-2 sets that register worse)
 * @property {number} [jpegQuality=0.93]
 * @property {(e: {stage: 'scan'|'capture', done: number, total: number}) => void} [onProgress]
 * @property {(msg: string) => void} [log]
 */

const until = (el, ev, err = 'error') => new Promise((res, rej) => {
  const ok = () => { cleanup(); res(); };
  const bad = (e) => { cleanup(); rej(new Error(`video ${err}: ${(e && e.message) || 'decode failed'}`)); };
  const cleanup = () => { el.removeEventListener(ev, ok); el.removeEventListener('error', bad); };
  el.addEventListener(ev, ok, { once: true });
  el.addEventListener('error', bad, { once: true });
});

/** Laplacian variance of a grayscale buffer (same sharpness measure the
 *  frame decoder uses for blur exclusion). */
function lapVar(gray, w, h) {
  let sum = 0, sq = 0;
  const n = (w - 2) * (h - 2);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - w] - gray[i + w];
      sum += lap; sq += lap * lap;
    }
  }
  const mean = sum / n;
  return sq / n - mean * mean;
}

/**
 * @param {File|Blob} file
 * @param {VideoExtractOptions} [opts]
 * @returns {Promise<{frames: Array<{source: Blob, name: string}>,
 *   duration: number, sampled: number, videoW: number, videoH: number}>}
 */
export async function extractSharpFrames(file, opts = {}) {
  const log = opts.log || (() => {});
  const onProgress = opts.onProgress || (() => {});
  const sps = opts.samplesPerSec ?? 10;

  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = url;
  try {
    await until(video, 'loadedmetadata');
    if (!isFinite(video.duration)) {
      // streamed/recorded webms report Infinity until forced to the end
      video.currentTime = 1e9;
      await until(video, 'seeked');
      video.currentTime = 0;
      await until(video, 'seeked');
    }
    const duration = video.duration;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh || !isFinite(duration) || duration <= 0) throw new Error('video has no decodable track');
    log(`video: ${vw}x${vh}, ${duration.toFixed(1)}s`);

    // ---- pass 1: sharpness scan on a small grayscale ----
    const sw = 320, sh = Math.max(2, Math.round(sw * vh / vw));
    const scanCv = mkCanvas(sw, sh);
    const scanCtx = scanCv.getContext('2d', { willReadFrequently: true });
    const gray = new Float32Array(sw * sh);
    const scoreNow = (t) => {
      scanCtx.drawImage(video, 0, 0, sw, sh);
      const d = scanCtx.getImageData(0, 0, sw, sh).data;
      for (let i = 0; i < sw * sh; i++) {
        gray[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255;
      }
      return { t, s: lapVar(gray, sw, sh) };
    };

    const samples = [];
    const totalSamples = Math.max(2, Math.floor(duration * sps));
    if (typeof video.requestVideoFrameCallback === 'function') {
      // real-time-ish scan: play fast, score presented frames ~1/sps apart
      video.playbackRate = Math.min(3, (video.canPlayType ? 3 : 1));
      let lastT = -1;
      let done = false;
      const onFrame = (_now, meta) => {
        if (done) return;
        const t = meta.mediaTime;
        if (t - lastT >= 1 / sps - 1e-3) {
          lastT = t;
          samples.push(scoreNow(t));
          onProgress({ stage: 'scan', done: Math.min(samples.length, totalSamples), total: totalSamples });
        }
        video.requestVideoFrameCallback(onFrame);
      };
      video.requestVideoFrameCallback(onFrame);
      await video.play();
      await until(video, 'ended', 'playback error');
      done = true;
      video.pause();
    } else {
      // fallback: seek-step through the video
      for (let k = 0; k < totalSamples; k++) {
        video.currentTime = Math.min(duration - 0.001, k / sps);
        await until(video, 'seeked');
        samples.push(scoreNow(video.currentTime));
        onProgress({ stage: 'scan', done: k + 1, total: totalSamples });
      }
    }
    if (samples.length < 2) throw new Error('could not decode frames from this video');
    log(`scanned ${samples.length} samples`);

    // ---- selection: best-n by sharpness with a minimum temporal buffer ----
    const target = opts.targetFrames ??
      Math.max(24, Math.min(140, Math.round(duration * 4)));
    const minGap = opts.minBufferSec ?? (2 / sps);
    const bySharp = [...samples].sort((a, b) => b.s - a.s);
    const picked = [];
    for (const cand of bySharp) {
      if (picked.length >= target) break;
      if (picked.some((p) => Math.abs(p.t - cand.t) < minGap)) continue;
      picked.push(cand);
    }
    picked.sort((a, b) => a.t - b.t);
    log(`selected ${picked.length} sharp frames (target ${target}, min gap ${minGap.toFixed(2)}s)`);

    // ---- pass 2: full-resolution capture at the selected times ----
    const capCv = mkCanvas(vw, vh);
    const capCtx = capCv.getContext('2d');
    const frames = [];
    for (let i = 0; i < picked.length; i++) {
      video.currentTime = picked[i].t;
      await until(video, 'seeked');
      capCtx.drawImage(video, 0, 0, vw, vh);
      const blob = await toBlob(capCv, opts.jpegQuality ?? 0.93);
      frames.push({ source: blob, name: `frame_${String(i + 1).padStart(5, '0')}.jpg` });
      onProgress({ stage: 'capture', done: i + 1, total: picked.length });
    }
    return { frames, duration, sampled: samples.length, videoW: vw, videoH: vh };
  } finally {
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}

function mkCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  return cv;
}

function toBlob(cv, quality) {
  if (cv.convertToBlob) return cv.convertToBlob({ type: 'image/jpeg', quality });
  return new Promise((res, rej) => cv.toBlob(
    (b) => (b ? res(b) : rej(new Error('jpeg encode failed'))), 'image/jpeg', quality));
}

/** Quick sniff: is this file a video the pipeline should extract from? */
export function isVideoFile(f) {
  return /^video\//.test(f.type) || /\.(mp4|mov|m4v|webm|mkv)$/i.test(f.name || '');
}
