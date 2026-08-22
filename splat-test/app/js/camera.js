// camera.js — record a capture walk with the device camera, in the app.
//
// getUserMedia (environment camera, 1080p-ish) -> MediaRecorder -> a video
// blob that feeds the same sharp-frame extraction as an uploaded file.
//
// Auto-exposure drift is poison for splat training (the same wall changes
// brightness between frames and the optimiser blames geometry), so once the
// preview has settled we LOCK exposure and white balance where the platform
// allows it (Android Chrome exposes exposureMode/whiteBalanceMode constraints;
// iOS Safari does not — reported in the UI either way).

function pickMime() {
  const cands = [
    'video/mp4;codecs=avc1', 'video/mp4',
    'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm',
  ];
  for (const m of cands) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  }
  return '';
}

export function cameraSupported() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder && pickMime());
}

/** Try to freeze exposure + white balance at their current auto values.
 *  Returns a short human-readable summary of what could be locked. */
async function lockExposure(track) {
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  const locked = [];
  const wants = [];
  if (Array.isArray(caps.exposureMode) && caps.exposureMode.includes('manual')) {
    const cur = track.getSettings().exposureTime;
    const c = { exposureMode: 'manual' };
    if (cur && caps.exposureTime) c.exposureTime = cur;
    wants.push([c, 'exposure']);
  }
  if (Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.includes('manual')) {
    const cur = track.getSettings().colorTemperature;
    const c = { whiteBalanceMode: 'manual' };
    if (cur && caps.colorTemperature) c.colorTemperature = cur;
    wants.push([c, 'white balance']);
  }
  for (const [c, name] of wants) {
    try {
      await track.applyConstraints({ advanced: [c] });
      locked.push(name);
    } catch { /* the capability lied — leave it on auto */ }
  }
  return locked.length ? `${locked.join(' + ')} locked` : 'auto exposure (lock not supported here)';
}

/**
 * Full-screen capture UI with two modes:
 *   video   record a walk, resolve { kind: 'video', file }
 *   photos  tap the shutter per shot (stills grabbed from the live stream —
 *           no codec loss, no walking blur), resolve { kind: 'photos', files }
 * Resolves null on cancel. Exposure locking is identical in both modes: it is
 * a property of the camera TRACK, so a photo series does not gain a lock on
 * platforms (iOS) that refuse the constraint — it gains clean, deliberate
 * frames instead.
 */
export async function recordCaptureVideo() {
  const mime = pickMime();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 1920 }, height: { ideal: 1080 },
      frameRate: { ideal: 30 },
    },
  });
  const track = stream.getVideoTracks()[0];

  const ui = document.createElement('div');
  ui.className = 'camrec';
  ui.innerHTML = `
    <video id="cam-view" autoplay muted playsinline></video>
    <div class="camrec-mode" id="cam-mode">
      <button data-m="video" aria-pressed="true">Video</button>
      <button data-m="photos" aria-pressed="false">Photos</button>
    </div>
    <button class="camrec-x" id="cam-cancel" aria-label="Cancel">&times;</button>
    <div class="camrec-hint" id="cam-hint">Move <b>sideways</b>, slowly. Keep the subject in frame — a wide arc beats a spin.</div>
    <div class="camrec-row">
      <span></span>
      <button class="camrec-btn" id="cam-rec" aria-label="Capture"></button>
      <span class="camrec-time" id="cam-time">0:00</span>
    </div>
    <button class="btn btn-accent camrec-done" id="cam-done" hidden>Use 0 photos</button>`;
  document.body.appendChild(ui);
  const view = ui.querySelector('#cam-view');
  view.srcObject = stream;
  const el = (id) => ui.querySelector('#' + id);

  const HINTS = {
    video: 'Move <b>sideways</b>, slowly. Keep the subject in frame — a wide arc beats a spin.',
    photos: 'A <b>step sideways</b> between shots. 30–80 overlapping photos of one place.',
  };

  return new Promise((resolve) => {
    // video capture is OFF for now — the sharp-frame extraction downstream is
    // not good enough yet. All video code below is kept working; flip this to
    // bring the mode pill and recording back.
    const VIDEO_CAPTURE = false;
    let mode = 'video';
    let rec = null, chunks = [], t0 = 0, timer = 0;
    const photos = [];
    // Space/Enter = the shutter (snap in photo mode, start/stop in video)
    const onKey = (e) => {
      if (e.code === 'Space' || e.code === 'Enter') {
        e.preventDefault();
        el('cam-rec').click();
      }
      if (e.code === 'Escape') el('cam-cancel').click();
    };
    addEventListener('keydown', onKey);
    const teardown = () => {
      removeEventListener('keydown', onKey);
      clearInterval(timer);
      track.stop();
      stream.getTracks().forEach((t) => t.stop());
      ui.remove();
    };

    // let auto-exposure settle on the scene for a moment, then freeze it
    // (silently — succeeds on Android, is refused on iOS)
    setTimeout(() => { lockExposure(track).catch(() => {}); }, 1200);

    el('cam-mode').addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b || (rec && rec.state !== 'inactive')) return;   // not mid-recording
      mode = b.dataset.m;
      [...el('cam-mode').children].forEach((x) =>
        x.setAttribute('aria-pressed', String(x === b)));
      el('cam-hint').innerHTML = HINTS[mode];
      el('cam-rec').dataset.mode = mode;
      el('cam-time').textContent = mode === 'photos' ? `${photos.length} shots` : '0:00';
      el('cam-done').hidden = !(mode === 'photos' && photos.length >= 2);
    });
    if (!VIDEO_CAPTURE) {
      // photos only: land in the stills mode through the normal switch, then
      // take the choice away
      el('cam-mode').querySelector('[data-m="photos"]').click();
      el('cam-mode').hidden = true;
    }

    el('cam-cancel').addEventListener('click', () => {
      if (rec && rec.state !== 'inactive') rec.stop();
      teardown();
      resolve(null);
    });

    el('cam-done').addEventListener('click', () => {
      teardown();
      resolve({ kind: 'photos', files: photos.slice() });
    });

    const snap = async () => {
      // capped at 640 wide for now — keeps phone solves/training light
      const sc = Math.min(1, 640 / view.videoWidth);
      const w = Math.round(view.videoWidth * sc), h = Math.round(view.videoHeight * sc);
      const cv = document.createElement('canvas');
      cv.width = w; cv.height = h;
      const c2 = cv.getContext('2d');
      c2.imageSmoothingEnabled = true;
      c2.imageSmoothingQuality = 'high';
      c2.drawImage(view, 0, 0, w, h);
      const blob = await new Promise((res, rej) => cv.toBlob(
        (b) => (b ? res(b) : rej(new Error('jpeg encode failed'))), 'image/jpeg', 0.95));
      photos.push(new File([blob], `shot_${String(photos.length + 1).padStart(4, '0')}.jpg`, { type: 'image/jpeg' }));
      el('cam-time').textContent = `${photos.length} shots`;
      el('cam-done').hidden = photos.length < 2;
      el('cam-done').textContent = `Use ${photos.length} photos`;
      ui.classList.add('camrec-flash');
      setTimeout(() => ui.classList.remove('camrec-flash'), 120);
    };

    el('cam-rec').addEventListener('click', function () {
      if (mode === 'photos') { snap(); return; }
      if (!rec) {
        rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 16e6 });
        chunks = [];
        rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
        rec.onstop = () => {
          const type = mime.split(';')[0];
          const ext = type.includes('mp4') ? 'mp4' : 'webm';
          teardown();
          resolve({ kind: 'video', file: new File([new Blob(chunks, { type })], `capture.${ext}`, { type }) });
        };
        rec.start(1000);
        t0 = performance.now();
        this.dataset.on = '1';
        timer = setInterval(() => {
          const s = Math.floor((performance.now() - t0) / 1000);
          const t = el('cam-time');
          if (t) t.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
        }, 500);
      } else if (rec.state !== 'inactive') {
        rec.stop();
      }
    });
  });
}
