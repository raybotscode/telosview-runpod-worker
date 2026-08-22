// develop.js — the comparison surface.
//
// There is only ever one render: the model, already on the canvas, drawn from
// this frame's own pose. This module lays that frame's photograph over it and
// takes some of it away again — a wipe, a hole, or a difference map. It never
// renders the model itself.

export function fitRect(iw, ih, w, h, pad = 0) {
  const s = Math.min((w - pad * 2) / iw, (h - pad * 2) / ih);
  const rw = iw * s, rh = ih * s;
  return { x: (w - rw) / 2, y: (h - rh) / 2, w: rw, h: rh, s };
}

export class Developer {
  constructor() {
    this.bmp = null;
    this.url = null;
    this.err = document.createElement('canvas');
    this.errKey = '';
  }

  /** hand it an already-decoded photograph (the app keeps one shared cache) */
  setBitmap(bitmap, url) {
    if (this.url === url) return;
    this.url = url;
    this.bmp = bitmap;
    this.errKey = '';
  }

  get ready() { return !!this.bmp; }

  /**
   * Per-pixel disagreement between what is on the canvas and the photograph.
   * Reads the render back rather than simulating it, which is exactly what the
   * wired-up version does with the trainer's own framebuffer.
   */
  _error(ctx, r, dpr, key, model) {
    if (this.errKey === key) return this.err;
    this.errKey = key;

    const ew = 260, eh = Math.max(2, Math.round(ew * r.h / r.w));
    this.err.width = ew; this.err.height = eh;
    const ec = this.err.getContext('2d', { willReadFrequently: true });

    // `model` is a dedicated same-camera render snapshot handed in by the app
    // (never the live WebGPU canvas: reading one back after presentation can
    // return blank pixels). Render and photo are the same camera image, so
    // both just downsample to the map's working size.
    ec.drawImage(model, 0, 0, ew, eh);
    const a = ec.getImageData(0, 0, ew, eh);
    ec.clearRect(0, 0, ew, eh);
    ec.drawImage(this.bmp, 0, 0, ew, eh);
    const b = ec.getImageData(0, 0, ew, eh);

    const out = ec.createImageData(ew, eh);
    for (let i = 0; i < a.data.length; i += 4) {
      const d = (Math.abs(a.data[i] - b.data[i]) + Math.abs(a.data[i + 1] - b.data[i + 1])
               + Math.abs(a.data[i + 2] - b.data[i + 2])) / 3;
      const u = Math.min(1, d / 70);
      // cool where the model agrees, hot where it does not
      out.data[i]     = 20 + 235 * Math.pow(u, .75);
      out.data[i + 1] = 40 + 150 * Math.max(0, Math.sin(u * Math.PI * .95));
      out.data[i + 2] = 45 + 90 * Math.max(0, 1 - u * 2.4);
      out.data[i + 3] = 255;
    }
    ec.putImageData(out, 0, 0);
    return this.err;
  }

  /**
   * @param o { mode:'render'|'photo'|'loupe'|'swipe'|'error', loupe:{x,y,r},
   *            swipe:0..1, dpr, key }  — `key` only decides when to recompute
   *            the error map, so pass something that changes with the render.
   * @returns the fit rect, so callers can align overlays
   */
  render(ctx, w, h, o) {
    const r = fitRect(this.bmp.width, this.bmp.height, w, h, 0);
    const drawPhoto = () => ctx.drawImage(this.bmp, r.x, r.y, r.w, r.h);

    if (o.mode === 'render') return r;              // the model is already there
    if (o.mode === 'photo') { drawPhoto(); return r; }
    if (o.mode === 'error') {
      if (!o.model) return r;   // snapshot not ready yet — next frame
      const e = this._error(ctx, r, o.dpr || 1, o.key || '', o.model);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(e, r.x, r.y, r.w, r.h);
      return r;
    }

    // The photograph is the top layer. Every reveal takes some of it away and
    // the render underneath shows through.
    const label = (text, x, align, tone) => {
      ctx.font = '500 10px "Spline Sans Mono", monospace';
      ctx.textAlign = align; ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = 'rgba(7,9,9,.75)';
      ctx.fillText(text, x + (align === 'left' ? 1 : -1), r.y + r.h - 11);
      ctx.fillStyle = tone;
      ctx.fillText(text, x, r.y + r.h - 12);
    };

    if (o.mode === 'swipe') {
      const x = r.x + r.w * (o.swipe ?? .5);
      ctx.save();                                  // photograph, up to the divider
      ctx.beginPath(); ctx.rect(r.x, r.y, x - r.x, r.h); ctx.clip();
      drawPhoto();
      ctx.restore();

      ctx.strokeStyle = 'rgba(47,212,193,.9)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, r.y); ctx.lineTo(x, r.y + r.h); ctx.stroke();
      ctx.fillStyle = 'rgba(47,212,193,.9)';
      ctx.beginPath(); ctx.arc(x, r.y + r.h / 2, 7, 0, 7); ctx.fill();
      ctx.fillStyle = '#04231f';
      ctx.font = '600 9px "Spline Sans Mono", monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('↔', x, r.y + r.h / 2 + .5);

      if (x - r.x > 96) label('PHOTOGRAPH', r.x + 12, 'left', 'rgba(230,236,235,.9)');
      if (r.x + r.w - x > 76) label('RENDER', r.x + r.w - 12, 'right', 'rgba(47,212,193,.95)');
      return r;
    }

    if (o.mode === 'loupe' && o.loupe) {
      const { x, y, r: rr } = o.loupe;
      ctx.save();                                  // photograph, with a hole in it
      ctx.beginPath();
      ctx.rect(r.x, r.y, r.w, r.h);
      ctx.arc(x, y, rr, 0, 7);
      ctx.clip('evenodd');
      drawPhoto();
      ctx.restore();

      ctx.strokeStyle = 'rgba(47,212,193,.95)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, rr, 0, 7); ctx.stroke();
      ctx.strokeStyle = 'rgba(7,9,9,.5)'; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(x, y, rr + 3, 0, 7); ctx.stroke();
      ctx.fillStyle = 'rgba(47,212,193,.95)';
      ctx.font = '500 10px "Spline Sans Mono", monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      ctx.fillText('RENDER', x, y - rr - 9);
      label('PHOTOGRAPH', r.x + 12, 'left', 'rgba(230,236,235,.9)');
    }
    return r;
  }
}
