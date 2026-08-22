// img.js — one decode per URL, shared across the page.

const pending = new Map();
const done = new Map();
const key = (url, w) => (w ? `${url}@${w}` : url);

export function bmp(url, w) {
  const k = key(url, w);
  if (pending.has(k)) return pending.get(k);
  const p = fetch(url).then((r) => r.blob())
    .then((b) => createImageBitmap(b, w ? { resizeWidth: w, resizeQuality: 'medium' } : undefined))
    .then((b) => { done.set(k, b); return b; })
    .catch(() => null);
  pending.set(k, p);
  return p;
}

/** the decoded bitmap if it is already here, otherwise null and a load starts */
export function readyBmp(url, w) {
  const k = key(url, w);
  if (done.has(k)) return done.get(k);
  bmp(url, w);
  return null;
}
