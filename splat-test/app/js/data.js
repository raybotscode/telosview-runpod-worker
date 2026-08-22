// data.js — the preset sets and the fixed copy. Every runtime number in the
// app is measured live; what lives here is only what a card needs to SAY
// before anything has run.

export const REPO = 'https://github.com/arrival-space/splat.js';
export const DATA = '../data/';

export const PRESETS = [
  {
    id: 'truck', name: 'Truck',
    kind: 'Standard test set',
    origin: 'The Truck sequence from the Tanks & Temples reconstruction benchmark.',
    links: [
      { label: 'Tanks & Temples', url: 'https://www.tanksandtemples.org/' },
      { label: 'image set', url: 'https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/' },
    ],
    // 000061 is the sequence's easter egg — the photographer in frame. Funny,
    // but a person that exists in ONE image only feeds the loss pure error.
    dir: 'truck', pattern: '{i:6}.jpg', start: 1, count: 100, maxCount: 250, skip: [61],
    blurb: 'A parked truck, circled once on foot. Even spacing, constant light, ' +
           'plenty of sideways movement — close to a textbook capture.',
    approx: '~6 min',
  },
  {
    id: 'camping', name: 'Camping',
    kind: 'Handheld phone video',
    origin: '113 frames pulled from a phone video. Not a benchmark — an ordinary capture, with everything that goes wrong in one.',
    links: [],
    dir: 'camping', pattern: 'frame_{i:5}.jpg', start: 1, count: 113,
    blurb: 'Every frame of a walking video. Neighbouring frames barely differ, so the ' +
           'solver has to chain landmarks across dozens of them.',
    approx: '~8 min',
  },
  {
    id: 'playroom', name: 'Playroom',
    kind: 'Indoor photo set',
    origin: 'A children\'s playroom in 225 photos — soft painted walls and ceilings make it a genuinely hard indoor solve.',
    links: [],
    // keep ALL frames: trimming the unplaced ones cascades more failures
    // (207 -> 197 -> 196 -> 192 measured) — even unplaceable frames carry
    // tracks that chain their neighbours together
    dir: 'playroom', list: 'files.json', count: 100, maxCount: 225,
    blurb: 'Indoor, warm light, and a lot of featureless wall. A few blank-wall shots ' +
           'sit out of every solve — they still help chain the rest together.',
    approx: '~12 min',
  },
  {
    id: 'train', name: 'Train',
    kind: 'Standard test set',
    origin: 'The Train sequence from the Tanks & Temples reconstruction benchmark.',
    links: [
      { label: 'Tanks & Temples', url: 'https://www.tanksandtemples.org/' },
      { label: 'image set', url: 'https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/' },
    ],
    dir: 'train', pattern: '{i:5}.jpg', start: 1, count: 84, maxCount: 301,
    blurb: 'More photos of a harder subject: dark paint, repeating windows, and a seam ' +
           'where the shoot was interrupted and picked up again.',
    approx: '~10 min',
  },
  {
    id: 'bicycle', name: 'Bicycle',
    kind: 'Standard test set',
    origin: 'The Bicycle scene from the Mip-NeRF 360 benchmark — a garden orbit around a bench.',
    links: [
      { label: 'Mip-NeRF 360', url: 'https://jonbarron.info/mipnerf360/' },
    ],
    dir: 'bicycle', list: 'files.json', count: 194,
    blurb: 'Spokes, grass and leaves — the thin geometry that breaks most ' +
           'reconstructions. A couple of photos sit out; the rest chain cleanly.',
    approx: '~6 min',
  },
  {
    id: 'synthetic', name: 'Synthetic Corner',
    kind: 'Rendered test set',
    origin: '12 rendered views of a small scene. The clean case, for comparison.',
    links: [],
    dir: 'synthetic', pattern: 'synthetic_{i:2}.png', start: 0, count: 12,
    blurb: 'Nothing is noisy, nothing is blurry. The cameras are solved from scratch ' +
           'like everywhere else — rendered input just makes every error visible.',
    approx: '~2 min',
  },
];

/** A set made of the visitor's own photographs — the primary path. */
export function ownSet(files, urls) {
  const n = files.length;
  return {
    id: '__own', name: 'Your photos',
    kind: 'Your own photos',
    origin: `${n} photos, read straight off your machine. Nothing is uploaded — they are ` +
            'decoded in this tab and go no further.',
    links: [],
    files, urls, count: n,
    blurb: '',
    approx: n <= 20 ? '~3 min' : n <= 60 ? '~6 min' : '~10 min',
  };
}

export const HOLD_HELP =
  'One photograph is held back from training and only ever scored, never learned from. ' +
  'Its score is the honest one — it says whether the model understood the scene or ' +
  'memorised the pictures.';
