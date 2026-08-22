// sw.js — the minimal service worker that makes the app installable.
//
// Deliberately NO caching: the app versions itself through HTTP cache
// headers (index/code are no-cache on the CDN), photos never leave the tab,
// and a stale trainer would be worse than a network round-trip. The empty
// fetch listener leaves every request on the browser's default network path.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(clients.claim()));
self.addEventListener('fetch', () => { /* default network handling */ });
