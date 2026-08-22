// store.js — keep the visitor's last own capture on the device.
//
// Camera shots and uploads live only in memory; a refresh used to erase a
// capture that might have taken minutes to walk. Every own set is therefore
// written into IndexedDB (blobs, origin-local, nothing leaves the machine)
// and offered back as a "Last capture" tile on the start card. One slot —
// each new capture replaces the previous one.

const DB = 'splatjs';
const STORE = 'captures';
const KEY = 'last';

function openDb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

/** rec: { kind: 'photos'|'video', created, files: [{ name, blob }] } */
export async function saveLastCapture(rec) {
  const d = await openDb();
  return new Promise((res, rej) => {
    const tx = d.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(rec, KEY);
    tx.oncomplete = () => { d.close(); res(); };
    tx.onerror = () => { d.close(); rej(tx.error); };
  });
}

export async function loadLastCapture() {
  try {
    const d = await openDb();
    return await new Promise((res, rej) => {
      const rq = d.transaction(STORE, 'readonly').objectStore(STORE).get(KEY);
      rq.onsuccess = () => { d.close(); res(rq.result || null); };
      rq.onerror = () => { d.close(); rej(rq.error); };
    });
  } catch { return null; }
}
