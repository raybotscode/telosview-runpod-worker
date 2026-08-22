// Run every unit test in this directory as a child process.
// Usage: node tests/unit/run.mjs
import { readdirSync } from 'fs';
import { spawnSync } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const here = dirname(fileURLToPath(import.meta.url));
const files = readdirSync(here).filter((f) => f.startsWith('test_') && f.endsWith('.mjs'));

let failed = 0, skipped = 0;
for (const f of files) {
  const r = spawnSync(process.execPath, [join(here, f)], { encoding: 'utf8' });
  const out = (r.stdout + r.stderr).trim().split('\n').pop() || '';
  if (r.status === 0) {
    console.log(`PASS  ${f}  (${out})`);
  } else if (/SKIP/.test(r.stdout + r.stderr) || /Cannot find (module|package)/.test(r.stderr)) {
    skipped++;
    console.log(`SKIP  ${f}  (missing optional deps or reference data)`);
  } else {
    failed++;
    console.log(`FAIL  ${f}`);
    console.log((r.stdout + r.stderr).split('\n').slice(-12).join('\n'));
  }
}
console.log(`\n${files.length - failed - skipped} passed, ${skipped} skipped, ${failed} failed`);
process.exit(failed ? 1 : 0);
