// Global setup for the oracle test project: run the oracle generators for the
// migrated algorithms into test/generated so the specs compare against goldens
// computed live by the pinned oracle environment. Requires uv; the first run
// downloads the Python dependencies (vtk is large), later runs reuse the venv.

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));

export default function setup() {
  const result = spawnSync(
    process.execPath,
    ['oracles/generate.ts', '--algo', 'A', '--out', 'test/generated'],
    { cwd: root, stdio: 'inherit' },
  );
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`oracle golden generation failed with status ${result.status}`);
  }
}
