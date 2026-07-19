// Global setup for the oracle test project: run every algorithm's oracle
// generators into test/generated so the specs compare against goldens computed
// live by the pinned oracle environment instead of the committed corpus, which
// agrees with itself even after an upstream oracle changes behavior.
//
// Requires uv; the first run downloads the Python dependencies (vtk is large),
// later runs reuse the venv. The generators are run with no --algo filter, so
// they execute in the dependency order declared in oracles/generate.ts (later
// algorithms read algorithm A's goldens as their inputs).

import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));

export default function setup() {
  const result = spawnSync(
    process.execPath,
    ['oracles/generate.ts', '--out', 'test/generated'],
    { cwd: root, stdio: 'inherit' },
  );
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`oracle golden generation failed with status ${result.status}`);
  }
}
