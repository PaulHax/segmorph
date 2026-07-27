// Single entry point for running the oracle generators. It syncs the Python
// oracle environment with uv, then drives each algorithm's oracle(s): the
// per-filter Python generators (vtk, itk) plus the composed Node oracles
// (@icr/polyseg-wasm for A/D, @itk-wasm/* for H). Pass `--algo <letter>` to
// generate a single algorithm.
//
// Output goes to `--out <dir>` (default `test/fixtures`, the committed corpus).
// The oracle test tier (`npm run test:oracle`) invokes this with
// `--out test/generated` so specs compare against goldens computed live by the
// pinned oracles instead of committed files that can go stale. Every generator
// resolves both its output root and any golden it reads back as input through
// SEGMORPH_FIXTURES_DIR, which this script exports, so a live run is
// self-consistent and never mixes the committed corpus into fresh goldens.
//
// Requires uv (https://docs.astral.sh/uv/). The Python dependencies live in
// oracles/py/pyproject.toml and are locked in oracles/py/uv.lock.

import { mkdir } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const oracleDirectory = fileURLToPath(new URL('./py', import.meta.url));
const nodeDirectory = fileURLToPath(new URL('./node', import.meta.url));

const outFlag = process.argv.indexOf('--out');
const fixturesDirectory = resolve(
  root,
  outFlag === -1 ? 'test/fixtures' : process.argv[outFlag + 1],
);

// Per-algorithm oracles. `py` lists the Python generators (run in order, so a
// second oracle can cross-check the first); `node` is a composed Node oracle
// (spawned after the Python goldens exist so it can cross-check them). An
// algorithm may have either or both.
const oracles: Record<string, { py?: string[]; node?: string[] }> = {
  A: { py: ['gen_surface.py'], node: [`${nodeDirectory}/polyseg.ts`, 'A'] },
  B: { py: ['gen_smooth.py'], node: [`${nodeDirectory}/vtkjs.ts`, 'B'] },
  C: { py: ['gen_decimate.py'] },
  D: { py: ['gen_voxelize.py'], node: [`${nodeDirectory}/polyseg.ts`, 'D'] },
  E: { py: ['gen_resample.py', 'gen_resample_itk.py'] },
  F: { py: ['gen_contour.py'], node: [`${nodeDirectory}/vtkjs.ts`, 'F'] },
  G: { py: ['gen_rasterize.py'] },
  H: { node: [`${nodeDirectory}/fillbetween.ts`] },
  I: { py: ['gen_fractional.py'] },
  J: { py: ['gen_surfacenets.py'] },
  K: { node: [`${nodeDirectory}/polyseg.ts`, 'K'] },
  // L is the clinical regression: a real chest CT converted by the rule Slicer
  // ships. Oracle-tier only, and the only oracle that needs network access (it
  // downloads the dataset once, then reads oracles/.cache).
  L: { node: [`${nodeDirectory}/clinical.ts`] },
  // P is the property-based sweep: seeded random blobs rather than hand-chosen
  // shapes. Oracle-tier only, so it is never written into the committed corpus.
  P: { py: ['gen_property.py'] },
};

// L and P belong to the oracle tier only: their corpora are regenerated for
// every live run and never committed (see .gitignore). `npm run fixtures`
// maintains the committed corpus, so a bulk run that targets it skips them --
// otherwise regenerating fixtures would download a clinical dataset and write
// megabytes of gitignored goldens for no reason. Naming one with --algo still
// runs it wherever it is pointed.
const liveOnlyAlgorithms = new Set(['L', 'P']);
const targetsCommittedCorpus = outFlag === -1;

const algorithmFlag = process.argv.indexOf('--algo');
const requestedAlgorithm = algorithmFlag === -1 ? undefined : process.argv[algorithmFlag + 1];
if (requestedAlgorithm && !(requestedAlgorithm in oracles)) {
  throw new Error(
    `Unsupported fixture algorithm: ${requestedAlgorithm}. Expected one of ${Object.keys(oracles).join(', ')}`,
  );
}
const algorithms = requestedAlgorithm
  ? [requestedAlgorithm]
  : Object.keys(oracles).filter(
      (algorithm) => !(targetsCommittedCorpus && liveOnlyAlgorithms.has(algorithm)),
    );

function run(command: string, args: string[]) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: 'inherit',
    env: { ...process.env, SEGMORPH_FIXTURES_DIR: fixturesDirectory },
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function runPython(script: string) {
  run('uv', ['run', '--project', oracleDirectory, 'python', `${oracleDirectory}/${script}`, root]);
}

await mkdir(fixturesDirectory, { recursive: true });
run('uv', ['sync', '--project', oracleDirectory]);

for (const algorithm of algorithms) {
  const oracle = oracles[algorithm];
  for (const script of oracle.py ?? []) runPython(script);
  if (oracle.node) run(process.execPath, oracle.node);
}
