import { fileURLToPath } from 'node:url';

import { configDefaults, defineConfig } from 'vitest/config';

// The oracle specs run in BOTH tiers, against two different corpora.
//
// `npm test` (unit) reads the committed goldens in test/fixtures, so the fast
// tier needs neither Python nor WASM and still gets the full differential
// suite.
//
// `npm run test:oracle` (oracle) regenerates every algorithm's goldens live
// with the pinned oracle environment into test/generated and points the same
// specs at them via SEGMORPH_FIXTURES_DIR. That is the anti-staleness check: a
// VTK, PolySeg, or ITK bump that changes an oracle's output fails here even
// though the committed corpus still agrees with itself. It needs uv (and, per
// algorithm, the WASM oracles), so it stays its own project and its own CI job.
const generatedDir = fileURLToPath(new URL('./test/generated', import.meta.url));

// These are oracle-tier ONLY, because their corpora are deliberately never
// committed: the property sweep is drawn from a seeded random distribution and
// regenerated live, and the clinical regression's input is a downloaded
// dataset. There is nothing for the fast tier to read.
const liveOnlySpecs = ['test/property-sweep.spec.ts', 'test/clinical-regression.spec.ts'];

const oracleSpecs = ['test/oracle-*.spec.ts', 'test/harness.spec.ts', ...liveOnlySpecs];

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          exclude: [...configDefaults.exclude, ...liveOnlySpecs],
        },
      },
      {
        test: {
          name: 'oracle',
          include: oracleSpecs,
          globalSetup: ['test/oracleSetup.ts'],
          env: { SEGMORPH_FIXTURES_DIR: generatedDir },
        },
      },
    ],
  },
});
