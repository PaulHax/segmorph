import { configDefaults, defineConfig } from 'vitest/config';

// Specs in the oracle tier compare our ports against goldens computed live by
// the canonical libraries: the project's global setup runs the oracle
// generators into the gitignored test/generated directory, so the goldens can
// never go stale against the pinned oracle versions. The tier needs uv (and,
// per algorithm, the WASM oracles), so it runs as its own project
// (`npm run test:oracle`) and its own CI job. Everything else is the fast
// tier: `npm test`, no Python or WASM required.
const oracleSpecs = ['test/oracle-surface.spec.ts', 'test/harness.spec.ts'];

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          exclude: [...configDefaults.exclude, ...oracleSpecs],
        },
      },
      {
        test: {
          name: 'oracle',
          include: oracleSpecs,
          globalSetup: ['test/oracleSetup.ts'],
        },
      },
    ],
  },
});
