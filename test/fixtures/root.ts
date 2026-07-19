import { pathToFileURL } from 'node:url';

// Specs resolve their golden corpus through here instead of hardcoding
// ./fixtures/. The fast tier (`npm test`) reads the committed corpus next to
// this file. The oracle tier sets SEGMORPH_FIXTURES_DIR to test/generated,
// where the project's global setup has just written goldens computed live by
// the pinned oracles, so the same specs double as the anti-staleness check.
const override = process.env.SEGMORPH_FIXTURES_DIR;

export const fixturesRoot = override
  ? pathToFileURL(`${override}/`)
  : new URL('./', import.meta.url);

export const fixtureUrl = (path: string) => new URL(path, fixturesRoot);

export const isLiveOracleRun = override !== undefined;
