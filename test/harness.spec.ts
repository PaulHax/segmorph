import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import {
  enclosedVolume,
  meanSurfaceDistance,
  symmetricHausdorffDistance,
} from './diff/mesh.js';
import { readMeshJson } from './fixtures/loaders.js';
import { fixtureUrl } from './fixtures/root.js';

// In the oracle tier both goldens are regenerated live by the project's global
// setup, so this calibration measures the installed oracles rather than the
// committed corpus it also guards in the fast tier.
const sphereUrl = fixtureUrl('A/sphere/');
const maxHausdorff = 0.057; // Measured oracle spread: 0.056527524923095714.
const maxMeanDistance = 0.023; // Measured oracle spread: 0.022649542874481864.
const maxVolumeRatioDelta = 0.0063; // Measured oracle spread: 0.006221478697251426.

async function loadMesh(filename: string) {
  return readMeshJson(await readFile(new URL(filename, sphereUrl), 'utf8'));
}

describe('oracle calibration harness', () => {
  it('compares the Python VTK and PolySeg sphere goldens', async () => {
    const [python, polyseg] = await Promise.all([
      loadMesh('golden.mesh.json'),
      loadMesh('golden.polyseg.mesh.json'),
    ]);
    const hausdorff = symmetricHausdorffDistance(python, polyseg);
    const meanDistance = meanSurfaceDistance(python, polyseg);
    const volumeRatio = enclosedVolume(polyseg) / enclosedVolume(python);
    const spread = { hausdorff, meanDistance, volumeRatio };
    const failureMessage = JSON.stringify(spread);

    expect(hausdorff, failureMessage).toBeLessThanOrEqual(maxHausdorff);
    expect(meanDistance, failureMessage).toBeLessThanOrEqual(maxMeanDistance);
    expect(Math.abs(volumeRatio - 1), failureMessage)
      .toBeLessThanOrEqual(maxVolumeRatioDelta);
  }, 20_000);
});
