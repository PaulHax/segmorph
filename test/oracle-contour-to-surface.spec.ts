import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { contourToSurface } from '../src/convert/contourToSurface.js';
import { contourToSurfaceCases } from './contourToSurfaceCases.js';
import {
  enclosedVolume,
  meanSurfaceDistance,
  symmetricHausdorffDistance,
  type Mesh,
} from './diff/mesh.js';
import { findFixtureEntries, readFixtureManifest, readMeshJson } from './fixtures/loaders.js';

const fixturesUrl = new URL('./fixtures/', import.meta.url);

// Tolerance calibration, measured against the committed
// golden.polyseg.mesh.json goldens (@icr/polyseg-wasm 0.4.0, which compiles
// the same MIT SlicerRT rule this port translates):
//
//   case        hausdorff   mean       |volumeRatio-1|
//   cylinder    4.03e-1     1.26e-2    6.2e-3
//   sphere      1.82e-2     6.80e-5    1.0e-5
//   cone        2.99e-1     6.24e-3    4.5e-3
//   branching   2.77e-1     8.36e-3    8.7e-4
//   keyhole     1.48e-1     1.69e-3    1.9e-4
//   tilted      5.72e-1     2.27e-2    1.4e-2
//
// The port matches the oracle's between-plane triangulation exactly (same
// points, same dynamic program); the differences live in the smooth end-cap
// pipeline, where marching-squares stripping order, priority-queue tie
// breaking in the decimator, and ear-cut order are implementation defined.
// Those differences move cap vertices along the cap contour, never off the
// eroded cap outline, so they are bounded by the cap raster spacing (about
// one pixel of the 28-pixel cap grid). Thresholds sit at roughly twice the
// measured spread.
const tolerances: Record<string, { hausdorff: number; mean: number; volume: number }> = {
  cylinder: { hausdorff: 0.8, mean: 0.03, volume: 0.015 },
  sphere: { hausdorff: 0.1, mean: 0.001, volume: 0.001 },
  cone: { hausdorff: 0.6, mean: 0.015, volume: 0.01 },
  branching: { hausdorff: 0.6, mean: 0.02, volume: 0.005 },
  keyhole: { hausdorff: 0.4, mean: 0.005, volume: 0.002 },
  tilted: { hausdorff: 1.2, mean: 0.05, volume: 0.03 },
};

// Vertex and triangle counts stay within a few cap-contour points of the
// oracle's (measured worst case: 6 of 362 triangles on tilted).
const countBand = 0.05;

async function loadGolden(caseName: string): Promise<Mesh> {
  return readMeshJson(
    await readFile(new URL(`K/${caseName}/golden.polyseg.mesh.json`, fixturesUrl), 'utf8'),
  );
}

const manifest = readFixtureManifest(
  await readFile(new URL('manifest.json', fixturesUrl), 'utf8'),
);

describe('polyseg-wasm contour-to-surface oracle', () => {
  for (const caseName of Object.keys(contourToSurfaceCases)) {
    it(`records a manifest entry for ${caseName}`, () => {
      const entries = findFixtureEntries(manifest, 'K', caseName);
      expect(entries.map((entry) => entry.oracle.name)).toContain('icr-polyseg-wasm');
    });

    it(`matches the composed PolySeg golden on ${caseName}`, async () => {
      const golden = await loadGolden(caseName);
      const ours = contourToSurface(contourToSurfaceCases[caseName].loops);

      const hausdorff = symmetricHausdorffDistance(ours, golden);
      const mean = meanSurfaceDistance(ours, golden);
      const volumeRatio = enclosedVolume(ours) / enclosedVolume(golden);
      const tolerance = tolerances[caseName];

      expect.soft(hausdorff, `hausdorff ${hausdorff}`).toBeLessThanOrEqual(tolerance.hausdorff);
      expect.soft(mean, `mean ${mean}`).toBeLessThanOrEqual(tolerance.mean);
      expect
        .soft(Math.abs(volumeRatio - 1), `volumeRatio ${volumeRatio}`)
        .toBeLessThanOrEqual(tolerance.volume);

      const vertexRatio = (ours.points.length / 3) / (golden.points.length / 3);
      const triangleRatio = (ours.polys.length / 4) / (golden.polys.length / 4);
      expect.soft(Math.abs(vertexRatio - 1), `vertexRatio ${vertexRatio}`)
        .toBeLessThanOrEqual(countBand);
      expect.soft(Math.abs(triangleRatio - 1), `triangleRatio ${triangleRatio}`)
        .toBeLessThanOrEqual(countBand);
    });
  }
});
