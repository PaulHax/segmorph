import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { meshDecimate } from '../src/convert/meshDecimate.js';
import { triangleCount, vertexCount } from '../src/geometry/mesh.js';
import {
  enclosedVolume,
  meanSurfaceDistance,
  symmetricHausdorffDistance,
  triangleIndices,
  type Mesh,
} from './diff/mesh.js';
import { hasConsistentOutwardOrientation, isManifold, isWatertight } from './diff/structure.js';
import { findFixtureEntries, readFixtureManifest, readMeshJson } from './fixtures/loaders.js';
import { fixtureUrl } from './fixtures/root.js';

const fixturesUrl = fixtureUrl('');

// Tolerances calibrated 2026-07-10 against the committed python-vtk 9.6.2
// goldens (vtkQuadricDecimation defaults) by measuring this port on every
// case. Distances are in mesh world units (bounding-box diagonals 31-41).
// Measured port-vs-golden spread per case:
//   a-sphere-r50   hausdorff 0.0000  mean 0.00000  |volumeRatio-1| 0.00085
//   a-sphere-r90   hausdorff 0.0000  mean 0.00000  |volumeRatio-1| 0.00931
//   ellipsoid-r50  hausdorff 0.0129  mean 0.00003  |volumeRatio-1| 0.00147
//   ellipsoid-r90  hausdorff 0.0059  mean 0.00009  |volumeRatio-1| 0.01333
//   torus-r50      hausdorff 0.1005  mean 0.00140  |volumeRatio-1| 0.00042
//   torus-r90      hausdorff 0.3185  mean 0.02420  |volumeRatio-1| 0.01899
// For scale, the legitimate golden-vs-input spread is much larger:
// hausdorff 0.0322/0.1649 (a-sphere), 0.0195/0.1090 (ellipsoid),
// 0.0638/0.3740 (torus). The a-sphere output is identical to VTK's; the
// ellipsoid and torus diverge only where the priority queue breaks exactly
// equal costs (both synthetic meshes are perfectly symmetric), so their
// thresholds sit just above the measured tie-break drift. Golden triangle
// counts matched this port exactly on all six cases.
const cases = [
  { name: 'a-sphere-r50', hausdorff: 0.001, mean: 0.0001, volumeBand: 0.005, euler: 2 },
  { name: 'a-sphere-r90', hausdorff: 0.001, mean: 0.0001, volumeBand: 0.02, euler: 2 },
  { name: 'ellipsoid-r50', hausdorff: 0.02, mean: 0.001, volumeBand: 0.005, euler: 2 },
  { name: 'ellipsoid-r90', hausdorff: 0.01, mean: 0.001, volumeBand: 0.02, euler: 2 },
  { name: 'torus-r50', hausdorff: 0.15, mean: 0.003, volumeBand: 0.005, euler: 0 },
  { name: 'torus-r90', hausdorff: 0.45, mean: 0.04, volumeBand: 0.03, euler: 0 },
] as const;

function eulerCharacteristic(mesh: Mesh) {
  const edges = new Set<string>();
  for (const [a, b, c] of triangleIndices(mesh)) {
    for (const [from, to] of [
      [a, b],
      [b, c],
      [c, a],
    ]) {
      edges.add(from < to ? `${from}:${to}` : `${to}:${from}`);
    }
  }
  return vertexCount(mesh) - edges.size + triangleCount(mesh);
}

async function loadCase(caseName: string) {
  const manifest = readFixtureManifest(
    await readFile(new URL('manifest.json', fixturesUrl), 'utf8'),
  );
  const entries = findFixtureEntries(manifest, 'C', caseName);
  expect(entries.map((entry) => entry.oracle.name)).toEqual(['python-vtk']);
  const [entry] = entries;

  const input = readMeshJson(
    await readFile(new URL(entry.params.input as string, fixturesUrl), 'utf8'),
  );
  const golden = readMeshJson(
    await readFile(new URL(entry.params.golden as string, fixturesUrl), 'utf8'),
  );
  return { entry, input, golden };
}

describe('quadric decimation vs python-vtk oracle', () => {
  it.each(cases)(
    'matches VTK on $name',
    async ({ name, hausdorff, mean, volumeBand, euler }) => {
      const { entry, input, golden } = await loadCase(name);
      const targetReduction = entry.params.targetReduction as number;
      const result = meshDecimate(input, { targetReduction });

      // Both stop at the first collapse that reaches the target reduction;
      // measured count difference was 0 on every case.
      expect(Math.abs(triangleCount(result) - triangleCount(golden))).toBeLessThanOrEqual(4);

      expect(symmetricHausdorffDistance(result, golden)).toBeLessThanOrEqual(hausdorff);
      expect(meanSurfaceDistance(result, golden)).toBeLessThanOrEqual(mean);

      const volumeRatio = enclosedVolume(result) / enclosedVolume(input);
      expect(volumeRatio).toBeGreaterThanOrEqual(1 - volumeBand);
      expect(volumeRatio).toBeLessThanOrEqual(1 + volumeBand);

      // Closed inputs must stay closed, manifold, outward oriented, and keep
      // their genus (Euler characteristic 2 for spheres, 0 for the torus).
      expect(isWatertight(result)).toBe(true);
      expect(isManifold(result)).toBe(true);
      expect(hasConsistentOutwardOrientation(result)).toBe(true);
      expect(eulerCharacteristic(result)).toBe(euler);
    },
    60000,
  );

  it('records every decimation case in the fixture manifest', async () => {
    const manifest = readFixtureManifest(
      await readFile(new URL('manifest.json', fixturesUrl), 'utf8'),
    );
    for (const { name } of cases) {
      expect(findFixtureEntries(manifest, 'C', name).map((entry) => entry.oracle.name)).toEqual([
        'python-vtk',
      ]);
    }
  });
});
