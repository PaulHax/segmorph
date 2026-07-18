import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import {
  fractionalLabelmapToSurface,
  surfaceToFractionalLabelmap,
} from '../src/convert/fractional.js';
import { readNrrd } from '../src/io/nrrd.js';
import {
  enclosedVolume,
  meanSurfaceDistance,
  symmetricHausdorffDistance,
} from './diff/mesh.js';
import {
  findFixtureEntries,
  readFixtureManifest,
  readMeshJson,
} from './fixtures/loaders.js';

const cases = ['isotropic', 'anisotropic', 'oblique'] as const;

const fixture = (caseName: string, name: string) => (
  new URL(`./fixtures/I/${caseName}/${name}`, import.meta.url)
);

async function loadCase(caseName: string) {
  const [meshJson, golden, surfaceJson] = await Promise.all([
    readFile(fixture(caseName, 'input.mesh.json'), 'utf8'),
    readNrrd(await readFile(fixture(caseName, 'golden.nrrd'))),
    readFile(fixture(caseName, 'golden.surface.mesh.json'), 'utf8'),
  ]);
  return {
    mesh: readMeshJson(meshJson),
    golden,
    goldenSurface: readMeshJson(surfaceJson),
  };
}

function occupancyDifference(actual: ArrayLike<number>, expected: ArrayLike<number>) {
  let maxAbs = 0;
  let mismatches = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const difference = Math.abs(actual[index] - expected[index]);
    if (difference > 0) mismatches += 1;
    maxAbs = Math.max(maxAbs, difference);
  }
  return { maxAbs, mismatches };
}

describe('fractional labelmap oracle', () => {
  it('records the generating VTK oracle in the fixture manifest', async () => {
    const manifest = readFixtureManifest(
      await readFile(new URL('./fixtures/manifest.json', import.meta.url), 'utf8'),
    );
    for (const caseName of cases) {
      expect(findFixtureEntries(manifest, 'I', caseName).map((entry) => entry.oracle.name))
        .toEqual(['python-vtk']);
    }
  });

  it.each(cases)('matches the 216-offset fractional occupancy on %s', async (caseName) => {
    const { mesh, golden } = await loadCase(caseName);
    const actual = surfaceToFractionalLabelmap(mesh, golden, {});
    expect(actual.data.length).toBe(golden.data.length);

    const { maxAbs, mismatches } = occupancyDifference(actual.data, golden.data);
    // Measured spread vs the VTK stencil oracle: maxAbs 1/216, 0, 1/216 on
    // isotropic / anisotropic / oblique. The bound admits only a single
    // sub-voxel sample of raster tie-break difference at the surface.
    expect(maxAbs).toBeLessThanOrEqual(1 / 216 + 1e-7);
    // Measured mismatching voxels: 1 of 8000, 0 of 5184, 1 of 4096.
    expect(mismatches).toBeLessThanOrEqual(Math.round(golden.data.length * 0.005));
  });

  it.each(cases)('extracts the thresholded surface on %s', async (caseName) => {
    const { golden, goldenSurface } = await loadCase(caseName);
    const surface = fractionalLabelmapToSurface(golden, {});

    // Both sides run linear-interpolated marching cubes on the same grid, so
    // only float32 rounding remains. Measured Hausdorff vs vtkFlyingEdges3D
    // at iso 0.5: 9.5e-7 / 1.9e-6 / 2.0e-6 world units (mean 1.3e-7 to
    // 1.5e-7). 1e-3 of a voxel keeps triangulation drift visible while
    // ignoring float noise.
    const spacingScale = Math.max(...golden.spacing);
    expect(symmetricHausdorffDistance(surface, goldenSurface))
      .toBeLessThan(1e-3 * spacingScale);
    expect(meanSurfaceDistance(surface, goldenSurface)).toBeLessThan(1e-3 * spacingScale);

    // Measured volume ratios actual/golden: 0.99999999 / 1.00000000 / 1.00000002.
    const goldenVolume = enclosedVolume(goldenSurface);
    expect(enclosedVolume(surface) / goldenVolume).toBeGreaterThan(0.999);
    expect(enclosedVolume(surface) / goldenVolume).toBeLessThan(1.001);
  });
});
