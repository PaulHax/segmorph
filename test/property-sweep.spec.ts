import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { labelmapToSurface } from '../src/convert/labelmapToSurface.js';
import { surfaceToLabelmap } from '../src/convert/surfaceToLabelmap.js';
import { createOrientedImage } from '../src/image/orientedImage.js';
import { readNrrd } from '../src/io/nrrd.js';
import { triangleCount, vertexCount } from '../src/geometry/mesh.js';
import { dice, mismatchCount } from './diff/image.js';
import { meanSurfaceDistance, symmetricHausdorffDistance } from './diff/mesh.js';
import {
  hasConsistentOutwardOrientation,
  isManifold,
  isWatertight,
} from './diff/structure.js';
import { findFixtureEntries, readFixtureManifest, readMeshJson } from './fixtures/loaders.js';
import { fixtureUrl } from './fixtures/root.js';

// Property-based sweep: 32 labelmaps drawn from a seeded random distribution
// (oracles/py/gen_property.py) rather than hand-picked shapes, each compared
// against vtkDiscreteFlyingEdges3D. The named cases pin the shapes someone
// thought to write down; this sweep covers the ones nobody did -- multi-
// component blobs, enclosed cavities, one-voxel slabs, foreground running into
// the volume wall, anisotropic and oblique geometry, and a distractor label
// packed against the target.
//
// Oracle-tier only: the corpus is regenerated live into test/generated and
// never committed, so the sweep can be wide without adding megabytes of
// goldens to the repository.
//
// Calibration across all 32 seeds (VTK 9.6.2, measured 2026-07-18):
//   - vertex and triangle counts: exact match on every seed
//   - worst symmetric Hausdorff: 1.07e-14 (float64 round-off)
//   - round-trip Dice: exactly 1.0 on every seed, 0 mismatched voxels
//   - watertight / manifold / outward-oriented: every seed
// VTK and our port emit the same points in a different ORDER, so the mesh
// comparison is geometric rather than element-wise -- deep equality holds on
// neither implementation's ordering and asserting it would be wrong.
const MAX_HAUSDORFF = 1e-9;

const caseName = (seed: number) => `blob-${String(seed).padStart(2, '0')}`;

type CaseParams = {
  labelValue: number;
  distractorLabel: number;
  geometry: string;
  voxelCount: number;
  pointCount: number;
  triangleCount: number;
};

async function loadCase(seed: number) {
  const directory = fixtureUrl(`P/${caseName(seed)}/`);
  const [params, image, golden] = await Promise.all([
    readFile(new URL('params.json', directory), 'utf8').then(JSON.parse) as Promise<CaseParams>,
    readFile(new URL('input.nrrd', directory)).then((bytes) => createOrientedImage(readNrrd(bytes))),
    readFile(new URL('golden.extract.mesh.json', directory), 'utf8').then(readMeshJson),
  ]);
  return { params, image, golden };
}

const manifest = readFixtureManifest(
  await readFile(fixtureUrl('manifest.json'), 'utf8'),
);
const seeds = manifest.fixtures
  .filter((entry) => entry.algorithm === 'P')
  .map((entry) => entry.seed)
  .sort((left, right) => left - right);

describe('labelmapToSurface property sweep vs VTK discrete flying edges', () => {
  it('generated a non-trivial sweep', () => {
    expect(seeds.length).toBeGreaterThanOrEqual(32);
    expect(findFixtureEntries(manifest, 'P', caseName(seeds[0])).map((entry) => entry.oracle.name))
      .toEqual(['python-vtk']);

    // A sweep of empty or identical blobs would pass everything below while
    // testing nothing, so assert the distribution actually varied.
    const geometries = new Set(manifest.fixtures
      .filter((entry) => entry.algorithm === 'P')
      .map((entry) => (entry.params as unknown as CaseParams).geometry));
    expect([...geometries].sort()).toEqual(['anisotropic', 'identity', 'oblique']);
  });

  it.each(seeds)('matches the VTK extraction for seed %i', async (seed) => {
    const { params, image, golden } = await loadCase(seed);

    // Guard the fixture itself: a degenerate blob would make the comparison
    // below vacuous.
    expect(params.voxelCount).toBeGreaterThan(0);
    expect(vertexCount(golden)).toBeGreaterThan(0);

    const mesh = labelmapToSurface(image, { labelValue: params.labelValue });

    expect(vertexCount(mesh)).toBe(vertexCount(golden));
    expect(triangleCount(mesh)).toBe(triangleCount(golden));
    expect(symmetricHausdorffDistance(mesh, golden)).toBeLessThan(MAX_HAUSDORFF);
    expect(meanSurfaceDistance(mesh, golden)).toBeLessThan(MAX_HAUSDORFF);
  }, 30_000);

  it.each(seeds)('extracts a closed, outward-oriented surface for seed %i', async (seed) => {
    const { params, image } = await loadCase(seed);

    const mesh = labelmapToSurface(image, { labelValue: params.labelValue });

    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
  }, 30_000);

  // The distractor label sits in the shell one voxel outside the target, so a
  // port that thresholds on "nonzero" instead of matching the requested label
  // produces a strictly larger mesh. Round-tripping back to a labelmap is what
  // makes that visible as an exact voxel count rather than a tolerance.
  it.each(seeds)('round-trips to the original voxel set for seed %i', async (seed) => {
    const { params, image } = await loadCase(seed);

    const mesh = labelmapToSurface(image, { labelValue: params.labelValue });
    const voxelized = surfaceToLabelmap(mesh, image, { labelValue: params.labelValue });

    const target = new Uint8Array(image.data.length);
    for (let index = 0; index < target.length; index += 1) {
      target[index] = image.data[index] === params.labelValue ? params.labelValue : 0;
    }

    const dims = image.dims as [number, number, number];
    expect(dice(target, voxelized.data, dims)).toBe(1);
    expect(mismatchCount(target, voxelized.data, dims)).toBe(0);
  }, 30_000);
});
