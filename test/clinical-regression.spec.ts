import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { labelmapToSurface } from '../src/convert/labelmapToSurface.js';
import { meshSmooth } from '../src/convert/meshSmooth.js';
import { createOrientedImage } from '../src/image/orientedImage.js';
import { readNrrd } from '../src/io/nrrd.js';
import { vertices } from './diff/mesh.js';
import type { Point } from './diff/mesh.js';
import {
  buildTriangleGrid,
  nearestTriangleDistance,
  sampledSurfaceDistances,
} from './diff/spatial.js';
import {
  hasConsistentOutwardOrientation,
  isManifold,
  isWatertight,
} from './diff/structure.js';
import { findFixtureEntries, readFixtureManifest, readMeshJson } from './fixtures/loaders.js';
import { fixtureUrl } from './fixtures/root.js';

// Clinical regression: our binary-labelmap-to-closed-surface chain against the
// one 3D Slicer actually ships (@icr/polyseg-wasm, vtkSegmentationCore's real
// conversion rule), run over a real chest CT rather than a constructed shape.
//
// The case is dense bone thresholded out of LIDC2 (256x256x133 at
// 1.40625 x 1.40625 x 2.5 mm): ribs, vertebrae, and scapulae, so dozens of
// disconnected components and thin curved shells, at ~412k triangles -- two
// orders of magnitude past anything else in the suite. See
// oracles/node/clinical.ts.
//
// KNOWN, EXPLAINED DIVERGENCE AT THE VOLUME BOUNDARY
// Our port follows VTK: it pads the volume with a one-sample background border,
// so foreground running into a volume face closes half a voxel outside it. That
// is what makes our output watertight, and the property sweep confirms we
// reproduce VTK's padded extraction exactly, edge-touching cases included.
// PolySeg does not pad, so it leaves those faces open -- its mesh is not
// watertight, which also makes its enclosed volume meaningless (the divergence
// theorem needs a closed surface), so this spec deliberately does not compare
// volumes.
//
// Rather than widening a tolerance until that difference fits, the tests below
// measure the interior tightly and assert the divergence is CONFINED to the
// boundary. An interior regression then cannot hide inside a boundary-sized
// tolerance.
//
// Calibration (polyseg-wasm 0.4.0, measured 2026-07-18):
//   whole mesh, all vertices:       Hausdorff 6.65 mm, mean 0.111 mm
//   excluding a 1-voxel boundary shell: Hausdorff 0.964 mm, mean 0.101 mm
//   excluding a 2-voxel boundary shell: Hausdorff 0.635 mm, mean 0.100 mm
//   vertices disagreeing by > 1 mm: 863 of 212042 (0.407%), every one of them
//   within one voxel of a volume face.
const MAX_INTERIOR_HAUSDORFF = 1.2; // measured 0.964
const MAX_INTERIOR_MEAN = 0.13; // measured 0.101
const FAR_THRESHOLD = 1.2; // a vertex farther than this must be a boundary vertex

const caseUrl = fixtureUrl('L/lidc2-bone/');

type ClinicalParams = {
  dims: [number, number, number];
  spacing: [number, number, number];
  labelValue: number;
  passBand: number;
  iterations: number;
  voxelCount: number;
  goldenPointCount: number;
  goldenTriangleCount: number;
};

const params: ClinicalParams = JSON.parse(
  await readFile(new URL('params.json', caseUrl), 'utf8'),
);

// World-space extent of the sample points, so "on the volume boundary" is a
// geometric statement rather than an index one.
const maxWorld = [0, 1, 2].map((axis) => (params.dims[axis] - 1) * params.spacing[axis]);
const isInterior = (margin: number) => (point: Point) => [0, 1, 2].every((axis) => (
  point[axis] > margin * params.spacing[axis]
  && point[axis] < maxWorld[axis] - margin * params.spacing[axis]
));

async function loadInputs() {
  const [image, golden] = await Promise.all([
    readFile(new URL('input.nrrd', caseUrl)).then((bytes) => createOrientedImage(readNrrd(bytes))),
    readFile(new URL('golden.polyseg.mesh.json', caseUrl), 'utf8').then(readMeshJson),
  ]);
  const ours = meshSmooth(
    labelmapToSurface(image, { labelValue: params.labelValue }),
    { passBand: params.passBand, numberOfIterations: params.iterations },
  );
  return { image, golden, ours };
}

describe('clinical regression: LIDC2 bone vs Slicer PolySeg', () => {
  it('regressed against the pinned clinical dataset', async () => {
    const manifest = readFixtureManifest(await readFile(fixtureUrl('manifest.json'), 'utf8'));
    const entries = findFixtureEntries(manifest, 'L', 'lidc2-bone');

    expect(entries.map((entry) => entry.oracle.name)).toEqual(['icr-polyseg-wasm']);
    // Guard the fixture: a threshold that selected nothing, or a truncated
    // download, would make every comparison below vacuously pass.
    expect(params.voxelCount).toBe(101_742);
    expect(params.goldenTriangleCount).toBeGreaterThan(400_000);
  });

  it('produces a closed, outward-oriented surface at clinical scale', async () => {
    const { ours } = await loadInputs();

    // Our padding guarantee, on real anatomy with components running into the
    // volume faces. PolySeg's own output is NOT watertight on this case, which
    // is exactly the property the padding exists to provide.
    expect(isWatertight(ours)).toBe(true);
    expect(isManifold(ours)).toBe(true);
    expect(hasConsistentOutwardOrientation(ours)).toBe(true);
  }, 120_000);

  it('matches Slicer away from the volume boundary', async () => {
    const { golden, ours } = await loadInputs();

    const distances = sampledSurfaceDistances(ours, golden, {
      maxSamples: 20_000,
      accept: isInterior(1),
    });

    const detail = JSON.stringify(distances);
    expect(distances.sampleCount).toBeGreaterThan(20_000);
    expect(distances.hausdorff, detail).toBeLessThan(MAX_INTERIOR_HAUSDORFF);
    expect(distances.mean, detail).toBeLessThan(MAX_INTERIOR_MEAN);
  }, 120_000);

  // The point of this test is that the boundary divergence is BOUNDED, not just
  // tolerated: every vertex that disagrees materially must be explainable as
  // the padding difference. If a future change moved interior geometry, the
  // disagreement would stop being boundary-local and this fails -- which the
  // whole-mesh Hausdorff, sitting at 6.65 mm to accommodate the boundary,
  // could never catch.
  it('confines its disagreement with Slicer to the volume boundary', async () => {
    const { golden, ours } = await loadInputs();

    const grid = buildTriangleGrid(golden);
    const nearBoundary = isInterior(1);
    const strays = vertices(ours)
      .filter((point) => nearestTriangleDistance(grid, point) > FAR_THRESHOLD)
      .filter((point) => nearBoundary(point));

    expect(strays.slice(0, 5)).toEqual([]);
    expect(strays).toHaveLength(0);
  }, 300_000);
});
