import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { surfaceNets } from '../src/convert/surfaceNets.js';
import { triangleCount, vertexCount, type Mesh } from '../src/geometry/mesh.js';
import { createOrientedImage } from '../src/image/orientedImage.js';
import { readNrrd } from '../src/io/nrrd.js';
import {
  boundingBox,
  enclosedVolume,
  meanSurfaceDistance,
  symmetricHausdorffDistance,
} from './diff/mesh.js';
import {
  hasConsistentOutwardOrientation,
  isManifold,
  isWatertight,
} from './diff/structure.js';
import { readMeshJson } from './fixtures/loaders.js';

const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const;

async function loadCase(name: string) {
  const directory = new URL(`./fixtures/J/${name}/`, import.meta.url);
  const nrrd = readNrrd(new Uint8Array(await readFile(new URL('input.nrrd', directory))));
  const image = createOrientedImage(nrrd);
  const params = JSON.parse(await readFile(new URL('params.json', directory), 'utf8')) as {
    labelValue: number;
  };
  const golden = readMeshJson(await readFile(new URL('golden.mesh.json', directory), 'utf8'));
  const unsmoothedGolden = readMeshJson(
    await readFile(new URL('golden.unsmoothed.mesh.json', directory), 'utf8'),
  );
  return { image, labelValue: params.labelValue, golden, unsmoothedGolden };
}

// Symmetric max distance from each vertex of one mesh to the nearest vertex
// of the other. Both meshes come from the same algorithm on the same grid, so
// vertices correspond one-to-one (modulo ordering) and this bounds the
// pointwise deviation far more tightly than a surface distance.
function maxNearestVertexDistance(a: Mesh, b: Mesh) {
  const directed = (source: Mesh, target: Mesh) => {
    let max = 0;
    for (let i = 0; i < source.points.length; i += 3) {
      let nearest = Infinity;
      for (let j = 0; j < target.points.length; j += 3) {
        const dx = source.points[i] - target.points[j];
        const dy = source.points[i + 1] - target.points[j + 1];
        const dz = source.points[i + 2] - target.points[j + 2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < nearest) nearest = d2;
      }
      max = Math.max(max, nearest);
    }
    return Math.sqrt(max);
  };
  return Math.max(directed(a, b), directed(b, a));
}

const cases = [
  'sphere',
  'boundary-blob',
  'anisotropic',
  'oblique',
  'multilabel-label1',
  'multilabel-label2',
] as const;

const outwardCases = new Set(['sphere', 'boundary-blob', 'anisotropic', 'oblique', 'multilabel-label1']);

describe('surfaceNets vs python vtk 9.6.2 vtkSurfaceNets3D', () => {
  for (const name of cases) {
    it(`matches the unsmoothed golden for ${name}`, async () => {
      const { image, labelValue, unsmoothedGolden } = await loadCase(name);
      const mesh = surfaceNets(image, { labelValue, smoothing: false });

      expect(vertexCount(mesh)).toBe(vertexCount(unsmoothedGolden));
      expect(triangleCount(mesh)).toBe(triangleCount(unsmoothedGolden));
      // Unsmoothed points are cell centers transformed to world space; the
      // only legitimate deviation is float32 transform rounding (~1 ulp).
      // Measured max nearest-vertex distance across all six cases: 1.91e-6
      // (oblique); threshold 1e-4 gives ~50x headroom while staying far
      // below half a voxel.
      expect(maxNearestVertexDistance(mesh, unsmoothedGolden)).toBeLessThanOrEqual(1e-4);
      expect(Math.abs(enclosedVolume(mesh) / enclosedVolume(unsmoothedGolden)) - 1)
        .toBeLessThanOrEqual(1e-6);
    });

    it(`matches the smoothed golden for ${name}`, async () => {
      const { image, labelValue, golden } = await loadCase(name);
      const mesh = surfaceNets(image, { labelValue });

      expect(vertexCount(mesh)).toBe(vertexCount(golden));
      expect(triangleCount(mesh)).toBe(triangleCount(golden));
      // Smoothing round-trips positions through float32 every iteration in
      // both implementations; drift stays at rounding scale. Measured max
      // nearest-vertex distance across all six cases: 4.66e-6 (oblique);
      // threshold 1e-3 gives large headroom yet stays well below spacing.
      expect(maxNearestVertexDistance(mesh, golden)).toBeLessThanOrEqual(1e-3);
      expect(Math.abs(enclosedVolume(mesh) / enclosedVolume(golden)) - 1)
        .toBeLessThanOrEqual(1e-4);

      if (outwardCases.has(name)) {
        expect(isWatertight(mesh)).toBe(true);
        expect(isManifold(mesh)).toBe(true);
        expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
      }
    });
  }

  it('stays within surface-distance tolerance of the smoothed sphere golden', async () => {
    const { image, labelValue, golden } = await loadCase('sphere');
    const mesh = surfaceNets(image, { labelValue });
    // Measured on this port: Hausdorff 4.2e-6, mean 6.7e-8 (float32 rounding
    // scale). Thresholds keep two orders of magnitude of headroom.
    expect(symmetricHausdorffDistance(mesh, golden)).toBeLessThanOrEqual(1e-3);
    expect(meanSurfaceDistance(mesh, golden)).toBeLessThanOrEqual(1e-4);
  }, 20_000);
});

describe('surfaceNets unit behavior', () => {
  it('returns an empty mesh when the label is absent', () => {
    const image = createOrientedImage({
      data: new Uint8Array(8).fill(1),
      dims: [2, 2, 2],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    });
    const mesh = surfaceNets(image, { labelValue: 5 });
    expect(vertexCount(mesh)).toBe(0);
    expect(triangleCount(mesh)).toBe(0);
  });

  it('closes an all-foreground volume via internal background padding', () => {
    // Raw vtkSurfaceNets3D returns an empty mesh here; polymorph pads one
    // background voxel per side (like the A-queue pipeline) so
    // boundary-touching foreground produces a closed surface.
    const image = createOrientedImage({
      data: new Uint8Array(8).fill(1),
      dims: [2, 2, 2],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    });
    const mesh = surfaceNets(image, { labelValue: 1, smoothing: false });
    expect(isWatertight(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
    expect(enclosedVolume(mesh)).toBeCloseTo(8);
    const bounds = boundingBox(mesh);
    expect(bounds.min).toEqual([-0.5, -0.5, -0.5]);
    expect(bounds.max).toEqual([1.5, 1.5, 1.5]);
  });

  it('extracts an oriented cube around one voxel without smoothing', () => {
    const diagonal = Math.SQRT1_2;
    const image = createOrientedImage({
      data: new Uint8Array([1]),
      dims: [1, 1, 1],
      spacing: [2, 4, 6],
      origin: [10, 20, 30],
      direction: [[diagonal, -diagonal, 0], [diagonal, diagonal, 0], [0, 0, 1]],
    });
    const mesh = surfaceNets(image, { labelValue: 1, smoothing: false });

    expect(vertexCount(mesh)).toBe(8);
    expect(triangleCount(mesh)).toBe(12);
    expect(enclosedVolume(mesh)).toBeCloseTo(2 * 4 * 6);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
    const bounds = boundingBox(mesh);
    // Cube corners sit at scaled offsets (+/-1, +/-2, +/-3) rotated 45
    // degrees about z: x and y extremes are (1 + 2) / sqrt(2).
    expect(bounds.min[0]).toBeCloseTo(10 - 3 * diagonal);
    expect(bounds.max[1]).toBeCloseTo(20 + 3 * diagonal);
    expect(bounds.min[2]).toBeCloseTo(27);
    expect(bounds.max[2]).toBeCloseTo(33);
  });

  it('keeps smoothing inside the constraint sphere and preserves topology', () => {
    const data = new Uint8Array(5 * 5 * 5);
    data[2 + 5 * (2 + 5 * 2)] = 1;
    const image = createOrientedImage({
      data,
      dims: [5, 5, 5],
      spacing: [1, 2, 3],
      origin: [0, 0, 0],
      direction: identity,
    });
    const unsmoothed = surfaceNets(image, { labelValue: 1, smoothing: false });
    const smoothed = surfaceNets(image, { labelValue: 1 });

    expect(vertexCount(smoothed)).toBe(vertexCount(unsmoothed));
    expect(isWatertight(smoothed)).toBe(true);
    expect(hasConsistentOutwardOrientation(smoothed)).toBe(true);
    const constraint = Math.hypot(1, 2, 3) + 1e-6;
    for (let index = 0; index < smoothed.points.length; index += 3) {
      const dx = smoothed.points[index] - unsmoothed.points[index];
      const dy = smoothed.points[index + 1] - unsmoothed.points[index + 1];
      const dz = smoothed.points[index + 2] - unsmoothed.points[index + 2];
      expect(Math.hypot(dx, dy, dz)).toBeLessThanOrEqual(constraint);
    }
  });

  it('treats zero smoothing iterations like smoothing disabled', async () => {
    const { image, labelValue } = await loadCase('boundary-blob');
    const unsmoothed = surfaceNets(image, { labelValue, smoothing: false });
    const zeroIterations = surfaceNets(image, { labelValue, smoothingIterations: 0 });
    expect(zeroIterations.points).toEqual(unsmoothed.points);
    expect(zeroIterations.polys).toEqual(unsmoothed.polys);
  });
});
