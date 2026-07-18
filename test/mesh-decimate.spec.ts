import { describe, expect, it } from 'vitest';

import { meshDecimate } from '../src/convert/meshDecimate.js';
import { triangleCount, validateMesh, vertexCount, type Mesh } from '../src/geometry/mesh.js';
import { boundingBox, symmetricHausdorffDistance } from './diff/mesh.js';
import {
  hasConsistentOutwardOrientation,
  isManifold,
  isWatertight,
} from './diff/structure.js';

function uvSphere(radius: number, latBands: number, lonBands: number): Mesh {
  const ringIndex = (ring: number, segment: number) => (
    1 + (ring - 1) * lonBands + (segment % lonBands)
  );
  const southPole = 1 + (latBands - 1) * lonBands;

  const points = new Float32Array((southPole + 1) * 3);
  points.set([0, 0, radius], 0);
  for (let ring = 1; ring < latBands; ring += 1) {
    const theta = (Math.PI * ring) / latBands;
    for (let segment = 0; segment < lonBands; segment += 1) {
      const phi = (2 * Math.PI * segment) / lonBands;
      points.set([
        radius * Math.sin(theta) * Math.cos(phi),
        radius * Math.sin(theta) * Math.sin(phi),
        radius * Math.cos(theta),
      ], ringIndex(ring, segment) * 3);
    }
  }
  points.set([0, 0, -radius], southPole * 3);

  const cells: number[] = [];
  for (let segment = 0; segment < lonBands; segment += 1) {
    cells.push(3, 0, ringIndex(1, segment), ringIndex(1, segment + 1));
  }
  for (let ring = 1; ring < latBands - 1; ring += 1) {
    for (let segment = 0; segment < lonBands; segment += 1) {
      const a = ringIndex(ring, segment);
      const b = ringIndex(ring, segment + 1);
      const c = ringIndex(ring + 1, segment + 1);
      const d = ringIndex(ring + 1, segment);
      cells.push(3, a, d, c, 3, a, c, b);
    }
  }
  for (let segment = 0; segment < lonBands; segment += 1) {
    cells.push(3, southPole, ringIndex(latBands - 1, segment + 1), ringIndex(latBands - 1, segment));
  }

  return { points, polys: new Uint32Array(cells) };
}

function planeGrid(resolution: number): Mesh {
  const stride = resolution + 1;
  const points = new Float32Array(stride * stride * 3);
  for (let y = 0; y <= resolution; y += 1) {
    for (let x = 0; x <= resolution; x += 1) {
      points.set([x / resolution, y / resolution, 0], (y * stride + x) * 3);
    }
  }

  const cells: number[] = [];
  for (let y = 0; y < resolution; y += 1) {
    for (let x = 0; x < resolution; x += 1) {
      const a = y * stride + x;
      const b = a + 1;
      const c = a + stride + 1;
      const d = a + stride;
      cells.push(3, a, b, c, 3, a, c, d);
    }
  }

  return { points, polys: new Uint32Array(cells) };
}

describe('meshDecimate invariants', () => {
  it('builds closed, outward-oriented spheres for the tests', () => {
    const sphere = uvSphere(10, 12, 16);
    validateMesh(sphere);
    expect(isWatertight(sphere)).toBe(true);
    expect(isManifold(sphere)).toBe(true);
    expect(hasConsistentOutwardOrientation(sphere)).toBe(true);
  });

  it('returns an equivalent mesh at targetReduction 0', () => {
    const sphere = uvSphere(10, 12, 16);
    const result = meshDecimate(sphere, { targetReduction: 0 });

    validateMesh(result);
    expect(triangleCount(result)).toBe(triangleCount(sphere));
    expect(vertexCount(result)).toBe(vertexCount(sphere));
    expect(symmetricHausdorffDistance(result, sphere)).toBeCloseTo(0);
  });

  it('reaches the requested reduction with valid triangles on a closed sphere', () => {
    const sphere = uvSphere(10, 24, 32);
    const inputTriangles = triangleCount(sphere);
    const result = meshDecimate(sphere, { targetReduction: 0.5 });

    validateMesh(result);
    const outputTriangles = triangleCount(result);
    // The collapse loop exits at the first collapse where the running reduction
    // meets the target, and one collapse deletes at most a handful of cells.
    expect(outputTriangles).toBeLessThanOrEqual(Math.ceil(inputTriangles * 0.5));
    expect(outputTriangles).toBeGreaterThanOrEqual(Math.floor(inputTriangles * 0.5) - 4);
    expect(isWatertight(result)).toBe(true);
    expect(isManifold(result)).toBe(true);
    expect(hasConsistentOutwardOrientation(result)).toBe(true);
  });

  it('is deterministic across runs', () => {
    const sphere = uvSphere(10, 24, 32);
    const first = meshDecimate(sphere, { targetReduction: 0.9 });
    const second = meshDecimate(sphere, { targetReduction: 0.9 });

    expect(second.points).toEqual(first.points);
    expect(second.polys).toEqual(first.polys);
  });

  it('decimates an open plane patch while keeping its footprint', () => {
    const plane = planeGrid(20);
    const inputTriangles = triangleCount(plane);
    const result = meshDecimate(plane, { targetReduction: 0.5 });

    validateMesh(result);
    expect(triangleCount(result)).toBeLessThanOrEqual(Math.ceil(inputTriangles * 0.5));
    expect(triangleCount(result)).toBeGreaterThan(0);

    const inputBounds = boundingBox(plane);
    const outputBounds = boundingBox(result);
    // Boundary quadric constraints keep the patch outline near the original
    // unit square; allow a small fraction of the unit extent for drift.
    for (let axis = 0; axis < 2; axis += 1) {
      expect(Math.abs(outputBounds.min[axis] - inputBounds.min[axis])).toBeLessThanOrEqual(0.1);
      expect(Math.abs(outputBounds.max[axis] - inputBounds.max[axis])).toBeLessThanOrEqual(0.1);
    }
  });

  it('survives a degenerate repeated-vertex triangle like VTK', () => {
    // Mirrors VTK's TestQuadricDecimationDegenerateTriangle: append a
    // repeated-vertex triangle to a triangulated plane and require every
    // output cell to still have three vertex ids.
    const plane = planeGrid(10);
    const polys = new Uint32Array(plane.polys.length + 4);
    polys.set(plane.polys, 0);
    polys.set([3, 0, 0, 1], plane.polys.length);
    const degenerate = { points: plane.points, polys };

    const result = meshDecimate(degenerate, { targetReduction: 0.5 });
    expect(result.polys.length % 4).toBe(0);
    for (let offset = 0; offset < result.polys.length; offset += 4) {
      expect(result.polys[offset]).toBe(3);
    }
    for (const index of result.polys) {
      expect(index).toBeLessThan(result.points.length / 3 + 1);
    }
  });

  it('returns an empty mesh for empty input', () => {
    const result = meshDecimate(
      { points: new Float32Array(0), polys: new Uint32Array(0) },
      { targetReduction: 0.5 },
    );
    expect(result.points.length).toBe(0);
    expect(result.polys.length).toBe(0);
  });

  it('clamps targetReduction to [0, 1] like VTK', () => {
    const sphere = uvSphere(10, 12, 16);
    const clamped = meshDecimate(sphere, { targetReduction: 2 });
    const full = meshDecimate(sphere, { targetReduction: 1 });

    expect(clamped.polys).toEqual(full.polys);
    expect(clamped.points).toEqual(full.points);

    const negative = meshDecimate(sphere, { targetReduction: -1 });
    expect(triangleCount(negative)).toBe(triangleCount(sphere));
  });

  it('rejects non-finite targetReduction', () => {
    const sphere = uvSphere(10, 6, 8);
    expect(() => meshDecimate(sphere, { targetReduction: Number.NaN })).toThrow(RangeError);
  });

  it('rejects malformed polys', () => {
    const broken = {
      points: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
      polys: new Uint32Array([4, 0, 1, 2, 3]),
    };
    expect(() => meshDecimate(broken, { targetReduction: 0.5 })).toThrow(RangeError);
  });
});
