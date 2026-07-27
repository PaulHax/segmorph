import { describe, expect, it } from 'vitest';

import {
  boundingBox,
  enclosedVolume,
  meanSurfaceDistance,
  symmetricHausdorffDistance,
  type Mesh,
} from './mesh.js';

function cube(offsetX = 0) {
  return {
    points: new Float32Array([
      offsetX,
      0,
      0,
      offsetX + 1,
      0,
      0,
      offsetX + 1,
      1,
      0,
      offsetX,
      1,
      0,
      offsetX,
      0,
      1,
      offsetX + 1,
      0,
      1,
      offsetX + 1,
      1,
      1,
      offsetX,
      1,
      1,
    ]),
    polys: new Uint32Array([
      3, 0, 2, 1, 3, 0, 3, 2, 3, 4, 5, 6, 3, 4, 6, 7, 3, 0, 1, 5, 3, 0, 5, 4, 3, 3, 7, 6, 3, 3, 6,
      2, 3, 0, 4, 7, 3, 0, 7, 3, 3, 1, 2, 6, 3, 1, 6, 5,
    ]),
  };
}

function reverseWinding(mesh: Mesh) {
  const polys = mesh.polys.slice();
  for (let offset = 0; offset < polys.length; offset += 4) {
    const swap = polys[offset + 2];
    polys[offset + 2] = polys[offset + 3];
    polys[offset + 3] = swap;
  }
  return { ...mesh, polys };
}

describe('mesh metrics', () => {
  it('returns zero surface distances for identical meshes', () => {
    const mesh = cube();

    expect(symmetricHausdorffDistance(mesh, mesh)).toBeCloseTo(0);
    expect(meanSurfaceDistance(mesh, mesh)).toBeCloseTo(0);
  });

  it('measures the symmetric distance between offset unit cubes', () => {
    const actual = cube();
    const expected = cube(0.1);

    // Interior (centroid) samples join the vertex samples, so the mean reflects
    // face centers as well as corners; the Hausdorff extreme is still 0.1.
    expect(symmetricHausdorffDistance(actual, expected)).toBeCloseTo(0.1);
    expect(meanSurfaceDistance(actual, expected)).toBeCloseTo(0.04);
  });

  it('detects interior deviation that vertex-only sampling misses', () => {
    // Two triangulations of the same four saddle corners: one splits the quad
    // along the 0-2 diagonal, the other along 1-3. Every vertex is shared, so
    // sampling vertices alone reports zero distance, yet the interior surfaces
    // diverge. Centroid sampling exposes the difference.
    const points = new Float32Array([0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]);
    const diagonal02 = { points, polys: new Uint32Array([3, 0, 1, 2, 3, 0, 2, 3]) };
    const diagonal13 = { points, polys: new Uint32Array([3, 1, 2, 3, 3, 1, 3, 0]) };

    expect(symmetricHausdorffDistance(diagonal02, diagonal13)).toBeGreaterThan(0.1);
  });

  it('stays finite when a mesh carries a degenerate zero-area triangle', () => {
    const good = cube();
    // A collinear (zero-area) triangle appended to an otherwise valid mesh must
    // not poison the metric with NaN from divide-by-zero in the distance math.
    const degenerate = {
      points: new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]),
      polys: new Uint32Array([3, 0, 1, 2]),
    };

    expect(Number.isFinite(symmetricHausdorffDistance(good, degenerate))).toBe(true);
    expect(Number.isFinite(meanSurfaceDistance(good, degenerate))).toBe(true);
  });

  it('computes signed enclosed volume and bounding box', () => {
    const mesh = cube();

    expect(enclosedVolume(mesh)).toBeCloseTo(1);
    expect(enclosedVolume(reverseWinding(mesh))).toBeCloseTo(-1);
    expect(boundingBox(mesh)).toEqual({ min: [0, 0, 0], max: [1, 1, 1] });
  });
});
