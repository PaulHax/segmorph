import { describe, expect, it } from 'vitest';

import { labelmapToSurface } from '../src/convert/labelmapToSurface.js';
import { triangleCount, vertexCount } from '../src/geometry/mesh.js';
import { createOrientedImage } from '../src/image/orientedImage.js';
import { boundingBox, enclosedVolume } from './diff/mesh.js';
import {
  hasConsistentOutwardOrientation,
  isManifold,
  isWatertight,
} from './diff/structure.js';

describe('labelmapToSurface', () => {
  const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const;

  function pointLabelmap(points: readonly (readonly [number, number, number])[], dims = [3, 3, 3]) {
    const data = new Uint8Array(dims[0] * dims[1] * dims[2]);
    for (const [x, y, z] of points) data[x + dims[0] * (y + dims[1] * z)] = 7;
    return createOrientedImage({
      data,
      dims,
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    });
  }

  it('extracts a closed outward-oriented surface around one voxel', () => {
    const diagonal = Math.SQRT1_2;
    const image = createOrientedImage({
      data: new Uint8Array([1]),
      dims: [1, 1, 1],
      spacing: [2, 4, 6],
      origin: [10, 20, 30],
      direction: [[diagonal, -diagonal, 0], [diagonal, diagonal, 0], [0, 0, 1]],
    });

    const mesh = labelmapToSurface(image, { labelValue: 1 });

    expect(vertexCount(mesh)).toBe(6);
    expect(triangleCount(mesh)).toBe(8);
    const bounds = boundingBox(mesh);
    expect(bounds.min[0]).toBeCloseTo(10 - Math.SQRT2);
    expect(bounds.min[1]).toBeCloseTo(20 - Math.SQRT2);
    expect(bounds.min[2]).toBeCloseTo(27);
    expect(bounds.max[0]).toBeCloseTo(10 + Math.SQRT2);
    expect(bounds.max[1]).toBeCloseTo(20 + Math.SQRT2);
    expect(bounds.max[2]).toBeCloseTo(33);
    expect(enclosedVolume(mesh)).toBeCloseTo(8);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
  });

  it('returns an empty mesh for an empty labelmap', () => {
    const image = createOrientedImage({
      data: new Uint8Array(8),
      dims: [2, 2, 2],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    });

    const mesh = labelmapToSurface(image, { labelValue: 1 });
    expect(vertexCount(mesh)).toBe(0);
    expect(triangleCount(mesh)).toBe(0);
  });

  it('selects an exact wide label value', () => {
    const image = createOrientedImage({
      data: new Uint16Array([256, 257]),
      dims: [2, 1, 1],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    });

    expect(enclosedVolume(labelmapToSurface(image, { labelValue: 256 }))).toBeCloseTo(1 / 6);
    expect(vertexCount(labelmapToSurface(image, { labelValue: 258 }))).toBe(0);
    expect(() => labelmapToSurface(image, { labelValue: 256.5 })).toThrow(/labelValue/);
  });

  it.each([
    ['face', [[1, 1, 1], [2, 1, 1]], 10, 16, 2 / 3],
    ['edge', [[1, 1, 1], [2, 2, 1]], 12, 16, 1 / 3],
    ['body diagonal', [[1, 1, 1], [2, 2, 2]], 12, 16, 1 / 3],
  ] as const)('handles %s adjacency', (_name, points, vertices, triangles, volume) => {
    const mesh = labelmapToSurface(pointLabelmap(points), { labelValue: 7 });
    expect(vertexCount(mesh)).toBe(vertices);
    expect(triangleCount(mesh)).toBe(triangles);
    expect(enclosedVolume(mesh)).toBeCloseTo(volume);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
  });

  it.each([
    ['one-voxel line', [[1, 1, 0], [1, 1, 1], [1, 1, 2]]],
    ['one-voxel plate', [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]],
  ] as const)('closes a thin %s', (_name, points) => {
    const mesh = labelmapToSurface(pointLabelmap(points), { labelValue: 7 });
    expect(enclosedVolume(mesh)).toBeGreaterThan(0);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
  });

  it('pads foreground touching every image border', () => {
    const points = [
      [0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0],
      [0, 0, 2], [2, 0, 2], [0, 2, 2], [2, 2, 2],
    ] as const;
    const mesh = labelmapToSurface(pointLabelmap(points), { labelValue: 7 });

    expect(boundingBox(mesh)).toEqual({ min: [-0.5, -0.5, -0.5], max: [2.5, 2.5, 2.5] });
    expect(isWatertight(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
  });

  it('keeps outward winding under every axis permutation and reflection', () => {
    const permutations = [
      [0, 1, 2], [0, 2, 1], [1, 0, 2],
      [1, 2, 0], [2, 0, 1], [2, 1, 0],
    ] as const;
    for (const permutation of permutations) {
      for (let signs = 0; signs < 8; signs += 1) {
        const direction = Array.from({ length: 3 }, (_, row) => (
          Array.from({ length: 3 }, (_, column) => (
            permutation[row] === column ? ((signs >> row) & 1 ? -1 : 1) : 0
          ))
        ));
        const image = createOrientedImage({
          data: new Uint8Array([7]),
          dims: [1, 1, 1],
          spacing: [1, 1, 1],
          origin: [0, 0, 0],
          direction,
        });
        const mesh = labelmapToSurface(image, { labelValue: 7 });
        expect(enclosedVolume(mesh)).toBeCloseTo(1 / 6);
        expect(isWatertight(mesh)).toBe(true);
        expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
      }
    }
  });
});
