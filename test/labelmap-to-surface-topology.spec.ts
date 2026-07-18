import { describe, expect, it } from 'vitest';

import { labelmapToSurface } from '../src/convert/labelmapToSurface.js';
import { triangleCount, vertexCount, type Mesh } from '../src/geometry/mesh.js';
import { boundingBox, enclosedVolume } from './diff/mesh.js';
import {
  hasConsistentOutwardOrientation,
  isManifold,
  isWatertight,
} from './diff/structure.js';

const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
] as const;

type Index = readonly [number, number, number];

function extract(
  foreground: readonly Index[],
  options: {
    dims?: readonly number[];
    spacing?: readonly number[];
    origin?: readonly number[];
    direction?: readonly (readonly number[])[];
  } = {},
) {
  const dims = options.dims ?? [5, 5, 5];
  const data = new Uint8Array(dims[0] * dims[1] * dims[2]);
  for (const [x, y, z] of foreground) data[x + dims[0] * (y + dims[1] * z)] = 7;
  return labelmapToSurface({
    data,
    dims,
    spacing: options.spacing ?? [1, 1, 1],
    origin: options.origin ?? [0, 0, 0],
    direction: options.direction ?? identity,
  }, { labelValue: 7 });
}

function expectClosedOutwardManifold(mesh: Mesh) {
  expect(isWatertight(mesh)).toBe(true);
  expect(isManifold(mesh)).toBe(true);
  expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
}

describe('VTK discrete surface topology acceptance', () => {
  it('places a single-voxel surface half a sample outside its center', () => {
    const mesh = extract([[0, 0, 0]], {
      dims: [1, 1, 1],
      spacing: [2, 3, 4],
      origin: [10, 20, 30],
    });

    expect(vertexCount(mesh)).toBe(6);
    expect(triangleCount(mesh)).toBe(8);
    expect(boundingBox(mesh)).toEqual({ min: [9, 18.5, 28], max: [11, 21.5, 32] });
    expect(enclosedVolume(mesh)).toBeCloseTo(4, 6);
    expectClosedOutwardManifold(mesh);
  });

  it.each([
    {
      name: 'one-voxel-wide line',
      points: [[1, 2, 2], [2, 2, 2], [3, 2, 2]] as Index[],
      volume: 7 / 6,
    },
    {
      name: 'one-voxel-thick sheet',
      points: [
        [1, 1, 2], [2, 1, 2], [3, 1, 2],
        [1, 2, 2], [2, 2, 2], [3, 2, 2],
      ] as Index[],
      volume: 11 / 3,
    },
  ])('preserves a closed $name', ({ points, volume }) => {
    const mesh = extract(points);

    expect(enclosedVolume(mesh)).toBeCloseTo(volume, 6);
    expectClosedOutwardManifold(mesh);
  });

  it('closes foreground that touches every image border', () => {
    const mesh = extract([[0, 0, 0], [1, 1, 1]], { dims: [2, 2, 2] });

    expect(enclosedVolume(mesh)).toBeCloseTo(1 / 3, 6);
    expect(isWatertight(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
  });

  it('keeps winding outward under an axis reflection', () => {
    const mesh = extract([[1, 1, 1], [2, 1, 1]], {
      direction: [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
    });

    expect(enclosedVolume(mesh)).toBeCloseTo(2 / 3, 6);
    expectClosedOutwardManifold(mesh);
  });

  it.each([
    {
      adjacency: 'face',
      offsets: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] as Index[],
      volume: 2 / 3,
    },
    {
      adjacency: 'edge',
      offsets: [[1, 1, 0], [-1, 0, 1], [0, 1, -1]] as Index[],
      volume: 1 / 3,
    },
    {
      adjacency: 'body diagonal',
      offsets: [[1, 1, 1], [-1, 1, -1]] as Index[],
      volume: 1 / 3,
    },
  ])('is symmetric for $adjacency adjacency under axis permutations and reflections', ({ offsets, volume }) => {
    const summaries = offsets.map(([dx, dy, dz]) => {
      const mesh = extract([[2, 2, 2], [2 + dx, 2 + dy, 2 + dz]]);
      expect(enclosedVolume(mesh)).toBeCloseTo(volume, 6);
      return {
        vertices: vertexCount(mesh),
        triangles: triangleCount(mesh),
        watertight: isWatertight(mesh),
        manifold: isManifold(mesh),
        outward: hasConsistentOutwardOrientation(mesh),
      };
    });

    expect(summaries.slice(1)).toEqual(summaries.slice(1).map(() => summaries[0]));
  });

  it('makes face-adjacent voxels one manifold component', () => {
    expectClosedOutwardManifold(extract([[2, 2, 2], [3, 2, 2]]));
  });

  // These stronger component-separation semantics are not guaranteed by classic VTK cases.
  it.todo('keeps edge-adjacent components topologically separate and manifold');
  it.todo('keeps body-diagonal components topologically separate and manifold');
});
