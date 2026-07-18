import { describe, expect, it } from 'vitest';
import {
  compositeSet,
  createMesh,
  indexToWorld,
  labelmapToSurface,
  surfaceToLabelmap,
  type ImageGeometry,
  type Point,
} from '../src/index.js';

const triangles = [
  [0, 2, 1], [0, 3, 2],
  [4, 5, 6], [4, 6, 7],
  [0, 1, 5], [0, 5, 4],
  [3, 7, 6], [3, 6, 2],
  [0, 4, 7], [0, 7, 3],
  [1, 2, 6], [1, 6, 5],
] as const;

function cube(geometry: ImageGeometry, minimum: number, maximum: number) {
  const indexPoints: Point[] = [
    [minimum, minimum, minimum], [maximum, minimum, minimum],
    [maximum, maximum, minimum], [minimum, maximum, minimum],
    [minimum, minimum, maximum], [maximum, minimum, maximum],
    [maximum, maximum, maximum], [minimum, maximum, maximum],
  ];
  const points = indexPoints.map((point) => indexToWorld(geometry, point));
  return createMesh(points, triangles);
}

describe('surfaceToLabelmap', () => {
  it('fills samples inside a closed surface', () => {
    const geometry = {
      dims: [4, 4, 4], spacing: [1, 1, 1], origin: [0, 0, 0],
      direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0.5, 2.5), geometry, { labelValue: 1 });
    expect([...result.data].filter(Boolean)).toHaveLength(8);
    expect(result.data[1 + 4 * (1 + 4 * 1)]).toBe(1);
    expect(result.data[3 + 4 * (1 + 4 * 1)]).toBe(0);
  });

  it('treats samples on the surface as foreground', () => {
    const geometry = {
      dims: [3, 3, 3], spacing: [1, 1, 1], origin: [0, 0, 0],
      direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0, 2), geometry, { labelValue: 1 });
    expect([...result.data]).toEqual(new Array(27).fill(1));
  });

  it('uses oblique reference geometry', () => {
    const geometry = {
      dims: [4, 4, 4], spacing: [2, 3, 4], origin: [10, -5, 7],
      direction: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0.5, 2.5), geometry, { labelValue: 1 });
    expect([...result.data].filter(Boolean)).toHaveLength(8);
    expect(result.data[2 + 4 * (2 + 4 * 2)]).toBe(1);
  });

  it('preserves label values that require wider integer storage', () => {
    const geometry = {
      dims: [3, 3, 3], spacing: [1, 1, 1], origin: [0, 0, 0],
      direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0.5, 1.5), geometry, { labelValue: 256 });

    expect(result.data).toBeInstanceOf(Uint16Array);
    expect(result.data[1 + 3 * (1 + 3 * 1)]).toBe(256);
  });

  it('includes boundary samples with small anisotropic spacing', () => {
    const geometry = {
      dims: [3, 3, 3], spacing: [0.001, 0.002, 0.004], origin: [20, -10, 3],
      direction: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0, 2), geometry, { labelValue: 1 });

    expect([...result.data]).toEqual(new Array(27).fill(1));
  });

  it('returns an empty labelmap for an empty mesh', () => {
    const geometry = {
      dims: [2, 2, 2], spacing: [1, 1, 1], origin: [0, 0, 0],
      direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    } as const;

    expect(surfaceToLabelmap(createMesh([], []), geometry, { labelValue: 1 }).data)
      .toEqual(new Uint8Array(8));
  });

  it('preserves the source element type through an A->D->composite round trip', () => {
    const geometry = {
      dims: [5, 5, 5], spacing: [1, 1, 1], origin: [0, 0, 0],
      direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    } as const;
    // A source Uint16 labelmap with a solid interior block.
    const source = new Uint16Array(5 * 5 * 5);
    for (let z = 1; z <= 3; z += 1) {
      for (let y = 1; y <= 3; y += 1) {
        for (let x = 1; x <= 3; x += 1) {
          source[x + 5 * (y + 5 * z)] = 1;
        }
      }
    }
    const surface = labelmapToSurface({ ...geometry, data: source }, { labelValue: 1 });

    // Default output width is derived from labelValue, so a small label narrows
    // the round trip to Uint8 and compositeSet rejects the type mismatch.
    const narrowed = surfaceToLabelmap(surface, geometry, { labelValue: 1 });
    expect(narrowed.data).toBeInstanceOf(Uint8Array);
    expect(() => compositeSet(source, narrowed.data as never, geometry.dims)).toThrow();

    // outputArray lets the caller preserve the source dtype so composite works.
    const preserved = surfaceToLabelmap(surface, geometry, {
      labelValue: 1,
      outputArray: Uint16Array,
    });
    expect(preserved.data).toBeInstanceOf(Uint16Array);
    const composed = compositeSet(source, preserved.data, geometry.dims);
    expect(composed).toBeInstanceOf(Uint16Array);
    expect(composed[2 + 5 * (2 + 5 * 2)]).toBe(1);
  });

  it('rejects malformed meshes and reference geometry', () => {
    const geometry = {
      dims: [2, 2, 2], spacing: [1, 1, 1], origin: [0, 0, 0],
      direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    } as const;
    expect(() => surfaceToLabelmap({
      points: new Float32Array([0, 0, 0]), polys: new Uint32Array([3, 0, 1, 2]),
    }, geometry, { labelValue: 1 })).toThrow();
    expect(() => surfaceToLabelmap(
      createMesh([], []),
      { ...geometry, dims: [0, 2, 2] },
      { labelValue: 1 },
    )).toThrow();
  });
});
