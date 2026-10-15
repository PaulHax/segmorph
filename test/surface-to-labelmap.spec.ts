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
  [0, 2, 1],
  [0, 3, 2],
  [4, 5, 6],
  [4, 6, 7],
  [0, 1, 5],
  [0, 5, 4],
  [3, 7, 6],
  [3, 6, 2],
  [0, 4, 7],
  [0, 7, 3],
  [1, 2, 6],
  [1, 6, 5],
] as const;

function cube(geometry: ImageGeometry, minimum: number, maximum: number) {
  const indexPoints: Point[] = [
    [minimum, minimum, minimum],
    [maximum, minimum, minimum],
    [maximum, maximum, minimum],
    [minimum, maximum, minimum],
    [minimum, minimum, maximum],
    [maximum, minimum, maximum],
    [maximum, maximum, maximum],
    [minimum, maximum, maximum],
  ];
  const points = indexPoints.map((point) => indexToWorld(geometry, point));
  return createMesh(points, triangles);
}

describe('surfaceToLabelmap', () => {
  it('fills samples inside a closed surface', () => {
    const geometry = {
      dims: [4, 4, 4],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0.5, 2.5), geometry, { labelValue: 1 });
    expect([...result.data].filter(Boolean)).toHaveLength(8);
    expect(result.data[1 + 4 * (1 + 4 * 1)]).toBe(1);
    expect(result.data[3 + 4 * (1 + 4 * 1)]).toBe(0);
  });

  it('treats samples on the surface as foreground', () => {
    const geometry = {
      dims: [3, 3, 3],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0, 2), geometry, { labelValue: 1 });
    expect([...result.data]).toEqual(new Array(27).fill(1));
  });

  it('preserves occupancy when mesh and reference geometry translate together', () => {
    const geometry = {
      dims: [7, 7, 7],
      spacing: [0.5, 0.5, 0.5],
      origin: [-0.5, -0.5, -0.5],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    const mesh = cube(geometry, 1, 5);
    const expected = surfaceToLabelmap(mesh, geometry, { labelValue: 1 });
    const translated = {
      ...geometry,
      origin: [-0.5, 10_000_000 - 0.5, -0.5],
    };
    const actual = surfaceToLabelmap(cube(translated, 1, 5), translated, { labelValue: 1 });

    expect(actual.data).toEqual(expected.data);
    expect([...actual.data].filter(Boolean)).toHaveLength(125);
    expect(actual.data[7 * (3 + 7 * 3)]).toBe(0);
  });

  it.each([1, 0.001, 1e-6])('preserves occupancy with sample spacing %s', (spacing) => {
    const geometry = {
      dims: [7, 7, 7],
      spacing: [spacing, spacing, spacing],
      origin: [-spacing, -spacing, -spacing],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    const actual = surfaceToLabelmap(cube(geometry, 1, 5), geometry, { labelValue: 1 });
    const expected = new Uint8Array(7 * 7 * 7);
    for (let z = 1; z <= 5; z += 1) {
      for (let y = 1; y <= 5; y += 1) {
        for (let x = 1; x <= 5; x += 1) expected[x + 7 * (y + 7 * z)] = 1;
      }
    }

    expect(actual.data).toEqual(expected);
  });

  it.each([0.1, 0.01, 0.001])(
    'includes translated float32 boundary samples with spacing %s',
    (spacing) => {
      const geometry = {
        dims: [3, 3, 3],
        spacing: [spacing, spacing, spacing],
        origin: [100, -100, 0],
        direction: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
      } as const;

      expect(surfaceToLabelmap(cube(geometry, 0, 2), geometry, { labelValue: 1 }).data).toEqual(
        new Uint8Array(27).fill(1),
      );
    },
  );

  it('includes translated float32 boundaries on oblique faces', () => {
    const diagonal = Math.SQRT1_2;
    const geometry = {
      dims: [3, 3, 3],
      spacing: [0.001, 0.001, 0.001],
      origin: [100, -100, 0],
      direction: [
        [diagonal, -diagonal, 0],
        [diagonal, diagonal, 0],
        [0, 0, 1],
      ],
    } as const;

    expect(surfaceToLabelmap(cube(geometry, 0, 2), geometry, { labelValue: 1 }).data).toEqual(
      new Uint8Array(27).fill(1),
    );
  });

  it('includes translated float32 boundaries with anisotropic spacing', () => {
    const geometry = {
      dims: [5, 5, 5],
      spacing: [0.05, 0.2, 1.1],
      origin: [100, -100, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    const actual = surfaceToLabelmap(cube(geometry, 1, 3), geometry, { labelValue: 1 });
    const expected = new Uint8Array(5 ** 3);
    for (let z = 1; z <= 3; z += 1) {
      for (let y = 1; y <= 3; y += 1) {
        for (let x = 1; x <= 3; x += 1) expected[x + 5 * (y + 5 * z)] = 1;
      }
    }

    expect(actual.data).toEqual(expected);
  });

  it('keeps exterior samples empty when a fine grid is translated far from zero', () => {
    const geometry = {
      dims: [43, 43, 43],
      spacing: [0.05, 0.05, 0.05],
      origin: [-0.05, 1_000_000 - 0.05, -0.05],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    const actual = surfaceToLabelmap(cube(geometry, 1, 41), geometry, { labelValue: 1 });
    const expected = new Uint8Array(43 ** 3);
    for (let z = 1; z <= 41; z += 1) {
      for (let y = 1; y <= 41; y += 1) {
        for (let x = 1; x <= 41; x += 1) expected[x + 43 * (y + 43 * z)] = 1;
      }
    }

    expect(actual.data).toEqual(expected);
  });

  it('keeps sub-step exterior samples empty across axes with different coordinate magnitudes', () => {
    const geometry = {
      dims: [23, 23, 23],
      spacing: [0.05, 0.05, 0.05],
      origin: [1_000_000, -0.01, -0.01],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    const mesh = cube({ ...geometry, origin: [1_000_000, 0, 0] }, 0, 20);
    const actual = surfaceToLabelmap(mesh, geometry, { labelValue: 1 });
    const expected = new Uint8Array(23 ** 3);
    for (let z = 1; z <= 20; z += 1) {
      for (let y = 1; y <= 20; y += 1) {
        for (let x = 0; x <= 20; x += 1) expected[x + 23 * (y + 23 * z)] = 1;
      }
    }

    expect(actual.data).toEqual(expected);
  });

  it('uses oblique reference geometry', () => {
    const geometry = {
      dims: [4, 4, 4],
      spacing: [2, 3, 4],
      origin: [10, -5, 7],
      direction: [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
      ],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0.5, 2.5), geometry, { labelValue: 1 });
    expect([...result.data].filter(Boolean)).toHaveLength(8);
    expect(result.data[2 + 4 * (2 + 4 * 2)]).toBe(1);
  });

  it('preserves label values that require wider integer storage', () => {
    const geometry = {
      dims: [3, 3, 3],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0.5, 1.5), geometry, { labelValue: 256 });

    expect(result.data).toBeInstanceOf(Uint16Array);
    expect(result.data[1 + 3 * (1 + 3 * 1)]).toBe(256);
  });

  describe('explicit output storage', () => {
    const geometry = {
      dims: [2, 1, 1],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    const mesh = cube(geometry, -0.5, 0.5);

    it.each([
      { outputArray: Uint8Array, labelValue: 255 },
      { outputArray: Uint8ClampedArray, labelValue: 255 },
      { outputArray: Uint16Array, labelValue: 256 },
      { outputArray: Uint16Array, labelValue: 65535 },
      { outputArray: Uint32Array, labelValue: 65536 },
      { outputArray: Uint32Array, labelValue: 4294967295 },
      { outputArray: Int8Array, labelValue: 127 },
      { outputArray: Int16Array, labelValue: 32767 },
      { outputArray: Int32Array, labelValue: 2147483647 },
      { outputArray: Float32Array, labelValue: 16777216 },
      { outputArray: Float32Array, labelValue: 16777218 },
      { outputArray: Float64Array, labelValue: 4294967295 },
    ])('preserves label $labelValue in $outputArray.name', ({ outputArray, labelValue }) => {
      const snapshot = structuredClone({ mesh, geometry });
      const result = surfaceToLabelmap(mesh, geometry, { labelValue, outputArray });

      expect(result.data.constructor).toBe(outputArray);
      expect([...result.data]).toEqual([labelValue, 0]);
      expect(labelmapToSurface(result, { labelValue }).points.length).toBeGreaterThan(0);
      expect({ mesh, geometry }).toEqual(snapshot);
    });

    it.each([
      { outputArray: Uint8Array, labelValue: 256 },
      { outputArray: Uint8Array, labelValue: 257 },
      { outputArray: Uint8ClampedArray, labelValue: 256 },
      { outputArray: Uint16Array, labelValue: 65536 },
      { outputArray: Int8Array, labelValue: 128 },
      { outputArray: Int16Array, labelValue: 32768 },
      { outputArray: Int32Array, labelValue: 2147483648 },
      { outputArray: Float32Array, labelValue: 16777217 },
      { outputArray: Float32Array, labelValue: 4294967295 },
    ])('rejects label $labelValue in $outputArray.name', ({ outputArray, labelValue }) => {
      const snapshot = structuredClone({ mesh, geometry });
      expect(() => surfaceToLabelmap(mesh, geometry, { labelValue, outputArray })).toThrow(
        new RangeError(`outputArray cannot represent labelValue ${labelValue} exactly`),
      );
      expect({ mesh, geometry }).toEqual(snapshot);
    });

    it('rejects an unrepresentable label before allocating the full output', () => {
      const lengths: number[] = [];
      class TrackedUint8Array extends Uint8Array {
        constructor(length: number) {
          super(length);
          lengths.push(length);
        }
      }

      expect(() =>
        surfaceToLabelmap(mesh, geometry, { labelValue: 256, outputArray: TrackedUint8Array }),
      ).toThrow(RangeError);
      expect(lengths).toEqual([1]);
    });
  });

  it('includes boundary samples with small anisotropic spacing', () => {
    const geometry = {
      dims: [3, 3, 3],
      spacing: [0.001, 0.002, 0.004],
      origin: [20, -10, 3],
      direction: [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
      ],
    } as const;
    const result = surfaceToLabelmap(cube(geometry, 0, 2), geometry, { labelValue: 1 });

    expect([...result.data]).toEqual(new Array(27).fill(1));
  });

  it('returns an empty labelmap for an empty mesh', () => {
    const geometry = {
      dims: [2, 2, 2],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;

    expect(surfaceToLabelmap(createMesh([], []), geometry, { labelValue: 1 }).data).toEqual(
      new Uint8Array(8),
    );
  });

  it('preserves the source element type through an A->D->composite round trip', () => {
    const geometry = {
      dims: [5, 5, 5],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
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
      dims: [2, 2, 2],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    } as const;
    expect(() =>
      surfaceToLabelmap(
        {
          points: new Float32Array([0, 0, 0]),
          polys: new Uint32Array([3, 0, 1, 2]),
        },
        geometry,
        { labelValue: 1 },
      ),
    ).toThrow();
    expect(() =>
      surfaceToLabelmap(createMesh([], []), { ...geometry, dims: [0, 2, 2] }, { labelValue: 1 }),
    ).toThrow();
  });
});
