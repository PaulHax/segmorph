import { describe, expect, it } from 'vitest';

import { contourToLabelmap } from '../src/convert/contourToLabelmap.js';
import type { PlanarContour } from '../src/geometry/contour.js';
import { indexToWorld, type ImageGeometry } from '../src/image/orientedImage.js';

/**
 * Boundary convention under test (matches vtkImageStencilRaster with zero
 * tolerance): voxel centers sit at integer index coordinates, edge crossings
 * count for rows j with edgeMinY < j <= edgeMaxY, and a crossing span fills
 * columns i with spanMinX < i <= spanMaxX. Half-open on the minimum side, so
 * two polygons sharing an edge never double-fill and never leave a gap.
 */

const identityDirection = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

const identityGeometry = (dims: readonly number[]): ImageGeometry => ({
  dims,
  spacing: [1, 1, 1],
  origin: [0, 0, 0],
  direction: identityDirection,
});

const sliceContour = (z: number, ...loops: readonly (readonly number[])[]): PlanarContour => ({
  plane: { origin: [0, 0, z], xAxis: [1, 0, 0], yAxis: [0, 1, 0] },
  loops: loops.map((points) => ({ points: Float64Array.from(points) })),
});

const rectangleLoop = (x0: number, y0: number, x1: number, y1: number) => (
  [x0, y0, x1, y0, x1, y1, x0, y1]
);

const regularLoop = (
  cx: number,
  cy: number,
  radius: number,
  sides: number,
  phase: number,
) => Array.from({ length: sides }, (_, vertex) => {
  const angle = (2 * Math.PI * vertex) / sides + phase;
  return [cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)];
}).flat();

const filledVoxels = (image: { data: ArrayLike<number>; dims: readonly number[] }) => {
  const [nx, ny, nz] = image.dims;
  const voxels: [number, number, number][] = [];
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        if (image.data[x + nx * (y + ny * z)] !== 0) voxels.push([x, y, z]);
      }
    }
  }
  return voxels;
};

const gridVoxels = (xs: readonly number[], ys: readonly number[], z: number) => (
  ys.flatMap((y) => xs.map((x): [number, number, number] => [x, y, z]))
);

const range = (first: number, last: number) => (
  Array.from({ length: last - first + 1 }, (_, offset) => first + offset)
);

const count = (image: { data: ArrayLike<number> }) => {
  let total = 0;
  for (let index = 0; index < image.data.length; index += 1) {
    total += Number(image.data[index] !== 0);
  }
  return total;
};

const multiply = (a: number[][], b: number[][]) => a.map((row, i) => (
  row.map((_, j) => a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j])
));

const rotationX = (angle: number) => [
  [1, 0, 0],
  [0, Math.cos(angle), -Math.sin(angle)],
  [0, Math.sin(angle), Math.cos(angle)],
];

const rotationZ = (angle: number) => [
  [Math.cos(angle), -Math.sin(angle), 0],
  [Math.sin(angle), Math.cos(angle), 0],
  [0, 0, 1],
];

const column = (matrix: readonly (readonly number[])[], index: number) => [
  matrix[0][index], matrix[1][index], matrix[2][index],
] as const;

describe('contourToLabelmap boundary convention', () => {
  it('fills voxel centers inside a fractional rectangle', () => {
    const image = contourToLabelmap(
      [sliceContour(1, rectangleLoop(1.25, 2.25, 5.75, 4.75))],
      identityGeometry([8, 8, 3]),
      { labelValue: 1 },
    );
    // x in (1.25, 5.75] -> 2..5, y in (2.25, 4.75] -> 3..4
    expect(filledVoxels(image)).toEqual(gridVoxels(range(2, 5), range(3, 4), 1));
  });

  it('is half-open on gridline-exact edges: min side excluded, max side included', () => {
    const image = contourToLabelmap(
      [sliceContour(1, rectangleLoop(2, 1, 5, 4))],
      identityGeometry([8, 8, 3]),
      { labelValue: 1 },
    );
    // x in (2, 5] -> 3..5, y in (1, 4] -> 2..4
    expect(filledVoxels(image)).toEqual(gridVoxels(range(3, 5), range(2, 4), 1));
  });

  it('never double-fills or gaps adjacent polygons sharing an edge', () => {
    const image = contourToLabelmap(
      [
        sliceContour(1, rectangleLoop(2, 1, 5, 4)),
        sliceContour(1, rectangleLoop(5, 1, 8, 4)),
      ],
      identityGeometry([10, 8, 3]),
      { labelValue: 1 },
    );
    expect(filledVoxels(image)).toEqual(gridVoxels(range(3, 8), range(2, 4), 1));
    expect(count(image)).toBe(18);
  });
});

describe('contourToLabelmap fill rule', () => {
  it('fills a concave polygon correctly', () => {
    const image = contourToLabelmap(
      [sliceContour(1, [1.5, 1.5, 6.5, 1.5, 6.5, 3.5, 3.5, 3.5, 3.5, 6.5, 1.5, 6.5])],
      identityGeometry([8, 8, 3]),
      { labelValue: 1 },
    );
    expect(filledVoxels(image)).toEqual([
      ...gridVoxels(range(2, 6), range(2, 3), 1),
      ...gridVoxels(range(2, 3), range(4, 6), 1),
    ]);
  });

  it('treats nested loops as holes (even-odd rule)', () => {
    const image = contourToLabelmap(
      [sliceContour(1, rectangleLoop(0.5, 0.5, 7.5, 7.5), rectangleLoop(2.5, 2.5, 5.5, 5.5))],
      identityGeometry([9, 9, 3]),
      { labelValue: 1 },
    );
    for (const [x, y] of gridVoxels(range(3, 5), range(3, 5), 1)) {
      expect(image.data[x + 9 * (y + 9 * 1)]).toBe(0);
    }
    expect(count(image)).toBe(7 * 7 - 3 * 3);
  });

  it('fills two disjoint loops of one contour', () => {
    const image = contourToLabelmap(
      [sliceContour(1, rectangleLoop(0.5, 0.5, 3.5, 3.5), rectangleLoop(5.5, 4.5, 8.5, 7.5))],
      identityGeometry([10, 9, 3]),
      { labelValue: 1 },
    );
    expect(filledVoxels(image)).toEqual([
      ...gridVoxels(range(1, 3), range(1, 3), 1),
      ...gridVoxels(range(6, 8), range(5, 7), 1),
    ]);
  });

  it('composes overlapping loops of one contour by even-odd parity', () => {
    const image = contourToLabelmap(
      [sliceContour(1, rectangleLoop(1.5, 1.5, 4.5, 4.5), rectangleLoop(2.5, 2.5, 5.5, 5.5))],
      identityGeometry([8, 8, 3]),
      { labelValue: 1 },
    );
    // 9 + 9 voxels minus the doubly-covered 2x2 overlap counted zero times
    expect(count(image)).toBe(10);
    expect(image.data[3 + 8 * (3 + 8 * 1)]).toBe(0);
  });

  it('composes overlapping separate contours by union', () => {
    const image = contourToLabelmap(
      [
        sliceContour(1, rectangleLoop(1.5, 1.5, 4.5, 4.5)),
        sliceContour(1, rectangleLoop(2.5, 2.5, 5.5, 5.5)),
      ],
      identityGeometry([8, 8, 3]),
      { labelValue: 1 },
    );
    expect(count(image)).toBe(14);
    expect(image.data[3 + 8 * (3 + 8 * 1)]).toBe(1);
  });

  it('rasterizes contours on their own slices only', () => {
    const image = contourToLabelmap(
      [
        sliceContour(0, rectangleLoop(0.5, 0.5, 3.5, 3.5)),
        sliceContour(2, rectangleLoop(3.5, 3.5, 6.5, 6.5)),
      ],
      identityGeometry([8, 8, 3]),
      { labelValue: 1 },
    );
    expect(filledVoxels(image)).toEqual([
      ...gridVoxels(range(1, 3), range(1, 3), 0),
      ...gridVoxels(range(4, 6), range(4, 6), 2),
    ]);
  });

  it('rasterizes sub-voxel polygons by center containment', () => {
    const image = contourToLabelmap(
      [sliceContour(1,
        [2.7, 2.8, 3.4, 2.7, 3.05, 3.45],
        [5.45, 5.3, 5.8, 5.25, 5.6, 5.6])],
      identityGeometry([8, 8, 3]),
      { labelValue: 1 },
    );
    expect(filledVoxels(image)).toEqual([[3, 3, 1]]);
  });

  it('clips polygons extending past the image extent', () => {
    const image = contourToLabelmap(
      [sliceContour(1, rectangleLoop(-3.5, -2.5, 2.5, 2.5))],
      identityGeometry([6, 6, 3]),
      { labelValue: 1 },
    );
    expect(filledVoxels(image)).toEqual(gridVoxels(range(0, 2), range(0, 2), 1));
  });
});

describe('contourToLabelmap geometry handling', () => {
  it('handles anisotropic spacing through world coordinates', () => {
    const geometry: ImageGeometry = {
      dims: [8, 5, 3],
      spacing: [0.5, 2, 1],
      origin: [10, -5, 2],
      direction: identityDirection,
    };
    const contour: PlanarContour = {
      plane: { origin: [10, -5, 3], xAxis: [1, 0, 0], yAxis: [0, 1, 0] },
      loops: [{ points: Float64Array.from(rectangleLoop(0.6, 1, 3.1, 7)) }],
    };
    const image = contourToLabelmap([contour], geometry, { labelValue: 1 });
    // i in (1.2, 6.2] -> 2..6, j in (0.5, 3.5] -> 1..3
    expect(filledVoxels(image)).toEqual(gridVoxels(range(2, 6), range(1, 3), 1));
  });

  it('produces the identical voxel pattern for oblique geometry with matching planes', () => {
    const direction = multiply(rotationZ(0.5), rotationX(0.3));
    const oblique: ImageGeometry = {
      dims: [10, 8, 4],
      spacing: [1.2, 0.8, 2.5],
      origin: [4, -3, 7],
      direction,
    };
    const identityTwin: ImageGeometry = {
      dims: [10, 8, 4],
      spacing: [1.2, 0.8, 2.5],
      origin: [0, 0, 0],
      direction: identityDirection,
    };
    const loop = regularLoop(5.1, 2.7, 2.3, 7, 0.15);
    const obliqueContour: PlanarContour = {
      plane: {
        origin: indexToWorld(oblique, [0, 0, 2]),
        xAxis: column(direction, 0),
        yAxis: column(direction, 1),
      },
      loops: [{ points: Float64Array.from(loop) }],
    };
    const identityContour: PlanarContour = {
      plane: { origin: [0, 0, 5], xAxis: [1, 0, 0], yAxis: [0, 1, 0] },
      loops: [{ points: Float64Array.from(loop) }],
    };

    const obliqueImage = contourToLabelmap([obliqueContour], oblique, { labelValue: 1 });
    const identityImage = contourToLabelmap([identityContour], identityTwin, { labelValue: 1 });

    expect(count(obliqueImage)).toBeGreaterThan(0);
    expect(obliqueImage.data).toEqual(identityImage.data);
    expect(obliqueImage.dims).toEqual(oblique.dims);
    expect(obliqueImage.spacing).toEqual(oblique.spacing);
    expect(obliqueImage.origin).toEqual(oblique.origin);
    expect(obliqueImage.direction).toEqual(oblique.direction);
  });

  it('returns all background with exact geometry for an empty contour list', () => {
    const geometry = identityGeometry([4, 5, 6]);
    const image = contourToLabelmap([], geometry, { labelValue: 1 });
    expect(image.data).toBeInstanceOf(Uint8Array);
    expect(count(image)).toBe(0);
    expect(image.dims).toEqual(geometry.dims);
    expect(image.spacing).toEqual(geometry.spacing);
    expect(image.origin).toEqual(geometry.origin);
    expect(image.direction).toEqual(geometry.direction);
  });
});

describe('contourToLabelmap slice alignment policy', () => {
  const geometry = identityGeometry([8, 8, 3]);
  const loop = rectangleLoop(1.5, 1.5, 4.5, 4.5);

  it('rejects contour planes that are not parallel to the image slices', () => {
    const tilted: PlanarContour = {
      plane: {
        origin: [0, 0, 1],
        xAxis: [Math.cos(0.01), 0, Math.sin(0.01)],
        yAxis: [0, 1, 0],
      },
      loops: [{ points: Float64Array.from(loop) }],
    };
    expect(() => contourToLabelmap([tilted], geometry, { labelValue: 1 }))
      .toThrow(RangeError);
  });

  it('rejects parallel contour planes that land between slices', () => {
    expect(() => contourToLabelmap([sliceContour(1.5, loop)], geometry, { labelValue: 1 }))
      .toThrow(RangeError);
  });

  it('snaps planes within the alignment tolerance onto the slice', () => {
    const image = contourToLabelmap(
      [sliceContour(1 + 5e-7, loop)],
      geometry,
      { labelValue: 1 },
    );
    expect(filledVoxels(image)).toEqual(gridVoxels(range(2, 4), range(2, 4), 1));
  });

  it('ignores aligned planes outside the image extent', () => {
    const image = contourToLabelmap(
      [sliceContour(-1, loop), sliceContour(3, loop)],
      geometry,
      { labelValue: 1 },
    );
    expect(count(image)).toBe(0);
  });
});

describe('contourToLabelmap label values', () => {
  const geometry = identityGeometry([8, 8, 3]);
  const contour = sliceContour(1, rectangleLoop(1.5, 1.5, 4.5, 4.5));

  it('writes the requested label value', () => {
    const image = contourToLabelmap([contour], geometry, { labelValue: 7 });
    expect(image.data[2 + 8 * (2 + 8 * 1)]).toBe(7);
    expect(new Set(image.data)).toEqual(new Set([0, 7]));
  });

  it('allocates the smallest sufficient scalar type', () => {
    expect(contourToLabelmap([contour], geometry, { labelValue: 255 }).data)
      .toBeInstanceOf(Uint8Array);
    expect(contourToLabelmap([contour], geometry, { labelValue: 300 }).data)
      .toBeInstanceOf(Uint16Array);
    expect(contourToLabelmap([contour], geometry, { labelValue: 70_000 }).data)
      .toBeInstanceOf(Uint32Array);
  });

  it('rejects invalid label values', () => {
    for (const labelValue of [0, -1, 1.5, Number.NaN]) {
      expect(() => contourToLabelmap([contour], geometry, { labelValue })).toThrow();
    }
  });
});

describe('contourToLabelmap area invariant', () => {
  it('fills approximately polygon area per slice', () => {
    const loop = regularLoop(7.6, 7.4, 5.3, 7, 0.15);
    const image = contourToLabelmap(
      [sliceContour(0, loop)],
      identityGeometry([16, 16, 1]),
      { labelValue: 1 },
    );

    let area = 0;
    let perimeter = 0;
    const sides = loop.length / 2;
    for (let vertex = 0; vertex < sides; vertex += 1) {
      const next = (vertex + 1) % sides;
      area += loop[2 * vertex] * loop[2 * next + 1] - loop[2 * next] * loop[2 * vertex + 1];
      perimeter += Math.hypot(
        loop[2 * next] - loop[2 * vertex],
        loop[2 * next + 1] - loop[2 * vertex + 1],
      );
    }
    area = Math.abs(area) / 2;

    // Digitization error of a convex region is bounded by its perimeter.
    // Measured: 79 filled voxels vs area 76.866 (diff 2.134) against the
    // perimeter / 2 = 16.097 band.
    expect(Math.abs(count(image) - area)).toBeLessThanOrEqual(perimeter / 2);
  });
});
