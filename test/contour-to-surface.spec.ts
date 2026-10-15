import { describe, expect, it } from 'vitest';

import { contourToLabelmap } from '../src/convert/contourToLabelmap.js';
import { contourToSurface } from '../src/convert/contourToSurface.js';
import { surfaceToContour } from '../src/convert/surfaceToContour.js';
import { createContourPlane } from '../src/geometry/contour.js';
import { triangleCount, type Mesh } from '../src/geometry/mesh.js';
import { contourToSurfaceCases } from './contourToSurfaceCases.js';
import { hasConsistentOutwardOrientation, isManifold, isWatertight } from './diff/structure.js';
import { boundingBox } from './diff/mesh.js';

describe('contourToSurface validation', () => {
  it('rejects an empty stack', () => {
    expect(() => contourToSurface([])).toThrow(RangeError);
  });

  it('rejects a loop with fewer than three points', () => {
    expect(() => contourToSurface([[0, 0, 0, 1, 0, 0]])).toThrow(RangeError);
  });

  it('rejects incomplete xyz coordinates', () => {
    expect(() => contourToSurface([[0, 0, 0, 1, 0, 0, 1, 1]])).toThrow(RangeError);
  });

  it('rejects non-finite coordinates', () => {
    expect(() => contourToSurface([[0, 0, 0, 1, 0, 0, Number.NaN, 1, 0]])).toThrow(RangeError);
  });
});

describe('contourToSurface structural invariants', () => {
  // The oracle's own output has open edges (its cap loops close on duplicated
  // coordinates with distinct ids); the port shares vertices along every seam
  // instead, so with default smooth capping the result is fully closed on
  // every fixture case.
  for (const [caseName, { loops }] of Object.entries(contourToSurfaceCases)) {
    it(`produces a watertight, manifold, outward-oriented surface for ${caseName}`, () => {
      const mesh = contourToSurface(loops);
      expect(isWatertight(mesh)).toBe(true);
      expect(isManifold(mesh)).toBe(true);
      expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
    });
  }
});

const axialPlane = (z: number) => createContourPlane([0, 0, z], [1, 0, 0], [0, 1, 0]);

function loopRadialRange(loop: { points: Float64Array }, centerX: number, centerY: number) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let offset = 0; offset < loop.points.length; offset += 2) {
    const radius = Math.hypot(loop.points[offset] - centerX, loop.points[offset + 1] - centerY);
    minimum = Math.min(minimum, radius);
    maximum = Math.max(maximum, radius);
  }
  return { minimum, maximum };
}

function loopArea(loop: { points: Float64Array }) {
  let area = 0;
  const count = loop.points.length / 2;
  for (let index = 0; index < count; index += 1) {
    const next = (index + 1) % count;
    area +=
      loop.points[index * 2] * loop.points[next * 2 + 1] -
      loop.points[next * 2] * loop.points[index * 2 + 1];
  }
  return area / 2;
}

describe('contour -> surface -> contour round trip', () => {
  // Cutting the reconstructed surface between two source planes must
  // reproduce the source loop shape: the stitch never moves points off the
  // source contours, so a mid-slice cut interpolates between two identical
  // circles. Chord error for 24 segments at radius 20 is
  // 20 * (1 - cos(pi/24)) = 0.17, so 0.25 bounds the radial deviation.
  it('reproduces the cylinder circle at a mid slice', () => {
    const mesh = contourToSurface(contourToSurfaceCases.cylinder.loops);
    const contour = surfaceToContour(mesh, axialPlane(2.25));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(1);
    const { minimum, maximum } = loopRadialRange(contour!.loops[0], 5, -3);
    expect(minimum).toBeGreaterThan(20 - 0.25);
    expect(maximum).toBeLessThan(20 + 0.25);
  });

  it('preserves the keyhole annulus hole', () => {
    const mesh = contourToSurface(contourToSurfaceCases.keyhole.loops);
    const contour = surfaceToContour(mesh, axialPlane(2.25));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(2);
    // The keyhole splitter leaves one channel remnant point on each ring
    // (the same remnants the original produces), so compare enclosed areas,
    // which the zero-width channel barely dents: outer near pi * 16^2 = 804
    // (32-segment polygon: 798), inner near pi * 8^2 = 201 (24-segment
    // polygon: 199).
    const areas = contour!.loops
      .map((loop) => Math.abs(loopArea(loop)))
      .sort((left, right) => left - right);
    expect(areas[0]).toBeGreaterThan(185);
    expect(areas[0]).toBeLessThan(210);
    expect(areas[1]).toBeGreaterThan(770);
    expect(areas[1]).toBeLessThan(810);
  });

  it('splits into two loops above the branch plane', () => {
    const mesh = contourToSurface(contourToSurfaceCases.branching.loops);
    const contour = surfaceToContour(mesh, axialPlane(11));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(2);
  });
});

describe('end capping modes', () => {
  const { loops } = contourToSurfaceCases.cylinder;
  const contourMinZ = 0;
  const contourMaxZ = 10.5;
  const spacing = 1.5;

  it('none leaves the tube open at the contour extents', () => {
    const mesh = contourToSurface(loops, { endCapping: 'none' });
    expect(isWatertight(mesh)).toBe(false);
    const bounds = boundingBox(mesh);
    expect(bounds.min[2]).toBeCloseTo(contourMinZ, 5);
    expect(bounds.max[2]).toBeCloseTo(contourMaxZ, 5);
  });

  it('smooth extends half a slice beyond the contour extents', () => {
    const mesh = contourToSurface(loops);
    const bounds = boundingBox(mesh);
    expect(bounds.min[2]).toBeCloseTo(contourMinZ - spacing / 2, 5);
    expect(bounds.max[2]).toBeCloseTo(contourMaxZ + spacing / 2, 5);
  });

  it('straight copies the contour half a slice out of plane', () => {
    const mesh = contourToSurface(loops, { endCapping: 'straight' });
    const bounds = boundingBox(mesh);
    expect(bounds.min[2]).toBeCloseTo(contourMinZ - spacing / 2, 5);
    expect(bounds.max[2]).toBeCloseTo(contourMaxZ + spacing / 2, 5);
    // Straight caps keep the source circle's footprint; smooth caps shrink it.
    const smooth = boundingBox(contourToSurface(loops));
    expect(bounds.max[0] - bounds.min[0]).toBeGreaterThanOrEqual(smooth.max[0] - smooth.min[0]);
  });

  it('defaultSliceThickness drives the caps when only one plane exists', () => {
    const single = [contourToSurfaceCases.cylinder.loops[0]];
    const mesh = contourToSurface(single, { defaultSliceThickness: 4 });
    const bounds = boundingBox(mesh);
    expect(bounds.min[2]).toBeCloseTo(-2, 5);
    expect(bounds.max[2]).toBeCloseTo(2, 5);
  });
});

const polygonAt = (xy: readonly (readonly number[])[], z: number) =>
  xy.flatMap(([x, y]) => [x, y, z]);
const diamond = (radius: number, z: number, centerX = 0) =>
  polygonAt(
    [
      [centerX + radius, 0],
      [centerX, radius],
      [centerX - radius, 0],
      [centerX, -radius],
    ],
    z,
  );
const triangle = (radius: number, z: number) =>
  polygonAt(
    [
      [radius / 2, (radius * 55) / 64],
      [-radius, 0],
      [radius / 2, (-radius * 55) / 64],
    ],
    z,
  );
const reverseLoop = (loop: number[]) =>
  Array.from({ length: loop.length / 3 }, (_, index) => loop.slice(index * 3, index * 3 + 3))
    .reverse()
    .flat();

function cutAreas(mesh: Mesh, z: number) {
  const cut = surfaceToContour(mesh, axialPlane(z));
  expect(cut).toBeDefined();
  return cut!.loops.map((loop) => Math.abs(loopArea(loop))).sort((a, b) => b - a);
}

function cutLabel(mesh: Mesh, z: number, x = 0, y = 0) {
  const cut = surfaceToContour(mesh, axialPlane(z));
  expect(cut).toBeDefined();
  return contourToLabelmap(
    [cut!],
    {
      dims: [1, 1, 1],
      spacing: [1, 1, 1],
      origin: [x, y, z],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    },
    { labelValue: 1 },
  ).data[0];
}

describe('closed straight caps', () => {
  it('closes smooth fallback caps when erosion removes a thin contour', () => {
    // This exact-binary triangle is thinner than a cap raster pixel. Erosion
    // removes the rasterized strip, so the fallback must keep its area-0.5
    // footprint and close both cap extensions with shared vertex ids.
    const thinTriangle = [
      [0, 0],
      [8, 8],
      [8, 7.875],
    ];
    const mesh = contourToSurface([polygonAt(thinTriangle, 0), polygonAt(thinTriangle, 2)]);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    for (const z of [-0.5, 1, 2.5]) expect(cutAreas(mesh, z)).toEqual([0.5]);
  });

  it('makes an 8 by 8 prism with closed area-64 cuts through both cap extensions', () => {
    const square = [
      [0, 0],
      [8, 0],
      [8, 8],
      [0, 8],
    ];
    const mesh = contourToSurface([polygonAt(square, 0), polygonAt(square, 2)], {
      endCapping: 'straight',
    });
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
    expect(boundingBox(mesh)).toEqual({ min: [0, 0, -1], max: [8, 8, 3] });
    for (const z of [-0.5, 1, 2.5]) {
      expect(cutAreas(mesh, z)).toEqual([64]);
      expect(cutLabel(mesh, z, 4, 4)).toBe(1);
    }
  });
});

describe('nested contour correspondence', () => {
  const nested = [diamond(8, 0), diamond(6, 0), triangle(8, 2), triangle(6, 2)];

  it.each(['none', 'smooth', 'straight'] as const)(
    'preserves rational nested rings with %s caps',
    (endCapping) => {
      const mesh = contourToSurface(nested, { endCapping });
      expect(cutAreas(mesh, 1)).toEqual([128.125, 72.0703125]);
      expect(cutLabel(mesh, 1)).toBe(0);
      if (endCapping !== 'none') {
        expect(isWatertight(mesh)).toBe(true);
        expect(isManifold(mesh)).toBe(true);
      }
    },
  );

  it('does not use independent winding or input order to identify holes', () => {
    for (let mask = 0; mask < 16; mask += 1) {
      const loops = nested.map((loop, index) => (mask & (1 << index) ? reverseLoop(loop) : loop));
      for (const order of [
        [0, 1, 2, 3],
        [3, 0, 2, 1],
      ]) {
        const mesh = contourToSurface(
          order.map((index) => loops[index]),
          { endCapping: 'none' },
        );
        expect(cutAreas(mesh, 1)).toEqual([128.125, 72.0703125]);
        expect(cutLabel(mesh, 1)).toBe(0);
      }
    }
  });

  it('distinguishes depth-two islands from root boundaries', () => {
    const mesh = contourToSurface([
      diamond(8, 0),
      diamond(6, 0),
      diamond(3, 0),
      triangle(8, 2),
      triangle(6, 2),
      triangle(3, 2),
    ]);
    expect(cutAreas(mesh, 1)).toEqual([128.125, 72.0703125, 18.017578125]);
    expect(cutLabel(mesh, 1)).toBe(1);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
  });

  // The C's centroid and the second root are in its notch. Both roots' and
  // both holes' bounding boxes overlap, although their polygon interiors do not.
  const cRoot = [
    [0, 0],
    [12, 0],
    [12, 2],
    [2, 2],
    [2, 10],
    [12, 10],
    [12, 12],
    [0, 12],
  ];
  const cHole = [
    [0.5, 0.5],
    [11.5, 0.5],
    [11.5, 1.5],
    [1.5, 1.5],
    [1.5, 10.5],
    [11.5, 10.5],
    [11.5, 11.5],
    [0.5, 11.5],
  ];
  const notchRoot = [
    [4, 4],
    [8, 4],
    [8, 8],
    [4, 8],
  ];
  const notchHole = [
    [5, 5],
    [7, 5],
    [7, 7],
    [5, 7],
  ];

  it('keeps holes with their polygon-compatible parents when sibling bounds overlap', () => {
    const loops = [0, 2].flatMap((z) =>
      [cRoot, cHole, notchRoot, notchHole].map((xy) => polygonAt(xy, z)),
    );
    const mesh = contourToSurface(loops, { endCapping: 'none' });
    expect(cutAreas(mesh, 1)).toEqual([64, 31, 16, 4]);
    expect(cutLabel(mesh, 1, 1, 6)).toBe(0);
    expect(cutLabel(mesh, 1, 1.75, 6)).toBe(1);
    expect(cutLabel(mesh, 1, 6, 6)).toBe(0);
  });

  it('recognizes a concave hole whose centroid lies outside its parent', () => {
    const upperHole = [
      [0.75, 10.75],
      [1.25, 10.75],
      [1.25, 11.25],
      [0.75, 11.25],
    ];
    const lowerHole = [
      [0.75, 0.75],
      [1.25, 0.75],
      [1.25, 1.25],
      [0.75, 1.25],
    ];
    expect(() =>
      contourToSurface([
        polygonAt(cRoot, 0),
        polygonAt(cHole, 0),
        polygonAt(cRoot, 2),
        polygonAt(upperHole, 2),
        polygonAt(lowerHole, 2),
      ]),
    ).toThrowError(/nested-ring split\/merge/);
  });

  it('rejects crossing edges even when all child vertices lie inside a concave parent', () => {
    const u = [
      [0, 0],
      [8, 0],
      [8, 8],
      [6, 8],
      [6, 2],
      [2, 2],
      [2, 8],
      [0, 8],
    ];
    const crossing = [
      [1, 7],
      [7, 7],
      [4, 1],
    ];
    expect(() =>
      contourToSurface([0, 2].flatMap((z) => [polygonAt(u, z), polygonAt(crossing, z)])),
    ).toThrowError(/intersecting or touching contours on the same slice/);
  });

  it('rejects boundary contact instead of assigning a strict nesting depth', () => {
    const outer = [
      [0, 0],
      [8, 0],
      [8, 8],
      [0, 8],
    ];
    const touching = [
      [0, 2],
      [2, 2],
      [2, 4],
      [0, 4],
    ];
    expect(() =>
      contourToSurface([0, 2].flatMap((z) => [polygonAt(outer, z), polygonAt(touching, z)])),
    ).toThrowError(/intersecting or touching contours on the same slice/);
  });

  it.each(['none', 'smooth', 'straight'] as const)(
    'rejects nested splits and merges before stitching with %s caps',
    (endCapping) => {
      const split = [
        diamond(20, 0),
        diamond(10, 0),
        diamond(20, 2),
        diamond(3, 2, -5),
        diamond(3, 2, 5),
      ];
      const merge = split.map((loop) =>
        loop.map((value, index) => (index % 3 === 2 ? 2 - value : value)),
      );
      for (const loops of [split, merge]) {
        expect(() => contourToSurface(loops, { endCapping })).toThrowError(
          new RangeError(
            'Contour to surface does not support nested-ring split/merge correspondence: a nested ring has multiple eligible partners on an adjacent slice',
          ),
        );
      }
    },
  );

  it.each(['smooth', 'straight'] as const)(
    'caps unmatched new and disappearing holes half a slice away with %s caps',
    (endCapping) => {
      const birth = [diamond(12, 0), diamond(12, 2), diamond(3, 2)];
      const death = birth.map((loop) =>
        loop.map((value, index) => (index % 3 === 2 ? 2 - value : value)),
      );
      for (const [loops, expected] of [
        [birth, [1, 1, 0, 0]],
        [death, [0, 0, 1, 1]],
      ] as const) {
        const mesh = contourToSurface(loops, { endCapping });
        expect(isWatertight(mesh)).toBe(true);
        expect(isManifold(mesh)).toBe(true);
        expect([0.25, 0.75, 1.25, 1.75].map((z) => cutLabel(mesh, z))).toEqual(expected);
        expect([0.25, 0.75, 1.25, 1.75].map((z) => cutAreas(mesh, z).length)).toEqual(
          expected.map((label) => (label === 0 ? 2 : 1)),
        );
      }
    },
  );

  it('preserves the keyhole fixture when input slice order is reversed', () => {
    const mesh = contourToSurface([...contourToSurfaceCases.keyhole.loops].reverse());
    expect(cutAreas(mesh, 2.25)).toHaveLength(2);
    expect(cutLabel(mesh, 2.25)).toBe(0);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
  });
});

describe('disjoint nested rings and irregular slice gaps', () => {
  const outer = [
    [-1, -1],
    [11, -1],
    [-1, 11],
  ];
  const firstHole = [
    [0, 0],
    [4, 0],
    [0, 4],
  ];
  const secondHole = [
    [4, 4],
    [1, 4],
    [4, 1],
  ];

  it('retains stationary disjoint holes whose bounding boxes overlap', () => {
    const loops = [0, 2].flatMap((z) =>
      [outer, firstHole, secondHole].map((xy) => polygonAt(xy, z)),
    );
    const mesh = contourToSurface(loops, { endCapping: 'straight' });
    expect(cutAreas(mesh, 1)).toEqual([72, 8, 4.5]);
    expect(cutLabel(mesh, 1, 1, 1)).toBe(0);
    expect(cutLabel(mesh, 1, 3, 3)).toBe(0);
    expect(cutLabel(mesh, 1, 5, 0)).toBe(1);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
  });

  it('caps a new disjoint hole without treating its overlapping bounds as a split', () => {
    const loops = [
      polygonAt(outer, 0),
      polygonAt(firstHole, 0),
      polygonAt(outer, 2),
      polygonAt(firstHole, 2),
      polygonAt(secondHole, 2),
    ];
    const mesh = contourToSurface(loops, { endCapping: 'straight' });
    expect(cutAreas(mesh, 0.5)).toEqual([72, 8]);
    expect(cutAreas(mesh, 1.5)).toEqual([72, 8, 4.5]);
    expect(cutLabel(mesh, 0.5, 3, 3)).toBe(1);
    expect(cutLabel(mesh, 1.5, 3, 3)).toBe(0);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
  });

  it.each([21, 1001])('keeps distinct planes separate across a gap to z=%s', (lastSlice) => {
    const ring = [
      [0, 0],
      [8, 0],
      [0, 8],
    ];
    const mesh = contourToSurface(
      [0, 1, lastSlice].map((z) => polygonAt(ring, z)),
      {
        endCapping: 'none',
      },
    );
    expect(cutAreas(mesh, 0.5)).toEqual([32]);
    expect(cutAreas(mesh, (1 + lastSlice) / 2)).toEqual([32]);
    expect(cutLabel(mesh, 0.5, 2, 2)).toBe(1);
    expect(isManifold(mesh)).toBe(true);
  });
});

describe('moving nested rings and numerical plane grouping', () => {
  const outer = [
    [-5, -5],
    [5, -5],
    [5, 5],
    [-5, 5],
  ];
  const movingDiamond = (radius: number, z: number) => {
    const xy =
      z === 0
        ? [
            [-radius, 0],
            [0, -radius],
            [radius, 0],
            [0, radius],
          ]
        : [
            [radius + 0.75, 1.5],
            [0.75, radius + 1.5],
            [0.75 - radius, 1.5],
            [0.75, 1.5 - radius],
          ];
    return polygonAt(xy, z);
  };

  it.each(['none', 'smooth', 'straight'] as const)(
    'retains an isolated translating hole with disjoint projected interiors and %s caps',
    (endCapping) => {
      const loops = [
        polygonAt(outer, 0),
        movingDiamond(1, 0),
        polygonAt(outer, 2),
        movingDiamond(1, 2),
      ];
      for (const ordered of [loops, [...loops].reverse()]) {
        const mesh = contourToSurface(ordered, { endCapping });
        expect(cutAreas(mesh, 1)).toEqual([100, 2]);
        expect(cutLabel(mesh, 1, 0.375, 0.75)).toBe(0);
        expect(isManifold(mesh)).toBe(true);
        if (endCapping !== 'none') expect(isWatertight(mesh)).toBe(true);
      }
    },
  );

  it.each(['none', 'smooth', 'straight'] as const)(
    'retains a translating annulus with a unique reciprocal root match and %s caps',
    (endCapping) => {
      // These cyclic starts give the existing stitcher a valid translated
      // surface. The outer and inner projections are both disjoint, although
      // each pair of bounds overlaps.
      const ring = (radius: number, z: number) =>
        movingDiamond(radius / 8, z).map((value, index) => (index % 3 === 2 ? value : value * 8));
      const loops = [ring(8, 0), ring(7, 0), ring(8, 2), ring(7, 2)];
      for (const ordered of [loops, [...loops].reverse()]) {
        const mesh = contourToSurface(ordered, { endCapping });
        expect(cutAreas(mesh, 1)).toEqual([128, 98]);
        expect(cutLabel(mesh, 1, 3, 6)).toBe(0);
        expect(cutLabel(mesh, 1, 10.5, 6)).toBe(1);
        expect(isManifold(mesh)).toBe(true);
        if (endCapping !== 'none') expect(isWatertight(mesh)).toBe(true);
      }
    },
  );

  it('keeps an island with its retained translating parent', () => {
    const loops = [0, 2].flatMap((z) => [
      polygonAt(outer, z),
      movingDiamond(1, z),
      movingDiamond(0.875, z),
    ]);
    const mesh = contourToSurface(loops, { endCapping: 'none' });
    expect(cutAreas(mesh, 1)).toEqual([100, 2, 1.53125]);
    expect(cutLabel(mesh, 1, 0.375, 0.75)).toBe(1);
    expect(isManifold(mesh)).toBe(true);
  });

  it.each([1000, 10000])(
    'groups separated coplanar oblique roots at offset %s independently of input order',
    (offset) => {
      const a = Math.sin(0.7);
      const b = Math.cos(0.7);
      const loops = [0, 2].flatMap((z) =>
        [0, offset].flatMap((center) =>
          [8, 3].map((radius) => {
            const xy = [
              [center + radius, center],
              [center, center + radius],
              [center - radius, center],
              [center, center - radius],
            ];
            return xy.flatMap(([x, y]) => [x, y, a * x + b * y + z]);
          }),
        ),
      );
      const xLength = Math.hypot(1, a);
      const xAxis = [1 / xLength, 0, a / xLength] as const;
      const normalLength = Math.hypot(a, b, 1);
      const normal = [-a / normalLength, -b / normalLength, 1 / normalLength];
      const yAxis = [
        normal[1] * xAxis[2],
        normal[2] * xAxis[0] - normal[0] * xAxis[2],
        -normal[1] * xAxis[0],
      ] as const;
      const plane = createContourPlane([0, 0, 1], xAxis, yAxis);
      for (const ordered of [
        loops,
        [...loops].reverse(),
        [loops[3], loops[0], loops[6], loops[1], loops[7], loops[4], loops[2], loops[5]],
      ]) {
        const mesh = contourToSurface(ordered, { endCapping: 'none' });
        const cut = surfaceToContour(mesh, plane);
        expect(cut?.loops).toHaveLength(4);
        const areas = cut!.loops
          .map((loop) => Math.abs(loopArea(loop)))
          .sort((left, right) => right - left);
        for (const [index, expected] of [128, 128, 18, 18].entries()) {
          expect(areas[index]).toBeCloseTo(expected * normalLength, 1);
        }
        expect(isManifold(mesh)).toBe(true);
      }
    },
  );

  it('uses resolved tiny plane gaps for both grouping and cap thickness', () => {
    const gap = 2 ** -12;
    const mesh = contourToSurface([diamond(8, 0), diamond(8, gap)], {
      endCapping: 'straight',
    });
    expect(boundingBox(mesh)).toEqual({ min: [-8, -8, -gap / 2], max: [8, 8, gap * 1.5] });
    for (const z of [-gap / 4, gap / 2, gap * 1.25]) {
      expect(cutAreas(mesh, z)).toEqual([128]);
    }
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
  });

  it('does not merge adjacent representable axial planes far from zero', () => {
    const z = 2 ** 24;
    const mesh = contourToSurface([diamond(8, z), diamond(8, z + 2)], { endCapping: 'none' });
    expect(mesh.polys.length / 4).toBe(8);
    expect(isManifold(mesh)).toBe(true);
  });

  it('groups a single plane even when default slice thickness is zero', () => {
    expect(() =>
      contourToSurface([diamond(8, 0), diamond(8, 0)], { endCapping: 'none' }),
    ).toThrowError(/intersecting or touching contours on the same slice/);
  });
});

describe('ambiguous nested correspondence', () => {
  it('rejects competing bbox-only partners without guessing a split or merge', () => {
    const root = [
      [-1, -1],
      [11, -1],
      [-1, 11],
    ];
    const original = [
      [0, 0],
      [4, 0],
      [0, 4],
    ];
    const first = [
      [2.5, 2.5],
      [3, 2.5],
      [3, 3],
      [2.5, 3],
    ];
    const second = [
      [3.25, 1.5],
      [3.75, 1.5],
      [3.75, 2],
      [3.25, 2],
    ];
    const split = [
      polygonAt(root, 0),
      polygonAt(original, 0),
      polygonAt(root, 2),
      polygonAt(first, 2),
      polygonAt(second, 2),
    ];
    const merge = split.map((loop) =>
      loop.map((value, index) => (index % 3 === 2 ? 2 - value : value)),
    );
    for (const loops of [split, merge]) {
      expect(() => contourToSurface(loops, { endCapping: 'none' })).toThrowError(
        /nested-ring split\/merge/,
      );
    }
  });
});

describe('resolved Float32 plane boundaries', () => {
  it.each([
    [1 - 2 ** -24, 1],
    [16777215, 16777216],
    [-16777216, -16777215],
  ])('keeps planes at %s and %s separate across an asymmetric bin boundary', (first, second) => {
    const loops = [first, second].map((z) => diamond(8, z));
    const mesh = contourToSurface(loops, { endCapping: 'none' });
    expect(triangleCount(mesh)).toBe(8);
    const planeCoordinates = new Set([...mesh.points].filter((_, index) => index % 3 === 2));
    expect([...planeCoordinates].sort((a, b) => a - b)).toEqual([first, second]);
    expect(isManifold(mesh)).toBe(true);
  });
});
