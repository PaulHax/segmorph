import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { labelmapToSurface } from '../src/convert/labelmapToSurface.js';
import type { Mesh } from '../src/geometry/mesh.js';
import { createOrientedImage, type OrientedImage } from '../src/image/orientedImage.js';
import { labelmapToSurfaceBase } from './baselines/labelmapToSurfaceBase.js';

// Exact-output regression net for performance work on labelmapToSurface.
// Every case asserts deep equality (same values, same order) against the
// frozen base implementation in test/baselines/labelmapToSurfaceBase.ts, and
// the small named cases additionally against committed JSON snapshots under
// test/fixtures/A/exact/. Regenerate snapshots with:
//   UPDATE_EXACT=1 npx vitest run test/labelmap-to-surface-exact.spec.ts
// (only legitimate when the observable output is intentionally changed, which
// requires coordinator approval).

const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
] as const;

const mulberry32 = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

// Rodrigues rotation about a unit axis; orthonormal to float64 round-off,
// well within validateImageGeometry's 1e-10 tolerance.
function rotationMatrix(axis: readonly [number, number, number], radians: number) {
  const [x, y, z] = axis;
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  const t = 1 - c;
  return [
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
  ] as const;
}

function fillEllipsoid(
  dims: readonly [number, number, number],
  center: readonly [number, number, number],
  radii: readonly [number, number, number],
  labelValue: number,
) {
  const data = new Uint8Array(dims[0] * dims[1] * dims[2]);
  let offset = 0;
  for (let z = 0; z < dims[2]; z += 1) {
    for (let y = 0; y < dims[1]; y += 1) {
      for (let x = 0; x < dims[0]; x += 1, offset += 1) {
        const dx = (x - center[0]) / radii[0];
        const dy = (y - center[1]) / radii[1];
        const dz = (z - center[2]) / radii[2];
        if (dx * dx + dy * dy + dz * dz <= 1) data[offset] = labelValue;
      }
    }
  }
  return data;
}

function multiLabelImage() {
  const dims = [8, 7, 6] as const;
  const data = new Uint8Array(dims[0] * dims[1] * dims[2]);
  const set = (x: number, y: number, z: number, label: number) => {
    data[x + dims[0] * (y + dims[1] * z)] = label;
  };
  for (let z = 1; z <= 3; z += 1) {
    for (let y = 1; y <= 3; y += 1) {
      for (let x = 1; x <= 3; x += 1) set(x, y, z, 1);
    }
  }
  for (let z = 1; z <= 4; z += 1) {
    for (let y = 2; y <= 4; y += 1) {
      for (let x = 3; x <= 5; x += 1) set(x, y, z, 2);
    }
  }
  set(7, 6, 5, 3);
  set(0, 6, 0, 3);
  return createOrientedImage({
    data,
    dims,
    spacing: [1, 1, 1],
    origin: [0, 0, 0],
    direction: identity,
  });
}

function boundarySlabImage() {
  const dims = [5, 4, 3] as const;
  const data = new Uint8Array(dims[0] * dims[1] * dims[2]);
  for (let z = 0; z < dims[2]; z += 1) {
    for (let x = 0; x < dims[0]; x += 1) data[x + dims[0] * dims[1] * z] = 5;
  }
  return createOrientedImage({
    data,
    dims,
    spacing: [1, 1, 1],
    origin: [0, 0, 0],
    direction: identity,
  });
}

function randomNoiseImage() {
  const dims = [20, 19, 18] as const;
  const random = mulberry32(42);
  const data = Uint8Array.from({ length: dims[0] * dims[1] * dims[2] }, () =>
    Math.floor(random() * 4),
  );
  return createOrientedImage({
    data,
    dims,
    spacing: [1, 1, 1],
    origin: [0, 0, 0],
    direction: identity,
  });
}

const oblique = rotationMatrix([Math.SQRT1_2, Math.SQRT1_2, 0], Math.PI / 6);
// Negative-determinant direction to exercise the reversed-winding branch.
const obliqueFlipped = oblique.map((row) => [-row[0], row[1], row[2]] as const);

const namedCases: readonly {
  name: string;
  image: OrientedImage;
  labelValue: number;
  snapshot: boolean;
}[] = [
  {
    name: 'empty',
    image: createOrientedImage({
      data: new Uint8Array(24),
      dims: [4, 3, 2],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    }),
    labelValue: 1,
    snapshot: true,
  },
  { name: 'multi-label-1', image: multiLabelImage(), labelValue: 1, snapshot: true },
  { name: 'multi-label-2', image: multiLabelImage(), labelValue: 2, snapshot: true },
  { name: 'multi-label-3', image: multiLabelImage(), labelValue: 3, snapshot: true },
  { name: 'boundary-slab', image: boundarySlabImage(), labelValue: 5, snapshot: true },
  {
    name: 'anisotropic-ellipsoid',
    image: createOrientedImage({
      data: fillEllipsoid([12, 10, 8], [5.5, 4.5, 3.5], [4.5, 3.5, 2.5], 4),
      dims: [12, 10, 8],
      spacing: [0.5, 1.25, 3],
      origin: [-4, 2.5, 10],
      direction: identity,
    }),
    labelValue: 4,
    snapshot: true,
  },
  {
    name: 'oblique-blob',
    image: createOrientedImage({
      data: fillEllipsoid([10, 10, 10], [4.5, 4.5, 4.5], [3.5, 3.5, 3.5], 2),
      dims: [10, 10, 10],
      spacing: [1, 0.75, 1.5],
      origin: [5, -2, 1],
      direction: oblique,
    }),
    labelValue: 2,
    snapshot: true,
  },
  {
    name: 'oblique-flipped-blob',
    image: createOrientedImage({
      data: fillEllipsoid([10, 10, 10], [4.5, 4.5, 4.5], [3.5, 3.5, 3.5], 2),
      dims: [10, 10, 10],
      spacing: [1, 0.75, 1.5],
      origin: [5, -2, 1],
      direction: obliqueFlipped,
    }),
    labelValue: 2,
    snapshot: true,
  },
  { name: 'random-noise-1', image: randomNoiseImage(), labelValue: 1, snapshot: false },
  { name: 'random-noise-3', image: randomNoiseImage(), labelValue: 3, snapshot: false },
  {
    name: 'sparse-blob',
    image: createOrientedImage({
      data: fillEllipsoid([64, 64, 64], [50, 12, 33], [4.5, 4.5, 4.5], 9),
      dims: [64, 64, 64],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    }),
    labelValue: 9,
    snapshot: false,
  },
];

const snapshotDir = new URL('./fixtures/A/exact/', import.meta.url);

function checkSnapshot(name: string, mesh: Mesh) {
  const path = fileURLToPath(new URL(`${name}.mesh.json`, snapshotDir));
  const serialized = { points: Array.from(mesh.points), polys: Array.from(mesh.polys) };
  if (process.env.UPDATE_EXACT) {
    mkdirSync(fileURLToPath(snapshotDir), { recursive: true });
    writeFileSync(path, `${JSON.stringify(serialized)}\n`);
  }
  expect(existsSync(path), `missing snapshot ${path}`).toBe(true);
  const golden = JSON.parse(readFileSync(path, 'utf8')) as typeof serialized;
  expect(serialized.points).toEqual(golden.points);
  expect(serialized.polys).toEqual(golden.polys);
}

describe('labelmapToSurface exact-output regression', () => {
  it.each(namedCases.map((entry) => [entry.name, entry] as const))(
    'matches the frozen base implementation for %s',
    (_name, { name, image, labelValue, snapshot }) => {
      const mesh = labelmapToSurface(image, { labelValue });
      const base = labelmapToSurfaceBase(image, { labelValue });
      expect(mesh.points).toEqual(base.points);
      expect(mesh.polys).toEqual(base.polys);
      if (snapshot) checkSnapshot(name, mesh);
    },
  );

  // The same 32-cubed radius-10 sphere the A oracle consumes (gen_surface.py),
  // built procedurally so the fast tier needs no oracle-generated input.
  it('matches the frozen base implementation for the oracle sphere', () => {
    const image = createOrientedImage({
      data: fillEllipsoid([32, 32, 32], [15.5, 15.5, 15.5], [10, 10, 10], 1),
      dims: [32, 32, 32],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    });
    const mesh = labelmapToSurface(image, { labelValue: 1 });
    const base = labelmapToSurfaceBase(image, { labelValue: 1 });
    expect(mesh.points).toEqual(base.points);
    expect(mesh.polys).toEqual(base.polys);
    expect(mesh.points.length).toBeGreaterThan(0);
  });

  it('rejects non-integer and out-of-range label values unchanged', () => {
    const image = createOrientedImage({
      data: new Uint8Array(1),
      dims: [1, 1, 1],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    });
    for (const labelValue of [0, -1, 1.5, 2 ** 32, Number.NaN]) {
      expect(() => labelmapToSurface(image, { labelValue })).toThrow(
        'labelValue must be an integer between 1 and 4294967295',
      );
    }
  });
});
