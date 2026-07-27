import { describe, expect, it } from 'vitest';

import { labelmapToSurface } from '../src/convert/labelmapToSurface.js';
import { surfaceToLabelmap } from '../src/convert/surfaceToLabelmap.js';
import type { ImageGeometry } from '../src/image/orientedImage.js';
import { dice, mismatchCount } from './diff/image.js';

const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
] as const;

const quarterTurn = [
  [0, -1, 0],
  [1, 0, 0],
  [0, 0, 1],
] as const;

// Rodrigues rotation of `angleDeg` degrees about the (non-axis) unit vector `axis`,
// giving a genuinely tilted orthonormal direction cosine matrix.
function rotationDirection(angleDeg: number, axis: readonly [number, number, number]) {
  const norm = Math.hypot(...axis);
  const [kx, ky, kz] = axis.map((value) => value / norm);
  const angle = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const t = 1 - cos;
  return [
    [cos + kx * kx * t, kx * ky * t - kz * sin, kx * kz * t + ky * sin],
    [ky * kx * t + kz * sin, cos + ky * ky * t, ky * kz * t - kx * sin],
    [kz * kx * t - ky * sin, kz * ky * t + kx * sin, cos + kz * kz * t],
  ] as const;
}

// ~30 degrees about (1, 1, 1): off every coordinate axis, exercising a fully oblique frame.
const tilted = rotationDirection(30, [1, 1, 1]);

function flatIndex(x: number, y: number, z: number, dims: readonly number[]) {
  return x + dims[0] * (y + dims[1] * z);
}

function labelmap(geometry: ImageGeometry, includes: (x: number, y: number, z: number) => boolean) {
  const data = new Uint8Array(geometry.dims[0] * geometry.dims[1] * geometry.dims[2]);
  for (let z = 0; z < geometry.dims[2]; z += 1) {
    for (let y = 0; y < geometry.dims[1]; y += 1) {
      for (let x = 0; x < geometry.dims[0]; x += 1) {
        if (includes(x, y, z)) {
          data[flatIndex(x, y, z, geometry.dims)] = 3;
        }
      }
    }
  }
  return { ...geometry, data };
}

function expectGeometry(actual: ImageGeometry, expected: ImageGeometry) {
  expect(actual.dims).toEqual(expected.dims);
  expect(actual.spacing).toEqual(expected.spacing);
  expect(actual.origin).toEqual(expected.origin);
  expect(actual.direction).toEqual(expected.direction);
}

describe('labelmap -> surface -> labelmap acceptance', () => {
  it.each([
    {
      name: 'interior cube with anisotropic spacing',
      geometry: {
        dims: [7, 7, 7],
        spacing: [0.7, 1.4, 2.5],
        origin: [-4, 8, 12],
        direction: identity,
      },
      includes: (x: number, y: number, z: number) =>
        x >= 2 && x <= 4 && y >= 2 && y <= 4 && z >= 2 && z <= 4,
    },
    {
      name: 'concave L-shape in an oblique geometry',
      geometry: {
        dims: [8, 8, 5],
        spacing: [0.8, 1.2, 2.2],
        origin: [17, -9, 3],
        direction: quarterTurn,
      },
      includes: (x: number, y: number, z: number) =>
        z >= 1 &&
        z <= 3 &&
        ((x >= 1 && x <= 2 && y >= 1 && y <= 6) || (x >= 1 && x <= 5 && y >= 1 && y <= 2)),
    },
    {
      name: 'interior cube in a fully tilted anisotropic geometry',
      geometry: {
        dims: [9, 9, 9],
        spacing: [0.6, 1.3, 2.1],
        origin: [3, -5, 7],
        direction: tilted,
      },
      includes: (x: number, y: number, z: number) =>
        x >= 2 && x <= 6 && y >= 2 && y <= 6 && z >= 2 && z <= 6,
    },
  ])('$name', ({ geometry, includes }) => {
    const input = labelmap(geometry, includes);

    const surface = labelmapToSurface(input, { labelValue: 3 });
    const output = surfaceToLabelmap(surface, geometry, { labelValue: 3 });

    expectGeometry(output, geometry);
    // Measured: the round trip is voxel-exact for every case above (mismatchCount 0,
    // Dice 1), so assert exactness rather than a padded fraction.
    expect(mismatchCount(output.data, input.data, geometry.dims as [number, number, number])).toBe(
      0,
    );
    expect(dice(output.data, input.data, geometry.dims as [number, number, number])).toBe(1);
  });
});
