import { describe, expect, it } from 'vitest';

import {
  createOrientedImage,
  indexToWorld,
  validateImageGeometry,
  worldToIndex,
} from '../src/index.js';

const obliqueGeometry = {
  dims: [2, 3, 4] as const,
  spacing: [2, 3, 5] as const,
  origin: [10, -4, 7] as const,
  direction: [
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
  ] as const,
};

describe('oriented images', () => {
  it('creates a validated image without copying its typed-array data', () => {
    const data = new Uint16Array(24);
    const image = createOrientedImage({ ...obliqueGeometry, data });

    expect(image.data).toBe(data);
    expect(image.dims).toEqual([2, 3, 4]);
  });

  it('maps continuous indices to world coordinates using direction, spacing, and origin', () => {
    expect(indexToWorld(obliqueGeometry, [1, 2, 3])).toEqual([4, -2, 22]);
    expect(indexToWorld(obliqueGeometry, [0.5, -1, 0.25])).toEqual([13, -3, 8.25]);
  });

  it('maps world coordinates back to continuous indices', () => {
    const index = [0.25, 1.5, -2] as const;
    const world = indexToWorld(obliqueGeometry, index);

    expect(worldToIndex(obliqueGeometry, world)).toEqual(index);
  });

  it('validates dimensions, spacing, finite values, and direction matrices', () => {
    expect(() => validateImageGeometry({ ...obliqueGeometry, dims: [2, 0, 4] })).toThrow(
      'dims must contain three positive integers',
    );
    expect(() => validateImageGeometry({ ...obliqueGeometry, spacing: [2, -3, 5] })).toThrow(
      'spacing must contain three positive finite numbers',
    );
    expect(() => validateImageGeometry({ ...obliqueGeometry, origin: [0, NaN, 0] })).toThrow(
      'origin must contain three finite numbers',
    );
    expect(() =>
      validateImageGeometry({
        ...obliqueGeometry,
        direction: [
          [1, 0, 0],
          [1, 0, 0],
          [0, 0, 1],
        ],
      }),
    ).toThrow('direction must be an orthonormal 3x3 matrix');
  });

  it('rejects image data whose length does not match its dimensions', () => {
    expect(() =>
      createOrientedImage({
        ...obliqueGeometry,
        data: new Uint8Array(23),
      }),
    ).toThrow('Image data length 23 does not match dimensions (expected 24)');
  });
});
