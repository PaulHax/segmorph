import { describe, expect, it } from 'vitest';

import { resampleNearest } from '../src/image/resample.js';
import type { ImageGeometry } from '../src/image/orientedImage.js';

const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
] as const;

const geometry = (overrides: Partial<ImageGeometry> = {}): ImageGeometry => ({
  dims: [2, 2, 1],
  spacing: [1, 1, 1],
  origin: [0, 0, 0],
  direction: identity,
  ...overrides,
});

describe('nearest-neighbor oriented-image resampling', () => {
  it('copies an image onto identical geometry without aliasing its data', () => {
    const input = { ...geometry(), data: new Uint8Array([1, 2, 3, 4]) };
    const output = resampleNearest(input, geometry());

    expect(output.data).toEqual(input.data);
    expect(output.data).not.toBe(input.data);
  });

  it('maps output voxel centers through world coordinates', () => {
    const input = { ...geometry({ dims: [3, 1, 1] }), data: new Uint8Array([10, 20, 30]) };
    const output = resampleNearest(input, geometry({ dims: [3, 1, 1], origin: [1, 0, 0] }));

    expect(output.data).toEqual(new Uint8Array([20, 30, 0]));
  });

  it('handles an oblique direction matrix', () => {
    const input = {
      ...geometry({ dims: [2, 1, 1] }),
      data: new Uint8Array([7, 9]),
    };
    const output = resampleNearest(
      input,
      geometry({
        dims: [1, 2, 1],
        direction: [
          [0, 1, 0],
          [-1, 0, 0],
          [0, 0, 1],
        ],
      }),
    );

    expect(output.data).toEqual(new Uint8Array([7, 9]));
  });

  it('uses the requested fill value outside the input geometry', () => {
    const input = { ...geometry({ dims: [1, 1, 1] }), data: new Int16Array([5]) };
    const output = resampleNearest(input, geometry({ dims: [2, 1, 1], origin: [-1, 0, 0] }), {
      fillValue: -2,
    });

    expect(output.data).toBeInstanceOf(Int16Array);
    expect(output.data).toEqual(new Int16Array([-2, 5]));
  });
});
