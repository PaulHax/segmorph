import { describe, expect, it } from 'vitest';

import {
  compositeMax,
  compositeMin,
  compositeSet,
  maskByLabelValue,
} from '../src/image/composite.js';
import type { ImageData } from '../src/image/orientedImage.js';

const dims = [3, 2, 2] as const;
const middleColumn = [1, 2, 0, 1, 0, 1] as const;

describe('bounded image compositing', () => {
  it('sets only voxels inside an inclusive extent and leaves both inputs unchanged', () => {
    const input = new Uint8Array(12).fill(1);
    const modifier = Uint8Array.from({ length: 12 }, (_, index) => index + 10);
    const originalInput = input.slice();
    const originalModifier = modifier.slice();

    const output = compositeSet(input, modifier, dims, middleColumn);

    expect(output).toBeInstanceOf(Uint8Array);
    expect([...output]).toEqual([1, 11, 12, 1, 14, 15, 1, 17, 18, 1, 20, 21]);
    expect(input).toEqual(originalInput);
    expect(modifier).toEqual(originalModifier);
  });

  it('takes the minimum and maximum only inside the extent', () => {
    const input = Uint16Array.from([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]);
    const modifier = Uint16Array.from([1, 9, 4, 7, 3, 10, 2, 12, 6, 5, 11, 0]);

    expect([...compositeMin(input, modifier, dims, middleColumn)])
      .toEqual([8, 8, 4, 8, 3, 8, 8, 8, 6, 8, 8, 0]);
    expect([...compositeMax(input, modifier, dims, middleColumn)])
      .toEqual([8, 9, 8, 8, 8, 10, 8, 12, 8, 8, 11, 8]);
  });

  it('keeps input voxels selected by a label and fills other voxels in the extent', () => {
    const input = Int16Array.from({ length: 12 }, (_, index) => index - 2);
    const labels = Uint8Array.from([0, 4, 4, 4, 2, 4, 4, 0, 4, 4, 4, 3]);

    expect([...maskByLabelValue(input, labels, dims, 4, -9, middleColumn)])
      .toEqual([-2, -1, 0, 1, -9, 3, 4, -9, 6, 7, 8, -9]);
    expect([...input]).toEqual([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('defaults to the complete image extent', () => {
    expect([...compositeSet(new Uint8Array(12), new Uint8Array(12).fill(5), dims)])
      .toEqual(new Array(12).fill(5));
  });

  it('validates dimensions, array sizes, extents, and compatible data types', () => {
    const input = new Uint8Array(12);
    const modifier = new Uint8Array(12);

    expect(() => compositeSet(input, modifier, [3, 0, 2])).toThrow(
      'dims must contain three positive integers',
    );
    expect(() => compositeSet(input.subarray(1), modifier, dims)).toThrow(
      'Input data length 11 does not match dimensions (expected 12)',
    );
    expect(() => compositeSet(input, modifier.subarray(1), dims)).toThrow(
      'Modifier data length 11 does not match dimensions (expected 12)',
    );
    expect(() => compositeSet(input, modifier, dims, [0, 3, 0, 1, 0, 1])).toThrow(
      'extent must be within image dimensions',
    );
    expect(() => compositeSet(input, modifier, dims, [2, 1, 0, 1, 0, 1])).toThrow(
      'extent bounds must be ordered',
    );
    expect(() => compositeSet<ImageData>(input, new Uint16Array(12), dims)).toThrow(
      'Input and modifier data types must match',
    );
  });
});
