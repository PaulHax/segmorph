import { describe, expect, it } from 'vitest';

import { dice, iou, mismatchCount, mismatchingVoxelCoordinates } from './image.js';

const dims = [4, 4, 1] as const;

describe('image metrics', () => {
  it('computes overlap metrics for hand-countable masks', () => {
    const actual = new Uint8Array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const expected = new Uint8Array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    expect(dice(actual, expected, dims)).toBe(0.5);
    expect(iou(actual, expected, dims)).toBe(1 / 3);
    expect(mismatchCount(actual, expected, dims)).toBe(4);
    expect(mismatchingVoxelCoordinates(actual, expected, dims)).toEqual([
      [0, 0, 0],
      [2, 0, 0],
      [0, 1, 0],
      [2, 1, 0],
    ]);
  });

  it('returns perfect overlap for identical masks', () => {
    const mask = new Uint8Array([0, 1, 0, 1]);
    const smallDims = [2, 2, 1] as const;

    expect(dice(mask, mask, smallDims)).toBe(1);
    expect(iou(mask, mask, smallDims)).toBe(1);
  });

  it('flags empty-vs-empty as NaN rather than a perfect match', () => {
    const empty = new Uint8Array(4);
    const smallDims = [2, 2, 1] as const;

    expect(dice(empty, empty, smallDims)).toBeNaN();
    expect(iou(empty, empty, smallDims)).toBeNaN();
  });

  it('scores an empty output against a non-empty expectation as zero', () => {
    const empty = new Uint8Array(4);
    const expected = new Uint8Array([1, 1, 0, 0]);
    const smallDims = [2, 2, 1] as const;

    expect(dice(empty, expected, smallDims)).toBe(0);
    expect(iou(empty, expected, smallDims)).toBe(0);
  });

  it('scores a label swap as zero overlap, not a perfect match', () => {
    const actual = new Uint8Array([1, 1, 0, 0]);
    const swapped = new Uint8Array([2, 2, 0, 0]);
    const smallDims = [2, 2, 1] as const;

    expect(dice(actual, swapped, smallDims)).toBe(0);
    expect(iou(actual, swapped, smallDims)).toBe(0);
  });

  it('scores partial label agreement by matching label only', () => {
    // Two voxels agree (label 1), one disagrees (label 2 vs 3), one is empty.
    const actual = new Uint8Array([1, 1, 2, 0]);
    const expected = new Uint8Array([1, 1, 3, 0]);
    const smallDims = [2, 2, 1] as const;

    // intersection 2, foregrounds 3 and 3.
    expect(dice(actual, expected, smallDims)).toBe(2 / 3);
    expect(iou(actual, expected, smallDims)).toBe(2 / 4);
  });

  it('returns zero overlap for disjoint masks', () => {
    const actual = new Uint8Array([1, 0, 0, 0]);
    const expected = new Uint8Array([0, 0, 0, 1]);
    const smallDims = [2, 2, 1] as const;

    expect(dice(actual, expected, smallDims)).toBe(0);
    expect(iou(actual, expected, smallDims)).toBe(0);
  });

  it('rejects arrays that do not match the dimensions', () => {
    expect(() => dice(new Uint8Array(3), new Uint8Array(4), [2, 2, 1])).toThrow(
      'Image data length must equal the dimensions product',
    );
  });
});
