import { describe, expect, it } from 'vitest';

import {
  EMPTY_EXTENT,
  extentFromDims,
  flatIndex,
  intersectExtents,
  isExtentEmpty,
  isIndexInExtent,
  iterateExtentIndices,
  validateExtent,
} from '../src/image/extent.js';

describe('image extents', () => {
  it('creates inclusive image bounds from dimensions', () => {
    expect(extentFromDims([4, 3, 2])).toEqual([0, 3, 0, 2, 0, 1]);
  });

  it('validates extents against image dimensions', () => {
    expect(validateExtent([1, 3, 0, 2, 1, 1], [4, 3, 2])).toEqual([1, 3, 0, 2, 1, 1]);
    expect(validateExtent(EMPTY_EXTENT, [4, 3, 2])).toBe(EMPTY_EXTENT);

    expect(() => validateExtent([0, 4, 0, 2, 0, 1], [4, 3, 2]))
      .toThrow('extent must be within image dimensions');
    expect(() => validateExtent([0, 2.5, 0, 2, 0, 1], [4, 3, 2]))
      .toThrow('extent values must be integers');
    expect(() => validateExtent([2, 1, 0, 2, 0, 1], [4, 3, 2]))
      .toThrow('extent bounds must be ordered');
    expect(() => validateExtent([0, 1, 0, 1, 0, 1], [2, 0, 2]))
      .toThrow('dims must contain three positive integers');
  });

  it('intersects inclusive bounds and canonicalizes empty results', () => {
    expect(intersectExtents([0, 3, 0, 3, 0, 3], [2, 4, 1, 2, -1, 1]))
      .toEqual([2, 3, 1, 2, 0, 1]);
    expect(intersectExtents([0, 1, 0, 1, 0, 1], [2, 3, 0, 1, 0, 1]))
      .toBe(EMPTY_EXTENT);
    expect(isExtentEmpty(EMPTY_EXTENT)).toBe(true);
    expect(isExtentEmpty([0, 0, 0, 0, 0, 0])).toBe(false);
  });

  it('checks whether integer indices are in inclusive bounds', () => {
    const extent = [1, 2, 3, 4, 5, 6] as const;
    expect(isIndexInExtent([1, 4, 6], extent)).toBe(true);
    expect(isIndexInExtent([0, 4, 6], extent)).toBe(false);
    expect(isIndexInExtent([1.5, 4, 6], extent)).toBe(false);
    expect(isIndexInExtent([0, 0, 0], EMPTY_EXTENT)).toBe(false);
  });

  it('computes x-fastest flat indices and rejects invalid indices', () => {
    expect(flatIndex([0, 0, 0], [4, 3, 2])).toBe(0);
    expect(flatIndex([3, 2, 1], [4, 3, 2])).toBe(23);
    expect(flatIndex([1, 2, 1], [4, 3, 2])).toBe(21);
    expect(() => flatIndex([4, 0, 0], [4, 3, 2])).toThrow('index must be within image dimensions');
    expect(() => flatIndex([1.5, 0, 0], [4, 3, 2])).toThrow('index values must be integers');
  });

  it('iterates deterministically with x varying fastest', () => {
    expect([...iterateExtentIndices([1, 2, 3, 4, 5, 5])]).toEqual([
      [1, 3, 5], [2, 3, 5], [1, 4, 5], [2, 4, 5],
    ]);
    expect([...iterateExtentIndices(EMPTY_EXTENT)]).toEqual([]);
  });
});
