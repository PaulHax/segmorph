import { describe, expect, it } from 'vitest';

import {
  addSegment,
  createSegment,
  createSegmentation,
  getRepresentation,
  removeSegment,
  renameSegment,
  reorderSegments,
  setRepresentation,
} from '../src/index.js';

describe('segment', () => {
  it('creates a segment and gets and sets representations immutably', () => {
    const segment = createSegment({
      id: 'tumor',
      name: 'Tumor',
      color: [1, 0.25, 0],
    });
    const labelmap = { data: new Uint8Array([0, 1]) };
    const updated = setRepresentation(segment, 'labelmap', labelmap);

    expect(segment).toEqual({
      id: 'tumor',
      name: 'Tumor',
      color: [1, 0.25, 0],
      locked: false,
      representations: {},
    });
    expect(getRepresentation(segment, 'labelmap')).toBeUndefined();
    expect(getRepresentation(segment, 'toString')).toBeUndefined();
    expect(getRepresentation(updated, 'labelmap')).toBe(labelmap);
    expect(updated).not.toBe(segment);
    expect(updated.representations).not.toBe(segment.representations);
  });
});

describe('segmentation', () => {
  const liver = createSegment({ id: 'liver', name: 'Liver', color: [0.5, 0, 0] });
  const tumor = createSegment({ id: 'tumor', name: 'Tumor', color: [1, 0, 0] });

  it('adds segments in order without mutating the input', () => {
    const empty = createSegmentation('labelmap');
    const withLiver = addSegment(empty, liver);
    const result = addSegment(withLiver, tumor);

    expect(empty).toEqual({ segments: {}, order: [], sourceRepresentation: 'labelmap' });
    expect(result.segments).toEqual({ liver, tumor });
    expect(result.order).toEqual(['liver', 'tumor']);
    expect(result).not.toBe(withLiver);
  });

  it('rejects duplicate segment ids', () => {
    const segmentation = addSegment(createSegmentation('labelmap'), liver);

    expect(() => addSegment(segmentation, { ...liver, name: 'Duplicate' })).toThrow(
      'Segment id already exists: liver',
    );
  });

  it('supports segment ids that match inherited object properties', () => {
    const prototypeIdSegment = createSegment({
      id: 'toString',
      name: 'Inherited name',
      color: [0, 0, 0],
    });
    const segmentation = addSegment(createSegmentation('labelmap'), prototypeIdSegment);
    const renamed = renameSegment(segmentation, 'toString', 'Renamed');

    expect(renamed.segments['toString'].name).toBe('Renamed');
    expect(removeSegment(renamed, 'toString').order).toEqual([]);
  });

  it('removes a segment while preserving the remaining order', () => {
    const segmentation = addSegment(addSegment(createSegmentation('surface'), liver), tumor);
    const result = removeSegment(segmentation, 'liver');

    expect(result.segments).toEqual({ tumor });
    expect(result.order).toEqual(['tumor']);
    expect(segmentation.order).toEqual(['liver', 'tumor']);
  });

  it('renames a segment without changing its id or order', () => {
    const segmentation = addSegment(addSegment(createSegmentation('labelmap'), liver), tumor);
    const result = renameSegment(segmentation, 'tumor', 'Lesion');

    expect(result.segments.tumor?.name).toBe('Lesion');
    expect(segmentation.segments.tumor?.name).toBe('Tumor');
    expect(result.order).toEqual(['liver', 'tumor']);
  });

  it('reorders segments and rejects an incomplete or unknown order', () => {
    const segmentation = addSegment(addSegment(createSegmentation('labelmap'), liver), tumor);

    expect(reorderSegments(segmentation, ['tumor', 'liver']).order).toEqual(['tumor', 'liver']);
    expect(() => reorderSegments(segmentation, ['liver'])).toThrow(
      'Order must contain every segment id exactly once',
    );
    expect(() => reorderSegments(segmentation, ['liver', 'other'])).toThrow(
      'Order must contain every segment id exactly once',
    );
    expect(() => reorderSegments(segmentation, ['toString', 'tumor'])).toThrow(
      'Order must contain every segment id exactly once',
    );
  });

  it('rejects operations for unknown segment ids', () => {
    const segmentation = createSegmentation('labelmap');

    expect(() => removeSegment(segmentation, 'missing')).toThrow('Unknown segment id: missing');
    expect(() => renameSegment(segmentation, 'missing', 'Name')).toThrow(
      'Unknown segment id: missing',
    );
  });
});
