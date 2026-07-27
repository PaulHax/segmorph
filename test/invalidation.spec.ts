import { describe, expect, it, vi } from 'vitest';

import {
  addSegment,
  createConversionGraph,
  createSegment,
  createSegmentation,
  editRepresentation,
  getOrCreateRepresentation,
  promoteRepresentationToSource,
  type ConversionRule,
} from '../src/index.js';

function segmentationWithRepresentations() {
  return addSegment(
    createSegmentation('A'),
    createSegment({
      id: 'segment',
      name: 'Segment',
      color: [1, 0, 0],
      representations: { A: 'source', B: 'derived-b', C: 'derived-c' },
    }),
  );
}

describe('source-authoritative invalidation', () => {
  it('edits the source and drops every derived representation immutably', () => {
    const segmentation = segmentationWithRepresentations();
    const result = editRepresentation(segmentation, 'segment', 'A', 'edited');

    expect(result.segments.segment.representations).toEqual({ A: 'edited' });
    expect(segmentation.segments.segment.representations).toEqual({
      A: 'source',
      B: 'derived-b',
      C: 'derived-c',
    });
  });

  it('re-derives a missing representation along the cheapest path', () => {
    const aToB = vi.fn((value: string) => `${value}-b`);
    const bToC = vi.fn((value: string) => `${value}-c`);
    const direct = vi.fn((value: string) => `${value}-direct`);
    const rules: ConversionRule<any, any>[] = [
      { source: 'A', target: 'B', cost: 1, convert: aToB },
      { source: 'B', target: 'C', cost: 1, convert: bToC },
      { source: 'A', target: 'C', cost: 5, convert: direct },
    ];
    const edited = editRepresentation(segmentationWithRepresentations(), 'segment', 'A', 'edited');
    const result = getOrCreateRepresentation(edited, 'segment', 'C', createConversionGraph(rules));

    expect(result.representation).toBe('edited-b-c');
    expect(result.segmentation.segments.segment.representations).toEqual({
      A: 'edited',
      B: 'edited-b',
      C: 'edited-b-c',
    });
    expect(aToB).toHaveBeenCalledOnce();
    expect(bToC).toHaveBeenCalledOnce();
    expect(direct).not.toHaveBeenCalled();
  });

  it('returns an existing representation without converting or changing the segmentation', () => {
    const segmentation = segmentationWithRepresentations();
    const convert = vi.fn();
    const graph = createConversionGraph([{ source: 'A', target: 'B', cost: 1, convert }]);
    const result = getOrCreateRepresentation(segmentation, 'segment', 'B', graph);

    expect(result).toEqual({ segmentation, representation: 'derived-b' });
    expect(result.segmentation).toBe(segmentation);
    expect(convert).not.toHaveBeenCalled();
  });

  it('rejects derived edits until that representation is promoted to source', () => {
    const segmentation = segmentationWithRepresentations();

    expect(() => editRepresentation(segmentation, 'segment', 'B', 'edited-b')).toThrow(
      'Cannot edit derived representation B; promote it to source first',
    );

    const promoted = promoteRepresentationToSource(segmentation, 'B');
    const edited = editRepresentation(promoted, 'segment', 'B', 'edited-b');

    expect(promoted.sourceRepresentation).toBe('B');
    expect(edited.segments.segment.representations).toEqual({ B: 'edited-b' });
  });

  it('rejects unknown segments, unavailable paths, and incomplete promotion', () => {
    const segmentation = segmentationWithRepresentations();
    const withoutB = addSegment(
      segmentation,
      createSegment({ id: 'other', name: 'Other', color: [0, 1, 0], representations: { A: 'a' } }),
    );

    expect(() => editRepresentation(segmentation, 'missing', 'A', 'value')).toThrow(
      'Unknown segment id: missing',
    );
    expect(() =>
      getOrCreateRepresentation(segmentation, 'segment', 'missing', createConversionGraph()),
    ).toThrow('No conversion path from A to missing');
    expect(() => promoteRepresentationToSource(withoutB, 'B')).toThrow(
      'Representation B is missing from segment other',
    );
  });
});
