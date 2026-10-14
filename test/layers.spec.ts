import { describe, expect, it } from 'vitest';

import { assignSegmentToLayer, compactLayers, createLayerModel } from '../src/index.js';

describe('layer overlap model', () => {
  it('places disjoint segments together on layer 0', () => {
    const model = assignSegmentToLayer(
      assignSegmentToLayer(createLayerModel(), 'liver', 1, new Uint8Array([1, 1, 0, 0])),
      'tumor',
      2,
      new Uint8Array([0, 0, 1, 0]),
    );

    expect(model.assignments).toEqual({
      liver: { layer: 0, labelValue: 1 },
      tumor: { layer: 0, labelValue: 2 },
    });
  });

  it('keeps disjoint segments with the same label distinguishable in shared labelmaps', () => {
    const model = assignSegmentToLayer(
      assignSegmentToLayer(createLayerModel(), 'left', 1, new Uint8Array([1, 0])),
      'right',
      1,
      new Uint8Array([0, 1]),
    );

    expect(model.assignments).toEqual({
      left: { layer: 0, labelValue: 1 },
      right: { layer: 1, labelValue: 1 },
    });
    expect(compactLayers(model).assignments).toEqual(model.assignments);
  });

  it('moves a segment when its new label conflicts with another segment in its layer', () => {
    const model = assignSegmentToLayer(
      assignSegmentToLayer(createLayerModel(), 'left', 1, new Uint8Array([1, 0])),
      'right',
      2,
      new Uint8Array([0, 1]),
    );
    const relabeled = assignSegmentToLayer(model, 'right', 1, new Uint8Array([0, 1]));

    expect(relabeled.assignments.right).toEqual({ layer: 1, labelValue: 1 });
    expect(model.assignments.right).toEqual({ layer: 0, labelValue: 2 });
  });

  it.each([0, -1, 0.5, NaN, Infinity, 0x1_0000_0000])(
    'rejects label values that cannot identify foreground: %s',
    (labelValue) => {
      expect(() =>
        assignSegmentToLayer(createLayerModel(), 'segment', labelValue, new Uint8Array([1])),
      ).toThrow('labelValue must be an integer between 1 and 4294967295');
    },
  );

  it('moves a painted segment to a new layer when its mask collides', () => {
    const disjoint = assignSegmentToLayer(
      assignSegmentToLayer(createLayerModel(), 'liver', 1, new Uint8Array([1, 1, 0, 0])),
      'tumor',
      2,
      new Uint8Array([0, 0, 1, 0]),
    );
    const painted = assignSegmentToLayer(disjoint, 'tumor', 2, new Uint8Array([0, 1, 1, 0]));

    expect(painted.assignments.tumor).toEqual({ layer: 1, labelValue: 2 });
    expect(disjoint.assignments.tumor).toEqual({ layer: 0, labelValue: 2 });
  });

  it('compacts a segment after its overlap is erased', () => {
    const overlapping = assignSegmentToLayer(
      assignSegmentToLayer(createLayerModel(), 'liver', 1, new Uint8Array([1, 1, 0, 0])),
      'tumor',
      2,
      new Uint8Array([0, 1, 1, 0]),
    );
    const erased = assignSegmentToLayer(overlapping, 'tumor', 2, new Uint8Array([0, 0, 1, 0]));

    expect(overlapping.assignments.tumor).toEqual({ layer: 1, labelValue: 2 });
    expect(erased.assignments.tumor).toEqual({ layer: 1, labelValue: 2 });
    expect(compactLayers(erased).assignments.tumor).toEqual({ layer: 0, labelValue: 2 });
  });

  it('supports segment ids that match inherited object properties', () => {
    const model = assignSegmentToLayer(createLayerModel(), 'toString', 1, new Uint8Array([1]));

    expect(model.assignments['toString']).toEqual({ layer: 0, labelValue: 1 });
    expect(model.order).toEqual(['toString']);
  });
});
