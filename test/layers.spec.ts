import { describe, expect, it } from 'vitest';

import {
  assignSegmentToLayer,
  compactLayers,
  createLayerModel,
} from '../src/index.js';

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

  it('moves a painted segment to a new layer when its mask collides', () => {
    const disjoint = assignSegmentToLayer(
      assignSegmentToLayer(createLayerModel(), 'liver', 1, new Uint8Array([1, 1, 0, 0])),
      'tumor',
      2,
      new Uint8Array([0, 0, 1, 0]),
    );
    const painted = assignSegmentToLayer(
      disjoint,
      'tumor',
      2,
      new Uint8Array([0, 1, 1, 0]),
    );

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
    const erased = assignSegmentToLayer(
      overlapping,
      'tumor',
      2,
      new Uint8Array([0, 0, 1, 0]),
    );

    expect(overlapping.assignments.tumor).toEqual({ layer: 1, labelValue: 2 });
    expect(erased.assignments.tumor).toEqual({ layer: 1, labelValue: 2 });
    expect(compactLayers(erased).assignments.tumor).toEqual({ layer: 0, labelValue: 2 });
  });

  it('supports segment ids that match inherited object properties', () => {
    const model = assignSegmentToLayer(
      createLayerModel(),
      'toString',
      1,
      new Uint8Array([1]),
    );

    expect(model.assignments['toString']).toEqual({ layer: 0, labelValue: 1 });
    expect(model.order).toEqual(['toString']);
  });
});
