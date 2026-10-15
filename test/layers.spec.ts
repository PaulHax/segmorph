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

  it('removes layer gaps without merging disjoint segments with the same label', () => {
    const model = {
      assignments: {
        left: { layer: 4, labelValue: 256 },
        right: { layer: 9, labelValue: 256 },
      },
      masks: { left: new Uint8Array([1, 0]), right: new Uint8Array([0, 1]) },
      order: ['left', 'right'],
    };
    const snapshot = structuredClone(model);
    const compacted = compactLayers(model);

    expect(compacted.assignments).toEqual({
      left: { layer: 0, labelValue: 256 },
      right: { layer: 1, labelValue: 256 },
    });
    expect(compacted.order).toEqual(snapshot.order);
    expect(compacted.masks.left).toBe(model.masks.left);
    expect(compacted.masks.right).toBe(model.masks.right);
    expect(model).toEqual(snapshot);
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

  it.each([
    [0, 1],
    [9, 3],
  ])(
    'avoids greedy layer growth and closes gaps in layers %i and %i',
    (firstLayer, secondLayer) => {
      let model = createLayerModel();
      for (const [id, labelValue, mask] of [
        ['__proto__', 1, [1, 0, 0, 1]],
        ['b', 2, [0, 0, 1, 1]],
        ['c', 3, [1, 1, 0, 0]],
        ['d', 4, [0, 1, 1, 0]],
      ] as const) {
        model = assignSegmentToLayer(model, id, labelValue, new Uint8Array(mask));
      }
      model = assignSegmentToLayer(model, '__proto__', 1, new Uint8Array([1, 0, 0, 0]));
      expect(model.order.map((id) => model.assignments[id].layer)).toEqual([0, 1, 1, 0]);

      // Also exercise sparse layer numbers whose numeric order differs from segment order.
      const layers = [firstLayer, secondLayer];
      model = {
        ...model,
        assignments: Object.fromEntries(
          model.order.map((id) => [
            id,
            { ...model.assignments[id], layer: layers[model.assignments[id].layer] },
          ]),
        ),
      };
      const snapshot = structuredClone(model);
      const compacted = compactLayers(model);
      const first = firstLayer < secondLayer ? 0 : 1;
      expect(compacted.assignments).toEqual({
        ['__proto__']: { layer: first, labelValue: 1 },
        b: { layer: 1 - first, labelValue: 2 },
        c: { layer: 1 - first, labelValue: 3 },
        d: { layer: first, labelValue: 4 },
      });
      expect(Object.hasOwn(compacted.assignments, '__proto__')).toBe(true);
      expect(Object.getPrototypeOf(compacted.assignments)).toBe(Object.prototype);
      expect(compacted.order).toEqual(snapshot.order);
      expect(compacted.masks).toEqual(snapshot.masks);
      for (const id of model.order) {
        expect(compacted.masks[id]).toBe(model.masks[id]);
      }
      expect(model).toEqual(snapshot);
    },
  );

  it('preserves the greedy reduction from three layers to two after an erase', () => {
    let model = createLayerModel();
    for (const [id, labelValue, mask] of [
      ['a', 1, [1, 0, 0, 1]],
      ['b', 2, [0, 1, 1, 1]],
      ['c', 3, [0, 1, 0, 0]],
      ['d', 4, [1, 0, 1, 0]],
    ] as const) {
      model = assignSegmentToLayer(model, id, labelValue, new Uint8Array(mask));
    }
    model = assignSegmentToLayer(model, 'a', 1, new Uint8Array([1, 0, 0, 0]));
    const snapshot = structuredClone(model);
    expect(model.order.map((id) => model.assignments[id].layer)).toEqual([0, 1, 0, 2]);

    const compacted = compactLayers(model);
    expect(compacted.assignments).toEqual({
      a: { layer: 0, labelValue: 1 },
      b: { layer: 0, labelValue: 2 },
      c: { layer: 1, labelValue: 3 },
      d: { layer: 1, labelValue: 4 },
    });
    expect(compacted.order).toEqual(snapshot.order);
    expect(compacted.masks).toEqual(snapshot.masks);
    for (const id of model.order) {
      expect(compacted.masks[id]).toBe(model.masks[id]);
    }
    expect(model).toEqual(snapshot);
  });

  it('supports segment ids that match inherited object properties', () => {
    const model = assignSegmentToLayer(createLayerModel(), 'toString', 1, new Uint8Array([1]));

    expect(model.assignments['toString']).toEqual({ layer: 0, labelValue: 1 });
    expect(model.order).toEqual(['toString']);
  });
});
