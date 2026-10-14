import { validateLabelValue } from '../image/labelData.js';

export type LabelmapArray = Uint8Array | Uint8ClampedArray | Uint16Array | Uint32Array;

export type LayerAssignment = Readonly<{
  layer: number;
  labelValue: number;
}>;

export type LayerModel = Readonly<{
  assignments: Readonly<Record<string, LayerAssignment>>;
  masks: Readonly<Record<string, LabelmapArray>>;
  order: readonly string[];
}>;

export function createLayerModel(): LayerModel {
  return { assignments: {}, masks: {}, order: [] };
}

function masksCollide(first: LabelmapArray, second: LabelmapArray) {
  for (let index = 0; index < first.length; index += 1) {
    if (first[index] !== 0 && second[index] !== 0) {
      return true;
    }
  }
  return false;
}

function layerHasConflict(
  model: LayerModel,
  segmentId: string,
  labelValue: number,
  mask: LabelmapArray,
  layer: number,
) {
  return model.order.some(
    (otherId) =>
      otherId !== segmentId &&
      model.assignments[otherId].layer === layer &&
      (model.assignments[otherId].labelValue === labelValue ||
        masksCollide(mask, model.masks[otherId])),
  );
}

function lowestAvailableLayer(
  model: LayerModel,
  segmentId: string,
  labelValue: number,
  mask: LabelmapArray,
) {
  let layer = 0;
  while (layerHasConflict(model, segmentId, labelValue, mask, layer)) {
    layer += 1;
  }
  return layer;
}

function validateMaskLength(model: LayerModel, mask: LabelmapArray) {
  const firstId = model.order[0];
  if (firstId !== undefined && model.masks[firstId].length !== mask.length) {
    throw new Error('All segment masks must have the same length');
  }
}

export function assignSegmentToLayer(
  model: LayerModel,
  segmentId: string,
  labelValue: number,
  mask: LabelmapArray,
): LayerModel {
  validateLabelValue(labelValue);
  validateMaskLength(model, mask);
  const existing = Object.hasOwn(model.assignments, segmentId)
    ? model.assignments[segmentId]
    : undefined;
  const layer =
    existing !== undefined && !layerHasConflict(model, segmentId, labelValue, mask, existing.layer)
      ? existing.layer
      : lowestAvailableLayer(model, segmentId, labelValue, mask);

  return {
    assignments: {
      ...model.assignments,
      [segmentId]: { layer, labelValue },
    },
    masks: { ...model.masks, [segmentId]: mask },
    order: existing === undefined ? [...model.order, segmentId] : model.order,
  };
}

export function compactLayers(model: LayerModel): LayerModel {
  let compacted = createLayerModel();
  for (const segmentId of model.order) {
    compacted = assignSegmentToLayer(
      compacted,
      segmentId,
      model.assignments[segmentId].labelValue,
      model.masks[segmentId],
    );
  }
  return compacted;
}
