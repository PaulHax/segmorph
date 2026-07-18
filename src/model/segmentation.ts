import type { Segment } from './segment.js';

export type Segmentation = Readonly<{
  segments: Readonly<Record<string, Segment>>;
  order: readonly string[];
  sourceRepresentation: string;
}>;

export function createSegmentation(sourceRepresentation: string): Segmentation {
  return {
    segments: {},
    order: [],
    sourceRepresentation,
  };
}

export function addSegment(segmentation: Segmentation, segment: Segment): Segmentation {
  if (Object.hasOwn(segmentation.segments, segment.id)) {
    throw new Error(`Segment id already exists: ${segment.id}`);
  }

  return {
    ...segmentation,
    segments: { ...segmentation.segments, [segment.id]: segment },
    order: [...segmentation.order, segment.id],
  };
}

function requireSegment(segmentation: Segmentation, id: string) {
  if (!Object.hasOwn(segmentation.segments, id)) {
    throw new Error(`Unknown segment id: ${id}`);
  }
  return segmentation.segments[id];
}

export function removeSegment(segmentation: Segmentation, id: string): Segmentation {
  requireSegment(segmentation, id);
  const { [id]: removed, ...segments } = segmentation.segments;
  void removed;

  return {
    ...segmentation,
    segments,
    order: segmentation.order.filter((segmentId) => segmentId !== id),
  };
}

export function renameSegment(segmentation: Segmentation, id: string, name: string): Segmentation {
  const segment = requireSegment(segmentation, id);

  return {
    ...segmentation,
    segments: {
      ...segmentation.segments,
      [id]: { ...segment, name },
    },
  };
}

export function reorderSegments(segmentation: Segmentation, order: readonly string[]): Segmentation {
  const uniqueIds = new Set(order);
  if (
    order.length !== segmentation.order.length ||
    uniqueIds.size !== order.length ||
    order.some((id) => !Object.hasOwn(segmentation.segments, id))
  ) {
    throw new Error('Order must contain every segment id exactly once');
  }

  return { ...segmentation, order: [...order] };
}
