import { applyConversionRule, findCheapestPath, type ConversionGraph } from './graph.js';
import type { Segment } from './segment.js';
import type { Segmentation } from './segmentation.js';

function requireSegment(segmentation: Segmentation, id: string) {
  if (!Object.hasOwn(segmentation.segments, id)) {
    throw new Error(`Unknown segment id: ${id}`);
  }
  return segmentation.segments[id];
}

function replaceSegment(segmentation: Segmentation, segment: Segment): Segmentation {
  return {
    ...segmentation,
    segments: { ...segmentation.segments, [segment.id]: segment },
  };
}

export function editRepresentation(
  segmentation: Segmentation,
  segmentId: string,
  representationName: string,
  representation: unknown,
): Segmentation {
  const segment = requireSegment(segmentation, segmentId);
  if (representationName !== segmentation.sourceRepresentation) {
    throw new Error(
      `Cannot edit derived representation ${representationName}; promote it to source first`,
    );
  }

  return replaceSegment(segmentation, {
    ...segment,
    representations: { [representationName]: representation },
  });
}

export function getOrCreateRepresentation(
  segmentation: Segmentation,
  segmentId: string,
  target: string,
  graph: ConversionGraph,
) {
  const segment = requireSegment(segmentation, segmentId);
  if (Object.hasOwn(segment.representations, target)) {
    return { segmentation, representation: segment.representations[target] };
  }

  const source = segmentation.sourceRepresentation;
  const path = findCheapestPath(graph, source, target);
  if (!path) {
    throw new Error(`No conversion path from ${source} to ${target}`);
  }
  if (!Object.hasOwn(segment.representations, source)) {
    throw new Error(`Source representation ${source} is missing from segment ${segmentId}`);
  }

  let representation = segment.representations[source];
  const representations = { ...segment.representations };
  for (const rule of path) {
    representation = applyConversionRule(rule, representation);
    representations[rule.target] = representation;
  }

  return {
    segmentation: replaceSegment(segmentation, { ...segment, representations }),
    representation,
  };
}

export function promoteRepresentationToSource(
  segmentation: Segmentation,
  representationName: string,
): Segmentation {
  for (const segmentId of segmentation.order) {
    if (!Object.hasOwn(segmentation.segments[segmentId].representations, representationName)) {
      throw new Error(`Representation ${representationName} is missing from segment ${segmentId}`);
    }
  }

  return { ...segmentation, sourceRepresentation: representationName };
}
