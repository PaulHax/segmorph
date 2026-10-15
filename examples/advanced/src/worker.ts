import {
  createOrientedImage,
  labelmapToSurface,
  meshDecimate,
  meshSmooth,
  resampleNearest,
  triangleCount,
  type Mesh,
  type OrientedImage,
} from 'segmorph';
import type {
  ConvertRequest,
  ConvertResponse,
  MeshPipeline,
  SegmentRequest,
  SegmentResult,
  WorkerRequest,
} from './protocol';

// Mesh extraction, smoothing, decimation, and preview resampling are
// synchronous and CPU-bound, so this pipeline runs off the main thread.

let sourceVolume: OrientedImage<Uint8Array> | undefined;

/**
 * Working volumes by stride. Converting a 256 x 256 x 133 CT at full
 * resolution is a real workload (millions of triangles), so the app offers a
 * coarser preview. `resampleNearest` retains the origin and orientation with
 * wider spacing and fewer samples in the same world coordinate system.
 */
const resampled = new Map<number, OrientedImage<Uint8Array>>();

function workingVolume(stride: number, input: OrientedImage<Uint8Array>) {
  if (stride <= 1) return input;
  const cached = resampled.get(stride);
  if (cached) return cached;
  const reference = {
    dims: input.dims.map((d) => Math.max(1, Math.floor(d / stride))),
    spacing: input.spacing.map((s) => s * stride),
    origin: input.origin,
    direction: input.direction,
  };
  const coarse = resampleNearest(input, reference);
  resampled.set(stride, coarse);
  return coarse;
}

/** Threshold the CT into a binary mask for one segment. */
function seedMask(source: Uint8Array, segment: SegmentRequest) {
  const [low, high] = segment.window;
  const mask = new Uint8Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    const value = source[i];
    if (value >= low && value <= high) mask[i] = segment.labelValue;
  }
  return mask;
}

function convertSegment(
  segment: SegmentRequest,
  pipeline: MeshPipeline,
  volume: OrientedImage<Uint8Array>,
): SegmentResult {
  const { data, ...imageGeometry } = volume;
  const extracted = labelmapToSurface(
    { data: seedMask(data, segment), ...imageGeometry },
    { labelValue: segment.labelValue },
  );

  // Marching cubes leaves voxel-staircase facets. Smoothing relaxes them
  // without changing the vertex count; decimation then removes the vertices
  // the flattened regions no longer need.
  const smoothed: Mesh =
    pipeline.smoothIterations > 0
      ? meshSmooth(extracted, { numberOfIterations: pipeline.smoothIterations })
      : extracted;

  const mesh: Mesh =
    pipeline.targetReduction > 0
      ? meshDecimate(smoothed, { targetReduction: pipeline.targetReduction })
      : smoothed;

  return {
    id: segment.id,
    points: mesh.points,
    polys: mesh.polys,
    extractedTriangles: triangleCount(extracted),
  };
}

function convert(request: ConvertRequest, source: OrientedImage<Uint8Array>) {
  const start = performance.now();
  const volume = workingVolume(request.stride, source);

  const segments = request.segments.map((segment) =>
    convertSegment(segment, request.pipeline, volume),
  );

  const response: ConvertResponse = {
    requestId: request.requestId,
    segments,
    durationMs: performance.now() - start,
  };
  const transfers = segments.flatMap((s) => [s.points.buffer, s.polys.buffer]);
  self.postMessage(response, transfers);
}

self.onmessage = ({ data: request }: MessageEvent<WorkerRequest>) => {
  if (request.kind === 'init') {
    sourceVolume = createOrientedImage({ data: request.data, ...request.geometry });
    resampled.clear();
    return;
  }
  if (!sourceVolume) throw new Error('convert requested before init');
  convert(request, sourceVolume);
};
