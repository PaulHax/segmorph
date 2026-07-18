import type { ImageGeometry } from 'segmorph';

/** Post-extraction mesh processing the worker applies before replying. */
export type MeshPipeline = {
  /** Windowed-sinc smoothing iterations; 0 skips smoothing entirely. */
  smoothIterations: number;
  /** Quadric decimation target in 0..1; 0 skips decimation entirely. */
  targetReduction: number;
};

/** Main thread -> worker, once: the volume every conversion reads from. */
export type InitRequest = {
  kind: 'init';
  /** CT scalars, transferred; the worker owns them from here on. */
  data: Uint8Array;
  geometry: ImageGeometry;
};

export type SegmentRequest = {
  id: string;
  labelValue: number;
  /** Inclusive intensity window that seeds this segment. */
  window: [number, number];
};

/** Main thread -> worker: rebuild these segments' surfaces. */
export type ConvertRequest = {
  kind: 'convert';
  requestId: number;
  segments: SegmentRequest[];
  pipeline: MeshPipeline;
  /** Voxel stride to convert at: 1 is full resolution, 2 is half, and so on. */
  stride: number;
};

export type WorkerRequest = InitRequest | ConvertRequest;

export type SegmentResult = {
  id: string;
  points: Float32Array;
  polys: Uint32Array;
  /** Triangle count straight out of extraction, before decimation. */
  extractedTriangles: number;
};

/** Worker -> main thread: one mesh per segment, arrays transferred back. */
export type ConvertResponse = {
  requestId: number;
  segments: SegmentResult[];
  durationMs: number;
};
