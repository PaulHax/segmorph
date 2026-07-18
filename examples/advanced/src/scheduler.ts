import type { ImageGeometry } from 'segmorph';
import type {
  ConvertRequest,
  ConvertResponse,
  InitRequest,
  MeshPipeline,
  SegmentRequest,
} from './protocol';

/**
 * Reference scheduling policy for segmorph conversions.
 *
 * segmorph itself is scheduler-free: conversions are pure synchronous
 * functions, so the app decides when and where they run. This module encodes
 * the recommended policy:
 *
 * - Web Worker: extraction, smoothing, and decimation are all CPU-bound over
 *   the whole volume, so they run off the main thread. The volume is sent
 *   once at init and the worker keeps it; requests carry only parameters.
 * - Debounced: dragging a slider fires a stream of input events, and
 *   converting on each one would waste work on states nobody sees. One
 *   conversion per settled gesture.
 * - Latest wins: if a conversion is in flight when another is requested, the
 *   new one waits and stale responses are dropped, so the rendered surface
 *   never goes backwards in time.
 */
export function createConversionScheduler(options: {
  debounceMs?: number;
  getSegments: () => SegmentRequest[];
  getPipeline: () => MeshPipeline;
  getStride: () => number;
  onResult: (response: ConvertResponse) => void;
}) {
  const debounceMs = options.debounceMs ?? 150;
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

  let nextRequestId = 0;
  let latestRequestId = -1;
  let inFlight = false;
  let pending = false;
  let debounceTimer: ReturnType<typeof setTimeout> | undefined;

  const send = () => {
    inFlight = true;
    nextRequestId += 1;
    latestRequestId = nextRequestId;
    const request: ConvertRequest = {
      kind: 'convert',
      requestId: nextRequestId,
      segments: options.getSegments(),
      pipeline: options.getPipeline(),
      stride: options.getStride(),
    };
    worker.postMessage(request);
  };

  worker.onmessage = ({ data: response }: MessageEvent<ConvertResponse>) => {
    inFlight = false;
    if (pending) {
      pending = false;
      send();
    }
    // Drop stale meshes: a newer request is already in flight.
    if (response.requestId !== latestRequestId) return;
    options.onResult(response);
  };

  return {
    /** Hand the worker the volume once; it owns that buffer afterwards. */
    init(data: Uint8Array, geometry: ImageGeometry) {
      // Copy before transferring: the main thread still renders this volume.
      const owned = data.slice();
      const request: InitRequest = { kind: 'init', data: owned, geometry };
      worker.postMessage(request, [owned.buffer]);
    },
    requestConvert() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        if (inFlight) {
          pending = true;
        } else {
          send();
        }
      }, debounceMs);
    },
    dispose() {
      clearTimeout(debounceTimer);
      worker.terminate();
    },
  };
}
