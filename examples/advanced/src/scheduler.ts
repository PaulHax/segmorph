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
  onError: (error: Error) => void;
}) {
  const debounceMs = options.debounceMs ?? 150;
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

  let initialized = false;
  let disposed = false;
  let latestRequestId = 0;
  let lastSentRequestId = 0;
  let inFlightRequestId: number | undefined;
  let debounceTimer: ReturnType<typeof setTimeout> | undefined;

  const send = () => {
    if (
      !initialized ||
      disposed ||
      inFlightRequestId !== undefined ||
      debounceTimer !== undefined ||
      lastSentRequestId === latestRequestId
    )
      return;
    const request: ConvertRequest = {
      kind: 'convert',
      requestId: latestRequestId,
      segments: options.getSegments(),
      pipeline: options.getPipeline(),
      stride: options.getStride(),
    };
    lastSentRequestId = request.requestId;
    inFlightRequestId = request.requestId;
    worker.postMessage(request);
  };

  worker.onmessage = ({ data: response }: MessageEvent<ConvertResponse>) => {
    if (disposed || response.requestId !== inFlightRequestId) return;
    inFlightRequestId = undefined;
    if (response.requestId === latestRequestId) options.onResult(response);
    send();
  };

  const fail = (error: Error) => {
    if (disposed) return;
    const failedRequestId = inFlightRequestId;
    inFlightRequestId = undefined;
    if (failedRequestId === undefined || failedRequestId === latestRequestId)
      options.onError(error);
    send();
  };

  worker.onerror = (event) => {
    event.preventDefault();
    fail(new Error(event.message || 'Conversion worker failed'));
  };

  worker.onmessageerror = () => {
    fail(new Error('Unable to read conversion worker response'));
  };

  return {
    /** Hand the worker the volume once; it owns that buffer afterwards. */
    init(data: Uint8Array, geometry: ImageGeometry) {
      // Copy before transferring: the main thread still renders this volume.
      const owned = data.slice();
      const request: InitRequest = { kind: 'init', data: owned, geometry };
      worker.postMessage(request, [owned.buffer]);
      initialized = true;
      send();
    },
    requestConvert() {
      if (disposed) return;
      latestRequestId += 1;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        debounceTimer = undefined;
        send();
      }, debounceMs);
    },
    dispose() {
      disposed = true;
      clearTimeout(debounceTimer);
      worker.terminate();
    },
  };
}
