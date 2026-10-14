import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  ConvertRequest,
  ConvertResponse,
  WorkerRequest,
} from '../examples/advanced/src/protocol.js';
import { createConversionScheduler } from '../examples/advanced/src/scheduler.js';

class FakeWorker {
  static instances: FakeWorker[] = [];
  messages: WorkerRequest[] = [];
  initialized = false;
  onmessage?: (event: { data: ConvertResponse }) => void;
  onerror?: (event: { message: string; preventDefault: () => void }) => void;
  onmessageerror?: () => void;

  constructor() {
    FakeWorker.instances.push(this);
  }

  postMessage(request: WorkerRequest) {
    if (request.kind === 'init') this.initialized = true;
    else if (!this.initialized) {
      this.onerror?.({ message: 'convert requested before init', preventDefault() {} });
      return;
    }
    this.messages.push(request);
  }

  terminate() {}

  reply(request: ConvertRequest) {
    this.onmessage?.({
      data: { requestId: request.requestId, segments: [], durationMs: request.stride },
    });
  }

  conversions() {
    return this.messages.filter((request) => request.kind === 'convert');
  }
}

const geometry = {
  dims: [1, 1, 1],
  spacing: [1, 1, 1],
  origin: [0, 0, 0],
  direction: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
} as const;

function setup() {
  const onResult = vi.fn();
  const onError = vi.fn();
  let stride = 1;
  const scheduler = createConversionScheduler({
    debounceMs: 150,
    getSegments: () => [],
    getPipeline: () => ({ smoothIterations: 0, targetReduction: 0 }),
    getStride: () => stride,
    onResult,
    onError,
  });
  return {
    scheduler,
    worker: FakeWorker.instances.at(-1)!,
    onResult,
    onError,
    setStride(value: number) {
      stride = value;
    },
  };
}

beforeEach(() => {
  FakeWorker.instances = [];
  vi.useFakeTimers();
  vi.stubGlobal('Worker', FakeWorker);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('advanced example conversion scheduling', () => {
  it('delivers a request made before initialization once the volume arrives', () => {
    const { scheduler, worker, onResult, onError } = setup();
    scheduler.requestConvert();
    vi.runAllTimers();
    expect(onError).not.toHaveBeenCalled();
    expect(onResult).not.toHaveBeenCalled();

    scheduler.init(new Uint8Array([1]), geometry);
    worker.reply(worker.conversions()[0]);
    expect(onError).not.toHaveBeenCalled();
    expect(onResult).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ durationMs: 1 }));
  });

  it('drops a stale result while the newer request is still debouncing', () => {
    const { scheduler, worker, onResult, setStride } = setup();
    scheduler.init(new Uint8Array([1]), geometry);
    scheduler.requestConvert();
    vi.runAllTimers();
    const first = worker.conversions()[0];

    setStride(2);
    scheduler.requestConvert();
    worker.reply(first);
    expect(onResult).not.toHaveBeenCalled();
    vi.runAllTimers();
    const second = worker.conversions()[1];
    worker.reply(second);
    expect(onResult).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ durationMs: 2 }));
  });

  it('delivers only the latest settled result after a busy worker replies', () => {
    const { scheduler, worker, onResult, setStride } = setup();
    scheduler.init(new Uint8Array([1]), geometry);
    scheduler.requestConvert();
    vi.runAllTimers();
    const first = worker.conversions()[0];
    setStride(2);
    scheduler.requestConvert();
    setStride(3);
    scheduler.requestConvert();
    vi.runAllTimers();

    worker.reply(first);
    expect(onResult).not.toHaveBeenCalled();
    worker.reply(worker.conversions()[1]);
    expect(onResult).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ durationMs: 3 }));
  });

  it.each(['error', 'messageerror'] as const)(
    'reports %s and allows another conversion',
    (kind) => {
      const { scheduler, worker, onResult, onError, setStride } = setup();
      scheduler.init(new Uint8Array([1]), geometry);
      scheduler.requestConvert();
      vi.runAllTimers();
      if (kind === 'error')
        worker.onerror?.({ message: 'Unable to convert volume', preventDefault() {} });
      else worker.onmessageerror?.();
      expect(onError).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({
          message:
            kind === 'error'
              ? 'Unable to convert volume'
              : 'Unable to read conversion worker response',
        }),
      );

      setStride(2);
      scheduler.requestConvert();
      vi.runAllTimers();
      worker.reply(worker.conversions()[1]);
      expect(onResult).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ durationMs: 2 }));
    },
  );

  it.each(['error', 'messageerror'] as const)(
    'drops a stale %s while a newer request debounces',
    (kind) => {
      const { scheduler, worker, onResult, onError, setStride } = setup();
      scheduler.init(new Uint8Array([1]), geometry);
      scheduler.requestConvert();
      vi.runAllTimers();
      setStride(2);
      scheduler.requestConvert();
      if (kind === 'error')
        worker.onerror?.({ message: 'Unable to convert volume', preventDefault() {} });
      else worker.onmessageerror?.();
      expect(onError).not.toHaveBeenCalled();

      vi.runAllTimers();
      worker.reply(worker.conversions()[1]);
      expect(onResult).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ durationMs: 2 }));
    },
  );

  it('ignores pending requests and late replies after disposal', () => {
    const { scheduler, worker, onResult } = setup();
    scheduler.init(new Uint8Array([1]), geometry);
    scheduler.requestConvert();
    vi.runAllTimers();
    const first = worker.conversions()[0];
    scheduler.requestConvert();
    scheduler.dispose();
    vi.runAllTimers();
    worker.reply(first);
    scheduler.requestConvert();
    vi.runAllTimers();
    expect(onResult).not.toHaveBeenCalled();
  });
});
