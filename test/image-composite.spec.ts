import { Buffer } from 'node:buffer';
import { describe, expect, it } from 'vitest';

import {
  compositeMax,
  compositeMin,
  compositeSet,
  maskByLabelValue,
} from '../src/image/composite.js';
import { EMPTY_EXTENT, intersectExtents, type Extent3D } from '../src/image/extent.js';
import type { ImageData } from '../src/image/orientedImage.js';

const dims = [3, 2, 2] as const;
const middleColumn = [1, 2, 0, 1, 0, 1] as const;

describe('bounded image compositing', () => {
  it('sets only voxels inside an inclusive extent and leaves both inputs unchanged', () => {
    const input = new Uint8Array(12).fill(1);
    const modifier = Uint8Array.from({ length: 12 }, (_, index) => index + 10);
    const originalInput = input.slice();
    const originalModifier = modifier.slice();

    const output = compositeSet(input, modifier, dims, middleColumn);

    expect(output).toBeInstanceOf(Uint8Array);
    expect([...output]).toEqual([1, 11, 12, 1, 14, 15, 1, 17, 18, 1, 20, 21]);
    expect(input).toEqual(originalInput);
    expect(modifier).toEqual(originalModifier);
  });

  it('takes the minimum and maximum only inside the extent', () => {
    const input = Uint16Array.from([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]);
    const modifier = Uint16Array.from([1, 9, 4, 7, 3, 10, 2, 12, 6, 5, 11, 0]);

    expect([...compositeMin(input, modifier, dims, middleColumn)]).toEqual([
      8, 8, 4, 8, 3, 8, 8, 8, 6, 8, 8, 0,
    ]);
    expect([...compositeMax(input, modifier, dims, middleColumn)]).toEqual([
      8, 9, 8, 8, 8, 10, 8, 12, 8, 8, 11, 8,
    ]);
  });

  it('keeps input voxels selected by a label and fills other voxels in the extent', () => {
    const input = Int16Array.from({ length: 12 }, (_, index) => index - 2);
    const labels = Uint8Array.from([0, 4, 4, 4, 2, 4, 4, 0, 4, 4, 4, 3]);

    expect([...maskByLabelValue(input, labels, dims, 4, -9, middleColumn)]).toEqual([
      -2, -1, 0, 1, -9, 3, 4, -9, 6, 7, 8, -9,
    ]);
    expect([...input]).toEqual([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('defaults to the complete image extent', () => {
    expect([...compositeSet(new Uint8Array(12), new Uint8Array(12).fill(5), dims)]).toEqual(
      new Array(12).fill(5),
    );
  });

  it('leaves input values unchanged when the requested extents do not intersect', () => {
    const input = Uint8Array.from({ length: 12 }, (_, index) => index + 1);
    const modifier = new Uint8Array(12).fill(99);
    const extent = intersectExtents([0, 0, 0, 1, 0, 1], [1, 2, 0, 1, 0, 1]);

    for (const composite of [compositeSet, compositeMin, compositeMax]) {
      const output = composite(input, modifier, dims, extent);
      expect(output).toEqual(input);
      expect(output).not.toBe(input);
    }
    expect(maskByLabelValue(input, modifier, dims, 1, 0, extent)).toEqual(input);
  });

  it('validates dimensions, array sizes, extents, and compatible data types', () => {
    const input = new Uint8Array(12);
    const modifier = new Uint8Array(12);

    expect(() => compositeSet(input, modifier, [3, 0, 2])).toThrow(
      'dims must contain three positive integers',
    );
    expect(() => compositeSet(input.subarray(1), modifier, dims)).toThrow(
      'Input data length 11 does not match dimensions (expected 12)',
    );
    expect(() => compositeSet(input, modifier.subarray(1), dims)).toThrow(
      'Modifier data length 11 does not match dimensions (expected 12)',
    );
    expect(() => compositeSet(input, modifier, dims, [0, 3, 0, 1, 0, 1])).toThrow(
      'extent must be within image dimensions',
    );
    expect(() => compositeSet(input, modifier, dims, [2, 1, 0, 1, 0, 1])).toThrow(
      'extent bounds must be ordered',
    );
    expect(() => compositeSet<ImageData>(input, new Uint16Array(12), dims)).toThrow(
      'Input and modifier data types must match',
    );
  });
});

const arrayCases = [
  Int8Array,
  Uint8Array,
  Uint8ClampedArray,
  Int16Array,
  Uint16Array,
  Int32Array,
  Uint32Array,
  Float32Array,
  Float64Array,
].map((Type) => ({ name: Type.name, create: (values: number[]) => new Type(values) }));
arrayCases.push({ name: 'Buffer', create: (values) => Buffer.from(values) });

const operations = [
  { name: 'set', run: compositeSet, full: [3, 9, 3, 7], bounded: [8, 9, 3, 4] },
  { name: 'minimum', run: compositeMin, full: [3, 2, 3, 4], bounded: [8, 2, 3, 4] },
  { name: 'maximum', run: compositeMax, full: [8, 9, 6, 7], bounded: [8, 9, 6, 4] },
  {
    name: 'mask',
    run: (input: ImageData, mask: ImageData, imageDims: readonly number[], extent?: Extent3D) =>
      maskByLabelValue(input, mask, imageDims, 3, 0, extent),
    full: [8, 0, 6, 0],
    bounded: [8, 0, 6, 4],
  },
];

describe.each(arrayCases)('$name input copies', ({ create }) => {
  it.each(['full', 'bounded', 'empty'] as const)(
    'preserves the dtype and isolates both inputs with the %s extent',
    (kind) => {
      const extent =
        kind === 'full'
          ? undefined
          : kind === 'empty'
            ? EMPTY_EXTENT
            : ([1, 2, 0, 0, 0, 0] as const);
      for (const { name, run, full, bounded } of operations) {
        const inputBacking = create([99, 8, 2, 6, 4, 88]);
        const modifierBacking = create([77, 3, 9, 3, 7, 66]);
        const input = inputBacking.subarray(1, 5);
        const modifier = modifierBacking.subarray(1, 5);
        const originalInput = [...inputBacking];
        const originalModifier = [...modifierBacking];

        const output = run(input, modifier, [4, 1, 1], extent);

        expect(output.constructor).toBe(input.constructor);
        expect(output).not.toBe(input);
        expect([...output], name).toEqual(
          kind === 'full' ? full : kind === 'bounded' ? bounded : [8, 2, 6, 4],
        );
        expect([...inputBacking], name).toEqual(originalInput);
        expect([...modifierBacking], name).toEqual(originalModifier);

        output[0] = 42;
        expect([...inputBacking], name).toEqual(originalInput);
        expect([...modifierBacking], name).toEqual(originalModifier);
        const originalOutput = [...output];
        input[1] = 43;
        modifier[2] = 44;
        expect([...output]).toEqual(originalOutput);
      }
    },
  );
});

describe('overlapping Buffer views', () => {
  it.each([
    { name: 'set', run: compositeSet, values: [7, 8, 9], expected: [7, 8] },
    { name: 'minimum', run: compositeMin, values: [7, 8, 9], expected: [7, 8] },
    { name: 'maximum', run: compositeMax, values: [9, 8, 7], expected: [9, 8] },
    {
      name: 'mask',
      run: (input: Buffer, mask: Buffer, imageDims: readonly number[]) =>
        maskByLabelValue(input, mask, imageDims, 1, 9),
      values: [0, 1, 2],
      expected: [9, 2],
    },
  ])(
    '$name reads original values and leaves shared storage unchanged',
    ({ run, values, expected }) => {
      const backing = Buffer.from(values);
      const input = backing.subarray(1);
      const modifier = backing.subarray(0, 2);

      const output = run(input, modifier, [2, 1, 1]);

      expect(Buffer.isBuffer(output)).toBe(true);
      expect([...output]).toEqual(expected);
      expect([...backing]).toEqual(values);
    },
  );
});

it.each([
  new Int8Array([-128, -1, 127]),
  new Uint8Array([0, 128, 255]),
  new Uint8ClampedArray([0, 128, 255]),
  new Int16Array([-32768, -1, 32767]),
  new Uint16Array([0, 256, 65535]),
  new Int32Array([-2147483648, -1, 2147483647]),
  new Uint32Array([0, 0x12345678, 0xffffffff]),
  new Float32Array([-1.25, -0, Infinity, NaN]),
  new Float64Array([-Math.PI, -0, -Infinity, NaN]),
])('copies numeric values without changing their dtype: %s', (input) => {
  const output = compositeSet(input, input, [input.length, 1, 1], EMPTY_EXTENT);
  expect(output.constructor).toBe(input.constructor);
  expect([...output]).toEqual([...input]);
});
