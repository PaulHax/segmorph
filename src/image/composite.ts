import type { ImageData } from './orientedImage.js';
import type { Extent3D } from './extent.js';

export type CompositeOperation = 'set' | 'minimum' | 'maximum';

function validateDimensions(dims: readonly number[]) {
  if (dims.length !== 3 || dims.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error('dims must contain three positive integers');
  }
}

function validateDataLength(
  data: ImageData,
  expectedLength: number,
  role: 'Input' | 'Modifier' | 'Mask',
) {
  if (data.length !== expectedLength) {
    throw new Error(
      `${role} data length ${data.length} does not match dimensions (expected ${expectedLength})`,
    );
  }
}

function resolveExtent(dims: readonly number[], extent?: Extent3D): Extent3D {
  const resolved = extent ?? [0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1];
  if (resolved.some((value) => !Number.isInteger(value))) {
    throw new Error('extent must contain six integers');
  }
  if (resolved[0] > resolved[1] || resolved[2] > resolved[3] || resolved[4] > resolved[5]) {
    throw new Error('extent bounds must be ordered');
  }
  if (resolved[0] < 0 || resolved[1] >= dims[0]
    || resolved[2] < 0 || resolved[3] >= dims[1]
    || resolved[4] < 0 || resolved[5] >= dims[2]) {
    throw new Error('extent must be within image dimensions');
  }
  return resolved;
}

function forEachExtentIndex(
  dims: readonly number[],
  extent: Extent3D,
  visit: (index: number) => void,
) {
  const rowStride = dims[0];
  const sliceStride = dims[0] * dims[1];
  for (let z = extent[4]; z <= extent[5]; z += 1) {
    for (let y = extent[2]; y <= extent[3]; y += 1) {
      let index = z * sliceStride + y * rowStride + extent[0];
      for (let x = extent[0]; x <= extent[1]; x += 1, index += 1) {
        visit(index);
      }
    }
  }
}

function copy<T extends ImageData>(data: T) {
  return data.slice() as T;
}

export function compositeImage<T extends ImageData>(
  input: T,
  modifier: T,
  dims: readonly number[],
  operation: CompositeOperation,
  extent?: Extent3D,
) {
  validateDimensions(dims);
  const expectedLength = dims[0] * dims[1] * dims[2];
  validateDataLength(input, expectedLength, 'Input');
  validateDataLength(modifier, expectedLength, 'Modifier');
  if (input.constructor !== modifier.constructor) {
    throw new Error('Input and modifier data types must match');
  }
  const boundedExtent = resolveExtent(dims, extent);
  const output = copy(input);

  forEachExtentIndex(dims, boundedExtent, (index) => {
    if (operation === 'set') {
      output[index] = modifier[index];
    } else if (operation === 'minimum') {
      output[index] = Math.min(input[index], modifier[index]);
    } else {
      output[index] = Math.max(input[index], modifier[index]);
    }
  });
  return output;
}

export function compositeSet<T extends ImageData>(
  input: T,
  modifier: T,
  dims: readonly number[],
  extent?: Extent3D,
) {
  return compositeImage(input, modifier, dims, 'set', extent);
}

export function compositeMin<T extends ImageData>(
  input: T,
  modifier: T,
  dims: readonly number[],
  extent?: Extent3D,
) {
  return compositeImage(input, modifier, dims, 'minimum', extent);
}

export function compositeMax<T extends ImageData>(
  input: T,
  modifier: T,
  dims: readonly number[],
  extent?: Extent3D,
) {
  return compositeImage(input, modifier, dims, 'maximum', extent);
}

export function maskByLabelValue<T extends ImageData>(
  input: T,
  mask: ImageData,
  dims: readonly number[],
  labelValue: number,
  fillValue = 0,
  extent?: Extent3D,
) {
  validateDimensions(dims);
  const expectedLength = dims[0] * dims[1] * dims[2];
  validateDataLength(input, expectedLength, 'Input');
  validateDataLength(mask, expectedLength, 'Mask');
  if (!Number.isFinite(labelValue) || !Number.isFinite(fillValue)) {
    throw new Error('labelValue and fillValue must be finite');
  }
  const boundedExtent = resolveExtent(dims, extent);
  const output = copy(input);

  forEachExtentIndex(dims, boundedExtent, (index) => {
    if (mask[index] !== labelValue) {
      output[index] = fillValue;
    }
  });
  return output;
}
