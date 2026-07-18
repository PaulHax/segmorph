export type ImageData =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export type ImageGeometry = {
  dims: readonly number[];
  spacing: readonly number[];
  origin: readonly number[];
  direction: readonly (readonly number[])[];
};

export type OrientedImage<T extends ImageData = ImageData> = ImageGeometry & {
  data: T;
};

export type Vector3 = readonly [number, number, number];

const directionTolerance = 1e-10;

function hasThreeFiniteValues(values: readonly number[]) {
  return values.length === 3 && values.every(Number.isFinite);
}

function dot(left: readonly number[], right: readonly number[]) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

export function validateImageGeometry(geometry: ImageGeometry) {
  if (geometry.dims.length !== 3
    || geometry.dims.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error('dims must contain three positive integers');
  }
  if (!hasThreeFiniteValues(geometry.spacing) || geometry.spacing.some((value) => value <= 0)) {
    throw new Error('spacing must contain three positive finite numbers');
  }
  if (!hasThreeFiniteValues(geometry.origin)) {
    throw new Error('origin must contain three finite numbers');
  }
  if (geometry.direction.length !== 3
    || geometry.direction.some((row) => !hasThreeFiniteValues(row))) {
    throw new Error('direction must be an orthonormal 3x3 matrix');
  }

  for (let row = 0; row < 3; row += 1) {
    for (let other = row; other < 3; other += 1) {
      const expected = row === other ? 1 : 0;
      if (Math.abs(dot(geometry.direction[row], geometry.direction[other]) - expected)
        > directionTolerance) {
        throw new Error('direction must be an orthonormal 3x3 matrix');
      }
    }
  }
}

export function createOrientedImage<T extends ImageData>(image: OrientedImage<T>) {
  validateImageGeometry(image);
  const expectedLength = image.dims[0] * image.dims[1] * image.dims[2];
  if (image.data.length !== expectedLength) {
    throw new Error(
      `Image data length ${image.data.length} does not match dimensions (expected ${expectedLength})`,
    );
  }
  return image;
}

/**
 * Determinant of the 3x3 direction matrix. Its sign is the geometry handedness:
 * a negative value marks a left-handed (mirrored) direction, which index-space
 * surface extractors use to decide whether to reverse triangle winding so
 * normals stay outward-facing in world space.
 */
export function directionDeterminant(direction: readonly (readonly number[])[]) {
  return direction[0][0] * (direction[1][1] * direction[2][2] - direction[1][2] * direction[2][1])
    - direction[0][1] * (direction[1][0] * direction[2][2] - direction[1][2] * direction[2][0])
    + direction[0][2] * (direction[1][0] * direction[2][1] - direction[1][1] * direction[2][0]);
}

export function indexToWorld(geometry: ImageGeometry, index: Vector3): Vector3 {
  const scaled = [
    index[0] * geometry.spacing[0],
    index[1] * geometry.spacing[1],
    index[2] * geometry.spacing[2],
  ];
  return [
    geometry.origin[0] + dot(geometry.direction[0], scaled),
    geometry.origin[1] + dot(geometry.direction[1], scaled),
    geometry.origin[2] + dot(geometry.direction[2], scaled),
  ];
}

export function worldToIndex(geometry: ImageGeometry, world: Vector3): Vector3 {
  const offset = [
    world[0] - geometry.origin[0],
    world[1] - geometry.origin[1],
    world[2] - geometry.origin[2],
  ];
  return [
    (geometry.direction[0][0] * offset[0]
      + geometry.direction[1][0] * offset[1]
      + geometry.direction[2][0] * offset[2]) / geometry.spacing[0],
    (geometry.direction[0][1] * offset[0]
      + geometry.direction[1][1] * offset[1]
      + geometry.direction[2][1] * offset[2]) / geometry.spacing[1],
    (geometry.direction[0][2] * offset[0]
      + geometry.direction[1][2] * offset[1]
      + geometry.direction[2][2] * offset[2]) / geometry.spacing[2],
  ];
}
