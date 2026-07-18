export type Dimensions3D = readonly [number, number, number];
export type Index3D = readonly [number, number, number];

/** Inclusive bounds in [xMin, xMax, yMin, yMax, zMin, zMax] order. */
export type Extent3D = readonly [number, number, number, number, number, number];

export const EMPTY_EXTENT: Extent3D = Object.freeze([0, -1, 0, -1, 0, -1]);

function validateDims(dims: Dimensions3D) {
  if (dims.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error('dims must contain three positive integers');
  }
}

function isCanonicalEmpty(extent: Extent3D) {
  return extent.every((value, position) => value === EMPTY_EXTENT[position]);
}

export function isExtentEmpty(extent: Extent3D) {
  return extent[0] > extent[1] || extent[2] > extent[3] || extent[4] > extent[5];
}

export function extentFromDims(dims: Dimensions3D): Extent3D {
  validateDims(dims);
  return [0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1];
}

export function validateExtent(extent: Extent3D, dims: Dimensions3D) {
  validateDims(dims);
  if (extent.some((value) => !Number.isInteger(value))) {
    throw new Error('extent values must be integers');
  }
  if (isCanonicalEmpty(extent)) {
    return EMPTY_EXTENT;
  }
  if (isExtentEmpty(extent)) {
    throw new Error('extent bounds must be ordered');
  }
  if (extent[0] < 0 || extent[1] >= dims[0]
    || extent[2] < 0 || extent[3] >= dims[1]
    || extent[4] < 0 || extent[5] >= dims[2]) {
    throw new Error('extent must be within image dimensions');
  }
  return extent;
}

export function intersectExtents(left: Extent3D, right: Extent3D): Extent3D {
  if (isExtentEmpty(left) || isExtentEmpty(right)) {
    return EMPTY_EXTENT;
  }
  const intersection: Extent3D = [
    Math.max(left[0], right[0]), Math.min(left[1], right[1]),
    Math.max(left[2], right[2]), Math.min(left[3], right[3]),
    Math.max(left[4], right[4]), Math.min(left[5], right[5]),
  ];
  return isExtentEmpty(intersection) ? EMPTY_EXTENT : intersection;
}

export function isIndexInExtent(index: Index3D, extent: Extent3D) {
  return index.every(Number.isInteger)
    && !isExtentEmpty(extent)
    && index[0] >= extent[0] && index[0] <= extent[1]
    && index[1] >= extent[2] && index[1] <= extent[3]
    && index[2] >= extent[4] && index[2] <= extent[5];
}

export function flatIndex(index: Index3D, dims: Dimensions3D) {
  validateDims(dims);
  if (index.some((value) => !Number.isInteger(value))) {
    throw new Error('index values must be integers');
  }
  if (!isIndexInExtent(index, extentFromDims(dims))) {
    throw new Error('index must be within image dimensions');
  }
  return index[0] + dims[0] * (index[1] + dims[1] * index[2]);
}

export function* iterateExtentIndices(extent: Extent3D): Generator<Index3D> {
  if (isExtentEmpty(extent)) {
    return;
  }
  for (let z = extent[4]; z <= extent[5]; z += 1) {
    for (let y = extent[2]; y <= extent[3]; y += 1) {
      for (let x = extent[0]; x <= extent[1]; x += 1) {
        yield [x, y, z];
      }
    }
  }
}
