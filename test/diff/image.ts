export type ImageDimensions = readonly [number, number, number];

export type NumericTypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export type VoxelCoordinate = readonly [number, number, number];

function voxelCount(
  actual: NumericTypedArray,
  expected: NumericTypedArray,
  dims: ImageDimensions,
) {
  if (dims.some((size) => !Number.isInteger(size) || size < 0)) {
    throw new RangeError('Image dimensions must be non-negative integers');
  }

  const count = dims[0] * dims[1] * dims[2];
  if (actual.length !== count || expected.length !== count) {
    throw new RangeError('Image data length must equal the dimensions product');
  }

  return count;
}

// Intersection counts only voxels whose labels *match* (both non-zero and
// equal), not merely both-foreground. A label 1 vs label 2 swap therefore
// scores zero overlap instead of a perfect match, so the metrics stay
// sensitive to label errors, not just presence/absence.
function overlapCounts(
  actual: NumericTypedArray,
  expected: NumericTypedArray,
  dims: ImageDimensions,
) {
  const count = voxelCount(actual, expected, dims);
  let actualForeground = 0;
  let expectedForeground = 0;
  let intersection = 0;

  for (let index = 0; index < count; index += 1) {
    const actualSet = actual[index] !== 0;
    const expectedSet = expected[index] !== 0;
    actualForeground += Number(actualSet);
    expectedForeground += Number(expectedSet);
    intersection += Number(actualSet && actual[index] === expected[index]);
  }

  return { actualForeground, expectedForeground, intersection };
}

// Empty-vs-empty returns NaN rather than 1: two silently-empty labelmaps are a
// degenerate comparison, not a perfect match. NaN fails any `toBeGreaterThan`
// threshold, forcing callers to assert non-emptiness explicitly instead of
// letting an empty conversion sail through as a 1.0.
export function dice(
  actual: NumericTypedArray,
  expected: NumericTypedArray,
  dims: ImageDimensions,
) {
  const { actualForeground, expectedForeground, intersection } = overlapCounts(
    actual,
    expected,
    dims,
  );
  const foregroundTotal = actualForeground + expectedForeground;

  return foregroundTotal === 0 ? NaN : (2 * intersection) / foregroundTotal;
}

export function iou(
  actual: NumericTypedArray,
  expected: NumericTypedArray,
  dims: ImageDimensions,
) {
  const { actualForeground, expectedForeground, intersection } = overlapCounts(
    actual,
    expected,
    dims,
  );
  const union = actualForeground + expectedForeground - intersection;

  return union === 0 ? NaN : intersection / union;
}

export function mismatchCount(
  actual: NumericTypedArray,
  expected: NumericTypedArray,
  dims: ImageDimensions,
) {
  return mismatchingVoxelCoordinates(actual, expected, dims).length;
}

export function mismatchingVoxelCoordinates(
  actual: NumericTypedArray,
  expected: NumericTypedArray,
  dims: ImageDimensions,
) {
  const count = voxelCount(actual, expected, dims);
  const coordinates: VoxelCoordinate[] = [];
  const [width, height] = dims;

  for (let index = 0; index < count; index += 1) {
    if (actual[index] === expected[index]) {
      continue;
    }

    const x = index % width;
    const y = Math.floor(index / width) % height;
    const z = Math.floor(index / (width * height));
    coordinates.push([x, y, z]);
  }

  return coordinates;
}
