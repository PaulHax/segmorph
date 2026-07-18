export type Vector2 = readonly [number, number];
export type Vector3 = readonly [number, number, number];

export type ContourPlane = {
  origin: Vector3;
  xAxis: Vector3;
  yAxis: Vector3;
};

export type ContourLoop = {
  /** Interleaved planar xy coordinates. The final vertex connects to the first. */
  points: Float64Array;
};

export type PlanarContour = {
  plane: ContourPlane;
  loops: readonly ContourLoop[];
};

const orthonormalTolerance = 1e-10;

function dot(left: Vector3, right: Vector3) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function hasThreeFiniteValues(vector: readonly number[]): vector is Vector3 {
  return vector.length === 3 && vector.every(Number.isFinite);
}

export function validateContourPlane(plane: ContourPlane) {
  if (!hasThreeFiniteValues(plane.origin)) {
    throw new RangeError('Contour plane origin must contain three finite coordinates');
  }
  if (!hasThreeFiniteValues(plane.xAxis) || !hasThreeFiniteValues(plane.yAxis)
    || Math.abs(dot(plane.xAxis, plane.xAxis) - 1) > orthonormalTolerance
    || Math.abs(dot(plane.yAxis, plane.yAxis) - 1) > orthonormalTolerance
    || Math.abs(dot(plane.xAxis, plane.yAxis)) > orthonormalTolerance) {
    throw new RangeError('Contour plane axes must form an orthonormal basis');
  }
}

export function createContourPlane(
  origin: Vector3,
  xAxis: Vector3,
  yAxis: Vector3,
): ContourPlane {
  const plane = {
    origin: [...origin] as Vector3,
    xAxis: [...xAxis] as Vector3,
    yAxis: [...yAxis] as Vector3,
  };
  validateContourPlane(plane);
  return plane;
}

export function planeToWorld(plane: ContourPlane, point: Vector2): Vector3 {
  validateContourPlane(plane);
  if (!point.every(Number.isFinite)) {
    throw new RangeError('Plane point must contain two finite coordinates');
  }
  return [
    plane.origin[0] + point[0] * plane.xAxis[0] + point[1] * plane.yAxis[0],
    plane.origin[1] + point[0] * plane.xAxis[1] + point[1] * plane.yAxis[1],
    plane.origin[2] + point[0] * plane.xAxis[2] + point[1] * plane.yAxis[2],
  ];
}

export function worldToPlane(plane: ContourPlane, point: Vector3): Vector2 {
  validateContourPlane(plane);
  if (!hasThreeFiniteValues(point)) {
    throw new RangeError('World point must contain three finite coordinates');
  }
  const offset: Vector3 = [
    point[0] - plane.origin[0],
    point[1] - plane.origin[1],
    point[2] - plane.origin[2],
  ];
  return [dot(offset, plane.xAxis), dot(offset, plane.yAxis)];
}

export function validateContourLoop(loop: ContourLoop) {
  if (loop.points.length % 2 !== 0) {
    throw new RangeError('Contour loop points must contain complete xy coordinates');
  }
  if (loop.points.length < 6) {
    throw new RangeError('Contour loop must contain at least three vertices');
  }
  if (loop.points.some((coordinate) => !Number.isFinite(coordinate))) {
    throw new RangeError('Contour loop coordinates must be finite');
  }
}

export function createContourLoop(points: readonly Vector2[]): ContourLoop {
  const flattened = new Float64Array(points.length * 2);
  for (let index = 0; index < points.length; index += 1) {
    flattened.set(points[index], index * 2);
  }
  const loop = { points: flattened };
  validateContourLoop(loop);
  return loop;
}

/**
 * Flatten a planar contour's loops into world-space polylines (interleaved
 * xyz, implicit closure) - the shape contourToSurface and DICOM RTSTRUCT
 * adapters consume.
 */
export function planarContourWorldLoops(contour: PlanarContour): Float64Array[] {
  validatePlanarContour(contour);
  const { origin, xAxis, yAxis } = contour.plane;
  return contour.loops.map((loop) => {
    const vertexCount = loop.points.length / 2;
    const world = new Float64Array(vertexCount * 3);
    for (let vertex = 0; vertex < vertexCount; vertex += 1) {
      const u = loop.points[vertex * 2];
      const v = loop.points[vertex * 2 + 1];
      world[vertex * 3] = origin[0] + u * xAxis[0] + v * yAxis[0];
      world[vertex * 3 + 1] = origin[1] + u * xAxis[1] + v * yAxis[1];
      world[vertex * 3 + 2] = origin[2] + u * xAxis[2] + v * yAxis[2];
    }
    return world;
  });
}

export function validatePlanarContour(contour: PlanarContour) {
  validateContourPlane(contour.plane);
  if (contour.loops.length === 0) {
    throw new RangeError('Planar contour must contain at least one loop');
  }
  contour.loops.forEach(validateContourLoop);
}

export function createPlanarContour(
  plane: ContourPlane,
  loops: readonly ContourLoop[],
): PlanarContour {
  const contour = { plane, loops: [...loops] };
  validatePlanarContour(contour);
  return contour;
}
