export type Point = readonly [number, number, number];
export type Triangle = readonly [number, number, number];

export type Mesh = {
  points: Float32Array;
  polys: Uint32Array;
};

function validatePoints(points: Float32Array) {
  if (points.length % 3 !== 0) {
    throw new RangeError('Mesh points must contain complete xyz coordinates');
  }
  if (points.some((coordinate) => !Number.isFinite(coordinate))) {
    throw new RangeError('Mesh point coordinates must be finite');
  }
}

function* uncheckedTriangles(mesh: Mesh): Generator<Triangle> {
  const count = mesh.points.length / 3;
  for (let offset = 0; offset < mesh.polys.length;) {
    const cellSize = mesh.polys[offset];
    if (cellSize !== 3 || offset + cellSize >= mesh.polys.length) {
      throw new RangeError('Mesh polys must contain complete triangle cells');
    }

    const triangle = [
      mesh.polys[offset + 1],
      mesh.polys[offset + 2],
      mesh.polys[offset + 3],
    ] as const;
    if (triangle.some((index) => index >= count)) {
      throw new RangeError('Mesh polygon index is out of bounds');
    }

    yield triangle;
    offset += cellSize + 1;
  }
}

export function validateMesh(mesh: Mesh) {
  validatePoints(mesh.points);
  for (const _triangle of uncheckedTriangles(mesh)) {
    // Iteration validates every VTK-style cell and vertex index.
  }
}

export function vertexCount(mesh: Mesh) {
  validatePoints(mesh.points);
  return mesh.points.length / 3;
}

export function triangleCount(mesh: Mesh) {
  validatePoints(mesh.points);
  let count = 0;
  for (const _triangle of uncheckedTriangles(mesh)) count += 1;
  return count;
}

export function getPoint(mesh: Mesh, index: number): Point {
  const count = vertexCount(mesh);
  if (!Number.isInteger(index) || index < 0 || index >= count) {
    throw new RangeError('Mesh point index is out of bounds');
  }
  const offset = index * 3;
  return [mesh.points[offset], mesh.points[offset + 1], mesh.points[offset + 2]];
}

export function* iteratePoints(mesh: Mesh): Generator<Point> {
  const count = vertexCount(mesh);
  for (let index = 0; index < count; index += 1) yield getPoint(mesh, index);
}

export function* iterateTriangles(mesh: Mesh): Generator<Triangle> {
  validatePoints(mesh.points);
  yield* uncheckedTriangles(mesh);
}

export function createMesh(points: readonly Point[], triangles: readonly Triangle[]): Mesh {
  const flatPoints = new Float32Array(points.length * 3);
  for (let index = 0; index < points.length; index += 1) {
    flatPoints.set(points[index], index * 3);
  }

  const flatPolys = new Uint32Array(triangles.length * 4);
  for (let index = 0; index < triangles.length; index += 1) {
    const triangle = triangles[index];
    if (triangle.some((vertex) => !Number.isInteger(vertex) || vertex < 0)) {
      throw new RangeError('Mesh polygon indices must be non-negative integers');
    }
    flatPolys.set([3, ...triangle], index * 4);
  }

  const mesh = { points: flatPoints, polys: flatPolys };
  validateMesh(mesh);
  return mesh;
}
