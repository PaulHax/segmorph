export type Mesh = {
  points: Float32Array;
  polys: Uint32Array;
};

export type Point = readonly [number, number, number];

export function vertices(mesh: Mesh) {
  if (mesh.points.length % 3 !== 0) {
    throw new RangeError('Mesh points must contain complete xyz coordinates');
  }

  const result: Point[] = [];
  for (let index = 0; index < mesh.points.length; index += 3) {
    result.push([mesh.points[index], mesh.points[index + 1], mesh.points[index + 2]]);
  }
  return result;
}

export type Triangle = readonly [number, number, number];

export function triangleIndices(mesh: Mesh) {
  const vertexCount = mesh.points.length / 3;
  if (!Number.isInteger(vertexCount)) {
    throw new RangeError('Mesh points must contain complete xyz coordinates');
  }

  const result: Triangle[] = [];
  for (let offset = 0; offset < mesh.polys.length;) {
    const count = mesh.polys[offset];
    if (count !== 3 || offset + count >= mesh.polys.length) {
      throw new RangeError('Mesh polys must contain complete triangle cells');
    }
    const triangle = [
      mesh.polys[offset + 1],
      mesh.polys[offset + 2],
      mesh.polys[offset + 3],
    ] as const;
    if (triangle.some((index) => index >= vertexCount)) {
      throw new RangeError('Mesh polygon index is out of bounds');
    }
    result.push(triangle);
    offset += count + 1;
  }
  return result;
}

export function triangles(mesh: Mesh) {
  const points = vertices(mesh);
  return triangleIndices(mesh).map(([a, b, c]) => [points[a], points[b], points[c]] as const);
}

function subtract(a: Point, b: Point): Point {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: Point, b: Point) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function squaredDistance(a: Point, b: Point) {
  const difference = subtract(a, b);
  return dot(difference, difference);
}

function addScaled(origin: Point, direction: Point, t: number): Point {
  return [
    origin[0] + t * direction[0],
    origin[1] + t * direction[1],
    origin[2] + t * direction[2],
  ];
}

// Closest-point regions from Real-Time Collision Detection, Christer Ericson.
export function squaredPointTriangleDistance(point: Point, [a, b, c]: readonly [Point, Point, Point]) {
  const ab = subtract(b, a);
  const ac = subtract(c, a);
  const ap = subtract(point, a);
  const d1 = dot(ab, ap);
  const d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return squaredDistance(point, a);

  const bp = subtract(point, b);
  const d3 = dot(ab, bp);
  const d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return squaredDistance(point, b);

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return squaredDistance(point, addScaled(a, ab, v));
  }

  const cp = subtract(point, c);
  const d5 = dot(ab, cp);
  const d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return squaredDistance(point, c);

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return squaredDistance(point, addScaled(a, ac, w));
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + d5 - d6);
    return squaredDistance(point, addScaled(b, subtract(c, b), w));
  }

  const denominator = 1 / (va + vb + vc);
  const v = vb * denominator;
  const w = vc * denominator;
  return squaredDistance(point, addScaled(addScaled(a, ab, v), ac, w));
}

// Barycentric weights for interior sample points. Vertices are sampled
// separately (they are shared across triangles), so this adds each triangle's
// centroid: the farthest interior point of a face that bows across a concavity.
// Without it a coarse triangle whose corners lie on the target surface but whose
// face cuts across the true surface reports a near-zero distance, hiding real
// error. One sample per triangle keeps the O(samples x triangles) sweep within
// the fine-mesh oracle comparisons' time budget.
const INTERIOR_BARYCENTRICS = [
  [1 / 3, 1 / 3, 1 / 3],
] as const;

function barycentricPoint(
  [a, b, c]: readonly [Point, Point, Point],
  u: number,
  v: number,
  w: number,
): Point {
  return [
    u * a[0] + v * b[0] + w * c[0],
    u * a[1] + v * b[1] + w * c[1],
    u * a[2] + v * b[2] + w * c[2],
  ];
}

function surfaceSamples(mesh: Mesh) {
  const samples = vertices(mesh);
  for (const triangle of triangles(mesh)) {
    for (const [u, v, w] of INTERIOR_BARYCENTRICS) {
      samples.push(barycentricPoint(triangle, u, v, w));
    }
  }
  return samples;
}

function directedDistances(source: Mesh, target: Mesh) {
  const samples = surfaceSamples(source);
  const surfaces = triangles(target);
  if (samples.length === 0 || surfaces.length === 0) {
    throw new RangeError('Surface distances require vertices and triangles');
  }

  // Skip non-finite distances: a degenerate (zero-area) target triangle can
  // divide by zero in the barycentric math and yield NaN, which would otherwise
  // poison the running minimum and silently void the whole metric.
  return samples.map((point) => Math.sqrt(surfaces.reduce(
    (nearest, triangle) => {
      const squared = squaredPointTriangleDistance(point, triangle);
      return Number.isFinite(squared) ? Math.min(nearest, squared) : nearest;
    },
    Infinity,
  )));
}

function symmetricDistances(a: Mesh, b: Mesh) {
  return [...directedDistances(a, b), ...directedDistances(b, a)];
}

export function symmetricHausdorffDistance(a: Mesh, b: Mesh) {
  return symmetricDistances(a, b).reduce((max, distance) => Math.max(max, distance), 0);
}

export function meanSurfaceDistance(a: Mesh, b: Mesh) {
  const distances = symmetricDistances(a, b);
  return distances.reduce((sum, distance) => sum + distance, 0) / distances.length;
}

export function enclosedVolume(mesh: Mesh) {
  return triangles(mesh).reduce((volume, [a, b, c]) => volume + dot(a, [
    b[1] * c[2] - b[2] * c[1],
    b[2] * c[0] - b[0] * c[2],
    b[0] * c[1] - b[1] * c[0],
  ]) / 6, 0);
}

export function boundingBox(mesh: Mesh) {
  const points = vertices(mesh);
  if (points.length === 0) throw new RangeError('Bounding box requires at least one vertex');

  const min = [...points[0]] as [number, number, number];
  const max = [...points[0]] as [number, number, number];
  for (const point of points) {
    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], point[axis]);
      max[axis] = Math.max(max[axis], point[axis]);
    }
  }
  return { min, max };
}
