import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { surfaceToContour } from '../src/convert/surfaceToContour.js';
import {
  createContourPlane,
  planeToWorld,
  type ContourPlane,
  type PlanarContour,
  type Vector3,
} from '../src/geometry/contour.js';
import { createMesh, type Point } from '../src/geometry/mesh.js';
import { readMeshJson } from './fixtures/loaders.js';

const sphereMeshUrl = new URL('./fixtures/A/sphere/golden.mesh.json', import.meta.url);

const cubeTriangles = [
  [0, 2, 1],
  [0, 3, 2],
  [4, 5, 6],
  [4, 6, 7],
  [0, 1, 5],
  [0, 5, 4],
  [3, 7, 6],
  [3, 6, 2],
  [0, 4, 7],
  [0, 7, 3],
  [1, 2, 6],
  [1, 6, 5],
] as const;

function cube(minimum: number, maximum: number) {
  const points: Point[] = [
    [minimum, minimum, minimum],
    [maximum, minimum, minimum],
    [maximum, maximum, minimum],
    [minimum, maximum, minimum],
    [minimum, minimum, maximum],
    [maximum, minimum, maximum],
    [maximum, maximum, maximum],
    [minimum, maximum, maximum],
  ];
  return createMesh(points, cubeTriangles);
}

const octahedronTriangles = [
  [0, 2, 3],
  [0, 3, 4],
  [0, 4, 5],
  [0, 5, 2],
  [1, 3, 2],
  [1, 4, 3],
  [1, 5, 4],
  [1, 2, 5],
] as const;

function octahedron(scale: Vector3 = [1, 1, 1], offset: Vector3 = [0, 0, 0]) {
  const base: Point[] = [
    [0, 0, 1],
    [0, 0, -1],
    [1, 0, 0],
    [0, 1, 0],
    [-1, 0, 0],
    [0, -1, 0],
  ];
  const points = base.map(
    ([x, y, z]): Point => [
      x * scale[0] + offset[0],
      y * scale[1] + offset[1],
      z * scale[2] + offset[2],
    ],
  );
  return createMesh(points, octahedronTriangles);
}

function axisPlane(z: number): ContourPlane {
  return createContourPlane([0, 0, z], [1, 0, 0], [0, 1, 0]);
}

function normalize(vector: Vector3): Vector3 {
  const length = Math.hypot(...vector);
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function cross(a: Vector3, b: Vector3): Vector3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function loopVertices(loop: { points: Float64Array }) {
  const vertices: [number, number][] = [];
  for (let index = 0; index < loop.points.length; index += 2) {
    vertices.push([loop.points[index], loop.points[index + 1]]);
  }
  return vertices;
}

function signedArea(loop: { points: Float64Array }) {
  const vertices = loopVertices(loop);
  let area = 0;
  for (let index = 0; index < vertices.length; index += 1) {
    const [x0, y0] = vertices[index];
    const [x1, y1] = vertices[(index + 1) % vertices.length];
    area += x0 * y1 - x1 * y0;
  }
  return area / 2;
}

function maxPlaneDistance(contour: PlanarContour) {
  const { plane } = contour;
  const normal = cross(plane.xAxis, plane.yAxis);
  let maximum = 0;
  for (const loop of contour.loops) {
    for (const [x, y] of loopVertices(loop)) {
      const world = planeToWorld(plane, [x, y]);
      const offset: Vector3 = [
        world[0] - plane.origin[0],
        world[1] - plane.origin[1],
        world[2] - plane.origin[2],
      ];
      const distance = Math.abs(
        offset[0] * normal[0] + offset[1] * normal[1] + offset[2] * normal[2],
      );
      maximum = Math.max(maximum, distance);
    }
  }
  return maximum;
}

function segmentsIntersect(
  a0: [number, number],
  a1: [number, number],
  b0: [number, number],
  b1: [number, number],
) {
  const orient = (p: [number, number], q: [number, number], r: [number, number]) =>
    Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
  return orient(a0, a1, b0) !== orient(a0, a1, b1) && orient(b0, b1, a0) !== orient(b0, b1, a1);
}

function isSimpleLoop(loop: { points: Float64Array }) {
  const vertices = loopVertices(loop);
  const count = vertices.length;
  for (let i = 0; i < count; i += 1) {
    for (let j = i + 2; j < count; j += 1) {
      if (i === 0 && j === count - 1) continue; // adjacent through closure
      if (
        segmentsIntersect(
          vertices[i],
          vertices[(i + 1) % count],
          vertices[j],
          vertices[(j + 1) % count],
        )
      ) {
        return false;
      }
    }
  }
  return true;
}

describe('surfaceToContour', () => {
  it('returns undefined when the plane misses the mesh', () => {
    expect(surfaceToContour(cube(0, 2), axisPlane(5))).toBeUndefined();
    expect(surfaceToContour(cube(0, 2), axisPlane(-1))).toBeUndefined();
  });

  it('treats points exactly on the plane as below (no intersection for tangent planes)', () => {
    // The plane z = 2 touches the cube's top face; every point evaluates <= 0.
    expect(surfaceToContour(cube(0, 2), axisPlane(2))).toBeUndefined();
  });

  it('drops open chains from meshes with boundaries', () => {
    const triangle = createMesh(
      [
        [0, 0, -1],
        [2, 0, 1],
        [0, 2, 1],
      ],
      [[0, 1, 2]],
    );
    expect(surfaceToContour(triangle, axisPlane(0))).toBeUndefined();
  });

  it('cuts a cube into a single closed square loop', () => {
    const contour = surfaceToContour(cube(0, 2), axisPlane(1));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(1);
    // Exact arithmetic: cut points sit on cube edges, so the area is exact.
    expect(signedArea(contour!.loops[0])).toBeCloseTo(4, 12);
    expect(isSimpleLoop(contour!.loops[0])).toBe(true);
    // Measured plane distance 0 for axis-aligned exact cuts.
    expect(maxPlaneDistance(contour!)).toBeLessThanOrEqual(1e-12);
  });

  it('orients every loop counterclockwise about xAxis cross yAxis', () => {
    const two = createMesh(
      [
        [0, 0, 1],
        [0, 0, -1],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [5, 0, 1],
        [5, 0, -1],
        [6, 0, 0],
        [5, 1, 0],
        [4, 0, 0],
        [5, -1, 0],
      ],
      [
        ...octahedronTriangles,
        ...octahedronTriangles.map(([a, b, c]) => [a + 6, b + 6, c + 6] as const),
      ],
    );
    const contour = surfaceToContour(two, axisPlane(0.25));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(2);
    for (const loop of contour!.loops) {
      expect(signedArea(loop)).toBeGreaterThan(0);
    }
  });

  it('passes exactly through on-plane vertices', () => {
    // Equator vertices sit exactly on z = 0; they classify as below, the top
    // apex is above, so only the four top faces are cut and the loop runs
    // exactly through the equator vertices.
    const contour = surfaceToContour(octahedron(), axisPlane(0));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(1);
    expect(signedArea(contour!.loops[0])).toBeCloseTo(2, 12);
    const radii = loopVertices(contour!.loops[0]).map(([x, y]) => Math.hypot(x, y));
    for (const radius of radii) expect(radius).toBeCloseTo(1, 12);
  });

  it('handles anisotropically scaled meshes', () => {
    const contour = surfaceToContour(octahedron([3, 0.5, 1], [10, -4, 2]), axisPlane(2));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(1);
    expect(signedArea(contour!.loops[0])).toBeCloseTo(2 * 3 * 0.5, 6);
    expect(isSimpleLoop(contour!.loops[0])).toBe(true);
  });

  it('cuts a cube with an oblique plane into a hexagon', () => {
    const normal = normalize([1, 1, 1]);
    const xAxis = normalize(cross(normal, [0, 0, 1]));
    const yAxis = cross(normal, xAxis);
    const plane = createContourPlane([1, 1, 1], xAxis, yAxis);
    const contour = surfaceToContour(cube(0, 2), plane);
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(1);
    // Regular hexagon cross-section of a cube with side 2: area = 3 * sqrt(3).
    expect(Math.abs(signedArea(contour!.loops[0]))).toBeCloseTo(3 * Math.sqrt(3), 6);
    expect(isSimpleLoop(contour!.loops[0])).toBe(true);
    // Measured max plane distance 2.3e-16 on this case; float32 mesh points
    // keep interpolated cut points within ~1e-7 of the plane in general.
    expect(maxPlaneDistance(contour!)).toBeLessThanOrEqual(1e-7);
  });

  it('cuts the sphere fixture through its center into one analytic circle', async () => {
    const mesh = readMeshJson(await readFile(sphereMeshUrl, 'utf8'));
    const center: Vector3 = [15.5, 15.5, 15.5];
    const plane = createContourPlane(center, [1, 0, 0], [0, 1, 0]);
    const contour = surfaceToContour(mesh, plane);
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(1);
    expect(isSimpleLoop(contour!.loops[0])).toBe(true);
    expect(maxPlaneDistance(contour!)).toBeLessThanOrEqual(1e-6);

    // The smoothed flying-edges sphere of voxel radius 10 measures loop radii
    // in [9.9724, 10.0751] on this cut; assert a band just outside it.
    const radii = loopVertices(contour!.loops[0]).map(([x, y]) => Math.hypot(x, y));
    for (const radius of radii) {
      expect(radius).toBeGreaterThan(9.9);
      expect(radius).toBeLessThan(10.15);
    }

    const area = signedArea(contour!.loops[0]);
    expect(area).toBeGreaterThan(Math.PI * 9.9 ** 2);
    expect(area).toBeLessThan(Math.PI * 10.15 ** 2);
  });
});
