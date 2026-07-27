import { boundingBox, squaredPointTriangleDistance, triangles, vertices } from './mesh.js';
import type { Mesh, Point } from './mesh.js';

// A triangle as three world-space corners, matching what ./mesh.ts's triangles()
// yields. (Its exported `Triangle` is an index triple, a different thing.)
type Face = readonly [Point, Point, Point];

// The exhaustive metrics in ./mesh.ts compare every sample against every
// triangle. That is the right default for the small named cases -- it has no
// tuning and no approximation -- but it is quadratic, and a clinical-scale mesh
// (hundreds of thousands of triangles) puts it out of reach entirely.
//
// This module keeps the same distance definition and only changes how the
// nearest triangle is found: bin triangles into a uniform grid, then search
// cells outward from the query point and stop once the unsearched cells are
// provably farther than the best hit so far. The result is exact, not
// approximate -- the sampling below is the only approximation, and it is
// explicit in the caller's sample budget.

type TriangleGrid = {
  cellSize: number;
  min: Point;
  counts: [number, number, number];
  cells: Map<number, Face[]>;
};

function cellKey(x: number, y: number, z: number, counts: [number, number, number]) {
  return x + counts[0] * (y + counts[1] * z);
}

export function buildTriangleGrid(mesh: Mesh): TriangleGrid {
  const faces = triangles(mesh);
  if (faces.length === 0) throw new RangeError('Cannot index a mesh with no triangles');

  const { min, max } = boundingBox(mesh);
  const extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];

  // Target a handful of triangles per occupied cell: small enough that a query
  // scans few triangles, large enough that binning does not explode memory on
  // a degenerate (flat) mesh.
  const volume = Math.max(extent[0], 1e-9) * Math.max(extent[1], 1e-9) * Math.max(extent[2], 1e-9);
  const cellSize = Math.max(Math.cbrt((volume * 4) / faces.length), 1e-9);

  const counts = extent.map((length) => Math.max(1, Math.ceil(length / cellSize) + 1)) as [
    number,
    number,
    number,
  ];

  const cells = new Map<number, Face[]>();
  const index = (value: number, axis: number) =>
    Math.min(counts[axis] - 1, Math.max(0, Math.floor((value - min[axis]) / cellSize)));

  for (const triangle of faces) {
    const lo = [0, 1, 2].map((axis) =>
      index(Math.min(triangle[0][axis], triangle[1][axis], triangle[2][axis]), axis),
    );
    const hi = [0, 1, 2].map((axis) =>
      index(Math.max(triangle[0][axis], triangle[1][axis], triangle[2][axis]), axis),
    );
    for (let z = lo[2]; z <= hi[2]; z += 1) {
      for (let y = lo[1]; y <= hi[1]; y += 1) {
        for (let x = lo[0]; x <= hi[0]; x += 1) {
          const key = cellKey(x, y, z, counts);
          const bucket = cells.get(key);
          if (bucket) bucket.push(triangle);
          else cells.set(key, [triangle]);
        }
      }
    }
  }

  return { cellSize, min: min as Point, counts, cells };
}

export function nearestTriangleDistance(grid: TriangleGrid, point: Point) {
  const { cellSize, min, counts, cells } = grid;
  const home = [0, 1, 2].map((axis) =>
    Math.min(counts[axis] - 1, Math.max(0, Math.floor((point[axis] - min[axis]) / cellSize))),
  );

  let bestSquared = Infinity;
  const maxRing = Math.max(...counts);

  for (let ring = 0; ring <= maxRing; ring += 1) {
    // Everything outside the searched box is at least this far away, so once
    // the best hit beats it no wider ring can improve on it.
    if (ring > 0 && bestSquared !== Infinity) {
      const guaranteed = (ring - 1) * cellSize;
      if (guaranteed > 0 && guaranteed * guaranteed > bestSquared) break;
    }

    let searchedAny = false;
    for (let z = home[2] - ring; z <= home[2] + ring; z += 1) {
      if (z < 0 || z >= counts[2]) continue;
      for (let y = home[1] - ring; y <= home[1] + ring; y += 1) {
        if (y < 0 || y >= counts[1]) continue;
        for (let x = home[0] - ring; x <= home[0] + ring; x += 1) {
          if (x < 0 || x >= counts[0]) continue;
          // Only the shell of the box is new on this ring.
          const onShell =
            ring === 0 ||
            Math.abs(x - home[0]) === ring ||
            Math.abs(y - home[1]) === ring ||
            Math.abs(z - home[2]) === ring;
          if (!onShell) continue;
          searchedAny = true;

          const bucket = cells.get(cellKey(x, y, z, counts));
          if (!bucket) continue;
          for (const triangle of bucket) {
            const squared = squaredPointTriangleDistance(point, triangle);
            // A degenerate target triangle can divide by zero in the
            // barycentric math; skipping keeps NaN out of the running minimum.
            if (Number.isFinite(squared) && squared < bestSquared) bestSquared = squared;
          }
        }
      }
    }
    if (!searchedAny && ring > 0 && bestSquared !== Infinity) break;
  }

  if (bestSquared === Infinity) throw new RangeError('No triangle found for query point');
  return Math.sqrt(bestSquared);
}

// Deterministic PRNG so a sampled metric is reproducible run to run: an
// unseeded sample would make a borderline tolerance flake.
const mulberry32 = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

function sampleVertices(
  mesh: Mesh,
  maxSamples: number,
  seed: number,
  accept?: (point: Point) => boolean,
) {
  const points = accept ? vertices(mesh).filter(accept) : vertices(mesh);
  if (points.length === 0) throw new RangeError('No vertices left to sample');
  if (points.length <= maxSamples) return points;
  const random = mulberry32(seed);
  const stride = points.length / maxSamples;
  return Array.from(
    { length: maxSamples },
    (_, index) => points[Math.min(points.length - 1, Math.floor((index + random()) * stride))],
  );
}

/**
 * Symmetric surface distance between two large meshes, measured from a seeded
 * sample of each mesh's vertices against a grid index of the other. Exact per
 * sampled point; the sample budget bounds the cost.
 *
 * `accept` restricts which vertices are sampled, which is how a caller compares
 * two meshes that legitimately differ in one region -- measuring the rest
 * tightly instead of loosening the tolerance until the known difference fits.
 */
export function sampledSurfaceDistances(
  a: Mesh,
  b: Mesh,
  {
    maxSamples = 20_000,
    seed = 0,
    accept,
  }: { maxSamples?: number; seed?: number; accept?: (point: Point) => boolean } = {},
) {
  const gridA = buildTriangleGrid(a);
  const gridB = buildTriangleGrid(b);
  const distances = [
    ...sampleVertices(a, maxSamples, seed, accept).map((point) =>
      nearestTriangleDistance(gridB, point),
    ),
    ...sampleVertices(b, maxSamples, seed + 1, accept).map((point) =>
      nearestTriangleDistance(gridA, point),
    ),
  ];

  return {
    hausdorff: distances.reduce((max, distance) => Math.max(max, distance), 0),
    mean: distances.reduce((sum, distance) => sum + distance, 0) / distances.length,
    sampleCount: distances.length,
  };
}
