// Port of VTK's vtkSurfaceNets3D (BSD-3-Clause, Filters/Core/vtkSurfaceNets3D.cxx)
// and its built-in vtkConstrainedSmoothingFilter, single-label case.
//
// The input is conceptually padded with one background voxel per side before
// extraction (matching the vtkImageConstantPad step used by the oracle and the
// A-queue pipeline). Raw vtkSurfaceNets3D treats volume-border edges as
// non-intersecting, which leaves boundary-touching foreground open; the pad
// closes it and is a geometric no-op for interior foreground.

import type { Mesh } from '../geometry/mesh.js';
import { indexToWorld, type OrientedImage } from '../image/orientedImage.js';

export type SurfaceNetsOptions = {
  labelValue: number;
  /** Run the constrained smoothing pass. vtkSurfaceNets3D default: true. */
  smoothing?: boolean;
  /** vtkSurfaceNets3D configures its smoother with 16 iterations (vtkSurfaceNets3D.cxx:2141). */
  smoothingIterations?: number;
  /** vtkSurfaceNets3D configures relaxation 0.5 (vtkSurfaceNets3D.cxx:2142). */
  relaxationFactor?: number;
  /**
   * Radius of the sphere around each unsmoothed point that constrains
   * smoothing. VTK computes (norm(spacing) / 2) * ConstraintScale with
   * ConstraintScale = 2, i.e. norm(spacing) (vtkSurfaceNets3D.cxx:1767, 2145).
   */
  constraintDistance?: number;
};

const backgroundLabel = 0;

// Triad classification bits (vtkSurfaceNets3D.cxx:182-192).
const inside = 1;
const xIntersection = 2;
const yIntersection = 4;
const zIntersection = 8;

// Each intersected voxel-cell edge activates its two adjacent cell faces,
// faces ordered -x,+x,-y,+y,-z,+z (GetFaceCase, vtkSurfaceNets3D.cxx:238-293).
const edgeFaces = [
  [2, 4],
  [3, 4],
  [2, 5],
  [3, 5],
  [0, 4],
  [1, 4],
  [0, 5],
  [1, 5],
  [0, 2],
  [1, 2],
  [0, 3],
  [1, 3],
] as const;

function popcount(value: number) {
  let bits = 0;
  for (let rest = value; rest > 0; rest >>= 1) bits += rest & 1;
  return bits;
}

// Stencil face case per 12-bit edge case, with the always-on "optimized
// stencil" rule: when any cell face has three or more edge intersections,
// connect only the faces with more than two intersections
// (GenerateEdgeStencils, vtkSurfaceNets3D.cxx:1056-1100; the constructor
// always calls GenerateEdgeStencils(1)).
const stencilTable = new Uint8Array(4096).map((_, edgeCase) => {
  const faceCounts = [0, 0, 0, 0, 0, 0];
  let faceCase = 0;
  for (let edge = 0; edge < 12; edge += 1) {
    if (edgeCase & (1 << edge)) {
      for (const face of edgeFaces[edge]) {
        faceCounts[face] += 1;
        faceCase |= 1 << face;
      }
    }
  }
  if (Math.max(...faceCounts) <= 2) return faceCase;
  return faceCounts.reduce(
    (junctions, count, face) => (count > 2 ? junctions | (1 << face) : junctions),
    0,
  );
});

// Constrained Laplacian smoothing (vtkConstrainedSmoothingFilter.cxx:134-350):
// per iteration, move each point toward the mean of its stencil neighbors'
// previous-iteration positions by the relaxation factor, then clamp inside a
// sphere of radius constraintDistance around the point's original position.
// Positions round-trip through float32 each iteration, matching VTK.
function smoothPoints(
  original: Float32Array,
  stencilOffsets: Int32Array,
  stencilConn: Int32Array,
  iterations: number,
  relaxation: number,
  constraintDistance: number,
) {
  const count = original.length / 3;
  const constraint2 = constraintDistance * constraintDistance;
  let previous = original;
  let next: Float32Array = new Float32Array(original.length);
  let spare: Float32Array | undefined;

  let maxMoved = Infinity;
  for (let iteration = 0; iteration < iterations && maxMoved > 0; iteration += 1) {
    let maxMoved2 = 0;
    for (let point = 0; point < count; point += 1) {
      const start = stencilOffsets[point];
      const end = stencilOffsets[point + 1];
      const size = end - start;
      let averageX = 0;
      let averageY = 0;
      let averageZ = 0;
      for (let entry = start; entry < end; entry += 1) {
        const neighbor = stencilConn[entry] * 3;
        averageX += previous[neighbor];
        averageY += previous[neighbor + 1];
        averageZ += previous[neighbor + 2];
      }
      averageX /= size;
      averageY /= size;
      averageZ /= size;

      const offset = point * 3;
      const currentX = previous[offset];
      const currentY = previous[offset + 1];
      const currentZ = previous[offset + 2];
      let x = currentX + relaxation * (averageX - currentX);
      let y = currentY + relaxation * (averageY - currentY);
      let z = currentZ + relaxation * (averageZ - currentZ);

      const anchorX = original[offset];
      const anchorY = original[offset + 1];
      const anchorZ = original[offset + 2];
      const distance2 = (x - anchorX) ** 2 + (y - anchorY) ** 2 + (z - anchorZ) ** 2;
      if (distance2 > constraint2) {
        const scale = Math.sqrt(constraint2 / distance2);
        x = anchorX + scale * (x - anchorX);
        y = anchorY + scale * (y - anchorY);
        z = anchorZ + scale * (z - anchorZ);
      }

      const moved2 = (x - currentX) ** 2 + (y - currentY) ** 2 + (z - currentZ) ** 2;
      if (moved2 > maxMoved2) maxMoved2 = moved2;

      next[offset] = x;
      next[offset + 1] = y;
      next[offset + 2] = z;
    }
    maxMoved = Math.sqrt(maxMoved2);

    const written = next;
    if (previous === original) {
      spare = new Float32Array(original.length);
      next = spare;
    } else {
      next = previous;
    }
    previous = written;
  }

  return previous === original ? new Float32Array(original) : previous;
}

// Quad-to-triangle conversion with the MIN_EDGE strategy: split along the
// strictly shorter diagonal; ties take the 1-3 diagonal
// (TransformMeshToTris, vtkSurfaceNets3D.cxx:1856-1934).
function triangulateQuads(points: Float32Array, quads: readonly number[]) {
  const distance2 = (a: number, b: number) => {
    const i = a * 3;
    const j = b * 3;
    return (
      (points[i] - points[j]) ** 2 +
      (points[i + 1] - points[j + 1]) ** 2 +
      (points[i + 2] - points[j + 2]) ** 2
    );
  };

  const quadCount = quads.length / 4;
  const polys = new Uint32Array(quadCount * 8);
  for (let quad = 0; quad < quadCount; quad += 1) {
    const c0 = quads[quad * 4];
    const c1 = quads[quad * 4 + 1];
    const c2 = quads[quad * 4 + 2];
    const c3 = quads[quad * 4 + 3];
    const diagonal02 = distance2(c0, c2) < distance2(c1, c3);
    const [a, b, c, d] = diagonal02 ? [c0, c2, c3, c1] : [c1, c3, c0, c2];
    polys.set([3, a, b, c, 3, b, a, d], quad * 8);
  }
  return polys;
}

export function surfaceNets(image: OrientedImage, options: SurfaceNetsOptions): Mesh {
  const { labelValue } = options;
  const [nx, ny, nz] = image.dims;
  const { data } = image;

  // Padded volume P: one background voxel per side; P voxel p maps to input
  // voxel p - 1, so P has dims m = dims + 2 and extent minimum -1.
  const mx = nx + 2;
  const my = ny + 2;
  const mz = nz + 2;
  const valueAt = (px: number, py: number, pz: number) => {
    const x = px - 1;
    const y = py - 1;
    const z = pz - 1;
    if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return backgroundLabel;
    return data[x + nx * (y + ny * z)];
  };

  // Triad grid pads P by one more layer per side; triad t maps to P voxel
  // t - 1 and carries inside/intersection bits for the +x/+y/+z edges
  // leaving that voxel (Pass1/Pass2, vtkSurfaceNets3D.cxx:1116-1278). Border
  // triads never intersect.
  const tdx = mx + 2;
  const tdy = my + 2;
  const tdz = mz + 2;
  const triads = new Uint8Array(tdx * tdy * tdz);
  const triadAt = (tx: number, ty: number, tz: number) => triads[tx + tdx * (ty + tdy * tz)];
  const splits = (v0: number, v1: number) => (v0 === labelValue || v1 === labelValue) && v0 !== v1;
  for (let pz = 0; pz < mz; pz += 1) {
    for (let py = 0; py < my; py += 1) {
      for (let px = 0; px < mx; px += 1) {
        const value = valueAt(px, py, pz);
        let triad = value === labelValue ? inside : 0;
        if (px + 1 < mx && splits(value, valueAt(px + 1, py, pz))) triad |= xIntersection;
        if (py + 1 < my && splits(value, valueAt(px, py + 1, pz))) triad |= yIntersection;
        if (pz + 1 < mz && splits(value, valueAt(px, py, pz + 1))) triad |= zIntersection;
        if (triad) triads[px + 1 + tdx * (py + 1 + tdy * (pz + 1))] = triad;
      }
    }
  }

  // One point per voxel cell with a non-zero 12-bit edge case, placed at the
  // cell center: paddedExtentMin(-1) + cellIndex - 0.5 in index space
  // (GetEdgeCase + GeneratePoint, vtkSurfaceNets3D.cxx:199-232, 730-736).
  const cdx = tdx - 1;
  const cdy = tdy - 1;
  const cdz = tdz - 1;
  const cellPointIds = new Int32Array(cdx * cdy * cdz).fill(-1);
  const cellIndex = (cx: number, cy: number, cz: number) => cx + cdx * (cy + cdy * cz);
  const pointCells: number[] = [];
  const pointEdgeCases: number[] = [];
  const indexPoints: number[] = [];
  for (let cz = 0; cz < cdz; cz += 1) {
    for (let cy = 0; cy < cdy; cy += 1) {
      for (let cx = 0; cx < cdx; cx += 1) {
        const t000 = triadAt(cx, cy, cz);
        const t100 = triadAt(cx + 1, cy, cz);
        const t010 = triadAt(cx, cy + 1, cz);
        const t110 = triadAt(cx + 1, cy + 1, cz);
        const t001 = triadAt(cx, cy, cz + 1);
        const t101 = triadAt(cx + 1, cy, cz + 1);
        const t011 = triadAt(cx, cy + 1, cz + 1);
        // vtkVoxel edge numbering: four x-edges, four y-edges, four z-edges
        // (GetEdgeCase, vtkSurfaceNets3D.cxx:199-232).
        const edgeCase =
          ((t000 & xIntersection) >> 1) |
          (t010 & xIntersection) |
          ((t001 & xIntersection) << 1) |
          ((t011 & xIntersection) << 2) |
          ((t000 & yIntersection) << 2) |
          ((t100 & yIntersection) << 3) |
          ((t001 & yIntersection) << 4) |
          ((t101 & yIntersection) << 5) |
          ((t000 & zIntersection) << 5) |
          ((t100 & zIntersection) << 6) |
          ((t010 & zIntersection) << 7) |
          ((t110 & zIntersection) << 8);
        if (edgeCase === 0) continue;
        cellPointIds[cellIndex(cx, cy, cz)] = pointCells.length;
        pointCells.push(cellIndex(cx, cy, cz));
        pointEdgeCases.push(edgeCase);
        indexPoints.push(cx - 1.5, cy - 1.5, cz - 1.5);
      }
    }
  }

  if (pointCells.length === 0) {
    return { points: new Float32Array(0), polys: new Uint32Array(0) };
  }

  // One quad per intersected triad edge, connecting the centers of the four
  // cells sharing the edge; c1/c3 swap orients the quad with the smaller
  // (or foreground) label outward (GenerateQuadsImpl,
  // vtkSurfaceNets3D.cxx:743-827).
  const quads: number[] = [];
  const pushQuad = (c0: number, c1: number, c2: number, c3: number, s0: number, s1: number) => {
    if (s0 === backgroundLabel || (s1 !== backgroundLabel && s0 > s1)) {
      quads.push(c0, c3, c2, c1);
    } else {
      quads.push(c0, c1, c2, c3);
    }
  };
  for (let tz = 1; tz <= mz; tz += 1) {
    for (let ty = 1; ty <= my; ty += 1) {
      for (let tx = 1; tx <= mx; tx += 1) {
        const triad = triadAt(tx, ty, tz);
        if ((triad & (xIntersection | yIntersection | zIntersection)) === 0) continue;
        const point = (cx: number, cy: number, cz: number) => cellPointIds[cellIndex(cx, cy, cz)];
        const s0 = valueAt(tx - 1, ty - 1, tz - 1);
        if (triad & zIntersection) {
          // x-y quad
          pushQuad(
            point(tx, ty, tz),
            point(tx - 1, ty, tz),
            point(tx - 1, ty - 1, tz),
            point(tx, ty - 1, tz),
            s0,
            valueAt(tx - 1, ty - 1, tz),
          );
        }
        if (triad & yIntersection) {
          // x-z quad
          pushQuad(
            point(tx, ty, tz),
            point(tx, ty, tz - 1),
            point(tx - 1, ty, tz - 1),
            point(tx - 1, ty, tz),
            s0,
            valueAt(tx - 1, ty, tz - 1),
          );
        }
        if (triad & xIntersection) {
          // y-z quad
          pushQuad(
            point(tx, ty, tz),
            point(tx, ty - 1, tz),
            point(tx, ty - 1, tz - 1),
            point(tx, ty, tz - 1),
            s0,
            valueAt(tx, ty - 1, tz - 1),
          );
        }
      }
    }
  }

  // Smoothing stencils: connect each point to the points in face-adjacent
  // cells selected by the stencil table; single-edge stencils reference the
  // point itself, locking it in place (GenerateStencilImpl,
  // vtkSurfaceNets3D.cxx:830-897).
  const pointCount = pointCells.length;
  const stencilOffsets = new Int32Array(pointCount + 1);
  const stencilConn: number[] = [];
  const faceOffsets = [-1, 1, -cdx, cdx, -cdx * cdy, cdx * cdy];
  for (let point = 0; point < pointCount; point += 1) {
    const faceCase = stencilTable[pointEdgeCases[point]];
    if (popcount(faceCase) === 1) {
      stencilConn.push(point);
    } else {
      for (let face = 0; face < 6; face += 1) {
        if (faceCase & (1 << face)) {
          stencilConn.push(cellPointIds[pointCells[point] + faceOffsets[face]]);
        }
      }
    }
    stencilOffsets[point + 1] = stencilConn.length;
  }

  // Transform cell centers into world space before smoothing
  // (vtkImageTransform::TransformPointSet, called from RequestData at
  // vtkSurfaceNets3D.cxx:2313). Points are float32, math in double.
  const worldPoints = new Float32Array(pointCount * 3);
  for (let point = 0; point < pointCount; point += 1) {
    const world = indexToWorld(image, [
      indexPoints[point * 3],
      indexPoints[point * 3 + 1],
      indexPoints[point * 3 + 2],
    ]);
    worldPoints.set(world, point * 3);
  }

  const iterations = options.smoothingIterations ?? 16;
  const smoothing = (options.smoothing ?? true) && iterations > 0;
  const points = smoothing
    ? smoothPoints(
        worldPoints,
        stencilOffsets,
        new Int32Array(stencilConn),
        iterations,
        options.relaxationFactor ?? 0.5,
        options.constraintDistance ??
          Math.hypot(image.spacing[0], image.spacing[1], image.spacing[2]),
      )
    : worldPoints;

  return { points, polys: triangulateQuads(points, quads) };
}
