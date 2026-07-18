import {
  createPlanarContour,
  validateContourPlane,
  type ContourPlane,
  type Vector3,
} from '../geometry/contour.js';
import { validateMesh, type Mesh } from '../geometry/mesh.js';

// Port of VTK's triangle-mesh plane cut and loop assembly:
// - vtkPolyDataPlaneCutter (Filters/Core/vtkPolyDataPlaneCutter.cxx): per-point
//   plane classification (strictly above > 0, on-plane counts as below), one
//   segment per cut triangle, exact topological point merging by canonical cut
//   edge (V0 < V1), and unclamped edge interpolation t = -v0 / (v1 - v0).
// - vtkContourLoopExtraction (Filters/Modeling/vtkContourLoopExtraction.cxx):
//   bidirectional traversal over 2-point segments, closure by point-id
//   equality, termination at boundaries (one incident segment) and
//   non-manifold junctions (three or more incident segments).
//
// Shipped policies (differences from raw VTK defaults, documented for G):
// - Open chains are dropped (vtkContourLoopExtraction's LoopClosure = OFF).
//   The default BOUNDARY mode closes chains only when the endpoints share a
//   world x or y coordinate, which is meaningless for an arbitrary cutting
//   plane, so it is not ported.
// - Loops are normalized to counterclockwise winding about the plane normal
//   xAxis cross yAxis (positive shoelace area in plane xy). VTK leaves the
//   winding unspecified. Holes are not encoded by winding; consumers should
//   use even-odd containment.
// - An empty intersection (plane misses the mesh, or only open chains) returns
//   undefined because validatePlanarContour rejects zero loops.

function cross(a: Vector3, b: Vector3): Vector3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

type CutEdge = {
  v0: number;
  v1: number;
  /** Index into the segment connectivity array this edge's merged point fills. */
  slot: number;
};

function classifyPoints(mesh: Mesh, origin: Vector3, normal: Vector3) {
  const count = mesh.points.length / 3;
  const above = new Uint8Array(count);
  let anyAbove = false;
  let anyBelow = false;
  for (let index = 0; index < count; index += 1) {
    const offset = index * 3;
    const value = normal[0] * (mesh.points[offset] - origin[0])
      + normal[1] * (mesh.points[offset + 1] - origin[1])
      + normal[2] * (mesh.points[offset + 2] - origin[2]);
    // vtkPolyDataPlaneCutter: strictly positive is above; exactly on the
    // plane classifies as below. No epsilon anywhere.
    if (value > 0) {
      above[index] = 1;
      anyAbove = true;
    } else {
      anyBelow = true;
    }
  }
  return { above, intersects: anyAbove && anyBelow };
}

function collectCutEdges(mesh: Mesh, above: Uint8Array) {
  const edges: CutEdge[] = [];
  let slot = 0;
  for (let offset = 0; offset < mesh.polys.length; offset += 4) {
    const a = mesh.polys[offset + 1];
    const b = mesh.polys[offset + 2];
    const c = mesh.polys[offset + 3];
    if (above[a] === above[b] && above[b] === above[c]) continue;
    // A cut triangle has exactly two edges whose endpoints straddle the plane.
    const vertices = [a, b, c];
    for (let corner = 0; corner < 3; corner += 1) {
      const v0 = vertices[corner];
      const v1 = vertices[(corner + 1) % 3];
      if (above[v0] !== above[v1]) {
        // Canonical edge ordering (EdgeTuple::Define): smaller id first.
        edges.push(v0 < v1 ? { v0, v1, slot } : { v0: v1, v1: v0, slot });
        slot += 1;
      }
    }
  }
  return edges;
}

function mergeCutEdges(edges: CutEdge[], segmentConn: Uint32Array) {
  // vtkStaticEdgeLocatorTemplate::MergeEdges: sort canonical edges; identical
  // (v0, v1) runs merge into one output point.
  const sorted = [...edges].sort((left, right) =>
    (left.v0 - right.v0) || (left.v1 - right.v1));
  const uniqueEdges: CutEdge[] = [];
  for (const edge of sorted) {
    const last = uniqueEdges[uniqueEdges.length - 1];
    if (!last || last.v0 !== edge.v0 || last.v1 !== edge.v1) {
      uniqueEdges.push(edge);
    }
    segmentConn[edge.slot] = uniqueEdges.length - 1;
  }
  return uniqueEdges;
}

function cutPointPositions(
  mesh: Mesh,
  uniqueEdges: CutEdge[],
  origin: Vector3,
  normal: Vector3,
) {
  const positions = new Float64Array(uniqueEdges.length * 3);
  for (let index = 0; index < uniqueEdges.length; index += 1) {
    const { v0, v1 } = uniqueEdges[index];
    const x0 = v0 * 3;
    const x1 = v1 * 3;
    const value0 = normal[0] * (mesh.points[x0] - origin[0])
      + normal[1] * (mesh.points[x0 + 1] - origin[1])
      + normal[2] * (mesh.points[x0 + 2] - origin[2]);
    const value1 = normal[0] * (mesh.points[x1] - origin[0])
      + normal[1] * (mesh.points[x1 + 1] - origin[1])
      + normal[2] * (mesh.points[x1 + 2] - origin[2]);
    const delta = value1 - value0;
    // Unclamped, computed in canonical edge order so shared points are
    // bitwise identical regardless of which triangle contributed the edge.
    const t = delta === 0 ? 0 : -value0 / delta;
    positions[index * 3] = mesh.points[x0] + t * (mesh.points[x1] - mesh.points[x0]);
    positions[index * 3 + 1] = mesh.points[x0 + 1] + t * (mesh.points[x1 + 1] - mesh.points[x0 + 1]);
    positions[index * 3 + 2] = mesh.points[x0 + 2] + t * (mesh.points[x1 + 2] - mesh.points[x0 + 2]);
  }
  return positions;
}

function buildPointToSegments(segmentConn: Uint32Array, pointCount: number) {
  const incident: number[][] = Array.from({ length: pointCount }, () => []);
  for (let segment = 0; segment < segmentConn.length / 2; segment += 1) {
    incident[segmentConn[segment * 2]].push(segment);
    incident[segmentConn[segment * 2 + 1]].push(segment);
  }
  return incident;
}

function traverse(
  segmentConn: Uint32Array,
  incident: number[][],
  seedSegment: number,
  start: number,
  visited: Uint8Array,
  emit: (pointId: number) => void,
) {
  // vtkContourLoopExtraction::TraverseLoop over 2-point segments.
  let lastCell = seedSegment;
  let last = start;
  for (;;) {
    const p0 = segmentConn[lastCell * 2];
    const p1 = segmentConn[lastCell * 2 + 1];
    last = p0 !== last ? p0 : p1;
    emit(last);
    const cells = incident[last];
    if (cells.length === 1 || last === start) return last;
    if (cells.length !== 2) return last; // non-manifold junction: stop
    const next = cells[0] !== lastCell ? cells[0] : cells[1];
    visited[next] = 1;
    lastCell = next;
  }
}

function extractClosedLoops(segmentConn: Uint32Array, pointCount: number) {
  const segmentCount = segmentConn.length / 2;
  const incident = buildPointToSegments(segmentConn, pointCount);
  const visited = new Uint8Array(segmentCount);
  const loops: number[][] = [];
  for (let seed = 0; seed < segmentCount; seed += 1) {
    if (visited[seed]) continue;
    visited[seed] = 1;
    const start = segmentConn[seed * 2];
    const chain = [start];
    const end = traverse(segmentConn, incident, seed, start, visited, (id) => chain.push(id));
    if (end === start) {
      chain.pop(); // implicit closure: drop the repeated start point
      if (chain.length >= 3) loops.push(chain);
    } else {
      // Open chain: walk the other direction only to mark segments visited,
      // then drop the chain (LoopClosure = OFF policy).
      traverse(segmentConn, incident, seed, segmentConn[seed * 2 + 1], visited, () => {});
    }
  }
  return loops;
}

function projectLoop(loop: number[], positions: Float64Array, plane: ContourPlane) {
  const planar = new Float64Array(loop.length * 2);
  for (let index = 0; index < loop.length; index += 1) {
    const offset = loop[index] * 3;
    const dx = positions[offset] - plane.origin[0];
    const dy = positions[offset + 1] - plane.origin[1];
    const dz = positions[offset + 2] - plane.origin[2];
    planar[index * 2] = dx * plane.xAxis[0] + dy * plane.xAxis[1] + dz * plane.xAxis[2];
    planar[index * 2 + 1] = dx * plane.yAxis[0] + dy * plane.yAxis[1] + dz * plane.yAxis[2];
  }
  return planar;
}

function signedArea(points: Float64Array) {
  let area = 0;
  const count = points.length / 2;
  for (let index = 0; index < count; index += 1) {
    const next = (index + 1) % count;
    area += points[index * 2] * points[next * 2 + 1]
      - points[next * 2] * points[index * 2 + 1];
  }
  return area / 2;
}

function orientCounterclockwise(points: Float64Array) {
  if (signedArea(points) >= 0) return points;
  const count = points.length / 2;
  const reversed = new Float64Array(points.length);
  reversed[0] = points[0];
  reversed[1] = points[1];
  for (let index = 1; index < count; index += 1) {
    const source = (count - index) * 2;
    reversed[index * 2] = points[source];
    reversed[index * 2 + 1] = points[source + 1];
  }
  return reversed;
}

/**
 * Cut a triangle mesh with the plane spanned by `plane.xAxis` and
 * `plane.yAxis` through `plane.origin`, assembling the intersection into
 * closed loops.
 *
 * Returns a PlanarContour whose loops are counterclockwise about the plane
 * normal (xAxis cross yAxis) in plane xy coordinates, with implicit closure.
 * Open chains (mesh boundary crossings, non-manifold junctions) are dropped.
 * Returns undefined when the intersection produces no closed loop.
 */
export function surfaceToContour(mesh: Mesh, plane: ContourPlane) {
  validateMesh(mesh);
  validateContourPlane(plane);
  const normal = cross(plane.xAxis, plane.yAxis);

  const { above, intersects } = classifyPoints(mesh, plane.origin, normal);
  if (!intersects) return undefined;

  const edges = collectCutEdges(mesh, above);
  if (edges.length === 0) return undefined;

  const segmentConn = new Uint32Array(edges.length);
  const uniqueEdges = mergeCutEdges(edges, segmentConn);
  const positions = cutPointPositions(mesh, uniqueEdges, plane.origin, normal);

  const loops = extractClosedLoops(segmentConn, uniqueEdges.length)
    .map((loop) => ({ points: orientCounterclockwise(projectLoop(loop, positions, plane)) }));
  if (loops.length === 0) return undefined;

  return createPlanarContour(plane, loops);
}
