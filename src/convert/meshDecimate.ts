import { validateMesh, type Mesh } from '../geometry/mesh.js';

// Port of VTK's vtkQuadricDecimation (BSD-3-Clause), geometry-only path:
// reference-repos/vtk/Filters/Core/vtkQuadricDecimation.cxx. VTK defaults
// confirmed from that source: AttributeErrorMetric off, VolumePreservation
// off, Regularize off, MaximumError = VTK_DOUBLE_MAX, boundary constraints
// weighted by squared edge length (WeighBoundaryConstraintsByLength off)
// with BoundaryWeightFactor 1.

export type MeshDecimateOptions = {
  /** Requested fraction of triangles to remove, clamped to [0, 1] like VTK. */
  targetReduction: number;
};

// vtkQuadricDecimation.cxx `errorNumber`: normalized-determinant singularity
// threshold in ComputeCost and the least-squares fallback test.
const SINGULARITY_TOLERANCE = 1e-10;
// TrianglePlaneCheck acceptance threshold on the normalized dot product.
const PLANE_CHECK_TOLERANCE = 1e-5;
// VTK_DOUBLE_MAX: MaximumError default and the "poor placement" re-queue cost.
const MAXIMUM_ERROR = Number.MAX_VALUE;

type PriorityQueue = {
  costs: number[];
  ids: number[];
  // edge id -> heap index, -1 when absent.
  locations: number[];
};

function queueSwap(queue: PriorityQueue, a: number, b: number) {
  const { costs, ids, locations } = queue;
  [costs[a], costs[b]] = [costs[b], costs[a]];
  [ids[a], ids[b]] = [ids[b], ids[a]];
  locations[ids[a]] = a;
  locations[ids[b]] = b;
}

function queueSiftUp(queue: PriorityQueue, start: number) {
  let index = start;
  while (index > 0) {
    const parent = (index - 1) >> 1;
    if (queue.costs[index] >= queue.costs[parent]) break;
    queueSwap(queue, index, parent);
    index = parent;
  }
}

function queueSiftDown(queue: PriorityQueue, start: number) {
  const size = queue.ids.length;
  let index = start;
  for (;;) {
    const left = 2 * index + 1;
    if (left >= size) break;
    const right = left + 1;
    let smallest = left;
    if (right < size && queue.costs[right] < queue.costs[left]) smallest = right;
    if (queue.costs[smallest] >= queue.costs[index]) break;
    queueSwap(queue, index, smallest);
    index = smallest;
  }
}

function queueInsert(queue: PriorityQueue, cost: number, id: number) {
  queue.costs.push(cost);
  queue.ids.push(id);
  while (queue.locations.length <= id) queue.locations.push(-1);
  queue.locations[id] = queue.ids.length - 1;
  queueSiftUp(queue, queue.ids.length - 1);
}

function queueRemoveAt(queue: PriorityQueue, index: number) {
  const last = queue.ids.length - 1;
  queue.locations[queue.ids[index]] = -1;
  if (index !== last) {
    queue.costs[index] = queue.costs[last];
    queue.ids[index] = queue.ids[last];
    queue.locations[queue.ids[index]] = index;
  }
  queue.costs.pop();
  queue.ids.pop();
  if (index < queue.ids.length) {
    queueSiftUp(queue, index);
    queueSiftDown(queue, index);
  }
}

function queuePop(queue: PriorityQueue): { id: number; cost: number } {
  if (queue.ids.length === 0) return { id: -1, cost: 0 };
  const id = queue.ids[0];
  const cost = queue.costs[0];
  queueRemoveAt(queue, 0);
  return { id, cost };
}

function queueDeleteId(queue: PriorityQueue, id: number) {
  const index = id < queue.locations.length ? queue.locations[id] : -1;
  if (index >= 0) queueRemoveAt(queue, index);
}

function clampTargetReduction(value: number) {
  if (!Number.isFinite(value)) {
    throw new RangeError('targetReduction must be a finite number');
  }
  return Math.min(1, Math.max(0, value));
}

// vtkMath::LinearSolve3x3: adjoint-based solve of A * x = b.
function linearSolve3x3(a: Float64Array, b: Float64Array, x: Float64Array) {
  const a1 = a[0];
  const b1 = a[1];
  const c1 = a[2];
  const a2 = a[3];
  const b2 = a[4];
  const c2 = a[5];
  const a3 = a[6];
  const b3 = a[7];
  const c3 = a[8];

  const d1 = b2 * c3 - b3 * c2;
  const d2 = -(a2 * c3 - a3 * c2);
  const d3 = a2 * b3 - a3 * b2;

  const e1 = -(b1 * c3 - b3 * c1);
  const e2 = a1 * c3 - a3 * c1;
  const e3 = -(a1 * b3 - a3 * b1);

  const f1 = b1 * c2 - b2 * c1;
  const f2 = -(a1 * c2 - a2 * c1);
  const f3 = a1 * b2 - a2 * b1;

  const det = a1 * d1 + b1 * d2 + c1 * d3;

  x[0] = (d1 * b[0] + e1 * b[1] + f1 * b[2]) / det;
  x[1] = (d2 * b[0] + e2 * b[1] + f2 * b[2]) / det;
  x[2] = (d3 * b[0] + e3 * b[1] + f3 * b[2]) / det;
}

function determinant3x3(a: Float64Array) {
  return (
    a[0] * (a[4] * a[8] - a[5] * a[7]) -
    a[1] * (a[3] * a[8] - a[5] * a[6]) +
    a[2] * (a[3] * a[7] - a[4] * a[6])
  );
}

function rowNorm(a: Float64Array, row: number) {
  const x = a[row * 3];
  const y = a[row * 3 + 1];
  const z = a[row * 3 + 2];
  return Math.sqrt(x * x + y * y + z * z);
}

export function meshDecimate(mesh: Mesh, options: MeshDecimateOptions): Mesh {
  validateMesh(mesh);
  const targetReduction = clampTargetReduction(options.targetReduction);

  const pointCount = mesh.points.length / 3;
  const triangleTotal = mesh.polys.length / 4;
  if (pointCount === 0 || triangleTotal === 0) {
    return { points: new Float32Array(0), polys: new Uint32Array(0) };
  }

  // Working mesh: coordinates stay float32 like VTK's deep-copied input
  // points, so every SetPoint write rounds the double target to float.
  const points = new Float32Array(mesh.points);
  const cells = new Uint32Array(triangleTotal * 3);
  for (let cell = 0; cell < triangleTotal; cell += 1) {
    cells[cell * 3] = mesh.polys[cell * 4 + 1];
    cells[cell * 3 + 1] = mesh.polys[cell * 4 + 2];
    cells[cell * 3 + 2] = mesh.polys[cell * 4 + 3];
  }
  const cellDeleted = new Uint8Array(triangleTotal);
  // vtkPolyData::BuildLinks equivalent: one entry per vertex slot, so a
  // repeated-vertex triangle appears repeatedly, exactly like vtkCellLinks.
  const pointCells: number[][] = Array.from({ length: pointCount }, () => []);
  for (let cell = 0; cell < triangleTotal; cell += 1) {
    for (let corner = 0; corner < 3; corner += 1) {
      pointCells[cells[cell * 3 + corner]].push(cell);
    }
  }

  // vtkEdgeTable equivalent keyed on the unordered vertex pair.
  const edgeIds = new Map<number, number>();
  const edgeKey = (a: number, b: number) => (a < b ? a * pointCount + b : b * pointCount + a);
  const endPoint1: number[] = [];
  const endPoint2: number[] = [];
  for (let cell = 0; cell < triangleTotal; cell += 1) {
    for (let corner = 0; corner < 3; corner += 1) {
      const from = cells[cell * 3 + corner];
      const to = cells[cell * 3 + ((corner + 1) % 3)];
      const key = edgeKey(from, to);
      if (!edgeIds.has(key)) {
        edgeIds.set(key, endPoint1.length);
        endPoint1.push(from);
        endPoint2.push(to);
      }
    }
  }

  // InitializeQuadrics: 11-entry symmetric homogeneous quadric per point,
  // accumulated per face weighted by triangle area.
  const quadrics = new Float64Array(pointCount * 11);
  for (let cell = 0; cell < triangleTotal; cell += 1) {
    const i0 = cells[cell * 3];
    const i1 = cells[cell * 3 + 1];
    const i2 = cells[cell * 3 + 2];
    const x0 = points[i0 * 3];
    const y0 = points[i0 * 3 + 1];
    const z0 = points[i0 * 3 + 2];
    const e1x = points[i1 * 3] - x0;
    const e1y = points[i1 * 3 + 1] - y0;
    const e1z = points[i1 * 3 + 2] - z0;
    const e2x = points[i2 * 3] - x0;
    const e2y = points[i2 * 3 + 1] - y0;
    const e2z = points[i2 * 3 + 2] - z0;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    const length = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (length !== 0) {
      nx /= length;
      ny /= length;
      nz /= length;
    }
    const area = length / 2;
    const d = -(nx * x0 + ny * y0 + nz * z0);

    const q0 = nx * nx;
    const q1 = nx * ny;
    const q2 = nx * nz;
    const q3 = d * nx;
    const q4 = ny * ny;
    const q5 = ny * nz;
    const q6 = d * ny;
    const q7 = nz * nz;
    const q8 = d * nz;
    const q9 = d * d;

    for (const point of [i0, i1, i2]) {
      const base = point * 11;
      quadrics[base] += q0 * area;
      quadrics[base + 1] += q1 * area;
      quadrics[base + 2] += q2 * area;
      quadrics[base + 3] += q3 * area;
      quadrics[base + 4] += q4 * area;
      quadrics[base + 5] += q5 * area;
      quadrics[base + 6] += q6 * area;
      quadrics[base + 7] += q7 * area;
      quadrics[base + 8] += q8 * area;
      quadrics[base + 9] += q9 * area;
      quadrics[base + 10] += area;
    }
  }

  // AddBoundaryConstraints: for every boundary edge add a quadric of the
  // plane through the edge, orthogonal to its triangle, weighted by the
  // squared edge length (WeighBoundaryConstraintsByLength off default).
  const isBoundaryEdge = (cell: number, from: number, to: number) =>
    !pointCells[from].some(
      (other) =>
        other !== cell &&
        (cells[other * 3] === to || cells[other * 3 + 1] === to || cells[other * 3 + 2] === to),
    );
  for (let cell = 0; cell < triangleTotal; cell += 1) {
    for (let corner = 0; corner < 3; corner += 1) {
      const from = cells[cell * 3 + corner];
      const to = cells[cell * 3 + ((corner + 1) % 3)];
      if (!isBoundaryEdge(cell, from, to)) continue;

      const opposite = cells[cell * 3 + ((corner + 2) % 3)];
      const t0x = points[opposite * 3];
      const t0y = points[opposite * 3 + 1];
      const t0z = points[opposite * 3 + 2];
      const t1x = points[from * 3];
      const t1y = points[from * 3 + 1];
      const t1z = points[from * 3 + 2];
      const t2x = points[to * 3];
      const t2y = points[to * 3 + 1];
      const t2z = points[to * 3 + 2];
      const e0x = t2x - t1x;
      const e0y = t2y - t1y;
      const e0z = t2z - t1z;
      const e1x = t0x - t1x;
      const e1y = t0y - t1y;
      const e1z = t0z - t1z;
      const c = (e0x * e1x + e0y * e1y + e0z * e1z) / (e0x * e0x + e0y * e0y + e0z * e0z);
      let nx = e1x - c * e0x;
      let ny = e1y - c * e0y;
      let nz = e1z - c * e0z;
      const length = Math.sqrt(nx * nx + ny * ny + nz * nz);
      if (length !== 0) {
        nx /= length;
        ny /= length;
        nz /= length;
      }
      const d = -(nx * t1x + ny * t1y + nz * t1z);
      const edgeLength = Math.sqrt(e0x * e0x + e0y * e0y + e0z * e0z);
      const weight = edgeLength * edgeLength;

      const constraint = [
        nx * nx,
        nx * ny,
        nx * nz,
        d * nx,
        ny * ny,
        ny * nz,
        d * ny,
        nz * nz,
        d * nz,
        d * d,
        1,
      ];
      for (const point of [from, to]) {
        for (let entry = 0; entry < 11; entry += 1) {
          quadrics[point * 11 + entry] += constraint[entry] * weight;
        }
      }
    }
  }

  // ComputeCost: optimal point of the summed quadric, falling back to a
  // least-squares point along the edge, then to the midpoint.
  const tempQuad = new Float64Array(11);
  const tempA = new Float64Array(9);
  const tempB = new Float64Array(3);
  const computeCost = (edgeId: number, x: Float64Array) => {
    const p0 = endPoint1[edgeId];
    const p1 = endPoint2[edgeId];
    for (let entry = 0; entry < 11; entry += 1) {
      tempQuad[entry] = quadrics[p0 * 11 + entry] + quadrics[p1 * 11 + entry];
    }

    tempA[0] = tempQuad[0];
    tempA[1] = tempA[3] = tempQuad[1];
    tempA[2] = tempA[6] = tempQuad[2];
    tempA[4] = tempQuad[4];
    tempA[5] = tempA[7] = tempQuad[5];
    tempA[8] = tempQuad[7];
    tempB[0] = -tempQuad[3];
    tempB[1] = -tempQuad[6];
    tempB[2] = -tempQuad[8];

    const norm = Math.max(rowNorm(tempA, 0), rowNorm(tempA, 1), rowNorm(tempA, 2));
    if (Math.abs(determinant3x3(tempA)) / (norm * norm * norm) > SINGULARITY_TOLERANCE) {
      linearSolve3x3(tempA, tempB, x);
    } else {
      // Singular quadric: least-squares fit of A * (pt1 + c * v) = b.
      const p0x = points[p0 * 3];
      const p0y = points[p0 * 3 + 1];
      const p0z = points[p0 * 3 + 2];
      const vx = points[p1 * 3] - p0x;
      const vy = points[p1 * 3 + 1] - p0y;
      const vz = points[p1 * 3 + 2] - p0z;
      const avx = tempA[0] * vx + tempA[1] * vy + tempA[2] * vz;
      const avy = tempA[3] * vx + tempA[4] * vy + tempA[5] * vz;
      const avz = tempA[6] * vx + tempA[7] * vy + tempA[8] * vz;
      const denominator = avx * avx + avy * avy + avz * avz;
      if (denominator > SINGULARITY_TOLERANCE) {
        const rx = tempB[0] - (tempA[0] * p0x + tempA[1] * p0y + tempA[2] * p0z);
        const ry = tempB[1] - (tempA[3] * p0x + tempA[4] * p0y + tempA[5] * p0z);
        const rz = tempB[2] - (tempA[6] * p0x + tempA[7] * p0y + tempA[8] * p0z);
        const c = (avx * rx + avy * ry + avz * rz) / denominator;
        x[0] = p0x + c * vx;
        x[1] = p0y + c * vy;
        x[2] = p0z + c * vz;
      } else {
        x[0] = 0.5 * (p0x + points[p1 * 3]);
        x[1] = 0.5 * (p0y + points[p1 * 3 + 1]);
        x[2] = 0.5 * (p0z + points[p1 * 3 + 2]);
      }
    }

    // cost = [x 1]' * quad * [x 1] via the upper-triangular sparse layout.
    const homogeneous = [x[0], x[1], x[2], 1];
    let cost = 0;
    let entry = 0;
    for (let i = 0; i < 4; i += 1) {
      cost += tempQuad[entry] * homogeneous[i] * homogeneous[i];
      entry += 1;
      for (let j = i + 1; j < 4; j += 1) {
        cost += 2 * tempQuad[entry] * homogeneous[i] * homogeneous[j];
        entry += 1;
      }
    }
    return cost;
  };

  const queue: PriorityQueue = { costs: [], ids: [], locations: [] };
  const targetPoints: number[] = [];
  const costWorkspace = new Float64Array(3);
  for (let edgeId = 0; edgeId < endPoint1.length; edgeId += 1) {
    const cost = computeCost(edgeId, costWorkspace);
    queueInsert(queue, cost, edgeId);
    targetPoints[edgeId * 3] = costWorkspace[0];
    targetPoints[edgeId * 3 + 1] = costWorkspace[1];
    targetPoints[edgeId * 3 + 2] = costWorkspace[2];
  }

  // TrianglePlaneCheck: is x on the same side as t0 of the plane through
  // t1-t2 that is orthogonal to the triangle?
  const trianglePlaneCheck = (t0: number, t1: number, t2: number, x: Float64Array) => {
    const t1x = points[t1 * 3];
    const t1y = points[t1 * 3 + 1];
    const t1z = points[t1 * 3 + 2];
    const e0x = points[t2 * 3] - t1x;
    const e0y = points[t2 * 3 + 1] - t1y;
    const e0z = points[t2 * 3 + 2] - t1z;
    const e1x = points[t0 * 3] - t1x;
    const e1y = points[t0 * 3 + 1] - t1y;
    const e1z = points[t0 * 3 + 2] - t1z;
    const c = (e0x * e1x + e0y * e1y + e0z * e1z) / (e0x * e0x + e0y * e0y + e0z * e0z);
    let nx = e1x - c * e0x;
    let ny = e1y - c * e0y;
    let nz = e1z - c * e0z;
    let e2x = x[0] - t1x;
    let e2y = x[1] - t1y;
    let e2z = x[2] - t1z;
    const nLength = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (nLength !== 0) {
      nx /= nLength;
      ny /= nLength;
      nz /= nLength;
    }
    const e2Length = Math.sqrt(e2x * e2x + e2y * e2y + e2z * e2z);
    if (e2Length !== 0) {
      e2x /= e2Length;
      e2y /= e2Length;
      e2z /= e2Length;
    }
    return nx * e2x + ny * e2y + nz * e2z > PLANE_CHECK_TOLERANCE;
  };

  // IsGoodPlacement: the target x must stay on the inside of every triangle
  // around either endpoint that does not contain the other endpoint.
  const isGoodPlacement = (pt0: number, pt1: number, x: Float64Array) => {
    for (const [anchor, excluded] of [
      [pt0, pt1],
      [pt1, pt0],
    ] as const) {
      for (const cell of pointCells[anchor]) {
        const base = cell * 3;
        if (
          cells[base] === excluded ||
          cells[base + 1] === excluded ||
          cells[base + 2] === excluded
        )
          continue;
        for (let corner = 0; corner < 3; corner += 1) {
          if (cells[base + corner] !== anchor) continue;
          if (
            !trianglePlaneCheck(
              cells[base + corner],
              cells[base + ((corner + 1) % 3)],
              cells[base + ((corner + 2) % 3)],
              x,
            )
          ) {
            return false;
          }
        }
      }
    }
    return true;
  };

  // FindAffectedEdges: existing edges incident to either endpoint, excluding
  // the collapsing edge itself.
  const findAffectedEdges = (pt0: number, pt1: number) => {
    const affected: number[] = [];
    const seen = new Set<number>();
    for (const [target, other] of [
      [pt1, pt0],
      [pt0, pt1],
    ] as const) {
      for (const cell of pointCells[target]) {
        for (let corner = 0; corner < 3; corner += 1) {
          const vertex = cells[cell * 3 + corner];
          if (vertex === other || vertex === target) continue;
          const edgeId = edgeIds.get(edgeKey(vertex, target));
          if (edgeId !== undefined && !seen.has(edgeId)) {
            seen.add(edgeId);
            affected.push(edgeId);
          }
        }
      }
    }
    return affected;
  };

  // UpdateEdgeData: retire affected edges from the queue, re-point edges that
  // referenced pt1 at pt0 (deduplicating against existing edges), recompute
  // costs and target points.
  const updateEdgeData = (pt0: number, pt1: number) => {
    for (const edgeId of findAffectedEdges(pt0, pt1)) {
      const from = endPoint1[edgeId];
      const to = endPoint2[edgeId];
      queueDeleteId(queue, edgeId);

      if (from === pt1 || to === pt1) {
        const survivor = from === pt1 ? to : from;
        if (!edgeIds.has(edgeKey(survivor, pt0))) {
          const newEdgeId = endPoint1.length;
          edgeIds.set(edgeKey(survivor, pt0), newEdgeId);
          endPoint1.push(survivor);
          endPoint2.push(pt0);
          const cost = computeCost(newEdgeId, costWorkspace);
          queueInsert(queue, cost, newEdgeId);
          targetPoints[newEdgeId * 3] = costWorkspace[0];
          targetPoints[newEdgeId * 3 + 1] = costWorkspace[1];
          targetPoints[newEdgeId * 3 + 2] = costWorkspace[2];
        }
      } else {
        const cost = computeCost(edgeId, costWorkspace);
        queueInsert(queue, cost, edgeId);
        targetPoints[edgeId * 3] = costWorkspace[0];
        targetPoints[edgeId * 3 + 1] = costWorkspace[1];
        targetPoints[edgeId * 3 + 2] = costWorkspace[2];
      }
    }
  };

  const removeCellReference = (cell: number) => {
    for (let corner = 0; corner < 3; corner += 1) {
      const references = pointCells[cells[cell * 3 + corner]];
      const index = references.indexOf(cell);
      if (index !== -1) references.splice(index, 1);
    }
  };

  // vtkPolyData::IsTriangle: does a live cell contain all three vertices?
  const isTriangle = (v0: number, v1: number, v2: number) => {
    for (const vertex of [v0, v1, v2]) {
      for (const cell of pointCells[vertex]) {
        const a = cells[cell * 3];
        const b = cells[cell * 3 + 1];
        const c = cells[cell * 3 + 2];
        if (
          (v0 === a || v0 === b || v0 === c) &&
          (v1 === a || v1 === b || v1 === c) &&
          (v2 === a || v2 === b || v2 === c)
        ) {
          return true;
        }
      }
    }
    return false;
  };

  // CollapseEdge: delete triangles containing both endpoints, re-point the
  // remaining triangles around pt1 at pt0 unless the re-pointed triangle
  // already exists, and drop pt1's cell links.
  const collapseEdge = (pt0: number, pt1: number) => {
    let deletedCount = 0;
    for (const cell of [...pointCells[pt0]]) {
      for (let corner = 0; corner < 3; corner += 1) {
        if (cells[cell * 3 + corner] === pt1) {
          removeCellReference(cell);
          cellDeleted[cell] = 1;
          deletedCount += 1;
        }
      }
    }

    for (const cell of [...pointCells[pt1]]) {
      const base = cell * 3;
      const duplicates =
        (cells[base] === pt1 && isTriangle(pt0, cells[base + 1], cells[base + 2])) ||
        (cells[base + 1] === pt1 && isTriangle(cells[base], pt0, cells[base + 2])) ||
        (cells[base + 2] === pt1 && isTriangle(cells[base], cells[base + 1], pt0));
      if (duplicates) {
        removeCellReference(cell);
        cellDeleted[cell] = 1;
        deletedCount += 1;
      } else {
        pointCells[pt0].push(cell);
        for (let corner = 0; corner < 3; corner += 1) {
          if (cells[base + corner] === pt1) {
            cells[base + corner] = pt0;
            break;
          }
        }
      }
    }
    pointCells[pt1] = [];

    return deletedCount;
  };

  // Main collapse loop from RequestData.
  let deletedTriangles = 0;
  let popped = queuePop(queue);
  while (
    popped.id >= 0 &&
    popped.cost < MAXIMUM_ERROR &&
    deletedTriangles / triangleTotal < targetReduction
  ) {
    const edgeId = popped.id;
    const pt0 = endPoint1[edgeId];
    const pt1 = endPoint2[edgeId];
    costWorkspace[0] = targetPoints[edgeId * 3];
    costWorkspace[1] = targetPoints[edgeId * 3 + 1];
    costWorkspace[2] = targetPoints[edgeId * 3 + 2];

    if (!isGoodPlacement(pt0, pt1, costWorkspace)) {
      // Re-queue at maximal cost; the edge is reconsidered only if a
      // neighboring collapse recomputes it.
      queueInsert(queue, MAXIMUM_ERROR, edgeId);
      popped = queuePop(queue);
      continue;
    }

    points[pt0 * 3] = costWorkspace[0];
    points[pt0 * 3 + 1] = costWorkspace[1];
    points[pt0 * 3 + 2] = costWorkspace[2];

    // AddQuadric: merge pt1's quadric into pt0.
    for (let entry = 0; entry < 11; entry += 1) {
      quadrics[pt0 * 11 + entry] += quadrics[pt1 * 11 + entry];
    }

    updateEdgeData(pt0, pt1);
    deletedTriangles += collapseEdge(pt0, pt1);
    popped = queuePop(queue);
  }

  // Output like vtkPolyData::CopyCells: surviving cells in input order with
  // points renumbered by first use.
  const pointMap = new Int32Array(pointCount).fill(-1);
  const outputPolys: number[] = [];
  const outputPointIds: number[] = [];
  for (let cell = 0; cell < triangleTotal; cell += 1) {
    if (cellDeleted[cell]) continue;
    outputPolys.push(3);
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = cells[cell * 3 + corner];
      if (pointMap[vertex] === -1) {
        pointMap[vertex] = outputPointIds.length;
        outputPointIds.push(vertex);
      }
      outputPolys.push(pointMap[vertex]);
    }
  }

  const outputPoints = new Float32Array(outputPointIds.length * 3);
  for (let index = 0; index < outputPointIds.length; index += 1) {
    outputPoints[index * 3] = points[outputPointIds[index] * 3];
    outputPoints[index * 3 + 1] = points[outputPointIds[index] * 3 + 1];
    outputPoints[index * 3 + 2] = points[outputPointIds[index] * 3 + 2];
  }

  return { points: outputPoints, polys: new Uint32Array(outputPolys) };
}
