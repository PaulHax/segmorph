// Windowed-sinc polygonal mesh smoothing, ported from VTK's
// vtkWindowedSincPolyDataFilter.cxx (BSD-3-Clause, Ken Martin, Will
// Schroeder, Bill Lorensen). Taubin et al., "Optimal Surface Smoothing as
// Filter Design", ECCV 1996: approximate an ideal low-pass filter on mesh
// coordinates with a Chebyshev polynomial expansion in the graph Laplacian.
//
// The port covers triangle meshes (the frozen Mesh contract), so the
// polyline, vertex-cell, and triangle-strip handling of the original filter
// is intentionally absent. Feature-edge smoothing (OptLevel 0) is not
// implemented and is rejected explicitly.
import type { Mesh } from '../geometry/mesh.js';
import { validateMesh } from '../geometry/mesh.js';

export type SmoothingWindowFunction = 'nuttall' | 'blackman' | 'hanning' | 'hamming';

export type MeshSmoothOptions = {
  /** Chebyshev polynomial degree (smoothing passes). VTK default: 20. */
  numberOfIterations?: number;
  /** Pass band of the windowed-sinc filter, in [0, 2]. VTK default: 0.1. */
  passBand?: number;
  /** Temporarily scale coordinates into the unit box. VTK default: false. */
  normalizeCoordinates?: boolean;
  /** Interpolation window. VTK default: 'nuttall'. */
  windowFunction?: SmoothingWindowFunction;
  /** Smooth boundary points along their two boundary edges. VTK default: true. */
  boundarySmoothing?: boolean;
  /** Allow smoothing across non-manifold points. VTK default: false. */
  nonManifoldSmoothing?: boolean;
  /**
   * Duplicate edges of non-manifold groups in the smoothing stencil.
   * Only relevant when nonManifoldSmoothing is true. VTK default: true.
   */
  weightNonManifoldEdges?: boolean;
  /**
   * Boundary points whose two boundary edges meet at more than this angle
   * (degrees, in [0, 180]) are fixed instead of smoothed. VTK default: 15.
   */
  edgeAngle?: number;
  /**
   * Feature-edge smoothing (VTK OptLevel 0) is not implemented by this
   * port; passing true throws. VTK default: false.
   */
  featureEdgeSmoothing?: boolean;
};

const FIXED = 0;
// VTK stores stencil sizes in an unsigned char (EDGE_COUNT_TYPE); points
// with 2 * 255 or more incident edges are fixed.
const MAX_EDGE_COUNT = 255;
const NEWTON_ERROR_TOLERANCE = 1e-3;

const WINDOW_WEIGHTS: Record<SmoothingWindowFunction, (a: number) => number> = {
  nuttall: (a) =>
    0.355768 + 0.487396 * Math.cos(a) + 0.144232 * Math.cos(2 * a) + 0.012604 * Math.cos(3 * a),
  blackman: (a) => 0.42 + 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a),
  hanning: (a) => 0.5 + 0.5 * Math.cos(a),
  hamming: (a) => 0.54 + 0.46 * Math.cos(a),
};

type ResolvedOptions = ReturnType<typeof resolveOptions>;

function resolveOptions(options: MeshSmoothOptions) {
  const resolved = {
    numberOfIterations: options.numberOfIterations ?? 20,
    passBand: options.passBand ?? 0.1,
    normalizeCoordinates: options.normalizeCoordinates ?? false,
    windowFunction: options.windowFunction ?? 'nuttall',
    boundarySmoothing: options.boundarySmoothing ?? true,
    nonManifoldSmoothing: options.nonManifoldSmoothing ?? false,
    weightNonManifoldEdges: options.weightNonManifoldEdges ?? true,
    edgeAngle: options.edgeAngle ?? 15,
  };

  if (options.featureEdgeSmoothing) {
    throw new RangeError('meshSmooth does not implement featureEdgeSmoothing');
  }
  if (!Number.isInteger(resolved.numberOfIterations) || resolved.numberOfIterations < 0) {
    throw new RangeError('numberOfIterations must be a non-negative integer');
  }
  if (!Number.isFinite(resolved.passBand) || resolved.passBand < 0 || resolved.passBand > 2) {
    throw new RangeError('passBand must be within [0, 2]');
  }
  if (!Number.isFinite(resolved.edgeAngle) || resolved.edgeAngle < 0 || resolved.edgeAngle > 180) {
    throw new RangeError('edgeAngle must be within [0, 180] degrees');
  }
  if (!(resolved.windowFunction in WINDOW_WEIGHTS)) {
    throw new RangeError(`Unknown window function: ${String(resolved.windowFunction)}`);
  }

  return resolved;
}

type Connectivity = {
  /** CSR offsets into edges, length numPts + 1. */
  offsets: Int32Array;
  /** Incident-edge neighbor ids; the first stencilSizes[p] entries of each
   * point's slice form the smoothing stencil after analysis. */
  edges: Int32Array;
  /** Number of smoothing edges per point; 0 means the point is fixed. */
  stencilSizes: Uint8Array;
};

// Build the CSR incident-edge structure. Every triangle contributes, for
// each of its points, the two neighbors along the cell perimeter; interior
// manifold edges therefore appear twice, and the duplicate count encodes the
// edge type (1 = boundary, 2 = manifold, >2 = non-manifold).
function buildIncidentEdges(mesh: Mesh, numPts: number) {
  const counts = new Int32Array(numPts + 1);
  const { polys } = mesh;
  for (let offset = 0; offset < polys.length; offset += 4) {
    counts[polys[offset + 1]] += 2;
    counts[polys[offset + 2]] += 2;
    counts[polys[offset + 3]] += 2;
  }

  const offsets = new Int32Array(numPts + 1);
  for (let ptId = 0; ptId < numPts; ptId += 1) {
    offsets[ptId + 1] = offsets[ptId] + counts[ptId];
  }

  const edges = new Int32Array(offsets[numPts]);
  const cursors = offsets.slice(0, numPts);
  const insert = (ptId: number, neighbor: number) => {
    edges[cursors[ptId]] = neighbor;
    cursors[ptId] += 1;
  };
  for (let offset = 0; offset < polys.length; offset += 4) {
    const a = polys[offset + 1];
    const b = polys[offset + 2];
    const c = polys[offset + 3];
    insert(a, c);
    insert(a, b);
    insert(b, a);
    insert(b, c);
    insert(c, b);
    insert(c, a);
  }

  return { offsets, edges };
}

// vtkWindowedSincPolyDataFilter ExceedsEdgeAngle: compare the dot product of
// the two normalized boundary-edge directions against cos(edgeAngle).
function exceedsEdgeAngle(
  points: Float32Array,
  ptId: number,
  pt0: number,
  pt1: number,
  cosEdgeAngle: number,
) {
  const l1 = [0, 0, 0];
  const l2 = [0, 0, 0];
  for (let k = 0; k < 3; k += 1) {
    l1[k] = points[ptId * 3 + k] - points[pt0 * 3 + k];
    l2[k] = points[pt1 * 3 + k] - points[ptId * 3 + k];
  }
  for (const l of [l1, l2]) {
    const norm = Math.sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
    if (norm > 0) {
      l[0] /= norm;
      l[1] /= norm;
      l[2] /= norm;
    }
  }
  return l1[0] * l2[0] + l1[1] * l2[1] + l1[2] * l2[2] < cosEdgeAngle;
}

// OptLevel 2 (no boundary, non-manifold, or feature smoothing): a point is
// smoothed only when its sorted incident edges pair up perfectly.
function buildO2Stencil(edges: Int32Array, start: number, nedges: number) {
  if (nedges % 2) return FIXED;

  const numPairs = nedges / 2;
  let previous = -1;
  for (let i = 0; i < numPairs; i += 1) {
    const value = edges[start + 2 * i];
    if (value === previous || value !== edges[start + 2 * i + 1]) return FIXED;
    previous = value;
    edges[start + i] = value;
  }
  return numPairs;
}

// OptLevel 1 (boundary and/or non-manifold smoothing, no feature edges):
// group the sorted incident edges, classify each group by its duplicate
// count, and compact the stencil to the front of the point's edge slice.
function buildO1Stencil(
  edges: Int32Array,
  start: number,
  nedges: number,
  ptId: number,
  points: Float32Array,
  options: ResolvedOptions,
  cosEdgeAngle: number,
) {
  if (nedges === 1) return FIXED; // end of a polyline in VTK; isolated here

  const { nonManifoldSmoothing, weightNonManifoldEdges } = options;
  const bEdges = [0, 0];
  let totalEdges = 0;
  let numBEdges = 0;
  let numNMEdges = 0;
  let eStart = 0;
  let eEnd = 1;

  while (true) {
    const groupValue = edges[start + eStart];
    while (eEnd < nedges && edges[start + eEnd] === groupValue) eEnd += 1;

    const num = eEnd - eStart;
    if (num === 1) {
      if (numBEdges === 2) return FIXED;
      bEdges[numBEdges] = groupValue;
      numBEdges += 1;
    } else if (num > 2) {
      numNMEdges += 1;
    }

    edges[start + totalEdges] = groupValue;
    totalEdges += 1;
    if (nonManifoldSmoothing && weightNonManifoldEdges) {
      for (let i = 0; i < num - 1; i += 1) {
        edges[start + totalEdges] = groupValue;
        totalEdges += 1;
      }
    }

    if (eEnd >= nedges) break;
    eStart = eEnd;
    eEnd += 1;
  }

  if (numBEdges === 0) {
    if (nonManifoldSmoothing || numNMEdges === 0) {
      // VTK narrows to unsigned char here; keep the modulo for parity.
      return totalEdges & 0xff;
    }
  } else if (numBEdges === 2 && numNMEdges === 0) {
    if (exceedsEdgeAngle(points, ptId, bEdges[0], bEdges[1], cosEdgeAngle)) {
      return FIXED;
    }
    edges[start] = bEdges[0];
    edges[start + 1] = bEdges[1];
    return 2;
  }

  return FIXED;
}

function analyzePoints(mesh: Mesh, numPts: number, options: ResolvedOptions): Connectivity {
  const { offsets, edges } = buildIncidentEdges(mesh, numPts);
  const stencilSizes = new Uint8Array(numPts);
  const optLevel = options.boundarySmoothing || options.nonManifoldSmoothing ? 1 : 2;
  const cosEdgeAngle = Math.cos((options.edgeAngle * Math.PI) / 180);

  for (let ptId = 0; ptId < numPts; ptId += 1) {
    const start = offsets[ptId];
    const nedges = offsets[ptId + 1] - start;
    // Group duplicate edges: manifold edges come in pairs, boundary edges
    // are single, non-manifold edges have more than two duplicates.
    edges.subarray(start, start + nedges).sort();

    if (nedges <= 0 || nedges >= 2 * MAX_EDGE_COUNT) {
      stencilSizes[ptId] = FIXED;
    } else if (optLevel === 2) {
      stencilSizes[ptId] = buildO2Stencil(edges, start, nedges);
    } else {
      stencilSizes[ptId] = buildO1Stencil(
        edges,
        start,
        nedges,
        ptId,
        mesh.points,
        options,
        cosEdgeAngle,
      );
    }
  }

  return { offsets, edges, stencilSizes };
}

// Chebyshev coefficients of the windowed-sinc filter, with the offset sigma
// found by Newton-Raphson so that the filter evaluates to 1 at the pass band.
function computeCoefficients(options: ResolvedOptions) {
  const numIters = options.numberOfIterations;
  const kPb = options.passBand;
  const thetaPb = Math.acos(1 - 0.5 * kPb);
  const basisAtKpb = 1 - 0.5 * kPb;

  const weight = WINDOW_WEIGHTS[options.windowFunction];
  const w = new Float64Array(numIters + 1);
  for (let i = 0; i <= numIters; i += 1) {
    w[i] = weight((i * Math.PI) / (numIters + 1));
  }

  const c = new Float64Array(numIters + 1);
  const cprime = new Float64Array(numIters + 1);
  let sigma = 0;
  let done = false;

  for (let j = 0; !done && j < 500; j += 1) {
    c[0] = (w[0] * (thetaPb + sigma)) / Math.PI;
    for (let i = 1; i <= numIters; i += 1) {
      c[i] = (2 * w[i] * Math.sin(i * (thetaPb + sigma))) / (i * Math.PI);
    }

    cprime[numIters] = 0;
    cprime[numIters - 1] = 0;
    if (numIters > 1) {
      cprime[numIters - 2] = 2 * (numIters - 1) * c[numIters - 1];
    }
    for (let i = numIters - 3; i >= 0; i -= 1) {
      cprime[i] = cprime[i + 2] + 2 * (i + 1) * c[i + 1];
    }

    let fKpb = c[0];
    let fprimeKpb = cprime[0];
    for (let i = 1; i <= numIters; i += 1) {
      const basis = i === 1 ? basisAtKpb : Math.cos(i * Math.acos(basisAtKpb));
      fKpb += c[i] * basis;
      fprimeKpb += cprime[i] * basis;
    }

    if (numIters > 1) {
      if (Math.abs(fKpb - 1) >= NEWTON_ERROR_TOLERANCE) {
        sigma -= (fKpb - 1) / fprimeKpb; // Newton-Raphson toward f(kPb) = 1
      } else {
        done = true;
      }
    } else {
      // Degree 1 cannot be tuned; the mesh will likely shrink (VTK behavior).
      done = true;
      sigma = 0;
    }
  }

  return c;
}

function boundingBoxNormalization(points: Float32Array) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let offset = 0; offset < points.length; offset += 3) {
    for (let k = 0; k < 3; k += 1) {
      const value = points[offset + k];
      min[k] = Math.min(min[k], value);
      max[k] = Math.max(max[k], value);
    }
  }
  // vtkPolyData::GetLength is the bounding-box diagonal; GetCenter its middle.
  const length = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  return { length, center };
}

// The smoothing passes. Four float32 point buffers (matching the input point
// precision, as VTK allocates buffers of the input data type) with all
// arithmetic in doubles: buffer 3 accumulates the Chebyshev expansion and
// buffers 0..2 rotate through x_{i-1}, x_i, x_{i+1}.
function smoothPoints(
  points: Float32Array,
  connectivity: Connectivity,
  c: Float64Array,
  options: ResolvedOptions,
) {
  const numPts = points.length / 3;
  const numIters = options.numberOfIterations;
  const { offsets, edges, stencilSizes } = connectivity;

  let length = 1;
  let center = [0, 0, 0];
  if (options.normalizeCoordinates) {
    ({ length, center } = boundingBoxNormalization(points));
  }

  const buffers = [
    new Float32Array(numPts * 3),
    new Float32Array(numPts * 3),
    new Float32Array(numPts * 3),
    new Float32Array(numPts * 3),
  ];
  const ptSelect = [0, 1, 2, 3];

  if (options.normalizeCoordinates) {
    for (let offset = 0; offset < points.length; offset += 3) {
      for (let k = 0; k < 3; k += 1) {
        buffers[0][offset + k] = (points[offset + k] - center[k]) / length;
      }
    }
  } else {
    buffers[0].set(points);
  }

  // First iteration: x1 = x - 0.5 * L(x), accumulator = c0 * x + c1 * x1.
  {
    const p0 = buffers[ptSelect[0]];
    const p1 = buffers[ptSelect[1]];
    const p3 = buffers[ptSelect[3]];
    for (let ptId = 0; ptId < numPts; ptId += 1) {
      const numEdges = stencilSizes[ptId];
      const start = offsets[ptId];
      const base = ptId * 3;
      const x0 = p0[base];
      const x1 = p0[base + 1];
      const x2 = p0[base + 2];
      let d0 = 0;
      let d1 = 0;
      let d2 = 0;
      for (let j = 0; j < numEdges; j += 1) {
        const neighbor = edges[start + j] * 3;
        d0 += (x0 - p0[neighbor]) / numEdges;
        d1 += (x1 - p0[neighbor + 1]) / numEdges;
        d2 += (x2 - p0[neighbor + 2]) / numEdges;
      }
      d0 = x0 - 0.5 * d0;
      d1 = x1 - 0.5 * d1;
      d2 = x2 - 0.5 * d2;
      p1[base] = d0;
      p1[base + 1] = d1;
      p1[base + 2] = d2;
      p3[base] = c[0] * x0 + c[1] * d0;
      p3[base + 1] = c[0] * x1 + c[1] * d1;
      p3[base + 2] = c[0] * x2 + c[1] * d2;
    }
  }

  // Remaining iterations: x2 = (x1 - x0) + (x1 - L(x1)), acc += c_i * x2.
  for (let iterNum = 2; iterNum <= numIters; iterNum += 1) {
    const p0 = buffers[ptSelect[0]];
    const p1 = buffers[ptSelect[1]];
    const p2 = buffers[ptSelect[2]];
    const p3 = buffers[ptSelect[3]];
    for (let ptId = 0; ptId < numPts; ptId += 1) {
      const numEdges = stencilSizes[ptId];
      const start = offsets[ptId];
      const base = ptId * 3;
      const x10 = p1[base];
      const x11 = p1[base + 1];
      const x12 = p1[base + 2];
      let d0 = 0;
      let d1 = 0;
      let d2 = 0;
      for (let j = 0; j < numEdges; j += 1) {
        const neighbor = edges[start + j] * 3;
        d0 += (x10 - p1[neighbor]) / numEdges;
        d1 += (x11 - p1[neighbor + 1]) / numEdges;
        d2 += (x12 - p1[neighbor + 2]) / numEdges;
      }
      d0 = x10 - p0[base] + x10 - d0;
      d1 = x11 - p0[base + 1] + x11 - d1;
      d2 = x12 - p0[base + 2] + x12 - d2;
      p2[base] = d0;
      p2[base + 1] = d1;
      p2[base + 2] = d2;
      p3[base] += c[iterNum] * d0;
      p3[base + 1] += c[iterNum] * d1;
      p3[base + 2] += c[iterNum] * d2;
    }
    ptSelect[0] = (1 + ptSelect[0]) % 3;
    ptSelect[1] = (1 + ptSelect[1]) % 3;
    ptSelect[2] = (1 + ptSelect[2]) % 3;
  }

  const output = buffers[ptSelect[3]];
  if (options.normalizeCoordinates) {
    for (let offset = 0; offset < output.length; offset += 3) {
      for (let k = 0; k < 3; k += 1) {
        output[offset + k] = output[offset + k] * length + center[k];
      }
    }
  }
  return output;
}

/**
 * Smooth a triangle mesh with VTK's windowed-sinc filter. Vertex count and
 * ordering are preserved and the polys array is copied bit-identically; only
 * point positions change. The input mesh is never mutated.
 *
 * Deviation from VTK: an input with no points or no triangles returns an
 * unchanged copy (VTK returns an empty vtkPolyData after a warning).
 */
export function meshSmooth(mesh: Mesh, options: MeshSmoothOptions = {}): Mesh {
  validateMesh(mesh);
  const resolved = resolveOptions(options);

  const numPts = mesh.points.length / 3;
  const passThrough = numPts < 1 || mesh.polys.length < 4 || resolved.numberOfIterations <= 0;
  if (passThrough) {
    return { points: new Float32Array(mesh.points), polys: new Uint32Array(mesh.polys) };
  }

  const connectivity = analyzePoints(mesh, numPts, resolved);
  const c = computeCoefficients(resolved);
  const points = smoothPoints(mesh.points, connectivity, c, resolved);

  return { points, polys: new Uint32Array(mesh.polys) };
}
