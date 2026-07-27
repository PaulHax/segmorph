import { getPoint, iterateTriangles, validateMesh, type Mesh } from '../geometry/mesh.js';
import {
  createLabelData,
  validateLabelValue,
  type LabelArrayConstructor,
} from '../image/labelData.js';
import {
  createOrientedImage,
  indexToWorld,
  validateImageGeometry,
  worldToIndex,
  type ImageGeometry,
  type Vector3,
} from '../image/orientedImage.js';

export type { LabelArrayConstructor };

const rayDirection: Vector3 = [1, 0.3713906763541037, 0.6947465906068658];

// All voxel rays share rayDirection, so u = y - slopeY * x and v = z - slopeZ * x
// are constant along every ray line. A triangle can only affect a voxel
// (on-surface hit or ray parity crossing) when the voxel's (u, v) falls inside
// the triangle's padded (u, v) bounding box, which lets a 2D grid prune
// candidate triangles without changing any output voxel.
const slopeY = rayDirection[1] / rayDirection[0];
const slopeZ = rayDirection[2] / rayDirection[0];

/**
 * Per-triangle constants of the base implementation, hoisted out of the voxel
 * loop. Every value is computed with the same expressions and operation order
 * as the original per-voxel code, so reusing them is bit-identical.
 */
function prepareTriangles(mesh: Mesh, tolerance: number) {
  const count = mesh.polys.length / 4;
  const vertex = new Float64Array(count * 3);
  const edge1 = new Float64Array(count * 3);
  const edge2 = new Float64Array(count * 3);
  const rayCross = new Float64Array(count * 3);
  const normal = new Float64Array(count * 3);
  const determinant = new Float64Array(count);
  const inverseDeterminant = new Float64Array(count);
  const planeTolerance = new Float64Array(count);
  const dot00 = new Float64Array(count);
  const dot01 = new Float64Array(count);
  const dot11 = new Float64Array(count);
  const denominator = new Float64Array(count);
  const barycentricTolerance = new Float64Array(count);
  const rayActive = new Uint8Array(count);
  const surfaceActive = new Uint8Array(count);

  const toleranceSquared = tolerance * tolerance;
  let index = 0;
  for (const triangle of iterateTriangles(mesh)) {
    const a = getPoint(mesh, triangle[0]);
    const b = getPoint(mesh, triangle[1]);
    const c = getPoint(mesh, triangle[2]);
    const offset = index * 3;
    vertex[offset] = a[0];
    vertex[offset + 1] = a[1];
    vertex[offset + 2] = a[2];
    const e1x = b[0] - a[0];
    const e1y = b[1] - a[1];
    const e1z = b[2] - a[2];
    const e2x = c[0] - a[0];
    const e2y = c[1] - a[1];
    const e2z = c[2] - a[2];
    edge1[offset] = e1x;
    edge1[offset + 1] = e1y;
    edge1[offset + 2] = e1z;
    edge2[offset] = e2x;
    edge2[offset + 1] = e2y;
    edge2[offset + 2] = e2z;

    const px = rayDirection[1] * e2z - rayDirection[2] * e2y;
    const py = rayDirection[2] * e2x - rayDirection[0] * e2z;
    const pz = rayDirection[0] * e2y - rayDirection[1] * e2x;
    rayCross[offset] = px;
    rayCross[offset + 1] = py;
    rayCross[offset + 2] = pz;
    const det = e1x * px + e1y * py + e1z * pz;
    determinant[index] = det;
    const squared1 = e1x * e1x + e1y * e1y + e1z * e1z;
    const squared2 = e2x * e2x + e2y * e2y + e2z * e2z;
    const determinantTolerance = Number.EPSILON * Math.max(1, squared1, squared2) * 64;
    rayActive[index] = Math.abs(det) <= determinantTolerance ? 0 : 1;
    inverseDeterminant[index] = 1 / det;

    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    normal[offset] = nx;
    normal[offset + 1] = ny;
    normal[offset + 2] = nz;
    const normalLengthSquared = nx * nx + ny * ny + nz * nz;
    planeTolerance[index] = toleranceSquared * normalLengthSquared;
    dot00[index] = squared2;
    dot01[index] = e2x * e1x + e2y * e1y + e2z * e1z;
    dot11[index] = squared1;
    denominator[index] = dot00[index] * dot11[index] - dot01[index] * dot01[index];
    surfaceActive[index] = normalLengthSquared === 0 ? 0 : 1;
    barycentricTolerance[index] = tolerance / Math.sqrt(Math.max(squared2, squared1));
    index += 1;
  }

  return {
    count,
    vertex,
    edge1,
    edge2,
    rayCross,
    normal,
    determinant,
    inverseDeterminant,
    planeTolerance,
    dot00,
    dot01,
    dot11,
    denominator,
    barycentricTolerance,
    rayActive,
    surfaceActive,
  };
}

type PreparedTriangles = ReturnType<typeof prepareTriangles>;

/**
 * Uniform 2D grid over the ray-invariant coordinates (u, v). Each triangle is
 * inserted into every cell overlapped by its padded (u, v) bounding box, so a
 * voxel only tests the triangles bucketed at its own (u, v) cell.
 *
 * Padding is conservative: 8 * tolerance covers the on-surface acceptance slab
 * (plane distance <= tolerance plus barycentric slack ~ tolerance in world
 * units) and the 1e-12 barycentric slack of ray hits, and the conditioning
 * terms bound how far floating-point error can move an accepted hit for a
 * nearly degenerate triangle (error ~ EPSILON * reach * scale / divisor). The
 * pad is capped at basePad + 2 * reach, which always suffices because
 * |u(P) - u(A)| <= 2 * |P - A| <= 2 * reach for every voxel center P and
 * triangle vertex A. Over-padding only adds candidates, and extra candidates
 * never change output because the candidate loop reproduces the base
 * implementation's per-triangle arithmetic exactly.
 */
function buildRayGrid(triangles: PreparedTriangles, tolerance: number, reach: number) {
  const { count } = triangles;
  const bounds = new Float64Array(count * 4);
  const basePad = 8 * tolerance;
  const padCap = basePad + 2 * reach;
  let uMin = Infinity;
  let uMax = -Infinity;
  let vMin = Infinity;
  let vMax = -Infinity;
  let inserted = 0;
  for (let index = 0; index < count; index += 1) {
    if (triangles.rayActive[index] === 0 && triangles.surfaceActive[index] === 0) {
      bounds[index * 4] = Infinity;
      bounds[index * 4 + 1] = -Infinity;
      continue;
    }
    const offset = index * 3;
    const ax = triangles.vertex[offset];
    const ay = triangles.vertex[offset + 1];
    const az = triangles.vertex[offset + 2];
    const bx = ax + triangles.edge1[offset];
    const by = ay + triangles.edge1[offset + 1];
    const bz = az + triangles.edge1[offset + 2];
    const cx = ax + triangles.edge2[offset];
    const cy = ay + triangles.edge2[offset + 1];
    const cz = az + triangles.edge2[offset + 2];
    const ua = ay - slopeY * ax;
    const ub = by - slopeY * bx;
    const uc = cy - slopeY * cx;
    const va = az - slopeZ * ax;
    const vb = bz - slopeZ * bx;
    const vc = cz - slopeZ * cx;

    let pad = basePad;
    if (triangles.rayActive[index] === 1) {
      const pLength = Math.sqrt(
        triangles.rayCross[offset] ** 2 +
          triangles.rayCross[offset + 1] ** 2 +
          triangles.rayCross[offset + 2] ** 2,
      );
      const edge1Length = Math.sqrt(triangles.dot11[index]);
      const edge2Length = Math.sqrt(triangles.dot00[index]);
      pad +=
        (Number.EPSILON * reach * (pLength * edge1Length + edge1Length * edge2Length) * 64) /
        Math.abs(triangles.determinant[index]);
    }
    if (triangles.surfaceActive[index] === 1 && triangles.denominator[index] !== 0) {
      pad +=
        (Number.EPSILON * reach * triangles.dot00[index] * triangles.dot11[index] * 64) /
        Math.abs(triangles.denominator[index]);
    }
    pad = Math.min(pad, padCap);

    const triangleUMin = Math.min(ua, ub, uc) - pad;
    const triangleUMax = Math.max(ua, ub, uc) + pad;
    const triangleVMin = Math.min(va, vb, vc) - pad;
    const triangleVMax = Math.max(va, vb, vc) + pad;
    bounds[index * 4] = triangleUMin;
    bounds[index * 4 + 1] = triangleUMax;
    bounds[index * 4 + 2] = triangleVMin;
    bounds[index * 4 + 3] = triangleVMax;
    uMin = Math.min(uMin, triangleUMin);
    uMax = Math.max(uMax, triangleUMax);
    vMin = Math.min(vMin, triangleVMin);
    vMax = Math.max(vMax, triangleVMax);
    inserted += 1;
  }

  const cellsPerAxis =
    inserted === 0 ? 1 : Math.min(128, Math.max(1, Math.ceil(Math.sqrt(inserted))));
  const uScale = uMax > uMin ? cellsPerAxis / (uMax - uMin) : 0;
  const vScale = vMax > vMin ? cellsPerAxis / (vMax - vMin) : 0;
  const cellCount = cellsPerAxis * cellsPerAxis;
  const columnOf = (u: number) =>
    Math.min(cellsPerAxis - 1, Math.max(0, Math.floor((u - uMin) * uScale)));
  const rowOf = (v: number) =>
    Math.min(cellsPerAxis - 1, Math.max(0, Math.floor((v - vMin) * vScale)));

  const cellSizes = new Int32Array(cellCount);
  for (let index = 0; index < count; index += 1) {
    if (bounds[index * 4] > bounds[index * 4 + 1]) continue;
    const column1 = columnOf(bounds[index * 4 + 1]);
    const row1 = rowOf(bounds[index * 4 + 3]);
    for (let row = rowOf(bounds[index * 4 + 2]); row <= row1; row += 1) {
      for (let column = columnOf(bounds[index * 4]); column <= column1; column += 1) {
        cellSizes[row * cellsPerAxis + column] += 1;
      }
    }
  }
  const cellStarts = new Int32Array(cellCount + 1);
  for (let cell = 0; cell < cellCount; cell += 1) {
    cellStarts[cell + 1] = cellStarts[cell] + cellSizes[cell];
  }
  const cellItems = new Int32Array(cellStarts[cellCount]);
  const cursor = cellStarts.slice(0, cellCount);
  for (let index = 0; index < count; index += 1) {
    if (bounds[index * 4] > bounds[index * 4 + 1]) continue;
    const column1 = columnOf(bounds[index * 4 + 1]);
    const row1 = rowOf(bounds[index * 4 + 3]);
    for (let row = rowOf(bounds[index * 4 + 2]); row <= row1; row += 1) {
      for (let column = columnOf(bounds[index * 4]); column <= column1; column += 1) {
        const cell = row * cellsPerAxis + column;
        cellItems[cursor[cell]] = index;
        cursor[cell] += 1;
      }
    }
  }

  return { uMin, uMax, vMin, vMax, cellsPerAxis, uScale, vScale, cellStarts, cellItems };
}

type RayGrid = ReturnType<typeof buildRayGrid>;

/**
 * Same decision procedure as the base implementation's containsPoint:
 * on-surface test first, then Moller-Trumbore parity along rayDirection, with
 * identical expressions and operand order per triangle. Only the candidate set
 * shrinks (via the grid); the set of contributing triangles is unchanged, and
 * the insertion sort produces the same ascending distances as the base
 * implementation's numeric sort of the same multiset.
 */
function containsPoint(
  x: number,
  y: number,
  z: number,
  triangles: PreparedTriangles,
  grid: RayGrid,
  tolerance: number,
  distances: Float64Array,
) {
  const u = y - slopeY * x;
  const v = z - slopeZ * x;
  if (u < grid.uMin || u > grid.uMax || v < grid.vMin || v > grid.vMax) return false;
  const column = Math.min(
    grid.cellsPerAxis - 1,
    Math.max(0, Math.floor((u - grid.uMin) * grid.uScale)),
  );
  const row = Math.min(
    grid.cellsPerAxis - 1,
    Math.max(0, Math.floor((v - grid.vMin) * grid.vScale)),
  );
  const cell = row * grid.cellsPerAxis + column;

  let hitCount = 0;
  const end = grid.cellStarts[cell + 1];
  for (let slot = grid.cellStarts[cell]; slot < end; slot += 1) {
    const index = grid.cellItems[slot];
    const offset = index * 3;
    const fromAX = x - triangles.vertex[offset];
    const fromAY = y - triangles.vertex[offset + 1];
    const fromAZ = z - triangles.vertex[offset + 2];

    if (triangles.surfaceActive[index] === 1) {
      const distanceNumerator =
        fromAX * triangles.normal[offset] +
        fromAY * triangles.normal[offset + 1] +
        fromAZ * triangles.normal[offset + 2];
      if (
        !(distanceNumerator * distanceNumerator > triangles.planeTolerance[index]) &&
        triangles.denominator[index] !== 0
      ) {
        const dot02 =
          triangles.edge2[offset] * fromAX +
          triangles.edge2[offset + 1] * fromAY +
          triangles.edge2[offset + 2] * fromAZ;
        const dot12 =
          triangles.edge1[offset] * fromAX +
          triangles.edge1[offset + 1] * fromAY +
          triangles.edge1[offset + 2] * fromAZ;
        const surfaceU =
          (triangles.dot11[index] * dot02 - triangles.dot01[index] * dot12) /
          triangles.denominator[index];
        const surfaceV =
          (triangles.dot00[index] * dot12 - triangles.dot01[index] * dot02) /
          triangles.denominator[index];
        const slack = triangles.barycentricTolerance[index];
        if (surfaceU >= -slack && surfaceV >= -slack && surfaceU + surfaceV <= 1 + slack) {
          return true;
        }
      }
    }

    if (triangles.rayActive[index] === 1) {
      const inverse = triangles.inverseDeterminant[index];
      const rayU =
        (fromAX * triangles.rayCross[offset] +
          fromAY * triangles.rayCross[offset + 1] +
          fromAZ * triangles.rayCross[offset + 2]) *
        inverse;
      if (rayU < -1e-12 || rayU > 1 + 1e-12) continue;
      const qx = fromAY * triangles.edge1[offset + 2] - fromAZ * triangles.edge1[offset + 1];
      const qy = fromAZ * triangles.edge1[offset] - fromAX * triangles.edge1[offset + 2];
      const qz = fromAX * triangles.edge1[offset + 1] - fromAY * triangles.edge1[offset];
      const rayV = (rayDirection[0] * qx + rayDirection[1] * qy + rayDirection[2] * qz) * inverse;
      if (rayV < -1e-12 || rayU + rayV > 1 + 1e-12) continue;
      const distance =
        (triangles.edge2[offset] * qx +
          triangles.edge2[offset + 1] * qy +
          triangles.edge2[offset + 2] * qz) *
        inverse;
      if (distance > tolerance) {
        let position = hitCount;
        while (position > 0 && distances[position - 1] > distance) {
          distances[position] = distances[position - 1];
          position -= 1;
        }
        distances[position] = distance;
        hitCount += 1;
      }
    }
  }

  let uniqueCount = 0;
  let previous = -Infinity;
  for (let hit = 0; hit < hitCount; hit += 1) {
    if (distances[hit] - previous > tolerance) {
      uniqueCount += 1;
      previous = distances[hit];
    }
  }
  return uniqueCount % 2 === 1;
}

function referenceBounds(mesh: Mesh, geometry: ImageGeometry, tolerance: number) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < mesh.points.length / 3; index += 1) {
    const point = worldToIndex(geometry, getPoint(mesh, index));
    for (let axis = 0; axis < 3; axis += 1) {
      minimum[axis] = Math.min(minimum[axis], point[axis]);
      maximum[axis] = Math.max(maximum[axis], point[axis]);
    }
  }
  return [0, 1, 2].map((axis) => ({
    minimum: Math.max(0, Math.ceil(minimum[axis] - tolerance / geometry.spacing[axis])),
    maximum: Math.min(
      geometry.dims[axis] - 1,
      Math.floor(maximum[axis] + tolerance / geometry.spacing[axis]),
    ),
  }));
}

/** Upper bound on |voxel center - triangle vertex| over all pairs. */
function worldReach(mesh: Mesh, geometry: ImageGeometry) {
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];
  const include = (point: readonly number[]) => {
    for (let axis = 0; axis < 3; axis += 1) {
      minimum[axis] = Math.min(minimum[axis], point[axis]);
      maximum[axis] = Math.max(maximum[axis], point[axis]);
    }
  };
  for (let index = 0; index < mesh.points.length / 3; index += 1) {
    include(getPoint(mesh, index));
  }
  for (let corner = 0; corner < 8; corner += 1) {
    include(
      indexToWorld(geometry, [
        corner & 1 ? geometry.dims[0] - 1 : 0,
        corner & 2 ? geometry.dims[1] - 1 : 0,
        corner & 4 ? geometry.dims[2] - 1 : 0,
      ]),
    );
  }
  return Math.hypot(maximum[0] - minimum[0], maximum[1] - minimum[1], maximum[2] - minimum[2]);
}

/**
 * Rasterize a closed triangle surface onto the sample points of a reference image.
 * Samples on the surface are foreground. All other samples use even-odd containment.
 *
 * By default the output storage width is the narrowest unsigned integer that
 * holds labelValue. Pass outputArray to force a specific element type, e.g. to
 * preserve a Uint16 source dtype across a labelmap -> surface -> labelmap round
 * trip so the result can composite against the original.
 */
export function surfaceToLabelmap(
  mesh: Mesh,
  geometry: ImageGeometry,
  options: { labelValue: number; outputArray?: LabelArrayConstructor },
) {
  validateMesh(mesh);
  validateImageGeometry(geometry);
  validateLabelValue(options.labelValue);

  const coordinates = mesh.points;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const coordinate of coordinates) {
    minimum = Math.min(minimum, coordinate);
    maximum = Math.max(maximum, coordinate);
  }
  const tolerance = Math.max(1, maximum - minimum) * 1e-7;
  const data = createLabelData(
    geometry.dims[0] * geometry.dims[1] * geometry.dims[2],
    options.labelValue,
    options.outputArray,
  );
  const bounds = referenceBounds(mesh, geometry, tolerance);
  const triangles = prepareTriangles(mesh, tolerance);
  const grid = buildRayGrid(triangles, tolerance, worldReach(mesh, geometry));
  const distances = new Float64Array(triangles.count);

  for (let z = bounds[2].minimum; z <= bounds[2].maximum; z += 1) {
    for (let y = bounds[1].minimum; y <= bounds[1].maximum; y += 1) {
      const rowOffset = geometry.dims[0] * (y + geometry.dims[1] * z);
      for (let x = bounds[0].minimum; x <= bounds[0].maximum; x += 1) {
        const world = indexToWorld(geometry, [x, y, z]);
        if (containsPoint(world[0], world[1], world[2], triangles, grid, tolerance, distances)) {
          data[rowOffset + x] = options.labelValue;
        }
      }
    }
  }

  return createOrientedImage({ ...geometry, data });
}
