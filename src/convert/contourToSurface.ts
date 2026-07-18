import { validateMesh, type Mesh } from '../geometry/mesh.js';

// Port of SlicerRT's vtkPlanarContourToClosedSurfaceConversionRule (MIT,
// https://github.com/SlicerRt/SlicerRT, originally developed by Kyle
// Sunderland, PerkLab, Queen's University), the algorithm behind RTSTRUCT ->
// closed surface in Slicer and in @icr/polyseg-wasm (which compiles this same
// rule; it is the oracle for test/oracle-contour-to-surface.spec.ts).
//
// Pipeline, mirrored stage for stage:
// 1. Fit a plane normal per contour, average, and rotate the stack so the
//    contour planes become z-normal (CalculateContourTransform).
// 2. Sort contours by mean z (SortContours).
// 3. Split keyholed loops - the RTSTRUCT technique that encodes a hole by a
//    zero-width channel of coincident point pairs - into separate loops
//    (FixKeyholes, epsilon 0.001, minimum index separation 3).
// 4. Normalize loops counterclockwise in plane xy (SetLinesCounterClockwise).
// 5. For each pair of adjacent planes, divide each loop by its closest
//    overlapping partner (Branch) and stitch the divided loops with a
//    dynamic-programming minimal-distance triangulation
//    (TriangulateBetweenContours).
// 6. Cap every loop edge not triangulated to a neighbor plane (EndCapping):
//    smooth mode rasterizes the loop onto a small binary image
//    (vtkPolyDataToImageStencil convention), erodes it with a 5x5x1
//    elliptical kernel until at most half the pixels remain
//    (vtkImageDilateErode3D), re-extracts the contour (vtkMarchingSquares at
//    value 1 + vtkStripper + FixLines), decimates it (DecimateLines), lifts
//    it a half slice out of plane, ear-cut-triangulates its interior
//    (TriangulateContourInterior) and stitches it to the source loop.
// 7. Rotate the result back.
//
// Deliberate deviations, all combinatorial rather than geometric (the oracle
// spec compares geometrically):
// - The per-contour plane fit uses a covariance eigenanalysis instead of
//   Eigen's BDCSVD (vtkAddonMathUtilities::FitPlaneToPoints); both return the
//   least-squares plane. The normal's sign is arbitrary in both; it only
//   mirrors the intermediate rotated frame, which the final inverse rotation
//   undoes.
// - The ear-cut triangulation picks ears smallest-angle-first rather than
//   reproducing vtkPolygon's priority order; any full triangulation of the
//   same planar polygon covers the same surface.
// - Point locator queries (closest point, radius search) are brute force;
//   ties resolve to the lowest index where VTK's binning order is
//   implementation defined. Only distances feed the algorithm, so ties do
//   not change the output.
//
// Points are stored at float32 precision throughout (the C++ stores vtkPoints
// as float32), so intermediate coordinates match the oracle bit for bit where
// the pipeline is exact.

export type EndCappingMode = 'none' | 'smooth' | 'straight';

export type ContourToSurfaceOptions = {
  /**
   * How to close the surface across the first and last contour planes (and
   * any loop with no overlapping partner on an adjacent plane). Matches the
   * rule's end capping parameter; 'smooth' is the Slicer default.
   */
  endCapping?: EndCappingMode;
  /**
   * Slice thickness used when it cannot be derived from the contours (fewer
   * than two distinct planes). Matches the rule's default slice thickness
   * parameter, default 0.
   */
  defaultSliceThickness?: number;
};

const f32 = Math.fround;

// FixKeyholes parameters (Convert hardcodes them).
const keyholeEpsilon = 0.001;
const keyholeMinimumSeparation = 3;

// CreateSmoothEndCapContour constants (constructor defaults).
const defaultCapSpacing = 1.0;
const alternativeCapDimensions = 28;
const capImagePadding = 4;

type Vec3 = [number, number, number];

/** Growable float32-precision point store shared by every stage. */
type PointStore = number[];

/** A contour as global point ids; closed when first === last. */
type Line = number[];

function insertPoint(points: PointStore, x: number, y: number, z: number) {
  const id = points.length / 3;
  points.push(f32(x), f32(y), f32(z));
  return id;
}

function distanceSquared(points: PointStore, a: number, b: number) {
  const dx = points[a * 3] - points[b * 3];
  const dy = points[a * 3 + 1] - points[b * 3 + 1];
  const dz = points[a * 3 + 2] - points[b * 3 + 2];
  return dx * dx + dy * dy + dz * dz;
}

function distanceSquaredToCoords(points: PointStore, id: number, coords: Vec3) {
  const dx = points[id * 3] - coords[0];
  const dy = points[id * 3 + 1] - coords[1];
  const dz = points[id * 3 + 2] - coords[2];
  return dx * dx + dy * dy + dz * dz;
}

function getPoint(points: PointStore, id: number): Vec3 {
  return [points[id * 3], points[id * 3 + 1], points[id * 3 + 2]];
}

// --- CalculateContourNormal / CalculateContourTransform ---------------------

/**
 * Least-squares plane normal of one loop's vertices: the eigenvector of the
 * centered covariance matrix with the smallest eigenvalue (equivalently the
 * smallest singular vector vtkAddonMathUtilities::FitPlaneToPoints takes from
 * its SVD). Jacobi rotations on the symmetric 3x3.
 */
function fitPlaneNormal(points: PointStore, ids: readonly number[]): Vec3 | undefined {
  const count = ids.length;
  if (count < 3) return undefined;
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (const id of ids) {
    cx += points[id * 3];
    cy += points[id * 3 + 1];
    cz += points[id * 3 + 2];
  }
  cx /= count;
  cy /= count;
  cz /= count;

  // Covariance (upper triangle).
  let xx = 0;
  let xy = 0;
  let xz = 0;
  let yy = 0;
  let yz = 0;
  let zz = 0;
  for (const id of ids) {
    const dx = points[id * 3] - cx;
    const dy = points[id * 3 + 1] - cy;
    const dz = points[id * 3 + 2] - cz;
    xx += dx * dx;
    xy += dx * dy;
    xz += dx * dz;
    yy += dy * dy;
    yz += dy * dz;
    zz += dz * dz;
  }

  const a = [
    [xx, xy, xz],
    [xy, yy, yz],
    [xz, yz, zz],
  ];
  const v = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 32; sweep += 1) {
    let off = 0;
    for (let p = 0; p < 3; p += 1) {
      for (let q = p + 1; q < 3; q += 1) off += a[p][q] * a[p][q];
    }
    if (off < 1e-30) break;
    for (let p = 0; p < 3; p += 1) {
      for (let q = p + 1; q < 3; q += 1) {
        if (a[p][q] === 0) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k += 1) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k += 1) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let smallest = 0;
  for (let index = 1; index < 3; index += 1) {
    if (a[index][index] < a[smallest][smallest]) smallest = index;
  }
  const normal: Vec3 = [v[0][smallest], v[1][smallest], v[2][smallest]];
  const norm = Math.hypot(...normal);
  if (norm === 0) return undefined;
  return [normal[0] / norm, normal[1] / norm, normal[2] / norm];
}

/** CalculateContourNormal: sign-aligned average of per-loop fitted normals. */
function calculateContourNormal(
  points: PointStore,
  lines: readonly Line[],
  minimumContourSize: number,
): Vec3 {
  const sum: Vec3 = [0, 0, 0];
  let smallContours = 0;
  for (const line of lines) {
    // vtkExtractCells counts each referenced point once, so the closure
    // duplicate does not contribute to the size check or the fit.
    const uniqueIds = line[0] === line[line.length - 1] ? line.slice(0, -1) : line.slice();
    if (uniqueIds.length > minimumContourSize) {
      const normal = fitPlaneNormal(points, uniqueIds);
      if (normal) {
        const dot = normal[0] * sum[0] + normal[1] * sum[1] + normal[2] * sum[2];
        const sign = dot < 0 ? -1 : 1;
        sum[0] += sign * normal[0];
        sum[1] += sign * normal[1];
        sum[2] += sign * normal[2];
        continue;
      }
    }
    smallContours += 1;
  }
  if (smallContours >= lines.length) return [0, 0, 0];
  const norm = Math.hypot(...sum);
  if (norm === 0) return [0, 0, 0];
  return [sum[0] / norm, sum[1] / norm, sum[2] / norm];
}

/**
 * CalculateContourTransform: the rotation taking the averaged contour normal
 * onto +z (vtkTransform::RotateWXYZ about normal x z). Returns undefined for
 * an identity rotation. Row-major 3x3.
 */
function calculateContourRotation(points: PointStore, lines: readonly Line[]) {
  let normal = calculateContourNormal(points, lines, 6);
  if (normal[0] === 0 && normal[1] === 0 && normal[2] === 0) {
    normal = calculateContourNormal(points, lines, 0);
  }

  const theta = Math.acos(Math.min(1, Math.max(-1, normal[2])));
  const axis: Vec3 = [normal[1], -normal[0], 0]; // normal x (0,0,1)
  const axisNorm = Math.hypot(axis[0], axis[1]);
  // vtkTransform::RotateWXYZ ignores a zero angle or zero axis.
  if (theta === 0 || axisNorm === 0) return undefined;

  const [ax, ay, az] = [axis[0] / axisNorm, axis[1] / axisNorm, 0];
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const t = 1 - c;
  return [
    [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
    [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
    [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
  ];
}

function applyRotation(points: PointStore, rotation: number[][] | undefined, inverse: boolean) {
  if (!rotation) return;
  for (let id = 0; id < points.length / 3; id += 1) {
    const x = points[id * 3];
    const y = points[id * 3 + 1];
    const z = points[id * 3 + 2];
    if (inverse) {
      points[id * 3] = f32(rotation[0][0] * x + rotation[1][0] * y + rotation[2][0] * z);
      points[id * 3 + 1] = f32(rotation[0][1] * x + rotation[1][1] * y + rotation[2][1] * z);
      points[id * 3 + 2] = f32(rotation[0][2] * x + rotation[1][2] * y + rotation[2][2] * z);
    } else {
      points[id * 3] = f32(rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z);
      points[id * 3 + 1] = f32(rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z);
      points[id * 3 + 2] = f32(rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z);
    }
  }
}

// --- SortContours / bounds --------------------------------------------------

type Bounds = [number, number, number, number, number, number];

function lineBounds(points: PointStore, line: Line): Bounds {
  const bounds: Bounds = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
  for (const id of line) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = points[id * 3 + axis];
      if (value < bounds[axis * 2]) bounds[axis * 2] = value;
      if (value > bounds[axis * 2 + 1]) bounds[axis * 2 + 1] = value;
    }
  }
  return bounds;
}

function lineMeanZ(points: PointStore, line: Line) {
  const bounds = lineBounds(points, line);
  return (bounds[4] + bounds[5]) / 2;
}

/** SortContours: ascending mean z, ties by original index (std::sort on pairs). */
function sortContours(points: PointStore, lines: Line[]) {
  return lines
    .map((line, index) => ({ line, index, z: lineMeanZ(points, line) }))
    .sort((left, right) => (left.z - right.z) || (left.index - right.index))
    .map((entry) => entry.line);
}

// --- FixKeyholes ------------------------------------------------------------

function fixKeyholes(points: PointStore, lines: Line[]) {
  const newLines: Line[] = [];
  for (const line of lines) {
    const count = line.length;
    // flags[i] >= 0 marks i as one end of a keyhole channel, paired with the
    // returned index. Brute-force radius search; matching the C++ loop, only
    // ordered pairs (i < j) can satisfy the separation test.
    const flags = new Int32Array(count).fill(-1);
    let keyholeExists = false;
    const epsilonSquared = keyholeEpsilon * keyholeEpsilon;
    for (let i = 0; i < count; i += 1) {
      for (let j = i + 1; j < count; j += 1) {
        if (distanceSquared(points, line[i], line[j]) > epsilonSquared) continue;
        const separation = Math.min(j - i, count - 1 - j + i);
        if (separation > keyholeMinimumSeparation) {
          keyholeExists = true;
          flags[i] = j;
          flags[j] = i;
        }
      }
    }

    if (!keyholeExists) {
      newLines.push(line);
      continue;
    }

    // Layered splitter, ported statement for statement, including the raw
    // list living in newLines from creation so output order is creation
    // order, and the pop-back that finishes the most recent raw line.
    let currentLayer = 0;
    let pointInChannel = false;
    const rawLines: Line[] = [];
    const finishedLines: Line[] = [];
    for (let index = 0; index < count; index += 1) {
      if (currentLayer === rawLines.length) {
        const created: Line = [];
        newLines.push(created);
        rawLines.push(created);
      }
      const pointId = line[index];
      if (flags[index] === -1) {
        rawLines[currentLayer].push(pointId);
        pointInChannel = false;
      } else if (flags[index] > index && !pointInChannel) {
        rawLines[currentLayer].push(pointId);
        currentLayer += 1;
        pointInChannel = true;
      } else if (flags[index] < index && !pointInChannel) {
        rawLines[currentLayer].push(pointId);
        finishedLines.push(rawLines[currentLayer]);
        rawLines.pop();
        if (currentLayer > 0) currentLayer -= 1;
        pointInChannel = true;
      }
    }
    finishedLines.push(...rawLines);
    for (const finished of finishedLines) {
      if (finished.length > 0 && finished[0] !== finished[finished.length - 1]) {
        finished.push(finished[0]);
      }
    }
  }
  return newLines.filter((line) => line.length > 1);
}

// --- SetLinesCounterClockwise ----------------------------------------------

function isLineClockwise(points: PointStore, line: Line) {
  let areaSum = 0;
  for (let index = 0; index < line.length - 1; index += 1) {
    const a = line[index] * 3;
    const b = line[index + 1] * 3;
    areaSum += (points[b] - points[a]) * (points[b + 1] + points[a + 1]);
  }
  return areaSum > 0;
}

function setLinesCounterClockwise(points: PointStore, lines: Line[]) {
  return lines.map((line) => (isLineClockwise(points, line) ? [...line].reverse() : line));
}

// --- GetSpacingBetweenLines / GetNumberOfLinesOnPlane -----------------------

function getSpacingBetweenLines(
  points: PointStore,
  lines: readonly Line[],
  defaultSliceThickness: number,
) {
  if (lines.length < 2) return defaultSliceThickness;

  const distances: number[] = [];
  let distanceSum = 0;
  for (let lineId = 0; lineId < lines.length - 1; lineId += 1) {
    const distance = Math.abs(
      lineMeanZ(points, lines[lineId]) - lineMeanZ(points, lines[lineId + 1]),
    );
    if (distance > 0.01) {
      distances.push(distance);
      distanceSum += distance;
    }
  }
  if (distances.length === 0) return defaultSliceThickness;

  let mean = distanceSum / distances.length;
  let keptSum = 0;
  let keptCount = 0;
  for (const distance of distances) {
    if (Math.abs(distance - mean) >= mean / 10) continue;
    keptSum += distance;
    keptCount += 1;
  }
  if (keptCount === 0) return mean;
  mean = keptSum / keptCount;
  return mean;
}

function getNumberOfLinesOnPlane(
  points: PointStore,
  lines: readonly Line[],
  originalLineIndex: number,
  spacing: number,
) {
  const threshold = 0.1 * spacing;
  const lineZ = lineMeanZ(points, lines[originalLineIndex]);
  let currentLineId = originalLineIndex + 1;
  while (currentLineId < lines.length
    && Math.abs(lineMeanZ(points, lines[currentLineId]) - lineZ) < threshold) {
    currentLineId += 1;
  }
  return currentLineId - originalLineIndex;
}

function doLinesOverlap(points: PointStore, line1: Line, line2: Line) {
  const bounds1 = lineBounds(points, line1);
  const bounds2 = lineBounds(points, line2);
  return bounds1[0] < bounds2[1]
    && bounds1[1] > bounds2[0]
    && bounds1[2] < bounds2[3]
    && bounds1[3] > bounds2[2];
}

// --- Branch -----------------------------------------------------------------

/** Squared distance from a point to the nearest vertex of a line. */
function distanceSquaredToLine(points: PointStore, coords: Vec3, line: Line) {
  let best = Infinity;
  for (const id of line) {
    const distance = distanceSquaredToCoords(points, id, coords);
    if (distance < best) best = distance;
  }
  return best;
}

function getClosestBranch(
  points: PointStore,
  coords: Vec3,
  overlapKeys: readonly number[],
  overlapLines: readonly Line[],
) {
  if (overlapKeys.length === 1) return overlapKeys[0];
  let minimumDistanceSquared = Infinity;
  let closest = overlapKeys[0];
  for (let index = 0; index < overlapKeys.length; index += 1) {
    const distance = distanceSquaredToLine(points, coords, overlapLines[index]);
    if (distance < minimumDistanceSquared) {
      minimumDistanceSquared = distance;
      closest = overlapKeys[index];
    }
  }
  return closest;
}

/**
 * Branch: keep the portion of `branchingLine` whose closest overlapping line
 * is `targetKey`, plus one extra point after each kept run to close the
 * surface; re-close when the source was closed.
 */
function branchLine(
  points: PointStore,
  branchingLine: Line,
  targetKey: number,
  overlapKeys: readonly number[],
  overlapLines: readonly Line[],
): Line {
  if (overlapKeys.length === 1) return [...branchingLine];

  const output: Line = [];
  let previousKept = false;
  for (const pointId of branchingLine) {
    const coords = getPoint(points, pointId);
    if (getClosestBranch(points, coords, overlapKeys, overlapLines) === targetKey) {
      output.push(pointId);
      previousKept = true;
    } else {
      if (previousKept) output.push(pointId);
      previousKept = false;
    }
  }
  if (output.length > 1) {
    const lineIsClosed = branchingLine[0] === branchingLine[branchingLine.length - 1];
    if (lineIsClosed && output[0] !== output[output.length - 1]) {
      output.push(output[0]);
    }
  }
  return output;
}

// --- TriangulateBetweenContours (dynamic programming stitch) ----------------

function getEndLoop(startLoopIndex: number, numberOfPoints: number, loopClosed: boolean) {
  if (startLoopIndex !== 0) {
    return loopClosed ? startLoopIndex : startLoopIndex - 1;
  }
  return numberOfPoints - 1;
}

function getNextLocation(currentLocation: number, numberOfPoints: number, loopClosed: boolean) {
  if (currentLocation + 1 === numberOfPoints) {
    return loopClosed ? 1 : 0;
  }
  return currentLocation + 1;
}

function getPreviousLocation(currentLocation: number, numberOfPoints: number, loopClosed: boolean) {
  if (currentLocation === 0) {
    return loopClosed ? numberOfPoints - 2 : numberOfPoints - 1;
  }
  return currentLocation - 1;
}

/** GetClosestPoint: position (not id) of the line vertex nearest to coords. */
function getClosestPosition(points: PointStore, coords: Vec3, line: Line) {
  let minimum = distanceSquaredToCoords(points, line[0], coords);
  let closest = 0;
  for (let position = 1; position < line.length; position += 1) {
    const distance = distanceSquaredToCoords(points, line[position], coords);
    if (distance < minimum) {
      minimum = distance;
      closest = position;
    }
  }
  return closest;
}

const BACKTRACK_UP = 0;
const BACKTRACK_LEFT = 1;

function triangulateBetweenContours(
  points: PointStore,
  pointsInLine1: Line,
  pointsInLine2: Line,
  polys: number[],
) {
  const numberOfPointsInLine1 = pointsInLine1.length;
  const numberOfPointsInLine2 = pointsInLine2.length;
  if (numberOfPointsInLine1 === 0 || numberOfPointsInLine2 === 0) return;

  // Pre-calculated closest points, position -> position.
  const closestFrom1To2 = pointsInLine1.map((id) =>
    getClosestPosition(points, getPoint(points, id), pointsInLine2));
  const closestFrom2To1 = pointsInLine2.map((id) =>
    getClosestPosition(points, getPoint(points, id), pointsInLine1));

  const startLine1Position = 0;
  const startLine2Position = closestFrom1To2[0];

  const firstPointLine1 = getPoint(points, pointsInLine1[startLine1Position]);
  const firstPointLine2 = getPoint(points, pointsInLine2[startLine2Position]);

  const line1Closed = pointsInLine1[0] === pointsInLine1[numberOfPointsInLine1 - 1];
  const line2Closed = pointsInLine2[0] === pointsInLine2[numberOfPointsInLine2 - 1];

  const line1EndPosition = getEndLoop(startLine1Position, numberOfPointsInLine1, line1Closed);
  const line2EndPosition = getEndLoop(startLine2Position, numberOfPointsInLine2, line2Closed);

  const score: number[][] = Array.from(
    { length: numberOfPointsInLine1 },
    () => new Array<number>(numberOfPointsInLine2).fill(0),
  );
  const backtrack: number[][] = Array.from(
    { length: numberOfPointsInLine1 },
    () => new Array<number>(numberOfPointsInLine2).fill(BACKTRACK_UP),
  );
  score[0][0] = distanceSquaredToCoords(points, pointsInLine2[startLine2Position], firstPointLine1);

  // First row.
  let currentLine2 = getNextLocation(startLine2Position, numberOfPointsInLine2, line2Closed);
  for (let j = 1; j < numberOfPointsInLine2; j += 1) {
    const distance = distanceSquaredToCoords(points, pointsInLine2[currentLine2], firstPointLine1);
    score[0][j] = score[0][j - 1] + distance;
    backtrack[0][j] = BACKTRACK_LEFT;
    currentLine2 = getNextLocation(currentLine2, numberOfPointsInLine2, line2Closed);
  }

  // First column. The C++ steps this walk with line 2's point count - a
  // faithful transcription of the original (only reachable with a one-point
  // line 2, where GetNextLocation(0, 1, closed) diverges from the correct
  // count).
  let currentLine1 = getNextLocation(startLine1Position, numberOfPointsInLine2, line1Closed);
  for (let i = 1; i < numberOfPointsInLine1; i += 1) {
    const distance = distanceSquaredToCoords(points, pointsInLine1[currentLine1], firstPointLine2);
    score[i][0] = score[i - 1][0] + distance;
    backtrack[i][0] = BACKTRACK_UP;
    currentLine1 = getNextLocation(currentLine1, numberOfPointsInLine1, line1Closed);
  }

  // Fill. The previous-position trackers deliberately carry across rows,
  // exactly as the source does.
  let previousLine1 = startLine1Position;
  let previousLine2 = startLine2Position;
  currentLine1 = getNextLocation(startLine1Position, numberOfPointsInLine1, line1Closed);
  currentLine2 = getNextLocation(startLine2Position, numberOfPointsInLine2, line2Closed);

  for (let i = 1; i < numberOfPointsInLine1; i += 1) {
    const pointOnLine1 = getPoint(points, pointsInLine1[currentLine1]);
    for (let j = 1; j < numberOfPointsInLine2; j += 1) {
      const distance = distanceSquaredToCoords(points, pointsInLine2[currentLine2], pointOnLine1);
      if (currentLine1 === closestFrom2To1[previousLine2]) {
        score[i][j] = score[i][j - 1] + distance;
        backtrack[i][j] = BACKTRACK_LEFT;
      } else if (currentLine2 === closestFrom1To2[previousLine1]) {
        score[i][j] = score[i - 1][j] + distance;
        backtrack[i][j] = BACKTRACK_UP;
      } else if (score[i][j - 1] <= score[i - 1][j]) {
        score[i][j] = score[i][j - 1] + distance;
        backtrack[i][j] = BACKTRACK_LEFT;
      } else {
        score[i][j] = score[i - 1][j] + distance;
        backtrack[i][j] = BACKTRACK_UP;
      }
      previousLine2 = currentLine2;
      currentLine2 = getNextLocation(currentLine2, numberOfPointsInLine2, line2Closed);
    }
    previousLine1 = currentLine1;
    currentLine1 = getNextLocation(currentLine1, numberOfPointsInLine1, line1Closed);
  }

  // Backtrack, emitting one triangle per step.
  currentLine1 = line1EndPosition;
  currentLine2 = line2EndPosition;
  let i = numberOfPointsInLine1 - 1;
  let j = numberOfPointsInLine2 - 1;
  while (i > 0 || j > 0) {
    const vertex0 = pointsInLine1[currentLine1];
    const vertex1 = pointsInLine2[currentLine2];
    let vertex2: number;
    if (backtrack[i][j] === BACKTRACK_LEFT) {
      const previous = getPreviousLocation(currentLine2, numberOfPointsInLine2, line2Closed);
      vertex2 = pointsInLine2[previous];
      j -= 1;
      currentLine2 = previous;
    } else {
      const previous = getPreviousLocation(currentLine1, numberOfPointsInLine1, line1Closed);
      vertex2 = pointsInLine1[previous];
      i -= 1;
      currentLine1 = previous;
    }
    polys.push(3, vertex0, vertex1, vertex2);
  }
}

// --- Smooth end cap: rasterize, erode, re-contour, decimate -----------------

/**
 * Even-odd scanline fill of one closed loop onto the cap grid, in continuous
 * index coordinates, following vtkImageStencilRaster's convention as ported
 * for contourToLabelmap: an edge crosses row j when minY < j <= maxY and a
 * crossing pair (x1, x2) fills columns floor(x1) + 1 through floor(x2).
 */
function rasterizeLoop(
  points: PointStore,
  line: Line,
  originX: number,
  originY: number,
  spacingX: number,
  spacingY: number,
  dimX: number,
  dimY: number,
) {
  const data = new Uint8Array(dimX * dimY);
  const xs: number[] = [];
  const ys: number[] = [];
  for (const id of line) {
    xs.push((points[id * 3] - originX) / spacingX);
    ys.push((points[id * 3 + 1] - originY) / spacingY);
  }
  const edgeCount = line.length - 1;
  const crossings: number[] = [];
  for (let row = 0; row < dimY; row += 1) {
    crossings.length = 0;
    for (let edge = 0; edge < edgeCount; edge += 1) {
      const y0 = ys[edge];
      const y1 = ys[edge + 1];
      if (y0 === y1) continue;
      const lower = Math.min(y0, y1);
      const upper = Math.max(y0, y1);
      if (!(lower < row && row <= upper)) continue;
      const x0 = xs[edge];
      const x1 = xs[edge + 1];
      const crossing = x0 + ((row - y0) * (x1 - x0)) / (y1 - y0);
      crossings.push(Math.min(Math.max(crossing, Math.min(x0, x1)), Math.max(x0, x1)));
    }
    crossings.sort((left, right) => left - right);
    const pairCount = crossings.length - (crossings.length % 2);
    for (let pair = 0; pair < pairCount; pair += 2) {
      const first = Math.max(0, Math.floor(crossings[pair]) + 1);
      const last = Math.min(dimX - 1, Math.floor(crossings[pair + 1]));
      for (let column = first; column <= last; column += 1) {
        data[row * dimX + column] = 1;
      }
    }
  }
  return data;
}

/**
 * vtkImageDilateErode3D erosion with kernel size 5x5x1: elliptical footprint
 * (radius 2.5 pixels about center 2), i.e. all offsets with |dx|,|dy| <= 2
 * except the four corners. Out-of-image neighbors are ignored (the spatial
 * filter shrinks its kernel at boundaries).
 */
function erode(data: Uint8Array, dimX: number, dimY: number) {
  const offsets: Array<[number, number]> = [];
  for (let dy = -2; dy <= 2; dy += 1) {
    for (let dx = -2; dx <= 2; dx += 1) {
      if ((dx / 2.5) ** 2 + (dy / 2.5) ** 2 <= 1) offsets.push([dx, dy]);
    }
  }
  const output = new Uint8Array(data);
  for (let y = 0; y < dimY; y += 1) {
    for (let x = 0; x < dimX; x += 1) {
      if (data[y * dimX + x] !== 1) continue;
      let anyBackground = false;
      for (const [dx, dy] of offsets) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= dimX || ny < 0 || ny >= dimY) continue;
        if (data[ny * dimX + nx] === 0) {
          anyBackground = true;
          break;
        }
      }
      if (anyBackground) output[y * dimX + x] = 0;
    }
  }
  return output;
}

function countForeground(data: Uint8Array) {
  let count = 0;
  for (const value of data) {
    if (value !== 0) count += 1;
  }
  return count;
}

/**
 * vtkMarchingSquares at value 1 on a 0/1 image: every contour vertex lands
 * exactly on a foreground pixel corner, points merge by exact grid position,
 * and degenerate segments are dropped. Returns segments as pairs of local
 * point ids plus the (index-space) point coordinates.
 */
function marchingSquares(data: Uint8Array, dimX: number, dimY: number) {
  // VTK's corner masks: pts0=(i,j)->1, pts1=(i+1,j)->2, pts2=(i,j+1)->8,
  // pts3=(i+1,j+1)->4; edges 0:(0,1) 1:(1,3) 2:(2,3) 3:(0,2).
  const caseTable = [
    [], [0, 3], [1, 0], [1, 3],
    [2, 1], [0, 3, 2, 1], [2, 0], [2, 3],
    [3, 2], [0, 2], [1, 0, 3, 2], [1, 2],
    [3, 1], [0, 1], [3, 0], [],
  ];
  const cornerOffsets = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ];
  const edgeCorners = [
    [0, 1],
    [1, 3],
    [2, 3],
    [0, 2],
  ];

  const pointIds = new Map<number, number>();
  const pointCoords: number[] = [];
  const segments: number[] = [];

  const insert = (x: number, y: number) => {
    const key = y * (dimX + 1) + x;
    const existing = pointIds.get(key);
    if (existing !== undefined) return existing;
    const id = pointCoords.length / 2;
    pointCoords.push(x, y);
    pointIds.set(key, id);
    return id;
  };

  for (let j = 0; j < dimY - 1; j += 1) {
    for (let i = 0; i < dimX - 1; i += 1) {
      const corners = cornerOffsets.map(([dx, dy]) => data[(j + dy) * dimX + (i + dx)]);
      let index = 0;
      if (corners[0] >= 1) index |= 1;
      if (corners[1] >= 1) index |= 2;
      if (corners[2] >= 1) index |= 8;
      if (corners[3] >= 1) index |= 4;
      const edges = caseTable[index];
      for (let pair = 0; pair < edges.length; pair += 2) {
        const ids: number[] = [];
        for (let end = 0; end < 2; end += 1) {
          const [c0, c1] = edgeCorners[edges[pair + end]];
          // t = (1 - s0) / (s1 - s0) is exactly 0 or 1 on a 0/1 image: the
          // vertex is the foreground corner of the edge.
          const corner = corners[c0] >= 1 ? c0 : c1;
          const [dx, dy] = cornerOffsets[corner];
          ids.push(insert(i + dx, j + dy));
        }
        if (ids[0] !== ids[1]) segments.push(ids[0], ids[1]);
      }
    }
  }
  return { pointCoords, segments };
}

/**
 * vtkStripper: join 2-point segments into polylines, starting from each
 * unused segment in cell order and extending forward; a loop that returns to
 * its start keeps the repeated first id.
 */
function stripSegments(segments: number[], pointCount: number) {
  const segmentCount = segments.length / 2;
  const incident: number[][] = Array.from({ length: pointCount }, () => []);
  for (let segment = 0; segment < segmentCount; segment += 1) {
    incident[segments[segment * 2]].push(segment);
    incident[segments[segment * 2 + 1]].push(segment);
  }
  const used = new Uint8Array(segmentCount);
  const lines: number[][] = [];
  for (let seed = 0; seed < segmentCount; seed += 1) {
    if (used[seed]) continue;
    used[seed] = 1;
    const strip = [segments[seed * 2], segments[seed * 2 + 1]];
    for (;;) {
      const tail = strip[strip.length - 1];
      const next = incident[tail].find((candidate) => !used[candidate]);
      if (next === undefined) break;
      used[next] = 1;
      const a = segments[next * 2];
      const b = segments[next * 2 + 1];
      strip.push(a === tail ? b : a);
    }
    lines.push(strip);
  }
  return lines;
}

/** FixLine: drop the first and last point. */
function fixLine(line: number[]) {
  return line.slice(1, -1);
}

/** FixLines: drop short lines and marching-squares self-loop artifacts. */
function fixLines(lines: number[][]) {
  const output: number[][] = [];
  for (const line of lines) {
    if (line.length <= 2) continue;
    if (line[0] === line[2] && line.length !== 3) {
      if (lines.length > 1) continue;
      const fixed = fixLine(line);
      if (fixed.length > 1) output.push(fixed);
    } else if (line.length > 1) {
      output.push(line);
    }
  }
  return output;
}

/** vtkLine::DistanceToLine: squared distance to the infinite line p1p2. */
function distanceToLineSquared(point: Vec3, p1: Vec3, p2: Vec3) {
  const direction = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
  const lengthSquared = direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2;
  const offset = [point[0] - p1[0], point[1] - p1[1], point[2] - p1[2]];
  if (lengthSquared === 0) {
    return offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2;
  }
  const t = (offset[0] * direction[0] + offset[1] * direction[1] + offset[2] * direction[2])
    / lengthSquared;
  const dx = offset[0] - t * direction[0];
  const dy = offset[1] - t * direction[1];
  const dz = offset[2] - t * direction[2];
  return dx * dx + dy * dy + dz * dz;
}

function capPoint(pointCoords: number[], id: number): Vec3 {
  return [pointCoords[id * 3], pointCoords[id * 3 + 1], pointCoords[id * 3 + 2]];
}

/** ComputeError: distance to the line through the neighbors, 0 when they coincide. */
function computeError(pointCoords: number[], lineIds: number[], position: number) {
  const isClosed = lineIds[0] === lineIds[lineIds.length - 1];
  const currentId = lineIds[position];
  const nextId = lineIds[getNextLocation(position, lineIds.length, isClosed)];
  const previousId = lineIds[getPreviousLocation(position, lineIds.length, isClosed)];
  const next = capPoint(pointCoords, nextId);
  const previous = capPoint(pointCoords, previousId);
  const gap = (next[0] - previous[0]) ** 2 + (next[1] - previous[1]) ** 2
    + (next[2] - previous[2]) ** 2;
  if (gap === 0) return 0;
  return distanceToLineSquared(capPoint(pointCoords, currentId), next, previous);
}

/**
 * DecimateLines: point-removal decimation driven by a smallest-error-first
 * priority queue (errors computed once, as in the original; VTK's tie order
 * inside vtkPriorityQueue is implementation defined - here insertion order).
 */
function decimateLines(
  pointCoords: number[],
  lines: number[][],
  decimationFactor: number,
) {
  return lines
    .map((inputLine) => {
      let ids = [...inputLine];
      if (ids.length > 2) {
        const queue = ids
          .map((id, position) => ({ priority: computeError(pointCoords, ids, position), id, position }))
          .sort((left, right) => (left.priority - right.priority) || (left.position - right.position));
        let head = 0;
        while (queue.length - head > 3
          && (queue[head].priority < Number.EPSILON
            || ids.length / inputLine.length > decimationFactor)) {
          const { id } = queue[head];
          head += 1;
          ids = ids.filter((candidate) => candidate !== id);
        }
      }
      if (ids.length > 1 && ids[0] !== ids[ids.length - 1]) {
        ids.push(ids[0]);
      }
      return ids;
    })
    .filter((ids) => ids.length > 1);
}

/**
 * CreateSmoothEndCapContour: shrink the loop to roughly half its area on a
 * small raster, re-extract and decimate the contour, and lift it half a slice
 * out of plane. Falls back to CreateStraightEndCapContour's shape when the
 * raster pipeline produces nothing. Returns new global cap lines.
 */
function createSmoothEndCapContour(
  points: PointStore,
  inputLine: Line,
  lineSpacing: number,
): Line[] {
  const bounds = lineBounds(points, inputLine);

  const alternativeSpacingX = (bounds[1] - bounds[0]) / alternativeCapDimensions;
  const alternativeSpacingY = (bounds[3] - bounds[2]) / alternativeCapDimensions;
  let spacingX = 1;
  let spacingY = 1;
  if (alternativeSpacingX > 0 && alternativeSpacingY > 0) {
    spacingX = Math.min(defaultCapSpacing, alternativeSpacingX);
    spacingY = Math.min(defaultCapSpacing, alternativeSpacingY);
  }

  const pad = capImagePadding / 2;
  const originX = bounds[0] - pad * spacingX;
  const maxX = bounds[1] + pad * spacingX;
  const originY = bounds[2] - pad * spacingY;
  const maxY = bounds[3] + pad * spacingY;
  const originZ = bounds[4];

  const dimX = Math.ceil((maxX - originX) / spacingX);
  const dimY = Math.ceil((maxY - originY) / spacingY);
  if (dimX < 2 || dimY < 2) {
    return [createStraightEndCapContour(points, inputLine, lineSpacing)];
  }

  let image = rasterizeLoop(points, inputLine, originX, originY, spacingX, spacingY, dimX, dimY);

  const totalVoxels = countForeground(image);
  let numberOfVoxels = totalVoxels;
  let voxelDifference = Infinity;
  while (numberOfVoxels > Math.trunc(totalVoxels / 2) && voxelDifference > 0) {
    image = erode(image, dimX, dimY);
    const count = countForeground(image);
    voxelDifference = numberOfVoxels - count;
    numberOfVoxels = count;
  }

  const { pointCoords: indexCoords, segments } = marchingSquares(image, dimX, dimY);
  // vtkImageTransform::TransformPointSet: index -> world by spacing/origin.
  const worldCoords: number[] = [];
  for (let id = 0; id < indexCoords.length / 2; id += 1) {
    worldCoords.push(
      originX + indexCoords[id * 2] * spacingX,
      originY + indexCoords[id * 2 + 1] * spacingY,
      originZ,
    );
  }

  const stripped = stripSegments(segments, worldCoords.length / 3);
  const capLines = fixLines(stripped);

  if (capLines.length > 0 && worldCoords.length > 0) {
    const decimationFactor = (capLines.length * inputLine.length + 1) / (worldCoords.length / 3);
    const decimated = decimateLines(worldCoords, capLines, decimationFactor);

    if (decimated.length > 0) {
      return decimated.map((capLine) => {
        const oriented = isCapLineClockwise(worldCoords, capLine)
          ? [...capLine].reverse()
          : capLine;
        const outputIds = oriented.map((localId) => insertPoint(
          points,
          worldCoords[localId * 3],
          worldCoords[localId * 3 + 1],
          worldCoords[localId * 3 + 2] + lineSpacing / 2,
        ));
        if (outputIds[0] !== outputIds[outputIds.length - 1]) {
          outputIds.push(outputIds[0]);
        }
        return outputIds;
      });
    }
  }
  return [createStraightEndCapContour(points, inputLine, lineSpacing, true)];
}

function isCapLineClockwise(pointCoords: number[], line: number[]) {
  let areaSum = 0;
  for (let index = 0; index < line.length - 1; index += 1) {
    const a = line[index] * 3;
    const b = line[index + 1] * 3;
    areaSum += (pointCoords[b] - pointCoords[a]) * (pointCoords[b + 1] + pointCoords[a + 1]);
  }
  return areaSum > 0;
}

/**
 * CreateStraightEndCapContour: copy the loop half a slice out of plane. The
 * direct mode keeps every point including the closure duplicate and emits an
 * id-open line; the smooth-pipeline fallback drops the closure duplicate and
 * re-closes by id, as the two code paths do in the original.
 */
function createStraightEndCapContour(
  points: PointStore,
  inputLine: Line,
  lineSpacing: number,
  smoothFallback = false,
): Line {
  const sourceIds = smoothFallback ? inputLine.slice(0, -1) : inputLine;
  const outputIds = sourceIds.map((id) => insertPoint(
    points,
    points[id * 3],
    points[id * 3 + 1],
    points[id * 3 + 2] + lineSpacing / 2,
  ));
  if (smoothFallback) outputIds.push(outputIds[0]);
  return outputIds;
}

// --- TriangulateContourInterior (ear-cut cap fill) --------------------------

/**
 * Ear-cut triangulation of a closed cap loop, filling its planar interior.
 * Emits triangles in loop order when normalsUp, reversed otherwise
 * (vtkPolygon triangulation + the original's normal flip).
 */
function triangulateContourInterior(
  points: PointStore,
  inputLine: Line,
  normalsUp: boolean,
  polys: number[],
) {
  let positions = inputLine.map((_, position) => position);
  if (inputLine[0] === inputLine[inputLine.length - 1]) {
    positions = positions.slice(0, -1);
  }
  if (positions.length < 3) return;

  const xy = (position: number): [number, number] => [
    points[inputLine[position] * 3],
    points[inputLine[position] * 3 + 1],
  ];

  // Polygon orientation from the signed area.
  let doubleArea = 0;
  for (let index = 0; index < positions.length; index += 1) {
    const [x0, y0] = xy(positions[index]);
    const [x1, y1] = xy(positions[(index + 1) % positions.length]);
    doubleArea += x0 * y1 - x1 * y0;
  }
  const orientation = doubleArea >= 0 ? 1 : -1;

  const remaining = [...positions];
  const emit = (a: number, b: number, c: number) => {
    if (normalsUp) {
      polys.push(3, inputLine[a], inputLine[b], inputLine[c]);
    } else {
      polys.push(3, inputLine[c], inputLine[b], inputLine[a]);
    }
  };

  const crossZ = (o: [number, number], p: [number, number], q: [number, number]) =>
    (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0]);

  const pointInTriangle = (
    p: [number, number],
    a: [number, number],
    b: [number, number],
    c: [number, number],
  ) => {
    const d1 = crossZ(a, b, p);
    const d2 = crossZ(b, c, p);
    const d3 = crossZ(c, a, p);
    const hasNegative = d1 < 0 || d2 < 0 || d3 < 0;
    const hasPositive = d1 > 0 || d2 > 0 || d3 > 0;
    return !(hasNegative && hasPositive);
  };

  while (remaining.length > 3) {
    let bestEar = -1;
    let bestMeasure = -Infinity;
    for (let index = 0; index < remaining.length; index += 1) {
      const previous = remaining[(index + remaining.length - 1) % remaining.length];
      const current = remaining[index];
      const next = remaining[(index + 1) % remaining.length];
      const a = xy(previous);
      const b = xy(current);
      const c = xy(next);
      const cross = crossZ(a, b, c) * orientation;
      if (cross <= 0) continue;
      let blocked = false;
      for (const other of remaining) {
        if (other === previous || other === current || other === next) continue;
        if (pointInTriangle(xy(other), a, b, c)) {
          blocked = true;
          break;
        }
      }
      if (blocked) continue;
      // Prefer well-shaped ears (largest minimum edge ratio to area), which
      // approximates vtkPolygon's priority without reproducing its queue.
      const measure = cross;
      if (measure > bestMeasure) {
        bestMeasure = measure;
        bestEar = index;
      }
    }
    if (bestEar === -1) {
      // Degenerate remainder (collinear or self-touching): clip the first
      // vertex to guarantee progress, as VTK's fallback fan does.
      bestEar = 0;
    }
    const previous = remaining[(bestEar + remaining.length - 1) % remaining.length];
    const current = remaining[bestEar];
    const next = remaining[(bestEar + 1) % remaining.length];
    emit(previous, current, next);
    remaining.splice(bestEar, 1);
  }
  emit(remaining[0], remaining[1], remaining[2]);
}

// --- EndCapping -------------------------------------------------------------

function endCapping(
  points: PointStore,
  lines: Line[],
  polys: number[],
  triangulatedAbove: boolean[],
  triangulatedBelow: boolean[],
  mode: EndCappingMode,
  defaultSliceThickness: number,
) {
  const numberOfLines = triangulatedAbove.length;
  const lineSpacing = getSpacingBetweenLines(points, lines, defaultSliceThickness);

  for (let lineIndex = 0; lineIndex < numberOfLines; lineIndex += 1) {
    const currentLine = lines[lineIndex];
    for (const direction of ['below', 'above'] as const) {
      const lineTriangulated = direction === 'above'
        ? triangulatedAbove[lineIndex]
        : triangulatedBelow[lineIndex];
      if (lineTriangulated) continue;

      const lineOffset = direction === 'below' ? -lineSpacing : lineSpacing;
      const capLines = mode === 'smooth'
        ? createSmoothEndCapContour(points, currentLine, lineOffset)
        : [createStraightEndCapContour(points, currentLine, lineOffset)];

      for (const capLine of capLines) {
        lines.push(capLine);
        triangulateContourInterior(points, capLine, direction === 'above', polys);
      }

      const overlapKeys = capLines.map((_, key) => key);
      for (let key = 0; key < capLines.length; key += 1) {
        const divided = branchLine(points, currentLine, key, overlapKeys, capLines);
        if (direction === 'above') {
          triangulateBetweenContours(points, divided, capLines[key], polys);
        } else {
          triangulateBetweenContours(points, capLines[key], divided, polys);
        }
      }
    }
  }
}

// --- Entry point ------------------------------------------------------------

/**
 * Convert a stack of planar contours (world-space loops, interleaved xyz,
 * implicit closure - the shape RTSTRUCT ROIs and polyseg-wasm use) into a
 * closed triangle surface.
 *
 * The contours may lie in any set of parallel planes (the stack is rotated so
 * they become z-normal and rotated back afterward); loops on the same plane
 * may branch to multiple loops on the neighbor plane, and RTSTRUCT keyhole
 * loops are split into their outer boundary and holes first.
 */
export function contourToSurface(
  loops: readonly ArrayLike<number>[],
  options?: ContourToSurfaceOptions,
): Mesh {
  if (loops.length === 0) {
    throw new RangeError('Contour to surface requires at least one loop');
  }
  for (const loop of loops) {
    if (loop.length % 3 !== 0 || loop.length < 9) {
      throw new RangeError('Every contour loop needs at least three xyz points');
    }
    for (let index = 0; index < loop.length; index += 1) {
      if (!Number.isFinite(loop[index])) {
        throw new RangeError('Contour loop coordinates must be finite');
      }
    }
  }
  const endCappingMode = options?.endCapping ?? 'smooth';
  const defaultSliceThickness = options?.defaultSliceThickness ?? 0;

  // Assemble the shared point store and closed lines (the polyseg-wasm
  // wrapper closes each cell by repeating its first point id).
  const points: PointStore = [];
  let lines: Line[] = loops.map((loop) => {
    const ids: number[] = [];
    for (let offset = 0; offset < loop.length; offset += 3) {
      ids.push(insertPoint(points, loop[offset], loop[offset + 1], loop[offset + 2]));
    }
    ids.push(ids[0]);
    return ids;
  });

  const rotation = calculateContourRotation(points, lines);
  applyRotation(points, rotation, false);

  lines = sortContours(points, lines);
  lines = fixKeyholes(points, lines);
  lines = setLinesCounterClockwise(points, lines);

  const numberOfLines = lines.length;
  const spacing = getSpacingBetweenLines(points, lines, defaultSliceThickness);

  const triangulatedAbove = new Array<boolean>(numberOfLines).fill(false);
  const triangulatedBelow = new Array<boolean>(numberOfLines).fill(false);
  const polys: number[] = [];

  let firstLineOnPlane1 = 0;
  let linesInPlane1 = numberOfLines > 0
    ? getNumberOfLinesOnPlane(points, lines, 0, spacing)
    : 0;

  while (firstLineOnPlane1 + linesInPlane1 < numberOfLines) {
    const firstLineOnPlane2 = firstLineOnPlane1 + linesInPlane1;
    const linesInPlane2 = getNumberOfLinesOnPlane(points, lines, firstLineOnPlane2, spacing);

    const plane1Overlaps: number[][] = Array.from({ length: linesInPlane1 }, () => []);
    const plane2Overlaps: number[][] = Array.from({ length: linesInPlane2 }, () => []);
    for (let line1 = 0; line1 < linesInPlane1; line1 += 1) {
      for (let line2 = 0; line2 < linesInPlane2; line2 += 1) {
        if (doLinesOverlap(
          points,
          lines[firstLineOnPlane1 + line1],
          lines[firstLineOnPlane2 + line2],
        )) {
          plane1Overlaps[line1].push(firstLineOnPlane2 + line2);
          plane2Overlaps[line2].push(firstLineOnPlane1 + line1);
        }
      }
    }

    for (let line1Index = firstLineOnPlane1;
      line1Index < firstLineOnPlane1 + linesInPlane1;
      line1Index += 1) {
      const line1 = lines[line1Index];
      const overlaps1 = plane1Overlaps[line1Index - firstLineOnPlane1];
      const overlap1Lines = overlaps1.map((id) => lines[id]);

      for (const line2Index of overlaps1) {
        const line2 = lines[line2Index];
        const overlaps2 = plane2Overlaps[line2Index - firstLineOnPlane2];
        const overlap2Lines = overlaps2.map((id) => lines[id]);

        const dividedLine1 = branchLine(points, line1, line2Index, overlaps1, overlap1Lines);
        const dividedLine2 = branchLine(points, line2, line1Index, overlaps2, overlap2Lines);

        if (dividedLine1.length > 1 && dividedLine2.length > 1) {
          triangulatedAbove[line1Index] = true;
          triangulatedBelow[line2Index] = true;
          triangulateBetweenContours(points, dividedLine1, dividedLine2, polys);
        }
      }
    }

    firstLineOnPlane1 = firstLineOnPlane2;
    linesInPlane1 = linesInPlane2;
  }

  if (endCappingMode !== 'none') {
    endCapping(
      points,
      lines,
      polys,
      triangulatedAbove,
      triangulatedBelow,
      endCappingMode,
      defaultSliceThickness,
    );
  }

  applyRotation(points, rotation, true);

  const mesh: Mesh = {
    points: Float32Array.from(points),
    polys: Uint32Array.from(polys),
  };
  validateMesh(mesh);
  return mesh;
}
