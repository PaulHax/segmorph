import { createMesh, type Point, type Triangle } from '../geometry/mesh.js';
import { validateLabelValue } from '../image/labelData.js';
import {
  createOrientedImage,
  directionDeterminant,
  indexToWorld,
  type ImageData,
  type OrientedImage,
} from '../image/orientedImage.js';
import { marchingCubesCases, voxelEdges } from './marchingCubesCases.js';

// Per-edge lattice deltas derived from the VTK_VOXEL edge numbering: each edge
// is a unit lattice edge whose min corner is voxelEdges[edge][0]. edgeAxis is
// the varying axis; edgeD* place the shared min corner in [-1, dim] key space.
const edgeAxis = Uint8Array.from(voxelEdges, ([a, b]) => ((a ^ b) === 1 ? 0 : (a ^ b) === 2 ? 1 : 2));
const edgeDx = Uint8Array.from(voxelEdges, ([a]) => a & 1);
const edgeDy = Uint8Array.from(voxelEdges, ([a]) => (a >> 1) & 1);
const edgeDz = Uint8Array.from(voxelEdges, ([a]) => (a >> 2) & 1);

// A corner-occupancy nibble holds the four lattice rows bounding a cell row
// (bit0 row y,z; bit1 row y+1,z; bit2 row y,z+1; bit3 row y+1,z+1). Spreading
// a nibble onto the even bits yields the x-face contribution of the VTK_VOXEL
// case index; the x+1 face lands on the odd bits via one extra shift.
const spreadNibble = Uint8Array.from({ length: 16 }, (_, nibble) => (
  (nibble & 1) | ((nibble & 2) << 1) | ((nibble & 4) << 2) | ((nibble & 8) << 3)
));

/**
 * Extract a label isosurface with VTK's table-driven discrete marching-cubes
 * cases. A one-sample background border closes foreground touching any edge.
 *
 * The scan is restricted to the foreground bounding extent plus that border:
 * per lattice row it visits only the span containing the label, composes each
 * cell's case index from four reads instead of eight, and dedups edge points
 * with numeric lattice keys. Output (point values, point order, triangle
 * order, winding) is bit-identical to the straightforward padded triple loop;
 * test/labelmap-to-surface-exact.spec.ts locks that equivalence.
 */
export function labelmapToSurface<T extends ImageData>(
  input: OrientedImage<T>,
  options: { labelValue: number },
) {
  const image = createOrientedImage(input);
  validateLabelValue(options.labelValue);
  const { labelValue } = options;
  const [width, height, depth] = image.dims;
  const { data } = image;

  // Per lattice row (y + height * z): first/last x holding the label, -1 when
  // absent, plus the foreground bounds along y and z.
  const rowCount = height * depth;
  const firstFg = new Int32Array(rowCount).fill(-1);
  const lastFg = new Int32Array(rowCount).fill(-1);
  let minY = height;
  let maxY = -1;
  let minZ = depth;
  let maxZ = -1;
  for (let row = 0; row < rowCount; row += 1) {
    const base = row * width;
    let first = -1;
    let last = -1;
    for (let x = 0; x < width; x += 1) {
      if (data[base + x] === labelValue) {
        if (first < 0) first = x;
        last = x;
      }
    }
    if (first < 0) continue;
    firstFg[row] = first;
    lastFg[row] = last;
    const y = row % height;
    const z = (row - y) / height;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  if (maxZ < 0) return createMesh([], []);

  const points: Point[] = [];
  const triangles: Triangle[] = [];
  const pointIds = new Map<number, number>();
  const reverseWinding = directionDeterminant(image.direction) < 0;

  // Numeric edge key on the padded lattice (corners range over [-1, dim] per
  // axis): min-corner position plus the edge axis. Exact well past 2048^3.
  const keyWidth = width + 2;
  const keyHeight = height + 2;
  const pointIdAt = (x: number, y: number, z: number, edge: number) => {
    const mx = x + edgeDx[edge];
    const my = y + edgeDy[edge];
    const mz = z + edgeDz[edge];
    const axis = edgeAxis[edge];
    const key = (((mz + 1) * keyHeight + (my + 1)) * keyWidth + (mx + 1)) * 3 + axis;
    const existing = pointIds.get(key);
    if (existing !== undefined) return existing;
    const midpoint = [
      axis === 0 ? mx + 0.5 : mx,
      axis === 1 ? my + 0.5 : my,
      axis === 2 ? mz + 0.5 : mz,
    ] as const;
    const id = points.length;
    points.push(indexToWorld(image, midpoint));
    pointIds.set(key, id);
    return id;
  };

  // Cells outside the visited spans have all eight corners off the label
  // (case 0), so skipping them cannot change the output.
  for (let z = minZ - 1; z <= maxZ; z += 1) {
    for (let y = minY - 1; y <= maxY; y += 1) {
      const rows = [
        y >= 0 && z >= 0 ? y + height * z : -1,
        y + 1 < height && z >= 0 ? y + 1 + height * z : -1,
        y >= 0 && z + 1 < depth ? y + height * (z + 1) : -1,
        y + 1 < height && z + 1 < depth ? y + 1 + height * (z + 1) : -1,
      ] as const;

      let spanFirst = width;
      let spanLast = -1;
      for (const row of rows) {
        if (row < 0 || firstFg[row] < 0) continue;
        if (firstFg[row] < spanFirst) spanFirst = firstFg[row];
        if (lastFg[row] > spanLast) spanLast = lastFg[row];
      }
      if (spanLast < 0) continue;

      const base0 = rows[0] * width;
      const base1 = rows[1] * width;
      const base2 = rows[2] * width;
      const base3 = rows[3] * width;
      const nibbleAt = (x: number) => {
        if (x < 0 || x >= width) return 0;
        return (rows[0] >= 0 && data[base0 + x] === labelValue ? 1 : 0)
          | (rows[1] >= 0 && data[base1 + x] === labelValue ? 2 : 0)
          | (rows[2] >= 0 && data[base2 + x] === labelValue ? 4 : 0)
          | (rows[3] >= 0 && data[base3 + x] === labelValue ? 8 : 0);
      };

      let left = nibbleAt(spanFirst - 1);
      for (let x = spanFirst - 1; x <= spanLast; x += 1) {
        const right = nibbleAt(x + 1);
        const caseIndex = spreadNibble[left] | (spreadNibble[right] << 1);
        left = right;
        if (caseIndex === 0 || caseIndex === 255) continue;

        const triangleEdges = marchingCubesCases[caseIndex];
        for (let offset = 0; triangleEdges[offset] !== -1; offset += 3) {
          const a = pointIdAt(x, y, z, triangleEdges[offset]);
          const b = pointIdAt(x, y, z, triangleEdges[offset + 1]);
          const c = pointIdAt(x, y, z, triangleEdges[offset + 2]);
          triangles.push(reverseWinding ? [a, c, b] : [a, b, c]);
        }
      }
    }
  }

  return createMesh(points, triangles);
}
