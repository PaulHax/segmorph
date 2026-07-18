import { createMesh, getPoint, iterateTriangles, validateMesh, type Mesh, type Point, type Triangle } from '../geometry/mesh.js';
import {
  createOrientedImage,
  directionDeterminant,
  indexToWorld,
  validateImageGeometry,
  worldToIndex,
  type ImageData,
  type ImageGeometry,
  type OrientedImage,
} from '../image/orientedImage.js';
import { marchingCubesCases, voxelCornerOffsets, voxelEdges } from './marchingCubesCases.js';

/**
 * Fractional labelmap conversions, ported from PolySeg (BSD-2):
 * vtkPolyDataToFractionalLabelmapFilter / the fractional conversion rules.
 *
 * Encoding: fractional occupancy is stored as Float32 in [0, 1], the fraction
 * of the 216 sub-voxel sample points inside the surface. This is PolySeg's
 * VTK_FLOAT compile-time option (FRACTIONAL_MIN 0.0, FRACTIONAL_MAX 1.0, step
 * 1/216); PolySeg ships the signed-char variant, related by
 * polysegChar = occupancy * 216 - 108. The 50% surface sits at 0.5 here
 * (0 in the signed-char encoding).
 */

// PolySeg vtkPolyDataToFractionalLabelmapFilter constructor default.
const defaultNumberOfOffsets = 6;

// PolySeg default "Threshold fraction" (surface at 50% occupancy).
const defaultThreshold = 0.5;

type IjkTriangle = {
  // Vertex coordinates in continuous index space.
  x0: number; y0: number; z0: number;
  x1: number; y1: number; z1: number;
  x2: number; y2: number; z2: number;
  minY: number; maxY: number;
  minZ: number; maxZ: number;
  // Surface normal x component and plane offset for solving x at (y, z).
  normalX: number; normalY: number; normalZ: number;
};

function prepareTriangles(mesh: Mesh, geometry: ImageGeometry) {
  const pointCount = mesh.points.length / 3;
  const ijk = new Float64Array(mesh.points.length);
  for (let index = 0; index < pointCount; index += 1) {
    const world = getPoint(mesh, index);
    const point = worldToIndex(geometry, world);
    ijk[index * 3] = point[0];
    ijk[index * 3 + 1] = point[1];
    ijk[index * 3 + 2] = point[2];
  }

  const triangles: IjkTriangle[] = [];
  for (const [a, b, c] of iterateTriangles(mesh)) {
    const x0 = ijk[a * 3]; const y0 = ijk[a * 3 + 1]; const z0 = ijk[a * 3 + 2];
    const x1 = ijk[b * 3]; const y1 = ijk[b * 3 + 1]; const z1 = ijk[b * 3 + 2];
    const x2 = ijk[c * 3]; const y2 = ijk[c * 3 + 1]; const z2 = ijk[c * 3 + 2];
    const normalX = (y1 - y0) * (z2 - z0) - (z1 - z0) * (y2 - y0);
    const normalY = (z1 - z0) * (x2 - x0) - (x1 - x0) * (z2 - z0);
    const normalZ = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    triangles.push({
      x0, y0, z0, x1, y1, z1, x2, y2, z2,
      minY: Math.min(y0, y1, y2),
      maxY: Math.max(y0, y1, y2),
      minZ: Math.min(z0, z1, z2),
      maxZ: Math.max(z0, z1, z2),
      normalX, normalY, normalZ,
    });
  }
  return triangles;
}

/**
 * Even-odd crossings of the +x line through (y, z) with a triangle, using a
 * pnpoly-style half-open edge rule in the (y, z) projection so shared edges
 * between adjacent triangles are counted exactly once.
 */
function lineCrossing(triangle: IjkTriangle, y: number, z: number) {
  const { y0, z0, y1, z1, y2, z2 } = triangle;
  let inside = false;
  if ((z0 > z) !== (z1 > z)
    && y < y0 + ((z - z0) / (z1 - z0)) * (y1 - y0)) inside = !inside;
  if ((z1 > z) !== (z2 > z)
    && y < y1 + ((z - z1) / (z2 - z1)) * (y2 - y1)) inside = !inside;
  if ((z2 > z) !== (z0 > z)
    && y < y2 + ((z - z2) / (z0 - z2)) * (y0 - y2)) inside = !inside;
  if (!inside) return undefined;
  // Solve the plane equation for x; skip triangles seen edge-on.
  if (Math.abs(triangle.normalX) < 1e-12) return undefined;
  return triangle.x0
    - (triangle.normalY * (y - triangle.y0) + triangle.normalZ * (z - triangle.z0))
    / triangle.normalX;
}

/**
 * Voxelize a closed triangle surface into fractional occupancy over the
 * sample points of a reference geometry.
 *
 * Port of PolySeg vtkPolyDataToFractionalLabelmapFilter (RequestData, cxx
 * 371-489): the surface is transformed world -> IJK, then rasterized
 * numberOfOffsets^3 times with per-axis sub-voxel shifts
 * idx / n - (n - 1) / (2n) (cxx 443-466), accumulating a per-voxel count of
 * inside samples. With the default n = 6 that is the 216-step occupancy.
 * Inside-ness per sample uses even-odd parity of surface crossings along the
 * +x index axis, the same classification vtkPolyDataToImageStencil applies
 * per slice.
 *
 * Unlike the PolySeg rule this takes the output geometry explicitly (the
 * caller chooses extent), so the rule's extent-expansion step, including its
 * known skip-K padding bug (vtkClosedSurfaceToFractionalLabelmapConversion
 * Rule.cxx 130), does not apply.
 */
export function surfaceToFractionalLabelmap(
  mesh: Mesh,
  geometry: ImageGeometry,
  options: { numberOfOffsets?: number } = {},
) {
  validateMesh(mesh);
  validateImageGeometry(geometry);
  const numberOfOffsets = options.numberOfOffsets ?? defaultNumberOfOffsets;
  if (!Number.isInteger(numberOfOffsets) || numberOfOffsets < 1) {
    throw new Error('numberOfOffsets must be a positive integer');
  }

  const [nx, ny, nz] = geometry.dims;
  const counts = new Uint32Array(nx * ny * nz);
  const triangles = prepareTriangles(mesh, geometry);

  const offsetStepSize = (numberOfOffsets - 1) / (2 * numberOfOffsets);
  const offsets = Array.from(
    { length: numberOfOffsets },
    (_, index) => index / numberOfOffsets - offsetStepSize,
  );

  const crossings: number[] = [];
  for (const zOffset of offsets) {
    for (const yOffset of offsets) {
      for (let k = 0; k < nz; k += 1) {
        const z = k + zOffset;
        for (let j = 0; j < ny; j += 1) {
          const y = j + yOffset;
          crossings.length = 0;
          for (const triangle of triangles) {
            if (y < triangle.minY || y > triangle.maxY
              || z < triangle.minZ || z > triangle.maxZ) continue;
            const crossing = lineCrossing(triangle, y, z);
            if (crossing !== undefined) crossings.push(crossing);
          }
          if (crossings.length === 0) continue;
          crossings.sort((left, right) => left - right);

          const rowStart = nx * (j + ny * k);
          for (const xOffset of offsets) {
            let next = 0;
            for (let i = 0; i < nx; i += 1) {
              const x = i + xOffset;
              while (next < crossings.length && crossings[next] < x) next += 1;
              if (next % 2 === 1) counts[rowStart + i] += 1;
            }
          }
        }
      }
    }
  }

  const samplesPerVoxel = numberOfOffsets ** 3;
  const data = new Float32Array(counts.length);
  for (let index = 0; index < counts.length; index += 1) {
    data[index] = counts[index] / samplesPerVoxel;
  }
  return createOrientedImage({ ...geometry, data });
}

/**
 * Extract the iso-surface of a fractional labelmap at a threshold fraction.
 *
 * Port of the marching pass of PolySeg
 * vtkFractionalLabelmapToClosedSurfaceConversionRule::Convert (cxx 184-203):
 * linear-interpolation marching cubes (vtkFlyingEdges3D) at
 * iso = threshold * (max - min) + min, which for the [0, 1] occupancy
 * encoding is the threshold itself; samples with value >= iso are inside
 * (vtkFlyingEdges3D.cxx 970). A one-sample zero border closes surfaces at
 * the image boundary, matching the rule's PadLabelmap step (cxx 294-305).
 * The rule's optional decimation and windowed-sinc smoothing stages are
 * separate algorithms in this library and are not applied here.
 *
 * An empty result (threshold above every sample) returns an empty mesh, as
 * the rule does (cxx 209-219).
 *
 * `threshold` must lie in (0, 1]; it defaults to 0.5. Zero is rejected: the
 * occupancy encoding has no iso-crossing at 0 (every sample and the zero
 * padding border satisfy value >= 0), so a zero threshold would silently
 * return an empty mesh rather than a surface.
 */
export function fractionalLabelmapToSurface<T extends ImageData>(
  input: OrientedImage<T>,
  options: { threshold?: number } = {},
) {
  const image = createOrientedImage(input);
  const threshold = options.threshold ?? defaultThreshold;
  // The occupancy encoding spans [0, 1] and the surface is the iso-crossing at
  // `threshold`. Zero is not a valid iso value: every voxel (and the zero
  // padding border) satisfies value >= 0, so nothing crosses and the result is
  // an empty mesh. Require a strictly positive threshold in (0, 1].
  if (!(threshold > 0 && threshold <= 1)) {
    throw new Error('threshold must be within (0, 1]');
  }

  const [width, height, depth] = image.dims;
  const points: Point[] = [];
  const triangles: Triangle[] = [];
  const pointIds = new Map<string, number>();
  const reverseWinding = directionDeterminant(image.direction) < 0;

  const sample = (x: number, y: number, z: number) => (
    x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth
      ? image.data[x + width * (y + height * z)]
      : 0
  );

  // Points are deduplicated by interpolated index position, not by edge:
  // samples exactly at the threshold put the interpolant on a shared corner,
  // and merging those coincident points (as vtkCleanPolyData does after the
  // rule's marching pass) keeps the mesh watertight once the resulting
  // zero-area triangles are skipped.
  const getPointId = (corners: readonly (readonly [number, number, number])[],
    left: number, right: number) => {
    const [ax, ay, az] = corners[left];
    const [bx, by, bz] = corners[right];
    const valueA = sample(ax, ay, az);
    const valueB = sample(bx, by, bz);
    const t = (threshold - valueA) / (valueB - valueA);
    const position = [
      ax + t * (bx - ax),
      ay + t * (by - ay),
      az + t * (bz - az),
    ] as const;
    const key = `${position[0]},${position[1]},${position[2]}`;
    const existing = pointIds.get(key);
    if (existing !== undefined) return existing;

    const id = points.length;
    points.push(indexToWorld(image, position));
    pointIds.set(key, id);
    return id;
  };

  for (let z = -1; z < depth; z += 1) {
    for (let y = -1; y < height; y += 1) {
      for (let x = -1; x < width; x += 1) {
        const corners = voxelCornerOffsets.map(
          ([dx, dy, dz]) => [x + dx, y + dy, z + dz] as const,
        );
        let caseIndex = 0;
        for (let corner = 0; corner < corners.length; corner += 1) {
          const [cx, cy, cz] = corners[corner];
          if (sample(cx, cy, cz) >= threshold) caseIndex |= 1 << corner;
        }
        if (caseIndex === 0 || caseIndex === 255) continue;

        const triangleEdges = marchingCubesCases[caseIndex];
        for (let offset = 0; triangleEdges[offset] !== -1; offset += 3) {
          const ids = [0, 1, 2].map((vertex) => {
            const edgeIndex: number | undefined = triangleEdges[offset + vertex];
            if (edgeIndex === undefined || edgeIndex < 0) {
              throw new Error('Invalid marching-cubes case table');
            }
            const [left, right] = voxelEdges[edgeIndex];
            return getPointId(corners, left, right);
          }) as [number, number, number];
          // Merged coincident interpolants collapse these to zero area.
          if (ids[0] === ids[1] || ids[1] === ids[2] || ids[2] === ids[0]) continue;
          triangles.push(reverseWinding ? [ids[0], ids[2], ids[1]] : ids);
        }
      }
    }
  }

  return createMesh(points, triangles);
}
