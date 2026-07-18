// Dev-only frozen copy of src/convert/labelmapToSurface.ts as of commit
// 4f62ccb (integration base). It is the exact-output reference for the
// performance work on the shipping implementation: any optimization must
// reproduce this function's points and polys arrays bit for bit.
// Do not modify this file when changing src/convert/labelmapToSurface.ts.
import { createMesh, type Point, type Triangle } from '../../src/geometry/mesh.js';
import {
  createOrientedImage,
  indexToWorld,
  type ImageData,
  type OrientedImage,
} from '../../src/image/orientedImage.js';
import { marchingCubesCases } from '../../src/convert/marchingCubesCases.js';

type GridPoint = readonly [number, number, number];

// VTK_VOXEL point ordering: x varies fastest, followed by y and z.
const cornerOffsets: readonly GridPoint[] = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];

// VTK's VoxelCases edge numbering from vtkVoxel.cxx.
const edges = [
  [0, 1], [1, 3], [2, 3], [0, 2],
  [4, 5], [5, 7], [6, 7], [4, 6],
  [0, 4], [1, 5], [2, 6], [3, 7],
] as const;

function determinant(direction: readonly (readonly number[])[]) {
  return direction[0][0] * (direction[1][1] * direction[2][2] - direction[1][2] * direction[2][1])
    - direction[0][1] * (direction[1][0] * direction[2][2] - direction[1][2] * direction[2][0])
    + direction[0][2] * (direction[1][0] * direction[2][1] - direction[1][1] * direction[2][0]);
}

function edgeKey(left: GridPoint, right: GridPoint) {
  const leftKey = left.join(',');
  const rightKey = right.join(',');
  return leftKey < rightKey ? `${leftKey}:${rightKey}` : `${rightKey}:${leftKey}`;
}

/**
 * Extract a label isosurface with VTK's table-driven discrete marching-cubes
 * cases. A one-sample background border closes foreground touching any edge.
 */
export function labelmapToSurfaceBase<T extends ImageData>(
  input: OrientedImage<T>,
  options: { labelValue: number },
) {
  const image = createOrientedImage(input);
  if (!Number.isInteger(options.labelValue)
    || options.labelValue < 1 || options.labelValue > 0xffff_ffff) {
    throw new Error('labelValue must be an integer between 1 and 4294967295');
  }
  const [width, height, depth] = image.dims;
  const points: Point[] = [];
  const triangles: Triangle[] = [];
  const pointIds = new Map<string, number>();
  const reverseWinding = determinant(image.direction) < 0;

  const isForeground = ([x, y, z]: GridPoint) => (
    x >= 0 && x < width && y >= 0 && y < height && z >= 0 && z < depth
      && image.data[x + width * (y + height * z)] === options.labelValue
  );

  const getPointId = (left: GridPoint, right: GridPoint) => {
    const key = edgeKey(left, right);
    const existing = pointIds.get(key);
    if (existing !== undefined) return existing;
    const midpoint = [
      (left[0] + right[0]) / 2,
      (left[1] + right[1]) / 2,
      (left[2] + right[2]) / 2,
    ] as const;
    const id = points.length;
    points.push(indexToWorld(image, midpoint));
    pointIds.set(key, id);
    return id;
  };

  for (let z = -1; z < depth; z += 1) {
    for (let y = -1; y < height; y += 1) {
      for (let x = -1; x < width; x += 1) {
        const corners = cornerOffsets.map(([dx, dy, dz]) => [x + dx, y + dy, z + dz] as const);
        let caseIndex = 0;
        for (let corner = 0; corner < corners.length; corner += 1) {
          if (isForeground(corners[corner])) caseIndex |= 1 << corner;
        }
        if (caseIndex === 0 || caseIndex === 255) continue;

        const triangleEdges = marchingCubesCases[caseIndex];
        for (let offset = 0; triangleEdges[offset] !== -1; offset += 3) {
          const ids = [0, 1, 2].map((vertex) => {
            const edgeIndex: number | undefined = triangleEdges[offset + vertex];
            if (edgeIndex === undefined || edgeIndex < 0) {
              throw new Error('Invalid marching-cubes case table');
            }
            const [left, right] = edges[edgeIndex];
            return getPointId(corners[left], corners[right]);
          }) as [number, number, number];
          triangles.push(reverseWinding ? [ids[0], ids[2], ids[1]] : ids);
        }
      }
    }
  }

  return createMesh(points, triangles);
}
