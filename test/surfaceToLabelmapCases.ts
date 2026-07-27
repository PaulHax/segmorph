import {
  createMesh,
  indexToWorld,
  type ImageGeometry,
  type Mesh,
  type Point,
} from '../src/index.js';

const cubeTriangles = [
  [0, 2, 1],
  [0, 3, 2],
  [4, 5, 6],
  [4, 6, 7],
  [0, 1, 5],
  [0, 5, 4],
  [3, 7, 6],
  [3, 6, 2],
  [0, 4, 7],
  [0, 7, 3],
  [1, 2, 6],
  [1, 6, 5],
] as const;

export function cubeMesh(geometry: ImageGeometry, minimum: number, maximum: number) {
  const indexPoints: Point[] = [
    [minimum, minimum, minimum],
    [maximum, minimum, minimum],
    [maximum, maximum, minimum],
    [minimum, maximum, minimum],
    [minimum, minimum, maximum],
    [maximum, minimum, maximum],
    [maximum, maximum, maximum],
    [minimum, maximum, maximum],
  ];
  return createMesh(
    indexPoints.map((point) => indexToWorld(geometry, point)),
    cubeTriangles,
  );
}

function spherePointsAndTriangles(
  center: Point,
  radius: number,
  polarSteps: number,
  azimuthSteps: number,
) {
  const points: Point[] = [];
  points.push([center[0], center[1], center[2] + radius]);
  for (let row = 1; row < polarSteps; row += 1) {
    const polar = (Math.PI * row) / polarSteps;
    for (let column = 0; column < azimuthSteps; column += 1) {
      const azimuth = (2 * Math.PI * column) / azimuthSteps;
      points.push([
        center[0] + radius * Math.sin(polar) * Math.cos(azimuth),
        center[1] + radius * Math.sin(polar) * Math.sin(azimuth),
        center[2] + radius * Math.cos(polar),
      ]);
    }
  }
  points.push([center[0], center[1], center[2] - radius]);

  const ring = (row: number, column: number) =>
    1 + (row - 1) * azimuthSteps + (column % azimuthSteps);
  const south = points.length - 1;
  const triangles: (readonly [number, number, number])[] = [];
  for (let column = 0; column < azimuthSteps; column += 1) {
    triangles.push([0, ring(1, column), ring(1, column + 1)]);
    triangles.push([south, ring(polarSteps - 1, column + 1), ring(polarSteps - 1, column)]);
  }
  for (let row = 1; row < polarSteps - 1; row += 1) {
    for (let column = 0; column < azimuthSteps; column += 1) {
      triangles.push([ring(row, column), ring(row + 1, column), ring(row + 1, column + 1)]);
      triangles.push([ring(row, column), ring(row + 1, column + 1), ring(row, column + 1)]);
    }
  }
  return { points, triangles };
}

export function sphereMesh(
  center: Point,
  radius: number,
  polarSteps: number,
  azimuthSteps: number,
) {
  const { points, triangles } = spherePointsAndTriangles(center, radius, polarSteps, azimuthSteps);
  return createMesh(points, triangles);
}

/** Two concentric sphere surfaces forming a thin closed shell. */
export function shellMesh(
  center: Point,
  outerRadius: number,
  innerRadius: number,
  polarSteps: number,
  azimuthSteps: number,
) {
  const outer = spherePointsAndTriangles(center, outerRadius, polarSteps, azimuthSteps);
  const inner = spherePointsAndTriangles(center, innerRadius, polarSteps, azimuthSteps);
  const offset = outer.points.length;
  return createMesh(
    [...outer.points, ...inner.points],
    [
      ...outer.triangles,
      ...inner.triangles.map(([a, b, c]) => [a + offset, c + offset, b + offset] as const),
    ],
  );
}

const cos30 = Math.sqrt(3) / 2;

export type RegressionCase = {
  name: string;
  mesh: Mesh;
  geometry: ImageGeometry;
  labelValue: number;
};

const anisotropicGeometry: ImageGeometry = {
  dims: [24, 20, 16],
  spacing: [0.5, 1, 1.5],
  origin: [1, 2, 3],
  direction: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};

const obliqueGeometry: ImageGeometry = {
  dims: [20, 20, 20],
  spacing: [1, 1.25, 0.75],
  origin: [-4, 6, 2],
  direction: [
    [cos30, -0.5, 0],
    [0.5, cos30, 0],
    [0, 0, 1],
  ],
};

const boundaryGeometry: ImageGeometry = {
  dims: [8, 8, 8],
  spacing: [1, 1, 1],
  origin: [0, 0, 0],
  direction: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};

const shellGeometry: ImageGeometry = {
  dims: [32, 32, 32],
  spacing: [0.5, 0.5, 0.5],
  origin: [0, 0, 0],
  direction: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};

export const regressionCases: RegressionCase[] = [
  {
    name: 'anisotropic-sphere',
    mesh: sphereMesh([7, 12, 15], 4, 9, 12),
    geometry: anisotropicGeometry,
    labelValue: 2,
  },
  {
    name: 'oblique-sphere',
    mesh: sphereMesh(indexToWorld(obliqueGeometry, [9.5, 9.5, 9.5]), 7, 11, 14),
    geometry: obliqueGeometry,
    labelValue: 300,
  },
  {
    name: 'boundary-touching-cube',
    mesh: cubeMesh(boundaryGeometry, 0, 7),
    geometry: boundaryGeometry,
    labelValue: 1,
  },
  {
    name: 'thin-shell',
    mesh: shellMesh([8, 8, 8], 6, 5, 13, 16),
    geometry: shellGeometry,
    labelValue: 70000,
  },
  {
    name: 'empty-mesh',
    mesh: createMesh([], []),
    geometry: boundaryGeometry,
    labelValue: 1,
  },
];
