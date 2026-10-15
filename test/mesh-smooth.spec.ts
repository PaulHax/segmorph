import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { meshSmooth } from '../src/convert/meshSmooth.js';
import { createMesh, type Mesh } from '../src/geometry/mesh.js';
import { enclosedVolume } from './diff/mesh.js';
import { hasConsistentOutwardOrientation, isManifold, isWatertight } from './diff/structure.js';
import { readMeshJson } from './fixtures/loaders.js';

const sphereUrl = new URL('./fixtures/A/sphere/golden.extract.mesh.json', import.meta.url);

async function loadSphere() {
  return readMeshJson(await readFile(sphereUrl, 'utf8'));
}

function octahedron() {
  return createMesh(
    [
      [1, 0, 0],
      [-1, 0, 0],
      [0, 1, 0],
      [0, -1, 0],
      [0, 0, 1],
      [0, 0, -1],
    ],
    [
      [0, 2, 4],
      [2, 1, 4],
      [1, 3, 4],
      [3, 0, 4],
      [2, 0, 5],
      [1, 2, 5],
      [3, 1, 5],
      [0, 3, 5],
    ],
  );
}

// An open 3x3 vertex grid with a gentle z bump (single interior vertex).
// The bump keeps adjacent boundary edges within the default 15 degree edge
// angle (about 5.7 degrees) so boundary smoothing is allowed to act.
function openGrid() {
  const points: [number, number, number][] = [];
  for (let y = 0; y < 3; y += 1) {
    for (let x = 0; x < 3; x += 1) {
      points.push([x, y, 0.05 * x * (2 - x) + 0.05 * y * (2 - y)]);
    }
  }
  const triangles: [number, number, number][] = [];
  for (let y = 0; y < 2; y += 1) {
    for (let x = 0; x < 2; x += 1) {
      const i = y * 3 + x;
      triangles.push([i, i + 1, i + 4], [i, i + 4, i + 3]);
    }
  }
  return createMesh(points, triangles);
}

function cloneMesh(mesh: Mesh) {
  return { points: new Float32Array(mesh.points), polys: new Uint32Array(mesh.polys) };
}

describe('meshSmooth invariants', () => {
  it('returns identical points and polys for zero iterations', async () => {
    const sphere = await loadSphere();
    const smoothed = meshSmooth(sphere, { numberOfIterations: 0 });

    expect(smoothed).not.toBe(sphere);
    expect(smoothed.points).not.toBe(sphere.points);
    expect(smoothed.polys).not.toBe(sphere.polys);
    expect(smoothed.points).toEqual(sphere.points);
    expect(smoothed.polys).toEqual(sphere.polys);
  });

  it('never mutates the input mesh', async () => {
    const sphere = await loadSphere();
    const pristine = cloneMesh(sphere);

    meshSmooth(sphere);

    expect(sphere.points).toEqual(pristine.points);
    expect(sphere.polys).toEqual(pristine.polys);
  });

  it('is deterministic across runs', async () => {
    const sphere = await loadSphere();
    const first = meshSmooth(sphere);
    const second = meshSmooth(sphere);

    expect(first.points).toEqual(second.points);
    expect(first.polys).toEqual(second.polys);
  });

  it('keeps topology bit-identical and vertex count unchanged', async () => {
    const sphere = await loadSphere();
    const smoothed = meshSmooth(sphere);

    expect(smoothed.polys).toEqual(sphere.polys);
    expect(smoothed.points.length).toBe(sphere.points.length);
  });

  it('actually moves points when smoothing a marching-cubes surface', async () => {
    const sphere = await loadSphere();
    const smoothed = meshSmooth(sphere);

    let maxShift = 0;
    for (let i = 0; i < smoothed.points.length; i += 1) {
      maxShift = Math.max(maxShift, Math.abs(smoothed.points[i] - sphere.points[i]));
    }
    expect(maxShift).toBeGreaterThan(0.01);
  });

  it('preserves closure, orientation, and volume on a closed sphere', async () => {
    const sphere = await loadSphere();
    const smoothed = meshSmooth(sphere);

    expect(isWatertight(smoothed)).toBe(true);
    expect(isManifold(smoothed)).toBe(true);
    expect(hasConsistentOutwardOrientation(smoothed)).toBe(true);

    // Calibration: defaults on the A sphere give a measured volume ratio of
    // 0.999822; band set well outside that.
    const ratio = enclosedVolume(smoothed) / enclosedVolume(sphere);
    expect(ratio).toBeGreaterThan(0.99);
    expect(ratio).toBeLessThan(1.01);
  });

  it('matches the VTK normalizeCoordinates path on a closed sphere', async () => {
    const sphere = await loadSphere();
    const plain = meshSmooth(sphere);
    const normalized = meshSmooth(sphere, { normalizeCoordinates: true });

    // Normalization changes rounding but not geometry: stay within float32
    // noise of the unnormalized result. Calibration: measured max component
    // deviation 3.815e-5 on the A sphere (bbox diagonal ~54).
    let maxDeviation = 0;
    for (let i = 0; i < plain.points.length; i += 1) {
      maxDeviation = Math.max(maxDeviation, Math.abs(plain.points[i] - normalized.points[i]));
    }
    expect(maxDeviation).toBeLessThan(5e-4);
  });

  it('smooths a coarse closed octahedron without breaking structure', () => {
    const mesh = octahedron();
    const smoothed = meshSmooth(mesh, { numberOfIterations: 40, passBand: 0.01 });

    expect(isWatertight(smoothed)).toBe(true);
    expect(hasConsistentOutwardOrientation(smoothed)).toBe(true);
    expect(smoothed.polys).toEqual(mesh.polys);
  });

  it('keeps boundary points pinned when boundary smoothing is off', () => {
    const mesh = openGrid();
    const smoothed = meshSmooth(mesh, {
      numberOfIterations: 20,
      passBand: 0.1,
      boundarySmoothing: false,
    });

    // All points except index 4 sit on the boundary; with boundary smoothing
    // off they are classified FIXED. Fixed points still accumulate
    // sum(c[i])*x = f(0)*x, so they drift by the Newton tolerance of the
    // window approximation, not zero. Calibration: measured max fixed-point
    // drift 4.29e-6 on this grid (coordinates of magnitude <= 2); the Newton
    // tolerance bounds it by about 1e-3 relative.
    for (let index = 0; index < 9; index += 1) {
      if (index === 4) continue;
      for (let k = 0; k < 3; k += 1) {
        const drift = Math.abs(smoothed.points[index * 3 + k] - mesh.points[index * 3 + k]);
        expect(drift).toBeLessThan(1e-4);
      }
    }

    // The interior point must genuinely move (it has a full stencil).
    const interiorShift = Math.hypot(
      smoothed.points[12] - mesh.points[12],
      smoothed.points[13] - mesh.points[13],
      smoothed.points[14] - mesh.points[14],
    );
    expect(interiorShift).toBeGreaterThan(1e-2);
  });

  it('moves boundary points along the boundary with default boundary smoothing', () => {
    const mesh = openGrid();
    const smoothed = meshSmooth(mesh, { numberOfIterations: 20, passBand: 0.1 });

    // Edge-midpoint boundary vertices (1, 3, 5, 7) smooth along the boundary
    // pair; corner vertices have two boundary edges too and may move.
    let boundaryShift = 0;
    for (const index of [1, 3, 5, 7]) {
      boundaryShift = Math.max(
        boundaryShift,
        Math.hypot(
          smoothed.points[index * 3] - mesh.points[index * 3],
          smoothed.points[index * 3 + 1] - mesh.points[index * 3 + 1],
          smoothed.points[index * 3 + 2] - mesh.points[index * 3 + 2],
        ),
      );
    }
    expect(boundaryShift).toBeGreaterThan(1e-3);
  });

  it('handles a degenerate repeated-vertex triangle without crashing', () => {
    const points = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]);
    const polys = new Uint32Array([
      3,
      0,
      1,
      2,
      3,
      1,
      3,
      2,
      3,
      3,
      3,
      2, // degenerate: repeated vertex
    ]);
    const smoothed = meshSmooth({ points, polys }, { numberOfIterations: 10 });

    expect(smoothed.polys).toEqual(polys);
    expect([...smoothed.points].every(Number.isFinite)).toBe(true);
  });

  it('returns an unchanged copy for an empty mesh', () => {
    const empty = { points: new Float32Array(0), polys: new Uint32Array(0) };
    const smoothed = meshSmooth(empty);

    expect(smoothed.points.length).toBe(0);
    expect(smoothed.polys.length).toBe(0);
    expect(smoothed.points).not.toBe(empty.points);
  });

  it('rejects invalid options instead of silently ignoring them', async () => {
    const sphere = await loadSphere();

    expect(() => meshSmooth(sphere, { featureEdgeSmoothing: true })).toThrow(
      /featureEdgeSmoothing/,
    );
    expect(() => meshSmooth(sphere, { passBand: -0.1 })).toThrow(RangeError);
    expect(() => meshSmooth(sphere, { passBand: 2.5 })).toThrow(RangeError);
    expect(() => meshSmooth(sphere, { passBand: Number.NaN })).toThrow(RangeError);
    expect(() => meshSmooth(sphere, { numberOfIterations: -1 })).toThrow(RangeError);
    expect(() => meshSmooth(sphere, { numberOfIterations: 2.5 })).toThrow(RangeError);
    expect(() => meshSmooth(sphere, { edgeAngle: -5 })).toThrow(RangeError);
    expect(() => meshSmooth(sphere, { edgeAngle: 200 })).toThrow(RangeError);
    // @ts-expect-error unknown window functions are rejected at runtime too
    expect(() => meshSmooth(sphere, { windowFunction: 'kaiser' })).toThrow(RangeError);
  });
});

describe('meshSmooth numerical failures', () => {
  const windows = ['nuttall', 'blackman', 'hanning', 'hamming'] as const;

  it.each(windows)(
    'rejects nonfinite filter gain with %s without mutating input',
    (windowFunction) => {
      const mesh = octahedron();
      const pristine = cloneMesh(mesh);

      // At passBand 0 the first Newton step divides by zero. Its NaN residual
      // must not be mistaken for a gain within tolerance.
      for (const numberOfIterations of [2, 3, 20, 40]) {
        const smooth = () => meshSmooth(mesh, { numberOfIterations, passBand: 0, windowFunction });
        expect(smooth).toThrow(RangeError);
        expect(smooth).toThrow(
          'meshSmooth filter gain became nonfinite; adjust passBand or numberOfIterations',
        );
        expect(mesh).toEqual(pristine);
      }
    },
  );

  it.each([
    { numberOfIterations: 2, passBand: 2, windowFunction: 'nuttall' },
    ...windows.map((windowFunction) => ({
      numberOfIterations: 3,
      passBand: 0.0001,
      windowFunction,
    })),
  ] as const)(
    'rejects exhausted coefficient tuning: $numberOfIterations iterations, passBand $passBand, $windowFunction',
    (options) => {
      const mesh = octahedron();
      const pristine = cloneMesh(mesh);

      // The two-iteration Nuttall case previously returned a finite radius of
      // 697792.0625. Finiteness alone does not establish filter convergence.
      const smooth = () => meshSmooth(mesh, options);
      expect(smooth).toThrow(RangeError);
      expect(smooth).toThrow(
        'meshSmooth filter did not converge after 500 attempts; adjust passBand or numberOfIterations',
      );
      expect(mesh).toEqual(pristine);
    },
  );

  it('rejects nonfinite final coordinates even when coefficient tuning converges', () => {
    const largestFloat32 = 3.4028234663852886e38;
    const mesh = createMesh(
      [
        [largestFloat32, 0, 0],
        [largestFloat32, 1, 0],
        [largestFloat32, 0, 1],
      ],
      [[0, 1, 2]],
    );
    const pristine = cloneMesh(mesh);

    // The default filter converges, but its accumulation overflows Float32.
    const smooth = () => meshSmooth(mesh);
    expect(smooth).toThrow(RangeError);
    expect(smooth).toThrow('meshSmooth produced nonfinite point coordinates');
    expect(mesh).toEqual(pristine);

    const normalized = meshSmooth(mesh, { normalizeCoordinates: true });
    expect(normalized.points.every(Number.isFinite)).toBe(true);
    expect(normalized.points[0]).toBe(largestFloat32);
    expect(normalized.polys).toEqual(mesh.polys);
    expect(mesh).toEqual(pristine);
  });

  it.each(windows)('preserves coincident normalized points with %s', (windowFunction) => {
    for (const position of [
      [0, 0, 0],
      [3, -4, 5],
    ] as const) {
      for (const numberOfIterations of [1, 20]) {
        const mesh = octahedron();
        for (let offset = 0; offset < mesh.points.length; offset += 3) {
          mesh.points.set(position, offset);
        }
        const pristine = cloneMesh(mesh);
        const smoothed = meshSmooth(mesh, {
          normalizeCoordinates: true,
          numberOfIterations,
          windowFunction,
        });

        expect(smoothed).toEqual(pristine);
        expect(smoothed.points).not.toBe(mesh.points);
        expect(smoothed.polys).not.toBe(mesh.polys);
        expect(mesh).toEqual(pristine);
      }
    }
  });

  it.each([
    { windowFunction: 'nuttall', scale: 0.12210766226053238 },
    { windowFunction: 'blackman', scale: 0.13487598299980164 },
    { windowFunction: 'hanning', scale: 0.15077874064445496 },
    { windowFunction: 'hamming', scale: 0.1547544300556183 },
  ] as const)(
    'preserves degree-one shrink behavior with $windowFunction',
    ({ windowFunction, scale }) => {
      const mesh = octahedron();
      const pristine = cloneMesh(mesh);
      const smoothed = meshSmooth(mesh, { numberOfIterations: 1, passBand: 0.1, windowFunction });

      // Float32 radii independently reproduced by Python VTK 9.6.2.
      expect(smoothed.points).toEqual(mesh.points.map((coordinate) => coordinate * scale));
      expect(smoothed.polys).toEqual(mesh.polys);

      const zeroPassBand = meshSmooth(mesh, { numberOfIterations: 1, passBand: 0, windowFunction });
      expect(zeroPassBand.points.every((coordinate) => coordinate === 0)).toBe(true);
      expect(zeroPassBand.polys).toEqual(mesh.polys);
      expect(mesh).toEqual(pristine);
    },
  );

  it('still passes through zero iterations with a zero pass band', () => {
    const mesh = octahedron();
    const pristine = cloneMesh(mesh);
    const smoothed = meshSmooth(mesh, {
      numberOfIterations: 0,
      passBand: 0,
      normalizeCoordinates: true,
    });

    expect(smoothed).toEqual(pristine);
    expect(smoothed.points).not.toBe(mesh.points);
    expect(smoothed.polys).not.toBe(mesh.polys);
    expect(mesh).toEqual(pristine);
  });
});
