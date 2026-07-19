import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { surfaceToContour } from '../src/convert/surfaceToContour.js';
import {
  createContourPlane,
  planeToWorld,
  worldToPlane,
  type ContourPlane,
  type Vector3,
} from '../src/geometry/contour.js';
import { findFixtureEntries, readFixtureManifest, readMeshJson } from './fixtures/loaders.js';
import { fixtureUrl } from './fixtures/root.js';

const fixturesUrl = fixtureUrl('');

// Tolerances, calibrated against the committed goldens:
// - The two Python oracle paths (vtkCutter delegation vs a direct
//   vtkPolyDataPlaneCutter pipeline) agree exactly on every case:
//   maxSymmetricDistance 0.0 recorded in each params.json calibration block.
// - Our port therefore only differs from the golden by output storage
//   precision: VTK stores cut points at input (float32) precision while we
//   keep float64. Measured max symmetric world distance ours-vs-golden across
//   all cases: 1.292e-6 (sphere-oblique); max loop area ratio deviation
//   3.29e-9 (sphere-oblique); exact-zero on the synthetic axis-aligned cases.
const maxLoopDistance = 2e-6;
const maxAreaRatioDeviation = 1e-7;

type ContourGolden = {
  plane: { origin: number[]; xAxis: number[]; yAxis: number[] };
  loops: number[][];
};

function readContourGolden(json: string): ContourGolden {
  const value = JSON.parse(json) as ContourGolden;
  if (!Array.isArray(value.loops) || !value.plane) {
    throw new Error('Invalid contour golden JSON');
  }
  return value;
}

function asVector3(values: number[]): Vector3 {
  return [values[0], values[1], values[2]];
}

type WorldLoop = Vector3[];

function goldenWorldLoops(golden: ContourGolden): WorldLoop[] {
  return golden.loops.map((flat) => {
    const loop: WorldLoop = [];
    for (let offset = 0; offset < flat.length; offset += 3) {
      loop.push([flat[offset], flat[offset + 1], flat[offset + 2]]);
    }
    return loop;
  });
}

function contourWorldLoops(plane: ContourPlane, loops: readonly { points: Float64Array }[]) {
  return loops.map((loop) => {
    const world: WorldLoop = [];
    for (let offset = 0; offset < loop.points.length; offset += 2) {
      world.push(planeToWorld(plane, [loop.points[offset], loop.points[offset + 1]]));
    }
    return world;
  });
}

function centroid(loop: WorldLoop): Vector3 {
  const sum = loop.reduce(
    (acc, point) => [acc[0] + point[0], acc[1] + point[1], acc[2] + point[2]],
    [0, 0, 0],
  );
  return [sum[0] / loop.length, sum[1] / loop.length, sum[2] / loop.length];
}

function distance(a: Vector3, b: Vector3) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function pointToPolylineDistance(point: Vector3, loop: WorldLoop) {
  let best = Infinity;
  for (let index = 0; index < loop.length; index += 1) {
    const start = loop[index];
    const end = loop[(index + 1) % loop.length];
    const direction: Vector3 = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    const lengthSquared = direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2;
    let closest = start;
    if (lengthSquared > 0) {
      const t = Math.min(1, Math.max(0, (
        (point[0] - start[0]) * direction[0]
        + (point[1] - start[1]) * direction[1]
        + (point[2] - start[2]) * direction[2]
      ) / lengthSquared));
      closest = [
        start[0] + t * direction[0],
        start[1] + t * direction[1],
        start[2] + t * direction[2],
      ];
    }
    best = Math.min(best, distance(point, closest));
  }
  return best;
}

function symmetricLoopDistance(a: WorldLoop, b: WorldLoop) {
  const forward = Math.max(...a.map((point) => pointToPolylineDistance(point, b)));
  const backward = Math.max(...b.map((point) => pointToPolylineDistance(point, a)));
  return Math.max(forward, backward);
}

function matchLoops(ours: WorldLoop[], golden: WorldLoop[]) {
  const remaining = golden.map((loop, index) => ({ loop, index }));
  return ours.map((loop) => {
    const ourCentroid = centroid(loop);
    let bestSlot = 0;
    for (let slot = 1; slot < remaining.length; slot += 1) {
      if (distance(ourCentroid, centroid(remaining[slot].loop))
        < distance(ourCentroid, centroid(remaining[bestSlot].loop))) {
        bestSlot = slot;
      }
    }
    const [match] = remaining.splice(bestSlot, 1);
    return { ours: loop, golden: match.loop };
  });
}

function enclosedArea(plane: ContourPlane, loop: WorldLoop) {
  const planar = loop.map((point) => worldToPlane(plane, point));
  let area = 0;
  for (let index = 0; index < planar.length; index += 1) {
    const [x0, y0] = planar[index];
    const [x1, y1] = planar[(index + 1) % planar.length];
    area += x0 * y1 - x1 * y0;
  }
  return area / 2;
}

async function loadCase(caseName: string, inputMesh: string) {
  const golden = readContourGolden(
    await readFile(new URL(`F/${caseName}/golden.contour.json`, fixturesUrl), 'utf8'),
  );
  const mesh = readMeshJson(await readFile(new URL(inputMesh, fixturesUrl), 'utf8'));
  const plane = createContourPlane(
    asVector3(golden.plane.origin),
    asVector3(golden.plane.xAxis),
    asVector3(golden.plane.yAxis),
  );
  return { golden, mesh, plane };
}

const manifest = readFixtureManifest(
  await readFile(new URL('manifest.json', fixturesUrl), 'utf8'),
);

const caseNames = [
  'sphere-center',
  'sphere-oblique',
  'sphere-miss',
  'torus-two-loops',
  'cube-axis',
  'octahedron-on-vertices',
];

describe('Python VTK contour oracle', () => {
  for (const caseName of caseNames) {
    const entries = findFixtureEntries(manifest, 'F', caseName);

    it(`records a manifest entry for ${caseName}`, () => {
      expect(entries.length).toBeGreaterThan(0);
    });

    // The golden.contour.json loops are the python-vtk pipeline's assembled
    // output; the vtk.js oracle is cross-checked separately (point sets) below.
    for (const entry of entries.filter((candidate) => candidate.oracle.name === 'python-vtk')) {
      it(`matches ${entry.oracle.name} on ${caseName}`, async () => {
        const inputMesh = entry.params.inputMesh as string;
        const { golden, mesh, plane } = await loadCase(caseName, inputMesh);
        const contour = surfaceToContour(mesh, plane);

        if (golden.loops.length === 0) {
          expect(contour).toBeUndefined();
          return;
        }

        expect(contour).toBeDefined();
        expect(contour!.loops).toHaveLength(golden.loops.length);

        const ourLoops = contourWorldLoops(plane, contour!.loops);
        const goldenLoops = goldenWorldLoops(golden);
        for (const pair of matchLoops(ourLoops, goldenLoops)) {
          expect(symmetricLoopDistance(pair.ours, pair.golden))
            .toBeLessThanOrEqual(maxLoopDistance);

          const ourArea = enclosedArea(plane, pair.ours);
          const goldenArea = Math.abs(enclosedArea(plane, pair.golden));
          // Winding: ours is counterclockwise about xAxis cross yAxis by
          // contract; the oracle's winding is arbitrary, so compare |area|.
          expect(ourArea).toBeGreaterThan(0);
          expect(Math.abs(ourArea / goldenArea - 1)).toBeLessThanOrEqual(maxAreaRatioDeviation);
        }
      });
    }
  }
});

describe('vtk.js contour oracle', () => {
  // vtk.js's vtkCutter is an independent JavaScript reimplementation of the VTK
  // cutter. It assembles loops differently, but the set of plane-mesh
  // intersection points it produces must match the python-vtk golden exactly, so
  // the golden our port is checked against is not a shared-VTK artifact. The
  // point set is order- and assembly-independent, so it needs no loop matching.
  const roundKey = (x: number, y: number, z: number) => (
    [x, y, z].map((value) => (Math.round(value * 1e6) / 1e6).toFixed(6)).join(',')
  );
  const pointSet = (flat: readonly number[]) => {
    const set = new Set<string>();
    for (let offset = 0; offset < flat.length; offset += 3) {
      set.add(roundKey(flat[offset], flat[offset + 1], flat[offset + 2]));
    }
    return [...set].sort();
  };

  for (const caseName of caseNames) {
    it(`vtk.js cut points match the python-vtk golden for ${caseName}`, async () => {
      expect(findFixtureEntries(manifest, 'F', caseName).map((entry) => entry.oracle.name).sort())
        .toEqual(['python-vtk', 'vtk-js']);

      const golden = readContourGolden(
        await readFile(new URL(`F/${caseName}/golden.contour.json`, fixturesUrl), 'utf8'),
      );
      const vtkjs = JSON.parse(
        await readFile(new URL(`F/${caseName}/golden.vtkjs.contour.json`, fixturesUrl), 'utf8'),
      ) as { points: number[] };

      expect(pointSet(vtkjs.points)).toEqual(pointSet(golden.loops.flat()));
    });
  }
});
