import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { meshSmooth, type MeshSmoothOptions } from '../src/convert/meshSmooth.js';
import { enclosedVolume } from './diff/mesh.js';
import { hasConsistentOutwardOrientation, isManifold, isWatertight } from './diff/structure.js';
import { findFixtureEntries, readFixtureManifest, readMeshJson } from './fixtures/loaders.js';
import { fixturesRoot } from './fixtures/root.js';

// Windowed-sinc smoothing preserves vertex count and ordering, so the
// comparison is per-vertex distance against the golden, not a sampled
// surface metric.
//
// Tolerance calibration (VTK 9.6.2 goldens, measured 2026-07):
// - Oracle precision floor: running the oracle itself with float64 instead
//   of float32 points moves sphere-default vertices by up to 9.50e-6
//   (mean 2.91e-6).
// - Measured TS-port max per-vertex distance per case: sphere-default
//   2.861e-6, sphere-slicer 2.697e-6, cubesphere-default 4.768e-7,
//   torus-default 1.192e-7, halfsphere-default 2.384e-7,
//   halfsphere-noboundary 2.384e-7, sliver-default 3.725e-9.
// 1e-5 sits just above the oracle's own precision spread and 3.5x above the
// worst measured port deviation.
const MAX_VERTEX_DISTANCE = 1e-5;

type SmoothCase = {
  name: string;
  closed: boolean;
};

const cases: SmoothCase[] = [
  { name: 'sphere-default', closed: true },
  { name: 'sphere-slicer', closed: true },
  { name: 'cubesphere-default', closed: true },
  { name: 'torus-default', closed: true },
  { name: 'halfsphere-default', closed: false },
  { name: 'halfsphere-noboundary', closed: false },
  { name: 'sliver-default', closed: false },
];

async function loadManifest() {
  return readFixtureManifest(await readFile(new URL('manifest.json', fixturesRoot), 'utf8'));
}

async function loadMesh(relativePath: string) {
  return readMeshJson(await readFile(new URL(relativePath, fixturesRoot), 'utf8'));
}

function toOptions(params: Record<string, unknown>) {
  const { input, ...options } = params;
  if (typeof input !== 'string') throw new Error('Fixture params must record their input');
  const inputPath = input === 'generated' ? null : input;
  return { inputPath, options: options as MeshSmoothOptions };
}

function maxVertexDistance(a: Float32Array, b: Float32Array) {
  let max = 0;
  for (let offset = 0; offset < a.length; offset += 3) {
    max = Math.max(
      max,
      Math.hypot(
        a[offset] - b[offset],
        a[offset + 1] - b[offset + 1],
        a[offset + 2] - b[offset + 2],
      ),
    );
  }
  return max;
}

describe('meshSmooth vs python-vtk windowed-sinc oracle', () => {
  for (const { name } of cases) {
    it(`matches the golden for ${name}`, async () => {
      const manifest = await loadManifest();
      const entries = findFixtureEntries(manifest, 'B', name);
      expect(entries.length).toBeGreaterThan(0);

      for (const entry of entries) {
        const { inputPath, options } = toOptions(entry.params);
        const input = await loadMesh(inputPath ?? `B/${name}/input.mesh.json`);
        const golden = await loadMesh(`B/${name}/golden.mesh.json`);

        const actual = meshSmooth(input, options);

        expect(actual.points.length).toBe(golden.points.length);
        expect(actual.polys).toEqual(golden.polys);
        expect(actual.polys).toEqual(input.polys);

        const distance = maxVertexDistance(actual.points, golden.points);
        expect(distance).toBeLessThan(MAX_VERTEX_DISTANCE);
      }
    });
  }

  for (const { name, closed } of cases) {
    if (!closed) continue;
    it(`preserves closed structure and volume for ${name}`, async () => {
      const manifest = await loadManifest();
      const [entry] = findFixtureEntries(manifest, 'B', name);
      const { inputPath, options } = toOptions(entry.params);
      const input = await loadMesh(inputPath ?? `B/${name}/input.mesh.json`);
      const golden = await loadMesh(`B/${name}/golden.mesh.json`);

      const actual = meshSmooth(input, options);

      expect(isWatertight(actual)).toBe(true);
      expect(isManifold(actual)).toBe(true);
      expect(hasConsistentOutwardOrientation(actual)).toBe(true);

      // With per-vertex parity at 1e-5, volumes agree far tighter than this
      // band; 1e-3 relative just guards against gross regressions.
      const ratio = enclosedVolume(actual) / enclosedVolume(golden);
      expect(ratio).toBeGreaterThan(0.999);
      expect(ratio).toBeLessThan(1.001);
    });
  }

  it('keeps the smoothed sphere volume within the calibrated band of the input', async () => {
    const manifest = await loadManifest();
    const [entry] = findFixtureEntries(manifest, 'B', 'sphere-default');
    const { options } = toOptions(entry.params);
    const input = await loadMesh('A/sphere/golden.extract.mesh.json');

    const actual = meshSmooth(input, options);

    // Calibration: measured volume ratio 0.999822 with VTK 9.6.2 defaults.
    const ratio = enclosedVolume(actual) / enclosedVolume(input);
    expect(ratio).toBeGreaterThan(0.99);
    expect(ratio).toBeLessThan(1.01);
  });
});

describe('meshSmooth vs vtk.js windowed-sinc oracle', () => {
  // vtk.js's vtkWindowedSincPolyDataFilter is an independent JavaScript
  // reimplementation of the same VTK filter, so agreeing with the python-vtk
  // golden to float precision confirms neither is carrying a shared-VTK bug or
  // a mis-read parameter. vtk.js hardcodes the Hamming window (no window
  // selection), so these cases are Hamming-configured. All three
  // implementations preserve point count and order, so the check is per-vertex.
  const hammingCases = ['cubesphere-hamming', 'torus-hamming', 'sphere-hamming'];

  for (const name of hammingCases) {
    it(`agrees across python-vtk, vtk.js, and our port for ${name}`, async () => {
      const manifest = await loadManifest();
      expect(
        findFixtureEntries(manifest, 'B', name)
          .map((entry) => entry.oracle.name)
          .sort(),
      ).toEqual(['python-vtk', 'vtk-js']);

      const [entry] = findFixtureEntries(manifest, 'B', name);
      const { inputPath, options } = toOptions(entry.params);
      const input = await loadMesh(inputPath ?? `B/${name}/input.mesh.json`);
      const python = await loadMesh(`B/${name}/golden.mesh.json`);
      const vtkjs = await loadMesh(`B/${name}/golden.vtkjs.mesh.json`);

      // Two independent implementations of the filter agree to float precision.
      expect(vtkjs.points.length).toBe(python.points.length);
      expect(maxVertexDistance(vtkjs.points, python.points)).toBeLessThan(MAX_VERTEX_DISTANCE);

      // Our port reproduces the independent vtk.js result vertex for vertex.
      const actual = meshSmooth(input, options);
      expect(actual.polys).toEqual(vtkjs.polys);
      expect(maxVertexDistance(actual.points, vtkjs.points)).toBeLessThan(MAX_VERTEX_DISTANCE);
    });
  }
});
