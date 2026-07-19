import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { readNrrd } from '../src/io/nrrd.js';
import { createOrientedImage } from '../src/image/orientedImage.js';
import { labelmapToSurface } from '../src/convert/labelmapToSurface.js';
import { meshSmooth } from '../src/convert/meshSmooth.js';
import { triangleCount, vertexCount } from '../src/geometry/mesh.js';
import {
  enclosedVolume,
  meanSurfaceDistance,
  symmetricHausdorffDistance,
} from './diff/mesh.js';
import {
  hasConsistentOutwardOrientation,
  isManifold,
  isWatertight,
} from './diff/structure.js';
import { readMeshJson } from './fixtures/loaders.js';
import { fixtureUrl } from './fixtures/root.js';

// This spec runs in both tiers. `npm test` reads the committed corpus, so the
// fast tier needs neither Python nor WASM. `npm run test:oracle` points
// SEGMORPH_FIXTURES_DIR at test/generated, where the global setup has just
// regenerated these goldens with the pinned oracles, so a VTK or PolySeg bump
// can never leave a stale committed golden silently passing.
const fixtureDir = fixtureUrl('A/sphere/');
const goldenUrl = new URL('golden.mesh.json', fixtureDir);
const inputUrl = new URL('input.nrrd', fixtureDir);
const extractionUrl = new URL('golden.extract.mesh.json', fixtureDir);
const polysegUrl = new URL('golden.polyseg.mesh.json', fixtureDir);

// Slicer's binary-labelmap-to-closed-surface rule derives the windowed-sinc
// parameters from smoothingFactor (passBand = 10^(-4 * f), iterations =
// 20 + 40 * f); the oracle records the values it actually used in params.json
// and our composed chain consumes the same ones.
const params = JSON.parse(await readFile(new URL('params.json', fixtureDir), 'utf8'));

// Read the recorded parameters strictly. Passing an absent key through as
// undefined silently falls back to meshSmooth's own defaults, which produces a
// plausible mesh that simply is not the one the oracle smoothed -- a corpus
// missing these keys should fail as a corpus error, not as a port mismatch.
function requireNumber(key: string) {
  const value = params[key];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Fixture params.json is missing the numeric "${key}" the oracle recorded`);
  }
  return value;
}

const smoothing = {
  passBand: requireNumber('passBand'),
  numberOfIterations: requireNumber('iterations'),
};

describe('Python VTK surface oracle', () => {
  it('records the unsmoothed Discrete Flying Edges stage separately', async () => {
    const mesh = readMeshJson(await readFile(extractionUrl, 'utf8'));

    expect(mesh.points.length / 3).toBe(1896);
    expect(mesh.polys.length / 4).toBe(3788);
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
  });

  it('produces a closed, outward-oriented sphere with the expected volume', async () => {
    const mesh = readMeshJson(await readFile(goldenUrl, 'utf8'));
    const analyticVolume = 4 / 3 * Math.PI * params.radius ** 3;
    const volume = enclosedVolume(mesh);

    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
    // Measured golden enclosed-volume ratio to the analytic sphere is 1.00114; keep it tight.
    expect(volume).toBeGreaterThanOrEqual(analyticVolume * 0.998);
    expect(volume).toBeLessThanOrEqual(analyticVolume * 1.002);
  });

  it('matches our unsmoothed labelmapToSurface port exactly', async () => {
    const image = createOrientedImage(readNrrd(await readFile(inputUrl)));
    const golden = readMeshJson(await readFile(extractionUrl, 'utf8'));

    const mesh = labelmapToSurface(image, { labelValue: 1 });

    expect(vertexCount(mesh)).toBe(vertexCount(golden));
    expect(triangleCount(mesh)).toBe(triangleCount(golden));
    expect(symmetricHausdorffDistance(mesh, golden)).toBeCloseTo(0);
    expect(meanSurfaceDistance(mesh, golden)).toBeCloseTo(0);
    expect(enclosedVolume(mesh)).toBeCloseTo(enclosedVolume(golden));
  }, 20_000);

  it('bounds our smoothed port against the smoothed golden', async () => {
    const image = createOrientedImage(readNrrd(await readFile(inputUrl)));
    const golden = readMeshJson(await readFile(goldenUrl, 'utf8'));
    const analyticVolume = 4 / 3 * Math.PI * params.radius ** 3;

    const mesh = labelmapToSurface(image, { labelValue: 1 });

    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
    // Measured enclosed-volume ratio to the analytic sphere is 1.00379; keep a ~1% band.
    expect(enclosedVolume(mesh)).toBeGreaterThanOrEqual(analyticVolume * 0.995);
    expect(enclosedVolume(mesh)).toBeLessThanOrEqual(analyticVolume * 1.005);
    expect(symmetricHausdorffDistance(mesh, golden)).toBeLessThanOrEqual(0.41);
    expect(meanSurfaceDistance(mesh, golden)).toBeLessThanOrEqual(0.15);
  }, 20_000);
});

describe('PolySeg surface oracle', () => {
  it('produces a closed, outward-oriented sphere', async () => {
    const mesh = readMeshJson(await readFile(polysegUrl, 'utf8'));

    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
  });

  // Oracle-vs-oracle agreement: the per-filter Python vtk chain and the composed
  // PolySeg WASM (Slicer's real conversion) must agree before either is trusted
  // as ground truth. Measured spread on this case: Hausdorff 0.0565, mean 0.0226,
  // volume ratio 0.9938 (spacing 1). Bounds are ~2x the spread.
  it('agrees with the Python VTK golden within the cross-implementation spread', async () => {
    const [python, polyseg] = await Promise.all([
      readFile(goldenUrl, 'utf8').then(readMeshJson),
      readFile(polysegUrl, 'utf8').then(readMeshJson),
    ]);

    expect(symmetricHausdorffDistance(python, polyseg)).toBeLessThan(0.12);
    expect(meanSurfaceDistance(python, polyseg)).toBeLessThan(0.05);
    const volumeRatio = enclosedVolume(python) / enclosedVolume(polyseg);
    expect(volumeRatio).toBeGreaterThan(0.98);
    expect(volumeRatio).toBeLessThan(1.02);
  }, 20_000);

  // Differential test: our labelmapToSurface + windowed-sinc smoothing port must
  // reproduce Slicer's real composed output (the whole point of the port), not
  // just the Python golden it was written against.
  it('matches our labelmapToSurface + meshSmooth port', async () => {
    const input = readNrrd(await readFile(inputUrl));
    const ours = meshSmooth(labelmapToSurface(input, { labelValue: 1 }), smoothing);
    const polyseg = readMeshJson(await readFile(polysegUrl, 'utf8'));

    expect(symmetricHausdorffDistance(ours, polyseg)).toBeLessThan(0.12);
    expect(meanSurfaceDistance(ours, polyseg)).toBeLessThan(0.05);
    const volumeRatio = enclosedVolume(ours) / enclosedVolume(polyseg);
    expect(volumeRatio).toBeGreaterThan(0.98);
    expect(volumeRatio).toBeLessThan(1.02);
  }, 20_000);
});
