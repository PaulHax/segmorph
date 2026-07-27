import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { enclosedVolume } from './diff/mesh.js';
import { hasConsistentOutwardOrientation, isManifold, isWatertight } from './diff/structure.js';
import { findFixtureEntries, readFixtureManifest, readMeshJson } from './fixtures/loaders.js';
import { fixtureUrl } from './fixtures/root.js';

const cases = [
  'sphere',
  'boundary-blob',
  'anisotropic',
  'oblique',
  'multilabel-label1',
  'multilabel-label2',
] as const;

// Cases whose surface encloses exactly one label against background; their
// goldens must be consistently outward oriented. multilabel-label2 shares an
// interface with label 1, and vtkSurfaceNets3D orients shared quads with the
// smaller label outward, so its orientation is intentionally mixed.
const outwardCases = cases.filter((name) => name !== 'multilabel-label2');

async function loadGolden(name: string, file: string) {
  const url = fixtureUrl(`J/${name}/${file}`);
  return readMeshJson(await readFile(url, 'utf8'));
}

describe('vtkSurfaceNets3D oracle fixtures', () => {
  it('registers every J case in the fixture manifest', async () => {
    const url = fixtureUrl('manifest.json');
    const manifest = readFixtureManifest(await readFile(url, 'utf8'));
    for (const name of cases) {
      const entries = findFixtureEntries(manifest, 'J', name);
      expect(entries, name).toHaveLength(1);
      expect(entries[0].oracle.name).toBe('python-vtk');
    }
  });

  for (const name of outwardCases) {
    it(`produces a closed outward-oriented golden for ${name}`, async () => {
      for (const file of ['golden.mesh.json', 'golden.unsmoothed.mesh.json']) {
        const mesh = await loadGolden(name, file);
        expect(isWatertight(mesh), `${file} watertight`).toBe(true);
        expect(isManifold(mesh), `${file} manifold`).toBe(true);
        expect(hasConsistentOutwardOrientation(mesh), `${file} outward`).toBe(true);
      }
    });
  }

  it('closes the boundary-touching blob at its exact voxel extent', async () => {
    // 16x16x16 box occupying x 0..15, y 0..7, z 0..5 => unsmoothed hull at
    // half-voxel offsets around the foreground.
    const mesh = await loadGolden('boundary-blob', 'golden.unsmoothed.mesh.json');
    expect(enclosedVolume(mesh)).toBeCloseTo(16 * 8 * 6, 5);
  });

  it('keeps the multilabel-label2 golden watertight and manifold', async () => {
    const mesh = await loadGolden('multilabel-label2', 'golden.mesh.json');
    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
  });

  it('smooths the sphere within its constraint distance of the net', async () => {
    const smoothed = await loadGolden('sphere', 'golden.mesh.json');
    const unsmoothed = await loadGolden('sphere', 'golden.unsmoothed.mesh.json');
    expect(smoothed.points.length).toBe(unsmoothed.points.length);
    // Constraint sphere radius is norm(spacing) = sqrt(3); every smoothed
    // point stays within that distance of its unsmoothed original.
    const constraint = Math.sqrt(3) + 1e-6;
    for (let index = 0; index < smoothed.points.length; index += 3) {
      const dx = smoothed.points[index] - unsmoothed.points[index];
      const dy = smoothed.points[index + 1] - unsmoothed.points[index + 1];
      const dz = smoothed.points[index + 2] - unsmoothed.points[index + 2];
      expect(Math.hypot(dx, dy, dz)).toBeLessThanOrEqual(constraint);
    }
  });
});
