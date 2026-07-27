import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { surfaceToLabelmap } from '../src/convert/surfaceToLabelmap.js';
import { readNrrd } from '../src/io/nrrd.js';
import { dice, mismatchCount } from './diff/image.js';
import { findFixtureEntries, readFixtureManifest, readMeshJson } from './fixtures/loaders.js';
import { fixtureUrl } from './fixtures/root.js';

const fixture = (algorithm: string, name: string) => fixtureUrl(`${algorithm}/sphere/${name}`);

describe('surface voxelization oracle', () => {
  it('regenerates the smoothed sphere close to its source labelmap', async () => {
    const [source, voxelized] = await Promise.all([
      readNrrd(await readFile(fixture('A', 'input.nrrd'))),
      readNrrd(await readFile(fixture('D', 'golden.nrrd'))),
    ]);

    // VTK's smoothed surface changes 26 boundary voxels: Dice 0.9969281663516069.
    const dims = source.dims as [number, number, number];
    expect(dice(source.data, voxelized.data, dims)).toBeGreaterThan(0.996);
    expect(mismatchCount(source.data, voxelized.data, dims)).toBe(26);
    expect(voxelized.dims).toEqual(source.dims);
  });

  it('records the generating VTK oracle in the fixture manifest', async () => {
    const manifest = readFixtureManifest(await readFile(fixtureUrl('manifest.json'), 'utf8'));

    expect(
      findFixtureEntries(manifest, 'D', 'sphere')
        .map((entry) => entry.oracle.name)
        .sort(),
    ).toEqual(['icr-polyseg-wasm', 'python-vtk']);
  });

  it('matches the VTK voxelization fixture', async () => {
    const [meshJson, golden] = await Promise.all([
      readFile(fixture('D', 'input.mesh.json'), 'utf8'),
      readNrrd(await readFile(fixture('D', 'golden.nrrd'))),
    ]);
    const actual = surfaceToLabelmap(readMeshJson(meshJson), golden, { labelValue: 1 });
    const dims = golden.dims as [number, number, number];

    expect(dice(actual.data, golden.data, dims)).toBe(1);
    expect(mismatchCount(actual.data, golden.data, dims)).toBe(0);
  }, 15_000);
});

describe('PolySeg voxelization oracle', () => {
  // PolySeg rasterizes onto its own super-resolution bounding-box grid, so the
  // fixture stores its occupancy resampled (nearest neighbor) onto the reference
  // geometry. This is the composed second oracle for D: Slicer's real
  // surface-to-labelmap, cross-checking the Python vtkPolyDataToImageStencil
  // golden and our port. Measured agreement on this case: Dice 0.99325, 57
  // boundary-voxel mismatches of 32768.
  const readPolysegLabelmap = async () => {
    const parsed = JSON.parse(await readFile(fixture('D', 'golden.polyseg.labelmap.json'), 'utf8'));
    return { data: Uint8Array.from(parsed.data), dims: parsed.dims as [number, number, number] };
  };

  it('records the composed PolySeg oracle in the fixture manifest', async () => {
    const manifest = readFixtureManifest(await readFile(fixtureUrl('manifest.json'), 'utf8'));

    expect(
      findFixtureEntries(manifest, 'D', 'sphere')
        .map((entry) => entry.oracle.name)
        .sort(),
    ).toEqual(['icr-polyseg-wasm', 'python-vtk']);
  });

  it('agrees with the Python VTK golden within the cross-implementation spread', async () => {
    const [polyseg, golden] = await Promise.all([
      readPolysegLabelmap(),
      readNrrd(await readFile(fixture('D', 'golden.nrrd'))),
    ]);

    expect(polyseg.dims).toEqual(golden.dims);
    expect(dice(polyseg.data, golden.data, polyseg.dims)).toBeGreaterThan(0.99);
    expect(mismatchCount(polyseg.data, golden.data, polyseg.dims)).toBeLessThan(120);
  });

  it('matches our surfaceToLabelmap port', async () => {
    const [meshJson, polyseg, golden] = await Promise.all([
      readFile(fixture('D', 'input.mesh.json'), 'utf8'),
      readPolysegLabelmap(),
      readNrrd(await readFile(fixture('D', 'golden.nrrd'))),
    ]);
    const ours = surfaceToLabelmap(readMeshJson(meshJson), golden, { labelValue: 1 });

    expect(dice(ours.data, polyseg.data, polyseg.dims)).toBeGreaterThan(0.99);
    expect(mismatchCount(ours.data, polyseg.data, polyseg.dims)).toBeLessThan(120);
  }, 15_000);
});

// The sphere case above uses unit spacing, identity direction, and origin 0, so
// it never exercises the oriented index<->world transform in surfaceToLabelmap.
// These grids voxelize the same sphere on oblique and anisotropic geometry with
// an external VTK golden (vtkPolyDataToImageStencil), so a transform or bounds
// bug in voxelization can no longer hide behind the trivial identity case.
describe('oriented surface voxelization oracle', () => {
  const cases = ['oblique', 'anisotropic', 'oblique-anisotropic'] as const;
  const orientedFixture = (name: string, file: string) => fixtureUrl(`D/${name}/${file}`);

  it('records each oriented VTK case in the fixture manifest', async () => {
    const manifest = readFixtureManifest(await readFile(fixtureUrl('manifest.json'), 'utf8'));

    for (const name of cases) {
      expect(findFixtureEntries(manifest, 'D', name).map((entry) => entry.oracle.name)).toEqual([
        'python-vtk',
      ]);
    }
  });

  it.each(cases)(
    'matches the VTK voxelization on the %s grid',
    async (name) => {
      const [meshJson, golden] = await Promise.all([
        readFile(orientedFixture(name, 'input.mesh.json'), 'utf8'),
        readNrrd(await readFile(orientedFixture(name, 'golden.nrrd'))),
      ]);
      const actual = surfaceToLabelmap(readMeshJson(meshJson), golden, { labelValue: 1 });
      const dims = golden.dims as [number, number, number];

      // Our world-space rasterizer reproduces vtkPolyDataToImageStencil voxel for
      // voxel on every oriented grid; any residual mismatch would be a transform
      // or bounds bug, so the tolerance is exact.
      expect(dice(actual.data, golden.data, dims)).toBe(1);
      expect(mismatchCount(actual.data, golden.data, dims)).toBe(0);
    },
    15_000,
  );
});
