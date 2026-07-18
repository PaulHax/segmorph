import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { contourToLabelmap } from '../src/convert/contourToLabelmap.js';
import { readNrrd } from '../src/io/nrrd.js';
import { dice, mismatchCount, mismatchingVoxelCoordinates } from './diff/image.js';
import { readContoursJson } from './fixtures/G/loaders.js';
import { findFixtureEntries, readFixtureManifest } from './fixtures/loaders.js';

const fixture = (caseName: string, file: string) => (
  new URL(`./fixtures/G/${caseName}/${file}`, import.meta.url)
);

const loadCase = async (caseName: string) => {
  const [input, golden] = await Promise.all([
    readFile(fixture(caseName, 'input.contours.json'), 'utf8'),
    readNrrd(await readFile(fixture(caseName, 'golden.nrrd'))),
  ]);
  return { ...readContoursJson(input), golden };
};

/**
 * The oracle is vtkLassoStencilSource (polygon) + vtkImageStencilToImage,
 * VTK 9.6.2. Its raster convention is identical to the TS port's half-open
 * (min-exclusive, max-inclusive) span rule except that VTK dilates every span
 * by VTK_STENCIL_TOL = 7.62939453125e-06 index units. The generator verified
 * that every case below keeps all scanline crossings and edge endpoint rows
 * at least 1e-3 index units away from integer voxel centers (measured minimum
 * clearance across cases: 0.008696, disjoint-loops), so the tolerance cannot
 * flip any voxel and exact voxel equality is required.
 */
const exactCases = [
  'convex-pentagon',
  'concave-l',
  'nested-hole',
  'disjoint-loops',
  'multi-slice',
  'anisotropic',
  'subvoxel',
  'oblique-pentagon',
];

describe('contour rasterization oracle', () => {
  it('records the generating VTK oracle for every case in the manifest', async () => {
    const manifest = readFixtureManifest(
      await readFile(new URL('./fixtures/manifest.json', import.meta.url), 'utf8'),
    );
    for (const caseName of [...exactCases, 'gridline']) {
      expect(findFixtureEntries(manifest, 'G', caseName).map((entry) => entry.oracle.name))
        .toEqual(['python-vtk']);
    }
  });

  it.each(exactCases)('matches the VTK rasterization exactly: %s', async (caseName) => {
    const { labelValue, contours, golden } = await loadCase(caseName);
    const actual = contourToLabelmap(contours, golden, { labelValue });
    const dims = golden.dims as [number, number, number];

    expect(mismatchCount(actual.data, golden.data, dims)).toBe(0);
    expect(dice(actual.data, golden.data, dims)).toBe(1);
  });

  it('echoes the oblique fixture geometry exactly', async () => {
    const { labelValue, contours, golden } = await loadCase('oblique-pentagon');
    const actual = contourToLabelmap(contours, golden, { labelValue });

    expect(actual.dims).toEqual(golden.dims);
    expect(actual.spacing).toEqual(golden.spacing);
    expect(actual.origin).toEqual(golden.origin);
    expect(actual.direction).toEqual(golden.direction);
  });

  it('differs from VTK only by its documented convention on gridline-exact edges', async () => {
    const { labelValue, contours, golden } = await loadCase('gridline');
    const actual = contourToLabelmap(contours, golden, { labelValue });
    const dims = golden.dims as [number, number, number];

    // The rectangle sits exactly on voxel-center gridlines x in [3, 8],
    // y in [2, 6] on slice 1. VTK's VTK_STENCIL_TOL dilation turns its
    // half-open spans into closed spans there, adding the min-x column and
    // min-y row (30 voxels total); the TS port's strict half-open rule fills
    // the 5x4 interior block (20 voxels). Measured disagreement: exactly
    // these 10 boundary voxels, all on the shared min-x/min-y boundary.
    expect(mismatchingVoxelCoordinates(actual.data, golden.data, dims)).toEqual([
      [3, 2, 1], [4, 2, 1], [5, 2, 1], [6, 2, 1], [7, 2, 1], [8, 2, 1],
      [3, 3, 1], [3, 4, 1], [3, 5, 1], [3, 6, 1],
    ]);

    // The port never fills a voxel the oracle leaves empty: interior equality.
    for (let index = 0; index < actual.data.length; index += 1) {
      if (actual.data[index] !== 0) expect(golden.data[index]).not.toBe(0);
    }
  });
});
