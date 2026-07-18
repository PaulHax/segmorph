import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import {
  fractionalLabelmapToSurface,
  surfaceToFractionalLabelmap,
} from '../src/convert/fractional.js';
import { createOrientedImage } from '../src/image/orientedImage.js';
import { readNrrd } from '../src/io/nrrd.js';
import { enclosedVolume } from './diff/mesh.js';
import { isManifold, isWatertight } from './diff/structure.js';
import { readMeshJson } from './fixtures/loaders.js';

const cases = ['isotropic', 'anisotropic', 'oblique'] as const;

const fixture = (caseName: string, name: string) => (
  new URL(`./fixtures/I/${caseName}/${name}`, import.meta.url)
);

const identityImage = (dims: readonly [number, number, number], data: Float32Array) => (
  createOrientedImage({
    dims,
    spacing: [1, 1, 1],
    origin: [0, 0, 0],
    direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    data,
  })
);

describe('surfaceToFractionalLabelmap', () => {
  it.each(cases)('conserves enclosed volume as summed occupancy on %s', async (caseName) => {
    const [meshJson, geometry] = await Promise.all([
      readFile(fixture(caseName, 'input.mesh.json'), 'utf8'),
      readNrrd(await readFile(fixture(caseName, 'golden.nrrd'))),
    ]);
    const mesh = readMeshJson(meshJson);
    const fractional = surfaceToFractionalLabelmap(mesh, geometry, {});

    let occupancySum = 0;
    for (const value of fractional.data) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
      occupancySum += value;
    }
    const voxelVolume = geometry.spacing[0] * geometry.spacing[1] * geometry.spacing[2];
    // Measured ratios summed occupancy volume / mesh volume:
    // 1.000090 / 0.999726 / 0.999750 across the three grids.
    expect(occupancySum * voxelVolume / enclosedVolume(mesh)).toBeCloseTo(1, 2);
  });

  it('preserves the reference geometry on the output image', async () => {
    const [meshJson, geometry] = await Promise.all([
      readFile(fixture('oblique', 'input.mesh.json'), 'utf8'),
      readNrrd(await readFile(fixture('oblique', 'golden.nrrd'))),
    ]);
    const fractional = surfaceToFractionalLabelmap(readMeshJson(meshJson), geometry, {});

    expect(fractional.dims).toEqual(geometry.dims);
    expect(fractional.spacing).toEqual(geometry.spacing);
    expect(fractional.origin).toEqual(geometry.origin);
    expect(fractional.direction).toEqual(geometry.direction);
    expect(fractional.data).toBeInstanceOf(Float32Array);
  });

  it('produces all-zero occupancy for a surface outside the grid', async () => {
    const meshJson = await readFile(fixture('isotropic', 'input.mesh.json'), 'utf8');
    const mesh = readMeshJson(meshJson);
    const farAway = {
      dims: [4, 4, 4],
      spacing: [1, 1, 1],
      origin: [500, 500, 500],
      direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    } as const;
    const fractional = surfaceToFractionalLabelmap(mesh, farAway, {});
    expect(fractional.data.every((value) => value === 0)).toBe(true);
  });

  it('rejects a non-positive or fractional numberOfOffsets', async () => {
    const meshJson = await readFile(fixture('isotropic', 'input.mesh.json'), 'utf8');
    const mesh = readMeshJson(meshJson);
    const geometry = {
      dims: [2, 2, 2],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    } as const;
    expect(() => surfaceToFractionalLabelmap(mesh, geometry, { numberOfOffsets: 0 }))
      .toThrow('numberOfOffsets');
    expect(() => surfaceToFractionalLabelmap(mesh, geometry, { numberOfOffsets: 2.5 }))
      .toThrow('numberOfOffsets');
  });
});

describe('fractionalLabelmapToSurface', () => {
  it.each(cases)('extracts a watertight manifold surface on %s', async (caseName) => {
    const geometry = readNrrd(await readFile(fixture(caseName, 'golden.nrrd')));
    const surface = fractionalLabelmapToSurface(geometry, {});

    expect(isWatertight(surface)).toBe(true);
    expect(isManifold(surface)).toBe(true);
    expect(enclosedVolume(surface)).toBeGreaterThan(0);
  });

  it('shrinks the surface as the threshold rises', async () => {
    const image = readNrrd(await readFile(fixture('isotropic', 'golden.nrrd')));
    const volumes = [0.25, 0.5, 0.75].map((threshold) => (
      enclosedVolume(fractionalLabelmapToSurface(image, { threshold }))
    ));
    expect(volumes[0]).toBeGreaterThan(volumes[1]);
    expect(volumes[1]).toBeGreaterThan(volumes[2]);
  });

  it('returns an empty mesh when nothing crosses the threshold', () => {
    const empty = identityImage([3, 3, 3], new Float32Array(27));
    const surface = fractionalLabelmapToSurface(empty, {});
    expect(surface.points.length).toBe(0);
    expect(surface.polys.length).toBe(0);
  });

  it('closes a fully occupied volume at the image boundary', () => {
    const full = identityImage([3, 3, 3], new Float32Array(27).fill(1));
    const surface = fractionalLabelmapToSurface(full, {});
    expect(isWatertight(surface)).toBe(true);
    expect(enclosedVolume(surface)).toBeGreaterThan(0);
  });

  it('rejects thresholds outside (0, 1]', () => {
    const image = identityImage([2, 2, 2], new Float32Array(8));
    expect(() => fractionalLabelmapToSurface(image, { threshold: -0.1 })).toThrow('threshold');
    expect(() => fractionalLabelmapToSurface(image, { threshold: 1.1 })).toThrow('threshold');
    expect(() => fractionalLabelmapToSurface(image, { threshold: Number.NaN })).toThrow('threshold');
  });

  it('rejects a zero threshold rather than returning an empty mesh', async () => {
    // At threshold 0 the >= test classifies the zero-valued padding border as
    // inside, so every cell is fully occupied and no surface is extracted: the
    // occupancy encoding has no iso-crossing at 0, so 0 is not a valid surface.
    const image = readNrrd(await readFile(fixture('isotropic', 'golden.nrrd')));
    expect(() => fractionalLabelmapToSurface(image, { threshold: 0 })).toThrow('threshold');
  });
});

describe('fractional round trip', () => {
  it.each(cases)('reproduces the input mesh volume on %s', async (caseName) => {
    const [meshJson, geometry] = await Promise.all([
      readFile(fixture(caseName, 'input.mesh.json'), 'utf8'),
      readNrrd(await readFile(fixture(caseName, 'golden.nrrd'))),
    ]);
    const mesh = readMeshJson(meshJson);
    const fractional = surfaceToFractionalLabelmap(mesh, geometry, {});
    const roundTrip = fractionalLabelmapToSurface(fractional, {});

    // Measured round-trip volume ratios: 0.98377 / 0.96515 / 0.95993.
    // Marching cubes at the 0.5 iso chops convex curvature, and the loss
    // grows with voxel size (the oblique grid resolves the sphere with only
    // ~9 voxels across). A 5% floor bounds that discretization while still
    // catching an offset or winding bug.
    const ratio = enclosedVolume(roundTrip) / enclosedVolume(mesh);
    expect(ratio).toBeGreaterThan(0.95);
    expect(ratio).toBeLessThan(1.01);
  });
});
