import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { indexToWorld } from '../src/image/orientedImage.js';
import { resampleNearest } from '../src/image/resample.js';
import { readNrrd } from '../src/io/nrrd.js';
import { mismatchCount, mismatchingVoxelCoordinates } from './diff/image.js';
import { findFixtureEntries, readFixtureManifest } from './fixtures/loaders.js';

const cases = [
  'anisotropic-translated',
  'oblique',
  'oblique-anisotropic',
  'axis-permutation',
  'upsample-half-spacing',
  'downsample-double-spacing',
  'outside-fill-uint8',
  'outside-fill-uint16',
  'mirrored',
  'half-voxel-ties-interior',
  'multi-label',
] as const;

const borderTiesCase = 'half-voxel-border-ties';

const fixture = (caseName: string, name: string) => (
  new URL(`./fixtures/E/${caseName}/${name}`, import.meta.url)
);

const loadCase = async (caseName: string) => {
  const [input, golden, params] = await Promise.all([
    readNrrd(await readFile(fixture(caseName, 'input.nrrd'))),
    readNrrd(await readFile(fixture(caseName, 'golden.nrrd'))),
    readFile(fixture(caseName, 'params.json'), 'utf8').then(JSON.parse),
  ]);
  return { input, golden, params };
};

const loadItkGolden = async (caseName: string) => (
  readNrrd(await readFile(fixture(caseName, 'golden.itk.nrrd')))
);

describe('oriented nearest-neighbor resample oracle', () => {
  it.each(cases)('matches Python VTK for %s geometry', async (caseName) => {
    const { input, golden, params } = await loadCase(caseName);

    const actual = resampleNearest(input, golden, { fillValue: params.fillValue });
    const dims = golden.dims as [number, number, number];

    expect(actual.data.constructor).toBe(golden.data.constructor);
    expect(actual.data).toEqual(golden.data);
    expect(mismatchCount(actual.data, golden.data, dims)).toBe(0);
    expect(actual.dims).toEqual(golden.dims);
    expect(actual.spacing).toEqual(golden.spacing);
    expect(actual.origin).toEqual(golden.origin);
    expect(actual.direction).toEqual(golden.direction);
  });

  it.each(['outside-fill-uint8', 'outside-fill-uint16'] as const)(
    '%s exercises the nonzero fill value outside the input grid',
    async (caseName) => {
      const { golden, params } = await loadCase(caseName);
      const filled = golden.data.filter((value) => value === params.fillValue).length;
      expect(filled).toBeGreaterThan(0);
      expect(filled).toBeLessThan(golden.data.length);
    },
  );

  it('keeps all eight labels of the multi-label fixture', async () => {
    const { golden } = await loadCase('multi-label');
    expect([...new Set(golden.data)].sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it('uses a negative-determinant direction in the mirrored fixture', async () => {
    const { golden } = await loadCase('mirrored');
    const [r0, r1, r2] = golden.direction;
    const determinant = r0[0] * (r1[1] * r2[2] - r1[2] * r2[1])
      - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0])
      + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]);
    expect(determinant).toBeCloseTo(-1, 10);
  });

  it('records the Python VTK and ITK oracles for every case in the manifest', async () => {
    const manifest = readFixtureManifest(
      await readFile(new URL('./fixtures/manifest.json', import.meta.url), 'utf8'),
    );
    const allCases = [...cases, borderTiesCase];

    expect(allCases.map((caseName) => (
      findFixtureEntries(manifest, 'E', caseName).map((entry) => entry.oracle.name).sort()
    ))).toEqual(allCases.map(() => ['itk', 'python-vtk']));
  });
});

describe('independent ITK resample oracle', () => {
  // ITK's ResampleImageFilter is a separate lineage from VTK, so agreement with
  // the vtkImageReslice golden confirms the golden our port is checked against is
  // not a shared-VTK artifact or a mis-read reslice parameter. On every geometry
  // except the upper-edge half-voxel ties (handled below) the two agree voxel for
  // voxel, and our port reproduces that independent result exactly.
  it.each(cases)('agrees with the Python VTK golden and our port for %s', async (caseName) => {
    const { input, golden, params } = await loadCase(caseName);
    const itk = await loadItkGolden(caseName);
    const dims = golden.dims as [number, number, number];

    expect(itk.data.constructor).toBe(golden.data.constructor);
    expect(mismatchCount(itk.data, golden.data, dims)).toBe(0);

    const actual = resampleNearest(input, itk, { fillValue: params.fillValue });
    expect(mismatchCount(actual.data, itk.data, dims)).toBe(0);
  });
});

describe('half-voxel upper-boundary ties', () => {
  // Every output voxel center of this fixture sits exactly half an input voxel past an input
  // voxel center, so each input index coordinate is an exact .5 tie. Interior ties round half
  // up (index -0.5 also rounds to 0). At the upper edge (index === dims - 0.5) vtkImageReslice
  // with its default Border=on accepts the sample and clamps the rounded index to the edge
  // voxel; resampleNearest now matches this via its default border: 'clamp' policy.

  it('clamps upper-edge ties to the edge voxel like VTK, no fill values', async () => {
    const { input, golden, params } = await loadCase(borderTiesCase);
    const dims = golden.dims as [number, number, number];
    const actual = resampleNearest(input, golden, { fillValue: params.fillValue });

    // The VTK golden equals round-half-up then clamp-to-extent of the input at every voxel.
    const clamped = new Uint8Array(golden.data.length);
    for (let z = 0; z < dims[2]; z += 1) {
      for (let y = 0; y < dims[1]; y += 1) {
        for (let x = 0; x < dims[0]; x += 1) {
          const sx = Math.min(x + 1, input.dims[0] - 1);
          const sy = Math.min(y + 1, input.dims[1] - 1);
          const sz = Math.min(z + 1, input.dims[2] - 1);
          clamped[x + dims[0] * (y + dims[1] * z)] = input.data[
            sx + input.dims[0] * (sy + input.dims[1] * sz)
          ];
        }
      }
    }
    expect(golden.data).toEqual(clamped);

    // With border-clamp parity, resampleNearest reproduces the clamped golden exactly: the
    // upper tie planes carry the clamped edge voxel, never fillValue.
    expect(mismatchingVoxelCoordinates(actual.data, golden.data, dims)).toEqual([]);
    const tiePlanes: [number, number, number][] = [];
    for (let z = 0; z < dims[2]; z += 1) {
      for (let y = 0; y < dims[1]; y += 1) {
        for (let x = 0; x < dims[0]; x += 1) {
          if (x === input.dims[0] - 1 || y === input.dims[1] - 1 || z === input.dims[2] - 1) {
            tiePlanes.push([x, y, z]);
          }
        }
      }
    }
    tiePlanes.forEach(([x, y, z]) => {
      expect(actual.data[x + dims[0] * (y + dims[1] * z)]).toBe(clamped[
        x + dims[0] * (y + dims[1] * z)
      ]);
      expect(actual.data[x + dims[0] * (y + dims[1] * z)]).not.toBe(params.fillValue);
    });
  });

  // Border-clamp parity: output samples whose nearest input index lands exactly on the upper
  // half-voxel boundary (index === dims - 0.5) clamp to the edge voxel to match vtkImageReslice
  // (Border=on, BorderThickness=0.5). Before the src/image/resample.ts fix this case had 18
  // mismatching voxels; after it has 0.
  it('matches Python VTK exactly on upper-edge half-voxel ties', async () => {
    const { input, golden, params } = await loadCase(borderTiesCase);
    const actual = resampleNearest(input, golden, { fillValue: params.fillValue });

    expect(mismatchCount(actual.data, golden.data, golden.dims as [number, number, number]))
      .toBe(0);
  });

  // This is the one geometry where the two oracles legitimately disagree, which
  // is exactly why a second oracle is worth having: it pins the convention rather
  // than letting it pass silently. ITK rounds each upper-edge tie (index ===
  // size - 0.5) outside the input and fills; VTK's Border=on clamps it to the
  // edge voxel. Our port follows VTK (Slicer's convention), so it diverges from
  // ITK on, and only on, the upper-edge planes.
  it('diverges from ITK only on the upper-edge tie planes', async () => {
    const { input, golden, params } = await loadCase(borderTiesCase);
    const itk = await loadItkGolden(borderTiesCase);
    const dims = golden.dims as [number, number, number];

    const divergences = mismatchingVoxelCoordinates(golden.data, itk.data, dims);
    const isUpperEdge = ([x, y, z]: readonly [number, number, number]) => (
      x === input.dims[0] - 1 || y === input.dims[1] - 1 || z === input.dims[2] - 1
    );
    expect(divergences.every(isUpperEdge)).toBe(true);
    // Every upper-edge voxel diverges: 24 total minus the 3x2x1 interior block.
    expect(divergences.length).toBe(18);

    // ITK fills those tie planes; VTK and our port carry the clamped edge voxel.
    divergences.forEach(([x, y, z]) => {
      expect(itk.data[x + dims[0] * (y + dims[1] * z)]).toBe(params.fillValue);
    });
    const actual = resampleNearest(input, golden, { fillValue: params.fillValue });
    expect(mismatchCount(actual.data, golden.data, dims)).toBe(0);
  });
});

describe('nearest-neighbor resample invariants (oracle-free)', () => {
  // Lehmer LCG; every product stays below 2^53, so the sequence is exact in doubles.
  const seededLabels = (length: number) => {
    let state = 20260710;
    const data = new Uint8Array(length);
    for (let index = 0; index < length; index += 1) {
      state = (state * 48271) % 2147483647;
      data[index] = state % 7;
    }
    return data;
  };

  const voxelCount = (dims: readonly number[]) => dims[0] * dims[1] * dims[2];

  const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  // 3-4-5 rotation about z: orthonormal to within one ulp in doubles.
  const oblique = [[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1]];
  const mirrored = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]];

  const geometries = [
    ['identity axis-aligned', {
      dims: [4, 3, 2], spacing: [1, 1, 1], origin: [0, 0, 0], direction: identity,
    }],
    ['oblique anisotropic', {
      dims: [5, 4, 3], spacing: [0.7, 1.3, 2.1], origin: [1, -2, 3], direction: oblique,
    }],
    ['mirrored', {
      dims: [4, 3, 2], spacing: [0.8, 1.1, 1.7], origin: [5, 6, 7], direction: mirrored,
    }],
  ] as const;

  it.each(geometries)('resampling onto its own geometry is the identity: %s', (_, geometry) => {
    const image = { ...geometry, data: seededLabels(voxelCount(geometry.dims)) };

    expect(resampleNearest(image, geometry).data).toEqual(image.data);
  });

  it('recovers every label after resampling to a half-spacing grid and back', () => {
    const geometry = {
      dims: [5, 4, 3], spacing: [0.8, 1.2, 2.0], origin: [3, -2, 7], direction: oblique,
    };
    // Fine voxel centers sit at input index (j - 0.75) / 2, an exact 0.125 away from the
    // nearest rounding boundary, so float noise cannot flip any nearest-neighbor pick.
    const fine = {
      dims: geometry.dims.map((size) => size * 2),
      spacing: geometry.spacing.map((step) => step / 2),
      origin: indexToWorld(geometry, [-0.375, -0.375, -0.375]),
      direction: geometry.direction,
    };
    const image = { ...geometry, data: seededLabels(voxelCount(geometry.dims)) };

    const restored = resampleNearest(resampleNearest(image, fine), geometry);

    expect(restored.data).toEqual(image.data);
  });
});
