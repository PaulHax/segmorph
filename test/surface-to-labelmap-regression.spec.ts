import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { surfaceToLabelmap } from '../src/convert/surfaceToLabelmap.js';
import { readNrrd } from '../src/io/nrrd.js';
import { readMeshJson } from './fixtures/loaders.js';
import { regressionCases } from './surfaceToLabelmapCases.js';

// Exact-output snapshots captured from the pre-optimization implementation at
// commit 4f62ccb. Any voxel-level difference changes the sha256 and fails here:
// optimizations must be output-preserving, byte for byte.
const expected = {
  'anisotropic-sphere': {
    hash: 'b390fa9e362d7f21031be307e7deea4feedc819dc70a12b032bb79659ae22874',
    foreground: 329,
    dtype: 'Uint8Array',
  },
  'oblique-sphere': {
    hash: 'b6b8710c6dbdbde92ec3dbdcae5ef24dfbcc259ee8e9806d7c287153ebb3b564',
    foreground: 1452,
    dtype: 'Uint16Array',
  },
  'boundary-touching-cube': {
    hash: '6caf38d537984e261527b8caef5f990fb91415a1db917198821a79ed28997973',
    foreground: 512,
    dtype: 'Uint8Array',
  },
  'thin-shell': {
    hash: '7607f2fd9ef310bff665c6abca76f0a57a2fce2ef5587d378639cb211936583f',
    foreground: 2946,
    dtype: 'Uint32Array',
  },
  'empty-mesh': {
    hash: '076a27c79e5ace2a3d47f9dd2e83e4ff6ea8872b3c2218f66c92b89b55f36560',
    foreground: 0,
    dtype: 'Uint8Array',
  },
} as const;

const sha256 = (data: ArrayBufferView) => createHash('sha256')
  .update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
  .digest('hex');

const summarize = (data: ArrayBufferView & ArrayLike<number>) => ({
  hash: sha256(data),
  foreground: Array.from(data).filter(Boolean).length,
  dtype: data.constructor.name,
});

describe('surfaceToLabelmap exact-output regression', () => {
  for (const { name, mesh, geometry, labelValue } of regressionCases) {
    it(`reproduces the base implementation byte for byte: ${name}`, () => {
      const result = surfaceToLabelmap(mesh, geometry, { labelValue });
      expect(summarize(result.data)).toEqual(expected[name as keyof typeof expected]);
    });
  }

  it('reproduces the base implementation byte for byte: D sphere fixture', async () => {
    const [meshJson, golden] = await Promise.all([
      readFile(new URL('./fixtures/D/sphere/input.mesh.json', import.meta.url), 'utf8'),
      readNrrd(await readFile(new URL('./fixtures/D/sphere/golden.nrrd', import.meta.url))),
    ]);
    const result = surfaceToLabelmap(readMeshJson(meshJson), golden, { labelValue: 1 });
    expect(summarize(result.data)).toEqual({
      hash: '70d9302475e0bab6eba9ab712d77212f5edac3eb2b0395756d531284f446e80e',
      foreground: 4240,
      dtype: 'Uint8Array',
    });
  }, 30_000);
});
