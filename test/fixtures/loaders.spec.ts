import { readFile } from 'node:fs/promises';

import { describe, expect, it } from 'vitest';

import { readNrrd } from '../../src/io/nrrd.js';
import {
  findFixtureEntries,
  fixtureManifestEntryId,
  readFixtureManifest,
  readMeshJson,
} from './loaders.js';

const fixture = (name: string) => new URL(`./loader-case/tiny/${name}`, import.meta.url);

function rawNrrd(header: string, payload: number[]) {
  const headerBytes = new TextEncoder().encode(`${header}\n\n`);
  const bytes = new Uint8Array(headerBytes.length + payload.length);
  bytes.set(headerBytes);
  bytes.set(payload, headerBytes.length);
  return bytes;
}

describe('fixture loaders', () => {
  it('reads raw NRRD data and oriented geometry', async () => {
    const parsed = readNrrd(await readFile(fixture('input.nrrd')));

    expect(parsed.dims).toEqual([2, 2, 2]);
    expect(parsed.spacing).toEqual([2, 3, 4]);
    expect(parsed.origin).toEqual([10, 20, 30]);
    expect(parsed.direction).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, -1]]);
    expect(parsed.data).toBeInstanceOf(Uint8Array);
    expect([...parsed.data]).toEqual([49, 50, 51, 52, 53, 54, 55, 10]);
  });

  it('reads flat mesh JSON into the mesh metric representation', async () => {
    const parsed = readMeshJson(await readFile(fixture('golden.mesh.json'), 'utf8'));

    expect(parsed.points).toEqual(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
    expect(parsed.polys).toEqual(new Uint32Array([3, 0, 1, 2]));
  });

  it('reads standard scalar aliases with the declared byte order', () => {
    const parsed = readNrrd(rawNrrd([
      'NRRD0005',
      'type: unsigned short',
      'dimension: 1',
      'sizes: 1',
      'encoding: raw',
      'endian: big',
      'space origin: (0)',
      'space directions: (1)',
    ].join('\n'), [0x12, 0x34]));

    expect(parsed.data).toEqual(new Uint16Array([0x1234]));
  });

  it('stores NRRD axis vectors as direction-matrix columns', () => {
    const parsed = readNrrd(rawNrrd([
      'NRRD0005',
      'type: unsigned char',
      'dimension: 3',
      'sizes: 1 1 1',
      'encoding: raw',
      'space origin: (0,0,0)',
      'space directions: (0,2,0) (-3,0,0) (0,0,4)',
    ].join('\n'), [1]));

    expect(parsed.spacing).toEqual([2, 3, 4]);
    expect(parsed.direction).toEqual([
      [0, -1, 0],
      [1, 0, 0],
      [0, 0, 1],
    ]);
  });

  it('validates and queries the fixture manifest', async () => {
    const manifest = readFixtureManifest(
      await readFile(new URL('./manifest.json', import.meta.url), 'utf8'),
    );
    const entries = findFixtureEntries(manifest, 'A', 'sphere');

    expect(entries).toHaveLength(2);
    expect(entries.map(fixtureManifestEntryId)).toEqual([
      'A/sphere/python-vtk',
      'A/sphere/icr-polyseg-wasm',
    ]);
  });

  it.each([
    ['unsupported schema', { schemaVersion: 2, fixtures: [] }],
    ['missing oracle version', {
      schemaVersion: 1,
      fixtures: [{ oracle: { name: 'vtk' }, algorithm: 'A', case: 'tiny', params: {}, seed: 0 }],
    }],
    ['non-object params', {
      schemaVersion: 1,
      fixtures: [{
        oracle: { name: 'vtk', version: '1' },
        algorithm: 'A',
        case: 'tiny',
        params: [],
        seed: 0,
      }],
    }],
    ['duplicate fixture identity', {
      schemaVersion: 1,
      fixtures: [
        { oracle: { name: 'vtk', version: '1' }, algorithm: 'A', case: 'tiny', params: {}, seed: 0 },
        { oracle: { name: 'vtk', version: '2' }, algorithm: 'A', case: 'tiny', params: {}, seed: 1 },
      ],
    }],
  ])('rejects %s', (_name, value) => {
    expect(() => readFixtureManifest(JSON.stringify(value))).toThrow(/fixture manifest/i);
  });
});
