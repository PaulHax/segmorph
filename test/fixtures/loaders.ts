import type { Mesh } from '../diff/mesh.js';

export type FixtureManifestEntry = {
  oracle: { name: string; version: string };
  algorithm: string;
  case: string;
  params: Record<string, unknown>;
  seed: number;
};

export type FixtureManifest = {
  schemaVersion: 1;
  fixtures: FixtureManifestEntry[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isJsonValue(value: unknown): boolean {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(isJsonValue);
  return isRecord(value) && Object.values(value).every(isJsonValue);
}

export function fixtureManifestEntryId(entry: FixtureManifestEntry): string {
  return `${entry.algorithm}/${entry.case}/${entry.oracle.name}`;
}

export function readFixtureManifest(json: string): FixtureManifest {
  const value: unknown = JSON.parse(json);
  if (!isRecord(value) || value.schemaVersion !== 1 || !Array.isArray(value.fixtures)) {
    throw new Error('Invalid fixture manifest: expected schemaVersion 1 and a fixtures array');
  }

  const fixtures = value.fixtures.map((fixture, index): FixtureManifestEntry => {
    if (!isRecord(fixture)
      || !isRecord(fixture.oracle)
      || typeof fixture.oracle.name !== 'string' || fixture.oracle.name.length === 0
      || typeof fixture.oracle.version !== 'string' || fixture.oracle.version.length === 0
      || typeof fixture.algorithm !== 'string' || fixture.algorithm.length === 0
      || typeof fixture.case !== 'string' || fixture.case.length === 0
      || !isRecord(fixture.params) || !isJsonValue(fixture.params)
      || typeof fixture.seed !== 'number' || !Number.isSafeInteger(fixture.seed)) {
      throw new Error(`Invalid fixture manifest entry at index ${index}`);
    }
    return {
      oracle: { name: fixture.oracle.name, version: fixture.oracle.version },
      algorithm: fixture.algorithm,
      case: fixture.case,
      params: fixture.params,
      seed: fixture.seed,
    };
  });

  const ids = fixtures.map(fixtureManifestEntryId);
  const duplicate = ids.find((id, index) => ids.indexOf(id) !== index);
  if (duplicate) throw new Error(`Duplicate fixture manifest entry: ${duplicate}`);

  return { schemaVersion: 1, fixtures };
}

export function findFixtureEntries(
  manifest: FixtureManifest,
  algorithm: string,
  caseName: string,
): FixtureManifestEntry[] {
  return manifest.fixtures.filter((entry) => entry.algorithm === algorithm && entry.case === caseName);
}

export function readMeshJson(json: string): Mesh {
  const value: unknown = JSON.parse(json);
  if (!value || typeof value !== 'object' || !('points' in value) || !('polys' in value)
    || !Array.isArray(value.points) || !Array.isArray(value.polys)
    || value.points.some((point) => typeof point !== 'number' || !Number.isFinite(point))
    || value.polys.some((index) => !Number.isInteger(index) || index < 0)) {
    throw new Error('Invalid flat mesh JSON');
  }
  return {
    points: new Float32Array(value.points),
    polys: new Uint32Array(value.polys),
  };
}
