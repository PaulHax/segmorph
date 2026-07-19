// Clinical regression oracle (algorithm L): the whole differential suite
// otherwise runs on volumes someone constructed -- spheres, tori, random blobs.
// Those are the right unit of comparison for pinning a case table, but they
// share a property real data does not: they are smooth, compact, and small.
// This oracle runs the conversion Slicer actually ships (@icr/polyseg-wasm,
// vtkSegmentationCore's binary-labelmap-to-closed-surface rule) over a real
// chest CT and hands the spec a golden to regress against.
//
// Dataset: LIDC2, from the Lung Image Database Consortium, published as vtk.js
// sample data -- the same volume examples/advanced already loads. 256x256x133
// at 1.40625 x 1.40625 x 2.5 mm, downloaded once and cached.
//
// The segmentation is a fixed intensity threshold rather than anything
// clever, chosen because it produces exactly the structure synthetic cases
// never do: dense bone, which comes out as ribs, vertebrae, and scapulae --
// dozens of disconnected components, thin curved shells a voxel or two across,
// and a surface an order of magnitude larger than any other case in the suite.
//
// Dev-only; the download is never committed and the goldens live in the
// gitignored generated corpus.
//
// Run: node oracles/node/clinical.ts

import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';

import launcher from '@icr/polyseg-wasm/js';

const ALGORITHM = 'L';
const ORACLE_NAME = 'icr-polyseg-wasm';
const CASE = 'lidc2-bone';

// vtk.js serves .vti as a directory: index.json plus one gzipped file per data
// array, addressed by content hash.
const DATASET_URL = 'https://kitware.github.io/vtk-js/data/volume/LIDC2.vti';
const SCALARS_ID = 'b1ad142a1ebc80f957fcdc329e876d51';

// Pinning the payload's digest is what makes this a regression rather than a
// measurement: if upstream ever republishes LIDC2, the goldens would quietly
// describe a different patient and every calibrated tolerance below would be
// meaningless. Fail loudly instead.
const SCALARS_SHA256 = 'd697a46dadafc1b2e08b9c088e7e78b905c3ddfa602f397992f54cd593ec8b25';

const DIMS = [256, 256, 133];
const SPACING = [1.40625, 1.40625, 2.5];
const ORIGIN = [0, 0, 0];
const DIRECTION = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

// Dense bone in this volume's 8-bit window. Measured occupancy: 101742 voxels.
const THRESHOLD = 128;
const LABEL_VALUE = 1;

const polysegPackageUrl = new URL('../package.json', import.meta.resolve('@icr/polyseg-wasm/js'));
const POLYSEG_VERSION: string = JSON.parse(await readFile(polysegPackageUrl, 'utf8')).version;

const fixturesUrl = process.env.SEGMORPH_FIXTURES_DIR
  ? pathToFileURL(`${process.env.SEGMORPH_FIXTURES_DIR}/`)
  : new URL('../../test/fixtures/', import.meta.url);
const manifestUrl = new URL('manifest.json', fixturesUrl);

// Cached outside the fixture corpus: the corpus is wiped and regenerated per
// oracle run, and re-downloading 1.4 MB every time would make the job depend on
// the network far more than it needs to.
const cacheUrl = new URL('../.cache/', import.meta.url);
const cachedScalarsUrl = new URL(`lidc2-${SCALARS_ID}.raw`, cacheUrl);

async function loadScalars() {
  const cached = await readFile(cachedScalarsUrl).catch((error) => {
    if (error.code !== 'ENOENT') throw error;
    return undefined;
  });
  if (cached) return new Uint8Array(cached);

  const url = `${DATASET_URL}/data/${SCALARS_ID}.gz`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Clinical oracle could not download LIDC2 (${response.status} ${response.statusText}) from ${url}. `
      + 'The oracle tier needs network access for this dataset.',
    );
  }
  const scalars = new Uint8Array(gunzipSync(new Uint8Array(await response.arrayBuffer())));
  await mkdir(cacheUrl, { recursive: true });
  await writeFile(cachedScalarsUrl, scalars);
  return scalars;
}

function writeNrrdBytes(data: Uint8Array) {
  const columns = [0, 1, 2].map((axis) => (
    `(${[0, 1, 2].map((row) => DIRECTION[row][axis] * SPACING[axis]).join(',')})`
  ));
  const header = [
    'NRRD0005',
    'type: uint8',
    'dimension: 3',
    `sizes: ${DIMS.join(' ')}`,
    'space: right-anterior-superior',
    `space directions: ${columns.join(' ')}`,
    'kinds: domain domain domain',
    'encoding: raw',
    'endian: little',
    `space origin: (${ORIGIN.join(',')})`,
    '',
    '',
  ].join('\n');
  const headerBytes = new TextEncoder().encode(header);
  const bytes = new Uint8Array(headerBytes.length + data.length);
  bytes.set(headerBytes);
  bytes.set(data, headerBytes.length);
  return bytes;
}

const scalars = await loadScalars();

const expectedLength = DIMS[0] * DIMS[1] * DIMS[2];
if (scalars.length !== expectedLength) {
  throw new Error(`LIDC2 payload is ${scalars.length} bytes, expected ${expectedLength}`);
}
const digest = createHash('sha256').update(scalars).digest('hex');
if (digest !== SCALARS_SHA256) {
  throw new Error(
    `LIDC2 payload digest ${digest} does not match the pinned ${SCALARS_SHA256}. `
    + 'The upstream dataset changed; re-calibrate the clinical goldens before updating the pin.',
  );
}

const data = new Uint8Array(scalars.length);
let voxelCount = 0;
for (let index = 0; index < scalars.length; index += 1) {
  if (scalars[index] >= THRESHOLD) {
    data[index] = LABEL_VALUE;
    voxelCount += 1;
  }
}
if (voxelCount === 0) throw new Error('Clinical threshold selected no voxels');

const wasmBinary = await readFile(new URL(import.meta.resolve('@icr/polyseg-wasm/wasm')));
const polyseg = await launcher({ wasmBinary });
const result = polyseg.convertLabelmapToSurface(
  Int32Array.from(data),
  DIMS,
  SPACING,
  DIRECTION.flat(),
  ORIGIN,
  [LABEL_VALUE],
);

const caseUrl = new URL(`${ALGORITHM}/${CASE}/`, fixturesUrl);
await mkdir(caseUrl, { recursive: true });

await writeFile(new URL('input.nrrd', caseUrl), writeNrrdBytes(data));
await writeFile(
  new URL('golden.polyseg.mesh.json', caseUrl),
  `${JSON.stringify({ points: Array.from(result.points), polys: Array.from(result.polys) })}\n`,
);

const params = {
  dataset: 'LIDC2',
  source: DATASET_URL,
  sha256: SCALARS_SHA256,
  dims: DIMS,
  spacing: SPACING,
  origin: ORIGIN,
  direction: DIRECTION,
  threshold: THRESHOLD,
  labelValue: LABEL_VALUE,
  voxelCount,
  // Slicer's default binary-labelmap-to-closed-surface parameters, the ones
  // convertLabelmapToSurface applies: smoothingFactor 0.5 maps to
  // passBand 10^(-4*0.5) and 20 + 40*0.5 iterations, with no decimation.
  smoothingFactor: 0.5,
  passBand: 10 ** (-4 * 0.5),
  iterations: 40,
  decimation: 0.0,
  goldenPointCount: result.points.length / 3,
  goldenTriangleCount: result.polys.length / 4,
};
await writeFile(new URL('params.json', caseUrl), `${JSON.stringify(params, null, 2)}\n`);

const manifestText = await readFile(manifestUrl, 'utf8').catch((error) => {
  if (error.code !== 'ENOENT') throw error;
  return JSON.stringify({ schemaVersion: 1, fixtures: [] });
});
const manifest = JSON.parse(manifestText);
type ManifestEntry = { algorithm: string; case: string; oracle: { name: string } };
manifest.fixtures = manifest.fixtures.filter((fixture: ManifestEntry) => !(
  fixture.algorithm === ALGORITHM
  && fixture.case === CASE
  && fixture.oracle.name === ORACLE_NAME
));
manifest.fixtures.push({
  oracle: { name: ORACLE_NAME, version: POLYSEG_VERSION },
  algorithm: ALGORITHM,
  case: CASE,
  params,
  seed: 0,
});
await writeFile(manifestUrl, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(
  `L/${CASE}: voxels=${voxelCount} goldenPoints=${params.goldenPointCount} `
  + `goldenTriangles=${params.goldenTriangleCount}`,
);
