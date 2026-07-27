// Composed second oracle: @icr/polyseg-wasm runs Slicer's real PolySeg
// conversion rules end to end. It is the "what Slicer actually returns"
// ground truth that cross-checks the per-filter Python-vtk goldens (and our
// TS port). Dev-only; WASM-backed; never shipped.
//
// Run: node oracles/node/polyseg.ts [A|D|K]

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

import launcher from '@icr/polyseg-wasm/js';

import { readNrrd } from '../../src/io/nrrd.ts';
import { indexToWorld, worldToIndex } from '../../src/image/orientedImage.ts';
import type { ImageGeometry, Vector3 } from '../../src/image/orientedImage.ts';
import { contourToSurfaceCases } from '../../test/contourToSurfaceCases.ts';

// The package exports map hides package.json, so resolve it from the exported
// entry. Reading the installed version keeps the manifest honest after a bump.
const polysegPackageUrl = new URL('../package.json', import.meta.resolve('@icr/polyseg-wasm/js'));
const POLYSEG_VERSION: string = JSON.parse(await readFile(polysegPackageUrl, 'utf8')).version;
const ORACLE_NAME = 'icr-polyseg-wasm';

const fixturesUrl = process.env.SEGMORPH_FIXTURES_DIR
  ? pathToFileURL(`${process.env.SEGMORPH_FIXTURES_DIR}/`)
  : new URL('../../test/fixtures/', import.meta.url);
const manifestUrl = new URL('manifest.json', fixturesUrl);

const requested = process.argv[2];
if (requested && !['A', 'D', 'K'].includes(requested)) {
  throw new Error(`polyseg oracle covers A, D and K only; got: ${requested}`);
}

const wasmBinary = await readFile(new URL(import.meta.resolve('@icr/polyseg-wasm/wasm')));
const polyseg = await launcher({ wasmBinary });

type ManifestEntry = {
  oracle: { name: string; version: string };
  algorithm: string;
  case: string;
  params: Record<string, unknown>;
  seed: number;
};

const written: ManifestEntry[] = [];

function caseUrl(algorithm: string, caseName: string) {
  return new URL(`${algorithm}/${caseName}/`, fixturesUrl);
}

// A: binary labelmap -> closed surface. The composed rule pads, runs flying
// edges, smooths, and decimates exactly as Slicer does.
async function generateSurface(caseName: string) {
  const url = caseUrl('A', caseName);
  const input = readNrrd(await readFile(new URL('input.nrrd', url)));
  const params = JSON.parse(await readFile(new URL('params.json', url), 'utf8'));
  const result = polyseg.convertLabelmapToSurface(
    Int32Array.from(input.data),
    input.dims,
    input.spacing,
    input.direction.flat(),
    input.origin,
    [1],
  );

  const mesh = { points: Array.from(result.points), polys: Array.from(result.polys) };
  await writeFile(new URL('golden.polyseg.mesh.json', url), `${JSON.stringify(mesh)}\n`);

  written.push({
    oracle: { name: ORACLE_NAME, version: POLYSEG_VERSION },
    algorithm: 'A',
    case: caseName,
    params: {
      dims: input.dims,
      radius: params.radius,
      smoothingFactor: params.smoothingFactor,
      decimation: params.decimation,
    },
    seed: 0,
  });
}

// Nearest-neighbor sample of a source labelmap at each voxel of a reference
// geometry. Uses only the geometry index<->world mapping (algorithm E, itself
// oracle-tested), never the surfaceToLabelmap rasterizer under test.
function resampleToReference(
  reference: ImageGeometry,
  source: ImageGeometry,
  sourceData: ArrayLike<number>,
) {
  const [rx, ry, rz] = reference.dims;
  const [sx, sy, sz] = source.dims;
  const out = new Uint8Array(rx * ry * rz);
  for (let z = 0; z < rz; z += 1) {
    for (let y = 0; y < ry; y += 1) {
      for (let x = 0; x < rx; x += 1) {
        const world = indexToWorld(reference, [x, y, z] as Vector3);
        const [ix, iy, iz] = worldToIndex(source, world).map(Math.round);
        if (ix < 0 || ix >= sx || iy < 0 || iy >= sy || iz < 0 || iz >= sz) continue;
        if (sourceData[ix + sx * (iy + sy * iz)]) out[x + rx * (y + ry * z)] = 1;
      }
    }
  }
  return out;
}

// D: closed surface -> binary labelmap (voxelization). PolySeg rasterizes onto
// its own tight, super-resolution grid (it ignores the caller's reference
// dims), so we resample its occupancy onto the reference geometry before
// comparing. Cross-checks the Python vtkPolyDataToImageStencil golden and our
// TS surfaceToLabelmap port.
async function generateLabelmap(caseName: string) {
  const url = caseUrl('D', caseName);
  const mesh = JSON.parse(await readFile(new URL('input.mesh.json', url), 'utf8'));
  const geometry = JSON.parse(await readFile(new URL('params.json', url), 'utf8'));

  const result = polyseg.convertSurfaceToLabelmap(
    Float32Array.from(mesh.points),
    Uint32Array.from(mesh.polys),
    geometry.dims,
    geometry.spacing,
    geometry.direction.flat(),
    geometry.origin,
  );

  const source: ImageGeometry = {
    dims: Array.from(result.dimensions),
    spacing: Array.from(result.spacing),
    origin: Array.from(result.origin),
    direction: [
      Array.from(result.direction.slice(0, 3)),
      Array.from(result.direction.slice(3, 6)),
      Array.from(result.direction.slice(6, 9)),
    ],
  };
  const data = resampleToReference(geometry, source, result.data);

  const labelmap = { data: Array.from(data), dims: geometry.dims };
  await writeFile(new URL('golden.polyseg.labelmap.json', url), `${JSON.stringify(labelmap)}\n`);

  written.push({
    oracle: { name: ORACLE_NAME, version: POLYSEG_VERSION },
    algorithm: 'D',
    case: caseName,
    params: {
      dims: geometry.dims,
      spacing: geometry.spacing,
      origin: geometry.origin,
      direction: geometry.direction,
    },
    seed: 0,
  });
}

// K: planar contours -> closed surface. The wasm wrapper feeds the world-space
// loops into the (MIT-licensed SlicerRT) planar-contour-to-closed-surface
// rule with its default parameters (smooth end capping, default slice
// thickness 0), which is exactly the chain our TS port reproduces. This
// oracle also writes the inputs, generated by test/contourToSurfaceCases.ts,
// so input and golden always come from the same run.
async function generateContourSurface(caseName: string) {
  const { loops, params } = contourToSurfaceCases[caseName];
  const url = caseUrl('K', caseName);
  await mkdir(url, { recursive: true });

  const flatPoints = Float32Array.from(loops.flat());
  const numPoints = Int32Array.from(loops.map((loop) => loop.length / 3));
  const result = polyseg.convertContourRoiToSurface(flatPoints, numPoints);

  const mesh = { points: Array.from(result.points), polys: Array.from(result.polys) };
  await writeFile(new URL('input.contours.json', url), `${JSON.stringify({ loops })}\n`);
  await writeFile(new URL('params.json', url), `${JSON.stringify(params, null, 2)}\n`);
  await writeFile(new URL('golden.polyseg.mesh.json', url), `${JSON.stringify(mesh)}\n`);

  written.push({
    oracle: { name: ORACLE_NAME, version: POLYSEG_VERSION },
    algorithm: 'K',
    case: caseName,
    params,
    seed: 0,
  });
}

if (!requested || requested === 'A') await generateSurface('sphere');
if (!requested || requested === 'D') await generateLabelmap('sphere');
if (!requested || requested === 'K') {
  for (const caseName of Object.keys(contourToSurfaceCases)) {
    await generateContourSurface(caseName);
  }
}

const manifestText = await readFile(manifestUrl, 'utf8').catch((error) => {
  if (error.code !== 'ENOENT') throw error;
  return JSON.stringify({ schemaVersion: 1, fixtures: [] });
});
const manifest = JSON.parse(manifestText);
manifest.fixtures = manifest.fixtures.filter(
  (fixture: ManifestEntry) =>
    !written.some(
      (entry) =>
        entry.algorithm === fixture.algorithm &&
        entry.case === fixture.case &&
        entry.oracle.name === fixture.oracle.name,
    ),
);
manifest.fixtures.push(...written);
await writeFile(manifestUrl, `${JSON.stringify(manifest, null, 2)}\n`);
