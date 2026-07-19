// Second oracle for algorithms B and F using vtk.js, a hand-written JavaScript
// reimplementation of the same VTK filters our port and the python-vtk goldens
// come from. An independent codebase agreeing to float precision rules out a
// shared-VTK bug or a mis-read parameter baked into a golden.
//
// - B (windowed-sinc smoothing): vtkWindowedSincPolyDataFilter. vtk.js hardcodes
//   the Hamming window (no SetWindowFunction), so it only cross-checks the
//   "-hamming" B cases; with matched windows it reproduces python-vtk to ~1e-7.
// - F (surface x plane -> contour): vtkCutter. Its cut point set matches the
//   python-vtk golden exactly on every case.
//
// Dev-only; never shipped. Run: node oracles/node/vtkjs.ts [B|F]

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData.js';
import vtkWindowedSincPolyDataFilter from '@kitware/vtk.js/Filters/General/WindowedSincPolyDataFilter.js';
import vtkCutter from '@kitware/vtk.js/Filters/Core/Cutter.js';
import vtkPlane from '@kitware/vtk.js/Common/DataModel/Plane.js';

const VTKJS_VERSION = '36.4.1';
const ORACLE_NAME = 'vtk-js';

const requested = process.argv[2];
if (requested && !['B', 'F'].includes(requested)) {
  throw new Error(`vtk.js oracle covers B and F only; got: ${requested}`);
}

// The Hamming-windowed B cases (see gen_smooth.py).
const SMOOTH_CASES = ['cubesphere-hamming', 'torus-hamming', 'sphere-hamming'];
const CUT_CASES = [
  'cube-axis', 'sphere-center', 'sphere-oblique',
  'torus-two-loops', 'octahedron-on-vertices', 'sphere-miss',
];

// SEGMORPH_FIXTURES_DIR redirects the whole corpus root, so a live regeneration
// run reads its inputs from and writes its goldens into the same tree. Inputs
// here (the B meshes, the F input meshes) come from the Python generators, so
// mixing roots would cross-check a fresh golden against a stale input.
const fixturesUrl = process.env.SEGMORPH_FIXTURES_DIR
  ? pathToFileURL(`${process.env.SEGMORPH_FIXTURES_DIR}/`)
  : new URL('../../test/fixtures/', import.meta.url);
const manifestUrl = new URL('manifest.json', fixturesUrl);

type ManifestEntry = {
  oracle: { name: string; version: string };
  algorithm: string;
  case: string;
  params: Record<string, unknown>;
  seed: number;
};

const written: ManifestEntry[] = [];

async function readMesh(url: URL) {
  const payload = JSON.parse(await readFile(url, 'utf8'));
  return {
    points: Float32Array.from(payload.points),
    polys: Uint32Array.from(payload.polys),
  };
}

function toPolyData(mesh: { points: Float32Array; polys: Uint32Array }) {
  const polydata = vtkPolyData.newInstance();
  polydata.getPoints().setData(mesh.points, 3);
  polydata.getPolys().setData(mesh.polys);
  return polydata;
}

function smooth(mesh: { points: Float32Array; polys: Uint32Array }, params: Record<string, unknown>) {
  const filter = vtkWindowedSincPolyDataFilter.newInstance({
    numberOfIterations: params.numberOfIterations,
    passBand: params.passBand,
    normalizeCoordinates: params.normalizeCoordinates,
    boundarySmoothing: params.boundarySmoothing,
    nonManifoldSmoothing: params.nonManifoldSmoothing,
    edgeAngle: params.edgeAngle,
    featureEdgeSmoothing: params.featureEdgeSmoothing,
    // vtk.js has no window selection; it always applies the Hamming window.
    featureAngle: 45,
  });
  filter.setInputData(toPolyData(mesh));
  const output = filter.getOutputData();
  return {
    points: Array.from(output.getPoints().getData()),
    polys: Array.from(output.getPolys().getData()),
  };
}

// The cut point set (order-independent) is what we cross-check: vtk.js assembles
// loops differently, but the intersection points themselves must match.
function cut(mesh: { points: Float32Array; polys: Uint32Array }, plane: Record<string, number[]>) {
  const cutter = vtkCutter.newInstance();
  cutter.setCutFunction(vtkPlane.newInstance({ origin: plane.origin, normal: plane.normal }));
  cutter.setInputData(toPolyData(mesh));
  return { points: Array.from(cutter.getOutputData().getPoints().getData()) };
}

async function generateSmooth(caseName: string) {
  const caseUrl = new URL(`B/${caseName}/`, fixturesUrl);
  const params = JSON.parse(await readFile(new URL('params.json', caseUrl), 'utf8'));
  const inputUrl = params.input === 'generated'
    ? new URL('input.mesh.json', caseUrl)
    : new URL(params.input, fixturesUrl);

  const golden = smooth(await readMesh(inputUrl), params);
  await mkdir(caseUrl, { recursive: true });
  await writeFile(new URL('golden.vtkjs.mesh.json', caseUrl), `${JSON.stringify(golden)}\n`);
  written.push({ oracle: { name: ORACLE_NAME, version: VTKJS_VERSION }, algorithm: 'B', case: caseName, params, seed: 0 });
}

async function generateCut(caseName: string) {
  const caseUrl = new URL(`F/${caseName}/`, fixturesUrl);
  const params = JSON.parse(await readFile(new URL('params.json', caseUrl), 'utf8'));
  const mesh = await readMesh(new URL(params.inputMesh, fixturesUrl));

  const golden = cut(mesh, params.plane);
  await mkdir(caseUrl, { recursive: true });
  await writeFile(new URL('golden.vtkjs.contour.json', caseUrl), `${JSON.stringify(golden)}\n`);
  written.push({ oracle: { name: ORACLE_NAME, version: VTKJS_VERSION }, algorithm: 'F', case: caseName, params, seed: 0 });
}

if (!requested || requested === 'B') for (const caseName of SMOOTH_CASES) await generateSmooth(caseName);
if (!requested || requested === 'F') for (const caseName of CUT_CASES) await generateCut(caseName);

const manifestText = await readFile(manifestUrl, 'utf8').catch((error) => {
  if (error.code !== 'ENOENT') throw error;
  return JSON.stringify({ schemaVersion: 1, fixtures: [] });
});
const manifest = JSON.parse(manifestText);
manifest.fixtures = manifest.fixtures.filter((fixture: ManifestEntry) => !written.some(
  (entry) => entry.algorithm === fixture.algorithm
    && entry.case === fixture.case
    && entry.oracle.name === fixture.oracle.name,
));
manifest.fixtures.push(...written);
await mkdir(fixturesUrl, { recursive: true });
await writeFile(manifestUrl, `${JSON.stringify(manifest, null, 2)}\n`);
