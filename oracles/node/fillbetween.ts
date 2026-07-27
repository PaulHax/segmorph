// Oracle runner for algorithm H (fill between slices).
// Generates sparse labelmap fixtures and goldens via
// @itk-wasm/morphological-contour-interpolation (wraps ITK's
// itkMorphologicalContourInterpolator, Zukic/Vicory/McCormick,
// after Albu et al. IEEE TBME 2008).
// Run: node oracles/node/fillbetween.ts

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

import { morphologicalContourInterpolationNode } from '@itk-wasm/morphological-contour-interpolation';
import { Image, ImageType, IntTypes, PixelTypes } from 'itk-wasm';

const ORACLE_NAME = 'itk-wasm-morphological-contour-interpolation';
const ORACLE_VERSION = '2.0.0';

type Geometry = {
  dims: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  direction: number[][];
};

type Disk = { z: number; cx: number; cy: number; r: number };

type Case = {
  name: string;
  geometry: Geometry;
  labelValue: number;
  disks: Disk[];
};

const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

const cases: Case[] = [
  {
    // Same center, shrinking radius: pure 1-to-1 morphological median.
    name: 'shrinking-disk',
    geometry: {
      dims: [24, 24, 9],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    },
    labelValue: 1,
    disks: [
      { z: 2, cx: 11.5, cy: 11.5, r: 7.6 },
      { z: 6, cx: 11.5, cy: 11.5, r: 3.6 },
    ],
  },
  {
    // Offset centers: exercises centroid alignment. Anisotropic spacing to
    // confirm the algorithm is index-space and geometry passes through.
    name: 'translated-blob',
    geometry: {
      dims: [24, 24, 9],
      spacing: [0.8, 1.2, 2.5],
      origin: [-3, 4, 10],
      direction: identity,
    },
    labelValue: 1,
    disks: [
      { z: 2, cx: 8, cy: 9, r: 5.2 },
      { z: 6, cx: 15, cy: 13, r: 4.2 },
    ],
  },
  {
    // One wide blob to two separate disks: 1-to-N correspondence split.
    name: 'split',
    geometry: {
      dims: [24, 24, 9],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: identity,
    },
    labelValue: 1,
    disks: [
      { z: 2, cx: 7.5, cy: 11.5, r: 4.6 },
      { z: 2, cx: 12.5, cy: 11.5, r: 4.6 },
      { z: 2, cx: 16.5, cy: 11.5, r: 4.6 },
      { z: 6, cx: 6.5, cy: 11.5, r: 3.1 },
      { z: 6, cx: 17.5, cy: 11.5, r: 3.1 },
    ],
  },
];

function rasterize(c: Case): Uint8Array {
  const [nx, ny, nz] = c.geometry.dims;
  const data = new Uint8Array(nx * ny * nz);
  for (const disk of c.disks) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const dx = x - disk.cx;
        const dy = y - disk.cy;
        if (dx * dx + dy * dy <= disk.r * disk.r) {
          data[disk.z * nx * ny + y * nx + x] = c.labelValue;
        }
      }
    }
  }
  return data;
}

function toItkImage(data: Uint8Array, geometry: Geometry): Image {
  const imageType = new ImageType(3, IntTypes.UInt8, PixelTypes.Scalar, 1);
  const image = new Image(imageType);
  image.size = [...geometry.dims];
  image.spacing = [...geometry.spacing];
  image.origin = [...geometry.origin];
  // itk-wasm direction is a dimension^2 row-major matrix; identity here.
  image.direction = new Float64Array(geometry.direction.flat());
  image.data = data;
  return image;
}

function imageJson(data: ArrayLike<number>, geometry: Geometry) {
  return `${JSON.stringify({
    dims: geometry.dims,
    spacing: geometry.spacing,
    origin: geometry.origin,
    direction: geometry.direction,
    dataType: 'uint8',
    data: Array.from(data),
  })}\n`;
}

// SEGMORPH_FIXTURES_DIR redirects the corpus root so a live regeneration run
// can write into an empty tree instead of the committed fixtures.
const corpusUrl = process.env.SEGMORPH_FIXTURES_DIR
  ? pathToFileURL(`${process.env.SEGMORPH_FIXTURES_DIR}/`)
  : new URL('../../test/fixtures/', import.meta.url);
const fixturesUrl = new URL('H/', corpusUrl);
const manifestUrl = new URL('manifest.json', corpusUrl);

const manifestText = await readFile(manifestUrl, 'utf8').catch((error) => {
  if (error.code !== 'ENOENT') throw error;
  return JSON.stringify({ schemaVersion: 1, fixtures: [] });
});
const manifest = JSON.parse(manifestText);

for (const c of cases) {
  const input = rasterize(c);
  const { outputImage } = await morphologicalContourInterpolationNode(
    toItkImage(input, c.geometry),
    { label: c.labelValue },
  );
  if (!outputImage.data) throw new Error(`Oracle returned no data for case ${c.name}`);

  const caseUrl = new URL(`${c.name}/`, fixturesUrl);
  await mkdir(caseUrl, { recursive: true });
  await writeFile(new URL('input.img.json', caseUrl), imageJson(input, c.geometry));
  await writeFile(
    new URL('golden.img.json', caseUrl),
    imageJson(outputImage.data as Uint8Array, c.geometry),
  );

  const params = {
    labelValue: c.labelValue,
    axis: 2,
    segmentedSlices: [...new Set(c.disks.map((d) => d.z))].sort((a, b) => a - b),
    disks: c.disks,
  };
  await writeFile(new URL('params.json', caseUrl), `${JSON.stringify(params, null, 2)}\n`);

  const entry = {
    oracle: { name: ORACLE_NAME, version: ORACLE_VERSION },
    algorithm: 'H',
    case: c.name,
    params: { ...params, ...c.geometry },
    seed: 0,
  };
  manifest.fixtures = manifest.fixtures.filter(
    (fixture: { algorithm: string; case: string; oracle: { name: string } }) =>
      !(
        fixture.algorithm === entry.algorithm &&
        fixture.case === entry.case &&
        fixture.oracle.name === entry.oracle.name
      ),
  );
  manifest.fixtures.push(entry);
  console.log(`H/${c.name}: golden written`);
}

await mkdir(corpusUrl, { recursive: true });
await writeFile(manifestUrl, `${JSON.stringify(manifest, null, 2)}\n`);
