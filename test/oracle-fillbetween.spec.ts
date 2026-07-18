import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import { fillBetween } from '../src/convert/fillBetween.js';
import type { OrientedImage } from '../src/image/orientedImage.js';
import { dice } from './diff/image.js';

// Per-slice Dice thresholds against the @itk-wasm oracle (defaults: distance
// transform median, label = 1). Calibration measured 2026-07-10 with itk-wasm
// package 2.0.0 by running the oracle against itself with the only legitimate
// implementation variant flipped (noUseDistanceTransform: true, the iterated
// dilation median our port uses). Minimum per-interpolated-slice Dice between
// the two ITK variants:
//   shrinking-disk   0.8533 (z=3)
//   translated-blob  0.7808 (z=5)
//   split            0.7442 (z=5)
// Thresholds sit just below the measured within-ITK spread; a port whose
// boundaries differ more than ITK differs from itself is wrong.
const cases = [
  { name: 'shrinking-disk', minSliceDice: 0.85 },
  { name: 'translated-blob', minSliceDice: 0.77 },
  { name: 'split', minSliceDice: 0.74 },
];

type ImageJson = {
  dims: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  direction: number[][];
  data: number[];
};

function loadImage(caseName: string, file: string): OrientedImage<Uint8Array> {
  const url = new URL(`./fixtures/H/${caseName}/${file}`, import.meta.url);
  const json: ImageJson = JSON.parse(readFileSync(url, 'utf8'));
  return {
    dims: json.dims,
    spacing: json.spacing,
    origin: json.origin,
    direction: json.direction,
    data: Uint8Array.from(json.data),
  };
}

function loadParams(caseName: string) {
  const url = new URL(`./fixtures/H/${caseName}/params.json`, import.meta.url);
  return JSON.parse(readFileSync(url, 'utf8')) as {
    labelValue: number;
    axis: number;
    segmentedSlices: number[];
  };
}

function slice(image: OrientedImage, z: number) {
  const [nx, ny] = image.dims;
  return image.data.subarray(z * nx * ny, (z + 1) * nx * ny);
}

describe('fillBetween vs itk-wasm morphological contour interpolation', () => {
  for (const { name, minSliceDice } of cases) {
    describe(name, () => {
      const input = loadImage(name, 'input.img.json');
      const golden = loadImage(name, 'golden.img.json');
      const params = loadParams(name);
      const [nx, ny, nz] = input.dims;
      const [first, last] = [
        params.segmentedSlices[0],
        params.segmentedSlices[params.segmentedSlices.length - 1],
      ];
      const actual = fillBetween(input, { labelValue: params.labelValue });

      it('preserves originally segmented slices exactly', () => {
        for (const z of params.segmentedSlices) {
          expect(slice(actual, z), `slice ${z}`).toEqual(slice(golden, z));
        }
      });

      it('leaves slices outside the segmented range empty', () => {
        for (let z = 0; z < nz; z += 1) {
          if (z >= first && z <= last) continue;
          expect(slice(actual, z).every((v) => v === 0), `slice ${z}`).toBe(true);
        }
      });

      it(`matches oracle interpolated slices with Dice >= ${minSliceDice}`, () => {
        for (let z = first + 1; z < last; z += 1) {
          if (params.segmentedSlices.includes(z)) continue;
          const d = dice(slice(actual, z), slice(golden, z), [nx, ny, 1]);
          expect(d, `slice ${z} dice`).toBeGreaterThanOrEqual(minSliceDice);
        }
      });

      it('matches oracle over the whole volume with Dice >= threshold', () => {
        const d = dice(actual.data, golden.data, [nx, ny, nz]);
        expect(d).toBeGreaterThanOrEqual(minSliceDice);
      });
    });
  }
});
