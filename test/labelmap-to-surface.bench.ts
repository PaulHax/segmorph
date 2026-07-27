import { bench, describe } from 'vitest';

import { labelmapToSurface } from '../src/convert/labelmapToSurface.js';
import { createOrientedImage } from '../src/image/orientedImage.js';

// Dev-only benchmark, excluded from `npm test`. Run with:
//   npx vitest bench --run test/labelmap-to-surface.bench.ts

const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
] as const;

function sphereImage(dim: number, radius: number, labelValue: number) {
  const data = new Uint8Array(dim * dim * dim);
  const center = (dim - 1) / 2;
  let offset = 0;
  for (let z = 0; z < dim; z += 1) {
    for (let y = 0; y < dim; y += 1) {
      for (let x = 0; x < dim; x += 1, offset += 1) {
        const dx = x - center;
        const dy = y - center;
        const dz = z - center;
        if (dx * dx + dy * dy + dz * dz <= radius * radius) data[offset] = labelValue;
      }
    }
  }
  return createOrientedImage({
    data,
    dims: [dim, dim, dim],
    spacing: [1, 1, 1],
    origin: [0, 0, 0],
    direction: identity,
  });
}

function sparseBlobImage(dim: number, radius: number, labelValue: number) {
  const data = new Uint8Array(dim * dim * dim);
  const center = [dim * 0.8, dim * 0.15, dim * 0.5] as const;
  let offset = 0;
  for (let z = 0; z < dim; z += 1) {
    for (let y = 0; y < dim; y += 1) {
      for (let x = 0; x < dim; x += 1, offset += 1) {
        const dx = x - center[0];
        const dy = y - center[1];
        const dz = z - center[2];
        if (dx * dx + dy * dy + dz * dz <= radius * radius) data[offset] = labelValue;
      }
    }
  }
  return createOrientedImage({
    data,
    dims: [dim, dim, dim],
    spacing: [1, 1, 1],
    origin: [0, 0, 0],
    direction: identity,
  });
}

const denseSphere = sphereImage(128, 48, 1);
const sparseBlob = sparseBlobImage(192, 6, 3);
const emptyVolume = createOrientedImage({
  data: new Uint8Array(128 ** 3),
  dims: [128, 128, 128],
  spacing: [1, 1, 1],
  origin: [0, 0, 0],
  direction: identity,
});

describe('labelmapToSurface', () => {
  bench('dense sphere r=48 in 128^3', () => {
    labelmapToSurface(denseSphere, { labelValue: 1 });
  });

  bench('sparse blob r=6 in 192^3', () => {
    labelmapToSurface(sparseBlob, { labelValue: 3 });
  });

  bench('empty 128^3', () => {
    labelmapToSurface(emptyVolume, { labelValue: 1 });
  });
});
