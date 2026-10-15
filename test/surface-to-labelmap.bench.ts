import { bench, describe } from 'vitest';

import { surfaceToLabelmap } from '../src/convert/surfaceToLabelmap.js';
import type { ImageGeometry } from '../src/index.js';
import { sphereMesh } from './surfaceToLabelmapCases.js';

const geometry: ImageGeometry = {
  dims: [96, 96, 96],
  spacing: [1, 1, 1],
  origin: [0, 0, 0],
  direction: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
};
const mesh = sphereMesh([48, 48, 48], 40, 48, 64);

describe('surfaceToLabelmap', () => {
  bench('dense sphere r=40 in 96^3', () => {
    surfaceToLabelmap(mesh, geometry, { labelValue: 1 });
  });
});
