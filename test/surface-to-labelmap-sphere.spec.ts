import { describe, expect, it } from 'vitest';

import { surfaceToLabelmap } from '../src/convert/surfaceToLabelmap.js';
import type { ImageGeometry } from '../src/index.js';
import { sphereMesh } from './surfaceToLabelmapCases.js';

describe('surfaceToLabelmap sphere coverage', () => {
  it('fills samples inside a triangulated sphere', () => {
    const geometry: ImageGeometry = {
      dims: [24, 24, 24],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    };
    const center = [12, 12, 12] as const;
    const mesh = sphereMesh(center, 10, 12, 16);

    const result = surfaceToLabelmap(mesh, geometry, { labelValue: 1 });

    const foreground = [...result.data].filter(Boolean).length;
    // A radius-10 sphere on unit spacing encloses about 4,189 voxels.
    expect(foreground).toBeGreaterThan(3_700);
    expect(foreground).toBeLessThan(4_500);
  });
});
