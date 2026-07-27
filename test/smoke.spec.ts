import { describe, expect, it } from 'vitest';

import { createOrientedImage, labelmapToSurface } from '../src/index.js';

describe('package', () => {
  it('exposes its public entry point', () => {
    const data = new Uint8Array(27);
    data[13] = 1;
    const labelmap = createOrientedImage({
      data,
      dims: [3, 3, 3],
      spacing: [1, 1, 1],
      origin: [0, 0, 0],
      direction: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    });

    expect(labelmapToSurface(labelmap, { labelValue: 1 }).points.length).toBeGreaterThan(0);
  });
});
