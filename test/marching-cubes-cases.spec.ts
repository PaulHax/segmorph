import { createHash } from 'node:crypto';

import { describe, expect, it } from 'vitest';

import { marchingCubesCases } from '../src/convert/marchingCubesCases.js';

describe('VTK voxel contour cases', () => {
  it('matches the complete canonical VTK table', () => {
    expect(marchingCubesCases).toHaveLength(256);
    expect(marchingCubesCases.every((row) => row.length === 16)).toBe(true);
    expect(createHash('sha256').update(JSON.stringify(marchingCubesCases)).digest('hex')).toBe(
      'a712f2dbb00f80e8e93ccfd52f45b83568646dc9c4b0f00b2bfff769cc9db41e',
    );
  });
});
