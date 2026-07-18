import { describe, expect, it } from 'vitest';

import type { Mesh } from './mesh.js';
import {
  hasConsistentOutwardOrientation,
  isManifold,
  isVolumeWithinBand,
  isWatertight,
} from './structure.js';

function cube(): Mesh {
  return {
    points: new Float32Array([
      0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
      0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
    ]),
    polys: new Uint32Array([
      3, 0, 2, 1, 3, 0, 3, 2,
      3, 4, 5, 6, 3, 4, 6, 7,
      3, 0, 1, 5, 3, 0, 5, 4,
      3, 3, 7, 6, 3, 3, 6, 2,
      3, 0, 4, 7, 3, 0, 7, 3,
      3, 1, 2, 6, 3, 1, 6, 5,
    ]),
  };
}

describe('mesh structural invariants', () => {
  it('accepts a closed outward-oriented cube', () => {
    const mesh = cube();

    expect(isWatertight(mesh)).toBe(true);
    expect(isManifold(mesh)).toBe(true);
    expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
  });

  it('rejects a cube missing one face as not watertight', () => {
    const mesh = cube();
    const missingFace = { ...mesh, polys: mesh.polys.slice(8) };

    expect(isWatertight(missingFace)).toBe(false);
    expect(isManifold(missingFace)).toBe(true);
  });

  it('rejects a cube with one flipped triangle as inconsistently oriented', () => {
    const mesh = cube();
    const polys = mesh.polys.slice();
    [polys[1], polys[2]] = [polys[2], polys[1]];

    expect(hasConsistentOutwardOrientation({ ...mesh, polys })).toBe(false);
  });

  it('rejects non-manifold edges and bow-tie vertices', () => {
    const edgePoints = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1]);
    const nonManifoldEdge = {
      points: edgePoints,
      polys: new Uint32Array([3, 0, 1, 2, 3, 1, 0, 3, 3, 0, 1, 4]),
    };
    const bowTie = {
      points: edgePoints,
      polys: new Uint32Array([3, 0, 1, 2, 3, 0, 3, 4]),
    };

    expect(isManifold(nonManifoldEdge)).toBe(false);
    expect(isManifold(bowTie)).toBe(false);
  });

  it('checks absolute volume ratio against an inclusive band', () => {
    expect(isVolumeWithinBand(cube(), 1, 0.1)).toBe(true);
    expect(isVolumeWithinBand(cube(), 0.9, 1 / 9)).toBe(true);
    expect(isVolumeWithinBand(cube(), 0.8, 0.1)).toBe(false);
  });
});
