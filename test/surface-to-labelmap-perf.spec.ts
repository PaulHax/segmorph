import { describe, expect, it } from 'vitest';

import { surfaceToLabelmap } from '../src/convert/surfaceToLabelmap.js';
import type { ImageGeometry } from '../src/index.js';
import { sphereMesh } from './surfaceToLabelmapCases.js';

// Performance guard. The ray-aligned 2D grid prunes candidate triangles so a
// dense sphere voxelized onto a 96^3 grid runs in roughly two seconds on
// developer hardware. The base linear scan (every voxel tested against every
// triangle) took minutes for the same input.
//
// The bound is the test timeout rather than an assertion on elapsed time. Both
// measure the host as much as the algorithm, so the point of the change is the
// headroom: at a 20 second assertion this flaked purely because the machine was
// loaded, despite finishing in about two seconds idle. 60 seconds stays well
// clear of a contended runner while remaining far below the minutes an
// O(voxels * triangles) regression would take.
//
// Note the conversion is synchronous, so the timeout is evaluated after it
// returns rather than interrupting it. A regression still runs to completion
// before the test fails; the timeout fails the build, it does not cut the work
// short.
const TIMEOUT_MS = 60_000;

describe('surfaceToLabelmap performance', () => {
  it(
    'voxelizes a dense 96^3 sphere with pruned ray tests',
    () => {
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
      const center = [48, 48, 48] as const;
      const mesh = sphereMesh(center, 40, 48, 64);

      const result = surfaceToLabelmap(mesh, geometry, { labelValue: 1 });

      const foreground = [...result.data].filter(Boolean).length;
      // A radius-40 sphere on unit spacing fills ~ (4/3) pi 40^3 ~ 268k voxels.
      expect(foreground).toBeGreaterThan(240_000);
      expect(foreground).toBeLessThan(290_000);
    },
    TIMEOUT_MS,
  );
});
