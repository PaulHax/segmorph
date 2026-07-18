// Deterministic synthetic clinical-ish data, in segmorph's native
// representations. main.ts encodes these to real DICOM bytes and everything
// rendered comes from decoding those bytes back - so the adapters, not this
// module, are what feed the pipeline.

import type { RtRoi } from './rtstruct';
import type { SegGeometry, SegSegment } from './seg';

function circleLoop(
  centerX: number,
  centerY: number,
  z: number,
  radiusX: number,
  count: number,
  options: { radiusY?: number; clockwise?: boolean; startAngle?: number } = {},
): number[] {
  const radiusY = options.radiusY ?? radiusX;
  const step = ((options.clockwise ? -1 : 1) * 2 * Math.PI) / count;
  const startAngle = options.startAngle ?? 0;
  const loop: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const angle = startAngle + index * step;
    loop.push(centerX + radiusX * Math.cos(angle), centerY + radiusY * Math.sin(angle), z);
  }
  return loop;
}

/** A tumor-like blob: sphere cross sections, slightly squashed. */
function tumorRoi(): RtRoi {
  const centerX = 40;
  const centerY = 0;
  const loops: Float64Array[] = [];
  for (let slice = 0; slice < 9; slice += 1) {
    const z = -16 + slice * 4;
    const radius = Math.sqrt(18 ** 2 - z ** 2);
    loops.push(Float64Array.from(
      circleLoop(centerX, centerY, z, radius, 28, { radiusY: radius * 0.8 }),
    ));
  }
  return { number: 1, name: 'Tumor', color: [230, 120, 40], loops };
}

/**
 * An annulus stored with the RTSTRUCT keyhole technique: each slice is a
 * single loop - outer ring, a zero-width channel of duplicated points, and
 * the inner ring wound the other way. contourToSurface's keyhole splitter
 * turns it back into a wall with a hole.
 */
function ringRoi(): RtRoi {
  const centerX = 40;
  const centerY = 52;
  const outer = 16;
  const inner = 8;
  const loops: Float64Array[] = [];
  for (let slice = 0; slice < 6; slice += 1) {
    const z = -7.5 + slice * 3;
    const loop: number[] = [];
    loop.push(...circleLoop(centerX, centerY, z, outer, 32));
    loop.push(centerX + 11.5, centerY, z);
    loop.push(centerX + 8.7, centerY, z);
    loop.push(...circleLoop(centerX, centerY, z, inner, 24, { clockwise: true, startAngle: -0.2 }));
    loop.push(centerX + 8.7, centerY, z);
    loop.push(centerX + 11.5, centerY, z);
    loops.push(Float64Array.from(loop));
  }
  return { number: 2, name: 'Ring (keyhole)', color: [60, 190, 180], loops };
}

export function buildRtStructRois(): RtRoi[] {
  return [tumorRoi(), ringRoi()];
}

/**
 * The SEG side: two deliberately overlapping segments on one grid - a sphere
 * inside a hollow box shell. A single fused labelmap could not store them
 * (one value per voxel); DICOM SEG gives each segment its own frames, and
 * segmorph gives each segment its own labelmap, so overlap survives the trip.
 */
export function buildSegData(): { geometry: SegGeometry; segments: SegSegment[] } {
  const dims: [number, number, number] = [44, 44, 26];
  const spacing: [number, number, number] = [1.4, 1.4, 2.2];
  const geometry: SegGeometry = {
    dims,
    spacing,
    origin: [-100, -28, -28],
    direction: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
  };

  const [nx, ny, nz] = dims;
  const sphere = new Uint8Array(nx * ny * nz);
  const shell = new Uint8Array(nx * ny * nz);
  const center = [(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2];
  const radius = 9;
  const boxMin = [7, 7, 4];
  const boxMax = [nx - 8, ny - 8, nz - 5];
  const wall = 2;

  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const index = x + nx * (y + ny * z);
        // Sphere in physical proportions (z voxels are thicker).
        const dx = (x - center[0]) * spacing[0];
        const dy = (y - center[1]) * spacing[1];
        const dz = (z - center[2]) * spacing[2];
        if (dx * dx + dy * dy + dz * dz <= (radius * spacing[0]) ** 2 * 2.2) {
          sphere[index] = 1;
        }
        const inBox = x >= boxMin[0] && x <= boxMax[0]
          && y >= boxMin[1] && y <= boxMax[1]
          && z >= boxMin[2] && z <= boxMax[2];
        const inCore = x >= boxMin[0] + wall && x <= boxMax[0] - wall
          && y >= boxMin[1] + wall && y <= boxMax[1] - wall
          && z >= boxMin[2] + wall && z <= boxMax[2] - wall;
        if (inBox && !inCore) shell[index] = 1;
      }
    }
  }

  return {
    geometry,
    segments: [
      { number: 1, label: 'Sphere', data: sphere },
      { number: 2, label: 'Box shell', data: shell },
    ],
  };
}

/** Display colors for SEG segments (this demo keeps color app-side). */
export const SEG_COLORS: Record<string, [number, number, number]> = {
  Sphere: [150, 110, 220],
  'Box shell': [120, 200, 90],
};
