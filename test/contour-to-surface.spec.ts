import { describe, expect, it } from 'vitest';

import { contourToSurface } from '../src/convert/contourToSurface.js';
import { surfaceToContour } from '../src/convert/surfaceToContour.js';
import { createContourPlane } from '../src/geometry/contour.js';
import { contourToSurfaceCases } from './contourToSurfaceCases.js';
import { hasConsistentOutwardOrientation, isManifold, isWatertight } from './diff/structure.js';
import { boundingBox } from './diff/mesh.js';

describe('contourToSurface validation', () => {
  it('rejects an empty stack', () => {
    expect(() => contourToSurface([])).toThrow(RangeError);
  });

  it('rejects a loop with fewer than three points', () => {
    expect(() => contourToSurface([[0, 0, 0, 1, 0, 0]])).toThrow(RangeError);
  });

  it('rejects incomplete xyz coordinates', () => {
    expect(() => contourToSurface([[0, 0, 0, 1, 0, 0, 1, 1]])).toThrow(RangeError);
  });

  it('rejects non-finite coordinates', () => {
    expect(() => contourToSurface([[0, 0, 0, 1, 0, 0, Number.NaN, 1, 0]])).toThrow(RangeError);
  });
});

describe('contourToSurface structural invariants', () => {
  // The oracle's own output has open edges (its cap loops close on duplicated
  // coordinates with distinct ids); the port shares vertices along every seam
  // instead, so with default smooth capping the result is fully closed on
  // every fixture case.
  for (const [caseName, { loops }] of Object.entries(contourToSurfaceCases)) {
    it(`produces a watertight, manifold, outward-oriented surface for ${caseName}`, () => {
      const mesh = contourToSurface(loops);
      expect(isWatertight(mesh)).toBe(true);
      expect(isManifold(mesh)).toBe(true);
      expect(hasConsistentOutwardOrientation(mesh)).toBe(true);
    });
  }
});

const axialPlane = (z: number) => createContourPlane([0, 0, z], [1, 0, 0], [0, 1, 0]);

function loopRadialRange(loop: { points: Float64Array }, centerX: number, centerY: number) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let offset = 0; offset < loop.points.length; offset += 2) {
    const radius = Math.hypot(loop.points[offset] - centerX, loop.points[offset + 1] - centerY);
    minimum = Math.min(minimum, radius);
    maximum = Math.max(maximum, radius);
  }
  return { minimum, maximum };
}

function loopArea(loop: { points: Float64Array }) {
  let area = 0;
  const count = loop.points.length / 2;
  for (let index = 0; index < count; index += 1) {
    const next = (index + 1) % count;
    area +=
      loop.points[index * 2] * loop.points[next * 2 + 1] -
      loop.points[next * 2] * loop.points[index * 2 + 1];
  }
  return area / 2;
}

describe('contour -> surface -> contour round trip', () => {
  // Cutting the reconstructed surface between two source planes must
  // reproduce the source loop shape: the stitch never moves points off the
  // source contours, so a mid-slice cut interpolates between two identical
  // circles. Chord error for 24 segments at radius 20 is
  // 20 * (1 - cos(pi/24)) = 0.17, so 0.25 bounds the radial deviation.
  it('reproduces the cylinder circle at a mid slice', () => {
    const mesh = contourToSurface(contourToSurfaceCases.cylinder.loops);
    const contour = surfaceToContour(mesh, axialPlane(2.25));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(1);
    const { minimum, maximum } = loopRadialRange(contour!.loops[0], 5, -3);
    expect(minimum).toBeGreaterThan(20 - 0.25);
    expect(maximum).toBeLessThan(20 + 0.25);
  });

  it('preserves the keyhole annulus hole', () => {
    const mesh = contourToSurface(contourToSurfaceCases.keyhole.loops);
    const contour = surfaceToContour(mesh, axialPlane(2.25));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(2);
    // The keyhole splitter leaves one channel remnant point on each ring
    // (the same remnants the original produces), so compare enclosed areas,
    // which the zero-width channel barely dents: outer near pi * 16^2 = 804
    // (32-segment polygon: 798), inner near pi * 8^2 = 201 (24-segment
    // polygon: 199).
    const areas = contour!.loops
      .map((loop) => Math.abs(loopArea(loop)))
      .sort((left, right) => left - right);
    expect(areas[0]).toBeGreaterThan(185);
    expect(areas[0]).toBeLessThan(210);
    expect(areas[1]).toBeGreaterThan(770);
    expect(areas[1]).toBeLessThan(810);
  });

  it('splits into two loops above the branch plane', () => {
    const mesh = contourToSurface(contourToSurfaceCases.branching.loops);
    const contour = surfaceToContour(mesh, axialPlane(11));
    expect(contour).toBeDefined();
    expect(contour!.loops).toHaveLength(2);
  });
});

describe('end capping modes', () => {
  const { loops } = contourToSurfaceCases.cylinder;
  const contourMinZ = 0;
  const contourMaxZ = 10.5;
  const spacing = 1.5;

  it('none leaves the tube open at the contour extents', () => {
    const mesh = contourToSurface(loops, { endCapping: 'none' });
    expect(isWatertight(mesh)).toBe(false);
    const bounds = boundingBox(mesh);
    expect(bounds.min[2]).toBeCloseTo(contourMinZ, 5);
    expect(bounds.max[2]).toBeCloseTo(contourMaxZ, 5);
  });

  it('smooth extends half a slice beyond the contour extents', () => {
    const mesh = contourToSurface(loops);
    const bounds = boundingBox(mesh);
    expect(bounds.min[2]).toBeCloseTo(contourMinZ - spacing / 2, 5);
    expect(bounds.max[2]).toBeCloseTo(contourMaxZ + spacing / 2, 5);
  });

  it('straight copies the contour half a slice out of plane', () => {
    const mesh = contourToSurface(loops, { endCapping: 'straight' });
    const bounds = boundingBox(mesh);
    expect(bounds.min[2]).toBeCloseTo(contourMinZ - spacing / 2, 5);
    expect(bounds.max[2]).toBeCloseTo(contourMaxZ + spacing / 2, 5);
    // Straight caps keep the source circle's footprint; smooth caps shrink it.
    const smooth = boundingBox(contourToSurface(loops));
    expect(bounds.max[0] - bounds.min[0]).toBeGreaterThanOrEqual(smooth.max[0] - smooth.min[0]);
  });

  it('defaultSliceThickness drives the caps when only one plane exists', () => {
    const single = [contourToSurfaceCases.cylinder.loops[0]];
    const mesh = contourToSurface(single, { defaultSliceThickness: 4 });
    const bounds = boundingBox(mesh);
    expect(bounds.min[2]).toBeCloseTo(-2, 5);
    expect(bounds.max[2]).toBeCloseTo(2, 5);
  });
});
