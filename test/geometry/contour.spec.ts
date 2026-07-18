import { describe, expect, it } from 'vitest';
import {
  createContourLoop,
  createContourPlane,
  createPlanarContour,
  planeToWorld,
  validateContourLoop,
  validateContourPlane,
  validatePlanarContour,
  worldToPlane,
} from '../../src/geometry/contour';

describe('contour plane', () => {
  it('constructs an orthonormal plane and transforms coordinates', () => {
    const plane = createContourPlane([10, 20, 30], [0, 1, 0], [0, 0, 1]);

    expect(plane).toEqual({
      origin: [10, 20, 30],
      xAxis: [0, 1, 0],
      yAxis: [0, 0, 1],
    });
    expect(planeToWorld(plane, [2, -3])).toEqual([10, 22, 27]);
    expect(worldToPlane(plane, [14, 22, 27])).toEqual([2, -3]);
  });

  it('rejects invalid origins and bases', () => {
    expect(() => createContourPlane([0, 0, Number.NaN], [1, 0, 0], [0, 1, 0]))
      .toThrow(/origin/);
    expect(() => createContourPlane([0, 0, 0], [2, 0, 0], [0, 1, 0]))
      .toThrow(/orthonormal/);
    expect(() => createContourPlane([0, 0, 0], [1, 0, 0], [1, 0, 0]))
      .toThrow(/orthonormal/);
  });

  it('validates plain plane records', () => {
    expect(() => validateContourPlane({
      origin: [0, 0, 0], xAxis: [1, 0, 0], yAxis: [0, 1, 0],
    })).not.toThrow();
  });
});

describe('contour loops', () => {
  it('stores planar vertices in a typed array with implicit closure', () => {
    const loop = createContourLoop([[0, 0], [4, 0], [0, 3]]);

    expect(loop.points).toBeInstanceOf(Float64Array);
    expect([...loop.points]).toEqual([0, 0, 4, 0, 0, 3]);
    expect(() => validateContourLoop(loop)).not.toThrow();
  });

  it('rejects incomplete, non-finite, and undersized loops', () => {
    expect(() => validateContourLoop({ points: new Float64Array([0, 0, 1]) }))
      .toThrow(/complete xy/);
    expect(() => createContourLoop([[0, 0], [1, 0]]))
      .toThrow(/at least three/);
    expect(() => createContourLoop([[0, 0], [1, 0], [Number.POSITIVE_INFINITY, 1]]))
      .toThrow(/finite/);
  });
});

describe('planar contour', () => {
  it('constructs and validates a plane with closed loops', () => {
    const plane = createContourPlane([0, 0, 2], [1, 0, 0], [0, 1, 0]);
    const first = createContourLoop([[0, 0], [2, 0], [0, 2]]);
    const second = createContourLoop([[3, 3], [4, 3], [3, 4]]);
    const contour = createPlanarContour(plane, [first, second]);

    expect(contour).toEqual({ plane, loops: [first, second] });
    expect(() => validatePlanarContour(contour)).not.toThrow();
  });

  it('rejects contours without loops', () => {
    const plane = createContourPlane([0, 0, 0], [1, 0, 0], [0, 1, 0]);
    expect(() => createPlanarContour(plane, [])).toThrow(/at least one loop/);
  });
});
