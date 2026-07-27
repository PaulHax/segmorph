import { describe, expect, it } from 'vitest';

import {
  createMesh,
  getPoint,
  iteratePoints,
  iterateTriangles,
  triangleCount,
  validateMesh,
  vertexCount,
  type Mesh,
} from '../../src/geometry/mesh.js';

function triangle(): Mesh {
  return {
    points: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    polys: new Uint32Array([3, 0, 1, 2]),
  };
}

describe('mesh geometry', () => {
  it('constructs flat typed arrays from points and triangles', () => {
    const mesh = createMesh(
      [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
      ],
      [[0, 1, 2]],
    );

    expect(mesh.points).toEqual(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
    expect(mesh.polys).toEqual(new Uint32Array([3, 0, 1, 2]));
  });

  it('constructs an empty mesh', () => {
    expect(createMesh([], [])).toEqual({
      points: new Float32Array(),
      polys: new Uint32Array(),
    });
  });

  it('iterates points and vtk-style triangle cells', () => {
    const mesh = createMesh(
      [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      [
        [0, 1, 2],
        [0, 2, 3],
      ],
    );

    expect([...iteratePoints(mesh)]).toEqual([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
    expect([...iterateTriangles(mesh)]).toEqual([
      [0, 1, 2],
      [0, 2, 3],
    ]);
    expect(vertexCount(mesh)).toBe(4);
    expect(triangleCount(mesh)).toBe(2);
  });

  it('reads a point by vertex index', () => {
    expect(getPoint(triangle(), 1)).toEqual([1, 0, 0]);
    expect(() => getPoint(triangle(), -1)).toThrow(RangeError);
    expect(() => getPoint(triangle(), 3)).toThrow(RangeError);
    expect(() => getPoint(triangle(), 0.5)).toThrow(RangeError);
  });

  it.each([
    [{ points: new Float32Array([0, 0]), polys: new Uint32Array() }, 'complete xyz'],
    [{ points: new Float32Array([0, 0, Number.NaN]), polys: new Uint32Array() }, 'finite'],
    [{ points: new Float32Array(9), polys: new Uint32Array([4, 0, 1, 2, 0]) }, 'triangle cells'],
    [{ points: new Float32Array(9), polys: new Uint32Array([3, 0, 1]) }, 'triangle cells'],
    [{ points: new Float32Array(9), polys: new Uint32Array([3, 0, 1, 3]) }, 'out of bounds'],
  ] as const)('rejects an invalid mesh: %s', (mesh, message) => {
    expect(() => validateMesh(mesh)).toThrow(message);
    expect(() => [...iterateTriangles(mesh)]).toThrow(message);
  });

  it('rejects invalid construction input', () => {
    expect(() => createMesh([[0, 0, Infinity]], [])).toThrow('finite');
    expect(() => createMesh([[0, 0, 0]], [[0, 1, 0]])).toThrow('out of bounds');
    expect(() => createMesh([[0, 0, 0]], [[0, -1, 0]])).toThrow('non-negative integers');
  });
});
