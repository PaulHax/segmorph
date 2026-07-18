import { describe, expect, it } from 'vitest';

import { fillBetween } from '../src/convert/fillBetween.js';
import type { OrientedImage } from '../src/image/orientedImage.js';

const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

function makeImage(
  dims: [number, number, number],
  spacing: [number, number, number] = [1, 1, 1],
): OrientedImage<Uint8Array> {
  return {
    dims,
    spacing,
    origin: [0, 0, 0],
    direction: identity,
    data: new Uint8Array(dims[0] * dims[1] * dims[2]),
  };
}

function drawDisk(
  image: OrientedImage<Uint8Array>,
  axis: number,
  sliceIndex: number,
  cu: number,
  cv: number,
  r: number,
  label = 1,
) {
  const [nx, ny] = image.dims;
  const axes = [0, 1, 2].filter((a) => a !== axis);
  const [nu, nv] = [image.dims[axes[0]], image.dims[axes[1]]];
  for (let v = 0; v < nv; v += 1) {
    for (let u = 0; u < nu; u += 1) {
      const du = u - cu;
      const dv = v - cv;
      if (du * du + dv * dv > r * r) continue;
      const idx = [0, 0, 0];
      idx[axis] = sliceIndex;
      idx[axes[0]] = u;
      idx[axes[1]] = v;
      image.data[idx[2] * nx * ny + idx[1] * nx + idx[0]] = label;
    }
  }
}

function sliceForeground(image: OrientedImage<Uint8Array>, axis: number, sliceIndex: number) {
  const [nx, ny, nz] = image.dims;
  const indices: number[] = [];
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        if ([x, y, z][axis] !== sliceIndex) continue;
        if (image.data[z * nx * ny + y * nx + x] !== 0) indices.push(z * nx * ny + y * nx + x);
      }
    }
  }
  return new Set(indices);
}

describe('fillBetween invariants', () => {
  it('returns a new image and does not mutate the input', () => {
    const image = makeImage([12, 12, 7]);
    drawDisk(image, 2, 1, 5.5, 5.5, 4.2);
    drawDisk(image, 2, 5, 5.5, 5.5, 2.2);
    const before = image.data.slice();
    const out = fillBetween(image, { labelValue: 1 });
    expect(image.data).toEqual(before);
    expect(out.data).not.toBe(image.data);
    expect(out.dims).toEqual(image.dims);
    expect(out.spacing).toEqual(image.spacing);
    expect(out.origin).toEqual(image.origin);
    expect(out.direction).toEqual(image.direction);
  });

  it('is a no-op for an empty image', () => {
    const image = makeImage([8, 8, 8]);
    expect(fillBetween(image, { labelValue: 1 }).data.every((v) => v === 0)).toBe(true);
  });

  it('is a no-op when only one slice is segmented', () => {
    const image = makeImage([12, 12, 7]);
    drawDisk(image, 2, 3, 5.5, 5.5, 3.2);
    expect(fillBetween(image, { labelValue: 1 }).data).toEqual(image.data);
  });

  it('rejects labelValue 0', () => {
    expect(() => fillBetween(makeImage([4, 4, 4]), { labelValue: 0 })).toThrow();
  });

  it('fills gap slices between concentric disks, bounded by intersection and union', () => {
    const image = makeImage([16, 16, 7]);
    drawDisk(image, 2, 1, 7.5, 7.5, 5.6);
    drawDisk(image, 2, 5, 7.5, 7.5, 2.6);
    const out = fillBetween(image, { labelValue: 1 });
    const big = sliceForeground(image, 2, 1);
    const small = sliceForeground(image, 2, 5);
    for (let z = 2; z <= 4; z += 1) {
      const mid = sliceForeground(out, 2, z);
      expect(mid.size, `slice ${z} non-empty`).toBeGreaterThan(0);
      const [nx, ny] = image.dims;
      for (const idx of mid) {
        const planar = idx % (nx * ny);
        expect(big.has(1 * nx * ny + planar), `slice ${z} inside union`).toBe(true);
      }
      for (const idx of small) {
        const planar = idx % (nx * ny);
        expect(mid.has(z * nx * ny + planar), `slice ${z} contains intersection`).toBe(true);
      }
    }
  });

  it('interpolates along a non-default axis, auto-detected', () => {
    const image = makeImage([12, 7, 12]);
    // Disks in x-z planes at y = 1 and y = 5.
    drawDisk(image, 1, 1, 5.5, 5.5, 4.2);
    drawDisk(image, 1, 5, 5.5, 5.5, 2.2);
    const out = fillBetween(image, { labelValue: 1 });
    for (let y = 2; y <= 4; y += 1) {
      expect(sliceForeground(out, 1, y).size, `y slice ${y}`).toBeGreaterThan(0);
    }
  });

  it('honors an explicit axis and ignores other axes', () => {
    const image = makeImage([12, 12, 7]);
    drawDisk(image, 2, 1, 5.5, 5.5, 4.2);
    drawDisk(image, 2, 5, 5.5, 5.5, 2.2);
    const alongZ = fillBetween(image, { labelValue: 1, axis: 2 });
    expect(sliceForeground(alongZ, 2, 3).size).toBeGreaterThan(0);
    const alongX = fillBetween(image, { labelValue: 1, axis: 0 });
    expect(alongX.data).toEqual(image.data);
  });

  it('interpolates blobs touching the image boundary', () => {
    const image = makeImage([12, 12, 7]);
    drawDisk(image, 2, 1, 0, 0, 5.2);
    drawDisk(image, 2, 5, 0, 0, 3.2);
    const out = fillBetween(image, { labelValue: 1 });
    for (let z = 2; z <= 4; z += 1) {
      expect(sliceForeground(out, 2, z).size, `slice ${z}`).toBeGreaterThan(0);
    }
  });

  it('preserves voxels of other labels on their original slices', () => {
    const image = makeImage([16, 16, 7]);
    drawDisk(image, 2, 1, 5.5, 5.5, 3.6);
    drawDisk(image, 2, 5, 5.5, 5.5, 3.6);
    drawDisk(image, 2, 3, 12.5, 12.5, 2.2, 7);
    const out = fillBetween(image, { labelValue: 1 });
    for (let i = 0; i < image.data.length; i += 1) {
      if (image.data[i] === 7) expect(out.data[i], `voxel ${i}`).toBe(7);
    }
    // Label 7 is not interpolated onto neighboring slices.
    for (const idx of sliceForeground(out, 2, 2)) {
      expect(out.data[idx]).toBe(1);
    }
  });
});
