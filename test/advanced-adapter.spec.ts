import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import { describe, expect, it } from 'vitest';

import { toOrientedImage, toVtkImageData } from '../examples/advanced/src/adapter.js';
import { createOrientedImage, indexToWorld } from '../src/image/orientedImage.js';

const geometry = {
  dims: [2, 3, 4],
  spacing: [0.7, 2, 4],
  origin: [7, -13, 19],
  direction: [
    [Math.SQRT1_2, -Math.SQRT1_2, 0],
    [Math.SQRT1_2, Math.SQRT1_2, 0],
    [0, 0, 1],
  ],
} as const;

describe('advanced example image adapters', () => {
  it('preserves vtk.js world coordinates and shares the scalar array', () => {
    const data = new Uint8Array(24);
    const vtkImage = vtkImageData.newInstance();
    vtkImage.setDimensions(2, 3, 4);
    vtkImage.setSpacing([...geometry.spacing]);
    vtkImage.setOrigin([...geometry.origin]);
    vtkImage.setDirection([Math.SQRT1_2, Math.SQRT1_2, 0, -Math.SQRT1_2, Math.SQRT1_2, 0, 0, 0, 1]);
    vtkImage.getPointData().setScalars(vtkDataArray.newInstance({ values: data }));

    const image = toOrientedImage(vtkImage);
    expect(image.data).toBe(data);
    for (const point of [
      [1, 0, 0],
      [0, 2, 0],
      [1, 2, 3],
    ] as const) {
      const actual = indexToWorld(image, point);
      const expected = vtkImage.indexToWorld([...point]);
      actual.forEach((coordinate, axis) => expect(coordinate).toBeCloseTo(expected[axis], 12));
    }
  });

  it('preserves segmorph world coordinates and shares the scalar array', () => {
    const image = createOrientedImage({ data: new Uint8Array(24), ...geometry });
    const vtkImage = toVtkImageData(image);
    expect(vtkImage.getPointData().getScalars().getData()).toBe(image.data);
    for (const point of [
      [1, 0, 0],
      [0, 2, 0],
      [1, 2, 3],
    ] as const) {
      const actual = vtkImage.indexToWorld([...point]);
      const expected = indexToWorld(image, point);
      actual.forEach((coordinate, axis) => expect(coordinate).toBeCloseTo(expected[axis], 12));
    }
  });
});
