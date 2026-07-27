import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import type { ImageData as SegmorphImageData, Mesh, OrientedImage } from 'segmorph';

/**
 * vtk.js <-> segmorph adapters.
 *
 * segmorph's data contracts are deliberately vtk.js shaped, so these are
 * reshapes rather than conversions: the voxel and vertex arrays are passed
 * through by reference, never copied or rewritten.
 */

/** vtkImageData -> OrientedImage. The scalar array is shared, not copied. */
export function toOrientedImage<T extends SegmorphImageData>(image: vtkImageData) {
  const direction = image.getDirection();
  return {
    data: image.getPointData().getScalars().getData() as T,
    dims: image.getDimensions(),
    spacing: image.getSpacing(),
    origin: image.getOrigin(),
    // vtk.js stores the direction as a flat 9-element row-major matrix;
    // segmorph wants it as three rows. This is the only reshape needed.
    direction: [direction.slice(0, 3), direction.slice(3, 6), direction.slice(6, 9)],
  } as OrientedImage<T>;
}

/** OrientedImage -> vtkImageData, sharing the same underlying typed array. */
export function toVtkImageData(image: OrientedImage) {
  const vtkImage = vtkImageData.newInstance();
  const [dx, dy, dz] = image.dims;
  vtkImage.setDimensions(dx, dy, dz);
  vtkImage.setSpacing(image.spacing as [number, number, number]);
  vtkImage.setOrigin(image.origin as [number, number, number]);
  // segmorph's three direction rows flatten straight into vtk.js's 3x3.
  vtkImage.setDirection(Float32Array.from(image.direction.flat()));
  vtkImage.getPointData().setScalars(
    vtkDataArray.newInstance({
      numberOfComponents: 1,
      values: image.data,
    }),
  );
  return vtkImage;
}

/** Mesh -> vtkPolyData. Points and polys are already in VTK layout. */
export function toPolyData(mesh: Mesh) {
  const polyData = vtkPolyData.newInstance();
  polyData.getPoints().setData(mesh.points, 3);
  polyData.getPolys().setData(mesh.polys);
  return polyData;
}
