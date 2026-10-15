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
    // vtk.js stores columns consecutively; segmorph stores matrix rows.
    direction: [
      [direction[0], direction[3], direction[6]],
      [direction[1], direction[4], direction[7]],
      [direction[2], direction[5], direction[8]],
    ],
  } as OrientedImage<T>;
}

/** OrientedImage -> vtkImageData, sharing the same underlying typed array. */
export function toVtkImageData(image: OrientedImage) {
  const vtkImage = vtkImageData.newInstance();
  const [dx, dy, dz] = image.dims;
  vtkImage.setDimensions(dx, dy, dz);
  vtkImage.setSpacing(image.spacing as [number, number, number]);
  vtkImage.setOrigin(image.origin as [number, number, number]);
  const [x, y, z] = image.direction;
  vtkImage.setDirection([x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]);
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
