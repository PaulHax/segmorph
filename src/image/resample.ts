import {
  createOrientedImage,
  indexToWorld,
  validateImageGeometry,
  worldToIndex,
} from './orientedImage.js';
import type { ImageData, ImageGeometry, OrientedImage } from './orientedImage.js';

type ImageDataConstructor<T extends ImageData> = {
  new (length: number): T;
};

// vtkInterpolationMath::Round biases the half-integer rounding boundary so a
// continuous index within this distance below a half-integer rounds UP rather
// than DOWN. Matches VTK_INTERPOLATE_FLOOR_TOL in
// reference-repos/vtk/Imaging/Core/vtkInterpolationMath.h.
const FLOOR_TOL = 7.62939453125e-6;

// vtkImageReslice default BorderThickness (Border=on) is the interpolator
// tolerance used to accept sub-voxel-edge samples before clamping. See
// vtkImageReslice.cxx (this->BorderThickness = 0.5).
const BORDER_THICKNESS = 0.5;

// vtkInterpolationMath::Round(x) == floor(x + 0.5 + FLOOR_TOL).
function nearestIndex(value: number) {
  return Math.floor(value + 0.5 + FLOOR_TOL);
}

function clampIndex(index: number, max: number) {
  if (index < 0) return 0;
  if (index > max) return max;
  return index;
}

function flatIndex(dims: readonly number[], x: number, y: number, z: number) {
  return x + dims[0] * (y + dims[1] * z);
}

export type NearestResampleOptions = {
  // Background value written where the sampled input index falls outside the
  // accepted region. Typed arrays default to 0 when this is omitted.
  fillValue?: number;
  // Out-of-bounds policy at the sub-voxel edge.
  //   'clamp' (default) matches vtkImageReslice with Border=on: a continuous
  //     input index within BORDER_THICKNESS (0.5) of the extent is accepted and
  //     the rounded index is clamped to the edge voxel; only indices past the
  //     tolerance receive fillValue.
  //   'fill' rejects any index that rounds strictly outside the input extent
  //     and writes fillValue (the pre-parity behavior).
  border?: 'clamp' | 'fill';
};

export function resampleNearest<T extends ImageData>(
  input: OrientedImage<T>,
  reference: ImageGeometry,
  options: NearestResampleOptions = {},
) {
  createOrientedImage(input);
  validateImageGeometry(reference);

  const length = reference.dims[0] * reference.dims[1] * reference.dims[2];
  const DataConstructor = input.data.constructor as ImageDataConstructor<T>;
  const data = new DataConstructor(length);
  if (options.fillValue !== undefined) data.fill(options.fillValue);

  const clamp = options.border !== 'fill';
  // vtkAbstractImageInterpolator raises the tolerance to 0.5 + FLOOR_TOL on
  // single-slice axes (extent min == max); other axes use BORDER_THICKNESS.
  const tol = [0, 1, 2].map((axis) =>
    input.dims[axis] === 1 ? BORDER_THICKNESS + FLOOR_TOL : BORDER_THICKNESS,
  );

  for (let z = 0; z < reference.dims[2]; z += 1) {
    for (let y = 0; y < reference.dims[1]; y += 1) {
      for (let x = 0; x < reference.dims[0]; x += 1) {
        const p = worldToIndex(input, indexToWorld(reference, [x, y, z]));

        let inputX: number;
        let inputY: number;
        let inputZ: number;
        if (clamp) {
          // Accept indices within tolerance of the extent, then clamp the
          // rounded index to the edge voxel (VTK Border=on clamp semantics).
          if (
            p[0] < -tol[0] ||
            p[0] > input.dims[0] - 1 + tol[0] ||
            p[1] < -tol[1] ||
            p[1] > input.dims[1] - 1 + tol[1] ||
            p[2] < -tol[2] ||
            p[2] > input.dims[2] - 1 + tol[2]
          )
            continue;
          inputX = clampIndex(nearestIndex(p[0]), input.dims[0] - 1);
          inputY = clampIndex(nearestIndex(p[1]), input.dims[1] - 1);
          inputZ = clampIndex(nearestIndex(p[2]), input.dims[2] - 1);
        } else {
          inputX = nearestIndex(p[0]);
          inputY = nearestIndex(p[1]);
          inputZ = nearestIndex(p[2]);
          if (
            inputX < 0 ||
            inputX >= input.dims[0] ||
            inputY < 0 ||
            inputY >= input.dims[1] ||
            inputZ < 0 ||
            inputZ >= input.dims[2]
          )
            continue;
        }

        data[flatIndex(reference.dims, x, y, z)] =
          input.data[flatIndex(input.dims, inputX, inputY, inputZ)];
      }
    }
  }

  return createOrientedImage({ ...reference, data });
}
