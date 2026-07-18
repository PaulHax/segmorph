import { validatePlanarContour, type PlanarContour } from '../geometry/contour.js';
import { createLabelData, validateLabelValue } from '../image/labelData.js';
import {
  createOrientedImage,
  validateImageGeometry,
  worldToIndex,
  type ImageGeometry,
  type OrientedImage,
} from '../image/orientedImage.js';

/**
 * Tolerance, in slice-index units (fractions of spacing[2]), for accepting a
 * contour plane as coincident with an image slice and parallel to it.
 * Upstream producers (the surface cutter) emit planes exactly on slice
 * planes; measured deviations are at float64 rounding (~1e-15), so 1e-6
 * accepts all legitimate inputs while still rejecting planes that sit a
 * meaningful fraction of a voxel away.
 */
const sliceAlignmentTolerance = 1e-6;

function dot(left: readonly number[], right: readonly number[]) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

/**
 * Map a contour onto its slice index, or undefined when the (aligned) plane
 * lies outside the image extent. Throws RangeError when the plane is not
 * parallel to the image slices or does not coincide with any slice plane
 * within sliceAlignmentTolerance.
 */
function matchSlice(contour: PlanarContour, geometry: ImageGeometry, contourIndex: number) {
  const sliceAxis = [
    geometry.direction[0][2],
    geometry.direction[1][2],
    geometry.direction[2][2],
  ];
  if (Math.abs(dot(contour.plane.xAxis, sliceAxis)) > sliceAlignmentTolerance
    || Math.abs(dot(contour.plane.yAxis, sliceAxis)) > sliceAlignmentTolerance) {
    throw new RangeError(
      `Contour ${contourIndex} plane is not parallel to the image slice planes`,
    );
  }

  const offset = [
    contour.plane.origin[0] - geometry.origin[0],
    contour.plane.origin[1] - geometry.origin[1],
    contour.plane.origin[2] - geometry.origin[2],
  ];
  const continuous = dot(offset, sliceAxis) / geometry.spacing[2];
  const slice = Math.round(continuous);
  if (Math.abs(continuous - slice) > sliceAlignmentTolerance) {
    throw new RangeError(
      `Contour ${contourIndex} plane does not coincide with an image slice plane`,
    );
  }
  return slice >= 0 && slice < geometry.dims[2] ? slice : undefined;
}

/** Contour loops projected to continuous (i, j) index coordinates. */
function projectLoops(contour: PlanarContour, geometry: ImageGeometry) {
  const { origin, xAxis, yAxis } = contour.plane;
  return contour.loops.map((loop) => {
    const projected = new Float64Array(loop.points.length);
    for (let vertex = 0; vertex < loop.points.length; vertex += 2) {
      const u = loop.points[vertex];
      const v = loop.points[vertex + 1];
      const index = worldToIndex(geometry, [
        origin[0] + u * xAxis[0] + v * yAxis[0],
        origin[1] + u * xAxis[1] + v * yAxis[1],
        origin[2] + u * xAxis[2] + v * yAxis[2],
      ]);
      projected[vertex] = index[0];
      projected[vertex + 1] = index[1];
    }
    return projected;
  });
}

/**
 * Even-odd scanline fill of one contour's loops onto one slice, following
 * vtkImageStencilRaster's convention with zero tolerance: voxel centers sit
 * at integer index coordinates, an edge contributes a crossing to row j when
 * edgeMinY < j <= edgeMaxY, and a crossing span (x1, x2] fills columns
 * floor(x1) + 1 through floor(x2). Half-open on the minimum side, so
 * polygons sharing an edge never double-fill and never leave a gap. (VTK
 * itself additionally dilates spans by VTK_STENCIL_TOL = 7.62939453125e-06,
 * which only differs when a crossing lies within that distance of a voxel
 * center; see test/oracle-rasterize.spec.ts for the measured comparison.)
 */
function fillSlice(
  loops: readonly Float64Array[],
  slice: number,
  geometry: ImageGeometry,
  data: Uint8Array | Uint16Array | Uint32Array,
  labelValue: number,
) {
  const [nx, ny] = geometry.dims;
  const sliceOffset = nx * ny * slice;

  let minY = Infinity;
  let maxY = -Infinity;
  for (const loop of loops) {
    for (let vertex = 1; vertex < loop.length; vertex += 2) {
      minY = Math.min(minY, loop[vertex]);
      maxY = Math.max(maxY, loop[vertex]);
    }
  }
  const rowStart = Math.max(0, Math.floor(minY) + 1);
  const rowEnd = Math.min(ny - 1, Math.floor(maxY));

  const crossings: number[] = [];
  for (let row = rowStart; row <= rowEnd; row += 1) {
    crossings.length = 0;
    for (const loop of loops) {
      const vertexCount = loop.length / 2;
      for (let vertex = 0; vertex < vertexCount; vertex += 1) {
        const next = (vertex + 1) % vertexCount;
        const x0 = loop[2 * vertex];
        const y0 = loop[2 * vertex + 1];
        const x1 = loop[2 * next];
        const y1 = loop[2 * next + 1];
        if (y0 === y1) continue;
        const lower = Math.min(y0, y1);
        const upper = Math.max(y0, y1);
        if (!(lower < row && row <= upper)) continue;
        const crossing = x0 + ((row - y0) * (x1 - x0)) / (y1 - y0);
        crossings.push(Math.min(Math.max(crossing, Math.min(x0, x1)), Math.max(x0, x1)));
      }
    }
    crossings.sort((left, right) => left - right);

    const pairCount = crossings.length - (crossings.length % 2);
    const rowOffset = sliceOffset + nx * row;
    for (let pair = 0; pair < pairCount; pair += 2) {
      const first = Math.max(0, Math.floor(crossings[pair]) + 1);
      const last = Math.min(nx - 1, Math.floor(crossings[pair + 1]));
      for (let column = first; column <= last; column += 1) {
        data[rowOffset + column] = labelValue;
      }
    }
  }
}

/**
 * Rasterize closed planar contours into a fresh binary labelmap with the
 * requested geometry. Every contour plane must be parallel to and coincide
 * with an image slice plane within sliceAlignmentTolerance (RangeError
 * otherwise); aligned planes outside the image extent are clipped away like
 * any other out-of-bounds geometry. Loops within one contour compose by the
 * even-odd rule (nested loops cut holes); separate contours compose by
 * union. Inside tests happen at voxel centers with a half-open
 * (min-exclusive, max-inclusive) boundary convention.
 */
export function contourToLabelmap(
  contours: readonly PlanarContour[],
  geometry: ImageGeometry,
  options: { labelValue: number },
): OrientedImage<Uint8Array | Uint16Array | Uint32Array> {
  validateImageGeometry(geometry);
  validateLabelValue(options.labelValue);
  contours.forEach(validatePlanarContour);

  const data = createLabelData(
    geometry.dims[0] * geometry.dims[1] * geometry.dims[2],
    options.labelValue,
  );
  contours.forEach((contour, contourIndex) => {
    const slice = matchSlice(contour, geometry, contourIndex);
    if (slice === undefined) return;
    fillSlice(projectLoops(contour, geometry), slice, geometry, data, options.labelValue);
  });

  return createOrientedImage({ ...geometry, data });
}
