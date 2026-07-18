import type { PlanarContour, Vector3 } from '../../../src/geometry/contour.js';

type RasterizeFixtureInput = {
  labelValue: number;
  contours: PlanarContour[];
};

function isVector3(value: unknown): value is Vector3 {
  return Array.isArray(value) && value.length === 3
    && value.every((component) => typeof component === 'number' && Number.isFinite(component));
}

function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value)
    && value.every((component) => typeof component === 'number' && Number.isFinite(component));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function readContoursJson(json: string): RasterizeFixtureInput {
  const value: unknown = JSON.parse(json);
  if (!isRecord(value) || typeof value.labelValue !== 'number'
    || !Number.isInteger(value.labelValue) || !Array.isArray(value.contours)) {
    throw new Error('Invalid rasterize fixture input');
  }

  const contours = value.contours.map((contour): PlanarContour => {
    if (!isRecord(contour) || !isRecord(contour.plane) || !Array.isArray(contour.loops)
      || !isVector3(contour.plane.origin)
      || !isVector3(contour.plane.xAxis)
      || !isVector3(contour.plane.yAxis)
      || !contour.loops.every(isNumberArray)) {
      throw new Error('Invalid rasterize fixture contour');
    }
    return {
      plane: {
        origin: contour.plane.origin,
        xAxis: contour.plane.xAxis,
        yAxis: contour.plane.yAxis,
      },
      loops: contour.loops.map((points) => ({ points: Float64Array.from(points) })),
    };
  });

  return { labelValue: value.labelValue, contours };
}
