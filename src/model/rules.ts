import { contourToSurface } from '../convert/contourToSurface.js';
import { contourToLabelmap } from '../convert/contourToLabelmap.js';
import { labelmapToSurface } from '../convert/labelmapToSurface.js';
import { surfaceToContour } from '../convert/surfaceToContour.js';
import { surfaceToLabelmap } from '../convert/surfaceToLabelmap.js';
import {
  createContourPlane,
  planarContourWorldLoops,
  type PlanarContour,
} from '../geometry/contour.js';
import type { Mesh } from '../geometry/mesh.js';
import { indexToWorld } from '../image/orientedImage.js';
import type {
  ImageData,
  ImageGeometry,
  OrientedImage,
  Vector3,
} from '../image/orientedImage.js';
import { createConversionGraph } from './graph.js';

/**
 * Representation names shared with the model layer (see createSegmentation
 * and the segment representation map in test/model.spec.ts).
 */
export const defaultRepresentations = {
  labelmap: 'labelmap',
  surface: 'surface',
  contour: 'contour',
} as const;

/**
 * Relative cost weights for findCheapestPath. Absolute values carry no unit;
 * only the ratios matter when picking among alternative paths. They are
 * proportional to expected work per conversion:
 * - labelmap -> surface: one marching-cubes table lookup per voxel cell plus
 *   linear output assembly.
 * - surface -> labelmap: per-sample even-odd containment tests against the
 *   triangle set, substantially more work than extraction on the same
 *   geometry, so it is weighted an order of magnitude higher.
 * - surface -> contour: one plane cut per reference slice, comparable to a
 *   single extraction pass.
 * - contour -> labelmap: per-slice scan fill, cheap.
 * - contour -> surface: dynamic-programming stitch between adjacent planes
 *   plus end-cap rasterization; kept below the contour -> labelmap ->
 *   surface total (2 + 1) so RTSTRUCT-style inputs take the direct stitch
 *   instead of a voxelization detour.
 */
const labelmapToSurfaceCost = 1;
const surfaceToLabelmapCost = 10;
const surfaceToContourCost = 2;
const contourToLabelmapCost = 2;
const contourToSurfaceCost = 2;

/**
 * The data type carried by each default representation. Adding a
 * representation (fractional labelmap, ...) means adding its name and type
 * here, after which representationRule below rejects any rule whose convert
 * signature does not match the source/target pairing.
 */
type RepresentationData = {
  labelmap: OrientedImage<ImageData>;
  surface: Mesh;
  contour: readonly PlanarContour[];
};

type DefaultRepresentation = keyof RepresentationData;

/**
 * Build one default rule with its source/target representation pairing checked
 * at compile time: convert must map the source representation's data type to
 * the target's. `input` is inferred from `source`, so a mismatched conversion
 * (e.g. a labelmap -> surface rule whose convert takes a Mesh) fails to
 * typecheck. The returned rule keeps its literal source/target and typed
 * convert (a subtype of the graph's ConversionRule), so callers can see the
 * pairing while the graph still stores it in its opaque rule slot.
 */
function representationRule<S extends DefaultRepresentation, T extends DefaultRepresentation>(
  rule: Readonly<{
    source: S;
    target: T;
    cost: number;
    convert: (input: RepresentationData[S]) => RepresentationData[T];
  }>,
) {
  return rule;
}

/**
 * One cutting plane per reference slice, spanned by the direction matrix's
 * first two columns through the slice origin.
 */
function referenceSlicePlanes(geometry: ImageGeometry) {
  const xAxis: Vector3 = [
    geometry.direction[0][0],
    geometry.direction[1][0],
    geometry.direction[2][0],
  ];
  const yAxis: Vector3 = [
    geometry.direction[0][1],
    geometry.direction[1][1],
    geometry.direction[2][1],
  ];
  return Array.from({ length: geometry.dims[2] }, (_, slice) =>
    createContourPlane(indexToWorld(geometry, [0, 0, slice]), xAxis, yAxis));
}

/**
 * Per-conversion parameters the graph API cannot carry: ConversionRule.convert
 * is fixed at (input) => output by graph.ts, so this factory closes over the
 * parameters and returns rules specialized to them. Build one rule set (or
 * graph) per (labelValue, referenceGeometry) pair.
 */
export type DefaultRuleOptions = Readonly<{
  /** Label written and extracted by every voxel-valued conversion. */
  labelValue: number;
  /** Target sampling grid for every conversion that produces a labelmap, and
   * the slice planes captured by every conversion that produces contours. */
  referenceGeometry: ImageGeometry;
}>;

/**
 * Default rules for the conversions landed so far. Extension list, in the
 * order the algorithms are expected to land (each is a new entry below, with
 * its parameters added to DefaultRuleOptions; smoothing and decimation are
 * post-processing options folded into the labelmap -> surface rule rather
 * than separate graph nodes):
 * - labelmap <-> fractional labelmap (fractional; captures oversampling).
 *
 * The inferred return type is the union of each rule's fully typed
 * ConversionRule, so the source/target/data-type pairings are visible to
 * callers and every rule is assignable to the graph's opaque rule slot.
 */
export function createDefaultRules(options: DefaultRuleOptions) {
  const { labelValue, referenceGeometry } = options;
  return [
    representationRule({
      source: defaultRepresentations.labelmap,
      target: defaultRepresentations.surface,
      cost: labelmapToSurfaceCost,
      convert: (input) => labelmapToSurface(input, { labelValue }),
    }),
    representationRule({
      source: defaultRepresentations.surface,
      target: defaultRepresentations.labelmap,
      cost: surfaceToLabelmapCost,
      convert: (input) => surfaceToLabelmap(input, referenceGeometry, { labelValue }),
    }),
    representationRule({
      source: defaultRepresentations.surface,
      target: defaultRepresentations.contour,
      cost: surfaceToContourCost,
      convert: (input) => referenceSlicePlanes(referenceGeometry)
        .map((plane) => surfaceToContour(input, plane))
        .filter((contour) => contour !== undefined),
    }),
    representationRule({
      source: defaultRepresentations.contour,
      target: defaultRepresentations.labelmap,
      cost: contourToLabelmapCost,
      convert: (input) => contourToLabelmap(input, referenceGeometry, { labelValue }),
    }),
    representationRule({
      source: defaultRepresentations.contour,
      target: defaultRepresentations.surface,
      cost: contourToSurfaceCost,
      convert: (input) => contourToSurface(input.flatMap(planarContourWorldLoops)),
    }),
  ];
}

export function createDefaultConversionGraph(options: DefaultRuleOptions) {
  return createConversionGraph(createDefaultRules(options));
}
