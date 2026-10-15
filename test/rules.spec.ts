import { describe, expect, expectTypeOf, it } from 'vitest';

import {
  addSegment,
  createContourLoop,
  createContourPlane,
  createPlanarContour,
  createSegment,
  contourToLabelmap,
  indexToWorld,
  surfaceToLabelmap,
  worldToIndex,
  createSegmentation,
  getOrCreateRepresentation,
} from '../src/index.js';
import type { PlanarContour } from '../src/geometry/contour.js';
import type { Mesh } from '../src/geometry/mesh.js';
import type { ImageData, ImageGeometry, OrientedImage } from '../src/image/orientedImage.js';
import {
  applyConversionRule,
  findCheapestPath,
  registerConversionRule,
  type StoredConversionRule,
} from '../src/model/graph.js';
import {
  createDefaultConversionGraph,
  createDefaultRules,
  defaultRepresentations,
} from '../src/model/rules.js';
import { dice } from './diff/image.js';
import { cubeMesh } from './surfaceToLabelmapCases.js';

// Compile-time coverage: the default rules encode each source/target
// representation pairing in its convert signature (representationRule in
// rules.ts). tsconfig includes test/, so a wrong pairing fails npm run
// typecheck, not just vitest --typecheck.
type Rules = ReturnType<typeof createDefaultRules>;
type LabelmapToSurfaceRule = Extract<Rules[number], { source: 'labelmap'; target: 'surface' }>;
type SurfaceToLabelmapRule = Extract<Rules[number], { source: 'surface'; target: 'labelmap' }>;
type ContourToLabelmapRule = Extract<Rules[number], { source: 'contour'; target: 'labelmap' }>;

expectTypeOf<LabelmapToSurfaceRule['convert']>()
  .parameter(0)
  .toEqualTypeOf<OrientedImage<ImageData>>();
expectTypeOf<LabelmapToSurfaceRule['convert']>().returns.toEqualTypeOf<Mesh>();
expectTypeOf<SurfaceToLabelmapRule['convert']>().parameter(0).toEqualTypeOf<Mesh>();
expectTypeOf<SurfaceToLabelmapRule['convert']>().returns.toEqualTypeOf<OrientedImage<ImageData>>();
expectTypeOf<ContourToLabelmapRule['convert']>()
  .parameter(0)
  .toEqualTypeOf<readonly PlanarContour[]>();
expectTypeOf<ContourToLabelmapRule['convert']>().returns.toEqualTypeOf<OrientedImage<ImageData>>();

const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
] as const;

const geometry: ImageGeometry = {
  dims: [11, 11, 11],
  spacing: [0.9, 1.3, 1.1],
  origin: [-5, 2, 7],
  direction: identity,
};

const contourGeometries = [
  { name: 'axial', direction: identity },
  {
    name: 'sagittal',
    direction: [
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
    ],
  },
  {
    name: 'oblique',
    direction: [
      [1, 0, 0],
      [0, Math.cos(0.4), -Math.sin(0.4)],
      [0, Math.sin(0.4), Math.cos(0.4)],
    ],
  },
  {
    name: 'reflected',
    direction: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, -1],
    ],
  },
].map(({ name, direction }) => ({
  name,
  referenceGeometry: { ...geometry, dims: [9, 9, 3], spacing: [1, 1, 4], direction },
}));

function square(low: number, high: number) {
  return createContourLoop([
    [low, low],
    [high, low],
    [high, high],
    [low, high],
  ]);
}

function referenceContour(
  referenceGeometry: ImageGeometry,
  slice: number,
  loops: PlanarContour['loops'],
) {
  const { direction } = referenceGeometry;
  return createPlanarContour(
    createContourPlane(
      indexToWorld(referenceGeometry, [0, 0, slice]),
      [direction[0][0], direction[1][0], direction[2][0]],
      [direction[0][1], direction[1][1], direction[2][1]],
    ),
    loops,
  );
}

function deriveContourSurface(
  contours: readonly PlanarContour[],
  referenceGeometry: ImageGeometry,
) {
  const source = addSegment(
    createSegmentation('contour'),
    createSegment({
      id: 'contours',
      name: 'Contours',
      color: [1, 0, 0],
      representations: { contour: contours },
    }),
  );
  return getOrCreateRepresentation(
    source,
    'contours',
    'surface',
    createDefaultConversionGraph({ labelValue: 256, referenceGeometry }),
  );
}

function sphereLabelmap(labelValue: number) {
  const [width, height, depth] = geometry.dims;
  const data = new Uint8Array(width * height * depth);
  const center = [(width - 1) / 2, (height - 1) / 2, (depth - 1) / 2];
  const radius = 3.4;
  for (let z = 0; z < depth; z += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const distance = Math.hypot(x - center[0], y - center[1], z - center[2]);
        if (distance <= radius) {
          data[x + width * (y + height * z)] = labelValue;
        }
      }
    }
  }
  return { ...geometry, data };
}

function applyPath(path: readonly StoredConversionRule[], input: unknown) {
  return path.reduce((value, rule) => applyConversionRule(rule, value), input);
}

describe('default conversion rules', () => {
  const labelValue = 3;
  const graph = createDefaultConversionGraph({
    labelValue,
    referenceGeometry: geometry,
  });

  it('includes every sample on a closed surface when deriving a labelmap', () => {
    const referenceGeometry = { ...geometry, dims: [3, 3, 3] as const };
    const segmentation = addSegment(
      createSegmentation(defaultRepresentations.surface),
      createSegment({
        id: 'cube',
        name: 'Cube',
        color: [1, 0, 0],
        representations: { surface: cubeMesh(referenceGeometry, 0, 2) },
      }),
    );
    const result = getOrCreateRepresentation(
      segmentation,
      'cube',
      defaultRepresentations.labelmap,
      createDefaultConversionGraph({ labelValue, referenceGeometry }),
    );

    expect((result.representation as OrientedImage<ImageData>).data).toEqual(
      new Uint8Array(27).fill(labelValue),
    );
  });

  it('round-trips a sphere labelmap through the surface representation', () => {
    const input = sphereLabelmap(labelValue);

    const toSurface = findCheapestPath(
      graph,
      defaultRepresentations.labelmap,
      defaultRepresentations.surface,
    );
    const toLabelmap = findCheapestPath(
      graph,
      defaultRepresentations.surface,
      defaultRepresentations.labelmap,
    );
    expect(toSurface).toBeDefined();
    expect(toLabelmap).toBeDefined();

    const surface = applyPath(toSurface!, input);
    const output = applyPath(toLabelmap!, surface) as ReturnType<typeof sphereLabelmap>;

    expect(output.dims).toEqual(geometry.dims);
    expect(output.spacing).toEqual(geometry.spacing);
    expect(output.origin).toEqual(geometry.origin);
    expect(output.direction).toEqual(geometry.direction);
    expect(
      dice(output.data, input.data, geometry.dims as [number, number, number]),
    ).toBeGreaterThanOrEqual(0.95);
  });

  it('reaches the contour representation from a labelmap', () => {
    const toContour = findCheapestPath(
      graph,
      defaultRepresentations.labelmap,
      defaultRepresentations.contour,
    );
    expect(toContour).toBeDefined();
    expect(toContour!.map((rule) => rule.target)).toEqual(['surface', 'contour']);

    const contours = applyPath(toContour!, sphereLabelmap(labelValue)) as unknown[];
    expect(contours.length).toBeGreaterThan(0);
  });

  it('derives contour surfaces through the reference labelmap', () => {
    const path = findCheapestPath(
      graph,
      defaultRepresentations.contour,
      defaultRepresentations.surface,
    );
    expect(path).toBeDefined();
    expect(path!.map((rule) => rule.target)).toEqual(['labelmap', 'surface']);
  });

  it.each(contourGeometries)(
    'preserves contour unions and grouped holes on $name reference slices',
    ({ referenceGeometry }) => {
      const outer = square(1, 7);
      const inner = square(3, 5);
      const separate = [1, 2].flatMap((slice) => [
        referenceContour(referenceGeometry, slice, [outer]),
        referenceContour(referenceGeometry, slice, [inner]),
      ]);
      const grouped = [1, 2].map((slice) =>
        referenceContour(referenceGeometry, slice, [outer, inner]),
      );

      for (const contours of [separate, grouped]) {
        const result = deriveContourSurface(contours, referenceGeometry);
        const expected = contourToLabelmap(contours, referenceGeometry, { labelValue: 256 });
        const actual = surfaceToLabelmap(result.representation as Mesh, referenceGeometry, {
          labelValue: 256,
        });

        expect(actual.data).toEqual(expected.data);
        expect(result.segmentation.segments.contours.representations.labelmap).toEqual(expected);
        expect(result.segmentation.segments.contours.representations.contour).toBe(contours);
      }
    },
  );

  it.each(contourGeometries)(
    'gives a single contour one reference slice of thickness on $name',
    ({ referenceGeometry }) => {
      const contours = [referenceContour(referenceGeometry, 1, [square(1, 7)])];
      const mesh = deriveContourSurface(contours, referenceGeometry).representation as Mesh;
      const sliceCoordinates = [];
      for (let offset = 0; offset < mesh.points.length; offset += 3) {
        sliceCoordinates.push(
          worldToIndex(referenceGeometry, [
            mesh.points[offset],
            mesh.points[offset + 1],
            mesh.points[offset + 2],
          ])[2],
        );
      }

      expect(Math.min(...sliceCoordinates)).toBeCloseTo(0.5);
      expect(Math.max(...sliceCoordinates)).toBeCloseTo(1.5);
    },
  );

  it('derives an empty surface from an empty contour representation', () => {
    const { referenceGeometry } = contourGeometries[0];
    const result = deriveContourSurface([], referenceGeometry);
    expect(result.representation).toEqual({
      points: new Float32Array(0),
      polys: new Uint32Array(0),
    });
    const cached = result.segmentation.segments.contours.representations.labelmap as OrientedImage;
    expect(cached.data).toEqual(new Uint16Array(9 * 9 * 3));
  });

  it('leaves omitted slices empty in the default contour reconstruction', () => {
    const { referenceGeometry } = contourGeometries[0];
    const contours = [0, 2].map((slice) =>
      referenceContour(referenceGeometry, slice, [square(1, 7)]),
    );
    const mesh = deriveContourSurface(contours, referenceGeometry).representation as Mesh;
    const output = surfaceToLabelmap(mesh, referenceGeometry, { labelValue: 256 });
    expect(output.data.slice(9 * 9, 2 * 9 * 9)).toEqual(new Uint16Array(9 * 9));
    expect(output.data.some((value) => value === 256)).toBe(true);
  });

  it('round-trips a sphere labelmap through the contour representation', () => {
    const input = sphereLabelmap(labelValue);
    const toContour = findCheapestPath(
      graph,
      defaultRepresentations.labelmap,
      defaultRepresentations.contour,
    );
    const toLabelmap = findCheapestPath(
      graph,
      defaultRepresentations.contour,
      defaultRepresentations.labelmap,
    );
    expect(toContour).toBeDefined();
    expect(toLabelmap).toBeDefined();

    const output = applyPath(toLabelmap!, applyPath(toContour!, input)) as ReturnType<
      typeof sphereLabelmap
    >;
    expect(
      dice(output.data, input.data, geometry.dims as [number, number, number]),
    ).toBeGreaterThanOrEqual(0.9);
  });

  it('returns undefined for an unreachable representation', () => {
    expect(findCheapestPath(graph, defaultRepresentations.labelmap, 'fractional')).toBeUndefined();
  });

  it('registers additional rules immutably', () => {
    const ruleCount = graph.rules.length;
    const extended = registerConversionRule(graph, {
      source: defaultRepresentations.surface,
      target: 'fractional',
      cost: 1,
      convert: (input) => input,
    });

    expect(extended).not.toBe(graph);
    expect(extended.rules).toHaveLength(ruleCount + 1);
    expect(graph.rules).toHaveLength(ruleCount);
  });
});
