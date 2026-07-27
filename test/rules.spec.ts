import { describe, expect, expectTypeOf, it } from 'vitest';

import type { Mesh } from '../src/geometry/mesh.js';
import type { ImageData, ImageGeometry, OrientedImage } from '../src/image/orientedImage.js';
import {
  findCheapestPath,
  registerConversionRule,
  type ConversionRule,
} from '../src/model/graph.js';
import {
  createDefaultConversionGraph,
  createDefaultRules,
  defaultRepresentations,
} from '../src/model/rules.js';
import { dice } from './diff/image.js';

// Compile-time coverage: the default rules encode each source/target
// representation pairing in its convert signature (representationRule in
// rules.ts). tsconfig includes test/, so a wrong pairing fails npm run
// typecheck, not just vitest --typecheck.
type Rules = ReturnType<typeof createDefaultRules>;
type LabelmapToSurfaceRule = Extract<Rules[number], { source: 'labelmap'; target: 'surface' }>;
type SurfaceToLabelmapRule = Extract<Rules[number], { source: 'surface'; target: 'labelmap' }>;
type ContourToSurfaceRule = Extract<Rules[number], { source: 'contour'; target: 'surface' }>;

expectTypeOf<LabelmapToSurfaceRule['convert']>()
  .parameter(0)
  .toEqualTypeOf<OrientedImage<ImageData>>();
expectTypeOf<LabelmapToSurfaceRule['convert']>().returns.toEqualTypeOf<Mesh>();
expectTypeOf<SurfaceToLabelmapRule['convert']>().parameter(0).toEqualTypeOf<Mesh>();
expectTypeOf<SurfaceToLabelmapRule['convert']>().returns.toEqualTypeOf<OrientedImage<ImageData>>();
expectTypeOf<ContourToSurfaceRule['convert']>().returns.toEqualTypeOf<Mesh>();

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

function applyPath(path: readonly ConversionRule<any, any>[], input: unknown) {
  return path.reduce((value, rule) => rule.convert(value), input);
}

describe('default conversion rules', () => {
  const labelValue = 3;
  const graph = createDefaultConversionGraph({
    labelValue,
    referenceGeometry: geometry,
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

  it('prefers the direct contour-to-surface stitch over a voxelization detour', () => {
    const path = findCheapestPath(
      graph,
      defaultRepresentations.contour,
      defaultRepresentations.surface,
    );
    expect(path).toBeDefined();
    expect(path!.map((rule) => rule.target)).toEqual(['surface']);
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
