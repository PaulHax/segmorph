# segmorph

segmorph is a pure TypeScript library for polymorphic medical image
segmentations: overlapping, individually identified segments that hold multiple
interchangeable representations (binary labelmap, closed surface, planar
contour) with conversions between them. Algorithms are ported from VTK and
PolySeg, proven correct by differential testing against those canonical
libraries, and the published package ships zero runtime dependencies, no
WebAssembly, and no Python. Data are plain records over typed arrays that map
1:1 onto vtk.js structures without importing vtk.js, and all conversions are
synchronous pure functions.

## Install and develop

```sh
npm install segmorph

# development, from a checkout:
npm install
npm run typecheck    # tsc --noEmit
npm test             # fast tier: unit tests and invariants; no Python or WASM needed
npm run test:oracle  # oracle tier: generates goldens live with the oracles; needs uv
npm run build        # tsup -> dist/
npm run fixtures     # dev only: regenerate the committed golden fixtures
```

## Data contracts

These shapes are frozen. They are deliberately vtk.js-shaped so adapters are
trivial, but nothing in the library imports vtk.js.

### Mesh

```ts
type Mesh = {
  points: Float32Array; // interleaved xyz world coordinates, 3 per vertex
  polys: Uint32Array; // VTK-style cell array: [3, i0, i1, i2, 3, i0, i1, i2, ...]
};
```

`points` holds vertex positions in world (physical) coordinates. `polys` is a
VTK cell array of triangles: each cell is a leading vertex count (always 3)
followed by three indices into `points`. This maps directly onto
`vtkPolyData` points and polys.

### OrientedImage / ImageGeometry

```ts
type ImageGeometry = {
  dims: readonly number[]; // voxel counts [x, y, z], positive integers
  spacing: readonly number[]; // physical size of one voxel step per axis
  origin: readonly number[]; // world position of voxel index [0, 0, 0]
  direction: readonly (readonly number[])[]; // orthonormal 3x3 axis matrix
};

type OrientedImage<T extends ImageData = ImageData> = ImageGeometry & {
  data: T; // typed array, x varies fastest, length = dims[0] * dims[1] * dims[2]
};
```

`direction` rows are the world-space directions of the image axes and must be
orthonormal (oblique orientations are fully supported). `ImageData` is any of
the standard numeric typed arrays. Voxels are stored x-fastest:
`data[x + dims[0] * (y + dims[1] * z)]`.

### PlanarContour

```ts
type PlanarContour = {
  plane: { origin: Vector3; xAxis: Vector3; yAxis: Vector3 };
  loops: readonly { points: Float64Array }[]; // interleaved planar xy per loop
};
```

`plane` is an origin plus an orthonormal 2D basis embedded in world space.
Each loop is a closed polygon in plane coordinates: interleaved xy pairs where
the final vertex connects back to the first.

## Quick start: the labelmap round trip

```ts
import {
  createOrientedImage,
  labelmapToSurface,
  surfaceToLabelmap,
} from "segmorph";

// A tiny labelmap with one foreground voxel labeled 2.
const data = new Uint8Array(27);
data[13] = 2;
const labelmap = createOrientedImage({
  data,
  dims: [3, 3, 3],
  spacing: [1, 1, 1],
  origin: [0, 0, 0],
  direction: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
});

// Binary labelmap -> closed triangle surface (world coordinates).
const surface = labelmapToSurface(labelmap, { labelValue: 2 });

// Closed surface -> binary labelmap on a reference geometry.
const voxelized = surfaceToLabelmap(surface, labelmap, { labelValue: 2 });
```

## Running conversions off the main thread

Every conversion is a synchronous pure function, which is a deliberate choice:
segmorph does not own a scheduler, so the calling app decides where the work
runs. On clinical-size volumes these are not instant. Extraction, voxelization,
and decimation are all superlinear in the data, so calling them directly from an
event handler will drop frames.

Run them in a worker. The data contracts are plain records over typed arrays, so
they cross the worker boundary by structured clone, and the buffers can be
transferred rather than copied. This is the property vtk.js objects do not have.

```ts
// worker.ts
import { labelmapToSurface } from "segmorph";

self.onmessage = ({ data }) => {
  const mesh = labelmapToSurface(data.labelmap, {
    labelValue: data.labelValue,
  });
  self.postMessage(mesh, [mesh.points.buffer, mesh.polys.buffer]);
};
```

`examples/advanced` runs this pattern end to end, including a scheduling policy
that debounces requests and drops stale responses.

## Examples

Both are Vite apps that consume the library through `file:../..`, so build the
package once (`npm run build`) before running them.

- [`examples/basic`](./examples/basic) is the smallest useful thing: a labelmap
  becomes a surface in a vtk.js view, in about forty lines, no worker.
- [`examples/advanced`](./examples/advanced) loads a real chest CT (LIDC2, from
  the vtk.js sample data), seeds two overlapping segments, and shows the same
  segmentation as surfaces inside a volume rendering and as contours cut onto a
  reslice, with smoothing, decimation, and resolution controls, all converted in
  a worker.
- [`examples/dicom`](./examples/dicom) shows the DICOM adapter pattern: RTSTRUCT
  ROIs (including a keyhole annulus) and a multi-frame DICOM SEG are encoded and
  decoded with dcmjs at the app edge, then converted with `contourToSurface` and
  `labelmapToSurface` and rendered together. The core never sees DICOM.

## API

Model functions return new objects instead of mutating their inputs.

### Convert (`src/convert`)

| Export                                                  | Description                                                                                                                                                                                                                                                    |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `labelmapToSurface(image, { labelValue })`              | Extract a closed triangle surface for one label using VTK's table-driven discrete marching cubes cases. A one-sample background border closes foreground that touches the volume edge. Returns a `Mesh` in world coordinates.                                  |
| `surfaceToLabelmap(mesh, geometry, { labelValue })`     | Rasterize a closed triangle surface onto the sample points of a reference `ImageGeometry` using ray-parity (even-odd) containment; samples on the surface are foreground. Returns an `OrientedImage` whose typed array width fits `labelValue`.                |
| `meshSmooth(mesh, options)`                             | Smooth a mesh with a windowed-sinc filter ported from `vtkWindowedSincPolyDataFilter`. Preserves vertex count and topology; returns a new `Mesh`.                                                                                                              |
| `meshDecimate(mesh, options)`                           | Reduce triangle count with quadric-error edge collapse ported from `vtkQuadricDecimation`. `options.targetReduction` in 0..1. Returns a new `Mesh`.                                                                                                            |
| `surfaceToContour(mesh, plane)`                         | Intersect a mesh with a `ContourPlane` (ported `vtkCutter` + contour-loop extraction) and return the closed planar loops as a `PlanarContour`.                                                                                                                 |
| `contourToLabelmap(contours, geometry, { labelValue })` | Rasterize planar contours into a binary labelmap with even-odd fill (nested loops make holes). Returns an `OrientedImage`.                                                                                                                                     |
| `contourToSurface(loops, options)`                      | Stitch a stack of planar contours (world-space loops, the RTSTRUCT shape) into a closed surface, ported from SlicerRT's planar-contour-to-closed-surface rule: keyhole splitting, branching, dynamic-programming stitching, smooth end caps. Returns a `Mesh`. |
| `surfaceToFractionalLabelmap(mesh, geometry, options)`  | Compute fractional (sub-voxel) occupancy of a surface using PolySeg's 216-offset supersampling. Returns an `OrientedImage` of fractional values.                                                                                                               |
| `fractionalLabelmapToSurface(image, options)`           | Extract a surface from a fractional labelmap by threshold, ported from PolySeg. `options.threshold` is the iso occupancy in `(0, 1]` (default `0.5`); zero is rejected. Returns a `Mesh`.                                                                      |
| `fillBetween(image, options)`                           | Interpolate contours between segmented slices, ported from ITK's morphological contour interpolation. Returns a new `OrientedImage`.                                                                                                                           |
| `surfaceNets(image, options)`                           | Extract a dual-grid surface net from a labelmap, ported from `vtkSurfaceNets3D`. Returns a `Mesh`.                                                                                                                                                             |

### Image (`src/image`)

| Export                                                                 | Description                                                                                                                                                                |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `createOrientedImage(image)`                                           | Validate an `OrientedImage` (geometry plus data length) and return it.                                                                                                     |
| `validateImageGeometry(geometry)`                                      | Throw unless dims, spacing, origin, and the orthonormal direction matrix are well formed.                                                                                  |
| `indexToWorld(geometry, index)`                                        | Map a continuous voxel index to world coordinates.                                                                                                                         |
| `worldToIndex(geometry, world)`                                        | Map a world position to a continuous voxel index.                                                                                                                          |
| `resampleNearest(image, reference, options?)`                          | Resample an image onto a reference geometry with nearest-neighbor sampling, following VTK `vtkImageReslice` behavior. `options.fillValue` fills samples outside the input. |
| `compositeImage(input, modifier, dims, operation, extent?)`            | Combine a modifier array into an input array voxel-wise with a `'set'`, `'minimum'`, or `'maximum'` operation, optionally bounded to an extent. Returns a new array.       |
| `compositeSet(input, modifier, dims, extent?)`                         | `compositeImage` with the `'set'` operation.                                                                                                                               |
| `compositeMin(input, modifier, dims, extent?)`                         | `compositeImage` with the `'minimum'` operation.                                                                                                                           |
| `compositeMax(input, modifier, dims, extent?)`                         | `compositeImage` with the `'maximum'` operation.                                                                                                                           |
| `maskByLabelValue(input, mask, dims, labelValue, fillValue?, extent?)` | Keep input voxels where the mask equals `labelValue`; write `fillValue` (default 0) elsewhere.                                                                             |
| `EMPTY_EXTENT`                                                         | The canonical empty extent constant.                                                                                                                                       |
| `extentFromDims(dims)`                                                 | Full-image extent for the given dimensions.                                                                                                                                |
| `validateExtent(extent, dims)`                                         | Throw unless the extent is well formed and inside the dimensions.                                                                                                          |
| `intersectExtents(left, right)`                                        | Intersection of two extents, or `EMPTY_EXTENT` when disjoint.                                                                                                              |
| `isExtentEmpty(extent)`                                                | True when the extent contains no voxels.                                                                                                                                   |
| `isIndexInExtent(index, extent)`                                       | True when a voxel index lies inside the extent.                                                                                                                            |
| `flatIndex(index, dims)`                                               | Flat data offset of an [x, y, z] index (x varies fastest).                                                                                                                 |
| `iterateExtentIndices(extent)`                                         | Generator over every [x, y, z] index in the extent.                                                                                                                        |

Extents are inclusive bounds in `[xMin, xMax, yMin, yMax, zMin, zMax]` order.

### Geometry (`src/geometry`)

| Export                                     | Description                                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `createMesh(points, triangles)`            | Build a validated `Mesh` from point tuples and triangle index tuples.                             |
| `validateMesh(mesh)`                       | Throw unless points and the VTK cell array are well formed.                                       |
| `vertexCount(mesh)`                        | Number of vertices.                                                                               |
| `triangleCount(mesh)`                      | Number of triangle cells.                                                                         |
| `getPoint(mesh, index)`                    | The xyz tuple of one vertex.                                                                      |
| `iteratePoints(mesh)`                      | Generator over vertex tuples.                                                                     |
| `iterateTriangles(mesh)`                   | Generator over triangle index tuples.                                                             |
| `createContourPlane(origin, xAxis, yAxis)` | Build a validated contour plane with an orthonormal basis.                                        |
| `validateContourPlane(plane)`              | Throw unless the plane basis is orthonormal and finite.                                           |
| `createContourLoop(points)`                | Build a validated closed loop from planar xy tuples (at least three).                             |
| `validateContourLoop(loop)`                | Throw unless the loop has complete, finite xy pairs and at least three vertices.                  |
| `createPlanarContour(plane, loops)`        | Build a validated `PlanarContour` from a plane and one or more loops.                             |
| `validatePlanarContour(contour)`           | Throw unless the plane and every loop validate.                                                   |
| `planarContourWorldLoops(contour)`         | Flatten a planar contour's loops to world-space xyz polylines (`contourToSurface`'s input shape). |
| `planeToWorld(plane, point)`               | Map a planar xy point to world coordinates.                                                       |
| `worldToPlane(plane, point)`               | Project a world point into plane xy coordinates.                                                  |

### Model (`src/model`)

| Export                                                              | Description                                                                                                                                                     |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `createSegment(options)`                                            | Create a segment record: id, name, color, optional locked flag and initial representations map.                                                                 |
| `getRepresentation(segment, name)`                                  | The named representation, or undefined.                                                                                                                         |
| `setRepresentation(segment, name, representation)`                  | New segment with the named representation set.                                                                                                                  |
| `createSegmentation(sourceRepresentation)`                          | Empty segmentation whose edits happen in the named source representation.                                                                                       |
| `addSegment(segmentation, segment)`                                 | New segmentation with the segment appended.                                                                                                                     |
| `removeSegment(segmentation, id)`                                   | New segmentation without the segment.                                                                                                                           |
| `renameSegment(segmentation, id, name)`                             | New segmentation with the segment renamed.                                                                                                                      |
| `reorderSegments(segmentation, order)`                              | New segmentation with segments in the given id order.                                                                                                           |
| `createLayerModel()`                                                | Empty `{layer, labelValue}` overlap model for shared labelmaps.                                                                                                 |
| `assignSegmentToLayer(model, segmentId, labelValue, mask)`          | Assign a segment mask to the lowest layer where it does not collide with another segment's foreground.                                                          |
| `compactLayers(model)`                                              | Reassign every segment to remove gaps left by removed or shrunken segments.                                                                                     |
| `createConversionGraph(rules?)`                                     | Conversion graph from an optional initial rule list.                                                                                                            |
| `registerConversionRule(graph, rule)`                               | New graph with the rule added. A rule is a data record: `{ source, target, cost, convert }`.                                                                    |
| `findCheapestPath(graph, source, target)`                           | Cheapest rule sequence from source to target representation, `[]` when source equals target, undefined when unreachable.                                        |
| `createDefaultRules(options)`                                       | Build the default conversion-rule set (labelmap and surface) as data records, capturing `{ labelValue, referenceGeometry }` for the conversions that need them. |
| `createDefaultConversionGraph(options)`                             | A conversion graph pre-populated with `createDefaultRules`.                                                                                                     |
| `defaultRepresentations`                                            | The canonical representation names (`labelmap`, `surface`) used by the default rules.                                                                           |
| `editRepresentation(segmentation, segmentId, name, representation)` | Write the source representation of a segment and drop its derived representations. Editing a non-source representation throws.                                  |
| `getOrCreateRepresentation(segmentation, segmentId, target, graph)` | Return the target representation, converting from the source along the cheapest path and caching intermediate results when absent.                              |
| `promoteRepresentationToSource(segmentation, name)`                 | Make an existing representation the source for future edits; every segment must already carry it.                                                               |

### IO (`src/io`)

| Export            | Description                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `readNrrd(bytes)` | Parse a NRRD file from bytes into `{ dims, spacing, origin, direction, data }`, ready to use as an `OrientedImage`. |

## Testing philosophy

Every ported algorithm is verified by differential testing against the
canonical implementation: dev-only oracles (Python `vtk`, `@icr/polyseg-wasm`,
vtk.js) produce golden outputs, and vitest specs compare the TypeScript port
to the goldens with geometric tolerances (Dice, Hausdorff, surface distance)
rather than bit equality, plus oracle-free round-trip and structural
invariants. The suite runs in two tiers: the default `npm test` needs neither
Python nor WASM, while `npm run test:oracle` regenerates goldens live with the
pinned oracle environment (its own CI job), so migrated algorithms can never
silently drift from the oracles.

## License

MIT, see [LICENSE](./LICENSE). Parts of the source are TypeScript ports of
BSD-licensed VTK and PolySeg algorithms and remain under those licenses;
upstream notices and license texts are collected in
[THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md), which ships inside the
package. 3D Slicer is used as a behavior reference only, and no code is ported
from it.
