# segmorph advanced example

A real chest CT, two overlapping segments, and the same segmentation shown
three ways at once: surfaces in a 3D view, contours on a reslice, and the
labelmaps they were extracted from.

```sh
# from the repo root, once
npm install && npm run build

# then here
npm install
npm run dev
```

The CT is fetched at runtime from the vtk.js sample data server, so nothing
large is committed here. First load needs network access.

## The data

LIDC2, a chest CT from the Lung Image Database Consortium, published as vtk.js
sample data. 256 x 256 x 133 voxels with anisotropic spacing
(1.40625, 1.40625, 2.5 mm), which is the kind of geometry the conversions are
differentially tested against.

Two segments are seeded by thresholding: `Body` and `Bone`. A real app gets its
segments from a paint tool or a model, but thresholding gives the demo
something anatomically real without shipping a segmentation file.

The two segments **overlap**: every bone voxel is also a body voxel. A single
fused labelmap cannot represent that, since one voxel holds one value. Each
segment carries its own mask and label value instead, which is the point of the
segmentation model.

## What each part demonstrates

| File | segmorph API | What it shows |
|---|---|---|
| `src/adapter.ts` | data contracts | vtkImageData to `OrientedImage`, and `Mesh` to `vtkPolyData`. Voxel and vertex arrays pass by reference; the only reshape is the 9-element direction matrix into three rows. |
| `src/worker.ts` | `labelmapToSurface`, `meshSmooth`, `meshDecimate`, `resampleNearest` | The whole conversion pipeline, off the main thread. |
| `src/sliceView.ts` | `surfaceToContour`, `createContourPlane`, `planeToWorld` | Cuts the 3D surfaces with the current slice plane and draws the loops over the reslice. |
| `src/volumeView.ts` | (rendering only) | vtk.js volume rendering with the segment surfaces inside it. Both live in the same world coordinates, because segmorph works in physical space, so there is no registration step. |
| `src/scheduler.ts` | (policy) | Worker ownership, debouncing, and latest-wins request handling. |
| `src/main.ts` | `createSegmentation`, `createSegment`, `addSegment` | The segment list drives the UI, rather than a hardcoded array. |

The contours on the slice view are worth dwelling on. They are not a second
stored segmentation. They are the same meshes shown in 3D, cut on demand by the
current plane. That is the polymorphic idea on one screen: one edited
representation, several views of it, converted when a view needs them.

## The controls

- **resolution**: converts at full, half, or quarter voxel stride, via
  `resampleNearest`. Full resolution on this volume is a real workload, a few
  million triangles, and takes seconds. Half is the interactive default.
- **smooth**: windowed-sinc iterations. Relaxes the voxel staircase that
  marching cubes leaves, without changing the vertex count.
- **decimate**: quadric edge-collapse target. Removes the vertices the
  flattened regions no longer need.
- **slice**: moves the reslice plane; the contours are recut to follow it.
- **volume** and the per-segment checkboxes toggle visibility.

The status line reports triangles before and after processing, how long the
worker took, and how many contour loops the current slice cut.

## Scheduling, and why the API is synchronous

segmorph ships no scheduler. Every conversion is a pure synchronous function,
so the app decides when and where the work runs. `src/scheduler.ts` is the
policy this example recommends:

- **Web Worker.** Extraction, smoothing, and decimation are CPU-bound over the
  whole volume. The volume is transferred to the worker once at init, and
  later requests carry only parameters.
- **Debounced.** Dragging a slider fires a stream of events; converting on each
  one wastes work on states nobody sees.
- **Latest wins.** If a conversion is in flight when another is requested, the
  new one waits and stale responses are dropped, so the rendered surface never
  goes backwards in time.

Plain records over typed arrays are what make this cheap: they cross the worker
boundary by structured clone, and the buffers transfer rather than copy. vtk.js
objects cannot cross that boundary at all.
