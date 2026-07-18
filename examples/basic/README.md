# segmorph basic example

The smallest useful thing: a binary labelmap becomes a closed surface in a
vtk.js 3D view. One file, about forty lines, no worker and no build ceremony.

```sh
# from the repo root, once
npm install && npm run build

# then here
npm install
npm run dev
```

## What it shows

`src/main.ts` does three things:

1. Builds a labelmap. A sphere painted into a 64^3 `Uint8Array`, wrapped with
   `createOrientedImage` so the geometry is validated.
2. Calls `labelmapToSurface`. One synchronous call. No async setup, no
   WebAssembly to initialize, no scheduler.
3. Hands the result to vtk.js. `mesh.points` and `mesh.polys` are already in
   VTK layout, so they go straight into a `vtkPolyData` without touching the
   numbers.

That third point is the reason the data contracts look the way they do.
segmorph never imports vtk.js, but its output is shaped so that adapting to it
costs nothing.

For real volumes, run the conversion in a worker instead of on the main
thread; see [`../advanced`](../advanced), which does that with a real chest CT.
