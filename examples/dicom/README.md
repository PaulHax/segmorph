# segmorph DICOM example

How to interface segmorph with the two DICOM segmentation interchange
formats, rendered with vtk.js:

- **RTSTRUCT** (RT Structure Set): planar contours per ROI. The adapter in
  [`src/rtstruct.ts`](./src/rtstruct.ts) encodes/decodes ROIs as world-space
  contour loops - exactly the input shape of segmorph's `contourToSurface`
  (the port of SlicerRT's planar-contour-to-closed-surface stitch). One demo
  ROI is stored with the RTSTRUCT *keyhole* technique (a hole encoded via a
  zero-width channel in a single loop), which the stitch splits back apart.
- **DICOM SEG** (Segmentation IOD, BINARY): bit-packed multi-frame masks. The
  adapter in [`src/seg.ts`](./src/seg.ts) encodes/decodes each segment's
  frames to/from a plain oriented labelmap for `labelmapToSurface`. The two
  demo segments deliberately overlap - possible because every SEG segment
  owns its own frames, mirroring segmorph's per-segment labelmaps.

Nothing rendered skips the DICOM layer: the app synthesizes segmorph-native
data, encodes it to part-10 bytes with [dcmjs](https://github.com/dcmjs-org/dcmjs),
decodes those bytes back, converts, and renders. The generated `.dcm` files
are downloadable from the toolbar for inspection in other viewers.

The adapters are example code, deliberately minimal (single orientation, no
DimensionOrganization, app-side colors instead of CIELab, no referenced image
series). They show the geometry, sequence, and PixelData handling a
production adapter needs; the segmorph core stays DICOM-free by design -
plain typed arrays in, plain typed arrays out.

## Run

```bash
npm install
npm run dev
```
