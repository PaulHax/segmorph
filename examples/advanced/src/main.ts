import { addSegment, createSegment, createSegmentation } from 'segmorph';

import { toOrientedImage } from './adapter';
import { loadChestCT, SEGMENT_SEEDS } from './data';
import { createConversionScheduler } from './scheduler';
import { createSliceView } from './sliceView';
import { createVolumeView } from './volumeView';

const query = <T extends HTMLElement>(selector: string) => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`missing element: ${selector}`);
  return element;
};

const meshStat = query<HTMLSpanElement>('#mesh-stat');
const loopStat = query<HTMLSpanElement>('#loop-stat');
const smoothInput = query<HTMLInputElement>('#smooth');
const decimateInput = query<HTMLInputElement>('#decimate');
const sliceInput = query<HTMLInputElement>('#slice');
const sliceLabel = query<HTMLSpanElement>('#slice-label');
const ctOpacityInput = query<HTMLInputElement>('#ct-opacity');
const strideInput = query<HTMLSelectElement>('#stride');

// The segmentation model: named, individually identified segments, each with
// its own label value. Segments are addressed by id rather than by the voxel
// value they happen to occupy, which is what lets them overlap.
const segmentation = SEGMENT_SEEDS.reduce(
  (acc, seed) =>
    addSegment(
      acc,
      createSegment({
        id: seed.id,
        name: seed.name,
        color: seed.color,
      }),
    ),
  createSegmentation('labelmap'),
);

const rgb = (color: readonly number[]) => `rgb(${color.map((c) => Math.round(c * 255)).join(',')})`;

const volumeView = createVolumeView(query('#volume'), SEGMENT_SEEDS);
const sliceView = createSliceView(query('#slice-view'), SEGMENT_SEEDS);

const scheduler = createConversionScheduler({
  debounceMs: 150,
  getSegments: () =>
    SEGMENT_SEEDS.map((seed, index) => ({
      id: seed.id,
      labelValue: index + 1,
      window: seed.window,
    })),
  getPipeline: () => ({
    smoothIterations: Number(smoothInput.value),
    targetReduction: Number(decimateInput.value) / 100,
  }),
  getStride: () => Number(strideInput.value),
  onResult(response) {
    let extracted = 0;
    let final = 0;
    for (const segment of response.segments) {
      const mesh = { points: segment.points, polys: segment.polys };
      volumeView.updateMesh(segment.id, mesh);
      sliceView.setMesh(segment.id, mesh);
      extracted += segment.extractedTriangles;
      final += segment.polys.length / 4;
    }
    volumeView.render();
    sliceView.render();

    // Each number sits next to the control that changes it: mesh counts by the
    // smoothing and decimation sliders, loop counts by the slice slider.
    const reduction = extracted > 0 ? Math.round((1 - final / extracted) * 100) : 0;
    meshStat.textContent =
      `${extracted.toLocaleString()} -> ${final.toLocaleString()} tris` +
      ` (${reduction}% fewer), ${(response.durationMs / 1000).toFixed(1)}s`;
    updateLoopStat();
  },
});

function updateLoopStat() {
  const loops = [...sliceView.getLoopCounts().values()].reduce((a, b) => a + b, 0);
  loopStat.textContent = `${loops} contour loops on slice`;
}

// Each toggle is drawn in its segment's color, so this row is also the key
// for the contours and surfaces in the views.
const toolbarLegend = query('#legend');
for (const id of segmentation.order) {
  const segment = segmentation.segments[id];
  const seed = SEGMENT_SEEDS.find((candidate) => candidate.id === id);
  if (!seed) continue;

  const label = document.createElement('label');
  label.style.color = rgb(seed.color);

  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = true;
  box.addEventListener('change', () => {
    volumeView.setSegmentVisible(id, box.checked);
    sliceView.setSegmentVisible(id, box.checked);
    updateLoopStat();
  });

  label.append(box, document.createTextNode(segment.name));
  toolbarLegend.append(label);
}

for (const input of [smoothInput, decimateInput, strideInput]) {
  input.addEventListener('input', () => {
    meshStat.textContent = 'converting...';
    scheduler.requestConvert();
  });
}

sliceInput.addEventListener('input', () => sliceView.setSlice(Number(sliceInput.value)));

// The wheel is the primary way to move through slices, so the slider follows it
// rather than owning the position.
sliceView.onSliceChange((z) => {
  sliceInput.value = String(z);
  sliceLabel.textContent = `slice ${z.toFixed(1)} mm`;
  updateLoopStat();
});
ctOpacityInput.addEventListener('input', () => {
  volumeView.setVolumeOpacity(Number(ctOpacityInput.value) / 100);
});

async function start() {
  const ct = await loadChestCT();

  // One reshape and the vtk.js volume is a segmorph OrientedImage; the scalar
  // array is shared, not copied.
  const image = toOrientedImage<Uint8Array>(ct);

  volumeView.setVolume(ct);
  sliceView.setVolume(ct);

  const bounds = sliceView.getBounds();
  if (bounds) {
    sliceInput.min = String(bounds[4]);
    sliceInput.max = String(bounds[5]);
    sliceInput.step = String(image.spacing[2]);
    sliceInput.value = String((bounds[4] + bounds[5]) / 2);
  }
  sliceView.setSlice(Number(sliceInput.value));

  scheduler.init(image.data, {
    dims: image.dims,
    spacing: image.spacing,
    origin: image.origin,
    direction: image.direction,
  });

  meshStat.textContent = 'converting...';
  scheduler.requestConvert();
}

start().catch((error) => {
  meshStat.textContent = `failed: ${error instanceof Error ? error.message : String(error)}`;
});
