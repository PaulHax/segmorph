// Registers the gzip-capable data access helper; the sample volume is served
// compressed, and the lite helper cannot decompress it.
import '@kitware/vtk.js/IO/Core/DataAccessHelper/HttpDataAccessHelper';
import vtkHttpDataSetReader from '@kitware/vtk.js/IO/Core/HttpDataSetReader';
import type vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';

/**
 * A real chest CT: LIDC2 from the Lung Image Database Consortium, served as
 * vtk.js sample data. 256 x 256 x 133, Uint8, anisotropic spacing
 * (1.40625, 1.40625, 2.5), which is exactly the oblique/anisotropic case the
 * conversions are tested against.
 */
const LIDC2_URL = 'https://kitware.github.io/vtk-js/data/volume/LIDC2.vti';

export async function loadChestCT(): Promise<vtkImageData> {
  const reader = vtkHttpDataSetReader.newInstance({ fetchGzip: true });
  await reader.setUrl(LIDC2_URL, { loadData: true });
  return reader.getOutputData() as vtkImageData;
}

export type SegmentSeed = {
  id: string;
  name: string;
  /** Display color, 0..1 per channel. */
  color: [number, number, number];
  /** Inclusive intensity window that seeds this segment. */
  window: [number, number];
  /** Surface opacity in the 3D view. */
  opacity: number;
};

/**
 * Two segments seeded straight off the CT intensities. A real app gets these
 * from a painting tool or a model; thresholding just gives the demo something
 * anatomically real to convert without shipping a segmentation file.
 *
 * Note that these two overlap: every bone voxel is also a body voxel. A single
 * fused labelmap cannot represent that, because one voxel holds one value.
 * Here each segment carries its own mask and its own label value, which is the
 * whole point of the segmentation model.
 */
export const SEGMENT_SEEDS: SegmentSeed[] = [
  // The body shell is drawn nearly transparent so the bone inside it reads.
  { id: 'body', name: 'Body', color: [0.4, 0.68, 0.95], window: [30, 255], opacity: 0.1 },
  { id: 'bone', name: 'Bone', color: [0.98, 0.72, 0.28], window: [150, 255], opacity: 1 },
];

/** Threshold a scalar volume into a binary mask for one seed. */
export function seedMask(scalars: Uint8Array, seed: SegmentSeed, labelValue: number) {
  const [low, high] = seed.window;
  const mask = new Uint8Array(scalars.length);
  for (let i = 0; i < scalars.length; i += 1) {
    const value = scalars[i];
    if (value >= low && value <= high) mask[i] = labelValue;
  }
  return mask;
}
