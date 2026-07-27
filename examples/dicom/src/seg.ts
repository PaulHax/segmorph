// DICOM SEG adapter: Segmentation IOD (BINARY) <-> segmorph's labelmap
// representation. One SEG object carries every segment; each segment owns its
// own run of frames, which is why DICOM SEG segments may overlap freely -
// exactly the property segmorph's per-segment labelmaps preserve. The decoded
// side hands `labelmapToSurface` a plain oriented image.
//
// Kept minimal on purpose (single orientation, one file, no
// DimensionOrganization or CIELab colors); it demonstrates the geometry and
// PixelData handling a production adapter needs.

import dcmjs from 'dcmjs';

import { demoIdentity, readPart10, toList, uid, writePart10, type DicomItem } from './dicom';

const { BitArray } = dcmjs.data;

const SEG_SOP_CLASS = '1.2.840.10008.5.1.4.1.1.66.4';

type Vec3 = [number, number, number];
type Direction = readonly [Vec3, Vec3, Vec3];

/** The shared sampling grid, shaped like segmorph's ImageGeometry. */
export type SegGeometry = {
  dims: [number, number, number];
  spacing: [number, number, number];
  origin: Vec3;
  /** Row-major; column j is the direction of image axis j. */
  direction: Direction;
};

export type SegSegment = {
  number: number;
  label: string;
  /** One byte per voxel, 0 or 1, x-fastest - a segmorph labelmap payload. */
  data: Uint8Array;
};

export type Seg = {
  geometry: SegGeometry;
  segments: SegSegment[];
};

const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

/** Encode per-segment binary masks on one grid as a BINARY DICOM SEG. */
export function encodeSeg(
  geometry: SegGeometry,
  segments: readonly SegSegment[],
  options: { frameOfReferenceUID?: string } = {},
): ArrayBuffer {
  const [nx, ny, nz] = geometry.dims;
  const sliceCount = nz;
  const { direction, spacing, origin } = geometry;
  const zAxis: Vec3 = [direction[0][2], direction[1][2], direction[2][2]];

  // Frames: for each segment, its slices bottom to top. BINARY SEG packs the
  // frames' bits continuously (frame boundaries may fall inside a byte), so
  // all frames are concatenated voxel-wise first and packed once.
  const frameVoxels = nx * ny;
  const allFrames = new Uint8Array(frameVoxels * sliceCount * segments.length);
  const perFrameGroups: unknown[] = [];
  segments.forEach((segment, segmentIndex) => {
    for (let slice = 0; slice < sliceCount; slice += 1) {
      allFrames.set(
        segment.data.subarray(slice * frameVoxels, (slice + 1) * frameVoxels),
        (segmentIndex * sliceCount + slice) * frameVoxels,
      );
      perFrameGroups.push({
        PlanePositionSequence: [
          {
            ImagePositionPatient: [
              origin[0] + slice * spacing[2] * zAxis[0],
              origin[1] + slice * spacing[2] * zAxis[1],
              origin[2] + slice * spacing[2] * zAxis[2],
            ],
          },
        ],
        SegmentIdentificationSequence: [{ ReferencedSegmentNumber: segment.number }],
      });
    }
  });
  const packed = BitArray.pack(allFrames);

  const dataset = {
    SOPClassUID: SEG_SOP_CLASS,
    SOPInstanceUID: uid(),
    SeriesInstanceUID: uid(),
    Modality: 'SEG',
    SeriesNumber: 2,
    InstanceNumber: 1,
    ImageType: ['DERIVED', 'PRIMARY'],
    ContentLabel: 'SEGMORPH',
    ContentDescription: 'segmorph demo segmentation',
    SegmentationType: 'BINARY',
    ...demoIdentity(options.frameOfReferenceUID ?? uid()),
    NumberOfFrames: perFrameGroups.length,
    Rows: ny,
    Columns: nx,
    BitsAllocated: 1,
    BitsStored: 1,
    HighBit: 0,
    SamplesPerPixel: 1,
    PixelRepresentation: 0,
    PhotometricInterpretation: 'MONOCHROME2',
    LossyImageCompression: '00',
    SharedFunctionalGroupsSequence: [
      {
        PixelMeasuresSequence: [
          {
            // PixelSpacing is row spacing then column spacing.
            PixelSpacing: [spacing[1], spacing[0]],
            SpacingBetweenSlices: spacing[2],
            SliceThickness: spacing[2],
          },
        ],
        PlaneOrientationSequence: [
          {
            ImageOrientationPatient: [
              direction[0][0],
              direction[1][0],
              direction[2][0],
              direction[0][1],
              direction[1][1],
              direction[2][1],
            ],
          },
        ],
      },
    ],
    PerFrameFunctionalGroupsSequence: perFrameGroups,
    SegmentSequence: segments.map((segment) => ({
      SegmentNumber: segment.number,
      SegmentLabel: segment.label,
      SegmentAlgorithmType: 'MANUAL',
      SegmentedPropertyCategoryCodeSequence: [
        {
          CodeValue: 'T-D0050',
          CodingSchemeDesignator: 'SRT',
          CodeMeaning: 'Tissue',
        },
      ],
      SegmentedPropertyTypeCodeSequence: [
        {
          CodeValue: 'T-D0050',
          CodingSchemeDesignator: 'SRT',
          CodeMeaning: 'Tissue',
        },
      ],
    })),
    PixelData: packed.buffer.slice(packed.byteOffset, packed.byteOffset + packed.byteLength),
    _vrMap: { PixelData: 'OB' },
  };
  return writePart10(dataset);
}

/** Decode a BINARY DICOM SEG into per-segment masks on their shared grid. */
export function decodeSeg(buffer: ArrayBuffer): Seg {
  const dataset = readPart10(buffer);
  if (dataset.SOPClassUID !== SEG_SOP_CLASS) {
    throw new Error(`Not a DICOM SEG: ${dataset.SOPClassUID}`);
  }
  if (dataset.SegmentationType !== 'BINARY') {
    throw new Error(`Only BINARY segmentations are handled here: ${dataset.SegmentationType}`);
  }

  const nx = Number(dataset.Columns);
  const ny = Number(dataset.Rows);
  const frameCount = Number(dataset.NumberOfFrames);

  const shared = toList<DicomItem>(dataset.SharedFunctionalGroupsSequence)[0];
  const measures = toList<DicomItem>(shared.PixelMeasuresSequence)[0];
  const orientation = toList<DicomItem>(shared.PlaneOrientationSequence)[0];
  const iop = toList<number>(orientation.ImageOrientationPatient);
  const xAxis: Vec3 = [iop[0], iop[1], iop[2]];
  const yAxis: Vec3 = [iop[3], iop[4], iop[5]];
  const zAxis = cross(xAxis, yAxis);
  const pixelSpacing = toList<number>(measures.PixelSpacing);

  // Per-frame positions and segment assignment.
  const frames = toList<DicomItem>(dataset.PerFrameFunctionalGroupsSequence).map((group, index) => {
    const position = toList<number>(
      toList<DicomItem>(group.PlanePositionSequence)[0].ImagePositionPatient,
    ) as Vec3;
    const segmentNumber = Number(
      toList<DicomItem>(group.SegmentIdentificationSequence)[0].ReferencedSegmentNumber,
    );
    return {
      index,
      position,
      segmentNumber,
      z: position[0] * zAxis[0] + position[1] * zAxis[1] + position[2] * zAxis[2],
    };
  });
  if (frames.length !== frameCount) {
    throw new Error(`Expected ${frameCount} per-frame groups, found ${frames.length}`);
  }

  const segmentNumbers = [...new Set(frames.map((frame) => frame.segmentNumber))].sort(
    (a, b) => a - b,
  );
  const slicesPerSegment = frames.length / segmentNumbers.length;
  const nz = slicesPerSegment;

  const first = [...frames].sort((a, b) => a.z - b.z)[0];
  const firstSegmentSlices = frames
    .filter((frame) => frame.segmentNumber === first.segmentNumber)
    .sort((a, b) => a.z - b.z);
  const sliceGap =
    firstSegmentSlices.length > 1
      ? firstSegmentSlices[1].z - firstSegmentSlices[0].z
      : Number(measures.SpacingBetweenSlices ?? measures.SliceThickness ?? 1);

  const geometry: SegGeometry = {
    dims: [nx, ny, nz],
    spacing: [pixelSpacing[1], pixelSpacing[0], sliceGap],
    origin: first.position,
    direction: [
      [xAxis[0], yAxis[0], zAxis[0]],
      [xAxis[1], yAxis[1], zAxis[1]],
      [xAxis[2], yAxis[2], zAxis[2]],
    ],
  };

  // Continuous bit unpacking; a frame's bits may straddle byte boundaries.
  const pixelData = toList<ArrayBuffer>(dataset.PixelData)[0];
  const bits = BitArray.unpack(new Uint8Array(pixelData));
  const frameVoxels = nx * ny;

  const labels = new Map<number, string>(
    toList<DicomItem>(dataset.SegmentSequence).map((segment) => [
      Number(segment.SegmentNumber),
      String(segment.SegmentLabel ?? `Segment ${segment.SegmentNumber}`),
    ]),
  );

  const segments = segmentNumbers.map((segmentNumber): SegSegment => {
    const data = new Uint8Array(frameVoxels * nz);
    const ownFrames = frames
      .filter((frame) => frame.segmentNumber === segmentNumber)
      .sort((a, b) => a.z - b.z);
    ownFrames.forEach((frame, slice) => {
      const bitOffset = frame.index * frameVoxels;
      for (let voxel = 0; voxel < frameVoxels; voxel += 1) {
        data[slice * frameVoxels + voxel] = bits[bitOffset + voxel] ? 1 : 0;
      }
    });
    return {
      number: segmentNumber,
      label: labels.get(segmentNumber) ?? `Segment ${segmentNumber}`,
      data,
    };
  });

  return { geometry, segments };
}
