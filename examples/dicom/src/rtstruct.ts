// RTSTRUCT adapter: DICOM RT Structure Set <-> segmorph's contour
// representation. This is app-edge code by design - the segmorph core stays
// DICOM-free and consumes/produces plain arrays; dcmjs does the DICOM
// encoding. The decoded `loops` are exactly what `contourToSurface` takes:
// world-space closed polylines, interleaved xyz, implicit closure.

import { demoIdentity, readPart10, toList, uid, writePart10, type DicomItem } from './dicom';

const RTSTRUCT_SOP_CLASS = '1.2.840.10008.5.1.4.1.1.481.3';

export type RtRoi = {
  number: number;
  name: string;
  /** 0-255 RGB (ROIDisplayColor). */
  color: [number, number, number];
  /** World-space closed contour loops, interleaved xyz. */
  loops: Float64Array[];
};

export type RtStruct = {
  label: string;
  rois: RtRoi[];
};

/**
 * Encode ROIs as a minimal RT Structure Set. Every loop becomes one
 * CLOSED_PLANAR ContourSequence item; RTSTRUCT implies closure, so the loops
 * are stored without a repeated end point, matching segmorph's convention.
 */
export function encodeRtStruct(
  rois: readonly RtRoi[],
  options: { label?: string; frameOfReferenceUID?: string } = {},
): ArrayBuffer {
  const frameOfReferenceUID = options.frameOfReferenceUID ?? uid();
  const label = options.label ?? 'segmorph demo';
  const dataset = {
    SOPClassUID: RTSTRUCT_SOP_CLASS,
    SOPInstanceUID: uid(),
    SeriesInstanceUID: uid(),
    Modality: 'RTSTRUCT',
    SeriesNumber: 1,
    InstanceNumber: 1,
    StructureSetLabel: label,
    StructureSetDate: '20260718',
    StructureSetTime: '000000',
    ...demoIdentity(frameOfReferenceUID),
    StructureSetROISequence: rois.map((roi) => ({
      ROINumber: roi.number,
      ReferencedFrameOfReferenceUID: frameOfReferenceUID,
      ROIName: roi.name,
      ROIGenerationAlgorithm: 'MANUAL',
    })),
    ROIContourSequence: rois.map((roi) => ({
      ReferencedROINumber: roi.number,
      ROIDisplayColor: roi.color,
      ContourSequence: roi.loops.map((loop) => ({
        ContourGeometricType: 'CLOSED_PLANAR',
        NumberOfContourPoints: loop.length / 3,
        ContourData: Array.from(loop),
      })),
    })),
    RTROIObservationsSequence: rois.map((roi, index) => ({
      ObservationNumber: index + 1,
      ReferencedROINumber: roi.number,
      RTROIInterpretedType: 'ORGAN',
      ROIInterpreter: '',
    })),
  };
  return writePart10(dataset);
}

/** Decode an RT Structure Set into per-ROI world-space contour loops. */
export function decodeRtStruct(buffer: ArrayBuffer): RtStruct {
  const dataset = readPart10(buffer);
  if (dataset.SOPClassUID !== RTSTRUCT_SOP_CLASS) {
    throw new Error(`Not an RT Structure Set: ${dataset.SOPClassUID}`);
  }

  const roisByNumber = new Map<number, { name: string }>();
  for (const item of toList<DicomItem>(dataset.StructureSetROISequence)) {
    roisByNumber.set(Number(item.ROINumber), { name: String(item.ROIName ?? 'ROI') });
  }

  const rois = toList<DicomItem>(dataset.ROIContourSequence).map((roiContour): RtRoi => {
    const number = Number(roiContour.ReferencedROINumber);
    const color = toList<number>(roiContour.ROIDisplayColor);
    const loops = toList<DicomItem>(roiContour.ContourSequence)
      .filter((contour) => contour.ContourGeometricType === 'CLOSED_PLANAR')
      .map((contour) => Float64Array.from(toList<number>(contour.ContourData)));
    return {
      number,
      name: roisByNumber.get(number)?.name ?? `ROI ${number}`,
      color: color.length === 3 ? [color[0], color[1], color[2]] : [255, 170, 0],
      loops,
    };
  });

  return { label: String(dataset.StructureSetLabel ?? 'RTSTRUCT'), rois };
}
