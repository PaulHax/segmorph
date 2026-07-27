import dcmjs from 'dcmjs';

const { DicomMessage, DicomMetaDictionary, datasetToDict } = dcmjs.data;

export const EXPLICIT_LITTLE_ENDIAN = '1.2.840.10008.1.2.1';

export const uid = () => DicomMetaDictionary.uid();

/**
 * A naturalized dataset or sequence item. dcmjs ships no declarations and the
 * naturalized shape is keyed by whichever tags the file happens to carry, so
 * every read lands as `unknown` and narrows where it is used.
 */
export type DicomItem = Record<string, unknown>;

/**
 * dcmjs naturalization may return a single-item sequence as the item itself;
 * normalize every sequence read through this. The parameter is `unknown`
 * because the argument is always a dynamic tag read, and the caller names the
 * element type it expects -- that expectation is the assumption being made
 * about the file, so it is written down at each call site.
 */
export function toList<T>(value: unknown): T[] {
  if (value === undefined) return [];
  return (Array.isArray(value) ? value : [value]) as T[];
}

/** Shared patient/study identity so both demo files land in one study. */
export function demoIdentity(frameOfReferenceUID: string) {
  return {
    PatientName: 'segmorph^demo',
    PatientID: 'segmorph-demo',
    StudyInstanceUID: '1.2.826.0.1.3680043.8.498.20260718.1',
    StudyID: '1',
    AccessionNumber: '1',
    FrameOfReferenceUID: frameOfReferenceUID,
  };
}

/** Serialize a naturalized dataset to a part-10 DICOM byte stream. */
export function writePart10(dataset: Record<string, unknown>): ArrayBuffer {
  const full = {
    ...dataset,
    _meta: {
      FileMetaInformationVersion: { Value: [new Uint8Array([0, 1]).buffer], vr: 'OB' },
      MediaStorageSOPClassUID: { Value: [dataset.SOPClassUID], vr: 'UI' },
      MediaStorageSOPInstanceUID: { Value: [dataset.SOPInstanceUID], vr: 'UI' },
      TransferSyntaxUID: { Value: [EXPLICIT_LITTLE_ENDIAN], vr: 'UI' },
      ImplementationClassUID: { Value: [uid()], vr: 'UI' },
      ImplementationVersionName: { Value: ['segmorph-dicom-demo'], vr: 'SH' },
    },
  };
  return datasetToDict(full).write();
}

/** Parse a part-10 DICOM byte stream into a naturalized dataset. */
export function readPart10(buffer: ArrayBuffer): DicomItem {
  return DicomMetaDictionary.naturalizeDataset(DicomMessage.readFile(buffer).dict);
}
