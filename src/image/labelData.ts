import type { ImageData } from './orientedImage.js';

/** Constructor for any labelmap storage array, e.g. Uint8Array or Uint16Array. */
export type LabelArrayConstructor = new (length: number) => ImageData;

/** Narrow unsigned-integer storage types a labelmap allocates by default. */
export type LabelData = Uint8Array | Uint16Array | Uint32Array;

const maxLabelValue = 0xffff_ffff;

/**
 * Throw unless labelValue is an integer usable as a labelmap foreground value
 * (1..4294967295). Shared by every converter that reads or writes a label.
 */
export function validateLabelValue(labelValue: number) {
  if (!Number.isInteger(labelValue) || labelValue < 1 || labelValue > maxLabelValue) {
    throw new Error('labelValue must be an integer between 1 and 4294967295');
  }
}

/**
 * Allocate labelmap storage. By default the width is the narrowest unsigned
 * integer that holds labelValue; pass outputArray to force a specific element
 * type, e.g. to preserve a source dtype across a round trip.
 */
export function createLabelData(length: number, labelValue: number): LabelData;
export function createLabelData(
  length: number,
  labelValue: number,
  outputArray?: LabelArrayConstructor,
): ImageData;
export function createLabelData(
  length: number,
  labelValue: number,
  outputArray?: LabelArrayConstructor,
) {
  if (outputArray) return new outputArray(length);
  if (labelValue <= 0xff) return new Uint8Array(length);
  if (labelValue <= 0xffff) return new Uint16Array(length);
  return new Uint32Array(length);
}
