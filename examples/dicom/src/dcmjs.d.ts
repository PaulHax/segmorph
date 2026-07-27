// Minimal typings for the parts of dcmjs this example touches. dcmjs ships no
// TypeScript declarations; naturalized datasets are inherently dynamic, so
// their values arrive as `unknown` and the adapters narrow them at the edges.
declare module 'dcmjs' {
  export type NaturalizedDataset = Record<string, unknown>;

  const dcmjs: {
    data: {
      DicomMessage: {
        readFile(buffer: ArrayBuffer): {
          dict: Record<string, unknown>;
          meta: Record<string, unknown>;
        };
      };
      DicomMetaDictionary: {
        naturalizeDataset(dict: Record<string, unknown>): NaturalizedDataset;
        uid(): string;
      };
      datasetToDict(dataset: NaturalizedDataset): { write(): ArrayBuffer };
      BitArray: {
        pack(values: Uint8Array): Uint8Array;
        unpack(packed: Uint8Array): Uint8Array;
      };
    };
  };
  export default dcmjs;
}
