export type NrrdData =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

export type Nrrd = {
  dims: number[];
  spacing: number[];
  origin: number[];
  direction: number[][];
  data: NrrdData;
};

type Scalar = {
  bytes: number;
  create(length: number): NrrdData;
  read(view: DataView, offset: number, littleEndian: boolean): number;
};

function aliases(names: string[], scalar: Scalar) {
  return Object.fromEntries(names.map((name) => [name, scalar]));
}

const scalarTypes: Record<string, Scalar> = {
  ...aliases(['signed char', 'int8', 'int8_t'], {
    bytes: 1,
    create: (length) => new Int8Array(length),
    read: (view, offset) => view.getInt8(offset),
  }),
  ...aliases(['uchar', 'unsigned char', 'uint8', 'uint8_t'], {
    bytes: 1,
    create: (length) => new Uint8Array(length),
    read: (view, offset) => view.getUint8(offset),
  }),
  ...aliases(['short', 'short int', 'signed short', 'signed short int', 'int16', 'int16_t'], {
    bytes: 2,
    create: (length) => new Int16Array(length),
    read: (view, offset, little) => view.getInt16(offset, little),
  }),
  ...aliases(['ushort', 'unsigned short', 'unsigned short int', 'uint16', 'uint16_t'], {
    bytes: 2,
    create: (length) => new Uint16Array(length),
    read: (view, offset, little) => view.getUint16(offset, little),
  }),
  ...aliases(['int', 'signed int', 'int32', 'int32_t'], {
    bytes: 4,
    create: (length) => new Int32Array(length),
    read: (view, offset, little) => view.getInt32(offset, little),
  }),
  ...aliases(['uint', 'unsigned int', 'uint32', 'uint32_t'], {
    bytes: 4,
    create: (length) => new Uint32Array(length),
    read: (view, offset, little) => view.getUint32(offset, little),
  }),
  ...aliases(['float'], {
    bytes: 4,
    create: (length) => new Float32Array(length),
    read: (view, offset, little) => view.getFloat32(offset, little),
  }),
  ...aliases(['double'], {
    bytes: 8,
    create: (length) => new Float64Array(length),
    read: (view, offset, little) => view.getFloat64(offset, little),
  }),
};

function parseVector(value: string) {
  const match = /^\(([^)]+)\)$/.exec(value.trim());
  if (!match) throw new Error(`Invalid NRRD vector: ${value}`);
  const vector = match[1].split(',').map(Number);
  if (vector.some((component) => !Number.isFinite(component))) {
    throw new Error(`Invalid NRRD vector: ${value}`);
  }
  return vector;
}

export function readNrrd(bytes: Uint8Array): Nrrd {
  let headerEnd = -1;
  let separatorLength = 0;
  for (let index = 0; index < bytes.length - 1; index += 1) {
    if (bytes[index] === 10 && bytes[index + 1] === 10) {
      headerEnd = index;
      separatorLength = 2;
      break;
    }
    if (index < bytes.length - 3 && bytes[index] === 13 && bytes[index + 1] === 10
      && bytes[index + 2] === 13 && bytes[index + 3] === 10) {
      headerEnd = index;
      separatorLength = 4;
      break;
    }
  }
  if (headerEnd < 0) throw new Error('NRRD header is not terminated');

  const lines = new TextDecoder().decode(bytes.subarray(0, headerEnd)).split(/\r?\n/);
  if (!/^NRRD\d+$/.test(lines[0])) throw new Error('Invalid NRRD magic');

  const fields = new Map<string, string>();
  for (const line of lines.slice(1)) {
    if (!line || line.startsWith('#')) continue;
    const separator = line.indexOf(':');
    if (separator < 0) throw new Error(`Invalid NRRD header line: ${line}`);
    fields.set(line.slice(0, separator).trim().toLowerCase(), line.slice(separator + 1).trim());
  }

  if (fields.get('encoding')?.toLowerCase() !== 'raw') {
    throw new Error('Only raw NRRD encoding is supported');
  }
  const dims = (fields.get('sizes') ?? '').split(/\s+/).map(Number);
  const dimension = Number(fields.get('dimension'));
  if (!Number.isInteger(dimension) || dims.length !== dimension
    || dims.some((size) => !Number.isInteger(size) || size <= 0)) {
    throw new Error('Invalid NRRD dimension or sizes');
  }

  const directions = (fields.get('space directions') ?? '').match(/\([^)]*\)|none/gi);
  if (!directions || directions.length !== dimension || directions.some((value) => value.toLowerCase() === 'none')) {
    throw new Error('NRRD space directions must contain one vector per dimension');
  }
  const spaceDirections = directions.map(parseVector);
  const spacing = spaceDirections.map((vector) => Math.hypot(...vector));
  const normalizedDirections = spaceDirections.map((vector, axis) => (
    vector.map((component) => component / spacing[axis])
  ));
  const direction = normalizedDirections[0].map((_, component) => (
    normalizedDirections.map((axis) => axis[component])
  ));
  const origin = parseVector(fields.get('space origin') ?? '');

  const scalar = scalarTypes[(fields.get('type') ?? '').toLowerCase()];
  if (!scalar) throw new Error(`Unsupported NRRD scalar type: ${fields.get('type') ?? ''}`);
  const length = dims.reduce((product, size) => product * size, 1);
  const dataOffset = headerEnd + separatorLength;
  if (bytes.length - dataOffset !== length * scalar.bytes) {
    throw new Error('NRRD payload length does not match sizes and type');
  }
  const littleEndian = fields.get('endian')?.toLowerCase() !== 'big';
  const view = new DataView(bytes.buffer, bytes.byteOffset + dataOffset, length * scalar.bytes);
  const data = scalar.create(length);
  for (let index = 0; index < length; index += 1) {
    data[index] = scalar.read(view, index * scalar.bytes, littleEndian);
  }

  return { dims, spacing, origin, direction, data };
}
