import { Buffer } from 'node:buffer';
import { describe, expect, it } from 'vitest';

import { readNrrd } from '../src/io/nrrd.js';

const fields = [
  'type: uint32',
  'dimension: 3',
  'sizes: 1 1 1',
  'encoding: raw',
  'endian: big',
  'space dimension: 3',
  'space directions: (1,0,0) (0,1,0) (0,0,1)',
  'space origin: (0,0,0)',
];

function nrrd(lines: string[], newline = '\n', littleEndian = false) {
  return Buffer.concat([
    Buffer.from(['NRRD0005', ...lines, '', ''].join(newline)),
    Buffer.from(littleEndian ? [0x78, 0x56, 0x34, 0x12] : [0x12, 0x34, 0x56, 0x78]),
  ]);
}

function expectImage(bytes: Uint8Array) {
  const original = [...bytes];
  const image = readNrrd(bytes);
  expect(image.data).toBeInstanceOf(Uint32Array);
  expect([...image.data]).toEqual([0x12345678]);
  expect(image.dims).toEqual([1, 1, 1]);
  expect(image.spacing).toEqual([1, 1, 1]);
  expect(image.origin).toEqual([0, 0, 0]);
  expect(image.direction).toEqual([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
  expect([...bytes]).toEqual(original);
}

describe.each([
  { name: 'LF', newline: '\n' },
  { name: 'CRLF', newline: '\r\n' },
])('NRRD metadata with $name', ({ newline }) => {
  it.each(['big', 'little'])(
    'preserves %s endian with metadata anywhere in the header',
    (endian) => {
      const structural = fields.map((line) =>
        line === 'endian: big' ? 'endian: ' + endian : line,
      );
      for (const position of [0, 4, structural.length]) {
        const lines = [...structural];
        lines.splice(position, 0, 'endian:=metadata', 'Endian:=different key', 'ENDIAN:=');
        expectImage(nrrd(lines, newline, endian === 'little'));
      }
    },
  );

  it('ignores repeated and mixed-case metadata matching structural field names', () => {
    const lines = fields.flatMap((line) => {
      const key = line.slice(0, line.indexOf(':'));
      return [key + ':=before', line.toUpperCase(), key.toUpperCase() + ':=after', key + ':='];
    });
    expectImage(nrrd(lines, newline));
  });

  it('handles colons in metadata keys and structural-looking text in metadata values', () => {
    expectImage(
      nrrd(
        [
          ...fields,
          'endian:description:=metadata',
          'type:description:=metadata',
          'endian:=notes: big:=value',
          'type:=notes: uint32',
          'space origin:=notes: (9,9,9)',
          ' endian := whitespace belongs to the key and value ',
          '# endian:=comment',
        ],
        newline,
      ),
    );
  });

  it.each([' ', '\t', '\t ', '\v'])(
    'rejects ambiguous endian fields with delimiter %j instead of changing byte order',
    (whitespace) => {
      // Teem recognizes a known field before looking for := in its value.
      for (const position of [0, 4, fields.length]) {
        const lines = [...fields];
        lines.splice(position, 0, 'endian:' + whitespace + 'note:=metadata');
        expect(() => readNrrd(nrrd(lines, newline))).toThrow(
          'Unsupported NRRD endian: note:=metadata',
        );
      }
    },
  );

  it.each([' ', '\t', '\t ', '\v'])(
    'preserves colon-whitespace fields with delimiter %j',
    (whitespace) => {
      const lines = fields.map((line) => line.replace(': ', ':' + whitespace));
      lines.push('content:' + whitespace + 'operation:=example');
      expectImage(nrrd(lines, newline));

      // A := in a structural value must still reach that field's validation.
      const invalid = lines.map((line) =>
        line.startsWith('type:') ? 'type:' + whitespace + 'uint32:=metadata' : line,
      );
      expect(() => readNrrd(nrrd(invalid, newline))).toThrow(
        'Unsupported NRRD scalar type: uint32:=metadata',
      );
    },
  );
});

it.each([
  ['type: uint32', 'Unsupported NRRD scalar type: '],
  ['encoding: raw', 'Only raw NRRD encoding is supported'],
  ['dimension: 3', 'Invalid NRRD dimension or sizes'],
  ['sizes: 1 1 1', 'Invalid NRRD dimension or sizes'],
  [
    'space directions: (1,0,0) (0,1,0) (0,0,1)',
    'NRRD space directions must contain one vector per dimension',
  ],
  ['space origin: (0,0,0)', 'Invalid NRRD vector: '],
])('metadata cannot supply the structural field %s', (field, message) => {
  const lines = fields.map((line) => (line === field ? line.replace(': ', ':=') : line));
  expect(() => readNrrd(nrrd(lines))).toThrow(message);
});

it('metadata cannot conceal an unsupported structural encoding', () => {
  const lines = fields.map((line) => (line === 'encoding: raw' ? 'encoding: gzip' : line));
  expect(() => readNrrd(nrrd([...lines, 'encoding:=raw']))).toThrow(
    'Only raw NRRD encoding is supported',
  );
});
