export type SegmentColor = readonly [number, number, number];

export type Segment = Readonly<{
  id: string;
  name: string;
  color: SegmentColor;
  locked: boolean;
  representations: Readonly<Record<string, unknown>>;
}>;

export type CreateSegmentOptions = Readonly<{
  id: string;
  name: string;
  color: SegmentColor;
  locked?: boolean;
  representations?: Readonly<Record<string, unknown>>;
}>;

export function createSegment(options: CreateSegmentOptions): Segment {
  return {
    id: options.id,
    name: options.name,
    color: options.color,
    locked: options.locked ?? false,
    representations: options.representations ?? {},
  };
}

export function getRepresentation<T = unknown>(segment: Segment, name: string) {
  return Object.hasOwn(segment.representations, name)
    ? (segment.representations[name] as T)
    : undefined;
}

export function setRepresentation(
  segment: Segment,
  name: string,
  representation: unknown,
): Segment {
  return {
    ...segment,
    representations: {
      ...segment.representations,
      [name]: representation,
    },
  };
}
