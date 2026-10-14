export function pointAt(points: Float32Array, index: number) {
  const offset = index * 3;
  return [points[offset], points[offset + 1], points[offset + 2]] as const;
}
