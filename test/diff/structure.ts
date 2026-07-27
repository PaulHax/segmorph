import { enclosedVolume, triangleIndices, type Mesh, type Triangle } from './mesh.js';

function distinctTriangleIndices(mesh: Mesh) {
  const triangles = triangleIndices(mesh);
  for (const triangle of triangles) {
    if (new Set(triangle).size !== 3) {
      throw new RangeError('Mesh triangles must contain three distinct vertices');
    }
  }
  return triangles;
}

function edgeKey(a: number, b: number) {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function edgeUses(triangles: readonly Triangle[]) {
  const uses = new Map<string, { from: number; to: number }[]>();
  for (const [a, b, c] of triangles) {
    for (const [from, to] of [
      [a, b],
      [b, c],
      [c, a],
    ]) {
      const key = edgeKey(from, to);
      const entries = uses.get(key) ?? [];
      entries.push({ from, to });
      uses.set(key, entries);
    }
  }
  return uses;
}

export function isWatertight(mesh: Mesh) {
  const uses = edgeUses(distinctTriangleIndices(mesh));
  return uses.size > 0 && [...uses.values()].every((entries) => entries.length === 2);
}

function hasManifoldVertexLinks(triangles: readonly Triangle[]) {
  const links = new Map<number, [number, number][]>();
  for (const triangle of triangles) {
    for (let index = 0; index < 3; index += 1) {
      const vertex = triangle[index];
      const edges = links.get(vertex) ?? [];
      edges.push([triangle[(index + 1) % 3], triangle[(index + 2) % 3]]);
      links.set(vertex, edges);
    }
  }

  for (const edges of links.values()) {
    const neighbors = new Map<number, Set<number>>();
    const connect = (from: number, to: number) => {
      const adjacent = neighbors.get(from) ?? new Set<number>();
      adjacent.add(to);
      neighbors.set(from, adjacent);
    };
    for (const [a, b] of edges) {
      connect(a, b);
      connect(b, a);
    }
    if ([...neighbors.values()].some((adjacent) => adjacent.size > 2)) return false;

    const [first] = neighbors.keys();
    const visited = new Set([first]);
    for (const current of visited) {
      for (const adjacent of neighbors.get(current) ?? []) {
        visited.add(adjacent);
      }
    }
    if (visited.size !== neighbors.size) return false;
  }
  return true;
}

export function isManifold(mesh: Mesh) {
  const triangles = distinctTriangleIndices(mesh);
  return (
    [...edgeUses(triangles).values()].every((entries) => entries.length <= 2) &&
    hasManifoldVertexLinks(triangles)
  );
}

export function hasConsistentOutwardOrientation(mesh: Mesh) {
  if (enclosedVolume(mesh) <= 0) return false;

  return [...edgeUses(distinctTriangleIndices(mesh)).values()].every(
    (entries) =>
      entries.length <= 2 &&
      (entries.length < 2 ||
        (entries[0].from === entries[1].to && entries[0].to === entries[1].from)),
  );
}

export function isVolumeWithinBand(mesh: Mesh, referenceVolume: number, band: number) {
  if (!Number.isFinite(referenceVolume) || referenceVolume <= 0) {
    throw new RangeError('Reference volume must be positive and finite');
  }
  if (!Number.isFinite(band) || band < 0) {
    throw new RangeError('Volume band must be non-negative and finite');
  }

  const ratio = Math.abs(enclosedVolume(mesh)) / referenceVolume;
  return ratio >= 1 - band && ratio <= 1 + band;
}
