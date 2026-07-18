// Fill-between-slices morphological contour interpolation.
// Port of ITK's itkMorphologicalContourInterpolator (Apache-2.0,
// ITKMorphologicalContourInterpolation remote module by Zukic, Vicory,
// McCormick; algorithm after Albu, Beugeling, Laurendeau, IEEE TBME 2008).
// Median between corresponding slice regions is the iterated conditional
// dilation variant (ITK UseDistanceTransform=false); region connectivity and
// the structuring element are "ball" (8-connected in-plane), matching the
// @itk-wasm/morphological-contour-interpolation effective default.

import type { ImageData, OrientedImage } from '../image/orientedImage.js';

export type FillBetweenOptions = {
  labelValue: number;
  /** Interpolation axis (0..2). Omit to auto-detect every axis that has more
   * than one isolated segmented slice for the label. */
  axis?: number;
};

type Mask = {
  data: Uint8Array;
  nu: number;
  nv: number;
};

function emptyLike(mask: Mask): Mask {
  return { data: new Uint8Array(mask.data.length), nu: mask.nu, nv: mask.nv };
}

function cloneMask(mask: Mask): Mask {
  return { data: mask.data.slice(), nu: mask.nu, nv: mask.nv };
}

function masksEqual(a: Mask, b: Mask) {
  return a.data.every((value, index) => value === b.data[index]);
}

function isEmpty(mask: Mask) {
  return mask.data.every((value) => value === 0);
}

// One dilation by a 3x3 ball structuring element, clipped to `within`.
function conditionalDilate(mask: Mask, within: Mask): Mask {
  const { nu, nv, data } = mask;
  const out = new Uint8Array(data.length);
  for (let v = 0; v < nv; v += 1) {
    for (let u = 0; u < nu; u += 1) {
      const index = v * nu + u;
      if (!within.data[index]) continue;
      let hit = data[index];
      for (let dv = -1; dv <= 1 && !hit; dv += 1) {
        const nvIdx = v + dv;
        if (nvIdx < 0 || nvIdx >= nv) continue;
        for (let du = -1; du <= 1; du += 1) {
          const nuIdx = u + du;
          if (nuIdx < 0 || nuIdx >= nu) continue;
          if (data[nvIdx * nu + nuIdx]) {
            hit = 1;
            break;
          }
        }
      }
      if (hit) out[index] = 1;
    }
  }
  return { data: out, nu, nv };
}

function intersect(a: Mask, b: Mask): Mask {
  const data = a.data.map((value, index) => (value && b.data[index] ? 1 : 0));
  return { data: new Uint8Array(data), nu: a.nu, nv: a.nv };
}

function union(a: Mask, b: Mask): Mask {
  const data = a.data.map((value, index) => (value || b.data[index] ? 1 : 0));
  return { data: new Uint8Array(data), nu: a.nu, nv: a.nv };
}

function translateMask(mask: Mask, tu: number, tv: number): Mask {
  if (tu === 0 && tv === 0) return mask;
  const { nu, nv, data } = mask;
  const out = new Uint8Array(data.length);
  for (let v = 0; v < nv; v += 1) {
    const sv = v - tv;
    if (sv < 0 || sv >= nv) continue;
    for (let u = 0; u < nu; u += 1) {
      const su = u - tu;
      if (su < 0 || su >= nu) continue;
      if (data[sv * nu + su]) out[v * nu + u] = 1;
    }
  }
  return { data: out, nu, nv };
}

function centroid(mask: Mask): [number, number] {
  let su = 0;
  let sv = 0;
  let count = 0;
  const { nu, nv, data } = mask;
  for (let v = 0; v < nv; v += 1) {
    for (let u = 0; u < nu; u += 1) {
      if (!data[v * nu + u]) continue;
      su += u;
      sv += v;
      count += 1;
    }
  }
  return count === 0 ? [0, 0] : [su / count, sv / count];
}

function overlapScore(a: Mask, b: Mask, tu: number, tv: number) {
  const { nu, nv } = a;
  let score = 0;
  for (let v = 0; v < nv; v += 1) {
    const sv = v - tv;
    if (sv < 0 || sv >= nv) continue;
    for (let u = 0; u < nu; u += 1) {
      const su = u - tu;
      if (su < 0 || su >= nu) continue;
      if (a.data[v * nu + u] && b.data[sv * nu + su]) score += 1;
    }
  }
  return score;
}

// Translation applied to `moving` that best overlaps `fixed`. Seeds at zero
// and at the centroid difference, then greedy 4-neighbor hill climb (a
// simplification of ITK's heuristic breadth-first alignment search).
function align(fixed: Mask, moving: Mask): [number, number] {
  const [fu, fv] = centroid(fixed);
  const [mu, mv] = centroid(moving);
  const seeds: [number, number][] = [
    [0, 0],
    [Math.round(fu - mu), Math.round(fv - mv)],
  ];
  let best = seeds[0];
  let bestScore = overlapScore(fixed, moving, best[0], best[1]);
  for (const seed of seeds.slice(1)) {
    const score = overlapScore(fixed, moving, seed[0], seed[1]);
    if (score > bestScore) {
      best = seed;
      bestScore = score;
    }
  }
  for (let iteration = 0; iteration < 100; iteration += 1) {
    let improved = false;
    for (const [du, dv] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const candidate: [number, number] = [best[0] + du, best[1] + dv];
      const score = overlapScore(fixed, moving, candidate[0], candidate[1]);
      if (score > bestScore) {
        best = candidate;
        bestScore = score;
        improved = true;
      }
    }
    if (!improved) break;
  }
  return best;
}

function symmetricDifferenceCount(a: Mask, b: Mask) {
  let count = 0;
  for (let index = 0; index < a.data.length; index += 1) {
    if ((a.data[index] !== 0) !== (b.data[index] !== 0)) count += 1;
  }
  return count;
}

// Conditional dilations from `seed` until stable inside `within`; the
// sequence excludes the seed and ends at the reachable subset of `within`.
function dilationSequence(seed: Mask, within: Mask): Mask[] {
  const sequence: Mask[] = [];
  let current = intersect(seed, within);
  for (;;) {
    const next = conditionalDilate(current, within);
    if (masksEqual(next, current)) return sequence;
    sequence.push(next);
    current = next;
  }
}

// ITK FindMedianImageDilations: morph from iMask to jMask through nested
// conditional dilations of their intersection; pick the step whose symmetric
// difference to the two endpoints is most balanced.
function dilationMedian(seed: Mask, iMask: Mask, jMask: Mask): Mask {
  const iSeq = dilationSequence(seed, iMask);
  const jSeq = dilationSequence(seed, jMask);
  let longer = [...iSeq].reverse();
  let shorter = jSeq;
  if (longer.length < shorter.length) {
    longer = [...jSeq].reverse();
    shorter = iSeq;
  }
  if (longer.length === 0) return cloneMask(seed);
  if (shorter.length === 0) shorter = [seed];

  const ratio = shorter.length / longer.length;
  let best: Mask | undefined;
  let bestBalance = Infinity;
  for (let step = 0; step < longer.length; step += 1) {
    const other = shorter[Math.min(shorter.length - 1, Math.floor(ratio * step))];
    const candidate = union(longer[step], other);
    const balance = Math.abs(
      symmetricDifferenceCount(candidate, iMask) - symmetricDifferenceCount(candidate, jMask),
    );
    if (balance < bestBalance) {
      bestBalance = balance;
      best = candidate;
    }
  }
  return best ?? cloneMask(seed);
}

// Connected components with ball (8) connectivity. Returns labels 1..count.
function connectedComponents(mask: Mask) {
  const { nu, nv, data } = mask;
  const labels = new Int32Array(data.length);
  let count = 0;
  const stack: number[] = [];
  for (let start = 0; start < data.length; start += 1) {
    if (!data[start] || labels[start]) continue;
    count += 1;
    labels[start] = count;
    stack.push(start);
    while (stack.length > 0) {
      const index = stack.pop()!;
      const u = index % nu;
      const v = (index - u) / nu;
      for (let dv = -1; dv <= 1; dv += 1) {
        const nvIdx = v + dv;
        if (nvIdx < 0 || nvIdx >= nv) continue;
        for (let du = -1; du <= 1; du += 1) {
          const nuIdx = u + du;
          if (nuIdx < 0 || nuIdx >= nu) continue;
          const neighbor = nvIdx * nu + nuIdx;
          if (data[neighbor] && !labels[neighbor]) {
            labels[neighbor] = count;
            stack.push(neighbor);
          }
        }
      }
    }
  }
  return { labels, count };
}

function componentMask(labels: Int32Array, id: number, nu: number, nv: number): Mask {
  const data = new Uint8Array(labels.length);
  for (let index = 0; index < labels.length; index += 1) {
    if (labels[index] === id) data[index] = 1;
  }
  return { data, nu, nv };
}

// Competitive conditional dilation: partition `whole` among the seeds.
function partition(whole: Mask, seeds: Mask[]): Mask[] {
  const parts = seeds.map((seed) => intersect(seed, whole));
  const claimed = emptyLike(whole);
  for (const part of parts) {
    for (let index = 0; index < claimed.data.length; index += 1) {
      if (part.data[index]) claimed.data[index] = 1;
    }
  }
  const maxRounds = whole.nu + whole.nv;
  for (let round = 0; round < maxRounds; round += 1) {
    let changed = false;
    for (const part of parts) {
      const grown = conditionalDilate(part, whole);
      for (let index = 0; index < grown.data.length; index += 1) {
        if (grown.data[index] && !claimed.data[index]) {
          part.data[index] = 1;
          claimed.data[index] = 1;
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  return parts;
}

type AxisView = {
  axis: number;
  nu: number;
  nv: number;
  nSlices: number;
  toLinear: (u: number, v: number, s: number) => number;
};

function axisView(dims: readonly number[], axis: number): AxisView {
  const [nx, ny, nz] = dims;
  const strides = [1, nx, nx * ny];
  const others = [0, 1, 2].filter((a) => a !== axis);
  const sizes = [nx, ny, nz];
  return {
    axis,
    nu: sizes[others[0]],
    nv: sizes[others[1]],
    nSlices: sizes[axis],
    toLinear: (u, v, s) => u * strides[others[0]] + v * strides[others[1]] + s * strides[axis],
  };
}

function extractSlice(
  data: ImageData,
  view: AxisView,
  sliceIndex: number,
  labelValue: number,
): Mask {
  const out = new Uint8Array(view.nu * view.nv);
  for (let v = 0; v < view.nv; v += 1) {
    for (let u = 0; u < view.nu; u += 1) {
      if (data[view.toLinear(u, v, sliceIndex)] === labelValue) out[v * view.nu + u] = 1;
    }
  }
  return { data: out, nu: view.nu, nv: view.nv };
}

function writeSlice(
  out: ImageData,
  view: AxisView,
  sliceIndex: number,
  mask: Mask,
  labelValue: number,
) {
  for (let v = 0; v < view.nv; v += 1) {
    for (let u = 0; u < view.nu; u += 1) {
      if (!mask.data[v * view.nu + u]) continue;
      const index = view.toLinear(u, v, sliceIndex);
      // ITK write rule: a smaller label never overwrites a larger one.
      if (out[index] < labelValue) out[index] = labelValue;
    }
  }
}

// ITK DetermineSliceOrientations: a voxel marks a segmented slice on an axis
// when it is isolated along exactly that axis (both axis neighbors are
// background) and connected to same-label voxels along every other axis.
function detectSegmentedSlices(
  image: OrientedImage,
  labelValue: number,
): [Set<number>, Set<number>, Set<number>] {
  const [nx, ny, nz] = image.dims;
  const { data } = image;
  const slices: [Set<number>, Set<number>, Set<number>] = [new Set(), new Set(), new Set()];
  const strides = [1, nx, nx * ny];
  for (let z = 0; z < nz; z += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const index = z * nx * ny + y * nx + x;
        if (data[index] !== labelValue) continue;
        const position = [x, y, z];
        const sizes = [nx, ny, nz];
        let isolatedAxis = -1;
        let isolatedCount = 0;
        let adjacentCount = 0;
        for (let a = 0; a < 3; a += 1) {
          const minus = position[a] > 0 ? data[index - strides[a]] : 0;
          const plus = position[a] < sizes[a] - 1 ? data[index + strides[a]] : 0;
          if (minus === 0 && plus === 0) {
            isolatedCount += 1;
            isolatedAxis = a;
          } else if (minus === labelValue && plus === labelValue) {
            adjacentCount += 1;
          }
        }
        if (isolatedCount === 1 && adjacentCount === 2) {
          slices[isolatedAxis].add(position[isolatedAxis]);
        }
      }
    }
  }
  return slices;
}

function interpolate1to1(
  out: ImageData,
  view: AxisView,
  labelValue: number,
  iIdx: number,
  iMask: Mask,
  jIdx: number,
  jMask: Mask,
) {
  if (jIdx - iIdx < 2) return;
  const [tu, tv] = align(iMask, jMask);
  const mid = Math.floor((iIdx + jIdx) / 2);
  const frac = (mid - iIdx) / (jIdx - iIdx);
  // Moving jMask by t aligns it onto iMask, so j sits at -t relative to i.
  const iMid = translateMask(iMask, Math.round(-tu * frac), Math.round(-tv * frac));
  const jMid = translateMask(jMask, Math.round(tu * (1 - frac)), Math.round(tv * (1 - frac)));
  let seed = intersect(iMid, jMid);
  if (isEmpty(seed)) {
    // Guard for extreme translations: seed at the blended centroid.
    const [cu, cv] = centroid(union(iMid, jMid));
    seed = emptyLike(iMid);
    seed.data[Math.round(cv) * seed.nu + Math.round(cu)] = 1;
    seed = union(intersect(seed, iMid), intersect(seed, jMid));
    if (isEmpty(seed)) return;
  }
  const median = dilationMedian(seed, iMid, jMid);
  writeSlice(out, view, mid, median, labelValue);
  interpolate1to1(out, view, labelValue, iIdx, iMask, mid, median);
  interpolate1to1(out, view, labelValue, mid, median, jIdx, jMask);
}

function interpolateBetweenTwo(
  out: ImageData,
  view: AxisView,
  input: ImageData,
  labelValue: number,
  iIdx: number,
  jIdx: number,
) {
  const iMask = extractSlice(input, view, iIdx, labelValue);
  const jMask = extractSlice(input, view, jIdx, labelValue);
  const iConn = connectedComponents(iMask);
  const jConn = connectedComponents(jMask);

  // Overlap pairs between components of the two slices.
  const overlaps = new Map<number, Set<number>>();
  for (let index = 0; index < iMask.data.length; index += 1) {
    const a = iConn.labels[index];
    const b = jConn.labels[index];
    if (a && b) {
      const set = overlaps.get(a) ?? new Set<number>();
      set.add(b);
      overlaps.set(a, set);
    }
  }
  const jCounts = new Map<number, number>();
  for (const set of overlaps.values()) {
    for (const b of set) jCounts.set(b, (jCounts.get(b) ?? 0) + 1);
  }

  const { nu, nv } = iMask;
  const pendingByJ = new Map<number, number[]>();
  for (const [a, bs] of overlaps) {
    const aMask = componentMask(iConn.labels, a, nu, nv);
    if (bs.size === 1) {
      const b = [...bs][0];
      if ((jCounts.get(b) ?? 0) === 1) {
        interpolate1to1(
          out, view, labelValue,
          iIdx, aMask, jIdx, componentMask(jConn.labels, b, nu, nv),
        );
      } else {
        const list = pendingByJ.get(b) ?? [];
        list.push(a);
        pendingByJ.set(b, list);
      }
      continue;
    }
    // 1-to-N split: partition the single region among its counterparts.
    const bMasks = [...bs].map((b) => componentMask(jConn.labels, b, nu, nv));
    const parts = partition(aMask, bMasks);
    for (let k = 0; k < bMasks.length; k += 1) {
      interpolate1to1(out, view, labelValue, iIdx, parts[k], jIdx, bMasks[k]);
    }
  }

  // M-to-1 merges: partition the single j region among its i counterparts.
  for (const [b, as] of pendingByJ) {
    const bMask = componentMask(jConn.labels, b, nu, nv);
    const aMasks = as.map((a) => componentMask(iConn.labels, a, nu, nv));
    const parts = partition(bMask, aMasks);
    for (let k = 0; k < aMasks.length; k += 1) {
      interpolate1to1(out, view, labelValue, iIdx, aMasks[k], jIdx, parts[k]);
    }
  }
}

export function fillBetween<T extends ImageData>(
  image: OrientedImage<T>,
  options: FillBetweenOptions,
): OrientedImage<T> {
  const { labelValue } = options;
  if (!Number.isFinite(labelValue) || labelValue === 0) {
    throw new Error('labelValue must be a nonzero finite number');
  }
  if (options.axis !== undefined && ![0, 1, 2].includes(options.axis)) {
    throw new Error('axis must be 0, 1, or 2');
  }

  const out = new (image.data.constructor as new (length: number) => T)(image.data.length);
  const slicesPerAxis = detectSegmentedSlices(image, labelValue);
  const axes = options.axis !== undefined
    ? [options.axis]
    : [0, 1, 2].filter((a) => slicesPerAxis[a].size > 1);

  for (const axis of axes) {
    const sliceIndices = [...slicesPerAxis[axis]].sort((a, b) => a - b);
    if (sliceIndices.length < 2) continue;
    const view = axisView(image.dims, axis);
    for (let pair = 0; pair + 1 < sliceIndices.length; pair += 1) {
      const [iIdx, jIdx] = [sliceIndices[pair], sliceIndices[pair + 1]];
      if (iIdx + 1 >= jIdx) continue;
      interpolateBetweenTwo(out, view, image.data, labelValue, iIdx, jIdx);
    }
  }

  // ITK final overwrite: original drawn voxels always win.
  for (let index = 0; index < image.data.length; index += 1) {
    if (image.data[index] !== 0) out[index] = image.data[index];
  }

  return {
    dims: [...image.dims],
    spacing: [...image.spacing],
    origin: [...image.origin],
    direction: image.direction.map((row) => [...row]),
    data: out,
  };
}
