// Deterministic synthetic contour stacks for algorithm K (planar contour ->
// closed surface). The polyseg oracle runner (oracles/node/polyseg.ts) writes
// these as input.contours.json next to the golden it computes, and the specs
// read the committed files, so this module is the single source of the inputs.
//
// Every coordinate is quantized to float32 (Math.fround) because both sides of
// the comparison consume float32: the polyseg-wasm wrapper converts the input
// to std::vector<float> and the SlicerRT rule stores points in float32
// vtkPoints, and our port emits a float32 Mesh.
//
// Loops are open (no repeated closure point) world-space polylines; the
// consumer closes them implicitly, matching the polyseg-wasm wrapper, which
// appends the first point id to every cell.

/** One contour loop as interleaved world xyz. */
export type ContourCaseLoop = number[];

export type ContourToSurfaceCase = {
  loops: ContourCaseLoop[];
  /** Copied into params.json for the manifest and spec. */
  params: Record<string, unknown>;
};

const f = Math.fround;

function circle(
  centerX: number,
  centerY: number,
  z: number,
  radius: number,
  count: number,
  options?: { clockwise?: boolean; startAngle?: number; radiusY?: number },
): ContourCaseLoop {
  const radiusY = options?.radiusY ?? radius;
  const startAngle = options?.startAngle ?? 0;
  const step = ((options?.clockwise ? -1 : 1) * 2 * Math.PI) / count;
  const loop: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const angle = startAngle + index * step;
    loop.push(f(centerX + radius * Math.cos(angle)), f(centerY + radiusY * Math.sin(angle)), f(z));
  }
  return loop;
}

/** Straight tube: identical circles on 8 axial slices. */
function cylinder(): ContourToSurfaceCase {
  const loops = Array.from({ length: 8 }, (_, slice) => circle(5, -3, slice * 1.5, 20, 24));
  return {
    loops,
    params: { shape: 'cylinder', slices: 8, radius: 20, spacing: 1.5 },
  };
}

/** Sphere cross-sections: radius varies per slice, small end contours. */
function sphere(): ContourToSurfaceCase {
  const sphereRadius = 15;
  const loops: ContourCaseLoop[] = [];
  for (let slice = 0; slice < 9; slice += 1) {
    const z = -12 + slice * 3;
    const radius = Math.sqrt(sphereRadius ** 2 - z ** 2);
    loops.push(circle(0, 0, z, radius, 24));
  }
  return {
    loops,
    params: { shape: 'sphere', slices: 9, sphereRadius, spacing: 3 },
  };
}

/** Cone: radius shrinks toward the apex, exercising asymmetric point counts. */
function cone(): ContourToSurfaceCase {
  const loops: ContourCaseLoop[] = [];
  for (let slice = 0; slice < 7; slice += 1) {
    const radius = 18 - slice * (16 / 6);
    loops.push(circle(0, 0, slice * 2, radius, 20));
  }
  return {
    loops,
    params: { shape: 'cone', slices: 7, baseRadius: 18, tipRadius: 2, spacing: 2 },
  };
}

/**
 * Branching: one wide ellipse on the lower slices splits into two circles on
 * the upper slices. Both circles' xy bounds overlap the ellipse bounds, so the
 * rule's Branch division applies; the circles do not overlap each other.
 */
function branching(): ContourToSurfaceCase {
  const loops: ContourCaseLoop[] = [];
  for (let slice = 0; slice < 4; slice += 1) {
    loops.push(circle(0, 0, slice * 2, 24, 32, { radiusY: 12 }));
  }
  for (let slice = 4; slice < 8; slice += 1) {
    loops.push(circle(-12, 0, slice * 2, 8, 16));
    loops.push(circle(12, 0, slice * 2, 8, 16));
  }
  return {
    loops,
    params: {
      shape: 'branching',
      trunkSlices: 4,
      branchSlices: 4,
      spacing: 2,
    },
  };
}

/**
 * Annulus expressed with the RTSTRUCT keyhole technique: outer ring, a
 * zero-width channel of exactly duplicated point pairs, and the inner ring
 * traversed the opposite way — exercises FixKeyholes (epsilon 0.001, minimum
 * separation 3).
 */
function keyhole(): ContourToSurfaceCase {
  const outerRadius = 16;
  const innerRadius = 8;
  const channelOuter = [f(11.5), f(0)];
  const channelInner = [f(8.7), f(0)];
  const loops: ContourCaseLoop[] = [];
  for (let slice = 0; slice < 6; slice += 1) {
    const z = f(slice * 1.5);
    const loop: number[] = [];
    // Outer ring, counterclockwise.
    loop.push(...circle(0, 0, z, outerRadius, 32));
    // Channel in: two points repeated exactly on the way back out.
    loop.push(channelOuter[0], channelOuter[1], z);
    loop.push(channelInner[0], channelInner[1], z);
    // Inner ring, clockwise (hole), starting just below the +x axis.
    loop.push(...circle(0, 0, z, innerRadius, 24, { clockwise: true, startAngle: -0.2 }));
    // Channel out: exact duplicates of the inbound channel points.
    loop.push(channelInner[0], channelInner[1], z);
    loop.push(channelOuter[0], channelOuter[1], z);
    loops.push(loop);
  }
  return {
    loops,
    params: {
      shape: 'keyhole-annulus',
      slices: 6,
      outerRadius,
      innerRadius,
      spacing: 1.5,
    },
  };
}

/** Cylinder whose contour planes are tilted 20 degrees about x. */
function tilted(): ContourToSurfaceCase {
  const angle = (20 * Math.PI) / 180;
  const normal = [0, -Math.sin(angle), Math.cos(angle)];
  const uAxis = [1, 0, 0];
  const vAxis = [0, Math.cos(angle), Math.sin(angle)];
  const radius = 14;
  const count = 20;
  const loops: ContourCaseLoop[] = [];
  for (let slice = 0; slice < 7; slice += 1) {
    const offset = slice * 1.8;
    const center = normal.map((component) => component * offset);
    const loop: number[] = [];
    for (let index = 0; index < count; index += 1) {
      const theta = (2 * Math.PI * index) / count;
      const u = radius * Math.cos(theta);
      const v = radius * Math.sin(theta);
      loop.push(
        f(center[0] + u * uAxis[0] + v * vAxis[0]),
        f(center[1] + u * uAxis[1] + v * vAxis[1]),
        f(center[2] + u * uAxis[2] + v * vAxis[2]),
      );
    }
    loops.push(loop);
  }
  return {
    loops,
    params: { shape: 'tilted-cylinder', slices: 7, radius, tiltDegrees: 20, spacing: 1.8 },
  };
}

export const contourToSurfaceCases: Record<string, ContourToSurfaceCase> = {
  cylinder: cylinder(),
  sphere: sphere(),
  cone: cone(),
  branching: branching(),
  keyhole: keyhole(),
  tilted: tilted(),
};
