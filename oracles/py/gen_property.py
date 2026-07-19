"""Property-based sweep for labelmapToSurface (algorithm P).

The named oracle cases pin behavior on hand-chosen shapes: a sphere, a torus, a
keyhole. They are good regressions and poor explorers -- every one of them was
written by someone who already knew which branch they wanted to exercise. This
generator instead draws labelmaps from a seeded random distribution built to
land on the awkward parts of the marching-cubes case table, runs VTK's
vtkDiscreteFlyingEdges3D over each one, and lets the spec assert the port
reproduces the oracle on inputs nobody designed.

What the distribution deliberately produces, across seeds:
  - multi-component blobs (unions of random ellipsoids that may not touch)
  - concavities and enclosed cavities (subtracted ellipsoids)
  - foreground running into the volume edge, exercising the background-border
    close that keeps edge-touching segments watertight
  - thin/sliver structures a voxel or two across
  - anisotropic spacing and oblique direction matrices, so the index->world
    transform is under test and not just the case table
  - a distractor label adjacent to the target, so extracting label N must
    ignore label M rather than treating all nonzero voxels as foreground

Goldens are the unsmoothed extraction, where our port claims to match VTK
exactly, so the sweep asserts equality rather than a tolerance and any drift is
unambiguous.

This algorithm is oracle-tier only: it is regenerated live into test/generated
and never committed, so the sweep can be wide without bloating the repository.
"""

import json
import math
import pathlib
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from fixtures import fixtures_root, read_manifest, write_manifest


ALGORITHM = "P"
ORACLE_NAME = "python-vtk"

# Wide enough to explore, small enough that the whole sweep regenerates and
# runs inside the oracle CI job in well under a minute.
SEED_COUNT = 32

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def rotation(axis, radians):
    """Rodrigues rotation about a unit axis, orthonormal to float64 round-off."""
    x, y, z = axis
    c, s = math.cos(radians), math.sin(radians)
    t = 1.0 - c
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def matmul(a, b):
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def ellipsoid_mask(dims, center, radii):
    z, y, x = np.ogrid[: dims[2], : dims[1], : dims[0]]
    return (
        ((x - center[0]) / radii[0]) ** 2
        + ((y - center[1]) / radii[1]) ** 2
        + ((z - center[2]) / radii[2]) ** 2
    ) <= 1.0


def random_case(rng, seed):
    """Draw one labelmap plus its geometry from the sweep distribution."""
    dims = tuple(int(rng.integers(18, 34)) for _ in range(3))

    # Geometry: a third identity, a third anisotropic, a third oblique. The
    # oblique rotations are composed off-axis so no case degenerates to a
    # permutation of the identity.
    geometry_kind = seed % 3
    if geometry_kind == 0:
        spacing = (1.0, 1.0, 1.0)
        direction = IDENTITY
    elif geometry_kind == 1:
        spacing = tuple(round(float(rng.uniform(0.4, 2.5)), 3) for _ in range(3))
        direction = IDENTITY
    else:
        spacing = tuple(round(float(rng.uniform(0.5, 2.0)), 3) for _ in range(3))
        direction = matmul(
            rotation((0.0, 0.0, 1.0), float(rng.uniform(0.15, 1.2))),
            rotation((1.0, 0.0, 0.0), float(rng.uniform(0.15, 1.2))),
        )

    origin = tuple(round(float(rng.uniform(-6.0, 6.0)), 3) for _ in range(3))

    # Target label varies so the extraction cannot assume 1, and a distractor
    # label sits in the same volume so it cannot assume "nonzero".
    label_value = int(rng.choice([1, 2, 7, 13]))
    distractor = label_value + 1 if label_value < 250 else 1

    data = np.zeros(dims[::-1], dtype=np.uint8)
    foreground = np.zeros(dims[::-1], dtype=bool)

    # Union of a few ellipsoids. `edge_bias` pulls some seeds' blobs into the
    # volume wall so the border-closing path is exercised.
    edge_bias = rng.random() < 0.35
    for _ in range(int(rng.integers(1, 5))):
        if edge_bias and rng.random() < 0.6:
            center = tuple(
                float(rng.choice([rng.uniform(-2.0, 2.0), dims[axis] - 1 + rng.uniform(-2.0, 2.0)]))
                for axis in range(3)
            )
        else:
            center = tuple(float(rng.uniform(0.25, 0.75) * dims[axis]) for axis in range(3))
        radii = tuple(max(1.2, float(rng.uniform(0.10, 0.32) * dims[axis])) for axis in range(3))
        foreground |= ellipsoid_mask(dims, center, radii)

    # Carve a cavity or concavity out of some seeds.
    if rng.random() < 0.45:
        center = tuple(float(rng.uniform(0.3, 0.7) * dims[axis]) for axis in range(3))
        radii = tuple(max(1.0, float(rng.uniform(0.06, 0.20) * dims[axis])) for axis in range(3))
        foreground &= ~ellipsoid_mask(dims, center, radii)

    # A thin slab on some seeds: one- and two-voxel structures are where the
    # case table and the border close are most likely to disagree.
    if rng.random() < 0.3:
        axis = int(rng.integers(0, 3))
        start = int(rng.integers(1, max(2, dims[axis] - 3)))
        thickness = int(rng.integers(1, 3))
        slab = [slice(None)] * 3
        slab[2 - axis] = slice(start, start + thickness)
        plate = np.zeros(dims[::-1], dtype=bool)
        plate[tuple(slab)] = True
        lo = tuple(int(rng.uniform(0.1, 0.4) * dims[axis]) for axis in range(3))
        hi = tuple(int(rng.uniform(0.6, 0.9) * dims[axis]) for axis in range(3))
        window = np.zeros(dims[::-1], dtype=bool)
        window[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = True
        foreground |= plate & window

    data[foreground] = label_value

    # Distractor: a shell one voxel outside the target, so a port that
    # thresholds instead of matching the label produces a visibly bigger mesh.
    neighborhood = np.zeros(dims[::-1], dtype=bool)
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1), (1, 2), (-1, 2)):
        neighborhood |= np.roll(foreground, shift, axis=axis)
    data[neighborhood & ~foreground] = distractor

    return {
        "dims": dims,
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "labelValue": label_value,
        "distractorLabel": distractor,
        "geometry": ("identity", "anisotropic", "oblique")[geometry_kind],
        "data": data,
        "voxelCount": int(foreground.sum()),
    }


def make_image(data, spacing, origin, direction):
    dims = data.shape[::-1]
    image = vtk.vtkImageData()
    image.SetDimensions(dims)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    flat = [direction[row][col] for row in range(3) for col in range(3)]
    image.SetDirectionMatrix(*flat)
    image.GetPointData().SetScalars(numpy_to_vtk(data.ravel(), deep=True))
    return image


def extract(image, label_value):
    """Unsmoothed discrete extraction, padded so edge-touching foreground closes."""
    dims = image.GetDimensions()
    pad = vtk.vtkImageConstantPad()
    pad.SetInputData(image)
    pad.SetOutputWholeExtent(-1, dims[0], -1, dims[1], -1, dims[2])
    pad.SetConstant(0)

    surface = vtk.vtkDiscreteFlyingEdges3D()
    surface.SetInputConnection(pad.GetOutputPort())
    surface.SetValue(0, label_value)
    surface.ComputeNormalsOff()
    surface.ComputeGradientsOff()
    surface.Update()
    return surface.GetOutput()


def write_nrrd(path, data, spacing, origin, direction):
    dims = data.shape[::-1]
    columns = [
        tuple(direction[row][axis] * spacing[axis] for row in range(3))
        for axis in range(3)
    ]

    def vec(values):
        return "(" + ",".join(repr(float(value)) for value in values) + ")"

    header = "\n".join([
        "NRRD0005",
        "type: uint8",
        "dimension: 3",
        f"sizes: {' '.join(map(str, dims))}",
        "space: right-anterior-superior",
        f"space directions: {' '.join(vec(column) for column in columns)}",
        "kinds: domain domain domain",
        "encoding: raw",
        "endian: little",
        f"space origin: {vec(origin)}",
        "",
        "",
    ]).encode("ascii")
    path.write_bytes(header + data.tobytes(order="C"))


def write_mesh(path, mesh):
    if mesh.GetNumberOfPoints() == 0:
        payload = {"points": [], "polys": []}
    else:
        points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32).ravel()
        legacy = vtk.vtkIdTypeArray()
        mesh.GetPolys().ExportLegacyFormat(legacy)
        polys = vtk_to_numpy(legacy).astype(np.uint32).ravel()
        payload = {"points": points.tolist(), "polys": polys.tolist()}
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    fixtures = fixtures_root(root)

    entries = []
    for seed in range(SEED_COUNT):
        case = random_case(np.random.default_rng(seed), seed)
        name = f"blob-{seed:02d}"
        directory = fixtures / ALGORITHM / name
        directory.mkdir(parents=True, exist_ok=True)

        image = make_image(case["data"], case["spacing"], case["origin"], case["direction"])
        mesh = extract(image, case["labelValue"])

        write_nrrd(
            directory / "input.nrrd",
            case["data"], case["spacing"], case["origin"], case["direction"],
        )
        write_mesh(directory / "golden.extract.mesh.json", mesh)

        params = {
            "dims": list(case["dims"]),
            "spacing": list(case["spacing"]),
            "origin": list(case["origin"]),
            "direction": [list(row) for row in case["direction"]],
            "labelValue": case["labelValue"],
            "distractorLabel": case["distractorLabel"],
            "geometry": case["geometry"],
            "voxelCount": case["voxelCount"],
            "pointCount": int(mesh.GetNumberOfPoints()),
            "triangleCount": int(mesh.GetNumberOfPolys()),
        }
        (directory / "params.json").write_text(json.dumps(params, indent=2) + "\n")

        entries.append({
            "oracle": {"name": ORACLE_NAME, "version": vtk.vtkVersion.GetVTKVersion()},
            "algorithm": ALGORITHM,
            "case": name,
            "params": params,
            "seed": seed,
        })
        print(
            f"{name}: {case['geometry']} dims={case['dims']} label={case['labelValue']} "
            f"voxels={case['voxelCount']} points={params['pointCount']} "
            f"triangles={params['triangleCount']}"
        )

    manifest_path = fixtures / "manifest.json"
    manifest = read_manifest(manifest_path)
    manifest["fixtures"] = [
        entry for entry in manifest["fixtures"] if entry["algorithm"] != ALGORITHM
    ] + entries
    write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
