"""Golden fixtures for algorithm G: planar contour -> binary labelmap rasterization.

Oracle: vtkLassoStencilSource (polygon shape, z slices) + vtkImageStencilToImage,
the canonical VTK per-slice polygon scan fill (vtkImageStencilRaster).

Conventions reproduced by the TS port (src/convert/contourToLabelmap.ts):
- pixel centers at integer index coordinates,
- half-open spans (x1, x2]: fill index i when floor(x1) + 1 <= i <= floor(x2),
- half-open rows (y1, y2] for edge crossings,
- horizontal edges skipped.

VTK additionally dilates every span and row interval by VTK_STENCIL_TOL
(7.62939453125e-06); the TS port uses zero tolerance so that shared polygon
edges never double-fill. The two agree exactly whenever no scanline crossing or
edge endpoint lies within the tolerance of an integer voxel center. Every
fixture except "gridline" is generated with all crossings and edge rows at
least MIN_CENTER_CLEARANCE from integer centers (verified below and printed as
the calibration record), so exact voxel equality is expected. The "gridline"
case intentionally puts polygon edges exactly on voxel-center gridlines to pin
the convention difference: VTK's tolerance turns its half-open spans into
closed spans there, so VTK also fills the min-x column and min-y row.

vtkImageStencilData/vtkLassoStencilSource support a single polygon per slice,
so even-odd multi-loop cases are composed here per contour by XOR of per-loop
fills (equivalent to even-odd for non-self-intersecting loops), and multiple
contours combine by union - the same composition contract as the TS port.

Usage: oracles/py/.venv/bin/python oracles/py/gen_rasterize.py <repo-root>
"""

import json
import math
import pathlib
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from fixtures import fixtures_root, read_manifest, write_manifest

STENCIL_TOL = 7.62939453125e-06
MIN_CENTER_CLEARANCE = 1e-3

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def regular_loop(cu, cv, ru, rv, sides, phase):
    return [
        (cu + ru * math.cos(2.0 * math.pi * m / sides + phase),
         cv + rv * math.sin(2.0 * math.pi * m / sides + phase))
        for m in range(sides)
    ]


def oblique_direction():
    ax = math.radians(17.0)
    az = math.radians(31.0)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(ax), -math.sin(ax)],
        [0.0, math.sin(ax), math.cos(ax)],
    ])
    rz = np.array([
        [math.cos(az), -math.sin(az), 0.0],
        [math.sin(az), math.cos(az), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return (rz @ rx).tolist()


CASES = [
    {
        "name": "convex-pentagon",
        "dims": [24, 20, 5],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": IDENTITY,
        "contours": [
            {"slice": 2, "loops": [regular_loop(11.3, 9.4, 7.6, 7.6, 5, 0.37)]},
        ],
    },
    {
        "name": "concave-l",
        "dims": [20, 18, 3],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": IDENTITY,
        "contours": [
            {"slice": 1, "loops": [[
                (2.3, 2.4), (14.6, 2.4), (14.6, 7.5),
                (7.4, 7.5), (7.4, 13.6), (2.3, 13.6),
            ]]},
        ],
    },
    {
        "name": "nested-hole",
        "dims": [20, 20, 3],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": IDENTITY,
        "contours": [
            {"slice": 1, "loops": [
                [(2.4, 2.4), (15.6, 2.4), (15.6, 15.6), (2.4, 15.6)],
                [(6.5, 6.5), (6.5, 11.5), (11.5, 11.5), (11.5, 6.5)],
            ]},
        ],
    },
    {
        "name": "disjoint-loops",
        "dims": [24, 16, 3],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": IDENTITY,
        "contours": [
            {"slice": 1, "loops": [
                [(3.2, 2.7), (9.8, 3.4), (5.6, 11.3)],
                [(13.4, 4.6), (20.2, 3.8), (21.1, 10.7), (14.3, 12.2)],
            ]},
        ],
    },
    {
        "name": "multi-slice",
        "dims": [20, 20, 4],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": IDENTITY,
        "contours": [
            {"slice": 0, "loops": [regular_loop(9.5, 9.5, 7.3, 7.3, 8, 0.21)]},
            {"slice": 1, "loops": [regular_loop(9.5, 9.5, 5.6, 5.6, 8, 0.21)]},
            {"slice": 2, "loops": [regular_loop(9.5, 9.5, 3.9, 3.9, 8, 0.21)]},
        ],
    },
    {
        "name": "anisotropic",
        "dims": [24, 12, 4],
        "spacing": [0.5, 2.0, 3.0],
        "origin": [-4.25, 3.5, -6.0],
        "direction": IDENTITY,
        "contours": [
            {"slice": 2, "loops": [regular_loop(5.85, 11.3, 4.3, 7.9, 8, 0.4)]},
        ],
    },
    {
        "name": "gridline",
        "dims": [12, 10, 3],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": IDENTITY,
        "skip_clearance_check": True,
        "contours": [
            {"slice": 1, "loops": [[(3.0, 2.0), (8.0, 2.0), (8.0, 6.0), (3.0, 6.0)]]},
        ],
    },
    {
        "name": "subvoxel",
        "dims": [12, 12, 3],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": IDENTITY,
        "contours": [
            {"slice": 1, "loops": [
                [(4.7, 4.8), (5.4, 4.7), (5.05, 5.45)],
                [(8.45, 8.3), (8.8, 8.25), (8.6, 8.6)],
            ]},
        ],
    },
    {
        "name": "oblique-pentagon",
        "dims": [24, 20, 5],
        "spacing": [1.2, 0.8, 2.5],
        "origin": [10.5, -22.25, 4.75],
        "direction": oblique_direction(),
        "contours": [
            {"slice": 2, "loops": [regular_loop(13.56, 7.52, 8.4, 6.1, 5, 0.37)]},
        ],
    },
]


def direction_column(direction, axis):
    return [direction[0][axis], direction[1][axis], direction[2][axis]]


def is_identity(direction):
    return direction == IDENTITY


def rasterize_loop(dims, spacing, lasso_origin, slice_index, loop):
    points = vtk.vtkPoints()
    z = lasso_origin[2] + slice_index * spacing[2]
    for u, v in loop:
        points.InsertNextPoint(lasso_origin[0] + u, lasso_origin[1] + v, z)

    lasso = vtk.vtkLassoStencilSource()
    lasso.SetShapeToPolygon()
    lasso.SetSliceOrientation(2)
    lasso.SetSlicePoints(slice_index, points)
    lasso.SetOutputOrigin(lasso_origin)
    lasso.SetOutputSpacing(spacing)
    lasso.SetOutputWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)

    to_image = vtk.vtkImageStencilToImage()
    to_image.SetInputConnection(lasso.GetOutputPort())
    to_image.SetInsideValue(1)
    to_image.SetOutsideValue(0)
    to_image.SetOutputScalarTypeToUnsignedChar()
    to_image.Update()
    flat = vtk_to_numpy(to_image.GetOutput().GetPointData().GetScalars())
    return flat.reshape(dims[::-1]).astype(bool)


def rasterize_case(case):
    dims = case["dims"]
    spacing = case["spacing"]
    lasso_origin = case["origin"] if is_identity(case["direction"]) else [0.0, 0.0, 0.0]
    golden = np.zeros(dims[::-1], dtype=bool)
    for contour in case["contours"]:
        mask = np.zeros(dims[::-1], dtype=bool)
        for loop in contour["loops"]:
            mask ^= rasterize_loop(dims, spacing, lasso_origin, contour["slice"], loop)
        golden |= mask
    return golden.astype(np.uint8)


def clearance(case):
    """Min distance of scanline crossings and edge endpoint rows from integer
    voxel centers, in index units. Must exceed MIN_CENTER_CLEARANCE for the
    zero-tolerance TS port to agree exactly with VTK's toleranced raster."""
    spacing = case["spacing"]
    best = math.inf
    for contour in case["contours"]:
        for loop in contour["loops"]:
            index_loop = [(u / spacing[0], v / spacing[1]) for u, v in loop]
            count = len(index_loop)
            for start in range(count):
                x0, y0 = index_loop[start]
                x1, y1 = index_loop[(start + 1) % count]
                best = min(best, abs(y0 - round(y0)))
                if y0 == y1:
                    continue
                lo, hi = min(y0, y1), max(y0, y1)
                for row in range(math.floor(lo) + 1, math.floor(hi) + 1):
                    crossing = x0 + (row - y0) * (x1 - x0) / (y1 - y0)
                    best = min(best, abs(crossing - round(crossing)))
    return best


def write_nrrd(path, case, data):
    dims = case["dims"]
    spacing = case["spacing"]
    columns = [
        [component * spacing[axis] for component in direction_column(case["direction"], axis)]
        for axis in range(3)
    ]
    vectors = " ".join(f"({','.join(repr(c) for c in column)})" for column in columns)
    origin = f"({','.join(repr(c) for c in case['origin'])})"
    header = "\n".join([
        "NRRD0005",
        "type: uint8",
        "dimension: 3",
        f"sizes: {' '.join(map(str, dims))}",
        "space: right-anterior-superior",
        f"space directions: {vectors}",
        "kinds: domain domain domain",
        "encoding: raw",
        "endian: little",
        f"space origin: {origin}",
        "",
        "",
    ]).encode("ascii")
    path.write_bytes(header + data.tobytes(order="C"))


def contour_json(case, contour):
    direction = case["direction"]
    origin = case["origin"]
    spacing = case["spacing"]
    k_axis = direction_column(direction, 2)
    z = contour["slice"] * spacing[2]
    plane_origin = [origin[axis] + z * k_axis[axis] for axis in range(3)]
    return {
        "plane": {
            "origin": plane_origin,
            "xAxis": direction_column(direction, 0),
            "yAxis": direction_column(direction, 1),
        },
        "loops": [[coord for point in loop for coord in point] for loop in contour["loops"]],
    }


def geometry_params(case):
    return {
        "dims": case["dims"],
        "spacing": case["spacing"],
        "origin": case["origin"],
        "direction": case["direction"],
    }


def update_manifest(path, entries):
    manifest = read_manifest(path)
    mine = {(entry["algorithm"], entry["case"], entry["oracle"]["name"]) for entry in entries}
    manifest["fixtures"] = [
        fixture for fixture in manifest["fixtures"]
        if (fixture["algorithm"], fixture["case"], fixture["oracle"]["name"]) not in mine
    ] + entries
    write_manifest(path, manifest)


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    fixtures = fixtures_root(root)
    entries = []
    for case in CASES:
        if not case.get("skip_clearance_check"):
            measured = clearance(case)
            print(f"{case['name']}: min center clearance {measured:.6f} index units")
            assert measured > MIN_CENTER_CLEARANCE, (case["name"], measured)
        else:
            print(f"{case['name']}: gridline case, clearance check skipped by design")

        golden = rasterize_case(case)
        print(f"{case['name']}: {int(golden.sum())} filled voxels")

        case_dir = fixtures / "G" / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "input.contours.json").write_text(json.dumps({
            "labelValue": 1,
            "contours": [contour_json(case, contour) for contour in case["contours"]],
        }, indent=2) + "\n")
        write_nrrd(case_dir / "golden.nrrd", case, golden)
        (case_dir / "params.json").write_text(
            json.dumps(geometry_params(case), indent=2) + "\n")

        entries.append({
            "oracle": {"name": "python-vtk", "version": vtk.vtkVersion.GetVTKVersion()},
            "algorithm": "G",
            "case": case["name"],
            "params": {
                **geometry_params(case),
                "labelValue": 1,
                "shape": "polygon",
                "stencilTolerance": STENCIL_TOL,
                "composition": "xor-per-loop, union-per-contour",
            },
            "seed": 0,
        })

    update_manifest(fixtures / "manifest.json", entries)


if __name__ == "__main__":
    main()
