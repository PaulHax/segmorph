"""Golden fixtures for meshSmooth (algorithm B).

Runs vtkWindowedSincPolyDataFilter (pinned python vtk) on the algorithm-A
sphere extraction mesh (read from the same fixture root this writes to) plus
small deterministic synthetic meshes generated here. Inputs and goldens are float32, matching the Mesh
contract. Deterministic: no randomness, no timestamps (seed 0).
"""

import json
import math
import pathlib
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from fixtures import fixtures_root, read_manifest, write_manifest


ALGORITHM = "B"
ORACLE_NAME = "python-vtk"

VTK_DEFAULTS = {
    "numberOfIterations": 20,
    "passBand": 0.1,
    "normalizeCoordinates": False,
    "windowFunction": "nuttall",
    "boundarySmoothing": True,
    "nonManifoldSmoothing": False,
    "weightNonManifoldEdges": True,
    "edgeAngle": 15.0,
    "featureEdgeSmoothing": False,
}

# Slicer's joint smoothing mapping for smoothingFactor = 0.5:
# passBand = 10^(-4 * 0.5), iterations = 20 + 40 * 0.5.
SLICER_FACTOR_HALF = {
    **VTK_DEFAULTS,
    "numberOfIterations": 40,
    "passBand": 10.0 ** (-4 * 0.5),
    "normalizeCoordinates": True,
    "boundarySmoothing": False,
    "nonManifoldSmoothing": True,
}


def cube_sphere(divisions=4, radius=8.0):
    """Coarse cube-sphere: subdivided cube lattice projected onto a sphere."""
    vertex_ids = {}
    points = []

    def vertex(lattice):
        if lattice not in vertex_ids:
            direction = np.array(lattice, dtype=np.float64) / divisions
            direction /= np.linalg.norm(direction)
            vertex_ids[lattice] = len(points)
            points.append(direction * radius)
        return vertex_ids[lattice]

    triangles = []
    n = divisions
    for axis in range(3):
        axis_u = (axis + 1) % 3
        axis_v = (axis + 2) % 3
        for sign in (1, -1):
            for u in range(-n, n):
                for v in range(-n, n):
                    corner = [0, 0, 0]
                    corner[axis] = sign * n

                    def lattice(du, dv):
                        p = list(corner)
                        p[axis_u] = u + du
                        p[axis_v] = v + dv
                        return tuple(p)

                    q00 = vertex(lattice(0, 0))
                    q10 = vertex(lattice(1, 0))
                    q11 = vertex(lattice(1, 1))
                    q01 = vertex(lattice(0, 1))
                    if sign == 1:
                        triangles += [(q00, q10, q11), (q00, q11, q01)]
                    else:
                        triangles += [(q00, q01, q11), (q00, q11, q10)]

    return np.array(points), triangles


def torus(major=8.0, minor=3.0, nu=24, nv=12):
    """Closed genus-1 torus from a parametric grid."""
    points = []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            ring = major + minor * math.cos(v)
            points.append((ring * math.cos(u), ring * math.sin(u), minor * math.sin(v)))

    triangles = []
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            triangles += [(a, b, c), (a, c, d)]

    return np.array(points), triangles


def half_sphere(radius=8.0, nu=32, rings=8, wobble=0.15):
    """Open hemisphere: pole cap plus latitude rings, boundary at the equator.

    The equator ring gets an alternating z wobble: a symmetric ring is nearly
    a fixed point of boundary-pair smoothing, so without the wobble the
    boundary-on and boundary-off goldens differ only at noise level. The
    wobbled boundary turns by about 24 degrees segment to segment (11.25
    in-plane plus the wobble slope), so the halfsphere cases run with
    edgeAngle 30 to keep boundary smoothing active.
    """
    points = [(0.0, 0.0, radius)]
    for ring in range(1, rings + 1):
        theta = (math.pi / 2) * ring / rings
        for i in range(nu):
            phi = 2 * math.pi * i / nu
            z_wobble = wobble * (-1) ** i if ring == rings else 0.0
            points.append((
                radius * math.sin(theta) * math.cos(phi),
                radius * math.sin(theta) * math.sin(phi),
                radius * math.cos(theta) + z_wobble,
            ))

    def ring_vertex(ring, i):
        return 1 + (ring - 1) * nu + (i % nu)

    triangles = []
    for i in range(nu):
        triangles.append((0, ring_vertex(1, i), ring_vertex(1, i + 1)))
    for ring in range(1, rings):
        for i in range(nu):
            a = ring_vertex(ring, i)
            b = ring_vertex(ring + 1, i)
            c = ring_vertex(ring + 1, i + 1)
            d = ring_vertex(ring, i + 1)
            triangles += [(a, b, c), (a, c, d)]

    return np.array(points), triangles


def sliver_grid():
    """Open grid with a sliver triangle and a degenerate repeated-vertex one."""
    points = []
    for y in range(2):
        for x in range(3):
            points.append((float(x), float(y), 0.05 * x * (2 - x) + 0.05 * y))
    points.append((1.5, -1e-6, 0.0))  # sliver apex, nearly colinear with edge 1-2

    triangles = [
        (0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4),
        (1, 2, 6),  # sliver triangle
        (5, 5, 2),  # degenerate: repeated vertex
    ]
    return np.array(points), triangles


def to_polydata(points_f32, polys):
    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points_f32, deep=True))
    polydata.SetPoints(vtk_points)

    cells = vtk.vtkCellArray()
    for triangle in polys:
        cells.InsertNextCell(3)
        for vertex in triangle:
            cells.InsertCellPoint(int(vertex))
    polydata.SetPolys(cells)
    return polydata


def smooth(polydata, params):
    windows = {"nuttall": 0, "blackman": 1, "hanning": 2, "hamming": 3}
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(polydata)
    smoother.SetNumberOfIterations(params["numberOfIterations"])
    smoother.SetPassBand(params["passBand"])
    smoother.SetWindowFunction(windows[params["windowFunction"]])
    smoother.SetNormalizeCoordinates(params["normalizeCoordinates"])
    smoother.SetBoundarySmoothing(params["boundarySmoothing"])
    smoother.SetNonManifoldSmoothing(params["nonManifoldSmoothing"])
    smoother.SetWeightNonManifoldEdges(params["weightNonManifoldEdges"])
    smoother.SetEdgeAngle(params["edgeAngle"])
    smoother.SetFeatureEdgeSmoothing(params["featureEdgeSmoothing"])
    smoother.Update()
    return smoother.GetOutput()


def write_mesh(path, points_f32, polys_u32):
    payload = {
        "points": [float(value) for value in points_f32.ravel()],
        "polys": [int(value) for value in polys_u32.ravel()],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def polydata_arrays(polydata):
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
    legacy = vtk.vtkIdTypeArray()
    polydata.GetPolys().ExportLegacyFormat(legacy)
    polys = vtk_to_numpy(legacy).astype(np.uint32)
    return points, polys


def read_mesh(path):
    payload = json.loads(path.read_text())
    points = np.array(payload["points"], dtype=np.float32).reshape(-1, 3)
    flat = payload["polys"]
    polys = [tuple(flat[i + 1 : i + 4]) for i in range(0, len(flat), 4)]
    return points, polys


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    fixtures = fixtures_root(root)
    sphere_points, sphere_polys = read_mesh(fixtures / "A" / "sphere" / "golden.extract.mesh.json")

    # The "-hamming" cases exist for the vtk.js second oracle: vtk.js's
    # WindowedSincPolyDataFilter hardcodes the Hamming window, so cross-checking
    # against it requires the Hamming window here too. Everything else matches
    # VTK_DEFAULTS.
    hamming = {**VTK_DEFAULTS, "windowFunction": "hamming"}

    synthetic = {
        "cubesphere-default": (cube_sphere, VTK_DEFAULTS),
        "cubesphere-hamming": (cube_sphere, hamming),
        "torus-default": (torus, VTK_DEFAULTS),
        "torus-hamming": (torus, hamming),
        "halfsphere-default": (half_sphere, {**VTK_DEFAULTS, "edgeAngle": 30.0}),
        "halfsphere-noboundary": (
            half_sphere,
            {**VTK_DEFAULTS, "edgeAngle": 30.0, "boundarySmoothing": False},
        ),
        "sliver-default": (sliver_grid, VTK_DEFAULTS),
    }
    sphere_cases = {
        "sphere-default": VTK_DEFAULTS,
        "sphere-slicer": SLICER_FACTOR_HALF,
        "sphere-hamming": hamming,
    }

    manifest_path = fixtures / "manifest.json"
    manifest = read_manifest(manifest_path)
    own_cases = set(synthetic) | set(sphere_cases)
    manifest["fixtures"] = [
        entry for entry in manifest["fixtures"]
        if not (entry["algorithm"] == ALGORITHM
                and entry["case"] in own_cases
                and entry["oracle"]["name"] == ORACLE_NAME)
    ]

    def emit(case, params, points_f32, polys, input_source):
        case_dir = fixtures / ALGORITHM / case
        case_dir.mkdir(parents=True, exist_ok=True)

        polydata = to_polydata(points_f32.reshape(-1, 3), polys)
        golden_points, golden_polys = polydata_arrays(smooth(polydata, params))
        write_mesh(case_dir / "golden.mesh.json", golden_points, golden_polys)

        if input_source == "generated":
            flat_polys = np.array(
                [value for triangle in polys for value in (3, *triangle)],
                dtype=np.uint32,
            )
            write_mesh(case_dir / "input.mesh.json", points_f32, flat_polys)

        case_params = {**params, "input": input_source}
        (case_dir / "params.json").write_text(json.dumps(case_params, indent=2) + "\n")
        manifest["fixtures"].append({
            "oracle": {"name": ORACLE_NAME, "version": vtk.vtkVersion.GetVTKVersion()},
            "algorithm": ALGORITHM,
            "case": case,
            "params": case_params,
            "seed": 0,
        })

    for case, params in sphere_cases.items():
        emit(case, params, sphere_points, sphere_polys, "A/sphere/golden.extract.mesh.json")

    for case, (generator, params) in synthetic.items():
        points, polys = generator()
        emit(case, params, points.astype(np.float32), polys, "generated")

    write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
