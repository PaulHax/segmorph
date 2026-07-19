import json
import pathlib
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from fixtures import fixtures_root, read_manifest, write_manifest


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# 3-4-5 rotation about z: orthonormal to within one ulp in doubles.
OBLIQUE = [[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]

# Each case voxelizes the shared sphere mesh onto a reference grid. The sphere
# case keeps the historical trivial geometry (unit spacing, identity direction,
# origin 0); the others exercise oblique and anisotropic grids so transform and
# bounds bugs in voxelization become visible against an external VTK oracle
# rather than only a self-referential round trip. Grids whose "centered" flag is
# set are positioned around the mesh bounding-box center at generation time.
CASES = {
    "sphere": {
        "dims": [32, 32, 32], "spacing": [1.0, 1.0, 1.0],
        "direction": IDENTITY, "origin": [0.0, 0.0, 0.0],
    },
    "oblique": {
        "dims": [26, 26, 26], "spacing": [1.0, 1.0, 1.0],
        "direction": OBLIQUE, "centered": True,
    },
    "anisotropic": {
        "dims": [18, 32, 24], "spacing": [1.4, 0.75, 1.05],
        "direction": IDENTITY, "centered": True,
    },
    "oblique-anisotropic": {
        "dims": [20, 28, 18], "spacing": [1.25, 0.9, 1.5],
        "direction": OBLIQUE, "centered": True,
    },
}


def read_points_and_mesh(path):
    payload = json.loads(path.read_text())
    points = np.asarray(payload["points"], dtype=np.float64).reshape((-1, 3))
    polys = np.asarray(payload["polys"], dtype=np.int64)
    return points, polys


def geometry_for(case, points):
    dims = case["dims"]
    spacing = np.asarray(case["spacing"], dtype=np.float64)
    direction = np.asarray(case["direction"], dtype=np.float64)
    if case.get("centered"):
        center = (points.min(axis=0) + points.max(axis=0)) / 2.0
        half = spacing * ((np.asarray(dims, dtype=np.float64) - 1.0) / 2.0)
        origin = center - direction @ half
    else:
        origin = np.asarray(case["origin"], dtype=np.float64)
    return {
        "dims": list(dims),
        "spacing": spacing.tolist(),
        "origin": origin.tolist(),
        "direction": direction.tolist(),
    }


def build_mesh(points, polys):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=True))
    legacy_polys = numpy_to_vtk(polys, deep=True, array_type=vtk.VTK_ID_TYPE)
    cells = vtk.vtkCellArray()
    cells.ImportLegacyFormat(legacy_polys)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)
    mesh.SetPolys(cells)
    return mesh


def voxelize(points, polys, geometry):
    # vtkPolyDataToImageStencil has no direction matrix: it samples voxel centers
    # at outputOrigin + outputSpacing * index. Transform the mesh into the grid's
    # aligned frame (p' = R^T (p - origin)) so the axis-aligned stencil evaluates
    # exactly the oriented grid's world sample points.
    direction = np.asarray(geometry["direction"], dtype=np.float64)
    origin = np.asarray(geometry["origin"], dtype=np.float64)
    aligned = (points - origin) @ direction

    dims = geometry["dims"]
    mesh = build_mesh(aligned, polys)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(mesh)
    stencil.SetOutputOrigin(0.0, 0.0, 0.0)
    stencil.SetOutputSpacing(*geometry["spacing"])
    stencil.SetOutputWholeExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)

    to_image = vtk.vtkImageStencilToImage()
    to_image.SetInputConnection(stencil.GetOutputPort())
    to_image.SetInsideValue(1)
    to_image.SetOutsideValue(0)
    to_image.SetOutputScalarTypeToUnsignedChar()
    to_image.Update()
    return vtk_to_numpy(
        to_image.GetOutput().GetPointData().GetScalars()
    ).reshape(dims[::-1])


def write_nrrd(path, data, geometry):
    columns = [
        [geometry["direction"][row][axis] * geometry["spacing"][axis] for row in range(3)]
        for axis in range(3)
    ]
    vectors = " ".join(f"({','.join(map(str, column))})" for column in columns)
    header = "\n".join([
        "NRRD0005", "type: uint8", "dimension: 3",
        f"sizes: {' '.join(map(str, geometry['dims']))}",
        "space: right-anterior-superior", f"space directions: {vectors}",
        "kinds: domain domain domain", "encoding: raw", "endian: little",
        f"space origin: ({','.join(map(str, geometry['origin']))})", "", "",
    ]).encode("ascii")
    path.write_bytes(header + data.tobytes(order="C"))


def manifest_entry(name, geometry):
    return {
        "oracle": {"name": "python-vtk", "version": vtk.vtkVersion.GetVTKVersion()},
        "algorithm": "D",
        "case": name,
        "params": {
            "dims": geometry["dims"],
            "spacing": geometry["spacing"],
            "origin": geometry["origin"],
            "direction": geometry["direction"],
            "insideValue": 1,
            "outsideValue": 0,
        },
        "seed": 0,
    }


def update_manifest(path, geometries):
    manifest = read_manifest(path)
    # Replace this generator's entries so regenerating after an oracle bump also
    # refreshes the recorded oracle version. Other oracles' entries (including
    # the composed PolySeg one for the sphere case) are kept.
    manifest["fixtures"] = [
        entry for entry in manifest["fixtures"]
        if not (entry["algorithm"] == "D" and entry["case"] in geometries
                and entry["oracle"]["name"] == "python-vtk")
    ] + [manifest_entry(name, geometry) for name, geometry in geometries.items()]
    write_manifest(path, manifest)


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    fixtures = fixtures_root(root)
    source_mesh = fixtures / "A" / "sphere" / "golden.mesh.json"
    mesh_json = source_mesh.read_text()
    points, polys = read_points_and_mesh(source_mesh)

    geometries = {}
    for name, case in CASES.items():
        geometry = geometry_for(case, points)
        geometries[name] = geometry
        fixture = fixtures / "D" / name
        fixture.mkdir(parents=True, exist_ok=True)
        (fixture / "input.mesh.json").write_text(mesh_json)
        write_nrrd(fixture / "golden.nrrd", voxelize(points, polys, geometry), geometry)
        (fixture / "params.json").write_text(json.dumps(geometry, indent=2) + "\n")

    update_manifest(fixtures / "manifest.json", geometries)


if __name__ == "__main__":
    main()
