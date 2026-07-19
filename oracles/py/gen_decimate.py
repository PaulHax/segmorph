import json
import math
import pathlib
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from fixtures import fixtures_root, read_manifest, write_manifest


TARGET_REDUCTIONS = (0.5, 0.9)

ELLIPSOID_RADII = (12.0, 8.0, 6.0)
ELLIPSOID_LAT_BANDS = 40
ELLIPSOID_LON_BANDS = 52

TORUS_MAJOR_RADIUS = 10.0
TORUS_MINOR_RADIUS = 4.0
TORUS_MAJOR_BANDS = 48
TORUS_MINOR_BANDS = 24


def ellipsoid_mesh():
    radii = ELLIPSOID_RADII
    lat_bands = ELLIPSOID_LAT_BANDS
    lon_bands = ELLIPSOID_LON_BANDS
    south = 1 + (lat_bands - 1) * lon_bands

    points = np.zeros((south + 1, 3), dtype=np.float64)
    points[0] = (0.0, 0.0, radii[2])
    for ring in range(1, lat_bands):
        theta = math.pi * ring / lat_bands
        for segment in range(lon_bands):
            phi = 2.0 * math.pi * segment / lon_bands
            points[1 + (ring - 1) * lon_bands + segment] = (
                radii[0] * math.sin(theta) * math.cos(phi),
                radii[1] * math.sin(theta) * math.sin(phi),
                radii[2] * math.cos(theta),
            )
    points[south] = (0.0, 0.0, -radii[2])

    def ring_index(ring, segment):
        return 1 + (ring - 1) * lon_bands + (segment % lon_bands)

    triangles = []
    for segment in range(lon_bands):
        triangles.append((0, ring_index(1, segment), ring_index(1, segment + 1)))
    for ring in range(1, lat_bands - 1):
        for segment in range(lon_bands):
            a = ring_index(ring, segment)
            b = ring_index(ring, segment + 1)
            c = ring_index(ring + 1, segment + 1)
            d = ring_index(ring + 1, segment)
            triangles.append((a, d, c))
            triangles.append((a, c, b))
    for segment in range(lon_bands):
        triangles.append((
            south,
            ring_index(lat_bands - 1, segment + 1),
            ring_index(lat_bands - 1, segment),
        ))

    return points.astype(np.float32), np.asarray(triangles, dtype=np.uint32)


def torus_mesh():
    major_bands = TORUS_MAJOR_BANDS
    minor_bands = TORUS_MINOR_BANDS

    points = np.zeros((major_bands * minor_bands, 3), dtype=np.float64)
    for i in range(major_bands):
        u = 2.0 * math.pi * i / major_bands
        for j in range(minor_bands):
            v = 2.0 * math.pi * j / minor_bands
            radius = TORUS_MAJOR_RADIUS + TORUS_MINOR_RADIUS * math.cos(v)
            points[i * minor_bands + j] = (
                radius * math.cos(u),
                radius * math.sin(u),
                TORUS_MINOR_RADIUS * math.sin(v),
            )

    def index(i, j):
        return (i % major_bands) * minor_bands + (j % minor_bands)

    triangles = []
    for i in range(major_bands):
        for j in range(minor_bands):
            a = index(i, j)
            b = index(i + 1, j)
            c = index(i + 1, j + 1)
            d = index(i, j + 1)
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    return points.astype(np.float32), np.asarray(triangles, dtype=np.uint32)


def signed_volume(points, triangles):
    a = points[triangles[:, 0]].astype(np.float64)
    b = points[triangles[:, 1]].astype(np.float64)
    c = points[triangles[:, 2]].astype(np.float64)
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def to_polydata(points, triangles):
    polydata = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    polydata.SetPoints(vtk_points)

    legacy = np.empty((len(triangles), 4), dtype=np.int64)
    legacy[:, 0] = 3
    legacy[:, 1:] = triangles
    cells = vtk.vtkCellArray()
    id_array = vtk.vtkIdTypeArray()
    id_array.DeepCopy(numpy_to_vtk(legacy.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE))
    cells.ImportLegacyFormat(id_array)
    polydata.SetPolys(cells)
    return polydata


def read_mesh(path):
    payload = json.loads(path.read_text())
    points = np.asarray(payload["points"], dtype=np.float32).reshape(-1, 3)
    polys = np.asarray(payload["polys"], dtype=np.uint32).reshape(-1, 4)
    assert (polys[:, 0] == 3).all(), "expected triangle cells"
    return points, polys[:, 1:].copy()


def write_mesh(path, mesh):
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32).ravel()
    legacy_polys = vtk.vtkIdTypeArray()
    mesh.GetPolys().ExportLegacyFormat(legacy_polys)
    polys = vtk_to_numpy(legacy_polys).astype(np.uint32).ravel()
    payload = {"points": points.tolist(), "polys": polys.tolist()}
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def write_input(path, points, triangles):
    legacy = np.empty((len(triangles), 4), dtype=np.uint32)
    legacy[:, 0] = 3
    legacy[:, 1:] = triangles
    payload = {
        "points": points.astype(np.float32).ravel().tolist(),
        "polys": legacy.ravel().tolist(),
    }
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def decimate(polydata, target_reduction):
    decimation = vtk.vtkQuadricDecimation()
    decimation.SetInputData(polydata)
    decimation.SetTargetReduction(target_reduction)
    # Defaults confirmed from vtkQuadricDecimation.cxx: AttributeErrorMetric
    # off, VolumePreservation off, Regularize off, MaximumError VTK_DOUBLE_MAX.
    decimation.Update()
    return decimation.GetOutput()


def reduction_label(reduction):
    return f"r{round(reduction * 100):02d}"


def update_manifest(path, entries):
    manifest = read_manifest(path)
    owned = {(entry["algorithm"], entry["case"], entry["oracle"]["name"]) for entry in entries}
    manifest["fixtures"] = [
        fixture for fixture in manifest["fixtures"]
        if (fixture["algorithm"], fixture["case"], fixture["oracle"]["name"]) not in owned
    ] + entries
    write_manifest(path, manifest)


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    fixtures = fixtures_root(root)
    version = vtk.vtkVersion.GetVTKVersion()

    ellipsoid_points, ellipsoid_triangles = ellipsoid_mesh()
    torus_points, torus_triangles = torus_mesh()
    assert signed_volume(ellipsoid_points, ellipsoid_triangles) > 0
    assert signed_volume(torus_points, torus_triangles) > 0

    sphere_points, sphere_triangles = read_mesh(fixtures / "A" / "sphere" / "golden.mesh.json")

    geometries = {
        "a-sphere": {
            "points": sphere_points,
            "triangles": sphere_triangles,
            "params": {"input": "A/sphere/golden.mesh.json"},
            "write_input": False,
        },
        "ellipsoid": {
            "points": ellipsoid_points,
            "triangles": ellipsoid_triangles,
            "params": {
                "radii": list(ELLIPSOID_RADII),
                "latBands": ELLIPSOID_LAT_BANDS,
                "lonBands": ELLIPSOID_LON_BANDS,
            },
            "write_input": True,
        },
        "torus": {
            "points": torus_points,
            "triangles": torus_triangles,
            "params": {
                "majorRadius": TORUS_MAJOR_RADIUS,
                "minorRadius": TORUS_MINOR_RADIUS,
                "majorBands": TORUS_MAJOR_BANDS,
                "minorBands": TORUS_MINOR_BANDS,
            },
            "write_input": True,
        },
    }

    entries = []
    for name, geometry in geometries.items():
        directory = fixtures / "C" / name
        directory.mkdir(parents=True, exist_ok=True)
        if geometry["write_input"]:
            write_input(directory / "input.mesh.json", geometry["points"], geometry["triangles"])
        (directory / "params.json").write_text(
            json.dumps({**geometry["params"], "targetReductions": list(TARGET_REDUCTIONS)}, indent=2)
            + "\n"
        )

        polydata = to_polydata(geometry["points"], geometry["triangles"])
        for reduction in TARGET_REDUCTIONS:
            label = reduction_label(reduction)
            golden = decimate(polydata, reduction)
            write_mesh(directory / f"golden.{label}.mesh.json", golden)
            entries.append({
                "oracle": {"name": "python-vtk", "version": version},
                "algorithm": "C",
                "case": f"{name}-{label}",
                "params": {
                    **geometry["params"],
                    "targetReduction": reduction,
                    "golden": f"C/{name}/golden.{label}.mesh.json",
                    **({} if not geometry["write_input"] else {"input": f"C/{name}/input.mesh.json"}),
                },
                "seed": 0,
            })

    update_manifest(fixtures / "manifest.json", entries)


if __name__ == "__main__":
    main()
