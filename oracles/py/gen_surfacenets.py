"""Golden fixtures for algorithm J: vtkSurfaceNets3D labelmap-to-surface extraction.

Each case is padded by one background voxel per side (vtkImageConstantPad) before
vtkSurfaceNets3D so boundary-touching foreground yields closed surfaces. Raw
vtkSurfaceNets3D treats padded-border edges as non-intersecting (verified: an
all-foreground image produces an empty mesh), so the pad is required for closure
and is a geometric no-op for interior blobs.

Two goldens per case: smoothed (VTK defaults: 16 iterations, relaxation 0.5,
constraint distance = norm(spacing)) and unsmoothed, both triangulated.
"""

import json
import pathlib
import re
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def read_nrrd(path):
    raw = path.read_bytes()
    end = raw.index(b"\n\n")
    header = raw[:end].decode("ascii").splitlines()
    fields = {}
    for line in header[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    assert fields["type"] == "uint8" and fields["encoding"] == "raw"
    dims = tuple(int(size) for size in fields["sizes"].split())
    data = np.frombuffer(raw[end + 2:], dtype=np.uint8).reshape(dims[::-1])
    return data, dims


def sphere_labelmap(dims, radius, dtype=np.uint8):
    center = tuple((dimension - 1) / 2 for dimension in dims)
    coords = np.indices(dims[::-1], dtype=np.float64)  # z, y, x order
    distance2 = sum(
        (coords[2 - axis] - center[axis]) ** 2 for axis in range(3)
    )
    return (distance2 <= radius ** 2).astype(dtype)


def boundary_blob(dims):
    data = np.zeros(dims[::-1], dtype=np.uint8)
    # Touches -x, +x, -y, and -z image faces; stays clear of +y and +z.
    data[0:6, 0:8, 0:dims[0]] = 1
    return data


def multilabel_boxes(dims):
    data = np.zeros(dims[::-1], dtype=np.uint8)
    data[4:14, 4:10, 3:12] = 1
    data[4:14, 10:16, 3:12] = 2  # shares a y-face with label 1
    return data


def rotation_z_then_x(angle_z, angle_x):
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return rx @ rz


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


def surface_nets(image, label_value, smoothing):
    dims = image.GetDimensions()
    pad = vtk.vtkImageConstantPad()
    pad.SetInputData(image)
    pad.SetOutputWholeExtent(-1, dims[0], -1, dims[1], -1, dims[2])
    pad.SetConstant(0)

    nets = vtk.vtkSurfaceNets3D()
    nets.SetInputConnection(pad.GetOutputPort())
    nets.SetValue(0, label_value)
    nets.SetOutputMeshTypeToTriangles()
    if not smoothing:
        nets.SmoothingOff()
    nets.Update()
    return nets.GetOutput()


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
        legacy_polys = vtk.vtkIdTypeArray()
        mesh.GetPolys().ExportLegacyFormat(legacy_polys)
        polys = vtk_to_numpy(legacy_polys).astype(np.uint32).ravel()
        payload = {"points": [float(value) for value in points], "polys": polys.tolist()}
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def update_manifest(path, entries):
    manifest = json.loads(path.read_text()) if path.exists() else {"schemaVersion": 1, "fixtures": []}
    own = {(entry["algorithm"], entry["case"], entry["oracle"]["name"]) for entry in entries}
    manifest["fixtures"] = [
        fixture for fixture in manifest["fixtures"]
        if (fixture["algorithm"], fixture["case"], fixture["oracle"]["name"]) not in own
    ] + entries
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    fixtures = root / "test" / "fixtures" / "J"
    vtk_version = vtk.vtkVersion.GetVTKVersion()

    sphere_data, sphere_dims = read_nrrd(root / "test" / "fixtures" / "A" / "sphere" / "input.nrrd")
    assert sphere_dims == (32, 32, 32)

    oblique_direction = tuple(
        tuple(float(value) for value in row)
        for row in rotation_z_then_x(np.deg2rad(30.0), np.deg2rad(20.0))
    )

    cases = [
        {
            "case": "sphere",
            "data": sphere_data,
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0.0, 0.0, 0.0),
            "direction": IDENTITY,
            "labelValue": 1,
            "note": "reuses the committed A sphere labelmap (dims 32, radius 10)",
        },
        {
            "case": "boundary-blob",
            "data": boundary_blob((16, 16, 16)),
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0.0, 0.0, 0.0),
            "direction": IDENTITY,
            "labelValue": 1,
            "note": "box touching -x,+x,-y,-z image faces; closure requires the pad",
        },
        {
            "case": "anisotropic",
            "data": sphere_labelmap((24, 24, 24), 8.0),
            "spacing": (0.5, 1.0, 2.0),
            "origin": (5.0, -3.0, 2.0),
            "direction": IDENTITY,
            "labelValue": 1,
            "note": "index-space sphere radius 8, anisotropic spacing",
        },
        {
            "case": "oblique",
            "data": sphere_labelmap((20, 20, 20), 6.0),
            "spacing": (1.0, 1.5, 0.8),
            "origin": (-4.0, 2.0, 7.0),
            "direction": oblique_direction,
            "labelValue": 1,
            "note": "index-space sphere radius 6, rotated direction matrix (Rz30 then Rx20)",
        },
        {
            "case": "multilabel-label1",
            "data": multilabel_boxes((20, 20, 20)),
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0.0, 0.0, 0.0),
            "direction": IDENTITY,
            "labelValue": 1,
            "note": "two face-adjacent boxes, extract label 1 only",
        },
        {
            "case": "multilabel-label2",
            "data": multilabel_boxes((20, 20, 20)),
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0.0, 0.0, 0.0),
            "direction": IDENTITY,
            "labelValue": 2,
            "note": "two face-adjacent boxes, extract label 2 only",
        },
    ]

    entries = []
    for case in cases:
        directory = fixtures / case["case"]
        directory.mkdir(parents=True, exist_ok=True)
        image = make_image(case["data"], case["spacing"], case["origin"], case["direction"])
        smoothed = surface_nets(image, case["labelValue"], smoothing=True)
        unsmoothed = surface_nets(image, case["labelValue"], smoothing=False)

        write_nrrd(directory / "input.nrrd", case["data"], case["spacing"], case["origin"], case["direction"])
        write_mesh(directory / "golden.mesh.json", smoothed)
        write_mesh(directory / "golden.unsmoothed.mesh.json", unsmoothed)

        params = {
            "labelValue": case["labelValue"],
            "backgroundLabel": 0,
            "padded": True,
            "smoothing": {
                "iterations": 16,
                "relaxationFactor": 0.5,
                "constraintDistance": float(np.linalg.norm(case["spacing"])),
            },
            "outputMeshType": "triangles",
            "triangulationStrategy": "min-edge",
            "note": case["note"],
        }
        (directory / "params.json").write_text(json.dumps(params, indent=2) + "\n")
        entries.append({
            "oracle": {"name": "python-vtk", "version": vtk_version},
            "algorithm": "J",
            "case": case["case"],
            "params": params,
            "seed": 0,
        })

    update_manifest(root / "test" / "fixtures" / "manifest.json", entries)


if __name__ == "__main__":
    main()
