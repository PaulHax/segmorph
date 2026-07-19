"""Golden fixtures for algorithm I: fractional labelmap conversions.

Reproduces PolySeg's vtkPolyDataToFractionalLabelmapFilter (BSD-2) behavior with
stock VTK filters, parameter-for-parameter:
- transform the surface world->IJK with the inverse image-to-world matrix, then
  vtkPolyDataNormals (ConsistencyOn), vtkTriangleFilter, vtkStripper
  (vtkPolyDataToFractionalLabelmapFilter.cxx lines 394-419);
- for NumberOfOffsets n=6, offsetStepSize=(n-1)/(2n), sample offsets
  idx/n - offsetStepSize per axis (cxx 443-466), i.e. 216 sub-voxel shifts;
- per offset, rasterize with vtkPolyDataToImageStencil in IJK space
  (spacing 1, origin = offset, default Tolerance 7.62939453125e-06); PolySeg's
  FillImageStencilData is a copy of that filter's ThreadedExecute;
- accumulate binary results (AddBinaryLabelMapToFractionalLabelMap, cxx 492-529).

Encoding: instead of PolySeg's signed char [-108, 108] we store float occupancy
count/216 in [0, 1] (PolySeg's own VTK_FLOAT compile-time option, filter .h 42-60).
occupancy = (polysegChar + 108) / 216.

Also writes a golden surface for the reverse direction: vtkFlyingEdges3D at
iso = threshold*(max-min)+min = 0.5 in IJK, points transformed back to world
(vtkFractionalLabelmapToClosedSurfaceConversionRule.cxx 184-203, 255-266),
with decimation 0 and smoothing skipped so the fixture pins the marching pass.
"""

import json
import math
import pathlib
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from fixtures import fixtures_root, read_manifest, write_manifest

NUMBER_OF_OFFSETS = 6
THRESHOLD = 0.5


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


CASES = {
    "isotropic": {
        "dims": [20, 20, 20],
        "spacing": [1.0, 1.0, 1.0],
        "origin": [-9.5, -9.5, -9.5],
        "direction": np.eye(3),
        "radius": 7.4,
        "center_offset": [0.3, -0.2, 0.15],
    },
    "anisotropic": {
        "dims": [24, 18, 12],
        "spacing": [0.7, 1.1, 1.9],
        "origin": [5.0, -3.0, 2.0],
        "direction": np.eye(3),
        "radius": 6.0,
        "center_offset": [0.2, 0.3, -0.4],
    },
    "oblique": {
        "dims": [16, 16, 16],
        "spacing": [1.2, 1.0, 0.8],
        "origin": [1.0, 2.0, -3.0],
        "direction": rotation([1.0, 2.0, 3.0], 33.0),
        "radius": 4.6,
        "center_offset": [-0.25, 0.15, 0.3],
    },
}


def image_to_world_matrix(case):
    matrix = np.eye(4)
    matrix[:3, :3] = case["direction"] @ np.diag(case["spacing"])
    matrix[:3, 3] = case["origin"]
    return matrix


def sphere_mesh(case):
    matrix = image_to_world_matrix(case)
    mid = (np.asarray(case["dims"], dtype=float) - 1.0) / 2.0
    center = matrix[:3, :3] @ mid + matrix[:3, 3] + np.asarray(case["center_offset"])
    source = vtk.vtkSphereSource()
    source.SetCenter(*center)
    source.SetRadius(case["radius"])
    source.SetThetaResolution(17)
    source.SetPhiResolution(13)
    source.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(source.GetOutputPort())
    triangles.Update()
    mesh = triangles.GetOutput()

    # Round through float32 so the oracle rasterizes the exact vertex values
    # the TS Mesh (Float32Array) will see.
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)
    mesh.GetPoints().SetData(numpy_to_vtk(points.astype(np.float64), deep=True))
    return mesh, points


def mesh_to_json(mesh, points):
    polys = vtk_to_numpy(mesh.GetPolys().GetData())
    return {
        "points": [float(value) for value in points.reshape(-1)],
        "polys": [int(value) for value in polys],
    }


def transform_to_ijk(mesh, case):
    matrix = image_to_world_matrix(case)
    inverse = np.linalg.inv(matrix)
    vtk_matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            vtk_matrix.SetElement(row, column, inverse[row, column])
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(mesh)
    transform_filter.SetTransform(transform)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(transform_filter.GetOutputPort())
    normals.ConsistencyOn()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(normals.GetOutputPort())
    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(triangles.GetOutputPort())
    stripper.Update()
    return stripper.GetOutput()


def fractional_counts(ijk_mesh, dims):
    counts = np.zeros(dims[::-1], dtype=np.int32)
    step = (NUMBER_OF_OFFSETS - 1.0) / (2 * NUMBER_OF_OFFSETS)
    for k in range(NUMBER_OF_OFFSETS):
        k_offset = k / NUMBER_OF_OFFSETS - step
        for j in range(NUMBER_OF_OFFSETS):
            j_offset = j / NUMBER_OF_OFFSETS - step
            for i in range(NUMBER_OF_OFFSETS):
                i_offset = i / NUMBER_OF_OFFSETS - step
                stencil = vtk.vtkPolyDataToImageStencil()
                stencil.SetInputData(ijk_mesh)
                stencil.SetOutputOrigin(i_offset, j_offset, k_offset)
                stencil.SetOutputSpacing(1.0, 1.0, 1.0)
                stencil.SetOutputWholeExtent(
                    0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
                to_image = vtk.vtkImageStencilToImage()
                to_image.SetInputConnection(stencil.GetOutputPort())
                to_image.SetInsideValue(1)
                to_image.SetOutsideValue(0)
                to_image.SetOutputScalarTypeToUnsignedChar()
                to_image.Update()
                binary = vtk_to_numpy(
                    to_image.GetOutput().GetPointData().GetScalars())
                counts += binary.reshape(dims[::-1])
    return counts


def golden_surface(occupancy, case):
    dims = case["dims"]
    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    scalars = numpy_to_vtk(
        occupancy.reshape(-1).astype(np.float64), deep=True,
        array_type=vtk.VTK_DOUBLE)
    image.GetPointData().SetScalars(scalars)

    marching = vtk.vtkFlyingEdges3D()
    marching.SetInputData(image)
    marching.SetNumberOfContours(1)
    marching.SetValue(0, THRESHOLD)
    marching.ComputeScalarsOff()
    marching.ComputeGradientsOff()
    marching.ComputeNormalsOff()
    marching.Update()

    matrix = image_to_world_matrix(case)
    vtk_matrix = vtk.vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            vtk_matrix.SetElement(row, column, matrix[row, column])
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    to_world = vtk.vtkTransformPolyDataFilter()
    to_world.SetInputConnection(marching.GetOutputPort())
    to_world.SetTransform(transform)
    # Samples exactly at the iso value give zero-area triangles; merge the
    # coincident points so the geometric diff metrics stay finite.
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(to_world.GetOutputPort())
    clean.PointMergingOn()
    clean.SetTolerance(0.0)
    clean.Update()
    mesh = clean.GetOutput()
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)
    return mesh_to_json(mesh, points)


def write_nrrd(path, occupancy, case):
    dims = case["dims"]
    directions = " ".join(
        "(" + ",".join(repr(float(case["direction"][row][axis]) * case["spacing"][axis])
                       for row in range(3)) + ")"
        for axis in range(3)
    )
    origin = "(" + ",".join(repr(float(value)) for value in case["origin"]) + ")"
    header = "\n".join([
        "NRRD0005",
        "type: float",
        "dimension: 3",
        f"sizes: {' '.join(map(str, dims))}",
        "space: right-anterior-superior",
        f"space directions: {directions}",
        "kinds: domain domain domain",
        "encoding: raw",
        "endian: little",
        f"space origin: {origin}",
        "",
        "",
    ]).encode("ascii")
    path.write_bytes(header + occupancy.astype(np.float32).tobytes(order="C"))


def case_params(case):
    return {
        "dims": list(case["dims"]),
        "spacing": [float(value) for value in case["spacing"]],
        "origin": [float(value) for value in case["origin"]],
        "direction": [[float(value) for value in row] for row in case["direction"]],
        "numberOfOffsets": NUMBER_OF_OFFSETS,
        "threshold": THRESHOLD,
        "encoding": "occupancy count/216 in [0,1]; polysegChar = occupancy*216 - 108",
        "stencilTolerance": 7.62939453125e-06,
    }


def update_manifest(path, names):
    manifest = read_manifest(path)
    keep = [
        fixture for fixture in manifest["fixtures"]
        if not (fixture["algorithm"] == "I"
                and fixture["oracle"]["name"] == "python-vtk")
    ]
    entries = [
        {
            "oracle": {"name": "python-vtk",
                       "version": vtk.vtkVersion.GetVTKVersion()},
            "algorithm": "I",
            "case": name,
            "params": case_params(CASES[name]),
            "seed": 0,
        }
        for name in names
    ]
    manifest["fixtures"] = keep + entries
    write_manifest(path, manifest)


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    fixtures = fixtures_root(root)
    for name, case in CASES.items():
        fixture = fixtures / "I" / name
        fixture.mkdir(parents=True, exist_ok=True)

        mesh, points = sphere_mesh(case)
        (fixture / "input.mesh.json").write_text(
            json.dumps(mesh_to_json(mesh, points)) + "\n")

        ijk_mesh = transform_to_ijk(mesh, case)
        counts = fractional_counts(ijk_mesh, case["dims"])
        occupancy = counts / float(NUMBER_OF_OFFSETS ** 3)
        write_nrrd(fixture / "golden.nrrd", occupancy, case)

        (fixture / "golden.surface.mesh.json").write_text(
            json.dumps(golden_surface(occupancy, case)) + "\n")
        (fixture / "params.json").write_text(
            json.dumps(case_params(case), indent=2) + "\n")
        print(name, "occupancy sum", float(occupancy.sum()))

    update_manifest(fixtures / "manifest.json", list(CASES.keys()))


if __name__ == "__main__":
    main()
