import json
import os
import pathlib
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


DIMS = (32, 32, 32)
RADIUS = 10.0
SMOOTHING_FACTOR = 0.5
PASS_BAND = 10 ** (-4 * SMOOTHING_FACTOR)
ITERATIONS = round(20 + 40 * SMOOTHING_FACTOR)


def sphere_labelmap():
    center = tuple((dimension - 1) / 2 for dimension in DIMS)
    data = np.zeros(DIMS[::-1], dtype=np.uint8)
    for z in range(DIMS[2]):
        for y in range(DIMS[1]):
            for x in range(DIMS[0]):
                distance_squared = sum(
                    (coordinate - midpoint) ** 2
                    for coordinate, midpoint in zip((x, y, z), center)
                )
                if distance_squared <= RADIUS ** 2:
                    data[z, y, x] = 1
    return data


def extract_surface(data):
    image = vtk.vtkImageData()
    image.SetDimensions(DIMS)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.GetPointData().SetScalars(numpy_to_vtk(data.ravel(), deep=True))

    pad = vtk.vtkImageConstantPad()
    pad.SetInputData(image)
    pad.SetOutputWholeExtent(-1, DIMS[0], -1, DIMS[1], -1, DIMS[2])
    pad.SetConstant(0)

    surface = vtk.vtkDiscreteFlyingEdges3D()
    surface.SetInputConnection(pad.GetOutputPort())
    surface.SetValue(0, 1)
    surface.ComputeNormalsOff()
    surface.ComputeGradientsOff()
    surface.Update()

    extracted = vtk.vtkPolyData()
    extracted.DeepCopy(surface.GetOutput())

    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(extracted)
    smooth.SetNumberOfIterations(ITERATIONS)
    smooth.SetPassBand(PASS_BAND)
    smooth.SetFeatureAngle(60.0)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    return extracted, smooth.GetOutput()


def write_nrrd(path, data):
    header = "\n".join([
        "NRRD0005",
        "type: uint8",
        "dimension: 3",
        f"sizes: {' '.join(map(str, DIMS))}",
        "space: right-anterior-superior",
        "space directions: (1,0,0) (0,1,0) (0,0,1)",
        "kinds: domain domain domain",
        "encoding: raw",
        "endian: little",
        "space origin: (0,0,0)",
        "",
        "",
    ]).encode("ascii")
    path.write_bytes(header + data.tobytes(order="C"))


def write_mesh(path, mesh):
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32).ravel()
    legacy_polys = vtk.vtkIdTypeArray()
    mesh.GetPolys().ExportLegacyFormat(legacy_polys)
    polys = vtk_to_numpy(legacy_polys).astype(np.uint32).ravel()
    payload = {"points": points.tolist(), "polys": polys.tolist()}
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def update_manifest(path):
    manifest = json.loads(path.read_text()) if path.exists() else {"schemaVersion": 1, "fixtures": []}
    entry = {
        "oracle": {"name": "python-vtk", "version": vtk.vtkVersion.GetVTKVersion()},
        "algorithm": "A",
        "case": "sphere",
        "params": {
            "dims": list(DIMS),
            "radius": RADIUS,
            "smoothingFactor": SMOOTHING_FACTOR,
            "passBand": PASS_BAND,
            "iterations": ITERATIONS,
            "decimation": 0.0,
        },
        "seed": 0,
    }
    manifest["fixtures"] = [
        fixture for fixture in manifest["fixtures"]
        if not (fixture["algorithm"] == "A" and fixture["case"] == "sphere"
                and fixture["oracle"]["name"] == "python-vtk")
    ] + [entry]
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def fixtures_root(root):
    override = os.environ.get("SEGMORPH_FIXTURES_DIR")
    return pathlib.Path(override) if override else root / "test" / "fixtures"


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    fixtures = fixtures_root(root)
    fixture = fixtures / "A" / "sphere"
    fixture.mkdir(parents=True, exist_ok=True)
    data = sphere_labelmap()
    # passBand and iterations are derived from smoothingFactor by Slicer's
    # conversion rule; recording them here lets specs consume the oracle's
    # actual filter parameters instead of re-deriving the mapping.
    params = {
        "radius": RADIUS,
        "smoothingFactor": SMOOTHING_FACTOR,
        "passBand": PASS_BAND,
        "iterations": ITERATIONS,
        "decimation": 0.0,
    }

    extracted, smoothed = extract_surface(data)
    write_nrrd(fixture / "input.nrrd", data)
    write_mesh(fixture / "golden.extract.mesh.json", extracted)
    write_mesh(fixture / "golden.mesh.json", smoothed)
    (fixture / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    update_manifest(fixtures / "manifest.json")


if __name__ == "__main__":
    main()
