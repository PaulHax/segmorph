import json
import pathlib
import sys

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from fixtures import fixtures_root, read_manifest, write_manifest
from resample_cases import CASES, DTYPES, make_values, write_nrrd


def direction_matrix(values):
    matrix = vtk.vtkMatrix3x3()
    for row in range(3):
        for column in range(3):
            matrix.SetElement(row, column, values[row][column])
    return matrix


def make_input(geometry, values):
    image = vtk.vtkImageData()
    image.SetDimensions(*geometry["dims"])
    image.SetSpacing(*geometry["spacing"])
    image.SetOrigin(*geometry["origin"])
    image.SetDirectionMatrix(direction_matrix(geometry["direction"]))
    image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(order="C"), deep=True))
    return image


def resample(image, geometry, fill_value):
    dims = geometry["dims"]
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(image)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.SetOutputExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    reslice.SetOutputSpacing(*geometry["spacing"])
    reslice.SetOutputOrigin(*geometry["origin"])
    reslice.SetOutputDirection(*sum(geometry["direction"], []))
    reslice.SetBackgroundLevel(fill_value)
    reslice.Update()
    return vtk_to_numpy(reslice.GetOutput().GetPointData().GetScalars()).reshape(dims[::-1])


def manifest_entry(name, case):
    params = {
        "input": case["input"], "output": case["output"],
        "interpolation": "nearest", "fillValue": case["fillValue"],
    }
    for key in ("dtype", "pattern"):
        if key in case:
            params[key] = case[key]
    return {
        "oracle": {"name": "python-vtk", "version": vtk.vtkVersion.GetVTKVersion()},
        "algorithm": "E", "case": name,
        "params": params,
        "seed": 0,
    }


def update_manifest(path):
    manifest = read_manifest(path)
    # Replace this generator's entries so regenerating after an oracle bump also
    # refreshes the recorded oracle version. Other oracles' entries are kept.
    manifest["fixtures"] = [
        entry for entry in manifest["fixtures"]
        if not (entry["algorithm"] == "E" and entry["case"] in CASES
                and entry["oracle"]["name"] == "python-vtk")
    ] + [manifest_entry(name, case) for name, case in CASES.items()]
    write_manifest(path, manifest)


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    fixtures = fixtures_root(root)
    for name, case in CASES.items():
        dtype_name = case.get("dtype", "uint8")
        pattern = case.get("pattern", "counter")
        fixture = fixtures / "E" / name
        fixture.mkdir(parents=True, exist_ok=True)
        values = make_values(case["input"], pattern, DTYPES[dtype_name])
        image = make_input(case["input"], values)
        write_nrrd(fixture / "input.nrrd", values, case["input"], dtype_name)
        write_nrrd(fixture / "golden.nrrd", resample(image, case["output"], case["fillValue"]),
                   case["output"], dtype_name)
        (fixture / "params.json").write_text(json.dumps({
            "input": case["input"], "output": case["output"], "fillValue": case["fillValue"],
        }, indent=2) + "\n")
    update_manifest(fixtures / "manifest.json")


if __name__ == "__main__":
    main()
