"""Golden fixtures for algorithm E: independent ITK nearest-neighbor resample.

Second oracle for E. ITK's ResampleImageFilter is a separate lineage from VTK,
so agreeing with the vtkImageReslice golden (gen_resample.py) confirms neither
a shared VTK bug nor a mis-read reslice parameter is baked into the golden our
port is tested against.

ITK resamples each output voxel by mapping its physical location (identity
transform) back into the input and taking the nearest sample; outside the input
buffer it returns the default pixel value. This matches vtkImageReslice's
nearest mode and background level on every case here except upper-edge
half-voxel ties: VTK's Border=on clamps a tie at index == size-0.5 to the edge
voxel, while ITK rounds it outside and fills. The "half-voxel-border-ties" case
pins exactly that divergence; the spec asserts it rather than hiding it.

Usage: uv run --project oracles/py python oracles/py/gen_resample_itk.py <repo-root>
"""

import json
import pathlib
import sys

import numpy as np
import itk

from resample_cases import CASES, DTYPES, make_values, write_nrrd

ORACLE_NAME = "itk"


def resample(values, input_geometry, output_geometry, fill_value):
    image = itk.image_from_array(np.ascontiguousarray(values))
    image.SetSpacing([float(s) for s in input_geometry["spacing"]])
    image.SetOrigin([float(o) for o in input_geometry["origin"]])
    image.SetDirection(itk.matrix_from_array(np.asarray(input_geometry["direction"], dtype=float)))

    resampled = itk.resample_image_filter(
        image,
        transform=itk.IdentityTransform[itk.D, 3].New(),
        interpolator=itk.NearestNeighborInterpolateImageFunction.New(image),
        size=[int(d) for d in output_geometry["dims"]],
        output_spacing=[float(s) for s in output_geometry["spacing"]],
        output_origin=[float(o) for o in output_geometry["origin"]],
        output_direction=itk.matrix_from_array(
            np.asarray(output_geometry["direction"], dtype=float)
        ),
        default_pixel_value=int(fill_value),
    )
    return np.asarray(resampled)


def manifest_entry(name, case):
    params = {
        "input": case["input"], "output": case["output"],
        "interpolation": "nearest", "fillValue": case["fillValue"],
    }
    for key in ("dtype", "pattern"):
        if key in case:
            params[key] = case[key]
    return {
        "oracle": {"name": ORACLE_NAME, "version": itk.Version.GetITKVersion()},
        "algorithm": "E", "case": name,
        "params": params,
        "seed": 0,
    }


def update_manifest(path):
    manifest = json.loads(path.read_text())
    assert manifest["schemaVersion"] == 1 and isinstance(manifest["fixtures"], list)
    # Replace this generator's entries so regenerating after an oracle bump also
    # refreshes the recorded oracle version. The python-vtk goldens are kept.
    manifest["fixtures"] = [
        entry for entry in manifest["fixtures"]
        if not (entry["algorithm"] == "E" and entry["case"] in CASES
                and entry["oracle"]["name"] == ORACLE_NAME)
    ] + [manifest_entry(name, case) for name, case in CASES.items()]
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    root = pathlib.Path(sys.argv[1]).resolve()
    for name, case in CASES.items():
        dtype_name = case.get("dtype", "uint8")
        pattern = case.get("pattern", "counter")
        fixture = root / "test" / "fixtures" / "E" / name
        fixture.mkdir(parents=True, exist_ok=True)
        values = make_values(case["input"], pattern, DTYPES[dtype_name])
        golden = resample(values, case["input"], case["output"], case["fillValue"])
        write_nrrd(fixture / "golden.itk.nrrd", golden.astype(DTYPES[dtype_name]),
                   case["output"], dtype_name)
    update_manifest(root / "test" / "fixtures" / "manifest.json")


if __name__ == "__main__":
    main()
