"""Shared case definitions for algorithm E (oriented nearest-neighbor resample).

Both resample oracles consume these: the per-filter vtk reslice generator
(gen_resample.py) and the independent ITK generator (gen_resample_itk.py). Pure
data plus numpy helpers, so importing it pulls in neither vtk nor itk.
"""

import numpy as np


IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
QUARTER_TURN = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
# 3-4-5 rotations about z: orthonormal to within one ulp in doubles.
OBLIQUE_A = [[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]
OBLIQUE_B = [[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]]
# Cyclic axis permutation (determinant +1): i -> world y, j -> world z, k -> world x.
PERMUTED = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
# Mirror across x (determinant -1).
MIRRORED = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

DTYPES = {"uint8": np.uint8, "uint16": np.uint16}

CASES = {
    "anisotropic-translated": {
        "input": {
            "dims": [4, 3, 2], "spacing": [0.7, 1.3, 2.1],
            "origin": [2.0, -3.0, 5.0], "direction": IDENTITY,
        },
        "output": {
            "dims": [5, 3, 2], "spacing": [0.7, 1.3, 2.1],
            "origin": [2.7, -3.0, 5.0], "direction": IDENTITY,
        },
        "fillValue": 241,
    },
    "oblique": {
        "input": {
            "dims": [4, 3, 2], "spacing": [0.8, 1.1, 1.7],
            "origin": [10.0, 20.0, -4.0], "direction": IDENTITY,
        },
        "output": {
            "dims": [3, 4, 2], "spacing": [1.1, 0.8, 1.7],
            "origin": [12.2, 20.0, -4.0], "direction": QUARTER_TURN,
        },
        "fillValue": 241,
    },
    "oblique-anisotropic": {
        "input": {
            "dims": [5, 4, 3], "spacing": [0.7, 1.3, 2.1],
            "origin": [1, 2, 3], "direction": OBLIQUE_A,
        },
        "output": {
            "dims": [4, 4, 3], "spacing": [1.1, 0.9, 1.6],
            "origin": [1.3, 2.4, 3.5], "direction": OBLIQUE_B,
        },
        "fillValue": 17,
    },
    "axis-permutation": {
        "input": {
            "dims": [4, 3, 2], "spacing": [1, 1, 1],
            "origin": [0, 0, 0], "direction": IDENTITY,
        },
        "output": {
            "dims": [3, 2, 4], "spacing": [1, 1, 1],
            "origin": [0, 0, 0], "direction": PERMUTED,
        },
        "fillValue": 7,
    },
    "upsample-half-spacing": {
        "input": {
            "dims": [4, 3, 2], "spacing": [1.0, 1.2, 1.4],
            "origin": [5, 6, 7], "direction": IDENTITY,
        },
        "output": {
            "dims": [8, 6, 4], "spacing": [0.5, 0.6, 0.7],
            "origin": [4.75, 5.7, 6.65], "direction": IDENTITY,
        },
        "fillValue": 11,
    },
    "downsample-double-spacing": {
        "input": {
            "dims": [8, 6, 4], "spacing": [0.5, 0.6, 0.7],
            "origin": [0, 0, 0], "direction": IDENTITY,
        },
        "output": {
            "dims": [4, 3, 2], "spacing": [1.0, 1.2, 1.4],
            "origin": [0.125, 0.15, 0.175], "direction": IDENTITY,
        },
        "fillValue": 13,
    },
    "outside-fill-uint8": {
        "input": {
            "dims": [4, 3, 2], "spacing": [1, 1, 1],
            "origin": [0, 0, 0], "direction": IDENTITY,
        },
        "output": {
            "dims": [6, 5, 4], "spacing": [1, 1, 1],
            "origin": [-1.25, -1.25, -1.25], "direction": IDENTITY,
        },
        "fillValue": 200,
    },
    "outside-fill-uint16": {
        "input": {
            "dims": [4, 3, 2], "spacing": [0.9, 1.1, 1.3],
            "origin": [2, 3, 4], "direction": IDENTITY,
        },
        "output": {
            "dims": [6, 4, 3], "spacing": [0.9, 1.1, 1.3],
            "origin": [0.875, 2.725, 3.025], "direction": IDENTITY,
        },
        "fillValue": 60000,
        "dtype": "uint16",
        "pattern": "counter16",
    },
    "mirrored": {
        "input": {
            "dims": [4, 3, 2], "spacing": [0.8, 1.1, 1.7],
            "origin": [5, 6, 7], "direction": IDENTITY,
        },
        "output": {
            "dims": [4, 3, 2], "spacing": [0.8, 1.1, 1.7],
            "origin": [7.4, 6, 7], "direction": MIRRORED,
        },
        "fillValue": 9,
    },
    "half-voxel-ties-interior": {
        "input": {
            "dims": [5, 4, 3], "spacing": [1, 1, 1],
            "origin": [0, 0, 0], "direction": IDENTITY,
        },
        "output": {
            "dims": [4, 3, 2], "spacing": [1, 1, 1],
            "origin": [0.5, 0.5, 0.5], "direction": IDENTITY,
        },
        "fillValue": 99,
    },
    "half-voxel-border-ties": {
        "input": {
            "dims": [4, 3, 2], "spacing": [1, 1, 1],
            "origin": [0, 0, 0], "direction": IDENTITY,
        },
        "output": {
            "dims": [4, 3, 2], "spacing": [1, 1, 1],
            "origin": [0.5, 0.5, 0.5], "direction": IDENTITY,
        },
        "fillValue": 99,
    },
    "multi-label": {
        "input": {
            "dims": [6, 6, 4], "spacing": [1, 1, 1],
            "origin": [0, 0, 0], "direction": IDENTITY,
        },
        "output": {
            "dims": [3, 3, 2], "spacing": [2, 2, 2],
            "origin": [0.25, 0.25, 0.25], "direction": IDENTITY,
        },
        "fillValue": 0,
        "pattern": "octants",
    },
}


def make_values(geometry, pattern, dtype):
    dims = geometry["dims"]
    if pattern == "octants":
        maker = lambda z, y, x: (
            1 + (x >= dims[0] // 2) + 2 * (y >= dims[1] // 2) + 4 * (z >= dims[2] // 2)
        )
    elif pattern == "counter16":
        maker = lambda z, y, x: 1 + x + 100 * y + 10000 * z
    else:
        maker = lambda z, y, x: 1 + x + 10 * y + 50 * z
    return np.fromfunction(maker, (dims[2], dims[1], dims[0]), dtype=int).astype(dtype)


def write_nrrd(path, data, geometry, dtype_name):
    columns = [
        [geometry["direction"][row][axis] * geometry["spacing"][axis] for row in range(3)]
        for axis in range(3)
    ]
    vectors = " ".join(f"({','.join(map(str, column))})" for column in columns)
    header = "\n".join([
        "NRRD0005", f"type: {dtype_name}", "dimension: 3",
        f"sizes: {' '.join(map(str, geometry['dims']))}",
        "space: right-anterior-superior", f"space directions: {vectors}",
        "kinds: domain domain domain", "encoding: raw", "endian: little",
        f"space origin: ({','.join(map(str, geometry['origin']))})", "", "",
    ]).encode("ascii")
    path.write_bytes(header + data.tobytes(order="C"))
