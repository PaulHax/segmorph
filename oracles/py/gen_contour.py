"""Golden generator for algorithm F: surface x plane -> planar contour loops.

Pipeline per case: vtkPlane + vtkCutter (which delegates triangle polydata to
vtkPolyDataPlaneCutter) + vtkContourLoopExtraction with LoopClosure OFF and
polygon output. Calibration runs vtkPolyDataPlaneCutter directly as a second
pipeline and records the spread between both oracle paths.

Inputs are deterministic synthetic float32 meshes (no timestamps, no RNG)
plus the committed A sphere golden mesh.
"""

import json
import math
import pathlib

import numpy as np
import vtk

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE.parent.parent / "test" / "fixtures"
OUT_ROOT = FIXTURES / "F"


def normalize(vector):
    array = np.asarray(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


def make_plane_frame(origin, x_axis, y_axis):
    x_axis = normalize(x_axis)
    y_axis = normalize(y_axis)
    normal = np.cross(x_axis, y_axis)
    return {
        "origin": [float(v) for v in origin],
        "xAxis": [float(v) for v in x_axis],
        "yAxis": [float(v) for v in y_axis],
        "normal": [float(v) for v in normal],
    }


def torus_mesh(ring_radius=6.0, tube_radius=2.0, n_u=24, n_v=12):
    points = []
    for i in range(n_u):
        u = 2.0 * math.pi * i / n_u
        for j in range(n_v):
            v = 2.0 * math.pi * j / n_v
            radial = ring_radius + tube_radius * math.cos(v)
            points.append((
                radial * math.cos(u),
                radial * math.sin(u),
                tube_radius * math.sin(v),
            ))
    triangles = []
    for i in range(n_u):
        for j in range(n_v):
            p00 = i * n_v + j
            p01 = i * n_v + (j + 1) % n_v
            p10 = ((i + 1) % n_u) * n_v + j
            p11 = ((i + 1) % n_u) * n_v + (j + 1) % n_v
            triangles.append((p00, p10, p11))
            triangles.append((p00, p11, p01))
    return np.asarray(points, dtype=np.float32), triangles


def cube_mesh(minimum=0.0, maximum=2.0):
    points = np.asarray([
        (minimum, minimum, minimum), (maximum, minimum, minimum),
        (maximum, maximum, minimum), (minimum, maximum, minimum),
        (minimum, minimum, maximum), (maximum, minimum, maximum),
        (maximum, maximum, maximum), (minimum, maximum, maximum),
    ], dtype=np.float32)
    triangles = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
    ]
    return points, triangles


def octahedron_mesh():
    points = np.asarray([
        (0, 0, 1), (0, 0, -1),
        (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
    ], dtype=np.float32)
    triangles = [
        (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 2),
        (1, 3, 2), (1, 4, 3), (1, 5, 4), (1, 2, 5),
    ]
    return points, triangles


def sphere_mesh():
    data = json.loads((FIXTURES / "A" / "sphere" / "golden.mesh.json").read_text())
    points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
    polys = data["polys"]
    triangles = [
        (polys[offset + 1], polys[offset + 2], polys[offset + 3])
        for offset in range(0, len(polys), 4)
    ]
    return points, triangles


def to_polydata(points, triangles):
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetDataTypeToFloat()
    for point in points:
        vtk_points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
    poly.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    for triangle in triangles:
        cells.InsertNextCell(3)
        for vertex in triangle:
            cells.InsertCellPoint(int(vertex))
    poly.SetPolys(cells)
    return poly


def extract_loops(polydata):
    loops = []
    polys = polydata.GetPolys()
    points = polydata.GetPoints()
    polys.InitTraversal()
    ids = vtk.vtkIdList()
    while polys.GetNextCell(ids):
        loop = []
        for index in range(ids.GetNumberOfIds()):
            loop.extend(points.GetPoint(ids.GetId(index)))
        loops.append([float(v) for v in loop])
    return loops


def run_cutter(polydata, frame):
    plane = vtk.vtkPlane()
    plane.SetOrigin(*frame["origin"])
    plane.SetNormal(*frame["normal"])

    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(polydata)
    cutter.Update()
    if cutter.GetOutput().GetNumberOfLines() == 0:
        return []

    loop_filter = vtk.vtkContourLoopExtraction()
    loop_filter.SetInputConnection(cutter.GetOutputPort())
    loop_filter.SetLoopClosureToOff()
    loop_filter.SetOutputModeToPolygons()
    loop_filter.Update()
    return extract_loops(loop_filter.GetOutput())


def run_plane_cutter(polydata, frame):
    plane = vtk.vtkPlane()
    plane.SetOrigin(*frame["origin"])
    plane.SetNormal(*frame["normal"])

    cutter = vtk.vtkPolyDataPlaneCutter()
    cutter.SetPlane(plane)
    cutter.SetInputData(polydata)
    cutter.Update()
    if cutter.GetOutput().GetNumberOfLines() == 0:
        return []

    loop_filter = vtk.vtkContourLoopExtraction()
    loop_filter.SetInputConnection(cutter.GetOutputPort())
    loop_filter.SetLoopClosureToOff()
    loop_filter.SetOutputModeToPolygons()
    loop_filter.Update()
    return extract_loops(loop_filter.GetOutput())


def loop_vertices(loop):
    return np.asarray(loop, dtype=np.float64).reshape(-1, 3)


def point_to_polyline_distance(point, vertices):
    best = math.inf
    count = len(vertices)
    for index in range(count):
        start = vertices[index]
        end = vertices[(index + 1) % count]
        direction = end - start
        length_squared = float(np.dot(direction, direction))
        if length_squared == 0.0:
            candidate = float(np.linalg.norm(point - start))
        else:
            t = float(np.dot(point - start, direction)) / length_squared
            t = min(1.0, max(0.0, t))
            candidate = float(np.linalg.norm(point - (start + t * direction)))
        best = min(best, candidate)
    return best


def symmetric_loop_distance(loop_a, loop_b):
    vertices_a = loop_vertices(loop_a)
    vertices_b = loop_vertices(loop_b)
    forward = max(point_to_polyline_distance(p, vertices_b) for p in vertices_a)
    backward = max(point_to_polyline_distance(p, vertices_a) for p in vertices_b)
    return max(forward, backward)


def match_loops(loops_a, loops_b):
    remaining = list(range(len(loops_b)))
    pairs = []
    for index_a, loop_a in enumerate(loops_a):
        centroid_a = loop_vertices(loop_a).mean(axis=0)
        best = min(
            remaining,
            key=lambda index_b: float(np.linalg.norm(
                centroid_a - loop_vertices(loops_b[index_b]).mean(axis=0))),
        )
        remaining.remove(best)
        pairs.append((index_a, best))
    return pairs


def calibrate(loops_a, loops_b):
    if len(loops_a) != len(loops_b):
        return {"loopCountA": len(loops_a), "loopCountB": len(loops_b)}
    if not loops_a:
        return {"maxSymmetricDistance": 0.0}
    spread = max(
        symmetric_loop_distance(loops_a[a], loops_b[b])
        for a, b in match_loops(loops_a, loops_b)
    )
    return {"maxSymmetricDistance": float(spread)}


def write_mesh_json(path, points, triangles):
    polys = []
    for triangle in triangles:
        polys.append(3)
        polys.extend(int(v) for v in triangle)
    flat_points = [float(v) for v in np.asarray(points, dtype=np.float32).ravel()]
    path.write_text(json.dumps({"points": flat_points, "polys": polys}) + "\n")


CASES = [
    {
        "case": "sphere-center",
        "mesh": sphere_mesh,
        "input": "A/sphere/golden.mesh.json",
        "frame": make_plane_frame((15.5, 15.5, 15.5), (1, 0, 0), (0, 1, 0)),
    },
    {
        "case": "sphere-oblique",
        "mesh": sphere_mesh,
        "input": "A/sphere/golden.mesh.json",
        "frame": make_plane_frame(
            (15.5, 15.5, 15.5),
            normalize(np.cross(normalize((1, 1, 1)), (0, 0, 1))),
            np.cross(
                normalize((1, 1, 1)),
                normalize(np.cross(normalize((1, 1, 1)), (0, 0, 1))),
            ),
        ),
    },
    {
        "case": "sphere-miss",
        "mesh": sphere_mesh,
        "input": "A/sphere/golden.mesh.json",
        "frame": make_plane_frame((15.5, 15.5, 50.0), (1, 0, 0), (0, 1, 0)),
    },
    {
        "case": "torus-two-loops",
        "mesh": torus_mesh,
        "input": "F/torus-two-loops/input.mesh.json",
        "frame": make_plane_frame((0, 0, 0), (0, 1, 0), (0, 0, 1)),
    },
    {
        "case": "cube-axis",
        "mesh": cube_mesh,
        "input": "F/cube-axis/input.mesh.json",
        "frame": make_plane_frame((1, 1, 1), (1, 0, 0), (0, 1, 0)),
    },
    {
        "case": "octahedron-on-vertices",
        "mesh": octahedron_mesh,
        "input": "F/octahedron-on-vertices/input.mesh.json",
        "frame": make_plane_frame((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    },
]


def append_manifest_entries(entries):
    # Append-only: preserve existing entries and the manifest's indent=2
    # formatting so regeneration produces a pure-append diff.
    manifest_path = FIXTURES / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schemaVersion"] == 1
    kept = [
        fixture for fixture in manifest["fixtures"]
        if fixture["algorithm"] != "F"
    ]
    manifest["fixtures"] = kept + entries
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    entries = []
    for spec in CASES:
        case_dir = OUT_ROOT / spec["case"]
        case_dir.mkdir(parents=True, exist_ok=True)

        points, triangles = spec["mesh"]()
        if spec["input"].startswith("F/"):
            write_mesh_json(case_dir / "input.mesh.json", points, triangles)

        polydata = to_polydata(points, triangles)
        loops = run_cutter(polydata, spec["frame"])
        loops_direct = run_plane_cutter(polydata, spec["frame"])
        calibration = calibrate(loops, loops_direct)

        golden = {"plane": spec["frame"], "loops": loops}
        (case_dir / "golden.contour.json").write_text(json.dumps(golden) + "\n")

        params = {
            "inputMesh": spec["input"],
            "plane": spec["frame"],
            "loopClosure": "off",
            "outputMode": "polygons",
            "calibration": calibration,
        }
        (case_dir / "params.json").write_text(json.dumps(params, indent=1) + "\n")

        entries.append({
            "oracle": {"name": "python-vtk", "version": vtk.VTK_VERSION},
            "algorithm": "F",
            "case": spec["case"],
            "params": params,
            "seed": 0,
        })
        print(spec["case"], "loops", len(loops), "calibration", calibration)

    append_manifest_entries(entries)


if __name__ == "__main__":
    main()
