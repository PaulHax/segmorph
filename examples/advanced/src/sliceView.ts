import '@kitware/vtk.js/Rendering/Profiles/Volume';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkImageResliceMapper from '@kitware/vtk.js/Rendering/Core/ImageResliceMapper';
import vtkImageSlice from '@kitware/vtk.js/Rendering/Core/ImageSlice';
import vtkInteractorStyleImage from '@kitware/vtk.js/Interaction/Style/InteractorStyleImage';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPlane from '@kitware/vtk.js/Common/DataModel/Plane';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import type vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import {
  createContourPlane,
  planeToWorld,
  surfaceToContour,
  type Mesh,
} from 'segmorph';

import type { SegmentSeed } from './data';

type Bounds = [number, number, number, number, number, number];

/**
 * 2D view: a reslice of the CT with each segment drawn as a contour.
 *
 * The contours are not a second segmentation. They are the same surface
 * meshes shown in the 3D view, cut by this slice plane with
 * `surfaceToContour`. That is the polymorphic idea on one screen: one edited
 * representation, several views of it, converted on demand.
 */
export function createSliceView(container: HTMLElement, seeds: SegmentSeed[]) {
  const grw = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
  grw.setContainer(container);
  grw.resize();
  const renderer = grw.getRenderer();
  const renderWindow = grw.getRenderWindow();

  // A 2D view must not tumble: the trackball style would rotate the slice out
  // of the screen on the first drag. Image style is pan and zoom only.
  grw.getInteractor().setInteractorStyle(vtkInteractorStyleImage.newInstance());

  const slicePlane = vtkPlane.newInstance({ normal: [0, 0, 1], origin: [0, 0, 0] });
  const sliceMapper = vtkImageResliceMapper.newInstance();
  sliceMapper.setSlicePlane(slicePlane);
  const imageSlice = vtkImageSlice.newInstance();
  imageSlice.setMapper(sliceMapper);
  imageSlice.getProperty().setColorWindow(255);
  imageSlice.getProperty().setColorLevel(127);

  // One line actor per segment for the cut contours.
  const contourActors = new Map<string, { actor: vtkActor; polyData: vtkPolyData }>();
  for (const seed of seeds) {
    const polyData = vtkPolyData.newInstance();
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(polyData);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(...seed.color);
    actor.getProperty().setLineWidth(2);
    renderer.addActor(actor);
    contourActors.set(seed.id, { actor, polyData });
  }

  const meshes = new Map<string, Mesh>();
  const loopCounts = new Map<string, number>();
  let bounds: Bounds | undefined;
  let currentZ = 0;
  let cameraInitialized = false;

  // The contours are coplanar with the image slice, so they would z-fight with
  // it. Lift them a hair toward the camera; the offset is far below one voxel.
  const CONTOUR_LIFT = 0.25;

  const clear = (entry: { polyData: vtkPolyData }) => {
    entry.polyData.getPoints().setData(new Float32Array(0), 3);
    entry.polyData.getLines().setData(new Uint32Array(0));
    entry.polyData.modified();
  };

  /** Cut one mesh with the current slice plane and upload the loops. */
  const updateContour = (id: string, z: number) => {
    const entry = contourActors.get(id);
    if (!entry) return;
    const mesh = meshes.get(id);

    if (!mesh || mesh.polys.length === 0) {
      clear(entry);
      loopCounts.set(id, 0);
      return;
    }

    // An axial plane through the current slice, in world coordinates.
    const plane = createContourPlane([0, 0, z], [1, 0, 0], [0, 1, 0]);
    // Undefined when the plane misses the mesh entirely.
    const contour = surfaceToContour(mesh, plane);
    if (!contour) {
      clear(entry);
      loopCounts.set(id, 0);
      return;
    }

    // Planar loops -> world-space line cells for vtk.js.
    const points: number[] = [];
    const lines: number[] = [];
    for (const loop of contour.loops) {
      const count = loop.points.length / 2;
      if (count < 2) continue;
      const first = points.length / 3;
      for (let i = 0; i < count; i += 1) {
        const world = planeToWorld(plane, [loop.points[2 * i], loop.points[2 * i + 1]]);
        points.push(world[0], world[1], world[2] + CONTOUR_LIFT);
      }
      // Close the loop by repeating the first index.
      lines.push(count + 1);
      for (let i = 0; i < count; i += 1) lines.push(first + i);
      lines.push(first);
    }

    entry.polyData.getPoints().setData(new Float32Array(points), 3);
    entry.polyData.getLines().setData(new Uint32Array(lines));
    entry.polyData.modified();
    loopCounts.set(id, contour.loops.length);
  };

  const refreshContours = () => {
    for (const id of contourActors.keys()) updateContour(id, currentZ);
  };

  let sliceStep = 1;
  let onSliceChange: ((z: number) => void) | undefined;

  const setSlice = (z: number) => {
    const clamped = bounds ? Math.min(Math.max(z, bounds[4]), bounds[5]) : z;
    currentZ = clamped;
    slicePlane.setOrigin(0, 0, clamped);
    refreshContours();
    if (!cameraInitialized && bounds) {
      // Look straight down the slice normal so the reslice fills the view.
      const camera = renderer.getActiveCamera();
      const cx = (bounds[0] + bounds[1]) / 2;
      const cy = (bounds[2] + bounds[3]) / 2;
      camera.setParallelProjection(true);
      camera.setPosition(cx, cy, clamped + 500);
      camera.setFocalPoint(cx, cy, clamped);
      camera.setViewUp(0, -1, 0);
      renderer.resetCamera();
      cameraInitialized = true;
    }
    renderWindow.render();
    onSliceChange?.(clamped);
  };

  // Zoom, for a parallel-projection view, is the camera's parallel scale.
  const zoomBy = (factor: number) => {
    const camera = renderer.getActiveCamera();
    camera.setParallelScale(camera.getParallelScale() * factor);
    renderWindow.render();
  };

  // The plain wheel steps through slices, which takes over the binding that
  // would normally zoom, so zoom moves to ctrl/cmd + wheel and right-drag.
  // Both are captured on the container to preempt the interactor's own
  // handlers bound to the same element.
  container.addEventListener('wheel', (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (event.ctrlKey || event.metaKey) {
      zoomBy(event.deltaY > 0 ? 1.1 : 1 / 1.1);
      return;
    }
    setSlice(currentZ + Math.sign(event.deltaY) * sliceStep);
  }, { capture: true, passive: false });

  // Right-drag zooms, the convention in most slice viewers. vtk.js's built-in
  // styles bind only the left button, so this is handled here.
  container.addEventListener('contextmenu', (event) => event.preventDefault());

  let zoomAnchorY: number | undefined;

  container.addEventListener('pointerdown', (event) => {
    if (event.button !== 2) return;
    event.preventDefault();
    event.stopPropagation();
    zoomAnchorY = event.clientY;
    // Capture keeps the drag alive outside the element, but is not essential:
    // it throws for pointers the element never actually received.
    try {
      container.setPointerCapture(event.pointerId);
    } catch {
      // Drag still works through the container's own move events.
    }
  }, { capture: true });

  container.addEventListener('pointermove', (event) => {
    if (zoomAnchorY === undefined) return;
    event.preventDefault();
    event.stopPropagation();
    const dy = event.clientY - zoomAnchorY;
    zoomAnchorY = event.clientY;
    zoomBy(Math.exp(dy * 0.005));
  }, { capture: true });

  const endZoom = (event: PointerEvent) => {
    if (zoomAnchorY === undefined) return;
    zoomAnchorY = undefined;
    if (container.hasPointerCapture(event.pointerId)) {
      container.releasePointerCapture(event.pointerId);
    }
  };
  container.addEventListener('pointerup', endZoom, { capture: true });
  container.addEventListener('pointercancel', endZoom, { capture: true });

  return {
    setVolume(image: vtkImageData) {
      sliceMapper.setInputData(image);
      renderer.addViewProp(imageSlice);
      bounds = image.getBounds() as Bounds;
      sliceStep = image.getSpacing()[2];
    },
    /** Notified whenever the slice moves, including by wheel. */
    onSliceChange(listener: (z: number) => void) {
      onSliceChange = listener;
    },
    getSlice() {
      return currentZ;
    },
    /** Move the slice; both the reslice and every contour follow it. */
    setSlice,
    /** Store a new mesh and recut it at the current slice. */
    setMesh(id: string, mesh: Mesh) {
      meshes.set(id, mesh);
      updateContour(id, currentZ);
    },
    setSegmentVisible(id: string, visible: boolean) {
      contourActors.get(id)?.actor.setVisibility(visible);
      renderWindow.render();
    },
    render() {
      renderWindow.render();
    },
    getBounds() {
      return bounds;
    },
    /** Loops cut at the current slice, per segment. */
    getLoopCounts() {
      return new Map(loopCounts);
    },
  };
}
