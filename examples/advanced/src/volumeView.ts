import '@kitware/vtk.js/Rendering/Profiles/Volume';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import type vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import type { Mesh } from 'segmorph';

import type { SegmentSeed } from './data';

/**
 * 3D view: the CT as a volume rendering, with each segment's extracted surface
 * drawn inside it. The meshes come from segmorph; the volume is plain vtk.js.
 * They share a world coordinate system because segmorph works in world
 * (physical) coordinates, so no registration step is needed.
 */
export function createVolumeView(container: HTMLElement, seeds: SegmentSeed[]) {
  const grw = vtkGenericRenderWindow.newInstance({ background: [0.08, 0.08, 0.1] });
  grw.setContainer(container);
  grw.resize();
  const renderer = grw.getRenderer();
  const renderWindow = grw.getRenderWindow();

  // Volume rendering of the CT itself.
  const volumeMapper = vtkVolumeMapper.newInstance();
  volumeMapper.setSampleDistance(1.2);
  const volume = vtkVolume.newInstance();
  volume.setMapper(volumeMapper);

  // Neutral gray for the scan. The segments are saturated (blue, amber), so a
  // desaturated volume reads as "the image" rather than competing with the
  // surfaces drawn inside it.
  const color = vtkColorTransferFunction.newInstance();
  color.addRGBPoint(0, 0, 0, 0);
  color.addRGBPoint(40, 0.22, 0.23, 0.26);
  color.addRGBPoint(110, 0.5, 0.52, 0.56);
  color.addRGBPoint(180, 0.78, 0.79, 0.82);
  color.addRGBPoint(255, 1, 1, 1);

  // Soft tissue sits roughly in 30..120 on this rescaled CT. The previous ramp
  // held zero until 90, which erased everything between the bones, so it now
  // starts just above air and stays faint rather than absent.
  const OPACITY_POINTS: [number, number][] = [
    [0, 0],
    [22, 0],
    [45, 0.012],
    [90, 0.03],
    [140, 0.06],
    [190, 0.14],
    [255, 0.35],
  ];

  const opacity = vtkPiecewiseFunction.newInstance();
  const applyOpacityScale = (scale: number) => {
    opacity.removeAllPoints();
    for (const [value, alpha] of OPACITY_POINTS) opacity.addPoint(value, alpha * scale);
    volume.getProperty().setScalarOpacity(0, opacity);
  };
  applyOpacityScale(1);

  volume.getProperty().setRGBTransferFunction(0, color);
  volume.getProperty().setInterpolationTypeToLinear();
  volume.getProperty().setShade(true);

  // One actor per segment, colored by the segment's display color.
  const actors = new Map<string, { actor: vtkActor; polyData: vtkPolyData }>();
  for (const seed of seeds) {
    const polyData = vtkPolyData.newInstance();
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(polyData);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(...seed.color);
    actor.getProperty().setOpacity(seed.opacity);
    renderer.addActor(actor);
    actors.set(seed.id, { actor, polyData });
  }

  let cameraInitialized = false;

  return {
    setVolume(image: vtkImageData) {
      volumeMapper.setInputData(image);
      renderer.addVolume(volume);
    },
    /** Overall CT opacity; 0 hides the volume entirely. */
    setVolumeOpacity(scale: number) {
      volume.setVisibility(scale > 0);
      if (scale > 0) applyOpacityScale(scale);
      renderWindow.render();
    },
    setSegmentVisible(id: string, visible: boolean) {
      actors.get(id)?.actor.setVisibility(visible);
      renderWindow.render();
    },
    updateMesh(id: string, mesh: Mesh) {
      const entry = actors.get(id);
      if (!entry) return;
      entry.polyData.getPoints().setData(mesh.points, 3);
      entry.polyData.getPolys().setData(mesh.polys);
      entry.polyData.modified();
    },
    render() {
      if (!cameraInitialized) {
        // Anterior view: look at the patient from the front, head up. Set the
        // direction first, then let resetCamera pick the distance.
        const camera = renderer.getActiveCamera();
        camera.setPosition(0, -1, 0);
        camera.setFocalPoint(0, 0, 0);
        camera.setViewUp(0, 0, 1);
        renderer.resetCamera();
        cameraInitialized = true;
      }
      renderWindow.render();
    },
  };
}
