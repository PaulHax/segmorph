import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';

import { createOrientedImage, labelmapToSurface } from 'segmorph';

// 1. A labelmap. In a real app this comes from a segmentation editor or a
//    DICOM SEG; here it is a sphere painted into a 64^3 volume.
const dims = [64, 64, 64];
const data = new Uint8Array(dims[0] * dims[1] * dims[2]);
for (let z = 0; z < dims[2]; z += 1) {
  for (let y = 0; y < dims[1]; y += 1) {
    for (let x = 0; x < dims[0]; x += 1) {
      const inside = (x - 32) ** 2 + (y - 32) ** 2 + (z - 32) ** 2 < 22 ** 2;
      if (inside) data[x + dims[0] * (y + dims[1] * z)] = 1;
    }
  }
}

const labelmap = createOrientedImage({
  data,
  dims,
  spacing: [1, 1, 1],
  origin: [0, 0, 0],
  direction: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
});

// 2. The conversion. One synchronous call, no setup and no await.
const mesh = labelmapToSurface(labelmap, { labelValue: 1 });

// 3. The mesh is already vtk.js shaped: interleaved xyz points and a VTK cell
//    array, so it becomes a vtkPolyData without touching the numbers.
const polyData = vtkPolyData.newInstance();
polyData.getPoints().setData(mesh.points, 3);
polyData.getPolys().setData(mesh.polys);

const fullScreen = vtkFullScreenRenderWindow.newInstance({
  background: [0.1, 0.1, 0.12],
});
const renderer = fullScreen.getRenderer();

const mapper = vtkMapper.newInstance();
mapper.setInputData(polyData);
const actor = vtkActor.newInstance();
actor.setMapper(mapper);
actor.getProperty().setColor(0.9, 0.5, 0.3);
renderer.addActor(actor);
renderer.resetCamera();
fullScreen.getRenderWindow().render();

const info = document.querySelector('#info');
if (info) info.textContent = `${mesh.polys.length / 4} triangles`;
