import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';

import { contourToSurface, createOrientedImage, labelmapToSurface, type Mesh } from 'segmorph';

import { buildRtStructRois, buildSegData, SEG_COLORS } from './data';
import { decodeRtStruct, encodeRtStruct } from './rtstruct';
import { decodeSeg, encodeSeg } from './seg';

// The full trip this demo takes, so nothing rendered skips the DICOM layer:
//   segmorph-native data
//     -> encodeRtStruct / encodeSeg   (dcmjs writes real part-10 bytes)
//     -> decodeRtStruct / decodeSeg   (parse those bytes back)
//     -> contourToSurface / labelmapToSurface  (segmorph conversions)
//     -> vtk.js actors.

const query = <T extends HTMLElement>(selector: string) => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`missing element: ${selector}`);
  return element;
};

const legend = query<HTMLDivElement>('#legend');
const statusLine = query<HTMLSpanElement>('#status');

// --- encode: segmorph representations -> DICOM bytes ------------------------

const rtBytes = encodeRtStruct(buildRtStructRois(), { label: 'segmorph demo' });
const segSource = buildSegData();
const segBytes = encodeSeg(segSource.geometry, segSource.segments);

const download = (id: string, bytes: ArrayBuffer, filename: string) => {
  const anchor = query<HTMLAnchorElement>(id);
  anchor.href = URL.createObjectURL(new Blob([bytes], { type: 'application/dicom' }));
  anchor.download = filename;
  anchor.textContent = `${filename} (${(bytes.byteLength / 1024).toFixed(1)} KB)`;
};
download('#rtstruct-download', rtBytes, 'segmorph-demo.rtstruct.dcm');
download('#seg-download', segBytes, 'segmorph-demo.seg.dcm');

// --- decode: DICOM bytes -> segmorph representations -> surfaces ------------

type Structure = {
  name: string;
  source: 'RTSTRUCT' | 'SEG';
  color: [number, number, number];
  mesh: Mesh;
};

const structures: Structure[] = [];

const rtStruct = decodeRtStruct(rtBytes);
for (const roi of rtStruct.rois) {
  structures.push({
    name: roi.name,
    source: 'RTSTRUCT',
    color: roi.color,
    // RTSTRUCT stores planar contours; the planar-contour-to-closed-surface
    // stitch (SlicerRT's algorithm) turns them into a watertight mesh.
    mesh: contourToSurface(roi.loops),
  });
}

const seg = decodeSeg(segBytes);
for (const segment of seg.segments) {
  const labelmap = createOrientedImage({ ...seg.geometry, data: segment.data });
  structures.push({
    name: segment.label,
    source: 'SEG',
    color: SEG_COLORS[segment.label] ?? [200, 200, 200],
    // SEG stores binary labelmaps; marching cubes extracts the surface.
    mesh: labelmapToSurface(labelmap, { labelValue: 1 }),
  });
}

// --- render ------------------------------------------------------------------

const grw = vtkGenericRenderWindow.newInstance({ background: [0.09, 0.09, 0.11] });
grw.setContainer(query('#view'));
grw.resize();
const renderer = grw.getRenderer();
const renderWindow = grw.getRenderWindow();

for (const structure of structures) {
  const polyData = vtkPolyData.newInstance();
  polyData.getPoints().setData(structure.mesh.points, 3);
  polyData.getPolys().setData(structure.mesh.polys);

  const mapper = vtkMapper.newInstance();
  mapper.setInputData(polyData);
  const actor = vtkActor.newInstance();
  actor.setMapper(mapper);
  const [r, g, b] = structure.color;
  actor.getProperty().setColor(r / 255, g / 255, b / 255);
  renderer.addActor(actor);

  const label = document.createElement('label');
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.checked = true;
  checkbox.addEventListener('change', () => {
    actor.setVisibility(checkbox.checked);
    renderWindow.render();
  });
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = `rgb(${r}, ${g}, ${b})`;
  const text = document.createElement('span');
  text.textContent = `${structure.name}`;
  const detail = document.createElement('span');
  detail.className = 'detail';
  detail.textContent = `${structure.source} · ${structure.mesh.polys.length / 4} tris`;
  label.append(checkbox, swatch, text, detail);
  legend.append(label);
}

renderer.resetCamera();
// Tilt off the top-down default so the stacks read as 3D shapes.
const camera = renderer.getActiveCamera();
camera.azimuth(30);
camera.elevation(-60);
camera.setViewUp(0, 0, 1);
renderer.resetCameraClippingRange();
renderWindow.render();
window.addEventListener('resize', () => grw.resize());

statusLine.textContent =
  `${rtStruct.rois.length} RTSTRUCT ROIs + ` +
  `${seg.segments.length} SEG segments decoded from ` +
  `${((rtBytes.byteLength + segBytes.byteLength) / 1024).toFixed(1)} KB of DICOM`;
