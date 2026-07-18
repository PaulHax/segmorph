export { surfaceToLabelmap } from './convert/surfaceToLabelmap.js';
export {
  createContourLoop,
  createContourPlane,
  createPlanarContour,
  planarContourWorldLoops,
  planeToWorld,
  validateContourLoop,
  validateContourPlane,
  validatePlanarContour,
  worldToPlane,
} from './geometry/contour.js';
export type {
  ContourLoop,
  ContourPlane,
  PlanarContour,
  Vector2,
} from './geometry/contour.js';
export {
  createMesh,
  getPoint,
  iteratePoints,
  iterateTriangles,
  triangleCount,
  validateMesh,
  vertexCount,
} from './geometry/mesh.js';
export type { Mesh, Point, Triangle } from './geometry/mesh.js';
export { labelmapToSurface } from './convert/labelmapToSurface.js';
export { meshSmooth } from './convert/meshSmooth.js';
export type { MeshSmoothOptions, SmoothingWindowFunction } from './convert/meshSmooth.js';
export { meshDecimate } from './convert/meshDecimate.js';
export type { MeshDecimateOptions } from './convert/meshDecimate.js';
export { surfaceToContour } from './convert/surfaceToContour.js';
export { contourToLabelmap } from './convert/contourToLabelmap.js';
export { contourToSurface } from './convert/contourToSurface.js';
export type {
  ContourToSurfaceOptions,
  EndCappingMode,
} from './convert/contourToSurface.js';
export {
  surfaceToFractionalLabelmap,
  fractionalLabelmapToSurface,
} from './convert/fractional.js';
export {
  createDefaultRules,
  createDefaultConversionGraph,
  defaultRepresentations,
} from './model/rules.js';
export type { DefaultRuleOptions } from './model/rules.js';
export { fillBetween } from './convert/fillBetween.js';
export type { FillBetweenOptions } from './convert/fillBetween.js';
export { surfaceNets } from './convert/surfaceNets.js';
export type { SurfaceNetsOptions } from './convert/surfaceNets.js';
export { readNrrd } from './io/nrrd.js';
export type { Nrrd, NrrdData } from './io/nrrd.js';
export {
  createOrientedImage,
  indexToWorld,
  validateImageGeometry,
  worldToIndex,
} from './image/orientedImage.js';
export type {
  ImageData,
  ImageGeometry,
  OrientedImage,
  Vector3,
} from './image/orientedImage.js';
export {
  compositeImage,
  compositeMax,
  compositeMin,
  compositeSet,
  maskByLabelValue,
} from './image/composite.js';
export type { CompositeOperation } from './image/composite.js';
export {
  EMPTY_EXTENT,
  extentFromDims,
  flatIndex,
  intersectExtents,
  isExtentEmpty,
  isIndexInExtent,
  iterateExtentIndices,
  validateExtent,
} from './image/extent.js';
export type { Dimensions3D, Extent3D, Index3D } from './image/extent.js';
export { resampleNearest } from './image/resample.js';
export type { NearestResampleOptions } from './image/resample.js';
export {
  createSegment,
  getRepresentation,
  setRepresentation,
} from './model/segment.js';
export type {
  CreateSegmentOptions,
  Segment,
  SegmentColor,
} from './model/segment.js';
export {
  addSegment,
  createSegmentation,
  removeSegment,
  renameSegment,
  reorderSegments,
} from './model/segmentation.js';
export type { Segmentation } from './model/segmentation.js';
export {
  assignSegmentToLayer,
  compactLayers,
  createLayerModel,
} from './model/layers.js';
export type {
  LabelmapArray,
  LayerAssignment,
  LayerModel,
} from './model/layers.js';
export {
  createConversionGraph,
  findCheapestPath,
  registerConversionRule,
} from './model/graph.js';
export type { ConversionGraph, ConversionRule } from './model/graph.js';
export {
  editRepresentation,
  getOrCreateRepresentation,
  promoteRepresentationToSource,
} from './model/invalidation.js';
