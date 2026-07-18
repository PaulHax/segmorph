# Third-party licenses

segmorph's own code is published under the MIT license (see [LICENSE](./LICENSE)).
Parts of the source are TypeScript ports of algorithms from permissively licensed
upstream libraries. A line-by-line translation is a derivative work, so those files
remain under their upstream license, and the upstream copyright notices and license
texts are retained here. This file is distributed inside the published package.

## Provenance

Modules derived from upstream sources at the current state of the library:

| segmorph module                     | Upstream source                                                                                             | Upstream license |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------- |
| `src/convert/marchingCubesCases.ts` | VTK, `vtkMarchingCellsContourCases.cxx` VoxelCases table                                                    | BSD-3-Clause     |
| `src/convert/labelmapToSurface.ts`  | VTK, table-driven discrete marching cubes (`vtkDiscreteMarchingCubes` semantics, `vtkVoxel` edge numbering) | BSD-3-Clause     |
| `src/convert/surfaceToLabelmap.ts`  | VTK, ray-parity voxelization semantics of `vtkPolyDataToImageStencil` and `vtkImageStencil`                 | BSD-3-Clause     |
| `src/convert/meshSmooth.ts`         | VTK, `vtkWindowedSincPolyDataFilter`                                                                        | BSD-3-Clause     |
| `src/convert/meshDecimate.ts`       | VTK, `vtkQuadricDecimation` geometry-only path                                                              | BSD-3-Clause     |
| `src/convert/surfaceNets.ts`        | VTK, `vtkSurfaceNets3D`                                                                                     | BSD-3-Clause     |
| `src/convert/surfaceToContour.ts`   | VTK, `vtkPolyDataPlaneCutter` plane cut and `vtkContourLoopExtraction` loop assembly                        | BSD-3-Clause     |
| `src/image/resample.ts`             | VTK, nearest-neighbor resample behavior of `vtkImageReslice`                                                | BSD-3-Clause     |
| `src/convert/fractional.ts`         | PolySeg, `vtkPolyDataToFractionalLabelmapFilter` and its marching pass                                      | BSD-2-Clause     |
| `src/convert/contourToLabelmap.ts`  | VTK, even-odd scan fill matching `vtkLassoStencilSource` boundary convention                                | BSD-3-Clause     |
| `src/convert/contourToSurface.ts`   | SlicerRT, `vtkPlanarContourToClosedSurfaceConversionRule` (plus VTK marching-squares line cases and `vtkImageDilateErode3D` footprint semantics) | MIT (SlicerRT), BSD-3-Clause (VTK) |

## VTK (BSD-3-Clause)

```
Copyright (c) 1993-2015 Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Ken Martin, Will Schroeder, or Bill Lorensen nor the names
   of any contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## SlicerRT (MIT)

Originally developed by Kyle Sunderland, PerkLab, Queen's University, supported
through the Applied Cancer Research Unit program of Cancer Care Ontario with
funds provided by the Ontario Ministry of Health and Long-Term Care.

```
MIT License

Copyright (c) 2026 SlicerRT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PolySeg (BSD-2-Clause)

```
BSD 2-Clause License

Copyright (c) 2018, The Perk Lab - Laboratory for Percutaneous Surgery
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
