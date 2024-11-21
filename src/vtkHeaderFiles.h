#pragma once

#include <vtkVersion.h>
#include <vtkNew.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkTransform.h>

#include <vtkCommand.h>
#include <vtkCallbackCommand.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include <vtkRenderer.h>
#include <vtkRendererCollection.h>

#include <vtkObject.h>
#include <vtkTable.h>

#include <vtkNamedColors.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>

#include <vtkCamera.h>
#include <vtkPointData.h>
#include <vtkLine.h>
#include <vtkTriangle.h>
#include <vtkPlane.h>
#include <vtkQuad.h>
#include <vtkCellData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkPolygon.h>
#include <vtkAppendPolyData.h>

#include <vtkCylinderSource.h>
#include <vtkSphereSource.h>
#include <vtkConeSource.h>
#include <vtkCubeSource.h>
#include <vtkArrowSource.h>
#include <vtkFrustumSource.h>
#include <vtkPlaneSource.h>
#include <vtkVectorText.h>

#include <vtkPolyDataMapper.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkGlyph3DMapper.h>
#include <vtkDataSetMapper.h>
#include <vtkLookupTable.h>
#include <vtkPolyDataNormals.h>
#include <vtkDataObjectToTable.h>
#include <vtkColorTransferFunction.h>

#include <vtkActor.h>
#include <vtkAssembly.h>
#include <vtkTextActor.h>
#include <vtkTextActor3D.h>
#include <vtkAxesActor.h>
#include <vtkCaptionActor2D.h>
#include <vtkTextProperty.h>
#include <vtkGlyph3D.h>

#include <vtkCellPicker.h>
#include <vtkPropPicker.h>

#include <vtkSelectionNode.h>
#include <vtkExtractSelection.h>

#include <vtkAnnotatedCubeActor.h>
#include <vtkOrientationMarkerWidget.h>

#include <vtkPolyDataAlgorithm.h>
#include <vtkInformation.h>
#include <vtkInformationVector.h>

#include <vtkKdTree.h>
#include <vtkOctreePointLocator.h>

#include <vtkDelaunay3D.h>
#include <vtkPolyDataConnectivityFilter.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkElevationFilter.h>
#include <vtkAdaptiveSubdivisionFilter.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkKdTreePointLocator.h>
#include <vtkLinearExtrusionFilter.h>
#include <vtkContourFilter.h>
#include <vtkCellLocator.h>
#include <vtkSignedDistance.h>
#include <vtkTransformFilter.h>
#include <vtkMarchingCubes.h>
#include <vtkDecimatePro.h>
#include <vtkWindowToImageFilter.h>
#include <vtkImageShiftScale.h>

#include <vtkExtractCells.h>

#include <vtkSTLReader.h>
#include <vtkSTLWriter.h>
#include <vtkOBJReader.h>
#include <vtkPLYReader.h>
#include <vtkPLYWriter.h>

#include <vtkImageCast.h>
#include <vtkImageData.h>
#include <vtkPNGReader.h>
#include <vtkPNGWriter.h>