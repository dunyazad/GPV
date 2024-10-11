#pragma once

#ifndef vtkPolygonalFrustumSource_h
#define vtkPolygonalFrustumSource_h

#include "vtkFiltersSourcesModule.h" // For export macro
#include "vtkPolyDataAlgorithm.h"

#include "vtkCell.h" // Needed for VTK_CELL_SIZE

#undef VTKFILTERSSOURCES_EXPORT
#define VTKFILTERSSOURCES_EXPORT

VTK_ABI_NAMESPACE_BEGIN
class VTKFILTERSSOURCES_EXPORT vtkPolygonalFrustumSource : public vtkPolyDataAlgorithm
{
public:
	vtkTypeMacro(vtkPolygonalFrustumSource, vtkPolyDataAlgorithm);
	void PrintSelf(ostream& os, vtkIndent indent) override;

	/**
	 * Construct with default resolution 6, height 1.0, radius 0.5, and
	 * capping on. The cone is centered at the origin and points down
	 * the x-axis.
	 */
	static vtkPolygonalFrustumSource* New();

	vtkSetClampMacro(Height, double, 0.0, VTK_DOUBLE_MAX);
	vtkGetMacro(Height, double);

	vtkSetClampMacro(TopRadius, double, 0.0, VTK_DOUBLE_MAX);
	vtkGetMacro(TopRadius, double);

	vtkSetClampMacro(BottomRadius, double, 0.0, VTK_DOUBLE_MAX);
	vtkGetMacro(BottomRadius, double);

	vtkSetClampMacro(NumberOfSides, int, 3, 512);
	vtkGetMacro(NumberOfSides, int);

protected:
	vtkPolygonalFrustumSource();
	~vtkPolygonalFrustumSource() override = default;

	int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
	int RequestInformation(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

	double Height;
	double TopRadius;
	double BottomRadius;
	int NumberOfSides;
	int OutputPointsPrecision;

private:
	vtkPolygonalFrustumSource(const vtkPolygonalFrustumSource&) = delete;
	void operator=(const vtkPolygonalFrustumSource&) = delete;
};

VTK_ABI_NAMESPACE_END
#endif
