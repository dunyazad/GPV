#include "vtkPolygonalFrustumSource.h"

#include "vtkCellArray.h"
#include "vtkDoubleArray.h"
#include "vtkFloatArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPolyData.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTransform.h"

VTK_ABI_NAMESPACE_BEGIN
vtkStandardNewMacro(vtkPolygonalFrustumSource);

vtkPolygonalFrustumSource::vtkPolygonalFrustumSource()
{
	this->Height = 1.0;
	this->TopRadius = 0.5;
	this->BottomRadius = 1.0;
	this->NumberOfSides = 32;

	this->OutputPointsPrecision = SINGLE_PRECISION;

	this->SetNumberOfInputPorts(0);
}

int vtkPolygonalFrustumSource::RequestData(vtkInformation* vtkNotUsed(request),
	vtkInformationVector** vtkNotUsed(inputVector), vtkInformationVector* outputVector)
{
	vtkInformation* outInfo = outputVector->GetInformationObject(0);

	auto newPoints = vtkPoints::New();
	auto newPolys = vtkCellArray::New();
	vtkPolyData* output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

	for (int i = 0; i < NumberOfSides; i++)
	{
		double angle = 2.0 * vtkMath::Pi() * (double)i / (double)NumberOfSides;
		double y = BottomRadius * cos(angle);
		double z = BottomRadius * sin(angle);
		newPoints->InsertNextPoint(0, y, z);
	}

	for (int i = 0; i < NumberOfSides; i++)
	{
		double angle = 2.0 * vtkMath::Pi() * (double)i / (double)NumberOfSides;
		double y = TopRadius * cos(angle);
		double z = TopRadius * sin(angle);
		newPoints->InsertNextPoint(Height, y, z);
	}

	newPoints->InsertNextPoint(0, 0, 0);
	newPoints->InsertNextPoint(Height, 0, 0);

	for (int i = 0; i < NumberOfSides - 1; i++) {
		auto i0 = i;
		auto i1 = i + NumberOfSides;
		auto i2 = i + NumberOfSides + 1;
		auto i3 = i + 1;

		newPolys->InsertNextCell(3);
		newPolys->InsertCellPoint(i0);
		newPolys->InsertCellPoint(i2);
		newPolys->InsertCellPoint(i1);
		newPolys->InsertNextCell(3);
		newPolys->InsertCellPoint(i0);
		newPolys->InsertCellPoint(i3);
		newPolys->InsertCellPoint(i2);
	}

	newPolys->InsertNextCell(3);
	newPolys->InsertCellPoint(NumberOfSides - 1);
	newPolys->InsertCellPoint(NumberOfSides);
	newPolys->InsertCellPoint(NumberOfSides + NumberOfSides - 1);
	newPolys->InsertNextCell(3);
	newPolys->InsertCellPoint(NumberOfSides - 1);
	newPolys->InsertCellPoint(0);
	newPolys->InsertCellPoint(NumberOfSides);

	for (int i = 0; i < NumberOfSides - 1; i++) {
		newPolys->InsertNextCell(3);
		newPolys->InsertCellPoint(i);
		newPolys->InsertCellPoint(NumberOfSides + NumberOfSides);
		newPolys->InsertCellPoint(i + 1);
		newPolys->InsertNextCell(3);
		newPolys->InsertCellPoint(NumberOfSides - 1);
		newPolys->InsertCellPoint(NumberOfSides + NumberOfSides);
		newPolys->InsertCellPoint(0);
	}

	for (int i = 0; i < NumberOfSides - 1; i++) {
		newPolys->InsertNextCell(3);
		newPolys->InsertCellPoint(NumberOfSides + i);
		newPolys->InsertCellPoint(NumberOfSides + i + 1);
		newPolys->InsertCellPoint(NumberOfSides + NumberOfSides + 1);
		newPolys->InsertNextCell(3);
		newPolys->InsertCellPoint(NumberOfSides + NumberOfSides - 1);
		newPolys->InsertCellPoint(NumberOfSides);
		newPolys->InsertCellPoint(NumberOfSides + NumberOfSides + 1);
	}

	output->SetPoints(newPoints);
	newPoints->Delete();

	output->SetPolys(newPolys);
	newPolys->Delete();

	return 1;
}

int vtkPolygonalFrustumSource::RequestInformation(vtkInformation* vtkNotUsed(request),
	vtkInformationVector** vtkNotUsed(inputVector), vtkInformationVector* outputVector)
{
	vtkInformation* outInfo = outputVector->GetInformationObject(0);
	outInfo->Set(CAN_HANDLE_PIECE_REQUEST(), 1);
	return 1;
}

void vtkPolygonalFrustumSource::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os, indent);

	os << indent << "Height: " << this->Height << "\n";
	os << indent << "TopRadius: " << this->TopRadius << "\n";
	os << indent << "BottomRadius: " << this->BottomRadius << "\n";
	os << indent << "NumberOfSides: " << this->NumberOfSides << "\n";

}
VTK_ABI_NAMESPACE_END
