#include <Algorithm/vtkQuantizingFilter.h>

vtkStandardNewMacro(vtkQuantizingFilter);

int vtkQuantizingFilter::RequestData(vtkInformation* request,
    vtkInformationVector** inputVector,
    vtkInformationVector* outputVector)
{
    vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
    vtkPolyData* input = vtkPolyData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));

    auto points = input->GetPoints();

    vtkNew<vtkPoints> newPoints;
    newPoints->SetNumberOfPoints(imageWidth * imageHeight);
    for (unsigned int h = 0; h < imageHeight; h++)
    {
        for (unsigned int w = 0; w < imageWidth; w++)
        {
            float point[3];
            point[0] = ((float)w - ((float)imageWidth * 0.5f)) * wInterval;
            point[1] = ((float)h - ((float)imageHeight * 0.5f)) * hInterval;
            point[2] = -1000;

            newPoints->SetPoint((vtkIdType)(h * imageWidth + w), point);
        }
    }

    auto nop = points->GetNumberOfPoints();
    for (size_t i = 0; i < nop; i++)
    {
        auto p = points->GetPoint(i);
        auto x = p[0];
        auto y = p[1];
        auto z = p[2];

        auto w = (unsigned int)(x / wInterval) + imageWidth / 2;
        auto h = (unsigned int)(y / hInterval) + imageHeight / 2;

        auto point = newPoints->GetPoint((vtkIdType)(h * imageWidth + w));
        point[2] = z;
        newPoints->SetPoint((vtkIdType)(h * imageWidth + w), point);
    }

    //input->SetPoints(newPoints);
    //input->Modified();

    vtkInformation* outInfo = outputVector->GetInformationObject(0);
    vtkPolyData* output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    output->SetPoints(newPoints);
    output->Modified();

    //output->ShallowCopy(input);

    return 1;
}