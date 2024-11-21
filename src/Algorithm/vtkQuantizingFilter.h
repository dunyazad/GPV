#pragma once

#include <Common.h>

class vtkQuantizingFilter : public vtkPolyDataAlgorithm
{
public:
    static vtkQuantizingFilter* New();
    vtkQuantizingFilter(vtkQuantizingFilter, vtkPolyDataAlgorithm);

protected:
    vtkQuantizingFilter() {}
    ~vtkQuantizingFilter() override {}

    int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;

    unsigned int imageWidth = 256;
    unsigned int imageHeight = 480;
    float wInterval = 0.1f;
    float hInterval = 0.1f;
};
