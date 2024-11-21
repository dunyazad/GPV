#pragma once

#include <Common.h>

class vtkMedianFilter : public vtkPolyDataAlgorithm
{
public:
    static vtkMedianFilter* New();
    vtkMedianFilter(vtkMedianFilter, vtkPolyDataAlgorithm);

protected:
    vtkMedianFilter() {}
    ~vtkMedianFilter() override {}

    int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
};
