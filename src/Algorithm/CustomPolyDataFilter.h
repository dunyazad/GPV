#pragma once

#include <Common.h>

class CustomPolyDataFilter : public vtkPolyDataAlgorithm
{
public:
    vtkTypeMacro(CustomPolyDataFilter, vtkPolyDataAlgorithm);
    static CustomPolyDataFilter* New();

protected:
    CustomPolyDataFilter() {}
    ~CustomPolyDataFilter() override {}

    // Override the RequestData method to process the data
    int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override
    {
        // Get the input and output
        vtkPolyData* input = vtkPolyData::GetData(this->GetInputInformation());
        vtkPolyData* output = vtkPolyData::GetData(this->GetOutputInformation(0));

        // Simple operation: just copy input to output
        output->ShallowCopy(input);

        return 1; // Return success
    }
};
