#pragma once

#include <vtkObjectFactory.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

class CustomTrackballStyle : public vtkInteractorStyleTrackballCamera
{
public:
    static CustomTrackballStyle* New();
    vtkTypeMacro(CustomTrackballStyle, vtkInteractorStyleTrackballCamera);

    CustomTrackballStyle();

    virtual void OnLeftButtonDown() override;
    virtual void OnRightButtonDown() override;
    virtual void OnLeftButtonUp() override;
    virtual void OnRightButtonUp() override;
    void HandleBothButtons();

private:
    bool LeftButtonPressed;
    bool RightButtonPressed;
};
