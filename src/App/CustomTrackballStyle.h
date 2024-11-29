#pragma once

#include <vtkObjectFactory.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>

class App;

class CustomTrackballStyle : public vtkInteractorStyleTrackballCamera
{
public:
    static CustomTrackballStyle* New();
    vtkTypeMacro(CustomTrackballStyle, vtkInteractorStyleTrackballCamera);

    CustomTrackballStyle();

    virtual void OnLeftButtonDown() override;
    virtual void OnMiddleButtonUp() override;
    virtual void OnRightButtonDown() override;
    virtual void OnLeftButtonUp() override;
    virtual void OnMiddleButtonDown() override;
    virtual void OnRightButtonUp() override;
    void HandleBothButtons();

    virtual void OnKeyPress() override;
    virtual void OnKeyRelease() override;
    virtual void OnMouseMove() override;
    virtual void OnMouseWheelForward() override;
    virtual void OnMouseWheelBackward() override;

    inline void SetApp(App* app) { this->app = app; }

private:
    App* app = nullptr;
    bool LeftButtonPressed = false;
    bool MiddleButtonPressed = false;
    bool RightButtonPressed = false;
};
