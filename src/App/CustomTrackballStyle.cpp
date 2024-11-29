#include <App/CustomTrackballStyle.h>
#include <App/App.h>

// Implement the New method
vtkStandardNewMacro(CustomTrackballStyle);

CustomTrackballStyle::CustomTrackballStyle() {
    LeftButtonPressed = false;
    RightButtonPressed = false;
}

void CustomTrackballStyle::OnLeftButtonDown()
{
    LeftButtonPressed = true;
    std::cout << "Left Button Pressed" << std::endl;

    if (LeftButtonPressed && RightButtonPressed) {
        std::cout << "Both Left and Right Buttons Pressed" << std::endl;

        HandleBothButtons();
    }

    bool propagateEvent = true;
    propagateEvent = app->OnMouseButtonPress(0);

    //vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}

void CustomTrackballStyle::OnMiddleButtonDown()
{
    MiddleButtonPressed = true;
    std::cout << "Middle Button Pressed" << std::endl;

    bool propagateEvent = true;
    propagateEvent = app->OnMouseButtonPress(1);

    if (propagateEvent)
    {
        vtkInteractorStyleTrackballCamera::OnMiddleButtonDown();
    }
}

void CustomTrackballStyle::OnRightButtonDown()
{
    RightButtonPressed = true;
    std::cout << "Right Button Pressed" << std::endl;

    if (LeftButtonPressed && RightButtonPressed) {
        std::cout << "Both Left and Right Buttons Pressed" << std::endl;

        HandleBothButtons();
    }

    bool propagateEvent = true;
    propagateEvent = app->OnMouseButtonPress(2);

    if (propagateEvent)
    {
        vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
    }
}

void CustomTrackballStyle::OnLeftButtonUp()
{
    LeftButtonPressed = false;
    std::cout << "Left Button Released" << std::endl;

    bool propagateEvent = true;
    propagateEvent = app->OnMouseButtonRelease(0);

    //vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
}

void CustomTrackballStyle::OnMiddleButtonUp()
{
    MiddleButtonPressed = false;
    std::cout << "Middle Button Released" << std::endl;

    bool propagateEvent = true;
    propagateEvent = app->OnMouseButtonRelease(1);

    if (propagateEvent)
    {
        vtkInteractorStyleTrackballCamera::OnMiddleButtonUp();
    }
}

void CustomTrackballStyle::OnRightButtonUp()
{
    RightButtonPressed = false;
    std::cout << "Right Button Released" << std::endl;

    bool propagateEvent = true;
    propagateEvent = app->OnMouseButtonRelease(2);

    if (propagateEvent)
    {
        vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
    }
}

void CustomTrackballStyle::HandleBothButtons()
{
    std::cout << "Custom behavior for both buttons being pressed" << std::endl;
}

void CustomTrackballStyle::OnKeyPress()
{
    std::string key = this->GetInteractor()->GetKeySym();
    //std::cout << "Key pressed: " << key << std::endl;

    bool propagateEvent = true;
    propagateEvent = app->OnKeyPress();

    if (propagateEvent)
    {
        vtkInteractorStyleTrackballCamera::OnKeyPress();
    }
}

void CustomTrackballStyle::OnKeyRelease()
{
    std::string key = this->GetInteractor()->GetKeySym();
    //std::cout << "Key Releaseed: " << key << std::endl;

    bool propagateEvent = true;
    propagateEvent = app->OnKeyRelease();

    if (propagateEvent)
    {
        vtkInteractorStyleTrackballCamera::OnKeyRelease();
    }
}

void CustomTrackballStyle::OnMouseMove()
{
    int* pos = this->GetInteractor()->GetEventPosition();
    int* lastPos = this->GetInteractor()->GetLastEventPosition();

    bool propagateEvent = true;
    propagateEvent = app->OnMouseMove(pos[0], pos[1], lastPos[0], lastPos[1], LeftButtonPressed, MiddleButtonPressed, RightButtonPressed);
    //printf("[%4d, %4d] - [%4d, %4d]\n", lastPos[0], lastPos[1], pos[0], pos[1]);
}

void CustomTrackballStyle::OnMouseWheelForward()
{
    bool propagateEvent = true;
    propagateEvent = app->OnMouseWheelScroll(true);

    if (propagateEvent)
    {
        vtkInteractorStyleTrackballCamera::OnMouseWheelForward();
    }
}

void CustomTrackballStyle::OnMouseWheelBackward()
{
    bool propagateEvent = true;
    propagateEvent = app->OnMouseWheelScroll(false);
    if (propagateEvent)
    {
        vtkInteractorStyleTrackballCamera::OnMouseWheelBackward();
    }
}
