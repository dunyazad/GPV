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

    app->OnMouseButtonPress(0);

    //vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}

void CustomTrackballStyle::OnRightButtonDown()
{
    RightButtonPressed = true;
    std::cout << "Right Button Pressed" << std::endl;

    if (LeftButtonPressed && RightButtonPressed) {
        std::cout << "Both Left and Right Buttons Pressed" << std::endl;

        HandleBothButtons();
    }

    app->OnMouseButtonPress(2);

    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}

void CustomTrackballStyle::OnLeftButtonUp()
{
    LeftButtonPressed = false;
    std::cout << "Left Button Released" << std::endl;

    app->OnMouseButtonRelease(0);

    //vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
}

void CustomTrackballStyle::OnRightButtonUp()
{
    RightButtonPressed = false;
    std::cout << "Right Button Released" << std::endl;

    app->OnMouseButtonRelease(2);

    vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
}

void CustomTrackballStyle::HandleBothButtons()
{
    std::cout << "Custom behavior for both buttons being pressed" << std::endl;
}

void CustomTrackballStyle::OnKeyPress()
{
    std::string key = this->GetInteractor()->GetKeySym();
    std::cout << "Key pressed: " << key << std::endl;

    app->OnKeyPress();

    vtkInteractorStyleTrackballCamera::OnKeyPress();
}

void CustomTrackballStyle::OnMouseMove()
{
    int* pos = this->GetInteractor()->GetEventPosition();
    int* lastPos = this->GetInteractor()->GetLastEventPosition();

    app->OnMouseMove(pos[0], pos[1], lastPos[0], lastPos[1], LeftButtonPressed, MiddleButtonPressed, RightButtonPressed);
    //printf("[%4d, %4d] - [%4d, %4d]\n", lastPos[0], lastPos[1], pos[0], pos[1]);
}