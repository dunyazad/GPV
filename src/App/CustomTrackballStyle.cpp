#include <App/CustomTrackballStyle.h>

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

    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}

void CustomTrackballStyle::OnRightButtonDown()
{
    RightButtonPressed = true;
    std::cout << "Right Button Pressed" << std::endl;

    if (LeftButtonPressed && RightButtonPressed) {
        std::cout << "Both Left and Right Buttons Pressed" << std::endl;

        HandleBothButtons();
    }

    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}

void CustomTrackballStyle::OnLeftButtonUp()
{
    LeftButtonPressed = false;
    std::cout << "Left Button Released" << std::endl;

    vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
}

void CustomTrackballStyle::OnRightButtonUp()
{
    RightButtonPressed = false;
    std::cout << "Right Button Released" << std::endl;

    vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
}

void CustomTrackballStyle::HandleBothButtons()
{
    std::cout << "Custom behavior for both buttons being pressed" << std::endl;
}