#pragma once

#include <Common.h>

class App;

void OnKeyPress(App* app);

void OnMouseButtonPress(App* app, int button);
void OnMouseButtonRelease(App* app, int button);
void OnMouseMove(App* app, int posx, int posy, int lastx, int lasty, bool lButton, bool mButton, bool rButton);