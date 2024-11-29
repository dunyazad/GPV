#pragma once

#include <Common.h>

class App;

class AppEventHandler
{
public:
	AppEventHandler();
	~AppEventHandler();

private:

};

bool OnKeyPress(App* app);
bool OnKeyRelease(App* app);

bool OnMouseButtonPress(App* app, int button);
bool OnMouseButtonRelease(App* app, int button);
bool OnMouseMove(App* app, int posx, int posy, int lastx, int lasty, bool lButton, bool mButton, bool rButton);
bool OnMouseWheelScroll(App* app, bool isForward);
#ifdef _WINDOWS
bool OnUSBEvent(App* app, USBEvent usbEvent);
#endif