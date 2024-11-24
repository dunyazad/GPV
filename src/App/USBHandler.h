#pragma once

#ifdef _WINDOWS

#include <Common.h>
#include <hidapi.h>

class App;

struct USBEvent
{
	bool valid = false;
	int data[6] = {0, 0, 0, 0, 0, 0};
};

struct USBDevice
{
	hid_device_info* deviceInfo;
	hid_device* handle;
};

class USBHandler
{
public:
	USBHandler(App* pApp);
	~USBHandler();

private:
	App* pApp = nullptr;
	map<string, USBDevice> devices;
	thread* mainThread = nullptr;
	bool needToQuit = false;
};

#endif