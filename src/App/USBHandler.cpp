#include <App/USBHandler.h>

#include <App/App.h>

#ifdef _WINDOWS

USBHandler::USBHandler(App* pApp)
	: pApp(pApp), mainThread(nullptr), needToQuit(false)
{
	if (hid_init() != 0) {
		// Handle initialization error
	}

	USBDevice device;
	device.deviceInfo = hid_enumerate(0x0483, 0x5750); // 0x0, 0x0 lists all devices


	struct hid_device_info* devs, * cur_dev;
	//devs = hid_enumerate(0x0, 0x0); // 0x0, 0x0 lists all devices
	devs = hid_enumerate(0x0483, 0x5750); // 0x0, 0x0 lists all devices
	cur_dev = devs;
	while (cur_dev) {
		// Access device information
		if (cur_dev->manufacturer_string) {
			printf("Manufacturer: %ls\n", cur_dev->manufacturer_string);
		}
		else {
			printf("Manufacturer: (null)\n");
		}
		if (cur_dev->product_string) {
			printf("Product: %ls\n", cur_dev->product_string);
		}
		else {
			printf("Product: (null)\n");
		}
		printf("Vendor ID: 0x%04hx, Product ID: 0x%04hx\n", cur_dev->vendor_id, cur_dev->product_id);
		cur_dev = cur_dev->next;
	}

	if (nullptr == devs) return;

	devices["INVAIZ"] = device;

	mainThread = new thread([&](USBHandler* usbHandler) {
		if (0 == usbHandler->devices.count("INVAIZ")) return;

		auto& device = usbHandler->devices["INVAIZ"];

		device.handle = hid_open(0x0483, 0x5750, nullptr);
		hid_set_nonblocking(device.handle, 1); // 1 for non-blocking, 0 for blocking

		if (!device.handle) {
			// Handle error
		}

		unsigned char buf[65]; // Buffer size depends on device
		int res;

		while (false == usbHandler->needToQuit) {
			// Read input report
			res = hid_read(device.handle, buf, sizeof(buf));
			if (res > 0) {
				// Print the received data
				int codes[6];
				printf("HID Event: ");
				for (int i = 0; i < res; i++) {
					printf("%02X ", buf[i]);

					codes[i] = buf[i];
				}
				printf("\n");

				pApp->OnUSBEvent({ true, codes[0], codes[1], codes[2], codes[3], codes[4], codes[5] });
			}
			else if (res < 0) {
				fprintf(stderr, "Error reading from HID device\n");
				break;
			}

			//printf("Alive\n");
			Sleep(1);
		}

		hid_close(device.handle);
		}, this);
}

USBHandler::~USBHandler()
{
	needToQuit = true;
	if (nullptr != mainThread)
	{
		mainThread->join();

		//auto& device = devices["INVAIZ"];

		//auto cur_dev = device.deviceInfo;
		//while (cur_dev) {
		//	struct hid_device_info* next = cur_dev->next;
		//	free(cur_dev);  // Free the current device
		//	cur_dev = next;
		//}
	}

	hid_exit();
}

#endif
