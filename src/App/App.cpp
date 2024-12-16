#include <App/App.h>
#include <CUDA/CUDA.cuh>
#include <App/Serialization.hpp>

set<App*> App::s_instances;

TimerCallback* TimerCallback::New() { return new TimerCallback; }
void TimerCallback::SetApp(App* app) { this->app = app; }

void TimerCallback::Execute(vtkObject* caller, unsigned long eventId, void* vtkNotUsed(callData))
{
	if (eventId == vtkCommand::TimerEvent)
	{
		OnTimer();
	}
	else
	{
		std::cerr << "Unexpected event ID: " << eventId << std::endl;
	}
}

void TimerCallback::OnTimer()
{
	app->OnUpdate();

	VisualDebugging::Update();
}

PostRenderCallback* PostRenderCallback::New() { return new PostRenderCallback; }
void PostRenderCallback::SetApp(App* app) { this->app = app; }

void PostRenderCallback::Execute(vtkObject* caller, unsigned long eventId, void* vtkNotUsed(callData))
{
	if (eventId == vtkCommand::EndEvent)
	{
		OnPostRender();
	}
	else
	{
		std::cerr << "Unexpected event ID: " << eventId << std::endl;
	}
}

void PostRenderCallback::OnPostRender()
{
	app->OnPostRender();
}


App::App()
#ifdef _WINDOWS
	: usbHandler(this)
#endif
{
	s_instances.insert(this);
}

App::~App()
{
	s_instances.erase(this);
}

void App::Run()
{
#ifdef _WINDOWS
	if (configuration.maximizeConsoleWindow)
	{
		MaximizeConsoleWindowOnMonitor(1);
	}
#endif

	vtkObject::GlobalWarningDisplayOff();

	renderer = vtkSmartPointer<vtkRenderer>::New();
	renderer->SetBackground(0.3, 0.5, 0.7);

	//renderer->GetActiveCamera()->SetClippingRange(0.001, 40.0);

	renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->SetSize(configuration.windowWidth, configuration.windowHeight);
	renderWindow->AddRenderer(renderer);

	interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	customTrackballStyle = vtkSmartPointer<CustomTrackballStyle>::New();
	customTrackballStyle->SetApp(this);
	//customTrackballStyle->SetMotionFactor(300);
	interactor->SetInteractorStyle(customTrackballStyle);
	interactor->SetRenderWindow(renderWindow);
	interactor->Initialize();

	VisualDebugging::Initialize(renderer);

#ifdef _WINDOWS
	if (configuration.maximizeRenderWindow)
	{
		MaximizeVTKWindowOnMonitor(renderWindow, 2);
	}
#endif

	timerCallback = vtkSmartPointer<TimerCallback>::New();
	timerCallback->SetApp(this);

	interactor->AddObserver(vtkCommand::TimerEvent, timerCallback);
	int timerId = interactor->CreateRepeatingTimer(16);
	if (timerId < 0) {
		std::cerr << "Error: Timer was not created!" << std::endl;
	}

	postRenderCallback = vtkSmartPointer<PostRenderCallback>::New();
	postRenderCallback->SetApp(this);

	renderWindow->AddObserver(vtkCommand::EndEvent, postRenderCallback);

	for (auto& kvp : appStartCallbacks)
	{
		kvp.second(this);
	}

	renderWindow->Render();
	interactor->Start();

	VisualDebugging::Terminate();
}

void App::Run(AppConfiguration configuration)
{
	this->configuration.windowWidth = configuration.windowWidth;
	this->configuration.windowHeight = configuration.windowHeight;
	this->configuration.maximizeRenderWindow = configuration.maximizeRenderWindow;
	this->configuration.maximizeConsoleWindow = configuration.maximizeConsoleWindow;

	Run();
}

void App::AddAppStartCallback(function<bool(App*)> f)
{
	AddAppStartCallback("Default", f);
}

void App::AddAppStartCallback(const string& name, function<bool(App*)> f)
{
	if (0 != appStartCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	appStartCallbacks[name] = f;
}

void App::RemoveAppStartCallback()
{
	RemoveAppStartCallback("Default");
}

void App::RemoveAppStartCallback(const string& name)
{
	if (0 != appStartCallbacks.count(name))
	{
		appStartCallbacks.erase(name);
	}
}

void App::AddAppUpdateCallback(function<bool(App*)> f)
{
	AddAppUpdateCallback("Default", f);
}

void App::AddAppUpdateCallback(const string& name, function<bool(App*)> f)
{
	if (0 != appUpdateCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	appUpdateCallbacks[name] = f;
}

void App::RemoveAppUpdateCallback()
{
	RemoveAppUpdateCallback("Default");
}

void App::RemoveAppUpdateCallback(const string& name)
{
	if (0 != appUpdateCallbacks.count(name))
	{
		appUpdateCallbacks.erase(name);
	}
}

void App::AddAppPostRenderCallback(function<bool(App*)> f)
{
	AddAppPostRenderCallback("Default", f);
}

void App::AddAppPostRenderCallback(const string& name, function<bool(App*)> f)
{
	if (0 != appPostRenderCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	appPostRenderCallbacks[name] = f;
}

void App::RemoveAppPostRenderCallback()
{
	RemoveAppPostRenderCallback("Default");
}

void App::RemoveAppPostRenderCallback(const string& name)
{
	if (0 != appPostRenderCallbacks.count(name))
	{
		appPostRenderCallbacks.erase(name);
	}
}

void App::AddKeyPressCallback(function<bool(App*)> f)
{
	AddKeyPressCallback("Default", f);
}

void App::AddKeyPressCallback(const string& name, function<bool(App*)> f)
{
	if (0 != keyPressCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	keyPressCallbacks[name] = f;
}

void App::RemoveKeyPressCallback()
{
	RemoveKeyPressCallback("Default");
}

void App::RemoveKeyPressCallback(const string& name)
{
	if (0 != keyPressCallbacks.count(name))
	{
		keyPressCallbacks.erase(name);
	}
}

void App::AddKeyReleaseCallback(function<bool(App*)> f)
{
	AddKeyReleaseCallback("Default", f);
}

void App::AddKeyReleaseCallback(const string& name, function<bool(App*)> f)
{
	if (0 != keyReleaseCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	keyReleaseCallbacks[name] = f;
}

void App::RemoveKeyReleaseCallback()
{
	RemoveKeyReleaseCallback("Default");
}

void App::RemoveKeyReleaseCallback(const string& name)
{
	if (0 != keyReleaseCallbacks.count(name))
	{
		keyReleaseCallbacks.erase(name);
	}
}

void App::AddMouseButtonPressCallback(function<bool(App*, int)> f)
{
	AddMouseButtonPressCallback("Default", f);
}

void App::AddMouseButtonPressCallback(const string& name, function<bool(App*, int)> f)
{
	if (0 != mouseButtonPressCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	mouseButtonPressCallbacks[name] = f;
}

void App::RemoveMouseButtonPressCallback()
{
	RemoveMouseButtonPressCallback("Default");
}

void App::RemoveMouseButtonPressCallback(const string& name)
{
	if (0 != mouseButtonReleaseCallbacks.count(name))
	{
		mouseButtonReleaseCallbacks.erase(name);
	}
}

void App::AddMouseButtonReleaseCallback(function<bool(App*, int)> f)
{
	AddMouseButtonReleaseCallback("Default", f);
}

void App::AddMouseButtonReleaseCallback(const string& name, function<bool(App*, int)> f)
{
	if (0 != mouseButtonReleaseCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	mouseButtonReleaseCallbacks[name] = f;
}

void App::RemoveMouseButtonReleaseCallback()
{
	RemoveMouseButtonReleaseCallback("Default");
}

void App::RemoveMouseButtonReleaseCallback(const string& name)
{
	if (0 != mouseButtonReleaseCallbacks.count(name))
	{
		mouseButtonReleaseCallbacks.erase(name);
	}
}

void App::AddMouseMoveCallback(function<bool(App*, int, int, int, int, bool, bool, bool)> f)
{
	AddMouseMoveCallback("Default", f);
}

void App::AddMouseMoveCallback(const string& name, function<bool(App*, int, int, int, int, bool, bool, bool)> f)
{
	if (0 != mouseMoveCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	mouseMoveCallbacks[name] = f;
}

void App::RemoveMouseMoveCallback()
{
	RemoveMouseMoveCallback("Default");
}

void App::RemoveMouseMoveCallback(const string& name)
{
	if (0 != mouseMoveCallbacks.count(name))
	{
		mouseMoveCallbacks.erase(name);
	}
}

void App::AddMouseWheelScrollCallback(function<bool(App*, bool)> f)
{
	AddMouseWheelScrollCallback("Default", f);
}

void App::AddMouseWheelScrollCallback(const string& name, function<bool(App*, bool)> f)
{
	if (0 != mouseWheelScrollCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	mouseWheelScrollCallbacks[name] = f;
}

void App::RemoveMouseWheelScrollCallback()
{
	RemoveMouseWheelScrollCallback("Default");
}

void App::RemoveMouseWheelScrollCallback(const string& name)
{
	if (0 != mouseWheelScrollCallbacks.count(name))
	{
		mouseWheelScrollCallbacks.erase(name);
	}
}

#ifdef _WINDOWS
void App::AddUSBEventCallback(function<bool(App*, USBEvent)> f)
{
	AddUSBEventCallback("Default", f);
}

void App::AddUSBEventCallback(const string& name, function<bool(App*, USBEvent)> f)
{
	if (0 != usbEventCallbacks.count(name))
	{
		printf("[Error] same name callback exists!");
	}
	usbEventCallbacks[name] = f;
}

void App::RemoveUSBEventCallback()
{
	RemoveUSBEventCallback("Default");
}

void App::RemoveUSBEventCallback(const string& name)
{
	if (0 != usbEventCallbacks.count(name))
	{
		usbEventCallbacks.erase(name);
	}
}
#endif

void App::OnUpdate()
{
#ifdef _WINDOWS
	for (auto& instance : s_instances)
	{
		USBEvent usbEvent;
		unique_lock<mutex> lock(instance->usbEventQueueLock);
		if (false == instance->usbEventQueue.empty())
		{
			usbEvent = instance->usbEventQueue.front();
			instance->usbEventQueue.pop();
		}
		lock.unlock();

		if (usbEvent.valid)
		{
			for (auto& kvp : instance->usbEventCallbacks)
			{
				kvp.second(instance, usbEvent);
			}
		}
	}
#endif

	for (auto& kvp : appUpdateCallbacks)
	{
		kvp.second(this);
	}
}

void App::OnPostRender()
{
	for (auto& kvp : appPostRenderCallbacks)
	{
		kvp.second(this);
	}
}

bool App::OnKeyPress()
{
	bool propagateEvent = true;

	for (auto& instance : s_instances)
	{
		for (auto& kvp : instance->keyPressCallbacks)
		{
			propagateEvent &= kvp.second(instance);
		}
	}

	return propagateEvent;
}

bool App::OnKeyRelease()
{
	bool propagateEvent = true;

	for (auto& instance : s_instances)
	{
		for (auto& kvp : instance->keyReleaseCallbacks)
		{
			propagateEvent &= kvp.second(instance);
		}
	}

	return propagateEvent;
}

bool App::OnMouseButtonPress(int button)
{
	bool propagateEvent = true;

	for (auto& instance : s_instances)
	{
		for (auto& kvp : instance->mouseButtonPressCallbacks)
		{
			propagateEvent &= kvp.second(instance, button);
		}
	}

	return propagateEvent;
}

bool App::OnMouseButtonRelease(int button)
{
	bool propagateEvent = true;

	for (auto& instance : s_instances)
	{
		for (auto& kvp : instance->mouseButtonReleaseCallbacks)
		{
			propagateEvent &= kvp.second(instance, button);
		}
	}

	return propagateEvent;
}

bool App::OnMouseMove(int posx, int posy, int lastx, int lasty, bool lButton, bool mButton, bool rButton)
{
	bool propagateEvent = true;

	for (auto& instance : s_instances)
	{
		for (auto& kvp : instance->mouseMoveCallbacks)
		{
			propagateEvent &= kvp.second(instance, posx, posy, lastx, lasty, lButton, mButton, rButton);
		}
	}

	return propagateEvent;
}

bool App::OnMouseWheelScroll(bool direction)
{
	bool propagateEvent = true;

	for (auto& instance : s_instances)
	{
		for (auto& kvp : instance->mouseWheelScrollCallbacks)
		{
			propagateEvent &= kvp.second(instance, direction);
		}
	}

	return propagateEvent;
}

#ifdef _WINDOWS
bool App::OnUSBEvent(USBEvent usbEvent)
{
	for (auto& instance : s_instances)
	{
		for (auto& kvp : instance->usbEventCallbacks)
		{
			unique_lock<mutex> lock(instance->usbEventQueueLock);

			instance->usbEventQueue.push(usbEvent);

			lock.unlock();
		}
	}

	return true;
}
#endif

void App::CaptureColorAndDepth(const string& saveDirectory)
{
	static int captureCount = 0;
	std::stringstream ss;
	ss << captureCount++;

	std::string depthDataFileName = saveDirectory + "\\depth_data_" + ss.str() + ".dpt";
	std::string depthmapFileName = saveDirectory + "\\depth_" + ss.str() + ".png";
	std::string colormapFileName = saveDirectory + "\\color_" + ss.str() + ".png";

	{
		vtkNew<vtkWindowToImageFilter> colorFilter;
		colorFilter->SetInput(renderWindow);
		colorFilter->SetInputBufferTypeToRGB();
		colorFilter->Update();

		vtkNew<vtkPNGWriter> colorWriter;
		colorWriter->SetFileName(colormapFileName.c_str());
		colorWriter->SetInputConnection(colorFilter->GetOutputPort());
		colorWriter->Write();
	}

	{
		vtkNew<vtkWindowToImageFilter> depthFilter;
		depthFilter->SetInput(renderWindow);
		depthFilter->SetInputBufferTypeToZBuffer();
		depthFilter->Update();

		vtkNew<vtkImageShiftScale> shiftScale;
		shiftScale->SetInputConnection(depthFilter->GetOutputPort());
		shiftScale->SetOutputScalarTypeToUnsignedChar();
		shiftScale->SetShift(0);
		shiftScale->SetScale(255);
		shiftScale->Update();

		vtkNew<vtkPNGWriter> depthWriter;
		depthWriter->SetFileName(depthmapFileName.c_str());
		depthWriter->SetInputConnection(shiftScale->GetOutputPort());
		depthWriter->Write();
	}

	{
		ofstream ofs;
		ofs.open(depthDataFileName, ios::out | ios::binary);

		int width = 256;
		int height = 480;
		ofs.write((char*)&width, sizeof(int));
		ofs.write((char*)&height, sizeof(int));

		Eigen::Matrix4f viewMatrix = vtkToEigen(renderer->GetActiveCamera()->GetViewTransformMatrix());
		Eigen::Matrix4f tm = viewMatrix.inverse();
		//auto& tm = cameraTransforms[0];

		vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
		windowToImageFilter->SetInput(GetRenderWindow());
		windowToImageFilter->SetInputBufferTypeToZBuffer(); // Get the depth buffer
		windowToImageFilter->Update();

		// Access the depth buffer as an image
		vtkSmartPointer<vtkImageData> depthImage = windowToImageFilter->GetOutput();

		// Get the depth values as a float array
		vtkSmartPointer<vtkFloatArray> depthArray = vtkFloatArray::SafeDownCast(depthImage->GetPointData()->GetScalars());

		double* clippingRange = renderer->GetActiveCamera()->GetClippingRange();
		float depthRatio = (float)(clippingRange[1] - clippingRange[0]);

		vtkIdType index = 0;
		for (float y = -24.0f; y < 24.0f; y += 0.1f)
		{
			for (float x = -12.8f; x < 12.8f; x += 0.1f)
			{
				float depth = depthArray->GetValue(index++);
				auto p = Transform(tm, { x, y, -depth * depthRatio });
				ofs.write((char*)p.data(), sizeof(Eigen::Vector3f));
			}
		}

		ofs.close();
	}
}

void App::CaptureAsPointCloud(const string& saveDirectory)
{
	static int captureCount = 0;
	std::stringstream ss;
	ss << captureCount++;
	std::string pointCloudFileName = saveDirectory + "\\point_" + ss.str() + ".ply";

	auto windowToColorImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
	windowToColorImageFilter->SetInput(renderWindow);
	//windowToColorImageFilter->SetMagnification(1); // Optional, set to >1 for higher resolution
	windowToColorImageFilter->SetInputBufferTypeToRGB(); // Capture only RGB, not alpha
	windowToColorImageFilter->Update();

	auto imageData = windowToColorImageFilter->GetOutput();

	auto colorArray = vtkUnsignedCharArray::SafeDownCast(imageData->GetPointData()->GetScalars());

	vtkNew<vtkPoints> points;

	Eigen::Matrix4f viewMatrix = vtkToEigen(renderer->GetActiveCamera()->GetViewTransformMatrix());
	Eigen::Matrix4f tm = viewMatrix.inverse();

	vtkSmartPointer<vtkWindowToImageFilter> windowToDepthImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
	windowToDepthImageFilter->SetInput(GetRenderWindow());
	windowToDepthImageFilter->SetInputBufferTypeToZBuffer(); // Get the depth buffer
	windowToDepthImageFilter->Update();

	vtkSmartPointer<vtkImageData> depthImage = windowToDepthImageFilter->GetOutput();

	vtkSmartPointer<vtkFloatArray> depthArray = vtkFloatArray::SafeDownCast(depthImage->GetPointData()->GetScalars());

	double* clippingRange = renderer->GetActiveCamera()->GetClippingRange();
	float depthRatio = (float)(clippingRange[1] - clippingRange[0]);

	vector<Eigen::Vector3f> inputPoints;

	PLYFormat ply;

	vtkIdType index = 0;
	for (float y = -24.0f; y < 24.0f; y += 0.1f)
	{
		for (float x = -12.8f; x < 12.8f; x += 0.1f)
		{
			float depth = depthArray->GetValue(index);
			if (depth < 1.0f)
			{
				auto p = Transform(tm, { x, y, -depth * depthRatio });
				//auto p = Eigen::Vector3f(x, y, -depth * depthRatio);

				ply.AddPointFloat3(p.data());

				inputPoints.push_back(p);

				//points->InsertNextPoint(p.x(), p.y(), p.z());

				unsigned char rgb[3];
				colorArray->GetTypedTuple(index, rgb);

				ply.AddColor((float)rgb[0] / 255.0f, (float)rgb[1] / 255.0f, (float)rgb[2] / 255.0f);
			}
			else
			{
				auto p = Transform(tm, { x, y, -depth * depthRatio });
				//auto p = Eigen::Vector3f(x, y, -depth * depthRatio);

				inputPoints.push_back(p);
			}
			index++;
		}
	}

	vector<Eigen::Vector3f> h_normals;
	CUDA::cuCache::GeneratePatchNormals(256, 480, inputPoints, h_normals);

	for (size_t i = 0; i < h_normals.size(); i++)
	{
		float depth = depthArray->GetValue(i);
		if (depth < 1.0f)
		{
			auto n = h_normals[i];
			ply.AddNormalFloat3(n.data());
		}
	}

	ply.Serialize(pointCloudFileName);
}
