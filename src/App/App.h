#include <Common.h>

#include <App/CustomTrackballStyle.h>
#include <App/USBHandler.h>
#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>

class App;

class TimerCallback : public vtkCommand
{
public:
    static TimerCallback* New();
    TimerCallback() = default;

    App* app;
    void SetApp(App* app);

    virtual void Execute(vtkObject* caller, unsigned long eventId, void* vtkNotUsed(callData)) override;

private:
    void OnTimer();
};

class PostRenderCallback : public vtkCommand
{
public:
    static PostRenderCallback* New();
    PostRenderCallback() = default;

    App* app;
    void SetApp(App* app);

    virtual void Execute(vtkObject * caller, unsigned long eventId, void* vtkNotUsed(callData)) override;

private:
    void OnPostRender();
};

struct AppConfiguration
{
    int windowWidth = 1920;
    int windowHeight = 1080;
    bool maximizeRenderWindow = true;
    bool maximizeConsoleWindow = true;

    AppConfiguration()
    {
        windowWidth = 1920;
        windowHeight = 1080;
        maximizeRenderWindow = true;
        maximizeConsoleWindow = true;
    }

    AppConfiguration(
        int windowWidth,
        int windowHeight,
        bool maximizeRenderWindow,
        bool maximizeConsoleWindow)
        : windowWidth(windowWidth),
        windowHeight(windowHeight),
        maximizeRenderWindow(maximizeRenderWindow),
        maximizeConsoleWindow(maximizeConsoleWindow)
    {
    }
};

class App
{
public:
	App();
	~App();

    void Run();
	void Run(AppConfiguration configuration);

    void AddAppStartCallback(function<bool(App*)> f);
    void AddAppStartCallback(const string& name, function<bool(App*)> f);
    void RemoveAppStartCallback();
    void RemoveAppStartCallback(const string& name);

    void AddAppUpdateCallback(function<bool(App*)> f);
    void AddAppUpdateCallback(const string& name, function<bool(App*)> f);
    void RemoveAppUpdateCallback();
    void RemoveAppUpdateCallback(const string& name);

    void AddAppPostRenderCallback(function<bool(App*)> f);
    void AddAppPostRenderCallback(const string& name, function<bool(App*)> f);
    void RemoveAppPostRenderCallback();
    void RemoveAppPostRenderCallback(const string& name);

    void AddKeyPressCallback(function<bool(App*)> f);
    void AddKeyPressCallback(const string& name, function<bool(App*)> f);
    void RemoveKeyPressCallback();
    void RemoveKeyPressCallback(const string& name);

    void AddKeyReleaseCallback(function<bool(App*)> f);
    void AddKeyReleaseCallback(const string& name, function<bool(App*)> f);
    void RemoveKeyReleaseCallback();
    void RemoveKeyReleaseCallback(const string& name);

    void AddMouseButtonPressCallback(function<bool(App*, int)> f);
    void AddMouseButtonPressCallback(const string& name, function<bool(App*, int)> f);
    void RemoveMouseButtonPressCallback();
    void RemoveMouseButtonPressCallback(const string& name);

    void AddMouseButtonReleaseCallback(function<bool(App*, int)> f);
    void AddMouseButtonReleaseCallback(const string& name, function<bool(App*, int)> f);
    void RemoveMouseButtonReleaseCallback();
    void RemoveMouseButtonReleaseCallback(const string& name);

    void AddMouseMoveCallback(function<bool(App*, int, int, int, int, bool, bool, bool)> f);
    void AddMouseMoveCallback(const string& name, function<bool(App*, int, int, int, int, bool, bool, bool)> f);
    void RemoveMouseMoveCallback();
    void RemoveMouseMoveCallback(const string& name);

    void AddMouseWheelScrollCallback(function<bool(App*, bool)> f);
    void AddMouseWheelScrollCallback(const string& name, function<bool(App*, bool)> f);
    void RemoveMouseWheelScrollCallback();
    void RemoveMouseWheelScrollCallback(const string& name);

#ifdef _WINDOWS
    void AddUSBEventCallback(function<bool(App*, USBEvent)> f);
    void AddUSBEventCallback(const string& name, function<bool(App*, USBEvent)> f);
    void RemoveUSBEventCallback();
    void RemoveUSBEventCallback(const string& name);
#endif

    void OnUpdate();
    void OnPostRender();

    void CaptureColorAndDepth(const string& saveDirectory);
    void CaptureAsPointCloud(const string& saveDirectory);

    static bool OnKeyPress();
    static bool OnKeyRelease();
    static bool OnMouseButtonPress(int button);
    static bool OnMouseButtonRelease(int button);
    static bool OnMouseMove(int posx, int posy, int lastx, int lasty, bool lButton, bool mButton, bool rButton);
    static bool OnMouseWheelScroll(bool direction);
#ifdef _WINDOWS
    static bool OnUSBEvent(USBEvent usbEvent);
#endif

    inline vtkSmartPointer<vtkRenderer> GetRenderer() const { return renderer; }
    inline vtkSmartPointer<vtkRenderWindow> GetRenderWindow() const { return renderWindow; }
    inline vtkSmartPointer<vtkRenderWindowInteractor> GetInteractor() const { return interactor; }

    inline AppConfiguration* Configuration() { return &configuration; }

    //inline map<string, bool> GetKeyStates() { return keyStates; }
    inline bool GetKeyState(string key) { return keyStates[key]; }
    inline void SetKeyState(string key, bool pressed) { keyStates[key] = pressed; }

    map<string, void*> registry;

private:
    static set<App*> s_instances;
    AppConfiguration configuration;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor;
    vtkSmartPointer<CustomTrackballStyle> customTrackballStyle;

    vtkSmartPointer<vtkCallbackCommand> keyPressCallback;
    map<string, bool> keyStates;

    vtkSmartPointer<TimerCallback> timerCallback;
    vtkSmartPointer<PostRenderCallback> postRenderCallback;

    map<string, function<bool(App*)>> appStartCallbacks;
    map<string, function<bool(App*)>> appUpdateCallbacks;
    map<string, function<bool(App*)>> appPostRenderCallbacks;
    map<string, function<bool(App*)>> keyPressCallbacks;
    map<string, function<bool(App*)>> keyReleaseCallbacks;
    map<string, function<bool(App*, int)>> mouseButtonPressCallbacks;
    map<string, function<bool(App*, int)>> mouseButtonReleaseCallbacks;
    map<string, function<bool(App*, int, int, int, int, bool, bool, bool)>> mouseMoveCallbacks;
    map<string, function<bool(App*, bool)>> mouseWheelScrollCallbacks;

#ifdef _WINDOWS
    map<string, function<bool(App*, USBEvent)>> usbEventCallbacks;

    USBHandler usbHandler;
    mutex usbEventQueueLock;
    queue<USBEvent> usbEventQueue;
#endif

    bool captureEnabled = false;
};
