#include <Common.h>

#include <App/CustomTrackballStyle.h>
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

    void AddAppStartCallback(function<void(App*)> f);
    void AddAppStartCallback(const string& name, function<void(App*)> f);
    void RemoveAppStartCallback();
    void RemoveAppStartCallback(const string& name);

    void AddAppUpdateCallback(function<void(App*)> f);
    void AddAppUpdateCallback(const string& name, function<void(App*)> f);
    void RemoveAppUpdateCallback();
    void RemoveAppUpdateCallback(const string& name);

    void AddAppPostRenderCallback(function<void(App*)> f);
    void AddAppPostRenderCallback(const string& name, function<void(App*)> f);
    void RemoveAppPostRenderCallback();
    void RemoveAppPostRenderCallback(const string& name);

    void AddKeyPressCallback(function<void(App*)> f);
    void AddKeyPressCallback(const string& name, function<void(App*)> f);
    void RemoveKeyPressCallback();
    void RemoveKeyPressCallback(const string& name);

    void AddMouseButtonPressCallback(function<void(App*, int)> f);
    void AddMouseButtonPressCallback(const string& name, function<void(App*, int)> f);
    void RemoveMouseButtonPressCallback();
    void RemoveMouseButtonPressCallback(const string& name);

    void AddMouseButtonReleaseCallback(function<void(App*, int)> f);
    void AddMouseButtonReleaseCallback(const string& name, function<void(App*, int)> f);
    void RemoveMouseButtonReleaseCallback();
    void RemoveMouseButtonReleaseCallback(const string& name);

    void AddMouseMoveReleaseCallback(function<void(App*, int, int, int, int, bool, bool, bool)> f);
    void AddMouseMoveReleaseCallback(const string& name, function<void(App*, int, int, int, int, bool, bool, bool)> f);
    void RemoveMouseMoveReleaseCallback();
    void RemoveMouseMoveReleaseCallback(const string& name);

    void OnUpdate();
    void OnPostRender();

    void CaptureColorAndDepth(const string& saveDirectory);
    void CaptureAsPointCloud(const string& saveDirectory);

    static void OnKeyPress();
    static void OnMouseButtonPress(int button);
    static void OnMouseButtonRelease(int button);
    static void OnMouseMove(int posx, int posy, int lastx, int lasty, bool lButton, bool mButton, bool rButton);

    inline vtkSmartPointer<vtkRenderer> GetRenderer() const { return renderer; }
    inline vtkSmartPointer<vtkRenderWindow> GetRenderWindow() const { return renderWindow; }
    inline vtkSmartPointer<vtkRenderWindowInteractor> GetInteractor() const { return interactor; }

    inline AppConfiguration* Configuration() { return &configuration; }

    map<string, void*> registry;
private:
    static set<App*> s_instances;
    AppConfiguration configuration;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor;
    vtkSmartPointer<CustomTrackballStyle> customTrackballStyle;

    vtkSmartPointer<vtkCallbackCommand> keyPressCallback;

    vtkSmartPointer<TimerCallback> timerCallback;
    vtkSmartPointer<PostRenderCallback> postRenderCallback;

    map<string, function<void(App*)>> appStartCallbacks;
    map<string, function<void(App*)>> appUpdateCallbacks;
    map<string, function<void(App*)>> appPostRenderCallbacks;
    map<string, function<void(App*)>> keyPressCallbacks;
    map<string, function<void(App*, int)>> mouseButtonPressCallbacks;
    map<string, function<void(App*, int)>> mouseButtonReleaseCallbacks;
    map<string, function<void(App*, int, int, int, int, bool, bool, bool)>> mouseMoveCallbacks;

    bool captureEnabled = false;
};
