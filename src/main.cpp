#include <Common.h>
#include <App/App.h>
#include <App/AppEventHandlers.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback(App* pApp);

int mode = 0;

int main()
{
	vtkActor* planeActor = nullptr;

	App app;
	app.AddKeyPressCallback(OnKeyPress);
	app.AddMouseButtonPressCallback(OnMouseButtonPress);
	app.AddMouseButtonReleaseCallback(OnMouseButtonRelease);
	app.AddMouseMoveReleaseCallback(OnMouseMove);
	app.AddAppStartCallback(AppStartCallback);

	if (mode == 0)
	{
		app.Run();
	}
	else if (mode == 1)
	{
		app.Run(AppConfiguration(256, 480, false, true));
	}

	return 0;
}
