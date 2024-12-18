#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback_Patches(App* pApp)
{
	auto renderer = pApp->GetRenderer();
	auto t = Time::Now();

	LoadPatch(0, renderer);
	t = Time::End(t, "Loading Patch");

	for (size_t i = 0; i < 400 * 480; i++)
	{
		auto& p = points_0[i];
		if (FLT_MAX != p.x() && FLT_MAX != p.y() && FLT_MAX != p.z())
		{
			VD::AddSphere("points_0", p, 0.05f, Color4::Red);
		}
	}

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
	t = Time::End(t, "Visualize");
}