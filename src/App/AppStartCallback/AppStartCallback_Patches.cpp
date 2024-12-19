#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback_Patches(App* pApp)
{
	auto renderer = pApp->GetRenderer();
	auto t = Time::Now();

	LoadPatch(0, renderer);
	t = Time::End(t, "Loading Patch");

	for (size_t i = 0; i < patchPoints_0.size(); i++)
	{
		auto& p = patchPoints_0[i];
		if (FLT_MAX != p.x() && FLT_MAX != p.y() && FLT_MAX != p.z())
		{
			VD::AddSphere("points_0", p, 0.05f, Color4::Red);
		}
	}

	//VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, 40.0f, 48.0f, 0.1f, Color4::Red);
	//VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f }, 40.0f, 48.0f, 0.1f, Color4::Yellow);
	//VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 40.0f, 48.0f, 0.1f, Color4::Green);
	//VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 1.0f }, 40.0f, 48.0f, 0.1f, Color4::Cyan);
	VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 40.0f, 48.0f, 0.1f, Color4::Blue);
	//VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 1.0f }, 40.0f, 48.0f, 0.1f, Color4::Magenta);

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
	t = Time::End(t, "Visualize");
}
