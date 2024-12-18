#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback_SaveTRN(App* pApp)
{
	auto renderer = pApp->GetRenderer();

	LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	//auto camera = renderer->GetActiveCamera();
	//camera->SetParallelProjection(true);
	//// Parallel Scale은 카메라 절반 높이
	//// 픽셀당 3D 공간의 유닛 * 창 높이 / 2
	//// 여기에선 256 x 480이므로 픽셀당 0.1, 창높이 480
	//// 480 * 0.1 / 2 = 24
	//camera->SetParallelScale(24);

	SaveTRNFile();
}
