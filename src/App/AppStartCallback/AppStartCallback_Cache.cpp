#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback_Cache(App* pApp)
{
	LoadPatch(10, pApp->GetRenderer());

	CUDA::TestCache();
}
