#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback_HalfEdges(App* pApp)
{
	CUDA::HalfEdge::TestHalfEdge();
}
