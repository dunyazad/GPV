#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback_ZSparseBlocks(App* pApp)
{
	CUDA::ZSparseBlocks::TestZSparseBlocks();
}