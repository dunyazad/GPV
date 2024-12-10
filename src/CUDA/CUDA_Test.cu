#include <CUDA/CUDA_Test.cuh>

#include <Algorithm/MarchingCubes.hpp>

#define USE_CUDA

void CUDA_Test()
{
#ifdef USE_CUDA
	MarchingCubes::MarchingCubesSurfaceExtractor<float3> mc(
		nullptr,
		make_float3(-5.0f, -5.0f, -5.0f),
		make_float3(5.0f, 5.0f, 5.0f),
		0.1f,
		0.5f,
		0.0f);

	mc.Extract();
#endif
}