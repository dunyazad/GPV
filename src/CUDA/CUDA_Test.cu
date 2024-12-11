#include <CUDA/CUDA_Test.cuh>

#include <Algorithm/MarchingCubes.hpp>

#include <thrust/async/for_each.h>
#include <thrust/async/reduce.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <App/Serialization.hpp>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void CUDA_Test()
{
	//float* d_volume;
	//cudaMallocManaged(&d_volume, sizeof(float) * 100 * 100 * 100);
	//thrust::device_ptr<float> dev_ptr(d_volume);
	//thrust::fill_n(d_volume, 1000000, 0.0f);

	//thrust::for_each(thrust::counting_iterator(0), thrust::counting_iterator(1000000),
	//	[=] __device__(size_t index) {
	//	size_t ix = index % 100;
	//	size_t iy = (index / 100) % 100;
	//	size_t iz = index / 10000;

	//	float dx = 5.0f - (float)ix;
	//	float dy = 5.0f - (float)iy;
	//	float dz = 5.0f - (float)iz;
	//	float distance = sqrtf(dx * dx + dy * dy + dz * dz);
	//	d_volume[index] = distance - 2.5f;

	//	//printf("distance : %f\n", distance);
	//});

	//cudaDeviceSynchronize();

	//float* h_volume = new float[1000000];
	//cudaMemcpy(h_volume, d_volume, sizeof(float) * 1000000, cudaMemcpyDeviceToHost);

	//cudaDeviceSynchronize();

	////PLYFormat ply;
	////for (size_t z = 0; z < 100; z++)
	////{
	////	for (size_t y = 0; y < 100; y++)
	////	{
	////		for (size_t x = 0; x < 100; x++)
	////		{
	////			size_t flatIndex = x + y * 100 + z * 10000;
	////			float distance = h_volume[flatIndex];

	////			if (distance < 50)
	////			{
	////				ply.AddPoint(x, y, z);
	////			}
	////		}
	////	}
	////}

	////ply.Serialize("C:\\Resources\\Debug\\Sphere.ply");

	//MarchingCubes::MarchingCubesSurfaceExtractor<float> mc(
	//	d_volume,
	//	make_float3(-5.0f, -5.0f, -5.0f),
	//	make_float3(5.0f, 5.0f, 5.0f),
	//	0.1f,
	//	2.5f);

	//auto result = mc.Extract();

	//PLYFormat ply;

	//for (size_t i = 0; i < result.numberOfVertices; i++)
	//{
	//	auto v0 = result.vertices[i];
	//	auto v1 = result.vertices[i];
	//	auto v2 = result.vertices[i];

	//	ply.AddPoint(v0.x, v0.y, v0.z);
	//	ply.AddPoint(v1.x, v1.y, v1.z);
	//	ply.AddPoint(v2.x, v2.y, v2.z);

	//	//VD::AddTriangle("Marching Cubes", { v0.x, v0.y, v0.z }, { v1.x, v1.y, v1.z }, { v2.x, v2.y, v2.z }, Color4::White);
	//}

	//for (size_t i = 0; i < result.numberOfTriangles; i++)
	//{
	//	auto i0 = result.triangles[i].x;
	//	auto i1 = result.triangles[i].y;
	//	auto i2 = result.triangles[i].z;

	//	ply.AddIndex(i0);
	//	ply.AddIndex(i1);
	//	ply.AddIndex(i2);

	//	//VD::AddTriangle("Marching Cubes", { v0.x, v0.y, v0.z }, { v1.x, v1.y, v1.z }, { v2.x, v2.y, v2.z }, Color4::White);
	//}

	//ply.Serialize("C:\\Resources\\Debug\\MC.ply");
}
