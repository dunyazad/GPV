#pragma once

#define USE_CUDA

#include <stdio.h>
#include <math.h>

#ifndef USE_CUDA
#if !defined(__device__)
#define __device__
#endif
#if !defined(__host__)
#define __host__
#if !defined(__global__)
#define __global__
#endif
#endif
#else
#include <cuda_runtime.h>
#include <vector_types.h>
#include <nvtx3/nvToolsExt.h>
#endif

#ifndef __VECTOR_TYPES_H__
struct uint3
{
	unsigned int x;
	unsigned int y;
	unsigned int z;
};
uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { return uint3{ x, y, z }; }

struct float3
{
	float x;
	float y;
	float z;
};

float3 make_float3(float x, float y, float z) { return float3{ x, y, z }; }
#endif

#pragma region float3 math
#ifndef __CUDA_FLOAT3_MATH__
#define __CUDA_FLOAT3_MATH__
__host__ __device__
inline float3 operator+(const float3& a, const float3& b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline float3& operator+=(float3& a, const float3& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__host__ __device__
inline float3 operator-(const float3& a, const float3& b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__
inline float3& operator-=(float3& a, const float3& b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

__host__ __device__
inline float3 operator*(const float3& a, float scalar) {
	return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__host__ __device__
inline float3 operator*(float scalar, const float3& a) {
	return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__host__ __device__
inline float3& operator*=(float3& a, float scalar) {
	a.x *= scalar;
	a.y *= scalar;
	a.z *= scalar;
	return a;
}

__host__ __device__
inline float3 operator/(const float3& a, float scalar) {
	return make_float3(a.x / scalar, a.y / scalar, a.z / scalar);
}

__host__ __device__
inline float3& operator/=(float3& a, float scalar) {
	a.x /= scalar;
	a.y /= scalar;
	a.z /= scalar;
	return a;
}

__host__ __device__
inline float length(const float3& a) {
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__
inline float dot(const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline float3 cross(const float3& a, const float3& b) {
	return make_float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

__host__ __device__
inline float3 normalize(const float3& a) {
	float len = length(a);
	if (len > 0.0f) {
		return a / len;
	}
	else {
		return make_float3(0.0f, 0.0f, 0.0f);
	}
}
#endif
#pragma endregion

namespace MarchingCubes
{
	template<typename T>
	class MarchingCubesSurfaceExtractor;

#ifdef USE_CUDA
	template<typename T>
	__global__ void Kernel_Extract(typename MarchingCubesSurfaceExtractor<T>::Internal* internal);
#else
	template<typename T>
	__global__ void Kernel_Extract(size_t index, typename MarchingCubesSurfaceExtractor<T>::Internal* internal);
#endif

	template<typename T>
	class MarchingCubesSurfaceExtractor
	{
	public:
		MarchingCubesSurfaceExtractor(
			T* data,
			float3 volumeMin,
			float3 volumeMax,
			float voxelSize,
			float truncationDistance,
			float isoValue)
		{
			h_internal = new Internal(data, volumeMin, volumeMax, voxelSize, truncationDistance, isoValue);

#ifdef USE_CUDA
			cudaMalloc(&d_internal, sizeof(Internal));
			cudaMemcpy(d_internal, h_internal, sizeof(Internal), cudaMemcpyHostToDevice);
#endif
		}

		~MarchingCubesSurfaceExtractor()
		{
#ifdef USE_CUDA
			cudaFree(d_internal);
#endif
			delete h_internal;
		}

		void Extract()
		{
#ifdef USE_CUDA
			dim3 blockSize(8, 8, 8);
			dim3 gridSize(
				(h_internal->dimensions.x + blockSize.x - 1) / blockSize.x,
				(h_internal->dimensions.y + blockSize.y - 1) / blockSize.y,
				(h_internal->dimensions.z + blockSize.z - 1) / blockSize.z);

			// Correct kernel launch syntax
			Kernel_Extract<T> << <gridSize, blockSize >> > (d_internal);

			// Synchronize CUDA device to ensure all operations complete
			cudaDeviceSynchronize();
#else
			for (size_t i = 0; i < h_internal->numberOfVoxels; i++)
			{
				Kernel_Extract<T>(i, h_internal);
			}
#endif
		}

		struct Internal
		{
			T* data = nullptr;
			float3 volumeMin = make_float3(-5.0f, -5.0f, -5.0f);
			float3 volumeMax = make_float3(5.0f, 5.0f, 5.0f);
			float3 volumeCenter = (volumeMax + volumeMin) * 0.5f;
			float voxelSize = 0.1f;
			uint3 dimensions = make_uint3(
				(volumeMax.x - volumeMin.x) / voxelSize,
				(volumeMax.y - volumeMin.y) / voxelSize,
				(volumeMax.z - volumeMin.z) / voxelSize);
			size_t numberOfVoxels = (dimensions.x * dimensions.y * dimensions.z);
			float truncationDistance = 0.5f;
			float isoValue = 0.0f;
			int voxelNeighborRange = (int)ceilf(truncationDistance / voxelSize);
			float3* vertices = nullptr;
			uint3* triangles = nullptr;

			__device__ __host__
				Internal(
					T* data,
					float3 volumeMin,
					float3 volumeMax,
					float voxelSize,
					float truncationDistance,
					float isoValue)
				:
				data(data),
				volumeMin(volumeMin),
				volumeMax(volumeMax),
				volumeCenter((volumeMax + volumeMin) * 0.5f),
				voxelSize(0.1f),
				dimensions(make_uint3(
					(unsigned int)(ceilf(volumeMax.x - volumeMin.x) / voxelSize),
					(unsigned int)(ceilf(volumeMax.y - volumeMin.y) / voxelSize),
					(unsigned int)(ceilf(volumeMax.z - volumeMin.z) / voxelSize))),
				numberOfVoxels(dimensions.x* dimensions.y* dimensions.z),
				truncationDistance(0.5f),
				isoValue(0.0f),
				voxelNeighborRange((int)ceilf(truncationDistance / voxelSize))
			{
			}
		};

		Internal* h_internal = nullptr;
		Internal* d_internal = nullptr;
	};

#ifdef USE_CUDA
	template<typename T>
	__global__ void Kernel_Extract(typename MarchingCubesSurfaceExtractor<T>::Internal* internal)
#else
	template<typename T>
	__global__ void Kernel_Extract(size_t index, typename MarchingCubesSurfaceExtractor<T>::Internal* internal)
#endif
	{
#ifdef USE_CUDA
		size_t threadX = blockIdx.x * blockDim.x + threadIdx.x;
		size_t threadY = blockIdx.y * blockDim.y + threadIdx.y;
		size_t threadZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (threadX >= internal->dimensions.x ||
			threadY >= internal->dimensions.y ||
			threadZ >= internal->dimensions.z) return;

		size_t index = threadZ * (internal->dimensions.x * internal->dimensions.y) +
			threadY * internal->dimensions.x + threadX;
#endif

		printf("Kernel_Extract : %llu\n", index);
	}
}
