#pragma once

#define USE_CUDA

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <tuple>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <vector_types.h>
#include <nvtx3/nvToolsExt.h>
#else
#define __host__
#define __device__
#define __global__
#define __constant__
#endif

#define FLT_VALID(x) ((x) < 3.402823466e+36F)
#define FLT_NOT_VALID(x) ((x) > 3.402823466e+36F)

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
	__device__ __constant__
		const float3 vertexOffsets[8] = {
			{0.0f, 0.0f, 0.0f},
			{1.0f, 0.0f, 0.0f},
			{1.0f, 0.0f, 1.0f},
			{0.0f, 0.0f, 1.0f},
			{0.0f, 1.0f, 0.0f},
			{1.0f, 1.0f, 0.0f},
			{1.0f, 1.0f, 1.0f},
			{0.0f, 1.0f, 1.0f} };

	__device__ __constant__
		const int edgeTable[256] = {
			0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
			0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
			0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
			0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
			0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
			0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
			0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
			0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
			0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
			0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
			0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
			0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
			0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
			0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
			0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
			0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
			0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
			0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
			0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
			0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
			0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
			0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
			0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
			0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
			0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
			0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
			0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
			0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
			0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
			0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
			0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
			0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };

	__device__ __constant__
		const int edgeVertexMap[12][2] = {
			{0, 1},  // Edge 0 connects Vertex 0 ¡æ Vertex 1
			{1, 2},  // Edge 1 connects Vertex 1 ¡æ Vertex 2
			{2, 3},  // Edge 2 connects Vertex 2 ¡æ Vertex 3
			{3, 0},  // Edge 3 connects Vertex 3 ¡æ Vertex 0
			{4, 5},  // Edge 4 connects Vertex 4 ¡æ Vertex 5
			{5, 6},  // Edge 5 connects Vertex 5 ¡æ Vertex 6
			{6, 7},  // Edge 6 connects Vertex 6 ¡æ Vertex 7
			{7, 4},  // Edge 7 connects Vertex 7 ¡æ Vertex 4
			{0, 4},  // Edge 8 connects Vertex 0 ¡æ Vertex 4
			{1, 5},  // Edge 9 connects Vertex 1 ¡æ Vertex 5
			{2, 6},  // Edge 10 connects Vertex 2 ¡æ Vertex 6
			{3, 7}   // Edge 11 connects Vertex 3 ¡æ Vertex 7
	};

	__device__ __constant__
		const int triTable[256][16] =
	{ {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };
}

namespace MarchingCubes
{
	struct MarchingCubesResult
	{
		float3* vertices;
		unsigned int numberOfVertices;
		uint3* triangles;
		unsigned int numberOfTriangles;
	};

	template<typename T>
	class MarchingCubesSurfaceExtractor;

#ifdef USE_CUDA
	template<typename T>
	__global__ void Kernel_ExtractVertices(typename MarchingCubesSurfaceExtractor<T>::Internal* internal);

	template<typename T>
	__global__ void Kernel_ExtractTriangles(typename MarchingCubesSurfaceExtractor<T>::Internal* internal);
#else
	template<typename T>
	void Kernel_ExtractVertices(size_t index, typename MarchingCubesSurfaceExtractor<T>::Internal* internal);

	template<typename T>
	void Kernel_ExtractTriangles(size_t index, typename MarchingCubesSurfaceExtractor<T>::Internal* internal);
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
			float isoValue)
		{
			h_internal = new Internal(data, volumeMin, volumeMax, voxelSize, isoValue);

			printf("h_internal->numberOfVoxels : %llu\n", h_internal->numberOfVoxels);

#ifdef USE_CUDA
			cudaMalloc(&h_internal->vertices, sizeof(float3) * h_internal->numberOfVoxels * 3);
			cudaMalloc(&h_internal->vertexMapping, sizeof(unsigned int) * h_internal->numberOfVoxels * 3);
			cudaMalloc(&h_internal->triangles, sizeof(uint3) * h_internal->numberOfVoxels * 4);

			cudaMemset(h_internal->vertices, 0, sizeof(float3) * h_internal->numberOfVoxels * 3);
			cudaMemset(h_internal->vertexMapping, 0, sizeof(unsigned int) * h_internal->numberOfVoxels * 3);
			cudaMemset(h_internal->triangles, 0, sizeof(uint3) * h_internal->numberOfVoxels * 4);

			cudaMalloc(&h_internal->vertexCounterPtr, sizeof(unsigned int));
			cudaMalloc(&h_internal->triangleCounterPtr, sizeof(unsigned int));

			cudaMalloc(&d_internal, sizeof(Internal));
			cudaMemcpy(d_internal, h_internal, sizeof(Internal), cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();
#else
			h_internal->vertexCounterPtr = new unsigned int;
			h_internal->triangleCounterPtr = new unsigned int;

			h_internal->vertices = new float3[h_internal->numberOfVoxels * 3];
			h_internal->vertexMapping = new unsigned int[h_internal->numberOfVoxels * 3];
			h_internal->triangles = new uint3[h_internal->numberOfVoxels * 4];
#endif
		}

		~MarchingCubesSurfaceExtractor()
		{
#ifdef USE_CUDA
			cudaFree(h_internal->vertexCounterPtr);
			cudaFree(h_internal->triangleCounterPtr);

			cudaFree(h_internal->vertices);
			cudaFree(h_internal->vertexMapping);
			cudaFree(h_internal->triangles);

			cudaFree(d_internal);
#else
			delete h_internal->vertexCounterPtr;
			delete h_internal->triangleCounterPtr;

			delete[] h_internal->vertices;
			delete[] h_internal->vertexMapping;
			delete[] h_internal->triangles;
#endif
			delete[] h_internal;
		}

		//std::tuple<float3*, unsigned int, uint3*, unsigned int> Extract()
		MarchingCubesResult Extract()
		{
#ifdef USE_CUDA
			nvtxRangePushA("Extract()");
			dim3 blockSize(8, 8, 8);
			dim3 gridSize(
				(h_internal->dimensions.x + blockSize.x - 1) / blockSize.x,
				(h_internal->dimensions.y + blockSize.y - 1) / blockSize.y,
				(h_internal->dimensions.z + blockSize.z - 1) / blockSize.z);

			printf("grid size : %d, %d, %d\n", gridSize.x, gridSize.y, gridSize.z);

			nvtxRangePushA("Kernel_ExtractVertices()");

			Kernel_ExtractVertices<T> << <gridSize, blockSize >> > (d_internal);

			cudaDeviceSynchronize();

			nvtxRangePop();

			nvtxRangePushA("Kernel_ExtractTriangles()");

			Kernel_ExtractTriangles<T> << <gridSize, blockSize >> > (d_internal);

			cudaDeviceSynchronize();
			nvtxRangePop();

			cudaError_t err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
			}

			unsigned int h_vertexCount;
			unsigned int h_triangleCount;
			cudaMemcpy(&h_vertexCount, h_internal->vertexCounterPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&h_triangleCount, h_internal->triangleCounterPtr, sizeof(unsigned int), cudaMemcpyDeviceToHost);
#else
			for (size_t i = 0; i < h_internal->numberOfVoxels; i++)
			{
				Kernel_ExtractVertices<T>(i, h_internal);
			}

			for (size_t i = 0; i < h_internal->numberOfVoxels; i++)
			{
				Kernel_ExtractTriangles<T>(i, h_internal);
			}

			unsigned int h_vertexCount;
			unsigned int h_triangleCount;
#endif

			float3* vertices = new float3[h_vertexCount];
			cudaMemcpy(vertices, h_internal->vertices, sizeof(float3) * h_vertexCount, cudaMemcpyDeviceToHost);

			uint3* triangles = new uint3[h_triangleCount];
			cudaMemcpy(triangles, h_internal->triangles, sizeof(uint3) * h_triangleCount, cudaMemcpyDeviceToHost);

			printf("Extracted %d triangle, %d vertices\n", h_triangleCount, h_vertexCount);

#ifdef USE_CUDA
			nvtxRangePop();
#endif
			//return make_tuple(vertices, h_vertexCount, triangles, h_triangleCount);
			return { vertices, h_vertexCount, triangles, h_triangleCount };
		}

		struct Internal
		{
			T* data = nullptr;
			float3 volumeMin = make_float3(-5.0f, -5.0f, -5.0f);
			float3 volumeMax = make_float3(5.0f, 5.0f, 5.0f);
			float3 volumeCenter = (volumeMax + volumeMin) * 0.5f;
			float voxelSize = 0.1f;
			uint3 dimensions = make_uint3(
				(unsigned int)(ceilf(volumeMax.x - volumeMin.x) / voxelSize),
				(unsigned int)(ceilf(volumeMax.y - volumeMin.y) / voxelSize),
				(unsigned int)(ceilf(volumeMax.z - volumeMin.z) / voxelSize));
			size_t numberOfVoxels = (dimensions.x * dimensions.y * dimensions.z);
			float isoValue = 0.0f;
			float3* vertices = nullptr;
			unsigned int* vertexMapping = nullptr;
			uint3* triangles = nullptr;
			unsigned int* vertexCounterPtr = nullptr;
			unsigned int* triangleCounterPtr = nullptr;

			__device__ __host__
				Internal(
					T* data,
					float3 volumeMin,
					float3 volumeMax,
					float voxelSize,
					float isoValue) :
				data(data),
				volumeMin(volumeMin),
				volumeMax(volumeMax),
				volumeCenter((volumeMax + volumeMin) * 0.5f),
				voxelSize(voxelSize),
				dimensions(make_uint3(
					(unsigned int)(ceilf(volumeMax.x - volumeMin.x) / voxelSize),
					(unsigned int)(ceilf(volumeMax.y - volumeMin.y) / voxelSize),
					(unsigned int)(ceilf(volumeMax.z - volumeMin.z) / voxelSize))),
				numberOfVoxels(dimensions.x* dimensions.y* dimensions.z),
				isoValue(isoValue)
			{
			}
		};

		Internal* h_internal = nullptr;
		Internal* d_internal = nullptr;
	};

	__host__ __device__
		inline size_t GetFlatIndex(const uint3& index, const uint3& dimensions)
	{
		if (index.x >= dimensions.x || index.y >= dimensions.y || index.z >= dimensions.z) {
			return UINT_MAX;
		}
		return index.z * dimensions.x * dimensions.y + index.y * dimensions.x + index.x;
	}

#ifdef USE_CUDA
	template<typename T>
	__global__ void Kernel_ExtractVertices(typename MarchingCubesSurfaceExtractor<T>::Internal* internal)
#else
	template<typename T>
	void Kernel_ExtractVertices(size_t index, typename MarchingCubesSurfaceExtractor<T>::Internal* internal)
#endif
	{
#ifdef USE_CUDA
		size_t indexX = blockIdx.x * blockDim.x + threadIdx.x;
		size_t indexY = blockIdx.y * blockDim.y + threadIdx.y;
		size_t indexZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (indexX >= internal->dimensions.x ||
			indexY >= internal->dimensions.y ||
			indexZ >= internal->dimensions.z) return;

		size_t index = indexZ * (internal->dimensions.x * internal->dimensions.y) +
			indexY * internal->dimensions.x + indexX;
#else
		size_t indexZ = index / (internal->dimensions.x * internal->dimensions.y);
		size_t indexY = (index / (internal->dimensions.x * internal->dimensions.y)) % internal->dimensions.x;
		size_t indexX = (index / (internal->dimensions.x * internal->dimensions.y)) / internal->dimensions.x;
#endif

		if (index >= internal->numberOfVoxels) return;

		float3 voxelPosition = internal->volumeMin + make_float3(
			(float)indexX * internal->voxelSize,
			(float)indexY * internal->voxelSize,
			(float)indexZ * internal->voxelSize);

		float tsdf[8];
		for (int i = 0; i < 8; ++i) {
			uint3 cornerIndex = make_uint3(
				indexX + (vertexOffsets[i].x > 0.0f ? 1 : 0),
				indexY + (vertexOffsets[i].y > 0.0f ? 1 : 0),
				indexZ + (vertexOffsets[i].z > 0.0f ? 1 : 0));
			size_t cornerFlatIndex = GetFlatIndex(cornerIndex, internal->dimensions);

			if (UINT_MAX == cornerFlatIndex)
			{
				//printf("%d, %d, %d\n", cornerIndex.x, cornerIndex.y, cornerIndex.z);
				tsdf[i] = FLT_MAX;
				continue;
			}

			if (cornerFlatIndex < internal->numberOfVoxels)
			{
				tsdf[i] = internal->data[cornerFlatIndex];
			}
			else {
				tsdf[i] = FLT_MAX;
			}
		}

		if (FLT_VALID(tsdf[0]) && FLT_VALID(tsdf[1]))
		{
			float alpha = 0.5f;
			float diff = tsdf[0] - tsdf[1];
			if (fabs(diff) > 1e-6) {
				alpha = (internal->isoValue - tsdf[0]) / (tsdf[1] - tsdf[0]);
			}

			if (0 <= alpha && alpha <= 1.0f)
			{
#ifdef USE_CUDA
				int vertexIndex = atomicAdd(internal->vertexCounterPtr, 1);
#else
				int vertexIndex = (*internal->vertexCounterPtr);
				(*internal->vertexCounterPtr) += 1;
#endif
				internal->vertexMapping[index * 3] = vertexIndex;
				internal->vertices[vertexIndex] = voxelPosition + alpha * make_float3(internal->voxelSize, 0.0f, 0.0f);
			}
		}

		if (FLT_VALID(tsdf[0]) && FLT_VALID(tsdf[4]))
		{
			float alpha = 0.5f;
			float diff = tsdf[0] - tsdf[4];
			if (fabs(diff) > 1e-6) {
				alpha = (internal->isoValue - tsdf[0]) / (tsdf[4] - tsdf[0]);
			}

			if (0 <= alpha && alpha <= 1.0f)
			{
#ifdef USE_CUDA
				int vertexIndex = atomicAdd(internal->vertexCounterPtr, 1);
#else
				int vertexIndex = (*internal->vertexCounterPtr);
				(*internal->vertexCounterPtr) += 1;
#endif
				internal->vertexMapping[index * 3 + 1] = vertexIndex;
				internal->vertices[vertexIndex] = voxelPosition + alpha * make_float3(0.0f, internal->voxelSize, 0.0f);
			}
		}

		if (FLT_VALID(tsdf[0]) && FLT_VALID(tsdf[3]))
		{
			float alpha = 0.5f;
			float diff = tsdf[0] - tsdf[3];
			if (fabs(diff) > 1e-6) {
				alpha = (internal->isoValue - tsdf[0]) / (tsdf[3] - tsdf[0]);
			}

			if (0 <= alpha && alpha <= 1.0f)
			{
#ifdef USE_CUDA
				int vertexIndex = atomicAdd(internal->vertexCounterPtr, 1);
#else
				int vertexIndex = (*internal->vertexCounterPtr);
				(*internal->vertexCounterPtr) += 1;
#endif
				internal->vertexMapping[index * 3 + 2] = vertexIndex;
				internal->vertices[vertexIndex] = voxelPosition + alpha * make_float3(0.0f, 0.0f, internal->voxelSize);
			}
		}
	}

#ifdef USE_CUDA
	template<typename T>
	__global__ void Kernel_ExtractTriangles(typename MarchingCubesSurfaceExtractor<T>::Internal* internal)
#else
	template<typename T>
	void Kernel_ExtractTriangles(size_t index, typename MarchingCubesSurfaceExtractor<T>::Internal* internal)
#endif
	{
#ifdef USE_CUDA
		size_t indexX = blockIdx.x * blockDim.x + threadIdx.x;
		size_t indexY = blockIdx.y * blockDim.y + threadIdx.y;
		size_t indexZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (indexX >= internal->dimensions.x ||
			indexY >= internal->dimensions.y ||
			indexZ >= internal->dimensions.z) return;

		size_t index = indexZ * (internal->dimensions.x * internal->dimensions.y) +
			indexY * internal->dimensions.x + indexX;
#else
		size_t indexZ = index / (internal->dimensions.x * internal->dimensions.y);
		size_t indexY = (index / (internal->dimensions.x * internal->dimensions.y)) % internal->dimensions.x;
		size_t indexX = (index / (internal->dimensions.x * internal->dimensions.y)) / internal->dimensions.x;
#endif

		if (index >= internal->numberOfVoxels) return;

		float3 voxelPosition = internal->volumeMin + make_float3(
			(float)indexX * internal->voxelSize,
			(float)indexY * internal->voxelSize,
			(float)indexZ * internal->voxelSize);
		
		float tsdf[8];
		for (int i = 0; i < 8; ++i) {
			uint3 cornerIndex = make_uint3(
				indexX + (vertexOffsets[i].x > 0.0f ? 1 : 0),
				indexY + (vertexOffsets[i].y > 0.0f ? 1 : 0),
				indexZ + (vertexOffsets[i].z > 0.0f ? 1 : 0));
			size_t cornerFlatIndex = GetFlatIndex(cornerIndex, internal->dimensions);

			if (UINT_MAX == cornerFlatIndex)
			{
				//printf("%d, %d, %d\n", cornerIndex.x, cornerIndex.y, cornerIndex.z);
				tsdf[i] = FLT_MAX;
				continue;
			}

			if (cornerFlatIndex < internal->numberOfVoxels) {
				tsdf[i] = internal->data[cornerFlatIndex];
			}
			else {
				tsdf[i] = FLT_MAX;
			}
		}

		int cubeIndex = 0;
		for (int i = 0; i < 8; ++i) {
			if (FLT_VALID(tsdf[i]) &&
				//(-truncationDistance <= tsdf[i] || tsdf[i] <= truncationDistance) &&
				tsdf[i] < internal->isoValue)
			{
				cubeIndex |= (1 << i);
			}
		}

		int edge = edgeTable[cubeIndex];
		if (edge == 0) return;

		size_t edgeVertexIndexMapping[12];
		if (edge & 1) edgeVertexIndexMapping[0] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 0, indexY + 0, indexZ + 0), internal->dimensions) * 3 + 0];
		if (edge & 2) edgeVertexIndexMapping[1] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 1, indexY + 0, indexZ + 0), internal->dimensions) * 3 + 2];
		if (edge & 4) edgeVertexIndexMapping[2] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 0, indexY + 0, indexZ + 1), internal->dimensions) * 3 + 0];
		if (edge & 8) edgeVertexIndexMapping[3] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 0, indexY + 0, indexZ + 0), internal->dimensions) * 3 + 2];
		if (edge & 16) edgeVertexIndexMapping[4] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 0, indexY + 1, indexZ + 0), internal->dimensions) * 3 + 0];
		if (edge & 32) edgeVertexIndexMapping[5] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 1, indexY + 1, indexZ + 0), internal->dimensions) * 3 + 2];
		if (edge & 64) edgeVertexIndexMapping[6] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 0, indexY + 1, indexZ + 1), internal->dimensions) * 3 + 0];
		if (edge & 128) edgeVertexIndexMapping[7] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 0, indexY + 1, indexZ + 0), internal->dimensions) * 3 + 2];
		if (edge & 256) edgeVertexIndexMapping[8] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 0, indexY + 0, indexZ + 0), internal->dimensions) * 3 + 1];
		if (edge & 512) edgeVertexIndexMapping[9] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 1, indexY + 0, indexZ + 0), internal->dimensions) * 3 + 1];
		if (edge & 1024) edgeVertexIndexMapping[10] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 1, indexY + 0, indexZ + 1), internal->dimensions) * 3 + 1];
		if (edge & 2048) edgeVertexIndexMapping[11] = internal->vertexMapping[GetFlatIndex(make_uint3(indexX + 0, indexY + 0, indexZ + 1), internal->dimensions) * 3 + 1];

		for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
			auto i0 = edgeVertexIndexMapping[triTable[cubeIndex][i]];
			auto i1 = edgeVertexIndexMapping[triTable[cubeIndex][i + 1]];
			auto i2 = edgeVertexIndexMapping[triTable[cubeIndex][i + 2]];

			if (0 == i0 || 0 == i1 || 0 == i2)
				continue;

#ifdef USE_CUDA
			int triangleIndex = atomicAdd(internal->triangleCounterPtr, 1);
#else
			int triangleIndex = (*internal->triangleCounterPtr);
			(*internal->triangleCounterPtr) += 1;
#endif
			internal->triangles[triangleIndex] = make_uint3(i0, i1, i2);
		}
	}
}
