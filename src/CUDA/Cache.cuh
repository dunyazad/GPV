#pragma once

#include "CUDA_Common.cuh"

#include <App/Serialization.hpp>

struct cached_allocator;
struct CUstream_st;

#define _XYZ(v) (v).x, (v).y, (v).z
#define _XYZ_(v) (v).x(), (v).y(), (v).z()


namespace Utilities
{
	struct Point
	{
		int tag;
		float x;
		float y;
		float z;
		float nx;
		float ny;
		float nz;
		float r;
		float g;
		float b;
	};

	namespace Device
	{
		unsigned int GetFrameNumber();

		class Debugging
		{
		public:
			static void Initialize();
			static void Terminate();
			static void BeginFrame();
			static void EndFrame();
			static void CaptureOneFrame();
			static void Flush();

			static bool captureOneFrame;

			__device__ static int IsCaptureEnabled();

			__device__ static void AddPointP(int tag, const Eigen::Vector3f& point);
			__device__ static void AddPointPN(int tag, const Eigen::Vector3f& point, const Eigen::Vector3f& normal);
			__device__ static void AddPointPC(int tag, const Eigen::Vector3f& point, const Eigen::Vector3f& color);
			__device__ static void AddPointPNC(int tag, const Eigen::Vector3f& point, const Eigen::Vector3f& normal, const Eigen::Vector3f& color);

			static void SerializePoints(
				Eigen::Vector3f* inputPoints,
				Eigen::Vector3f* inputNormals,
				Eigen::Vector<unsigned char, 3>* inputColors, size_t numberOfInputPoints,
				const string& tagName);
		};

		class Cache
		{
		public:
			CUstream_st* stream = 0;

			Eigen::Vector3f min;
			float voxelSize = 0.1f;
			float pointMag = 1.0f;
			unsigned int xVoxelCount = 200;
			unsigned int yVoxelCount = 300;
			unsigned int zVoxelCount = 620;
			unsigned int zOffset = 100;
			unsigned int* cacheData = nullptr;
			float* cacheData2D = nullptr;

			Cache();

			~Cache();

			void Initialize(
				const Eigen::Vector3f& min,
				unsigned int voxelCountX = 200,
				unsigned int voxelCountY = 300,
				unsigned int voxelCountZ = 620,
				unsigned int zOffset = 100,
				float voxelSize = 0.1f,
				float pointMag = 1.0f);

			void Terminate();

			void SetStream(CUstream_st* st);

			void Clear();

			void Clustering();
		};
	}
}

namespace CUDA
{
	void TestCache();
}