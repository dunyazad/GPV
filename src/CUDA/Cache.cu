#include "Cache.cuh"

#include <App/Utility.h>

#include <App/Serialization.hpp>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

#include "nvapi.h"
#include "NvApiDriverSettings.h"

namespace Utilities
{
	namespace Device
	{
		const unsigned int POINT_COUNT = 50000000;

		unsigned int h_frameNumber = 0;
		unsigned int GetFrameNumber() { return h_frameNumber; }

		__device__ unsigned int frameNumber = 0;
		__device__ int captureEnabled = 0;
		__device__ Point* points = nullptr;
		__device__ unsigned int numberOfPoints = 0;
		__device__ unsigned int frameCount = 0;
		__device__ unsigned int flushCount = 0;

		Point* d_points;

#pragma region Debugging
		bool Debugging::captureOneFrame = false;

		void Debugging::Initialize()
		{
			cudaMalloc(&d_points, sizeof(Point) * POINT_COUNT);
			cudaMemcpyToSymbol(points, &d_points, sizeof(Point*));

			int enabled = 0;
			cudaMemcpyToSymbol(captureEnabled, &enabled, sizeof(int));
		}

		void Debugging::Terminate()
		{
			cudaFree(d_points);
		}

		void Debugging::BeginFrame()
		{
			alog("[%4d] Debugging::BeginFrame() Begin\n", h_frameNumber);

			if (captureOneFrame)
			{
				int enabled = 1;
				cudaMemcpyToSymbol(captureEnabled, &enabled, sizeof(int));

				alog("[%4d] set captureEnabled to 1\n", h_frameNumber);
			}

			alog("[%4d] Debugging::BeginFrame() End\n", h_frameNumber);
		}

		void Debugging::EndFrame()
		{
			alog("[%4d] Debugging::EndFrame() Begin\n", h_frameNumber);

			if (captureOneFrame)
			{
				Flush();
			}

			cudaMemcpyFromSymbol(&h_frameNumber, frameNumber, sizeof(unsigned int));
			h_frameNumber++;
			cudaMemcpyToSymbol(frameNumber, &h_frameNumber, sizeof(unsigned int));

			alog("[%4d] Debugging::EndFrame() End\n", h_frameNumber);
		}

		void Debugging::CaptureOneFrame()
		{
			captureOneFrame = true;
		}

		int Debugging::IsCaptureEnabled()
		{
			return captureEnabled;
		}

		void Debugging::Flush()
		{
			alog("[%4d] Debugging::Flush() Begin\n", h_frameNumber);

			unsigned int h_numberOfPoints = 0;
			cudaMemcpyFromSymbol(&h_numberOfPoints, numberOfPoints, sizeof(unsigned int));

			unsigned int h_flushCount = 0;
			cudaMemcpyFromSymbol(&h_flushCount, flushCount, sizeof(unsigned int));

			cudaMemcpyFromSymbol(&d_points, points, sizeof(Point*));
			Point* h_points = new Point[h_numberOfPoints];
			cudaMemcpy(h_points, d_points, sizeof(Point) * h_numberOfPoints, cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();

			alog("[%4d] h_numberOfPoints : %d\n", h_frameNumber, h_numberOfPoints);

			vector<PLYFormat> plys(30);

			for (size_t i = 0; i < h_numberOfPoints; i++)
			{
				int tag = h_points[i].tag;
				plys[tag].AddPoint(h_points[i].x, h_points[i].y, h_points[i].z);
			}

			for (size_t i = 0; i < plys.size(); i++)
			{
				if (0 < plys[i].GetPoints().size())
				{
					stringstream ss;
					ss << "C:\\Resources\\Debug\\Serialized\\Serialized_" << h_flushCount << "_" << i << ".ply";
					plys[i].Serialize(ss.str());
				}
			}

			delete h_points;

			h_numberOfPoints = 0;
			cudaMemcpyToSymbol(numberOfPoints, &h_numberOfPoints, sizeof(unsigned int));

			h_flushCount++;
			cudaMemcpyToSymbol(flushCount, &h_flushCount, sizeof(unsigned int));




			alog("[%4d] Debugging::Flush() End\n", h_frameNumber);

			{
				int enabled = 0;
				cudaMemcpyFromSymbol(&enabled, captureEnabled, sizeof(int));

				if (1 == enabled)
				{
					enabled = 0;
					cudaMemcpyToSymbol(captureEnabled, &enabled, sizeof(int));

					alog("[%4d] set captureEnabled to 0\n", h_frameNumber);

					captureOneFrame = false;
				}
			}
		}

		__device__ unsigned int lastFrameNumber = 0;

		__device__ void Debugging::AddPointP(int tag, const Eigen::Vector3f& point)
		{
			unsigned int oldFrameNumber = atomicCAS(&lastFrameNumber, lastFrameNumber, frameNumber);

			bool printLog = oldFrameNumber != frameNumber;
			lastFrameNumber = frameNumber;

			if (printLog) alog("[%4d] Debugging::AddPointP() Begin\n", frameNumber);

			if (captureEnabled != 1) return;

			int index = atomicAdd(&numberOfPoints, 1);
			if (index < POINT_COUNT) {
				points[index].tag = tag;
				points[index].x = point.x();
				points[index].y = point.y();
				points[index].z = point.z();
			}

			if (printLog) alog("[%4d] Debugging::AddPointP() end\n", frameNumber);
		}

		__device__ void Debugging::AddPointPN(int tag, const Eigen::Vector3f& point, const Eigen::Vector3f& normal)
		{
			if (captureEnabled != 1) return;

			int index = atomicAdd(&numberOfPoints, 1);
			if (index < POINT_COUNT) {
				points[index].tag = tag;
				points[index].x = point.x();
				points[index].y = point.y();
				points[index].z = point.z();
				points[index].nx = normal.x();
				points[index].ny = normal.y();
				points[index].nz = normal.z();
			}
		}

		__device__ void Debugging::AddPointPC(int tag, const Eigen::Vector3f& point, const Eigen::Vector3f& color)
		{
			if (captureEnabled != 1) return;

			int index = atomicAdd(&numberOfPoints, 1);
			if (index < POINT_COUNT) {
				points[index].tag = tag;
				points[index].x = point.x();
				points[index].y = point.y();
				points[index].z = point.z();
				points[index].r = color.x();
				points[index].g = color.y();
				points[index].b = color.z();
			}
		}

		__device__ void Debugging::AddPointPNC(int tag, const Eigen::Vector3f& point, const Eigen::Vector3f& normal, const Eigen::Vector3f& color)
		{
			if (captureEnabled != 1) return;

			int index = atomicAdd(&numberOfPoints, 1);
			if (index < POINT_COUNT) {
				points[index].tag = tag;
				points[index].x = point.x();
				points[index].y = point.y();
				points[index].z = point.z();
				points[index].nx = normal.x();
				points[index].ny = normal.y();
				points[index].nz = normal.z();
				points[index].r = color.x();
				points[index].g = color.y();
				points[index].b = color.z();
			}
		}

		void Debugging::SerializePoints(
			Eigen::Vector3f* inputPoints,
			Eigen::Vector3f* inputNormals,
			Eigen::Vector<unsigned char, 3>* inputColors, size_t numberOfInputPoints,
			const string& tagName)
		{
			Eigen::Vector3f* h_inputPoints = new Eigen::Vector3f[numberOfInputPoints];
			Eigen::Vector3f* h_inputNormals = new Eigen::Vector3f[numberOfInputPoints];
			Eigen::Vector<unsigned char, 3>* h_inputColors = new Eigen::Vector<unsigned char, 3>[numberOfInputPoints];

			cudaMemcpy(h_inputPoints, inputPoints, sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_inputNormals, inputNormals, sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_inputColors, inputColors, sizeof(Eigen::Vector<unsigned char, 3>) * numberOfInputPoints, cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();

			unsigned int h_frameCount = 0;
			cudaMemcpyFromSymbol(&h_frameCount, frameCount, sizeof(unsigned int));

			PLYFormat ply;

			for (size_t i = 0; i < numberOfInputPoints; i++)
			{
				auto& p = h_inputPoints[i];
				auto& n = h_inputNormals[i];
				auto& c = h_inputColors[i];

				if (FLT_MAX == p.x() || FLT_MAX == p.y() || FLT_MAX == p.z()) continue;

				ply.AddPointFloat3(p.data());
				ply.AddNormalFloat3(n.data());
				ply.AddColor(c.x() / 255.f, c.y() / 255.f, c.z() / 255.f);
			}

			stringstream ss;
			ss << "C:\\Resources\\Debug\\Serialized\\Serialized_" << h_frameCount << "_pointCloud_" << tagName << ".ply";
			ply.Serialize(ss.str());
		}
#pragma endregion

#pragma region Cache
		__global__ void Kernel_ClearCache(
			unsigned int* cacheData,
			unsigned int xVoxelCount,
			unsigned int yVoxelCount,
			unsigned int zVoxelCount,
			unsigned int value)
		{
			auto threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid > xVoxelCount * yVoxelCount * zVoxelCount - 1) return;

			auto index = threadid;

			cacheData[index] = value;
		}

		__global__ void Kernel_ClearCache2D(
			float* cacheData2D,
			unsigned int xVoxelCount,
			unsigned int yVoxelCount,
			float value)
		{
			auto threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid > xVoxelCount * yVoxelCount - 1) return;

			auto index = threadid;

			cacheData2D[index] = value;
		}

		__device__ unsigned int clusteringGroupIndex = 0;

		__global__ void Kernel_Clustering(
			unsigned int* cacheData,
			unsigned int xVoxelCount,
			unsigned int yVoxelCount,
			unsigned int zVoxelCount,
			unsigned int offset)
		{
			auto threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid > xVoxelCount * yVoxelCount * zVoxelCount - 1) return;

			auto index = threadid;

			auto zIndex = index / (xVoxelCount * yVoxelCount);
			auto yIndex = (index % (xVoxelCount * yVoxelCount)) / xVoxelCount;
			auto xIndex = (index % (xVoxelCount * yVoxelCount)) % xVoxelCount;

			auto cv = cacheData[index];
			if (UINT32_MAX == cv) return;

			for (int zDelta = -(int)offset; zDelta <= (int)offset; zDelta++)
			{
				if (zDelta + (int)zIndex < 0 || (int)zVoxelCount - 1 < (int)zIndex + zDelta) continue;

				for (int yDelta = -(int)offset; yDelta <= (int)offset; yDelta++)
				{
					if (yDelta + (int)yIndex < 0 || (int)yVoxelCount - 1 < (int)yIndex + yDelta) continue;

					for (int xDelta = -(int)offset; xDelta <= (int)offset; xDelta++)
					{
						if (xDelta + (int)xIndex < 0 || (int)xVoxelCount - 1 < (int)xIndex + xDelta) continue;
						if (xDelta == 0 && yDelta == 0 && zDelta == 0) continue;

						auto neighborIndex = (zIndex + zDelta) * xVoxelCount * yVoxelCount +
							(yIndex + yDelta) * xVoxelCount + xIndex + xDelta;

						auto nv = cacheData[neighborIndex];
						if (UINT32_MAX != nv)
						{
							if (Debugging::IsCaptureEnabled())
							{
								//Debugging::AddPointP(7, {
								//	(float)(xIndex + xDelta),
								//	(float)(yIndex + yDelta),
								//	(float)(zIndex + zDelta) });
							}

							return;
						}
					}
				}
			}

			if (Debugging::IsCaptureEnabled())
			{
				//Debugging::AddPointP(4, {
				//	(float)(xIndex),
				//	(float)(yIndex),
				//	(float)(zIndex) });
			}
		}

		Cache::Cache()
		{
		}

		Cache::~Cache()
		{
			Terminate();
		}

		void Cache::Initialize(
			const Eigen::Vector3f& min,
			unsigned int voxelCountX,
			unsigned int voxelCountY,
			unsigned int voxelCountZ,
			unsigned int zOffset,
			float voxelSize,
			float pointMag)
		{
			this->min = min;
			this->voxelSize = voxelSize;
			this->pointMag = pointMag;

			xVoxelCount = voxelCountX;
			yVoxelCount = voxelCountY;
			zVoxelCount = voxelCountZ;
			this->zOffset = zOffset;

			cudaMalloc(&cacheData, sizeof(unsigned int) * xVoxelCount * yVoxelCount * zVoxelCount);
			cudaMemset(cacheData, 0xffffffffui32, sizeof(unsigned int) * xVoxelCount * yVoxelCount * zVoxelCount);

			cudaMalloc(&cacheData2D, sizeof(float) * xVoxelCount * yVoxelCount);
			cudaMemset(cacheData, 0xffffffffui32, sizeof(float) * xVoxelCount * yVoxelCount);

			checkCudaErrors(cudaDeviceSynchronize());

			alog("Cache Initialized\n");
		}

		void Cache::Terminate()
		{
			if (nullptr != cacheData)
			{
				cudaFree(cacheData);
			}

			if (nullptr != cacheData2D)
			{
				cudaFree(cacheData2D);
			}

			checkCudaErrors(cudaDeviceSynchronize());

			alog("Cache Terminated\n");
		}

		void Cache::SetStream(CUstream_st* st)
		{
			stream = st;
		}

		void Cache::Clear()
		{
			nvtxRangePushA("Cache Clear");

			{
				nvtxRangePushA("Cache Clear 2D");

				auto numberOfThreads = xVoxelCount * yVoxelCount;
				int mingridsize;
				int threadblocksize;
				checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_ClearCache, 0, 0));
				int gridsize = (numberOfThreads + threadblocksize - 1) / threadblocksize;

				Kernel_ClearCache2D << <gridsize, threadblocksize, 0, stream >> > (cacheData2D, xVoxelCount, yVoxelCount, FLT_MAX);

				checkCudaErrors(cudaStreamSynchronize(stream));

				nvtxRangePop();
			}

			{
				nvtxRangePushA("Cache Clear 3D");

				auto numberOfThreads = xVoxelCount * yVoxelCount * zVoxelCount;
				int mingridsize;
				int threadblocksize;
				checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_ClearCache, 0, 0));
				int gridsize = (numberOfThreads + threadblocksize - 1) / threadblocksize;

				Kernel_ClearCache << <gridsize, threadblocksize, 0, stream >> > (cacheData, xVoxelCount, yVoxelCount, zVoxelCount, UINT32_MAX);

				checkCudaErrors(cudaStreamSynchronize(stream));

				nvtxRangePop();
			}

			nvtxRangePop();

			//cudaMemset(cacheData, 0xffffffffui32, sizeof(T) * xVoxelCount * yVoxelCount * zVoxelCount);

			//checkCudaErrors(cudaStreamSynchronize(stream));

			//nvtxRangePop();
		}

		void Cache::Clustering()
		{
			alog("Cache::Clustering() Begin\n");

			nvtxRangePushA("Cache::Clustering");

			auto numberOfThreads = xVoxelCount * yVoxelCount * zVoxelCount;
			int mingridsize;
			int threadblocksize;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_Clustering, 0, 0));
			int gridsize = (numberOfThreads + threadblocksize - 1) / threadblocksize;

			Kernel_Clustering << <gridsize, threadblocksize, 0, stream >> > (cacheData, xVoxelCount, yVoxelCount, zVoxelCount, 1);

			checkCudaErrors(cudaStreamSynchronize(stream));

			nvtxRangePop();

			alog("Cache::Clustering() End\n");
		}
#pragma endregion

	}
}

namespace CUDA
{
	void TestCache()
	{
	}
}
