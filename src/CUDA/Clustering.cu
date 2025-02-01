#include "Clustering.cuh"

#include <App/Utility.h>

#include <App/Serialization.hpp>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

#include "Cache.cuh"

namespace CUDA
{
	namespace Clustering
	{
		__device__ __host__
			size_t GetKeyFromIndex(size_t xIndex, size_t yIndex, size_t zIndex)
		{
			return xIndex << 32 | yIndex << 16 | zIndex;
		}

		__device__ __host__
			size_t GetKey(const Eigen::Vector3f& volumeMin, float voxelSize, const Eigen::Vector3f& position)
		{
			Eigen::Vector3f localPosition = position - volumeMin;
			size_t xIndex = (size_t)(unsigned int)floorf(localPosition.x() / voxelSize);
			size_t yIndex = (size_t)(unsigned int)floorf(localPosition.y() / voxelSize);
			size_t zIndex = (size_t)(unsigned int)floorf(localPosition.z() / voxelSize);
			return GetKeyFromIndex(xIndex, yIndex, zIndex);
		}

		__device__ __host__
			tuple<size_t, size_t, size_t> GetIndex(const Eigen::Vector3f& volumeMin, float voxelSize, size_t key)
		{
			size_t xIndex = key >> 32 & ((1 << 16) - 1);
			size_t yIndex = key >> 16 & ((1 << 16) - 1);
			size_t zIndex = key & ((1 << 16) - 1);

			return make_tuple(xIndex, yIndex, zIndex);
		}

		__device__ __host__
			Eigen::Vector3f GetPosition(const Eigen::Vector3f& volumeMin, float voxelSize, size_t key)
		{
			auto [xIndex, yIndex, zIndex] = GetIndex(volumeMin, voxelSize, key);
			return volumeMin + Eigen::Vector3f(xIndex * voxelSize, yIndex * voxelSize, zIndex * voxelSize);
		}

		/*struct ClusteringNode
		{
			int neigborCount = 0;
			ClusteringNode* neighbors[26] = { 0 };
			int tag = -1;
		};*/

		__global__
			void Kernel_InsertPoints(
				const Eigen::Matrix4f transform, unsigned int* cacheData, float* cacheData2D, float voxelSize,
				unsigned int cacheVoxelCountX, unsigned int cacheVoxelCountY, unsigned int cacheVoxelCountZ, unsigned int zOffset,
				Eigen::Vector3f* points, unsigned int numberOfPoints)
		{
			unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid > numberOfPoints - 1) return;

			Eigen::Vector3f gp = points[threadid];
			Eigen::Vector3f lp = (transform.inverse() * Eigen::Vector4f(gp.x(), gp.y(), gp.z(), 1.0f)).head(3);

			lp.x() += (float)cacheVoxelCountX * voxelSize * 0.5f;
			lp.y() += (float)cacheVoxelCountY * voxelSize * 0.5f;
			lp.z() += (float)zOffset * voxelSize;

			if (0 > lp.x() || 0 > lp.y() || 0 > lp.z()) return;

			auto xLocalIndex = (unsigned int)floorf(lp.x() / voxelSize);
			auto yLocalIndex = (unsigned int)floorf(lp.y() / voxelSize);
			auto zLocalIndex = (unsigned int)floorf(lp.z() / voxelSize);

			if (cacheVoxelCountX - 1 < xLocalIndex ||
				cacheVoxelCountY - 1 < yLocalIndex ||
				cacheVoxelCountZ - 1 < zLocalIndex) return;

			unsigned int index3D =
				zLocalIndex * cacheVoxelCountX * cacheVoxelCountY +
				yLocalIndex * cacheVoxelCountX + xLocalIndex;

			cacheData[index3D] = 1;
		}

		void InsertPoints(PLYFormat& ply, const Utilities::Device::Cache& filteringCache, const Eigen::Matrix4f& transform)
		{
			alog("InsertPoints Begin\n");

			unsigned int numberOfPoints = ply.GetPoints().size() / 3;
			Eigen::Vector3f* h_points = new Eigen::Vector3f[numberOfPoints];
			memcpy(h_points, ply.GetPoints().data(), sizeof(Eigen::Vector3f) * numberOfPoints);

			Eigen::Vector3f* d_points;
			cudaMalloc(&d_points, sizeof(Eigen::Vector3f) * numberOfPoints);
			cudaMemcpy(d_points, h_points, sizeof(Eigen::Vector3f) * numberOfPoints, cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();

			nvtxRangePushA("InsertPoints");

			int mingridsize;
			int threadblocksize;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_InsertPoints, 0, 0));
			int gridsize = ((uint32_t)numberOfPoints + threadblocksize - 1) / threadblocksize;

			Kernel_InsertPoints << <gridsize, threadblocksize >> > (
				transform, filteringCache.cacheData, filteringCache.cacheData2D, filteringCache.voxelSize,
				filteringCache.xVoxelCount, filteringCache.yVoxelCount, filteringCache.zVoxelCount, filteringCache.zOffset,
				d_points, numberOfPoints);

			cudaDeviceSynchronize();

			nvtxRangePop();

			delete[] h_points;

			cudaFree(d_points);

			alog("InsertPoints End\n");
		}

		__global__
			void Kernel_FloodFillUsingPatch(
				const Eigen::Matrix4f transform, unsigned int* cacheData, float* cacheData2D, float voxelSize,
				unsigned int cacheVoxelCountX, unsigned int cacheVoxelCountY, unsigned int cacheVoxelCountZ, unsigned int zOffset,
				Eigen::Vector3f* points, unsigned int numberOfPoints)
		{
			unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid > numberOfPoints - 1) return;

			Eigen::Vector3f gp = points[threadid];
			Eigen::Vector3f lp = (transform.inverse() * Eigen::Vector4f(gp.x(), gp.y(), gp.z(), 1.0f)).head(3);

			lp.x() += (float)cacheVoxelCountX * voxelSize * 0.5f;
			lp.y() += (float)cacheVoxelCountY * voxelSize * 0.5f;
			lp.z() += (float)zOffset * voxelSize;

			if (0 > lp.x() || 0 > lp.y() || 0 > lp.z()) return;

			auto xLocalIndex = (unsigned int)floorf(lp.x() / voxelSize);
			auto yLocalIndex = (unsigned int)floorf(lp.y() / voxelSize);
			auto zLocalIndex = (unsigned int)floorf(lp.z() / voxelSize);

			if (cacheVoxelCountX - 1 < xLocalIndex ||
				cacheVoxelCountY - 1 < yLocalIndex ||
				cacheVoxelCountZ - 1 < zLocalIndex) return;

			const int stackSize = 64;
			uint3 stack[stackSize];
			int stackIndex = 0;

			stack[stackIndex++] = make_uint3(xLocalIndex, yLocalIndex, zLocalIndex);

			while (0 < stackIndex && stackIndex < stackSize)
			{
				if (1 == threadid)
				{
					alog("stackIndex : %d\n", stackIndex);
				}

				uint3 top = stack[(stackIndex--) - 1];

				auto currentIndex = top.z * cacheVoxelCountX * cacheVoxelCountY + top.y * cacheVoxelCountX + top.x;

				if (2 == cacheData[currentIndex]) continue;

				cacheData[currentIndex] = 2;

				for (int zDelta = -1; zDelta <= 1; zDelta++)
				{
					if (top.z == 0 || top.z == cacheVoxelCountZ - 1)continue;

					for (int yDelta = -1; yDelta <= 1; yDelta++)
					{
						if (top.y == 0 || top.y == cacheVoxelCountY - 1)continue;

						for (int xDelta = -1; xDelta <= 1; xDelta++)
						{
							if (top.x == 0 || top.x == cacheVoxelCountX - 1)continue;
							if (top.x == 0 && top.y == 0 && top.z == 0)continue;

							unsigned int neighborIndex =
								(top.z + zDelta) * cacheVoxelCountX * cacheVoxelCountY +
								(top.y + yDelta) * cacheVoxelCountX + top.x + xDelta;

							if (1 == cacheData[neighborIndex])
							{
								if (stackIndex == stackSize)
								{
									continue;
								}
								else
								{
									stack[stackIndex++] = make_uint3(top.x + xDelta, top.y + yDelta, top.z + zDelta);
								}
							}
						}
					}
				}
			}
		}

		void FloodFillUsingPatch(PLYFormat& ply, const Utilities::Device::Cache& filteringCache, const Eigen::Matrix4f& transform)
		{
			alog("FloodFillUsingPatch Begin\n");

			unsigned int numberOfPoints = ply.GetPoints().size() / 3;
			Eigen::Vector3f* h_points = new Eigen::Vector3f[numberOfPoints];
			memcpy(h_points, ply.GetPoints().data(), sizeof(Eigen::Vector3f) * numberOfPoints);

			Eigen::Vector3f* d_points;
			cudaMalloc(&d_points, sizeof(Eigen::Vector3f) * numberOfPoints);
			cudaMemcpy(d_points, h_points, sizeof(Eigen::Vector3f) * numberOfPoints, cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();

			nvtxRangePushA("FloodFillUsingPatch");

			int mingridsize;
			int threadblocksize;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_FloodFillUsingPatch, 0, 0));
			int gridsize = ((uint32_t)numberOfPoints + threadblocksize - 1) / threadblocksize;

			Kernel_FloodFillUsingPatch << <gridsize, threadblocksize >> > (
				transform, filteringCache.cacheData, filteringCache.cacheData2D, filteringCache.voxelSize,
				filteringCache.xVoxelCount, filteringCache.yVoxelCount, filteringCache.zVoxelCount, filteringCache.zOffset,
				d_points, numberOfPoints);

			cudaDeviceSynchronize();

			nvtxRangePop();

			delete[] h_points;

			cudaFree(d_points);

			alog("FloodFillUsingPatch End\n");
		}

		void ShowCache(const Utilities::Device::Cache& filteringCache, const Eigen::Matrix4f& transform)
		{
			alog("ShowCache Begin\n");

			unsigned int numberOfPoints = filteringCache.xVoxelCount * filteringCache.yVoxelCount * filteringCache.zVoxelCount;
			unsigned int* h_cacheData = new unsigned int[numberOfPoints];

			cudaMemcpy(h_cacheData, filteringCache.cacheData, sizeof(unsigned int) * numberOfPoints, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			for (size_t i = 0; i < numberOfPoints; i++)
			{
				auto& value = h_cacheData[i];
				if (0 < value)
				{
					auto zLocalIndex = i / (filteringCache.xVoxelCount * filteringCache.yVoxelCount);
					auto yLocalIndex = (i % (filteringCache.xVoxelCount * filteringCache.yVoxelCount)) / filteringCache.xVoxelCount;
					auto xLocalIndex = (i % (filteringCache.xVoxelCount * filteringCache.yVoxelCount)) % filteringCache.xVoxelCount;

					auto x = (float)xLocalIndex * filteringCache.voxelSize - (float)filteringCache.xVoxelCount * 0.5f * filteringCache.voxelSize;
					auto y = (float)yLocalIndex * filteringCache.voxelSize - (float)filteringCache.yVoxelCount * 0.5f * filteringCache.voxelSize;
					auto z = (float)zLocalIndex * filteringCache.voxelSize - (float)filteringCache.zOffset * filteringCache.voxelSize;

					Eigen::Vector3f gp = (transform * Eigen::Vector4f(x, y, z, 1.0f)).head(3);

					if (1 == value)
					{
						VD::AddSphere("Source Points", gp, 0.05f, Color4::Magenta);
					}
					else if (2 == value)
					{
						VD::AddSphere("Alive Points", gp, 0.05f, Color4::Cyan);
					}
				}
			}

			delete[] h_cacheData;

			alog("ShowCache End\n");
		}

		Utilities::Device::Cache filteringCache;

		void TestClustering_old()
		{
			//filteringCache.Initialize({0.0f, 0.0f, 0.0f}, 64, 64, 64, 0, 0.1f, 1.0f);
			filteringCache.Initialize({ 0.0f, 0.0f, 0.0f });

			//VD::AddLine("axes", { 0, 0, 0 }, { 100 * 0.5f, 0.0f, 0.0f }, Color4::Red);
			//VD::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100 * 0.5f, 0.0f }, Color4::Green);
			//VD::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100 * 0.5f }, Color4::Blue);

			auto t = Time::Now();

			{
				FILE* fs;
				fopen_s(&fs, "C:\\Debug\\GPV\\transform.bin", "rb");
				float m[16];
				fread(m, sizeof(float) * 16, 1, fs);
				Eigen::Matrix4f transform(m);
				fclose(fs);


				{
					Eigen::Vector3f origin = (transform * Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f)).head(3);
					Eigen::Vector3f xEnd = (transform * Eigen::Vector4f(100.0f, 0.0f, 0.0f, 1.0f)).head(3);
					Eigen::Vector3f yEnd = (transform * Eigen::Vector4f(0.0f, 100.0f, 0.0f, 1.0f)).head(3);
					Eigen::Vector3f zEnd = (transform * Eigen::Vector4f(0.0f, 0.0f, 100.0f, 1.0f)).head(3);


					VD::AddLine("axes", origin, xEnd, Color4::Red);
					VD::AddLine("axes", origin, yEnd, Color4::Green);
					VD::AddLine("axes", origin, zEnd, Color4::Blue);
				}

				{
					PLYFormat ply;
					ply.Deserialize("C:\\Debug\\GPV\\Serialized_1_0.ply");

					float voxelSize = 0.1f;
					auto aabbMin = ply.GetAABBMin();
					auto aabbMax = ply.GetAABBMax();

					//auto aabbMin = Eigen::Vector3f(aabb.min().minCoeff(), aabb.min().minCoeff(), aabb.min().minCoeff());
					//auto aabbMax = Eigen::Vector3f(aabb.max().maxCoeff(), aabb.max().maxCoeff(), aabb.max().maxCoeff());

					auto volumeMinX = floorf(aabbMin.x() / voxelSize) * voxelSize;
					auto volumeMinY = floorf(aabbMin.y() / voxelSize) * voxelSize;
					auto volumeMinZ = floorf(aabbMin.z() / voxelSize) * voxelSize;
					auto volumeMaxX = ceilf(aabbMax.x() / voxelSize) * voxelSize;
					auto volumeMaxY = ceilf(aabbMax.y() / voxelSize) * voxelSize;
					auto volumeMaxZ = ceilf(aabbMax.z() / voxelSize) * voxelSize;

					auto volumeMin = Eigen::Vector3f(volumeMinX, volumeMinY, volumeMinZ);
					auto volumeMax = Eigen::Vector3f(volumeMaxX, volumeMaxY, volumeMaxZ);

					cout << "volumeMin : " << volumeMin << endl;
					cout << "volumeMax : " << volumeMax << endl;

					//VD::AddBox("aabb", aabbMin, aabbMax, Color4::Blue);
					//VD::AddBox("volume", volumeMin, volumeMax, Color4::Red);

					//VD::AddLine("aabb", aabbMin, aabbMax, Color4::Blue);
					//VD::AddLine("volume", volumeMin, volumeMax, Color4::Red);

					//vector<Eigen::Vector3f> loadedPoints;

					for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
					{
						auto x = ply.GetPoints()[i * 3];
						auto y = ply.GetPoints()[i * 3 + 1];
						auto z = ply.GetPoints()[i * 3 + 2];

						VD::AddSphere("Points", { x, y, z }, 0.05f, Color4::White);

						//loadedPoints.push_back(Eigen::Vector3f(x, y, z));
					}

					InsertPoints(ply, filteringCache, transform);
				}

				{
					PLYFormat ply;
					ply.Deserialize("C:\\Debug\\GPV\\Serialized_1_29.ply");

					for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
					{
						auto x = ply.GetPoints()[i * 3];
						auto y = ply.GetPoints()[i * 3 + 1];
						auto z = ply.GetPoints()[i * 3 + 2];

						VD::AddSphere("Points", { x, y, z }, 0.05f, Color4::Red);
					}

					FloodFillUsingPatch(ply, filteringCache, transform);
				}

				ShowCache(filteringCache, transform);
			}
		}

		void TestClustering()
		{
			unsigned int xLength = 500;
			unsigned int yLength = 500;
			unsigned int zLength = 500;
			unsigned int voxelCount = xLength * yLength * zLength;

			float* d_regularGrid;

			cudaMalloc(&d_regularGrid, sizeof(float) * voxelCount);

			cudaMemset(d_regularGrid, 0, sizeof(float) * voxelCount);

			cudaDeviceSynchronize();

			float* h_regularGrid = new float[voxelCount];

			cudaMemcpy(h_regularGrid, d_regularGrid, sizeof(float) * voxelCount, cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();

			for (unsigned int zIndex = 0; zIndex < zLength; zIndex++)
			{
				for (unsigned int yIndex = 0; yIndex < yLength; yIndex++)
				{
					for (unsigned int xIndex = 0; xIndex < xLength; xIndex++)
					{
						auto index = zIndex * xLength * yLength + yIndex * xLength + xIndex;
						if (0.0f != h_regularGrid[index])
						{
							alog("[1] WTF!!!!\n");
						}
					}
				}
			}

			float fmax = FLT_MAX;
			unsigned int temp;
			memcpy(&temp, &fmax, sizeof(float));

			cudaMemset(d_regularGrid, -1, sizeof(float) * voxelCount);

			cudaDeviceSynchronize();

			cudaMemcpy(h_regularGrid, d_regularGrid, sizeof(float) * voxelCount, cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();

			for (unsigned int zIndex = 0; zIndex < zLength; zIndex++)
			{
				for (unsigned int yIndex = 0; yIndex < yLength; yIndex++)
				{
					for (unsigned int xIndex = 0; xIndex < xLength; xIndex++)
					{
						auto index = zIndex * xLength * yLength + yIndex * xLength + xIndex;
						if (false == isnan(h_regularGrid[index]))
						{
							printf("%f\n", h_regularGrid[index]);
							alog("[2] WTF!!!!\n");
						}
					}
				}
			}

			delete[] h_regularGrid;

			cudaFree(d_regularGrid);

			alog("Done\n");
		}

		//void TestClustering()
		//{
		//	float fmax = FLT_MAX;
		//	unsigned int temp;
		//	memcpy(&temp, &fmax, sizeof(float));
		//	
		//	temp = -1;
		//	cout << "fmax : " << bitset<16>(temp) << endl;
		//}
	}
}
