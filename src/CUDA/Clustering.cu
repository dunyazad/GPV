#include "Clustering.cuh"

#include <App/Utility.h>

#include <App/Serialization.hpp>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

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

		struct ClusteringNode
		{
			int neigborCount = 0;
			ClusteringNode* neighbors[26] = { 0 };
			int tag = -1;
		};

		void TestClustering()
		{
			VD::AddLine("axes", { 0, 0, 0 }, { 100 * 0.5f, 0.0f, 0.0f }, Color4::Red);
			VD::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100 * 0.5f, 0.0f }, Color4::Green);
			VD::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100 * 0.5f }, Color4::Blue);

			auto t = Time::Now();

			PLYFormat ply;
			ply.Deserialize("C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply");

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
		}
	}
}
