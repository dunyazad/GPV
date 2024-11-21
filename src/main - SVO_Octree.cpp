#include <Common.h>
#include <App/App.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <Algorithm/Octree.hpp>

#include <Algorithm/CustomPolyDataFilter.h>
#include <Algorithm/vtkMedianFilter.h>
#include <Algorithm/vtkQuantizingFilter.h>

#include <Debugging/VisualDebugging.h>

#include <CUDA/HashTable.cuh>
#include <CUDA/KDTree.cuh>
#include <CUDA/KDITree.cuh>
#include <CUDA/SVO.cuh>
#include <CUDA/AOctree.cuh>

#include <CUDA/Processing.cuh>

#include <CUDA/CUDA.cuh>

int depthIndex = 0;

namespace SVO
{
	struct Triangle
	{
		Eigen::Vector3f vertices[3];
	};

	struct NearestNeighborResult {
		Eigen::Vector3f nearestPoint;
		float distanceSquared = std::numeric_limits<float>::max();
	};

	const int edgeConnection[12][2] = {
		{0, 1}, {1, 3}, {3, 2}, {2, 0},
		{4, 5}, {5, 7}, {7, 6}, {6, 4},
		{0, 4}, {1, 5}, {2, 6}, {3, 7}
	};

	std::vector<Eigen::Vector3f> GetVoxelCorners(const Eigen::Vector3f& center, float halfSize)
	{
		return {
			center + Eigen::Vector3f(-halfSize, -halfSize, -halfSize),
			center + Eigen::Vector3f(halfSize, -halfSize, -halfSize),
			center + Eigen::Vector3f(-halfSize,  halfSize, -halfSize),
			center + Eigen::Vector3f(halfSize,  halfSize, -halfSize),
			center + Eigen::Vector3f(-halfSize, -halfSize,  halfSize),
			center + Eigen::Vector3f(halfSize, -halfSize,  halfSize),
			center + Eigen::Vector3f(-halfSize,  halfSize,  halfSize),
			center + Eigen::Vector3f(halfSize,  halfSize,  halfSize)
		};
	}

	struct SVONode {
		bool isLeaf = false;
		size_t parentIndex = Max.U64;
		size_t childIndex[8] = { Max.U64, Max.U64, Max.U64, Max.U64, Max.U64, Max.U64, Max.U64, Max.U64 };
		size_t voxelIndex = Max.U64;

		Eigen::Vector3f center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		float size = 0.0f;

		float tsdfValue = Max.FLOAT;
		float weight = 0.0f;
		bool occupied = false;
		int updatedCount = 0;

		// 각 코너의 SDF 값과 초기화 여부
		std::array<float, 8> cornerSDFs = { Max.FLOAT, Max.FLOAT, Max.FLOAT, Max.FLOAT, Max.FLOAT, Max.FLOAT, Max.FLOAT, Max.FLOAT };
	};

	void InitializeNode(SVONode* node, const Eigen::Vector3f& center, float size)
	{
		node->isLeaf = false;
		node->parentIndex = Max.U64;
		node->voxelIndex = Max.U64;
		node->childIndex[0] = Max.U64;
		node->childIndex[1] = Max.U64;
		node->childIndex[2] = Max.U64;
		node->childIndex[3] = Max.U64;
		node->childIndex[4] = Max.U64;
		node->childIndex[5] = Max.U64;
		node->childIndex[6] = Max.U64;
		node->childIndex[7] = Max.U64;
		node->center = center;
		node->size = size;
		node->tsdfValue = Max.FLOAT;
		node->occupied = false;
		node->weight = 0.0f;
		node->updatedCount = 0;
	}

	struct Octree
	{
		SVONode* nodes = nullptr;
		size_t numberOfNodes = 0;

		size_t nodeBufferIndex = 0;
		size_t rootIndex = Max.U64;
		size_t maxDepth = 0;
	};

	int GetChildIndex(const Eigen::Vector3f& point, const Eigen::Vector3f& nodeCenter);
	void UpdateNodeWithSDF(SVONode& node, const Eigen::Vector3f& point);
	void UpdateNodeWithTSDF(SVONode& node, const Eigen::Vector3f& point, float sdfValue, float weight);
	void UpdateNeighboringNodesWithTSDF(Octree* octree, size_t nodeIndex, const Eigen::Vector3f& point, float truncationDistance, float weight, int maxDepth, int currentDepth);
	float ComputeSDFValue(const SVONode& node, const Eigen::Vector3f& point);
	float ComputeTSDFValue(const SVONode& node, const Eigen::Vector3f& point, float truncationDistance);
	void IntegratePointCloudWithTSDF(Octree* octree, const Eigen::Vector3f& center, const Eigen::Matrix4f& transform, const Eigen::Vector3f* pointCloud, size_t numberOfPoints, float rootVoxelSize, int maxDepth, float truncationDistance);

	void InitializeSVO(Octree* octree, size_t numberOfNodes)
	{
		octree->nodes = new SVONode[numberOfNodes];
		octree->numberOfNodes = numberOfNodes;
		octree->nodeBufferIndex = 0;
		octree->rootIndex = Max.U64;
		octree->maxDepth = 0;
	}

	void IntegratePointCloud(Octree* octree, const Eigen::Vector3f& center, const Eigen::Matrix4f& transform, const Eigen::Vector3f* pointCloud, size_t numberOfPoints, float rootVoxelSize, int maxDepth)
	{
		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			const Eigen::Vector3f& patchPoint = pointCloud[i];
			Eigen::Vector4f tpp = transform * Eigen::Vector4f(patchPoint.x(), patchPoint.y(), patchPoint.z(), 1.0f);
			Eigen::Vector3f point(tpp.x(), tpp.y(), tpp.z());

			size_t currentIndex = octree->rootIndex;

			if (currentIndex == Max.U64)
			{
				octree->rootIndex = 0;
				InitializeNode(&octree->nodes[octree->rootIndex], center, rootVoxelSize);
				currentIndex = octree->rootIndex;
			}

			int currentDepth = 0;

			while (!octree->nodes[currentIndex].isLeaf)
			{
				if (currentDepth >= maxDepth)
				{
					octree->nodes[currentIndex].isLeaf = true;
					break;
				}

				Eigen::Vector3f nodeCenter = octree->nodes[currentIndex].center;
				float currentVoxelSize = octree->nodes[currentIndex].size;

				int childIdx = GetChildIndex(point, nodeCenter);

				if (octree->nodes[currentIndex].childIndex[childIdx] == Max.U64)
				{
					octree->nodes[currentIndex].childIndex[childIdx] = ++octree->nodeBufferIndex;

					float halfSize = currentVoxelSize / 2.0f;
					Eigen::Vector3f childCenter = nodeCenter;

					childCenter.x() += (childIdx & 1) ? halfSize / 2.0f : -halfSize / 2.0f;
					childCenter.y() += (childIdx & 2) ? halfSize / 2.0f : -halfSize / 2.0f;
					childCenter.z() += (childIdx & 4) ? halfSize / 2.0f : -halfSize / 2.0f;

					//printf("octree->nodeBufferIndex : %llu, halfSize : %f\n", octree->nodeBufferIndex, halfSize);

					InitializeNode(&octree->nodes[octree->nodeBufferIndex], childCenter, halfSize);
				}

				currentIndex = octree->nodes[currentIndex].childIndex[childIdx];
				currentDepth++;

				if (currentDepth > octree->maxDepth)
				{
					octree->maxDepth = currentDepth;
					//printf("octree->maxDepth : %llu\n", octree->maxDepth);
				}
			}

			UpdateNodeWithSDF(octree->nodes[currentIndex], point);
		}
	}

	void IntegratePointCloudWithTSDF(Octree* octree, const Eigen::Vector3f& center, const Eigen::Matrix4f& transform, const Eigen::Vector3f* pointCloud, size_t numberOfPoints, float rootVoxelSize, int maxDepth, float truncationDistance)
	{
		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			const Eigen::Vector3f& patchPoint = pointCloud[i];
			Eigen::Vector4f tpp = transform * Eigen::Vector4f(patchPoint.x(), patchPoint.y(), patchPoint.z(), 1.0f);
			Eigen::Vector3f point(tpp.x(), tpp.y(), tpp.z());

			size_t currentIndex = octree->rootIndex;

			if (currentIndex == Max.U64)
			{
				octree->rootIndex = 0;
				InitializeNode(&octree->nodes[octree->rootIndex], center, rootVoxelSize);
				currentIndex = octree->rootIndex;
			}

			int currentDepth = 0;

			while (!octree->nodes[currentIndex].isLeaf)
			{
				if (currentDepth >= maxDepth)
				{
					octree->nodes[currentIndex].isLeaf = true;
					break;
				}

				Eigen::Vector3f nodeCenter = octree->nodes[currentIndex].center;
				float currentVoxelSize = octree->nodes[currentIndex].size;

				int childIdx = GetChildIndex(point, nodeCenter);

				if (octree->nodes[currentIndex].childIndex[childIdx] == Max.U64)
				{
					octree->nodes[currentIndex].childIndex[childIdx] = ++octree->nodeBufferIndex;

					float halfSize = currentVoxelSize / 2.0f;
					Eigen::Vector3f childCenter = nodeCenter;

					childCenter.x() += (childIdx & 1) ? halfSize / 2.0f : -halfSize / 2.0f;
					childCenter.y() += (childIdx & 2) ? halfSize / 2.0f : -halfSize / 2.0f;
					childCenter.z() += (childIdx & 4) ? halfSize / 2.0f : -halfSize / 2.0f;

					InitializeNode(&octree->nodes[octree->nodeBufferIndex], childCenter, halfSize);
				}

				currentIndex = octree->nodes[currentIndex].childIndex[childIdx];
				currentDepth++;
			}

			float sdfValue = ComputeTSDFValue(octree->nodes[currentIndex], point, truncationDistance);
			float weight = 1.0f;
			UpdateNodeWithTSDF(octree->nodes[currentIndex], point, sdfValue, weight);
		}
	}

	void IntegratePointCloudWithNeighborUpdate(Octree* octree, const Eigen::Vector3f& center, const Eigen::Matrix4f& transform, const Eigen::Vector3f* pointCloud, size_t numberOfPoints, float rootVoxelSize, int maxDepth, float truncationDistance)
	{
		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			const Eigen::Vector3f& patchPoint = pointCloud[i];
			Eigen::Vector4f tpp = transform * Eigen::Vector4f(patchPoint.x(), patchPoint.y(), patchPoint.z(), 1.0f);
			Eigen::Vector3f point(tpp.x(), tpp.y(), tpp.z());

			size_t currentIndex = octree->rootIndex;

			if (currentIndex == Max.U64)
			{
				octree->rootIndex = 0;
				InitializeNode(&octree->nodes[octree->rootIndex], center, rootVoxelSize);
				currentIndex = octree->rootIndex;
			}

			int currentDepth = 0;

			while (!octree->nodes[currentIndex].isLeaf)
			{
				if (currentDepth >= maxDepth)
				{
					octree->nodes[currentIndex].isLeaf = true;
					break;
				}

				Eigen::Vector3f nodeCenter = octree->nodes[currentIndex].center;
				float currentVoxelSize = octree->nodes[currentIndex].size;

				int childIdx = GetChildIndex(point, nodeCenter);

				if (octree->nodes[currentIndex].childIndex[childIdx] == Max.U64)
				{
					octree->nodes[currentIndex].childIndex[childIdx] = ++octree->nodeBufferIndex;

					float halfSize = currentVoxelSize / 2.0f;
					Eigen::Vector3f childCenter = nodeCenter;

					childCenter.x() += (childIdx & 1) ? halfSize / 2.0f : -halfSize / 2.0f;
					childCenter.y() += (childIdx & 2) ? halfSize / 2.0f : -halfSize / 2.0f;
					childCenter.z() += (childIdx & 4) ? halfSize / 2.0f : -halfSize / 2.0f;

					InitializeNode(&octree->nodes[octree->nodeBufferIndex], childCenter, halfSize);
				}

				currentIndex = octree->nodes[currentIndex].childIndex[childIdx];
				currentDepth++;
			}

			float sdfValue = ComputeTSDFValue(octree->nodes[currentIndex], point, truncationDistance);
			float weight = 1.0f;
			UpdateNodeWithTSDF(octree->nodes[currentIndex], point, sdfValue, weight);

			UpdateNeighboringNodesWithTSDF(octree, currentIndex, point, truncationDistance, weight, maxDepth, currentDepth);
		}
	}

	int GetChildIndex(const Eigen::Vector3f& point, const Eigen::Vector3f& nodeCenter)
	{
		int childIndex = 0;

		if (point.x() >= nodeCenter.x()) childIndex |= 1;
		if (point.y() >= nodeCenter.y()) childIndex |= 2;
		if (point.z() >= nodeCenter.z()) childIndex |= 4;

		return childIndex;
	}

	void UpdateNodeWithSDF(SVONode& node, const Eigen::Vector3f& point)
	{
		float tsdfValue = ComputeSDFValue(node, point);
		//if (tsdfValue != 0.0f)
		//{
		//	printf("tsdfValue : %f\n", tsdfValue);
		//}

		if (node.updatedCount > 0) {
			node.tsdfValue = (node.tsdfValue * node.updatedCount + tsdfValue) / (node.updatedCount + 1);
		}
		else {
			node.tsdfValue = tsdfValue;
		}
		node.updatedCount++;

		node.occupied = (node.tsdfValue < 0.0f);

		//if (node.occupied)
		//{
		//	printf("OK\n");
		//}
	}

	void UpdateNodeWithTSDF(SVONode& node, const Eigen::Vector3f& point, float sdfValue, float weight) {
		// TSDF 업데이트
		float totalWeight = node.weight + weight;
		node.tsdfValue = (node.tsdfValue * node.weight + sdfValue * weight) / totalWeight;
		node.weight = totalWeight;
		node.occupied = (node.tsdfValue < 0.0f && std::abs(node.tsdfValue) < node.size);

		// 각 코너의 SDF 값을 업데이트
		float halfSize = node.size / 2.0f;
		std::vector<Eigen::Vector3f> corners = {
			node.center + Eigen::Vector3f(-halfSize, -halfSize, -halfSize),
			node.center + Eigen::Vector3f(halfSize, -halfSize, -halfSize),
			node.center + Eigen::Vector3f(-halfSize, halfSize, -halfSize),
			node.center + Eigen::Vector3f(halfSize, halfSize, -halfSize),
			node.center + Eigen::Vector3f(-halfSize, -halfSize, halfSize),
			node.center + Eigen::Vector3f(halfSize, -halfSize, halfSize),
			node.center + Eigen::Vector3f(-halfSize, halfSize, halfSize),
			node.center + Eigen::Vector3f(halfSize, halfSize, halfSize)
		};

		// 각 코너에 대해 주어진 포인트로부터의 거리를 바탕으로 SDF 값을 업데이트
		for (int i = 0; i < 8; ++i) {
			float distance = (corners[i] - point).norm();
			float cornerSDF = distance - halfSize;

			if (node.cornerSDFs[i] == Max.FLOAT) {
				// 코너가 초기화되지 않은 경우 초기값 설정
				node.cornerSDFs[i] = cornerSDF;
			}
			else {
				// 가중치를 반영하여 기존 SDF 값과 보간하여 업데이트
				node.cornerSDFs[i] = (node.cornerSDFs[i] * node.weight + cornerSDF * weight) / totalWeight;
			}
		}
	}

	void EnsureNeighboringNodesExist(Octree* octree, size_t nodeIndex, const Eigen::Vector3f& point, float truncationDistance, int maxDepth, int currentDepth)
	{
		if (nodeIndex == Max.U64 || currentDepth > maxDepth)
		{
			return;
		}

		std::queue<std::pair<size_t, int>> nodesToVisit;
		nodesToVisit.push({ nodeIndex, currentDepth });

		while (!nodesToVisit.empty())
		{
			auto [currentIndex, currentDepth] = nodesToVisit.front();
			nodesToVisit.pop();

			SVONode& currentNode = octree->nodes[currentIndex];

			float halfSize = currentNode.size / 2.0f;

			for (int i = 0; i < 8; ++i)
			{
				size_t& childIdx = currentNode.childIndex[i];
				Eigen::Vector3f childCenter = currentNode.center;

				childCenter.x() += (i & 1) ? halfSize / 2.0f : -halfSize / 2.0f;
				childCenter.y() += (i & 2) ? halfSize / 2.0f : -halfSize / 2.0f;
				childCenter.z() += (i & 4) ? halfSize / 2.0f : -halfSize / 2.0f;

				float distanceToChild = (point - childCenter).norm();

				if (distanceToChild < truncationDistance)
				{
					if (childIdx == Max.U64)
					{
						if (currentDepth < maxDepth)
						{
							childIdx = ++octree->nodeBufferIndex;
							InitializeNode(&octree->nodes[childIdx], childCenter, halfSize);
						}
						else if (currentDepth == maxDepth)
						{
							childIdx = ++octree->nodeBufferIndex;
							InitializeNode(&octree->nodes[childIdx], childCenter, halfSize);
						}
					}

					if (childIdx != Max.U64 && currentDepth < maxDepth)
					{
						nodesToVisit.push({ childIdx, currentDepth + 1 });
					}
				}
			}
		}
	}

	void UpdateNeighboringNodesWithTSDF(Octree* octree, size_t nodeIndex, const Eigen::Vector3f& point, float truncationDistance, float weight, int maxDepth, int currentDepth)
	{
		EnsureNeighboringNodesExist(octree, nodeIndex, point, truncationDistance, maxDepth, currentDepth);

		std::queue<size_t> nodesToVisit;
		nodesToVisit.push(nodeIndex);

		while (!nodesToVisit.empty())
		{
			size_t currentIndex = nodesToVisit.front();
			nodesToVisit.pop();

			SVONode& currentNode = octree->nodes[currentIndex];

			float sdfValue = ComputeTSDFValue(currentNode, point, truncationDistance);
			UpdateNodeWithTSDF(currentNode, point, sdfValue, weight);

			if (!currentNode.isLeaf)
			{
				for (int i = 0; i < 8; ++i)
				{
					size_t& childIdx = currentNode.childIndex[i];
					if (childIdx != Max.U64 && (currentDepth + 1) <= maxDepth)
					{
						nodesToVisit.push(childIdx);
					}
				}
			}
		}
	}

	float ComputeSDFValue(const SVONode& node, const Eigen::Vector3f& point) {
		Eigen::Vector3f toPoint = point - node.center;
		float halfSize = node.size / 2.0f;

		// 각 축에서 거리 차이를 계산
		float dx = std::max(0.0f, std::abs(toPoint.x()) - halfSize);
		float dy = std::max(0.0f, std::abs(toPoint.y()) - halfSize);
		float dz = std::max(0.0f, std::abs(toPoint.z()) - halfSize);

		float sdfValue = std::sqrt(dx * dx + dy * dy + dz * dz);

		// 포인트가 박스 내부에 있는 경우 음수로 설정
		if (std::abs(toPoint.x()) < halfSize &&
			std::abs(toPoint.y()) < halfSize &&
			std::abs(toPoint.z()) < halfSize) {
			sdfValue = -sdfValue;
		}

		return sdfValue;
	}

	float ComputeTSDFValue(const SVONode& node, const Eigen::Vector3f& point, float truncationDistance)
	{
		Eigen::Vector3f toPoint = point - node.center;
		float distance = toPoint.norm();

		float halfSize = node.size / 2.0f;
		float sdfValue = distance - halfSize;

		return std::min(std::max(sdfValue, -truncationDistance), truncationDistance);
	}

	void TraverseOctree(Octree* octree, size_t nodeIndex, int currentDepth, const std::function<void(const SVONode&, int)>& operation)
	{
		if (nodeIndex == Max.U64)
		{
			return;
		}

		const SVONode& currentNode = octree->nodes[nodeIndex];

		operation(currentNode, currentDepth);

		if (currentNode.isLeaf)
		{
			return;
		}

		for (int i = 0; i < 8; ++i)
		{
			if (currentNode.childIndex[i] != Max.U64)
			{
				TraverseOctree(octree, currentNode.childIndex[i], currentDepth + 1, operation);
			}
		}
	}

	void TraverseOctree(Octree* octree, const std::function<void(const SVONode&, int)>& operation)
	{
		if (octree->rootIndex != Max.U64)
		{
			TraverseOctree(octree, octree->rootIndex, 0, operation);
		}
		else
		{
			std::cout << "The octree is empty." << std::endl;
		}
	}

	void NearestNeighborDFS(Octree* octree, size_t nodeIndex, const Eigen::Vector3f& queryPoint, NearestNeighborResult& result) {
		if (nodeIndex == Max.U64) {
			return;
		}

		const SVONode& currentNode = octree->nodes[nodeIndex];

		float distSquaredToCenter = (queryPoint - currentNode.center).squaredNorm();

		if (distSquaredToCenter > result.distanceSquared) {
			return;
		}

		if (currentNode.isLeaf && currentNode.occupied) {
			if (distSquaredToCenter < result.distanceSquared) {
				result.distanceSquared = distSquaredToCenter;
				result.nearestPoint = currentNode.center;
			}
			return;
		}

		std::vector<std::pair<size_t, float>> childrenDistances;
		for (int i = 0; i < 8; ++i) {
			if (currentNode.childIndex[i] != Max.U64) {
				const SVONode& childNode = octree->nodes[currentNode.childIndex[i]];
				float childDistSquared = (queryPoint - childNode.center).squaredNorm();
				childrenDistances.emplace_back(currentNode.childIndex[i], childDistSquared);
			}
		}

		std::sort(childrenDistances.begin(), childrenDistances.end(),
			[](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
				return a.second < b.second;
			});

		for (const auto& child : childrenDistances) {
			NearestNeighborDFS(octree, child.first, queryPoint, result);
		}
	}

	void GetCornerSDFValues(const SVONode& node, std::array<float, 8>& cornerSDFs) {
		float halfSize = node.size / 2.0f;
		const Eigen::Vector3f center = node.center;

		std::vector<Eigen::Vector3f> corners = {
			center + Eigen::Vector3f(-halfSize, -halfSize, -halfSize),
			center + Eigen::Vector3f(halfSize, -halfSize, -halfSize),
			center + Eigen::Vector3f(-halfSize,  halfSize, -halfSize),
			center + Eigen::Vector3f(halfSize,  halfSize, -halfSize),
			center + Eigen::Vector3f(-halfSize, -halfSize,  halfSize),
			center + Eigen::Vector3f(halfSize, -halfSize,  halfSize),
			center + Eigen::Vector3f(-halfSize,  halfSize,  halfSize),
			center + Eigen::Vector3f(halfSize,  halfSize,  halfSize)
		};

		for (int i = 0; i < 8; ++i) {
			Eigen::Vector3f toCorner = corners[i] - center;
			float distance = toCorner.norm();
			cornerSDFs[i] = node.tsdfValue - distance;
			//std::cout << "Corner [" << i << "] Position: " << corners[i].transpose()
			//	<< ", SDF Value: " << cornerSDFs[i] << std::endl;
		}
	}

	const int MARCHING_CUBES_TABLE[256] = {
	0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xb3c, 0xa35, 0xd3f, 0xc36, 0xf3a, 0xe33, 0x939, 0x830,
	0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
	0x830, 0x939, 0xa33, 0xb3a, 0xc36, 0xd3f, 0xe35, 0xf3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
	};

	const int TRIANGLE_TABLE[256][16] =
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

	void MarchingCubes(const SVONode& node, std::vector<Triangle>& triangles) {
		const std::array<float, 8>& cornerSDFs = node.cornerSDFs;

		// 각 코너의 SDF 값을 확인하고, cubeIndex를 설정
		int cubeIndex = 0;
		for (int i = 0; i < 8; ++i) {
			if (cornerSDFs[i] < 0.0f) {
				cubeIndex |= (1 << i);
			}
		}

		// 삼각형이 없는 경우
		if (MARCHING_CUBES_TABLE[cubeIndex] == 0) {
			return;
		}

		float halfSize = node.size / 2.0f;
		const auto corners = GetVoxelCorners(node.center, halfSize);

		Eigen::Vector3f vertList[12];

		// 각 에지에 대해 보간하여 삼각형의 버텍스를 계산
		for (int i = 0; i < 12; ++i) {
			if (MARCHING_CUBES_TABLE[cubeIndex] & (1 << i)) {
				int idx1 = edgeConnection[i][0];
				int idx2 = edgeConnection[i][1];

				const Eigen::Vector3f& p1 = corners[idx1];
				const Eigen::Vector3f& p2 = corners[idx2];
				float val1 = cornerSDFs[idx1];
				float val2 = cornerSDFs[idx2];

				// 보간을 통해 에지의 중간 점을 계산
				float t = val1 / (val1 - val2);
				vertList[i] = p1 + t * (p2 - p1);
			}
		}

		// 삼각형 생성
		for (int i = 0; TRIANGLE_TABLE[cubeIndex][i] != -1; i += 3) {
			Triangle tri;
			tri.vertices[0] = vertList[TRIANGLE_TABLE[cubeIndex][i]];
			tri.vertices[1] = vertList[TRIANGLE_TABLE[cubeIndex][i + 1]];
			tri.vertices[2] = vertList[TRIANGLE_TABLE[cubeIndex][i + 2]];
			triangles.push_back(tri);

			//// 디버깅용 삼각형 시각화
			//std::cout << "Generating triangle with vertices: "
			//	<< tri.vertices[0].transpose() << ", "
			//	<< tri.vertices[1].transpose() << ", "
			//	<< tri.vertices[2].transpose() << std::endl;

			VisualDebugging::AddTriangle("Triangles", tri.vertices[0], tri.vertices[1], tri.vertices[2], Color4::White);
		}
	}

	void ExtractTrianglesFromOctree(Octree* octree, std::vector<Triangle>& triangles)
	{
		TraverseOctree(octree, [&](const SVONode& node, int depth)
			{
				if (node.isLeaf && node.occupied)
				{
					MarchingCubes(node, triangles);
				}
			});
	}
}

int pid = 0;
size_t size_0 = 0;
size_t size_45 = 0;
float transform_0[16];
float transform_45[16];
unsigned char image_0[400 * 480];
unsigned char image_45[400 * 480];
Eigen::Vector3f points_0[400 * 480];
Eigen::Vector3f points_45[400 * 480];
Eigen::AlignedBox3f aabb_0(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f aabb_45(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f gaabb_0(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f gaabb_45(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f taabb(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
Eigen::AlignedBox3f lmax(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));

vector<Eigen::Vector3f> patchPoints_0;
vector<Eigen::Vector3f> patchPoints_45;
vector<Eigen::Vector3f> inputPoints;

void LoadPatch(int patchID, vtkRenderer* renderer)
{
	stringstream ss;
	ss << "C:\\Debug\\Patches\\patch_" << patchID << ".pat";

	ifstream ifs;
	ifs.open(ss.str(), ios::in | ios::binary);

	ifs.read((char*)&pid, sizeof(int));
	ifs.read((char*)&size_0, sizeof(size_t));
	ifs.read((char*)&size_45, sizeof(size_t));
	ifs.read((char*)&transform_0, sizeof(float) * 16);
	ifs.read((char*)&transform_45, sizeof(float) * 16);
	ifs.read((char*)&image_0, sizeof(unsigned char) * 400 * 480);
	ifs.read((char*)&image_45, sizeof(unsigned char) * 400 * 480);
	ifs.read((char*)&points_0, sizeof(Eigen::Vector3f) * size_0);
	ifs.read((char*)&points_45, sizeof(Eigen::Vector3f) * size_45);

	ifs.close();

	vtkNew<vtkPoints> points;
	//points->SetNumberOfPoints(size_0);

	Eigen::Matrix4f t0(transform_0);
	Eigen::Matrix4f t45(transform_45);

	aabb_0 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
	aabb_45 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
	gaabb_0 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
	gaabb_45 = Eigen::AlignedBox3f(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));

	int count = 0;

	patchPoints_0.clear();
	patchPoints_45.clear();

	for (size_t i = 0; i < size_0; i++)
	{
		auto& p = points_0[i];

		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
		{
			patchPoints_0.push_back(p);
			
			Eigen::Vector4f p4(p.x(), p.y(), p.z(), 1.0f);
			Eigen::Vector4f tp = t0 * p4;
			Eigen::Vector3f tp3 = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

			points->InsertNextPoint(tp3.data());

			aabb_0.extend(p);
			gaabb_0.extend(tp3);
			taabb.extend(tp3);

			lmax.extend(aabb_0.max() - aabb_0.min());

			inputPoints.push_back(tp3);

			count++;
		}
	}

	for (size_t i = 0; i < size_45; i++)
	{
		auto& p = points_45[i];

		if ((p.x() > -1000 && p.y() > -1000 && p.z() > -1000) &&
			(p.x() < 1000 && p.y() < 1000 && p.z() < 1000))
		{
			patchPoints_45.push_back(p);
	
			Eigen::Vector4f p4(p.x(), p.y(), p.z(), 1.0f);
			Eigen::Vector4f tp = t45 * p4;
			Eigen::Vector3f tp3 = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

			points->InsertNextPoint(tp3.data());

			aabb_45.extend(p);
			gaabb_45.extend(tp3);
			taabb.extend(tp3);

			lmax.extend(aabb_45.max() - aabb_45.min());
	
			inputPoints.push_back(tp3);

			count++;
		}
	}

	std::cout << aabb_0.min().transpose() << std::endl;
	std::cout << aabb_0.max().transpose() << std::endl;
	std::cout << gaabb_0.min().transpose() << std::endl;
	std::cout << gaabb_0.max().transpose() << std::endl;
	std::cout << aabb_45.min().transpose() << std::endl;
	std::cout << aabb_45.max().transpose() << std::endl;
	std::cout << gaabb_45.min().transpose() << std::endl;
	std::cout << gaabb_45.max().transpose() << std::endl;

	return;

	//VisualDebugging::AddCube("aabb", (aabb.min() + aabb.max()) * 0.5f, aabb.max() - aabb.min(), { 0.0f, 0.0f, 0.0f }, Color4::Red);

	vtkNew<vtkPolyData> polyData;
	polyData->SetPoints(points);

	//WritePLY(polyData, "C:\\Debug\\GPV\\Original.ply");

	//double spatialSigma = 0.5;  // adjust this based on the point cloud scale
	//double featureSigma = 0.1;  // adjust based on feature variance
	//double neighborhoodSize = 0.5;  // adjust based on the density of the point cloud

	//// Apply bilateral filter
	//vtkSmartPointer<vtkPoints> newPoints = BilateralFilter(polyData, spatialSigma, featureSigma, neighborhoodSize);

	//vtkNew<vtkPolyData> newPolyData;
	//newPolyData->SetPoints(newPoints);

	//WritePLY(newPolyData, "C:\\Debug\\GPV\\Filtered.ply");

	vtkNew<vtkVertexGlyphFilter> vertexFilter;
	vertexFilter->SetInputData(polyData);
	vertexFilter->Update();

	vtkNew<vtkPolyDataMapper> mapper;
	mapper->SetInputData(vertexFilter->GetOutput());

	vtkNew<vtkActor> actor;
	actor->SetObjectName("points");
	actor->SetMapper(mapper);

	actor->GetProperty()->SetPointSize(5.0f);
	actor->GetProperty()->SetColor(1.0, 1.0, 1.0);

	renderer->AddActor(actor);
}

void OnKeyPress(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData) {
	vtkRenderWindowInteractor* interactor = static_cast<vtkRenderWindowInteractor*>(caller);
	vtkRenderWindow* renderWindow = interactor->GetRenderWindow();
	vtkRenderer* renderer = renderWindow->GetRenderers()->GetFirstRenderer();

	std::string key = interactor->GetKeySym();

	printf("%s\n", key.c_str());

	if (key == "r")
	{
		std::cout << "Key 'r' was pressed. Resetting camera." << std::endl;
		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);

		vtkCamera* camera = renderer->GetActiveCamera();
		renderer->ResetCamera();

		// Reset the camera position, focal point, and view up vector to their defaults
		camera->SetPosition(0, 0, 1);          // Reset position to default
		camera->SetFocalPoint(0, 0, 0);        // Reset focal point to origin
		camera->SetViewUp(0, 1, 0);            // Reset the up vector to default (Y-axis up)

		interactor->Render();
	}
	else if (key == "c")
	{
		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);

		vtkCamera* camera = renderer->GetActiveCamera();
		camera->SetParallelProjection(!camera->GetParallelProjection());

		interactor->Render();
	}
	else if (key == "Escape")
	{
		std::cout << "Key 'Escape' was pressed. Exiting." << std::endl;
		interactor->TerminateApp();
	}
	else if (key == "minus")
	{
		VisualDebugging::SetLineWidth("Spheres", 1);
		vtkSmartPointer<vtkActor> actor = VisualDebugging::GetSphereActor("Spheres");
		vtkSmartPointer<vtkMapper> mapper = actor->GetMapper();
		vtkSmartPointer<vtkPolyDataMapper> polyDataMapper =
			vtkPolyDataMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkGlyph3DMapper> glyph3DMapper = vtkGlyph3DMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::SafeDownCast(glyph3DMapper->GetInputDataObject(0, 0));
		vtkSmartPointer<vtkPointData> pointData = polyData->GetPointData();
		vtkSmartPointer<vtkDoubleArray> scaleArray =
			vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
		for (vtkIdType i = 0; i < scaleArray->GetNumberOfTuples(); ++i)
		{
			double scale[3]; // Assuming 3-component scale array (X, Y, Z)
			scaleArray->GetTuple(i, scale);
			//std::cout << "Scale for point " << i << ": "
			//	<< scale[0 ] << ", " << scale[1] << ", " << scale[2] << std::endl;
			scale[0] *= 0.9;
			scale[1] *= 0.9;
			scale[2] *= 0.9;
			scaleArray->SetTuple(i, scale);
		}
		polyData->Modified();
		glyph3DMapper->SetScaleArray("Scales");
		glyph3DMapper->Update();
	}
	else if (key == "equal")
	{
		VisualDebugging::SetLineWidth("Spheres", 1);
		vtkSmartPointer<vtkActor> actor = VisualDebugging::GetSphereActor("Spheres");
		vtkSmartPointer<vtkMapper> mapper = actor->GetMapper();
		vtkSmartPointer<vtkPolyDataMapper> polyDataMapper =
			vtkPolyDataMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkGlyph3DMapper> glyph3DMapper = vtkGlyph3DMapper::SafeDownCast(mapper);
		vtkSmartPointer<vtkPolyData> polyData = vtkPolyData::SafeDownCast(glyph3DMapper->GetInputDataObject(0, 0));
		vtkSmartPointer<vtkPointData> pointData = polyData->GetPointData();
		vtkSmartPointer<vtkDoubleArray> scaleArray =
			vtkDoubleArray::SafeDownCast(pointData->GetArray("Scales"));
		for (vtkIdType i = 0; i < scaleArray->GetNumberOfTuples(); ++i)
		{
			double scale[3]; // Assuming 3-component scale array (X, Y, Z)
			scaleArray->GetTuple(i, scale);
			//std::cout << "Scale for point " << i << ": "
			//	<< scale[0 ] << ", " << scale[1] << ", " << scale[2] << std::endl;
			scale[0] *= 1.1;
			scale[1] *= 1.1;
			scale[2] *= 1.1;
			scaleArray->SetTuple(i, scale);
		}
		polyData->Modified();
		glyph3DMapper->SetScaleArray("Scales");
		glyph3DMapper->Update();
	}
	else if (key == "Return")
	{
		vtkRenderer* renderer = static_cast<vtkRenderer*>(clientData);
		vtkRenderWindow* renderWindow = renderer->GetRenderWindow();

		// Grab the depth buffer (z-values)
		int width = renderWindow->GetSize()[0];
		int height = renderWindow->GetSize()[1];
		vtkSmartPointer<vtkFloatArray> depthBuffer = vtkSmartPointer<vtkFloatArray>::New();
		depthBuffer->SetNumberOfComponents(1);
		depthBuffer->SetNumberOfTuples(width * height);

		renderWindow->GetZbufferData(0, 0, width - 1, height - 1, depthBuffer);

		// Save depth map to an image
		float minDepth = std::numeric_limits<float>::max();
		float maxDepth = std::numeric_limits<float>::lowest();

		for (vtkIdType i = 0; i < depthBuffer->GetNumberOfTuples(); i++) {
			float depthValue = depthBuffer->GetValue(i);
			minDepth = std::min(minDepth, depthValue);
			maxDepth = std::max(maxDepth, depthValue);
		}

		vtkSmartPointer<vtkImageData> depthImage = vtkSmartPointer<vtkImageData>::New();
		depthImage->SetDimensions(width, height, 1);
		depthImage->AllocateScalars(VTK_FLOAT, 1);

		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				float depthValue = depthBuffer->GetValue((height - y - 1) * width + x);
				float normalizedDepth = (depthValue - minDepth) / (maxDepth - minDepth);

				depthImage->SetScalarComponentFromFloat(x, y, 0, 0, normalizedDepth * 255);

				if (0.0f != normalizedDepth)
				{
					VisualDebugging::AddSphere("Depth", { (float)x / (float)width * 100.0f, (float)y / (float)height * 100.0f, normalizedDepth * 100.0f }, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 0.0f }, Color4::White);
				}
			}
		}

		// Optional: Cast the float image to unsigned char to save as PNG
		vtkSmartPointer<vtkImageCast> castFilter = vtkSmartPointer<vtkImageCast>::New();
		castFilter->SetInputData(depthImage);
		castFilter->SetOutputScalarTypeToUnsignedChar();
		castFilter->Update();

		vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
		writer->SetFileName("c:\\Debug\\2D\\depthmap.png");
		writer->SetInputData(castFilter->GetOutput());
		writer->Write();
	}
	else if (key == "1")
	{
		VisualDebugging::ToggleVisibility("Original");
	}
	else if (key == "2")
	{
		VisualDebugging::ToggleVisibility("DownSample");
	}
	else if (key == "3")
	{
		VisualDebugging::ToggleVisibility("Filter");
	}
	else if (key == "4")
	{
		VisualDebugging::ToggleVisibility("OctreeNode");
	}
	else if (key == "5")
	{
		VisualDebugging::ToggleVisibility("WiredBox_4");
	}
	else if (key == "6")
	{
		VisualDebugging::ToggleVisibility("WiredBox_5");
	}
	else if (key == "7")
	{
		VisualDebugging::ToggleVisibility("WiredBox_6");
	}
	else if (key == "8")
	{
		VisualDebugging::ToggleVisibility("WiredBox_7");
	}
	else if (key == "9")
	{
		VisualDebugging::ToggleVisibility("WiredBox_8");
	}
	else if (key == "0")
	{
		VisualDebugging::ToggleVisibility("WiredBox_9");
	}
	else if (key == "Left")
	{
		depthIndex--;
		if (depthIndex < 0) depthIndex = 0;

		for (int i = 0; i < 14; i++)
		{
			stringstream ss;
			ss << "Cubes_" << i;
			VisualDebugging::SetVisibility(ss.str(), false);
		}
		{
			stringstream ss;
			ss << "Cubes_" << depthIndex;
			VisualDebugging::SetVisibility(ss.str(), true);
		}

		printf("Depth : %d\n", depthIndex);
	}
	else if (key == "Right")
	{
		depthIndex++;
		if (depthIndex > 14) depthIndex = 13;

		for (int i = 0; i < 14; i++)
		{
			stringstream ss;
			ss << "Cubes_" << i;
			VisualDebugging::SetVisibility(ss.str(), false);
		}
		{
			stringstream ss;
			ss << "Cubes_" << depthIndex;
			VisualDebugging::SetVisibility(ss.str(), true);
		}

		printf("Depth : %d\n", depthIndex);
	}
	else if (key == "space")
	{
	}
}

int main()
{
	App app;
	app.AddAppStartCallback([&](App* pApp) {
		auto renderer = pApp->GetRenderer();

		SVO::Octree octree;
		SVO::InitializeSVO(&octree, 3000000);

		{
			Processing::PatchProcessor pp;
			float sampleSize = 0.1f;
			pp.Initialize(2000, 2000, 256 * 480);

			//int i = 3;
			for (int i = 3; i < 244; i++)
				//for (int i = 3; i < 57; i++)
				//for (int cnt = 0; cnt < 4; cnt++)
			{
				printf("======= [%4d] =======\n", i);

				auto te = Time::Now();
				LoadPatch(i, renderer);
				Time::End(te, "Loading PointCloud Patch");

				//for (size_t i = 0; i < patchPoints_0.size(); i++)
				//{
				//	auto& p = patchPoints_0[i];
				//	VisualDebugging::AddSphere("Original_0", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
				//}
				//for (size_t i = 0; i < patchPoints_45.size(); i++)
				//{
				//	auto& p = patchPoints_45[i];
				//	VisualDebugging::AddSphere("Original_45", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Blue);

				//SVO::IntegratePointCloud(&octree, aabb_0.center(), Eigen::Matrix4f(transform_0), patchPoints_0.data(), patchPoints_0.size(), 200, 14);
				//SVO::IntegratePointCloud(&octree, aabb_45.center(), Eigen::Matrix4f(transform_45), patchPoints_45.data(), patchPoints_45.size(), 200, 14);

				//SVO::IntegratePointCloudWithTSDF(&octree, aabb_0.center(), Eigen::Matrix4f(transform_0), patchPoints_0.data(), patchPoints_0.size(), 200, 14, 1.0f);
				//SVO::IntegratePointCloudWithTSDF(&octree, aabb_45.center(), Eigen::Matrix4f(transform_45), patchPoints_45.data(), patchPoints_45.size(), 200, 14, 1.0f);

				//SVO::IntegratePointCloudWithNeighborUpdate(&octree, aabb_0.center(), Eigen::Matrix4f(transform_0), patchPoints_0.data(), patchPoints_0.size(), 200, 13, 1.0f);
				//SVO::IntegratePointCloudWithNeighborUpdate(&octree, aabb_45.center(), Eigen::Matrix4f(transform_45), patchPoints_45.data(), patchPoints_45.size(), 200, 13, 1.0f);

				pp.DownSample(patchPoints_0.data(), patchPoints_0.size(), sampleSize, aabb_0.min(), aabb_0.max());
				pp.MedianFilter(pp.h_resultPoints, pp.h_numberOfResultPoints, sampleSize, aabb_0.min(), aabb_0.max());

				SVO::IntegratePointCloudWithTSDF(&octree, aabb_0.center(), Eigen::Matrix4f(transform_0), pp.h_resultPoints, pp.h_numberOfResultPoints, 200, 12, 1.0f);

				pp.DownSample(patchPoints_45.data(), patchPoints_45.size(), sampleSize, aabb_45.min(), aabb_45.max());
				pp.MedianFilter(pp.h_resultPoints, pp.h_numberOfResultPoints, sampleSize, aabb_45.min(), aabb_45.max());

				SVO::IntegratePointCloudWithTSDF(&octree, aabb_45.center(), Eigen::Matrix4f(transform_45), pp.h_resultPoints, pp.h_numberOfResultPoints, 200, 12, 1.0f);
			}

			//{
			//	float sampleSize = 0.1f;

			//	printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints_0.size());

			//	auto t = Time::Now();

			//	pp.DownSample(patchPoints_0.data(), patchPoints_0.size(), sampleSize, aabb_0.min(), aabb_0.max());

			//	t = Time::End(t, "DownSampling");

			//	printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

			//	//for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
			//	//{
			//	//	auto& p = pp.h_resultPoints[i];

			//	//	//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

			//	//	VisualDebugging::AddSphere("DownSample", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Blue);
			//	//}
			//}

			//{
			//	float sampleSize = 0.1f;

			//	printf("[[[ patchPoints.size() ]]] : %llu\n", patchPoints_0.size());

			//	auto t = Time::Now();

			//	pp.MedianFilter(patchPoints_0.data(), patchPoints_0.size(), sampleSize, aabb_45.min(), aabb_45.max());

			//	t = Time::End(t, "MedianFilter");

			//	printf("[[[ *pp.h_numberOfResultPoints ]]] : %llu\n", pp.h_numberOfResultPoints);

			//	//for (size_t i = 0; i < pp.h_numberOfResultPoints; i++)
			//	//{
			//	//	auto& p = pp.h_resultPoints[i];

			//	//	//p += Eigen::Vector3f(0.0f, 0.0f, 10.0f);

			//	//	VisualDebugging::AddSphere("Filter", p, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 0.0f }, Color4::Red);
			//	//}
			//}
		}

		SVO::TraverseOctree(&octree, [&](const SVO::SVONode& node, int depth) {

			//std::cout << "Node TSDF Value: " << node.tsdfValue << std::endl;

			//if (node.occupied)
			{
				stringstream ss;
				ss << "Cubes_" << depth;
				VisualDebugging::AddCube(ss.str(), node.center, { node.size, node.size, node.size }, { 0.0f, 0.0f, 1.0f }, Color4::White);
			}
			});

		vector<SVO::Triangle> triangles;
		SVO::ExtractTrianglesFromOctree(&octree, triangles);
		printf("triangles : %llu\n", triangles.size());

		for (auto& t : triangles)
		{
			VisualDebugging::AddTriangle("t", t.vertices[0], t.vertices[1], t.vertices[2], Color4::White);
		}

		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
		VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
		});

	app.AddKeyPressCallback(OnKeyPress);
	app.Run();

	return 0;
}
