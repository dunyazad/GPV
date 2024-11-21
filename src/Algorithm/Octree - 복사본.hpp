#pragma once

#ifndef __CUDA_ARCH__
#include <atomic>
#include <functional>
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Spatial
{
#define FLT_MAX          3.402823466e+38F        // max value

	typedef size_t OctreeNodePtr;
	const OctreeNodePtr InvalidPointer = UINT64_MAX;

	class OctreeNode
	{
	public:
		bool isLeaf = false;
		OctreeNodePtr children[8] = {
			InvalidPointer, InvalidPointer,
			InvalidPointer, InvalidPointer,
			InvalidPointer, InvalidPointer,
			InvalidPointer, InvalidPointer };

		Eigen::Vector3f min = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
		Eigen::Vector3f max = Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);

		float tsdf = FLT_MAX;
		float weight = 0.0f;
	};

	class Octree
	{
	public:
		std::vector<OctreeNode> nodes;
		OctreeNodePtr root = InvalidPointer;
		std::atomic<OctreeNodePtr> availableNodePtr = 0;
		int maxDepth = 13;

		Eigen::Vector3f min = Eigen::Vector3f(-250.0f, -250.0f, -250.0f);
		Eigen::Vector3f max = Eigen::Vector3f(250.0f, 250.0f, 250.0f);

		void Initialize(size_t numberOfNodes)
		{
			nodes.resize(numberOfNodes);
		}

		void Insert(const Eigen::Vector3f& position)
		{
			// If the root is not initialized, create the root node
			if (root == InvalidPointer) {
				root = NewNode();
				nodes.emplace_back();  // Add the root node to the vector
			}

			OctreeNodePtr currentNode = root;
			Eigen::Vector3f currentMin = min;
			Eigen::Vector3f currentMax = max;

			// Traverse the octree to find or create the appropriate leaf
			for (int depth = 0; depth < maxDepth; ++depth) {
				OctreeNode& node = nodes[currentNode];

				// Determine which child the point falls into
				int childIndex = 0;
				Eigen::Vector3f mid = (currentMin + currentMax) * 0.5f;
				if (position.x() > mid.x()) childIndex |= 1;
				if (position.y() > mid.y()) childIndex |= 2;
				if (position.z() > mid.z()) childIndex |= 4;

				// If the child does not exist, create it
				if (node.children[childIndex] == InvalidPointer) {
					node.children[childIndex] = NewNode();
					nodes.emplace_back(); // Add a new child node to the vector

					// Initialize the child node bounds
					OctreeNode& childNode = nodes[node.children[childIndex]];
					childNode.min = currentMin;
					childNode.max = currentMax;

					if (position.x() > mid.x()) childNode.min.x() = mid.x(); else childNode.max.x() = mid.x();
					if (position.y() > mid.y()) childNode.min.y() = mid.y(); else childNode.max.y() = mid.y();
					if (position.z() > mid.z()) childNode.min.z() = mid.z(); else childNode.max.z() = mid.z();
				}

				// Update current node pointer and bounds for the next level
				currentNode = node.children[childIndex];
				if (position.x() > mid.x()) currentMin.x() = mid.x(); else currentMax.x() = mid.x();
				if (position.y() > mid.y()) currentMin.y() = mid.y(); else currentMax.y() = mid.y();
				if (position.z() > mid.z()) currentMin.z() = mid.z(); else currentMax.z() = mid.z();
			}

			// Mark the final node as a leaf
			OctreeNode& leafNode = nodes[currentNode];
			leafNode.isLeaf = true;
			leafNode.tsdf = 0.0f; // Set appropriate TSDF value here
			leafNode.weight += 1.0f; // Update weight as necessary
		}

		OctreeNodePtr NewNode()
		{
			return availableNodePtr.fetch_add(1);
		}

		void Traverse(OctreeNodePtr nodePtr, std::function<void(OctreeNode&)> onLeaf)
		{
			if (nodePtr == InvalidPointer) return;

			// Retrieve the node from the nodes vector
			OctreeNode& node = nodes[nodePtr];

			// Process the current node (for example, printing its info)
			if (node.isLeaf)
			{
				//std::cout << "Leaf Node: TSDF = " << node.tsdf << ", Weight = " << node.weight << std::endl;

				onLeaf(node);
			}
			else
			{
				//std::cout << "Internal Node" << std::endl;
			}

			// Recursively traverse each child
			for (int i = 0; i < 8; ++i)
			{
				if (node.children[i] != InvalidPointer)
				{
					Traverse(node.children[i], onLeaf);
				}
			}
		}
	};

	void Test(const std::vector<Eigen::Vector3f>& inputPoints);
}