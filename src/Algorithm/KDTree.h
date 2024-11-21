#pragma once

#include <Common.h>

namespace Algorithm
{
	struct Ray {
		float origin[3];
		float direction[3]; // Must be normalized
	};

	float RayPointDistanceSquared(const Ray& ray, const float* point);
	
	// Helper for max-heap (priority queue) comparison
	struct MaxHeapEntry {
		unsigned int pointIndex;
		float distanceSquared;

		bool operator<(const MaxHeapEntry& other) const {
			return distanceSquared < other.distanceSquared; // Max-heap
		}
	};

	class KDTreeNode
	{
	public:
		KDTreeNode(unsigned int point_index) : point_index(point_index) {}

		inline unsigned int GetPointIndex() const { return point_index; }

	private:
		unsigned int point_index = UINT_MAX;
		KDTreeNode* left = nullptr;
		KDTreeNode* right = nullptr;

	public:
		friend class KDTree;
	};

	class KDTree
	{
	public:
		KDTree() : points(nullptr) {}
		KDTree(const float* points) : points(points) {}
		~KDTree() { Clear(); }

		void Clear()
		{
			if (nullptr != root)
			{
				ClearRecursive(root);
				root = nullptr;
			}

			nearestNeighborNode = nullptr;
			nearestNeighbor = UINT_MAX;
			nearestNeighborDistance = FLT_MAX;
		}

		void Insert(unsigned int point_index)
		{
			root = InsertRecursive(root, point_index, 0);
		}

		void BuildTree(std::vector<unsigned int>& pointIndices)
		{
			if(nullptr != root)
			{
				Clear();
			}

			root = BuildTreeRecursive(pointIndices, 0);
		}

		unsigned int FindNearestNeighbor(const float* query) const
		{
			nearestNeighbor = root->point_index;
			auto dx = points[root->point_index * 3 + 0] - query[0];
			auto dy = points[root->point_index * 3 + 1] - query[1];
			auto dz = points[root->point_index * 3 + 2] - query[2];
			nearestNeighborDistance = dx * dx + dy * dy + dz * dz;
			nearestNeighborNode = nullptr;
			FindNearestNeighborRecursive(root, query, 0);
			return nearestNeighbor;
		}

		KDTreeNode* FindNearestNeighborNode(const float* query) const
		{
			if (nullptr == root)
				return nullptr;

			nearestNeighbor = UINT_MAX;
			nearestNeighborDistance = FLT_MAX;
			nearestNeighborNode = nullptr;
			FindNearestNeighborRecursive(root, query, 0);
			return nearestNeighborNode;
		}

		vector<unsigned int> RangeSearchSquaredDistance(const float* query, float squaredRadius) const
		{
			vector<unsigned int> result;
			RangeSearchRecursive(root, query, squaredRadius, result, 0);
			return result;
		}

		inline bool IsEmpty() const { return nullptr == root; }

		inline const float* GetPoints() const { return points; }
		inline void SetPoints(const float* points) { this->points = points; }

		std::vector<unsigned int> RayKNearestNeighbors(const Ray& ray, int k, float maxDistance = FLT_MAX) const {
			std::priority_queue<MaxHeapEntry> maxHeap; // Max-heap for K nearest neighbors
			RayKNNRecursive(root, ray, k, maxDistance, 0, maxHeap);

			// Extract indices from the heap
			std::vector<unsigned int> result;
			while (!maxHeap.empty()) {
				result.push_back(maxHeap.top().pointIndex);
				maxHeap.pop();
			}
			std::reverse(result.begin(), result.end()); // Sort closest first
			return result;
		}

		void TraversePreOrder(function<void(KDTreeNode*)> f) const
		{
			TraversePreOrderRecursive(root, f);
		}

	private:
		const float* points;

		KDTreeNode* root = nullptr;
		mutable KDTreeNode* nearestNeighborNode = nullptr;
		mutable unsigned int nearestNeighbor = UINT_MAX;
		mutable float nearestNeighborDistance = FLT_MAX;

		void ClearRecursive(KDTreeNode* node)
		{
			if (nullptr != node->left)
			{
				ClearRecursive(node->left);
			}

			if (nullptr != node->right)
			{
				ClearRecursive(node->right);
			}

			delete node;
		}

		KDTreeNode* InsertRecursive(KDTreeNode* node, unsigned int point_index, int depth)
		{
			if (node == nullptr)
			{
				auto newNode = new KDTreeNode(point_index);
				return newNode;
			}

			int currentDimension = depth % 3;
			float pointValue = points[point_index * 3 + currentDimension];
			float nodeValue = points[node->point_index * 3 + currentDimension];

			if (pointValue < nodeValue)
			{
				node->left = InsertRecursive(node->left, point_index, depth + 1);
			}
			else
			{
				node->right = InsertRecursive(node->right, point_index, depth + 1);
			}

			return node;
		}

		KDTreeNode* BuildTreeRecursive(std::vector<unsigned int>& pointIndices, int depth)
		{
			if (pointIndices.empty()) {
				return nullptr;
			}

			// Determine the current axis to split on
			int axis = depth % 3;

			// Sort the points along the current axis
			std::sort(pointIndices.begin(), pointIndices.end(), [&](unsigned int lhs, unsigned int rhs) {
				return points[lhs * 3 + axis] < points[rhs * 3 + axis];
				});

			// Find the median index
			size_t medianIndex = pointIndices.size() / 2;

			// Create a new node with the median point
			KDTreeNode* node = new KDTreeNode(pointIndices[medianIndex]);

			// Prepare left and right subsets
			std::vector<unsigned int> leftIndices(pointIndices.begin(), pointIndices.begin() + medianIndex);
			std::vector<unsigned int> rightIndices(pointIndices.begin() + medianIndex + 1, pointIndices.end());

			// Recursively build the left and right subtrees
			node->left = BuildTreeRecursive(leftIndices, depth + 1);
			node->right = BuildTreeRecursive(rightIndices, depth + 1);

			return node;
		}

		void FindNearestNeighborRecursive(KDTreeNode* node, const float* query, int depth) const
		{
			if (node == nullptr)
			{
				return;
			}

			int currentDimension = depth % 3;
			auto dx = points[node->point_index * 3 + 0] - query[0];
			auto dy = points[node->point_index * 3 + 1] - query[1];
			auto dz = points[node->point_index * 3 + 2] - query[2];
			float nodeDistance = sqrtf(dx * dx + dy * dy + dz * dz);
			if (nodeDistance < nearestNeighborDistance)
			{
				nearestNeighborNode = node;
				nearestNeighbor = node->point_index;
				nearestNeighborDistance = nodeDistance;
			}

			auto queryValue = query[currentDimension];
			auto nodeValue = points[node->point_index * 3 + currentDimension];

			KDTreeNode* closerNode = (queryValue < nodeValue) ? node->left : node->right;
			KDTreeNode* otherNode = (queryValue < nodeValue) ? node->right : node->left;

			FindNearestNeighborRecursive(closerNode, query, depth + 1);

			float planeDistance = queryValue - nodeValue;
			if (planeDistance * planeDistance < nearestNeighborDistance) {
				FindNearestNeighborRecursive(otherNode, query, depth + 1);
			}
		}

		void RangeSearchRecursive(KDTreeNode* node, const float* query, float squaredRadius, std::vector<unsigned int>& result, int depth) const
		{
			if (node == nullptr)
			{
				return;
			}

			auto dx = points[node->point_index * 3 + 0] - query[0];
			auto dy = points[node->point_index * 3 + 1] - query[1];
			auto dz = points[node->point_index * 3 + 2] - query[2];
			float nodeDistance = sqrtf(dx * dx + dy * dy + dz * dz);

			if (nodeDistance <= squaredRadius)
			{
				result.push_back(node->point_index);
			}

			int currentDimension = depth % 3;
			float queryValue = query[currentDimension];
			float nodeValue = points[node->point_index * 3 + currentDimension];

			KDTreeNode* closerNode = (queryValue < nodeValue) ? node->left : node->right;
			KDTreeNode* otherNode = (queryValue < nodeValue) ? node->right : node->left;

			RangeSearchRecursive(closerNode, query, squaredRadius, result, depth + 1);

			float distanceToPlane = queryValue - nodeValue;
			if (distanceToPlane * distanceToPlane <= squaredRadius)
			{
				RangeSearchRecursive(otherNode, query, squaredRadius, result, depth + 1);
			}
		}

		void RayKNNRecursive(
			KDTreeNode* node,
			const Ray& ray,
			int k,
			float maxDistance,
			int depth,
			std::priority_queue<MaxHeapEntry>& maxHeap
		) const {
			if (node == nullptr) {
				return;  // Base case: reached a leaf node
			}

			// Step 1: Compute the squared distance from the ray to the current node's point
			float point[3] = {
				points[node->point_index * 3 + 0],
				points[node->point_index * 3 + 1],
				points[node->point_index * 3 + 2]
			};

			float distanceSquared = RayPointDistanceSquared(ray, point);

			// Step 2: Update the maxHeap if the current point is within maxDistance
			if (distanceSquared <= maxDistance * maxDistance) {
				if (maxHeap.size() < k) {
					// Heap has fewer than k points, so add this point
					maxHeap.push({ node->point_index, distanceSquared });
				}
				else if (distanceSquared < maxHeap.top().distanceSquared) {
					// Replace the farthest point in the heap if this one is closer
					maxHeap.pop();
					maxHeap.push({ node->point_index, distanceSquared });
				}
			}

			// Step 3: Determine which subtree to explore first
			int currentDimension = depth % 3;

			float nodeValue = points[node->point_index * 3 + currentDimension];
			float rayOriginValue = ray.origin[currentDimension];

			KDTreeNode* closerNode = nullptr;
			KDTreeNode* fartherNode = nullptr;

			// Choose which child node is closer based on the ray origin value
			if (rayOriginValue < nodeValue) {
				closerNode = node->left;
				fartherNode = node->right;
			}
			else {
				closerNode = node->right;
				fartherNode = node->left;
			}

			// Step 4: Explore the closer node first
			RayKNNRecursive(closerNode, ray, k, maxDistance, depth + 1, maxHeap);

			// Step 5: Determine whether we need to explore the farther node
			float planeDistance = rayOriginValue - nodeValue;  // Distance to the splitting plane
			float planeDistanceSquared = planeDistance * planeDistance;

			// We need to explore the farther node if:
			// 1. The heap has fewer than k points (we need more neighbors)
			// 2. The distance to the splitting plane is less than or equal to the current farthest distance in the heap
			if (maxHeap.size() < k || planeDistanceSquared <= maxHeap.top().distanceSquared) {
				RayKNNRecursive(fartherNode, ray, k, maxDistance, depth + 1, maxHeap);
			}
		}

		void TraversePreOrderRecursive(KDTreeNode* node, function<void(KDTreeNode*)> f) const
		{
			if (node == nullptr) return;

			// Visit the current node
			f(node);

			// Traverse the left subtree
			TraversePreOrderRecursive(node->left, f);

			// Traverse the right subtree
			TraversePreOrderRecursive(node->right, f);
		}
	};
}
