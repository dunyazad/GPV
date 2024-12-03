#include "SVO.cuh"

#include <cufft.h>

#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
	namespace SVO
	{
#pragma region PatchBuffers
		PatchBuffers::PatchBuffers(int width, int height)
			: width(width), height(height),
			inputPoints(thrust::device_vector<Eigen::Vector3f>(width* height)),
			inputNormals(thrust::device_vector<Eigen::Vector3f>(width* height)),
			inputColors(thrust::device_vector<Eigen::Vector3f>(width* height))
		{
		}

		PatchBuffers::~PatchBuffers()
		{
		}

		void PatchBuffers::Clear()
		{
			numberOfInputPoints = 0;
			Eigen::Vector3f zeroVector(0.0f, 0.0f, 0.0f);
			thrust::fill(inputPoints.begin(), inputPoints.end(), zeroVector);
			thrust::fill(inputNormals.begin(), inputNormals.end(), zeroVector);
			thrust::fill(inputColors.begin(), inputColors.end(), zeroVector);
		}

		void PatchBuffers::FromPLYFile(const PLYFormat& ply)
		{
			Clear();

			numberOfInputPoints = ply.GetPoints().size() / 3;

			cudaMemcpy(thrust::raw_pointer_cast(inputPoints.data()),
				ply.GetPoints().data(),
				sizeof(Eigen::Vector3f) * numberOfInputPoints,
				cudaMemcpyHostToDevice);

			cudaMemcpy(thrust::raw_pointer_cast(inputNormals.data()),
				ply.GetPoints().data(),
				sizeof(Eigen::Vector3f) * numberOfInputPoints,
				cudaMemcpyHostToDevice);

			cudaMemcpy(thrust::raw_pointer_cast(inputColors.data()),
				ply.GetPoints().data(),
				sizeof(Eigen::Vector3f) * numberOfInputPoints,
				cudaMemcpyHostToDevice);

			for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
			{
				auto px = ply.GetPoints()[i * 3 + 0];
				auto py = ply.GetPoints()[i * 3 + 1];
				auto pz = ply.GetPoints()[i * 3 + 2];

				auto nx = ply.GetNormals()[i * 3 + 0];
				auto ny = ply.GetNormals()[i * 3 + 1];
				auto nz = ply.GetNormals()[i * 3 + 2];

				auto cx = ply.GetColors()[i * 3 + 0];
				auto cy = ply.GetColors()[i * 3 + 1];
				auto cz = ply.GetColors()[i * 3 + 2];

				auto c4 = Color4::FromNormalized(cx, cy, cz, 1.0f);
				VD::AddSphere("points", { px, py, pz }, { 0.05f, 0.05f, 0.05f }, { nx, ny, nz }, c4);
			}
		}
#pragma endregion

		//__host__ __device__
		//	uint64_t GetMortonCode(
		//		const Eigen::Vector3f& min,
		//		const Eigen::Vector3f& max,
		//		int maxDepth,
		//		const Eigen::Vector3f& position) {
		//	// Validate and compute range
		//	Eigen::Vector3f range = max - min;
		//	range = range.cwiseMax(Eigen::Vector3f::Constant(1e-6f)); // Avoid zero range

		//	// Normalize position
		//	Eigen::Vector3f relativePos = (position - min).cwiseQuotient(range);

		//	// Clamp to [0, 1]
		//	relativePos = relativePos.cwiseMax(0.0f).cwiseMin(1.0f);

		//	// Scale to Morton grid size
		//	uint32_t maxCoordinateValue = (1 << maxDepth) - 1; // maxCoordinateValue = 1 for maxDepth = 1
		//	//uint32_t x = static_cast<uint32_t>(roundf(relativePos.x() * maxCoordinateValue * 1000)) / 1000;
		//	//uint32_t y = static_cast<uint32_t>(roundf(relativePos.y() * maxCoordinateValue * 1000)) / 1000;
		//	//uint32_t z = static_cast<uint32_t>(roundf(relativePos.z() * maxCoordinateValue * 1000)) / 1000;
		//	uint32_t x = static_cast<uint32_t>(roundf(relativePos.x() * maxCoordinateValue));
		//	uint32_t y = static_cast<uint32_t>(roundf(relativePos.y() * maxCoordinateValue));
		//	uint32_t z = static_cast<uint32_t>(roundf(relativePos.z() * maxCoordinateValue));

		//	// Compute Morton code
		//	uint64_t mortonCode = 0;
		//	for (int i = 0; i < maxDepth; ++i) {
		//		mortonCode |= ((x >> i) & 1ULL) << (3 * i);
		//		mortonCode |= ((y >> i) & 1ULL) << (3 * i + 1);
		//		mortonCode |= ((z >> i) & 1ULL) << (3 * i + 2);
		//	}

		//	return mortonCode;
		//}

		__host__ __device__
			void printBinary(uint64_t num)
		{
			char buffer[100];  // Buffer large enough to hold the entire output, including binary, thread info, and null terminator
			int offset = 0;

			// Construct the binary representation
			const int BITS = sizeof(num) * 8;
			for (int i = BITS - 1; i >= 0; i--) {
				uint64_t mask = 1ULL << i;
				buffer[offset++] = (num & mask) ? '1' : '0';
			}

			buffer[offset] = '\0';  // Null-terminate the string

			// Print the entire buffer in one printf call
			printf("%s\n", buffer);
		}

		struct OctreeNode {
			uint64_t mortonCode;
			bool isLeaf;
			int children[8];  // �ڽ� ����� �ε����� ����. -1�� �ڽ��� ������ �ǹ�.
			int parentIndex;  // �θ� ����� �ε���. ��Ʈ�� -1�� ����.
			Eigen::Vector3f minBound;  // ����� �ּ� ���.
			Eigen::Vector3f maxBound;  // ����� �ִ� ���.
			int depth;  // ����� ����.

			__host__ __device__
				OctreeNode(uint64_t code = 0, int parent = -1, const Eigen::Vector3f& min = Eigen::Vector3f::Zero(), const Eigen::Vector3f& max = Eigen::Vector3f::Zero(), int depth = 0)
				: mortonCode(code), isLeaf(true), parentIndex(parent), minBound(min), maxBound(max), depth(depth) {
				for (int i = 0; i < 8; ++i) {
					children[i] = -1;  // ��� �ڽ� ��带 -1�� �ʱ�ȭ
				}
			}
		};

		__host__ __device__
			uint64_t GetMortonCode(
				const Eigen::Vector3f& min,
				const Eigen::Vector3f& max,
				int maxDepth,
				const Eigen::Vector3f& position) {
			// Validate and compute range
			Eigen::Vector3f range = max - min;
			range = range.cwiseMax(Eigen::Vector3f::Constant(1e-6f)); // Avoid zero range

			// Normalize position
			Eigen::Vector3f relativePos = (position - min).cwiseQuotient(range);

			// Clamp to [0, 1]
			relativePos = relativePos.cwiseMax(0.0f).cwiseMin(1.0f);

			// Scale to Morton grid size
			uint32_t maxCoordinateValue = (1 << maxDepth) - 1; // maxCoordinateValue = 1 for maxDepth = 1
			uint32_t x = static_cast<uint32_t>(roundf(relativePos.x() * maxCoordinateValue));
			uint32_t y = static_cast<uint32_t>(roundf(relativePos.y() * maxCoordinateValue));
			uint32_t z = static_cast<uint32_t>(roundf(relativePos.z() * maxCoordinateValue));

			// Compute Morton code
			uint64_t mortonCode = 0;
			for (int i = 0; i < maxDepth; ++i) {
				mortonCode |= ((x >> i) & 1ULL) << (3 * i);
				mortonCode |= ((y >> i) & 1ULL) << (3 * i + 1);
				mortonCode |= ((z >> i) & 1ULL) << (3 * i + 2);
			}

			return mortonCode;
		}

		Eigen::Vector3f MortonCodeToPosition(uint64_t mortonCode, int depth, const Eigen::Vector3f& minBound, const Eigen::Vector3f& maxBound) {
			Eigen::Vector3f range = maxBound - minBound;

			// �ʱ� ��ġ�� ũ�� ����
			Eigen::Vector3f position(0.0f, 0.0f, 0.0f);
			float stepSize = 1.0f;

			// Morton �ڵ带 ������� ��ġ ���
			for (int i = 0; i < depth; ++i) {
				int shift = (depth - 1 - i) * 3;
				uint64_t xBit = (mortonCode >> (shift + 2)) & 1;
				uint64_t yBit = (mortonCode >> (shift + 1)) & 1;
				uint64_t zBit = (mortonCode >> shift) & 1;

				// �� ��Ʈ�� ����� ������ �����ϰ� ��ġ�� ����
				stepSize /= 2.0f;
				position += Eigen::Vector3f(xBit * stepSize, yBit * stepSize, zBit * stepSize);
			}

			// ���� ��ġ�� ������ Morton �ڵ忡���� ��� ��ġ�� ���� ����
			return minBound + position.cwiseProduct(range);
		}

		// Octree ���� �Լ� ���� (thrust ���� ���)
		void BuildOctreeWithThrust(
			thrust::device_vector<uint64_t>& mortonCodes,
			thrust::device_vector<OctreeNode>& octreeNodes,
			int maxDepth,
			const Eigen::Vector3f& minBound,
			const Eigen::Vector3f& maxBound) {

			octreeNodes[0] = OctreeNode(0, -1, minBound, maxBound, 0);

			int numNodes = mortonCodes.size();

			// raw pointer�� ����� GPU �޸𸮿� ���� �����ϵ��� ����
			uint64_t* mortonCodesPtr = thrust::raw_pointer_cast(mortonCodes.data());
			OctreeNode* octreeNodesPtr = thrust::raw_pointer_cast(octreeNodes.data());

			thrust::for_each(
				thrust::make_counting_iterator<int>(0),
				thrust::make_counting_iterator<int>(numNodes),
				[=] __device__(int idx) {
				uint64_t mortonCode = mortonCodesPtr[idx];
				int parentIndex = 0; // ��Ʈ���� ���� (��Ʈ �ε����� 0)

				Eigen::Vector3f currentMin = minBound;
				Eigen::Vector3f currentMax = maxBound;

				for (int depth = 0; depth < maxDepth; ++depth) {
					if (depth == 6) return;

					// ���� ���̿��� 3��Ʈ�� �����Ͽ� ��� �ڽ����� ���������� ����
					int shift = (maxDepth - 1 - depth) * 3;
					uint64_t childIndex = (mortonCode >> shift) & 0b111;

					// ��� ������Ʈ
					Eigen::Vector3f midPoint = (currentMin + currentMax) * 0.5f;
					if (childIndex & 0b001) currentMin.x() = midPoint.x(); else currentMax.x() = midPoint.x();
					if (childIndex & 0b010) currentMin.y() = midPoint.y(); else currentMax.y() = midPoint.y();
					if (childIndex & 0b100) currentMin.z() = midPoint.z(); else currentMax.z() = midPoint.z();

					// �θ� ���κ��� �ڽ� ��带 ����
					int* childPtr = &(octreeNodesPtr[parentIndex].children[childIndex]);
					int expected = -1;  // �ڽ��� ���ٰ� ���
					int newChildIndex;

					// atomicCAS�� �̿��Ͽ� �ڽ� ��带 ���������� ����
					if (atomicCAS(childPtr, expected, idx) == expected) {
						// �ڽ��� ���������� ������ ��� ���ο� �ڽ� ��带 �ʱ�ȭ
						newChildIndex = idx;
						octreeNodesPtr[newChildIndex] = OctreeNode(mortonCode, parentIndex, currentMin, currentMax, depth + 1);
						octreeNodesPtr[parentIndex].isLeaf = false;
					}
					else {
						// �̹� �ٸ� �����忡 ���� �ڽ��� ������ ���
						newChildIndex = *childPtr;
					}

					// ���� �θ� �ε����� �����Ͽ� ��� �Ʒ��� Ž��
					parentIndex = newChildIndex;
				}
			}
			);
		}

		void Traverse(const thrust::host_vector<OctreeNode>& nodes, int nodeIndex, int depth)
		{
			auto& node = nodes[nodeIndex];
			Eigen::Vector3f center = (node.minBound + node.maxBound) * 0.5f;
			Eigen::Vector3f scale = (node.maxBound - node.minBound) * 0.5f;
			//if (depth == 0)
			{
				//printBinary(node.mortonCode);
				//printf("%6.4f, %6.4f, %6.4f <---> %6.4f, %6.4f, %6.4f\n",
				//	node.mortonCode,
				//	node.minBound.x(), node.minBound.y(), node.minBound.z(),
				//	node.maxBound.x(), node.maxBound.y(), node.maxBound.z());

				stringstream ss;
				ss << "Cubes_" << depth;
				VD::AddCube(ss.str(), center, scale, {0.0f, 0.0f, 1.0f}, Color4::White);
			}

			for (int i = 0; i < 8; ++i) {
				int childIndex = nodes[nodeIndex].children[i];
				if (childIndex != -1) {

					Traverse(nodes, childIndex, depth + 1);
				}
			}
		}

		void TestSVO()
		{
			auto t = Time::Now();

			stringstream ss;
			ss << "C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply";

			PLYFormat ply;
			ply.Deserialize(ss.str());
			cout << "ply min : " << ply.GetAABB().min().transpose() << endl;
			cout << "ply max : " << ply.GetAABB().max().transpose() << endl;

			t = Time::End(t, "Load ply");

			PatchBuffers patchBuffers(ply.GetPoints().size() / 3, 1);
			patchBuffers.FromPLYFile(ply);

			t = Time::End(t, "Copy data to device");

			Eigen::Vector3f min(-100.0f, -100.0f, -100.0f);
			Eigen::Vector3f max(100.0f, 100.0f, 100.0f);
			int maxDepth = 13;

			printf("patchBuffers.numberOfInputPoints : %d\n", patchBuffers.numberOfInputPoints);

			thrust::device_vector<uint64_t> mortonCodes(patchBuffers.numberOfInputPoints);

			t = Time::Now();
			nvtxRangePushA("ConvertToMortonCode");
			thrust::transform(
				patchBuffers.inputPoints.begin(), patchBuffers.inputPoints.end(),
				mortonCodes.begin(),
				[=] __device__(const Eigen::Vector3f & point) {
				return GetMortonCode(min, max, maxDepth, point);
			});
			nvtxRangePop();
			t = Time::End(t, "Convert To MortonCode");

			nvtxRangePushA("sort");
			thrust::sort(mortonCodes.begin(), mortonCodes.end());
			nvtxRangePop();
			t = Time::End(t, "Sort");

			nvtxRangePushA("unique");
			auto newEnd = thrust::unique(mortonCodes.begin(), mortonCodes.end());
			nvtxRangePop();
			t = Time::End(t, "Unique");

			nvtxRangePushA("erase");
			mortonCodes.erase(newEnd, mortonCodes.end());
			nvtxRangePop();
			t = Time::End(t, "Erase");

			nvtxRangePushA("ready octreeNodes");
			int numNodes = mortonCodes.size();
			thrust::device_vector<OctreeNode> octreeNodes(numNodes * maxDepth);
			nvtxRangePop();
			t = Time::End(t, "ready octreeNodes");

			nvtxRangePushA("BuildOctreeWithThrust");
			BuildOctreeWithThrust(mortonCodes, octreeNodes, maxDepth, min, max);
			nvtxRangePop();
			t = Time::End(t, "BuildOctreeWithThrust");

			// Octree ��带 ȣ��Ʈ�� �����ϰ� �ð�ȭ
			thrust::host_vector<OctreeNode> hostOctreeNodes = octreeNodes;

			//string layerName = "OctreeNodes";
			//Color4 color(1.0f, 0.0f, 0.0f, 0.5f); // ������ ������ ť��
			//VisualizeOctreeNode(hostOctreeNodes, 0, maxDepth, min, max, layerName, color);

			Traverse(hostOctreeNodes, 0, 0);
		}
	}
}
