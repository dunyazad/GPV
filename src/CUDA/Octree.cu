#include "Octree.cuh"

#include <cufft.h>

#include <App/Serialization.hpp>
#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
	template<typename T>
	struct Octant;

	template<typename T>
	class Octree;

	class PatchBuffers
	{
	public:
		PatchBuffers(int width = 256, int height = 480)
			: width(width), height(height)
		{
			cudaMallocManaged(&inputPoints, sizeof(Eigen::Vector3f) * width * height);
			cudaMallocManaged(&inputNormals, sizeof(Eigen::Vector3f) * width * height);
			cudaMallocManaged(&inputColors, sizeof(Eigen::Vector3f) * width * height);
		}

		~PatchBuffers()
		{
			cudaFree(inputPoints);
			cudaFree(inputNormals);
			cudaFree(inputColors);
		}

		int width;
		int height;
		size_t numberOfInputPoints;
		Eigen::Vector3f* inputPoints;
		Eigen::Vector3f* inputNormals;
		Eigen::Vector3f* inputColors;

		void Clear()
		{
			numberOfInputPoints = 0;
			cudaMemset(inputPoints, 0, sizeof(Eigen::Vector3f) * width * height);
			cudaMemset(inputNormals, 0, sizeof(Eigen::Vector3f) * width * height);
			cudaMemset(inputColors, 0, sizeof(Eigen::Vector3f) * width * height);
		}

		void FromPLYFile(const PLYFormat& ply)
		{
			Clear();

			numberOfInputPoints = ply.GetPoints().size() / 3;
			cudaMemcpy(inputPoints, ply.GetPoints().data(), sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyHostToDevice);
			cudaMemcpy(inputNormals, ply.GetNormals().data(), sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyHostToDevice);
			cudaMemcpy(inputColors, ply.GetColors().data(), sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyHostToDevice);

			//for (size_t i = 0; i < numberOfInputPoints; i++)
			//{
			//	auto p = inputPoints[i];
			//	auto n = inputNormals[i];
			//	auto c = inputColors[i];
			//	Color4 c4;
			//	c4.FromNormalized(c.x(), c.y(), c.z(), 1.0f);
			//	VD::AddSphere("points", p, { 0.05f, 0.05f, 0.05f }, n, c4);
			//}
		}
	};

	struct Point
	{
		float tsdfValue;
		float weight;
		Eigen::Vector3f normal;
		Eigen::Vector3f color;
	};

	template<typename T>
	struct Octant
	{
		T data;
		int depth;
		unsigned int lock;
		size_t mortonCode;
		Octant<T>* children[8];
	};

	template<typename T>
	__global__ void Kernel_Insert(Octree<T>::Internal* octree, PatchBuffers patchBuffers);

	template<typename T>
	class Octree
	{
	public:
		struct Internal
		{
			bool initialized;
			int maxDepth;
			Eigen::Vector3f min;
			Eigen::Vector3f max;
			Eigen::Vector3f center;
			Octant<T>* root;

			size_t allocatedCount;
			Octant<T>* allocated;
			size_t* nextAllocationIndex;

			Internal()
				: initialized(false),
				maxDepth(0),
				min(FLT_MAX, FLT_MAX, FLT_MAX),
				max(-FLT_MAX, -FLT_MAX, -FLT_MAX),
				center(0.0f, 0.0f, 0.0f),
				root(nullptr),
				allocatedCount(0),
				allocated(nullptr),
				nextAllocationIndex(nullptr)
			{
			}

			__device__
				uint64_t GetMortonCode(const Eigen::Vector3f& position) {
				// Validate and compute range
				Eigen::Vector3f range = max - min;
				range = range.cwiseMax(Eigen::Vector3f::Constant(1e-6f)); // Avoid zero range

				// Normalize position
				Eigen::Vector3f relativePos = (position - min).cwiseQuotient(range);

				// Clamp to [0, 1]
				relativePos = relativePos.cwiseMax(0.0f).cwiseMin(1.0f);

				// Scale to Morton grid size
				uint32_t maxCoordinateValue = (1 << maxDepth) - 1; // maxCoordinateValue = 1 for maxDepth = 1
				uint32_t x = static_cast<uint32_t>(roundf(relativePos.x() * maxCoordinateValue * 1000)) / 1000;
				uint32_t y = static_cast<uint32_t>(roundf(relativePos.y() * maxCoordinateValue * 1000)) / 1000;
				uint32_t z = static_cast<uint32_t>(roundf(relativePos.z() * maxCoordinateValue * 1000)) / 1000;

				// Compute Morton code
				uint64_t mortonCode = 0;
				for (int i = 0; i < maxDepth; ++i) {
					mortonCode |= ((x >> i) & 1ULL) << (3 * i);
					mortonCode |= ((y >> i) & 1ULL) << (3 * i + 1);
					mortonCode |= ((z >> i) & 1ULL) << (3 * i + 2);
				}

				return mortonCode;
			}

			__device__
				Octant<T>* AllocateOctant()
			{
				auto index = atomicAdd(nextAllocationIndex, 1);
				if (index >= allocatedCount) return nullptr;
				allocated[index].depth = 0;
				allocated[index].lock = 0;
				allocated[index].mortonCode = 0;
				allocated[index].children[0]= nullptr;
				allocated[index].children[1] = nullptr;
				allocated[index].children[2] = nullptr;
				allocated[index].children[3] = nullptr;
				allocated[index].children[4] = nullptr;
				allocated[index].children[5] = nullptr;
				allocated[index].children[6] = nullptr;
				allocated[index].children[7] = nullptr;
				return &allocated[index];
			}


			__device__ Octant<T>* AllocateOctantsUsingMortonCode(size_t mortonCode) {
				Octant<T>* current = root;

				for (int level = 0; level < maxDepth; ++level) {
					int childIndex = (mortonCode >> (3 * level)) & 0x7;

					if (current->children[childIndex] == nullptr) {
						unsigned int expected = 0;
						while (atomicCAS(&(current->lock), expected, 1) != expected) {
							// Wait until lock is available
						}

						if (current->children[childIndex] == nullptr) {
							Octant<T>* newChild = AllocateOctant();
							if (newChild == nullptr) {
								printf("Failed to allocate child octant at depth %d.\n", level);
								atomicExch(&(current->lock), 0); // Release lock
								return nullptr;
							}

							newChild->depth = current->depth + 1;
							newChild->mortonCode = mortonCode & ((1ULL << (3 * (level + 1))) - 1);

							current->children[childIndex] = newChild;

							printBinary(newChild->mortonCode);
							printf("Allocated child octant at depth %d, Morton Code: %llu, Index: %d\n",
								level + 1, newChild->mortonCode, childIndex);
						}

						atomicExch(&(current->lock), 0); // Release lock
					}

					current = current->children[childIndex];
				}

				return current;
			}


		};

		Internal* internal;

		Octree()
		{
			cudaError_t err = cudaMallocManaged(&internal, sizeof(Internal));
			if (err != cudaSuccess)
			{
				printf("Failed to allocate memory for internal: %s\n", cudaGetErrorString(err));
				internal = nullptr;  // Set to null to avoid accidental access
			}
			else
			{
				memset(internal, 0, sizeof(Internal));

				internal->initialized = false;
				internal->min = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
				internal->max = Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
				internal->center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				internal->root = nullptr;
				internal->allocatedCount = 0;
				internal->allocated = nullptr;
				internal->nextAllocationIndex = nullptr;
			}
		}

		~Octree()
		{
			TerminateOctree();
		
			cudaFree(internal);
		}

		void Initialize(int maxDepth, const Eigen::Vector3f& min, const Eigen::Vector3f& max, size_t maxNumberOfOctants)
		{
			internal->maxDepth = maxDepth;
			internal->min = min;
			internal->max = max;
			internal->center = (min + max) * 0.5f;
			internal->root = nullptr;  // Initially no nodes

			internal->allocatedCount = maxNumberOfOctants;

			cudaError_t err;

			err = cudaMallocManaged(&internal->allocated, sizeof(Octant<T>) * internal->allocatedCount);
			if (err != cudaSuccess)
			{
				printf("Error allocating internal->allocated: %s\n", cudaGetErrorString(err));
				return;
			}

			err = cudaMallocManaged(&internal->nextAllocationIndex, sizeof(size_t));
			if (err != cudaSuccess)
			{
				printf("Error allocating internal->nextAllocationIndex: %s\n", cudaGetErrorString(err));
				cudaFree(internal->allocated); // Clean up any allocated memory if further allocation fails
				return;
			}

			// Ensure root allocation if required
			if (internal->allocatedCount > 0)
			{
				internal->root = &internal->allocated[0];  // Assign first octant as root
				internal->root->depth = 0;
				internal->root->lock = 0;
				internal->root->mortonCode = 0;
				for (int i = 0; i < 8; ++i)
				{
					internal->root->children[i] = nullptr;  // Initialize children as null
				}

				*(internal->nextAllocationIndex) = 1;
			}

			internal->initialized = true;
		}

		void TerminateOctree()
		{
			cudaDeviceSynchronize();  // Ensure all CUDA operations are complete before deallocating

			if (internal != nullptr)
			{
				if (internal->initialized)
				{
					if (internal->allocated != nullptr)
					{
						cudaError_t err = cudaFree(internal->allocated);
						if (err != cudaSuccess)
						{
							printf("Error freeing allocated memory: %s\n", cudaGetErrorString(err));
						}
						internal->allocated = nullptr;  // Avoid invalid future accesses
					}

					if (internal->nextAllocationIndex != nullptr)
					{
						cudaError_t err = cudaFree(internal->nextAllocationIndex);
						if (err != cudaSuccess)
						{
							printf("Error freeing allocation index: %s\n", cudaGetErrorString(err));
						}
						internal->nextAllocationIndex = nullptr;  // Avoid invalid future accesses
					}

					internal->initialized = false;
				}

				cudaError_t err = cudaFree(internal);
				if (err != cudaSuccess)
				{
					printf("Error freeing internal: %s\n", cudaGetErrorString(err));
				}
				internal = nullptr;  // Prevent further usage
			}
		}

		void Insert(const PatchBuffers& patchBuffers);
	};

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

	// Utility Functions for Morton Code
	uint32_t ExtractBitsFromMorton(uint64_t mortonCode, int startBit, int depth) {
		uint32_t value = 0;
		for (int i = 0; i < depth; ++i) {
			value |= ((mortonCode >> (3 * i + startBit)) & 1ULL) << i;
		}
		return value;
	}

	Eigen::Vector3f CalculatePositionFromMortonCode(uint64_t mortonCode, int depth, const Eigen::Vector3f& min, const Eigen::Vector3f& max) {
		uint32_t x = ExtractBitsFromMorton(mortonCode, 0, depth);
		uint32_t y = ExtractBitsFromMorton(mortonCode, 1, depth);
		uint32_t z = ExtractBitsFromMorton(mortonCode, 2, depth);

		uint32_t numSubdivisions = 1 << depth;
		Eigen::Vector3f voxelSize = (max - min) / numSubdivisions;

		return min + Eigen::Vector3f(x, y, z).cwiseProduct(voxelSize) + (voxelSize * 0.5f);
	}

	Eigen::Vector3f CalculateVoxelSizeFromMortonCode(uint64_t mortonCode, int depth, const Eigen::Vector3f& min, const Eigen::Vector3f& max)
	{
		Eigen::Vector3f span = max - min;
		float subdivisions = static_cast<float>(1 << depth); // 2^depth
		Eigen::Vector3f voxelSize = span / subdivisions;
		return voxelSize;
	}

	template<typename T>
	__global__ void Kernel_Insert(Octree<T>::Internal* octree, PatchBuffers patchBuffers) {
		unsigned int threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid >= patchBuffers.numberOfInputPoints) return;

		// Initialize cuRAND for each thread
		curandState state;
		curand_init(1234, threadid, 0, &state); // 1234 is the seed, threadid ensures unique seed per thread

		Eigen::Vector3f p = patchBuffers.inputPoints[threadid];

		// Generate small perturbation using curand_uniform to avoid coincident Morton codes
		float epsilon = 1e-3f;  // Increased perturbation
		p.x() += epsilon * curand_uniform(&state);
		p.y() += epsilon * curand_uniform(&state);
		p.z() += epsilon * curand_uniform(&state);

		// Get Morton code for the perturbed point
		auto mortonCode = octree->GetMortonCode(p);

		auto node = octree->AllocateOctantsUsingMortonCode(mortonCode);
		if (node != nullptr) {
			node->mortonCode = mortonCode;
		}
		else {
			printf("Failed to allocate node for point at Morton code %llu.\n", mortonCode);
		}
	}

	template<typename T>
	void Octree<T>::Insert(const PatchBuffers& patchBuffers)
	{
		unsigned int threadblocksize = 512;
		int gridsize = ((uint32_t)patchBuffers.numberOfInputPoints + threadblocksize - 1) / threadblocksize;

		Kernel_Insert<T> << <gridsize, threadblocksize >> > (internal, patchBuffers);
		cudaDeviceSynchronize(); // Ensure kernel execution is complete
	}

	void TestOctree()
	{
		auto t = Time::Now();

		Octree<Point> octree;
		//octree.Initialize(Eigen::Vector3f(-25.0f, -25.0f, -25.0f), Eigen::Vector3f(25.0f, 25.0f, 25.0f), 15000000);
		//octree.Initialize(Eigen::Vector3f(-50.0f, -50.0f, -50.0f), Eigen::Vector3f(50.0f, 50.0f, 50.0f), 150000000);
		//octree.Initialize(1, Eigen::Vector3f(-100.0f, -100.0f, -100.0f), Eigen::Vector3f(100.0f, 100.0f, 100.0f), 15000000);
		octree.Initialize(2, Eigen::Vector3f(-100.0f, -100.0f, -100.0f), Eigen::Vector3f(100.0f, 100.0f, 100.0f), 50);

		t = Time::End(t, "Octants allocation");

		//PatchBuffers patchBuffers;

//for (size_t i = 0; i < 2; i++)
////for (size_t i = 0; i < 4252; i++)
//{
//	t = Time::Now();

//	stringstream ss;
//	ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_" << i << ".ply";

//	PLYFormat ply;
//	ply.Deserialize(ss.str());

//	t = Time::End(t, "Load ply");

//	patchBuffers.FromPLYFile(ply);

//	t = Time::End(t, "Copy data to device");

//	nvtxRangePushA("Insert");

//	octree.Insert(patchBuffers);

//	nvtxRangePop();

//	t = Time::End(t, "Insert using PatchBuffers");
//}

/*
{
	t = Time::Now();

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

	nvtxRangePushA("Insert");

	octree.Insert(patchBuffers);

	nvtxRangePop();

	t = Time::End(t, "Insert using PatchBuffers");
}
*/

		{
			t = Time::Now();

			PatchBuffers patchBuffers(8, 1);
			patchBuffers.inputPoints[0] = Eigen::Vector3f(-50.0f, -50.0f, -50.0f);
			patchBuffers.inputPoints[1] = Eigen::Vector3f(50.0f, -50.0f, -50.0f);
			patchBuffers.inputPoints[2] = Eigen::Vector3f(-50.0f, 50.0f, -50.0f);
			patchBuffers.inputPoints[3] = Eigen::Vector3f(50.0f, 50.0f, -50.0f);
			patchBuffers.inputPoints[4] = Eigen::Vector3f(-50.0f, -50.0f, 50.0f);
			patchBuffers.inputPoints[5] = Eigen::Vector3f(50.0f, -50.0f, 50.0f);
			patchBuffers.inputPoints[6] = Eigen::Vector3f(-50.0f, 50.0f, 50.0f);
			patchBuffers.inputPoints[7] = Eigen::Vector3f(50.0f, 50.0f, 50.0f);

			patchBuffers.inputNormals[0] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
			patchBuffers.inputNormals[1] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
			patchBuffers.inputNormals[2] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
			patchBuffers.inputNormals[3] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
			patchBuffers.inputNormals[4] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
			patchBuffers.inputNormals[5] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
			patchBuffers.inputNormals[6] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
			patchBuffers.inputNormals[7] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);

			patchBuffers.inputColors[0] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
			patchBuffers.inputColors[1] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
			patchBuffers.inputColors[2] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
			patchBuffers.inputColors[3] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
			patchBuffers.inputColors[4] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
			patchBuffers.inputColors[5] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
			patchBuffers.inputColors[6] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
			patchBuffers.inputColors[7] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);

			t = Time::End(t, "Copy data to device");

			for (int i = 0; i < 8; ++i) {
				std::cout << "Host point[" << i << "]: " << patchBuffers.inputPoints[i].transpose() << std::endl;
				VD::AddSphere("center", patchBuffers.inputPoints[i], { 1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);
			}

			cudaDeviceSynchronize();

			nvtxRangePushA("Insert");

			octree.Insert(patchBuffers);

			nvtxRangePop();

			t = Time::End(t, "Insert using PatchBuffers");
		}

		t = Time::Now();
		
		printf("GetAllLeafPositions begins\n");

		auto root = octree.internal->root;
		printf("root : %d\n", root->mortonCode);

		auto current = root;
		stack<Octant<Point>*> octantStack;
		octantStack.push(root);

		while (false == octantStack.empty())
		{
			current = octantStack.top();
			octantStack.pop();

			auto p = CalculatePositionFromMortonCode(current->mortonCode, current->depth, octree.internal->min, octree.internal->max);
			auto voxelSize = CalculateVoxelSizeFromMortonCode(current->mortonCode, current->depth, octree.internal->min, octree.internal->max);
			cout << "Depth : " << current->depth << endl;
			//cout << "Morton Code : " << current->mortonCode << endl;
			printBinary(current->mortonCode);
			cout << "Position : " << p.transpose() << endl;
			cout << "VoxelSize : " << voxelSize.transpose() << endl;

			if (0 < current->depth)
			{
				stringstream ss;
				ss << "cube_" << current->depth;
				VD::AddCube(ss.str(), p, voxelSize * 0.5f, { 0.0f, 0.0f, 1.0f }, Color4::White);
			}

			printf("----------------------------------------------------------------------------------------------------\n");

			for (int i = 0; i < 8; i++)
			{
				if (nullptr != current->children[i])
				{
					octantStack.push(current->children[i]);
				}
			}
		}

		//auto nov = *(octree.internal->nextAllocationIndex);
		//for (size_t i = 0; i < nov; i++)
		//{
		//	auto& octant = octree.internal->allocated[i];

		//	bool isLeaf = true;
		//	for (size_t i = 0; i < 8; i++)
		//	{
		//		if (nullptr != octant.children[i])
		//		{
		//			isLeaf = false;
		//			break;
		//		}
		//	}

		//	//if (isLeaf)
		//	{
		//		if (0 != octant.mortonCode)
		//		{
		//			printBinary(octant.mortonCode);

		//			auto p = CalculatePositionFromMortonCode(octant.mortonCode, octree.internal->maxDepth, octree.internal->min, octree.internal->max);
		//			cout << p.transpose() << endl;
		//			auto voxelSize = CalculateVoxelSizeFromMortonCode(octant.mortonCode, octree.internal->maxDepth, octree.internal->min, octree.internal->max);
		//			//printf("voxelSize : %f, %f, %f\n", voxelSize.x(), voxelSize.y(), voxelSize.z());
		//			VD::AddCube("cube", p, voxelSize * 0.5f, { 0.0f, 0.0f, 1.0f }, Color4::White);
		//		}
		//	}
		//}

		//VD::AddCube("cube", { 0.0f, 0.0f, 0.0f }, { 50.0f, 50.0f, 50.0f }, { 0.0f, 0.0f, 1.0f }, Color4::Red);
	}
}
