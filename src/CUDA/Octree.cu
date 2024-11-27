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
			cudaMemset(&inputPoints, 0, sizeof(Eigen::Vector3f) * width * height);
			cudaMemset(&inputNormals, 0, sizeof(Eigen::Vector3f) * width * height);
			cudaMemset(&inputColors, 0, sizeof(Eigen::Vector3f) * width * height);
		}

		void FromPLYFile(const PLYFormat& ply)
		{
			Clear();

			numberOfInputPoints = ply.GetPoints().size() / 3;
			cudaMemcpy(inputPoints, ply.GetPoints().data(), sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyHostToDevice);
			cudaMemcpy(inputNormals, ply.GetNormals().data(), sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyHostToDevice);
			cudaMemcpy(inputColors, ply.GetColors().data(), sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyHostToDevice);
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

			__device__ uint64_t GetMortonCode(const Eigen::Vector3f& position)
			{
				// Step 1: Normalize the position to be within the range [0, 1] relative to the bounds of the octree
				Eigen::Vector3f relativePos = (position - min).cwiseQuotient(max - min);

				// Clamp the relative position to the range [0, 1]
				relativePos.x() = fminf(fmaxf(relativePos.x(), 0.0f), 1.0f);
				relativePos.y() = fminf(fmaxf(relativePos.y(), 0.0f), 1.0f);
				relativePos.z() = fminf(fmaxf(relativePos.z(), 0.0f), 1.0f);

				// Step 2: Scale relative position to integer coordinates within the given depth
				// Convert [0, 1] to a value in [0, 2^depth - 1]
				uint32_t maxCoordinateValue = (1 << maxDepth) - 1;

				// Map the relative positions to integer grid
				uint32_t x = static_cast<uint32_t>(relativePos.x() * maxCoordinateValue);
				uint32_t y = static_cast<uint32_t>(relativePos.y() * maxCoordinateValue);
				uint32_t z = static_cast<uint32_t>(relativePos.z() * maxCoordinateValue);

				// Step 3: Interleave the bits of x, y, and z to generate the Morton code
				uint64_t mortonCode = 0;
				for (int i = 0; i < maxDepth; ++i)
				{
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
				return &allocated[index];
			}

			__device__
				Octant<T>* AllocateOctantsUsingMortonCode(size_t mortonCode)
			{
				// Start at the root octant
				Octant<T>* current = root;

				if (current == nullptr)
				{
					// If the root is null, allocate it first
					current = AllocateOctant();
					if (current == nullptr)
					{
						printf("Root allocation failed.\n");
						return nullptr;
					}

					// Attempt to lock the newly allocated root
					unsigned int expected = 0;
					if (atomicCAS(&(current->lock), expected, 1) != expected)
					{
						printf("Failed to acquire lock for the root.\n");
						return nullptr;
					}

					root = current;
					atomicExch(&(current->lock), 0); // Release the lock
				}

				// Traverse the octree using the Morton code, allocating octants if necessary
				for (int level = 0; level < sizeof(size_t) * 8 / 3; ++level)
				{
					// Extract the next 3 bits from the Morton code, which tell us which child to move to
					int childIndex = (mortonCode >> (3 * level)) & 0x7; // Get 3 bits for each level, range 0-7

					// If the current octant's child is null, allocate it
					if (current->children[childIndex] == nullptr)
					{
						// Lock the current octant to allocate a new child safely
						unsigned int expected = 0;
						if (atomicCAS(&(current->lock), expected, 1) != expected)
						{
							// If we cannot acquire the lock, it means another thread is modifying it
							// Retry the current level
							--level;  // Retry the current level
							continue;
						}

						// Double-check if the child is still null after acquiring the lock
						if (current->children[childIndex] == nullptr)
						{
							// Allocate a new child octant
							Octant<T>* newChild = AllocateOctant();
							if (newChild == nullptr)
							{
								// Allocation failed, print error and stop further allocations
								printf("Failed to allocate child octant at level %d.\n", level);
								atomicExch(&(current->lock), 0); // Release the lock
								return nullptr;
							}

							// Initialize the new child octant pointer
							current->children[childIndex] = newChild;
						}

						// Release the lock after child allocation
						atomicExch(&(current->lock), 0);
					}

					// Move to the child octant for the next iteration
					current = current->children[childIndex];
				}

				// Return the pointer to the leaf octant
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

			*(internal->nextAllocationIndex) = 0;

			// Ensure root allocation if required
			if (internal->allocatedCount > 0)
			{
				internal->root = &internal->allocated[0];  // Assign first octant as root
				internal->root->lock = 0;
				for (int i = 0; i < 8; ++i)
				{
					internal->root->children[i] = nullptr;  // Initialize children as null
				}
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

	Eigen::Vector3f CalculatePositionFromMortonCode(uint64_t mortonCode, int depth, const Eigen::Vector3f& min, const Eigen::Vector3f& max)
	{
		// Step 1: Extract the x, y, z coordinates from the Morton code
		uint32_t x = 0, y = 0, z = 0;

		for (int i = 0; i < depth; ++i)
		{
			x |= ((mortonCode >> (3 * i)) & 1ULL) << i;
			y |= ((mortonCode >> (3 * i + 1)) & 1ULL) << i;
			z |= ((mortonCode >> (3 * i + 2)) & 1ULL) << i;
		}

		// Step 2: Convert the extracted coordinates to normalized values in [0, 1]
		uint32_t maxCoordinateValue = (1 << depth) - 1;

		float fx = static_cast<float>(x) / static_cast<float>(maxCoordinateValue);
		float fy = static_cast<float>(y) / static_cast<float>(maxCoordinateValue);
		float fz = static_cast<float>(z) / static_cast<float>(maxCoordinateValue);

		// Step 3: Scale the normalized coordinates to the actual space defined by min and max
		Eigen::Vector3f relativePosition(fx, fy, fz);
		Eigen::Vector3f position = min + relativePosition.cwiseProduct(max - min);

		return position;
	}

	Eigen::Vector3f CalculateVoxelSizeFromMortonCode(uint64_t mortonCode, int depth, const Eigen::Vector3f& min, const Eigen::Vector3f& max)
	{
		// Calculate the full span of the octree bounding box
		Eigen::Vector3f span = max - min;

		// Calculate the number of subdivisions along each axis at the given depth
		// At each depth level, we split each axis by 2. Thus, there are 2^depth subdivisions per axis.
		float subdivisions = static_cast<float>(1 << depth);  // This computes 2^depth as a float

		// Calculate the size of the voxel in each dimension
		// The size of each voxel along a dimension is the span of that dimension divided by the number of subdivisions
		Eigen::Vector3f voxelSize = span / subdivisions;

		// Return the voxel size along x, y, and z
		return voxelSize;
	}


	template<typename T>
	__global__ void Kernel_Insert(Octree<T>::Internal* octree, PatchBuffers patchBuffers)
	{
		unsigned int threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid > patchBuffers.numberOfInputPoints - 1) return;

		//auto node = octree->AllocateOctant();
		//if (nullptr == node)
		//{
		//	printf("AllocatedOctant Failed.\n");
		//	return;
		//}

		auto& p = patchBuffers.inputPoints[threadid];
		auto mortonCode = octree->GetMortonCode(p);

		//printBinary(mortonCode);
		auto node = octree->AllocateOctantsUsingMortonCode(mortonCode);
		node->mortonCode = mortonCode;
	}

	template<typename T>
	void Octree<T>::Insert(const PatchBuffers& patchBuffers)
	{
		unsigned int threadblocksize = 512;
		int gridsize = ((uint32_t)patchBuffers.numberOfInputPoints + threadblocksize - 1) / threadblocksize;

		Kernel_Insert<T> << <gridsize, threadblocksize >> > (internal, patchBuffers);

		cudaDeviceSynchronize();
	}

	void TestOctree()
	{
		auto t = Time::Now();

		Octree<Point> octree;
		//octree.Initialize(Eigen::Vector3f(-25.0f, -25.0f, -25.0f), Eigen::Vector3f(25.0f, 25.0f, 25.0f), 15000000);
		//octree.Initialize(Eigen::Vector3f(-50.0f, -50.0f, -50.0f), Eigen::Vector3f(50.0f, 50.0f, 50.0f), 150000000);
		octree.Initialize(14, Eigen::Vector3f(-100.0f, -100.0f, -100.0f), Eigen::Vector3f(100.0f, 100.0f, 100.0f), 150000000);

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

		{
				t = Time::Now();

				stringstream ss;
				ss << "C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply";

				PLYFormat ply;
				ply.Deserialize(ss.str());

				t = Time::End(t, "Load ply");

				PatchBuffers patchBuffers(ply.GetPoints().size() / 3, 1);
				patchBuffers.FromPLYFile(ply);

				t = Time::End(t, "Copy data to device");

				nvtxRangePushA("Insert");

				octree.Insert(patchBuffers);

				nvtxRangePop();

				t = Time::End(t, "Insert using PatchBuffers");
		}

		t = Time::Now();

		printf("GetAllLeafPositions begins\n");

		auto nov = *(octree.internal->nextAllocationIndex);
		for (size_t i = 0; i < nov; i++)
		{
			auto& octant = octree.internal->allocated[i];

			bool isLeaf = true;
			for (size_t i = 0; i < 8; i++)
			{
				if (nullptr != octant.children[i])
				{
					isLeaf = false;
					break;
				}
			}

			if (isLeaf)
			{
				if (0 != octant.mortonCode)
				{
					//printBinary(octant.mortonCode);

					auto p = CalculatePositionFromMortonCode(octant.mortonCode, octree.internal->maxDepth, octree.internal->min, octree.internal->max);
					auto voxelSize = CalculateVoxelSizeFromMortonCode(octant.mortonCode, octree.internal->maxDepth, octree.internal->min, octree.internal->max);
					VD::AddCube("cube", p, voxelSize * 2.0f, { 0.0f, 0.0f, 1.0f }, Color4::White);
				}
			}
		}
	}
}
