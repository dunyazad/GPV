#include "RegularGrid.cuh"

#include <App/Serialization.hpp>
#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
	template<typename T> class RegularGrid;

	class PatchBuffers
	{
	public:
		PatchBuffers(int width = 256, int height = 480)
			: width(width), height(height)
		{
			checkCudaErrors(cudaMallocManaged(&inputPoints, sizeof(float3) * width * height));
			checkCudaErrors(cudaMallocManaged(&inputNormals, sizeof(float3) * width * height));
			checkCudaErrors(cudaMallocManaged(&inputColors, sizeof(float3) * width * height));
		}

		~PatchBuffers()
		{
			checkCudaErrors(cudaFree(inputPoints));
			checkCudaErrors(cudaFree(inputNormals));
			checkCudaErrors(cudaFree(inputColors));
		}

		int width;
		int height;
		size_t numberOfInputPoints;
		float3* inputPoints;
		float3* inputNormals;
		float3* inputColors;

		void Clear()
		{
			numberOfInputPoints = 0;
			checkCudaErrors(cudaMemset(inputPoints, 0, sizeof(float3) * width * height));
			checkCudaErrors(cudaMemset(inputNormals, 0, sizeof(float3) * width * height));
			checkCudaErrors(cudaMemset(inputColors, 0, sizeof(float3) * width * height));
		}

		void FromPLYFile(const PLYFormat& ply)
		{
			Clear();

			numberOfInputPoints = ply.GetPoints().size() / 3;
			checkCudaErrors(cudaMemcpy(inputPoints, ply.GetPoints().data(), sizeof(float3) * numberOfInputPoints, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(inputNormals, ply.GetNormals().data(), sizeof(float3) * numberOfInputPoints, cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(inputColors, ply.GetColors().data(), sizeof(float3) * numberOfInputPoints, cudaMemcpyHostToDevice));

			//for (size_t i = 0; i < numberOfInputPoints; i++)
			//{
			//	auto p = inputPoints[i];
			//	auto n = inputNormals[i];
			//	auto c = inputColors[i];
			//	Color4 c4;
			//	c4.FromNormalized(c.x, c.y, c.z, 1.0f);
			//    VD::AddSphere("points", { p.x, p.y, p.z }, { 0.05f, 0.05f, 0.05f }, { n.x, n.y, n.z }, c4);
			//}
		}
	};

	struct Voxel
	{
		float tsdfValue;
		float weight;
		float3 normal;
		float3 color;
	};

	template<typename T>
	__global__ void Kernel_Clear(RegularGrid<T>::Internal* regularGrid);

	template<typename T>
	__global__ void Kernel_Integrate(RegularGrid<T>::Internal* regularGrid, PatchBuffers);

	__host__ __device__
		uint3 GetIndex(const float3& gridCenter, uint3 gridDimensions, float voxelSize, const float3& position);

	__host__ __device__
		float3 GetPosition(const float3& gridCenter, uint3 gridDimensions, float voxelSize, const uint3& index);

	template<typename T>
	class RegularGrid
	{
	public:
		struct Internal
		{
			T* elements;
			float3 center;
			uint3 dimensions;
			float voxelSize;
			size_t numberOfVoxels;
			float truncationDistance;
		};

		RegularGrid(const float3& center, uint32_t dimensionX, uint32_t dimensionY, uint32_t dimensionZ, float voxelSize)
		{
			checkCudaErrors(cudaMallocManaged(&internal, sizeof(Internal)));

			internal->center = center;
			internal->elements = nullptr;
			internal->dimensions = make_uint3(dimensionX, dimensionY, dimensionZ);
			internal->voxelSize = voxelSize;
			internal->numberOfVoxels = dimensionX * dimensionY * dimensionZ;
			internal->truncationDistance = 1.0f;

			checkCudaErrors(cudaMallocManaged(&(internal->elements), sizeof(T) * internal->numberOfVoxels));
		}

		~RegularGrid()
		{
			checkCudaErrors(cudaFree(internal->elements));
			checkCudaErrors(cudaFree(internal));
		}

		void Clear()
		{
			nvtxRangePushA("Clear");

			dim3 threadsPerBlock(8, 8, 8);  // 8x8x8 threads per block
			dim3 blocksPerGrid(
				(internal->dimensions.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(internal->dimensions.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
				(internal->dimensions.z + threadsPerBlock.z - 1) / threadsPerBlock.z
			);

			Kernel_Clear<T> << <blocksPerGrid, threadsPerBlock >> > (internal);

			checkCudaErrors(cudaDeviceSynchronize());

			nvtxRangePop();
		}

		void Integrate(const PatchBuffers& patchBuffers)
		{
			nvtxRangePushA("Insert");

			unsigned int threadblocksize = 512;
			int gridsize = ((uint32_t)patchBuffers.numberOfInputPoints + threadblocksize - 1) / threadblocksize;

			Kernel_Integrate<T> << <gridsize, threadblocksize >> > (internal, patchBuffers);

			checkCudaErrors(cudaDeviceSynchronize());

			nvtxRangePop();
		}

		Internal* internal;
	};

	//---------------------------------------------------------------------------------------------------
	// Kernel Functions
	//---------------------------------------------------------------------------------------------------

	template<typename T>
	__global__ void Kernel_Clear(RegularGrid<T>::Internal* regularGrid)
	{
		size_t threadX = blockIdx.x * blockDim.x + threadIdx.x;
		size_t threadY = blockIdx.y * blockDim.y + threadIdx.y;
		size_t threadZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (threadX >= regularGrid->dimensions.x ||
			threadY >= regularGrid->dimensions.y ||
			threadZ >= regularGrid->dimensions.z) return;

		size_t flatIndex = threadZ * (regularGrid->dimensions.x * regularGrid->dimensions.y) +
			threadY * regularGrid->dimensions.x + threadX;

		regularGrid->elements[flatIndex].tsdfValue = 1.0f;
		regularGrid->elements[flatIndex].weight = 0.0f;
		regularGrid->elements[flatIndex].normal = make_float3(0.0f, 0.0f, 0.0f);
		regularGrid->elements[flatIndex].color = make_float3(0.5f, 0.5f, 0.5f);
	}

	template<typename T>
	__global__ void Kernel_Integrate(RegularGrid<T>::Internal* regularGrid, PatchBuffers patchBuffers)
	{
		unsigned int threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid >= patchBuffers.numberOfInputPoints) return;

		float3 p = patchBuffers.inputPoints[threadid];
		float3 color = patchBuffers.inputColors[threadid];
		float3 normal = patchBuffers.inputNormals[threadid];

		auto gridIndex = GetIndex(regularGrid->center, regularGrid->dimensions, regularGrid->voxelSize, p);

		if (gridIndex.x == UINT_MAX || gridIndex.y == UINT_MAX || gridIndex.z == UINT_MAX) return;

		size_t flatIndex = gridIndex.z * (regularGrid->dimensions.x * regularGrid->dimensions.y) +
			gridIndex.y * regularGrid->dimensions.x + gridIndex.x;

		float3 voxelCenter = GetPosition(regularGrid->center, regularGrid->dimensions, regularGrid->voxelSize, gridIndex);

		float distance = length(p - voxelCenter);
		float signedDistance = distance;

		float tsdfValue = signedDistance / regularGrid->truncationDistance;
		tsdfValue = fminf(1.0f, fmaxf(-1.0f, tsdfValue));

		Voxel* voxel = &(regularGrid->elements[flatIndex]);

		float newWeight = 1.0f;

		float previousTsdf = voxel->tsdfValue;
		float previousWeight = voxel->weight;

		// Weighted average for TSDF
		float updatedTsdf = (previousTsdf * previousWeight + tsdfValue * newWeight) / (previousWeight + newWeight);
		float updatedWeight = previousWeight + newWeight;

		voxel->tsdfValue = updatedTsdf;
		voxel->weight = updatedWeight;

		voxel->color = make_float3(
			fminf(1.0f, fmaxf(0.0f, (voxel->color.x * previousWeight + color.x * newWeight) / updatedWeight)),
			fminf(1.0f, fmaxf(0.0f, (voxel->color.y * previousWeight + color.y * newWeight) / updatedWeight)),
			fminf(1.0f, fmaxf(0.0f, (voxel->color.z * previousWeight + color.z * newWeight) / updatedWeight))
		);

		voxel->normal = make_float3(
			(voxel->normal.x * previousWeight + normal.x * newWeight) / updatedWeight,
			(voxel->normal.y * previousWeight + normal.y * newWeight) / updatedWeight,
			(voxel->normal.z * previousWeight + normal.z * newWeight) / updatedWeight
		);

		voxel->normal = normalize(voxel->normal);

		//printf("voxel->tsdfValue : %f\n", voxel->tsdfValue);

		//if (-0.05f <= voxel->tsdfValue && voxel->tsdfValue <= 0.05f)
		//if (-0.2f <= voxel->tsdfValue && voxel->tsdfValue <= 0.2f)
		//{
		//	printf("%f, %f, %f\n", p.x, p.y, p.z);
		//}
	}

	//---------------------------------------------------------------------------------------------------
	// Utility Functions
	//---------------------------------------------------------------------------------------------------

	__host__ __device__
		uint3 GetIndex(const float3& gridCenter, uint3 gridDimensions, float voxelSize, const float3& position)
	{
		float3 halfGridSize = make_float3(
			(float)gridDimensions.x * voxelSize * 0.5f,
			(float)gridDimensions.y * voxelSize * 0.5f,
			(float)gridDimensions.z * voxelSize * 0.5f
		);

		float3 gridMin = gridCenter - halfGridSize;
		float3 relativePosition = position - gridMin;

		uint3 index = make_uint3(UINT_MAX, UINT_MAX, UINT_MAX);

		if (relativePosition.x < 0.0f || relativePosition.x >= (float)gridDimensions.x * voxelSize ||
			relativePosition.y < 0.0f || relativePosition.y >= (float)gridDimensions.y * voxelSize ||
			relativePosition.z < 0.0f || relativePosition.z >= (float)gridDimensions.z * voxelSize)
		{
			return index;
		}
		else
		{
			index.x = (uint32_t)floorf(relativePosition.x / voxelSize);
			index.y = (uint32_t)floorf(relativePosition.y / voxelSize);
			index.z = (uint32_t)floorf(relativePosition.z / voxelSize);
		}

		return index;
	}

	__host__ __device__
		float3 GetPosition(const float3& gridCenter, uint3 gridDimensions, float voxelSize, const uint3& index)
	{
		float3 halfGridSize = make_float3(
			(float)gridDimensions.x * voxelSize * 0.5f,
			(float)gridDimensions.y * voxelSize * 0.5f,
			(float)gridDimensions.z * voxelSize * 0.5f
		);

		float3 gridMin = gridCenter - halfGridSize;

		// Calculate the position of the given voxel using the provided index
		float3 position = make_float3(
			gridMin.x + (float)index.x * voxelSize + voxelSize * 0.5f,
			gridMin.y + (float)index.y * voxelSize + voxelSize * 0.5f,
			gridMin.z + (float)index.z * voxelSize + voxelSize * 0.5f
		);

		return position;
	}

	////////////////////////////////////////////////////////////////////////////////////

	void TestRegularGrid()
	{
		auto t = Time::Now();

		RegularGrid<Voxel> rg({ 0.0f, 0.0f, 0.0f }, 100, 100, 100, 0.1f);

		t = Time::End(t, "RegularGrid allocation");

		rg.Clear();

		t = Time::End(t, "RegularGrid Clear");

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

			rg.Integrate(patchBuffers);

			nvtxRangePop();

			t = Time::End(t, "Insert using PatchBuffers");
		}

		{
			t = Time::Now();

			for (uint32_t z = 0; z < rg.internal->dimensions.z; z++)
			{
				for (uint32_t y = 0; y < rg.internal->dimensions.y; y++)
				{
					for (uint32_t x = 0; x < rg.internal->dimensions.x; x++)
					{
						size_t flatIndex = z * (rg.internal->dimensions.x * rg.internal->dimensions.y) +
							y * rg.internal->dimensions.x + x;

						//printf("flatIndex : %llu\n", flatIndex);

						auto voxel = rg.internal->elements[flatIndex];

						if (-0.1f <= voxel.tsdfValue && voxel.tsdfValue <= 0.1f)
						//if (-0.2f <= voxel.tsdfValue && voxel.tsdfValue <= 0.2f)
						{
							auto p = GetPosition(rg.internal->center, rg.internal->dimensions, rg.internal->voxelSize, make_uint3(x, y, z));
							auto n = voxel.normal;
							//n = make_float3(0.0f, 0.0f, 1.0f);
							auto c = voxel.color;
							Color4 c4;
							c4.FromNormalized(c.x, c.y, c.z, 1.0f);
							VD::AddCube("voxels", { p.x, p.y, p.z },
								{ rg.internal->voxelSize * 0.5f, rg.internal->voxelSize * 0.5f,rg.internal->voxelSize * 0.5f },
								{ n.x, n.y, n.z }, c4);
						}
					}
				}
			}
		}
	}
}