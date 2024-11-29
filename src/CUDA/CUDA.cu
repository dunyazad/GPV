#include "CUDA.cuh"

namespace MarchingCubes
{
	typedef struct {
		Eigen::Vector3f p[8];
		float val[8];
		Eigen::Vector<unsigned char, 3> c[8];
	} GRIDCELL;

	typedef struct {
		Eigen::Vector3f p[3];
		Eigen::Vector<unsigned char, 3> c[3];
	} TRIANGLE;
}

namespace CUDA
{
	__global__ void Kernel_InitializeCache(Voxel* cache, int3 cacheSize)
	{
		int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
		int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

		int index = zIndex * cacheSize.z * cacheSize.y + yIndex * cacheSize.x + xIndex;
		if (index > cacheSize.x * cacheSize.y * cacheSize.z - 1) return;

		cache[index].tsdfValue = FLT_MAX;
		cache[index].weight = 0.0f;
		cache[index].normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		cache[index].color = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
	}

	__global__ void Kernel_Integrate(
		Voxel* cache,
		int3 cacheSize,
		float voxelSize,
		float tsdfThreshold,
		Eigen::Vector3f cacheMin,
		Eigen::Vector3f camPos,
		Eigen::Matrix4f transform,
		Eigen::Matrix4f inverseTransform,
		uint32_t numberOfPoints,
		Eigen::Vector3f* points,
		Eigen::Vector3f* normals,
		Eigen::Vector3f* colors)
	{
		uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid > numberOfPoints - 1) return;

		if (nullptr == cache) return;
		if (nullptr == points) return;

		auto& op = points[threadid];
		auto tp = transform * Eigen::Vector4f(op.x(), op.y(), op.z(), 1.0f);
		auto cp = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

		auto dp = op.z() - camPos.z();

		float offsetX = cp.x() - cacheMin.x();
		float offsetY = cp.y() - cacheMin.y();
		float offsetZ = cp.z() - cacheMin.z();

		int xIndex = (int)floorf(offsetX / voxelSize);
		int yIndex = (int)floorf(offsetY / voxelSize);
		int zIndex = (int)floorf(offsetZ / voxelSize);

		int currentIndex = zIndex * cacheSize.z * cacheSize.y + yIndex * cacheSize.x + xIndex;
		cache[currentIndex].weight = 1.0f;
		cache[currentIndex].tsdfValue = 0.0f;

		return;

		//if (xIndex < 0 || xIndex >= cacheSize.x || yIndex < 0 || yIndex >= cacheSize.y || zIndex < 0 || zIndex >= cacheSize.z)	return;

		int offset = (int)ceilf(tsdfThreshold / voxelSize);
		for (int iz = zIndex - offset; iz <= zIndex + offset; iz++)
		{
			if (iz < 0 || iz > cacheSize.z) continue;
			for (int iy = yIndex - offset; iy <= yIndex + offset; iy++)
			{
				if (iy < 0 || iy > cacheSize.y) continue;
				for (int ix = xIndex - offset; ix <= xIndex + offset; ix++)
				{
					if (ix < 0 || ix > cacheSize.x) continue;

					//int index = iz * cacheSize.z * cacheSize.y + iy * cacheSize.x + ix;
					//cache[index].weight = 1.0f;
					//cache[index].tsdfValue = 0.0f;

					//continue;

					auto vp4 = Eigen::Vector4f(
						cacheMin.x() + ix * voxelSize,
						cacheMin.y() + iy * voxelSize,
						cacheMin.z() + iz * voxelSize,
						1.0f);

					auto ivp4 = inverseTransform * vp4;
					auto dvp = ivp4.z() - camPos.z();

					auto distance = norm3df(ivp4.x() - op.x(), ivp4.y() - op.y(), ivp4.z() - op.z());

					if (tsdfThreshold > distance)
					{
						if (0 > dp - dvp) distance = -distance;

						//printf("distance : %f\n", distance);

						int index = iz * cacheSize.z * cacheSize.y + iy * cacheSize.x + ix;

						if (0.0f == cache[index].weight)
						{
							cache[index].tsdfValue = distance;
							cache[index].weight = 1.0f;
						}
						else
						{
							float newTsdfValue = distance;
							float tsdfValue = (cache[index].tsdfValue * cache[index].weight + newTsdfValue) / (cache[index].weight + 1);
							if (fabsf(tsdfValue) < fabsf(cache[index].tsdfValue))
							{
								cache[index].tsdfValue = tsdfValue;
								cache[index].weight += 1.0f;
							}
						}
					}
				}
			}
		}
	}

	__global__ void Kernel_Integrate_CurrentOnly(
		Voxel* cache,
		int3 cacheSize,
		float voxelSize,
		float tsdfThreshold,
		Eigen::Vector3f cacheMin,
		Eigen::Vector3f camPos,
		Eigen::Matrix4f transform,
		Eigen::Matrix4f inverseTransform,
		uint32_t numberOfPoints,
		Eigen::Vector3f* points,
		Eigen::Vector3f* normals,
		Eigen::Vector3f* colors)
	{
		uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid > numberOfPoints - 1) return;

		if (nullptr == cache) return;
		if (nullptr == points) return;

		auto& op = points[threadid];
		auto tp = transform * Eigen::Vector4f(op.x(), op.y(), op.z(), 1.0f);
		auto p = Eigen::Vector3f(tp.x(), tp.y(), tp.z());

		auto dp = p.z() - camPos.z();

		float offsetX = p.x() - cacheMin.x();
		float offsetY = p.y() - cacheMin.y();
		float offsetZ = p.z() - cacheMin.z();

		int xIndex = (int)floorf(offsetX / voxelSize);
		int yIndex = (int)floorf(offsetY / voxelSize);
		int zIndex = (int)floorf(offsetZ / voxelSize);

		int index = zIndex * cacheSize.z * cacheSize.y + yIndex * cacheSize.x + xIndex;
		if (0.0f == cache[index].weight)
		{
			cache[index].tsdfValue = dp;
			cache[index].weight = 1.0f;
		}
		else
		{
			float newTsdfValue = dp;
			float tsdfValue = (cache[index].tsdfValue * cache[index].weight + newTsdfValue) / (cache[index].weight + 1);
			if (fabsf(tsdfValue) < fabsf(cache[index].tsdfValue))
			{
				cache[index].tsdfValue = tsdfValue;
				cache[index].weight += 1.0f;
			}
		}
	}

	cuCache::cuCache(int xLength, int yLength, int zLength, float voxelSize)
		: voxelSize(voxelSize)
	{
		cacheSize = make_int3(xLength, yLength, zLength);
		cudaMallocManaged(&cache, sizeof(Voxel) * xLength * yLength * zLength);
		//cudaMalloc(&cache, sizeof(Voxel) * xLength * yLength * zLength);

		ClearCache();
	}

	cuCache::~cuCache()
	{
		cudaFree(cache);
	}

	void cuCache::ClearCache()
	{
		dim3 blockDim(8, 8, 8);
		dim3 gridDim(
			(cacheSize.x + blockDim.x - 1) / blockDim.x,
			(cacheSize.y + blockDim.y - 1) / blockDim.y,
			(cacheSize.z + blockDim.z - 1) / blockDim.z);

		Kernel_InitializeCache << < gridDim, blockDim >> > (cache, cacheSize);

		cudaDeviceSynchronize();
	}

	void cuCache::Integrate(
		const Eigen::Vector3f& cacheMin,
		const Eigen::Vector3f& camPos,
		const Eigen::Matrix4f& transform,
		uint32_t numberOfPoints,
		Eigen::Vector3f* points,
		Eigen::Vector3f* normals,
		Eigen::Vector3f* colors)
	{
		nvtxRangePushA("Integrate");

		this->cacheMin = cacheMin;

		int threadblocksize = 512;
		uint32_t gridsize = (numberOfPoints - 1) / threadblocksize;
		Kernel_Integrate << < gridsize, threadblocksize >> > (
			cache, cacheSize, voxelSize, tsdfThreshold, cacheMin,
			camPos, transform, transform.inverse(),
			numberOfPoints, points, normals, colors);

		cudaDeviceSynchronize();

		nvtxRangePop();
	}

	void cuCache::Serialize(Voxel* h_voxels)
	{
		cudaMemcpy(h_voxels, cache, sizeof(Voxel) * cacheSize.x * cacheSize.y * cacheSize.z, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}

	__device__ float3 getNormalFromCovariance(const float* covarianceMatrix) {
		// Extract elements of the 3x3 covariance matrix
		float Cxx = covarianceMatrix[0];
		float Cxy = covarianceMatrix[1];
		float Cxz = covarianceMatrix[2];
		float Cyy = covarianceMatrix[4];
		float Cyz = covarianceMatrix[5];
		float Czz = covarianceMatrix[8];

		// To approximate the smallest eigenvector, use a simple heuristic:
		// We calculate two vectors that roughly align with the largest two eigenvectors
		// and take the cross product to approximate the smallest eigenvector.
		float3 v1 = { Cxx, Cxy, Cxz };
		float3 v2 = { Cxy, Cyy, Cyz };

		// The normal vector is the cross product of v1 and v2
		float3 normal = cross(v1, v2);

		// Normalize the resulting vector
		return normalize(normal);
	}

	__global__ void Kernel_GeneratePatchNormals(int width, int height, float3* points, size_t numberOfPoints, float3* normals)
	{
		uint32_t threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid > width * height - 1) return;

		int xIndex = threadid % width;
		int yIndex = threadid / width;

		auto currentPoint = points[threadid];

		uint32_t found = 0;
		float3 mean = { 0.0f, 0.0f, 0.0f };

		float voxelSize = 0.1f;
		int offset = 5;
		int currentOffset = 0;
		while (currentOffset <= offset)
		{
			for (int y = yIndex - currentOffset; y <= yIndex + currentOffset; y++)
			{
				if (y < 0 || y > height) continue;

				for (int x = xIndex - currentOffset; x <= xIndex + currentOffset; x++)
				{
					if (x < 0 || x > width) continue;

					if ((x == xIndex - currentOffset || x == xIndex + currentOffset) ||
						(y == yIndex - currentOffset || y == yIndex + currentOffset))
					{
						auto npoint = points[y * width + x];

						auto distance = norm3d(
							npoint.x - currentPoint.x,
							npoint.y - currentPoint.y,
							npoint.z - currentPoint.z);

						if (distance <= (float)offset * voxelSize)
						{
							mean += npoint;
							found++;
						}
					}
				}
			}
			currentOffset++;
		}

		mean /= (float)found;

		currentOffset = 0;

		float covarianceMatrix[9];
		float Cxx = 0, Cxy = 0, Cxz = 0, Cyy = 0, Cyz = 0, Czz = 0;
		while (currentOffset <= offset)
		{
			for (int y = yIndex - currentOffset; y <= yIndex + currentOffset; y++)
			{
				if (y < 0 || y > height) continue;

				for (int x = xIndex - currentOffset; x <= xIndex + currentOffset; x++)
				{
					if (x < 0 || x > width) continue;

					if ((x == xIndex - currentOffset || x == xIndex + currentOffset) ||
						(y == yIndex - currentOffset || y == yIndex + currentOffset))
					{
						auto npoint = points[y * width + x];
						auto distance = norm3d(
							npoint.x - currentPoint.x,
							npoint.y - currentPoint.y,
							npoint.z - currentPoint.z);

						if (distance <= (float)offset * voxelSize)
						{
							Cxx += npoint.x * npoint.x;
							Cxy += npoint.x * npoint.y;
							Cxz += npoint.x * npoint.z;
							Cyy += npoint.y * npoint.y;
							Cyz += npoint.y * npoint.z;
							Czz += npoint.z * npoint.z;
						}
					}
				}
			}
			currentOffset++;
		}

		covarianceMatrix[0] = Cxx / (float)found;
		covarianceMatrix[1] = Cxy / (float)found;
		covarianceMatrix[2] = Cxz / (float)found;
		covarianceMatrix[3] = Cxy / (float)found;
		covarianceMatrix[4] = Cyy / (float)found;
		covarianceMatrix[5] = Cyz / (float)found;
		covarianceMatrix[6] = Cxz / (float)found;
		covarianceMatrix[7] = Cyz / (float)found;
		covarianceMatrix[8] = Czz / (float)found;

		auto normal = getNormalFromCovariance(covarianceMatrix);
		normals[threadid] = normal;
	}

	__global__ void Kernel_EverageNormals(int width, int height, float3* points, size_t numberOfPoints, float3* normals)
	{
		uint32_t threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid > width * height - 1) return;

		int xIndex = threadid % width;
		int yIndex = threadid / width;

		auto currentPoint = points[threadid];
		auto currentNormal = normals[threadid];

		uint32_t found = 1;
		float3 mean = normals[threadid];

		int offset = 5;
		int currentOffset = 0;
		while (currentOffset <= offset)
		{
			for (int y = yIndex - currentOffset; y <= yIndex + currentOffset; y++)
			{
				if (y < 0 || y > height) continue;

				for (int x = xIndex - currentOffset; x <= xIndex + currentOffset; x++)
				{
					if (x < 0 || x > width) continue;

					if ((x == xIndex - currentOffset || x == xIndex + currentOffset) ||
						(y == yIndex - currentOffset || y == yIndex + currentOffset))
					{
						auto npoint = points[y * width + x];
						auto nnormal = normals[y * width + x];

						auto distance = norm3d(
							npoint.x - currentPoint.x,
							npoint.y - currentPoint.y,
							npoint.z - currentPoint.z);

						if (distance <= 0.5f)
						{
							mean += nnormal;
							found++;
						}
					}
				}
			}
			currentOffset++;
		}

		mean /= (float)found;

		normals[threadid] = mean;
	}
	void GeneratePatchNormals(int width, int height, float3* points, size_t numberOfPoints, float3* normals)
	{
		{
			nvtxRangePushA("GeneratePatchNormals");

			int mingridsize;
			int threadblocksize;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_GeneratePatchNormals, 0, 0));
			auto gridsize = (numberOfPoints - 1) / threadblocksize;

			Kernel_GeneratePatchNormals << <gridsize, threadblocksize >> > (width, height, points, numberOfPoints, normals);

			checkCudaErrors(cudaDeviceSynchronize());

			nvtxRangePop();
		}

		//{
		//	nvtxRangePushA("EverageNormals");

		//	int mingridsize;
		//	int threadblocksize;
		//	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_EverageNormals, 0, 0));
		//	auto gridsize = (numberOfPoints - 1) / threadblocksize;

		//	Kernel_EverageNormals << <gridsize, threadblocksize >> > (width, height, points, numberOfPoints, normals);

		//	checkCudaErrors(cudaDeviceSynchronize());

		//	nvtxRangePop();
		//}
	}

	__global__ void Kernel_ClearVolume(Voxel* volume, uint3 volumeDimension)
	{
		uint32_t threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid > volumeDimension.x * volumeDimension.y * volumeDimension.z - 1) return;
		volume[threadid].tsdfValue = FLT_MAX;
		volume[threadid].weight = 0.0f;
		volume[threadid].normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		volume[threadid].color = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
	}

	void ClearVolume(Voxel* volume, uint3 volumeDimension)
	{
		nvtxRangePushA("ClearVolume");

		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_ClearVolume, 0, 0));
		auto gridsize = (volumeDimension.x * volumeDimension.y * volumeDimension.z - 1) / threadblocksize;

		Kernel_ClearVolume << <gridsize, threadblocksize >> > (
			volume, volumeDimension);

		checkCudaErrors(cudaDeviceSynchronize());

		nvtxRangePop();
	}

	__global__ void Kernel_IntegrateInputPoints(
		Voxel* volume,
		uint3 volumeDimension,
		float voxelSize,
		size_t numberOfInputPoints,
		Eigen::Vector3f* inputPoints,
		Eigen::Vector3f* inputNormals,
		Eigen::Vector3f* inputColors)
	{
		uint32_t threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid > numberOfInputPoints - 1) return;

		auto point = inputPoints[threadid];
		auto pointNormal = inputNormals[threadid];
		auto pointColor = inputColors[threadid];

		int xKey = (int)floorf(point.x() / voxelSize);
		int yKey = (int)floorf(point.y() / voxelSize);
		int zKey = (int)floorf(point.z() / voxelSize);

		if (xKey < 0 || xKey >= volumeDimension.x) return;
		if (yKey < 0 || yKey >= volumeDimension.y) return;
		if (zKey < 0 || zKey >= volumeDimension.z) return;

		//size_t index = zKey * volumeDimension.x * volumeDimension.y + yKey * volumeDimension.x + xKey;
		//volume[index].tsdfValue = 1.0f;

		int offset = 5;
		float weight = 1.0f;
		float truncationDistance = 1.0f;

		for (int z = zKey - offset; z <= zKey + offset; z++)
		{
			if (z < 0 || z >= volumeDimension.z) return;
			for (int y = yKey - offset; y <= yKey + offset; y++)
			{
				if (y < 0 || y >= volumeDimension.y) return;
				for (int x = xKey - offset; x <= xKey + offset; x++)
				{
					if (x < 0 || x >= volumeDimension.x) return;

					Eigen::Vector3f position((float)x * voxelSize, (float)y * voxelSize, (float)z * voxelSize);
					float distance = (position - point).norm();

					float tsdfValue = 0.0f;
					if (distance <= truncationDistance) {
						tsdfValue = distance / truncationDistance;
						if ((position - point).dot(point) < 0.0f) {
							tsdfValue = -tsdfValue;
						}

						size_t index = z * volumeDimension.x * volumeDimension.y + y * volumeDimension.x + x;

						auto& voxel = volume[index];

						float oldTSDF = voxel.tsdfValue;
						if (FLT_MAX == oldTSDF)
						{
							voxel.tsdfValue = tsdfValue;
							voxel.weight = 1.0f;
							voxel.normal = pointNormal;
							voxel.color = pointColor;
						}
						else
						{
							float oldWeight = voxel.weight;
							float newTSDF = (oldTSDF * oldWeight + tsdfValue * weight) / (oldWeight + weight);
							float newWeight = oldWeight + weight;
							if (fabsf(newTSDF) < fabsf(oldTSDF))
							{
								voxel.tsdfValue = newTSDF;
								voxel.weight = oldWeight + weight;

								Eigen::Vector3f oldNormal = voxel.normal;
								Eigen::Vector3f oldColor = voxel.color;
								voxel.normal = (oldNormal + pointNormal) * 0.5f;
								voxel.color = (oldColor + pointColor) * 0.5f;
							}
						}
					}
				}
			}
		}
	}

	void IntegrateInputPoints(
		Voxel* volume, uint3 volumeDimension, float voxelSize,
		size_t numberOfInputPoints, Eigen::Vector3f* inputPoints, Eigen::Vector3f* inputNormals, Eigen::Vector3f* inputColors)
	{
		nvtxRangePushA("IntegrateInputPoints");

		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_IntegrateInputPoints, 0, 0));
		auto gridsize = (numberOfInputPoints - 1) / threadblocksize;

		Kernel_IntegrateInputPoints << <gridsize, threadblocksize >> > (
			volume, volumeDimension, voxelSize,
			numberOfInputPoints, inputPoints, inputNormals, inputColors);

		checkCudaErrors(cudaDeviceSynchronize());

		nvtxRangePop();
	}

	__global__ void Kernel_GetNumberOfSurfaceVoxels(Voxel* volume, uint3 volumeDimension, float voxelSize, size_t* numberOfSurfaceVoxel)
	{
		uint32_t threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid > volumeDimension.x * volumeDimension.y * volumeDimension.z - 1) return;

		if (-voxelSize * 0.5f < volume[threadid].tsdfValue && volume[threadid].tsdfValue < voxelSize * 0.5f)
		{
			atomicAdd(numberOfSurfaceVoxel, 1);
		}
	}

	size_t GetNumberOfSurfaceVoxels(Voxel* volume, uint3 volumeDimension, float voxelSize)
	{
		nvtxRangePushA("GetNumberOfSurfaceVoxel");

		size_t* numberOfSurfaceVoxel = nullptr;
		cudaMallocManaged(&numberOfSurfaceVoxel, sizeof(size_t));

		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_GetNumberOfSurfaceVoxels, 0, 0));
		auto gridsize = (volumeDimension.x * volumeDimension.y * volumeDimension.z - 1) / threadblocksize;

		Kernel_GetNumberOfSurfaceVoxels << <gridsize, threadblocksize >> > (volume, volumeDimension, voxelSize, numberOfSurfaceVoxel);

		checkCudaErrors(cudaDeviceSynchronize());

		size_t result = *numberOfSurfaceVoxel;

		cudaFree(numberOfSurfaceVoxel);

		nvtxRangePop();

		return result;
	}

	__global__ void Kernel_ExtractSurfacePoints(
		Voxel* volume, uint3 volumeDimension, float voxelSize, Eigen::Vector3f volumeMin, Point* resultPoints, size_t* numberOfResultPoints)
	{
		uint32_t threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid > volumeDimension.x * volumeDimension.y * volumeDimension.z - 1) return;

		auto zIndex = threadid / (volumeDimension.x * volumeDimension.y);
		auto yIndex = (threadid % (volumeDimension.x * volumeDimension.y)) / volumeDimension.x;
		auto xIndex = (threadid % (volumeDimension.x * volumeDimension.y)) % volumeDimension.x;

		if (-voxelSize * 0.025f < volume[threadid].tsdfValue && volume[threadid].tsdfValue < voxelSize * 0.025f)
		{
			float x = volumeMin.x() + (float)xIndex * voxelSize;
			float y = volumeMin.y() + (float)yIndex * voxelSize;
			float z = volumeMin.z() + (float)zIndex * voxelSize;

			auto index = atomicAdd(numberOfResultPoints, 1);
			resultPoints[index].position = Eigen::Vector3f(x, y, z);
			resultPoints[index].normal = volume[threadid].normal;
			resultPoints[index].color = volume[threadid].color;
		}
	}

	//__global__ void Kernel_ExtractSurfacePoints(
	//	Voxel* volume, uint3 volumeDimension, float voxelSize, Eigen::Vector3f volumeMin, Point* resultPoints, size_t* numberOfResultPoints)
	//{
	//	uint32_t threadid = blockDim.x * blockIdx.x + threadIdx.x;
	//	if (threadid > volumeDimension.x * volumeDimension.y * volumeDimension.z - 1) return;

	//	auto zIndex = threadid / (volumeDimension.x * volumeDimension.y);
	//	auto yIndex = (threadid % (volumeDimension.x * volumeDimension.y)) / volumeDimension.x;
	//	auto xIndex = (threadid % (volumeDimension.x * volumeDimension.y)) % volumeDimension.x;
	//	auto& voxel = volume[threadid];
	//	if (FLT_MAX == voxel.tsdfValue) return;

	//	auto position = volumeMin + Eigen::Vector3f((float)xIndex * voxelSize, (float)yIndex * voxelSize, (float)zIndex * voxelSize);

	//	Eigen::Vector3f mp(FLT_MAX, FLT_MAX, FLT_MAX);
	//	int mc = 0;

	//	int offset = 1;
	//	for (int z = zIndex - offset; z <= zIndex + offset; z++)
	//	{
	//		if (z < 0 || z > volumeDimension.z - 1) continue;
	//		for (int y = yIndex - offset; y <= yIndex + offset; y++)
	//		{
	//			if (y < 0 || y > volumeDimension.y - 1) continue;
	//			for (int x = xIndex - offset; x <= xIndex + offset; x++)
	//			{
	//				if (x < 0 || x > volumeDimension.x - 1) continue;

	//				if (x == xIndex && y == yIndex && z == zIndex) continue;

	//				auto neighborVoxelIndex = z * volumeDimension.x * volumeDimension.y + y * volumeDimension.x + x;
	//				auto& neighborVoxel = volume[neighborVoxelIndex];

	//				if (FLT_MAX == neighborVoxel.tsdfValue) continue;

	//				if (0 > voxel.tsdfValue * neighborVoxel.tsdfValue)
	//				{
	//					auto neighborPosition = volumeMin + Eigen::Vector3f((float)x * voxelSize, (float)y * voxelSize, (float)z * voxelSize);

	//					if (0.0f == neighborVoxel.tsdfValue - voxel.tsdfValue)
	//					{

	//					}
	//					else
	//					{
	//						float ratio = voxel.tsdfValue / (neighborVoxel.tsdfValue - voxel.tsdfValue);
	//						Eigen::Vector3f ip = position * ratio + neighborPosition * (1 - ratio);

	//						if (0 == mc)
	//						{
	//							mp = ip;
	//							mc = 1;
	//						}
	//						else
	//						{
	//							mp += ip;
	//							mc++;
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}

	//	if (0 < mc)
	//	{
	//		auto index = atomicAdd(numberOfResultPoints, 1);
	//		resultPoints[index].position = mp / (float)mc;
	//		//printf("index : %llu\n", index);
	//		//resultPoints[index].position = position;
	//		resultPoints[index].normal = volume[threadid].normal;
	//		resultPoints[index].color = volume[threadid].color;
	//	}
	//}

	void ExtractSurfacePoints(
		Voxel* volume, uint3 volumeDimension, float voxelSize, Eigen::Vector3f volumeMin, Point* resultPoints, size_t* numberOfResultPoints)
	{
		nvtxRangePushA("ExtractSurfacePoints");

		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_ExtractSurfacePoints, 0, 0));
		auto gridsize = (volumeDimension.x * volumeDimension.y * volumeDimension.z - 1) / threadblocksize;

		Kernel_ExtractSurfacePoints << <gridsize, threadblocksize >> > (
			volume, volumeDimension, voxelSize, volumeMin, resultPoints, numberOfResultPoints);

		checkCudaErrors(cudaDeviceSynchronize());

		nvtxRangePop();
	}

#if 0
	__global__ void Kernel_PopulateExtractionVoxels(
		Voxel* volume, uint3 volumeDimension, float voxelSiz,
		Eigen::Vector3f& volumeMin,
		ExtractionVoxel* extractionVoxels, unsigned int* numberOfExtractionVoxels, ExtractionEdge* extractionEdges, unsigned int* numberOfExtractionEdges)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid > volumeDimension.x * volumeDimension.y * volumeDimension.z - 1) return;

		if (FLT_MAX == volume[threadid].tsdfValue) return;
		
		extractionVoxels[threadid].value = volume[threadid].tsdfValue;

		auto globalIndexX = (key >> 32) & 0xffff;
		auto globalIndexY = (key >> 16) & 0xffff;
		auto globalIndexZ = (key) & 0xffff;

		extractionVoxels[i].globalIndexX = globalIndexX;
		extractionVoxels[i].globalIndexY = globalIndexY;
		extractionVoxels[i].globalIndexZ = globalIndexZ;

		auto x = (float)globalIndexX * exeInfo.global.voxelSize
			+ exeInfo.global.voxelSize * 0.5f
			- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountX * 0.5f;
		auto y = (float)globalIndexY * exeInfo.global.voxelSize
			+ exeInfo.global.voxelSize * 0.5f
			- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountY * 0.5f;
		auto z = (float)globalIndexZ * exeInfo.global.voxelSize
			+ exeInfo.global.voxelSize * 0.5f
			- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountZ * 0.5f;

		extractionVoxels[i].position = Eigen::Vector3f(x, y, z);

		extractionVoxels[i].edgeIndexX = i * 3;
		extractionVoxels[i].edgeIndexY = i * 3 + 1;
		extractionVoxels[i].edgeIndexZ = i * 3 + 2;

		extractionVoxels[i].numberOfTriangles = 0;

		extractionVoxels[i].triangles[0].edgeIndices[0] = UINT32_MAX;
		extractionVoxels[i].triangles[0].edgeIndices[1] = UINT32_MAX;
		extractionVoxels[i].triangles[0].edgeIndices[2] = UINT32_MAX;

		extractionVoxels[i].triangles[1].edgeIndices[0] = UINT32_MAX;
		extractionVoxels[i].triangles[1].edgeIndices[1] = UINT32_MAX;
		extractionVoxels[i].triangles[1].edgeIndices[2] = UINT32_MAX;

		extractionVoxels[i].triangles[2].edgeIndices[0] = UINT32_MAX;
		extractionVoxels[i].triangles[2].edgeIndices[1] = UINT32_MAX;
		extractionVoxels[i].triangles[2].edgeIndices[2] = UINT32_MAX;

		extractionVoxels[i].triangles[3].edgeIndices[0] = UINT32_MAX;
		extractionVoxels[i].triangles[3].edgeIndices[1] = UINT32_MAX;
		extractionVoxels[i].triangles[3].edgeIndices[2] = UINT32_MAX;

		extractionEdges[i * 3 + 0].startVoxelIndex = i;
		extractionEdges[i * 3 + 0].edgeDirection = 0;
		extractionEdges[i * 3 + 0].zeroCrossing = false;
		extractionEdges[i * 3 + 0].zeroCrossingPointIndex = UINT32_MAX;
		extractionEdges[i * 3 + 1].startVoxelIndex = i;
		extractionEdges[i * 3 + 1].edgeDirection = 1;
		extractionEdges[i * 3 + 1].zeroCrossing = false;
		extractionEdges[i * 3 + 1].zeroCrossingPointIndex = UINT32_MAX;
		extractionEdges[i * 3 + 2].startVoxelIndex = i;
		extractionEdges[i * 3 + 2].edgeDirection = 2;
		extractionEdges[i * 3 + 2].zeroCrossing = false;
		extractionEdges[i * 3 + 2].zeroCrossingPointIndex = UINT32_MAX;

		// Next X
		{
			auto voxel_key = HASH_KEY_GEN_CUBE_64_(globalIndexX + 1, globalIndexY, globalIndexZ);
			uint32_t hashSlot_idx = get_hashtable_lookup_idx_func64_v4(globalHash_info, globalHash, globalHash_value, voxel_key);
			if (hashSlot_idx != kEmpty32)
			{
				extractionEdges[i * 3 + 0].endVoxelIndex = hashSlot_idx;
			}
			else
			{
				extractionEdges[i * 3 + 0].endVoxelIndex = UINT32_MAX;
			}
		}

		// Next Y
		{
			auto voxel_key = HASH_KEY_GEN_CUBE_64_(globalIndexX, globalIndexY + 1, globalIndexZ);
			uint32_t hashSlot_idx = get_hashtable_lookup_idx_func64_v4(globalHash_info, globalHash, globalHash_value, voxel_key);
			if (hashSlot_idx != kEmpty32)
			{
				extractionEdges[i * 3 + 1].endVoxelIndex = hashSlot_idx;
			}
			else
			{
				extractionEdges[i * 3 + 1].endVoxelIndex = UINT32_MAX;
			}
		}

		// Next Z
		{
			auto voxel_key = HASH_KEY_GEN_CUBE_64_(globalIndexX, globalIndexY, globalIndexZ + 1);
			uint32_t hashSlot_idx = get_hashtable_lookup_idx_func64_v4(globalHash_info, globalHash, globalHash_value, voxel_key);
			if (hashSlot_idx != kEmpty32)
			{
				extractionEdges[i * 3 + 2].endVoxelIndex = hashSlot_idx;
			}
			else
			{
				extractionEdges[i * 3 + 2].endVoxelIndex = UINT32_MAX;
			}
		}
	}

	__global__ void Kernel_ExtractVolume(
		Voxel* volume, uint3 volumeDimension, float voxelSiz,
		Eigen::Vector3f* resultPositions, Eigen::Vector3f* resultNormals, Eigen::Vector4f* resultColors,
		unsigned int* numberOfVoxelPositions, ExtractionVoxel* extractionVoxels, ExtractionEdge* extractionEdges)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid > volumeDimension.x * volumeDimension.y * volumeDimension.z - 1) return;

		auto& voxel = volume[threadid];
		if (FLT_MAX == voxel.tsdfValue) return;

		int zIndex = threadid / (volumeDimension.x * volumeDimension.y);
		int yIndex = (threadid % (volumeDimension.x * volumeDimension.y)) / volumeDimension.x;
		int xIndex = (threadid % (volumeDimension.x * volumeDimension.y)) % volumeDimension.x;

		auto key = globalHash[threadid];
		if ((kEmpty64 != key) && (kEmpty8 != globalHash_value[threadid]))
		{
			auto vv = voxelValues[threadid];
			if (VOXEL_INVALID != vv)
			{
				auto v = VV2D(vv);

				auto x_idx = (key >> 32) & 0xffff;
				auto y_idx = (key >> 16) & 0xffff;
				auto z_idx = (key) & 0xffff;

				if (x_idx < exeInfo.local.GetGlobalIndexX(0)) return;
				if (x_idx > exeInfo.local.GetGlobalIndexX(exeInfo.local.voxelCountX)) return;
				if (y_idx < exeInfo.local.GetGlobalIndexY(0)) return;
				if (y_idx > exeInfo.local.GetGlobalIndexY(exeInfo.local.voxelCountY)) return;
				if (z_idx < exeInfo.local.GetGlobalIndexZ(0)) return;
				if (z_idx > exeInfo.local.GetGlobalIndexZ(exeInfo.local.voxelCountZ)) return;

				auto x = (float)x_idx * exeInfo.global.voxelSize
					+ exeInfo.global.voxelSize * 0.5f
					- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountX * 0.5f;
				auto y = (float)y_idx * exeInfo.global.voxelSize
					+ exeInfo.global.voxelSize * 0.5f
					- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountY * 0.5f;
				auto z = (float)z_idx * exeInfo.global.voxelSize
					+ exeInfo.global.voxelSize * 0.5f
					- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountZ * 0.5f;

				auto nx = x + exeInfo.global.voxelSize;
				auto ny = y + exeInfo.global.voxelSize;
				auto nz = z + exeInfo.global.voxelSize;

				MarchingCubes::GRIDCELL gridcell;
				gridcell.p[0] = Eigen::Vector3f(x, y, z);
				gridcell.p[1] = Eigen::Vector3f(nx, y, z);
				gridcell.p[2] = Eigen::Vector3f(nx, y, nz);
				gridcell.p[3] = Eigen::Vector3f(x, y, nz);
				gridcell.p[4] = Eigen::Vector3f(x, ny, z);
				gridcell.p[5] = Eigen::Vector3f(nx, ny, z);
				gridcell.p[6] = Eigen::Vector3f(nx, ny, nz);
				gridcell.p[7] = Eigen::Vector3f(x, ny, nz);

				uint32_t  hs_voxel_hashIdx_core;

				for (size_t idx = 0; idx < 8; idx++)
				{


					//	mscho	@20240214
					//auto xGlobalIndex = (size_t)(floorf((gridcell.p[idx].x() - globalMinX) / voxelSize));
					//auto yGlobalIndex = (size_t)(floorf((gridcell.p[idx].y() - globalMinY) / voxelSize));
					//auto zGlobalIndex = (size_t)(floorf((gridcell.p[idx].z() - globalMinZ) / voxelSize));

					//printf("----- %llu %llu %llu\n", xGlobalIndex, yGlobalIndex, zGlobalIndex);

					auto voxel_ = 10.f;

					auto xGlobalIndex = (size_t)floorf((gridcell.p[idx].x()) * voxel_ + 2500.f);
					auto yGlobalIndex = (size_t)floorf((gridcell.p[idx].y()) * voxel_ + 2500.f);
					auto zGlobalIndex = (size_t)floorf((gridcell.p[idx].z()) * voxel_ + 2500.f);



					auto key = HASH_KEY_GEN_CUBE_64_(xGlobalIndex, yGlobalIndex, zGlobalIndex);
					auto i = get_hashtable_lookup_idx_func64_v4(globalHash_info, globalHash, globalHash_value, key);

					if (idx == 0)
					{
						hs_voxel_hashIdx_core = i;
						if (hs_voxel_hashIdx_core == kEmpty32)	return;
					}

					if (i != kEmpty32)
					{
						auto voxelValueCount = voxelValueCounts[i];

						//gridcell.val[idx] = VV2D(values[i]);
						gridcell.val[idx] = VV2D(voxelValues[i]) / (float)voxelValueCount;
						// 
						//printf("values[%d] : %d\tgridcell.val[%llu] : %f\n", i, values[i], idx, gridcell.val[idx]);
						//printf("GlobalIndex : %llu, %llu, %llu\tvalues[%d] : %d\t VV2D[%d] : %f\n", xGlobalIndex, yGlobalIndex, zGlobalIndex, i, values[i], values[i], VV2D(values[i]));
					}
					else
					{
						gridcell.val[idx] = FLT_MAX;
					}
				}

				auto oldIndex = atomicAdd(numberOfVoxelPositions, 1);
				resultPositions[oldIndex] = Eigen::Vector3f(x, y, z);

				resultNormals[oldIndex] = voxelNormals[threadid];

				auto r = (float)(voxelColors[threadid].x()) / 255.0f;
				auto g = (float)(voxelColors[threadid].y()) / 255.0f;
				auto b = (float)(voxelColors[threadid].z()) / 255.0f;
				auto a = v;
				if (a > 1.0f) a = 1.0f;
				if (a < -1.0f) a = -1.0f;
				a = a + 1.0f;
				a = (a / 2) * 255.0f;

				//r = a;
				//g = a;
				//b = a;

				if (v > 1.0f) v = 1.0f;
				if (v < -1.0f) v = -1.0f;

				r = r * (v + 1.0f) * 0.5f;
				g = 0.5 * g;
				b = b * (1.0f - (v + 1.0f) * 0.5f);
				a = 1.0f;

				resultColors[oldIndex] = Eigen::Vector4f(r, g, b, a);




				float isoValue = 0.0f;
				int cubeindex = 0;
				float isolevel = isoValue;
				Eigen::Vector3f vertlist[12];

				if (FLT_VALID(gridcell.val[0]) && gridcell.val[0] < isolevel) cubeindex |= 1;
				if (FLT_VALID(gridcell.val[1]) && gridcell.val[1] < isolevel) cubeindex |= 2;
				if (FLT_VALID(gridcell.val[2]) && gridcell.val[2] < isolevel) cubeindex |= 4;
				if (FLT_VALID(gridcell.val[3]) && gridcell.val[3] < isolevel) cubeindex |= 8;
				if (FLT_VALID(gridcell.val[4]) && gridcell.val[4] < isolevel) cubeindex |= 16;
				if (FLT_VALID(gridcell.val[5]) && gridcell.val[5] < isolevel) cubeindex |= 32;
				if (FLT_VALID(gridcell.val[6]) && gridcell.val[6] < isolevel) cubeindex |= 64;
				if (FLT_VALID(gridcell.val[7]) && gridcell.val[7] < isolevel) cubeindex |= 128;

				if (edgeTable[cubeindex] == 0)
				{
					return;
				}

				if (edgeTable[cubeindex] & 1)
					vertlist[0] =
					VertexInterp(isolevel, gridcell.p[0], gridcell.p[1], gridcell.val[0], gridcell.val[1]);
				if (edgeTable[cubeindex] & 2)
					vertlist[1] =
					VertexInterp(isolevel, gridcell.p[1], gridcell.p[2], gridcell.val[1], gridcell.val[2]);
				if (edgeTable[cubeindex] & 4)
					vertlist[2] =
					VertexInterp(isolevel, gridcell.p[2], gridcell.p[3], gridcell.val[2], gridcell.val[3]);
				if (edgeTable[cubeindex] & 8)
					vertlist[3] =
					VertexInterp(isolevel, gridcell.p[3], gridcell.p[0], gridcell.val[3], gridcell.val[0]);
				if (edgeTable[cubeindex] & 16)
					vertlist[4] =
					VertexInterp(isolevel, gridcell.p[4], gridcell.p[5], gridcell.val[4], gridcell.val[5]);
				if (edgeTable[cubeindex] & 32)
					vertlist[5] =
					VertexInterp(isolevel, gridcell.p[5], gridcell.p[6], gridcell.val[5], gridcell.val[6]);
				if (edgeTable[cubeindex] & 64)
					vertlist[6] =
					VertexInterp(isolevel, gridcell.p[6], gridcell.p[7], gridcell.val[6], gridcell.val[7]);
				if (edgeTable[cubeindex] & 128)
					vertlist[7] =
					VertexInterp(isolevel, gridcell.p[7], gridcell.p[4], gridcell.val[7], gridcell.val[4]);
				if (edgeTable[cubeindex] & 256)
					vertlist[8] =
					VertexInterp(isolevel, gridcell.p[0], gridcell.p[4], gridcell.val[0], gridcell.val[4]);
				if (edgeTable[cubeindex] & 512)
					vertlist[9] =
					VertexInterp(isolevel, gridcell.p[1], gridcell.p[5], gridcell.val[1], gridcell.val[5]);
				if (edgeTable[cubeindex] & 1024)
					vertlist[10] =
					VertexInterp(isolevel, gridcell.p[2], gridcell.p[6], gridcell.val[2], gridcell.val[6]);
				if (edgeTable[cubeindex] & 2048)
					vertlist[11] =
					VertexInterp(isolevel, gridcell.p[3], gridcell.p[7], gridcell.val[3], gridcell.val[7]);

				MarchingCubes::TRIANGLE tris[4];
				Eigen::Vector3f nm;
				int ntriang = 0;
				for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
					auto v0 = vertlist[triTable[cubeindex][i]];
					auto v1 = vertlist[triTable[cubeindex][i + 1]];
					auto v2 = vertlist[triTable[cubeindex][i + 2]];

					tris[ntriang].p[0] = v0;
					tris[ntriang].p[1] = v1;
					tris[ntriang].p[2] = v2;
					ntriang++;
				}

			}
		}
	}

	__global__ void Kernel_CalculateZeroCrossingPoints(
		MarchingCubes::ExecutionInfo exeInfo,
		HashKey64* globalHash_info, uint64_t* globalHash, uint8_t* globalHash_value,
		voxel_value_t* voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f* voxelNormals, Eigen::Vector<unsigned char, 3>* voxelColors,
		//Eigen::Vector3f* resultPositions, Eigen::Vector3f* resultNormals, Eigen::Vector4f* resultColors, unsigned int* numberOfVoxelPositions,
		ExtractionVoxel* extractionVoxels, unsigned int* numberOfExtractionVoxels, ExtractionEdge* extractionEdges, unsigned int* numberOfExtractionEdges,
		Eigen::Vector3f* zeroCrossingPositions, unsigned int* numberOfZeroCrossingPositions)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid > globalHash_info->HashTableCapacity * 3 - 1) return;

		auto& extractionEdge = extractionEdges[threadid];
		if (UINT32_MAX == extractionEdge.startVoxelIndex || UINT32_MAX == extractionEdge.endVoxelIndex) return;

		auto startVoxel = extractionVoxels[extractionEdge.startVoxelIndex];
		auto endVoxel = extractionVoxels[extractionEdge.endVoxelIndex];
		if ((FLT_MAX == startVoxel.value) || (FLT_MAX == endVoxel.value)) return;

		if ((startVoxel.value > 0 && endVoxel.value < 0) || (startVoxel.value < 0 && endVoxel.value > 0)) {
			extractionEdge.zeroCrossing = true;

			float ratio = startVoxel.value / (startVoxel.value - endVoxel.value);
			auto zeroCrossingPoint = startVoxel.position + ratio * (endVoxel.position - startVoxel.position);

			auto zeroCrossingIndex = atomicAdd(numberOfZeroCrossingPositions, 1);
			zeroCrossingPositions[zeroCrossingIndex] = zeroCrossingPoint;
			extractionEdge.zeroCrossingPointIndex = zeroCrossingIndex;
		}
	}

	__device__ ExtractionVoxel* GetVoxel(MarchingCubes::ExecutionInfo exeInfo,
		HashKey64* globalHash_info, uint64_t* globalHash, uint8_t* globalHash_value,
		voxel_value_t* voxelValues,
		ExtractionVoxel* extractionVoxels, uint32_t globalIndexX, uint32_t globalIndexY, uint32_t globalIndexZ)
	{
		auto key = HASH_KEY_GEN_CUBE_64_(globalIndexX, globalIndexY, globalIndexZ);
		auto i = get_hashtable_lookup_idx_func64_v4(globalHash_info, globalHash, globalHash_value, key);
		if (i != kEmpty32)
		{
			return extractionVoxels + i;
		}
		else
		{
			return nullptr;
		}
	}

	__device__ int calcCubeIndex(MarchingCubes::ExecutionInfo exeInfo,
		HashKey64* globalHash_info, uint64_t* globalHash, uint8_t* globalHash_value, voxel_value_t* voxelValues,
		ExtractionVoxel* extractionVoxels, ExtractionEdge* extractionEdges, size_t voxelIndex, ExtractionVoxel** voxels)
	{
		int cubeIndex = 0;
		float isolevel = 0.0f;

		auto currentVoxel = extractionVoxels + voxelIndex;
		voxels[0] = currentVoxel;
		voxels[1] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
			currentVoxel->globalIndexX + 1, currentVoxel->globalIndexY, currentVoxel->globalIndexZ);
		voxels[2] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
			currentVoxel->globalIndexX + 1, currentVoxel->globalIndexY, currentVoxel->globalIndexZ + 1);
		voxels[3] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
			currentVoxel->globalIndexX, currentVoxel->globalIndexY, currentVoxel->globalIndexZ + 1);
		voxels[4] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
			currentVoxel->globalIndexX, currentVoxel->globalIndexY + 1, currentVoxel->globalIndexZ);
		voxels[5] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
			currentVoxel->globalIndexX + 1, currentVoxel->globalIndexY + 1, currentVoxel->globalIndexZ);
		voxels[6] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
			currentVoxel->globalIndexX + 1, currentVoxel->globalIndexY + 1, currentVoxel->globalIndexZ + 1);
		voxels[7] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
			currentVoxel->globalIndexX, currentVoxel->globalIndexY + 1, currentVoxel->globalIndexZ + 1);

		if (nullptr != voxels[0]) if (FLT_VALID(voxels[0]->value)) if (voxels[0]->value < isolevel) cubeIndex |= 1;
		if (nullptr != voxels[1]) if (FLT_VALID(voxels[1]->value)) if (voxels[1]->value < isolevel) cubeIndex |= 2;
		if (nullptr != voxels[2]) if (FLT_VALID(voxels[2]->value)) if (voxels[2]->value < isolevel) cubeIndex |= 4;
		if (nullptr != voxels[3]) if (FLT_VALID(voxels[3]->value)) if (voxels[3]->value < isolevel) cubeIndex |= 8;
		if (nullptr != voxels[4]) if (FLT_VALID(voxels[4]->value)) if (voxels[4]->value < isolevel) cubeIndex |= 16;
		if (nullptr != voxels[5]) if (FLT_VALID(voxels[5]->value)) if (voxels[5]->value < isolevel) cubeIndex |= 32;
		if (nullptr != voxels[6]) if (FLT_VALID(voxels[6]->value)) if (voxels[6]->value < isolevel) cubeIndex |= 64;
		if (nullptr != voxels[7]) if (FLT_VALID(voxels[7]->value)) if (voxels[7]->value < isolevel) cubeIndex |= 128;

		return cubeIndex;
	}

	__global__ void Kernel_MarchingCubes(
		MarchingCubes::ExecutionInfo exeInfo,
		HashKey64* globalHash_info, uint64_t* globalHash, uint8_t* globalHash_value,
		voxel_value_t* voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f* voxelNormals, Eigen::Vector<unsigned char, 3>* voxelColors,
		//Eigen::Vector3f* resultPositions, Eigen::Vector3f* resultNormals, Eigen::Vector4f* resultColors, unsigned int* numberOfVoxelPositions,
		ExtractionVoxel* extractionVoxels, unsigned int* numberOfExtractionVoxels, ExtractionEdge* extractionEdges, unsigned int* numberOfExtractionEdges)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid > exeInfo.globalHashInfo->HashTableCapacity - 1) return;

		auto key = exeInfo.globalHash[threadid];
		auto i = get_hashtable_lookup_idx_func64_v4(exeInfo.globalHashInfo, exeInfo.globalHash, exeInfo.globalHashValue, key);
		if (i == kEmpty32) return;

		auto vv = voxelValues[i];
		if (VOXEL_INVALID == vv) return;

		float isolevel = 0.0f;
		ExtractionVoxel* voxels[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };

		int cubeIndex = calcCubeIndex(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues,
			extractionVoxels, extractionEdges, i, (ExtractionVoxel**)voxels);

		uint32_t edgelist[12]
			= { UINT32_MAX, UINT32_MAX, UINT32_MAX,
				UINT32_MAX, UINT32_MAX, UINT32_MAX,
				UINT32_MAX, UINT32_MAX, UINT32_MAX,
				UINT32_MAX, UINT32_MAX, UINT32_MAX };

		uint32_t vertlist[12]
			= { UINT32_MAX, UINT32_MAX, UINT32_MAX,
				UINT32_MAX, UINT32_MAX, UINT32_MAX,
				UINT32_MAX, UINT32_MAX, UINT32_MAX,
				UINT32_MAX, UINT32_MAX, UINT32_MAX };

		if (edgeTable[cubeIndex] == 0)
		{
			return;
		}

		if (edgeTable[cubeIndex] & 1)
		{
			if (nullptr != voxels[0])
			{
				auto edge = extractionEdges[voxels[0]->edgeIndexX];
				edgelist[0] = voxels[0]->edgeIndexX;
				vertlist[0] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 2)
		{
			if (nullptr != voxels[1])
			{
				auto edge = extractionEdges[voxels[1]->edgeIndexZ];
				edgelist[1] = voxels[1]->edgeIndexZ;
				vertlist[1] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 4)
		{
			if (nullptr != voxels[3])
			{
				auto edge = extractionEdges[voxels[3]->edgeIndexX];
				edgelist[2] = voxels[3]->edgeIndexX;
				vertlist[2] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 8)
		{
			if (nullptr != voxels[0])
			{
				auto edge = extractionEdges[voxels[0]->edgeIndexZ];
				edgelist[3] = voxels[0]->edgeIndexZ;
				vertlist[3] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 16)
		{
			if (nullptr != voxels[4])
			{
				auto edge = extractionEdges[voxels[4]->edgeIndexX];
				edgelist[4] = voxels[4]->edgeIndexX;
				vertlist[4] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 32)
		{
			if (nullptr != voxels[5])
			{
				auto edge = extractionEdges[voxels[5]->edgeIndexZ];
				edgelist[5] = voxels[5]->edgeIndexZ;
				vertlist[5] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 64)
		{
			if (nullptr != voxels[7])
			{
				auto edge = extractionEdges[voxels[7]->edgeIndexX];
				edgelist[6] = voxels[7]->edgeIndexX;
				vertlist[6] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 128)
		{
			if (nullptr != voxels[4])
			{
				auto edge = extractionEdges[voxels[4]->edgeIndexZ];
				edgelist[7] = voxels[4]->edgeIndexZ;
				vertlist[7] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 256)
		{
			if (nullptr != voxels[0])
			{
				auto edge = extractionEdges[voxels[0]->edgeIndexY];
				edgelist[8] = voxels[0]->edgeIndexY;
				vertlist[8] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 512)
		{
			if (nullptr != voxels[1])
			{
				auto edge = extractionEdges[voxels[1]->edgeIndexY];
				edgelist[9] = voxels[1]->edgeIndexY;
				vertlist[9] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 1024)
		{
			if (nullptr != voxels[2])
			{
				auto edge = extractionEdges[voxels[2]->edgeIndexY];
				edgelist[10] = voxels[2]->edgeIndexY;
				vertlist[10] = edge.zeroCrossingPointIndex;
			}
		}
		if (edgeTable[cubeIndex] & 2048)
		{
			if (nullptr != voxels[3])
			{
				auto edge = extractionEdges[voxels[3]->edgeIndexY];
				edgelist[11] = voxels[3]->edgeIndexY;
				vertlist[11] = edge.zeroCrossingPointIndex;
			}
		}

		for (int ti = 0; ti < 4; ti++) {
			auto ti0 = triTable[cubeIndex][ti * 3];
			auto ti1 = triTable[cubeIndex][ti * 3 + 1];
			auto ti2 = triTable[cubeIndex][ti * 3 + 2];

			if (-1 == ti0 || -1 == ti1 || -1 == ti2) break;

			auto ei0 = edgelist[ti0];
			auto ei1 = edgelist[ti1];
			auto ei2 = edgelist[ti2];

			auto vi0 = vertlist[ti0];
			auto vi1 = vertlist[ti1];
			auto vi2 = vertlist[ti2];

			if (UINT32_MAX == ei0 || UINT32_MAX == ei1 || UINT32_MAX == ei2) break;
			if (0 == ei0 || 0 == ei1 || 0 == ei2) break;
			if (ei0 == ei1 || ei0 == ei2 || ei1 == ei2) break;

			if (UINT32_MAX == vi0 || UINT32_MAX == vi1 || UINT32_MAX == vi2) break;
			if (0 == vi0 || 0 == vi1 || 0 == vi2) break;
			if (vi0 == vi1 || vi0 == vi2 || vi1 == vi2) break;

			voxels[0]->triangles[ti].edgeIndices[0] = ei0;
			voxels[0]->triangles[ti].edgeIndices[1] = ei1;
			voxels[0]->triangles[ti].edgeIndices[2] = ei2;

			voxels[0]->triangles[ti].vertexIndices[0] = vi0;
			voxels[0]->triangles[ti].vertexIndices[1] = vi1;
			voxels[0]->triangles[ti].vertexIndices[2] = vi2;

			if (extractionEdges[ei0].zeroCrossingPointIndex != vi0)
			{
				printf("vi0 != ei0");
			}
			if (extractionEdges[ei1].zeroCrossingPointIndex != vi1)
			{
				printf("vi1 != ei1");
			}
			if (extractionEdges[ei2].zeroCrossingPointIndex != vi2)
			{
				printf("vi2 != ei2");
			}

			voxels[0]->numberOfTriangles++;
		}
	}

	__global__ void Kernel_MarchingCubes_Verify(
		MarchingCubes::ExecutionInfo exeInfo,
		HashKey64* globalHash_info, uint64_t* globalHash, uint8_t* globalHash_value,
		voxel_value_t* voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f* voxelNormals, Eigen::Vector<unsigned char, 3>* voxelColors,
		//Eigen::Vector3f* resultPositions, Eigen::Vector3f* resultNormals, Eigen::Vector4f* resultColors, unsigned int* numberOfVoxelPositions,
		ExtractionVoxel* extractionVoxels, unsigned int* numberOfExtractionVoxels, ExtractionEdge* extractionEdges, unsigned int* numberOfExtractionEdges)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid > globalHash_info->HashTableCapacity) return;

		auto& voxel = extractionVoxels[threadid];
		for (int i = 0; i < voxel.numberOfTriangles; i++)
		{
			auto& triangle = voxel.triangles[i];

			auto ei0 = triangle.edgeIndices[0];
			auto ei1 = triangle.edgeIndices[1];
			auto ei2 = triangle.edgeIndices[2];

			auto e0 = extractionEdges[ei0];
			auto e1 = extractionEdges[ei1];
			auto e2 = extractionEdges[ei2];

			auto v0 = triangle.vertexIndices[0];
			auto v1 = triangle.vertexIndices[1];
			auto v2 = triangle.vertexIndices[2];

			if (e0.zeroCrossingPointIndex != v0)
			{
				printf("e0.zeroCrossingPointIndex != v0\n");
			}
			if (e1.zeroCrossingPointIndex != v1)
			{
				printf("e1.zeroCrossingPointIndex != v1\n");
			}
			if (e2.zeroCrossingPointIndex != v2)
			{
				printf("e2.zeroCrossingPointIndex != v2\n");
			}
		}
	}

	__global__ void Kernel_MeshSmooth(MarchingCubes::ExecutionInfo exeInfo,
		HashKey64* globalHash_info, uint64_t* globalHash, uint8_t* globalHash_value, voxel_value_t* voxelValues,
		ExtractionVoxel* extractionVoxels, unsigned int* numberOfExtractionVoxels,
		ExtractionEdge* extractionEdges, unsigned int* numberOfExtractionEdges,
		Eigen::Vector3f* zeroCrossingPositions, unsigned int* numberOfZeroCrossingPositions,
		Eigen::Vector3f* resultZeroCrossingPositions)
	{
		unsigned int edgeIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (edgeIndex > *numberOfExtractionEdges - 1) return;

		ExtractionEdge* edge = extractionEdges + edgeIndex;
		if (false == edge->zeroCrossing) return;

		if (UINT32_MAX == edge->zeroCrossingPointIndex) return;

		ExtractionVoxel* voxel = extractionVoxels + edge->startVoxelIndex;

		ExtractionVoxel* voxels[4] = { nullptr, nullptr, nullptr, nullptr };
		if (0 == edge->edgeDirection)
		{
			voxels[0] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
				voxel->globalIndexX, voxel->globalIndexY - 1, voxel->globalIndexZ - 1);

			voxels[1] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
				voxel->globalIndexX, voxel->globalIndexY - 1, voxel->globalIndexZ);

			voxels[2] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
				voxel->globalIndexX, voxel->globalIndexY, voxel->globalIndexZ - 1);

			voxels[3] = voxel;
		}
		else if (1 == edge->edgeDirection)
		{
			voxels[0] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
				voxel->globalIndexX - 1, voxel->globalIndexY, voxel->globalIndexZ - 1);

			voxels[1] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
				voxel->globalIndexX - 1, voxel->globalIndexY, voxel->globalIndexZ);

			voxels[2] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
				voxel->globalIndexX, voxel->globalIndexY, voxel->globalIndexZ - 1);

			voxels[3] = voxel;
		}
		else if (2 == edge->edgeDirection)
		{
			voxels[0] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
				voxel->globalIndexX - 1, voxel->globalIndexY - 1, voxel->globalIndexZ);

			voxels[1] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
				voxel->globalIndexX, voxel->globalIndexY - 1, voxel->globalIndexZ);

			voxels[2] = GetVoxel(exeInfo, globalHash_info, globalHash, globalHash_value, voxelValues, extractionVoxels,
				voxel->globalIndexX - 1, voxel->globalIndexY, voxel->globalIndexZ);

			voxels[3] = voxel;
		}

		Eigen::Vector3f accumulatedPosition = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		int positionCount = 0;
		for (int voxelIndex = 0; voxelIndex < 4; voxelIndex++)
		{
			auto incidentVoxel = voxels[voxelIndex];
			if (nullptr == incidentVoxel) continue;

			for (int i = 0; i < incidentVoxel->numberOfTriangles; i++)
			{
				auto triangle = incidentVoxel->triangles[i];
				if (*numberOfExtractionEdges <= triangle.edgeIndices[0] ||
					*numberOfExtractionEdges <= triangle.edgeIndices[1] ||
					*numberOfExtractionEdges <= triangle.edgeIndices[2])
					continue;

				if (edgeIndex == triangle.edgeIndices[0])
				{
					auto& ea = extractionEdges[triangle.edgeIndices[1]];
					auto& pa = zeroCrossingPositions[ea.zeroCrossingPointIndex];
					if (VECTOR3F_VALID_(pa))
					{
						accumulatedPosition += pa;
						positionCount++;
					}

					auto& eb = extractionEdges[triangle.edgeIndices[2]];
					auto& pb = zeroCrossingPositions[eb.zeroCrossingPointIndex];
					if (VECTOR3F_VALID_(pb))
					{
						accumulatedPosition += pb;
						positionCount++;
					}
				}
				else if (edgeIndex == triangle.edgeIndices[1])
				{
					auto& ea = extractionEdges[triangle.edgeIndices[0]];
					auto& pa = zeroCrossingPositions[ea.zeroCrossingPointIndex];
					if (VECTOR3F_VALID_(pa))
					{
						accumulatedPosition += pa;
						positionCount++;
					}

					auto& eb = extractionEdges[triangle.edgeIndices[2]];
					auto& pb = zeroCrossingPositions[eb.zeroCrossingPointIndex];
					if (VECTOR3F_VALID_(pb))
					{
						accumulatedPosition += pb;
						positionCount++;
					}
				}
				else if (edgeIndex == triangle.edgeIndices[2])
				{
					auto& ea = extractionEdges[triangle.edgeIndices[0]];
					auto& pa = zeroCrossingPositions[ea.zeroCrossingPointIndex];
					if (VECTOR3F_VALID_(pa))
					{
						accumulatedPosition += pa;
						positionCount++;
					}

					auto& eb = extractionEdges[triangle.edgeIndices[1]];
					auto& pb = zeroCrossingPositions[eb.zeroCrossingPointIndex];
					if (VECTOR3F_VALID_(pb))
					{
						accumulatedPosition += pb;
						positionCount++;
					}
				}
			}
		}

		if (positionCount > 0)
		{
			resultZeroCrossingPositions[edge->zeroCrossingPointIndex] = (Eigen::Vector3f)((accumulatedPosition / (float)positionCount));// +Eigen::Vector3f(0.0f, 1.0f, 0.0f);

			//printf("%f, %f, %f\n", meanPosition.x(), meanPosition.y(), meanPosition.z());
		}

		//if (positionCount > 0)
		//{
		//	resultZeroCrossingPositions[edge->zeroCrossingPointIndex] = meanPosition / (float)positionCount;
		//}
		//else
		//{
		//	resultZeroCrossingPositions[edge->zeroCrossingPointIndex] = zeroCrossingPositions[edge->zeroCrossingPointIndex];
		//}
	}

	__global__ void Kernel_CountNeighborPoints(
		MarchingCubes::ExecutionInfo exeInfo,
		HashKey64* globalHash_info, uint64_t* globalHash, uint8_t* globalHash_value,
		voxel_value_t* voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f* voxelNormals, Eigen::Vector<unsigned char, 3>* voxelColors,
		Eigen::Vector3f* resultPositions, Eigen::Vector3f* resultNormals, Eigen::Vector4f* resultColors,
		unsigned int* numberOfVoxelPositions, ExtractionVoxel* extractionVoxels, ExtractionEdge* extractionEdges,
		Eigen::Vector3f* tempPositions, unsigned int* numberOfTemp)
	{
		unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid > globalHash_info->HashTableCapacity - 1) return;

		auto extractionEdge = extractionEdges[threadid];

		auto startVoxel = extractionVoxels[extractionEdge.startVoxelIndex];
		auto endVoxel = extractionVoxels[extractionEdge.endVoxelIndex];
	}

	void MarchingCubes::ExtractVolume(const std::string& filename, Eigen::Vector3f& localMin, Eigen::Vector3f& localMax, cached_allocator* alloc_, CUstream_st* st)
	{
		if (globalHash_info_host == nullptr) return;

		qDebug("ExtractVolume()");

		Eigen::Vector3f* zeroCrossingPositions = nullptr;
		cudaMalloc(&zeroCrossingPositions, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity * 3);

		unsigned int* numberOfzeroCrossingPositions = nullptr;
		cudaMalloc(&numberOfzeroCrossingPositions, sizeof(unsigned int));
		cudaMemset(numberOfzeroCrossingPositions, 0, sizeof(unsigned int));

		Eigen::Vector3f* resultZeroCrossingPositions = nullptr;
		cudaMalloc(&resultZeroCrossingPositions, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity * 3);



		//Eigen::Vector3f* voxelPositions = nullptr;
		//cudaMalloc(&voxelPositions, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity);

		//Eigen::Vector3f* voxelNormals = nullptr;
		//cudaMalloc(&voxelNormals, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity);

		//Eigen::Vector4f* voxelColors = nullptr;
		//cudaMalloc(&voxelColors, sizeof(Eigen::Vector4f) * globalHash_info_host->HashTableCapacity);

		//unsigned int* numberOfVoxelPositions = nullptr;
		//cudaMalloc(&numberOfVoxelPositions, sizeof(unsigned int));
		//cudaMemset(numberOfVoxelPositions, 0, sizeof(unsigned int));

		checkCudaSync(st);

		exeInfo.local.SetLocalMinMax(localMin, localMax);
		//exeInfo.local.SetLocalMinMax(exeInfo.global.globalMin, exeInfo.global.globalMax);

		ExtractionVoxel* extractionVoxels = nullptr;
		cudaMalloc(&extractionVoxels, sizeof(ExtractionVoxel) * globalHash_info_host->HashTableCapacity);

		unsigned int* numberOfExtractionVoxels = nullptr;
		cudaMalloc(&numberOfExtractionVoxels, sizeof(unsigned int));
		cudaMemset(numberOfExtractionVoxels, 0, sizeof(unsigned int));

		ExtractionEdge* extractionEdges = nullptr;
		cudaMalloc(&extractionEdges, sizeof(ExtractionEdge) * globalHash_info_host->HashTableCapacity * 3);

		unsigned int* numberOfExtractionEdges = nullptr;
		cudaMalloc(&numberOfExtractionEdges, sizeof(unsigned int));
		cudaMemset(numberOfExtractionEdges, 0xFFFFFFFF, sizeof(unsigned int));

		Eigen::Vector3f* tempPositions = nullptr;
		cudaMalloc(&tempPositions, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity * 3);

		unsigned int* numberOfTemp = nullptr;
		cudaMalloc(&numberOfTemp, sizeof(unsigned int));
		cudaMemset(numberOfTemp, 0, sizeof(unsigned int));

		checkCudaSync(st);

		nvtxRangePushA("Kernel_PopulateExtractionVoxels");
		{
			int mingridsize;
			int threadblocksize;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_PopulateExtractionVoxels, 0, 0));
			int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

			Kernel_PopulateExtractionVoxels << <gridsize, threadblocksize, 0, st >> > (exeInfo, globalHash_info, globalHash, globalHash_value,
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelValues.data()),
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelValueCounts.data()),
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelNormals.data()),
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelColors.data()),
				//voxelPositions, voxelNormals, voxelColors, numberOfVoxelPositions,
				extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges);

			checkCudaSync(st);
		}
		nvtxRangePop();

		nvtxRangePushA("Kernel_CalculateZeroCrossingPoints");
		{
			int mingridsize;
			int threadblocksize;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_CalculateZeroCrossingPoints, 0, 0));
			int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity * 3 + threadblocksize - 1) / threadblocksize;

			Kernel_CalculateZeroCrossingPoints << <gridsize, threadblocksize, 0, st >> > (exeInfo, globalHash_info, globalHash, globalHash_value,
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelValues.data()),
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelValueCounts.data()),
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelNormals.data()),
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelColors.data()),
				//voxelPositions, voxelNormals, voxelColors, numberOfVoxelPositions,
				extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges,
				zeroCrossingPositions, numberOfzeroCrossingPositions);

			checkCudaSync(st);
		}
		nvtxRangePop();

		nvtxRangePushA("Kernel_MarchingCubes");
		{
			int mingridsize;
			int threadblocksize;
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_MarchingCubes, 0, 0));
			int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

			Kernel_MarchingCubes << <gridsize, threadblocksize, 0, st >> > (exeInfo, globalHash_info, globalHash, globalHash_value,
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelValues.data()),
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelValueCounts.data()),
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelNormals.data()),
				thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelColors.data()),
				//voxelPositions, voxelNormals, voxelColors, numberOfVoxelPositions,
				extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges);

			checkCudaSync(st);
		}
		nvtxRangePop();

		//unsigned int host_numberOfExtractionEdges = 0;
		//cudaMemcpyAsync(&host_numberOfExtractionEdges, numberOfExtractionEdges, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

		//checkCudaErrors(cudaStreamSynchronize(st));

		//unsigned int* numberOfResultIndices;
		//cudaMalloc(&numberOfResultIndices, sizeof(unsigned int));

		//unsigned int* resultIndices;
		//cudaMalloc(&resultIndices, sizeof(unsigned int)* host_numberOfExtractionEdges * 3 * 4);


		for (size_t count = 0; count < 10; count++)
		{
			nvtxRangePushA("Kernel_MeshSmooth");
			{
				int mingridsize;
				int threadblocksize;
				checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_MeshSmooth, 0, 0));
				int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity * 3 + threadblocksize - 1) / threadblocksize;

				Kernel_MeshSmooth << <gridsize, threadblocksize, 0, st >> > (
					exeInfo, globalHash_info, globalHash, globalHash_value,
					thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelValues.data()),
					extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges,
					zeroCrossingPositions, numberOfzeroCrossingPositions, resultZeroCrossingPositions);

				checkCudaErrors(cudaStreamSynchronize(st));

				cudaMemcpy(
					zeroCrossingPositions,
					resultZeroCrossingPositions,
					sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity * 3,
					cudaMemcpyDeviceToDevice);
			}
			nvtxRangePop();
		}

		//nvtxRangePushA("Kernel_MarchingCubes_Verify");
		//{
		//	int mingridsize;
		//	int threadblocksize;
		//	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_MarchingCubes, 0, 0));
		//	int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

		//	Kernel_MarchingCubes << <gridsize, threadblocksize, 0, st >> > (exeInfo, globalHash_info, globalHash, globalHash_value,
		//		thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelValues.data()),
		//		thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelValueCounts.data()),
		//		thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelNormals.data()),
		//		thrust::raw_pointer_cast(CUDA_MANAGER->regModule->m_MC_voxelColors.data()),
		//		//voxelPositions, voxelNormals, voxelColors, numberOfVoxelPositions,
		//		extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges);

		//	checkCudaErrors(cudaStreamSynchronize(st));
		//}
		//nvtxRangePop();

		unsigned int host_numberOfZeroCrossingValues = 0;
		cudaMemcpyAsync(&host_numberOfZeroCrossingValues, numberOfzeroCrossingPositions, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

		Eigen::Vector3f* host_ZeroCrossingPositions = new Eigen::Vector3f[host_numberOfZeroCrossingValues];
		cudaMemcpyAsync(host_ZeroCrossingPositions, zeroCrossingPositions,
			sizeof(Eigen::Vector3f) * host_numberOfZeroCrossingValues, cudaMemcpyDeviceToHost, st);

		printf("host_numberOfZeroCrossingValues : %d\n", host_numberOfZeroCrossingValues);

		//unsigned int host_numberOfVoxelValues = 0;
		//cudaMemcpyAsync(&host_numberOfVoxelValues, numberOfVoxelPositions, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

		//Eigen::Vector3f* host_voxelPositions = new Eigen::Vector3f[host_numberOfVoxelValues];
		//cudaMemcpyAsync(host_voxelPositions, voxelPositions,
		//	sizeof(Eigen::Vector3f) * host_numberOfVoxelValues, cudaMemcpyDeviceToHost, st);

		//Eigen::Vector3f* host_voxelNormals = new Eigen::Vector3f[host_numberOfVoxelValues];
		//cudaMemcpyAsync(host_voxelNormals, voxelNormals,
			//	sizeof(Eigen::Vector3f) * host_numberOfVoxelValues, cudaMemcpyDeviceToHost, st);

			//Eigen::Vector4f* host_voxelColors = new Eigen::Vector4f[host_numberOfVoxelValues];
			//cudaMemcpyAsync(host_voxelColors, voxelColors,
			//	sizeof(Eigen::Vector4f) * host_numberOfVoxelValues, cudaMemcpyDeviceToHost, st);

		checkCudaSync(st);

		qDebug("host_numberOfZeroCrossingValues : %d", host_numberOfZeroCrossingValues);

		{
			ExtractionVoxel* host_extractionVoxels = new ExtractionVoxel[globalHash_info_host->HashTableCapacity];
			cudaMemcpy(host_extractionVoxels, extractionVoxels, sizeof(ExtractionVoxel) * globalHash_info_host->HashTableCapacity, cudaMemcpyDeviceToHost);

			ExtractionEdge* host_extractionEdges = new ExtractionEdge[globalHash_info_host->HashTableCapacity * 3];
			cudaMemcpy(host_extractionEdges, extractionEdges, sizeof(ExtractionEdge) * globalHash_info_host->HashTableCapacity * 3, cudaMemcpyDeviceToHost);

			PLYFormat ply;

			for (size_t i = 0; i < host_numberOfZeroCrossingValues; i++)
			{
				ply.AddPointFloat3(host_ZeroCrossingPositions[i].data());
			}

			for (size_t i = 0; i < globalHash_info_host->HashTableCapacity; i++)
			{
				auto voxel = host_extractionVoxels[i];
				for (size_t j = 0; j < voxel.numberOfTriangles; j++)
				{
					auto& triangle = voxel.triangles[j];
					auto e0 = triangle.vertexIndices[0];
					auto e1 = triangle.vertexIndices[1];
					auto e2 = triangle.vertexIndices[2];

					if (e0 == e1 || e0 == e2 || e1 == e2) continue;
					if (UINT32_MAX == e0 || UINT32_MAX == e1 || UINT32_MAX == e2) continue;
					//if (0 == e0 || 0 == e1 || 0 == e2) continue;
					if (e0 >= host_numberOfZeroCrossingValues ||
						e1 >= host_numberOfZeroCrossingValues ||
						e2 >= host_numberOfZeroCrossingValues) continue;

					ply.AddIndex(e0);
					ply.AddIndex(e1);
					ply.AddIndex(e2);
				}
			}
			ply.Serialize("C:\\Resources\\Debug\\MarchingCubes.ply");

			delete[] host_extractionVoxels;
			delete[] host_extractionEdges;
		}

		PLYFormat plyZeroCrossing;
		for (size_t i = 0; i < host_numberOfZeroCrossingValues; i++)
		{
			plyZeroCrossing.AddPointFloat3(host_ZeroCrossingPositions[i].data());
		}
		plyZeroCrossing.Serialize("C:\\Resources\\Debug\\ZeroCrossingPoints.ply");

#pragma region Save VoxelValues
		//{
	//	PLYFormat ply;

	//	for (size_t i = 0; i < host_numberOfVoxelValues; i++)
	//	{
	//		auto& v = host_voxelPositions[i];
			//		if (VECTOR3F_VALID_(v))
			//		{
			//			ply.AddPointFloat3(v.data());
			//			auto& normal = host_voxelNormals[i];
			//			normal.normalize();
			//			ply.AddNormalFloat3(normal.data());
			//			auto& color = host_voxelColors[i];
			//			ply.AddColorFloat4(color.data());
			//		}
			//	}
			//	ply.Serialize(filename);
			//}

			//{
			//	unsigned int host_numberOfTemp = 0;
			//	cudaMemcpyAsync(&host_numberOfTemp, numberOfTemp, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

			//	Eigen::Vector3f* host_tempPositions = new Eigen::Vector3f[host_numberOfTemp];
			//	cudaMemcpyAsync(host_tempPositions, tempPositions,
			//		sizeof(Eigen::Vector3f) * host_numberOfTemp, cudaMemcpyDeviceToHost, st);

			//	PLYFormat ply;

			//	for (size_t i = 0; i < host_numberOfTemp; i++)
			//	{
			//		auto& v = host_tempPositions[i];
			//		if (VECTOR3F_VALID_(v))
			//		{
			//			ply.AddPointFloat3(v.data());
			//		}
			//	}
			//	ply.Serialize(filename);

			//	delete[] host_tempPositions;
			//}

	//delete[] host_voxelPositions;
	//delete[] host_voxelNormals;
	//delete[] host_voxelColors;  
#pragma endregion

		cudaFree(zeroCrossingPositions);
		cudaFree(numberOfzeroCrossingPositions);

		cudaFree(resultZeroCrossingPositions);

		//cudaFree(voxelPositions);
		//cudaFree(voxelNormals);
		//cudaFree(voxelColors);
		//cudaFree(numberOfVoxelPositions);

		cudaFree(extractionVoxels);
		cudaFree(extractionEdges);
	}

#endif // 0


}
