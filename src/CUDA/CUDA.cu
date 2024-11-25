#include "CUDA.cuh"

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

	// Helper device function to add two float3 vectors
	__device__ float3 operator+(const float3& a, const float3& b) {
		return { a.x + b.x, a.y + b.y, a.z + b.z };
	}

	// Helper device function to add two float3 vectors
	__device__ float3 operator+=(float3& a, float3& b) {
		a.x += b.x;
		a.y += b.y;
		a.z += b.z;
		return { a.x + b.x, a.y + b.y, a.z + b.z };
	}

	// Helper device function to divide a float3 vector by a scalar
	__device__ float3 operator/(const float3& a, float b) {
		return { a.x / b, a.y / b, a.z / b };
	}

	// Helper device function to divide a float3 vector by a scalar
	__device__ float3 operator/=(float3& a, float b) {
		a.x /= b;
		a.y /= b;
		a.z /= b;
		return { a.x / b, a.y / b, a.z / b };
	}

	// Helper device function to calculate the cross product of two vectors
	__device__ float3 crossProduct(const float3& a, const float3& b) {
		return { a.y * b.z - a.z * b.y,
				a.z * b.x - a.x * b.z,
				a.x * b.y - a.y * b.x };
	}

	// Helper device function to normalize a float3 vector
	__device__ float3 normalize(const float3& v) {
		float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
		return len > 0 ? float3{ v.x / len, v.y / len, v.z / len } : float3{ 0, 0, 1 };
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
		float3 normal = crossProduct(v1, v2);

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

	__global__ void Kernel_IntegrateInputPoints(
		Voxel* volume,
		int3 volumeDimension,
		float voxelSize,
		Eigen::Vector3f* inputPoints,
		size_t numberOfInputPoints)
	{
		uint32_t threadid = blockDim.x * blockIdx.x + threadIdx.x;
		if (threadid > numberOfInputPoints - 1) return;

		auto p = inputPoints[threadid];

		int xKey = (int)floorf(p.x() / voxelSize);
		int yKey = (int)floorf(p.y() / voxelSize);
		int zKey = (int)floorf(p.z() / voxelSize);

		if (xKey < 0 || xKey >= volumeDimension.x) return;
		if (yKey < 0 || yKey >= volumeDimension.y) return;
		if (zKey < 0 || zKey >= volumeDimension.z) return;

		volume[zKey * volumeDimension.x * volumeDimension.y + yKey * volumeDimension.x + xKey].tsdfValue = 1.0f;

		int offset = 5;

		for (int z = zKey - offset; z <= zKey + offset; z++)
		{
			if (zKey < 0 || zKey >= volumeDimension.z) return;
			for (int y = yKey - offset; y <= yKey + offset; y++)
			{
				if (yKey < 0 || yKey >= volumeDimension.y) return;
				for (int x = xKey - offset; x <= xKey + offset; x++)
				{
					if (xKey < 0 || xKey >= volumeDimension.x) return;

					volume[z * volumeDimension.x * volumeDimension.y + y * volumeDimension.x + x].tsdfValue = 1.0f;
				}
			}
		}
	}

	void IntegrateInputPoints(Voxel* volume, int3 volumeDimension, float voxelSize, Eigen::Vector3f* inputPoints, size_t numberOfInputPoints)
	{
		nvtxRangePushA("IntegrateInputPoints");

		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_IntegrateInputPoints, 0, 0));
		auto gridsize = (numberOfInputPoints - 1) / threadblocksize;

		Kernel_IntegrateInputPoints << <gridsize, threadblocksize >> > (volume, volumeDimension, voxelSize, inputPoints, numberOfInputPoints);

		checkCudaErrors(cudaDeviceSynchronize());

		nvtxRangePop();
	}
}