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
		}
	};

	struct Voxel
	{
		float tsdfValue;
		float weight;
		float3 normal;
		float3 color;
	};

	struct Vertex
	{
		float3 position;
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

		// 메시 추출 함수 추가
		std::vector<Vertex> ExtractMesh()
		{
			std::vector<Vertex> meshVertices;

			for (uint32_t z = 0; z < internal->dimensions.z - 1; ++z)
			{
				for (uint32_t y = 0; y < internal->dimensions.y - 1; ++y)
				{
					for (uint32_t x = 0; x < internal->dimensions.x - 1; ++x)
					{
						// 현재 그리드 셀의 8개 코너 정점 TSDF 값 가져오기
						std::array<float, 8> tsdfValues;
						std::array<float3, 8> positions;

						for (int i = 0; i < 8; ++i)
						{
							uint3 cornerOffset = getCornerOffset(i);
							uint3 cornerIndex = make_uint3(x + cornerOffset.x, y + cornerOffset.y, z + cornerOffset.z);
							size_t flatIndex = cornerIndex.z * (internal->dimensions.x * internal->dimensions.y) +
								cornerIndex.y * internal->dimensions.x +
								cornerIndex.x;

							tsdfValues[i] = internal->elements[flatIndex].tsdfValue;
							positions[i] = GetPosition(internal->center, internal->dimensions, internal->voxelSize, cornerIndex);
						}

						// Marching Cubes 알고리즘을 위한 인덱스 계산
						int cubeIndex = 0;
						for (int i = 0; i < 8; ++i)
						{
							if (tsdfValues[i] < 0)
							{
								cubeIndex |= (1 << i);
							}
						}

						// 삼각형이 없는 경우 스킵
						if (cubeIndex == 0 || cubeIndex == 255)
						{
							continue;
						}

						// 삼각형 목록을 가져와 처리 (triTable 사용)
						for (int i = 0; triTable[cubeIndex][i] != -1; i += 3)
						{
							int edgeA = triTable[cubeIndex][i];
							int edgeB = triTable[cubeIndex][i + 1];
							int edgeC = triTable[cubeIndex][i + 2];

							Vertex vertexA = interpolateVertex(edgeA, positions, tsdfValues);
							Vertex vertexB = interpolateVertex(edgeB, positions, tsdfValues);
							Vertex vertexC = interpolateVertex(edgeC, positions, tsdfValues);

							meshVertices.push_back(vertexA);
							meshVertices.push_back(vertexB);
							meshVertices.push_back(vertexC);
						}
					}
				}
			}

			return meshVertices;
		}

		// 각 코너의 오프셋을 구하는 함수 (Marching Cubes용)
		uint3 getCornerOffset(int cornerIndex)
		{
			static const uint3 offsets[8] = {
				make_uint3(0, 0, 0), make_uint3(1, 0, 0),
				make_uint3(1, 1, 0), make_uint3(0, 1, 0),
				make_uint3(0, 0, 1), make_uint3(1, 0, 1),
				make_uint3(1, 1, 1), make_uint3(0, 1, 1)
			};
			return offsets[cornerIndex];
		}

		// 두 정점 사이의 보간된 정점을 생성하는 함수
		Vertex interpolateVertex(int edgeIndex, const std::array<float3, 8>& positions, const std::array<float, 8>& tsdfValues)
		{
			static const int edgeVertexMap[12][2] = {
				{0, 1}, {1, 2}, {2, 3}, {3, 0},
				{4, 5}, {5, 6}, {6, 7}, {7, 4},
				{0, 4}, {1, 5}, {2, 6}, {3, 7}
			};

			const auto& edge = edgeVertexMap[edgeIndex];
			int v0 = edge[0];
			int v1 = edge[1];

			float3 p0 = positions[v0];
			float3 p1 = positions[v1];
			float t = (0 - tsdfValues[v0]) / (tsdfValues[v1] - tsdfValues[v0]);

			Vertex vertex;
			vertex.position = make_float3(
				p0.x + t * (p1.x - p0.x),
				p0.y + t * (p1.y - p0.y),
				p0.z + t * (p1.z - p0.z)
			);

			vertex.normal = normalize(interpolateNormal(v0, v1));
			vertex.color = make_float3(0.5f, 0.5f, 0.5f); // 컬러는 임시로 설정, 필요시 보간할 수 있음

			return vertex;
		}

		// 두 정점 사이의 노말을 보간하는 함수
		float3 interpolateNormal(int v0, int v1)
		{
			Voxel voxel0 = internal->elements[v0];
			Voxel voxel1 = internal->elements[v1];
			return make_float3(
				(voxel0.normal.x + voxel1.normal.x) * 0.5f,
				(voxel0.normal.y + voxel1.normal.y) * 0.5f,
				(voxel0.normal.z + voxel1.normal.z) * 0.5f
			);
		}

		// Divergence 기반으로 메시를 추출하는 함수 추가
		// 기존의 ExtractMeshFromPotentialField 코드
		std::vector<Vertex> ExtractMeshFromPotentialField(float* potentialField)
		{
			std::vector<Vertex> meshVertices;

			float minValue = FLT_MAX;
			float maxValue = -FLT_MAX;

			// 잠재 필드의 최소/최대 값 계산
			for (uint32_t z = 0; z < internal->dimensions.z; ++z)
			{
				for (uint32_t y = 0; y < internal->dimensions.y; ++y)
				{
					for (uint32_t x = 0; x < internal->dimensions.x; ++x)
					{
						size_t flatIndex = z * (internal->dimensions.x * internal->dimensions.y) +
							y * internal->dimensions.x + x;
						float value = potentialField[flatIndex];
						minValue = fminf(minValue, value);
						maxValue = fmaxf(maxValue, value);
					}
				}
			}

			float intensityRange = maxValue - minValue;

			for (uint32_t z = 0; z < internal->dimensions.z - 1; ++z)
			{
				for (uint32_t y = 0; y < internal->dimensions.y - 1; ++y)
				{
					for (uint32_t x = 0; x < internal->dimensions.x - 1; ++x)
					{
						std::array<float, 8> potentialValues;
						std::array<float3, 8> positions;

						for (int i = 0; i < 8; ++i)
						{
							uint3 cornerOffset = getCornerOffset(i);
							uint3 cornerIndex = make_uint3(x + cornerOffset.x, y + cornerOffset.y, z + cornerOffset.z);
							size_t flatIndex = cornerIndex.z * (internal->dimensions.x * internal->dimensions.y) +
								cornerIndex.y * internal->dimensions.x +
								cornerIndex.x;

							potentialValues[i] = potentialField[flatIndex];
							positions[i] = GetPosition(internal->center, internal->dimensions, internal->voxelSize, cornerIndex);
						}

						// Marching Cubes 알고리즘을 위한 인덱스 계산
						int cubeIndex = 0;
						for (int i = 0; i < 8; ++i)
						{
							if (potentialValues[i] < 0)
							{
								cubeIndex |= (1 << i);
							}
						}

						if (cubeIndex == 0 || cubeIndex == 255)
						{
							continue;
						}

						// 삼각형 처리
						for (int i = 0; triTable[cubeIndex][i] != -1; i += 3)
						{
							int edgeA = triTable[cubeIndex][i];
							int edgeB = triTable[cubeIndex][i + 1];
							int edgeC = triTable[cubeIndex][i + 2];

							Vertex vertexA = interpolateVertex_Potential(edgeA, positions, potentialValues);
							Vertex vertexB = interpolateVertex_Potential(edgeB, positions, potentialValues);
							Vertex vertexC = interpolateVertex_Potential(edgeC, positions, potentialValues);

							// 노말 계산
							float3 normal = normalize(cross(
								make_float3(vertexB.position.x - vertexA.position.x,
									vertexB.position.y - vertexA.position.y,
									vertexB.position.z - vertexA.position.z),
								make_float3(vertexC.position.x - vertexA.position.x,
									vertexC.position.y - vertexA.position.y,
									vertexC.position.z - vertexA.position.z)
							));

							vertexA.normal = normal;
							vertexB.normal = normal;
							vertexC.normal = normal;

							// 컬러 계산
							float colorIntensityA = (potentialValues[edgeVertexMap[edgeA][0]] - minValue) / intensityRange;
							float colorIntensityB = (potentialValues[edgeVertexMap[edgeB][0]] - minValue) / intensityRange;
							float colorIntensityC = (potentialValues[edgeVertexMap[edgeC][0]] - minValue) / intensityRange;

							colorIntensityA = fmaxf(0.0f, fminf(1.0f, colorIntensityA));
							colorIntensityB = fmaxf(0.0f, fminf(1.0f, colorIntensityB));
							colorIntensityC = fmaxf(0.0f, fminf(1.0f, colorIntensityC));

							// 컬러 적용 (빨강-초록-파랑 그라디언트)
							vertexA.color = make_float3(1.0f - colorIntensityA, colorIntensityA, 0.0f);
							vertexB.color = make_float3(1.0f - colorIntensityB, colorIntensityB, 0.0f);
							vertexC.color = make_float3(1.0f - colorIntensityC, colorIntensityC, 0.0f);

							// 정점 추가
							meshVertices.push_back(vertexA);
							meshVertices.push_back(vertexB);
							meshVertices.push_back(vertexC);
						}
					}
				}
			}

			return meshVertices;
		}
		


		// 두 정점 사이의 보간된 정점을 생성하는 함수 (포텐셜 필드 보간)
		Vertex interpolateVertex_Potential(int edgeIndex, const std::array<float3, 8>& positions, const std::array<float, 8>& potentialValues)
		{
			static const int edgeVertexMap[12][2] = {
				{0, 1}, {1, 2}, {2, 3}, {3, 0},
				{4, 5}, {5, 6}, {6, 7}, {7, 4},
				{0, 4}, {1, 5}, {2, 6}, {3, 7}
			};

			const auto& edge = edgeVertexMap[edgeIndex];
			int v0 = edge[0];
			int v1 = edge[1];

			float3 p0 = positions[v0];
			float3 p1 = positions[v1];

			// 분모가 0이 되는 경우를 방지하기 위해 epsilon을 도입합니다.
			float epsilon = 1e-6f;
			float denom = potentialValues[v1] - potentialValues[v0];
			if (fabs(denom) < epsilon) {
				denom = (denom < 0) ? -epsilon : epsilon;  // 부호 유지하며 epsilon으로 설정
			}

			// 보간 비율 t 계산
			float t = (0 - potentialValues[v0]) / denom;

			// 보간된 정점 계산
			Vertex vertex;
			vertex.position = make_float3(
				p0.x + t * (p1.x - p0.x),
				p0.y + t * (p1.y - p0.y),
				p0.z + t * (p1.z - p0.z)
			);

			// 노말 계산 (임시로 각 정점의 노말을 보간하거나, 두 위치의 방향성 이용)
			// 여기서는 p0와 p1의 위치 차이를 이용하여 간단히 노말 방향성을 결정
			float3 direction = normalize(make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z));
			vertex.normal = direction;

			// 색상 설정: 잠재 필드 값에 따라 그라데이션 컬러를 적용할 수 있습니다.
			float color_intensity = fmaxf(0.0f, fminf(1.0f, (potentialValues[v0] + potentialValues[v1]) * 0.5f));  // 잠재 필드 값에 따라 0-1로 정규화
			vertex.color = make_float3(color_intensity, color_intensity, color_intensity);  // 잠재 필드 값에 기반하여 색상 설정

			return vertex;
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
	__global__ void ComputeDivergence(RegularGrid<T>::Internal* regularGrid, float* divergenceOutput)
	{
		size_t threadX = blockIdx.x * blockDim.x + threadIdx.x;
		size_t threadY = blockIdx.y * blockDim.y + threadIdx.y;
		size_t threadZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (threadX >= regularGrid->dimensions.x ||
			threadY >= regularGrid->dimensions.y ||
			threadZ >= regularGrid->dimensions.z) return;

		size_t flatIndex = threadZ * (regularGrid->dimensions.x * regularGrid->dimensions.y) +
			threadY * regularGrid->dimensions.x + threadX;

		Voxel voxel = regularGrid->elements[flatIndex];

		float dx = 0.0f, dy = 0.0f, dz = 0.0f;

		if (threadX > 0 && threadX < regularGrid->dimensions.x - 1)
		{
			size_t xPlus = flatIndex + 1;
			size_t xMinus = flatIndex - 1;
			dx = (regularGrid->elements[xPlus].tsdfValue - regularGrid->elements[xMinus].tsdfValue) / (2.0f * regularGrid->voxelSize);
		}

		if (threadY > 0 && threadY < regularGrid->dimensions.y - 1)
		{
			size_t yPlus = flatIndex + regularGrid->dimensions.x;
			size_t yMinus = flatIndex - regularGrid->dimensions.x;
			dy = (regularGrid->elements[yPlus].tsdfValue - regularGrid->elements[yMinus].tsdfValue) / (2.0f * regularGrid->voxelSize);
		}

		if (threadZ > 0 && threadZ < regularGrid->dimensions.z - 1)
		{
			size_t zPlus = flatIndex + regularGrid->dimensions.x * regularGrid->dimensions.y;
			size_t zMinus = flatIndex - regularGrid->dimensions.x * regularGrid->dimensions.y;
			dz = (regularGrid->elements[zPlus].tsdfValue - regularGrid->elements[zMinus].tsdfValue) / (2.0f * regularGrid->voxelSize);
		}

		divergenceOutput[flatIndex] = dx + dy + dz;
	}

	void SolvePoissonEquation(RegularGrid<Voxel>::Internal* regularGrid, float* divergence)
	{
		cusolverSpHandle_t cusolverHandle;
		cusparseMatDescr_t descrA;

		cusolverSpCreate(&cusolverHandle);
		cusparseCreateMatDescr(&descrA);
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

		int numRows = regularGrid->dimensions.x * regularGrid->dimensions.y * regularGrid->dimensions.z;
		int nnz = numRows * 7; // Example: Assuming a 7-point stencil for each grid cell, adjust as needed

		float tol = 1e-6f;
		int reorder = 0;
		int singularity = 0;

		// 포아송 방정식을 풀기 위한 cuSolver 호출 예제
		float* d_A_values;    // Laplacian matrix values in CSR format
		int* d_A_rowPtr;      // Row pointer for CSR
		int* d_A_colInd;      // Column indices for CSR
		float* d_x;           // Solution vector

		// Memory allocation for CSR matrix and solution vector
		checkCudaErrors(cudaMalloc((void**)&d_A_values, sizeof(float) * nnz));
		checkCudaErrors(cudaMalloc((void**)&d_A_rowPtr, sizeof(int) * (numRows + 1)));
		checkCudaErrors(cudaMalloc((void**)&d_A_colInd, sizeof(int) * nnz));
		checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(float) * numRows));

		cusolverSpScsrlsvqr(
			cusolverHandle, numRows, nnz,
			descrA, d_A_values, d_A_rowPtr, d_A_colInd,
			divergence, tol, reorder, d_x, &singularity
		);

		if (singularity >= 0) {
			printf("WARNING: A is singular at row %d\n", singularity);
		}

		// cuSolver 해제
		cusolverSpDestroy(cusolverHandle);
		cusparseDestroyMatDescr(descrA);

		// 메모리 해제
		checkCudaErrors(cudaFree(d_A_values));
		checkCudaErrors(cudaFree(d_A_rowPtr));
		checkCudaErrors(cudaFree(d_A_colInd));
		checkCudaErrors(cudaFree(d_x));
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

		// signedDistance는 포인트와 보폭 센터의 상대적 위치에 따라 음수 또는 양수가 될 수 있어야 합니다.
		float signedDistance = dot(normal, voxelCenter - p) < 0 ? -distance : distance;

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

		//{
		//    t = Time::Now();

		//    for (uint32_t z = 0; z < rg.internal->dimensions.z; z++)
		//    {
		//        for (uint32_t y = 0; y < rg.internal->dimensions.y; y++)
		//        {
		//            for (uint32_t x = 0; x < rg.internal->dimensions.x; x++)
		//            {
		//                size_t flatIndex = z * (rg.internal->dimensions.x * rg.internal->dimensions.y) +
		//                    y * rg.internal->dimensions.x + x;

		//                //printf("flatIndex : %llu\n", flatIndex);

		//                auto voxel = rg.internal->elements[flatIndex];

		//                if (-0.1f <= voxel.tsdfValue && voxel.tsdfValue <= 0.1f)
		//                    //if (-0.2f <= voxel.tsdfValue && voxel.tsdfValue <= 0.2f)
		//                {
		//                    auto p = GetPosition(rg.internal->center, rg.internal->dimensions, rg.internal->voxelSize, make_uint3(x, y, z));
		//                    auto n = voxel.normal;
		//                    //n = make_float3(0.0f, 0.0f, 1.0f);
		//                    auto c = voxel.color;
		//                    Color4 c4;
		//                    c4.FromNormalized(c.x, c.y, c.z, 1.0f);
		//                    VD::AddCube("voxels", { p.x + 20.0f, p.y, p.z },
		//                        { rg.internal->voxelSize * 0.5f, rg.internal->voxelSize * 0.5f,rg.internal->voxelSize * 0.5f },
		//                        { n.x, n.y, n.z }, c4);
		//                }
		//            }
		//        }
		//    }

		//    t = Time::End(t, "?????");
		//}

		//{
		//	t = Time::Now();

		//	// Divergence 계산을 위한 출력 배열
		//	float* divergence;
		//	checkCudaErrors(cudaMallocManaged(&divergence, sizeof(float) * rg.internal->numberOfVoxels));

		//	dim3 threadsPerBlock(8, 8, 8);
		//	dim3 blocksPerGrid(
		//		(rg.internal->dimensions.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
		//		(rg.internal->dimensions.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
		//		(rg.internal->dimensions.z + threadsPerBlock.z - 1) / threadsPerBlock.z
		//	);

		//	ComputeDivergence<Voxel> << <blocksPerGrid, threadsPerBlock >> > (rg.internal, divergence);
		//	checkCudaErrors(cudaDeviceSynchronize());

		//	t = Time::End(t, "Compute Divergence");

		//	// 포아송 방정식 풀기
		//	SolvePoissonEquation(rg.internal, divergence);

		//	// 결과 시각화 (간단한 예시)
		//	for (uint32_t z = 0; z < rg.internal->dimensions.z; z++)
		//	{
		//		for (uint32_t y = 0; y < rg.internal->dimensions.y; y++)
		//		{
		//			for (uint32_t x = 0; x < rg.internal->dimensions.x; x++)
		//			{
		//				size_t flatIndex = z * (rg.internal->dimensions.x * rg.internal->dimensions.y) +
		//					y * rg.internal->dimensions.x + x;

		//				float potential = divergence[flatIndex]; // 잠재 필드 값

		//				if (fabs(potential) > 0.01f) // 잠재 필드에 대해 임계값을 적용
		//				{
		//					size_t flatIndex = z * (rg.internal->dimensions.x * rg.internal->dimensions.y) +
		//						y * rg.internal->dimensions.x + x;

		//					//printf("flatIndex : %llu\n", flatIndex);

		//					auto voxel = rg.internal->elements[flatIndex];

		//					auto p = GetPosition(rg.internal->center, rg.internal->dimensions, rg.internal->voxelSize, make_uint3(x, y, z));
		//					auto n = voxel.normal;
		//					//n = make_float3(0.0f, 0.0f, 1.0f);
		//					auto c = voxel.color;
		//					Color4 c4;
		//					c4.FromNormalized(c.x, c.y, c.z, 1.0f);

		//					VD::AddCube("potential_field", { p.x, p.y, p.z },
		//						{ rg.internal->voxelSize * 0.5f, rg.internal->voxelSize * 0.5f, rg.internal->voxelSize * 0.5f },
		//						{ n.x, n.y, n.z }, c4);
		//				}
		//			}
		//		}
		//	}

		//	// 메모리 해제
		//	checkCudaErrors(cudaFree(divergence));
		//}

		//{
		//	std::vector<Vertex> mesh = rg.ExtractMesh();

		//	printf("size of mesh : %llu\n", mesh.size());

		//	for (size_t i = 0; i < mesh.size() / 3; i++)
		//	{
		//		auto v0 = mesh[i * 3];
		//		auto v1 = mesh[i * 3 + 1];
		//		auto v2 = mesh[i * 3 + 2];

		//		VD::AddTriangle("mesh",
		//			{ v0.position.x, v0.position.y, v0.position.z },
		//			{ v1.position.x, v1.position.y, v1.position.z },
		//			{ v2.position.x, v2.position.y, v2.position.z },
		//			Color4::White);
		//	}
		//}
		{
			// Divergence 계산
			float* divergence;
			checkCudaErrors(cudaMallocManaged(&divergence, sizeof(float)* rg.internal->numberOfVoxels));

			dim3 threadsPerBlock(8, 8, 8);
			dim3 blocksPerGrid(
				(rg.internal->dimensions.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(rg.internal->dimensions.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
				(rg.internal->dimensions.z + threadsPerBlock.z - 1) / threadsPerBlock.z
			);

			ComputeDivergence<Voxel> << <blocksPerGrid, threadsPerBlock >> > (rg.internal, divergence);
			checkCudaErrors(cudaDeviceSynchronize());

			t = Time::End(t, "Compute Divergence");

			// 포아송 방정식을 풀기
			SolvePoissonEquation(rg.internal, divergence);

			// 포텐셜 필드 기반 메시 추출
			std::vector<Vertex> mesh = rg.ExtractMeshFromPotentialField(divergence);
			printf("size of potential field mesh: %llu\n", mesh.size());

			// 삼각형 단위로 시각화
			for (size_t i = 0; i < mesh.size() / 3; i++)
			{
				auto v0 = mesh[i * 3];
				auto v1 = mesh[i * 3 + 1];
				auto v2 = mesh[i * 3 + 2];

				float x = (v0.color.x + v1.color.x + v2.color.x) / 3.0f;
				float y = (v0.color.y + v1.color.y + v2.color.y) / 3.0f;
				float z = (v0.color.z + v1.color.z + v2.color.z) / 3.0f;

				Color4 c;
				c.FromNormalized(x, y, z, 1.0f);

				VD::AddTriangle("PotentialFieldMesh",
					{ v0.position.x, v0.position.y, v0.position.z },
					{ v1.position.x, v1.position.y, v1.position.z },
					{ v2.position.x, v2.position.y, v2.position.z },
					c);
			}

			t = Time::End(t, "Extract and Visualize Mesh from Potential Field");

			checkCudaErrors(cudaFree(divergence));
		}

		//// VisualDebugging를 사용해 메시를 그리기 위한 코드
		//for (const auto& vertex : mesh)
		//{
		//	VD::AddCube("ExtractedMesh",
		//		{ vertex.position.x, vertex.position.y, vertex.position.z },
		//		{ 0.01f, 0.01f, 0.01f },
		//		{ vertex.normal.x, vertex.normal.y, vertex.normal.z },
		//		Color4::FromNormalized(vertex.color.x, vertex.color.y, vertex.color.z, 1.0f));
		//}

		t = Time::End(t, "Mesh Extraction and Visualization");
	}
}
