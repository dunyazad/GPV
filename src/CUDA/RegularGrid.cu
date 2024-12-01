#include "RegularGrid.cuh"

#include <vtkHeaderFiles.h>

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

	template<typename T>
	__global__ void Kernel_SmoothTSDF(RegularGrid<T>::Internal* regularGrid, T* smoothedElements);

	template<typename T>
	__global__ void Kernel_ExtractMesh(typename RegularGrid<T>::Internal* regularGrid, Vertex* meshVertices, int* vertexCount);

	__host__ __device__
		uint3 GetIndex(const float3& gridCenter, uint3 gridDimensions, float voxelSize, const float3& position);

	__host__ __device__
		float3 GetPosition(const float3& gridCenter, uint3 gridDimensions, float voxelSize, const uint3& index);

	__host__ __device__
		size_t GetFlatIndex(const uint3& index, const uint3& dimensions);
	
	// Declare the offsets array in constant memory
	__device__ __constant__ uint3 offsets[8] = {
		{0, 0, 0}, {1, 0, 0},
		{1, 1, 0}, {0, 1, 0},
		{0, 0, 1}, {1, 0, 1},
		{1, 1, 1}, {0, 1, 1}
	};

	__host__ __device__
		uint3 getCornerOffset(int cornerIndex)
	{
		return offsets[cornerIndex];
	}

	__host__ __device__
		Vertex interpolateVertex(int edgeIndex, float3* positions, float* tsdfValues)
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

		vertex.normal = normalize(make_float3(1.0f, 0.0f, 0.0f)); // Example default normal for device compatibility.
		vertex.color = make_float3(0.5f, 0.5f, 0.5f); // Example default color.

		return vertex;
	}

	//---------------------------------------------------------------------------------------------------
	// Helper Functions ->
	//---------------------------------------------------------------------------------------------------

	Vertex ComputeOptimalVertex(const std::vector<float3>& edgeVertices, const std::vector<float3>& edgeNormals);

	//uint3 getCornerOffset(int cornerIndex);

	Vertex interpolateVertex(int edgeIndex, float3* positions, float* tsdfValues);

	Vertex interpolateVertex_Potential(int edgeIndex, const std::array<float3, 8>& positions, const std::array<float, 8>& potentialValues);

	//---------------------------------------------------------------------------------------------------
	// <- Helper Functions
	//---------------------------------------------------------------------------------------------------


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
			float maxWeight;
		};

		RegularGrid(const float3& center, uint32_t dimensionX, uint32_t dimensionY, uint32_t dimensionZ, float voxelSize)
		{
#ifdef MALLOC_MANAGED
			checkCudaErrors(cudaMallocManaged(&internal, sizeof(Internal)));

			internal->center = center;
			internal->elements = nullptr;
			internal->dimensions = make_uint3(dimensionX, dimensionY, dimensionZ);
			internal->voxelSize = voxelSize;
			internal->numberOfVoxels = dimensionX * dimensionY * dimensionZ;
			internal->truncationDistance = 2.0f;
			internal->maxWeight = 20.0f;

			checkCudaErrors(cudaMallocManaged(&(internal->elements), sizeof(T) * internal->numberOfVoxels));
#else
			checkCudaErrors(cudaMalloc(&internal, sizeof(Internal)));

			h_internal.center = center;
			h_internal.elements = nullptr;
			h_internal.dimensions = make_uint3(dimensionX, dimensionY, dimensionZ);
			h_internal.voxelSize = voxelSize;
			h_internal.numberOfVoxels = dimensionX * dimensionY * dimensionZ;
			h_internal.truncationDistance = 2.0f;
			h_internal.maxWeight = 20.0f;

			checkCudaErrors(cudaMalloc(&(h_internal.elements), sizeof(T) * h_internal.numberOfVoxels));

			cudaMemcpy(internal, &h_internal, sizeof(Internal), cudaMemcpyHostToDevice);
#endif
		}

		~RegularGrid()
		{
			checkCudaErrors(cudaFree(h_internal.elements));
			checkCudaErrors(cudaFree(internal));
		}

		void Clear()
		{
			nvtxRangePushA("Clear");

			dim3 threadsPerBlock(8, 8, 8);  // 8x8x8 threads per block
			dim3 blocksPerGrid(
				(h_internal.dimensions.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(h_internal.dimensions.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
				(h_internal.dimensions.z + threadsPerBlock.z - 1) / threadsPerBlock.z
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

		// Function to launch the smoothing kernel and apply smoothing to the entire TSDF grid
		void SmoothTSDF()
		{
			// Allocate buffer for smoothed elements
			T* smoothedElements;
			checkCudaErrors(cudaMallocManaged(&smoothedElements, sizeof(T) * internal->numberOfVoxels));

			// Set up the CUDA kernel launch parameters
			dim3 threadsPerBlock(8, 8, 8); // You can optimize these values
			dim3 blocksPerGrid(
				(internal->dimensions.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(internal->dimensions.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
				(internal->dimensions.z + threadsPerBlock.z - 1) / threadsPerBlock.z
			);

			// Launch the smoothing kernel
			Kernel_SmoothTSDF<T> << <blocksPerGrid, threadsPerBlock >> > (internal, smoothedElements);
			checkCudaErrors(cudaDeviceSynchronize());

			// Copy the smoothed values back to the original TSDF grid
			cudaMemcpy(internal->elements, smoothedElements, sizeof(T) * internal->numberOfVoxels, cudaMemcpyDeviceToDevice);

			// Free the temporary smoothed elements buffer
			checkCudaErrors(cudaFree(smoothedElements));
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
						float  tsdfValues[8];
						float3 positions[8];
						float3 normals[8];

						for (int i = 0; i < 8; ++i)
						{
							uint3 cornerOffset = getCornerOffset(i);
							uint3 cornerIndex = make_uint3(x + cornerOffset.x, y + cornerOffset.y, z + cornerOffset.z);
							size_t flatIndex = cornerIndex.z * (internal->dimensions.x * internal->dimensions.y) +
								cornerIndex.y * internal->dimensions.x +
								cornerIndex.x;

							tsdfValues[i] = internal->elements[flatIndex].tsdfValue;
							positions[i] = GetPosition(internal->center, internal->dimensions, internal->voxelSize, cornerIndex);
							normals[i] = internal->elements[flatIndex].normal;
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

							// 삼각형의 노말 계산
							float3 edge1 = make_float3(vertexB.position.x - vertexA.position.x,
								vertexB.position.y - vertexA.position.y,
								vertexB.position.z - vertexA.position.z);
							float3 edge2 = make_float3(vertexC.position.x - vertexA.position.x,
								vertexC.position.y - vertexA.position.y,
								vertexC.position.z - vertexA.position.z);
							float3 triangleNormal = normalize(cross(edge1, edge2));

							//// 삼각형 노말과 복셀 노말이 반대 방향이면 삼각형 생성을 건너뜀
							//float3 voxelNormal = normals[edgeA];
							//if (dot(triangleNormal, voxelNormal) < 0)
							//{
							//	continue;
							//}

							//// 삼각형이 내부 방향으로 생성되지 않도록 TSDF 값을 사용하여 확인
							//if (tsdfValues[edgeA] < 0 && tsdfValues[edgeB] < 0 && tsdfValues[edgeC] < 0)
							//{
							//	// 모든 정점이 내부에 있는 경우, 삼각형을 생성하지 않음
							//	continue;
							//}

							meshVertices.push_back(vertexA);
							meshVertices.push_back(vertexB);
							meshVertices.push_back(vertexC);
						}
					}
				}
			}

			return meshVertices;
		}

		std::vector<Vertex> ExtractMeshCUDA()
		{
			// Allocate device memory for vertices and vertex count
			Vertex* d_meshVertices;
			int* d_vertexCount;
			int maxVertices = h_internal.numberOfVoxels * 15;  // An estimated upper limit for the number of vertices

			checkCudaErrors(cudaMalloc(&d_meshVertices, sizeof(Vertex) * maxVertices));
			checkCudaErrors(cudaMalloc(&d_vertexCount, sizeof(int)));
			checkCudaErrors(cudaMemset(d_vertexCount, 0, sizeof(int)));

			// Launch the CUDA kernel for mesh extraction
			dim3 threadsPerBlock(8, 8, 8);
			dim3 blocksPerGrid(
				(h_internal.dimensions.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(h_internal.dimensions.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
				(h_internal.dimensions.z + threadsPerBlock.z - 1) / threadsPerBlock.z
			);

			auto t = Time::Now();

			nvtxRangePushA("ExtractMeshCUDA_Kernel");

			Kernel_ExtractMesh<T> << <blocksPerGrid, threadsPerBlock >> > (internal, d_meshVertices, d_vertexCount);
			checkCudaErrors(cudaDeviceSynchronize());

			nvtxRangePop();

			t = Time::End(t, "ExtractMeshCUDA_Kernel");

			// Copy the resulting vertices back to the host
			int h_vertexCount;
			checkCudaErrors(cudaMemcpy(&h_vertexCount, d_vertexCount, sizeof(int), cudaMemcpyDeviceToHost));

			std::vector<Vertex> meshVertices(h_vertexCount);
			checkCudaErrors(cudaMemcpy(meshVertices.data(), d_meshVertices, sizeof(Vertex) * h_vertexCount, cudaMemcpyDeviceToHost));

			// Free device memory
			checkCudaErrors(cudaFree(d_meshVertices));
			checkCudaErrors(cudaFree(d_vertexCount));

			return meshVertices;
		}


		std::vector<Vertex> ExtractMeshUsingDualContouring() {
			std::vector<Vertex> meshVertices;

			// 3D 배열로 각 셀의 정점을 저장
			std::vector<std::vector<std::vector<float3>>> cellVertices(
				internal->dimensions.z,
				std::vector<std::vector<float3>>(
					internal->dimensions.y,
					std::vector<float3>(internal->dimensions.x, make_float3(0.0f, 0.0f, 0.0f))
					)
			);

			// Iterate over all cells in the grid
			for (uint32_t z = 0; z < internal->dimensions.z - 1; ++z) {
				for (uint32_t y = 0; y < internal->dimensions.y - 1; ++y) {
					for (uint32_t x = 0; x < internal->dimensions.x - 1; ++x) {
						// 1. Collect Hermite data (zero-crossing points and normals)
						std::vector<float3> edgeVertices;
						std::vector<float3> edgeNormals;

						for (int edgeIndex = 0; edgeIndex < 12; ++edgeIndex) {
							int v0 = edgeVertexMap[edgeIndex][0];
							int v1 = edgeVertexMap[edgeIndex][1];

							auto cornerOffset0 = getCornerOffset(v0);
							auto cornerOffset1 = getCornerOffset(v1);

							uint3 corner0 = make_uint3(cornerOffset0.x + x, cornerOffset0.y + y, cornerOffset0.z + z);
							uint3 corner1 = make_uint3(cornerOffset1.x + x, cornerOffset1.y + y, cornerOffset1.z + z);

							size_t index0 = GetFlatIndex(corner0, internal->dimensions);
							size_t index1 = GetFlatIndex(corner1, internal->dimensions);

							float tsdf0 = internal->elements[index0].tsdfValue;
							float tsdf1 = internal->elements[index1].tsdfValue;

							if ((tsdf0 < 0 && tsdf1 > 0) || (tsdf0 > 0 && tsdf1 < 0)) {
								// Interpolate the zero-crossing position
								float t = tsdf0 / (tsdf0 - tsdf1);
								float3 p0 = GetPosition(internal->center, internal->dimensions, internal->voxelSize, corner0);
								float3 p1 = GetPosition(internal->center, internal->dimensions, internal->voxelSize, corner1);

								float3 zeroCrossing = make_float3(
									p0.x + t * (p1.x - p0.x),
									p0.y + t * (p1.y - p0.y),
									p0.z + t * (p1.z - p0.z)
								);

								float3 normal = normalize(internal->elements[index0].normal);

								edgeVertices.push_back(zeroCrossing);
								edgeNormals.push_back(normal);
							}
						}

						// 2. Compute optimal vertex inside the cell
						if (!edgeVertices.empty()) {
							Vertex optimalVertex = ComputeOptimalVertex(edgeVertices, edgeNormals);

							// Save the optimal vertex for the current cell
							cellVertices[z][y][x] = optimalVertex.position;

							// Link vertices to form triangles with neighboring cells
							if (x > 0 && y > 0 && z > 0) {
								Vertex v0 = optimalVertex;
								Vertex v1, v2, v3;

								// Fetch neighboring vertices
								v1.position = cellVertices[z][y][x - 1]; // Left neighbor
								v2.position = cellVertices[z][y - 1][x]; // Below neighbor
								v3.position = cellVertices[z - 1][y][x]; // Back neighbor

								// Add triangles
								meshVertices.push_back(v0);
								meshVertices.push_back(v1);
								meshVertices.push_back(v2);

								meshVertices.push_back(v0);
								meshVertices.push_back(v2);
								meshVertices.push_back(v3);
							}
						}
					}
				}
			}

			return meshVertices;
		}

		// Divergence 기반으로 메시를 추출하는 함수 추가
		// 기존의 ExtractMeshFromPotentialField 코드
		std::vector<Vertex> ExtractMeshFromPotentialField(float* potentialField) {
			std::vector<Vertex> meshVertices;

			for (uint32_t z = 0; z < internal->dimensions.z - 1; ++z) {
				for (uint32_t y = 0; y < internal->dimensions.y - 1; ++y) {
					for (uint32_t x = 0; x < internal->dimensions.x - 1; ++x) {
						std::array<float, 8> potentialValues;
						std::array<float3, 8> positions;

						for (int i = 0; i < 8; ++i) {
							uint3 cornerOffset = getCornerOffset(i);
							uint3 cornerIndex = make_uint3(x + cornerOffset.x, y + cornerOffset.y, z + cornerOffset.z);
							size_t flatIndex = GetFlatIndex(cornerIndex, internal->dimensions);

							potentialValues[i] = potentialField[flatIndex];
							positions[i] = GetPosition(internal->center, internal->dimensions, internal->voxelSize, cornerIndex);
						}

						// Relaxed condition for cubeIndex
						int cubeIndex = 0;
						for (int i = 0; i < 8; ++i) {
							if (potentialValues[i] < 0.05f) {  // Changed threshold
								cubeIndex |= (1 << i);
							}
						}

						if (cubeIndex == 0 || cubeIndex == 255) {
							continue; // No zero crossing
						}

						for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
							int edgeA = triTable[cubeIndex][i];
							int edgeB = triTable[cubeIndex][i + 1];
							int edgeC = triTable[cubeIndex][i + 2];

							Vertex vertexA = interpolateVertex_Potential(edgeA, positions, potentialValues);
							Vertex vertexB = interpolateVertex_Potential(edgeB, positions, potentialValues);
							Vertex vertexC = interpolateVertex_Potential(edgeC, positions, potentialValues);

							// Calculate normals if needed
							float3 normal = normalize(cross(
								vertexB.position - vertexA.position,
								vertexC.position - vertexA.position
							));
							vertexA.normal = vertexB.normal = vertexC.normal = normal;

							// Add to mesh
							meshVertices.push_back(vertexA);
							meshVertices.push_back(vertexB);
							meshVertices.push_back(vertexC);
						}
					}
				}
			}

			return meshVertices;
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

		// 두 정점 사이의 보간된 정점을 생성하는 함수
		Vertex interpolateVertex(int edgeIndex, float3* positions, float* tsdfValues)
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
		Internal h_internal;
	};

	//---------------------------------------------------------------------------------------------------
	// Helper Functions
	//---------------------------------------------------------------------------------------------------

	// Helper function to compute the optimal vertex
	Vertex ComputeOptimalVertex(const std::vector<float3>& edgeVertices, const std::vector<float3>& edgeNormals) {
		// Use Hermite data (positions and normals) to minimize quadratic error
		float3 optimalPosition = make_float3(0.0f, 0.0f, 0.0f);
		float3 optimalNormal = make_float3(0.0f, 0.0f, 0.0f);

		// Solve the quadratic error metric using Hermite data
		for (size_t i = 0; i < edgeVertices.size(); ++i) {
			optimalPosition.x += edgeVertices[i].x;
			optimalPosition.y += edgeVertices[i].y;
			optimalPosition.z += edgeVertices[i].z;

			optimalNormal.x += edgeNormals[i].x;
			optimalNormal.y += edgeNormals[i].y;
			optimalNormal.z += edgeNormals[i].z;
		}

		optimalPosition.x /= edgeVertices.size();
		optimalPosition.y /= edgeVertices.size();
		optimalPosition.z /= edgeVertices.size();

		optimalNormal = normalize(optimalNormal);

		// Return the optimal vertex
		Vertex vertex;
		vertex.position = optimalPosition;
		vertex.normal = optimalNormal;
		vertex.color = make_float3(0.5f, 0.5f, 0.5f); // Set a default color or compute based on some criteria.

		return vertex;
	}

	// 각 코너의 오프셋을 구하는 함수 (Marching Cubes용)
	/*uint3 getCornerOffset(int cornerIndex)
	{
		static const uint3 offsets[8] = {
			make_uint3(0, 0, 0), make_uint3(1, 0, 0),
			make_uint3(1, 1, 0), make_uint3(0, 1, 0),
			make_uint3(0, 0, 1), make_uint3(1, 0, 1),
			make_uint3(1, 1, 1), make_uint3(0, 1, 1)
		};
		return offsets[cornerIndex];
	}*/

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

		//divergenceOutput[flatIndex] = dx + dy + dz;

		if (fabs(dx) > 0.001f || fabs(dy) > 0.001f || fabs(dz) > 0.001f) {
			divergenceOutput[flatIndex] = dx + dy + dz;
		}
		else {
			divergenceOutput[flatIndex] = 0.0f;
		}
	}

	__global__ void ComputeScreenedPotential(
		RegularGrid<Voxel>::Internal* regularGrid,
		float* divergence,
		float* inputField,
		float* screenedField,
		float lambda)
	{
		size_t threadX = blockIdx.x * blockDim.x + threadIdx.x;
		size_t threadY = blockIdx.y * blockDim.y + threadIdx.y;
		size_t threadZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (threadX >= regularGrid->dimensions.x ||
			threadY >= regularGrid->dimensions.y ||
			threadZ >= regularGrid->dimensions.z) return;

		size_t flatIndex = threadZ * (regularGrid->dimensions.x * regularGrid->dimensions.y) +
			threadY * regularGrid->dimensions.x + threadX;

		float divergenceValue = divergence[flatIndex];
		float inputValue = inputField[flatIndex];

		// 스크리닝 조건에 따라 잠재 필드 업데이트
		screenedField[flatIndex] = divergenceValue + lambda * inputValue;
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

	void SolveScreenedPoissonEquation(
		RegularGrid<Voxel>::Internal* regularGrid,
		float* divergence,
		float* inputField,
		float* potentialField,
		float lambda)
	{
		// cusolver 및 cusparse 핸들 생성
		cusolverSpHandle_t cusolverHandle;
		cusparseMatDescr_t descrA;

		cusolverSpCreate(&cusolverHandle);
		cusparseCreateMatDescr(&descrA);
		cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

		// 잠재 필드 초기화
		int numRows = regularGrid->dimensions.x * regularGrid->dimensions.y * regularGrid->dimensions.z;
		checkCudaErrors(cudaMemset(potentialField, 0, sizeof(float) * numRows));
		checkCudaErrors(cudaMemset(inputField, 0, sizeof(float) * numRows));

		// 경계 조건 설정: 경계 부분의 잠재 필드를 고정하여 안정화
		for (uint32_t z = 0; z < regularGrid->dimensions.z; ++z) {
			for (uint32_t y = 0; y < regularGrid->dimensions.y; ++y) {
				for (uint32_t x = 0; x < regularGrid->dimensions.x; ++x) {
					if (x == 0 || y == 0 || z == 0 ||
						x == regularGrid->dimensions.x - 1 ||
						y == regularGrid->dimensions.y - 1 ||
						z == regularGrid->dimensions.z - 1) {
						size_t flatIndex = GetFlatIndex(make_uint3(x, y, z), regularGrid->dimensions);
						potentialField[flatIndex] = 0.0f; // 경계 조건 값 설정
					}
				}
			}
		}

		// CUDA 커널을 사용하여 스크리닝 잠재 필드 계산
		dim3 threadsPerBlock(8, 8, 8);
		dim3 blocksPerGrid(
			(regularGrid->dimensions.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(regularGrid->dimensions.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
			(regularGrid->dimensions.z + threadsPerBlock.z - 1) / threadsPerBlock.z
		);

		float* screenedField;
		checkCudaErrors(cudaMallocManaged(&screenedField, sizeof(float) * numRows));
		checkCudaErrors(cudaMemset(screenedField, 0, sizeof(float) * numRows));

		// 스크리닝 잠재 필드 계산
		ComputeScreenedPotential << <blocksPerGrid, threadsPerBlock >> > (
			regularGrid, divergence, inputField, screenedField, lambda
			);
		checkCudaErrors(cudaDeviceSynchronize());

		// 경계 조건 유지 (ComputeScreenedPotential 이후에도 경계값은 고정해야 함)
		for (uint32_t z = 0; z < regularGrid->dimensions.z; ++z) {
			for (uint32_t y = 0; y < regularGrid->dimensions.y; ++y) {
				for (uint32_t x = 0; x < regularGrid->dimensions.x; ++x) {
					if (x == 0 || y == 0 || z == 0 ||
						x == regularGrid->dimensions.x - 1 ||
						y == regularGrid->dimensions.y - 1 ||
						z == regularGrid->dimensions.z - 1) {
						size_t flatIndex = GetFlatIndex(make_uint3(x, y, z), regularGrid->dimensions);
						screenedField[flatIndex] = 0.0f; // 경계 조건 값 설정
					}
				}
			}
		}

		// Poisson 방정식을 풀기 위한 cuSolver 설정
		float tol = 1e-6f;
		int reorder = 0;
		int singularity = 0;

		// CSR 포맷을 위한 메모리 할당
		int nnz = numRows * 7; // 7-point stencil
		float* d_A_values;
		int* d_A_rowPtr;
		int* d_A_colInd;
		float* d_x;

		checkCudaErrors(cudaMalloc((void**)&d_A_values, sizeof(float) * nnz));
		checkCudaErrors(cudaMalloc((void**)&d_A_rowPtr, sizeof(int) * (numRows + 1)));
		checkCudaErrors(cudaMalloc((void**)&d_A_colInd, sizeof(int) * nnz));
		checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(float) * numRows));

		// cuSolver를 사용하여 스크린드 포아송 방정식 풀기
		cusolverSpScsrlsvqr(
			cusolverHandle, numRows, nnz,
			descrA, d_A_values, d_A_rowPtr, d_A_colInd,
			screenedField, tol, reorder, d_x, &singularity
		);

		if (singularity >= 0) {
			printf("WARNING: A is singular at row %d\n", singularity);
		}

		// 결과를 잠재 필드에 복사
		checkCudaErrors(cudaMemcpy(potentialField, d_x, sizeof(float) * numRows, cudaMemcpyDeviceToDevice));

		// 메모리 해제
		cusolverSpDestroy(cusolverHandle);
		cusparseDestroyMatDescr(descrA);
		checkCudaErrors(cudaFree(screenedField));
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

		//if (normal.z < 0) return;

		auto currentIndex = GetIndex(regularGrid->center, regularGrid->dimensions, regularGrid->voxelSize, p);
		if (currentIndex.x == UINT_MAX || currentIndex.y == UINT_MAX || currentIndex.z == UINT_MAX) return;

		int offset = 2;
		for (uint32_t nzi = currentIndex.z - offset; nzi < currentIndex.z + offset; nzi++)
		{
			if (currentIndex.z < offset || nzi >= regularGrid->dimensions.z) continue;

			for (uint32_t nyi = currentIndex.y - offset; nyi < currentIndex.y + offset; nyi++)
			{
				if (currentIndex.y < offset || nyi >= regularGrid->dimensions.y) continue;

				for (uint32_t nxi = currentIndex.x - offset; nxi < currentIndex.x + offset; nxi++)
				{
					if (currentIndex.x < offset || nxi >= regularGrid->dimensions.x) continue;

					size_t index = nzi * (regularGrid->dimensions.x * regularGrid->dimensions.y) +
						nyi * regularGrid->dimensions.x + nxi;

					float3 voxelCenter = GetPosition(regularGrid->center, regularGrid->dimensions, regularGrid->voxelSize, make_uint3(nxi, nyi, nzi));

					float distance = length(p - voxelCenter);

					float signedDistance = dot(normal, voxelCenter - p) < 0 ? -distance : distance;

					float tsdfValue = signedDistance / regularGrid->truncationDistance;
					tsdfValue = fminf(1.0f, fmaxf(-1.0f, tsdfValue));

					Voxel* voxel = &(regularGrid->elements[index]);

					float newWeight = 1.0f;

					float previousTsdf = voxel->tsdfValue;
					float previousWeight = voxel->weight;

					// Weighted average for TSDF
					float updatedWeight = fminf(previousWeight + newWeight, regularGrid->maxWeight); // 최대 가중치 제한
					float updatedTsdf = (previousTsdf * previousWeight + tsdfValue * newWeight) / updatedWeight;

					updatedTsdf = fmax(-regularGrid->truncationDistance, fmin(regularGrid->truncationDistance, updatedTsdf));

					voxel->tsdfValue = updatedTsdf;
					voxel->weight = updatedWeight;

					// 컬러 및 노멀 업데이트
					voxel->color = make_float3(
						fminf(1.0f, fmaxf(0.0f, (voxel->color.x * previousWeight + color.x * newWeight) / updatedWeight)),
						fminf(1.0f, fmaxf(0.0f, (voxel->color.y * previousWeight + color.y * newWeight) / updatedWeight)),
						fminf(1.0f, fmaxf(0.0f, (voxel->color.z * previousWeight + color.z * newWeight) / updatedWeight))
					);
					voxel->normal = normalize(make_float3(
						(voxel->normal.x * previousWeight + normal.x * newWeight) / updatedWeight,
						(voxel->normal.y * previousWeight + normal.y * newWeight) / updatedWeight,
						(voxel->normal.z * previousWeight + normal.z * newWeight) / updatedWeight
					));

					voxel->normal = normalize(voxel->normal);
				}
			}
		}

		//if (gridIndex.x == UINT_MAX || gridIndex.y == UINT_MAX || gridIndex.z == UINT_MAX) return;

		//size_t flatIndex = gridIndex.z * (regularGrid->dimensions.x * regularGrid->dimensions.y) +
		//	gridIndex.y * regularGrid->dimensions.x + gridIndex.x;

		//float3 voxelCenter = GetPosition(regularGrid->center, regularGrid->dimensions, regularGrid->voxelSize, gridIndex);

		//float distance = length(p - voxelCenter);

		//float signedDistance = dot(normal, voxelCenter - p) < 0 ? -distance : distance;

		//float tsdfValue = signedDistance / regularGrid->truncationDistance;
		//tsdfValue = fminf(1.0f, fmaxf(-1.0f, tsdfValue));

		//Voxel* voxel = &(regularGrid->elements[flatIndex]);

		//float newWeight = 1.0f;

		//float previousTsdf = voxel->tsdfValue;
		//float previousWeight = voxel->weight;

		//// Weighted average for TSDF
		//float updatedWeight = fminf(previousWeight + newWeight, regularGrid->maxWeight); // 최대 가중치 제한
		//float updatedTsdf = (previousTsdf * previousWeight + tsdfValue * newWeight) / updatedWeight;

		//voxel->tsdfValue = updatedTsdf;
		//voxel->weight = updatedWeight;

		//// 컬러 및 노멀 업데이트
		//voxel->color = make_float3(
		//	fminf(1.0f, fmaxf(0.0f, (voxel->color.x * previousWeight + color.x * newWeight) / updatedWeight)),
		//	fminf(1.0f, fmaxf(0.0f, (voxel->color.y * previousWeight + color.y * newWeight) / updatedWeight)),
		//	fminf(1.0f, fmaxf(0.0f, (voxel->color.z * previousWeight + color.z * newWeight) / updatedWeight))
		//);
		//voxel->normal = normalize(make_float3(
		//	(voxel->normal.x * previousWeight + normal.x * newWeight) / updatedWeight,
		//	(voxel->normal.y * previousWeight + normal.y * newWeight) / updatedWeight,
		//	(voxel->normal.z * previousWeight + normal.z * newWeight) / updatedWeight
		//));

		//voxel->normal = normalize(voxel->normal);

		//printf("voxel->tsdfValue : %f\n", voxel->tsdfValue);

		//if (-0.05f <= voxel->tsdfValue && voxel->tsdfValue <= 0.05f)
		//if (-0.2f <= voxel->tsdfValue && voxel->tsdfValue <= 0.2f)
		//{
		//	printf("%f, %f, %f\n", p.x, p.y, p.z);
		//}
	}

	template<typename T>
	__global__ void Kernel_SmoothTSDF(RegularGrid<T>::Internal* regularGrid, T* smoothedElements)
	{
		// Calculate thread coordinates
		size_t threadX = blockIdx.x * blockDim.x + threadIdx.x;
		size_t threadY = blockIdx.y * blockDim.y + threadIdx.y;
		size_t threadZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (threadX >= regularGrid->dimensions.x ||
			threadY >= regularGrid->dimensions.y ||
			threadZ >= regularGrid->dimensions.z) return;

		// Current flat index for the voxel
		size_t flatIndex = GetFlatIndex(make_uint3(threadX, threadY, threadZ), regularGrid->dimensions);

		// Initialize smoothing variables
		float tsdfSum = 0.0f;
		float weightSum = 0.0f;
		float3 colorSum = make_float3(0.0f, 0.0f, 0.0f);
		float3 normalSum = make_float3(0.0f, 0.0f, 0.0f);
		int count = 0;

		// Define smoothing radius (e.g., 1 for a simple box filter)
		const int radius = 1;

		// Iterate over the neighboring voxels within the smoothing radius
		for (int zOffset = -radius; zOffset <= radius; ++zOffset)
		{
			for (int yOffset = -radius; yOffset <= radius; ++yOffset)
			{
				for (int xOffset = -radius; xOffset <= radius; ++xOffset)
				{
					int neighborX = threadX + xOffset;
					int neighborY = threadY + yOffset;
					int neighborZ = threadZ + zOffset;

					// Skip out-of-bounds neighbors
					if (neighborX < 0 || neighborX >= regularGrid->dimensions.x ||
						neighborY < 0 || neighborY >= regularGrid->dimensions.y ||
						neighborZ < 0 || neighborZ >= regularGrid->dimensions.z)
					{
						continue;
					}

					// Get the neighbor index
					uint3 neighborIndex = make_uint3(neighborX, neighborY, neighborZ);
					size_t neighborFlatIndex = GetFlatIndex(neighborIndex, regularGrid->dimensions);

					// Fetch the neighbor voxel
					Voxel neighborVoxel = regularGrid->elements[neighborFlatIndex];

					// Accumulate TSDF values, weights, and colors
					tsdfSum += neighborVoxel.tsdfValue * neighborVoxel.weight;
					weightSum += neighborVoxel.weight;
					colorSum.x += neighborVoxel.color.x * neighborVoxel.weight;
					colorSum.y += neighborVoxel.color.y * neighborVoxel.weight;
					colorSum.z += neighborVoxel.color.z * neighborVoxel.weight;
					normalSum += neighborVoxel.normal;

					++count;
				}
			}
		}

		// Calculate smoothed values for the current voxel
		Voxel smoothedVoxel;
		if (weightSum > 0.0f)
		{
			smoothedVoxel.tsdfValue = tsdfSum / weightSum;
			smoothedVoxel.weight = weightSum / count; // Normalizing by number of neighbors considered
			smoothedVoxel.color = make_float3(colorSum.x / weightSum, colorSum.y / weightSum, colorSum.z / weightSum);
		}
		else
		{
			// If there was no effective weight, keep the original value
			smoothedVoxel.tsdfValue = regularGrid->elements[flatIndex].tsdfValue;
			smoothedVoxel.weight = regularGrid->elements[flatIndex].weight;
			smoothedVoxel.color = regularGrid->elements[flatIndex].color;
		}

		smoothedVoxel.normal = normalize(normalSum);

		// Write the smoothed voxel to the output buffer
		smoothedElements[flatIndex] = smoothedVoxel;
	}

	template<typename T>
	__global__ void Kernel_ExtractMesh(typename RegularGrid<T>::Internal* regularGrid, Vertex* meshVertices, int* vertexCount)
	{
		size_t threadX = blockIdx.x * blockDim.x + threadIdx.x;
		size_t threadY = blockIdx.y * blockDim.y + threadIdx.y;
		size_t threadZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (threadX >= regularGrid->dimensions.x - 1 ||
			threadY >= regularGrid->dimensions.y - 1 ||
			threadZ >= regularGrid->dimensions.z - 1) return;

		uint3 currentIndex = make_uint3(threadX, threadY, threadZ);
		size_t flatIndex = GetFlatIndex(currentIndex, regularGrid->dimensions);

		float tsdfValues[8];
		float3 positions[8];
		float3 normals[8];

		uint3 offsets[8] = {
		{0, 0, 0}, {1, 0, 0},
		{1, 1, 0}, {0, 1, 0},
		{0, 0, 1}, {1, 0, 1},
		{1, 1, 1}, {0, 1, 1}
		};

		for (int i = 0; i < 8; ++i)
		{
			uint3 cornerOffset = offsets[i];
			uint3 cornerIndex = make_uint3(currentIndex.x + cornerOffset.x, currentIndex.y + cornerOffset.y, currentIndex.z + cornerOffset.z);
			size_t cornerFlatIndex = GetFlatIndex(cornerIndex, regularGrid->dimensions);

			tsdfValues[i] = regularGrid->elements[cornerFlatIndex].tsdfValue;
			positions[i] = GetPosition(regularGrid->center, regularGrid->dimensions, regularGrid->voxelSize, cornerIndex);
			normals[i] = regularGrid->elements[cornerFlatIndex].normal;
		}

		int cubeIndex = 0;
		for (int i = 0; i < 8; ++i)
		{
			if (tsdfValues[i] < 0.0f)
			{
				cubeIndex |= (1 << i);
			}
		}

		if (cubeIndex == 0 || cubeIndex == 255)
		{
			return;
		}

		for (int i = 0; triTable[cubeIndex][i] != -1; i += 3)
		{
			int edgeA = triTable[cubeIndex][i];
			int edgeB = triTable[cubeIndex][i + 1];
			int edgeC = triTable[cubeIndex][i + 2];

			Vertex vertexA = interpolateVertex(edgeA, positions, tsdfValues);
			Vertex vertexB = interpolateVertex(edgeB, positions, tsdfValues);
			Vertex vertexC = interpolateVertex(edgeC, positions, tsdfValues);

			int idx = atomicAdd(vertexCount, 3);
			meshVertices[idx] = vertexA;
			meshVertices[idx + 1] = vertexB;
			meshVertices[idx + 2] = vertexC;
		}
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

	__host__ __device__
		size_t GetFlatIndex(const uint3& index, const uint3& dimensions)
	{
		return index.z * dimensions.x * dimensions.y + index.y * dimensions.x + index.x;
	}

	void SaveRegularGridToVTK(const CUDA::RegularGrid<CUDA::Voxel>& rg, const std::string& filename) {
		auto& internal = *(rg.internal);

		// 파일 열기
		std::ofstream vtkFile(filename);
		if (!vtkFile.is_open()) {
			std::cerr << "Error: Cannot open file " << filename << std::endl;
			return;
		}

		// VTK 헤더 작성
		vtkFile << "# vtk DataFile Version 4.2\n";
		vtkFile << "vtk output\n";
		vtkFile << "ASCII\n";
		vtkFile << "DATASET STRUCTURED_POINTS\n";
		vtkFile << "DIMENSIONS " << rg.h_internal.dimensions.x << " "
			<< rg.h_internal.dimensions.y << " " << rg.h_internal.dimensions.z << "\n";
		vtkFile << "SPACING " << rg.h_internal.voxelSize << " "
			<< rg.h_internal.voxelSize << " " << rg.h_internal.voxelSize << "\n";
		vtkFile << "ORIGIN 0.0 0.0 0.0\n";
		vtkFile << "POINT_DATA " << (rg.h_internal.dimensions.x * rg.h_internal.dimensions.y * rg.h_internal.dimensions.z) << "\n";
		vtkFile << "SCALARS TSDF float 1\n";
		vtkFile << "LOOKUP_TABLE default\n";

		CUDA::Voxel* elements = new CUDA::Voxel[rg.h_internal.numberOfVoxels];
		cudaMemcpy(elements, rg.h_internal.elements, sizeof(CUDA::Voxel) * rg.h_internal.numberOfVoxels, cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaDeviceSynchronize());

		// TSDF 값 저장
		for (uint32_t z = 0; z < rg.h_internal.dimensions.z; ++z) {
			for (uint32_t y = 0; y < rg.h_internal.dimensions.y; ++y) {
				for (uint32_t x = 0; x < rg.h_internal.dimensions.x; ++x) {
					size_t flatIndex = z * rg.h_internal.dimensions.x * rg.h_internal.dimensions.y
						+ y * rg.h_internal.dimensions.x + x;
					vtkFile << elements[flatIndex].tsdfValue << "\n";
				}
			}
		}

		delete[] elements;
		vtkFile.close();
		std::cout << "VTK file saved to " << filename << std::endl;
	}


	////////////////////////////////////////////////////////////////////////////////////

	void TestRegularGrid()
	{
		auto t = Time::Now();

		RegularGrid<Voxel> rg({ 0.0f, 0.0f, 0.0f }, 200, 200, 200, 0.1f);

		t = Time::End(t, "RegularGrid allocation");

		rg.Clear();

		t = Time::End(t, "RegularGrid Clear");

		{
			t = Time::Now();

			stringstream ss;
			//ss << "C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply";
			ss << "D:\\Resources\\3D\\PLY\\Lower_pointcloud_crop.ply";

			PLYFormat ply;
			ply.Deserialize(ss.str());
			cout << "ply min : " << ply.GetAABB().min().transpose() << endl;
			cout << "ply max : " << ply.GetAABB().max().transpose() << endl;

			t = Time::End(t, "Load ply");

			//for (size_t i = 0; i < ply.GetNumberOfPoints() / 3; i++)
			//{

			//}

			PatchBuffers patchBuffers(ply.GetPoints().size() / 3, 1);
			patchBuffers.FromPLYFile(ply);

			t = Time::End(t, "Copy data to device");

			rg.Integrate(patchBuffers);

			t = Time::End(t, "Insert using PatchBuffers");

			/*rg.SmoothTSDF();

			t = Time::End(t, "SmoothTSDF");*/
		}

		SaveRegularGridToVTK(rg, "D:\\Resources\\3D\\VTK\\rg.vtk");

#ifdef MARCHING_CUBES
		{
			//// VisualDebugging를 사용해 메시를 그리기 위한 코드
			//for (const auto& vertex : mesh)
			//{
			//	VD::AddCube("ExtractedMesh",
			//		{ vertex.position.x, vertex.position.y, vertex.position.z },
			//		{ 0.01f, 0.01f, 0.01f },
			//		{ vertex.normal.x, vertex.normal.y, vertex.normal.z },
			//		Color4::FromNormalized(vertex.color.x, vertex.color.y, vertex.color.z, 1.0f));
			//}

			//t = Time::End(t, "Mesh Extraction and Visualization");
		}
#endif // MARCHING_CUBES

#ifdef SHOW_VOXELS
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
							VD::AddCube("voxels", { p.x + 20.0f, p.y, p.z },
								{ rg.internal->voxelSize * 0.5f, rg.internal->voxelSize * 0.5f,rg.internal->voxelSize * 0.5f },
								{ n.x, n.y, n.z }, c4);
						}
					}
				}
			}

			t = Time::End(t, "?????");
		}
#endif // SHOW_VOXELS

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

#ifdef POISSON_SURFACE_RECONSTRUCTION
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

			//for (size_t i = 0; i < rg.internal->numberOfVoxels; ++i) {
			//	printf("Potential field value at voxel %zu: %f\n", i, divergence[i]);
			//}


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
#endif // POISSON_SURFACE_RECONSTRUCTION

#ifdef SCREENED_POISSON_SURFACE_RECONSTRUCTION
		{
			dim3 threadsPerBlock(8, 8, 8);
			dim3 blocksPerGrid(
				(rg.internal->dimensions.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
				(rg.internal->dimensions.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
				(rg.internal->dimensions.z + threadsPerBlock.z - 1) / threadsPerBlock.z
			);

			// 다이버전스 계산
			float* divergence;
			checkCudaErrors(cudaMallocManaged(&divergence, sizeof(float) * rg.internal->numberOfVoxels));
			ComputeDivergence<Voxel> << <blocksPerGrid, threadsPerBlock >> > (rg.internal, divergence);
			checkCudaErrors(cudaDeviceSynchronize());

			// Screened Poisson 방정식 해결
			float* potentialField;
			float* inputField;  // 원 데이터 잠재 필드
			float lambda = 0.1f;  // 스크리닝 강도

			checkCudaErrors(cudaMallocManaged(&potentialField, sizeof(float) * rg.internal->numberOfVoxels));
			checkCudaErrors(cudaMallocManaged(&inputField, sizeof(float) * rg.internal->numberOfVoxels));

			SolveScreenedPoissonEquation(rg.internal, divergence, inputField, potentialField, lambda);

			// 메시 추출
			std::vector<Vertex> mesh = rg.ExtractMeshFromPotentialField(potentialField);
			//std::vector<Vertex> mesh = rg.ExtractMesh();
			//std::vector<Vertex> mesh = rg.ExtractMeshUsingDualContouring();

			// 메시 시각화
			for (size_t i = 0; i < mesh.size() / 3; i++)
			{
				auto& v0 = mesh[i * 3];
				auto& v1 = mesh[i * 3 + 1];
				auto& v2 = mesh[i * 3 + 2];

				VD::AddTriangle("mesh",
					{ v0.position.x, v0.position.y, v0.position.z },
					{ v1.position.x, v1.position.y, v1.position.z },
					{ v2.position.x, v2.position.y, v2.position.z },
					Color4::FromNormalized(v0.color.x, v0.color.y, v0.color.z, 1.0f)
				);
			}

			// 메모리 해제
			checkCudaErrors(cudaFree(divergence));
			checkCudaErrors(cudaFree(potentialField));
			checkCudaErrors(cudaFree(inputField));
		}
#endif // 

		{
			t = Time::Now();

			std::vector<Vertex> mesh = rg.ExtractMeshCUDA();

			t = Time::End(t, "ExtractMeshCUDA");

			printf("size of mesh : %llu\n", mesh.size());

			for (size_t i = 0; i < mesh.size() / 3; i++)
			{
				auto v0 = mesh[i * 3];
				auto v1 = mesh[i * 3 + 1];
				auto v2 = mesh[i * 3 + 2];

				VD::AddTriangle("mesh",
					{ v0.position.x, v0.position.y, v0.position.z },
					{ v1.position.x, v1.position.y, v1.position.z },
					{ v2.position.x, v2.position.y, v2.position.z },
					Color4::White);
			}
		}
	}

}
