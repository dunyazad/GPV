#include "PSR.cuh"

#include <vtkHeaderFiles.h>

#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
	namespace PSR
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
				ply.GetNormals().data(),
				sizeof(Eigen::Vector3f) * numberOfInputPoints,
				cudaMemcpyHostToDevice);

			cudaMemcpy(thrust::raw_pointer_cast(inputColors.data()),
				ply.GetColors().data(),
				sizeof(Eigen::Vector3f) * numberOfInputPoints,
				cudaMemcpyHostToDevice);

			//for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
			//{
			//	auto px = ply.GetPoints()[i * 3 + 0];
			//	auto py = ply.GetPoints()[i * 3 + 1];
			//	auto pz = ply.GetPoints()[i * 3 + 2];

			//	auto nx = ply.GetNormals()[i * 3 + 0];
			//	auto ny = ply.GetNormals()[i * 3 + 1];
			//	auto nz = ply.GetNormals()[i * 3 + 2];

			//	auto cx = ply.GetColors()[i * 3 + 0];
			//	auto cy = ply.GetColors()[i * 3 + 1];
			//	auto cz = ply.GetColors()[i * 3 + 2];

			//	auto c4 = Color4::FromNormalized(cx, cy, cz, 1.0f);
			//	VD::AddSphere("points", { px, py, pz }, { 0.05f, 0.05f, 0.05f }, { nx, ny, nz }, c4);
			//}
		}
#pragma endregion

		struct Voxel
		{
			Eigen::Vector3f normal;
			int weight;
			__host__ __device__ Voxel() : normal(0.0f, 0.0f, 0.0f), weight(0) {}
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

		__host__ __device__
			uint3 GetIndex(const Eigen::Vector3f& gridCenter, uint3 gridDimensions, float voxelSize, const Eigen::Vector3f& position)
		{
			Eigen::Vector3f halfGridSize = Eigen::Vector3f(
				(float)gridDimensions.x * voxelSize * 0.5f,
				(float)gridDimensions.y * voxelSize * 0.5f,
				(float)gridDimensions.z * voxelSize * 0.5f
			);

			Eigen::Vector3f gridMin = gridCenter - halfGridSize;
			Eigen::Vector3f relativePosition = position - gridMin;

			uint3 index = make_uint3(UINT_MAX, UINT_MAX, UINT_MAX);

			if (relativePosition.x() < 0.0f || relativePosition.x() >= (float)gridDimensions.x * voxelSize ||
				relativePosition.y() < 0.0f || relativePosition.y() >= (float)gridDimensions.y * voxelSize ||
				relativePosition.z() < 0.0f || relativePosition.z() >= (float)gridDimensions.z * voxelSize)
			{
				return index;
			}
			else
			{
				index.x = (uint32_t)floorf(relativePosition.x() / voxelSize);
				index.y = (uint32_t)floorf(relativePosition.y() / voxelSize);
				index.z = (uint32_t)floorf(relativePosition.z() / voxelSize);
			}

			return index;
		}

		__host__ __device__
			Eigen::Vector3f GetPosition(const Eigen::Vector3f& gridCenter, uint3 gridDimensions, float voxelSize, const uint3& index)
		{
			Eigen::Vector3f halfGridSize = Eigen::Vector3f(
				(float)gridDimensions.x * voxelSize * 0.5f,
				(float)gridDimensions.y * voxelSize * 0.5f,
				(float)gridDimensions.z * voxelSize * 0.5f
			);

			Eigen::Vector3f gridMin = gridCenter - halfGridSize;

			// Calculate the position of the given voxel using the provided index
			Eigen::Vector3f position = Eigen::Vector3f(
				gridMin.x() + (float)index.x * voxelSize/* + voxelSize * 0.5f*/,
				gridMin.y() + (float)index.y * voxelSize/* + voxelSize * 0.5f*/,
				gridMin.z() + (float)index.z * voxelSize/* + voxelSize * 0.5f*/
			);

			return position;
		}

		__host__ __device__
			size_t GetFlatIndex(const uint3& index, const uint3& dimensions)
		{
			return index.z * dimensions.x * dimensions.y + index.y * dimensions.x + index.x;
		}

		cusolverSpHandle_t cusolverHandle;

		void InitializeCuSolver()
		{
			// cuSolver 핸들 초기화
			cusolverStatus_t status = cusolverSpCreate(&cusolverHandle);
			if (status != CUSOLVER_STATUS_SUCCESS) {
				cerr << "Failed to create cuSolver handle" << endl;
				exit(EXIT_FAILURE);
			}
		}

		void DestroyCuSolver()
		{
			// cuSolver 핸들 해제
			cusolverSpDestroy(cusolverHandle);
		}

		void TestPSR()
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

			Eigen::Vector3f total_min(-17.5f, -17.5f, -17.5f);
			Eigen::Vector3f total_max(17.5f, 17.5f, 17.5f);
			//Eigen::Vector3f total_min(-50.0f, -50.0f, -50.0f);
			//Eigen::Vector3f total_max(50.0f, 50.0f, 50.0f);
			Eigen::Vector3f total_diff = total_max - total_min;
			Eigen::Vector3f total_center = (total_max + total_min) * 0.5f;
			float voxelSize = 0.1f;

			uint3 total_dimensions;
			total_dimensions.x = (uint32_t)ceilf(total_diff.x() / voxelSize);
			total_dimensions.y = (uint32_t)ceilf(total_diff.y() / voxelSize);
			total_dimensions.z = (uint32_t)ceilf(total_diff.z() / voxelSize);

			thrust::device_vector<Voxel> volume(total_dimensions.x / 2 * total_dimensions.y / 2 * total_dimensions.z / 2);
			auto d_volume = thrust::raw_pointer_cast(volume.data());
			thrust::device_vector<float> divergences(total_dimensions.x / 2 * total_dimensions.y / 2 * total_dimensions.z / 2);
			auto d_divergences = thrust::raw_pointer_cast(divergences.data());
			thrust::device_vector<float> potentials(total_dimensions.x / 2 * total_dimensions.y / 2 * total_dimensions.z / 2);
			auto d_potentials = thrust::raw_pointer_cast(potentials.data());

			for (size_t i = 0; i < 8; i++)
			{
				float minX = 0.0f;
				float maxX = 0.0f;
				float minY = 0.0f;
				float maxY = 0.0f;
				float minZ = 0.0f;
				float maxZ = 0.0f;

				if (i & 0b001)
				{
					minX = total_center.x();
					maxX = total_max.x();
				}
				else
				{
					minX = total_min.x();
					maxX = total_center.x();
				}
				if (i & 0b010)
				{
					minY = total_center.y();
					maxY = total_max.y();
				}
				else
				{
					minY = total_min.y();
					maxY = total_center.y();
				}
				if (i & 0b100)
				{
					minZ = total_center.y();
					maxZ = total_max.y();
				}
				else
				{
					minZ = total_min.y();
					maxZ = total_center.y();
				}

				Eigen::Vector3f min(minX, minY, minZ);
				Eigen::Vector3f max(maxX, maxY, maxZ);
				Eigen::Vector3f diff = max - min;
				Eigen::Vector3f center = (max + min) * 0.5f;

				uint3 dimensions;
				dimensions.x = (uint32_t)ceilf(diff.x() / voxelSize);
				dimensions.y = (uint32_t)ceilf(diff.y() / voxelSize);
				dimensions.z = (uint32_t)ceilf(diff.z() / voxelSize);

				thrust::for_each(thrust::counting_iterator((size_t)0), thrust::counting_iterator(volume.size()),
					[=] __device__(size_t index) {
					d_volume[index].normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
					d_volume[index].weight = 0;
					d_divergences[index] = 0.0f;
					d_potentials[index] = 0.0f;
				});

				{
					t = Time::Now();
					nvtxRangePushA("Insert Points");
					thrust::for_each(
						thrust::make_zip_iterator(thrust::make_tuple(patchBuffers.inputPoints.begin(), patchBuffers.inputNormals.begin())),
						thrust::make_zip_iterator(thrust::make_tuple(patchBuffers.inputPoints.end(), patchBuffers.inputNormals.end())),
						[=] __device__(thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> t) {
						Eigen::Vector3f point = thrust::get<0>(t);
						Eigen::Vector3f normal = thrust::get<1>(t);

						auto index = GetIndex(center, dimensions, voxelSize, point);
						if (index.x == UINT_MAX || index.y == UINT_MAX || index.z == UINT_MAX) return;

						auto flatIndex = GetFlatIndex(index, dimensions);

						atomicAdd(&(d_volume[flatIndex].normal.x()), normal.x());
						atomicAdd(&(d_volume[flatIndex].normal.y()), normal.y());
						atomicAdd(&(d_volume[flatIndex].normal.z()), normal.z());
						atomicAdd(&(d_volume[flatIndex].weight), 1);
					});
					nvtxRangePop();
					t = Time::End(t, "Insert Points");
				}

				{
					t = Time::Now();
					nvtxRangePushA("Compute Divergence");
					thrust::for_each(thrust::counting_iterator((size_t)0), thrust::counting_iterator(volume.size()),
						[=] __device__(size_t index) {
						size_t indexZ = index / (dimensions.y * dimensions.x);
						size_t indexY = (index % (dimensions.y * dimensions.x)) / dimensions.x;
						size_t indexX = (index % (dimensions.y * dimensions.x)) % dimensions.x;

						Voxel cv = d_volume[index];  // 현재 복셀 값 복사

						// 경계 조건에 따른 이전/다음 인덱스 계산
						size_t piX = (indexX > 0) ? indexX - 1 : indexX;
						size_t niX = (indexX < dimensions.x - 1) ? indexX + 1 : indexX;
						size_t piY = (indexY > 0) ? indexY - 1 : indexY;
						size_t niY = (indexY < dimensions.y - 1) ? indexY + 1 : indexY;
						size_t piZ = (indexZ > 0) ? indexZ - 1 : indexZ;
						size_t niZ = (indexZ < dimensions.z - 1) ? indexZ + 1 : indexZ;

						// 이전 복셀과 다음 복셀의 인덱스와 데이터 가져오기
						size_t flatIndexX1 = GetFlatIndex(make_uint3(piX, indexY, indexZ), dimensions);
						size_t flatIndexX2 = GetFlatIndex(make_uint3(niX, indexY, indexZ), dimensions);
						size_t flatIndexY1 = GetFlatIndex(make_uint3(indexX, piY, indexZ), dimensions);
						size_t flatIndexY2 = GetFlatIndex(make_uint3(indexX, niY, indexZ), dimensions);
						size_t flatIndexZ1 = GetFlatIndex(make_uint3(indexX, indexY, piZ), dimensions);
						size_t flatIndexZ2 = GetFlatIndex(make_uint3(indexX, indexY, niZ), dimensions);

						// 이웃 복셀의 노멀 벡터와 가중치를 확인하여 유효한 값만 사용
						Eigen::Vector3f normX1 = d_volume[flatIndexX1].weight > 0 ? d_volume[flatIndexX1].normal / (float)d_volume[flatIndexX1].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normX2 = d_volume[flatIndexX2].weight > 0 ? d_volume[flatIndexX2].normal / (float)d_volume[flatIndexX2].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normY1 = d_volume[flatIndexY1].weight > 0 ? d_volume[flatIndexY1].normal / (float)d_volume[flatIndexY1].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normY2 = d_volume[flatIndexY2].weight > 0 ? d_volume[flatIndexY2].normal / (float)d_volume[flatIndexY2].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normZ1 = d_volume[flatIndexZ1].weight > 0 ? d_volume[flatIndexZ1].normal / (float)d_volume[flatIndexZ1].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normZ2 = d_volume[flatIndexZ2].weight > 0 ? d_volume[flatIndexZ2].normal / (float)d_volume[flatIndexZ2].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);

						// 발산 계산 (중심 차분 방식 - 경계에서는 단측 차분 사용)
						float divX = 0.0f, divY = 0.0f, divZ = 0.0f;

						if (indexX > 0 && indexX < dimensions.x - 1) {
							divX = (normX2.x() - normX1.x()) / (2.0f * voxelSize);
						}
						else if (indexX == 0) {
							divX = (normX2.x() - cv.normal.x()) / voxelSize;
						}
						else if (indexX == dimensions.x - 1) {
							divX = (cv.normal.x() - normX1.x()) / voxelSize;
						}

						// Y와 Z 축에 대해서도 동일하게 처리
						if (indexY > 0 && indexY < dimensions.y - 1) {
							divY = (normY2.y() - normY1.y()) / (2.0f * voxelSize);
						}
						else if (indexY == 0) {
							divY = (normY2.y() - cv.normal.y()) / voxelSize;
						}
						else if (indexY == dimensions.y - 1) {
							divY = (cv.normal.y() - normY1.y()) / voxelSize;
						}

						if (indexZ > 0 && indexZ < dimensions.z - 1) {
							divZ = (normZ2.z() - normZ1.z()) / (2.0f * voxelSize);
						}
						else if (indexZ == 0) {
							divZ = (normZ2.z() - cv.normal.z()) / voxelSize;
						}
						else if (indexZ == dimensions.z - 1) {
							divZ = (cv.normal.z() - normZ1.z()) / voxelSize;
						}

						// 발산 결과 계산 및 저장
						d_divergences[index] = divX + divY + divZ;

						// 결과를 다시 d_volume에 저장
						d_volume[index] = cv;
					});
					nvtxRangePop();
					t = Time::End(t, "Compute Divergence");
				}

				{
					//// Solving Poisson System for Potentials
					//t = Time::Now();
					//nvtxRangePushA("SolvePoissonWithCuSolver");

					//// Prepare CSR matrix and vectors for cuSolver
					//int numRows = dimensions.x * dimensions.y * dimensions.z;
					//int nnz = numRows * 7; // Adjust as needed

					//thrust::device_vector<int> rowPtr(numRows + 1);
					//thrust::device_vector<int> colInd(nnz);
					//thrust::device_vector<float> values(nnz);
					//thrust::device_vector<float> b(volume.size());

					//// Example: Initialization (constructing a CSR matrix is context-specific)
					//thrust::host_vector<int> h_rowPtr(numRows + 1, 0);
					//for (int i = 0; i < numRows; i++) {
					//	h_rowPtr[i] = i * 7; // Example increment, adjust accordingly
					//}
					//h_rowPtr[numRows] = nnz;

					//// Copy rowPtr to device
					//rowPtr = h_rowPtr;

					//// Fill divergence vector `b`
					//thrust::transform(volume.begin(), volume.end(), b.begin(), [] __device__(const Voxel & voxel) {
					//	return (voxel.divergence != FLT_MAX) ? voxel.divergence : 0.0f;
					//});

					//// Create matrix descriptor
					//cusparseMatDescr_t descrA;
					//cusparseCreateMatDescr(&descrA);
					//cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
					//cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

					//// Get raw pointers from device vectors
					//int* d_rowPtr = thrust::raw_pointer_cast(rowPtr.data());
					//int* d_colInd = thrust::raw_pointer_cast(colInd.data());
					//float* d_values = thrust::raw_pointer_cast(values.data());
					//float* d_b = thrust::raw_pointer_cast(b.data());
					//thrust::device_vector<float> d_potentials(volume.size());
					//float* d_phi = thrust::raw_pointer_cast(d_potentials.data());

					//int singularity;
					//cusolverStatus_t status = cusolverSpScsrlsvqr(
					//	cusolverHandle,
					//	numRows,
					//	nnz,
					//	descrA,
					//	d_values,
					//	d_rowPtr,
					//	d_colInd,
					//	d_b,
					//	1e-10,
					//	0,
					//	d_phi,
					//	&singularity
					//);

					//if (status != CUSOLVER_STATUS_SUCCESS) {
					//	printf("Solver failed with status code %d\n", status);
					//}

					//if (singularity >= 0) {
					//	printf("The matrix is singular at row %d\n", singularity);
					//}

					//nvtxRangePop();
					//t = Time::End(t, "SolvePoissonWithCuSolver");

					//nvtxRangePushA("Update voxel potentials");
					//// Update voxel potentials
					//thrust::for_each(
					//	thrust::counting_iterator<size_t>(0),
					//	thrust::counting_iterator<size_t>(volume.size()),
					//	[=] __device__(size_t index) mutable {
					//	Voxel voxel = volume[index];
					//	voxel.potential = d_potentials[index];
					//	volume[index] = voxel;
					//});
					//nvtxRangePop();
					//t = Time::End(t, "Update voxel potentials");
				}

				{
					// Add cubes where volume value is not zero
					nvtxRangePushA("Add Cubes");
					thrust::host_vector<Voxel> h_volume = volume; // Copy device vector to host
					thrust::host_vector<float> h_divergences = divergences; // Copy device vector to host

					for (uint32_t z = 0; z < dimensions.z; ++z)
					{
						for (uint32_t y = 0; y < dimensions.y; ++y)
						{
							for (uint32_t x = 0; x < dimensions.x; ++x)
							{
								uint3 index = make_uint3(x, y, z);
								size_t flatIndex = GetFlatIndex(index, dimensions);
								Voxel& voxel = h_volume[flatIndex];
								float divergence = h_divergences[flatIndex];

								// 발산 값이 유효한지 확인하는 조건 강화
								if (!isnan(divergence) && divergence != FLT_MAX)
								{
									if (fabsf(divergence) > 0.0f)  // 발산 값이 일정 범위 내에 있는 경우에만 큐브 추가
									{
										Eigen::Vector3f position = GetPosition(center, dimensions, voxelSize, index);
										VD::AddCube("volume", position, { voxelSize, voxelSize, voxelSize },
											{ 0.0f, 0.0f, 1.0f }, Color4::White);
									}
								}
							}
						}
					}
					nvtxRangePop();
					t = Time::End(t, "Add Cubes");
				}
			}
		}
	}
}
