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
			size_t GetFlatIndex(const uint3& index, const uint3& dimensions) {
			// Check if the index is within bounds
			if (index.x >= dimensions.x || index.y >= dimensions.y || index.z >= dimensions.z) {
				return UINT_MAX; // Invalid index
			}
			return index.z * dimensions.x * dimensions.y + index.y * dimensions.x + index.x;
		}

		__host__ __device__
			bool isBoundary(uint32_t x, uint32_t y, uint32_t z, const uint3& dimensions) {
			return (x == 0 || y == 0 || z == 0 ||
				x == dimensions.x - 1 ||
				y == dimensions.y - 1 ||
				z == dimensions.z - 1);
		}

		// 행렬과 벡터의 곱
		__host__ __device__
			void matVecMul(const std::vector<float>& A, const std::vector<float>& x, std::vector<float>& b, int size) {
			for (int i = 0; i < size; ++i) {
				b[i] = 0;
				for (int j = 0; j < size; ++j) {
					b[i] += A[i * size + j] * x[j];
				}
			}
		}

		// 직접 해법: 가우스 소거법
		__host__ __device__
			void gaussSolve(float* A, float* x, float* b, int size) {
			for (int i = 0; i < size; ++i) {
				// 대각선 요소를 1로 만듦
				float diag = A[i * size + i];
				for (int j = 0; j < size; ++j) {
					A[i * size + j] /= diag;
				}
				b[i] /= diag;

				// 다른 행 제거
				for (int k = i + 1; k < size; ++k) {
					float factor = A[k * size + i];
					for (int j = 0; j < size; ++j) {
						A[k * size + j] -= factor * A[i * size + j];
					}
					b[k] -= factor * b[i];
				}
			}

			// 역방향 대입
			for (int i = size - 1; i >= 0; --i) {
				x[i] = b[i];
				for (int j = i + 1; j < size; ++j) {
					x[i] -= A[i * size + j] * x[j];
				}
			}
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

			/*Eigen::Vector3f total_min(-17.5f, -17.5f, -17.5f);
			Eigen::Vector3f total_max(17.5f, 17.5f, 17.5f);*/
			Eigen::Vector3f total_min(-5.0f, -5.0f, -5.0f);
			Eigen::Vector3f total_max(5.0f, 5.0f, 5.0f);
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

				size_t numberOfVoxels = dimensions.x * dimensions.y * dimensions.z;

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

				cudaDeviceSynchronize();

				{
					t = Time::Now();
					nvtxRangePushA("Compute Divergence");
					/*
					thrust::for_each(thrust::counting_iterator((size_t)0), thrust::counting_iterator(volume.size()),
						[=] __device__(size_t index) {
						size_t indexZ = index / (dimensions.y * dimensions.x);
						size_t indexY = (index % (dimensions.y * dimensions.x)) / dimensions.x;
						size_t indexX = (index % (dimensions.y * dimensions.x)) % dimensions.x;

						Voxel cv = d_volume[index];

						size_t piX = (indexX > 0) ? indexX - 1 : indexX;
						size_t niX = (indexX < dimensions.x - 1) ? indexX + 1 : indexX;
						size_t piY = (indexY > 0) ? indexY - 1 : indexY;
						size_t niY = (indexY < dimensions.y - 1) ? indexY + 1 : indexY;
						size_t piZ = (indexZ > 0) ? indexZ - 1 : indexZ;
						size_t niZ = (indexZ < dimensions.z - 1) ? indexZ + 1 : indexZ;

						size_t flatIndexX1 = GetFlatIndex(make_uint3(piX, indexY, indexZ), dimensions);
						size_t flatIndexX2 = GetFlatIndex(make_uint3(niX, indexY, indexZ), dimensions);
						size_t flatIndexY1 = GetFlatIndex(make_uint3(indexX, piY, indexZ), dimensions);
						size_t flatIndexY2 = GetFlatIndex(make_uint3(indexX, niY, indexZ), dimensions);
						size_t flatIndexZ1 = GetFlatIndex(make_uint3(indexX, indexY, piZ), dimensions);
						size_t flatIndexZ2 = GetFlatIndex(make_uint3(indexX, indexY, niZ), dimensions);

						Eigen::Vector3f normX1 = d_volume[flatIndexX1].weight > 0 ? d_volume[flatIndexX1].normal / (float)d_volume[flatIndexX1].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normX2 = d_volume[flatIndexX2].weight > 0 ? d_volume[flatIndexX2].normal / (float)d_volume[flatIndexX2].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normY1 = d_volume[flatIndexY1].weight > 0 ? d_volume[flatIndexY1].normal / (float)d_volume[flatIndexY1].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normY2 = d_volume[flatIndexY2].weight > 0 ? d_volume[flatIndexY2].normal / (float)d_volume[flatIndexY2].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normZ1 = d_volume[flatIndexZ1].weight > 0 ? d_volume[flatIndexZ1].normal / (float)d_volume[flatIndexZ1].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);
						Eigen::Vector3f normZ2 = d_volume[flatIndexZ2].weight > 0 ? d_volume[flatIndexZ2].normal / (float)d_volume[flatIndexZ2].weight : Eigen::Vector3f(0.0f, 0.0f, 0.0f);

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

						d_divergences[index] = divX + divY + divZ;

						d_volume[index] = cv;
					});
					nvtxRangePop();
					*/

					thrust::for_each(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(numberOfVoxels),
						[=] __device__(size_t index) {
						// Skip empty voxels
						if (d_volume[index].weight == 0) {
							d_divergences[index] = 0.0f;
							return;
						}

						// Compute indices
						size_t x = index % dimensions.x;
						size_t y = (index / dimensions.x) % dimensions.y;
						size_t z = index / (dimensions.x * dimensions.y);

						// Compute divergence
						float divX = 0.0f, divY = 0.0f, divZ = 0.0f;

						// X-direction
						if (x == 0) {
							divX = (d_volume[GetFlatIndex(make_uint3(x + 1, y, z), dimensions)].normal.x() -
								d_volume[index].normal.x()) / voxelSize;
						}
						else if (x == dimensions.x - 1) {
							divX = (d_volume[index].normal.x() -
								d_volume[GetFlatIndex(make_uint3(x - 1, y, z), dimensions)].normal.x()) / voxelSize;
						}
						else {
							divX = (d_volume[GetFlatIndex(make_uint3(x + 1, y, z), dimensions)].normal.x() -
								d_volume[GetFlatIndex(make_uint3(x - 1, y, z), dimensions)].normal.x()) / (2.0f * voxelSize);
						}

						// Y-direction
						if (y == 0) {
							divY = (d_volume[GetFlatIndex(make_uint3(x, y + 1, z), dimensions)].normal.y() -
								d_volume[index].normal.y()) / voxelSize;
						}
						else if (y == dimensions.y - 1) {
							divY = (d_volume[index].normal.y() -
								d_volume[GetFlatIndex(make_uint3(x, y - 1, z), dimensions)].normal.y()) / voxelSize;
						}
						else {
							divY = (d_volume[GetFlatIndex(make_uint3(x, y + 1, z), dimensions)].normal.y() -
								d_volume[GetFlatIndex(make_uint3(x, y - 1, z), dimensions)].normal.y()) / (2.0f * voxelSize);
						}

						// Z-direction
						if (z == 0) {
							divZ = (d_volume[GetFlatIndex(make_uint3(x, y, z + 1), dimensions)].normal.z() -
								d_volume[index].normal.z()) / voxelSize;
						}
						else if (z == dimensions.z - 1) {
							divZ = (d_volume[index].normal.z() -
								d_volume[GetFlatIndex(make_uint3(x, y, z - 1), dimensions)].normal.z()) / voxelSize;
						}
						else {
							divZ = (d_volume[GetFlatIndex(make_uint3(x, y, z + 1), dimensions)].normal.z() -
								d_volume[GetFlatIndex(make_uint3(x, y, z - 1), dimensions)].normal.z()) / (2.0f * voxelSize);
						}

						// Set divergence
						d_divergences[index] = divX + divY + divZ;

						// Clamp divergence for stability
						float maxDivergenceThreshold = 100.0f;
						if (fabsf(d_divergences[index]) > maxDivergenceThreshold) {
							d_divergences[index] = copysignf(maxDivergenceThreshold, d_divergences[index]);
						}
					});


					t = Time::End(t, "Compute Divergence");
				}

				cudaDeviceSynchronize();

				//{
				//	thrust::for_each(thrust::counting_iterator((size_t)0), thrust::counting_iterator(volume.size()),
				//		[=] __device__(size_t index) {
				//		d_potentials[index] = d_divergences[index];
				//	});
				//}

				{
					t = Time::Now();
					nvtxRangePushA("Compute Potential");

					// 라플라스 방정식의 연산을 직접 수행
					thrust::for_each(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(numberOfVoxels),
						[=] __device__(size_t idx) {
						//printf("%d\n", idx);

						// 3D 인덱스를 계산
						size_t z = idx / (dimensions.x * dimensions.y);
						size_t y = (idx / dimensions.x) % dimensions.y;
						size_t x = idx % dimensions.x;

						//printf("%d %d %d\n", x, y, z);

						// 경계 조건 처리
						if (x == 0 || x == dimensions.x - 1 || y == 0 || y == dimensions.y - 1 || z == 0 || z == dimensions.z - 1) {
							// Dirichlet 경계 조건: 경계 포텐셜은 0
							return;
						}

						//// 발산값 가져오기
						float divergence = d_divergences[idx];

						//// 6-연결 이웃에서의 포텐셜 계산
						float neighborSum = 0.0f;
						neighborSum += d_potentials[idx - 1];                  // Left
						neighborSum += d_potentials[idx + 1];                  // Right
						neighborSum += d_potentials[idx - dimensions.x];           // Bottom
						neighborSum += d_potentials[idx + dimensions.x];           // Top
						neighborSum += d_potentials[idx - dimensions.x * dimensions.y]; // Front
						neighborSum += d_potentials[idx + dimensions.x * dimensions.y]; // Back

						////// 현재 위치의 포텐셜 계산
						d_potentials[idx] = (neighborSum - divergence * voxelSize * voxelSize) / 6.0f;

						//printf("d_potentials[idx] : %f\n", d_potentials[idx]);
					});

					nvtxRangePop();
					t = Time::End(t, "Compute Potential");
				}



				cudaDeviceSynchronize();

				{
					// Add cubes where volume value is not zero
					nvtxRangePushA("Add Cubes");
					thrust::host_vector<Voxel> h_volume = volume; // Copy device vector to host
					thrust::host_vector<float> h_divergences = divergences; // Copy device vector to host
					thrust::host_vector<float> h_potentials = potentials; // Copy device vector to host

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
									if (fabsf(divergence) > 0.05f)  // 발산 값이 일정 범위 내에 있는 경우에만 화살표 추가
									{
										Eigen::Vector3f position = GetPosition(center, dimensions, voxelSize, index);
										VD::AddArrow("Divergences", position, voxel.normal, voxelSize, Color4::Red);
									}
								}
							}
						}
					}
					nvtxRangePop();
					t = Time::End(t, "Add Cubes");
				}

				{
					const float isovalue = 0.5f;

					Eigen::Vector3f gridCenter = (max + min) * 0.5f;

					std::vector<Eigen::Vector3f> vertices;
					std::vector<Eigen::Vector3f> normals;

					// 삼각형 버퍼를 GPU 메모리에 할당
					thrust::device_vector<Triangle> triangles(numberOfVoxels * 4);
					Triangle* d_triangles = thrust::raw_pointer_cast(triangles.data());

					cudaDeviceSynchronize();

					thrust::for_each(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(numberOfVoxels),
						[=] __device__(size_t idx) {
						if (idx >= numberOfVoxels) return;

						size_t z = idx / (dimensions.x * dimensions.y);
						size_t y = (idx / dimensions.x) % dimensions.y;
						size_t x = idx % dimensions.x;

						//printf("%d, %d, %d\n", x, y, z);

						// 8개 코너의 3D 인덱스 계산
						uint3 cornerIndices[8] = {
							make_uint3(x, y, z),
							make_uint3(x + 1, y, z),
							make_uint3(x + 1, y + 1, z),
							make_uint3(x, y + 1, z),
							make_uint3(x, y, z + 1),
							make_uint3(x + 1, y, z + 1),
							make_uint3(x + 1, y + 1, z + 1),
							make_uint3(x, y + 1, z + 1)
						};

						// 상태 값 계산
						int cubeIndex = 0;
						float cornerValues[8];
						for (int i = 0; i < 8; ++i) {
							// Get the flat index for the corner
							size_t flatIndex = GetFlatIndex(cornerIndices[i], dimensions);
							if (UINT_MAX == flatIndex) continue;

							//// Debugging output to check corner indices and flatIndex
							//printf("Debug: cornerIndex[%d] = (%u, %u, %u), flatIndex = %llu\n",
							//	i, cornerIndices[i].x, cornerIndices[i].y, cornerIndices[i].z, flatIndex);

							// Check if the flatIndex is valid
							if (flatIndex == UINT_MAX) {
								//printf("Error: flatIndex for cornerIndex[%d] is invalid. Skipping this voxel.\n", i);
								return; // Skip further processing for this voxel
							}

							// Ensure flatIndex is within bounds of d_potentials
							size_t numPotentials = dimensions.x * dimensions.y * dimensions.z;
							if (flatIndex >= numPotentials) {
								//printf("Error: flatIndex = %zu is out of bounds (max = %zu). Skipping this voxel.\n",
								//	flatIndex, numPotentials - 1);
								return; // Skip further processing for this voxel
							}

							// Fetch the potential value
							cornerValues[i] = d_potentials[flatIndex];

							if (true == isnan(cornerValues[i]) ||
								true == isinf(cornerValues[i]))
							{
								printf("cornerValues[%d] : %f\n", i, cornerValues[i]);
							}

							// Check if the potential value is below the isovalue
							if (cornerValues[i] < isovalue) {
								cubeIndex |= (1 << i); // Update cube index
							}

							// Debugging output for potential values
							//printf("Debug: cornerValues[%d] = %f, cubeIndex = %d\n", i, cornerValues[i], cubeIndex);
						}

						if (cubeIndex == 0 || edgeTable[cubeIndex] == 0) return;

						//printf("Debug: cubeIndex = %d\n", cubeIndex);


						Eigen::Vector3f vertList[12];
						for (int i = 0; i < 12; ++i) {
							if (edgeTable[cubeIndex] & (1 << i)) {
								int v0 = edgeVertexMap[i][0];
								int v1 = edgeVertexMap[i][1];
								Eigen::Vector3f p0 = GetPosition(gridCenter, dimensions, voxelSize, cornerIndices[v0]);
								Eigen::Vector3f p1 = GetPosition(gridCenter, dimensions, voxelSize, cornerIndices[v1]);
								float val0 = cornerValues[v0];
								float val1 = cornerValues[v1];

								if (true == isnan(val0) ||
									true == isinf(val0))
								{
									printf("val0 : %f, cornerValues[v0] : %f\n", val0, cornerValues[v0]);
								}

								float denom = val1 - val0;

								if (fabsf(val0 - val1) < 0.000001f)
								{
									denom = 0.5f;
								}

								float t = (isovalue - val0) / denom;

								vertList[i] = p0 + t * (p1 - p0);

								//if (true == isnan(vertList[i].x()) ||
								//	true == isnan(vertList[i].y()) ||
								//	true == isnan(vertList[i].z()) ||
								//	true == isinf(vertList[i].x()) ||
								//	true == isinf(vertList[i].y()) ||
								//	true == isinf(vertList[i].z()))
								//{
								//	//printf("%f, %f, %f\n", vertList[i].x(), vertList[i].y(), vertList[i].z());
								//	printf("val0 : %f, val1 : %f, denom : %f, t : %f\n", val0, val1, denom, t);
								//}
							}
						}

						for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
							int t0 = triTable[cubeIndex][i];
							int t1 = triTable[cubeIndex][i + 1];
							int t2 = triTable[cubeIndex][i + 2];

							//printf("%d %d %d\n", t0, t1, t2);

							//printf("%f, %f, %f\n", vertList[t0], vertList[t1], vertList[t2]);

	/*						if (idx < triangles.size()) {
								d_triangles[idx].v0 = vertList[t0];
								d_triangles[idx].v1 = vertList[t1];
								d_triangles[idx].v2 = vertList[t2];
							}*/
						}
					});

					float minPotential = *thrust::min_element(potentials.begin(), potentials.end());
					float maxPotential = *thrust::max_element(potentials.begin(), potentials.end());
					std::cout << "Min potential: " << minPotential << ", Max potential: " << maxPotential << std::endl;

					thrust::host_vector<Triangle> hostTriangles(triangles);
					for (const auto& tri : hostTriangles)
					{
						VD::AddTriangle("marching_cubes", tri.v0, tri.v1, tri.v2, Color4::White);
					}
				}

				{
					//// Add cubes based on potentials
					//nvtxRangePushA("Visualize Potentials");
					//thrust::host_vector<Voxel> h_volume = volume; // Copy device vector to host
					//thrust::host_vector<float> h_divergences = divergences; // Copy device vector to host
					//thrust::host_vector<float> h_potentials = potentials; // Copy device vector to host

					//// Define thresholds for potential visualization
					////float minThreshold = -0.005f; // Minimum potential value to display
					////float maxThreshold = 0.005f;  // Maximum potential value to display

					//float minPotential = *thrust::min_element(potentials.begin(), potentials.end());
					//float maxPotential = *thrust::max_element(potentials.begin(), potentials.end());

					//float minThreshold = minPotential + 0.01f * (maxPotential - minPotential);
					//float maxThreshold = maxPotential - 0.01f * (maxPotential - minPotential);

					//for (uint32_t z = 1; z < dimensions.z - 1; ++z)
					//{
					//	for (uint32_t y = 1; y < dimensions.y - 1; ++y)
					//	{
					//		for (uint32_t x = 1; x < dimensions.x - 1; ++x)
					//		{
					//			uint3 index = make_uint3(x, y, z);
					//			size_t flatIndex = GetFlatIndex(index, dimensions);

					//			float potential = h_potentials[flatIndex];

					//			//if (FLT_MAX != potential && 0.0f != potential)
					//			//{
					//			//	printf("potential : %f\n", potential);
					//			//}

					//			// Check if the potential value is within the threshold range
					//			if (!isnan(potential) && potential != FLT_MAX)
					//			{
					//				float normalized = (potential - minThreshold) / (maxThreshold - minThreshold);

					//				//if (minThreshold <= potential && potential <= maxThreshold)
					//				if(0.45f < potential && potential <= maxThreshold)
					//				{
					//					// Get the position of the voxel
					//					Eigen::Vector3f position = GetPosition(center, dimensions, voxelSize, index);

					//					// Normalize the potential to a color scale (e.g., blue to red)
					//					float normalized = (potential - minThreshold) / (maxThreshold - minThreshold);
					//					//Eigen::Vector3f color = { 1.0f - normalized, 0.0f, normalized }; // Red to Blue gradient

					//					// Add cube
					//					//VD::AddCube("potentials", position, { voxelSize, voxelSize, voxelSize },
					//					//	{0.0f, 0.0f, 1.0f}, Color4::FromNormalized(color.x(), color.y(), color.z(), 1.0f));

					//					VD::AddCube("potentials", position, { voxelSize, voxelSize, voxelSize },
					//						{ 0.0f, 0.0f, 1.0f }, Color4::White);
					//				}
					//			}
					//		}
					//	}
					//}
					//nvtxRangePop();
					//t = Time::End(t, "Visualize Potentials");
				}
			}
		}
	}
}
