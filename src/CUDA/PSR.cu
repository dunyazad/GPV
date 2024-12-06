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

		cusolverSpHandle_t cusolverHandle;
		cusparseMatDescr_t descrA;

		void InitializeCuSolver()
		{
			// cuSolver 핸들 초기화
			cusolverStatus_t status = cusolverSpCreate(&cusolverHandle);
			if (status != CUSOLVER_STATUS_SUCCESS) {
				cerr << "Failed to create cuSolver handle" << endl;
				exit(EXIT_FAILURE);
			}

			// cuSparse 행렬 설명자 초기화
			cusparseStatus_t cusparseStatus = cusparseCreateMatDescr(&descrA);
			if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
				cerr << "Failed to create cuSparse matrix descriptor" << endl;
				exit(EXIT_FAILURE);
			}

			cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
		}

		void DestroyCuSolver()
		{
			cusparseDestroyMatDescr(descrA);
			cusolverSpDestroy(cusolverHandle);
		}


		void CheckCusolverStatus(cusolverStatus_t status, const char* msg)
		{
			if (status != CUSOLVER_STATUS_SUCCESS) {
				std::cerr << "CUSOLVER ERROR: " << msg << " (status code: " << status << ")" << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		void CheckCusparseStatus(cusparseStatus_t status, const char* msg)
		{
			if (status != CUSPARSE_STATUS_SUCCESS) {
				std::cerr << "CUSPARSE ERROR: " << msg << " (status code: " << status << ")" << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		void TestCusolverSpScsrlsvqr(
			int n,
			int nnz,
			thrust::device_vector<float>& csrValA,
			thrust::device_vector<int>& csrRowPtrA,
			thrust::device_vector<int>& csrColIndA,
			thrust::device_vector<float>& b,
			float tol
		)
		{
			// Create cuSolver handle
			cusolverSpHandle_t cusolverHandle;
			CheckCusolverStatus(cusolverSpCreate(&cusolverHandle), "Failed to create cuSolver handle");

			// Create cuSparse matrix descriptor
			cusparseMatDescr_t descrA;
			CheckCusparseStatus(cusparseCreateMatDescr(&descrA), "Failed to create cuSparse matrix descriptor");

			cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

			// Output vector for solution
			thrust::device_vector<float> x(n, 0.0f);

			// Singular value to indicate whether the system is singular
			int singularity = -1;

			// Perform the solve using cuSolverSpScsrlsvqr
			cusolverStatus_t solveStatus = cusolverSpScsrlsvqr(
				cusolverHandle,
				n,                                      // Number of rows/columns
				nnz,                                    // Number of non-zero elements
				descrA,                                 // Matrix descriptor
				thrust::raw_pointer_cast(csrValA.data()),   // CSR value array
				thrust::raw_pointer_cast(csrRowPtrA.data()), // CSR row pointer array
				thrust::raw_pointer_cast(csrColIndA.data()), // CSR column index array
				thrust::raw_pointer_cast(b.data()),         // Right-hand side vector
				tol,                                    // Tolerance
				0,                                      // Reorder (0: No, 1: Yes)
				thrust::raw_pointer_cast(x.data()),     // Solution vector
				&singularity                            // Singularity indicator
			);

			if (solveStatus != CUSOLVER_STATUS_SUCCESS) {
				std::cerr << "ERROR: Failed to solve the linear system with cusolverSpScsrlsvqr." << std::endl;
				exit(EXIT_FAILURE);
			}

			// Check for singularity
			if (singularity >= 0) {
				std::cerr << "WARNING: The system is singular at row " << singularity << "." << std::endl;
			}
			else {
				std::cout << "System solved successfully!" << std::endl;
			}

			// Clean up
			cusparseDestroyMatDescr(descrA);
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

			InitializeCuSolver();

			Eigen::Vector3f total_min(-17.5f, -17.5f, -17.5f);
			Eigen::Vector3f total_max(17.5f, 17.5f, 17.5f);
			//Eigen::Vector3f total_min(-5.0f, -5.0f, -5.0f);
			//Eigen::Vector3f total_max(5.0f, 5.0f, 5.0f);
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

					// Initialize maximum difference for convergence check
					thrust::device_vector<float> max_diff(1);

					// Iterative solver parameters
					int max_iterations = 10;
					float tolerance = 1e-5;

					for (int iter = 0; iter < max_iterations; iter++) {
						max_diff[0] = 0.0f; // Reset max_diff

						// Update potential using thrust::for_each
						thrust::for_each(
							thrust::counting_iterator<size_t>(0),
							thrust::counting_iterator<size_t>(numberOfVoxels),
							[=, d_max_diff = thrust::raw_pointer_cast(max_diff.data())] __device__(size_t index) {
							size_t x = index % dimensions.x;
							size_t y = (index / dimensions.x) % dimensions.y;
							size_t z = index / (dimensions.x * dimensions.y);

							// Skip boundary voxels
							if (x == 0 || x == dimensions.x - 1 ||
								y == 0 || y == dimensions.y - 1 ||
								z == 0 || z == dimensions.z - 1) {
								d_potentials[index] = 0.0f;
								return;
							}

							// Compute new potential value
							float sum_neighbors =
								d_potentials[GetFlatIndex(make_uint3(x - 1, y, z), dimensions)] +
								d_potentials[GetFlatIndex(make_uint3(x + 1, y, z), dimensions)] +
								d_potentials[GetFlatIndex(make_uint3(x, y - 1, z), dimensions)] +
								d_potentials[GetFlatIndex(make_uint3(x, y + 1, z), dimensions)] +
								d_potentials[GetFlatIndex(make_uint3(x, y, z - 1), dimensions)] +
								d_potentials[GetFlatIndex(make_uint3(x, y, z + 1), dimensions)];

							float new_potential = (1.0f / 6.0f) * (sum_neighbors - d_divergences[index]);

							// Compute difference
							float diff = fabsf(new_potential - d_potentials[index]);

							// Update maximum difference
							atomicMax(reinterpret_cast<int*>(d_max_diff), __float_as_int(diff));

							// Update the potential value
							d_potentials[index] = new_potential;
						});

						// Check convergence
						float max_diff_host = 0.0f;
						cudaMemcpy(&max_diff_host, thrust::raw_pointer_cast(max_diff.data()), sizeof(float), cudaMemcpyDeviceToHost);

						if (max_diff_host < tolerance) {
							std::cout << "Converged after " << iter + 1 << " iterations with max_diff = " << max_diff_host << std::endl;
							break;
						}
					}

					nvtxRangePop();
					t = Time::End(t, "Compute Potential");
				}

				cudaDeviceSynchronize();

				//{
				//	// Add cubes where volume value is not zero
				//	nvtxRangePushA("Add Cubes");
				//	thrust::host_vector<Voxel> h_volume = volume; // Copy device vector to host
				//	thrust::host_vector<float> h_divergences = divergences; // Copy device vector to host
				//	thrust::host_vector<float> h_potentials = potentials; // Copy device vector to host

				//	for (uint32_t z = 0; z < dimensions.z; ++z)
				//	{
				//		for (uint32_t y = 0; y < dimensions.y; ++y)
				//		{
				//			for (uint32_t x = 0; x < dimensions.x; ++x)
				//			{
				//				uint3 index = make_uint3(x, y, z);
				//				size_t flatIndex = GetFlatIndex(index, dimensions);
				//				Voxel& voxel = h_volume[flatIndex];
				//				float divergence = h_divergences[flatIndex];

				//				// 발산 값이 유효한지 확인하는 조건 강화
				//				if (!isnan(divergence) && divergence != FLT_MAX)
				//				{
				//					if (fabsf(divergence) > 0.05f)  // 발산 값이 일정 범위 내에 있는 경우에만 큐브 추가
				//					{
				//						Eigen::Vector3f position = GetPosition(center, dimensions, voxelSize, index);
				//						VD::AddCube("volume", position, { voxelSize, voxelSize, voxelSize },
				//							{ 0.0f, 0.0f, 1.0f }, Color4::White);
				//					}
				//				}
				//			}
				//		}
				//	}
				//	nvtxRangePop();
				//	t = Time::End(t, "Add Cubes");
				//}

				{
					// Add cubes based on potentials
					nvtxRangePushA("Visualize Potentials");
					thrust::host_vector<Voxel> h_volume = volume; // Copy device vector to host
					thrust::host_vector<float> h_divergences = divergences; // Copy device vector to host
					thrust::host_vector<float> h_potentials = potentials; // Copy device vector to host

					// Define thresholds for potential visualization
					//float minThreshold = -0.005f; // Minimum potential value to display
					//float maxThreshold = 0.005f;  // Maximum potential value to display

					float minPotential = *thrust::min_element(potentials.begin(), potentials.end());
					float maxPotential = *thrust::max_element(potentials.begin(), potentials.end());

					float minThreshold = minPotential + 0.01f * (maxPotential - minPotential);
					float maxThreshold = maxPotential - 0.01f * (maxPotential - minPotential);

					for (uint32_t z = 1; z < dimensions.z - 1; ++z)
					{
						for (uint32_t y = 1; y < dimensions.y - 1; ++y)
						{
							for (uint32_t x = 1; x < dimensions.x - 1; ++x)
							{
								uint3 index = make_uint3(x, y, z);
								size_t flatIndex = GetFlatIndex(index, dimensions);

								float potential = h_potentials[flatIndex];

								//if (FLT_MAX != potential)
								//{
								//	printf("potential : %f\n", potential);
								//}

								// Check if the potential value is within the threshold range
								if (!isnan(potential) && potential != FLT_MAX)
								{
									if (minThreshold <= potential && potential <= maxThreshold)
									{
										// Get the position of the voxel
										Eigen::Vector3f position = GetPosition(center, dimensions, voxelSize, index);

										// Normalize the potential to a color scale (e.g., blue to red)
										float normalized = (potential - minThreshold) / (maxThreshold - minThreshold);
										Eigen::Vector3f color = { 1.0f - normalized, 0.0f, normalized }; // Red to Blue gradient

										// Add cube
										VD::AddCube("potentials", position, { voxelSize, voxelSize, voxelSize },
											color, Color4::White);
									}
								}
							}
						}
					}
					nvtxRangePop();
					t = Time::End(t, "Visualize Potentials");
				}

			}

			DestroyCuSolver();
		}
	}
}
