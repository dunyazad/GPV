#include "MarchingCubes.cuh"

#include <vtkHeaderFiles.h>

#include <App/Utility.h>

#include <App/Serialization.hpp>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

#include <App/AppStartCallback.h>

#define USE_CUDA
#include <Algorithm/MarchingCubes.hpp>

namespace CUDA
{
	namespace MarchingCubes
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

			for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
			{
				auto px = ply.GetPoints()[i * 3 + 0];
				auto py = ply.GetPoints()[i * 3 + 1];
				auto pz = ply.GetPoints()[i * 3 + 2];

				auto nx = ply.GetNormals()[i * 3 + 0];
				auto ny = ply.GetNormals()[i * 3 + 1];
				auto nz = ply.GetNormals()[i * 3 + 2];

				auto cx = ply.GetColors()[i * 3 + 0];
				auto cy = ply.GetColors()[i * 3 + 1];
				auto cz = ply.GetColors()[i * 3 + 2];

				auto c4 = Color4::FromNormalized(cx, cy, cz, 1.0f);
				VD::AddSphere("points", { px, py, pz }, { 0.05f, 0.05f, 0.05f }, { nx, ny, nz }, c4);
			}
		}
#pragma endregion

		struct Voxel
		{
			float minDistance;
			float tsdfValue;
			Eigen::Vector3f normal;
			Eigen::Vector3f color;
			float weight;
			__host__ __device__ Voxel() : minDistance(FLT_MAX), tsdfValue(FLT_MAX), normal(0.0f, 0.0f, 0.0f), color(1.0f, 1.0f, 1.0f), weight(0.0f) {}
		};

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

		__device__ float atomicMinFloat(float* addr, float value) {
			int* intAddr = (int*)addr;
			int oldInt = *intAddr;
			float oldFloat = __int_as_float(oldInt);
			while (oldFloat > value) {
				int assumedInt = oldInt;
				oldInt = atomicCAS(intAddr, assumedInt, __float_as_int(value));
				oldFloat = __int_as_float(oldInt);
			}
			return oldFloat;
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

		void TestMarchingCubes_PointCloud()
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
				});

				{
					t = Time::Now();
					nvtxRangePushA("Insert Points");

					Eigen::Vector3f* d_points = thrust::raw_pointer_cast(patchBuffers.inputPoints.data());
					Eigen::Vector3f* d_normals = thrust::raw_pointer_cast(patchBuffers.inputNormals.data());

					thrust::for_each(thrust::counting_iterator((size_t)0), thrust::counting_iterator(patchBuffers.inputPoints.size()),
						[=] __device__(size_t i) {
						Eigen::Vector3f point = d_points[i];
						Eigen::Vector3f normal = d_normals[i];

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
					nvtxRangePushA("Visualize Potentials");
					thrust::host_vector<Voxel> h_volume = volume; // Copy device vector to host

					for (uint32_t z = 0; z < dimensions.z; ++z)
					{
						for (uint32_t y = 0; y < dimensions.y; ++y)
						{
							for (uint32_t x = 0; x < dimensions.x; ++x)
							{
								uint3 index = make_uint3(x, y, z);
								size_t flatIndex = GetFlatIndex(index, dimensions);

								auto& voxel = h_volume[flatIndex];

								if (voxel.weight > 0.0f)
								{
									Eigen::Vector3f position = GetPosition(center, dimensions, voxelSize, index);

									VD::AddCube("Cubes", position, { voxelSize, voxelSize, voxelSize },
										{ 0.0f, 0.0f, 1.0f }, Color4::White);
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

		void TestMarchingCubes_Patches()
		{
			auto t = Time::Now();

			Eigen::Vector3f volumeMin(-5.0f, -5.0f, -5.0f);
			Eigen::Vector3f volumeMax(5.0f, 5.0f, 5.0f);
			Eigen::Vector3f diff = volumeMax - volumeMin;
			Eigen::Vector3f volumeCenter = (volumeMax + volumeMin) * 0.5f;
			float voxelSize = 0.1f;
			float truncationDistance = 0.5f;
			float isoValue = 0.0f;
			int voxelNeighborRange = (int)ceilf(truncationDistance / voxelSize);
			//voxelNeighborRange = 1;

			uint3 dimensions;
			dimensions.x = (uint32_t)ceilf(diff.x() / voxelSize);
			dimensions.y = (uint32_t)ceilf(diff.y() / voxelSize);
			dimensions.z = (uint32_t)ceilf(diff.z() / voxelSize);

			size_t numberOfVoxels = dimensions.x * dimensions.y * dimensions.z;

			thrust::device_vector<Voxel> volume(dimensions.x * dimensions.y * dimensions.z);
			auto d_volume = thrust::raw_pointer_cast(volume.data());

			thrust::for_each(thrust::counting_iterator((size_t)0), thrust::counting_iterator(volume.size()),
				[=] __device__(size_t index) {
				d_volume[index].minDistance = FLT_MAX;
				d_volume[index].tsdfValue = FLT_MAX;
				d_volume[index].weight = 0;
				d_volume[index].normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				d_volume[index].color = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
			});

			Time::End(t, "Initialize Volume");

			LoadTRNFile();

			PatchBuffers patchBuffers(256, 480);

			//size_t i = 3;
			for (size_t i = 0; i < 10; i++)
				//for (size_t i = 0; i < 4252; i++)
			{
				stringstream ss;
				ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_" << i << ".ply";

				PLYFormat ply;
				ply.Deserialize(ss.str());

				patchBuffers.FromPLYFile(ply);

				Time::End(t, "Loading PointCloud Patch", i);

				{
					t = Time::Now();
					nvtxRangePushA("Insert Points");

					thrust::for_each(
						thrust::make_zip_iterator(thrust::make_tuple(
							patchBuffers.inputPoints.begin(),
							patchBuffers.inputNormals.begin(),
							patchBuffers.inputColors.begin())),
						thrust::make_zip_iterator(thrust::make_tuple(
							patchBuffers.inputPoints.end(),
							patchBuffers.inputNormals.end(),
							patchBuffers.inputColors.end())),
						[=] __device__(thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f> t) {
						Eigen::Vector3f point = thrust::get<0>(t);
						Eigen::Vector3f normal = thrust::get<1>(t);
						Eigen::Vector3f color = thrust::get<2>(t);

						for (int dx = -voxelNeighborRange; dx <= voxelNeighborRange; ++dx) {
							for (int dy = -voxelNeighborRange; dy <= voxelNeighborRange; ++dy) {
								for (int dz = -voxelNeighborRange; dz <= voxelNeighborRange; ++dz) {
									Eigen::Vector3f offset(dx * voxelSize, dy * voxelSize, dz * voxelSize);
									Eigen::Vector3f voxelCenter = point + offset;

									uint3 index = GetIndex(volumeCenter, dimensions, voxelSize, voxelCenter);
									if (index.x == UINT_MAX || index.y == UINT_MAX || index.z == UINT_MAX) continue;

									size_t flatIndex = GetFlatIndex(index, dimensions);

									Voxel& voxel = d_volume[flatIndex];
									float distance = (voxelCenter - point).norm();
									atomicMinFloat(&voxel.minDistance, distance);

									if (fabs(distance - voxel.minDistance) < 1e-6f) {
										float tsdfValue = distance / truncationDistance;
										tsdfValue = (voxelCenter - point).dot(normal) > 0 ? tsdfValue : -tsdfValue;

										voxel.tsdfValue = tsdfValue;

										//if (1.0f < voxel.tsdfValue) voxel.tsdfValue = 1.0f;
										//if (-1.0f > voxel.tsdfValue) voxel.tsdfValue = -1.0f;

										voxel.weight++;

										voxel.color.x() = (voxel.color.x() + color.x()) / 2.0f;
										voxel.color.y() = (voxel.color.y() + color.y()) / 2.0f;
										voxel.color.z() = (voxel.color.z() + color.z()) / 2.0f;

										voxel.normal.x() = (voxel.normal.x() + normal.x()) / 2.0f;
										voxel.normal.y() = (voxel.normal.y() + normal.y()) / 2.0f;
										voxel.normal.z() = (voxel.normal.z() + normal.z()) / 2.0f;
									}
								}
							}
						}
					});

					nvtxRangePop();
					t = Time::End(t, "Insert Points");
				}
			}

			thrust::host_vector<Voxel> h_volume(volume);
			//for (size_t i = 0; i < numberOfVoxels; i++)
			//{
			//	Voxel& voxel = h_volume[i];

			//	auto tsdfValue = h_volume[i].tsdfValue;

			//	if (-0.05f < tsdfValue && tsdfValue < 0.05f)
			//	{
			//		if (fabs(voxel.tsdfValue) < truncationDistance) {
			//			int zKey = i / (dimensions.x * dimensions.y);
			//			int yKey = (i % (dimensions.x * dimensions.y)) / dimensions.x;
			//			int xKey = (i % (dimensions.x * dimensions.y)) % dimensions.x;

			//			float x = volumeMin.x() + (float)xKey * voxelSize;
			//			float y = volumeMin.y() + (float)yKey * voxelSize;
			//			float z = volumeMin.z() + (float)zKey * voxelSize;

			//			Color4 color = Color4::FromNormalized(voxel.color.x(), voxel.color.y(), voxel.color.z(), 1.0f);
			//			//VD::AddCube("temp", { x, y, z }, { voxelSize, voxelSize, voxelSize }, voxel.normal, color);
			//			VD::AddCube("temp", { x, y, z }, voxelSize, color);
			//		}
			//	}
			//}
			//Time::End(t, "Show Voxels");

			{
				thrust::device_vector<Eigen::Vector3f> vertices(volume.size() * 12);
				auto d_vertices = thrust::raw_pointer_cast(vertices.data());
				thrust::device_vector<uint3> triangles(volume.size() * 4);
				auto d_triangles = thrust::raw_pointer_cast(triangles.data());
				thrust::device_vector<int> vertexCounter(1, 0);
				thrust::device_vector<int> triangleCounter(1, 0);
				int* vertexCounterPtr = thrust::raw_pointer_cast(vertexCounter.data());
				int* triangleCounterPtr = thrust::raw_pointer_cast(triangleCounter.data());

				thrust::for_each(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(volume.size()),
					[=] __device__(size_t index) {

					unsigned int zKey = (unsigned int)index / (dimensions.x * dimensions.y);
					unsigned int yKey = ((unsigned int)index % (dimensions.x * dimensions.y)) / dimensions.x;
					unsigned int xKey = ((unsigned int)index % (dimensions.x * dimensions.y)) % dimensions.x;

					uint3 voxelIndex = make_uint3(xKey, yKey, zKey);
					auto voxel = d_volume[GetFlatIndex(voxelIndex, dimensions)];
					if (FLT_MAX == voxel.tsdfValue) return;
					if (-truncationDistance > voxel.tsdfValue || voxel.tsdfValue > truncationDistance) return;

					Eigen::Vector3f voxelNormal = voxel.normal;

					Eigen::Vector3f voxelPos = volumeMin + Eigen::Vector3f(
						(float)voxelIndex.x * voxelSize,
						(float)voxelIndex.y * voxelSize,
						(float)voxelIndex.z * voxelSize);

					const Eigen::Vector3f vertexOffsets[8] = {
						{0.0f, 0.0f, 0.0f},
						{1.0f, 0.0f, 0.0f},
						{1.0f, 0.0f, 1.0f},
						{0.0f, 0.0f, 1.0f},
						{0.0f, 1.0f, 0.0f},
						{1.0f, 1.0f, 0.0f},
						{1.0f, 1.0f, 1.0f},
						{0.0f, 1.0f, 1.0f} 
					};

					float tsdf[8];
					for (int i = 0; i < 8; ++i) {
						uint3 cornerIndex = make_uint3(
							voxelIndex.x + (vertexOffsets[i].x() > 0.0f ? 1 : 0),
							voxelIndex.y + (vertexOffsets[i].y() > 0.0f ? 1 : 0),
							voxelIndex.z + (vertexOffsets[i].z() > 0.0f ? 1 : 0));
						size_t cornerFlatIndex = GetFlatIndex(cornerIndex, dimensions);

						if (cornerFlatIndex < dimensions.x * dimensions.y * dimensions.z) {
							tsdf[i] = d_volume[cornerFlatIndex].tsdfValue;
						}
						else {
							tsdf[i] = FLT_MAX;
						}
					}

					int cubeIndex = 0;
					for (int i = 0; i < 8; ++i) {
						if (tsdf[i] != FLT_MAX &&
							(-truncationDistance <= tsdf[i] || tsdf[i] <= truncationDistance) &&
							tsdf[i] < isoValue)
						{
							cubeIndex |= (1 << i);
						}
					}

					int edges = edgeTable[cubeIndex];
					if (edges == 0) return;

					Eigen::Vector3f edgeVertices[12];
					for (int i = 0; i < 12; ++i) {
						if (edges & (1 << i)) {
							int i0 = edgeVertexMap[i][0];
							int i1 = edgeVertexMap[i][1];

							Eigen::Vector3f p0 = voxelPos + vertexOffsets[edgeVertexMap[i][0]] * voxelSize;
							Eigen::Vector3f p1 = voxelPos + vertexOffsets[edgeVertexMap[i][1]] * voxelSize;

							float alpha = 0.5f;
							float diff = tsdf[i0] - tsdf[i1];
							if (fabs(diff) > 1e-6) {
								alpha = (isoValue - tsdf[i0]) / (tsdf[i1] - tsdf[i0]);
								//alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
							}

							edgeVertices[i] = p0 + alpha * (p1 - p0);
						}
					}

					for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
						auto& v0 = edgeVertices[triTable[cubeIndex][i]];
						auto& v1 = edgeVertices[triTable[cubeIndex][i + 1]];
						auto& v2 = edgeVertices[triTable[cubeIndex][i + 2]];

						int localVertexIndex = atomicAdd(vertexCounterPtr, 3);
						int localTriangleIndex = atomicAdd(triangleCounterPtr, 1);

						// Assign vertices and triangles
						d_vertices[localVertexIndex] = v0;
						d_vertices[localVertexIndex + 1] = v1;
						d_vertices[localVertexIndex + 2] = v2;

						d_triangles[localTriangleIndex] = make_uint3(localVertexIndex, localVertexIndex + 1, localVertexIndex + 2);
					}
				});

				//{
				//	thrust::host_vector<Voxel> h_volume(volume);
				//	for (size_t i = 0; i < numberOfVoxels; i++)
				//	{
				//		Voxel& voxel = h_volume[i];

				//		auto tsdfValue = h_volume[i].tsdfValue;

				//		//if (tsdfValue > 1.0f)
				//		//{
				//		//	//printf("tsdfValue : %f\n", tsdfValue);

				//		//	int zKey = i / (dimensions.x * dimensions.y);
				//		//	int yKey = (i % (dimensions.x * dimensions.y)) / dimensions.x;
				//		//	int xKey = (i % (dimensions.x * dimensions.y)) % dimensions.x;

				//		//	float x = volumeMin.x() + (float)xKey * voxelSize;
				//		//	float y = volumeMin.y() + (float)yKey * voxelSize;
				//		//	float z = volumeMin.z() + (float)zKey * voxelSize;

				//		//	VD::AddCube("temp", { x, y, z }, voxelSize, Color4::Red);
				//		//	continue;
				//		//}

				//		if (-0.05f < tsdfValue && tsdfValue < 0.05f)
				//		{
				//			if (fabs(voxel.tsdfValue) < truncationDistance) {
				//				int zKey = i / (dimensions.x * dimensions.y);
				//				int yKey = (i % (dimensions.x * dimensions.y)) / dimensions.x;
				//				int xKey = (i % (dimensions.x * dimensions.y)) % dimensions.x;

				//				float x = volumeMin.x() + (float)xKey * voxelSize;
				//				float y = volumeMin.y() + (float)yKey * voxelSize;
				//				float z = volumeMin.z() + (float)zKey * voxelSize;

				//				Color4 color = Color4::FromNormalized(voxel.color.x(), voxel.color.y(), voxel.color.z(), 1.0f);
				//				//VD::AddCube("temp", { x, y, z }, { voxelSize, voxelSize, voxelSize }, voxel.normal, color);
				//				VD::AddCube("temp", { x, y, z }, voxelSize * 0.25, color);
				//			}
				//		}
				//	}
				//	Time::End(t, "Show Voxels");
				//}

				{
					// Resize vectors to fit actual counts
					vertices.resize(vertexCounter[0]);
					triangles.resize(triangleCounter[0]);


					thrust::host_vector<Eigen::Vector3f> h_vertices(vertices);
					thrust::host_vector<uint3> h_triangles(triangles);

					for (auto& t : h_triangles)
					{
						auto& i0 = t.x;
						auto& i1 = t.y;
						auto& i2 = t.z;

						printf("%d, %d, %d\n", i0, i1, i2);

						auto& v0 = h_vertices[i0];
						auto& v1 = h_vertices[i1];
						auto& v2 = h_vertices[i2];

						if (volumeMin.x() > v0.x() || v0.x() > volumeMax.x()) continue;
						if (volumeMin.y() > v0.y() || v0.y() > volumeMax.y()) continue;
						if (volumeMin.z() > v0.z() || v0.z() > volumeMax.z()) continue;

						if (volumeMin.x() > v1.x() || v1.x() > volumeMax.x()) continue;
						if (volumeMin.y() > v1.y() || v1.y() > volumeMax.y()) continue;
						if (volumeMin.z() > v1.z() || v1.z() > volumeMax.z()) continue;

						if (volumeMin.x() > v2.x() || v2.x() > volumeMax.x()) continue;
						if (volumeMin.y() > v2.y() || v2.y() > volumeMax.y()) continue;
						if (volumeMin.z() > v2.z() || v2.z() > volumeMax.z()) continue;


						printf("%f, %f, %f\n", v0.x(), v0.y(), v0.z());

						VD::AddTriangle("Mesh", v0, v1, v2, Color4::White);
					}
				
					Time::End(t, "Marching Cubes");
				}
			}
		}

		void TestMarchingCubes_HPP()
		{
			float* d_volume;
			cudaMallocManaged(&d_volume, sizeof(float) * 100 * 100 * 100);
			thrust::device_ptr<float> dev_ptr(d_volume);
			thrust::fill_n(d_volume, 1000000, 0.0f);

			thrust::for_each(thrust::counting_iterator(0), thrust::counting_iterator(1000000),
				[=] __device__(size_t index) {
				size_t ix = index % 100;
				size_t iy = (index / 100) % 100;
				size_t iz = index / 10000;

				float amplitude = 1.0f;
				float frequency = 1.0f;

				float x = (float)ix * 0.1f;
				float y = (float)iy * 0.1f;
				float z = (float)iz * 0.1f;
				float distance = amplitude * sinf(frequency * x) * sinf(frequency * (y - 3.0f)) * sinf(frequency * z);
				d_volume[index] = (y - distance);

				//printf("distance : %f\n", distance);
			});

			cudaDeviceSynchronize();

			float* h_volume = new float[1000000];
			cudaMemcpy(h_volume, d_volume, sizeof(float) * 1000000, cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();

			//PLYFormat ply;
			//for (size_t z = 0; z < 100; z++)
			//{
			//	for (size_t y = 0; y < 100; y++)
			//	{
			//		for (size_t x = 0; x < 100; x++)
			//		{
			//			size_t flatIndex = x + y * 100 + z * 10000;
			//			float distance = h_volume[flatIndex];

			//			if (distance < 50)
			//			{
			//				ply.AddPoint(x, y, z);
			//			}
			//		}
			//	}
			//}

			//ply.Serialize("C:\\Resources\\Debug\\Sphere.ply");

			::MarchingCubes::MarchingCubesSurfaceExtractor<float> mc(
				d_volume,
				make_float3(-5.0f, -5.0f, -5.0f),
				make_float3(5.0f, 5.0f, 5.0f),
				0.1f,
				1.0f);
			
			auto result = mc.Extract();
			
			PLYFormat ply;

			for (size_t i = 0; i < result.numberOfVertices; i++)
			{
				auto v = result.vertices[i];
				ply.AddPoint(v.x, v.y, v.z);
			}

			for (size_t i = 0; i < result.numberOfTriangles; i++)
			{
				auto t = result.triangles[i];
				ply.AddIndex(t.x);
				ply.AddIndex(t.y);
				ply.AddIndex(t.z);
			}
 
			ply.Serialize("C:\\Resources\\Debug\\TestSphere.ply");

			for (size_t i = 0; i < result.numberOfTriangles; i++)
			{
				auto i0 = result.triangles[i].x;
				auto i1 = result.triangles[i].y;
				auto i2 = result.triangles[i].z;

				auto v0 = result.vertices[i0];
				auto v1 = result.vertices[i1];
				auto v2 = result.vertices[i2];

				VD::AddTriangle("Marching Cubes", { v0.x, v0.y, v0.z }, { v1.x, v1.y, v1.z }, { v2.x, v2.y, v2.z }, Color4::White);
			}

			delete result.vertices;
			delete result.triangles;
		}
		
		void TestMarchingCubes_Fuse()
		{
			auto t = Time::Now();

			Eigen::Vector3f volumeMin(-20.0f, -20.0f, -20.0f);
			Eigen::Vector3f volumeMax(20.0f, 20.0f, 20.0f);
			Eigen::Vector3f diff = volumeMax - volumeMin;
			Eigen::Vector3f volumeCenter = (volumeMax + volumeMin) * 0.5f;
			float voxelSize = 0.1f;
			float truncationDistance = 0.5f;
			float isoValue = 0.0f;
			int voxelNeighborRange = (int)ceilf(truncationDistance / voxelSize);
			//voxelNeighborRange = 1;

			uint3 dimensions;
			dimensions.x = (uint32_t)ceilf(diff.x() / voxelSize);
			dimensions.y = (uint32_t)ceilf(diff.y() / voxelSize);
			dimensions.z = (uint32_t)ceilf(diff.z() / voxelSize);

			size_t numberOfVoxels = dimensions.x * dimensions.y * dimensions.z;

			thrust::device_vector<Voxel> volume(dimensions.x * dimensions.y * dimensions.z);
			auto d_volume = thrust::raw_pointer_cast(volume.data());

			thrust::for_each(thrust::counting_iterator((size_t)0), thrust::counting_iterator(volume.size()),
				[=] __device__(size_t index) {
				d_volume[index].minDistance = FLT_MAX;
				d_volume[index].tsdfValue = FLT_MAX;
				d_volume[index].weight = 0;
				d_volume[index].normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				d_volume[index].color = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
			});

			Time::End(t, "Initialize Volume");

			LoadTRNFile();

			PatchBuffers patchBuffers(256, 480);

			//size_t i = 0;
			for (size_t i = 0; i < 10; i++)
			//for (size_t i = 0; i < 4252; i++)
			{
				stringstream ss;
				ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_" << i << ".ply";
				//ss << "C:\\Resources\\2D\\Captured\\PointCloud\\point_crop.ply";

				PLYFormat ply;
				ply.Deserialize(ss.str());

				patchBuffers.FromPLYFile(ply);
				auto d_patchBuffersPoints = thrust::raw_pointer_cast(patchBuffers.inputPoints.data());
				auto d_patchBuffersNormals = thrust::raw_pointer_cast(patchBuffers.inputNormals.data());
				auto d_patchBuffersColors = thrust::raw_pointer_cast(patchBuffers.inputColors.data());

				Time::End(t, "Loading PointCloud Patch", i);

				{
					t = Time::Now();
					nvtxRangePushA("Insert Points");

					thrust::for_each(
						thrust::counting_iterator<size_t>(0),
						thrust::counting_iterator<size_t>(patchBuffers.numberOfInputPoints),
						[=] __device__(size_t pointIndex) {

						Eigen::Vector3f& point = d_patchBuffersPoints[pointIndex];
						Eigen::Vector3f& normal = d_patchBuffersNormals[pointIndex];
						Eigen::Vector3f& color = d_patchBuffersColors[pointIndex];

						uint3 index = GetIndex(volumeCenter, dimensions, voxelSize, point);
						if (index.x == UINT_MAX || index.y == UINT_MAX || index.z == UINT_MAX) return;

						for (int nz = index.z - voxelNeighborRange; nz < index.z + voxelNeighborRange; nz++)
						{
							if (dimensions.z <= nz) continue;

							for (int ny = index.y - voxelNeighborRange; ny < index.y + voxelNeighborRange; ny++)
							{
								if (dimensions.y <= ny) continue;

								for (int nx = index.x - voxelNeighborRange; nx < index.x + voxelNeighborRange; nx++)
								{
									if (dimensions.x <= nx) continue;

									auto neighborIndex = make_uint3(nx, ny, nz);
									auto voxelPosition = GetPosition(volumeCenter, dimensions, voxelSize, neighborIndex);

									auto flatIndex = GetFlatIndex(neighborIndex, dimensions);

									Voxel& voxel = d_volume[flatIndex];
									float distance = (voxelPosition - point).norm();
									atomicMinFloat(&voxel.minDistance, distance);

									float tsdfValue = distance;// / truncationDistance;
									tsdfValue = (voxelPosition - point).dot(normal) > 0 ? tsdfValue : -tsdfValue;

									float newTSDFValue = (voxel.weight * voxel.tsdfValue + tsdfValue) / (voxel.weight + 1.0f);
									voxel.tsdfValue = newTSDFValue;

									if (1.0f < voxel.tsdfValue) voxel.tsdfValue = 1.0f;
									if (-1.0f > voxel.tsdfValue) voxel.tsdfValue = -1.0f;

									voxel.weight = voxel.weight + 1.0f;

									voxel.color.x() = (voxel.color.x() + color.x()) / 2.0f;
									voxel.color.y() = (voxel.color.y() + color.y()) / 2.0f;
									voxel.color.z() = (voxel.color.z() + color.z()) / 2.0f;

									voxel.normal.x() = (voxel.normal.x() + normal.x()) / 2.0f;
									voxel.normal.y() = (voxel.normal.y() + normal.y()) / 2.0f;
									voxel.normal.z() = (voxel.normal.z() + normal.z()) / 2.0f;
								}
							}
						}
					});

					nvtxRangePop();
					t = Time::End(t, "Insert Points");
				}
			}

			thrust::device_vector<float> field(dimensions.x * dimensions.y * dimensions.z);
			auto d_field = thrust::raw_pointer_cast(field.data());
			thrust::for_each(thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(dimensions.x * dimensions.y * dimensions.z),
				[=] __device__(unsigned int index) {
				d_field[index] = d_volume[index].tsdfValue;

				//auto zIndex =  index / (dimensions.x * dimensions.y);
				//auto yIndex = (index % (dimensions.x * dimensions.y)) / dimensions.x;
				//auto xIndex = (index % (dimensions.x * dimensions.y)) % dimensions.x;

				//if (200 == xIndex && 200 == yIndex && 200 == zIndex)
				//{
				//	if (FLT_MAX == d_field[index])
				//	{
				//		printf("tsdfValue : %f\n", d_field[index]);
				//	}
				//}
			});

			::MarchingCubes::MarchingCubesSurfaceExtractor<float> mc(
				d_field,
				make_float3(-20.0f, -20.0f, -20.0f),
				make_float3(20.0f, 20.0f, 20.0f),
				0.1f,
				0.0f);
			
			auto result = mc.Extract();

			{
				thrust::host_vector<float> h_field(mc.h_internal->numberOfVoxels);
				auto t_field = thrust::raw_pointer_cast(h_field.data());
				cudaMemcpy(t_field, mc.h_internal->data, sizeof(float)* mc.h_internal->numberOfVoxels, cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				
				for (size_t i = 0; i < h_field.size(); i++)
				{
					auto zIndex = i / (dimensions.x * dimensions.y);
					auto yIndex = (i % (dimensions.x * dimensions.y)) / dimensions.x;
					auto xIndex = (i % (dimensions.x * dimensions.y)) % dimensions.x;

					if (FLT_MAX != h_field[i])
					{
						auto position = GetPosition({ 0.0f, 0.0f, 0.0f }, dimensions, voxelSize, make_uint3(xIndex, yIndex, zIndex));
						VD::AddCube("occupied", position, { 0.05f, 0.05f, 0.05f }, { 0.0f, 0.0f, 1.0f }, Color4::White);
					}
				}
			}

			PLYFormat ply;

			for (size_t i = 0; i < result.numberOfVertices; i++)
			{
				auto v = result.vertices[i];
				ply.AddPoint(v.x, v.y, v.z);
			}

			for (size_t i = 0; i < result.numberOfTriangles; i++)
			{
				auto t = result.triangles[i];
				ply.AddIndex(t.x);
				ply.AddIndex(t.y);
				ply.AddIndex(t.z);
			}

			ply.Serialize("C:\\Resources\\Debug\\Field.ply");

			for (size_t i = 0; i < result.numberOfTriangles; i++)
			{
				auto i0 = result.triangles[i].x;
				auto i1 = result.triangles[i].y;
				auto i2 = result.triangles[i].z;

				auto v0 = result.vertices[i0];
				auto v1 = result.vertices[i1];
				auto v2 = result.vertices[i2];

				VD::AddTriangle("Marching Cubes", { v0.x, v0.y, v0.z }, { v1.x, v1.y, v1.z }, { v2.x, v2.y, v2.z }, Color4::White);
			}

			delete result.vertices;
			delete result.triangles;
		}

		void TestMarchingCubes()
		{
			//TestMarchingCubes_Patches();
			//TestMarchingCubes_HPP();

			TestMarchingCubes_Fuse();
		}
	}
}
