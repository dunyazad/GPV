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
				ply.GetPoints().data(),
				sizeof(Eigen::Vector3f) * numberOfInputPoints,
				cudaMemcpyHostToDevice);

			cudaMemcpy(thrust::raw_pointer_cast(inputColors.data()),
				ply.GetPoints().data(),
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
			Eigen::Vector3f normal;
			int weight;
			float divergence;
			__host__ __device__ Voxel() : normal(0.0f, 0.0f, 0.0f), weight(0), divergence(0.0f) {}
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

			Eigen::Vector3f min(-5.0f, -5.0f, -5.0f);
			Eigen::Vector3f max(5.0f, 5.0f, 5.0f);
			//Eigen::Vector3f min(-75.0f, -75.0f, -75.0f);
			//Eigen::Vector3f max(75.0f, 75.0f, 75.0f);
			Eigen::Vector3f diff = max - min;
			Eigen::Vector3f center = (max + min) * 0.5f;
			float voxelSize = 0.1f;
			uint3 dimensions;
			dimensions.x = (uint32_t)ceilf(diff.x() / voxelSize);
			dimensions.y = (uint32_t)ceilf(diff.y() / voxelSize);
			dimensions.z = (uint32_t)ceilf(diff.z() / voxelSize);
			

			thrust::device_vector<Voxel> volume(dimensions.x * dimensions.y * dimensions.z);
			Voxel defaultVoxel;
			thrust::fill_n(volume.begin(), dimensions.x * dimensions.y * dimensions.z, defaultVoxel);
			auto d_volume = thrust::raw_pointer_cast(volume.data());

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

					Voxel& cv = d_volume[index];

					size_t piX = indexX; if (indexX > 0) piX--;
					size_t niX = indexX; if (indexX < dimensions.x - 1) niX++;
					size_t piY = indexY; if (indexY > 0) piY--;
					size_t niY = indexY; if (indexY < dimensions.y - 1) niY++;
					size_t piZ = indexZ; if (indexZ > 0) piZ--;
					size_t niZ = indexZ; if (indexZ < dimensions.z - 1) niZ++;

					uint3 indexP = make_uint3(piX, piY, piZ);
					size_t flatIndexP = GetFlatIndex(indexP, dimensions);
					Voxel& pv = d_volume[flatIndexP];

					uint3 indexN = make_uint3(niX, niY, niZ);
					size_t flatIndexN = GetFlatIndex(indexN, dimensions);
					Voxel& nv = d_volume[flatIndexN];

					Eigen::Vector3f dn = (nv.normal / (float)nv.weight - pv.normal / (float)pv.weight) / (2.0f * voxelSize);
					cv.divergence = dn.x() + dn.y() + dn.z();
				});
				nvtxRangePop();
				t = Time::End(t, "Compute Divergence");
			}

			{
				// Add cubes where volume value is not zero
				nvtxRangePushA("Add Cubes");
				thrust::host_vector<Voxel> h_volume = volume; // Copy device vector to host
				for (uint32_t z = 0; z < dimensions.z; ++z)
				{
					for (uint32_t y = 0; y < dimensions.y; ++y)
					{
						for (uint32_t x = 0; x < dimensions.x; ++x)
						{
							uint3 index = make_uint3(x, y, z);
							size_t flatIndex = GetFlatIndex(index, dimensions);
							if (h_volume[flatIndex].weight > 0) // Only add cubes where the count is greater than 0
							{
								Eigen::Vector3f position = GetPosition(center, dimensions, voxelSize, index);
								VD::AddCube("volume", position, { voxelSize, voxelSize, voxelSize },
									{ 0.0f, 0.0f, 1.0f }, Color4::FromNormalized(0.0f, 0.5f, 1.0f, 0.5f));
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
