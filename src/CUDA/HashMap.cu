#include "HashMap.cuh"

#include <vtkHeaderFiles.h>

#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
	namespace HashMap
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

		//const int numBuckets = 10;

		//// 1차 해싱 함수
		//struct FirstHash
		//{
		//	int numBuckets;
		//	FirstHash(int _numBuckets) : numBuckets(_numBuckets) {}

		//	__host__ __device__
		//		int operator()(int key) const
		//	{
		//		return key % numBuckets;
		//	}
		//};

		//// 2차 해싱 함수: 각 버킷 내에서 고유한 인덱스를 계산
		//struct SecondHash
		//{
		//	int bucketStartOffset;
		//	int numSlots;

		//	SecondHash(int offset, int _numSlots) : bucketStartOffset(offset), numSlots(_numSlots) {}

		//	__host__ __device__
		//		int operator()(int key) const
		//	{
		//		return bucketStartOffset + ((key * key) % numSlots); // 간단히 제곱 연산으로 충돌 해결
		//	}
		//};

		void TestHashMap()
		{
	/*		auto t = Time::Now();

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

			Eigen::Vector3f min(-100.0f, -100.0f, -100.0f);
			Eigen::Vector3f max(100.0f, 100.0f, 100.0f);
			int maxDepth = 13;

			printf("patchBuffers.numberOfInputPoints : %d\n", patchBuffers.numberOfInputPoints);

			thrust::device_vector<uint64_t> mortonCodes(patchBuffers.numberOfInputPoints);

			t = Time::Now();
			nvtxRangePushA("ConvertToMortonCode");
			thrust::transform(
				patchBuffers.inputPoints.begin(), patchBuffers.inputPoints.end(),
				mortonCodes.begin(),
				[=] __device__(const Eigen::Vector3f & point) {
				return GetMortonCode(min, max, maxDepth, point);
			});
			nvtxRangePop();
			t = Time::End(t, "Convert To MortonCode");*/
		}
	}
}
