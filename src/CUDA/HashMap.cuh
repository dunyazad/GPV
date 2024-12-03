#pragma once

#include "CUDA_Common.cuh"

#include <App/Serialization.hpp>

namespace CUDA
{
	namespace HashMap
	{
		class PatchBuffers
		{
		public:
			PatchBuffers(int width = 256, int height = 480);
			~PatchBuffers();

			void Clear();
			void FromPLYFile(const PLYFormat& ply);

			int width;
			int height;
			size_t numberOfInputPoints;

			thrust::device_vector<Eigen::Vector3f> inputPoints;
			thrust::device_vector<Eigen::Vector3f> inputNormals;
			thrust::device_vector<Eigen::Vector3f> inputColors;
		};

		void TestHashMap();
	}
}
