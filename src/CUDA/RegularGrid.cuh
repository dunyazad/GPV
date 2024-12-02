#pragma once

#include "CUDA_Common.cuh"

#include <App/Serialization.hpp>

namespace CUDA
{
    template<typename T> class RegularGrid;

	class PatchBuffers
	{
	public:
		PatchBuffers(int width = 256, int height = 480);
		~PatchBuffers();

		int width;
		int height;
		size_t numberOfInputPoints;
		float3* inputPoints;
		float3* inputNormals;
		float3* inputColors;

		void Initalize();
		void Terminate();

		void Clear();

		void FromPLYFile(const PLYFormat& ply);
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

	struct ExtractionEdge
	{
		uint3 startVoxelIndex;
		uint3 endVoxelIndex;
		unsigned char edgeDirection = 0;
		bool zeroCrossing = false;
		//Eigen::Vector3f zeroCrossingPoint = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
		uint32_t zeroCrossingPointIndex = UINT32_MAX;
		int neighborCount = 0;
	};

	struct ExtractionTriangle
	{
		uint32_t edgeIndices[3] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
		uint32_t vertexIndices[3] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
	};

	struct ExtractionVoxel
	{
		uint32_t globalIndexX = UINT32_MAX;
		uint32_t globalIndexY = UINT32_MAX;
		uint32_t globalIndexZ = UINT32_MAX;

		float3 position = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		float value = FLT_MAX;

		uint32_t edgeIndexX = UINT32_MAX;
		uint32_t edgeIndexY = UINT32_MAX;
		uint32_t edgeIndexZ = UINT32_MAX;

		ExtractionTriangle triangles[4] = { ExtractionTriangle(), ExtractionTriangle(), ExtractionTriangle(), ExtractionTriangle() };
		uint32_t numberOfTriangles = 0;
	};

    void TestRegularGrid();
}