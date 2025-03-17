#include "Clustering.cuh"

#include <App/Utility.h>

#include <App/Serialization.hpp>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

#include "Cache.cuh"

namespace CUDA
{
	namespace Clustering
	{
        struct Voxel
        {
            float3 position;
            unsigned int label;
        };

        __global__ void Kernel_ClearVoxels(
            Voxel* d_voxels,
            unsigned int numberOfVoxels,
            dim3 volumeDimensions,
            float voxelSize,
            float3 volumeMin,
            float3 volumeCenter)
        {
            unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
            if (threadid >= volumeDimensions.x * volumeDimensions.y * volumeDimensions.z) return;

            d_voxels[threadid].position = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
            d_voxels[threadid].label = threadid;
        }

        void ClearVoxels(
            Voxel* d_voxels,
            unsigned int numberOfVoxels,
            dim3 volumeDimensions,
            float voxelSize,
            float3 volumeMin,
            float3 volumeCenter)
        {
            nvtxRangePush("ClearVoxels");

            unsigned int blockSize = 256;
            unsigned int gridSize = (numberOfVoxels + blockSize - 1) / blockSize;
            Kernel_ClearVoxels<<<gridSize, blockSize>>>(d_voxels, numberOfVoxels, volumeDimensions, voxelSize, volumeMin, volumeCenter);

            cudaDeviceSynchronize();
            nvtxRangePop();
        }

        __global__ void Kernel_OccupyVoxels(
            float* d_points,
            unsigned int numberOfPoints,
            Voxel* d_voxels,
            unsigned int numberOfVoxels,
            dim3 volumeDimensions,
            float voxelSize,
            float3 volumeMin,
            float3 volumeCenter,
            dim3* occupiedVoxelIndices,
            unsigned int* numberOfOccupiedVoxelIndices)
        {
            unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
            if (threadid >= numberOfPoints) return;

            auto gx = d_points[threadid * 3];
            auto gy = d_points[threadid * 3 + 1];
            auto gz = d_points[threadid * 3 + 2];

            if (gx < volumeMin.x || gx > volumeMin.x + volumeDimensions.x * voxelSize ||
                gy < volumeMin.y || gy > volumeMin.y + volumeDimensions.y * voxelSize ||
                gz < volumeMin.z || gz > volumeMin.z + volumeDimensions.z * voxelSize)
            {
                return;
            }

            unsigned int ix = (unsigned int)floorf((gx - volumeMin.x) / voxelSize);
            unsigned int iy = (unsigned int)floorf((gy - volumeMin.y) / voxelSize);
            unsigned int iz = (unsigned int)floorf((gz - volumeMin.z) / voxelSize);

            if (ix >= volumeDimensions.x  || iy >= volumeDimensions.y || iz >= volumeDimensions.z) return;

            unsigned int volumeIndex = iz * volumeDimensions.x * volumeDimensions.y + iy * volumeDimensions.x + ix;
            auto& voxel = d_voxels[volumeIndex];
            
            voxel.position.x = volumeMin.x + ix * voxelSize;
            voxel.position.y = volumeMin.y + iy * voxelSize;
            voxel.position.z = volumeMin.z + iz * voxelSize;
            voxel.label = volumeIndex;

            //alog("%f, %f, %f\n", voxel.position.x, voxel.position.y, voxel.position.z);

            auto index = atomicAdd(numberOfOccupiedVoxelIndices, 1);
            occupiedVoxelIndices[index] = dim3(ix, iy, iz);
            //alog("%d\n", index);
        }

        void OccupyVoxels(
            float* d_points,
            unsigned int numberOfPoints,
            Voxel* d_voxels,
            unsigned int numberOfVoxels,
            dim3 volumeDimensions,
            float voxelSize,
            float3 volumeMin,
            float3 volumeCenter,
            dim3* occupiedVoxelIndices,
            unsigned int* numberOfOccupiedVoxelIndices)
        {
            nvtxRangePush("OccupyVoxels");

            unsigned int blockSize = 256;
            unsigned int gridSize = (numberOfPoints + blockSize - 1) / blockSize;

            Kernel_OccupyVoxels<<<gridSize, blockSize>>>(
                d_points,
                numberOfPoints,
                d_voxels,
                numberOfVoxels,
                volumeDimensions,
                voxelSize,
                volumeMin,
                volumeCenter,
                occupiedVoxelIndices,
                numberOfOccupiedVoxelIndices);

            cudaDeviceSynchronize();
            nvtxRangePop();
        }

        __device__ __forceinline__ unsigned int FindRoot(Voxel* d_voxels, unsigned int index)
        {
            while (d_voxels[index].label != index)
            {
                unsigned int parent = d_voxels[index].label;
                unsigned int grandparent = d_voxels[parent].label;

                if (parent != grandparent)
                {
                    atomicCAS(&d_voxels[index].label, parent, grandparent);
                }
                index = d_voxels[index].label;
            }
            return index;
        }

        __device__ __forceinline__ void Union(Voxel* d_voxels, unsigned int a, unsigned int b)
        {
            unsigned int rootA = FindRoot(d_voxels, a);
            unsigned int rootB = FindRoot(d_voxels, b);

            if (rootA != rootB)
            {
                if (rootA < rootB)
                    atomicMin(&d_voxels[rootB].label, rootA);
                else
                    atomicMin(&d_voxels[rootA].label, rootB);
            }
        }

        __global__ void Kernel_ConnectedComponentLabeling(
            Voxel* d_voxels,
            dim3* occupiedVoxelIndices,
            unsigned int numberOfOccupiedVoxels,
            dim3 volumeDimensions)
        {
            unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
            if (threadid >= numberOfOccupiedVoxels) return;

            dim3 voxelIdx = occupiedVoxelIndices[threadid];
            unsigned int index = voxelIdx.z * volumeDimensions.x * volumeDimensions.y + voxelIdx.y * volumeDimensions.x + voxelIdx.x;

            // Ensure the voxel is occupied
            if (d_voxels[index].position.x == FLT_MAX) return;

            // 6-connected neighborhood (¡¾X, ¡¾Y, ¡¾Z)
            int dx[6] = { 1, -1, 0, 0, 0, 0 };
            int dy[6] = { 0, 0, 1, -1, 0, 0 };
            int dz[6] = { 0, 0, 0, 0, 1, -1 };

            for (int i = 0; i < 6; i++)
            {
                int nx = voxelIdx.x + dx[i];
                int ny = voxelIdx.y + dy[i];
                int nz = voxelIdx.z + dz[i];

                if (nx >= 0 && nx < volumeDimensions.x &&
                    ny >= 0 && ny < volumeDimensions.y &&
                    nz >= 0 && nz < volumeDimensions.z)
                {
                    unsigned int neighborIndex = nz * volumeDimensions.x * volumeDimensions.y + ny * volumeDimensions.x + nx;

                    // Check if the neighbor is occupied
                    if (d_voxels[neighborIndex].position.x != FLT_MAX)
                    {
                        Union(d_voxels, index, neighborIndex);
                    }
                }
            }
        }

        void ConnectedComponentLabeling(
            Voxel* d_voxels,
            dim3* occupiedVoxelIndices,
            unsigned int numberOfOccupiedVoxelIndices,
            dim3 volumeDimensions)
        {
            nvtxRangePush("ConnectedComponentLabeling");

            unsigned int blockSize = 256;
            unsigned int gridSize = (numberOfOccupiedVoxelIndices + blockSize - 1) / blockSize;

            for (int i = 0; i < 20; i++) // Increase iterations to ensure full convergence
            {
                Kernel_ConnectedComponentLabeling << <gridSize, blockSize >> > (
                    d_voxels, occupiedVoxelIndices, numberOfOccupiedVoxelIndices, volumeDimensions);
                cudaDeviceSynchronize();
            }

            cudaDeviceSynchronize();
            nvtxRangePop();
        }

        void VisualizeVoxels(
            Voxel* d_voxels,
            unsigned int numberOfVoxels,
            dim3 volumeDimensions,
            float voxelSize,
            float3 volumeMin)
        {
            nvtxRangePush("VisualizeVoxels");

            Voxel* h_voxels = new Voxel[numberOfVoxels];
            cudaMemcpy(h_voxels, d_voxels, sizeof(Voxel) * numberOfVoxels, cudaMemcpyDeviceToHost);

            std::unordered_map<unsigned int, std::tuple<unsigned char, unsigned char, unsigned char>> labelToColor;

            for (size_t i = 0; i < numberOfVoxels; i++)
            {
                auto& voxel = h_voxels[i];

                if (voxel.position.x != FLT_MAX) // Only visualize occupied voxels
                {
                    unsigned int label = voxel.label;

                    // Assign a unique color per label using a hash function
                    if (labelToColor.find(label) == labelToColor.end())
                    {
                        unsigned char r = (label * 53) % 256;
                        unsigned char g = (label * 97) % 256;
                        unsigned char b = (label * 151) % 256;
                        labelToColor[label] = std::make_tuple(r, g, b);
                    }

                    // Get the assigned color
                    auto [r, g, b] = labelToColor[label];

                    // Visualize the voxel with the computed color
                    VD::AddCube("labeled voxels", { voxel.position.x, voxel.position.y, voxel.position.z },
                        0.05f, { r, g, b, 255 });
                }
            }

            delete[] h_voxels;

            cudaDeviceSynchronize();
            nvtxRangePop();
        }

		void TestClustering()
		{
			PLYFormat ply;

			ply.Deserialize("C:\\Resources\\Debug\\Serialized\\Debugging_1_1002.ply");

			for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
			{
				auto x = ply.GetPoints()[i * 3];
				auto y = ply.GetPoints()[i * 3 + 1];
				auto z = ply.GetPoints()[i * 3 + 2];

				auto r = ply.GetColors()[i * 3];
				auto g = ply.GetColors()[i * 3 + 1];
				auto b = ply.GetColors()[i * 3 + 2];

				VD::AddSphere("points", { x, y, z }, 0.05f, {(unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f), (unsigned char)(b * 255.0f), 255});

                //VD::AddCube("occupid voxels", { x, y, z }, 0.05f);
			}

            nvtxRangePush("TestClustering");

            float* d_points = nullptr;
            cudaMalloc(&d_points, sizeof(float) * ply.GetPoints().size());
            cudaMemcpy(d_points, ply.GetPoints().data(), sizeof(float) * ply.GetPoints().size(), cudaMemcpyHostToDevice);

            unsigned int numberOfPoints = ply.GetPoints().size() / 3;
            dim3 volumeDimensions(400, 400, 400);
            unsigned int numberOfVoxels = volumeDimensions.x * volumeDimensions.y * volumeDimensions.z;
            float voxelSize = 0.1f;
            float3 volumeCenter = make_float3(3.9904f, -15.8357f, -7.2774f);
            float3 volumeMin = make_float3(
                volumeCenter.x - (float)(volumeDimensions.x / 2) * voxelSize,
                volumeCenter.y - (float)(volumeDimensions.y / 2) * voxelSize,
                volumeCenter.z - (float)(volumeDimensions.z / 2) * voxelSize);

            Voxel* d_voxels = nullptr;
            cudaMalloc(&d_voxels, sizeof(Voxel) * numberOfVoxels);

            unsigned int* d_labels = nullptr;
            cudaMalloc(&d_labels, sizeof(unsigned int) * numberOfVoxels);

            dim3* occupiedVoxelIndices = nullptr;
            cudaMalloc(&occupiedVoxelIndices, sizeof(dim3) * 5000000);
            unsigned int* numberOfOccupiedVoxelIndices = nullptr;
            cudaMalloc(&numberOfOccupiedVoxelIndices, sizeof(unsigned int));
            cudaMemset(numberOfOccupiedVoxelIndices, 0, sizeof(unsigned int));

            ClearVoxels(d_voxels, numberOfVoxels, volumeDimensions, voxelSize, volumeMin, volumeCenter);

            OccupyVoxels(
                d_points,
                numberOfPoints,
                d_voxels,
                numberOfVoxels,
                volumeDimensions,
                voxelSize,
                volumeMin,
                volumeCenter,
                occupiedVoxelIndices,
                numberOfOccupiedVoxelIndices);

            unsigned int h_numberOfOccupiedVoxelIndices = 0;
            cudaMemcpy(&h_numberOfOccupiedVoxelIndices, numberOfOccupiedVoxelIndices, sizeof(unsigned int), cudaMemcpyDeviceToHost);

            ConnectedComponentLabeling(d_voxels, occupiedVoxelIndices, h_numberOfOccupiedVoxelIndices, volumeDimensions);

            VisualizeVoxels(
                d_voxels,
                numberOfVoxels,
                volumeDimensions,
                voxelSize,
                volumeMin);

            //connectedComponentLabelingCUDA(pointCloud);

            //for (auto& p : pointCloud)
            //{
            //    auto x = p.x;
            //    auto y = p.y;
            //    auto z = p.z;

            //    auto r = p.label / 255.0f;
            //    auto g = p.label / 255.0f;
            //    auto b = p.label / 255.0f;

            //    VD::AddSphere("points_result", { x, y, z }, 0.05f, { (unsigned char)(r * 255.0f), (unsigned char)(g * 255.0f), (unsigned char)(b * 255.0f), 255 });
            //}

            cudaFree(d_points);
            cudaFree(d_voxels);
            cudaFree(d_labels);
            cudaFree(occupiedVoxelIndices);
            cudaFree(numberOfOccupiedVoxelIndices);

            cudaDeviceSynchronize();
            nvtxRangePop();
		}
	}
}
