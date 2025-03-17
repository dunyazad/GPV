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
#define VOXEL_SIZE 0.1f
#define GRID_SIZE 400 // 400x400x400 Voxel Grid

        struct Point3D {
            float x, y, z;
            int label;  // 클러스터 ID
        };

        // CUDA 커널: Voxel Grid 매핑
        __global__ void mapToVoxelGrid(Point3D* d_points, int* d_voxelGrid, int numPoints) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numPoints) return;

            // Voxel Index 계산
            int vx = int(d_points[idx].x / VOXEL_SIZE);
            int vy = int(d_points[idx].y / VOXEL_SIZE);
            int vz = int(d_points[idx].z / VOXEL_SIZE);

            int voxelIndex = vx + vy * GRID_SIZE + vz * GRID_SIZE * GRID_SIZE;
            d_voxelGrid[idx] = voxelIndex;
        }

        // CUDA 커널: Union-Find 초기화
        __global__ void initLabels(int* d_labels, int numPoints) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < numPoints) {
                d_labels[idx] = idx;  // 초기에는 자기 자신을 루트로 설정
            }
        }

        // CUDA 커널: Find 함수 (경로 압축 적용)
        __device__ int find(int* labels, int i) {
            while (labels[i] != i) {
                labels[i] = labels[labels[i]]; // 경로 압축
                i = labels[i];
            }
            return i;
        }

        // CUDA 커널: Union-Find 병합
        __global__ void unionFind(int* d_labels, int* d_voxelGrid, int numPoints) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numPoints) return;

            int root1 = find(d_labels, idx);

            // 인접 Voxel 확인 (6-방향 연결)
            for (int i = 0; i < numPoints; i++) {
                if (idx != i && d_voxelGrid[idx] == d_voxelGrid[i]) {
                    int root2 = find(d_labels, i);
                    if (root1 != root2) {
                        d_labels[root2] = root1; // 병합
                    }
                }
            }
        }

        // CPU에서 실행하는 코드
        void connectedComponentLabelingCUDA(std::vector<Point3D>& points) {
            int numPoints = points.size();

            // CUDA 메모리 할당
            Point3D* d_points;
            int* d_voxelGrid;
            int* d_labels;

            cudaMalloc(&d_points, numPoints * sizeof(Point3D));
            cudaMalloc(&d_voxelGrid, numPoints * sizeof(int));
            cudaMalloc(&d_labels, numPoints * sizeof(int));

            // 데이터 복사 (CPU → GPU)
            cudaMemcpy(d_points, points.data(), numPoints * sizeof(Point3D), cudaMemcpyHostToDevice);

            int blockSize = 256;
            int gridSize = (numPoints + blockSize - 1) / blockSize;

            // Voxel Grid 매핑
            mapToVoxelGrid << <gridSize, blockSize >> > (d_points, d_voxelGrid, numPoints);
            cudaDeviceSynchronize();

            // Union-Find 초기화
            initLabels << <gridSize, blockSize >> > (d_labels, numPoints);
            cudaDeviceSynchronize();

            // Union-Find 병합
            unionFind << <gridSize, blockSize >> > (d_labels, d_voxelGrid, numPoints);
            cudaDeviceSynchronize();

            // 결과 복사 (GPU → CPU)
            std::vector<int> labels(numPoints);
            cudaMemcpy(labels.data(), d_labels, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

            // GPU 메모리 해제
            cudaFree(d_points);
            cudaFree(d_voxelGrid);
            cudaFree(d_labels);

            // 클러스터 ID 적용
            for (size_t i = 0; i < points.size(); i++) {
                points[i].label = labels[i];
            }
        }

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
            if (threadid > volumeDimensions.x * volumeDimensions.y * volumeDimensions.z - 1) return;

            d_voxels[threadid].position = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
            d_voxels[threadid].label = UINT32_MAX;
        }

        void ClearVoxels(
            Voxel* d_voxels,
            unsigned int numberOfVoxels,
            dim3 volumeDimensions,
            float voxelSize,
            float3 volumeMin,
            float3 volumeCenter)
        {
            unsigned int blockSize = 256;
            unsigned int gridSize = (numberOfVoxels + blockSize - 1) / blockSize;
            Kernel_ClearVoxels<<<gridSize, blockSize>>>(d_voxels, numberOfVoxels, volumeDimensions, voxelSize, volumeMin, volumeCenter);
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
            if (threadid > numberOfPoints - 1) return;

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

            voxel.label = threadid;

            auto index = atomicAdd(numberOfOccupiedVoxelIndices, 1);
            occupiedVoxelIndices[index] = dim3(ix, iy, iz);
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
        }

        __device__ int findRoot(int* labels, int i)
        {
            while (labels[i] != i)
            {
                labels[i] = labels[labels[i]];
                i = labels[i];
            }
            return i;
        }

        void VisualizeVoxels(
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
            Voxel* h_voxels = new Voxel[numberOfVoxels];
            cudaMemcpy(h_voxels, d_voxels, sizeof(Voxel) * numberOfVoxels, cudaMemcpyDeviceToHost);

            unsigned int h_numberOfOccupiedVoxelIndices = 0;
            cudaMemcpy(&h_numberOfOccupiedVoxelIndices, numberOfOccupiedVoxelIndices, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            
            dim3* h_occupiedVoxelIndices = new dim3[h_numberOfOccupiedVoxelIndices];
            cudaMemcpy(h_occupiedVoxelIndices, occupiedVoxelIndices, sizeof(dim3) * h_numberOfOccupiedVoxelIndices, cudaMemcpyDeviceToHost);

            alog("h_numberOfOccupiedVoxelIndices : %d\n", h_numberOfOccupiedVoxelIndices);

            for (size_t i = 0; i < h_numberOfOccupiedVoxelIndices; i++)
            {
                auto& index = h_occupiedVoxelIndices[i];
                //alog("%d, %d, %d\n", index.x, index.y, index.z);
                unsigned int flattenIndex = index.z * volumeDimensions.x * volumeDimensions.y + index.y * volumeDimensions.x + index.x;
                auto& voxel = h_voxels[flattenIndex];

                if (FLT_MAX != voxel.position.x && FLT_MAX != voxel.position.y && FLT_MAX != voxel.position.z)
                {
                    //alog("%f, %f, %f\n", voxel.position.x, voxel.position.y, voxel.position.z);
                    VD::AddCube("occupid voxels", { voxel.position.x, voxel.position.y, voxel.position.z }, 0.05f);
                }
            }

            delete h_voxels;
        }

		void TestClustering()
		{
            vector<Point3D> pointCloud;

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

                pointCloud.push_back({ x,y,z,(int)0 });
			}

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

            dim3* occupiedVoxelIndices = nullptr;
            cudaMalloc(&occupiedVoxelIndices, sizeof(dim3) * 5000000);
            unsigned int* numberOfOccupiedVoxelIndices = nullptr;
            cudaMalloc(&numberOfOccupiedVoxelIndices, sizeof(unsigned int));

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

            VisualizeVoxels(
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
            cudaFree(occupiedVoxelIndices);
            cudaFree(numberOfOccupiedVoxelIndices);

		}
	}
}
