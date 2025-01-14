#include "ZSparseBlocks.cuh"

#include <App/Utility.h>

#include <App/Serialization.hpp>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
	namespace ZSparseBlocks
	{
		struct Node;
		struct Grid;

		struct Node
		{
			size_t pointIndex;
			unsigned int zIndex;
			Node* previous;
			Node* next;
		};

		struct Grid
		{
			Eigen::Vector3f gridMin;
			float voxelSize;

			unsigned int xLength;
			unsigned int yLength;

			unsigned int numberOfNodes;
			Node** nodes;

			unsigned int allocatedNodeIndex;
			Node* allocatedNodes;
		};

		tuple<Grid* , Grid*> InitializeGrid(
			const Eigen::Vector3f& gridMin,
			float voxelSize,
			unsigned int xLength,
			unsigned int yLength,
			unsigned int numberOfNodes = 3000000)
		{
			Grid* h_grid = new Grid;
			h_grid->gridMin = gridMin;
			h_grid->voxelSize = voxelSize;
			h_grid->xLength = xLength;
			h_grid->yLength = yLength;

			h_grid->numberOfNodes = numberOfNodes;
			
			h_grid->allocatedNodeIndex = 0;

			cudaMalloc(&h_grid->nodes, sizeof(Node*) * xLength * yLength);
			//cudaMemset(h_grid->nodes, 0, sizeof(Node*) * xLength * yLength);

			cudaMalloc(&h_grid->allocatedNodes, sizeof(Node) * numberOfNodes);
			//cudaMemset(h_grid->allocatedNodes, 0, sizeof(Node*) * numberOfNodes);

			Grid* d_grid;
			cudaMalloc(&d_grid, sizeof(Grid));
			cudaMemcpy(d_grid, h_grid, sizeof(Grid), cudaMemcpyHostToDevice);

			return make_tuple(h_grid, d_grid);
		}

		__global__ void Kernel_Initialize_Test(Grid* grid)
		{
			printf("grid->gridMin : %f, %f, %f\n", grid->gridMin.x(), grid->gridMin.y(), grid->gridMin.z());
			printf("grid->voxelSize : %f\n", grid->voxelSize);
			printf("grid->xLength : %d\n", grid->xLength);
			printf("grid->yLength : %d\n", grid->yLength);
			printf("grid->numberOfNodes : %d\n", grid->numberOfNodes);

			printf("grid->nodes : %p\n", grid->nodes);

			printf("grid->allocatedNodeIndex : %d\n", grid->allocatedNodeIndex);

			printf("grid->allocatedNodes : %p\n", grid->allocatedNodes);
		}

		void TerminateGrid(Grid* h_grid, Grid* d_grid)
		{
			if (nullptr != h_grid)
			{
				cudaFree(h_grid->nodes);
				cudaFree(h_grid->allocatedNodes);

				delete h_grid;
			}

			if (nullptr != d_grid)
			{
				cudaFree(d_grid);
			}
		}

		__device__
			uint3 GetIndex(
				Grid* grid, const Eigen::Vector3f& position)
		{
			unsigned int xIndex = (unsigned int)floorf((position.x() - grid->gridMin.x()) / grid->voxelSize);
			unsigned int yIndex = (unsigned int)floorf((position.y() - grid->gridMin.y()) / grid->voxelSize);
			unsigned int zIndex = (unsigned int)floorf((position.z() - grid->gridMin.z()) / grid->voxelSize);

			return make_uint3(xIndex, yIndex, zIndex);
		}

		__device__
			Eigen::Vector3f GetPosition(Grid* grid, unsigned int xIndex, unsigned int yIndex, unsigned int zIndex)
		{
			Eigen::Vector3f localPosition((float)xIndex * grid->voxelSize, (float)yIndex * grid->voxelSize, (float)zIndex * grid->voxelSize);
			return grid->gridMin + localPosition;
		}

		__device__
		Node* GetNode(Grid* grid, unsigned int xIndex, unsigned int yIndex, unsigned int zIndex)
		{
			Node* currentNode = grid->nodes[yIndex * grid->xLength + xIndex];
			while (nullptr != currentNode)
			{
				if (zIndex == currentNode->zIndex) return currentNode;

				if (zIndex < currentNode->zIndex) break;

				currentNode = currentNode->next;
			}

			return nullptr;
		}

		__device__
			void AddNode(Grid* grid, unsigned int xIndex, unsigned int yIndex, unsigned int zIndex, size_t pointIndex)
		{
			Node* currentNode = grid->nodes[yIndex * grid->xLength + xIndex];
			if (nullptr == currentNode)
			{
				auto nodeIndex = atomicAdd(&grid->allocatedNodeIndex, 1);
				auto newNode = &grid->allocatedNodes[nodeIndex];
				newNode->pointIndex = pointIndex;
				newNode->zIndex = zIndex;
				newNode->previous = nullptr;
				newNode->next = nullptr;
				grid->nodes[yIndex * grid->xLength + xIndex] = newNode;
				//printf("grid->nodes[yIndex * grid->xLength + xIndex] : %p\n", grid->nodes[yIndex * grid->xLength + xIndex]);
				return;
			}
			else
			{
				while (nullptr != currentNode)
				{
					if (zIndex > currentNode->zIndex)
					{
						auto nodeIndex = atomicAdd(&grid->allocatedNodeIndex, 1);
						auto newNode = &grid->allocatedNodes[nodeIndex];
						newNode->pointIndex = pointIndex;
						newNode->zIndex = zIndex;

						if (nullptr == currentNode->next)
						{
							currentNode->next = newNode;
							newNode->previous = currentNode;
							newNode->next = nullptr;
							return;
						}
						else
						{
							if (zIndex < currentNode->next->zIndex)
							{
								currentNode->next->previous = newNode;
								newNode->next = currentNode->next;

								currentNode->next = newNode;
								newNode->previous = currentNode;
							}
							//else
							//{
							//	printf("zIndex : %d, currentNode->zIndex : %d, currentNode->next->zIndex : %d\n",
							//		zIndex, currentNode->zIndex, currentNode->next->zIndex);
							//}
						}
					}
					
					currentNode = currentNode->next;
				}
			}
		}

		__global__
			void Kernel_InsertPoints(Grid* grid, Eigen::Vector3f* d_points, size_t numberOfPoints)
		{
			unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid > numberOfPoints - 1) return;

			//printf("numberOfPoints : %d\n", numberOfPoints);

			auto index = GetIndex(grid, d_points[threadid]);
			auto node = GetNode(grid, index.x, index.y, index.z);
			if (nullptr != node)
			{
				printf("%d, %d, %d\n", index.x, index.y, index.z);
			}
			if (nullptr == node)
			{
				AddNode(grid, index.x, index.y, index.z, threadid);
			}
		}

		__global__
			void Kernel_InsertPoints_Test(Grid* grid)
		{
			printf("grid->allocatedNodeIndex : %d\n", grid->allocatedNodeIndex);

			//for (size_t i = 0; i < grid->allocatedNodeIndex; i++)
			//{
			//	auto& node = grid->allocatedNodes[i];
			//	//printf("zIndex : %d\n", node.zIndex);
			//}

			for (size_t y = 0; y < grid->yLength; y++)
			{
				for (size_t x = 0; x < grid->xLength; x++)
				{
					auto node = grid->nodes[y * grid->xLength + x];
					if (nullptr != node)
					{
						printf("zIndex : %d\n", node->zIndex);
					}
				}
			}
		}

		void TestZSparseBlocks()
		{
			auto t = Time::Now();

			PLYFormat ply;
			ply.Deserialize("C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply");

			float voxelSize = 0.1f;
			auto aabbMin = ply.GetAABBMin();
			auto aabbMax = ply.GetAABBMax();

			//auto aabbMin = Eigen::Vector3f(aabb.min().minCoeff(), aabb.min().minCoeff(), aabb.min().minCoeff());
			//auto aabbMax = Eigen::Vector3f(aabb.max().maxCoeff(), aabb.max().maxCoeff(), aabb.max().maxCoeff());

			auto volumeMinX = floorf(aabbMin.x() / voxelSize) * voxelSize;
			auto volumeMinY = floorf(aabbMin.y() / voxelSize) * voxelSize;
			auto volumeMinZ = floorf(aabbMin.z() / voxelSize) * voxelSize;
			auto volumeMaxX = ceilf(aabbMax.x() / voxelSize) * voxelSize;
			auto volumeMaxY = ceilf(aabbMax.y() / voxelSize) * voxelSize;
			auto volumeMaxZ = ceilf(aabbMax.z() / voxelSize) * voxelSize;

			auto volumeMin = Eigen::Vector3f(volumeMinX, volumeMinY, volumeMinZ);
			auto volumeMax = Eigen::Vector3f(volumeMaxX, volumeMaxY, volumeMaxZ);

			//VD::AddBox("aabb", aabbMin, aabbMax, Color4::Blue);
			//VD::AddBox("volume", volumeMin, volumeMax, Color4::Red);

			//VD::AddLine("aabb", aabbMin, aabbMax, Color4::Blue);
			//VD::AddLine("volume", volumeMin, volumeMax, Color4::Red);

			vector<Eigen::Vector3f> loadedPoints;

			size_t numberOfPoints = ply.GetPoints().size() / 3;

			for (size_t i = 0; i < numberOfPoints; i++)
			{
				auto x = ply.GetPoints()[i * 3];
				auto y = ply.GetPoints()[i * 3 + 1];
				auto z = ply.GetPoints()[i * 3 + 2];

				VD::AddSphere("Points", { x, y, z }, 0.05f, Color4::White);

				loadedPoints.push_back({ x,y, z });
			}

			Eigen::Vector3f* d_points;
			cudaMalloc(&d_points, sizeof(Eigen::Vector3f) * numberOfPoints);
			cudaMemcpy(d_points, loadedPoints.data(), sizeof(Eigen::Vector3f) * numberOfPoints, cudaMemcpyHostToDevice);

			auto [h_grid, d_grid] = InitializeGrid({ -250.f, -250.f, -250.f }, 0.1f, 5000, 5000);

			checkCudaErrors(cudaDeviceSynchronize());

			Kernel_Initialize_Test << <1, 1 >> > (d_grid);

			checkCudaErrors(cudaDeviceSynchronize());

			int threadblocksize = 512;
			uint32_t gridsize = (numberOfPoints - 1) / threadblocksize;
			Kernel_InsertPoints << < gridsize, threadblocksize >> > (d_grid, d_points, numberOfPoints);

			checkCudaErrors(cudaDeviceSynchronize());

			printf("After InsertPoints\n");

			Kernel_InsertPoints_Test << <1, 1 >> > (d_grid);

			checkCudaErrors(cudaDeviceSynchronize());

			TerminateGrid(h_grid, d_grid);

			cudaFree(d_points);
		}
	}
}
