#include "RegularGrid.cuh"

#include <vtkHeaderFiles.h>

#include <App/ResourceIO.h>
#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
	namespace RegularGrid
	{
        struct RegularGrid
        {
            float3 globalMinPosition = make_float3(0.0f, 0.0f, 0.0f);
            dim3 dimensions = dim3(400, 400, 400);
            float voxelSize = 0.1f;
            cudaArray* d_voxels = nullptr;
            cudaSurfaceObject_t surfaceObject;
        };

        void InitializeRegularGrid(RegularGrid* regularGrid)
        {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaExtent volumeSize = make_cudaExtent(regularGrid->dimensions.x, regularGrid->dimensions.y, regularGrid->dimensions.z);

            checkCudaErrors(cudaMalloc3DArray(&regularGrid->d_voxels, &channelDesc, volumeSize, cudaArraySurfaceLoadStore));

            struct cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = regularGrid->d_voxels;
            checkCudaErrors(cudaCreateSurfaceObject(&regularGrid->surfaceObject, &resDesc));
        }

        __global__ void Kernel_ClearCache(RegularGrid regularGrid)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z) return;

            int zIndex = (idx / (regularGrid.dimensions.x * regularGrid.dimensions.y));
            int yIndex = (idx % (regularGrid.dimensions.x * regularGrid.dimensions.y)) / regularGrid.dimensions.x;
            int xIndex = (idx % (regularGrid.dimensions.x * regularGrid.dimensions.y)) % regularGrid.dimensions.x;

            if (xIndex < 0 || xIndex >= regularGrid.dimensions.x ||
                yIndex < 0 || yIndex >= regularGrid.dimensions.y ||
                zIndex < 0 || zIndex >= regularGrid.dimensions.z)
                return;

            float tsdf = FLT_MAX;

            surf3Dwrite<float>(tsdf, regularGrid.surfaceObject, xIndex * sizeof(float), yIndex, zIndex);
        }

        __global__ void Kernel_IntegrateInputPoints(RegularGrid regularGrid, Eigen::Vector3f* inputPoints, unsigned int numberOfInputPoints)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= numberOfInputPoints) return;

            float3 p = make_float3(inputPoints[idx].x(), inputPoints[idx].y(), inputPoints[idx].z());
            if (p.x == FLT_MAX || p.y == FLT_MAX || p.z == FLT_MAX) return;

            int xIndex = __float2int_rd((p.x - regularGrid.globalMinPosition.x + (regularGrid.dimensions.x / 2) * regularGrid.voxelSize) / regularGrid.voxelSize);
            int yIndex = __float2int_rd((p.y - regularGrid.globalMinPosition.y + (regularGrid.dimensions.y / 2) * regularGrid.voxelSize) / regularGrid.voxelSize);
            int zIndex = __float2int_rd((p.z - regularGrid.globalMinPosition.z + (regularGrid.dimensions.z / 2) * regularGrid.voxelSize) / regularGrid.voxelSize);

            if (xIndex < 0 || xIndex >= regularGrid.dimensions.x ||
                yIndex < 0 || yIndex >= regularGrid.dimensions.y ||
                zIndex < 0 || zIndex >= regularGrid.dimensions.z)
                return;

            float3 vp = make_float3(
                regularGrid.globalMinPosition.x - (regularGrid.dimensions.x / 2) * regularGrid.voxelSize + xIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize,
                regularGrid.globalMinPosition.y - (regularGrid.dimensions.y / 2) * regularGrid.voxelSize + yIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize,
                regularGrid.globalMinPosition.z - (regularGrid.dimensions.z / 2) * regularGrid.voxelSize + zIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize);

            //Debugging::AddPointP(0, p);
            //Debugging::AddPointP(1, vp);

            float tsdf = vp.z - p.z;
            float oldTSDF = surf3Dread<float>(regularGrid.surfaceObject, xIndex * sizeof(float), yIndex, zIndex);
            if (oldTSDF > 100000.0f)
                oldTSDF = 100.0f;
            tsdf = oldTSDF < tsdf ? oldTSDF : tsdf;

            surf3Dwrite<float>(tsdf, regularGrid.surfaceObject, xIndex * sizeof(float), yIndex, zIndex);

            int offset = 10;

            for (int zOffset = -offset; zOffset <= offset; zOffset++)
            {
                int nzIndex = zIndex + zOffset;
                if (zIndex + zOffset < 0 || zIndex + zOffset >= regularGrid.dimensions.z) continue;

                for (int yOffset = -offset; yOffset <= offset; yOffset++)
                {
                    int nyIndex = yIndex + yOffset;
                    if (yIndex + yOffset < 0 || yIndex + yOffset >= regularGrid.dimensions.y) continue;

                    for (int xOffset = -offset; xOffset <= offset; xOffset++)
                    {
                        int nxIndex = xIndex + xOffset;
                        if (xIndex + xOffset < 0 || xIndex + xOffset >= regularGrid.dimensions.x) continue;
                        if (0 == xOffset && 0 == yOffset && 0 == zOffset) continue;

                        if (nxIndex < 0 || nxIndex >= regularGrid.dimensions.x ||
                            nyIndex < 0 || nyIndex >= regularGrid.dimensions.y ||
                            nzIndex < 0 || nzIndex >= regularGrid.dimensions.z)
                            return;

                        auto np = vp + make_float3(
                            (float)xOffset * regularGrid.voxelSize,
                            (float)yOffset * regularGrid.voxelSize,
                            (float)zOffset * regularGrid.voxelSize);

                        float ntsdf = np.z - p.z;

                        float oldNTSDF = surf3Dread<float>(regularGrid.surfaceObject, (xIndex + xOffset) * sizeof(float), yIndex + yOffset, zIndex + zOffset);
                        if (oldNTSDF > 100000.0f)
                            oldNTSDF = 100.0f;
                        ntsdf = oldNTSDF < ntsdf ? oldNTSDF : ntsdf;
                        surf3Dwrite<float>(ntsdf, regularGrid.surfaceObject, (xIndex + xOffset) * sizeof(float), yIndex + yOffset, zIndex + zOffset);
                    }
                }
            }
        }

        __global__ void Kernel_SerializeGrid(RegularGrid regularGrid, Eigen::Vector3f* outputPoints)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z) return;

            int zIndex = idx / (regularGrid.dimensions.x * regularGrid.dimensions.y);
            int yIndex = (idx % (regularGrid.dimensions.x * regularGrid.dimensions.y)) / regularGrid.dimensions.x;
            int xIndex = (idx % (regularGrid.dimensions.x * regularGrid.dimensions.y)) % regularGrid.dimensions.x;

            float tsdf = surf3Dread<float>(regularGrid.surfaceObject, xIndex * sizeof(float), yIndex, zIndex);

            {
                if (1000000.0f > tsdf && -1000000.0f < tsdf)
                {
                    //if (0 > voxel.tsdfValue)
                    //if(FLT_MAX != tsdf || 0.0f != tsdf)
                    if (-0.05f <= tsdf && tsdf <= 0.05f)
                    {
                        //printf("tsdf : %f\n", tsdf);

                        float x = xIndex * regularGrid.voxelSize + regularGrid.globalMinPosition.x;
                        float y = yIndex * regularGrid.voxelSize + regularGrid.globalMinPosition.y;
                        float z = zIndex * regularGrid.voxelSize + regularGrid.globalMinPosition.z;
                        outputPoints[idx] = Eigen::Vector3f(x, y, z);
                    }
                }
            }
        }

		void TestRegularGrid()
		{
            auto transforms = ResourceIO::ReadTransformsFile("Debug/maxillar_transforms.bin");

            int patchIndex = 0;

            vector<Eigen::Vector3f> inputPoints;

            PLYFormat ply;
            //auto modelFilePath = ResourceIO::GetPath("Debug/Maxillar.ply");
            stringstream ss;
            ss << "Debug/Patches/point_" << patchIndex << ".ply";
            auto modelFilePath = ResourceIO::GetPath(ss.str());
            if (filesystem::exists(modelFilePath))
            {
                ply.Deserialize(modelFilePath.string());
            }

            for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
            {
                float x = ply.GetPoints()[i * 3];
                float y = ply.GetPoints()[i * 3 + 1];
                float z = ply.GetPoints()[i * 3 + 2];
                //VD::AddCube("points", {x,y,z}, 0.05f);

                //printf("%f, %f, %f\n", x,y,z);

                auto p = ((Eigen::Vector4f)(transforms[patchIndex] * Eigen::Vector4f(x, y, z, 1.0f))).head<3>();

                //cout << transforms[patchIndex] << endl;

                inputPoints.push_back(p);

                //printf("%f, %f, %f\n", p.x(), p.y(), p.z());
                
                VD::AddSphere("points", p, 0.05f);
            }

            uint32_t numberOfInputPoints = inputPoints.size();



            RegularGrid regularGrid;
            InitializeRegularGrid(&regularGrid);

            RegularGrid* d_regularGrid;
            checkCudaErrors(cudaMalloc(&d_regularGrid, sizeof(RegularGrid)));
            checkCudaErrors(cudaMemcpy(d_regularGrid, &regularGrid, sizeof(RegularGrid), cudaMemcpyHostToDevice));

            Eigen::Vector3f* d_points;
            checkCudaErrors(cudaMalloc(&d_points, sizeof(Eigen::Vector3f) * numberOfInputPoints));
            checkCudaErrors(cudaMemcpy(d_points, inputPoints.data(), sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyHostToDevice));

            int threads = 512;
            int blocks = (regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z + threads - 1) / threads;

            nvtxRangePushA("ClearCache");
            Kernel_ClearCache << <blocks, threads >> > (regularGrid);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaGetLastError());
            nvtxRangePop();

            blocks = (numberOfInputPoints + threads - 1) / threads;

            nvtxRangePushA("IntegrateInputPoints");
            Kernel_IntegrateInputPoints << <blocks, threads >> > (regularGrid, d_points, numberOfInputPoints);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaGetLastError());
            nvtxRangePop();

            Eigen::Vector3f* d_grid;
            checkCudaErrors(cudaMalloc(&d_grid, sizeof(Eigen::Vector3f) * regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z));
            checkCudaErrors(cudaMemset(d_grid, 0, sizeof(Eigen::Vector3f) * regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z));

            blocks = (regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z + threads - 1) / threads;

            nvtxRangePushA("SerializeGrid");
            Kernel_SerializeGrid << <blocks, threads >> > (regularGrid, d_grid);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaGetLastError());
            nvtxRangePop();

            Eigen::Vector3f* h_grid = new Eigen::Vector3f[regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z];
            checkCudaErrors(cudaMemcpy(h_grid, d_grid, sizeof(Eigen::Vector3f) * regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z, cudaMemcpyDeviceToHost));

            cudaDeviceSynchronize();

            for (size_t i = 0; i < regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z; i++)
            {
                auto p = h_grid[i];
                if ((p.x() != 0.0f || p.y() != 0.0f || p.z() != 0.0f) &&
                    (p.x() != FLT_MAX || p.y() != FLT_MAX || p.z() != FLT_MAX))
                {
                    VD::AddCube("occupied", p, 0.05f, Color4::White);
                }
            }

            delete[] h_grid;
            checkCudaErrors(cudaFree(d_grid));
            checkCudaErrors(cudaFree(d_points));
            checkCudaErrors(cudaFree(d_regularGrid));
		}
	}
}
