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
        struct Voxel
        {
            float tsdfValue = FLT_MAX;
            float weight = 1.0f;
            float3 normal = make_float3(0.0f, 0.0f, 0.0f);
            float3 color = make_float3(1.0f, 1.0f, 1.0f);
        };

        struct RegularGrid
        {
            float3 globalMinPosition = make_float3(0.0f, 0.0f, 0.0f);
            dim3 dimensions = dim3(400, 400, 400);
            float voxelSize = 0.1f;
            cudaArray* d_voxels1 = nullptr;
            cudaArray* d_voxels2 = nullptr;
            cudaSurfaceObject_t surfaceObject1;
            cudaSurfaceObject_t surfaceObject2;
        };

        void InitializeRegularGrid(RegularGrid* regularGrid)
        {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            cudaExtent volumeSize = make_cudaExtent(regularGrid->dimensions.x, regularGrid->dimensions.y, regularGrid->dimensions.z);

            checkCudaErrors(cudaMalloc3DArray(&regularGrid->d_voxels1, &channelDesc, volumeSize, cudaArraySurfaceLoadStore));
            checkCudaErrors(cudaMalloc3DArray(&regularGrid->d_voxels2, &channelDesc, volumeSize, cudaArraySurfaceLoadStore));

            struct cudaResourceDesc resDesc1 = {};
            resDesc1.resType = cudaResourceTypeArray;
            resDesc1.res.array.array = regularGrid->d_voxels1;
            checkCudaErrors(cudaCreateSurfaceObject(&regularGrid->surfaceObject1, &resDesc1));

            struct cudaResourceDesc resDesc2 = {};
            resDesc2.resType = cudaResourceTypeArray;
            resDesc2.res.array.array = regularGrid->d_voxels2;
            checkCudaErrors(cudaCreateSurfaceObject(&regularGrid->surfaceObject2, &resDesc2));
        }

        __device__ float AtomicMinFloat(float* addr, float value)
        {
            int* addr_as_int = (int*)addr;
            int old = *addr_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
            } while (assumed != old);
            return __int_as_float(old);
        }

        __global__ void Kernel_ClearCache(RegularGrid regularGrid)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= regularGrid.dimensions.x * regularGrid.dimensions.y * regularGrid.dimensions.z) return;

            int zIndex = idx / (regularGrid.dimensions.x * regularGrid.dimensions.y);
            int yIndex = (idx % (regularGrid.dimensions.x * regularGrid.dimensions.y)) / regularGrid.dimensions.x;
            int xIndex = idx % regularGrid.dimensions.x;

            if (xIndex < 0 || xIndex >= regularGrid.dimensions.x ||
                yIndex < 0 || yIndex >= regularGrid.dimensions.y ||
                zIndex < 0 || zIndex >= regularGrid.dimensions.z)
                return;

            Voxel voxel;
            voxel.tsdfValue = FLT_MAX;
            voxel.weight = 1.0f;
            voxel.normal = make_float3(0.0f, 0.0f, 0.0f);
            voxel.color = make_float3(1.0f, 1.0f, 1.0f);

            float4 data1 = make_float4(voxel.tsdfValue, voxel.weight, voxel.normal.x, voxel.normal.y);
            float4 data2 = make_float4(voxel.normal.z, voxel.color.x, voxel.color.y, voxel.color.z);

            surf3Dwrite(data1, regularGrid.surfaceObject1, xIndex * sizeof(float4), yIndex, zIndex);
            surf3Dwrite(data2, regularGrid.surfaceObject2, xIndex * sizeof(float4), yIndex, zIndex);
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
                regularGrid.globalMinPosition.x + xIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize,
                regularGrid.globalMinPosition.y + yIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize,
                regularGrid.globalMinPosition.z + zIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize);

            float4 data1 = surf3Dread<float4>(regularGrid.surfaceObject1, xIndex * sizeof(float4), yIndex, zIndex);
            float4 data2 = surf3Dread<float4>(regularGrid.surfaceObject2, xIndex * sizeof(float4), yIndex, zIndex);

            //printf("%f, %f, %f\n", vp.x, vp.y, vp.z);

            Voxel voxel;
            voxel.tsdfValue = data1.x;
            voxel.weight = data1.y;
            voxel.normal = make_float3(data1.z, data1.w, data2.x);
            voxel.color = make_float3(data2.y, data2.z, data2.w);



            float3 voxelCenter = make_float3(
                regularGrid.globalMinPosition.x + xIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize,
                regularGrid.globalMinPosition.y + yIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize,
                regularGrid.globalMinPosition.z + zIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize);

            float distance = length(voxelCenter - p);
            float truncation = 1.0f;
            float newTSDF = fmaxf(-truncation, fminf(distance / truncation, truncation));
            newTSDF = (dot(voxelCenter - p, voxel.normal) >= 0.0f) ? newTSDF : -newTSDF;

            if (FLT_MAX == voxel.tsdfValue)
            {
                voxel.tsdfValue = newTSDF;
            }
            else
            {
                voxel.tsdfValue = (voxel.tsdfValue * voxel.weight + newTSDF) / (voxel.weight + 1.0f);
            }
            voxel.weight += 1.0f;
            
            //Voxel voxel;
            //voxel.tsdfValue = 1.0f;
            //voxel.weight = 1.0f;
            //voxel.normal = make_float3(1.0f, 0.0f, 0.0f);
            //voxel.color = make_float3(1.0f, 0.0f, 0.0f);
            
            data1 = make_float4(voxel.tsdfValue, voxel.weight, voxel.normal.x, voxel.normal.y);
            data2 = make_float4(voxel.normal.z, voxel.color.x, voxel.color.y, voxel.color.z);

            surf3Dwrite(data1, regularGrid.surfaceObject1, xIndex * sizeof(float4), yIndex, zIndex);
            surf3Dwrite(data2, regularGrid.surfaceObject2, xIndex * sizeof(float4), yIndex, zIndex);
            
            //printf("%f, %f, %f : %f\n", vp.x, vp.y, vp.z, voxel.tsdfValue);

            int offset = 1;

            for (int zOffset = -offset; zOffset <= offset; zOffset++)
            {
                if (zIndex + zOffset < 0 || zIndex + zOffset >= regularGrid.dimensions.z) continue;

                for (int yOffset = -offset; yOffset <= offset; yOffset++)
                {
                    if (yIndex + yOffset < 0 || yIndex + yOffset >= regularGrid.dimensions.y) continue;

                    for (int xOffset = -offset; xOffset <= offset; xOffset++)
                    {
                        if (xIndex + xOffset < 0 || xIndex + xOffset >= regularGrid.dimensions.x) continue;
                        if (0 == xOffset && 0 == yOffset && 0 == zOffset) continue;

                        int nxIndex = xIndex + xOffset;
                        int nyIndex = yIndex + yOffset;
                        int nzIndex = zIndex + zOffset;

                        if (nxIndex < 0 || nxIndex >= regularGrid.dimensions.x ||
                            nyIndex < 0 || nyIndex >= regularGrid.dimensions.y ||
                            nzIndex < 0 || nzIndex >= regularGrid.dimensions.z)
                            return;

                        float3 nvp = make_float3(
                            regularGrid.globalMinPosition.x + nxIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize,
                            regularGrid.globalMinPosition.y + nyIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize,
                            regularGrid.globalMinPosition.z + nzIndex * regularGrid.voxelSize + 0.5f * regularGrid.voxelSize);

                        float4 ndata1 = surf3Dread<float4>(regularGrid.surfaceObject1, nxIndex * sizeof(float4), nyIndex, nzIndex);
                        float4 ndata2 = surf3Dread<float4>(regularGrid.surfaceObject2, nxIndex * sizeof(float4), nyIndex, nzIndex);

                        Voxel nvoxel;
                        nvoxel.tsdfValue = ndata1.x;
                        nvoxel.weight = ndata1.y;
                        nvoxel.normal = make_float3(ndata1.z, ndata1.w, ndata2.x);
                        nvoxel.color = make_float3(ndata2.y, ndata2.z, ndata2.w);

                        float ndx = nvp.x - p.x;
                        float ndy = nvp.y - p.y;
                        float ndz = nvp.z - p.z;

                        float ndistance = sqrtf(ndx * ndx + ndy * ndy + ndz * ndz);

                        float truncation = 1.0f;
                        float newTSDF = fmaxf(-truncation, fminf(ndistance / truncation, truncation));

                        //printf("newTSDF write: %f %f %f -> %f\n", nvp.x, nvp.y, nvp.z, newTSDF);

                        if (FLT_MAX == nvoxel.tsdfValue)
                        {
                            nvoxel.tsdfValue = newTSDF;
                        }
                        else
                        {
                            nvoxel.tsdfValue = (nvoxel.tsdfValue * nvoxel.weight + newTSDF) / (nvoxel.weight + 1.0f);
                        }

                        //printf("voxel.tsdfValue write: %f %f %f -> %f\n", nvp.x, nvp.y, nvp.z, voxel.tsdfValue);

                        nvoxel.weight += 1.0f;

                        //Voxel voxel;
                        //voxel.tsdfValue = 1.0f;
                        //voxel.weight = 1.0f;
                        //voxel.normal = make_float3(1.0f, 0.0f, 0.0f);
                        //voxel.color = make_float3(1.0f, 0.0f, 0.0f);

                        ndata1 = make_float4(nvoxel.tsdfValue, nvoxel.weight, nvoxel.normal.x, nvoxel.normal.y);
                        ndata2 = make_float4(nvoxel.normal.z, nvoxel.color.x, nvoxel.color.y, nvoxel.color.z);

                        //surf3Dwrite(ndata1, regularGrid.surfaceObject1, nxIndex * sizeof(float4), nyIndex, nzIndex);
                        //surf3Dwrite(ndata2, regularGrid.surfaceObject2, nxIndex * sizeof(float4), nyIndex, nzIndex);

                        //printf("Before write: %f %f %f -> %f\n", nvp.x, nvp.y, nvp.z, nvoxel.tsdfValue);
                        surf3Dwrite(ndata1, regularGrid.surfaceObject1, nxIndex * sizeof(float4), nyIndex, nzIndex);
                        surf3Dwrite(ndata2, regularGrid.surfaceObject2, nxIndex * sizeof(float4), nyIndex, nzIndex);
                        //printf("After write: %f %f %f -> %f\n", nvp.x, nvp.y, nvp.z, nvoxel.tsdfValue);
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

            float4 data1 = surf3Dread<float4>(regularGrid.surfaceObject1, xIndex * sizeof(float4), yIndex, zIndex);
            float4 data2 = surf3Dread<float4>(regularGrid.surfaceObject2, xIndex * sizeof(float4), yIndex, zIndex);

            Voxel voxel;
            voxel.tsdfValue = data1.x;
            voxel.weight = data1.y;
            voxel.normal = make_float3(data1.z, data1.w, data2.x);
            voxel.color = make_float3(data2.y, data2.z, data2.w);

            if (voxel.tsdfValue != FLT_MAX)
            {
                //if (-0.05f <= voxel.tsdfValue && voxel.tsdfValue <= 0.05f)
                {
                    //if (0 > voxel.tsdfValue)
                    {
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

            ui32 numberOfInputPoints = inputPoints.size();



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
