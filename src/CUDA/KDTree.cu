#include "KDTree.cuh"

#define WINDOW_SIZE 3

namespace Algorithm
{
	KDTreeNode::KDTreeNode()
	{
	}

	KDTreeNode::~KDTreeNode()
	{
	}

	KDTree::KDTree()
	{
	}

	KDTree::~KDTree()
	{
	}
}

namespace CUDA
{
	void Test()
	{
		printf("Test\n");
	}

    __device__ void swap(float& a, float& b) {
        float tmp = a;
        a = b;
        b = tmp;
    }

    __global__ void bitonicSort(float* data, int numberOfPoints, int step, int stage) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= numberOfPoints)
            return;

        int ixj = idx ^ step;

        if (ixj > idx) {
            if ((idx & stage) == 0) {
                if (data[idx] > data[ixj]) {
                    swap(data[idx], data[ixj]);
                }
            }
            else {
                if (data[idx] < data[ixj]) {
                    swap(data[idx], data[ixj]);
                }
            }
        }
    }

    std::vector<Eigen::Vector3f> BitonicSort(Eigen::Vector3f* points, int numberOfPoints)
    {
        float* d_data;
        cudaMalloc(&d_data, sizeof(Eigen::Vector3f) * numberOfPoints);
        cudaMemcpy(d_data, points, sizeof(Eigen::Vector3f) * numberOfPoints, cudaMemcpyHostToDevice);

        dim3 blocks((numberOfPoints + 255) / 256);
        dim3 threads(256);

        nvtxRangePushA("Bitonic Sort");

        for (int stage = 2; stage <= numberOfPoints; stage <<= 1) {
            for (int step = stage >> 1; step > 0; step >>= 1) {
                bitonicSort << <blocks, threads >> > (d_data, numberOfPoints, step, stage);
                cudaDeviceSynchronize();

                printf("step : %d\n", step);
            }
        }

        nvtxRangePop();

        std::vector<Eigen::Vector3f> result(numberOfPoints);
        cudaMemcpy(result.data(), d_data, sizeof(Eigen::Vector3f) * numberOfPoints, cudaMemcpyDeviceToHost);

        cudaFree(d_data);

        return result;
    }

    __device__ void bubbleSort(float* window, int windowSize) {
        for (int i = 0; i < windowSize - 1; i++) {
            for (int j = 0; j < windowSize - i - 1; j++) {
                if (window[j] > window[j + 1]) {
                    float temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }
    }

    // CUDA kernel to perform median filtering on an Eigen::Vector3f array
    __global__ void medianFilter3D(Eigen::Vector3f* input, Eigen::Vector3f* output, int width, int height) {
        // Compute the x, y index of the current thread
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Ensure the thread is within the bounds of the array
        if (x >= width || y >= height) {
            return;
        }

        // Define the window size for median filtering
        const int halfWindowSize = WINDOW_SIZE / 2;

        // Initialize arrays to hold the window values for x, y, z components
        float windowX[WINDOW_SIZE * WINDOW_SIZE];
        float windowY[WINDOW_SIZE * WINDOW_SIZE];
        float windowZ[WINDOW_SIZE * WINDOW_SIZE];

        int windowIndex = 0;

        // Loop through the window around the current pixel
        for (int dy = -halfWindowSize; dy <= halfWindowSize; ++dy) {
            for (int dx = -halfWindowSize; dx <= halfWindowSize; ++dx) {
                int nx = min(max(x + dx, 0), width - 1); // Clamp to array bounds
                int ny = min(max(y + dy, 0), height - 1); // Clamp to array bounds

                // Load the window values for each component (x, y, z)
                Eigen::Vector3f neighbor = input[ny * width + nx];

                if (FLT_MAX == neighbor.x() || FLT_MAX == neighbor.y() || FLT_MAX == neighbor.z())
                    continue;

                windowX[windowIndex] = neighbor.x();
                windowY[windowIndex] = neighbor.y();
                windowZ[windowIndex] = neighbor.z();
                windowIndex++;
            }
        }

        // Sort each window to find the median
        bubbleSort(windowX, WINDOW_SIZE * WINDOW_SIZE);
        bubbleSort(windowY, WINDOW_SIZE * WINDOW_SIZE);
        bubbleSort(windowZ, WINDOW_SIZE * WINDOW_SIZE);

        // Store the median values in the output array
        output[y * width + x] = Eigen::Vector3f(windowX[WINDOW_SIZE * WINDOW_SIZE / 2],
            windowY[WINDOW_SIZE * WINDOW_SIZE / 2],
            windowZ[WINDOW_SIZE * WINDOW_SIZE / 2]);
    }

    std::vector<Eigen::Vector3f> CUDA::DoFilter(Eigen::Vector3f* points)
    {
        std::vector<Eigen::Vector3f> result;

        int width = 256;
        int height = 480;

        // Create an Eigen array to hold the input data
        Eigen::Matrix<Eigen::Vector3f, Eigen::Dynamic, Eigen::Dynamic> input(height, width);
        Eigen::Matrix<Eigen::Vector3f, Eigen::Dynamic, Eigen::Dynamic> output(height, width);

        // Initialize the input array with some data (this should be replaced with actual data)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                ///input(y, x) = Eigen::Vector3f(x, y, x + y); // Example values
                input(y, x) = points[y * 256 + x];
            }
        }

        // Allocate device memory for input and output arrays
        Eigen::Vector3f* d_input;
        Eigen::Vector3f* d_output;
        cudaMalloc(&d_input, width * height * sizeof(Eigen::Vector3f));
        cudaMalloc(&d_output, width * height * sizeof(Eigen::Vector3f));

        // Copy input data to device
        cudaMemcpy(d_input, input.data(), width * height * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);

        // Define block and grid sizes
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        // Launch the kernel
        medianFilter3D << <gridDim, blockDim >> > (d_input, d_output, width, height);

        // Copy the output data back to the host
        cudaMemcpy(output.data(), d_output, width * height * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);

        // Print the result for testing (this can be replaced with further processing)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Eigen::Vector3f p = output(y, x).transpose();

                if (p.x() == FLT_MAX || p.y() == FLT_MAX || p.z() == FLT_MAX)
                    continue;

                result.push_back(p);
                //std::cout << p << " ";
            }
            //std::cout << std::endl;
        }

        return result;
    }
}