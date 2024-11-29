#pragma once

#include <assert.h>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <stack>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <nvtx3/nvToolsExt.h>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

static const char* _cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const* const func, const char* const file,
    int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        assert(false);
        exit(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define checkCudaSync(st)  {checkCudaErrors(cudaStreamSynchronize(st));checkCudaErrors(cudaGetLastError());}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char* errorMessage, const char* file,
    const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
        assert(false);
        exit(EXIT_FAILURE);
    }
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char* errorMessage, const char* file,
    const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    }
}
//
//// CUDA error check macro
//#define cudaCheckError() { \
//    cudaError_t e = cudaGetLastError(); \
//    if (e != cudaSuccess) { \
//        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << "\n"; \
//        exit(1); \
//    } \
//}


__host__ __device__
inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__
inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__
inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__host__ __device__
inline float3 operator*(const float3& a, float scalar) {
    return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__host__ __device__
inline float3 operator*(float scalar, const float3& a) {
    return make_float3(a.x * scalar, a.y * scalar, a.z * scalar);
}

__host__ __device__
inline float3& operator*=(float3& a, float scalar) {
    a.x *= scalar;
    a.y *= scalar;
    a.z *= scalar;
    return a;
}

__host__ __device__
inline float3 operator/(const float3& a, float scalar) {
    return make_float3(a.x / scalar, a.y / scalar, a.z / scalar);
}

__host__ __device__
inline float3& operator/=(float3& a, float scalar) {
    a.x /= scalar;
    a.y /= scalar;
    a.z /= scalar;
    return a;
}

__host__ __device__
inline float length(const float3& a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__
inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__
inline float3 normalize(const float3& a) {
    float len = length(a);
    if (len > 0.0f) {
        return a / len;
    }
    else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}
