#pragma once

typedef char i8;
typedef unsigned char ui8;
typedef short i16;
typedef unsigned short ui16;
typedef int i32;
typedef unsigned int ui32;
typedef long i64;
typedef unsigned long ui64;
typedef float f32;
typedef double f64;

#define BUILD_FOR_DEVELOPMENT

#define NOMINMAX

#include <stdHeaderFiles.h>
#include <vtkHeaderFiles.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <Max.h>

typedef unsigned char ubyte;

namespace Time
{
	chrono::steady_clock::time_point Now();

    uint64_t Microseconds(chrono::steady_clock::time_point& from, chrono::steady_clock::time_point& now);

    chrono::steady_clock::time_point End(chrono::steady_clock::time_point& from, const string& message = "", int number = -1);

    string DateTime();
}

Eigen::Vector3f Transform(const Eigen::Matrix4f& tm, const Eigen::Vector3f& p);

Eigen::Matrix4f vtkToEigen(const vtkMatrix4x4* vtkMat);
vtkSmartPointer<vtkMatrix4x4> eigenToVtk(const Eigen::Matrix4f& eigenMat);
