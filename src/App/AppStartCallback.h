#pragma once

#include <Common.h>
#include <App/App.h>
#include <App/AppEventHandlers.h>

#include <CUDA/CUDA.cuh>
#include <CUDA/HashMap.cuh>
#include <CUDA/MarchingCubes.cuh>
#include <CUDA/Octree.cuh>
#include <CUDA/PSR.cuh>
#include <CUDA/RegularGrid.cuh>
#include <CUDA/SVO.cuh>

extern int pid;
extern size_t size_0;
extern size_t size_45;
extern float transform_0[16];
extern float transform_45[16];
extern int transformIndex;
extern Eigen::Vector3f cameraPosition;
extern unsigned char image_0[400 * 480];
extern unsigned char image_45[400 * 480];
extern Eigen::Vector3f points_0[400 * 480];
extern Eigen::Vector3f points_45[400 * 480];
extern bool enabledToCapture;
extern Eigen::AlignedBox3f aabb_0;
extern Eigen::AlignedBox3f aabb_45;
extern Eigen::AlignedBox3f gaabb_0;
extern Eigen::AlignedBox3f gaabb_45;
extern Eigen::AlignedBox3f taabb;
extern Eigen::AlignedBox3f lmax;

extern vector<Eigen::Vector3f> patchPoints_0;
extern vector<Eigen::Vector3f> patchPoints_45;
extern vector<Eigen::Vector3f> inputPoints;

extern vector<Eigen::Matrix4f> cameraTransforms;

void LoadPatch(int patchID, vtkRenderer* renderer);

tuple<Eigen::Matrix4f, Eigen::Vector3f> LoadPatchTransform(int patchID);

void SaveTRNFile();

void LoadTRNFile();

void LoadModel(vtkRenderer* renderer, const string& filename);

void MoveCamera(App* pApp, vtkCamera* camera, const Eigen::Matrix4f& tm);

void CaptureNextFrame(App* pApp);

void LoadDepthImage();

bool AppStartCallback(App* pApp);
void AppStartCallback_Capture(App* pApp);
void AppStartCallback_HashMap(App* pApp);
void AppStartCallback_Integrate(App* pApp);
void AppStartCallback_MarchingCubes(App* pApp);
void AppStartCallback_Octree(App* pApp);
void AppStartCallback_PSR(App* pApp);
void AppStartCallback_RegularGrid(App* pApp);
void AppStartCallback_Simple(App* pApp);
void AppStartCallback_SVO(App* pApp);
