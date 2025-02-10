#include "CUDA_Common.cuh"

#include "../App/Serialization.hpp"

//#include <math_functions.h>

/*
#pragma region Debugging
const unsigned int POINT_COUNT = 50000000;

unsigned int h_frameNumber = 0;
unsigned int GetFrameNumber() { return h_frameNumber; }

__device__ unsigned int frameNumber = 0;
__device__ int captureEnabled = 0;
__device__ Point* points = nullptr;
__device__ unsigned int numberOfPoints = 0;
__device__ unsigned int frameCount = 0;
__device__ unsigned int flushCount = 0;

Point* d_points;

bool Debugging::captureOneFrame = false;
bool Debugging::recording = false;
std::map<int, std::string> Debugging::tagNameMapping;
//Eigen::Matrix4f Debugging::transform;

void Debugging::Initialize()
{
	cudaMalloc(&d_points, sizeof(Point) * POINT_COUNT);
	cudaMemcpyToSymbol(points, &d_points, sizeof(Point*));

	int enabled = 0;
	cudaMemcpyToSymbol((const void*)captureEnabled, &enabled, sizeof(int));
}

void Debugging::Terminate()
{
	cudaFree(d_points);
}

void Debugging::BeginFrame()
{
	//alog("[%4d] Debugging::BeginFrame() Begin\n", h_frameNumber);

	if (recording)
	{
		captureOneFrame = true;

		//alog("[%4d] Debugging::BeginFrame() Recording\n", h_frameNumber);
	}

	if (captureOneFrame)
	{
		int enabled = 1;
		cudaMemcpyToSymbol((const void*)captureEnabled, &enabled, sizeof(int));

		//alog("[%4d] set captureEnabled to 1\n", h_frameNumber);
	}

	//alog("[%4d] Debugging::BeginFrame() End\n", h_frameNumber);
}

void Debugging::EndFrame()
{
	//alog("[%4d] Debugging::EndFrame() Begin\n", h_frameNumber);

	if (captureOneFrame)
	{
		Flush();
	}

	cudaMemcpyFromSymbol(&h_frameNumber, (const void*)frameNumber, sizeof(unsigned int));
	h_frameNumber++;
	cudaMemcpyToSymbol((const void*)frameNumber, &h_frameNumber, sizeof(unsigned int));

	//alog("[%4d] Debugging::EndFrame() End\n", h_frameNumber);
}

void Debugging::CaptureOneFrame()
{
	captureOneFrame = true;
}

int Debugging::IsCaptureEnabled()
{
	return captureEnabled;
}

void Debugging::Flush()
{
	//alog("[%4d] Debugging::Flush() Begin\n", h_frameNumber);

	unsigned int h_numberOfPoints = 0;
	cudaMemcpyFromSymbol(&h_numberOfPoints, (const void*)numberOfPoints, sizeof(unsigned int));

	unsigned int h_flushCount = 0;
	cudaMemcpyFromSymbol(&h_flushCount, (const void*)flushCount, sizeof(unsigned int));

	cudaMemcpyFromSymbol(&d_points, points, sizeof(Point*));
	Point* h_points = new Point[h_numberOfPoints];
	cudaMemcpy(h_points, d_points, sizeof(Point) * h_numberOfPoints, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	//alog("[%4d] h_numberOfPoints : %d\n", h_frameNumber, h_numberOfPoints);

	map<int, PLYFormat> plys;

	for (size_t i = 0; i < h_numberOfPoints; i++)
	{
		int tag = h_points[i].tag;
		plys[tag].AddPoint(h_points[i].x, h_points[i].y, h_points[i].z);

		auto pointType = h_points[i].pointType;
		if (Point::PN == pointType || Point::PNC == pointType)
		{
			plys[tag].AddNormal(h_points[i].nx, h_points[i].ny, h_points[i].nz);
		}
		if (Point::PC == pointType || Point::PNC == pointType)
		{
			plys[tag].AddColor(h_points[i].r, h_points[i].g, h_points[i].b);
		}
	}

	for (auto& kvp : plys)
	{
		if (0 < kvp.second.GetPoints().size())
		{
			stringstream ss;
			if (0 != tagNameMapping.count(kvp.first))
			{
				ss << "C:\\Resources\\Debug\\Serialized\\Debugging_" << h_flushCount << "_" << kvp.first << "_" << tagNameMapping[kvp.first] << ".ply";
			}
			else
			{
				ss << "C:\\Resources\\Debug\\Serialized\\Debugging_" << h_flushCount << "_" << kvp.first << ".ply";
			}
			kvp.second.Serialize(ss.str());
		}
	}

	delete h_points;

	h_numberOfPoints = 0;
	cudaMemcpyToSymbol((const void*)numberOfPoints, &h_numberOfPoints, sizeof(unsigned int));

	h_flushCount++;
	cudaMemcpyToSymbol((const void*)flushCount, &h_flushCount, sizeof(unsigned int));


	//{
	//	stringstream ss;
	//	ss << "C:\\Resources\\Debug\\Serialized\\transform_" << h_flushCount << ".bin";
	//	FILE* fs;
	//	fopen_s(&fs, ss.str().c_str(), "wb");
	//	fwrite(transform.data(), sizeof(float) * 16, 1, fs);
	//	fclose(fs);
	//}

	//alog("[%4d] Debugging::Flush() End\n", h_frameNumber);

	{
		int enabled = 0;
		cudaMemcpyFromSymbol(&enabled, (const void*)captureEnabled, sizeof(int));

		if (1 == enabled)
		{
			enabled = 0;
			cudaMemcpyToSymbol((const void*)captureEnabled, &enabled, sizeof(int));

			//alog("[%4d] set captureEnabled to 0\n", h_frameNumber);

			captureOneFrame = false;
		}
	}
}

void Debugging::StartRecording()
{
	recording = true;
}

void Debugging::EndRecording()
{
	recording = false;
}

void Debugging::ToggleRecording()
{
	recording = !recording;
}

__device__ unsigned int lastFrameNumber = 0;

__device__ void Debugging::AddPointP(int tag, const float3& point)
{
	unsigned int oldFrameNumber = atomicCAS(&lastFrameNumber, lastFrameNumber, frameNumber);

	bool printLog = oldFrameNumber != frameNumber;
	lastFrameNumber = frameNumber;

	//if (printLog) alog("[%4d] Debugging::AddPointP() Begin\n", frameNumber);

	if (captureEnabled != 1) return;

	int index = atomicAdd(&numberOfPoints, 1);
	if (index < POINT_COUNT) {
		points[index].tag = tag;
		points[index].pointType = Point::P;
		points[index].x = point.x;
		points[index].y = point.y;
		points[index].z = point.z;
	}

	//if (printLog) alog("[%4d] Debugging::AddPointP() end\n", frameNumber);
}

__device__ void Debugging::AddPointPN(int tag, const float3& point, const float3& normal)
{
	if (captureEnabled != 1) return;

	int index = atomicAdd(&numberOfPoints, 1);
	if (index < POINT_COUNT) {
		points[index].tag = tag;
		points[index].pointType = Point::PN;
		points[index].x = point.x;
		points[index].y = point.y;
		points[index].z = point.z;
		points[index].nx = normal.x;
		points[index].ny = normal.y;
		points[index].nz = normal.z;
	}
}

__device__ void Debugging::AddPointPC(int tag, const float3& point, const float3& color)
{
	if (captureEnabled != 1) return;

	int index = atomicAdd(&numberOfPoints, 1);
	if (index < POINT_COUNT) {
		points[index].tag = tag;
		points[index].pointType = Point::PN;
		points[index].x = point.x;
		points[index].y = point.y;
		points[index].z = point.z;
		points[index].r = color.x;
		points[index].g = color.y;
		points[index].b = color.z;
	}
}

__device__ void Debugging::AddPointPNC(int tag, const float3& point, const float3& normal, const float3& color)
{
	if (captureEnabled != 1) return;

	int index = atomicAdd(&numberOfPoints, 1);
	if (index < POINT_COUNT) {
		points[index].tag = tag;
		points[index].pointType = Point::PNC;
		points[index].x = point.x;
		points[index].y = point.y;
		points[index].z = point.z;
		points[index].nx = normal.x;
		points[index].ny = normal.y;
		points[index].nz = normal.z;
		points[index].r = color.x;
		points[index].g = color.y;
		points[index].b = color.z;
	}
}

void Debugging::SerializePoints(
	float3* inputPoints,
	float3* inputNormals,
	uint3* inputColors,
	size_t numberOfInputPoints,
	const string& tagName)
{
	float3* h_inputPoints = new float3[numberOfInputPoints];
	float3* h_inputNormals = new float3[numberOfInputPoints];
	uint3* h_inputColors = new uint3[numberOfInputPoints];

	cudaMemcpy(h_inputPoints, inputPoints, sizeof(float3) * numberOfInputPoints, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_inputNormals, inputNormals, sizeof(float3) * numberOfInputPoints, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_inputColors, inputColors, sizeof(uint3) * numberOfInputPoints, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	unsigned int h_frameCount = 0;
	cudaMemcpyFromSymbol(&h_frameCount, (const void*)frameCount, sizeof(unsigned int));

	PLYFormat ply;

	for (size_t i = 0; i < numberOfInputPoints; i++)
	{
		auto& p = h_inputPoints[i];
		auto& n = h_inputNormals[i];
		auto& c = h_inputColors[i];

		if (FLT_MAX == p.x || FLT_MAX == p.y || FLT_MAX == p.z) continue;

		ply.AddPoint(p.x, p.y, p.z);
		ply.AddNormal(n.x, n.y, n.z);
		ply.AddColor(c.x / 255.f, c.y / 255.f, c.z / 255.f);
	}

	std::stringstream ss;
	ss << "C:\\Resources\\Debug\\Serialized\\Serialized_" << h_frameCount << "_pointCloud_" << tagName << ".ply";
	ply.Serialize(ss.str());
}

void Debugging::SetTagName(int tag, const string& name)
{
	tagNameMapping[tag] = name;
}
#pragma endregion
*/