#include "HalfEdge.cuh"

#include <App/Serialization.hpp>
#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
	namespace HalfEdge
	{
		struct Vertex;
		struct HalfEdge;
		struct Face;

		struct Vertex
		{
			int state;
			size_t pointIndex;
			size_t halfEdgeIndex;
		};

		struct HalfEdge
		{
			int state;
			size_t vertexIndex;
			size_t faceIndex;
			size_t next;
			size_t pair;
		};

		struct Face
		{
			int state;
			size_t halfEdgeIndex;
		};

		struct HalfEdgeMesh
		{
			Eigen::Vector3f* points;
			Vertex* vertices;
			HalfEdge* halfEdges;
			Face* faces;

			size_t numberOfPoints;
			size_t numberOfVertices;
			size_t numberOfHalfEdges;
			size_t numberOfFaces;

			size_t numberOfValidPoints;
			size_t numberOfValidVertices;
			size_t numberOfValidHalfEdges;
			size_t numberOfValidFaces;
		};

		void InitializeHalfEdgeMesh(HalfEdgeMesh* d_mesh, size_t numberOfPoints, size_t numberOfFaces)
		{
			HalfEdgeMesh mesh;

			cudaMallocManaged(&mesh.points, sizeof(Eigen::Vector3f) * numberOfPoints);
			cudaMallocManaged(&mesh.vertices, sizeof(Vertex) * numberOfPoints);
			cudaMallocManaged(&mesh.halfEdges, sizeof(HalfEdge) * numberOfFaces * 3);
			cudaMallocManaged(&mesh.faces, sizeof(Face) * numberOfFaces);

			mesh.numberOfPoints = numberOfPoints;
			mesh.numberOfVertices = numberOfPoints;
			mesh.numberOfHalfEdges = numberOfFaces * 3;
			mesh.numberOfFaces = numberOfFaces;

			mesh.numberOfValidPoints = 0;
			mesh.numberOfValidVertices = 0;
			mesh.numberOfValidHalfEdges = 0;
			mesh.numberOfValidFaces = 0;

			cudaMemcpy(d_mesh, &mesh, sizeof(HalfEdgeMesh), cudaMemcpyHostToDevice);
		}

		void TerminateHalfEdgeMesh(HalfEdgeMesh* d_mesh)
		{
			HalfEdgeMesh mesh;
			cudaMemcpy(&mesh, d_mesh, sizeof(HalfEdgeMesh), cudaMemcpyDeviceToHost);

			cudaFree(mesh.points);
			cudaFree(mesh.vertices);
			cudaFree(mesh.halfEdges);
			cudaFree(mesh.faces);
		}

		__global__ void Kernel_FromMesh(
			HalfEdgeMesh* d_mesh,
			Eigen::Vector3f* points,
			size_t numberOfPoints,
			Eigen::Vector3i* triangles,
			size_t numberOfTriangles)
		{

		}

		void FromMesh(
			HalfEdgeMesh* h_mesh,
			HalfEdgeMesh* d_mesh,
			Eigen::Vector3f* points,
			size_t numberOfPoints,
			Eigen::Vector3i* triangles,
			size_t numberOfTriangles)
		{


			//cudaMallocManaged(&d_mesh->points, sizeof(Eigen::Vector3f) * numberOfPoints);
			//cudaMallocManaged(&d_mesh->vertices, sizeof(Vertex) * numberOfPoints);
			//cudaMallocManaged(&d_mesh->halfEdges, sizeof(HalfEdge) * numberOfTriangles * 3);
			//cudaMallocManaged(&d_mesh->faces, sizeof(Face) * numberOfTriangles);

			//mesh->numberOfPoints = numberOfPoints;
			//mesh->numberOfVertices = numberOfPoints;
			//mesh->numberOfHalfEdges = numberOfTriangles * 3;
			//mesh->numberOfFaces = numberOfTriangles;
		}
		
		void TestHalfEdge()
		{
			auto t = Time::Now();

			size_t hCount = 10;
			size_t vCount = 10;
			float interval = 0.1f;

			size_t numberOfVertices = hCount * vCount;
			size_t numberOfTriangles = (hCount - 1) * (vCount - 1) * 2;

			HalfEdgeMesh* d_mesh;
			cudaMallocManaged(&d_mesh, sizeof(HalfEdgeMesh));
			InitializeHalfEdgeMesh(d_mesh, numberOfVertices, numberOfTriangles);

			Eigen::Vector3f* points = new Eigen::Vector3f[hCount * vCount];
			for (size_t y = 0; y < vCount; y++)
			{
				for (size_t x = 0; x < hCount; x++)
				{
					points[y * hCount + x] = Eigen::Vector3f(
						(float)x * interval - 5.0f,
						(float)y * interval - 5.0f,
						0.0f);

					VD::AddSphere("points", points[y * hCount + x], 0.05f, Color4::White);
				}
			}

			Eigen::Vector3i* triangles = new Eigen::Vector3i[numberOfTriangles];
			for (size_t y = 0; y < vCount - 1; y++)
			{
				for (size_t x = 0; x < hCount - 1; x++)
				{
					triangles[y * (hCount - 1) * 2 + x * 2] = Eigen::Vector3i(
						y * hCount + x,
						y * hCount + x + 1,
						(y + 1) * hCount + x + 1);
					triangles[y * (hCount - 1) * 2 + x * 2 + 1] = Eigen::Vector3i(
						y * hCount + x,
						(y + 1) * hCount + x + 1,
						(y + 1) * hCount + x);
					
					Eigen::Vector3i& t0 = triangles[y * (hCount - 1) * 2 + x * 2];
					Eigen::Vector3i& t1 = triangles[y * (hCount - 1) * 2 + x * 2 + 1];
					VD::AddTriangle("triangles", points[t0.x()], points[t0.y()], points[t0.z()], Color4::White);
					VD::AddTriangle("triangles", points[t1.x()], points[t1.y()], points[t1.z()], Color4::White);
				}
			}

			//FromMesh(d_mesh, points, numberOfVertices, triangles, numberOfTriangles);
			
			


			delete[] points;
			delete[] triangles;

			TerminateHalfEdgeMesh(d_mesh);
			cudaFree(d_mesh);
			
			
			
			//LoadPatch(0, renderer);
			//t = Time::End(t, "Loading Patch");

			//for (size_t i = 0; i < patchPoints_0.size(); i++)
			//{
			//	auto& p = patchPoints_0[i];
			//	if (FLT_MAX != p.x() && FLT_MAX != p.y() && FLT_MAX != p.z())
			//	{
			//		VD::AddSphere("points_0", p, 0.05f, Color4::Red);
			//	}
			//}

			////VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, 40.0f, 48.0f, 0.1f, Color4::Red);
			////VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 0.0f }, 40.0f, 48.0f, 0.1f, Color4::Yellow);
			////VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, 40.0f, 48.0f, 0.1f, Color4::Green);
			////VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 1.0f }, 40.0f, 48.0f, 0.1f, Color4::Cyan);
			//VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, 40.0f, 48.0f, 0.1f, Color4::Blue);
			////VD::AddGrid("grid", { 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 1.0f }, 40.0f, 48.0f, 0.1f, Color4::Magenta);

			VD::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
			VD::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
			VD::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
			t = Time::End(t, "Visualize");
		}
	}
}
