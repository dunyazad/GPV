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

			PLYFormat ply;
			ply.Deserialize("C:\\Resources\\Debug\\Field.ply");

			map<pair<unsigned int, unsigned int>, int> edges;

			for (size_t i = 0; i < ply.GetTriangleIndices().size() / 3; i++)
			{
				auto i0 = ply.GetTriangleIndices()[i * 3];
				auto i1 = ply.GetTriangleIndices()[i * 3 + 1];
				auto i2 = ply.GetTriangleIndices()[i * 3 + 2];

				auto v0x = ply.GetPoints()[i0 * 3];
				auto v0y = ply.GetPoints()[i0 * 3 + 1];
				auto v0z = ply.GetPoints()[i0 * 3 + 2];

				auto v1x = ply.GetPoints()[i1 * 3];
				auto v1y = ply.GetPoints()[i1 * 3 + 1];
				auto v1z = ply.GetPoints()[i1 * 3 + 2];

				auto v2x = ply.GetPoints()[i2 * 3];
				auto v2y = ply.GetPoints()[i2 * 3 + 1];
				auto v2z = ply.GetPoints()[i2 * 3 + 2];

				VD::AddTriangle("mesh", { v0x, v0y, v0z }, { v1x, v1y, v1z }, { v2x, v2y, v2z }, Color4::White);

				edges[i0 < i1 ? make_pair(i0, i1) : make_pair(i1, i0)]++;
				edges[i1 < i2 ? make_pair(i1, i2) : make_pair(i2, i1)]++;
				edges[i2 < i0 ? make_pair(i2, i0) : make_pair(i0, i2)]++;
			}

			VD::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
			VD::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
			VD::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);
			t = Time::End(t, "Visualize");

			int to = 0;
			int tc = 0;
			int minp = INT_MAX;
			int maxp = -INT_MAX;
			for (auto& kvp : edges)
			{
				auto v = kvp.second;
				if (minp > v) minp = v;
				if (maxp < v) maxp = v;

				if (v == 1) to++;
				if (v == 2) tc++;
			}

			printf("minp : %d, maxp: %d, tc : %d, tc : %d\n", minp, maxp, to, tc);
		}
	}
}
