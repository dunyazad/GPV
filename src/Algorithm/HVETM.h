#pragma once

#include "MiniMath.h"
using namespace MiniMath;

#include <map>
#include <set>
#include <stack>
#include <tuple>
#include <chrono>
using namespace std;

//class HNode;

namespace HVETM
{
	using namespace std;
	struct Vertex;
	struct Edge;
	struct Triangle;
	class Mesh;

	chrono::steady_clock::time_point Now();
	string Miliseconds(const chrono::steady_clock::time_point beginTime, const char* tag = nullptr);
#define TS(name) auto time_##name = HVETM::Now();
#define TE(name) std::cout << HVETM::Miliseconds(time_##name, #name) << std::endl;


	struct Vertex
	{
		int id = -1;
		V3 p = V3{ 0.0f, 0.0f, 0.0f };
		V3 n = V3{ 0.0f, 0.0f, 0.0f };
		set<Edge*> edges;
		set<Triangle*> triangles;
		V3 diffuse = V3{ 0.7f, 0.7f, 0.7f };

		int tempFlag = 0;
	};

	struct Edge
	{
		int id = -1;
		Vertex* v0 = nullptr;
		Vertex* v1 = nullptr;
		set<Triangle*> triangles;
		float length = 0.0f;
	};

	struct Triangle
	{
		int id = -1;
		Vertex* v0 = nullptr;
		Vertex* v1 = nullptr;
		Vertex* v2 = nullptr;
		V3 centroid = V3{ 0.0f, 0.0f, 0.0f };
		V3 normal = V3{ 0.0f, 0.0f, 0.0f };
	};

	class HKDTreeNode
	{
	public:
		HKDTreeNode(Vertex* vertex) : v(vertex) {}

		Vertex* GetVertex() const { return v; }

	private:
		Vertex* v;
		HKDTreeNode* left = nullptr;
		HKDTreeNode* right = nullptr;

	public:
		friend class HKDTree;
	};

	class HKDTree
	{
	public:
		HKDTree() {}

		void Clear();

		void Insert(Vertex* vertex);

		Vertex* FindNearestNeighbor(const V3& query) const;
		HKDTreeNode* FindNearestNeighborNode(const V3& query) const;

		vector<Vertex*> RangeSearch(const V3& query, float squaredRadius) const;

		inline bool IsEmpty() const { return nullptr == root; }

	private:
		HKDTreeNode* root = nullptr;
		mutable HKDTreeNode* nearestNeighborNode = nullptr;
		mutable Vertex* nearestNeighbor = nullptr;
		mutable float nearestNeighborDistance = FLT_MAX;

		void ClearRecursive(HKDTreeNode* node);
		HKDTreeNode* InsertRecursive(HKDTreeNode* node, Vertex* vertex, int depth);
		void FindNearestNeighborRecursive(HKDTreeNode* node, const V3& query, int depth) const;
		void RangeSearchRecursive(HKDTreeNode* node, const V3& query, float squaredRadius, std::vector<Vertex*>& result, int depth) const;
	};

	void TestHKDTree();

	template<typename V, typename T>
	class RegularGridCell : public AABB {
	public:
		enum class CellFlag {
			None,
			PartiallySelected = 128,
			Extruded = 512,
			FloodFillVisited = 1024,
		};

		bool selected = false;

		CellFlag tempFlag = { CellFlag::None };


		RegularGridCell(const V3& minPoint = { FLT_MAX, FLT_MAX, FLT_MAX },
			const V3& maxPoint = { -FLT_MAX, -FLT_MAX, -FLT_MAX })
			: AABB({ minPoint, maxPoint }), index({ IntInfinity, IntInfinity, IntInfinity }) {}

		RegularGridCell(tuple<size_t, size_t, size_t> index,
			const V3& minPoint = { FLT_MAX, FLT_MAX, FLT_MAX },
			const V3& maxPoint = { -FLT_MAX, -FLT_MAX, -FLT_MAX })
			: AABB(minPoint, maxPoint), index(index) {}

		inline set<V*>& GetVertices() { return vertices; }
		inline set<T*>& GetTriangles() { return triangles; }

		bool operator==(const RegularGridCell& other) const
		{
			return get<0>(index) == get<0>(other.index) &&
				get<1>(index) == get<1>(other.index) &&
				get<2>(index) == get<2>(other.index);
		}

		bool operator!=(const RegularGridCell& other) const
		{
			return get<0>(index) != get<0>(other.index) ||
				get<1>(index) != get<1>(other.index) ||
				get<2>(index) != get<2>(other.index);
		}

	protected:
		tuple<size_t, size_t, size_t> index;

		set<V*> vertices;
		set<T*> triangles;
	};

	class RegularGrid : public AABB {
	public:
		typedef RegularGridCell<Vertex, Triangle> CellType;

		RegularGrid();
		RegularGrid(const Mesh* mesh, float cellSize);
		~RegularGrid();

		inline float GetCellSize() const { return cellSize; }
		inline size_t GetCellCountX() const { return cellCountX; }
		inline size_t GetCellCountY() const { return cellCountY; }
		inline size_t GetCellCountZ() const { return cellCountZ; }

		inline tuple<size_t, size_t, size_t> GetIndex(const V3& position) const
		{






			auto x = (size_t)floorf((position.x - this->GetMinPoint().x) / cellSize);
			auto y = (size_t)floorf((position.y - this->GetMinPoint().y) / cellSize);
			auto z = (size_t)floorf((position.z - this->GetMinPoint().z) / cellSize);
			if ((x < 0 || x > cellCountX) || (y < 0 || y > cellCountY) || (z < 0 || z > cellCountZ))
			{
				//cout << "WTF" << endl;
			}
			return make_tuple(x, y, z);
		}

		inline CellType* GetCell(const tuple<size_t, size_t, size_t>& index) const
		{
			auto x = get<0>(index);
			auto y = get<1>(index);
			auto z = get<2>(index);

			return GetCell(x, y, z);
		}

		inline CellType* GetCell(size_t x, size_t y, size_t z) const
		{
			if ((0 <= x && x < cellCountX) &&
				(0 <= y && y < cellCountY) &&
				(0 <= z && z < cellCountZ))
			{
				return cells[z][y][x];
			}
			else
			{
				return nullptr;
			}
		}
		/*
		inline CellType* GetCell(size_t x, size_t y, size_t z) 
		{
			return const_cast<RegularGrid*>(this)->GetCell(x, y, z);
		}*/

		void Build();

		tuple<size_t, size_t, size_t> InsertVertex(Vertex* vertex);
		void InsertTriangle(Triangle* t);

		inline const vector<Vertex*>& GetVertices() const { return vertices; }

		/*
			void RemoveTriangle(Triangle* t);

			set<CellType*> GetCellsWithRay(const Ray& ray);
		*/

		inline const vector<vector<vector<CellType*>>>& GetCells() const { return cells; }

		vector<vector<V3>> ExtractSurface(float isolevel) const;

		//void ForEachCell(function<void(CellType*, size_t, size_t, size_t)> callback);

		void SelectOutsideCells();
		void InvertSelectedCells();
		void ShrinkSelectedCells(int iteration);
		void ExtrudeSelectedCells(const V3& direction, int iteration);

	private:
		const Mesh* mesh = nullptr;
		float cellSize = 0.5;
		float cellHalfSize = 0.25;
		size_t cellCountX = 0;
		size_t cellCountY = 0;
		size_t cellCountZ = 0;

		vector<vector<vector<CellType*>>> cells;

		//map<tuple<int, int, int>, CellType*> cells;

		vector<Vertex*> vertices;
		vector<Triangle*> triangles;
	};

	class Mesh
	{
	public:
		static vector<uint32_t> Triangulate(const vector<vector<V2>>& pointsList);

		// construction
		Mesh(float vertexEpsilon = 0.00001f);
		virtual ~Mesh();
		void Clear();
		void Clone(Mesh& clone);
		inline void Swap(const Mesh& other) { kdtree = other.kdtree; }

		inline HKDTree& GetKDTree() { return kdtree; }
		inline const HKDTree& GetKDTree() const { return kdtree; }

		// vertex primitive access
		inline float GetTotalArea() const { return totalArea; }
		inline const vector<Vertex*>& GetVertices() const { return vertices; }
		inline const set<Edge*>& GetEdges() const { return edges; }
		inline const set<Triangle*>& GetTriangles() const { return triangles; }
		inline const AABB& GetAABB() const { return aabb; }
		Vertex* GetVertex(const V3& position) const;
		Edge*	GetEdge(const Vertex* v0, const Vertex* v1) const;
		Vertex* GetCommonVertex(const Edge* e0, const Edge* e1) const;
		set<Vertex*> GetAdjacentVertices(Vertex* vertex) const;
		Triangle* GetTriangle(Vertex* v0, Vertex* v1, Vertex* v2) const;
		set<Vertex*> GetVerticesInRadius(const V3& position, float radius) const;
		//float GetDistanceFromEdge(Edge* edge, const V3& position);
		tuple<V3, V3, V3> GetTrianglePoints(const Triangle* triangle) const;
		V3 GetTriangleCentroid(const Triangle* triangle) const;
		float GetTriangleArea(const Triangle* triangle) const;
		//V3 GetNearestPointOnEdge(Edge* edge, const V3& position);
		Vertex* GetNearestVertex(const V3& position) const;
		Vertex* GetNearestVertexOnTriangle(const Triangle* triangle, const V3& position) const;
		//Edge* GetNearestEdgeOnTriangle(Triangle* triangle, const V3& position);
		set<Triangle*> GetAdjacentTrianglesByEdge(const Triangle* triangle) const;
		set<Triangle*> GetAdjacentTrianglesByVertex(const Triangle* triangle) const;
		set<Triangle*> GetConnectedTriangles(Triangle* triangle) const;
		vector<vector<Edge*>> GetBorderEdges() const;
		vector<Vertex*> GetBorderVerticesFromBorderEdges(const vector<Edge*>& borderEdges) const;

		// mesh editing
		Vertex* AddVertex(const V3& position, const V3& normal);
		Edge* AddEdge(Vertex* v0, Vertex* v1);
		Triangle* AddTriangle(Vertex* v0, Vertex* v1, Vertex* v2);
		void RemoveTriangle(Triangle* triangle);
		void AddTriangles(const set<Triangle*>& triangles);
		void AddTriangles(const vector<Vertex*>& vertices, const vector<uint32_t>& indices, bool clockwise);
		void AddInnerTriangles(vector<vector<V3>>& triangles);
		void FlipTriangle(Triangle* triangle);

		vector<Mesh*> SeparateConnectedGroup();

		// mesh editing
		void FillTrianglesToMakeBorderSmooth(float maxAngle);
		void ExtrudeBorder(const V3& direction, int segments);
		void MakeSmooth(size_t iteration, const V3& direction);
		void DeleteSelfintersectingTriangles();
		void SliceAndRemove(const V3& planePosition, const V3& planeNormal);

		// base mesh generation
		float GetMinTotalHeight(bool isMaxillar=false) const;

		//// conversion
		//bool CreateFromHNode(const HNode& node);
		//HNode* ToHNode() const;

		V3 DetermineBorderExtrudeDirection(const vector<Edge*>& borderEdges) const;
		void BorderVertexSmoothing(const vector<Vertex*>& borderVertices, int iteration);
		void SimpleFillHole(bool leaveLargeHole = true);

#ifdef _DEBUG
		void DumpBorderEdges(const std::string& message) const;
#endif

	protected:
		float vertexEpsilon = 0.00001f;

		HKDTree kdtree;
		AABB aabb;

		int vid = 0;
		int eid = 0;
		int tid = 0;

		vector<Vertex*> vertices;
		vector<Vertex*> ProjectedBottomVertices;
		set<Edge*> edges;
		map<tuple<const Vertex*, const Vertex*>, Edge*> edgeMapping;
		set<Triangle*> triangles;
		map<tuple<Edge*, Edge*, Edge*>, Triangle*> triangleMapping;
		float totalArea = 0.0;
	};
}
