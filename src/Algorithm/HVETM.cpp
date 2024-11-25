#include "HVETM.h"
#include <array>
#include <sstream>
#include <chrono>
#include <vtkSTLReader.h>
#include <vtkPLYReader.h>
#include "earcut.hpp"
//#include "HCommonTopology.h"

namespace HVETM
{
	chrono::steady_clock::time_point Now()
	{
		return chrono::high_resolution_clock::now();
	}

	string Miliseconds(const chrono::steady_clock::time_point beginTime, const char* tag)
	{
		auto now = Now();
		auto timeSpan = chrono::duration_cast<chrono::nanoseconds>(now - beginTime).count();
		stringstream ss;
		ss << "[[[ ";
		if (nullptr != tag)
		{
			ss << tag << " - ";
		}
		ss << (float)timeSpan / 1000000.0 << " ms ]]]";
		return ss.str();
	}

	void HKDTree::Clear()
	{
		if (nullptr != root)
		{
			ClearRecursive(root);
			root = nullptr;
		}

		nearestNeighborNode = nullptr;
		nearestNeighbor = nullptr;
		nearestNeighborDistance = FLT_MAX;
	}

	void HKDTree::Insert(Vertex* vertex)
	{
		root = InsertRecursive(root, vertex, 0);
	}

	Vertex* HKDTree::FindNearestNeighbor(const V3& query) const
	{
		nearestNeighbor = root->v;
		nearestNeighborDistance = magnitude(query - root->v->p);
		nearestNeighborNode = nullptr;
		FindNearestNeighborRecursive(root, query, 0);
		return nearestNeighbor;
	}

	HKDTreeNode* HKDTree::FindNearestNeighborNode(const V3& query) const
	{
		if (nullptr == root)
			return nullptr;

		nearestNeighbor = nullptr;
		nearestNeighborDistance = FLT_MAX;
		nearestNeighborNode = nullptr;
		FindNearestNeighborRecursive(root, query, 0);
		return nearestNeighborNode;
	}

	vector<Vertex*> HKDTree::RangeSearch(const V3& query, float squaredRadius) const
	{
		vector<Vertex*> result;
		RangeSearchRecursive(root, query, squaredRadius, result, 0);
		return result;
	}

	void HKDTree::ClearRecursive(HKDTreeNode* node)
	{
		if (nullptr != node->left)
		{
			ClearRecursive(node->left);
		}

		if (nullptr != node->right)
		{
			ClearRecursive(node->right);
		}

		delete node;
	}

	HKDTreeNode* HKDTree::InsertRecursive(HKDTreeNode* node, Vertex* vertex, int depth) {
		if (node == nullptr) {
			auto newNode = new HKDTreeNode(vertex);
			return newNode;
		}

		int currentDimension = depth % 3;
		if (((float*)&vertex->p)[currentDimension] < ((float*)&node->GetVertex()->p)[currentDimension])
		{
			node->left = InsertRecursive(node->left, vertex, depth + 1);
		}
		else {
			node->right = InsertRecursive(node->right, vertex, depth + 1);
		}

		return node;
	}

	void HKDTree::FindNearestNeighborRecursive(HKDTreeNode* node, const V3& query, int depth) const{
		if (node == nullptr) {
			return;
		}

		int currentDimension = depth % 3;

		float nodeDistance = magnitude(query - node->GetVertex()->p);
		if (nodeDistance < nearestNeighborDistance) {
			nearestNeighborNode = node;
			nearestNeighbor = node->GetVertex();
			nearestNeighborDistance = nodeDistance;
		}

		auto queryValue = ((float*)&query)[currentDimension];
		auto nodeValue = ((float*)&node->GetVertex()->p)[currentDimension];

		HKDTreeNode* closerNode = (queryValue < nodeValue) ? node->left : node->right;
		HKDTreeNode* otherNode = (queryValue < nodeValue) ? node->right : node->left;

		FindNearestNeighborRecursive(closerNode, query, depth + 1);

		// Check if the other subtree could have a closer point
		float planeDistance = queryValue - nodeValue;
		if (planeDistance * planeDistance < nearestNeighborDistance) {
			FindNearestNeighborRecursive(otherNode, query, depth + 1);
		}
	}

	void HKDTree::RangeSearchRecursive(HKDTreeNode* node, const V3& query, float squaredRadius, std::vector<Vertex*>& result, int depth) const {
		if (node == nullptr) {
			return;
		}

		float nodeDistance = magnitude(query - node->GetVertex()->p);
		if (nodeDistance <= squaredRadius) {
			result.push_back(node->GetVertex());
		}

		int currentDimension = depth % 3;
		auto queryValue = ((float*)&query)[currentDimension];
		auto nodeValue = ((float*)&node->GetVertex()->p)[currentDimension];

		HKDTreeNode* closerNode = (queryValue < nodeValue) ? node->left : node->right;
		HKDTreeNode* otherNode = (queryValue < nodeValue) ? node->right : node->left;

		RangeSearchRecursive(closerNode, query, squaredRadius, result, depth + 1);

		// Check if the other subtree could have points within the range
		if (std::abs(queryValue - nodeValue) * std::abs(queryValue - nodeValue) <= squaredRadius) {
			RangeSearchRecursive(otherNode, query, squaredRadius, result, depth + 1);
		}
	}

	typedef struct {
		V3 p[3];
	} TRIANGLE;

	typedef struct {
		V3 p[8];
		float val[8];

	} GRIDCELL;


	V3 VertexInterp(float isolevel, const V3& p1, const V3& p2, float valp1, float valp2);

	/*
	   Given a grid cell and an isolevel, calculate the triangular
	   facets required to represent the isosurface through the cell.
	   Return the number of triangular facets, the array "triangles"
	   will be loaded up with the vertices at most 5 triangular facets.
		0 will be returned if the grid cell is either totally above
	   of totally below the isolevel.
	*/
	int Polygonise(GRIDCELL grid, float isolevel, TRIANGLE* triangles)
	{
		int i, ntriang;
		int cubeindex;
		V3 vertlist[12];

		const int edgeTable[256] = {
		0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
		0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
		0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
		0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
		0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
		0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
		0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
		0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
		0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
		0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
		0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
		0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
		0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
		0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
		0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
		0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
		0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };
		const int triTable[256][16] =
		{ {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
		{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
		{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
		{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
		{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
		{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
		{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
		{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
		{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
		{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
		{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
		{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
		{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
		{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
		{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
		{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
		{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
		{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
		{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
		{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
		{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
		{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
		{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
		{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
		{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
		{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
		{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
		{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
		{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
		{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
		{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
		{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
		{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
		{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
		{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
		{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
		{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
		{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
		{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
		{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
		{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
		{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
		{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
		{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
		{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
		{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
		{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
		{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
		{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
		{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
		{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
		{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
		{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
		{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
		{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
		{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
		{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
		{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
		{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
		{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
		{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
		{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
		{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
		{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
		{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
		{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
		{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
		{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
		{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
		{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
		{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
		{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
		{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
		{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
		{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
		{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
		{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
		{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
		{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
		{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
		{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
		{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
		{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
		{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
		{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
		{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
		{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
		{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
		{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
		{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
		{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
		{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
		{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
		{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
		{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
		{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
		{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
		{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
		{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
		{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
		{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
		{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
		{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
		{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
		{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
		{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
		{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
		{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
		{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
		{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
		{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
		{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
		{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
		{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
		{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
		{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
		{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
		{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
		{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
		{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
		{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
		{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
		{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
		{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
		{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
		{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
		{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
		{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };

		/*
		   Determine the index into the edge table which
		   tells us which vertices are inside of the surface
		*/
		cubeindex = 0;
		if (grid.val[0] < isolevel) cubeindex |= 1;
		if (grid.val[1] < isolevel) cubeindex |= 2;
		if (grid.val[2] < isolevel) cubeindex |= 4;
		if (grid.val[3] < isolevel) cubeindex |= 8;
		if (grid.val[4] < isolevel) cubeindex |= 16;
		if (grid.val[5] < isolevel) cubeindex |= 32;
		if (grid.val[6] < isolevel) cubeindex |= 64;
		if (grid.val[7] < isolevel) cubeindex |= 128;

		/* Cube is entirely in/out of the surface */
		if (edgeTable[cubeindex] == 0)
			return(0);

		/* Find the vertices where the surface intersects the cube */
		if (edgeTable[cubeindex] & 1)
			vertlist[0] =
			VertexInterp(isolevel, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
		if (edgeTable[cubeindex] & 2)
			vertlist[1] =
			VertexInterp(isolevel, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
		if (edgeTable[cubeindex] & 4)
			vertlist[2] =
			VertexInterp(isolevel, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
		if (edgeTable[cubeindex] & 8)
			vertlist[3] =
			VertexInterp(isolevel, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
		if (edgeTable[cubeindex] & 16)
			vertlist[4] =
			VertexInterp(isolevel, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
		if (edgeTable[cubeindex] & 32)
			vertlist[5] =
			VertexInterp(isolevel, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
		if (edgeTable[cubeindex] & 64)
			vertlist[6] =
			VertexInterp(isolevel, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
		if (edgeTable[cubeindex] & 128)
			vertlist[7] =
			VertexInterp(isolevel, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
		if (edgeTable[cubeindex] & 256)
			vertlist[8] =
			VertexInterp(isolevel, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
		if (edgeTable[cubeindex] & 512)
			vertlist[9] =
			VertexInterp(isolevel, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
		if (edgeTable[cubeindex] & 1024)
			vertlist[10] =
			VertexInterp(isolevel, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
		if (edgeTable[cubeindex] & 2048)
			vertlist[11] =
			VertexInterp(isolevel, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

		/* Create the triangle */
		ntriang = 0;
		for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
			triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i]];
			triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i + 1]];
			triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i + 2]];
			ntriang++;
		}

		return(ntriang);
	}

	/*
	   Linearly interpolate the position where an isosurface cuts
	   an edge between two vertices, each with their own scalar value
	*/

	inline V3 VertexInterp(float isolevel, const V3& p1, const V3& p2, float valp1, float valp2)
	{
		float mu;
		V3 p;

		if (fabsf(isolevel - valp1) < 0.00001)
			return(p1);
		if (fabsf(isolevel - valp2) < 0.00001)
			return(p2);
		if (fabsf(valp1 - valp2) < 0.00001)
			return(p1);
		mu = (isolevel - valp1) / (valp2 - valp1);
		p.x = p1.x + mu * (p2.x - p1.x);
		p.y = p1.y + mu * (p2.y - p1.y);
		p.z = p1.z + mu * (p2.z - p1.z);

		return(p);
	}

	RegularGrid::RegularGrid()
		: mesh(nullptr), cellSize(0.1f), cellHalfSize(cellSize * 0.5f)
	{
	}

	RegularGrid::RegularGrid(const Mesh* mesh, float cellSize)
		: mesh(mesh), cellSize(cellSize), cellHalfSize(cellSize * 0.5f)
	{
	}

	RegularGrid::~RegularGrid()
	{
		for (size_t z = 0; z < cellCountZ; z++)
		{
			for (size_t y = 0; y < cellCountY; y++)
			{
				for (size_t x = 0; x < cellCountX; x++)
				{
					if (nullptr != cells[z][y][x])
					{
						delete cells[z][y][x];
						cells[z][y][x] = nullptr;
					}
				}
			}
		}
		cells.clear();

		for (auto& v : vertices)
		{
			if (nullptr != v)
			{
				delete v;
				v = nullptr;
			}
		}
		vertices.clear();

		for (auto& t : triangles)
		{
			if (nullptr != t)
			{
				delete t;
				t = nullptr;
			}
		}
		triangles.clear();
	}

	void RegularGrid::Build()
	{
		auto meshAABB = mesh->GetAABB();
		cellCountX = (size_t)ceilf(meshAABB.GetXLength() / cellSize) + 1;
		cellCountY = (size_t)ceilf(meshAABB.GetYLength() / cellSize) + 1;
		cellCountZ = (size_t)ceilf(meshAABB.GetZLength() / cellSize) + 1;

		float nx = -((float)cellCountX * cellSize) * 0.5f;
		float px = ((float)cellCountX * cellSize) * 0.5f;
		float ny = -((float)cellCountY * cellSize) * 0.5f;
		float py = ((float)cellCountY * cellSize) * 0.5f;
		float nz = -((float)cellCountZ * cellSize) * 0.5f;
		float pz = ((float)cellCountZ * cellSize) * 0.5f;
		float cx = (px + nx) * 0.5f;
		float cy = (py + ny) * 0.5f;
		float cz = (pz + nz) * 0.5f;
		this->Expand(V3(nx, ny, nz) + meshAABB.GetCenter());
		this->Expand(V3(px, py, pz) + meshAABB.GetCenter());

		// allocate cells
		cells.resize(cellCountZ);
		for (size_t z = 0; z < cellCountZ; z++)
		{
			cells[z].resize(cellCountY);
			for (size_t y = 0; y < cellCountY; y++)
			{
				cells[z][y].resize(cellCountX);
				for (size_t x = 0; x < cellCountX; x++)
				{
					auto minPoint = V3(xyz.x + (float)x * cellSize, xyz.y + (float)y * cellSize, xyz.z + (float)z * cellSize);
					auto maxPoint = V3(xyz.x + (float)(x + 1) * cellSize, xyz.y + (float)(y + 1) * cellSize, xyz.z + (float)(z + 1) * cellSize);
					auto cell = new CellType(minPoint, maxPoint);
					cells[z][y][x] = cell;
				}
			}
		}

		for (auto& v : mesh->GetVertices())
		{
			InsertVertex(v);
		}

		for (auto& t : mesh->GetTriangles())
		{
			InsertTriangle(t);
		}
	}

	tuple<size_t, size_t, size_t> RegularGrid::InsertVertex(Vertex* vertex)
	{
		auto index = GetIndex(vertex->p);
		auto cell = GetCell(index);
		if (nullptr != cell) {
			cell->GetVertices().insert(vertex);
		}
		return index;
	}

	void RegularGrid::InsertTriangle(Triangle* t)
	{
		auto p0 = t->v0->p;
		auto p1 = t->v1->p;
		auto p2 = t->v2->p;

		AABB taabb;
		taabb.Expand(p0);
		taabb.Expand(p1);
		taabb.Expand(p2);
		auto minIndex = GetIndex(taabb.GetMinPoint());
		auto maxIndex = GetIndex(taabb.GetMaxPoint());
		for (size_t z = get<2>(minIndex); z <= get<2>(maxIndex); z++) {
			for (size_t y = get<1>(minIndex); y <= get<1>(maxIndex); y++) {
				for (size_t x = get<0>(minIndex); x <= get<0>(maxIndex); x++) {
					auto cell = cells[(int)z][(int)y][(int)x];
					if (cell->IntersectsTriangle(p0, p1, p2))
					{
						cell->GetTriangles().insert(t);
					}
				}
			}
		}
	}

	vector<vector<V3>> RegularGrid::ExtractSurface(float isolevel) const
	{
		auto cellHalfSize = cellSize * 0.5f;

		vector<vector<V3>> result;

		for (size_t z = 0; z < cellCountZ; z++)
		{
			for (size_t y = 0; y < cellCountY; y++)
			{
				for (size_t x = 0; x < cellCountX; x++)
				{
					CellType* cell = GetCell(x, y, z);
					if (nullptr == cell)
						continue;

					{
						GRIDCELL gridCell;
						gridCell.p[0] = cell->xyz - cellHalfSize;
						gridCell.p[1] = cell->Xyz - cellHalfSize;
						gridCell.p[2] = cell->XyZ - cellHalfSize;
						gridCell.p[3] = cell->xyZ - cellHalfSize;
						gridCell.p[4] = cell->xYz - cellHalfSize;
						gridCell.p[5] = cell->XYz - cellHalfSize;
						gridCell.p[6] = cell->XYZ - cellHalfSize;
						gridCell.p[7] = cell->xYZ - cellHalfSize;
						gridCell.val[0] = 1.0f;
						gridCell.val[1] = 1.0f;
						gridCell.val[2] = 1.0f;
						gridCell.val[3] = 1.0f;
						gridCell.val[4] = 1.0f;
						gridCell.val[5] = 1.0f;
						gridCell.val[6] = 1.0f;
						gridCell.val[7] = 1.0f;

						for (size_t i = 0; i < 8; i++)
						{
							const CellType* pcell = GetCell(GetIndex(gridCell.p[i]));
							if (nullptr != pcell)
							{
								if (pcell->selected)
								{
									gridCell.val[i] = -1.0f;
								}
							}
						}

						TRIANGLE triangles[16];
						int not = Polygonise(gridCell, isolevel, (TRIANGLE*)triangles);
						for (size_t i = 0; i < not; i++)
						{
							auto v0 = triangles[i].p[0];
							auto v1 = triangles[i].p[1];
							auto v2 = triangles[i].p[2];

							result.push_back({ triangles[i].p[0], triangles[i].p[1], triangles[i].p[2] });
						}
					}
					if (x == cellCountX - 1 || y == cellCountY - 1 || z == cellCountZ - 1)
					{
						GRIDCELL gridCell;
						gridCell.p[0] = cell->xyz + cellHalfSize;
						gridCell.p[1] = cell->Xyz + cellHalfSize;
						gridCell.p[2] = cell->XyZ + cellHalfSize;
						gridCell.p[3] = cell->xyZ + cellHalfSize;
						gridCell.p[4] = cell->xYz + cellHalfSize;
						gridCell.p[5] = cell->XYz + cellHalfSize;
						gridCell.p[6] = cell->XYZ + cellHalfSize;
						gridCell.p[7] = cell->xYZ + cellHalfSize;
						gridCell.val[0] = 1.0f;
						gridCell.val[1] = 1.0f;
						gridCell.val[2] = 1.0f;
						gridCell.val[3] = 1.0f;
						gridCell.val[4] = 1.0f;
						gridCell.val[5] = 1.0f;
						gridCell.val[6] = 1.0f;
						gridCell.val[7] = 1.0f;

						for (size_t i = 0; i < 8; i++)
						{
							auto pcell = GetCell(GetIndex(gridCell.p[i]));
							if (nullptr != pcell)
							{
								if (pcell->selected)
								{
									gridCell.val[i] = -1.0f;
								}
							}
						}

						TRIANGLE triangles[16];
						int not = Polygonise(gridCell, isolevel, (TRIANGLE*)triangles);
						for (size_t i = 0; i < not; i++)
						{
							auto v0 = triangles[i].p[0];
							auto v1 = triangles[i].p[1];
							auto v2 = triangles[i].p[2];

							result.push_back({ triangles[i].p[0], triangles[i].p[1], triangles[i].p[2] });
						}
					}
				}
			}
		}

		return result;
	}

	//void RegularGrid::ForEachCell(function<void(CellType*, size_t, size_t, size_t)> callback)
	//{
	//	auto t = Time("ForEachCell");

	//	for (size_t z = 0; z < cellCountZ; z++)
	//	{
	//		for (size_t y = 0; y < cellCountY; y++)
	//		{
	//			for (size_t x = 0; x < cellCountX; x++)
	//			{
	//				auto cell = GetCell(x, y, z);
	//				callback(cell, x, y, z);
	//			}
	//		}
	//	}
	//}

	void RegularGrid::SelectOutsideCells()
	{
		stack<tuple<size_t, size_t, size_t>> toCheck;
		toCheck.push(make_tuple(0, 0, 0));
		while (false == toCheck.empty())
		{
			auto currentIndex = toCheck.top();
			toCheck.pop();

			auto cell = GetCell(currentIndex);
			if (nullptr == cell) 
				continue;

			cell->tempFlag = CellType::CellFlag::FloodFillVisited;

			if (0 < cell->GetTriangles().size())
				continue;

			if (nullptr != cell)
			{
				if (false == cell->selected)
				{
					cell->selected = true;

					auto x = get<0>(currentIndex);
					auto y = get<1>(currentIndex);
					auto z = get<2>(currentIndex);

					auto mx = x - 1; auto px = x + 1;
					auto my = y - 1; auto py = y + 1;
					auto mz = z - 1; auto pz = z + 1;

					clamp(mx, 0, cellCountX - 1);
					clamp(px, 0, cellCountX - 1);

					clamp(my, 0, cellCountY - 1);
					clamp(py, 0, cellCountY - 1);

					clamp(mz, 0, cellCountZ - 1);
					clamp(pz, 0, cellCountZ - 1);

					toCheck.push(make_tuple(mx, y, z));
					toCheck.push(make_tuple(px, y, z));

					toCheck.push(make_tuple(x, my, z));
					toCheck.push(make_tuple(x, py, z));

					toCheck.push(make_tuple(x, y, mz));
					toCheck.push(make_tuple(x, y, pz));
				}
			}
		}
	}

	void RegularGrid::InvertSelectedCells()
	{
		for (size_t z = 0; z < cellCountZ; z++)
		{
			for (size_t y = 0; y < cellCountY; y++)
			{
				for (size_t x = 0; x < cellCountX; x++)
				{
					auto cell = GetCell(x, y, z);
					if (nullptr != cell)
					{
						cell->selected = !cell->selected;
					}
				}
			}
		}
	}

	void RegularGrid::ShrinkSelectedCells(int iteration)
	{
		for (size_t i = 0; i < iteration; i++)
		{
			for (size_t z = 0; z < cellCountZ; z++)
			{
				for (size_t y = 0; y < cellCountY; y++)
				{
					for (size_t x = 0; x < cellCountX; x++)
					{
						auto cell = GetCell(x, y, z);
						if (nullptr != cell)
						{
							auto mx = x - 1; auto px = x + 1;
							auto my = y - 1; auto py = y + 1;
							auto mz = z - 1; auto pz = z + 1;

							auto cellMX = GetCell(mx, y, z);
							auto cellPX = GetCell(px, y, z);

							auto cellMY = GetCell(x, my, z);
							auto cellPY = GetCell(x, py, z);

							auto cellMZ = GetCell(x, y, mz);
							auto cellPZ = GetCell(x, y, pz);

							bool allSelected = true;
							if (nullptr != cellMX) { if (false == cellMX->selected) allSelected = false; }
							else { allSelected = false; }
							if (nullptr != cellPX) { if (false == cellPX->selected) allSelected = false; }
							else { allSelected = false; }
							if (nullptr != cellMY) { if (false == cellMY->selected) allSelected = false; }
							else { allSelected = false; }
							if (nullptr != cellPY) { if (false == cellPY->selected) allSelected = false; }
							else { allSelected = false; }
							if (nullptr != cellMZ) { if (false == cellMZ->selected) allSelected = false; }
							else { allSelected = false; }
							if (nullptr != cellPZ) { if (false == cellPZ->selected) allSelected = false; }
							else { allSelected = false; }

							if (false == allSelected)
							{
								//cell->tempFlag = 128;
								cell->tempFlag = CellType::CellFlag::PartiallySelected;
							}
						}
					}
				}
			}

			for (size_t z = 0; z < cellCountZ; z++)
			{
				for (size_t y = 0; y < cellCountY; y++)
				{
					for (size_t x = 0; x < cellCountX; x++)
					{
						auto cell = GetCell(x, y, z);
						if (nullptr != cell)
						{
							if (cell->tempFlag == CellType::CellFlag::PartiallySelected)
							{
								cell->tempFlag = CellType::CellFlag::None;
								cell->selected = false;
							}
						}
					}
				}
			}
		}
	}

	void RegularGrid::ExtrudeSelectedCells(const V3& direction, int iteration)
	{
		for (size_t i = 0; i < iteration; i++)
		{
			for (size_t z = 0; z < cellCountZ; z++)
			{
				for (size_t y = 0; y < cellCountY; y++)
				{
					for (size_t x = 0; x < cellCountX; x++)
					{
						auto cell = GetCell(x, y, z);
						if (cell->selected)
						{
							auto offset = direction;
							offset.x *= cell->GetXLength();
							offset.y *= cell->GetYLength();
							offset.z *= cell->GetZLength();
							auto index = GetIndex(cell->GetCenter() + offset);
							auto nextCell = GetCell(index);
							if (nullptr != nextCell)
							{
								nextCell->tempFlag = CellType::CellFlag::Extruded;
							}
						}
					}
				}
			}

			for (size_t z = 0; z < cellCountZ; z++)
			{
				for (size_t y = 0; y < cellCountY; y++)
				{
					for (size_t x = 0; x < cellCountX; x++)
					{
						auto cell = GetCell(x, y, z);
						if (cell->tempFlag == CellType::CellFlag::Extruded)
						{
							cell->selected = true;
						}
					}
				}
			}
		}
	}

	vector<uint32_t> Mesh::Triangulate(const vector<vector<V2>>& pointsList)
	{
		vector<uint32_t> result;

		using Point = std::array<float, 2>;
		vector<vector<Point>> polygon;

		for (auto& points : pointsList)
		{
			vector<Point> contour;
			for (auto& p : points)
			{
				contour.push_back({ p.x, p.y });
			}
			polygon.push_back(contour);
		}

		return mapbox::earcut<uint32_t>(polygon);
	}

	Mesh::Mesh(float vertexEpsilon)
		: vertexEpsilon(vertexEpsilon)
	{
	}

	Mesh::~Mesh()
	{
		Clear();
	}

	void Mesh::Clear()
	{
		kdtree.Clear();

		for (auto vertex : vertices)
		{
			delete vertex;
		}
		vertices.clear();
		vid = 0;

		for (auto edge : edges)
		{
			delete edge;
		}
		edges.clear();
		eid = 0;
		edgeMapping.clear();

		for (auto triangle : triangles)
		{
			delete triangle;
		}
		triangles.clear();
		tid = 0;
		triangleMapping.clear();

		totalArea = 0.0;
	}

	void Mesh::Clone(Mesh& clone)
	{
		clone.Clear();

		map<Vertex*, Vertex*> vertexMapping;
		map<Vertex*, Vertex*> vertexDiffMapping;
		for (auto& v : vertices)
		{
			vertexMapping[v] = clone.AddVertex(v->p, { 0.0f, 0.0f, 0.0f, });
			vertexDiffMapping[v] = vertexMapping[v];
		}

		for (auto& t : triangles)
		{
			clone.AddTriangle(vertexMapping[t->v0], vertexMapping[t->v1], vertexMapping[t->v2]);
			vertexMapping[t->v0]->diffuse = vertexDiffMapping[vertexMapping[t->v0]]->diffuse;
		}
	}

	Vertex* Mesh::GetVertex(const V3& position) const
	{
		auto nn = kdtree.FindNearestNeighborNode(position);
		if (nullptr != nn)
		{
			if (vertexEpsilon > magnitude(nn->GetVertex()->p - position))
			{
				return nn->GetVertex();
			}
		}

		return nullptr;
	}

	Vertex* Mesh::AddVertex(const V3& position, const V3& normal)
	{
		aabb.Expand(position);

		auto vertex = GetVertex(position);
		if (nullptr == vertex)
		{
			Vertex* vertex = new Vertex;
			vertex->id = vid++;
			vertex->p = position;
			vertex->n = normal;
			vertices.push_back(vertex);

			kdtree.Insert(vertex);

			return vertex;
		}
		else
		{
			return vertex;
		}
	}

	Edge* Mesh::GetEdge(const Vertex* v0, const Vertex* v1) const
	{
		tuple<const Vertex*, const Vertex*> t0 = make_tuple(v0, v1);
		tuple<const Vertex*, const Vertex*> t1 = make_tuple(v1, v0);
		if (edgeMapping.count(t0) != 0)
		{
			return edgeMapping.at(t0);
			//return edgeMapping[t0];
		}
		else if (edgeMapping.count(t1) != 0)
		{
			return edgeMapping.at(t1);
			//return edgeMapping[t1];
		}
		else
		{
			return nullptr;
		}
	}

	Edge* Mesh::AddEdge(Vertex* v0, Vertex* v1)
	{
		auto edge = GetEdge(v0, v1);
		if (nullptr == edge)
		{
			edge = new Edge;
			edge->id = eid++;
			edge->v0 = v0;
			edge->v1 = v1;
			edge->length = magnitude(v0->p - v1->p);
			edges.insert(edge);
			v0->edges.insert(edge);
			v1->edges.insert(edge);
			edgeMapping[make_tuple(v0, v1)] = edge;
			edgeMapping[make_tuple(v1, v0)] = edge;
			return edge;
		}
		else
		{
			return edge;
		}
	}

	Vertex* Mesh::GetCommonVertex(const Edge* e0, const Edge* e1) const
	{
		auto e0v0 = e0->v0;
		auto e0v1 = e0->v1;
		auto e1v0 = e1->v0;
		auto e1v1 = e1->v1;

		if (e0v0 == e1v0)
			return e0v0;
		if (e0v0 == e1v1)
			return e0v0;
		if (e0v1 == e1v0)
			return e0v1;
		if (e0v1 == e1v1)
			return e0v1;

		return nullptr;
	}

	Triangle* Mesh::GetTriangle(Vertex* v0, Vertex* v1, Vertex* v2) const
	{
		auto e0 = GetEdge(v0, v1);
		auto e1 = GetEdge(v1, v2);
		auto e2 = GetEdge(v2, v0);

		auto t0 = make_tuple(e0, e1, e2);
		auto t1 = make_tuple(e1, e2, e0);
		auto t2 = make_tuple(e2, e0, e1);
		if (triangleMapping.count(t0) != 0)
		{
			return triangleMapping.at(t0);
		}
		else if (triangleMapping.count(t1) != 0)
		{
			return triangleMapping.at(t1);
		}
		else if (triangleMapping.count(t2) != 0)
		{
			return triangleMapping.at(t2);
		}
		else
		{
			return nullptr;
		}
	}

	Triangle* Mesh::AddTriangle(Vertex* v0, Vertex* v1, Vertex* v2)
	{
		if (v0 == v1 || v1 == v2 || v2 == v0)
			return nullptr;

		auto triangle = GetTriangle(v0, v1, v2);
		if (nullptr == triangle)
		{
			triangle = new Triangle;
			triangle->id = tid++;

			auto e0 = AddEdge(v0, v1);
			auto e1 = AddEdge(v1, v2);
			auto e2 = AddEdge(v2, v0);

			triangle->v0 = v0;
			triangle->v1 = v1;
			triangle->v2 = v2;

			if (triangle->v0 == triangle->v1 || triangle->v1 == triangle->v2 || triangle->v2 == triangle->v0)
			{
				std::cout << "YYYYY" << std::endl;
			}
			if (triangle->v0 == triangle->v1 && triangle->v1 == triangle->v2)
			{
				std::cout << "XXXXX" << std::endl;
			}

			triangles.insert(triangle);
			triangleMapping[make_tuple(e0, e1, e2)] = triangle;
			triangleMapping[make_tuple(e1, e2, e0)] = triangle;
			triangleMapping[make_tuple(e2, e0, e1)] = triangle;

			v0->triangles.insert(triangle);
			v1->triangles.insert(triangle);
			v2->triangles.insert(triangle);

			e0->triangles.insert(triangle);
			e1->triangles.insert(triangle);
			e2->triangles.insert(triangle);

			triangle->centroid = { (triangle->v0->p.x + triangle->v1->p.x + triangle->v2->p.x) / 3,
									(triangle->v0->p.y + triangle->v1->p.y + triangle->v2->p.y) / 3,
									(triangle->v0->p.z + triangle->v1->p.z + triangle->v2->p.z) / 3 };

			auto d01 = (triangle->v1->p - triangle->v0->p);
			auto d02 = (triangle->v2->p - triangle->v0->p);
			auto area = magnitude(cross(d01, d02)) * float(0.5);
			totalArea += area;
			d01 = normalize(d01);
			d02 = normalize(d02);
			triangle->normal = normalize(cross(d01, d02));

			return triangle;
		}
		else
		{
			return triangle;
		}
	}

	void Mesh::RemoveTriangle(Triangle* triangle)
	{
		auto v0 = triangle->v0;
		auto v1 = triangle->v1;
		auto v2 = triangle->v2;

		v0->triangles.erase(triangle);
		v1->triangles.erase(triangle);
		v2->triangles.erase(triangle);

		auto e0 = GetEdge(triangle->v0, triangle->v1);
		auto e1 = GetEdge(triangle->v1, triangle->v2);
		auto e2 = GetEdge(triangle->v2, triangle->v0);

		e0->triangles.erase(triangle);
		e1->triangles.erase(triangle);
		e2->triangles.erase(triangle);

		triangleMapping.erase(make_tuple(e0, e1, e2));
		triangleMapping.erase(make_tuple(e1, e2, e0));
		triangleMapping.erase(make_tuple(e2, e0, e1));
		triangles.erase(triangle);

		delete triangle;
	}

	void Mesh::AddInnerTriangles(vector<vector<V3>>& triangles)
	{
		for (auto& vs : triangles)
		{
#pragma region Skip floor
			//if (direction.x > 0 || direction.y > 0 || direction.z > 0)
			//{
			//	if ((vs[0].x + 0.0001f >= rg.GetMaxPoint().x) ||
			//		(vs[1].x + 0.0001f >= rg.GetMaxPoint().x) ||
			//		(vs[2].x + 0.0001f >= rg.GetMaxPoint().x))
			//	{
			//		continue;
			//	}

			//	if ((vs[0].y + 0.0001f >= rg.GetMaxPoint().y) ||
			//		(vs[1].y + 0.0001f >= rg.GetMaxPoint().y) ||
			//		(vs[2].y + 0.0001f >= rg.GetMaxPoint().y))
			//	{
			//		continue;
			//	}

			//	if ((vs[0].z + 0.0001f >= rg.GetMaxPoint().z) ||
			//		(vs[1].z + 0.0001f >= rg.GetMaxPoint().z) ||
			//		(vs[2].z + 0.0001f >= rg.GetMaxPoint().z))
			//	{
			//		continue;
			//	}
			//}
			//else if(direction.x < 0 || direction.y < 0 || direction.z < 0)
			//{
			//	if ((vs[0].x - 0.0001f <= rg.GetMinPoint().x) ||
			//		(vs[1].x - 0.0001f <= rg.GetMinPoint().x) ||
			//		(vs[2].x - 0.0001f <= rg.GetMinPoint().x))
			//	{
			//		continue;
			//	}

			//	if ((vs[0].y - 0.0001f <= rg.GetMinPoint().y) ||
			//		(vs[1].y - 0.0001f <= rg.GetMinPoint().y) ||
			//		(vs[2].y - 0.0001f <= rg.GetMinPoint().y))
			//	{
			//		continue;
			//	}

			//	if ((vs[0].z - 0.0001f <= rg.GetMinPoint().z) ||
			//		(vs[1].z - 0.0001f <= rg.GetMinPoint().z) ||
			//		(vs[2].z - 0.0001f <= rg.GetMinPoint().z))
			//	{
			//		continue;
			//	}
			//}
#pragma endregion

			//scene->Debug("Result")->AddTriangle(vs[0], vs[1], vs[2], glm::vec4(0.7f, 0.6f, 0.4f, 1.0f), glm::vec4(0.7f, 0.6f, 0.4f, 1.0f), glm::vec4(0.7f, 0.6f, 0.4f, 1.0f));
			auto v0 = AddVertex(vs[0], "zero");
			auto v1 = AddVertex(vs[1], "zero");
			auto v2 = AddVertex(vs[2], "zero");

			AddTriangle(v0, v2, v1);
		}
	}

	void Mesh::AddTriangles(const set<Triangle*>& triangles)
	{
		for (auto& triangle : triangles)
		{
			auto v0 = AddVertex(triangle->v0->p, triangle->v0->n);
			auto v1 = AddVertex(triangle->v1->p, triangle->v1->n);
			auto v2 = AddVertex(triangle->v2->p, triangle->v2->n);

			AddTriangle(v0, v1, v2);
		}
	}

	void Mesh::	AddTriangles(const vector<Vertex*>& vertices, const vector<uint32_t>& indices, bool clockwise)
	{
		for (size_t i = 0; i < indices.size() / 3; i++)
		{
			Vertex* v0 = vertices[indices[i * 3 + 0]];
			Vertex* v1 = vertices[indices[i * 3 + 1]];
			Vertex* v2 = vertices[indices[i * 3 + 2]];

			if (clockwise)
			{
				AddTriangle(v0, v2, v1);
			}
			else
			{
				AddTriangle(v0, v1, v2);
			}
		}
	}

	set<Vertex*> Mesh::GetAdjacentVertices(Vertex* vertex) const
	{
		set<Vertex*> adjacentVertices;
		for (auto e : vertex->edges)
		{
			if (e->triangles.size() != 0)
			{
				if (e->v0 != vertex)
				{
					if (e->v0->triangles.size() != 0)
					{
						adjacentVertices.insert(e->v0);
					}
				}
				else if (e->v1 != vertex)
				{
					if (e->v1->triangles.size() != 0)
					{
						adjacentVertices.insert(e->v1);
					}
				}
			}
		}
		return adjacentVertices;
	}

	set<Vertex*> Mesh::GetVerticesInRadius(const V3& position, float radius) const
	{
		set<Vertex*> result;

		auto vertices = kdtree.RangeSearch(position, radius * radius);
		result.insert(vertices.begin(), vertices.end());

		return result;
	}

	//float Mesh::GetDistanceFromEdge(Edge* edge, const V3& position)
	//{
	//	auto ray = HRay(edge->v0->position, edge->v1->position - edge->v0->position);
	//	auto p = ray.GetNearestPointOnRay(position);
	//	return glm::distance(p, position);
	//}

	tuple<V3, V3, V3>
		Mesh::GetTrianglePoints(const Triangle* triangle) const
	{
		auto p0 = triangle->v0->p;
		auto p1 = triangle->v1->p;
		auto p2 = triangle->v2->p;
		return make_tuple(p0, p1, p2);
	}

	bool compareByfloat(const std::tuple<float, Triangle*, V3>& tuple1,
		const std::tuple<float, Triangle*, V3>& tuple2)
	{
		return std::get<0>(tuple1) < std::get<0>(tuple2);
	}

	V3 Mesh::GetTriangleCentroid(const Triangle* triangle) const
	{
		auto tps = GetTrianglePoints(triangle);
		auto& p0 = get<0>(tps);
		auto& p1 = get<1>(tps);
		auto& p2 = get<2>(tps);
		return { (p0.x + p1.x + p2.x) / 3, (p0.y + p1.y + p2.y) / 3,
				(p0.z + p1.z + p2.z) / 3 };
	}

	float Mesh::GetTriangleArea(const Triangle* triangle) const
	{
		auto d01 = (triangle->v1->p - triangle->v0->p);
		auto d02 = (triangle->v2->p - triangle->v0->p);
		return magnitude(cross(d01, d02)) * float(0.5);
	}

	void Mesh::FlipTriangle(Triangle* triangle)
	{
		auto t1 = triangle->v1;
		auto t2 = triangle->v2;
		triangle->v1 = t2;
		triangle->v2 = t1;

		triangle->normal = -triangle->normal;
	}

	//V3 Mesh::GetNearestPointOnEdge(Edge* edge, const V3& position)
	//{
	//	auto ray = HRay(edge->v0->position, edge->v1->position - edge->v0->position);
	//	return ray.GetNearestPointOnRay(position);
	//}

	HVETM::Vertex* Mesh::GetNearestVertex(const V3& position) const
	{
		auto node = kdtree.FindNearestNeighborNode(position);
		return node->GetVertex();
	}

	Vertex* Mesh::GetNearestVertexOnTriangle(const Triangle* triangle, const V3& position) const
	{
		auto di0 = magnitude(position - triangle->v0->p);
		auto di1 = magnitude(position - triangle->v1->p);
		auto di2 = magnitude(position - triangle->v2->p);

		Vertex* nv = nullptr;
		if (di0 < di1 && di0 < di2)
		{
			nv = triangle->v0;
		}
		else if (di1 < di0 && di1 < di2)
		{
			nv = triangle->v1;
		}
		else if (di2 < di0 && di2 < di1)
		{
			nv = triangle->v2;
		}

		return nv;
	}

	//Edge* Mesh::GetNearestEdgeOnTriangle(Triangle* triangle, const V3& position)
	//{
	//	//auto dp = point - origin;
	//	//auto distanceFromOrigin = direction * dp;
	//	//return origin + direction * distanceFromOrigin;

	//	auto d0 = GetDistanceFromEdge(triangle->e0, position);
	//	auto d1 = GetDistanceFromEdge(triangle->e1, position);
	//	auto d2 = GetDistanceFromEdge(triangle->e2, position);

	//	Edge* e = nullptr;
	//	if (d0 < d1 && d0 < d2)
	//	{
	//		e = triangle->e0;
	//	}
	//	else if (d1 < d0 && d1 < d2)
	//	{
	//		e = triangle->e1;
	//	}
	//	else if (d2 < d0 && d2 < d1)
	//	{
	//		e = triangle->e2;
	//	}
	//	return e;
	//}

	set<Triangle*> Mesh::GetAdjacentTrianglesByEdge(const Triangle* triangle) const
	{
		set<Triangle*> adjacentTriangles;

		//adjacentTriangles.insert(triangle->e0->triangles.begin(),
		//	triangle->e0->triangles.end());
		//adjacentTriangles.insert(triangle->e1->triangles.begin(),
		//	triangle->e1->triangles.end());
		//adjacentTriangles.insert(triangle->e2->triangles.begin(),
		//	triangle->e2->triangles.end());

		adjacentTriangles.erase(const_cast<Triangle*>(triangle));

		return adjacentTriangles;
	}

	set<Triangle*> Mesh::GetAdjacentTrianglesByVertex(const Triangle* triangle) const
	{
		set<Triangle*> adjacentTriangles;

		for (auto edge : triangle->v0->edges)
		{
			adjacentTriangles.insert(edge->triangles.begin(), edge->triangles.end());
		}

		for (auto edge : triangle->v1->edges)
		{
			adjacentTriangles.insert(edge->triangles.begin(), edge->triangles.end());
		}

		for (auto edge : triangle->v2->edges)
		{
			adjacentTriangles.insert(edge->triangles.begin(), edge->triangles.end());
		}

		adjacentTriangles.erase(const_cast<Triangle*>(triangle));

		return adjacentTriangles;
	}

	set<Triangle*> Mesh::GetConnectedTriangles(Triangle* triangle) const
	{
		set<Triangle*> visited;
		stack<Triangle*> triangleStack;
		triangleStack.push(triangle);
		while (triangleStack.empty() == false)
		{
			auto currentTriangle = triangleStack.top();
			triangleStack.pop();

			if (visited.count(currentTriangle) != 0)
			{
				continue;
			}
			visited.insert(currentTriangle);

			auto ats = GetAdjacentTrianglesByVertex(currentTriangle);
			for (auto at : ats)
			{
				if (visited.count(at) == 0)
				{
					triangleStack.push(at);
				}
			}
		}

		return visited;
	}

	vector<Mesh*> Mesh::SeparateConnectedGroup()
	{
		vector<Mesh*> result;

		//set<Triangle*> visited;
		//for (auto triangle : triangles)
		//{
		//	if (visited.count(triangle) != 0)
		//	{
		//		continue;
		//	}

		//	auto group = GetConnectedTriangles(triangle);
		//	visited.insert(group.begin(), group.end());

		//	auto model = new Mesh(volume.GetVoxelSize());
		//	result.push_back(model);
		//	ExtractTriangles(*model, group);
		//}

		//sort(result.begin(), result.end(), [](Mesh* a, Mesh* b)
		//	{ return a->GetTotalArea() > b->GetTotalArea(); });

		return result;
	}

	vector<vector<Edge*>> Mesh::GetBorderEdges() const
	{
		vector<vector<Edge*>> result;

		set<Edge*> borderEdges;

		for (auto& edge : edges)
		{
			if (edge->triangles.size() < 2)
			{
				borderEdges.insert(edge);
			}
		}

		while (borderEdges.empty() == false)
		{
			vector<Edge*> border;
			Edge* seed = *borderEdges.begin();
			Edge* currentEdge = seed;
			set<Edge*> visited;
			do
			{
				if (0 != visited.count(currentEdge))
					break;

				visited.insert(currentEdge);
				border.push_back(currentEdge);
				borderEdges.erase(currentEdge);

				for (auto& ne : currentEdge->v1->edges)
				{
					if (ne->triangles.size() < 2)
					{
						if (ne->id != currentEdge->id)
						{
							//cout << "currentEdge->id : " << currentEdge->id << endl;

							currentEdge = ne;
							break;
						}
					}
				}
			} while (nullptr != currentEdge && currentEdge != seed);

			//auto d00 = distance(border.front()->v0->p, border.back()->v0->p);
			//auto d01 = distance(border.front()->v0->p, border.back()->v1->p);
			//auto d10 = distance(border.front()->v1->p, border.back()->v0->p);
			//auto d11 = distance(border.front()->v1->p, border.back()->v1->p);

			//if (1 > d00 && 1 > d01 && 1 > d10 && 1 > d11)
			//{
				result.push_back(border);
			//}
		}
		return result;
	}

	vector<Vertex*> Mesh::GetBorderVerticesFromBorderEdges(const vector<Edge*>& borderEdges) const
	{
		vector<Vertex*> borderVertices;

		for (size_t i = 0; i < borderEdges.size(); i++)
		{
			auto ce = borderEdges[i];
			auto ne = borderEdges[(i + 1) % borderEdges.size()];
			auto cv = GetCommonVertex(ce, ne);
			HVETM::Vertex* v0 = nullptr;
			HVETM::Vertex* v1 = nullptr;
			HVETM::Vertex* v2 = nullptr;
			if (ce->v0 == cv) v0 = ce->v1;
			else v0 = ce->v0;
			v1 = cv;
			if (ne->v0 == cv) v2 = ne->v1;
			else v2 = ne->v0;

			if (nullptr != v1)
			{
				borderVertices.push_back(v1);
			}
			else
			{
				//printf("???\n");
			}
		}

		return borderVertices;
	}

	void Mesh::FillTrianglesToMakeBorderSmooth(float maxAngle)
	{
		while (true)
		{
			bool triangleAdded = false;
			auto foundBorderEdges = GetBorderEdges();

			for (size_t k = 0; k < foundBorderEdges.size(); k++)
			{
				auto borderEdges = foundBorderEdges[k];
				for (size_t i = 0; i < borderEdges.size() - 1; i++)
				{
					auto ce = borderEdges[i];
					auto ne = borderEdges[i + 1];
					auto cv = GetCommonVertex(ce, ne);
					HVETM::Vertex* v0 = nullptr;
					HVETM::Vertex* v1 = nullptr;
					HVETM::Vertex* v2 = nullptr;
					if (ce->v0 == cv) v0 = ce->v1;
					else v0 = ce->v0;
					v1 = cv;
					if (ne->v0 == cv) v2 = ne->v1;
					else v2 = ne->v0;

					auto radian = angle(normalize(v0->p - v1->p), normalize(v2->p - v1->p));
					if (radian < maxAngle)
					{
						auto n = normalize(cross(normalize(v1->p - v0->p), normalize(v2->p - v0->p)));
						auto t = (*ce->triangles.begin());
						t->normal = normalize(cross(normalize(t->v1->p - t->v0->p), normalize(t->v2->p - t->v0->p)));
						if (dot(n, t->normal) < 0)
						{
							AddTriangle(v0, v2, v1);
							triangleAdded = true;

							i++;
						}
					}
				}
			}
			if (false == triangleAdded)
				break;
		}
	}

	void Mesh::ExtrudeBorder(const V3& direction, int segments)
	{
		auto foundBorderEdges = GetBorderEdges();

		for (size_t k = 0; k < foundBorderEdges.size(); k++)
		{
			vector<Edge*> borderEdges;
			vector<Edge*> newBorderEdges;


			for (size_t n = 0; n < segments; n++)
			{
				if (0 == n)
				{
					borderEdges = foundBorderEdges[k];
					newBorderEdges.resize(borderEdges.size());
				}
				for (size_t i = 0; i < borderEdges.size(); i++)
				{
					auto ce = borderEdges[i];
					auto ne = borderEdges[(i + 1) % borderEdges.size()];
					auto cv = GetCommonVertex(ce, ne);
					HVETM::Vertex* v0 = nullptr;
					HVETM::Vertex* v1 = nullptr;
					if (ce->v0 == cv) v0 = ce->v1;
					else v0 = ce->v0;
					v1 = cv;

					auto nv0 = AddVertex(v0->p + direction, { 0.0f, 0.0f, 0.0f });
					auto nv1 = AddVertex(v1->p + direction, { 0.0f, 0.0f, 0.0f });
					AddTriangle(v0, nv1, v1);
					AddTriangle(v0, nv0, nv1);

					newBorderEdges[i] = GetEdge(nv0, nv1);
				}
				swap(borderEdges, newBorderEdges);
				newBorderEdges.clear();
				newBorderEdges.resize(borderEdges.size());
			}
		}
	}

	void Mesh::MakeSmooth(size_t iteration, const V3& direction)
	{
		for (size_t i = 0; i < iteration; i++)
		{
			for (auto& v : GetVertices())
			{
				set<Vertex*> avs = GetAdjacentVertices(v);

				V3 p = "zero";
				for (auto& av : avs)
				{
					p += av->p;
				}
				p /= (float)(avs.size());

				if (2 == v->tempFlag)
				{
					if (fabsf(direction.x) > 0)
					{
						p.x = v->p.x;
					}
					if (fabsf(direction.y) > 0)
					{
						p.y = v->p.y;
					}
					if (fabsf(direction.z) > 0)
					{
						p.z = v->p.z;
					}
				}

				v->p = p;
			}
		}
	}

	V3 Mesh::DetermineBorderExtrudeDirection(const vector<Edge*>& borderEdges) const
	{
		V3 direction;
		AABB aabb;

		for (size_t i = 0; i < borderEdges.size(); i++)
		{
			auto ce = borderEdges[i];
			auto ne = borderEdges[(i + 1) % borderEdges.size()];
			auto cv = GetCommonVertex(ce, ne);
			HVETM::Vertex* v0 = nullptr;
			HVETM::Vertex* v1 = nullptr;
			HVETM::Vertex* v2 = nullptr;
			if (ce->v0 == cv) v0 = ce->v1;
			else v0 = ce->v0;
			v1 = cv;
			if (ne->v0 == cv) v2 = ne->v1;
			else v2 = ne->v0;

			auto e = borderEdges[i];
			auto t = *e->triangles.begin();
			auto d = V3{ 0.0f, 0.0f, 0.0f, };
			aabb.Expand(e->v0->p);
			aabb.Expand(e->v1->p);

			if (t->v0 == e->v0 && t->v1 == e->v1 || t->v0 == e->v1 && t->v1 == e->v0)
			{
				d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v2->p);
			}
			else if (t->v1 == e->v0 && t->v2 == e->v1 || t->v1 == e->v1 && t->v2 == e->v0)
			{
				d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v0->p);
			}
			else if (t->v2 == e->v0 && t->v0 == e->v1 || t->v2 == e->v1 && t->v0 == e->v0)
			{
				d = normalize((e->v0->p + e->v1->p) * 0.5f - t->v1->p);
			}

			direction += d;
		}

		direction = normalize(direction);
		if (abs(direction.x) > abs(direction.y) && abs(direction.x) > abs(direction.z))
		{
			if (0 < direction.x) direction = V3{ 1.0f, 0.0f, 0.0f };
			else direction = V3{ -1.0f, 0.0f, 0.0f };
		}
		else if (abs(direction.y) > abs(direction.x) && abs(direction.y) > abs(direction.z))
		{
			if (0 < direction.y) direction = V3{ 0.0f, 1.0f, 0.0f };
			else direction = V3{ 0.0f, -1.0f, 0.0f };
		}
		else if (abs(direction.z) > abs(direction.x) && abs(direction.z) > abs(direction.y))
		{
			if (0 < direction.z) direction = V3{ 0.0f, 0.0f, 1.0f };
			else direction = V3{ 0.0f, 0.0f, -1.0f };
		}

		return direction;
	}

	float Mesh::GetMinTotalHeight(bool isMaxillar) const
	{
		auto foundBorderEdges = GetBorderEdges();

		for (size_t k = 0; k < foundBorderEdges.size(); k++)
		{
			auto borderEdges = foundBorderEdges[k];
			//auto direction = DetermineBorderExtrudeDirection(borderEdges);

			V3 direction;
			direction.x = 0;
			direction.y = -1;
			direction.z = 0;
			if (isMaxillar)
			{
				direction.y = 1;
			}

			if (0 < fabsf(direction.x))
			{
				return aabb.GetXLength();
			}
			else if (0 < fabsf(direction.y))
			{
				return aabb.GetYLength();
			}
			else if (0 < fabsf(direction.z))
			{
				return aabb.GetZLength();
			}
		}

		return 0.0f;
	}

	void Mesh::BorderVertexSmoothing(const vector<Vertex*>& borderVertices, int iteration)
	{
		for (size_t n = 0; n < iteration; n++)
		{
			for (size_t i = 0; i < borderVertices.size(); i++)
			{
				auto v0 = borderVertices[i];
				auto v1 = borderVertices[(i + 1) % borderVertices.size()];
				auto v2 = borderVertices[(i + 2) % borderVertices.size()];

				v1->p = 0.5f * (v0->p + v2->p);
			}
		}
	}

	void Mesh::SimpleFillHole(bool leaveLargeHole)
	{
		auto foundBorderEdges = GetBorderEdges();
		for (size_t k = 0; k < foundBorderEdges.size(); k++)
		{
			vector<Edge*> borderEdges = foundBorderEdges[k];
			if (leaveLargeHole)
			{
				if (borderEdges.size() > 100)
					continue;
			}
			auto borderVertices = GetBorderVerticesFromBorderEdges(borderEdges);
			V3 center;
			for (auto& v : borderVertices)
			{
				center += v->p;
			}
			center /= borderVertices.size();

			auto cv = AddVertex(center, "zero");
			for (size_t i = 0; i < borderVertices.size(); i++)
			{
				auto v = borderVertices[i];
				auto nv = borderVertices[(i + 1) % borderVertices.size()];
				AddTriangle(v, cv, nv);
			}
		}
	}

	bool lineTriangleIntersection(const V3& P1, const V3& P2, const V3& A, const V3& B, const V3& C, V3& intersection_point) {
		// Calculate line direction and normalize it
		V3 dir = P2 - P1;
		dir = normalize(dir);

		// Calculate the normal vector of the triangle's plane
		V3 N = cross(B - A, C - A);
		N = normalize(N);

		// Calculate the distance from the origin to the plane
		float d = -dot(N, A);

		// Calculate the parameter t of the intersection point
		float t = -(dot(N, P1) + d) / dot(N, dir);

		// Calculate the intersection point
		intersection_point = P1 + t * dir;

		// Calculate the barycentric coordinates of the intersection point
		float u, v, w;
		V3 v0 = C - A;
		V3 v1 = B - A;
		V3 v2 = intersection_point - A;

		float dot00 = dot(v0, v0);
		float dot01 = dot(v0, v1);
		float dot02 = dot(v0, v2);
		float dot11 = dot(v1, v1);
		float dot12 = dot(v1, v2);

		float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);

		u = (dot11 * dot02 - dot01 * dot12) * invDenom;
		v = (dot00 * dot12 - dot01 * dot02) * invDenom;
		w = 1.0f - u - v;

		// Check if the intersection point is inside the triangle
		if (u >= 0 && v >= 0 && u + v <= 1) {
			return true;
		}

		return false;
	}

	void Mesh::DeleteSelfintersectingTriangles()
	{
		TS(DeleteSelfintersectingTriangles)
			set<Triangle*> toDelete;

		for (auto& t0 : triangles)
		{
			if (0 != toDelete.count(t0))
				continue;

			bool intersects = false;
			for (auto& t1 : triangles)
			{
				if (0 != toDelete.count(t1))
					continue;

				V3 intersection;
				if (lineTriangleIntersection(t0->v0->p, t0->v1->p, t1->v0->p, t1->v2->p, t1->v2->p, intersection))
				{
					toDelete.insert(t0);
					toDelete.insert(t1);

					intersects = true;
					break;
				}
			}

			if (true == intersects)
				break;
		}

		for (auto& t : toDelete)
		{
			RemoveTriangle(t);
		}
		TE(DeleteSelfintersectingTriangles)
	}

	void Mesh::SliceAndRemove(const V3& planePosition, const V3& planeNormal)
	{
		TS(SliceAndRemove);

		vector<tuple<V3, V3, V3>> toCreate;
		vector<tuple<V3, V3, V3>> toCreateColor;
		V3 d1(0.7, 0.7, 0.7);

		for (Triangle* t : triangles)
		{
			auto dp0 = t->v0->p - planePosition;
			auto dot0 = dot(dp0, planeNormal);
			auto dp1 = t->v1->p - planePosition;
			auto dot1 = dot(dp1, planeNormal);
			auto dp2 = t->v2->p - planePosition;
			auto dot2 = dot(dp2, planeNormal);
			if (-Epsilon <= dot0 && -Epsilon <= dot1 && -Epsilon <= dot2) // 모든 점이 평면의 앞에 있다. 점 유지
			{ 
				toCreate.push_back(make_tuple(t->v0->p, t->v1->p, t->v2->p));
				toCreateColor.push_back(make_tuple(t->v0->diffuse, t->v1->diffuse, t->v2->diffuse));

			}
			else if (-Epsilon > dot0 && -Epsilon > dot1 && -Epsilon > dot2) // 모든 점이 평면의 뒤에 있다. 점 삭제
			{ 
				continue;
			}
			else if (-Epsilon > dot0 && -Epsilon > dot1) // V2 remains
			{
				auto in0 = LinePlaneIntersection(t->v2->p, t->v0->p, planePosition, planeNormal);
				auto in1 = LinePlaneIntersection(t->v2->p, t->v1->p, planePosition, planeNormal);
				toCreate.push_back(make_tuple(t->v2->p, in0, in1));
				toCreateColor.push_back(make_tuple(t->v2->diffuse, t->v2->diffuse, t->v2->diffuse));
			}
			else if (-Epsilon > dot1 && -Epsilon > dot2) // V0 remains
			{
				auto in0 = LinePlaneIntersection(t->v0->p, t->v1->p, planePosition, planeNormal);
				auto in1 = LinePlaneIntersection(t->v0->p, t->v2->p, planePosition, planeNormal);
				toCreate.push_back(make_tuple(t->v0->p, in0, in1));
				toCreateColor.push_back(make_tuple(t->v0->diffuse, t->v0->diffuse, t->v0->diffuse));
			}
			else if (-Epsilon > dot2 && -Epsilon > dot0) // V1 remains
			{
				auto in0 = LinePlaneIntersection(t->v1->p, t->v2->p, planePosition, planeNormal);
				auto in1 = LinePlaneIntersection(t->v1->p, t->v0->p, planePosition, planeNormal);
				toCreate.push_back(make_tuple(t->v1->p, in0, in1));
				toCreateColor.push_back(make_tuple(t->v1->diffuse, t->v1->diffuse, t->v1->diffuse));
			}
			else if (-Epsilon > dot0) // v1, v2 remains
			{
				auto in0 = LinePlaneIntersection(t->v0->p, t->v1->p, planePosition, planeNormal);
				auto in1 = LinePlaneIntersection(t->v0->p, t->v2->p, planePosition, planeNormal);
				toCreate.push_back(make_tuple(t->v1->p, t->v2->p, in1));
				toCreate.push_back(make_tuple(t->v1->p, in1, in0));
				toCreateColor.push_back(make_tuple(t->v1->diffuse, t->v2->diffuse, d1));
				toCreateColor.push_back(make_tuple(t->v1->diffuse, t->v1->diffuse, t->v1->diffuse));
			}
			else if (-Epsilon > dot1) // v0, v2 remains
			{
				auto in0 = LinePlaneIntersection(t->v1->p, t->v2->p, planePosition, planeNormal);
				auto in1 = LinePlaneIntersection(t->v1->p, t->v0->p, planePosition, planeNormal);
				toCreate.push_back(make_tuple(t->v2->p, t->v0->p, in1));
				toCreate.push_back(make_tuple(t->v2->p, in1, in0));
				toCreateColor.push_back(make_tuple(t->v2->diffuse, t->v0->diffuse, d1));
				toCreateColor.push_back(make_tuple(t->v2->diffuse, t->v2->diffuse, t->v2->diffuse));
			}
			else if (-Epsilon > dot2) // v0, v1 remains
			{
				auto in0 = LinePlaneIntersection(t->v2->p, t->v0->p, planePosition, planeNormal);
				auto in1 = LinePlaneIntersection(t->v2->p, t->v1->p, planePosition, planeNormal);
				toCreate.push_back(make_tuple(t->v0->p, t->v1->p, in1));
				toCreate.push_back(make_tuple(t->v0->p, in1, in0));
				toCreateColor.push_back(make_tuple(t->v0->diffuse, t->v1->diffuse, d1));
				toCreateColor.push_back(make_tuple(t->v0->diffuse, t->v0->diffuse, t->v0->diffuse));
			}
		}

		this->Clear();

		int i = 0;
		for (auto& t : toCreate)
		{
			auto dif=toCreateColor.at(i);
			auto v0 = AddVertex(get<0>(t), "zero");
			auto v1 = AddVertex(get<1>(t), "zero");
			auto v2 = AddVertex(get<2>(t), "zero");
			AddTriangle(v0, v1, v2);

			v0->diffuse = get<0>(dif);
			v1->diffuse = get<1>(dif);
			v2->diffuse = get<2>(dif);


			//v0->diffuse.x = get<0>(dif).x;
			/*v0->diffuse.y = get<0>(dif).y;
			v0->diffuse.z = get<0>(dif).z;

			v1->diffuse.x = get<1>(dif).x;
			v1->diffuse.y = get<1>(dif).x;
			v1->diffuse.z = get<1>(dif).x;

			v2->diffuse.x = get<2>(dif).x;
			v2->diffuse.y = get<2>(dif).x;
			v2->diffuse.z = get<2>(dif).x;*/
			i++;
		}

		TE(SliceAndRemove);
	}

	//bool Mesh::CreateFromHNode(const HNode& node)
	//{
	//	map<size_t, HVETM::Vertex*> vertexMapping;
	//	for (size_t i = 0; i < node.GetVertexCount(); i++)
	//	{
	//		auto x = node.m_fpPositionBuffer[i * 3 + 0];
	//		auto y = node.m_fpPositionBuffer[i * 3 + 1];
	//		auto z = node.m_fpPositionBuffer[i * 3 + 2];

	//		auto vertex = AddVertex({ x,y,z }, { 0.0f, 0.0f, 0.0f });
	//		vertexMapping[i] = vertex;

	//		vertex->diffuse.x = node.m_fpDiffuseBuffer[i * 3 + 0];
	//		vertex->diffuse.y = node.m_fpDiffuseBuffer[i * 3 + 1];
	//		vertex->diffuse.z = node.m_fpDiffuseBuffer[i * 3 + 2];
	//	}

	//	for (size_t i = 0; i < node.GetMeshIndexCount() / 3; i++)
	//	{
	//		auto i0 = node.m_uipMeshIndexBuffer[i * 3 + 0];
	//		auto i1 = node.m_uipMeshIndexBuffer[i * 3 + 1];
	//		auto i2 = node.m_uipMeshIndexBuffer[i * 3 + 2];

	//		auto v0 = vertexMapping[i0];
	//		auto v1 = vertexMapping[i1];
	//		auto v2 = vertexMapping[i2];

	//		AddTriangle(v0, v1, v2);


	//	}
	//	return true;
	//}

	//HNode* Mesh::ToHNode() const
	//{
	//	if (HNode* node = new HNode())
	//	{
	//		node->SetPrimitiveType(PT_TRIANGLE_LIST_COLOR);
	//		node->SetShow(true);

	//		node->SetVertexCount(GetVertices().size());
	//		node->SetMeshIndexCount(GetTriangles().size()*3);

	//		const vector<Vertex*>& vertices = GetVertices();
	//		node->AllocateVertices(vertices.size(), HNode::VertexAttribute::PositionNormalDiffuse);

	//		map<const HVETM::Vertex*, int> vertexMapping;
	//		for (size_t i = 0; i < vertices.size(); i++)
	//		{
	//			const Vertex* vertex = vertices[i];
	//			node->m_fpPositionBuffer[i * 3 + 0] = vertex->p.x;
	//			node->m_fpPositionBuffer[i * 3 + 1] = vertex->p.y;
	//			node->m_fpPositionBuffer[i * 3 + 2] = vertex->p.z;

	//			vertexMapping[vertex] = i;

	//			node->m_fpDiffuseBuffer[i * 3 + 0] = vertex->diffuse.x;
	//			node->m_fpDiffuseBuffer[i * 3 + 1] = vertex->diffuse.y;
	//			node->m_fpDiffuseBuffer[i * 3 + 2] = vertex->diffuse.z;

	//			V3 normal; 
	//			if (vertices[i]->triangles.size() > 0) {
	//				for (auto& t : vertices[i]->triangles)
	//				{
	//					normal += t->normal;
	//				}
	//				normal /= (float)(vertices[i]->triangles.size());
	//				node->m_fpNormalBuffer[i * 3 + 0] = normal.x;
	//				node->m_fpNormalBuffer[i * 3 + 1] = normal.y;
	//				node->m_fpNormalBuffer[i * 3 + 2] = normal.z;
	//			}
	//			else {
	//				node->m_fpNormalBuffer[i * 3 + 0] = vertices[i]->n.x;
	//				node->m_fpNormalBuffer[i * 3 + 1] = vertices[i]->n.y;
	//				node->m_fpNormalBuffer[i * 3 + 2] = vertices[i]->n.z;
	//			}
	//		}

	//		
	//		const set<Triangle*>& triangles = GetTriangles();
	//		node->AllocateMeshIndices(triangles.size() * 3);

	//		size_t index = 0;
	//		for (auto& t : triangles)
	//		{
	//			node->m_uipMeshIndexBuffer[index * 3 + 0] = vertexMapping[t->v0];
	//			node->m_uipMeshIndexBuffer[index * 3 + 1] = vertexMapping[t->v1];
	//			node->m_uipMeshIndexBuffer[index * 3 + 2] = vertexMapping[t->v2];
	//			index++;
	//		}

	//		return node;
	//	}
	//	return nullptr;
	//}

#ifdef _DEBUG
	void Mesh::DumpBorderEdges(const std::string& message) const
	{
		vector<vector<Edge*>> foundBorderEdges = GetBorderEdges();
		cout << message << " border edge groups[" << foundBorderEdges.size() << "] ";
		for (vector<Edge*>& borderEdges : foundBorderEdges) {
			cout << borderEdges.size() << ", ";
		}
		cout << std::endl;
	}

	void TestHKDTree() {

	}
#endif


}
