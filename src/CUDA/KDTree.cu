#include "Octree.cuh"

#include <cufft.h>

#include <App/Serialization.hpp>
#include <App/Utility.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

namespace CUDA
{
	namespace KDTree
	{
        // Point 구조체 정의
        struct Point {
            float x, y, z;
            __host__ __device__
                Point(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}
        };

        // 축에 따라 비교하기 위한 functor
        struct CompareByDimension {
            int dim;
            CompareByDimension(int _dim) : dim(_dim) {}
            __host__ __device__
                bool operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const {
                if (dim == 0) return a.x() < b.x();
                if (dim == 1) return a.y() < b.y();
                return a.z() < b.z();
            }
        };

        // KDTree 노드 구조
        struct KDNode {
            Eigen::Vector3f point;
            int left, right;
        };

        // KDTree 빌드 함수
        void buildKDTree(thrust::device_vector<Eigen::Vector3f>& points, thrust::device_vector<KDNode>& nodes, int start, int end, int depth) {
            if (start >= end) return;

            int dim = depth % 3; // 차원 선택
            int mid = (start + end) / 2;

            // 데이터 정렬
            thrust::sort(points.begin() + start, points.begin() + end, CompareByDimension(dim));

            // KDNode에 추가
            KDNode node;
            node.point = points[mid];
            node.left = -1;
            node.right = -1;
            nodes.push_back(node);

            // 서브트리 재귀 호출
            buildKDTree(points, nodes, start, mid, depth + 1);
            buildKDTree(points, nodes, mid + 1, end, depth + 1);
        }

		void TestKDTree(std::vector<Eigen::Vector3f> inputPoints)
		{
            auto t = Time::Now();

            thrust::host_vector<Eigen::Vector3f> h_points(inputPoints);
            //// 데이터 초기화
            //thrust::host_vector<Point> h_points = {
            //    {3.0f, 6.0f, 7.0f}, {17.0f, 15.0f, 13.0f},
            //    {13.0f, 15.0f, 6.0f}, {6.0f, 12.0f, 10.0f},
            //    {9.0f, 1.0f, 2.0f}, {2.0f, 7.0f, 3.0f}
            //};

            // Device Vector로 복사
            thrust::device_vector<Eigen::Vector3f> d_points = h_points;
            thrust::device_vector<KDNode> d_nodes;

            // KDTree 빌드
            buildKDTree(d_points, d_nodes, 0, d_points.size(), 0);

            t = Time::End(t, "Build KDTree");

            // 결과 출력
            //thrust::host_vector<KDNode> h_nodes = d_nodes;
            //for (const auto& node : h_nodes) {
            //    std::cout << "Point: (" << node.point.x() << ", " << node.point.y() << ", " << node.point.z() << ")\n";
            //}

		}
	}
}
