#pragma once

#include <limits>
#include <atomic>
#include <functional>

using namespace std;

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace Spatial
{
    struct Ray {
        Eigen::Vector3f origin;
        Eigen::Vector3f direction;

        Ray(const Eigen::Vector3f& origin_, const Eigen::Vector3f& direction_)
            : origin(origin_), direction(direction_.normalized()) {}
    };

    struct OctreeNode {
        unique_ptr<OctreeNode> children[8];
        vector<size_t> pointIndices;  // Only leaf nodes will have this populated.
        Eigen::AlignedBox3f aabb;     // Axis-aligned bounding box for this node's region.
        int depth;                    // Depth level of this node in the octree.

        // Constructor
        OctreeNode(const Eigen::AlignedBox3f& aabb_, int depth_)
            : aabb(aabb_), depth(depth_) {}
    };

    class Octree {
    public:
        Octree(const Eigen::AlignedBox3f& rootAABB, int maxDepth, int maxPointsPerNode)
            : root(make_unique<OctreeNode>(rootAABB, 0)), maxDepth(maxDepth), maxPointsPerNode(maxPointsPerNode) {}

        void setPoints(Eigen::Vector3f* points, Eigen::Vector3f* normals, size_t numberOfPoints) {
            this->points = points;
            this->normals = normals;
            this->numberOfPoints = numberOfPoints;
        }

        void insert(size_t index) {
            if (nullptr == points)
            {
                printf("points is not set.\n");
                return;
            }
            insert(root.get(), index);
        }

        void traverse(function<void(OctreeNode* node)> f) {
            traverseNode(root.get(), f);
        }

        size_t pickPoint(const Ray& ray) {
            float closestDist = FLT_MAX;
            size_t closestPointIndex = -1;
            bool found = pickPointNode_OriginDistance(root.get(), ray, closestDist, closestPointIndex);
            if (found) {
                return closestPointIndex;
            }
            return -1;
        }

        vector<size_t> searchPointsNearRay(const Ray& ray, float maxDistance) {
            vector<size_t> result;
            searchPointsNearRayNode(root.get(), ray, maxDistance, result);
            return result;
        }

        float distanceToRay(const Ray& ray, const Eigen::Vector3f& point) {
            // Distance from point to ray (perpendicular distance)
            Eigen::Vector3f pToOrigin = point - ray.origin;
            float t = pToOrigin.dot(ray.direction);
            Eigen::Vector3f closestPointOnRay = ray.origin + t * ray.direction;
            return (closestPointOnRay - point).norm();
        }

        vector<size_t> radiusSearch(const Eigen::Vector3f& queryPoint, float radius) {
            vector<size_t> result;
            radiusSearchNode(root.get(), queryPoint, radius, result);
            return result;
        }

        Eigen::Vector3f* points = nullptr;
        Eigen::Vector3f* normals = nullptr;
        size_t numberOfPoints = 0;

    private:
        unique_ptr<OctreeNode> root;
        int maxDepth;
        int maxPointsPerNode;

        void insert(OctreeNode* node, size_t index) {
            Eigen::Vector3f& point = points[index];
            // Ensure point lies within the node's bounding box
            if (!node->aabb.contains(point)) {
                return;
            }

            // If we are at the maximum depth, add the point index to the leaf node
            if (node->depth == maxDepth) {
                node->pointIndices.push_back(index);
                return;
            }

            // If this node has no children, we need to subdivide to reach max depth
            if (node->children[0] == nullptr) {
                subdivide(node);
            }

            // Determine the correct child index for the point and insert it
            Eigen::Vector3f mid = (node->aabb.min() + node->aabb.max()) * 0.5f;
            int childIndex = determineChildIndex(point, mid);
            insert(node->children[childIndex].get(), index);
        }

        void subdivide(OctreeNode* node) {
            // Calculate the mid-point of the current bounding box
            Eigen::Vector3f min = node->aabb.min();
            Eigen::Vector3f max = node->aabb.max();
            Eigen::Vector3f mid = (min + max) * 0.5f;

            // Create 8 child AABBs for each sub-region
            for (int i = 0; i < 8; ++i) {
                Eigen::Vector3f childMin = min;
                Eigen::Vector3f childMax = max;

                // Determine the boundaries of the child AABB
                if (i & 1) childMin.x() = mid.x(); else childMax.x() = mid.x();
                if (i & 2) childMin.y() = mid.y(); else childMax.y() = mid.y();
                if (i & 4) childMin.z() = mid.z(); else childMax.z() = mid.z();

                Eigen::AlignedBox3f childAABB(childMin, childMax);
                node->children[i] = make_unique<OctreeNode>(childAABB, node->depth + 1);
            }
        }

        int determineChildIndex(const Eigen::Vector3f& point, const Eigen::Vector3f& mid) {
            int index = 0;
            if (point.x() > mid.x()) index |= 1;
            if (point.y() > mid.y()) index |= 2;
            if (point.z() > mid.z()) index |= 4;
            return index;
        }

        bool pickPointNode_RayDistance(OctreeNode* node, const Ray& ray, float& closestDist, size_t& closestPointIndex) {
            if (node == nullptr) {
                return false;
            }

            // Check if the ray intersects with the node's AABB
            if (!rayIntersectsAABB(ray, node->aabb)) {
                return false;
            }

            bool found = false;

            // If the node is a leaf, check each point
            if (node->depth == maxDepth) {
                for (size_t pointIndex : node->pointIndices) {
                    Eigen::Vector3f point = getPointFromIndex(pointIndex);
                    float dist = distanceToRay(ray, point);
                    if (dist < closestDist) {
                        closestDist = dist;
                        closestPointIndex = pointIndex;
                        found = true;
                    }
                }
            }
            else {
                // Recursively check each child node
                for (int i = 0; i < 8; ++i) {
                    if (pickPointNode_RayDistance(node->children[i].get(), ray, closestDist, closestPointIndex)) {
                        found = true;
                    }
                }
            }

            return found;
        }

        bool pickPointNode_OriginDistance(OctreeNode* node, const Ray& ray, float& closestDist, size_t& closestPointIndex) {
            if (node == nullptr) {
                return false;
            }

            // Check if the ray intersects with the node's AABB
            if (!rayIntersectsAABB(ray, node->aabb)) {
                return false;
            }

            bool found = false;

            // If the node is a leaf, check each point
            if (node->depth == maxDepth) {
                for (size_t pointIndex : node->pointIndices) {
                    Eigen::Vector3f point = getPointFromIndex(pointIndex);
                    float dist = (ray.origin - point).norm();  // Distance from ray origin to the point
                    if (dist < closestDist) {
                        closestDist = dist;
                        closestPointIndex = pointIndex;
                        found = true;
                    }
                }
            }
            else {
                // Recursively check each child node
                for (int i = 0; i < 8; ++i) {
                    if (pickPointNode_OriginDistance(node->children[i].get(), ray, closestDist, closestPointIndex)) {
                        found = true;
                    }
                }
            }

            return found;
        }

        bool rayIntersectsAABB(const Ray& ray, const Eigen::AlignedBox3f& aabb) {
            // Slab method for ray-AABB intersection
            float tmin = (aabb.min().x() - ray.origin.x()) / ray.direction.x();
            float tmax = (aabb.max().x() - ray.origin.x()) / ray.direction.x();

            if (tmin > tmax) swap(tmin, tmax);

            float tymin = (aabb.min().y() - ray.origin.y()) / ray.direction.y();
            float tymax = (aabb.max().y() - ray.origin.y()) / ray.direction.y();

            if (tymin > tymax) swap(tymin, tymax);

            if ((tmin > tymax) || (tymin > tmax)) return false;

            if (tymin > tmin) tmin = tymin;
            if (tymax < tmax) tmax = tymax;

            float tzmin = (aabb.min().z() - ray.origin.z()) / ray.direction.z();
            float tzmax = (aabb.max().z() - ray.origin.z()) / ray.direction.z();

            if (tzmin > tzmax) swap(tzmin, tzmax);

            if ((tmin > tzmax) || (tzmin > tmax)) return false;

            return true;
        }

        void searchPointsNearRayNode(OctreeNode* node, const Ray& ray, float maxDistance, vector<size_t>& result) {
            if (node == nullptr) {
                return;
            }

            // Check if the ray intersects with an expanded version of the node's AABB
            if (!rayIntersectsExpandedAABB(ray, node->aabb, maxDistance)) {
                return;
            }

            // If the node is a leaf, check each point
            if (node->depth == maxDepth) {
                for (size_t pointIndex : node->pointIndices) {
                    Eigen::Vector3f point = getPointFromIndex(pointIndex);
                    float dist = distanceToRay(ray, point);
                    float dot = ray.direction.dot(normals[pointIndex]);
                    if (dist <= maxDistance && 0 > dot) {
                        result.push_back(pointIndex);
                    }
                }
            }
            else {
                // Recursively check each child node
                for (int i = 0; i < 8; ++i) {
                    searchPointsNearRayNode(node->children[i].get(), ray, maxDistance, result);
                }
            }
        }

        bool rayIntersectsExpandedAABB(const Ray& ray, const Eigen::AlignedBox3f& aabb, float maxDistance) {
            // Expanding the AABB by maxDistance
            Eigen::AlignedBox3f expandedAABB(aabb.min() - Eigen::Vector3f(maxDistance, maxDistance, maxDistance),
                aabb.max() + Eigen::Vector3f(maxDistance, maxDistance, maxDistance));

            // Slab method for ray-AABB intersection
            float tmin = (expandedAABB.min().x() - ray.origin.x()) / ray.direction.x();
            float tmax = (expandedAABB.max().x() - ray.origin.x()) / ray.direction.x();

            if (tmin > tmax) swap(tmin, tmax);

            float tymin = (expandedAABB.min().y() - ray.origin.y()) / ray.direction.y();
            float tymax = (expandedAABB.max().y() - ray.origin.y()) / ray.direction.y();

            if (tymin > tymax) swap(tymin, tymax);

            if ((tmin > tymax) || (tymin > tmax)) return false;

            if (tymin > tmin) tmin = tymin;
            if (tymax < tmax) tmax = tymax;

            float tzmin = (expandedAABB.min().z() - ray.origin.z()) / ray.direction.z();
            float tzmax = (expandedAABB.max().z() - ray.origin.z()) / ray.direction.z();

            if (tzmin > tzmax) swap(tzmin, tzmax);

            if ((tmin > tzmax) || (tzmin > tmax)) return false;

            return true;
        }

        Eigen::Vector3f getPointFromIndex(size_t index) const {
            return points[index];
        }

        // Recursive function to perform radius search on each node
        void radiusSearchNode(OctreeNode* node, const Eigen::Vector3f& queryPoint, float radius, vector<size_t>& result) {
            if (node == nullptr) {
                return;
            }

            // Check if the search sphere intersects the node's AABB
            if (!sphereIntersectsAABB(queryPoint, radius, node->aabb)) {
                return;
            }

            // If it's a leaf node, check all points within it
            if (node->depth == maxDepth) {
                for (size_t pointIndex : node->pointIndices) {
                    Eigen::Vector3f point = points[pointIndex];
                    if ((point - queryPoint).squaredNorm() <= radius * radius) {
                        result.push_back(pointIndex);
                    }
                }
            }
            else {
                // Recursively check each child node
                for (int i = 0; i < 8; ++i) {
                    radiusSearchNode(node->children[i].get(), queryPoint, radius, result);
                }
            }
        }

        // Utility function to check if a sphere intersects an AABB
        bool sphereIntersectsAABB(const Eigen::Vector3f& center, float radius, const Eigen::AlignedBox3f& aabb) {
            // Calculate the closest point on the AABB to the sphere's center manually
            Eigen::Vector3f closestPoint;
            for (int i = 0; i < 3; ++i) {
                closestPoint[i] = max(aabb.min()[i], min(center[i], aabb.max()[i]));
            }

            // Calculate the distance from the sphere's center to this closest point
            float distanceSquared = (closestPoint - center).squaredNorm();

            // The sphere intersects the AABB if this distance is less than or equal to the radius squared
            return distanceSquared <= radius * radius;
        }

        void traverseNode(OctreeNode* node, const function<void(OctreeNode*)>& f) {
            if (!node) return;

            // Apply the function to the current node
            f(node);

            // Recursively traverse children nodes if they exist
            for (int i = 0; i < 8; ++i) {
                traverseNode(node->children[i].get(), f);
            }
        }
    };
}
