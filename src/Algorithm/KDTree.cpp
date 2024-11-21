#include <Algorithm/KDTree.h>

namespace Algorithm
{
	//float RayPointDistanceSquared(const Ray& ray, const float* point)
	//{
	//	// Ray equation: P(t) = origin + t * direction
	//	// Closest point: (point - origin) dot direction
	//	float diff[3] = {
	//		point[0] - ray.origin[0],
	//		point[1] - ray.origin[1],
	//		point[2] - ray.origin[2]
	//	};

	//	// Project onto ray direction
	//	float t = diff[0] * ray.direction[0] + diff[1] * ray.direction[1] + diff[2] * ray.direction[2];

	//	t = std::max(0.0f, t); // Clamp t >= 0

	//	// Closest point on the ray
	//	float closestPoint[3] = {
	//		ray.origin[0] + t * ray.direction[0],
	//		ray.origin[1] + t * ray.direction[1],
	//		ray.origin[2] + t * ray.direction[2]
	//	};

	//	// Distance from closest point to the input point
	//	float dx = closestPoint[0] - point[0];
	//	float dy = closestPoint[1] - point[1];
	//	float dz = closestPoint[2] - point[2];

	//	return dx * dx + dy * dy + dz * dz; // Squared distance
	//}

	//float RayPointDistanceSquared(const Ray& ray, const float* point) {
	//	float v[3] = {
	//		point[0] - ray.origin[0],
	//		point[1] - ray.origin[1],
	//		point[2] - ray.origin[2]
	//	};

	//	float t = v[0] * ray.direction[0] + v[1] * ray.direction[1] + v[2] * ray.direction[2];

	//	float proj[3] = {
	//		ray.origin[0] + t * ray.direction[0],
	//		ray.origin[1] + t * ray.direction[1],
	//		ray.origin[2] + t * ray.direction[2]
	//	};

	//	float dSquared = (proj[0] - point[0]) * (proj[0] - point[0]) +
	//		(proj[1] - point[1]) * (proj[1] - point[1]) +
	//		(proj[2] - point[2]) * (proj[2] - point[2]);

	//	return dSquared;
	//}

	float RayPointDistanceSquared(const Ray& ray, const float* point) {
		// Calculate vector from ray origin to point
		float v[3] = {
			point[0] - ray.origin[0],
			point[1] - ray.origin[1],
			point[2] - ray.origin[2]
		};

		// Project v onto the ray direction
		float t = v[0] * ray.direction[0] + v[1] * ray.direction[1] + v[2] * ray.direction[2];

		// Find the projection point on the ray
		float proj[3] = {
			ray.origin[0] + t * ray.direction[0],
			ray.origin[1] + t * ray.direction[1],
			ray.origin[2] + t * ray.direction[2]
		};

		// Calculate squared distance from the point to the projection point on the ray
		float dSquared = (proj[0] - point[0]) * (proj[0] - point[0]) +
			(proj[1] - point[1]) * (proj[1] - point[1]) +
			(proj[2] - point[2]) * (proj[2] - point[2]);

		return dSquared;
	}


	float GetDistanceSquared(const float* points, unsigned int point_index, const float* query)
	{
		float distanceSquared = 0.0f;
		for (int i = 0; i < 3; ++i) {
			float diff = points[point_index * 3 + i] - query[i];
			distanceSquared += diff * diff;
		}
		return distanceSquared;
	}

	float GetDistance(const float* points, unsigned int point_index, const float* query)
	{
		auto dx = points[point_index * 3] - query[0];
		auto dy = points[point_index * 3 + 1] - query[1];
		auto dz = points[point_index * 3 + 2] - query[2];
		return sqrtf(dx * dx + dy * dy + dz * dz);
	}

	Eigen::AlignedBox3f TransformAABB(const Eigen::AlignedBox3f& aabb, const Eigen::Matrix4f& transform)
	{
		Eigen::AlignedBox3f transformedAABB(
			Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
		for (size_t i = 0; i < 8; i++)
		{
			const auto& corner = aabb.corner((Eigen::AlignedBox3f::CornerType)i);
			Eigen::Vector4f homogenousCorner(corner.x(), corner.y(), corner.z(), 1.0f);
			Eigen::Vector4f transformedCorner = transform * homogenousCorner;
			transformedAABB.extend(transformedCorner.head<3>());
		}

		return transformedAABB;
	}

	bool Intersects(const Eigen::AlignedBox3f& a, const Eigen::AlignedBox3f& b, const Eigen::Matrix4f& transformB)
	{
		return a.intersects(TransformAABB(b, transformB));
	}

}
