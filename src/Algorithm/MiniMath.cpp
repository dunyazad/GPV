#include "MiniMath.h"
#include <sstream>
#include <algorithm>

namespace MiniMath
{
	float clamp(float f, float minf, float maxf)
	{
		return min(max(f, minf), maxf);
	}

	float Trimax(float a, float b, float c) {
		return std::max(std::max(a, b), c);
	}

	float Trimin(float a, float b, float c) {
		return std::min(std::min(a, b), c);
	}

	V2::V2()
		: x(0.0f), y(0.0f) {}
	V2::V2(int ix, int iy)
		: x(ix), y(iy) {}
	V2::V2(float fx, float fy)
		: x(fx), y(fy) {}
	V2::V2(double dx, double dy)
		: x(dx), y(dy) {}
	V2::V2(float* fs)
		: x(fs[0]), y(fs[1]) {}

	V2::V2(const char* c)
	{
		string code(c);
		std::istringstream iss(code);
		float value;
		iss >> std::noskipws >> value;

		if (iss.eof() && !iss.fail())
		{
			x = y = value;
		}
		else
		{
			std::transform(code.begin(), code.end(), code.begin(), [](unsigned char c) { return std::tolower(c); });

			if (code == "nan")
			{
				x = y = std::numeric_limits<float>::quiet_NaN();
			}
			else if (code == "zero")
			{
				x = y = 0.0;
			}
			else if (code == "one")
			{
				x = y = 1.0;
			}
			else if (code == "half")
			{
				x = y = 0.5;
			}
		}
	}

	V3::V3()
		: x(0.0f), y(0.0f), z(0.0f) {}
	V3::V3(int ix, int iy, int iz)
		: x(ix), y(iy), z(iz) {}
	V3::V3(float fx, float fy, float fz)
		: x(fx), y(fy), z(fz) {}
	V3::V3(double dx, double dy, double dz)
		: x(dx), y(dy), z(dz) {}
	V3::V3(float* fs)
		: x(fs[0]), y(fs[1]), z(fs[2]) {}
	V3::V3(const char* c)
	{
		string code(c);
		std::istringstream iss(code);
		float value;
		iss >> std::noskipws >> value;

		if (iss.eof() && !iss.fail())
		{
			x = y = z = value;
		}
		else
		{
			std::transform(code.begin(), code.end(), code.begin(), [](unsigned char c) { return std::tolower(c); });

			if (code == "nan")
			{
				x = y = std::numeric_limits<float>::quiet_NaN();
			}
			else if (code == "zero")
			{
				x = y = 0.0;
			}
			else if (code == "one")
			{
				x = y = 1.0;
			}
			else if (code == "half")
			{
				x = y = 0.5;
			}
			else if (code == "red")
			{
				x = 1.0f;
			}
			else if (code == "green")
			{
				y = 1.0f;
			}
			else if (code == "blue")
			{
				z = 1.0f;
			}
			else if (code == "black")
			{
				x = y = z = 0.0f;
			}
			else if (code == "gray")
			{
				x = y = z = 0.5f;
			}
			else if (code == "white")
			{
				x = y = z = 1.0f;
			}
			else if (code == "yellow")
			{
				x = y = 1.0f;
				z = 0.0f;
			}
			else if (code == "magenta")
			{
				x = z = 1.0f;
				y = 0.0f;
			}
			else if (code == "cyan")
			{
				y = z = 1.0f;
				x = 0.0f;
			}
			else if (code == "positive x")
			{
				x = 1.0f;
				y = z = 0.0f;
			}
			else if (code == "negative x")
			{
				x = -1.0f;
				y = z = 0.0f;
			}
			else if (code == "positive y")
			{
				y = 1.0f;
				x = z = 0.0f;
			}
			else if (code == "negative y")
			{
				y = -1.0f;
				x = z = 0.0f;
			}
			else if (code == "positive z")
			{
				z = 1.0f;
				x = y = 0.0f;
			}
			else if (code == "negative z")
			{
				z = -1.0f;
				x = y = 0.0f;
			}
		}
	}


	const V3& V3::operator += (float scalar) { x += scalar; y += scalar; z += scalar; return *this; }
	const V3& V3::operator -= (float scalar) { x -= scalar; y -= scalar; z -= scalar; return *this; }
	const V3& V3::operator *= (float scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
	const V3& V3::operator /= (float scalar) { x /= scalar; y /= scalar; z /= scalar; return *this; }

	const V3& V3::operator += (const V3& other) { x += other.x; y += other.y; z += other.z; return *this; }
	const V3& V3::operator -= (const V3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }

	float V3::operator[](int index) { return *(&this->x + index); }

	V3 operator - (const V3& v) { return { -v.x, -v.y, -v.z }; }
	V3 operator + (const V3& v, float scalar) { return { v.x + scalar, v.y + scalar, v.z + scalar }; }
	V3 operator + (float scalar, const V3& v) { return { v.x + scalar, v.y + scalar, v.z + scalar }; }
	V3 operator - (const V3& v, float scalar) { return { v.x - scalar, v.y - scalar, v.z - scalar }; }
	V3 operator - (float scalar, const V3& v) { return { v.x - scalar, v.y - scalar, v.z - scalar }; }
	V3 operator * (const V3& v, float scalar) { return { v.x * scalar, v.y * scalar, v.z * scalar }; }
	V3 operator * (float scalar, const V3& v) { return { v.x * scalar, v.y * scalar, v.z * scalar }; }
	V3 operator / (const V3& v, float scalar) { return { v.x / scalar, v.y / scalar, v.z / scalar }; }
	V3 operator / (float scalar, const V3& v) { return { v.x / scalar, v.y / scalar, v.z / scalar }; }

	V3 operator + (const V3& a, const V3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
	V3 operator - (const V3& a, const V3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }

	float magnitude(const V3& v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
	V3 normalize(const V3& v)
	{
		float mag = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
		if (mag != 0.0f)
		{
			return { v.x / mag, v.y / mag, v.z / mag };
		}
		return v;
	}

	float dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
	V3 cross(const V3& a, const V3& b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

	float distance(const V3& a, const V3& b)
	{
		return sqrtf((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y) + (b.z - a.z) * (b.z - a.z));
	}

	float angle(const V3& a, const V3& b)
	{
		return acosf(clamp(dot(a, b), -1.0f, 1.0f));
	}

	V3 calculateMean(const std::vector<V3>& vectors)
	{
		V3 mean;
		for (const V3& vec : vectors)
		{
			mean.x += vec.x;
			mean.y += vec.y;
			mean.z += vec.z;
		}
		mean.x /= vectors.size();
		mean.y /= vectors.size();
		mean.z /= vectors.size();
		return mean;
	}

	void centerData(std::vector<V3>& vectors, const V3& mean)
	{
		for (V3& vec : vectors)
		{
			vec.x -= mean.x;
			vec.y -= mean.y;
			vec.z -= mean.z;
		}
	}

	void calculateCovarianceMatrix(const std::vector<V3>& vectors, V3& eigenvalues, V3& eigenvector1, V3& eigenvector2, V3& eigenvector3)
	{
		int n = vectors.size();

		for (const V3& vec : vectors)
		{
			eigenvalues.x += vec.x * vec.x;
			eigenvalues.y += vec.y * vec.y;
			eigenvalues.z += vec.z * vec.z;

			eigenvector1.x += vec.x * vec.x;
			eigenvector2.x += vec.x * vec.y;
			eigenvector3.x += vec.x * vec.z;

			eigenvector2.y += vec.y * vec.y;
			eigenvector3.y += vec.y * vec.z;

			eigenvector3.z += vec.z * vec.z;
		}

		eigenvalues.x /= n;
		eigenvalues.y /= n;
		eigenvalues.z /= n;

		eigenvector1.x /= n;
		eigenvector2.x /= n;
		eigenvector3.x /= n;

		eigenvector2.y /= n;
		eigenvector3.y /= n;

		eigenvector3.z /= n;
	}

	Quaternion::Quaternion(float scalar, float i, float j, float k)
		: w(scalar), x(i), y(j), z(k) {}

	Quaternion::Quaternion(float radian, const V3& axis)
	{
		float halfAngle = radian * 0.5f;
		float sinHalfAngle = sinf(halfAngle);
		w = cosf(halfAngle);
		x = axis.x * sinHalfAngle;
		y = axis.y * sinHalfAngle;
		z = axis.z * sinHalfAngle;
	}

	Quaternion conjugate(const Quaternion q) { return { q.w, -q.x, -q.y, -q.z }; }

	Quaternion operator*(const Quaternion& q1, const Quaternion& q2)
	{
		float w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
		float x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
		float y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
		float z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;

		return Quaternion(w, x, y, z);
	}

	V3 rotate(const V3& v, const Quaternion& q)
	{
		auto qv = V3{ q.x, q.y, q.z };
		auto uv = cross(qv, v);
		auto uuv = cross(qv, uv);

		return v + ((uv * q.w) + uuv) * 2.0f;
	}

	bool AABB::IntersectsTriangle(const V3& tp0, const V3& tp1, const V3& tp2)
	{
		V3 v0 = { tp0.x - center.x, tp0.y - center.y, tp0.z - center.z };
		V3 v1 = { tp1.x - center.x, tp1.y - center.y, tp1.z - center.z };
		V3 v2 = { tp2.x - center.x, tp2.y - center.y, tp2.z - center.z };

		// Compute edge vectors for triangle
		V3 f0 = { tp1.x - tp0.x, tp1.y - tp0.y, tp1.z - tp0.z };
		V3 f1 = { tp2.x - tp1.x, tp2.y - tp1.y, tp2.z - tp1.z };
		V3 f2 = { tp0.x - tp2.x, tp0.y - tp2.y, tp0.z - tp2.z };

		//// region Test axes a00..a22 (category 3)

		// Test axis a00
		V3 a00 = { 0.0f, -f0.z, f0.y };
		float p0 = v0.x * a00.x + v0.y * a00.y + v0.z * a00.z;
		float p1 = v1.x * a00.x + v1.y * a00.y + v1.z * a00.z;
		float p2 = v2.x * a00.x + v2.y * a00.y + v2.z * a00.z;
		float r = extents.y * std::fabs(f0.z) + extents.z * std::fabs(f0.y);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a01
		V3 a01 = { 0.0f, -f1.z, f1.y };
		p0 = v0.x * a01.x + v0.y * a01.y + v0.z * a01.z;
		p1 = v1.x * a01.x + v1.y * a01.y + v1.z * a01.z;
		p2 = v2.x * a01.x + v2.y * a01.y + v2.z * a01.z;
		r = extents.y * std::fabs(f1.z) + extents.z * std::fabs(f1.y);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a02
		V3 a02 = { 0.0f, -f2.z, f2.y };
		p0 = v0.x * a02.x + v0.y * a02.y + v0.z * a02.z;
		p1 = v1.x * a02.x + v1.y * a02.y + v1.z * a02.z;
		p2 = v2.x * a02.x + v2.y * a02.y + v2.z * a02.z;
		r = extents.y * std::fabs(f2.z) + extents.z * std::fabs(f2.y);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a10
		V3 a10 = { f0.z, 0.0f, -f0.x };
		p0 = v0.x * a10.x + v0.y * a10.y + v0.z * a10.z;
		p1 = v1.x * a10.x + v1.y * a10.y + v1.z * a10.z;
		p2 = v2.x * a10.x + v2.y * a10.y + v2.z * a10.z;
		r = extents.x * std::fabs(f0.z) + extents.z * std::fabs(f0.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a11
		V3 a11 = { f1.z, 0.0f, -f1.x };
		p0 = v0.x * a11.x + v0.y * a11.y + v0.z * a11.z;
		p1 = v1.x * a11.x + v1.y * a11.y + v1.z * a11.z;
		p2 = v2.x * a11.x + v2.y * a11.y + v2.z * a11.z;
		r = extents.x * std::fabs(f1.z) + extents.z * std::fabs(f1.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a12
		V3 a12 = { f2.z, 0.0f, -f2.x };
		p0 = v0.x * a12.x + v0.y * a12.y + v0.z * a12.z;
		p1 = v1.x * a12.x + v1.y * a12.y + v1.z * a12.z;
		p2 = v2.x * a12.x + v2.y * a12.y + v2.z * a12.z;
		r = extents.x * std::fabs(f2.z) + extents.z * std::fabs(f2.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a20
		V3 a20 = { -f0.y, f0.x, 0.0f };
		p0 = v0.x * a20.x + v0.y * a20.y + v0.z * a20.z;
		p1 = v1.x * a20.x + v1.y * a20.y + v1.z * a20.z;
		p2 = v2.x * a20.x + v2.y * a20.y + v2.z * a20.z;
		r = extents.x * std::fabs(f0.y) + extents.y * std::fabs(f0.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a21
		V3 a21 = { -f1.y, f1.x, 0.0f };
		p0 = v0.x * a21.x + v0.y * a21.y + v0.z * a21.z;
		p1 = v1.x * a21.x + v1.y * a21.y + v1.z * a21.z;
		p2 = v2.x * a21.x + v2.y * a21.y + v2.z * a21.z;
		r = extents.x * std::fabs(f1.y) + extents.y * std::fabs(f1.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		// Test axis a22
		V3 a22 = { -f2.y, f2.x, 0.0f };
		p0 = v0.x * a22.x + v0.y * a22.y + v0.z * a22.z;
		p1 = v1.x * a22.x + v1.y * a22.y + v1.z * a22.z;
		p2 = v2.x * a22.x + v2.y * a22.y + v2.z * a22.z;
		r = extents.x * std::fabs(f2.y) + extents.y * std::fabs(f2.x);
		if (std::max(-Trimax(p0, p1, p2), Trimin(p0, p1, p2)) > r + FLT_EPSILON)
			return false;

		//// endregion

		//// region Test the three axes corresponding to the face normals of AABB b (category 1)

		// Exit if...
		// ... [-extents.X, extents.X] and [Min(v0.X,v1.X,v2.X), Max(v0.X,v1.X,v2.X)] do not overlap
		if (Trimax(v0.x, v1.x, v2.x) < -extents.x || Trimin(v0.x, v1.x, v2.x) > extents.x) {
			if (Trimax(v0.x, v1.x, v2.x) - (-extents.x) > FLT_EPSILON ||
				Trimin(v0.x, v1.x, v2.x) - (extents.x) > FLT_EPSILON) {
				return false;
			}
		}

		// ... [-extents.Y, extents.Y] and [Min(v0.Y,v1.Y,v2.Y), Max(v0.Y,v1.Y,v2.Y)] do not overlap
		if (Trimax(v0.y, v1.y, v2.y) < -extents.y || Trimin(v0.y, v1.y, v2.y) > extents.y) {
			if (Trimax(v0.y, v1.y, v2.y) - (-extents.y) > FLT_EPSILON ||
				Trimin(v0.y, v1.y, v2.y) - (extents.y) > FLT_EPSILON) {
				return false;
			}
		}

		// ... [-extents.Z, extents.Z] and [Min(v0.Z,v1.Z,v2.Z), Max(v0.Z,v1.Z,v2.Z)] do not overlap
		if (Trimax(v0.z, v1.z, v2.z) < -extents.z || Trimin(v0.z, v1.z, v2.z) > extents.z) {
			if (Trimax(v0.z, v1.z, v2.z) - (-extents.z) > FLT_EPSILON ||
				Trimin(v0.z, v1.z, v2.z) - (extents.z) > FLT_EPSILON) {
				return false;
			}
		}

		//// endregion

		//// region Test separating axis corresponding to triangle face normal (category 2)

		V3 plane_normal = { f0.y * f1.z - f0.z * f1.y, f0.z * f1.x - f0.x * f1.z, f0.x * f1.y - f0.y * f1.x };
		float plane_distance = std::fabs(plane_normal.x * v0.x + plane_normal.y * v0.y + plane_normal.z * v0.z);

		// Compute the projection interval radius of b onto L(t) = b.c + t * p.n
		r = extents.x * std::fabs(plane_normal.x) + extents.y * std::fabs(plane_normal.y) + extents.z * std::fabs(plane_normal.z);

		// Intersection occurs when plane distance falls within [-r,+r] interval
		if (plane_distance > r + FLT_EPSILON)
			return false;

		//// endregion

		return true;
	}

	V3 LinePlaneIntersection(const V3& l0, const V3& l1, const V3& planePosition, const V3& planeNormal)
	{
		auto ld = l1 - l0;

		float denominator = dot(ld, planeNormal);

		if (denominator == 0.0f) {
			std::cout << "The line is parallel to the plane." << std::endl;
			// You might want to handle this case differently based on your requirements
			return "nan";
		}

		auto t = ((planePosition.x - l0.x) * planeNormal.x +
			(planePosition.y - l0.y) * planeNormal.y +
			(planePosition.z - l0.z) * planeNormal.z) / denominator;

		return V3(l0.x + t * ld.x, l0.y + t * ld.y, l0.z + t * ld.z);
	}
}
