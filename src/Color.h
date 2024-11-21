#pragma once

#include <Common.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

class Color3
{
public:
	Color3(unsigned char r, unsigned char g, unsigned char b);
	
	void FromNormalzed(float r, float g, float b);

	operator Eigen::Vector3f() const
	{
		return Eigen::Vector3f(
			(float)components[0] / 255.0f,
			(float)components[1] / 255.0f,
			(float)components[2] / 255.0f);
	}

	operator Eigen::Vector4f() const
	{
		return Eigen::Vector4f(
			(float)components[0] / 255.0f,
			(float)components[1] / 255.0f,
			(float)components[2] / 255.0f,
			1.0f);
	}

	inline unsigned char* data() { return components; }
	inline unsigned char& x() { return components[0]; }
	inline unsigned char& y() { return components[1]; }
	inline unsigned char& z() { return components[2]; }

	static const Color3 Black;
	static const Color3 Red;
	static const Color3 Green;
	static const Color3 Blue;
	static const Color3 Yellow;
	static const Color3 Cyan;
	static const Color3 Magenta;
	static const Color3 White;

	static Color3 Random();

private:
	unsigned char components[3] = { 255, 255, 255 };
};

class Color4
{
public:
	Color4(unsigned char r, unsigned char g, unsigned char b, unsigned char a);

	void FromNormalzed(float r, float g, float b, float a);

	operator Eigen::Vector3f() const
    {
        return Eigen::Vector3f(
            (float)components[0] / 255.0f,
            (float)components[1] / 255.0f,
            (float)components[2] / 255.0f);
    }

	operator Eigen::Vector4f() const
	{
		return Eigen::Vector4f(
			(float)components[0] / 255.0f,
			(float)components[1] / 255.0f,
			(float)components[2] / 255.0f,
			(float)components[3] / 255.0f);
	}

	inline unsigned char* data() { return components; }
	inline unsigned char& x() { return components[0]; }
	inline unsigned char& y() { return components[1]; }
	inline unsigned char& z() { return components[2]; }
	inline unsigned char& a() { return components[3]; }

	static const Color4 Black;
	static const Color4 Red;
	static const Color4 Green;
	static const Color4 Blue;
	static const Color4 Yellow;
	static const Color4 Cyan;
	static const Color4 Magenta;
	static const Color4 White;

	static Color4 Random();

private:
	unsigned char components[4] = { 255, 255, 255, 255 };
};
