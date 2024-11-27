#pragma once

#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <tuple>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

//#define FLT_VALID(x) ((x) < FLT_MAX / 2)
#define FLT_VALID(x) ((x) < 3.402823466e+36F)
#define VECTOR3F_VALID(v) (FLT_VALID((v).x) && FLT_VALID((v).y) && FLT_VALID((v).z))
#define VECTOR3F_VALID_(v) (FLT_VALID((v).x()) && FLT_VALID((v).y()) && FLT_VALID((v).z()))
#define SHORT_VALID(x) ((x) != SHRT_MAX)
#define USHORT_VALID(x) ((x) != USHRT_MAX)
#define INT_VALID(x) ((x) != INT_MAX)
#define UINT_VALID(x) ((x) != UINT_MAX)
#define VECTOR3U_VALID(v) (UINT_VALID((v).x) && UINT_VALID((v).y) && UINT_VALID((v).z))
#define VECTOR3U_VALID_(v) (UINT_VALID((v).x()) && UINT_VALID((v).y()) && UINT_VALID((v).z()))

//#define USE_CHAR_AS_VOXEL_VALUE
//#define USE_SHORT_AS_VOXEL_VALUE
//#define USE_INT_AS_VOXEL_VALUE
#define USE_FLOAT_AS_VOXEL_VALUE

#ifdef USE_FLOAT_AS_VOXEL_VALUE
typedef float voxel_value_t;
#define VOXEL_INVALID FLT_MAX
#define VV2D(x) (x)
#define D2VV(x) (x)
#elif defined USE_CHAR_AS_VOXEL_VALUE
typedef char voxel_value_t;
#define VOXEL_INVALID -128
#define VV2D(x) ((x) == VOXEL_INVALID ? FLT_MAX : ((float)(x) / 100.0f))
#define D2VV(x) (FLT_VALID(x) ? ((voxel_value_t)((float)(x) * 100.0f )) : VOXEL_INVALID)
#elif defined USE_SHORT_AS_VOXEL_VALUE
typedef short voxel_value_t;
#define VOXEL_INVALID SHRT_MAX
#define VV2D(x) ((x) == VOXEL_INVALID ? FLT_MAX : ((float)(x) / 100.0f))
#define D2VV(x) (FLT_VALID(x) ? ((voxel_value_t)((float)(x) * 100.0f)) : VOXEL_INVALID)
#elif defined USE_INT_AS_VOXEL_VALUE
typedef int voxel_value_t;
#define VOXEL_INVALID INT_MAX
#define VV2D(x) ((x) == VOXEL_INVALID ? FLT_MAX : ((float)(x) / 100.0f))
#define D2VV(x) (FLT_VALID(x) ? ((voxel_value_t)((float)(x) * 100.0f)) : VOXEL_INVALID)
#endif

#define VOXELCNT_VALUE(x) (x)
#define VOXELCNT_TOOTH(x) (x)
#define VOXELCNT_ISTOOTH(x) (VOXELCNT_TOOTH(x) == 0)
#define VOXELCNT_NONTOOTH(x) (x)
#define VOXELCNT_ADD(x, y) ((x) = (x) + (y))
#define VOXELCNT_REP(x, y) ((x) = (y))

#define VOXELCNT_VALUE_OLD(x) ((x) & 0x7fff)
#define VOXELCNT_TOOTH_OLD(x) ((x) & 0x8000)
#define VOXELCNT_ISTOOTH_OLD(x) (VOXELCNT_TOOTH_OLD(x) == 0)
#define VOXELCNT_NONTOOTH_OLD(x) ((x) | 0x8000)
#define VOXELCNT_ADD_OLD(x, y) ((x) = (x) + (y))
#define VOXELCNT_REP_OLD(x, y) ((x) = (VOXELCNT_TOOTH_OLD(x)|VOXELCNT_VALUE_OLD(y)))

inline int safe_stoi(const string& input)
{
	if (input.empty())
	{
		return INT_MAX;
	}
	else
	{
		return stoi(input);
	}
}

inline float safe_stof(const string& input)
{
	if (input.empty())
	{
		return FLT_MAX;
	}
	else
	{
		return stof(input);
	}
}

inline vector<string> split(const string& input, const string& delimiters, bool includeEmptyString = false)
{
	vector<string> result;
	string piece;
	for (auto c : input)
	{
		bool contains = false;
		for (auto d : delimiters)
		{
			if (d == c)
			{
				contains = true;
				break;
			}
		}

		if (contains == false)
		{
			piece += c;
		}
		else
		{
			if (includeEmptyString || piece.empty() == false)
			{
				result.push_back(piece);
				piece.clear();
			}
		}
	}
	if (piece.empty() == false)
	{
		result.push_back(piece);
	}

	return result;
}

inline void ParseOneLine(
	const string& line,
	vector<float>& vertices,
	vector<float>& uvs,
	vector<float>& vertex_normals,
	vector<float>& vertex_colors,
	vector<uint32_t>& faces,
	float scaleX, float scaleY, float scaleZ)
{
	if (line.empty())
		return;

	auto words = split(line, " \t");

	if (words[0] == "v")
	{
		float x = safe_stof(words[1]) * scaleX;
		float y = safe_stof(words[2]) * scaleY;
		float z = safe_stof(words[3]) * scaleZ;
		vertices.push_back(x);
		vertices.push_back(y);
		vertices.push_back(z);

		if (words.size() > 3)
		{
			float r = safe_stof(words[4]);
			float g = safe_stof(words[5]);
			float b = safe_stof(words[6]);
			vertex_colors.push_back(r);
			vertex_colors.push_back(g);
			vertex_colors.push_back(b);
		}

	}
	else if (words[0] == "vt")
	{
		float u = safe_stof(words[1]);
		float v = safe_stof(words[2]);
		uvs.push_back(u);
		uvs.push_back(v);
	}
	else if (words[0] == "vn")
	{
		float x = safe_stof(words[1]);
		float y = safe_stof(words[2]);
		float z = safe_stof(words[3]);
		vertex_normals.push_back(x);
		vertex_normals.push_back(y);
		vertex_normals.push_back(z);
	}
	else if (words[0] == "f")
	{
		if (words.size() == 4)
		{
			auto fe0 = split(words[1], "/", true);
			auto fe1 = split(words[2], "/", true);
			auto fe2 = split(words[3], "/", true);

			if (fe0.size() == 1 && fe1.size() == 1 && fe2.size() == 1) {
				faces.push_back(safe_stoi(fe0[0]));
				//faces.push_back(INT_MAX);
				//faces.push_back(INT_MAX);
				faces.push_back(safe_stoi(fe1[0]));
				//faces.push_back(INT_MAX);
				//faces.push_back(INT_MAX);
				faces.push_back(safe_stoi(fe2[0]));
				//faces.push_back(INT_MAX);
				//faces.push_back(INT_MAX);
			}
			else {
				faces.push_back(safe_stoi(fe0[0]));
				//faces.push_back(safe_stoi(fe0[1]));
				//faces.push_back(safe_stoi(fe0[2]));
				faces.push_back(safe_stoi(fe1[0]));
				//faces.push_back(safe_stoi(fe1[1]));
				//faces.push_back(safe_stoi(fe1[2]));
				faces.push_back(safe_stoi(fe2[0]));
				//faces.push_back(safe_stoi(fe2[1]));
				//faces.push_back(safe_stoi(fe2[2]));
			}
		}
	}
}

class HSerializable
{
public:
	virtual bool Serialize(const string& filename) = 0;
	virtual bool Deserialize(const string& filename) = 0;

	virtual inline void AddPoint(float x, float y, float z)
	{
		points.push_back(x);
		points.push_back(y);
		points.push_back(z);

		if (FLT_VALID(x) && FLT_VALID(y) && FLT_VALID(z))
		{
			aabb.extend(Eigen::Vector3f(x, y, z));
		}
	}

	virtual inline void AddPoint(float x, float y, float z, float w)
	{
		points.push_back(x);
		points.push_back(y);
		points.push_back(z);
		points.push_back(w);

		if (FLT_VALID(x) && FLT_VALID(y) && FLT_VALID(z))
		{
			aabb.extend(Eigen::Vector3f(x, y, z));
		}
	}

	virtual inline void AddPointFloat3(const float* point)
	{
		points.push_back(point[0]);
		points.push_back(point[1]);
		points.push_back(point[2]);

		if (FLT_VALID(point[0]) && FLT_VALID(point[1]) && FLT_VALID(point[2]))
		{
			aabb.extend(Eigen::Vector3f(point[0], point[1], point[2]));
		}
	}

	virtual inline void AddPointFloat4(const float* point)
	{
		points.push_back(point[0]);
		points.push_back(point[1]);
		points.push_back(point[2]);
		points.push_back(point[3]);

		if (FLT_VALID(point[0]) && FLT_VALID(point[1]) && FLT_VALID(point[2]))
		{
			aabb.extend(Eigen::Vector3f(point[0], point[1], point[2]));
		}
	}

	inline const vector<float>& GetPoints() const { return points; }
	inline vector<float>& GetPoints() { return points; }

	inline const Eigen::AlignedBox3f GetAABB() const { return aabb; }

protected:
	vector<float> points;
	Eigen::AlignedBox3f aabb;
};

class XYZFormat : public HSerializable
{
public:
	virtual bool Serialize(const string& filename)
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err)
		{
			printf("[Serialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		fprintf(fp, "%d\n", (int)points.size() / 3);
		for (size_t i = 0; i < points.size() / 3; i++)
		{
			fprintf(fp, "%.6f %.6f %.6f\n", points[i * 3 + 0], points[i * 3 + 1], points[i * 3 + 2]);
		}

		fclose(fp);

		return true;
	}

	virtual bool Deserialize(const string& filename)
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err)
		{
			printf("[Deserialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		int size = 0;
		fscanf(fp, "%d\n", &size);

		for (size_t i = 0; i < size; i++)
		{
			float x, y, z;
			fscanf(fp, "%f %f %f\n", &x, &y, &z);

			points.push_back(x);
			points.push_back(y);
			points.push_back(z);
		}

		fclose(fp);

		return true;
	}
};

class OFFFormat : public HSerializable
{
public:
	virtual bool Serialize(const string& filename)
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err)
		{
			printf("[Serialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		fprintf(fp, "OFF\n");
		auto pointCount = 0 < points.size() ? points.size() / 3 : 0;
		auto indexCount = 0 < indices.size() ? indices.size() / 3 : 0;
		if (0 == indexCount && 0 < colors.size())
		{
			indexCount = colors.size() / 3;
		}
		fprintf(fp, "%d %d %d\n", pointCount, indexCount, 0);

		for (size_t i = 0; i < points.size() / 3; i++)
		{
			auto x = points[3 * i + 0];
			auto y = points[3 * i + 1];
			auto z = points[3 * i + 2];

			fprintf(fp, "%4.6f %4.6f %4.6f\n", x, y, z);

			if (0 == i % 10000)
			{
				auto percent = ((double)i / (double)(points.size() / 3)) * 100.0;
				printf("[%llu / %llu] %f percent\n", i, points.size() / 3, percent);
			}
		}

#pragma region To remove float precision error
		//for (size_t i = 0; i < points.size() / 3; i++)
		//{
		//	auto fx = floorf((points[3 * i + 0]) * 100.0f + 0.001f);
		//	auto fy = floorf((points[3 * i + 1]) * 100.0f + 0.001f);
		//	auto fz = floorf((points[3 * i + 2]) * 100.0f + 0.001f);

		//	int x = (int)fx;
		//	int y = (int)fy;
		//	int z = (int)fz;

		//	//printf("%7d %7d %7d\n", x, y, z);
		//	fprintf(fp, "%4.6f %4.6f %4.6f\n", points[3 * i + 0], points[3 * i + 1], points[3 * i + 2]);
		//	fprintf(fp, "%7d %7d %7d\n", x, y, z);

		//	if (0 == i % 10000)
		//	{
		//		auto percent = ((double)i / (double)(points.size() / 3)) * 100.0;
		//		printf("[%llu / %llu] %f percent\n", i, points.size() / 3, percent);
		//	}
		//}
#pragma endregion


		for (size_t i = 0; i < indices.size() / 3; i++)
		{
			auto i0 = indices[i * 3 + 0];
			auto i1 = indices[i * 3 + 1];
			auto i2 = indices[i * 3 + 2];

			if (colors.empty())
			{
				fprintf(fp, "3 %7d %7d %7d 255 255 255\n", i0, i1, i2);
			}
			else
			{
				auto red = unsigned char(colors[i0 * 3 + 0] * 255);
				auto green = unsigned char(colors[i0 * 3 + 1] * 255);
				auto blue = unsigned char(colors[i0 * 3 + 2] * 255);

				fprintf(fp, "3 %7d %7d %7d %3d %3d %3d\n", i0, i1, i2, red, green, blue);
			}
		}

		if (0 == indices.size() && 0 < colors.size())
		{
			for (size_t i = 0; i < colors.size() / 3; i++)
			{
				auto red = unsigned char(colors[i * 3 + 0] * 255);
				auto green = unsigned char(colors[i * 3 + 1] * 255);
				auto blue = unsigned char(colors[i * 3 + 2] * 255);

				fprintf(fp, "1 %3d %3d %3d\n", red, green, blue);
			}
		}

		fclose(fp);

		return true;
	}

	virtual bool Deserialize(const string& filename)
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err)
		{
			printf("[Deserialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		char buffer[1024];
		memset(buffer, 0, 1024);
		auto line = fgets(buffer, 1024, fp);
		if (0 != strcmp(line, "OFF\n"))
			return false;

		line = fgets(buffer, 1024, fp);
		while ('#' == line[0])
		{
			line = fgets(buffer, 1024, fp);
		}

		size_t vertexCount = 0;
		size_t triangleCount = 0;
		sscanf_s(line, "%d %d", &vertexCount, &triangleCount);

		printf("vertexCount : %d, triangleCount : %d\n", vertexCount, triangleCount);

		for (size_t i = 0; i <= vertexCount; i++)
		{
			line = fgets(buffer, 1024, fp);
			if (nullptr != line)
			{
				if ('#' == line[0])
				{
					i--;
					continue;
				}
				else
				{
					float x, y, z;
					sscanf_s(line, "%f %f %f\n", &x, &y, &z);

					AddPoint(x, y, z);
				}
			}
		}

		for (size_t i = 0; i < triangleCount; i++)
		{
			line = fgets(buffer, 1024, fp);
			if ('#' == line[0])
			{
				i--;
				continue;
			}
			else
			{
				size_t count, i0, i1, i2;
				sscanf_s(line, "%d %d %d %d\n", &count, &i0, &i1, &i2);

				AddIndex(i0);
				AddIndex(i1);
				AddIndex(i2);
			}
		}

		//return false;

		while (nullptr != line)
		{
			printf("%s", line);

			line = fgets(buffer, 1024, fp);
		}

		fclose(fp);

		return true;
	}

	inline const vector<unsigned int>& GetIndices() const { return indices; }
	inline const vector<float>& GetColors() const { return colors; }

	virtual inline void AddIndex(unsigned int index) { indices.push_back(index); }

	virtual inline void AddColor(float r, float g, float b)
	{
		colors.push_back(r);
		colors.push_back(g);
		colors.push_back(b);
	}

	virtual inline void SetColor(size_t index, float color) { if (index < colors.size() - 1) colors[index] = color; }

protected:
	vector<unsigned int> indices;
	vector<float> colors;
};

class CustomMeshFormat : public HSerializable
{
public:
	virtual bool Serialize(const string& filename)
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "wb");
		if (0 != err)
		{
			printf("[Serialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		if (0 < points.size())
		{
			fprintf_s(fp, "%ul\n", points.size() / 3);
			for (size_t i = 0; i < points.size() / 3; i++)
			{
				fprintf(fp, "%f, %f, %f\n", points[i * 3 + 0], points[i * 3 + 1], points[i * 3 + 2]);
			}
		}
		else
		{
			fprintf_s(fp, "%ul\n", 0);
		}

		if (0 < normals.size())
		{
			fprintf_s(fp, "%ul\n", normals.size() / 3);
			for (size_t i = 0; i < normals.size() / 3; i++)
			{
				fprintf(fp, "%f, %f, %f\n", normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);
			}
		}
		else
		{
			fprintf_s(fp, "%ul\n", 0);
		}

		if (0 < colors.size())
		{
			fprintf_s(fp, "%ul\n", colors.size() / 3);
			for (size_t i = 0; i < colors.size() / 3; i++)
			{
				fprintf(fp, "%f, %f, %f\n", colors[i * 3 + 0], colors[i * 3 + 1], colors[i * 3 + 2]);
			}
		}
		else
		{
			fprintf_s(fp, "%ul\n", 0);
		}

		if (0 < indices.size())
		{
			fprintf_s(fp, "%ul\n", indices.size() / 3);
			for (size_t i = 0; i < indices.size() / 3; i++)
			{
				fprintf(fp, "%d, %d, %d\n", indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2]);
			}
		}
		else
		{
			fprintf_s(fp, "%ul\n", 0);
		}

		fclose(fp);

		return true;
	}

	virtual bool Deserialize(const string& filename)
	{
		FILE* fp = nullptr;
		auto err = fopen_s(&fp, filename.c_str(), "rb");
		if (0 != err)
		{
			printf("[Deserialize] File \"%s\" open failed.", filename.c_str());
			return false;
		}

		char buffer[1024];
		while (nullptr != fgets(buffer, sizeof(buffer), fp))
		{
			size_t nop = 0;
			sscanf(buffer, "%ul\n", &nop);
			for (size_t i = 0; i < nop; i++)
			{
				fgets(buffer, sizeof(buffer), fp);
				if (nullptr == buffer) break;

				float x, y, z;
				sscanf(buffer, "%f, %f, %f\n", &x, &y, &z);
				AddPoint(x, y, z);
			}

			size_t non = 0;
			fgets(buffer, sizeof(buffer), fp);
			if (nullptr == buffer) break;
			sscanf(buffer, "%ul\n", &non);
			for (size_t i = 0; i < non; i++)
			{
				fgets(buffer, sizeof(buffer), fp);
				if (nullptr == buffer) break;

				float x, y, z;
				sscanf(buffer, "%f, %f, %f\n", &x, &y, &z);
				AddNormal(x, y, z);
			}

			size_t noc = 0;
			fgets(buffer, sizeof(buffer), fp);
			if (nullptr == buffer) break;
			sscanf(buffer, "%ul\n", &noc);
			for (size_t i = 0; i < noc; i++)
			{
				fgets(buffer, sizeof(buffer), fp);
				if (nullptr == buffer) break;

				float r, g, b;
				sscanf(buffer, "%f, %f, %f\n", &r, &g, &b);
				AddColor(r, g, b);
			}

			size_t noi = 0;
			fgets(buffer, sizeof(buffer), fp);
			if (nullptr == buffer) break;
			sscanf(buffer, "%ul\n", &noi);
			for (size_t i = 0; i < noc; i++)
			{
				fgets(buffer, sizeof(buffer), fp);
				if (nullptr == buffer) break;

				size_t i0, i1, i2;
				sscanf(buffer, "%ul, %ul, %ul\n", &i0, &i1, &i2);

				AddIndex(i0);
				AddIndex(i1);
				AddIndex(i2);
			}
		}

		return true;
	}

	inline const vector<unsigned int>& GetIndices() const { return indices; }
	inline const vector<float>& GetColors() const { return colors; }

	virtual inline void AddNormal(float x, float y, float z)
	{
		normals.push_back(x);
		normals.push_back(y);
		normals.push_back(z);
	}

	virtual inline void AddNormalFloat3(const float* normal)
	{
		normals.push_back(normal[0]);
		normals.push_back(normal[1]);
		normals.push_back(normal[2]);
	}

	virtual inline void AddIndex(unsigned int index) { indices.push_back(index); }

	virtual inline void AddColor(float r, float g, float b)
	{
		colors.push_back(r);
		colors.push_back(g);
		colors.push_back(b);
	}

	virtual inline void SetColor(size_t index, float color) { if (index < colors.size() - 1) colors[index] = color; }

protected:
	vector<float> normals;
	vector<unsigned int> indices;
	vector<float> colors;
};

class OBJFormat : public HSerializable
{
public:
	virtual bool Serialize(const string& filename)
	{
		ofstream ofs(filename);
		stringstream ss;
		ss.precision(6);

		ss << "# cuTSDF::ResourceIO::OBJ" << endl;
		for (size_t i = 0; i < points.size() / 3; i++)
		{
			auto x = points[3 * i + 0];
			auto y = points[3 * i + 1];
			auto z = points[3 * i + 2];

			if (colors.size() == 0)
			{
				ss << "v " << x << " " << y << " " << z << endl;
			}
			else if (colors.size() == points.size())
			{
				auto r = colors[3 * i + 0];
				auto g = colors[3 * i + 1];
				auto b = colors[3 * i + 2];

				ss << "v " << x << " " << y << " " << z << " " << r << " " << g << " " << b << endl;
			}

			if (normals.size() == points.size())
			{
				auto x = normals[3 * i + 0];
				auto y = normals[3 * i + 1];
				auto z = normals[3 * i + 2];

				ss << "vn " << x << " " << y << " " << z << endl;

				//printf("%f %f %f\n", x, y, z);
			}
		}

		for (size_t i = 0; i < uvs.size() / 2; i++)
		{
			auto u = uvs[2 * i + 0];
			auto v = uvs[2 * i + 1];

			ss << "vt " << u << " " << v << endl;
		}

		//for (size_t i = 0; i < normals.size() / 3; i++)
		//{
		//	auto x = normals[3 * i + 0];
		//	auto y = normals[3 * i + 1];
		//	auto z = normals[3 * i + 2];

		//	ss << "vn " << x << " " << y << " " << z << endl;
		//}

		bool has_uv = uvs.size() != 0;
		bool has_vn = normals.size() != 0;

		auto nof = indices.size() / 3;
		if (nof == 0)
		{
			//for (size_t i = 0; i < points.size(); i++)
			//{
			//	if (has_uv && has_vn)
			//	{
			//		ss << "f "
			//			<< i + 1 << "/" << i + 1 << "/" << i + 1 << " "
			//			<< i + 1 << "/" << i + 1 << "/" << i + 1 << " "
			//			<< i + 1 << "/" << i + 1 << "/" << i + 1 << endl;
			//	}
			//	else if (has_uv)
			//	{
			//		ss << "f "
			//			<< i + 1 << "/" << i + 1 << " "
			//			<< i + 1 << "/" << i + 1 << " "
			//			<< i + 1 << "/" << i + 1 << endl;
			//	}
			//	else if (has_vn)
			//	{
			//		ss << "f "
			//			<< i + 1 << "//" << i + 1 << " "
			//			<< i + 1 << "//" << i + 1 << " "
			//			<< i + 1 << "//" << i + 1 << endl;
			//	}
			//	else
			//	{
			//		ss << "f " << i + 1 << " " << i + 1 << " " << i + 1 << endl;
			//	}

			//	if (0 == i % 10000)
			//	{
			//		auto percent = ((double)i / (double)(points.size())) * 100.0;
			//		printf("[%llu / %llu] %f percent\n", i, points.size(), percent);
			//	}
			//}
		}
		else
		{
			for (size_t i = 0; i < nof; i++)
			{
				//uint32_t face[3] = { (uint32_t)i * 3 + 1, (uint32_t)i * 3 + 2, (uint32_t)i * 3 + 3 };
				uint32_t face[3] = { indices[i * 3 + 0],indices[i * 3 + 1],indices[i * 3 + 2] };

				if (has_uv && has_vn)
				{
					ss << "f "
						<< face[0] << "/" << face[0] << "/" << face[0] << " "
						<< face[1] << "/" << face[1] << "/" << face[1] << " "
						<< face[2] << "/" << face[2] << "/" << face[2] << endl;
				}
				else if (has_uv)
				{
					ss << "f "
						<< face[0] << "/" << face[0] << " "
						<< face[1] << "/" << face[1] << " "
						<< face[2] << "/" << face[2] << endl;
				}
				else if (has_vn)
				{
					ss << "f "
						<< face[0] << "//" << face[0] << " "
						<< face[1] << "//" << face[1] << " "
						<< face[2] << "//" << face[2] << endl;
				}
				else
				{
					ss << "f " << face[0] << " " << face[1] << " " << face[2] << endl;
				}

				if (0 == i % 10000)
				{
					auto percent = ((double)i / (double)(nof)) * 100.0;
					printf("[%llu / %llu] %f percent\n", i, nof, percent);
				}
			}
		}

		ofs << ss.rdbuf();
		ofs.close();

		return true;
	}

	virtual bool Deserialize(const string& filename)
	{
		ifstream ifs(filename);
		if (false == ifs.is_open())
		{
			printf("filename : %s is not open\n", filename.c_str());
			return false;
		}

		stringstream buffer;
		buffer << ifs.rdbuf();

		string line;
		while (buffer.good())
		{
			getline(buffer, line);
			ParseOneLine(line, points, uvs, normals, colors, indices, 1.0f, 1.0f, 1.0f);
		}

		return true;
	}

	inline const vector<float>& GetNormals() const { return normals; }
	inline const vector<unsigned int>& GetIndices() const { return indices; }
	inline const vector<float>& GetColors() const { return colors; }

	virtual inline void AddUV(float u, float v)
	{
		uvs.push_back(u);
		uvs.push_back(v);
	}

	virtual inline void AddUVFloat2(const float* uv)
	{
		uvs.push_back(uv[0]);
		uvs.push_back(uv[1]);
	}

	virtual inline void AddNormal(float x, float y, float z)
	{
		normals.push_back(x);
		normals.push_back(y);
		normals.push_back(z);
	}

	virtual inline void AddNormalFloat3(const float* normal)
	{
		normals.push_back(normal[0]);
		normals.push_back(normal[1]);
		normals.push_back(normal[2]);
	}

	virtual inline void AddIndex(unsigned int index) { indices.push_back(index); }

	virtual inline void AddColor(float r, float g, float b)
	{
		colors.push_back(r);
		colors.push_back(g);
		colors.push_back(b);
	}

	virtual inline void AddColorFloat3(const float* color)
	{
		colors.push_back(color[0]);
		colors.push_back(color[1]);
		colors.push_back(color[2]);
	}

	virtual inline void SetColor(size_t index, float color) { if (index < colors.size() - 1) colors[index] = color; }

protected:
	vector<float> uvs;
	vector<float> normals;
	vector<unsigned int> indices;
	vector<float> colors;
};

class PLYFormat : public HSerializable
{
public:
	virtual bool Serialize(const string& filename)
	{
		ofstream ofs(filename);
		stringstream ss;
		ss.precision(6);

		ss << "ply" << endl;
		ss << "format ascii 1.0" << endl;
		ss << "element vertex " << points.size() / 3 << endl;
		ss << "property float x" << endl;
		ss << "property float y" << endl;
		ss << "property float z" << endl;

		if (normals.size() == points.size())
		{
			ss << "property float nx" << endl;
			ss << "property float ny" << endl;
			ss << "property float nz" << endl;
		}
		if (colors.size() == points.size() || colors.size() / 4 == points.size() / 3)
		{
			ss << "property uchar red" << endl;
			ss << "property uchar green" << endl;
			ss << "property uchar blue" << endl;
			if (useAlpha)
			{
				ss << "property uchar alpha" << endl;
			}
		}
		if (uvs.size() == points.size())
		{
			ss << "property float u" << endl;
			ss << "property float v" << endl;
		}

		if (indices.size() > 0)
		{
			ss << "element face " << indices.size() / 3 << endl;
			ss << "property list uchar int vertex_indices" << endl;
		}

		ss << "end_header" << endl;

		for (size_t i = 0; i < points.size() / 3; i++)
		{
			auto x = points[3 * i + 0];
			auto y = points[3 * i + 1];
			auto z = points[3 * i + 2];

			ss << x << " " << y << " " << z << " ";

			if (normals.size() == points.size())
			{
				auto nx = normals[3 * i + 0];
				auto ny = normals[3 * i + 1];
				auto nz = normals[3 * i + 2];

				ss << nx << " " << ny << " " << nz << " ";
			}

			if (false == useAlpha)
			{
				if (colors.size() == points.size())
				{
					auto red = (unsigned char)(colors[3 * i + 0] * 255.0f);
					auto green = (unsigned char)(colors[3 * i + 1] * 255.0f);
					auto blue = (unsigned char)(colors[3 * i + 2] * 255.0f);

					ss << (int)red << " " << (int)green << " " << (int)blue << " ";
				}
			}
			else
			{
				if (colors.size() / 4 == points.size() / 3)
				{
					auto red = (unsigned char)(colors[4 * i + 0] * 255.0f);
					auto green = (unsigned char)(colors[4 * i + 1] * 255.0f);
					auto blue = (unsigned char)(colors[4 * i + 2] * 255.0f);
					auto alpha = (unsigned char)(colors[4 * i + 3] * 255.0f);

					ss << (int)red << " " << (int)green << " " << (int)blue << " " << (int)alpha << " ";
				}
			}

			if (uvs.size() == points.size())
			{
				auto u = uvs[3 * i + 0];
				auto v = uvs[3 * i + 1];

				ss << u << " " << v << " ";
			}

			ss << endl;

			if (0 == i % 10000 && i != 0)
			{
				auto percent = ((double)i / (double)(points.size() / 3)) * 100.0;
				printf("[%llu / %llu] %f percent\n", i, points.size(), percent);
			}
		}

		if (indices.size() > 0)
		{
			for (size_t i = 0; i < indices.size() / 3; i++)
			{
				auto i0 = indices[3 * i + 0];
				auto i1 = indices[3 * i + 1];
				auto i2 = indices[3 * i + 2];

				ss << "3 " << i0 << " " << i1 << " " << i2 << endl;

				if (0 == i % 10000 && i != 0)
				{
					auto percent = ((double)i / (double)(indices.size() / 3)) * 100.0;
					printf("[%llu / %llu] %f percent\n", i, indices.size() / 3, percent);
				}
			}
		}

		ofs << ss.rdbuf();
		ofs.close();

		printf("\"%s\" saved.\n", filename.c_str());

		return true;
	}

	virtual bool Deserialize(const string& filename)
	{
		ifstream ifs(filename);
		if (false == ifs.is_open())
		{
			printf("filename : %s is not open\n", filename.c_str());
			return false;
		}

		stringstream buffer;
		buffer << ifs.rdbuf();

		string line;
		vector<string> elementNames;
		vector<size_t> elementCounts;
		vector<vector<string>> elementPropertyTypes;
		vector<vector<string>> elementPropertyNames;

		if (buffer.good())
		{
			getline(buffer, line);
			if (false == ("ply" == line || "PLY" == line))
			{
				printf("Not a ply file");
				return false;
			}
		}
		while (buffer.good())
		{
			getline(buffer, line);
			stringstream ss(line);
			auto words = split(line, " \t");
			if (words[0] == "format")
			{

			}
			else if (words[0] == "element")
			{
				elementNames.push_back(words[1]);
				elementCounts.push_back(atoi(words[2].c_str()));
			}
			else if (words[0] == "property")
			{
				auto index = elementNames.size() - 1;
				if (elementPropertyTypes.size() <= index)
				{
					elementPropertyTypes.push_back(vector<string>());
					elementPropertyNames.push_back(vector<string>());
				}
				if ("list" != words[1])
				{
					elementPropertyTypes[index].push_back(words[1]);
					elementPropertyNames[index].push_back(words[2]);
					if ("a" == words[2] || "alpha" == words[2])
					{
						useAlpha = true;
					}
				}
				else
				{

				}
			}
			else if (words[0] == "end_header")
			{
				break;
			}
		}

		for (size_t i = 0; i < elementNames.size(); i++)
		{
			float x, y, z, nx, ny, nz;
			unsigned char red, green, blue, alpha;
			for (size_t j = 0; j < elementCounts[i]; j++)
			{
				getline(buffer, line);
				auto words = split(line, " \t");

				for (size_t k = 0; k < words.size(); k++)
				{
					if (elementPropertyNames[i][k] == "x")
					{
						x = atof(words[k].c_str());
					}
					else if (elementPropertyNames[i][k] == "y")
					{
						y = atof(words[k].c_str());
					}
					else if (elementPropertyNames[i][k] == "z")
					{
						z = atof(words[k].c_str());

						AddPoint(x, y, z);
					}

					else if (elementPropertyNames[i][k] == "nx")
					{
						nx = atof(words[k].c_str());
					}
					else if (elementPropertyNames[i][k] == "ny")
					{
						ny = atof(words[k].c_str());
					}
					else if (elementPropertyNames[i][k] == "nz")
					{
						nz = atof(words[k].c_str());

						AddNormal(nx, ny, nz);
					}

					else if (elementPropertyNames[i][k] == "red")
					{
						red = (unsigned char)atoi(words[k].c_str());
					}
					else if (elementPropertyNames[i][k] == "green")
					{
						green = (unsigned char)atoi(words[k].c_str());
					}
					else if (elementPropertyNames[i][k] == "blue")
					{
						blue = (unsigned char)atoi(words[k].c_str());
						if (false == useAlpha)
						{
							AddColor((float)red / 255.0f, (float)green / 255.0f, (float)blue / 255.0f);
						}
					}
					else if (elementPropertyNames[i][k] == "alpha")
					{
						alpha = (unsigned char)atoi(words[k].c_str());

						AddColor((float)red / 255.0f, (float)green / 255.0f, (float)blue / 255.0f, (float)alpha / 255.0f);
					}
				}
			}
		}

		printf("\"%s\" loaded.\n", filename.c_str());

		return true;
	}

	virtual bool SerializeAsync(const string& filename)
	{
		async(launch::async, [&, filename]() {
			Serialize(filename);
			});
		return true;
	}

	inline const vector<float>& GetNormals() const { return normals; }
	inline const vector<unsigned int>& GetIndices() const { return indices; }
	inline const vector<float>& GetColors() const { return colors; }
	inline const vector<uint8_t>& GetMaterialIDs() const { return materialIDs; }
	inline const vector<unsigned short>& GetStartPatchIDs() const { return startPatchIDs; }
	inline bool UseAlpha() const { return useAlpha; }

	virtual inline void AddUV(float u, float v)
	{
		uvs.push_back(u);
		uvs.push_back(v);
	}

	virtual inline void AddUVFloat2(const float* uv)
	{
		uvs.push_back(uv[0]);
		uvs.push_back(uv[1]);
	}

	virtual inline void AddNormal(float x, float y, float z)
	{
		normals.push_back(x);
		normals.push_back(y);
		normals.push_back(z);
	}

	virtual inline void AddNormalFloat3(const float* normal)
	{
		normals.push_back(normal[0]);
		normals.push_back(normal[1]);
		normals.push_back(normal[2]);
	}

	virtual inline void AddIndex(unsigned int index) { indices.push_back(index); }

	virtual inline void AddColor(float r, float g, float b)
	{
		colors.push_back(r);
		colors.push_back(g);
		colors.push_back(b);
	}

	virtual inline void AddColorFloat3(const float* color)
	{
		colors.push_back(color[0]);
		colors.push_back(color[1]);
		colors.push_back(color[2]);
	}

	virtual inline void AddColor(float r, float g, float b, float a)
	{
		colors.push_back(r);
		colors.push_back(g);
		colors.push_back(b);
		colors.push_back(a);
		useAlpha = true;
	}

	virtual inline void AddColorFloat4(const float* color)
	{
		colors.push_back(color[0]);
		colors.push_back(color[1]);
		colors.push_back(color[2]);
		colors.push_back(color[3]);
		useAlpha = true;
	}

	virtual inline void SetColor(size_t index, float color) { if (index < colors.size() - 1) colors[index] = color; }

	virtual inline void AddMaterialId(uint8_t materialId) {
		materialIDs.push_back(materialId);
	}

	virtual inline void AddStartPatchID(const unsigned short patchID) {
		startPatchIDs.push_back(patchID);
	}

protected:
	vector<float> uvs;
	vector<float> normals;
	vector<unsigned int> indices;
	vector<float> colors;
	vector<uint8_t> materialIDs;
	vector<unsigned short> startPatchIDs;
	bool useAlpha = false;
};
