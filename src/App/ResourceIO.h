#pragma once

#include <Common.h>

#include <App/Serialization.hpp>

template<typename T, typename U>
class ResourceReader
{
public:
	ResourceReader(const filesystem::path& filepath)
		: filepath(filepath)
	{
		ifs.open(filepath.string(), ios::in | ios::binary);
	}

	~ResourceReader()
	{
		ifs.close();
	}

	bool ReadHeader(T* header)
	{
		return ifs.read((i8*)header, sizeof(T)) ? true : false;
	}

	bool ReadData(U* elements, ui32 numberOfElements)
	{
		bool result = true;
		for (ui32 i = 0; i < numberOfElements; i++)
		{
			result = ifs.read((i8*)elements + i * sizeof(U), sizeof(U)) ? true : false;
			if (false == result) return false;
		}
		return true;
	}

private:
	filesystem::path filepath;
	ifstream ifs;
};

class ResourceIO
{
public:
	static inline filesystem::path GetResourceRoot() { return resourceRoot; };
	static inline filesystem::path GetPath(const filesystem::path& name) { return resourceRoot / name; };

	static vector<Eigen::Matrix4f> ReadTransformsFile(const filesystem::path& filepath);
private:
	static filesystem::path resourceRoot;
};
