#include "ResourceIO.h"

#ifdef BUILD_FOR_DEVELOPMENT
filesystem::path ResourceIO::resourceRoot = filesystem::current_path() / "../../res";
#else
filesystem::path ResourceIO::resourceRoot = filesystem::current_path() / "./res";
#endif

vector<Eigen::Matrix4f> ResourceIO::ReadTransformsFile(const filesystem::path& filepath)
{
	filesystem::path absolute_filepath(filepath);

	if (absolute_filepath.is_relative())
	{
		absolute_filepath = resourceRoot / filepath;
	}

	struct Header
	{
		ui64 numberOfElements;
	};

	struct Data
	{
		Eigen::Matrix4f transform0;
		Eigen::Matrix4f transform45;
	};

	ResourceReader<Header, Data> rr(absolute_filepath);
	
	Header header;
	rr.ReadHeader(&header);
	
	vector<Data> datas(header.numberOfElements);
	rr.ReadData(datas.data(), header.numberOfElements);

	vector<Eigen::Matrix4f> result;
	for (ui64 i = 0; i < header.numberOfElements; i++)
	{
		result.push_back(datas[i].transform0);
	}

	return result;
}