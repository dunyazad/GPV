#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

size_t GetKeyFromIndex(size_t xIndex, size_t yIndex, size_t zIndex)
{
	return xIndex << 32 | yIndex << 16 | zIndex;
}

size_t GetKey(const Eigen::Vector3f& volumeMin, float voxelSize, const Eigen::Vector3f& position)
{
	Eigen::Vector3f localPosition = position - volumeMin;
	size_t xIndex = (size_t)(unsigned int)floorf(localPosition.x() / voxelSize);
	size_t yIndex = (size_t)(unsigned int)floorf(localPosition.y() / voxelSize);
	size_t zIndex = (size_t)(unsigned int)floorf(localPosition.z() / voxelSize);
	return GetKeyFromIndex(xIndex, yIndex, zIndex);
}

tuple<size_t, size_t, size_t> GetIndex(const Eigen::Vector3f& volumeMin, float voxelSize, size_t key)
{
	size_t xIndex = key >> 32 & ((1 << 16) - 1);
	size_t yIndex = key >> 16 & ((1 << 16) - 1);
	size_t zIndex = key & ((1 << 16) - 1);

	return make_tuple(xIndex, yIndex, zIndex);
}

Eigen::Vector3f GetPosition(const Eigen::Vector3f& volumeMin, float voxelSize, size_t key)
{
	auto [xIndex, yIndex, zIndex] = GetIndex(volumeMin, voxelSize, key);
	return volumeMin + Eigen::Vector3f(xIndex * voxelSize, yIndex * voxelSize, zIndex * voxelSize);
}

struct Node
{
	int neigborCount = 0;
	Node* neighbors[26] = { 0 };
	int tag = -1;
};

size_t FloodFill(Node* node, int tag) {

	stack<Node*> nodes;
	nodes.push(node);

	size_t count = 0;

	while (false == nodes.empty())
	{
		auto currentNode = nodes.top();
		nodes.pop();

		if (-1 != currentNode->tag)
			continue;

		currentNode->tag = tag;
		count++;

		for (int i = 0; i < currentNode->neigborCount; i++)
		{
			nodes.push(currentNode->neighbors[i]);
		}
	}

	//if (count > 1000)
	//{
	//	printf("[%d] count : %d\n", tag, count);
	//}

	return count;
};

void AppStartCallback_Clustering(App* pApp)
{
	auto renderer = pApp->GetRenderer();

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100 * 0.5f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100 * 0.5f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100 * 0.5f }, Color4::Blue);

	auto t = Time::Now();
	
	map<size_t, Node*> quantizingMap;

	PLYFormat ply;
	ply.Deserialize("C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply");
	
	float voxelSize = 0.1f;
	auto aabbMin = ply.GetAABBMin();
	auto aabbMax = ply.GetAABBMax();

	//auto aabbMin = Eigen::Vector3f(aabb.min().minCoeff(), aabb.min().minCoeff(), aabb.min().minCoeff());
	//auto aabbMax = Eigen::Vector3f(aabb.max().maxCoeff(), aabb.max().maxCoeff(), aabb.max().maxCoeff());

	auto volumeMinX = floorf(aabbMin.x() / voxelSize) * voxelSize;
	auto volumeMinY = floorf(aabbMin.y() / voxelSize) * voxelSize;
	auto volumeMinZ = floorf(aabbMin.z() / voxelSize) * voxelSize;
	auto volumeMaxX = ceilf(aabbMax.x() / voxelSize) * voxelSize;
	auto volumeMaxY = ceilf(aabbMax.y() / voxelSize) * voxelSize;
	auto volumeMaxZ = ceilf(aabbMax.z() / voxelSize) * voxelSize;

	auto volumeMin = Eigen::Vector3f(volumeMinX, volumeMinY, volumeMinZ);
	auto volumeMax = Eigen::Vector3f(volumeMaxX, volumeMaxY, volumeMaxZ);

	//VD::AddBox("aabb", aabbMin, aabbMax, Color4::Blue);
	//VD::AddBox("volume", volumeMin, volumeMax, Color4::Red);

	//VD::AddLine("aabb", aabbMin, aabbMax, Color4::Blue);
	//VD::AddLine("volume", volumeMin, volumeMax, Color4::Red);

	vector<Eigen::Vector3f> loadedPoints;

	for (size_t i = 0; i < ply.GetPoints().size() / 3; i++)
	{
		auto x = ply.GetPoints()[i * 3];
		auto y = ply.GetPoints()[i * 3 + 1];
		auto z = ply.GetPoints()[i * 3 + 2];

		VD::AddSphere("Points", { x, y, z }, 0.0005f, Color4::Red);

		loadedPoints.push_back(Eigen::Vector3f(x, y, z));
	}

	t = Time::End(t, "Loading");

	for (size_t i = 0; i < loadedPoints.size(); i++)
	{
		auto& p = loadedPoints[i];
		auto key = GetKey(volumeMin, voxelSize, p);

		if (0 == quantizingMap.count(key))
		{
			quantizingMap[key] = new Node;
		}
	}

	for (auto& kvp : quantizingMap)
	{
		auto key = kvp.first;
		auto& position = GetPosition(volumeMin, voxelSize, key);

		//VD::AddCube("cubes", position + Eigen::Vector3f(voxelSize * 0.5f, voxelSize * 0.5f, voxelSize * 0.5f), voxelSize * 0.5f, Color4::White);

		auto [xIndex, yIndex, zIndex] = GetIndex(volumeMin, voxelSize, key);

		for (int z = -1; z <=1; z++)
		{
			if (zIndex == 0) continue;

			for (int y = -1; y <= 1; y++)
			{
				if (yIndex == 0) continue;

				for (int x = -1; x <= 1; x++)
				{
					if (xIndex == 0) continue;

					if (0 == x && 0 == y && 0 == z) continue;

					auto neighborKey = GetKeyFromIndex(xIndex + x, yIndex + y, zIndex + z);
					if (0 != quantizingMap.count(neighborKey))
					{
						kvp.second->neighbors[kvp.second->neigborCount++] = quantizingMap[neighborKey];
					}
				}
			}
		}
	}

	t = Time::End(t, "Quantizing");

	int tagCount = 0;
	vector<int> tagVoxelCount;
	for (auto& kvp : quantizingMap)
	{
		if (-1 != kvp.second->tag)
		{
			continue;
		}
		else
		{
			auto count = FloodFill(kvp.second, tagCount++);
			tagVoxelCount.push_back(count);
		}
	}

	t = Time::End(t, "Clustering");

	Color4 colors[6] = { Color4::Green, Color4::Blue, Color4::Black,
		                 Color4::Yellow, Color4::Magenta, Color4::Cyan };

	for (auto& kvp : quantizingMap)
	{
		auto key = kvp.first;
		auto& position = GetPosition(volumeMin, voxelSize, key);

		if (-1 == kvp.second->tag)
		{
			VD::AddCube("cubes", position + Eigen::Vector3f(voxelSize * 0.5f, voxelSize * 0.5f, voxelSize * 0.5f), voxelSize * 0.5f, Color4::Red);
		}
		else
		{
			if (tagVoxelCount[kvp.second->tag] < 10000)
			{
				VD::AddCube("cubes", position + Eigen::Vector3f(voxelSize * 0.5f, voxelSize * 0.5f, voxelSize * 0.5f), voxelSize * 0.5f, Color4::Red);
			}
			else
			{
				VD::AddCube("cubes", position + Eigen::Vector3f(voxelSize * 0.5f, voxelSize * 0.5f, voxelSize * 0.5f), voxelSize * 0.5f, colors[kvp.second->tag % 6]);
			}
		}
	}

	//printf("tagCount : %d\n", tagCount);
	//printf("quantizingMap.size() : %d\n", quantizingMap.size());

	t = Time::End(t, "Visualizing");

	for (auto& kvp : quantizingMap)
	{
		delete kvp.second;
	}
}
