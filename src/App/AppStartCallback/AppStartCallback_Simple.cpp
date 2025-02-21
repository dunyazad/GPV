#include <App/AppStartCallback/AppStartCallback.h>

#include <App/ResourceIO.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

struct SpatialIndex
{
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const SpatialIndex& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std {
    template <>
    struct hash<SpatialIndex> {
        size_t operator()(const SpatialIndex& s) const {
            return ((size_t)s.x * 73856093) ^ ((size_t)s.y * 19349663) ^ ((size_t)s.z * 83492791);
        }
    };
}

bool operator<(const SpatialIndex& a, const SpatialIndex& b)
{
	if (a.x != b.x) return a.x < b.x;
	if (a.y != b.y) return a.y < b.y;
	return a.z < b.z;
}

void AppStartCallback_Simple(App* pApp)
{
	auto t = Time::Now();

	auto renderer = pApp->GetRenderer();
	
	float voxelSize = 0.1f;
	std::mutex volumeMutex;
	unordered_map<SpatialIndex, float> volume;

	PLYFormat ply;
	ply.Deserialize(ResourceIO::GetPath("Debug/Patches/point_0.ply").string());

    t = Time::Now();

    std::vector<float>& points = ply.GetPoints();
    size_t numPoints = points.size() / 3;

    int offset = 0; // Voxel 범위를 확장하는 정도

#pragma omp parallel for
    for (size_t i = 0; i < numPoints; i++)
    {
        auto x = points[i * 3];
        auto y = points[i * 3 + 1];
        auto z = points[i * 3 + 2];

        auto xIndex = static_cast<int>(floorf(x / voxelSize));
        auto yIndex = static_cast<int>(floorf(y / voxelSize));
        auto zIndex = static_cast<int>(floorf(z / voxelSize));

        std::vector<SpatialIndex> localUpdates;

        for (int zOffset = -offset; zOffset <= offset; zOffset++)
        {
            for (int yOffset = -offset; yOffset <= offset; yOffset++)
            {
                for (int xOffset = -offset; xOffset <= offset; xOffset++)
                {
                    localUpdates.push_back({ xIndex + xOffset, yIndex + yOffset, zIndex + zOffset });
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(volumeMutex);
            for (const auto& key : localUpdates)
            {
                volume[key] = 1.0f;
            }
        }
    }

    t = Time::End(t, "Loading");

    auto cameraPosition = Eigen::Vector3f(16.584496, -11.332185, 117.763481);

	for (auto& kvp : volume)
	{
		auto x = (float)kvp.first.x * voxelSize;
		auto y = (float)kvp.first.y * voxelSize;
		auto z = (float)kvp.first.z * voxelSize;

        auto p = Eigen::Vector3f(x, y, z);

		VD::AddSphere("quantized points", p, 0.05f, Color4::White);
	}


	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	t = Time::End(t, "Visualize");
}