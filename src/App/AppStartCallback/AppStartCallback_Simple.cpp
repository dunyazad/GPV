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

    {
        PLYFormat ply;
        ply.Deserialize(ResourceIO::GetPath("Debug/Compound.ply").string());

        t = Time::Now();

        std::vector<float>& points = ply.GetPoints();
        size_t numPoints = points.size() / 3;

#pragma omp parallel for
        for (size_t i = 0; i < numPoints; i++)
        {
            auto x = points[i * 3];
            auto y = points[i * 3 + 1];
            auto z = points[i * 3 + 2];

            auto p = Eigen::Vector3f(x, y, z);

            VD::AddSphere("compound", p, 0.05f, Color4::White);
        }
    }
    {
        PLYFormat ply;
        ply.Deserialize(ResourceIO::GetPath("Debug/Patch.ply").string());

        t = Time::Now();

        std::vector<float>& points = ply.GetPoints();
        size_t numPoints = points.size() / 3;

#pragma omp parallel for
        for (size_t i = 0; i < numPoints; i++)
        {
            auto x = points[i * 3];
            auto y = points[i * 3 + 1];
            auto z = points[i * 3 + 2];

            auto p = Eigen::Vector3f(x, y, z);

            VD::AddSphere("patch", p, 0.05f, Color4::Red);
        }
    }
    t = Time::End(t, "Loading");

    auto cameraPosition = Eigen::Vector3f(16.584496, -11.332185, 117.763481);

    auto rotation = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), cameraPosition);

    //VD::AddLine("to camposition", { 0.0f, 0.0f, 0.0f }, cameraPosition, Color4::Yellow);

    //VD::AddLine("to camposition", { 0.0f, 0.0f, 0.0f }, rotation * Eigen::Vector3f::UnitZ() * 20.0f, Color4::Magenta);

    float width = 400.0f * voxelSize * 0.33333f;
    float height = 480.0f * voxelSize * 0.33333f;

    auto lu = Eigen::Vector3f(-width * 0.5f, height * 0.5f, 0.0f);
    auto ll = Eigen::Vector3f(-width * 0.5f, -height * 0.5f, 0.0f);
    auto ru = Eigen::Vector3f(width * 0.5f, height * 0.5f, 0.0f);
    auto rl = Eigen::Vector3f(width * 0.5f, -height * 0.5f, 0.0f);
    auto lc = Eigen::Vector3f(0.0f, 0.0f, cameraPosition.norm());

    VD::AddLine("CamSpace", lu, ll, Color4::Yellow);
    VD::AddLine("CamSpace", ll, rl, Color4::Yellow);
    VD::AddLine("CamSpace", rl, ru, Color4::Yellow);
    VD::AddLine("CamSpace", ru, lu, Color4::Yellow);
    VD::AddLine("CamSpace", lu, lc, Color4::Yellow);
    VD::AddLine("CamSpace", ll, lc, Color4::Yellow);
    VD::AddLine("CamSpace", rl, lc, Color4::Yellow);
    VD::AddLine("CamSpace", ru, lc, Color4::Yellow);


	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	t = Time::End(t, "Visualize");
}