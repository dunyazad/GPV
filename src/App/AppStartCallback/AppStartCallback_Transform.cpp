#include <App/AppStartCallback/AppStartCallback.h>

#include <App/ResourceIO.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

Eigen::Vector3f ComputePosition(float width, float height, float centerDistance, float distance, Eigen::Quaternionf rotation)
{
    //centerData: width * 0.5f = distance : x
    //width * 0.5f * distance = centerDistance * x
    float x = width * 0.5f * distance / centerDistance;
    float y = height * 0.5f * distance / centerDistance;
    float z = distance;
    Eigen::Vector3f point(x, y, z);
    //return rotation * point;
    return point;
}

__host__ __device__
bool rayPlaneIntersection(
	const Eigen::Vector3f& rayOrigin,
	const Eigen::Vector3f& rayDirection,
	const Eigen::Vector3f& planeOrigin,
	const Eigen::Vector3f& planeNormal,
	Eigen::Vector3f& intersection)
{
	double denom = planeNormal.dot(rayDirection);

	if (fabsf(denom) < 1e-6) {
		return false;
	}

	double t = (planeNormal.dot(planeOrigin - rayOrigin)) / denom;

	if (t < 0) {
		return false;
	}

	intersection = rayOrigin + t * rayDirection;
	return true;
}

Eigen::Vector2f ProjectPointToPlane(
    Eigen::Vector3f cameraPosition, Eigen::Vector3f planePosition,
    Eigen::Vector3f planeNormal, const Eigen::Vector3f& point3D) {
    Eigen::Vector3f viewDir = point3D - cameraPosition;

    float t = -(planeNormal.dot(cameraPosition)) / (planeNormal.dot(viewDir));

    Eigen::Vector3f projectedPoint = cameraPosition + t * viewDir;

    Eigen::Vector3f U, V;
    if (planeNormal.x() != 0 || planeNormal.y() != 0) {
        U = Eigen::Vector3f(-planeNormal.y(), planeNormal.x(), 0).normalized();
    }
    else {
        U = Eigen::Vector3f(1, 0, 0);
    }
    V = planeNormal.cross(U).normalized();

    float u = (projectedPoint - planePosition).dot(U);
    float v = (projectedPoint - planePosition).dot(V);

    return Eigen::Vector2f(u, v);
}

__host__ __device__
Eigen::Vector2i GetUV(
    const Eigen::Vector3f& point,
    const Eigen::Matrix3f& rotation, float width, float height)
{
    Eigen::Vector3f rp = rotation * point;
    unsigned int u = floorf((rp.x() + width * 0.5f));
    unsigned int v = floorf((rp.y() + height * 0.5f));
    return Eigen::Vector2i(u, v);
}

__host__ __device__
Eigen::Vector2i ComputeUV(float width, float height, const Eigen::Vector3f& refPoint, const Eigen::Quaternionf& rotation, const Eigen::Vector3f& point)
{
    auto rayDirection = (point - refPoint).normalized();
    Eigen::Vector3f intersection;
    rayPlaneIntersection(refPoint, rayDirection, Eigen::Vector3f::Zero(), refPoint.normalized(), intersection);
    Eigen::Vector3f rp = rotation * intersection;
    rp.x() += width * 0.5f;
    rp.y() += height * 0.5f;

    return Eigen::Vector2i((int)(rp.x() * 1000.0f), (int)(rp.y() * 1000.0f));
}

void AppStartCallback_Transform(App* pApp)
{
	auto t = Time::Now();

	auto renderer = pApp->GetRenderer();
	
	PLYFormat ply;
	ply.Deserialize(ResourceIO::GetPath("Debug/Patches/point_0.ply").string());

    t = Time::Now();

    std::vector<float>& points = ply.GetPoints();
    size_t numPoints = points.size() / 3;

    auto cameraPosition = Eigen::Vector3f(16.584496, -11.332185, 117.763481);
    auto planeNormal = cameraPosition.normalized();
    auto planeOrigin = Eigen::Vector3f::Zero();

    VD::AddLine("cameraPosition Line", Eigen::Vector3f::Zero(), cameraPosition, Color4::Red);

    for (size_t i = 0; i < numPoints; i++)
    {
        auto x = points[i * 3];
        auto y = points[i * 3 + 1];
        auto z = points[i * 3 + 2];

        auto p = Eigen::Vector3f(x, y, z);

        VD::AddSphere("quantized points", p, 0.05f, Color4::White);

        auto rayDirection = (p - cameraPosition).normalized();
        Eigen::Vector3f intersection;
        rayPlaneIntersection(cameraPosition, rayDirection, planeOrigin, planeNormal, intersection);

        //VD::AddLine("Line", cameraPosition, intersection, Color4::Red);
        VD::AddSphere("intersection points", intersection, 0.05f, Color4::Red);


        auto rotation = Eigen::Quaternionf::FromTwoVectors(cameraPosition.normalized(), Eigen::Vector3f::UnitZ());
        Eigen::Vector3f rp = rotation * intersection;
        rp.x() += 20.0f;
        rp.y() += 24.0f;
        VD::AddSphere("rotated points", rp, 0.05f, Color4::Green);


        auto uv = ComputeUV(400.0f * 0.1f, 480.0f * 0.1f, cameraPosition, rotation, p);
        VD::AddSphere("rotated points   ", {(float)uv.x() / 1000.0f, (float)uv.y() / 1000.0f, 0.0f}, 0.05f, Color4::Blue);

        //auto uv = ProjectPointToPlane(cameraPosition, Eigen::Vector3f::Zero(), cameraPosition.normalized(), p);

        //VD::AddSphere("uv", { uv.x(), uv.y(), 0.0f }, 0.05f, Color4::Green);
    }

 /*   auto rotationMatrix = GetRotationMatrix(cameraPosition.normalized(), Eigen::Vector3f::UnitZ());

    auto rotated = (Eigen::Vector3f)(rotationMatrix * cameraPosition.normalized());
    rotated = rotated.normalized();

    VD::AddLine("Rotated", Eigen::Vector3f::Zero(), rotated * 100.0f);

    for (size_t i = 0; i < numPoints; i++)
    {
        auto x = points[i * 3];
        auto y = points[i * 3 + 1];
        auto z = points[i * 3 + 2];

        auto p = Eigen::Vector3f(x, y, z);
        auto uv = GetUV(p, rotationMatrix, 400.0f, 480.0f);

        VD::AddSphere("quantized points", p, 0.05f, Color4::White);

        VD::AddSphere("projected", { (float)uv.x() - 200.0f, (float)uv.y() - 240.0f, 0.0f }, 0.05f, Color4::Red);
    }*/

    t = Time::End(t, "Loading");

    VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	t = Time::End(t, "Visualize");
}