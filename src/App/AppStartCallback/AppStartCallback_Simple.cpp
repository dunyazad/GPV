#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback_Simple(App* pApp)
{
	auto renderer = pApp->GetRenderer();
	//LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	vtkNew<vtkPLYReader> reader;
	reader->SetFileName("C:\\Resources\\3D\\PLY\\Complete\\Lower_pointcloud.ply");
	//reader->SetFileName("./../../res/3D/Lower_pointcloud.ply");
	reader->Update();

	vtkPolyData* polyData = reader->GetOutput();

	auto plyPoints = polyData->GetPoints();
	float* rawPoints = static_cast<float*>(plyPoints->GetData()->GetVoidPointer(0));
	vtkDataArray* plyNormals = polyData->GetPointData()->GetNormals();
	float* rawNormals = static_cast<float*>(plyNormals->GetVoidPointer(0));
	vtkUnsignedCharArray* plyColors = vtkUnsignedCharArray::SafeDownCast(polyData->GetPointData()->GetScalars());

	static vector<Eigen::Vector3f> points;
	static vector<Eigen::Vector3f> normals;
	static vector<Color4> colors;

	vector<unsigned int> pointIndices;

	auto bounds = polyData->GetBounds();
	Eigen::AlignedBox3f aabb(
		Eigen::Vector3f{ (float)bounds[0], (float)bounds[2], (float)bounds[4] },
		Eigen::Vector3f{ (float)bounds[1], (float)bounds[3], (float)bounds[5] });
	Eigen::Vector3f center = aabb.center();

	Eigen::Vector3f aabbDelta = aabb.max() - aabb.min() - center;
	float axisMax = aabbDelta.x();
	if (axisMax < aabbDelta.y()) axisMax = aabbDelta.y();
	if (axisMax < aabbDelta.z()) axisMax = aabbDelta.z();

	auto t = Time::Now();

	//float voxelSize = 0.1f;
	//float truncationDistance = 1.0f;
	//float weight = 1.0f;
	//vector<Voxel> volume;
	//volume.resize(1000 * 1000 * 1000);

	//t = Time::End(t, "Initialize volume");

	//for (size_t pi = 0; pi < plyPoints->GetNumberOfPoints(); pi++)
	//{
	//	pointIndices.push_back((unsigned int)pi);

	//	auto dp = plyPoints->GetPoint(pi);
	//	auto normal = plyNormals->GetTuple(pi);
	//	unsigned char color[3];
	//	plyColors->GetTypedTuple(pi, color);

	//	Eigen::Vector3f point((float)dp[0] - center.x(), (float)dp[1] - center.y(), (float)dp[2] - center.z());
	//	points.push_back(point);
	//	Eigen::Vector3f pointNormal((float)normal[0], (float)normal[1], (float)normal[2]);
	//	normals.push_back(pointNormal);
	//	auto color4 = Color4(color[0], color[1], color[2], 255);
	//	colors.push_back(color4);
	//	Eigen::Vector3f pointColor((float)color4.x() / 255.0f, (float)color4.y() / 255.0f, (float)color4.z() / 255.0f);

	//	//VD::AddSphere("points",
	//	//	point,
	//	//	{ 0.1f,0.1f,0.1f },
	//	//	{ 0.0f, 0.0f, 1.0f },
	//	//	Color4(color[0], color[1], color[2], 255));

	//	int xIndex = (int)floorf((point.x() + 50.0f) / voxelSize);
	//	int yIndex = (int)floorf((point.y() + 50.0f) / voxelSize);
	//	int zIndex = (int)floorf((point.z() + 50.0f) / voxelSize);

	//	for (int z = zIndex - 5; z <= zIndex + 5; z++)
	//	{
	//		for (int y = yIndex - 5; y <= yIndex + 5; y++)
	//		{
	//			for (int x = xIndex - 5; x <= xIndex + 5; x++)
	//			{
	//				size_t index = z * 1000 * 1000 + y * 1000 + x;

	//				Eigen::Vector3f position((float)x* voxelSize - 50.0f, (float)y* voxelSize - 50.0f, (float)z* voxelSize - 50.0f);
	//				float distance = (position - point).norm();

	//				float tsdfValue = 0.0f;
	//				if (distance <= truncationDistance) {
	//					tsdfValue = distance / truncationDistance;
	//					if ((position - point).dot(point) < 0.0f) {
	//						tsdfValue = -tsdfValue;
	//					}


	//					auto& voxel = volume[index];

	//					float oldTSDF = voxel.tsdfValue;
	//					if (FLT_MAX == oldTSDF)
	//					{
	//						voxel.tsdfValue = tsdfValue;
	//						voxel.weight = 1.0f;
	//						voxel.normal = pointNormal;
	//						voxel.color = pointColor;
	//					}
	//					else
	//					{
	//						float oldWeight = voxel.weight;
	//						float newTSDF = (oldTSDF * oldWeight + tsdfValue * weight) / (oldWeight + weight);
	//						float newWeight = oldWeight + weight;
	//						if (fabsf(newTSDF) < fabsf(oldTSDF))
	//						{
	//							voxel.tsdfValue = newTSDF;
	//							voxel.weight = oldWeight + weight;

	//							Eigen::Vector3f oldNormal = voxel.normal;
	//							Eigen::Vector3f oldColor = voxel.color;
	//							voxel.normal = (oldNormal + pointNormal) * 0.5f;
	//							voxel.color = (oldColor + pointColor) * 0.5f;
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
	//}

	//t = Time::End(t, "Integrate points");

	//for (size_t i = 0; i < volume.size(); i++)
	//{
	//	int zIndex = i / (1000 * 1000);
	//	int yIndex = (i % (1000 * 1000)) / 1000;
	//	int xIndex = (i % (1000 * 1000)) % 1000;

	//	if (-0.05f < volume[i].tsdfValue && volume[i].tsdfValue < 0.05f)
	//	{
	//		Eigen::Vector3f position((float)xIndex * voxelSize, (float)yIndex * voxelSize, (float)zIndex * voxelSize);

	//		Eigen::Vector3f c = volume[i].color.normalized();
	//		Color4 c4(255, 255, 255, 255);

	//		c4.FromNormalzed(c.x(), c.y(), c.z(), 1.0f);

	//		VD::AddCube("voxel", { (float)xIndex * voxelSize - 50.0f, (float)yIndex * voxelSize - 50.0f, (float)zIndex * voxelSize - 50.0f }, 0.05f, c4);
	//	}
	//}

	//auto mesh = GenerateMeshFromVolume(volume, 1000, 0.1f);

	//printf("mesh.vertices : %llu\n", mesh.vertices.size());
	//printf("mesh.indices : %llu\n", mesh.indices.size());
	//printf("mesh.normals : %llu\n", mesh.normals.size());
	//printf("mesh.colors : %llu\n", mesh.colors.size());

	//for (size_t i = 0; i < mesh.indices.size() / 3; i++)
	//{
	//	auto& i0 = mesh.indices[i * 3];
	//	auto& i1 = mesh.indices[i * 3 + 1];
	//	auto& i2 = mesh.indices[i * 3 + 2];

	//	auto& v0 = mesh.vertices[i * 3];
	//	auto& v1 = mesh.vertices[i * 3 + 1];
	//	auto& v2 = mesh.vertices[i * 3 + 2];

	//	auto& n0 = mesh.normals[i * 3];
	//	auto& n1 = mesh.normals[i * 3 + 1];
	//	auto& n2 = mesh.normals[i * 3 + 2];

	//	auto& c0 = mesh.colors[i * 3];
	//	auto& c1 = mesh.colors[i * 3 + 1];
	//	auto& c2 = mesh.colors[i * 3 + 2];

	//	VD::AddTriangle("Mesh", v0, v1, v2, Color4::White);
	//}

	////VD::AddCube("AABC", { 0.0f, 0.0f, 0.0f }, axisMax * 0.5f, Color4::White);
	////printf("axisMax : %f\n", axisMax);

	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { axisMax * 0.5f, 0.0f, 0.0f }, Color4::Red);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, axisMax * 0.5f, 0.0f }, Color4::Green);
	VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, axisMax * 0.5f }, Color4::Blue);

	t = Time::End(t, "Visualize");
}