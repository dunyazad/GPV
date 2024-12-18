#include <App/AppStartCallback/AppStartCallback.h>

#include <Debugging/VisualDebugging.h>
using VD = VisualDebugging;

void AppStartCallback_Capture(App* pApp)
{
	pApp->AddAppUpdateCallback([&](App* pApp) -> bool {
		if (enabledToCapture)
		{
			CaptureNextFrame(pApp);
		}

		//static size_t index = 0;

		//auto kdTreePoints = (vector<Eigen::Vector3f>*)pApp->registry["kdTreePoints"];
		//auto kdTreeColors = (vector<Color4>*)pApp->registry["kdTreeColors"];

		//auto& p = (*kdTreePoints)[index];
		//auto& c = (*kdTreeColors)[index];

		//VD::AddSphere("points",
		//	p,
		//	{ 0.05f,0.05f,0.05f },
		//	{ 0.0f, 0.0f, 1.0f },
		//	c);
		//index++;

		return true;
	});

	auto renderer = pApp->GetRenderer();

	//{
		//	vtkNew<vtkPLYReader> reader;
		//	reader->SetFileName("C:\\Resources\\2D\\Captured\\PointCloud\\point_0.ply");
		//	reader->Update();

		//	vtkSmartPointer<vtkPolyData> polyData = reader->GetOutput();

		//	vector<Eigen::Vector3f> points;
		//	auto plyPoints = polyData->GetPoints();
		//	for (size_t i = 0; i < plyPoints->GetNumberOfPoints(); i++)
		//	{
		//		auto dp = plyPoints->GetPoint(i);
		//		auto p = Eigen::Vector3f(dp[0], dp[1], dp[2]);
		//		points.push_back(p);
		//		VD::AddSphere("points", p, { 0.1f, 0.1f, 0.1f }, { 0.0f, 0.0f, 1.0f }, Color4::White);
		//	}

		//	{
		//		float3* d_points;
		//		float3* d_normals;
		//		cudaMallocManaged(&d_points, sizeof(float3) * points.size());
		//		cudaMallocManaged(&d_normals, sizeof(float3) * points.size());

		//		cudaDeviceSynchronize();

		//		cudaMemcpy(d_points, points.data(), sizeof(float3) * points.size(), cudaMemcpyHostToDevice);

		//		CUDA::GeneratePatchNormals(256, 480, d_points, points.size(), d_normals);

		//		for (size_t i = 0; i < points.size(); i++)
		//		{
		//			auto n = d_normals[i];
		//			VD::AddLine("normals", points[i], points[i] + Eigen::Vector3f(n.x, n.y, n.z), Color4::Red);
		//		}
		//	}
		//	return;
		//}

	LoadModel(renderer, "C:\\Resources\\3D\\PLY\\Complete\\Lower.ply");

	auto camera = renderer->GetActiveCamera();
	camera->SetParallelProjection(true);
	// Parallel Scale은 카메라 절반 높이
	// 픽셀당 3D 공간의 유닛 * 창 높이 / 2
	// 여기에선 256 x 480이므로 픽셀당 0.1, 창높이 480
	// 480 * 0.1 / 2 = 24
	camera->SetParallelScale(24);

	LoadTRNFile();

	//LoadDepthImage();

	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 100.0f, 0.0f, 0.0f }, Color4::Red);
	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 100.0f, 0.0f }, Color4::Green);
	//VisualDebugging::AddLine("axes", { 0, 0, 0 }, { 0.0f, 0.0f, 100.0f }, Color4::Blue);

	enabledToCapture = true;
}
